#!/usr/bin/env python3
"""Offline calibration of the EdgePace PMU admission thresholds.

During an active foreground frame, the EdgePace policy admits the next quantum only
if the foreground CPU's memory signals stay below fixed gates (stall <= 0.70
and demand <= 0.35 for the balanced preset).  ``pmu_sampler`` maps raw
PMU ratios into those units with constants that were hand-set for one Jetson
workload, so the same gates mean different things for another CPU or another
foreground.  This tool measures the user's own foreground and derives them:

1. solo       foreground alone                  -> baseline level B, and the
              demand reference scales of paper §3.2:
              r0 / m0 = max(p99, 2 * p95) of the per-window LLC miss rate / MPKI
2. contended  foreground + unmanaged LLM load    -> contended level C
3. sweep      managed EdgePace with stall cutoffs R(x) = B + x * (C - B)
              -> keep the candidate with the most LLM service whose excess
                 deadline-miss rate (vs. solo) stays within the budget

The foreground is launched by the sampler (``monitor/``), which attaches the
inherit counters before exec so every foreground thread is counted.  Levels
use only valid windows (paper Eq. 1) inside active foreground frames, after
the same EWMA (alpha 0.2) the sampler applies.  The sampler smooths the
normalized value clamped to [0, 1]; this tool smooths the raw ratio, so the
two differ only after spikes above the threshold.  Policy gates are left
unchanged; only the sampler normalization moves:

    --miss-rate-threshold = r0, --mpki-threshold = m0       (paper rule)
    --stall-threshold = R_stall(x) / stall_gate              (swept s0)
    so "stall <= stall_gate"  <=>  "raw backend-memory stall ratio <= R_stall(x)"

Frame timing comes from the task page (``begin_frame``/``complete`` ledger),
so any foreground that publishes that page can be calibrated.  Run it under
``scripts/locked_run.sh``.

Example::

    scripts/locked_run.sh runs/calib -- python3 python/edgepace_phase.py \\
        --model <model.gguf> --plan runs/profile/plan/cut_plan.json \\
        --output-dir runs/calib/out \\
        --foreground '<fg-binary> --video <v.mp4> --frames {frames} \\
            --task-page {task_page} --output {phase_dir}/foreground.csv'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import os
import shlex
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from edgepace_plan import (
    DEFAULT_POLICY_LIBRARY,
    DEFAULT_REQUEST,
    DEFAULT_SERVER,
    clean_environment,
    require_locked_platform,
    resolved_priority,
    runtime_environment,
    server_command,
    terminate,
    wait_server,
)


DEFAULT_SAMPLER = Path(__file__).resolve().parents[1] / "monitor/build/pmu_sampler"
SCHEMA_VERSION = "edgepace.pmu_calibration.v2"
DEMAND_GATE = 0.35          # EDGEPACE_ONLINE_DEMAND_SAFE_MAX in the runtime env
EWMA_ALPHA = 0.2            # pmu_sampler EWMA
# Sampler defaults; only used where the phase does not depend on them.
DEFAULT_THRESHOLDS = {"stall": 0.288896292, "miss_rate": 87578405.875, "mpki": 32.651825}

# Shared-page ABI (edgepace/abi.h == full_sched/abi.h).
ABI_VERSION = 1
TASK_MAGIC = 0x46535431
REQUEST_MAGIC = 0x46535231
TASK_BYTES = 512
REQUEST_BYTES = 256
HEADER = struct.Struct("<IHHII")
SEQUENCE_OFFSET = 16
LEDGER_OFFSET = 264
LEDGER_CAPACITY = 4
FRAME = struct.Struct("<6Q2I")   # frame_id, source, release, dequeue, complete, deadline, flags, reserved
FRAME_VALID = 1 << 0
FRAME_COMPLETE = 1 << 1
REQUEST_FIELDS = struct.Struct("<11Q")   # sequence .. last_submit_ns
REQUEST_STATE = struct.Struct("<6I")     # phase, running, completed, remaining, terminal, flags
REQUEST_STATE_OFFSET = 104


# --- shared pages ----------------------------------------------------------

def read_seqlock(buffer: mmap.mmap, size: int, magic: int) -> bytes | None:
    for _ in range(8):
        before = struct.unpack_from("<Q", buffer, SEQUENCE_OFFSET)[0]
        if before & 1:
            continue
        data = bytes(buffer[:size])
        after = struct.unpack_from("<Q", buffer, SEQUENCE_OFFSET)[0]
        if before != after:
            continue
        page_magic, version, page_bytes, _, _ = HEADER.unpack_from(data, 0)
        if page_magic == magic and version == ABI_VERSION and page_bytes == size:
            return data
    return None


@dataclass
class Frame:
    frame_id: int
    release_ns: int
    dequeue_ns: int
    complete_ns: int
    deadline_ns: int

    @property
    def missed(self) -> bool:
        return self.complete_ns > self.deadline_ns

    @property
    def response_ms(self) -> float:
        return (self.complete_ns - self.release_ns) / 1e6


class FramePoller(threading.Thread):
    """Collects completed frames from the task-page ledger.

    The ledger holds the last four frames (~130 ms at 30 Hz), so a 10 ms poll
    sees every completion without adding a busy reader next to the foreground.
    """

    def __init__(self, path: Path, interval_s: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.interval_s = interval_s
        self.frames: dict[int, Frame] = {}
        self.first_frame = threading.Event()
        self.stop_event = threading.Event()
        self.overwrites = 0

    def run(self) -> None:
        handle = None
        buffer = None
        try:
            while not self.stop_event.is_set():
                if buffer is None:
                    try:
                        if self.path.stat().st_size >= TASK_BYTES:
                            handle = self.path.open("rb")
                            buffer = mmap.mmap(handle.fileno(), TASK_BYTES,
                                               access=mmap.ACCESS_READ)
                    except OSError:
                        pass
                if buffer is not None:
                    self.collect(buffer)
                self.stop_event.wait(self.interval_s)
            if buffer is not None:
                self.collect(buffer)
        finally:
            if buffer is not None:
                buffer.close()
            if handle is not None:
                handle.close()

    def collect(self, buffer: mmap.mmap) -> None:
        data = read_seqlock(buffer, TASK_BYTES, TASK_MAGIC)
        if data is None:
            return
        self.overwrites = struct.unpack_from("<I", data, 256)[0]
        for index in range(LEDGER_CAPACITY):
            frame_id, _, release, dequeue, complete, deadline, flags, _ = FRAME.unpack_from(
                data, LEDGER_OFFSET + index * FRAME.size)
            if not flags & FRAME_VALID:
                continue
            self.first_frame.set()
            if flags & FRAME_COMPLETE and frame_id not in self.frames and complete:
                self.frames[frame_id] = Frame(frame_id, release, dequeue, complete, deadline)


class RequestPage:
    """Single-writer request page; EdgePace only splits graphs while running != 0."""

    def __init__(self, path: Path) -> None:
        # an existing page of the right size is reused, never truncated
        # under a server that may already have it mapped
        if not path.is_file() or path.stat().st_size != REQUEST_BYTES:
            path.write_bytes(b"\0" * REQUEST_BYTES)
        self.handle = path.open("r+b")
        self.buffer = mmap.mmap(self.handle.fileno(), REQUEST_BYTES)
        self.publish(0, running=False, enqueue_ns=0, freshness_deadline_ns=0)

    def publish(self, request_id: int, running: bool, enqueue_ns: int,
                freshness_deadline_ns: int, terminal: int = 0) -> None:
        sequence = struct.unpack_from("<Q", self.buffer, SEQUENCE_OFFSET)[0]
        writing = sequence | 1
        now = time.monotonic_ns()
        value = bytearray(REQUEST_BYTES)
        HEADER.pack_into(value, 0, REQUEST_MAGIC, ABI_VERSION, REQUEST_BYTES, 0, 0)
        REQUEST_FIELDS.pack_into(value, SEQUENCE_OFFSET, writing, now, request_id,
                                 enqueue_ns, freshness_deadline_ns, 0, 0, 0, 0, now, now)
        REQUEST_STATE.pack_into(value, REQUEST_STATE_OFFSET, 0, 1 if running else 0,
                                0, 0, terminal, 0)
        struct.pack_into("<Q", self.buffer, SEQUENCE_OFFSET, writing)
        self.buffer[SEQUENCE_OFFSET + 8:] = value[SEQUENCE_OFFSET + 8:]
        self.buffer[:SEQUENCE_OFFSET] = value[:SEQUENCE_OFFSET]
        struct.pack_into("<Q", self.buffer, SEQUENCE_OFFSET, writing + 1)

    def close(self) -> None:
        self.buffer.close()
        self.handle.close()


# --- LLM load ----------------------------------------------------------------

@dataclass
class RequestRecord:
    request_id: int
    start_ns: int
    latency_ms: float
    status: int
    fresh: bool
    # output end and the server's token counts (usage), for the
    # output-tokens / active-time throughput
    complete_ns: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    # completion of the frame the request was aligned to (0 = not aligned)
    aligned_frame_complete_ns: int = 0


class LoadGenerator(threading.Thread):
    """Issues the profile request periodically and mirrors it on the request page.

    ``align_task_page``: once a period has elapsed, wait for the next
    foreground frame completion on that task page and send right after it, as
    the paper-era client did (``four_camera_semantic_generalized_llm_client.py
    --wait-next-snapshot`` sends after each frame's semantic snapshot), so
    every request starts at the same point of the frame cycle.  Without it the
    phase of a request relative to the frames is arbitrary.
    """

    def __init__(self, host: str, port: int, payload: dict[str, Any],
                 page: RequestPage, period_ms: float, freshness_ms: float,
                 align_task_page: Path | None = None) -> None:
        super().__init__(daemon=True)
        self.url = f"http://{host}:{port}" + (
            "/v1/completions" if "prompt" in payload else "/v1/chat/completions")
        self.body = json.dumps(payload).encode("utf-8")
        self.page = page
        self.period_ns = int(period_ms * 1e6)
        self.freshness_ns = int(freshness_ms * 1e6)
        self.align_task_page = align_task_page
        self.records: list[RequestRecord] = []
        self.stop_event = threading.Event()

    def _latest_completion(self, buffer: mmap.mmap) -> int:
        data = read_seqlock(buffer, TASK_BYTES, TASK_MAGIC)
        if data is None:
            return 0
        latest = 0
        for index in range(LEDGER_CAPACITY):
            _, _, _, _, complete, _, flags, _ = FRAME.unpack_from(data, LEDGER_OFFSET + index * FRAME.size)
            if flags & FRAME_VALID and flags & FRAME_COMPLETE:
                latest = max(latest, complete)
        return latest

    def _wait_next_completion(self, buffer: mmap.mmap) -> int:
        """Blocks until a frame completes after now; returns its completion stamp."""
        seen = self._latest_completion(buffer)
        while not self.stop_event.is_set():
            latest = self._latest_completion(buffer)
            if latest > seen:
                return latest
            self.stop_event.wait(0.001)
        return 0

    def run(self) -> None:
        buffer = handle = None
        if self.align_task_page is not None:
            handle = self.align_task_page.open("rb")
            buffer = mmap.mmap(handle.fileno(), TASK_BYTES, access=mmap.ACCESS_READ)
        try:
            self._run(buffer)
        finally:
            if buffer is not None:
                buffer.close()
                handle.close()

    def _run(self, buffer: mmap.mmap | None) -> None:
        request_id = 0
        next_start = time.monotonic_ns()
        while not self.stop_event.is_set():
            frame_complete = self._wait_next_completion(buffer) if buffer is not None else 0
            if self.stop_event.is_set():
                break
            request_id += 1
            start = time.monotonic_ns()
            deadline = start + self.freshness_ns
            self.page.publish(request_id, True, start, deadline)
            status = 0
            body = b""
            try:
                request = urllib.request.Request(
                    self.url, data=self.body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120.0) as response:
                    status = response.status
                    body = response.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                status = 0
            complete = time.monotonic_ns()
            self.page.publish(request_id, False, start, deadline, terminal=1)
            usage: dict[str, Any] = {}
            try:
                usage = json.loads(body).get("usage", {}) if body else {}
            except (ValueError, AttributeError):
                usage = {}
            self.records.append(RequestRecord(
                request_id, start, (complete - start) / 1e6, status,
                200 <= status < 300 and complete <= deadline, complete,
                int(usage.get("completion_tokens", 0)), int(usage.get("prompt_tokens", 0)),
                frame_complete))
            next_start = max(next_start + self.period_ns, complete)
            self.stop_event.wait(max(0.0, (next_start - time.monotonic_ns()) / 1e9))


class BulkStream(threading.Thread):
    """an extra back-to-back client of the same server, not on the
    request page (its freshness is not tracked); it only adds batched decode
    load next to the LoadGenerator's request."""

    def __init__(self, host: str, port: int, payload: dict[str, Any]) -> None:
        super().__init__(daemon=True)
        self.url = f"http://{host}:{port}" + (
            "/v1/completions" if "prompt" in payload else "/v1/chat/completions")
        self.body = json.dumps(payload).encode("utf-8")
        self.output_tokens = 0
        self.completed = 0
        self.first_start_ns = self.last_complete_ns = 0
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                start = time.monotonic_ns()
                request = urllib.request.Request(
                    self.url, data=self.body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120.0) as response:
                    usage = json.loads(response.read()).get("usage", {})
                self.output_tokens += int(usage.get("completion_tokens", 0))
                self.completed += 1
                self.first_start_ns = self.first_start_ns or start
                self.last_complete_ns = time.monotonic_ns()
            except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
                self.stop_event.wait(0.1)


# --- PMU samples -------------------------------------------------------------

@dataclass
class Levels:
    """Foreground memory signals over valid windows inside active frames.

    ``stall``/``miss_rate``/``mpki`` are EWMA-smoothed (what the policy reads);
    ``raw_*`` are per-window values (what the paper's reference scales use).
    """
    samples: int = 0
    frame_samples: int = 0          # all sampler ticks inside frames, valid or not
    zero_count_samples: int = 0     # in-frame ticks invalid for zero cycles/instructions
    stall: dict[str, float] = field(default_factory=dict)
    miss_rate: dict[str, float] = field(default_factory=dict)
    mpki: dict[str, float] = field(default_factory=dict)
    raw_stall: dict[str, float] = field(default_factory=dict)
    raw_miss_rate: dict[str, float] = field(default_factory=dict)
    raw_mpki: dict[str, float] = field(default_factory=dict)


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]

    return {"p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90),
            "p95": pick(0.95), "p99": pick(0.99), "max": ordered[-1]}


def window_valid(row: dict[str, str]) -> bool:
    """Paper Eq. (1); samplers record the verdict in ``window_valid``."""
    if "window_valid" in row:
        return row["window_valid"] == "1"
    return (row["pmu_valid"] == "1" and row["pmu_counter_reset"] != "1"
            and int(row["interval_ns"]) > 0 and int(row["delta_cycles"]) > 0
            and int(row["delta_inst"]) > 0)


def frame_levels(sampler_csv: Path, frames: list[Frame]) -> Levels:
    intervals = sorted((frame.dequeue_ns, frame.complete_ns) for frame in frames)
    names = ("stall", "miss_rate", "mpki")
    smoothed: dict[str, float] = {}
    inside: dict[str, list[float]] = {name: [] for name in names}
    raw_inside: dict[str, list[float]] = {name: [] for name in names}
    cursor = 0
    frame_samples = 0
    zero_count_samples = 0
    with sampler_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["timestamp_ns"])
            while cursor < len(intervals) and intervals[cursor][1] < timestamp:
                cursor += 1
            in_frame = cursor < len(intervals) and intervals[cursor][0] <= timestamp
            frame_samples += in_frame
            if not window_valid(row):
                zero_count_samples += in_frame and row["pmu_valid"] == "1" and (
                    row["delta_cycles"] == "0" or row["delta_inst"] == "0")
                continue
            cycles = int(row["delta_cycles"])
            raw = {
                "stall": int(row["delta_stall"]) / cycles,
                "miss_rate": int(row["delta_llc"]) / (int(row["interval_ns"]) / 1e9),
                "mpki": 1000.0 * int(row["delta_llc"]) / int(row["delta_inst"]),
            }
            for name, value in raw.items():
                # Eq. (3), seeded by the first valid window (as the sampler)
                previous = smoothed.get(name)
                smoothed[name] = value if previous is None else (
                    EWMA_ALPHA * value + (1.0 - EWMA_ALPHA) * previous)
            if in_frame:
                for name in names:
                    inside[name].append(smoothed[name])
                    raw_inside[name].append(raw[name])
    return Levels(len(inside["stall"]), frame_samples, zero_count_samples,
                  percentiles(inside["stall"]), percentiles(inside["miss_rate"]),
                  percentiles(inside["mpki"]), percentiles(raw_inside["stall"]),
                  percentiles(raw_inside["miss_rate"]), percentiles(raw_inside["mpki"]))


# --- one measured phase ------------------------------------------------------

def stop_process(process: subprocess.Popen | None) -> None:
    if process is not None:
        terminate(process)


def pinned(command: list[str], cpus: str | None) -> list[str]:
    return ["taskset", "-c", cpus, *command] if cpus else command


def run_phase(args: argparse.Namespace, name: str, server_env: dict[str, str] | None,
              llm_load: bool, thresholds: dict[str, float],
              affinity: dict[str, str] | None = None) -> dict[str, Any]:
    """Run one measured phase.

    ``affinity`` optionally pins ``foreground``, ``server`` and ``tooling``
    (sampler, load generator and frame poller) to CPU lists.
    """
    if getattr(args, "device", "local") == "android":   # the phase runs on the phone
        from edgepace_android import run_phase_android
        return run_phase_android(args, name, server_env, llm_load, thresholds)
    affinity = affinity or {}
    original_affinity = os.sched_getaffinity(0)
    if affinity.get("tooling"):
        os.sched_setaffinity(0, parse_cpus(affinity["tooling"]))
    try:
        return run_phase_pinned(args, name, server_env, llm_load, thresholds, affinity)
    finally:
        os.sched_setaffinity(0, original_affinity)


def parse_cpus(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def run_phase_pinned(args: argparse.Namespace, name: str, server_env: dict[str, str] | None,
                     llm_load: bool, thresholds: dict[str, float],
                     affinity: dict[str, str]) -> dict[str, Any]:
    phase_dir = (args.output_dir / name).resolve()
    phase_dir.mkdir(parents=True, exist_ok=True)
    task_page = phase_dir / "task.page"
    request_page = phase_dir / "request.page"
    signal_page = phase_dir / "signal.page"
    sampler_csv = phase_dir / "pmu_signal.csv"
    trace = phase_dir / "edgepace_trace.csv"
    for path in (task_page, request_page, signal_page, sampler_csv, trace):
        path.unlink(missing_ok=True)

    print(f"[{name}] starting", file=sys.stderr, flush=True)
    page = RequestPage(request_page) if llm_load else None
    server = foreground = None
    poller = FramePoller(task_page)
    load = None
    bulk: list[BulkStream] = []
    logs = []
    try:
        if server_env is not None:
            environment = clean_environment()
            environment.update(server_env)
            environment.update({
                "EDGEPACE_TASK_PAGE": str(task_page),
                "EDGEPACE_REQUEST_PAGE": str(request_page),
                "EDGEPACE_SIGNAL_PAGE": str(signal_page),
            })
            if "EDGEPACE_POLICY" in server_env:
                environment["EDGEPACE_LOG_PATH"] = str(trace)
            logs.append((phase_dir / "server.log").open("w", encoding="utf-8"))
            server = subprocess.Popen(pinned(server_command(args), affinity.get("server")),
                                      stdout=logs[-1], stderr=subprocess.STDOUT,
                                      env=environment)
            wait_server(args.host, args.port, server, args.server_timeout)

        foreground_command = pinned(shlex.split(args.foreground.format(
            frames=args.frames, task_page=task_page, phase_dir=phase_dir)),
            affinity.get("foreground"))
        # The sampler launches the foreground: counters are attached
        # before exec, it exits with the foreground and returns its exit code.
        # The sampler sits on the tooling CPUs, the foreground on its own.
        # --no-sampler: the foreground runs alone (no PMU monitoring at
        # all), for baselines where monitoring must not add interference.
        if getattr(args, "no_sampler", False):
            command = foreground_command
        else:
            command = pinned([
                str(args.sampler), "--snapshot", str(signal_page), "--csv", str(sampler_csv),
                "--sample-interval-ms", "2", "--no-mc",
                "--stall-threshold", repr(thresholds["stall"]),
                "--miss-rate-threshold", repr(thresholds["miss_rate"]),
                "--mpki-threshold", repr(thresholds["mpki"]),
                *shlex.split((getattr(args, "sampler_extra", "") or "").format(phase_dir=phase_dir)),
                "--", *foreground_command,
            ], affinity.get("tooling"))
        (phase_dir / "foreground.command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8")
        logs.append((phase_dir / "foreground.log").open("w", encoding="utf-8"))
        foreground = subprocess.Popen(command, stdout=logs[-1], stderr=subprocess.STDOUT)
        poller.start()
        deadline = time.monotonic() + args.foreground_start_timeout
        while not poller.first_frame.is_set():
            if foreground.poll() is not None:
                raise RuntimeError(f"[{name}] foreground exited before its first frame "
                                   f"(code {foreground.returncode}); see {logs[-1].name}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"[{name}] no frame on {task_page}")
            time.sleep(0.05)

        if llm_load:
            payload = json.loads(args.request.read_text(encoding="utf-8"))
            load = LoadGenerator(args.host, args.port, payload, page,
                                 args.llm_period_ms, args.freshness_ms,
                                 task_page if getattr(args, "llm_align", "none") == "frame" else None)
            load.start()
            bulk = [BulkStream(args.host, args.port, payload)
                    for _ in range(getattr(args, "llm_extra_streams", 0))]
            for stream in bulk:
                stream.start()

        budget_s = args.frames * args.period_us / 1e6 + 120.0
        foreground.wait(timeout=budget_s)
    finally:
        for stream in bulk:
            stream.stop_event.set()
        if load is not None:
            load.stop_event.set()
            load.join(timeout=130.0)
        for stream in bulk:
            stream.join(timeout=130.0)
        poller.stop_event.set()
        if poller.is_alive():
            poller.join(timeout=2.0)
        stop_process(foreground)
        stop_process(server)
        if page is not None:
            page.close()
        for log in logs:
            log.close()

    if foreground.returncode != 0:
        raise RuntimeError(f"[{name}] foreground exited with code {foreground.returncode}")
    frames = [frame for frame_id, frame in sorted(poller.frames.items())
              if frame_id > args.skip_frames]
    if not frames:
        raise RuntimeError(f"[{name}] no completed frames recorded")
    return summarize_phase(args, name, phase_dir, frames, Path(str(signal_page) + ".status"),
                           sampler_csv, thresholds, load.records if load is not None else None,
                           bulk, trace, poller.overwrites)


def summarize_phase(args: argparse.Namespace, name: str, phase_dir: Path, frames: list[Frame],
                    status_path: Path, sampler_csv: Path, thresholds: dict[str, float],
                    records: list[RequestRecord] | None, bulk: list[Any], trace: Path,
                    ledger_overwrites: int) -> dict[str, Any]:
    """Phase result from its frames, sampler CSV/status, requests and trace
    (shared by local phases and device phases)."""
    status = {}
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            status[key] = value
    if status.get("pmu_open_failures", "0") != "0":
        raise RuntimeError(f"[{name}] sampler could not attach counters: {status}")
    no_sampler = getattr(args, "no_sampler", False)
    levels = Levels() if no_sampler else frame_levels(sampler_csv, frames)
    # Most in-frame ticks must still be valid windows (paper Eq. 1).
    coverage = levels.samples / levels.frame_samples if levels.frame_samples else 0.0
    if not no_sampler and coverage < args.min_pmu_coverage:
        raise RuntimeError(
            f"[{name}] only {100 * coverage:.1f}% of in-frame sampler ticks have valid "
            f"PMU data (sampler status: {status})")
    responses = [frame.response_ms for frame in frames]
    misses = sum(frame.missed for frame in frames)
    result: dict[str, Any] = {
        "frames": len(frames),
        "misses": misses,
        "miss_rate": misses / len(frames),
        "response_ms": percentiles(responses),
        "ledger_overwrites": ledger_overwrites,
        "levels": asdict(levels),
        "pmu_coverage": coverage,
        "zero_count_share": (levels.zero_count_samples / levels.frame_samples
                             if levels.frame_samples else 0.0),
        "sampler_status": status,
        "sampler_thresholds": thresholds,
    }
    if records is not None:
        ok = [record for record in records if 200 <= record.status < 300]
        duration_s = max(1e-9, (frames[-1].complete_ns - frames[0].release_ns) / 1e9)
        # active time = arrival to output end, summed over requests
        active_s = sum((record.complete_ns - record.start_ns) / 1e9 for record in ok)
        output_tokens = sum(record.completion_tokens for record in ok)
        result["llm"] = {
            "requests": len(records),
            "completed": len(ok),
            "fresh": sum(record.fresh for record in records),
            "completed_per_s": len(ok) / duration_s,
            "latency_ms": percentiles([record.latency_ms for record in ok]),
            "output_tokens": output_tokens,
            "active_s": active_s,
            "tokens_per_active_s": output_tokens / active_s if active_s > 0 else 0.0,
        }
        if bulk:
            extra = sum(stream.output_tokens for stream in bulk)
            # per back-to-back stream: completed tokens over first start .. last completion
            spans = [(output_tokens, ok[-1].complete_ns - ok[0].start_ns)] if ok else []
            spans += [(s.output_tokens, s.last_complete_ns - s.first_start_ns) for s in bulk]
            result["llm"].update({
                "extra_streams": len(bulk),
                "extra_completed": sum(stream.completed for stream in bulk),
                "extra_output_tokens": extra,
                # all streams, over the measured frames' span (coarse: whole requests)
                "total_tokens_per_s": (output_tokens + extra) / duration_s,
                "stream_tokens_per_s": sum(n / (span / 1e9) for n, span in spans if span > 0),
            })
        with (phase_dir / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["request_id", "start_ns", "complete_ns", "latency_ms", "status",
                             "fresh", "completion_tokens", "prompt_tokens",
                             "aligned_frame_complete_ns"])
            for record in records:
                writer.writerow([record.request_id, record.start_ns, record.complete_ns,
                                 f"{record.latency_ms:.3f}", record.status, int(record.fresh),
                                 record.completion_tokens, record.prompt_tokens,
                                 record.aligned_frame_complete_ns])
    if trace.is_file():
        events: dict[str, int] = {}
        with trace.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                events[row["event"]] = events.get(row["event"], 0) + 1
        result["trace_events"] = events
    (phase_dir / "phase.json").write_text(json.dumps(result, indent=2) + "\n",
                                          encoding="utf-8")
    llm = result.get("llm", {})
    print(f"[{name}] frames={len(frames)} miss={misses} ({100 * result['miss_rate']:.2f}%) "
          f"resp p95={result['response_ms'].get('p95', 0):.2f}ms "
          f"valid-window cov={100 * coverage:.1f}% zero-count={100 * result['zero_count_share']:.1f}% "
          f"stall p50/p90={levels.stall.get('p50', 0):.3f}/{levels.stall.get('p90', 0):.3f} "
          f"llm={llm.get('completed', '-')}", file=sys.stderr, flush=True)
    return result


# --- calibration ------------------------------------------------------------

def reference_scales(solo: dict[str, Any]) -> dict[str, Any]:
    """Paper §3.2: r0 and m0 are the larger of the baseline p99 and twice its p95.

    The baseline is the solo phase's per-window miss rate and MPKI over valid
    windows inside active frames.
    """
    result: dict[str, Any] = {"rule": "max(p99, 2 * p95) of solo in-frame valid windows"}
    for name, raw in (("miss_rate", "raw_miss_rate"), ("mpki", "raw_mpki")):
        stats = solo["levels"][raw]
        result[name] = max(stats["p99"], 2.0 * stats["p95"])
        result[f"{name}_inputs"] = {"p95": stats["p95"], "p99": stats["p99"]}
    return result


def stall_cutoff(solo: dict[str, Any], contended: dict[str, Any], x: float) -> float:
    """Raw stall cutoff R(x) = B_p50 + x * (C_p90 - B_p50) on the policy's EWMA."""
    base = solo["levels"]["stall"]["p50"]
    top = contended["levels"]["stall"]["p90"]
    # a signal that does not rise under contention cannot gate anything
    return math.inf if top <= base else base + x * (top - base)


def stall_separation(solo: dict[str, Any], contended: dict[str, Any]) -> dict[str, Any]:
    """Whether in-frame stall separates interference from solo variation.

    Interpolating cutoffs between B and C is only meaningful when the two
    distributions are distinct: the typical contended frame must sit above
    what the foreground alone already reaches in its upper tail.
    """
    solo_levels = solo["levels"]["stall"]
    contended_levels = contended["levels"]["stall"]
    return {
        "solo_p50": solo_levels["p50"],
        "solo_p90": solo_levels["p90"],
        "contended_p50": contended_levels["p50"],
        "contended_p90": contended_levels["p90"],
        "median_rise": (contended_levels["p50"] - solo_levels["p50"]) / solo_levels["p50"]
        if solo_levels["p50"] > 0 else math.inf,
        "discriminative": contended_levels["p50"] > solo_levels["p90"],
    }


def sampler_thresholds(scales: dict[str, Any], raw_stall_cutoff: float,
                       stall_gate: float) -> dict[str, float]:
    """Sampler normalization: demand by the paper's r0/m0, stall by the swept s0.

    The sampler computes D = clip(min(r / r0, m / m0)) and S = clip(s / s0), so
    the reference scales are passed through unchanged.
    """
    return {
        "stall": raw_stall_cutoff / stall_gate if math.isfinite(raw_stall_cutoff) else 1e18,
        "miss_rate": scales["miss_rate"],
        "mpki": scales["mpki"],
    }


def managed_environment(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, str]:
    namespace = SimpleNamespace(
        policy="adaptive_slo_online", policy_library=args.policy_library,
        priority=args.priority, priority_bias=args.priority_bias,
        max_tokens=args.max_tokens, period_us=args.period_us,
        deadline_us=args.deadline_us, freshness_ms=args.freshness_ms,
        task_page=None, request_page=None, signal_page=None, actmon_page=None,
        trace=None,
    )
    return runtime_environment(namespace, plan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True,
                        help="cut_plan.json from `edgepace_plan.py plan`")
    parser.add_argument("--foreground", required=True,
                        help="foreground command; placeholders {frames} {task_page} {phase_dir}")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--candidates", default="0.25,0.5,0.75,1.0",
                        help="positions x between solo (0) and contended (1)")
    parser.add_argument("--miss-budget-pp", type=float, default=1.0,
                        help="allowed deadline-miss increase over solo, percentage points")
    parser.add_argument("--min-pmu-coverage", type=float, default=0.90,
                        help="minimum share of in-frame sampler ticks with valid PMU data")
    parser.add_argument("--force-sweep", action="store_true",
                        help="sweep cutoffs even if the stall signal is not discriminative")
    parser.add_argument("--priority", choices=("miss", "fresh", "balanced", "manual"),
                        default="balanced")
    parser.add_argument("--priority-bias", type=float)
    parser.add_argument("--llm-period-ms", type=float, default=1600.0)
    parser.add_argument("--llm-align", choices=("none", "frame"), default="none",
                        help="frame: send each request right after a foreground frame completes")
    parser.add_argument("--contended-period-ms", type=float, default=0.0,
                        help="request period for the contended phase (0 = back-to-back)")
    parser.add_argument("--period-us", type=int, default=33333)
    parser.add_argument("--deadline-us", type=int, default=33333)
    parser.add_argument("--freshness-ms", type=float, default=1800.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--policy-library", type=Path, default=DEFAULT_POLICY_LIBRARY)
    parser.add_argument("--sampler", type=Path, default=DEFAULT_SAMPLER)
    parser.add_argument("--sampler-extra", default="",
                        help="extra sampler arguments placed before the launched command; "
                             "placeholder {phase_dir}")
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18432)
    parser.add_argument("--server-timeout", type=float, default=180.0)
    parser.add_argument("--foreground-start-timeout", type=float, default=120.0)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--prompt-cache", action="store_true")
    parser.add_argument("--required-power-mode", default="MAXN_SUPER")
    parser.add_argument("--allow-unlocked-profile", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    platform = require_locked_platform(args)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    stall_gate = float(resolved_priority(args.priority, args.priority_bias)["stall_safe_max"])
    candidates = sorted(float(value) for value in args.candidates.split(","))

    solo = run_phase(args, "solo", None, False, DEFAULT_THRESHOLDS)
    scales = reference_scales(solo)
    measured = {"stall": DEFAULT_THRESHOLDS["stall"],
                "miss_rate": scales["miss_rate"], "mpki": scales["mpki"]}
    contended_args = argparse.Namespace(**vars(args))
    contended_args.llm_period_ms = args.contended_period_ms
    contended = run_phase(contended_args, "contended", {}, True, measured)

    separation = stall_separation(solo, contended)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": str(args.model.resolve()),
        "plan": str(args.plan.resolve()),
        "foreground": args.foreground,
        "platform": platform,
        "priority": args.priority,
        "gates": {"stall": stall_gate, "demand": DEMAND_GATE},
        "reference_scales": scales,
        "miss_budget_pp": args.miss_budget_pp,
        "solo": solo,
        "contended": contended,
        "stall_separation": separation,
        "sweep": [],
    }

    if not separation["discriminative"] and not args.force_sweep:
        report["decision"] = {
            "status": "pmu_gate_ineffective",
            "reason": (
                f"contended in-frame stall median {separation['contended_p50']:.3f} does not "
                f"exceed solo p90 {separation['solo_p90']:.3f} (median rise "
                f"{100 * separation['median_rise']:.1f}%): the signal cannot tell "
                "interference from the foreground's own variation; gate on deadline "
                "laxity alone or fix PMU coverage of the foreground workers"),
        }
    else:
        environment = managed_environment(args, plan)
        for x in candidates:
            raw_stall = stall_cutoff(solo, contended, x)
            thresholds = sampler_thresholds(scales, raw_stall, stall_gate)
            result = run_phase(args, f"managed_x{x:g}", environment, True, thresholds)
            result["x"] = x
            result["raw_stall_cutoff"] = raw_stall
            result["excess_miss_pp"] = 100.0 * (result["miss_rate"] - solo["miss_rate"])
            result["within_budget"] = result["excess_miss_pp"] <= args.miss_budget_pp
            report["sweep"].append(result)

        feasible = [value for value in report["sweep"] if value["within_budget"]]
        if feasible:
            chosen = max(feasible, key=lambda value: (value["llm"]["completed_per_s"], value["x"]))
            report["decision"] = {
                "status": "calibrated",
                "x": chosen["x"],
                "raw_stall_cutoff": chosen["raw_stall_cutoff"],
                "sampler_args": [
                    "--stall-threshold", repr(chosen["sampler_thresholds"]["stall"]),
                    "--miss-rate-threshold", repr(chosen["sampler_thresholds"]["miss_rate"]),
                    "--mpki-threshold", repr(chosen["sampler_thresholds"]["mpki"]),
                ],
                "excess_miss_pp": chosen["excess_miss_pp"],
                "llm_completed_per_s": chosen["llm"]["completed_per_s"],
            }
        else:
            tightest = min(report["sweep"], key=lambda value: value["excess_miss_pp"])
            report["decision"] = {
                "status": "budget_unattainable",
                "reason": (f"lowest excess miss {tightest['excess_miss_pp']:.2f}pp at "
                           f"x={tightest['x']:g} exceeds {args.miss_budget_pp}pp; "
                           "add smaller candidates or relax the budget"),
            }

    (args.output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2), file=sys.stderr)
    return 0 if report["decision"]["status"] != "budget_unattainable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
