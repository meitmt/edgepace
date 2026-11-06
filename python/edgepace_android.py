"""Android device phases for the EdgePace tools.

With ``--device android`` the profile and evaluation tools run each phase on a
rooted phone over adb and compute everything else on the host with the same
code as on the Jetson (``edgepace_phase.summarize_phase``):

* one shell script per phase on the device: llama-server (geniex Hexagon
  backend + EdgePace ``libggml-base.so``), the PMU sampler launching the foreground,
  ``edgepace_client`` issuing the LLM load; it waits for the foreground and
  stops the client and the server;
* the phase's CSVs (prediction, PMU, requests, policy trace, sampler status)
  are copied back into the local phase directory;
* the platform lock is ``android/locked_run_android.sh``, taken on the device for
  each device run and released after it; between phases the phone cools down
  unlocked (``require_locked_platform``).

Device paths: ``--device-root`` (default /data/local/tmp/edgepace) holds bin/,
lib/ (``android/build_android.sh`` output) and runs/; ``--model`` and the
foreground command are device paths; ``--plan`` and ``--request`` are local and
are pushed.  Frame windows come from the sampler's prediction.csv (release to
completion) instead of the task-page poller.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from edgepace_fineplan import inspect_gguf, inspect_mmproj, make_manifest, profile_from_trace
from edgepace_phase import Frame, RequestRecord, summarize_phase
from edgepace_plan import ROOT, prefill_boundaries, profile_environment, server_command, vision_info

DEFAULT_ROOT = "/data/local/tmp/edgepace"
# Oryon: S from the all-cause dispatch stall share (stall_backend_mem does not follow
# CPU or NPU memory contention there), D from L2 (last-level CPU cache) misses
ANDROID_SAMPLER = ("--pmu-dir /sys/bus/event_source/devices/armv8_pmuv3 "
                   "--stall-event stall --llc-event l2d_cache_lmiss_rd")
PULLED = ("prediction.csv", "pmu_signal.csv", "signal.page.status", "requests.csv", "requests.csv.streams",
          "edgepace_trace.csv", "foreground.csv", "foreground.log", "server.log", "client.log", "phase.log",
          "canary_server.log")


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("local", "android"), default="local",
                        help="android: run each phase on a phone over adb")
    parser.add_argument("--adb", default=os.environ.get("EDGEPACE_ADB", "adb"), help="adb executable")
    parser.add_argument("--adb-serial", default=os.environ.get("ANDROID_SERIAL"))
    parser.add_argument("--device-root", default=DEFAULT_ROOT,
                        help="device directory with bin/ and lib/ of android/build_android.sh")


class Device:
    """Root shell on a phone through adb; stdin/stdout only, no host paths."""

    def __init__(self, adb: str, serial: str | None, root: str) -> None:
        self.base = [adb, *(["-s", serial] if serial else [])]
        self.root = root

    def run(self, script: str, timeout: float = 120.0, check: bool = True) -> str:
        """Runs a /system/bin/sh script as root (sent on stdin, no quoting)."""
        name = f"/data/local/tmp/edgepace_host_{uuid.uuid4().hex[:8]}.sh"
        self.write(name, script.encode())
        try:
            done = subprocess.run([*self.base, "shell", f"su -c 'sh {name}'"], capture_output=True,
                                  timeout=timeout)
        finally:
            subprocess.run([*self.base, "shell", f"su -c 'rm -f {name}'"], capture_output=True)
        output = done.stdout.decode(errors="replace").replace("\r\n", "\n")
        if check and done.returncode != 0:
            raise RuntimeError(f"device script failed ({done.returncode}): {output[-2000:]}"
                               f"{done.stderr.decode(errors='replace')[-500:]}")
        return output

    def write(self, path: str, data: bytes) -> None:
        subprocess.run([*self.base, "exec-in", f"su -c 'cat > {path}'"], input=data, check=True,
                       capture_output=True)

    def read(self, path: str) -> bytes | None:
        done = subprocess.run([*self.base, "exec-out", f"su -c 'cat {path} 2>/dev/null || echo __MISSING__'"],
                              capture_output=True)
        return None if done.stdout.endswith(b"__MISSING__\n") else done.stdout


_device: Device | None = None


def device(args: argparse.Namespace) -> Device:
    global _device
    if _device is None:
        _device = Device(args.adb, args.adb_serial, args.device_root)
    return _device


def prepare(args: argparse.Namespace) -> None:
    """Device paths for the server, sampler and policy; push the request body."""
    root = args.device_root
    args.server = Path(f"{root}/bin/llama-server")
    args.sampler = Path(f"{root}/bin/pmu_sampler")
    args.policy_library = Path(f"{root}/lib/libedgepace_policy.so")
    args.device_request = f"{root}/requests/{args.request.name}"
    d = device(args)
    d.run(f"mkdir -p {root}/requests {root}/runs")
    d.write(args.device_request, args.request.read_bytes())


def push_request(args: argparse.Namespace, request: Path) -> str:
    """Another request body to the device; its device path."""
    path = f"{args.device_root}/requests/{request.name}"
    device(args).write(path, request.read_bytes())
    return path


# a lock taken within about a minute of releasing the previous one puts
# the phone in its slow state (see SLOW_LOAD_RATIO) more often than not; with two
# minutes in between, 17 of 18 phases ran fast
MIN_UNLOCKED_S = 120.0
_last_unlock: float | None = None


def start_lock(d: Device, root: str, name: str) -> tuple[str, str]:
    """Starts android/locked_run_android.sh holding the lock on the device, at
    least MIN_UNLOCKED_S after the last release; returns its session directory and
    log up to LOCK_VERIFIED."""
    if _last_unlock is not None:
        time.sleep(max(0.0, MIN_UNLOCKED_S - (time.monotonic() - _last_unlock)))
    session = f"{root}/runs/lock_{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    lock = f"{root}/locked_run_android.sh"
    # the hold loop also ends by itself after 6 h, so a lost host cannot pin the clocks
    d.run(f"mkdir -p {session}; nohup sh {lock} {session} -- sh -c "
          f"'i=0; while [ ! -f {session}/release ] && [ $i -lt 21600 ]; do sleep 1; i=$((i+1)); done' "
          f"> {session}/locked.log 2>&1 < /dev/null &")
    log = ""
    for _ in range(100):
        log = (d.read(f"{session}/locked.log") or b"").decode()
        if "LOCK_VERIFIED=" in log:
            break
        time.sleep(0.2)
    return session, log


def stop_lock(d: Device, session: str) -> str:
    """Releases a start_lock session; returns its log with RESTORE_VERIFIED."""
    global _last_unlock
    d.run(f"touch {session}/release")
    for _ in range(100):
        text = (d.read(f"{session}/locked.log") or b"").decode()
        if "RESTORE_VERIFIED=" in text:
            _last_unlock = time.monotonic()
            return text
        time.sleep(0.2)
    raise RuntimeError(f"device lock {session} did not report its restore")


def check_lock(log: str, args: argparse.Namespace) -> None:
    if "LOCK_VERIFIED=1" not in log and not args.allow_unlocked_profile:
        raise RuntimeError(f"device lock failed: {log[-1000:]}")


def require_locked_platform(args: argparse.Namespace) -> dict[str, Any]:
    """Checks that the device locks, and records its locked clocks.

    The lock is then taken per device run (each phase, the plan profiling) and
    released after it: left locked while idle, or locked again soon after a
    release, the phone drops into a state in which memory copies (video
    decode output, model loading) and the NPU run at about half speed, invisible in
    the CPU, GPU and bus clocks it reports (see SLOW_LOAD_RATIO).  Between phases
    the phone cools down unlocked, at least MIN_UNLOCKED_S."""
    prepare(args)
    d = device(args)
    if getattr(args, "hold_lock", False):
        # one lock for the whole run (hold_lock): it is the check
        held = hold_lock(args)
        log = d.run(f"cat {held['lock_session']}/locked.log", check=False)
        clocks = next((l for l in log.splitlines() if l.startswith("LOCKED_CLOCKS")), "")
        return {"device": "android", "locked": "LOCK_VERIFIED=1" in log, "clocks": clocks,
                "power_mode": "fixed-performance", "lock": "held for the run", **held}
    cool_down(d)   # a hot phone fails the check (thermal caps on the CPU policies)
    session, log = start_lock(d, args.device_root, "check")
    log = stop_lock(d, session)
    check_lock(log, args)
    args.android_lock_per_run = True
    clocks = next((l for l in log.splitlines() if l.startswith("LOCKED_CLOCKS")), "")
    return {"device": "android", "lock_check_session": session, "locked": "LOCK_VERIFIED=1" in log,
            "clocks": clocks, "power_mode": "fixed-performance", "lock": "per run"}


def hold_lock(args: argparse.Namespace) -> dict[str, Any]:
    """One lock for the rest of the tool run, released at exit (the quick
    profile): its phases run back to back, so the phone neither idles under the lock
    nor is locked again soon after a release, and a minute of load stays well
    inside the ~4 min before thermal mitigation.  Starts once the phone is cool."""
    d = device(args)
    args.android_lock_per_run = False
    cool = cool_down(d)
    session, log = start_lock(d, args.device_root, "run")
    check_lock(log, args)
    args.android_lock_session = session

    def release() -> None:
        text = stop_lock(d, session)
        (args.output_dir / "device_lock.log").write_text(text)
        print(" ".join(l for l in text.splitlines()
                       if l.startswith(("LOCK_VERIFIED", "LOCK_CONTINUOUS", "RESTORE", "THERMAL"))),
              file=sys.stderr, flush=True)

    atexit.register(release)
    return {"lock_session": session, **cool}


def device_libraries(root: str, vision_device: str = "HTP0") -> str:
    """The runtime's library environment on the phone.  /vendor/lib64 (after
    /system/lib64, whose binder libraries the Hexagon RPC needs) for libOpenCL.so; the
    OpenCL shim loads geniex's Adreno backend; a VLM's vision encoder runs on
    vision_device (--vision-device)."""
    return (f"export LD_LIBRARY_PATH={root}/lib:/system/lib64:/vendor/lib64 "
            f"GGML_HEXAGON_IMPL={root}/lib/libggml-hexagon.so ADSP_LIBRARY_PATH={root}/lib "
            f"GGML_OPENCL_IMPL={root}/lib/libggml-opencl.so MTMD_BACKEND_DEVICE={vision_device}")


DEVICE_BINARIES = ("bin/llama-server", "lib/libedgepace_policy.so", "bin/pmu_sampler",
                   "lib/libggml-base.so", "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so", "bin/edgepace_client",
                   "lib/libggml-opencl.so", "bin/libggml-opencl-dl.so")


def device_fingerprint(d: Device, model: str, clocks: str = "") -> dict[str, Any]:
    """What a phone profile was measured with: the locked clocks, the
    sha256 of the runtime binaries on the device, the model file's size/mtime and
    the sampler's PMU events (the scales and thresholds belong to them)."""
    sums = {}
    for line in d.run(f"cd {d.root} && sha256sum {' '.join(DEVICE_BINARIES)} 2>/dev/null", check=False).splitlines():
        digest, _, name = line.partition("  ")
        if name:
            sums[name.strip()] = digest
    stat = d.run(f"stat -c '%s %Y' {model} 2>/dev/null", check=False).split()
    return {"device": "android", "locked_clocks": clocks, "binaries_sha256": sums, "sampler_pmu": ANDROID_SAMPLER,
            "model": {"path": model, "bytes": int(stat[0]) if stat else None,
                      "mtime_s": int(stat[1]) if len(stat) > 1 else None}}


def device_profile_fields(args: argparse.Namespace, platform: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Workload fields and fingerprint of a phone profile directory; ({}, None)
    for a local one."""
    if getattr(args, "device", "local") != "android":
        return {}, None
    workload = {"device": "android", "device_root": args.device_root, "adb": args.adb,
                "adb_serial": args.adb_serial}
    return workload, device_fingerprint(device(args), str(args.model), platform.get("clocks", ""))


def device_gguf_head(args: argparse.Namespace, path: str) -> tuple[Path, int, int]:
    """The first 32 MiB of a device GGUF copied to the host (metadata only is parsed),
    with the device file's size and mtime."""
    import hashlib
    d = device(args)
    size, mtime = (int(x) for x in d.run(f"stat -c '%s %Y' {path}").split()[:2])
    cache = ROOT / "build/android-models" / f"{hashlib.sha256(path.encode()).hexdigest()[:16]}.gguf-head"
    stamp = cache.with_suffix(".stat")
    if not cache.is_file() or not stamp.is_file() or stamp.read_text() != f"{size} {mtime}":
        cache.parent.mkdir(parents=True, exist_ok=True)
        head = subprocess.run([*d.base, "exec-out", f"su -c 'dd if={path} bs=1048576 count=32 2>/dev/null'"],
                              capture_output=True, check=True).stdout
        cache.write_bytes(head)
        stamp.write_text(f"{size} {mtime}")
    return cache, size, mtime


def model_info(args: argparse.Namespace) -> Any:
    """GGUF metadata of the device model; path, size and mtime are the device file's,
    so the cut-plan cache key names the model on the phone."""
    import dataclasses
    model = str(args.model)
    cache, size, mtime = device_gguf_head(args, model)
    return dataclasses.replace(inspect_gguf(cache, size=size), path=model, size=size, mtime_ns=mtime * 10**9)


def mmproj_info(args: argparse.Namespace, boundaries: tuple[str, ...]) -> Any:
    """inspect_mmproj of the device --mmproj, as model_info."""
    import dataclasses
    mmproj = str(args.mmproj)
    cache, size, mtime = device_gguf_head(args, mmproj)
    return dataclasses.replace(inspect_mmproj(cache, boundaries, size=size), path=mmproj, size=size,
                               mtime_ns=mtime * 10**9)


def admit_start_trace(trace: Path, out: Path) -> Path:
    """The Hexagon backend computes a quantum synchronously inside
    graph_compute, so the trace's ``submit`` (recorded after that call returns)
    is already past the work and complete - submit is ~20 us.  With
    profile_static (one quantum outstanding) the quantum starts at its
    ``admit``: rewrite each completion's submit_ns to that admit time.  CUDA
    launches asynchronously; Jetson plans keep submit -> complete."""
    with trace.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), reader.fieldnames or []
    admitted: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["graph_uid"], row["quantum_id"])
        if row["event"] == "admit":
            admitted[key] = int(row["timestamp_ns"])
        elif row["event"] == "complete" and key in admitted:
            row["submit_ns"] = str(admitted.pop(key))
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def profile_plan_android(args: argparse.Namespace, info: Any, output_dir: Path) -> dict[str, Any]:
    """edgepace_plan.profile_plan on the phone: profile_static server, three
    profile requests by edgepace_client, the trace copied back."""
    platform = require_locked_platform(args)
    d = device(args)
    run = f"{args.device_root}/runs/plan_{time.strftime('%Y%m%d_%H%M%S')}"
    trace = f"{run}/cut_profile_trace.csv"
    env = profile_environment(args, info.layer_count, Path(trace))
    command = server_command(args)
    (output_dir / "cut_profile_server.command.json").write_text(
        json.dumps({"device": "android", "command": command, "environment": env}, indent=2) + "\n")
    script = "\n".join([
        f"RUN={run}", "mkdir -p $RUN",
        device_libraries(args.device_root, getattr(args, "vision_device", "HTP0")),
        "(", *(f"export {k}={shlex.quote(str(v))}" for k, v in env.items()),
        f"exec {shlex.join(command)} > $RUN/server.log 2>&1", ") &", "server=$!",
        "i=0; while [ $i -lt 900 ] && ! grep -q 'server is listening' $RUN/server.log; do "
        "kill -0 $server 2>/dev/null || break; sleep 0.2; i=$((i+1)); done",
        "grep -q 'server is listening' $RUN/server.log || { echo SERVER_FAILED; tail -20 $RUN/server.log; exit 3; }",
        f"{args.device_root}/bin/edgepace_client --port {args.port} --body {args.device_request} --count 3 "
        "--period-ms 0 --freshness-ms 60000 --request-page $RUN/request.page --csv $RUN/requests.csv",
        "kill $server; wait $server 2>/dev/null", "echo PLAN_DONE", ""])
    session, log = start_lock(d, args.device_root, "plan")
    try:
        check_lock(log, args)
        output = d.run(script, timeout=900, check=False)
    finally:
        (output_dir / "device_lock.log").write_text(stop_lock(d, session))
    for f in ("cut_profile_trace.csv", "server.log", "requests.csv"):
        data = d.read(f"{run}/{f}")
        if data is not None:
            (output_dir / (f if f.startswith("cut_") else f"cut_profile_{f}")).write_bytes(data)
    if "PLAN_DONE" not in output:
        raise RuntimeError(f"device plan profiling failed: {output[-1500:]}")
    timed = admit_start_trace(output_dir / "cut_profile_trace.csv", output_dir / "cut_profile_trace.admit_start.csv")
    vision, prefill = vision_info(args), prefill_boundaries(args)   # as edgepace_plan.profile_plan
    manifest = make_manifest(info, profile_from_trace(timed, info.layer_count, vision, prefill),
                             args.context_size, args.batch_size, args.ubatch_size, vision, prefill)
    manifest["profile_backend"] = {
        "runtime": "EdgePace (android overlay) on geniex llama.cpp, Hexagon HTP",
        "server": str(args.server), "policy_library": str(args.policy_library), "policy": "profile_static",
        "requests": 3, "request_payload": str(args.request.resolve()), "platform": platform,
    }
    return manifest


def _expand(command: str, values: dict[str, Any]) -> str:
    return command.format(**values)


# under the lock the thermal engine caps the CPUs (and the GPU, which then
# stays capped for many minutes) after a few minutes of load, and its status is
# back at 0 well before the phone has cooled.  Each phase therefore starts, unlocked,
# once the shell (skin) and CPU sensors are near idle temperature (idle, unlocked:
# ~31 / ~34 C); its lock is then taken and the CPU clocks checked over the phase.
# The GPU carries no load in the phone runs, so its clock is recorded only.
COOL_SHELL_C = 35.0
COOL_CPU_C = 50.0
TEMPS = ("for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/type); case $t in "
         "shell_*|cpu-*) echo $t $(cat $z/temp);; esac; done")


def locked_cpu_clocks(session: str, d: Device) -> list[str]:
    log = d.run(f"cat {session}/locked.log", check=False)
    line = next((l for l in log.splitlines() if l.startswith("LOCKED_CLOCKS")), "")
    return re.findall(r"policy\d+=(\d+)", line)


def temperatures(d: Device) -> tuple[float, float]:
    """Hottest shell and CPU sensor, deg C."""
    shell = cpu = 0.0
    for line in d.run(TEMPS, check=False).splitlines():
        kind, _, value = line.partition(" ")
        if value.strip().lstrip("-").isdigit():
            celsius = int(value) / 1000.0
            if kind.startswith("shell_"):
                shell = max(shell, celsius)
            else:
                cpu = max(cpu, celsius)
    return shell, cpu


def cool_down(d: Device, timeout_s: float = 1800.0) -> dict[str, float]:
    """Wait (unlocked) until the phone is cool."""
    start = time.monotonic()
    while True:
        shell, cpu = temperatures(d)
        if shell <= COOL_SHELL_C and cpu <= COOL_CPU_C:
            return {"cooldown_s": round(time.monotonic() - start, 1), "shell_c": shell, "cpu_c": cpu}
        if time.monotonic() - start > timeout_s:
            raise RuntimeError(f"phone not cool within {timeout_s:.0f} s: shell {shell} C, cpu {cpu} C")
        time.sleep(10)


def lock_during(d: Device, session: str, t0: int, t1: int) -> dict[str, int]:
    """The lock's once-a-second samples (device wall clock: ts state gpu cpu...)
    within [t0, t1], with the seconds the CPU clocks and the GPU clock were off it."""
    cpus = locked_cpu_clocks(session, d)
    samples = [line.split() for line in d.run(f"cat {session}/platform_samples.txt", check=False).splitlines()]
    inside = [s for s in samples if len(s) > 3 and s[0].isdigit() and t0 <= int(s[0]) <= t1]
    gpu = inside[0][2] if inside else ""
    return {"samples": len(inside), "cpu_off_lock": sum(s[3:] != cpus for s in inside),
            "off_lock": sum(s[1] != "locked" for s in inside), "gpu_hz_at_start": gpu}


# the phone has a slow state in which memory copies (video decode
# output, model loading) and the NPU take about twice as long, invisible in its CPU,
# GPU, bus clocks and thermal status.  An idle stretch under the lock puts it there,
# and so does a lock taken soon after the previous one was released (MIN_UNLOCKED_S);
# it then holds for that lock.  Model loading shows it (1.3 vs 2.7 s for Qwen2.5-1.5B
# F16 on the OnePlus 13).  Warm-up phases run before the first measured one until
# the load time is steady and fast, and a measured phase that loads SLOW_LOAD_RATIO
# times slower than the fastest load seen is run again.  A phase without the LLM
# (solo, stress) loads the model once inside its lock before the foreground starts,
# as a canary of that lock's state.
WARMUP_FRAMES = 900
WARMUP_MIN = 2
WARMUP_MAX = 6
SLOW_LOAD_RATIO = 1.4
PHASE_RETRIES = 2
# per process, not per argparse copy (the profile tool runs phases on copies)
_state: dict[str, Any] = {"warm": False, "fastest_load": None}


def _slow(load: float | None) -> bool:
    fastest = _state["fastest_load"]
    return load is not None and fastest is not None and load > SLOW_LOAD_RATIO * fastest


def _note_load(load: float | None) -> None:
    if load is not None:
        _state["fastest_load"] = min(load, _state["fastest_load"] or load)


def run_phase_android(args: argparse.Namespace, name: str, server_env: dict[str, str] | None,
                      llm_load: bool, thresholds: dict[str, float]) -> dict[str, Any]:
    if not getattr(args, "android_lock_per_run", False):
        return run_phase_android_once(args, name, server_env, llm_load, thresholds)
    if not _state["warm"]:
        warm_up(args, thresholds)
    for attempt in range(PHASE_RETRIES + 1):
        result = run_phase_android_once(args, name, server_env, llm_load, thresholds)
        lock = result["device_lock"]
        load = lock.get("model_load_s", lock.get("canary_load_s"))
        slow = _slow(load)
        _note_load(load)
        if not slow or attempt == PHASE_RETRIES:
            break
        print(f"[{name}] model load {load:.2f} s vs fastest {_state['fastest_load']:.2f} s: phone in its "
              "slow state, running the phase again", file=sys.stderr, flush=True)
    result["device_lock"].update({"attempts": attempt + 1, "slow_state": slow, "fastest_load_s": _state["fastest_load"]})
    return result


def warm_up(args: argparse.Namespace, thresholds: dict[str, float]) -> None:
    """Unmanaged phases with the LLM load until model loading is steady and fast:
    at least WARMUP_MIN, the last two within 10% and not slow against the fastest."""
    _state["warm"] = True
    loads = []
    copy = argparse.Namespace(**vars(args))
    copy.frames = min(args.frames, WARMUP_FRAMES)
    for k in range(WARMUP_MAX):
        result = run_phase_android_once(copy, f"warmup_{k + 1}", {}, True, thresholds)
        loads.append(result["device_lock"].get("model_load_s") or 0.0)
        _note_load(loads[-1] or None)
        print(f"[warmup_{k + 1}] model load {loads[-1]:.2f} s", file=sys.stderr, flush=True)
        steady = (len(loads) >= WARMUP_MIN and abs(loads[-1] - loads[-2]) <= 0.1 * loads[-2]
                  and not _slow(loads[-1]))
        if steady:
            break


def run_phase_android_once(args: argparse.Namespace, name: str, server_env: dict[str, str] | None,
                           llm_load: bool, thresholds: dict[str, float]) -> dict[str, Any]:
    d = device(args)
    per_run = getattr(args, "android_lock_per_run", False)
    lock = cool_down(d) if per_run else {}
    root = args.device_root
    local = (args.output_dir / name).resolve()
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    run = f"{root}/runs/{args.output_dir.resolve().parent.name}_{args.output_dir.name}/{name}"
    pages = {k: f"{run}/{k}.page" for k in ("task", "request", "signal")}
    port = args.port
    lines = ["set -u", f"RUN={run}", "mkdir -p $RUN; rm -rf $RUN/*", "echo PHASE_T0=$(date +%s)",
             device_libraries(root, getattr(args, "vision_device", "HTP0"))]
    if server_env is not None:
        env = {**server_env, "EDGEPACE_TASK_PAGE": pages["task"], "EDGEPACE_REQUEST_PAGE": pages["request"],
               "EDGEPACE_SIGNAL_PAGE": pages["signal"]}
        if "EDGEPACE_POLICY" in server_env:
            env["EDGEPACE_LOG_PATH"] = f"{run}/edgepace_trace.csv"
        command = server_command(args)
        lines += ["(", *(f"export {k}={shlex.quote(str(v))}" for k, v in env.items()),
                  f"exec {shlex.join(command)} > $RUN/server.log 2>&1", ") &", "server=$!",
                  "i=0; while [ $i -lt 900 ] && ! grep -q 'server is listening' $RUN/server.log; do "
                  "kill -0 $server 2>/dev/null || break; sleep 0.2; i=$((i+1)); done",
                  "grep -q 'server is listening' $RUN/server.log || { echo SERVER_FAILED; tail -20 $RUN/server.log; exit 3; }"]
    elif per_run:
        # the load canary of this lock: the model loaded once, then stopped
        lines += [f"{shlex.join(server_command(args))} > $RUN/canary_server.log 2>&1 &", "canary=$!",
                  "i=0; while [ $i -lt 600 ] && ! grep -q 'model loaded' $RUN/canary_server.log; do "
                  "kill -0 $canary 2>/dev/null || break; sleep 0.1; i=$((i+1)); done",
                  "kill $canary 2>/dev/null; wait $canary 2>/dev/null; sleep 1"]
    foreground = _expand(args.foreground, {"frames": args.frames, "task_page": pages["task"], "phase_dir": run})
    sampler = (f"{args.sampler} --snapshot {pages['signal']} --csv $RUN/pmu_signal.csv "
               f"--sample-interval-ms 2 --no-mc --stall-threshold {thresholds['stall']!r} "
               f"--miss-rate-threshold {thresholds['miss_rate']!r} --mpki-threshold {thresholds['mpki']!r} "
               f"{ANDROID_SAMPLER} {(args.sampler_extra or '').format(phase_dir=run)} -- {foreground}")
    lines += [f"{sampler} > $RUN/foreground.log 2>&1 &", "fg=$!",
              f"i=0; while [ $i -lt 2400 ] && [ ! -s {pages['task']} ]; do kill -0 $fg 2>/dev/null || break; "
              "sleep 0.05; i=$((i+1)); done"]
    if llm_load:
        align = f"--align-task-page {pages['task']}" if getattr(args, "llm_align", "none") == "frame" else ""
        lines += [f"{root}/bin/edgepace_client --port {port} --body {args.device_request} "
                  f"--request-page {pages['request']} {align} --period-ms {args.llm_period_ms} "
                  f"--freshness-ms {args.freshness_ms} --extra-streams {getattr(args, 'llm_extra_streams', 0)} "
                  "--stop-file $RUN/stop --csv $RUN/requests.csv > $RUN/client.log 2>&1 &", "client=$!"]
    lines += ["wait $fg; echo $? > $RUN/foreground.exit", "touch $RUN/stop"]
    if llm_load:
        lines += ["wait $client"]
    if server_env is not None:
        lines += ["kill $server 2>/dev/null; wait $server 2>/dev/null"]
    lines += ["echo PHASE_T1=$(date +%s)", "echo PHASE_DONE"]
    (local / "phase.sh").write_text("\n".join(lines) + "\n")
    print(f"[{name}] starting on {d.base[-1] if len(d.base) > 1 else 'device'}", file=sys.stderr, flush=True)
    budget = args.frames * args.period_us / 1e6 + 400.0
    session = None
    if per_run:
        session, log = start_lock(d, root, name)
        check_lock(log, args)
    try:
        output = d.run("\n".join(lines) + "\n", timeout=budget, check=False)
    finally:
        if session:
            text = stop_lock(d, session)
            (local / "device_lock.log").write_text(text)
            print(f"[{name}] " + " ".join(l for l in text.splitlines()
                                          if l.startswith(("LOCK_VERIFIED", "LOCK_CONTINUOUS", "RESTORE", "THERMAL"))),
                  file=sys.stderr, flush=True)
            for f in ("platform_samples.txt", "platform_bus.txt"):
                data = d.read(f"{session}/{f}")
                if data is not None:
                    (local / f"device_lock_{f}").write_bytes(data)
    (local / "phase.log").write_text(output)
    for f in PULLED:
        data = d.read(f"{run}/{f}")
        if data is not None:
            (local / f).write_bytes(data)
    if "PHASE_DONE" not in output:
        raise RuntimeError(f"[{name}] device phase failed: {output[-1500:]}")
    code = (d.read(f"{run}/foreground.exit") or b"1").decode().strip()
    if code != "0":
        raise RuntimeError(f"[{name}] foreground exited with code {code}; see {local / 'foreground.log'}")
    marks = dict(line.split("=", 1) for line in output.splitlines() if line.startswith(("PHASE_T0=", "PHASE_T1=")))
    session = session or getattr(args, "android_lock_session", None)   # held for the run (hold_lock)
    if session and len(marks) == 2:
        lock.update(lock_during(d, session, int(marks["PHASE_T0"]), int(marks["PHASE_T1"])))
        if lock["cpu_off_lock"]:
            print(f"[{name}] WARNING CPU clocks off the lock for {lock['cpu_off_lock']}/{lock['samples']} s "
                  "(thermal mitigation)", file=sys.stderr, flush=True)

    frames = []
    with (local / "prediction.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame_id"]) > args.skip_frames and row["complete_ns"]:
                release = int(row["release_ns"])
                frames.append(Frame(int(row["frame_id"]), release, release,
                                    int(row["complete_ns"]), int(row["deadline_ns"])))
    if not frames:
        raise RuntimeError(f"[{name}] no completed frames recorded")
    records = bulk = None
    if llm_load and (local / "requests.csv").is_file():
        records = [RequestRecord(int(r["request_id"]), int(r["start_ns"]), float(r["latency_ms"]),
                                 int(r["status"]), r["fresh"] == "1", int(r["complete_ns"]),
                                 int(r["completion_tokens"]), int(r["prompt_tokens"]),
                                 int(r["aligned_frame_complete_ns"]))
                   for r in csv.DictReader((local / "requests.csv").open(newline=""))]
        streams = local / "requests.csv.streams"
        bulk = [SimpleNamespace(completed=int(r["completed"]), output_tokens=int(r["output_tokens"]),
                                first_start_ns=int(r["first_start_ns"]), last_complete_ns=int(r["last_complete_ns"]))
                for r in csv.DictReader(streams.open(newline=""))] if streams.is_file() else []
    result = summarize_phase(args, name, local, frames, local / "signal.page.status", local / "pmu_signal.csv",
                             thresholds, records, bulk or [], local / "edgepace_trace.csv", 0)
    load = model_load_s(local / "server.log")
    if load is not None:
        # the half-speed state after an idle lock doubles it (1.3 -> 2.7 s on
        # the OnePlus 13 with Qwen2.5-1.5B F16); compare it across phases
        lock["model_load_s"] = load
    canary = model_load_s(local / "canary_server.log")
    if canary is not None:
        lock["canary_load_s"] = canary
    result["device_lock"] = lock
    return result


def model_load_s(server_log: Path) -> float | None:
    """Seconds from llama-server's 'loading model' to 'model loaded' (log timestamps
    m.ss.mmm.uuu)."""
    if not server_log.is_file():
        return None
    stamps = {}
    for line in server_log.read_text(errors="replace").splitlines():
        for key in ("llama_server: loading model", "llama_server: model loaded"):
            if key in line and key not in stamps:
                minutes, seconds, millis, _ = (line.split(" ", 1)[0].split(".") + ["0"] * 4)[:4]
                stamps[key] = int(minutes) * 60 + int(seconds) + int(millis) / 1000
    if len(stamps) < 2:
        return None
    return round(stamps["llama_server: model loaded"] - stamps["llama_server: loading model"], 3)
