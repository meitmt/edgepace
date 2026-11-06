#!/usr/bin/env python3
"""Offline parameter profile for EdgePace.

The policy's parameters depend on the platform, the foreground and the LLM
load: the §3.2 reference scales and the foreground's own quiet D/S levels
differ per foreground, and how early to hold (the warning horizon H) depends
on how the LLM load interferes.  This tool measures them on the target device
with the target workload, before deployment, and writes ``profile.json``:

1. solo     the foreground alone, sampled exactly as at run time
            (``--sample-interval-ms``, predictor, gate rule)
            -> r0, m0, s0 = max(p99, 2 * p95) of the per-window LLC miss rate,
               MPKI and stall share inside the pipeline (paper §3.2), and the
               medians D0, S0 of the indicators the sampler then publishes
Quick mode (default): only 1. and the unmanaged phase of 4., ``--quick-frames``
each, back to back (one lock on a phone); wS from the cross comparison (the unmanaged
frames the LLM slowed vs solo, stall AUC), H = ``--horizon-ms``.  ``--mode sweep`` runs
all steps.

2. stress   (``--stress-binary``) the foreground beside a GPU memory stressor, or
            (``--stress-llm-request``, the phone) beside back-to-back LLM decode streams
            -> AUC of the frame stall share vs solo; below ``--stall-auc-min``
               S does not see memory contention for this foreground and the
               stall weight is set to 0 (D is reported, its weight is kept: it
               describes the foreground's demand, not contention)
3. thresholds  theta = theta_U + wD * D0 + wS * S0: theta_U are the thresholds
            on wU * U alone (default 0.2226 / 0.5226, i.e. Admit below a laxity
            of 0.68 H and Hold below 0.25 H with wU 0.70), shifted by what D and
            S contribute when the foreground runs undisturbed
4. sweep    the target LLM load unmanaged, then managed with each candidate H
            -> by ``--objective``: ``miss`` (default) the fewest deadline misses
               among candidates keeping at least ``--min-retention`` of the
               unmanaged LLM throughput, within ``--miss-tolerance`` the most
               throughput; ``throughput`` the most throughput among
               candidates within ``--miss-budget``.  ``--reselect
               profile_report.json`` re-picks from a finished sweep without runs.
               A pick at the end of the grid is flagged (``profiled_at_grid_edge``)

Run it under ``scripts/locked_run.sh``; the result is used with
``edgepace_eval.py --profile profile.json --conditions edgepace_profile``.

Example::

    scripts/locked_run.sh runs/profile_x/locked -- python3 python/edgepace_profile.py \\
        --model <model.gguf> --plan <cut_plan.json> --output-dir runs/profile_x/out \\
        --foreground '<fg> --frames {frames} --task-page {task_page} \\
            --progress-page {phase_dir}/progress.page --output {phase_dir}/foreground.csv' \\
        --stress-binary <gpu_stress>
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from edgepace_phase import DEFAULT_THRESHOLDS, managed_environment, run_phase
from edgepace_plan import require_locked_platform
from edgepace_eval import PREDICTOR_ARGS, add_run_arguments
from edgepace_android import device_profile_fields, push_request
from edgepace_cli import update_profile_dir, write_profile_dir

SCHEMA_VERSION = 1
EWMA_ALPHA = 0.2          # pmu_sampler's smoothing of D and S (paper Eq. 3)


# --- measurements -------------------------------------------------------------

def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def auc(negative: list[float], positive: list[float]) -> float:
    """P(positive > negative), ties counted half (Mann-Whitney)."""
    labelled = sorted([(v, 0) for v in negative] + [(v, 1) for v in positive])
    rank_sum, i = 0.0, 0
    while i < len(labelled):
        j = i
        while j + 1 < len(labelled) and labelled[j + 1][0] == labelled[i][0]:
            j += 1
        rank = (i + j) / 2 + 1
        rank_sum += rank * sum(label for _, label in labelled[i:j + 1])
        i = j + 1
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def read_windows(phase_dir: Path, skip: int) -> tuple[list[dict], list[list[dict]]]:
    """Valid sampler windows in time order, flagged when inside a frame's
    pipeline, and the pipeline windows grouped per frame."""
    frames = sorted((int(r["pipeline_ns"]), int(r["complete_ns"]))
                    for r in csv.DictReader((phase_dir / "prediction.csv").open())
                    if int(r["frame_id"]) > skip)
    starts = [lo for lo, _ in frames]
    windows, per_frame = [], [[] for _ in frames]
    for row in csv.DictReader((phase_dir / "pmu_signal.csv").open()):
        if row.get("window_valid", "1") != "1":
            continue
        end, interval = int(row["timestamp_ns"]), int(row["interval_ns"])
        w = {"interval_ns": interval, "cycles": float(row["delta_cycles"]),
             "inst": float(row["delta_inst"]), "stall": float(row["delta_stall"]),
             "llc": float(row["delta_llc"])}
        i = bisect.bisect_right(starts, end - interval) - 1
        w["pipeline"] = i >= 0 and end <= frames[i][1]
        if w["pipeline"]:
            per_frame[i].append(w)
        windows.append(w)
    return windows, [f for f in per_frame if f]


def raw_signals(w: dict) -> tuple[float, float, float]:
    """Paper Eq. 1: LLC misses per s, MPKI, memory-stall share."""
    return (w["llc"] / (w["interval_ns"] / 1e9), 1000.0 * w["llc"] / w["inst"], w["stall"] / w["cycles"])


def reference_scales(windows: list[dict]) -> dict[str, Any]:
    """Paper §3.2: max(p99, 2 * p95) of the solo pipeline windows (also for s0)."""
    columns = list(zip(*(raw_signals(w) for w in windows if w["pipeline"])))
    out: dict[str, Any] = {"rule": "max(p99, 2 * p95) of solo pipeline windows"}
    for name, values in zip(("miss_rate", "mpki", "stall"), columns):
        out[name] = max(quantile(values, 0.99), 2.0 * quantile(values, 0.95))
        out[f"{name}_p50"] = quantile(values, 0.50)
    return out


def published_levels(windows: list[dict], scales: dict[str, Any]) -> dict[str, float]:
    """D and S as the sampler would publish them with these scales (clip, then
    EWMA over all valid windows), at the windows inside a pipeline."""
    clip = lambda x: min(1.0, max(0.0, x))
    d_bar = s_bar = None
    d_in, s_in = [], []
    for w in windows:
        rate, mpki, stall = raw_signals(w)
        d = clip(min(rate / scales["miss_rate"], mpki / scales["mpki"]))
        s = clip(stall / scales["stall"])
        d_bar = d if d_bar is None else EWMA_ALPHA * d + (1 - EWMA_ALPHA) * d_bar
        s_bar = s if s_bar is None else EWMA_ALPHA * s + (1 - EWMA_ALPHA) * s_bar
        if w["pipeline"]:
            d_in.append(d_bar); s_in.append(s_bar)
    return {"demand_p50": quantile(d_in, 0.5), "demand_p90": quantile(d_in, 0.9),
            "stall_p50": quantile(s_in, 0.5), "stall_p90": quantile(s_in, 0.9)}


def frame_signals(per_frame: list[list[dict]]) -> dict[str, list[float]]:
    """Per-frame stall share and MPKI over the pipeline windows."""
    stall = [sum(w["stall"] for w in f) / sum(w["cycles"] for w in f) for f in per_frame]
    mpki = [1000.0 * sum(w["llc"] for w in f) / sum(w["inst"] for w in f) for f in per_frame]
    return {"stall": stall, "mpki": mpki}


def pipeline_ms(phase_dir: Path, skip: int) -> list[float]:
    """Per-frame pipeline time (pipeline start to completion), sorted."""
    return sorted((int(r["complete_ns"]) - int(r["pipeline_ns"])) / 1e6
                  for r in csv.DictReader((phase_dir / "prediction.csv").open())
                  if int(r["frame_id"]) > skip and r["complete_ns"])


# experimental: H from the slowdown the LLM causes, fitted on the four
# sweeps (H = slowdown p99 + 10 ms, within 16..24 ms); to be checked on held-out loads
RULE_B = (10.0, 16.0, 24.0)


def cross_compare(solo_dir: Path, unmanaged_dir: Path, skip: int,
                  solo_frames: list[list[dict]]) -> dict[str, Any]:
    """The foreground alone vs beside the unmanaged LLM load.

    slowed frames  unmanaged frames whose pipeline window span exceeds the solo p99;
                   their frame stall AUC against solo says whether S sees the LLM's
                   contention (it replaces a stressor run)
    excess_ms      unmanaged minus solo pipeline-time quantiles: how much the LLM
                   slows the foreground
    """
    _, unmanaged_frames = read_windows(unmanaged_dir.resolve(), skip)
    span = lambda f: sum(w["interval_ns"] for w in f)
    cut = quantile([span(f) for f in solo_frames], 0.99)
    slowed = [f for f in unmanaged_frames if span(f) > cut]
    base = frame_signals(solo_frames)
    stall_auc = auc(base["stall"], frame_signals(slowed)["stall"]) if slowed else 0.5
    solo_ms, unmanaged_ms = pipeline_ms(solo_dir, skip), pipeline_ms(unmanaged_dir, skip)
    excess = {f"p{int(100 * p)}": quantile(unmanaged_ms, p) - quantile(solo_ms, p) for p in (0.5, 0.9, 0.95, 0.99)}
    margin, low, high = RULE_B
    return {"frames": len(unmanaged_frames), "slowed_frames": len(slowed), "stall_auc_slowed": stall_auc,
            "excess_ms": excess, "h_rule_b": float(min(high, max(low, round(excess["p99"] + margin)))),
            "interference_k": interference_k(solo_dir, unmanaged_dir, skip)}


def interference_k(solo_dir: Path, unmanaged_dir: Path, skip: int) -> float:
    """how much longer the foreground's pipeline runs while an LLM request is in
    flight: per unmanaged frame, the pipeline time over the solo mean against the time a
    request overlapped the pipeline, least squares through the origin, within [0, 1].
    The policy's laxity then counts (1 + k) times the remaining foreground work."""
    def rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    frames = lambda d: [(int(r["pipeline_ns"]), int(r["complete_ns"]))
                        for r in rows(d / "prediction.csv")
                        if int(r["frame_id"]) > skip and r["complete_ns"]]
    solo = [(c - p) / 1e6 for p, c in frames(solo_dir)]
    requests_csv = unmanaged_dir / "requests.csv"
    if not solo or not requests_csv.is_file():
        return 0.0
    base = sum(solo) / len(solo)
    active = [(int(r["start_ns"]), int(r["start_ns"]) + int(float(r["latency_ms"]) * 1e6))
              for r in rows(requests_csv) if r.get("start_ns")]
    sxy = sxx = 0.0
    for p, c in frames(unmanaged_dir):
        overlap = sum(max(0, min(c, e) - max(p, s)) for s, e in active) / 1e6
        sxy += overlap * ((c - p) / 1e6 - base)
        sxx += overlap * overlap
    return min(1.0, max(0.0, sxy / sxx)) if sxx else 0.0


def llm_rate(result: dict[str, Any]) -> float:
    """LLM throughput of a phase: all streams when extra streams ran,
    else output tokens per active second of the tracked request."""
    llm = result.get("llm", {})
    return llm.get("stream_tokens_per_s") or llm.get("tokens_per_active_s", 0.0)


def miss_tolerance(args: argparse.Namespace, sweep_frames: int, skip_frames: int) -> float:
    """--miss-tolerance, by default two frames of a sweep phase."""
    if args.miss_tolerance is not None:
        return args.miss_tolerance
    return 2.0 / max(1, sweep_frames - skip_frames)


def choose(sweep: list[dict[str, Any]], args: argparse.Namespace,
           tolerance: float = 0.0) -> tuple[dict[str, Any], str]:
    """Pick H from the sweep by the workload's objective.

    miss        fewest misses with retention >= --min-retention; candidates within
                ``tolerance`` of the fewest count as equal (a sweep phase's misses
                differ by a frame or two from run to run) and the one with
                the most LLM throughput among them is taken
    throughput  highest retention with miss rate <= --miss-budget
    """
    if args.objective == "miss":
        feasible = [c for c in sweep if c["retention"] >= args.min_retention]

        def best(pool: list[dict[str, Any]]) -> dict[str, Any]:
            fewest = min(c["miss_rate"] for c in pool)
            tied = [c for c in pool if c["miss_rate"] <= fewest + tolerance + 1e-12]
            return max(tied, key=lambda c: (c["retention"], -c["miss_rate"]))
        fallback, unattainable = (lambda: max(sweep, key=lambda c: c["retention"])), "retention_unattainable"
    else:
        feasible = [c for c in sweep if c["miss_rate"] <= args.miss_budget]
        best = lambda pool: max(pool, key=lambda c: (c["retention"], -c["miss_rate"]))
        fallback, unattainable = (lambda: min(sweep, key=lambda c: c["miss_rate"])), "miss_budget_unattainable"
    if not feasible:
        return fallback(), unattainable
    chosen = best(feasible)
    horizons = [c["horizon_ms"] for c in sweep]
    # at either end of the grid a candidate beyond it may be better still
    edge = chosen["horizon_ms"] in (min(horizons), max(horizons)) and len(horizons) > 1
    return chosen, "profiled_at_grid_edge" if edge else "profiled"


def decision(chosen: dict[str, Any], status: str, args: argparse.Namespace, tolerance: float) -> dict[str, Any]:
    return {"status": status, "objective": args.objective, "min_retention": args.min_retention,
            "miss_budget": args.miss_budget, "miss_tolerance": tolerance, "horizon_ms": chosen["horizon_ms"],
            "miss_rate": chosen["miss_rate"], "retention": chosen["retention"]}


def reselect(args: argparse.Namespace) -> int:
    """Re-pick H from a finished profile's sweep with another objective."""
    out = args.reselect.parent
    report = json.loads(args.reselect.read_text())
    profile = json.loads((out / "profile.json").read_text())
    tolerance = miss_tolerance(args, int(report["args"]["sweep_frames"]), int(report["args"]["skip_frames"]))
    chosen, status = choose(report["sweep"], args, tolerance)
    report.setdefault("reselections", []).append(decision(chosen, status, args, tolerance))
    profile["status"] = status
    profile["server_env"]["EDGEPACE_RHO_HORIZON_MS"] = f"{chosen['horizon_ms']:g}"
    args.reselect.write_text(json.dumps(report, indent=2) + "\n")
    (out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    if (out / "profile" / "profile.json").is_file():
        update_profile_dir(out / "profile", profile["server_env"], status, report["reselections"][-1], report)
    print(f"-> {status}: H={chosen['horizon_ms']:g} ms (miss {100 * chosen['miss_rate']:.2f}%, "
          f"retention {100 * chosen['retention']:.1f}%); {out / 'profile.json'}", file=sys.stderr)
    return 0


# --- phases -------------------------------------------------------------------

def phase_args(args: argparse.Namespace, frames: int, sampler_extra: str) -> argparse.Namespace:
    copy = argparse.Namespace(**vars(args))
    copy.frames = frames
    copy.sampler_extra = sampler_extra
    return copy


def stress_phase(args: argparse.Namespace, thresholds: dict[str, float]) -> dict[str, Any]:
    phase_dir = (args.output_dir / "stress").resolve()
    phase_dir.mkdir(parents=True, exist_ok=True)
    ready = phase_dir / "stress.ready"
    ready.unlink(missing_ok=True)
    duration = int(args.frames * args.period_us / 1e6) + 120
    with (phase_dir / "stress.log").open("w") as log:
        stressor = subprocess.Popen(
            [str(args.stress_binary), "--mode", "memory", "--memory-mb", str(args.stress_memory_mb),
             "--duration-sec", str(duration), "--ready-file", str(ready)],
            stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30.0
            while not ready.exists():
                if stressor.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"GPU stressor did not start; see {log.name}")
                time.sleep(0.1)
            time.sleep(2.0)
            return run_phase(args, "stress", None, False, thresholds)
        finally:
            stressor.terminate()
            try:
                stressor.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                stressor.kill()


def stress_llm_phase(args: argparse.Namespace, thresholds: dict[str, float]) -> dict[str, Any]:
    """The foreground beside the LLM itself as the memory load: back-to-back
    decode streams of --stress-llm-request on an unmanaged server.  On the phone this
    is the NPU traffic the foreground meets; there is no GPU stressor there."""
    copy = argparse.Namespace(**vars(args))
    copy.request = args.stress_llm_request
    copy.llm_period_ms, copy.llm_align, copy.freshness_ms = 0.0, "none", 120000.0
    copy.parallel = args.stress_llm_streams
    copy.llm_extra_streams = args.stress_llm_streams - 1
    copy.context_size = max(args.context_size, 1024 * args.stress_llm_streams)
    if args.device == "android":
        copy.device_request = push_request(copy, copy.request)
    return run_phase(copy, "stress", {}, True, thresholds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objective", choices=("miss", "throughput"), default="miss",
                        help="miss: fewest misses at >= --min-retention; "
                             "throughput: most LLM throughput at <= --miss-budget")
    parser.add_argument("--miss-budget", type=float, default=0.02, help="deadline-miss rate, 0-1")
    parser.add_argument("--min-retention", type=float, default=0.60,
                        help="minimum LLM throughput of a candidate relative to unmanaged")
    parser.add_argument("--miss-tolerance", type=float,
                        help="objective miss: candidates within this miss rate (0-1) of the fewest "
                             "count as equal, the most throughput among them wins "
                             "(default: two frames of a sweep phase)")
    parser.add_argument("--reselect", type=Path, metavar="PROFILE_REPORT",
                        help="only re-pick H from this profile_report.json's sweep (no runs)")
    if "--reselect" in sys.argv:
        args, _ = parser.parse_known_args()
        return reselect(args)
    add_run_arguments(parser)
    parser.add_argument("--mode", choices=("quick", "sweep"), default="quick",
                        help="quick: solo and unmanaged, --quick-frames each, back to back, H "
                             "--horizon-ms; sweep: solo --frames, unmanaged and each of --horizons "
                             "--sweep-frames, H chosen by --objective")
    parser.add_argument("--quick-frames", type=int, default=600, help="frames per quick-mode phase")
    parser.add_argument("--horizon-ms", type=float, default=20.0, help="quick mode: the warning horizon H")
    parser.add_argument("--interference", action="store_true",
                        help="put the measured foreground slowdown k (reported always) into the "
                             "laxity, L = (d - t) - (1 + k) W_fg")
    parser.add_argument("--sweep-frames", type=int, default=900, help="frames per sweep phase")
    parser.add_argument("--stress-frames", type=int, default=900)
    parser.add_argument("--runtime-sampler", default="--sample-interval-ms 4 --gate-persistence 2",
                        help="sampler arguments of the managed run time; every phase uses them")
    parser.add_argument("--horizons", default="4,8,12", help="candidate warning horizons H, ms")
    parser.add_argument("--w-demand", type=float, default=0.15)
    parser.add_argument("--w-stall", type=float, default=0.70)
    parser.add_argument("--w-urgency", type=float, default=0.70)
    parser.add_argument("--u-thresholds", default="0.2226,0.5226",
                        help="theta_low,theta_high on wU * U alone, before the D/S shift")
    parser.add_argument("--stress-binary", type=Path,
                        help="GPU memory stressor (llm_ros/foreground gpu_stress); omit to skip step 2")
    parser.add_argument("--stress-memory-mb", type=int, default=384)
    parser.add_argument("--stress-llm-request", type=Path,
                        help="instead of --stress-binary: the LLM as the memory load, back-to-back "
                             "decode streams of this request body (e.g. requests/text_256tok.json)")
    parser.add_argument("--stress-llm-streams", type=int, default=4)
    parser.add_argument("--stall-auc-min", type=float, default=0.60,
                        help="frame stall AUC (slowed unmanaged frames, or the stressor, vs solo) "
                             "below which wS is set to 0")
    parser.add_argument("--profile-dir", type=Path,
                        help="profile directory to write (default OUTPUT_DIR/profile)")
    parser.add_argument("--profile-name", help="profile name (default: the run's directory name)")
    args = parser.parse_args()
    if "{phase_dir}/progress.page" not in args.foreground:
        parser.error("--foreground must publish the progress page at {phase_dir}/progress.page")
    if args.device == "android" and args.stress_binary:
        parser.error("--stress-binary starts a host process; on the phone use --stress-llm-request")
    if args.stress_binary and args.stress_llm_request:
        parser.error("--stress-binary and --stress-llm-request are alternatives")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quick = args.mode == "quick"
    if quick and (args.stress_binary or args.stress_llm_request):
        print("quick mode: the stressor is not run; the cross comparison decides wS", file=sys.stderr)
        args.stress_binary = args.stress_llm_request = None
    args.hold_lock = quick   # on a phone: one lock, phases back to back
    platform = require_locked_platform(args)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    managed = managed_environment(args, plan)
    sampler_extra = f"{PREDICTOR_ARGS} {args.gate} {args.sampler_args} {args.runtime_sampler}"
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "platform": platform,
                              "args": {k: str(v) for k, v in vars(args).items()}}

    def save() -> None:
        (args.output_dir / "profile_report.json").write_text(json.dumps(report, indent=2) + "\n")

    solo_frames_n = args.quick_frames if quick else args.frames
    unmanaged_frames_n = args.quick_frames if quick else args.sweep_frames

    # 1. solo: reference scales and the quiet D/S levels
    solo = run_phase(phase_args(args, solo_frames_n, sampler_extra), "solo", None, False, DEFAULT_THRESHOLDS)
    windows, solo_frames = read_windows((args.output_dir / "solo").resolve(), args.skip_frames)
    scales = reference_scales(windows)
    thresholds = {"stall": scales["stall"], "miss_rate": scales["miss_rate"], "mpki": scales["mpki"]}
    quiet = published_levels(windows, scales)
    report["solo"] = {"miss_rate": solo["miss_rate"], "response_ms": solo["response_ms"],
                      "pipeline_windows": sum(w["pipeline"] for w in windows),
                      "reference_scales": scales, "published": quiet,
                      **({"device_lock": solo["device_lock"]} if "device_lock" in solo else {})}
    save()

    # 2. the target LLM load unmanaged, cross-compared with solo: does S see
    # the frames the LLM slows, and by how much does it slow them
    unmanaged = run_phase(phase_args(args, unmanaged_frames_n, sampler_extra), "unmanaged", {}, True, thresholds)
    reference = llm_rate(unmanaged)
    report["unmanaged"] = {"miss_rate": unmanaged["miss_rate"], "llm_rate": reference,
                           "llm": unmanaged.get("llm", {}),
                           **({"device_lock": unmanaged["device_lock"]} if "device_lock" in unmanaged else {})}
    cross = cross_compare(args.output_dir / "solo", args.output_dir / "unmanaged", args.skip_frames, solo_frames)
    report["cross"] = cross
    w_stall = args.w_stall if cross["stall_auc_slowed"] >= args.stall_auc_min else 0.0
    cross["w_stall"] = w_stall
    save()

    # 3. optional stressor: does S see memory contention for this foreground?  It
    # decides wS when it runs
    if args.stress_binary or args.stress_llm_request:
        stress_args = phase_args(args, args.stress_frames, sampler_extra)
        stress = (stress_phase(stress_args, thresholds) if args.stress_binary
                  else stress_llm_phase(stress_args, thresholds))
        _, stress_frames = read_windows((args.output_dir / "stress").resolve(), args.skip_frames)
        base, hit = frame_signals(solo_frames), frame_signals(stress_frames)
        separation = {"stall_auc": auc(base["stall"], hit["stall"]),
                      "mpki_auc": auc(base["mpki"], hit["mpki"]),
                      "stall_rise": quantile(hit["stall"], 0.5) / quantile(base["stall"], 0.5) - 1,
                      "miss_rate": stress["miss_rate"],
                      "load": "gpu_stress" if args.stress_binary else f"llm:{args.stress_llm_request.name}"
                              f" x{args.stress_llm_streams}",
                      "llm_rate": llm_rate(stress) if args.stress_llm_request else None}
        w_stall = args.w_stall if separation["stall_auc"] >= args.stall_auc_min else 0.0
        separation["w_stall"] = w_stall
        report["stress"] = separation
        save()

    # 4. thresholds shifted by the quiet D/S contribution
    u_low, u_high = (float(x) for x in args.u_thresholds.split(","))
    shift = args.w_demand * quiet["demand_p50"] + w_stall * quiet["stall_p50"]
    rho = {"EDGEPACE_ONLINE_RHO": "1", "EDGEPACE_RHO_W_DEMAND": f"{args.w_demand:g}",
           "EDGEPACE_RHO_W_STALL": f"{w_stall:g}", "EDGEPACE_RHO_W_URGENCY": f"{args.w_urgency:g}",
           "EDGEPACE_RHO_THETA_LOW": f"{u_low + shift:.4f}", "EDGEPACE_RHO_THETA_HIGH": f"{u_high + shift:.4f}"}
    if args.interference:   # the measured slowdown enters the laxity
        rho["EDGEPACE_RHO_INTERFERENCE"] = f"{cross['interference_k']:.3f}"
    report["thresholds"] = {"u_thresholds": [u_low, u_high], "shift": shift,
                            "theta_low": u_low + shift, "theta_high": u_high + shift}

    # 5. H: the default (quick), or swept on the target LLM load
    sweep_args = phase_args(args, args.sweep_frames, sampler_extra)
    report["sweep"] = []
    for horizon in ([] if quick else (float(h) for h in args.horizons.split(","))):
        env = {**managed, **rho, "EDGEPACE_RHO_HORIZON_MS": f"{horizon:g}"}
        result = run_phase(sweep_args, f"edgepace_h{horizon:g}", env, True, thresholds)
        rate = llm_rate(result)
        report["sweep"].append({"horizon_ms": horizon, "miss_rate": result["miss_rate"],
                                "response_ms": result["response_ms"], "llm_rate": rate,
                                "retention": rate / reference if reference > 0 else 0.0,
                                "trace_events": result.get("trace_events", {}),
                                **({"device_lock": result["device_lock"]} if "device_lock" in result else {})})
        save()

    if quick:
        chosen, status = {"horizon_ms": args.horizon_ms}, "profiled_quick"
        report["decision"] = {"status": status, "objective": "default_h", "horizon_ms": args.horizon_ms,
                              "h_rule_b": cross["h_rule_b"]}
    else:
        tolerance = miss_tolerance(args, args.sweep_frames, args.skip_frames)
        chosen, status = choose(report["sweep"], args, tolerance)
        report["decision"] = {**decision(chosen, status, args, tolerance), "h_rule_b": cross["h_rule_b"]}
    save()

    profile = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "platform": platform,
        "workload": {"model": str(args.model.resolve()), "plan": str(args.plan.resolve()),
                     "foreground": args.foreground, "request": str(args.request.resolve()),
                     "llm_period_ms": args.llm_period_ms, "llm_align": args.llm_align,
                     "parallel": args.parallel, "llm_extra_streams": args.llm_extra_streams,
                     **({"mmproj": str(args.mmproj), "vision_device": args.vision_device}
                        if args.mmproj else {})},
        "sampler_args": [*args.runtime_sampler.split(),
                         "--stall-threshold", f"{scales['stall']:.6g}",
                         "--miss-rate-threshold", f"{scales['miss_rate']:.6g}",
                         "--mpki-threshold", f"{scales['mpki']:.6g}"],
        "server_env": {**rho, "EDGEPACE_RHO_HORIZON_MS": f"{chosen['horizon_ms']:g}"},
        "report": "profile_report.json",
    }
    (args.output_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    device_workload, device_print = device_profile_fields(args, platform)   # ({}, None) locally
    # the profile directory the runtime takes (tools/edgepace serve/launch)
    profile_dir = write_profile_dir(
        args.profile_dir or args.output_dir / "profile",
        name=args.profile_name or args.output_dir.resolve().parent.name, status=status,
        source=str(args.output_dir.resolve()),
        workload={**profile["workload"], "period_us": args.period_us, "deadline_us": args.deadline_us,
                  "freshness_ms": args.freshness_ms, "max_tokens": args.max_tokens,
                  "context_size": args.context_size, "batch_size": args.batch_size,
                  "ubatch_size": args.ubatch_size, "prompt_cache": args.prompt_cache, **device_workload},
        sampler_args=[*shlex.split(args.gate), *shlex.split(args.sampler_args), *profile["sampler_args"]],
        server_env=profile["server_env"], decision=report["decision"], plan=args.plan, report=report,
        binaries={"server": args.server, "policy": args.policy_library, "sampler": args.sampler},
        platform=platform, fingerprint_data=device_print)

    print(f"solo: miss {100 * solo['miss_rate']:.2f}%  scales r0={scales['miss_rate']:.4g} "
          f"m0={scales['mpki']:.4g} s0={scales['stall']:.4g}  D0={quiet['demand_p50']:.3f} "
          f"S0={quiet['stall_p50']:.3f}", file=sys.stderr)
    print(f"unmanaged: miss {100 * unmanaged['miss_rate']:.2f}%  llm {reference:.2f}; slowed frames "
          f"{cross['slowed_frames']}/{cross['frames']}, stall AUC {cross['stall_auc_slowed']:.3f}; slowdown p99 "
          f"{cross['excess_ms']['p99']:.1f} ms -> rule B H {cross['h_rule_b']:g} ms; interference k "
          f"{cross['interference_k']:.3f}{' (used)' if args.interference else ''}", file=sys.stderr)
    if "stress" in report:
        s = report["stress"]
        print(f"stress: stall AUC {s['stall_auc']:.3f} (rise {100 * s['stall_rise']:+.1f}%), "
              f"MPKI AUC {s['mpki_auc']:.3f}", file=sys.stderr)
    print(f"wS {w_stall:g}; thresholds: {u_low + shift:.4f} / {u_high + shift:.4f} (shift {shift:.4f})",
          file=sys.stderr)
    for c in report["sweep"]:
        print(f"H={c['horizon_ms']:g} ms: miss {100 * c['miss_rate']:.2f}%  llm {c['llm_rate']:.2f} "
              f"(retention {100 * c['retention']:.1f}%)", file=sys.stderr)
    print(f"-> {status}: H={chosen['horizon_ms']:g} ms; profile directory {profile_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
