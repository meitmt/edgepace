#!/usr/bin/env python3
"""Online validation of the progress gate.

Runs the same foreground + LLM load under several admission conditions, each
as one phase of ``edgepace_phase.run_phase`` (launch-mode sampler with the
progress predictor, EdgePace server and policy):

  unmanaged      LLM without the EdgePace policy
  v6_stall       EdgePace as shipped: in-frame admission needs demand <= 0.35 and
                 stall <= the priority's gate, plus deadline fit (or rescue)
  laxity_only    same, without the PMU condition (deadline fit or rescue)
  progress_gate  same, the PMU condition is "no progress alarm for this frame"
                 (sampler gate rule, default shape d=+5% k=3 a=6 ms)
  edgepace_rho   paper §3.3 as written: rho = wD*D + wS*S + wU*U selects
                 Bypass / Admit / Hold at every boundary (Hold rechecked every
                 1 ms), U from W_fg = the sampler's smoothed progress prediction
                 (historical estimate before it has one); 4 ms sampling
  edgepace_rho_2ms  the same with 2 ms sampling

laxity_only vs progress_gate isolates what the online predictor adds.  The
predictor runs in every condition, so its alarms are recorded even where no
policy uses them.  Conditions rotate order between repeats.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from edgepace_phase import DEFAULT_SAMPLER, DEFAULT_THRESHOLDS, managed_environment, run_phase
from edgepace_plan import (DEFAULT_POLICY_LIBRARY, DEFAULT_REQUEST, DEFAULT_SERVER, add_vision_arguments,
                          require_locked_platform)
from edgepace_android import add_device_arguments

# rho parameters; the policy defaults changed in so the historical
# conditions pin them to keep their meaning
RHO_BASE = {"EDGEPACE_ONLINE_RHO": "1", "EDGEPACE_RHO_W_DEMAND": "0.15", "EDGEPACE_RHO_W_STALL": "0.15",
             "EDGEPACE_RHO_W_URGENCY": "0.70", "EDGEPACE_RHO_THETA_LOW": "0.30", "EDGEPACE_RHO_THETA_HIGH": "0.60",
             "EDGEPACE_RHO_HORIZON_MS": "2"}
CONDITIONS: dict[str, dict[str, str] | None] = {
    "unmanaged": None,
    "v6_stall": {},
    "laxity_only": {"EDGEPACE_ONLINE_DEMAND_SAFE_MAX": "1e9", "EDGEPACE_ONLINE_STALL_SAFE_MAX": "1e9"},
    "progress_gate": {"EDGEPACE_ONLINE_PROGRESS_GATE": "1"},
    # the policy's rho defaults (wS 0.70, theta 0.4755 / 0.7755, H 8 ms)
    "edgepace": {"EDGEPACE_ONLINE_RHO": "1"},
    # paper §3.3 rho = wD*D + wS*S + wU*U, U from the progress-based W_fg
    "edgepace_rho": RHO_BASE,
    "edgepace_rho_2ms": RHO_BASE,
    # warning horizon H of U(t) ( default 2 ms): Hold engages at larger laxity
    "edgepace_rho_h5": {**RHO_BASE, "EDGEPACE_RHO_HORIZON_MS": "5"},
    "edgepace_rho_h8": {**RHO_BASE, "EDGEPACE_RHO_HORIZON_MS": "8"},
    # ablation: D and S replaced by their light-load medians (D 0.197, S 0.319
    # at the rho scales, runs): wD = wS = 0 and both thresholds lowered by
    # 0.15 * (0.197 + 0.319) = 0.077, so only their variation is removed
    "edgepace_rho_h8_nods": {**RHO_BASE, "EDGEPACE_RHO_HORIZON_MS": "8",
                             "EDGEPACE_RHO_W_DEMAND": "0", "EDGEPACE_RHO_W_STALL": "0",
                             "EDGEPACE_RHO_THETA_LOW": "0.2226", "EDGEPACE_RHO_THETA_HIGH": "0.5226"},
    # larger wS; both thresholds raised by (wS - 0.15) * 0.319 (light-load S
    # median) so light-load decisions stay as with wS = 0.15
    "edgepace_rho_h8_ws40": {**RHO_BASE, "EDGEPACE_RHO_HORIZON_MS": "8", "EDGEPACE_RHO_W_STALL": "0.40",
                             "EDGEPACE_RHO_THETA_LOW": "0.3798", "EDGEPACE_RHO_THETA_HIGH": "0.6798"},
    "edgepace_rho_h8_ws70": {**RHO_BASE, "EDGEPACE_RHO_HORIZON_MS": "8", "EDGEPACE_RHO_W_STALL": "0.70",
                             "EDGEPACE_RHO_THETA_LOW": "0.4755", "EDGEPACE_RHO_THETA_HIGH": "0.7755"},
}
# per-condition sampler arguments, after the common ones; the rho conditions also
# get --rho-scales (the §3.2 reference scales for D and S)
CONDITION_SAMPLER = {
    "edgepace": "--sample-interval-ms 4 --gate-persistence 2",
    "edgepace_rho": "--sample-interval-ms 4 --gate-persistence 2",
    "edgepace_rho_2ms": "--gate-persistence 3",
    "edgepace_rho_h5": "--sample-interval-ms 4 --gate-persistence 2",
    "edgepace_rho_h8": "--sample-interval-ms 4 --gate-persistence 2",
    "edgepace_rho_h8_nods": "--sample-interval-ms 4 --gate-persistence 2",
    "edgepace_rho_h8_ws40": "--sample-interval-ms 4 --gate-persistence 2",
    "edgepace_rho_h8_ws70": "--sample-interval-ms 4 --gate-persistence 2",
}
PREDICTOR_ARGS = "--progress-page {phase_dir}/progress.page --prediction-csv {phase_dir}/prediction.csv"


def gate_outcome(phase_dir: Path, skip: int) -> dict[str, Any]:
    """Frames with a gate alarm and their misses, from the sampler's prediction.csv."""
    path = phase_dir / "prediction.csv"
    if not path.is_file():
        return {}
    alarmed = alarmed_miss = quiet = quiet_miss = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame_id"]) <= skip:
                continue
            miss = row["miss"] == "1"
            if row["gate_alarm_elapsed_ns"]:
                alarmed += 1; alarmed_miss += miss
            else:
                quiet += 1; quiet_miss += miss
    return {"alarmed_frames": alarmed, "alarmed_misses": alarmed_miss,
            "quiet_frames": quiet, "quiet_misses": quiet_miss}


def trace_summary(phase_dir: Path) -> dict[str, Any]:
    """Policy events by kind, and in-frame holds split by the gate alarm.

    The trace does not carry the progress state (see edgepace_policy_engine.h),
    so a hold counts as after the alarm when it begins at or after that
    frame's alarm time in prediction.csv (pipeline start + alarm elapsed).
    """
    path = phase_dir / "edgepace_trace.csv"
    if not path.is_file():
        return {}
    alarm_at: dict[int, int] = {}
    prediction = phase_dir / "prediction.csv"
    if prediction.is_file():
        with prediction.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["gate_alarm_elapsed_ns"]:
                    alarm_at[int(row["frame_id"])] = (int(row["pipeline_ns"]) +
                                                      int(row["gate_alarm_elapsed_ns"]))
    counts: dict[str, int] = {}
    after_alarm = other = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['event']}:{row['mode']}:{row['fg_phase']}"
            counts[key] = counts.get(key, 0) + 1
            if row["event"] == "hold_begin" and row["fg_phase"] == "fg_active":
                frame = int(row.get("frame_id") or 0)
                if frame in alarm_at and int(row["timestamp_ns"]) >= alarm_at[frame]:
                    after_alarm += 1
                else:
                    other += 1
    return {"events": counts, "in_frame_holds_after_alarm": after_alarm,
            "in_frame_holds_other": other}


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Workload, LLM load and platform options shared with edgepace_profile.py."""
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--foreground", required=True,
                        help="foreground command; placeholders {frames} {task_page} {phase_dir}; "
                             "must pass --progress-page {phase_dir}/progress.page")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--gate", default="--gate-margin 0.05 --gate-persistence 3 --gate-min-ms 6",
                        help="sampler gate-rule arguments")
    parser.add_argument("--no-sampler", action="store_true",
                        help="run the foreground without the PMU sampler (unmanaged baselines only)")
    parser.add_argument("--sampler-args", default="",
                        help="extra sampler arguments after the defaults, e.g. '--sample-interval-ms 10'")
    parser.add_argument("--rho-scales", default="",
                        help="sampler D/S reference scales for the rho conditions "
                             "(analysis/rho_scales.py output)")
    parser.add_argument("--min-pmu-coverage", type=float, default=0.90)
    parser.add_argument("--priority", choices=("miss", "fresh", "balanced", "manual"), default="balanced")
    parser.add_argument("--priority-bias", type=float)
    parser.add_argument("--llm-period-ms", type=float, default=1600.0)
    parser.add_argument("--llm-align", choices=("none", "frame"), default="none",
                        help="frame: send each request right after a foreground frame completes "
                             "(paper-era client), none: free-running period")
    parser.add_argument("--period-us", type=int, default=33333)
    parser.add_argument("--deadline-us", type=int, default=33333)
    parser.add_argument("--freshness-ms", type=float, default=1800.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=1, help="server slots (--parallel)")
    parser.add_argument("--llm-extra-streams", type=int, default=0,
                        help="extra back-to-back clients beside the tracked request")
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--policy-library", type=Path, default=DEFAULT_POLICY_LIBRARY)
    parser.add_argument("--sampler", type=Path, default=DEFAULT_SAMPLER)
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
    add_vision_arguments(parser)
    add_device_arguments(parser)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="append", default=[], metavar="[NAME=]PROFILE_JSON",
                        help="profile.json from edgepace_profile.py, run as condition "
                             "edgepace_profile; NAME=path (repeatable) runs it as profile_NAME")
    args = parser.parse_args()

    conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]
    profiles: dict[str, dict[str, Any]] = {}
    for entry in args.profile:
        name, _, path = entry.rpartition("=")
        profiles[f"profile_{name}" if name else "edgepace_profile"] = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [name for name in conditions if name not in CONDITIONS and name not in profiles]
    if missing:
        parser.error(f"unknown conditions {missing}; known: {[*CONDITIONS, *profiles]} "
                     "(edgepace_profile / profile_NAME need --profile [NAME=]path)")
    if "{phase_dir}/progress.page" not in args.foreground:
        parser.error("--foreground must publish the progress page at {phase_dir}/progress.page")
    if args.no_sampler and any(name != "unmanaged" for name in conditions):
        parser.error("--no-sampler leaves the policy without PMU signals; use it with unmanaged only")
    args.sampler_extra = f"{PREDICTOR_ARGS} {args.gate} {args.sampler_args}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    platform = require_locked_platform(args)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    managed = managed_environment(args, plan)

    summary: dict[str, Any] = {"platform": platform, "args": {k: str(v) for k, v in vars(args).items()},
                               "phases": []}
    for repeat in range(args.repeats):
        order = conditions[repeat % len(conditions):] + conditions[:repeat % len(conditions)]
        for name in order:
            if name in profiles:
                # the profile carries the server variables and sampler arguments
                server_env = {**managed, **profiles[name]["server_env"]}
                extra = " ".join(profiles[name]["sampler_args"])
            else:
                overrides = CONDITIONS[name]
                server_env = {} if overrides is None else {**managed, **overrides}
                extra = CONDITION_SAMPLER.get(name, "")
                if name.startswith("edgepace"):
                    extra += f" {args.rho_scales}"
            phase = f"r{repeat + 1}_{name}"
            args.sampler_extra = f"{PREDICTOR_ARGS} {args.gate} {args.sampler_args} {extra}"
            result = run_phase(args, phase, server_env, True, DEFAULT_THRESHOLDS)
            phase_dir = (args.output_dir / phase).resolve()
            result["gate"] = gate_outcome(phase_dir, args.skip_frames)
            result["trace"] = trace_summary(phase_dir)
            (phase_dir / "phase.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            summary["phases"].append({"phase": phase, "condition": name, "repeat": repeat + 1,
                                      "frames": result["frames"], "misses": result["misses"],
                                      "miss_rate": result["miss_rate"],
                                      "response_ms": result["response_ms"],
                                      "llm": result.get("llm", {}), "gate": result["gate"],
                                      **({"device_lock": result["device_lock"]} if "device_lock" in result else {})})
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                          encoding="utf-8")

    print(f"{'phase':24s} {'miss%':>6s} {'p95ms':>6s} {'llm/s':>6s} {'fresh':>6s} {'lat p50':>8s} "
          f"{'tok/act-s':>9s} {'alarmed(miss)':>14s} {'quiet(miss)':>12s}", file=sys.stderr)
    for p in summary["phases"]:
        llm, gate = p["llm"], p["gate"]
        print(f"{p['phase']:24s} {100 * p['miss_rate']:6.2f} {p['response_ms'].get('p95', 0):6.1f} "
              f"{llm.get('completed_per_s', 0):6.3f} {llm.get('fresh', 0):6d} "
              f"{llm.get('latency_ms', {}).get('p50', 0):8.1f} "
              f"{llm.get('tokens_per_active_s', 0):9.2f} "
              f"{gate.get('alarmed_frames', 0):6d}({gate.get('alarmed_misses', 0):4d}) "
              f"{gate.get('quiet_frames', 0):6d}({gate.get('quiet_misses', 0):4d})"
              + (f"  cpu off-lock {p['device_lock'].get('cpu_off_lock', '?')}/{p['device_lock'].get('samples', '?')} s"
                 f"  model load {p['device_lock'].get('model_load_s', '?')} s"
                 if "device_lock" in p else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
