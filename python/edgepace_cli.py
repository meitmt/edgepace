#!/usr/bin/env python3
"""EdgePace command line: hand a profile to the runtime.

A *profile directory* is the single artefact between profiling and deployment:

    profile.json          parameters, workload, platform and binary fingerprint
    server.env            complete llama-server environment (KEY=VALUE), with the
                          shared pages under the default run directory
    sampler.args          pmu_sampler arguments, one per line
    cut_plan.json         the LLM's Fine Plan (Fine Plan)
    profile_report.json   every measurement behind the parameters
    report.md             the same, for people

Commands (``tools/edgepace <command> --help``):

    init     write a commented workload description (edgepace.toml)
    profile  measure the parameters for a workload description on this device:
             Fine Plan (if none given) and edgepace_profile.py under
             scripts/locked_run.sh (clocks locked, then restored) -> profile dir
    export   turn a finished edgepace_profile.py run into a profile directory
    show     print a profile's parameters and decision
    check    compare the profile's platform and binaries with this machine
    env      print the server environment for a run directory (shell or json)
    serve    exec llama-server with the profile (EdgePace backend + policy DSO)
    launch   exec the PMU sampler, which starts the foreground with counters
             attached; {task_page} {progress_page} {run_dir} {phase_dir} {frames}
             in the foreground command are filled in (without a command after
             --, the profiled foreground command is used)
    request  send requests the way EdgePace expects: the request page marks a
             request as running (EdgePace splits graphs only then) and carries its
             freshness deadline
    stop     phone profiles: end the processes serve/launch left on the device

Phone profiles (``[platform] device = "android"``) run every command on
the device over adb; serve/launch/request start detached device processes.

``serve`` and ``launch`` meet in a run directory (default
``/dev/shm/edgepace/<profile name>``) holding task.page, request.page,
signal.page and progress.page.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shlex
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from edgepace_plan import (DEFAULT_POLICY_LIBRARY, DEFAULT_SERVER, ROOT, clean_environment,
                          platform_snapshot, runtime_environment, server_command)

PROFILE_DIR_SCHEMA = 2
DEFAULT_SAMPLER = ROOT / "monitor/build/pmu_sampler"
RUN_ROOT = Path("/dev/shm/edgepace")
PAGES = ("task", "request", "signal", "progress")


# --- profile directory --------------------------------------------------------

def sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def fingerprint(server: Path, policy: Path, sampler: Path, model: Path,
                snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the parameters were measured with; ``check`` compares it.  The
    platform part comes from ``snapshot`` (taken while profiling) when given."""
    snapshot = snapshot or platform_snapshot()
    stat = model.stat() if model.exists() else None
    return {
        "power_mode": snapshot["power_mode"],
        "cpu_khz": sorted({v["max"] for v in snapshot["cpu"].values()}),
        "gpu_hz": snapshot["gpu"].get("max"),
        "cpu_gpu_pinned": snapshot["cpu_gpu_pinned"],
        "server_sha256": sha256(server), "policy_sha256": sha256(policy),
        "sampler_sha256": sha256(sampler),
        "model": {"path": str(model), "bytes": stat.st_size if stat else None,
                  "mtime_ns": stat.st_mtime_ns if stat else None},
    }


def run_dir_pages(run_dir: Path) -> dict[str, Path]:
    return {name: run_dir / f"{name}.page" for name in PAGES}


def report_markdown(profile: dict[str, Any], report: dict[str, Any]) -> str:
    lines = [f"# EdgePace profile `{profile['name']}`", "",
             f"status **{profile['status']}**, created {profile['created']}", ""]
    w = profile["workload"]
    lines += ["## Workload", "", f"- model `{w['model']}`", f"- foreground `{w['foreground']}`",
              f"- LLM request `{w['request']}`, period {w['llm_period_ms']:g} ms ({w['llm_align']}), "
              f"{w['parallel']} slot(s), {w['llm_extra_streams']} extra stream(s), "
              f"max_tokens {w['max_tokens']}, freshness {w['freshness_ms']:g} ms",
              f"- foreground period / deadline {w['period_us']} / {w['deadline_us']} us", ""]
    solo = report.get("solo", {})
    if solo:
        s, q = solo["reference_scales"], solo["published"]
        lines += ["## Foreground alone", "",
                  f"- misses {100 * solo['miss_rate']:.2f}%, response p95 {solo['response_ms']['p95']:.1f} ms",
                  f"- reference scales r0 {s['miss_rate']:.4g}/s, m0 {s['mpki']:.4g}, s0 {s['stall']:.4g}",
                  f"- quiet D0 {q['demand_p50']:.3f}, S0 {q['stall_p50']:.3f}", ""]
    if "stress" in report:
        st = report["stress"]
        lines += [f"## Memory stress ({st.get('load', 'gpu_stress')})", "",
                  f"- frame stall AUC {st['stall_auc']:.3f} (median {100 * st['stall_rise']:+.1f}%), "
                  f"MPKI AUC {st['mpki_auc']:.3f} -> wS {st['w_stall']:g}", ""]
    if "cross" in report:
        c, u = report["cross"], report["unmanaged"]
        lines += ["## Foreground alone vs beside the unmanaged LLM", "",
                  f"- unmanaged: misses {100 * u['miss_rate']:.2f}%, LLM {u['llm_rate']:.2f}",
                  f"- frames the LLM slowed (pipeline > solo p99): {c['slowed_frames']}/{c['frames']}, "
                  f"their stall AUC vs solo {c['stall_auc_slowed']:.3f}"
                  + (f" -> wS {c['w_stall']:g}" if "w_stall" in c else ""),
                  f"- pipeline slowdown p95 / p99 {c['excess_ms']['p95']:.1f} / {c['excess_ms']['p99']:.1f} ms; "
                  f"H rule B (experimental) {c['h_rule_b']:g} ms",
                  f"- interference k {c.get('interference_k', 0):.3f} (pipeline time per ms of LLM overlap)"
                  + (" -> EDGEPACE_RHO_INTERFERENCE" if "EDGEPACE_RHO_INTERFERENCE" in profile["server_env"] else ""), ""]
    if report.get("decision", {}).get("objective") == "default_h":
        lines += ["## Warning horizon", "", f"default H {report['decision']['horizon_ms']:g} ms "
                  "(quick profile; `--mode sweep` measures it)", ""]
    if "thresholds" in report:
        t = report["thresholds"]
        lines += ["## Thresholds", "",
                  f"- theta_U {t['u_thresholds'][0]:g} / {t['u_thresholds'][1]:g} + shift {t['shift']:.4f}"
                  f" = {t['theta_low']:.4f} / {t['theta_high']:.4f}", ""]
    if report.get("sweep"):
        u = report["unmanaged"]
        lines += ["## Warning horizon", "", f"unmanaged: misses {100 * u['miss_rate']:.2f}%, LLM {u['llm_rate']:.2f}", "",
                  "| H (ms) | misses | LLM | retention |", "|---|---|---|---|"]
        for c in report["sweep"]:
            mark = " ←" if c["horizon_ms"] == float(profile["server_env"]["EDGEPACE_RHO_HORIZON_MS"]) else ""
            lines.append(f"| {c['horizon_ms']:g}{mark} | {100 * c['miss_rate']:.2f}% | {c['llm_rate']:.2f} | "
                         f"{100 * c['retention']:.1f}% |")
        d = report.get("decision", {})
        lines += ["", f"objective `{d.get('objective', 'miss')}` (min retention {d.get('min_retention', '-')}, "
                      f"miss budget {d.get('miss_budget', '-')})", ""]
    lines += ["## Parameters", "", "```", *(f"{k}={v}" for k, v in profile["server_env"].items()), "```", "",
              "sampler: `" + " ".join(profile["sampler_args"]) + "`", ""]
    return "\n".join(lines)


def write_profile_dir(out: Path, *, name: str, status: str, source: str, workload: dict[str, Any],
                      sampler_args: list[str], server_env: dict[str, str], decision: dict[str, Any],
                      plan: Path, report: dict[str, Any], binaries: dict[str, Path],
                      platform: dict[str, Any] | None = None,
                      fingerprint_data: dict[str, Any] | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan, out / "cut_plan.json")
    workload = {**workload, "plan": "cut_plan.json"}
    profile = {
        "schema_version": PROFILE_DIR_SCHEMA, "name": name, "status": status,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": source,
        "workload": workload, "sampler_args": sampler_args, "server_env": server_env,
        "decision": decision,
        "fingerprint": fingerprint_data or fingerprint(binaries["server"], binaries["policy"], binaries["sampler"],
                                                       Path(workload["model"]), platform),
    }
    (out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (out / "profile_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "sampler.args").write_text("\n".join(sampler_args) + "\n")
    write_server_env(out, profile)
    (out / "report.md").write_text(report_markdown(profile, report))
    return out


def update_profile_dir(path: Path, server_env: dict[str, str], status: str,
                       decision: dict[str, Any], report: dict[str, Any]) -> None:
    """After a re-selection: new parameters, same measurements and fingerprint."""
    profile = load_profile(path)
    profile.update(server_env=server_env, status=status, decision=decision)
    (path / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (path / "profile_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_server_env(path, profile)
    (path / "report.md").write_text(report_markdown(profile, report))


def on_android(profile: dict[str, Any]) -> bool:
    return profile["workload"].get("device") == "android"


def run_root(profile: dict[str, Any]) -> Path:
    """Where serve and launch meet: /dev/shm/edgepace, or DEVICE_ROOT/run on a phone."""
    return Path(f"{profile['workload']['device_root']}/run") if on_android(profile) else RUN_ROOT


def write_server_env(path: Path, profile: dict[str, Any]) -> None:
    run_dir = run_root(profile) / profile["name"]
    environment = full_environment(path, profile, run_dir, None)
    (path / "server.env").write_text(
        f"# llama-server environment for profile {profile['name']}; pages under {run_dir}\n"
        + "".join(f"{k}={v}\n" for k, v in environment.items()))


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads((path / "profile.json").read_text())
    if profile.get("schema_version") != PROFILE_DIR_SCHEMA:
        raise SystemExit(f"{path}: not a profile directory (schema {profile.get('schema_version')}); "
                         "use `edgepace export` on a edgepace_profile.py run")
    return profile


def runtime_namespace(profile: dict[str, Any], run_dir: Path | None, trace: Path | None,
                      host: str = "127.0.0.1", port: int = 8080,
                      server: Path = DEFAULT_SERVER, policy: Path = DEFAULT_POLICY_LIBRARY) -> SimpleNamespace:
    w = profile["workload"]
    pages = run_dir_pages(run_dir) if run_dir else {}
    if on_android(profile):   # the runtime lives on the phone
        server = Path(f"{w['device_root']}/bin/llama-server")
        policy = Path(f"{w['device_root']}/lib/libedgepace_policy.so")
    return SimpleNamespace(
        server=server, model=Path(w["model"]), context_size=w["context_size"],
        batch_size=w["batch_size"], ubatch_size=w["ubatch_size"], parallel=w["parallel"],
        prompt_cache=w["prompt_cache"], host=host, port=port,
        policy="adaptive_slo_online", policy_library=policy, priority="balanced", priority_bias=None,
        max_tokens=w["max_tokens"], period_us=w["period_us"], deadline_us=w["deadline_us"],
        freshness_ms=w["freshness_ms"], mmproj=w.get("mmproj"),
        device="android" if on_android(profile) else "local",
        task_page=pages.get("task"), request_page=pages.get("request"),
        signal_page=pages.get("signal"), actmon_page=None, trace=trace)


def full_environment(path: Path, profile: dict[str, Any], run_dir: Path, trace: Path | None,
                     **kwargs: Any) -> dict[str, str]:
    """EdgePace runtime variables from the Fine Plan, then the profile's rho parameters."""
    plan = json.loads((path / "cut_plan.json").read_text())
    return {**runtime_environment(runtime_namespace(profile, run_dir, trace, **kwargs), plan),
            **profile["server_env"]}


# --- commands -----------------------------------------------------------------

def parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def cmd_export(args: argparse.Namespace) -> int:
    """From a edgepace_profile.py output directory (schema 1 profile.json)."""
    source = args.run.resolve()
    old = json.loads((source / "profile.json").read_text())
    report = json.loads((source / "profile_report.json").read_text())
    a = report["args"]
    workload = {**old["workload"],
                "period_us": int(a["period_us"]), "deadline_us": int(a["deadline_us"]),
                "freshness_ms": float(a["freshness_ms"]), "max_tokens": int(a["max_tokens"]),
                "context_size": int(a["context_size"]), "batch_size": int(a["batch_size"]),
                "ubatch_size": int(a["ubatch_size"]), "prompt_cache": parse_bool(a["prompt_cache"])}
    sampler_args = [*shlex.split(a["gate"]), *shlex.split(a.get("sampler_args", "")), *old["sampler_args"]]
    name = args.name or source.parent.name
    device_print = None
    if a.get("device") == "android":   # a phone run (its fingerprint is taken on the device now)
        from edgepace_android import device_profile_fields
        run = SimpleNamespace(device="android", device_root=a["device_root"], model=a["model"],
                              adb=getattr(args, "adb", None) or a["adb"],
                              adb_serial=None if a.get("adb_serial") in (None, "None") else a["adb_serial"])
        extra, device_print = device_profile_fields(run, old.get("platform") or {})
        workload.update(extra)
    out = write_profile_dir(
        args.output, name=name, status=old["status"], source=str(source), workload=workload,
        sampler_args=sampler_args, server_env=old["server_env"], decision=report.get("decision", {}),
        plan=Path(old["workload"]["plan"]), report=report,
        binaries={"server": Path(a["server"]), "policy": Path(a["policy_library"]), "sampler": Path(a["sampler"])},
        platform=old.get("platform"),   # as measured during the profile; binaries hashed now
        fingerprint_data=device_print)
    print(f"profile {name} -> {out}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    print((args.profile / "report.md").read_text())
    return 0


def cmd_plans(args: argparse.Namespace) -> int:
    """The cached Fine Plans: a plan is reused for the same model content,
    graph shape, cut points and device, whatever the workload."""
    caches = args.cache or [ROOT / "build/plan-cache", ROOT / "build/plan-cache-android"]
    for cache in caches:
        entries = sorted(cache.glob("plan_*.json"), key=lambda p: p.stat().st_mtime)
        print(f"== {cache} ({len(entries)} plans)")
        for path in entries:
            try:
                plan = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  {path.name}: unreadable ({exc})")
                continue
            model, phases = plan["model"], plan["phases"]
            backend = plan.get("profile_backend", {})
            parts = [f"{model['architecture']}/{model['layer_count']}L {Path(model['path']).name}",
                     f"ctx {plan['context']} b {plan['batch']} ub {plan['ubatch']}"]
            if "vision" in phases:
                v = phases["vision"]
                parts.append(f"vision {Path(plan['vision_model']['path']).name} {len(v['chunks'])} quanta "
                             f"(body max {v['body_max_runtime_ms']:.2f} ms)")
            p = phases["prefill"]
            parts.append(f"prefill {len(p['chunks'])} quanta (body max {p['body_max_runtime_ms']:.2f} ms"
                         + (f", {','.join(p['boundaries'])}" if p.get("boundaries") else "") + ")")
            parts.append(f"decode {len(phases['decode']['chunks'])} quanta")
            parts.append(f"device {backend.get('device', '?')}, runtime {backend.get('runtime_fingerprint', '?')}, "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(path.stat().st_mtime))}")
            old = "_m" not in path.name   # key before adopted on first use, then unused
            print(f"  {path.name}{'  (old key)' if old else ''}\n    " + "; ".join(parts))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    if on_android(profile):
        return android_check(args, profile)
    then = profile["fingerprint"]
    now = fingerprint(args.server, args.policy_library, args.sampler, Path(profile["workload"]["model"]))
    problems = []
    for key, label in (("power_mode", "power mode"), ("cpu_khz", "CPU clock"), ("gpu_hz", "GPU clock"),
                       ("cpu_gpu_pinned", "CPU/GPU pinning"), ("server_sha256", "llama-server binary"),
                       ("policy_sha256", "policy DSO"), ("sampler_sha256", "PMU sampler"), ("model", "model file")):
        if then.get(key) != now.get(key):
            problems.append(f"{label}: profiled {then.get(key)!r}, now {now.get(key)!r}")
    if not (args.profile / "cut_plan.json").is_file():
        problems.append("cut_plan.json missing")
    for line in problems:
        print(f"MISMATCH {line}")
    print("EMC lock is not visible without root; profile and run under the same "
          "scripts/locked_run.sh settings")
    print("OK" if not problems else f"{len(problems)} mismatch(es): re-profile or accept the difference")
    return 0 if not problems else 1


def run_dir_of(args: argparse.Namespace, profile: dict[str, Any]) -> Path:
    if on_android(profile):
        return Path(str(args.run_dir or run_root(profile) / profile["name"]))
    return (args.run_dir or RUN_ROOT / profile["name"]).resolve()


def cmd_env(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    env = full_environment(args.profile, profile, run_dir_of(args, profile), args.trace,
                           host=args.host, port=args.port, server=args.server, policy=args.policy_library)
    if args.format == "json":
        print(json.dumps(env, indent=2))
    else:
        for k, v in env.items():
            print(f"export {k}={shlex.quote(v)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    if on_android(profile):
        return android_serve(args, profile)
    run_dir = run_dir_of(args, profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(host=args.host, port=args.port, server=args.server, policy=args.policy_library)
    runtime = full_environment(args.profile, profile, run_dir, args.trace, **kwargs)
    command = server_command(runtime_namespace(profile, run_dir, args.trace, **kwargs)) + args.extra
    (run_dir / "server.command.json").write_text(json.dumps({"command": command, "environment": runtime}, indent=2) + "\n")
    if args.dry_run:
        print(shlex.join(command))
        return 0
    environment = clean_environment()
    environment.update(runtime)
    os.execve(command[0], command, environment)
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    if on_android(profile):
        return android_launch(args, profile)
    run_dir = run_dir_of(args, profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    pages = run_dir_pages(run_dir)
    # {phase_dir} is the run directory, so a profiled "{phase_dir}/progress.page" is the progress page
    fill = {"task_page": str(pages["task"]), "progress_page": str(pages["progress"]),
            "run_dir": str(run_dir), "phase_dir": str(run_dir), "frames": str(args.frames)}
    tokens = args.command or shlex.split(profile["workload"]["foreground"])
    foreground = [token.format(**fill) for token in tokens]
    command = [str(args.sampler), "--snapshot", str(pages["signal"]), "--no-mc",
               "--progress-page", str(pages["progress"]), "--prediction-csv", str(run_dir / "prediction.csv"),
               *(["--csv", str(run_dir / "pmu_signal.csv")] if args.log else []),
               *profile["sampler_args"], "--", *foreground]
    (run_dir / "sampler.command.json").write_text(json.dumps(command, indent=2) + "\n")
    if args.dry_run:
        print(shlex.join(command))
        return 0
    os.execv(command[0], command)
    return 0


def cmd_request(args: argparse.Namespace) -> int:
    from edgepace_phase import RequestPage   # page layout shared with the evaluation tools
    profile = load_profile(args.profile)
    if on_android(profile):
        return android_request(args, profile)
    run_dir = run_dir_of(args, profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    body = args.body.read_bytes()
    url = f"http://{args.host}:{args.port}" + ("/v1/completions" if b'"prompt"' in body else "/v1/chat/completions")
    freshness_ns = int((args.freshness_ms or profile["workload"]["freshness_ms"]) * 1e6)
    page = RequestPage(run_dir_pages(run_dir)["request"])
    try:
        for request_id in range(1, args.count + 1):
            start = time.monotonic_ns()
            deadline = start + freshness_ns
            page.publish(request_id, True, start, deadline)
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        url, data=body, headers={"Content-Type": "application/json"}), timeout=args.timeout) as r:
                    reply = json.loads(r.read())
                status = "ok"
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                reply, status = {}, f"error {exc}"
            done = time.monotonic_ns()
            page.publish(request_id, False, start, deadline, terminal=1)
            usage = reply.get("usage", {})
            print(json.dumps({"request": request_id, "status": status, "latency_ms": (done - start) / 1e6,
                              "fresh": done <= deadline, "completion_tokens": usage.get("completion_tokens")}),
                  flush=True)
            if args.print_reply and reply:
                print(json.dumps(reply.get("choices", reply), ensure_ascii=False))
            if request_id < args.count:
                time.sleep(max(0.0, args.period_ms / 1e3 - (done - start) / 1e9))
    finally:
        page.close()
    return 0


# --- phone: the same commands, run on the device over adb -----------------
# Processes started by serve/launch are nohup'ed on the phone, so they outlive
# the adb session; `edgepace stop` ends them.

def android_device(args: argparse.Namespace, profile: dict[str, Any]) -> Any:
    from edgepace_android import Device
    w = profile["workload"]
    return Device(getattr(args, "adb", None) or w.get("adb") or os.environ.get("EDGEPACE_ADB", "adb"),
                  getattr(args, "adb_serial", None) or w.get("adb_serial"), w["device_root"])


def android_spawn(d: Any, run_dir: Path, name: str, lines: list[str]) -> None:
    d.run(f"mkdir -p {run_dir}")
    d.write(f"{run_dir}/{name}.sh", ("\n".join(lines) + "\n").encode())
    d.run(f"nohup sh {run_dir}/{name}.sh > {run_dir}/{name}.log 2>&1 < /dev/null & echo $! > {run_dir}/{name}.pid")


def android_libraries(profile: dict[str, Any]) -> str:
    from edgepace_android import device_libraries
    w = profile["workload"]
    return device_libraries(w["device_root"], w.get("vision_device") or "HTP0")


def android_check(args: argparse.Namespace, profile: dict[str, Any]) -> int:
    from edgepace_android import device_fingerprint
    then = profile["fingerprint"]
    now = device_fingerprint(android_device(args, profile), profile["workload"]["model"])
    problems = [f"{name}: profiled {then['binaries_sha256'].get(name, '-')[:12]}, now {now['binaries_sha256'].get(name, '-')[:12]}"
                for name in sorted(set(then["binaries_sha256"]) | set(now["binaries_sha256"]))
                if then["binaries_sha256"].get(name) != now["binaries_sha256"].get(name)]
    if then["model"] != now["model"]:
        problems.append(f"model file: profiled {then['model']}, now {now['model']}")
    # profiles before do not record it; they were measured with stall_backend_mem
    pmu = then.get("sampler_pmu", "--llc-event l2d_cache_lmiss_rd (stall_backend_mem)")
    if pmu != now["sampler_pmu"]:
        problems.append(f"sampler PMU events: profiled {pmu}, now {now['sampler_pmu']}")
    for line in problems:
        print(f"MISMATCH {line}")
    print(f"profiled with locked clocks {then.get('locked_clocks') or '?'}; run under "
          "android/locked_run_android.sh for the same platform state")
    print("OK" if not problems else f"{len(problems)} mismatch(es): re-profile or accept the difference")
    return 0 if not problems else 1


def android_serve(args: argparse.Namespace, profile: dict[str, Any]) -> int:
    d = android_device(args, profile)
    run_dir = run_dir_of(args, profile)
    kwargs = dict(host=args.host, port=args.port)
    runtime = full_environment(args.profile, profile, run_dir, args.trace, **kwargs)
    command = server_command(runtime_namespace(profile, run_dir, args.trace, **kwargs)) + args.extra
    if args.dry_run:
        print(shlex.join(command))
        return 0
    android_spawn(d, run_dir, "server", [android_libraries(profile),
                                         *(f"export {k}={shlex.quote(str(v))}" for k, v in runtime.items()),
                                         f"exec {shlex.join(command)}"])
    for _ in range(600):
        log = (d.read(f"{run_dir}/server.log") or b"").decode(errors="replace")
        if "server is listening" in log:
            print(f"llama-server listening on the phone at {args.host}:{args.port}; pages and logs in {run_dir}")
            return 0
        time.sleep(0.2)
    print(f"llama-server did not come up; see {run_dir}/server.log on the phone", file=sys.stderr)
    return 1


def android_launch(args: argparse.Namespace, profile: dict[str, Any]) -> int:
    from edgepace_android import ANDROID_SAMPLER
    d = android_device(args, profile)
    run_dir = run_dir_of(args, profile)
    root = profile["workload"]["device_root"]
    pages = run_dir_pages(run_dir)
    fill = {"task_page": str(pages["task"]), "progress_page": str(pages["progress"]),
            "run_dir": str(run_dir), "phase_dir": str(run_dir), "frames": str(args.frames)}
    tokens = args.command or shlex.split(profile["workload"]["foreground"])
    command = [f"{root}/bin/pmu_sampler", "--snapshot", str(pages["signal"]), "--no-mc",
               *shlex.split(ANDROID_SAMPLER), "--progress-page", str(pages["progress"]),
               "--prediction-csv", f"{run_dir}/prediction.csv",
               *(["--csv", f"{run_dir}/pmu_signal.csv"] if args.log else []),
               *profile["sampler_args"], "--", *(token.format(**fill) for token in tokens)]
    if args.dry_run:
        print(shlex.join(command))
        return 0
    android_spawn(d, run_dir, "foreground", [f"exec {shlex.join(command)}"])
    print(f"foreground started on the phone under the PMU sampler; logs in {run_dir}")
    return 0


def android_request(args: argparse.Namespace, profile: dict[str, Any]) -> int:
    d = android_device(args, profile)
    run_dir = run_dir_of(args, profile)
    root = profile["workload"]["device_root"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d.run(f"mkdir -p {run_dir}")
    d.write(f"{run_dir}/request_body.json", args.body.read_bytes())
    freshness = args.freshness_ms or profile["workload"]["freshness_ms"]
    d.run(f"{root}/bin/edgepace_client --port {args.port} --body {run_dir}/request_body.json "
          f"--request-page {run_dir}/request.page --count {args.count} --period-ms {args.period_ms} "
          f"--freshness-ms {freshness} --timeout-s {args.timeout} --csv {run_dir}/requests_{stamp}.csv",
          timeout=args.count * (args.timeout + args.period_ms / 1e3) + 30)
    rows = list(csv.DictReader(io.StringIO((d.read(f"{run_dir}/requests_{stamp}.csv") or b"").decode())))
    for r in rows:
        print(json.dumps({"request": int(r["request_id"]), "status": int(r["status"]),
                          "latency_ms": float(r["latency_ms"]), "fresh": r["fresh"] == "1",
                          "completion_tokens": int(r["completion_tokens"])}))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    if not on_android(profile):
        print("local serve/launch run in the foreground of their terminal; stop them there")
        return 0
    d = android_device(args, profile)
    run_dir = run_dir_of(args, profile)
    print(d.run(f"for f in {run_dir}/*.pid; do [ -f $f ] || continue; p=$(cat $f); "
                "kill $p 2>/dev/null && echo \"stopped $(basename $f .pid) ($p)\"; rm -f $f; done", check=False).strip())
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    import edgepace_workload
    if args.path.exists() and not args.force:
        raise SystemExit(f"{args.path} exists (--force to overwrite)")
    foreground = args.foreground or ("/path/to/app --frames {frames} --task-page {task_page} "
                                     "--progress-page {progress_page}")
    args.path.write_text(edgepace_workload.template(args.path, args.name, args.model or "/path/to/model.gguf",
                                                    foreground))
    print(f"wrote {args.path}; edit it, then: tools/edgepace profile {args.path}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Validate the workload, then run the inner steps under locked_run.sh."""
    import subprocess
    import edgepace_workload
    config = edgepace_workload.load(args.config)
    out = (args.output or ROOT / "runs" / f"profile_{config['name']}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    if args.dry_run:
        plan = config["llm"].get("plan") or out / "plan" / "cut_plan.json"
        if not config["llm"].get("plan"):
            print(shlex.join(["edgepace_plan.py", *edgepace_workload.plan_arguments(config, out / "plan")]))
        print(shlex.join(["edgepace_profile.py", *edgepace_workload.profile_arguments(
            config, out / "measure", Path(plan), out / "profile")]))
        return 0
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    inner = [sys.executable, str(Path(__file__).resolve()), "_profile-inner", str(args.config.resolve()), str(out)]
    android = config["platform"].get("device", "local") == "android"
    if args.no_lock:
        command, env = inner + ["--allow-unlocked-profile"], os.environ
    elif android:   # each step holds android/locked_run_android.sh on the phone
        command, env = inner, os.environ
    else:
        platform = config["platform"]
        env = {**os.environ, "VIDEO_CLOCKS": platform.get("video_clocks", "dynamic")}
        if platform.get("default_conf"):
            env["DEFAULT_CONF"] = platform["default_conf"]
        command = ["bash", str(ROOT / "scripts/locked_run.sh"), str(out / "locked"), "--", *inner]
    print(f"profiling {config['name']} -> {out}"
          + ("" if args.no_lock or android else f"\nprogress: tail -f {out / 'locked/command.log'}"),
          file=sys.stderr, flush=True)
    code = subprocess.call(command, env=env)
    profile_dir = out / "profile"
    if code != 0 or not (profile_dir / "profile.json").is_file():
        print(f"profile failed (exit {code}); see {out}", file=sys.stderr)
        return code or 1
    print((profile_dir / "report.md").read_text())
    print(f"profile directory: {profile_dir}\nnext: tools/edgepace check {profile_dir}; "
          f"tools/edgepace serve {profile_dir}; tools/edgepace launch {profile_dir}", file=sys.stderr)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """The profile's workload with the LLM unmanaged and under the profile, alternating
    (python/edgepace_eval.py), under the same clock lock as the profile."""
    import subprocess
    profile = load_profile(args.profile)
    w = profile["workload"]
    out = (args.output or ROOT / "runs" / f"eval_{profile['name']}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    command = [sys.executable, str(ROOT / "python/edgepace_eval.py"),
               "--model", w["model"], "--plan", str(args.profile.resolve() / w.get("plan", "cut_plan.json")),
               "--profile", str(args.profile.resolve() / "profile.json"),
               "--conditions", args.conditions, "--repeats", str(args.repeats), "--frames", str(args.frames),
               "--foreground", w["foreground"], "--request", w["request"],
               "--llm-period-ms", str(w["llm_period_ms"]), "--llm-align", w["llm_align"],
               "--max-tokens", str(w["max_tokens"]), "--freshness-ms", str(w["freshness_ms"]),
               "--parallel", str(w["parallel"]), "--llm-extra-streams", str(w["llm_extra_streams"]),
               "--period-us", str(w["period_us"]), "--deadline-us", str(w["deadline_us"]),
               "--context-size", str(w["context_size"]), "--batch-size", str(w["batch_size"]),
               "--ubatch-size", str(w["ubatch_size"]), "--output-dir", str(out)]
    if w.get("mmproj"):
        command += ["--mmproj", w["mmproj"], "--vision-device", w.get("vision_device") or "HTP0"]
    env = os.environ
    if on_android(profile):   # each phase holds android/locked_run_android.sh on the phone
        command += ["--device", "android", "--device-root", w["device_root"],
                    "--adb", args.adb or w.get("adb") or os.environ.get("EDGEPACE_ADB", "adb")]
        if args.adb_serial or w.get("adb_serial"):
            command += ["--adb-serial", args.adb_serial or w["adb_serial"]]
    elif args.no_lock:
        command += ["--allow-unlocked-profile"]
    else:
        command = ["bash", str(ROOT / "scripts/locked_run.sh"), str(out / "locked"), "--", *command]
    if args.dry_run:
        print(shlex.join(command))
        return 0
    out.mkdir(parents=True)
    print(f"evaluating {profile['name']} -> {out}", file=sys.stderr, flush=True)
    return subprocess.call(command, env=env)


def cmd_profile_inner(args: argparse.Namespace) -> int:
    """Runs on the locked platform: Fine Plan if needed, then the parameter profile."""
    import subprocess
    import edgepace_workload
    config = edgepace_workload.load(args.config)
    out = args.output
    extra = ["--allow-unlocked-profile"] if args.allow_unlocked_profile else []
    plan = Path(config["llm"]["plan"]) if config["llm"].get("plan") else None
    if plan is None:
        code = subprocess.call([sys.executable, str(ROOT / "python/edgepace_plan.py"),
                                *edgepace_workload.plan_arguments(config, out / "plan"), *extra])
        if code != 0:
            return code
        plan = out / "plan" / "cut_plan.json"
    code = subprocess.call([sys.executable, str(ROOT / "python/edgepace_profile.py"),
                            *edgepace_workload.profile_arguments(config, out / "measure", plan, out / "profile"),
                            *extra])
    if code == 0:
        shutil.copyfile(args.config, out / "profile" / "workload.toml")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edgepace", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="cmd", required=True)

    def with_profile(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("profile", type=Path, help="profile directory")
        return sub

    def with_adb(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--adb", help="phone profiles: adb executable (default: the profile's, $EDGEPACE_ADB, adb)")
        sub.add_argument("--adb-serial")

    def with_runtime(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--run-dir", type=Path,
                         help=f"shared pages (default {RUN_ROOT}/<profile name>, on a phone DEVICE_ROOT/run/<name>)")
        sub.add_argument("--host", default="127.0.0.1")
        sub.add_argument("--port", type=int, default=8080)
        with_adb(sub)

    def with_binaries(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--server", type=Path, default=DEFAULT_SERVER)
        sub.add_argument("--policy-library", type=Path, default=DEFAULT_POLICY_LIBRARY)
        sub.add_argument("--sampler", type=Path, default=DEFAULT_SAMPLER)

    sub = commands.add_parser("init", help="write a workload description template")
    sub.add_argument("path", type=Path, nargs="?", default=Path("edgepace.toml"))
    sub.add_argument("--name", default="my_workload")
    sub.add_argument("--model", help="GGUF model path")
    sub.add_argument("--foreground", help="foreground command with {frames} {task_page} {progress_page}")
    sub.add_argument("--force", action="store_true")
    sub.set_defaults(func=cmd_init)

    sub = commands.add_parser("profile", help="profile a workload description -> profile directory")
    sub.add_argument("config", type=Path, help="workload description (edgepace init)")
    sub.add_argument("--output", type=Path, help="run directory (default runs/profile_<name>_<time>)")
    sub.add_argument("--no-lock", action="store_true",
                     help="do not lock clocks (no sudo; parameters are then not comparable)")
    sub.add_argument("--dry-run", action="store_true", help="validate and print the steps only")
    sub.set_defaults(func=cmd_profile)

    sub = commands.add_parser("_profile-inner")    # internal: runs under locked_run.sh
    sub.add_argument("config", type=Path)
    sub.add_argument("output", type=Path)
    sub.add_argument("--allow-unlocked-profile", action="store_true")
    sub.set_defaults(func=cmd_profile_inner)

    sub = commands.add_parser("export", help="edgepace_profile.py run -> profile directory")
    sub.add_argument("run", type=Path, help="the run's output directory (with profile.json, profile_report.json)")
    sub.add_argument("output", type=Path, help="profile directory to write")
    sub.add_argument("--name", help="profile name (default: the run's directory name)")
    with_adb(sub)
    sub.set_defaults(func=cmd_export)

    with_profile("show", "print the profile report").set_defaults(func=cmd_show)

    sub = with_profile("evaluate", "measure the profile: its workload unmanaged vs under EdgePace")
    sub.add_argument("--conditions", default="unmanaged,edgepace_profile",
                     help="unmanaged, edgepace_profile (this profile), ... (edgepace_eval.py)")
    sub.add_argument("--repeats", type=int, default=2)
    sub.add_argument("--frames", type=int, default=1800, help="frames per phase")
    sub.add_argument("--output", type=Path, help="output directory (default runs/eval_<name>_<time>)")
    sub.add_argument("--no-lock", action="store_true", help="Jetson: do not lock clocks (not comparable)")
    sub.add_argument("--dry-run", action="store_true", help="print the command, do not run")
    with_adb(sub)
    sub.set_defaults(func=cmd_evaluate)

    sub = commands.add_parser("plans", help="list the cached Fine Plans (reused by plan/profile)")
    sub.add_argument("--cache", type=Path, action="append",
                     help="cache directory (default: build/plan-cache and build/plan-cache-android)")
    sub.set_defaults(func=cmd_plans)

    sub = with_profile("check", "compare platform and binaries with the profile")
    with_binaries(sub); with_adb(sub)
    sub.set_defaults(func=cmd_check)

    sub = with_profile("stop", "phone profiles: stop what serve/launch started on the device")
    sub.add_argument("--run-dir", type=Path)
    with_adb(sub)
    sub.set_defaults(func=cmd_stop)

    sub = with_profile("env", "print the llama-server environment")
    with_runtime(sub); with_binaries(sub)
    sub.add_argument("--trace", type=Path, help="EDGEPACE_LOG_PATH (policy decision trace)")
    sub.add_argument("--format", choices=("shell", "json"), default="shell")
    sub.set_defaults(func=cmd_env)

    sub = with_profile("serve", "exec llama-server with the profile")
    with_runtime(sub); with_binaries(sub)
    sub.add_argument("--trace", type=Path, help="EDGEPACE_LOG_PATH (policy decision trace)")
    sub.add_argument("--dry-run", action="store_true", help="print the command, do not start")
    sub.set_defaults(func=cmd_serve)   # extra llama-server arguments after --

    sub = with_profile("launch", "exec the PMU sampler with the foreground")
    with_runtime(sub); with_binaries(sub)
    sub.add_argument("--log", action="store_true", help="also write pmu_signal.csv to the run directory")
    sub.add_argument("--frames", type=int, default=1800, help="{frames} in the foreground command")
    sub.add_argument("--dry-run", action="store_true", help="print the command, do not start")
    sub.set_defaults(func=cmd_launch)  # foreground command after --

    sub = with_profile("request", "send LLM requests and publish them on the request page")
    with_runtime(sub)
    sub.add_argument("--body", type=Path, required=True, help="JSON request body")
    sub.add_argument("--count", type=int, default=1)
    sub.add_argument("--period-ms", type=float, default=0.0, help="start-to-start period for --count > 1")
    sub.add_argument("--freshness-ms", type=float, help="default: the profile's")
    sub.add_argument("--timeout", type=float, default=120.0)
    sub.add_argument("--print-reply", action="store_true")
    sub.set_defaults(func=cmd_request)

    argv = list(sys.argv[1:] if argv is None else argv)
    tail = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    args.extra = args.command = tail     # serve: llama-server arguments; launch: foreground
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
