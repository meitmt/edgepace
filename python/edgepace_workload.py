"""Workload description for ``edgepace profile``.

One TOML file names the platform, the foreground, the LLM load and the profiling
objective; ``edgepace init`` writes a commented template, ``edgepace profile``
turns it into a profile directory.  Placeholders in ``foreground.command``:
``{frames}`` (the foreground must stop after that many frames), ``{task_page}``
and ``{progress_page}`` (the pages it publishes), and optionally ``{phase_dir}``
(a per-run directory for the application's own outputs); ``{repo}`` is this
repository.  A relative executable is relative to the workload file.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path
from typing import Any

try:                                   # Python >= 3.11
    import tomllib
except ModuleNotFoundError:            # pragma: no cover - older Pythons
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = '''# EdgePace workload description.  `tools/edgepace profile {path}` measures the
# parameters for it on this device (~10-20 min, needs sudo for clock locking)
# and writes a profile directory for `edgepace serve` / `edgepace launch`.
# Relative paths are relative to this file.

name = "{name}"                     # profile name; also the run directory /dev/shm/edgepace/<name>

[platform]
power_mode = "MAXN_SUPER"           # profile and deployment must use the same mode and clocks
default_conf = "{default_conf}"     # `jetson_clocks --store` reference, restored afterwards
video_clocks = "dynamic"            # dynamic: NVDEC/VIC under their governor; locked: pinned at max
# A rooted Android phone instead (android/README): every path in [foreground]
# and llm.model is then a device path; the lock is android/locked_run_android.sh.
# device = "android"
# adb = "adb"
# device_root = "/data/local/tmp/edgepace"

[foreground]
# The real-time application.  It must publish the task and progress pages
# (edgepace foreground header) and stop after {{frames}} frames; {{phase_dir}} is a
# per-run directory for its own outputs.
command = "{foreground}"
period_ms = 33.333                  # release period of a job (frame)
deadline_ms = 33.333                # relative deadline of a job

[llm]
model = "{model}"
plan = ""                           # cut_plan.json; empty: profile the Fine Plan first (cached)
request = "{request}"               # JSON body of one request, as your client sends it
period_ms = 1600                    # request period; 0 = back-to-back
align = "frame"                     # frame: send right after a foreground frame completes; none
max_tokens = 4                      # output tokens per request
freshness_ms = 1800                 # an answer older than this is stale
parallel = 1                        # server slots
extra_streams = 0                   # extra back-to-back clients beside the tracked one
context_size = 2048

[profile]
mode = "quick"                      # quick: foreground alone + unmanaged, quick_frames each (~1 min);
                                    #   sweep: also measure H over `horizons` (long)
quick_frames = 600
horizon_ms = 20                     # quick mode: the warning horizon H
objective = "miss"                  # sweep mode: miss: fewest misses at >= min_retention; throughput: most LLM
min_retention = 0.6                 #   throughput at <= miss_budget misses
miss_budget = 0.02
horizons = [12, 16, 20, 24]         # candidate warning horizons H (ms)
frames = 1800                       # frames of the solo phase
sweep_frames = 900                  # frames per sweep phase (more = less noise, slower)
# miss_tolerance = 0.002            # misses within this of the fewest count as equal (default 2 frames)
stress_binary = "{stress}"          # GPU memory stressor for the S check; empty to skip
# stress_llm_request = "text_256tok.json"   # or the LLM itself as the load (phones)
'''

REQUIRED = {"name": str, "platform": dict, "foreground": dict, "llm": dict, "profile": dict}
FOREGROUND_PLACEHOLDERS = ("{frames}", "{task_page}", "{progress_page}")


def template(path: Path, name: str, model: str, foreground: str) -> str:
    stress = ROOT / "build/stress/gpu_stress"     # scripts/build.sh
    return TEMPLATE.format(
        path=path, name=name, model=model, foreground=foreground.replace('"', '\\"'),
        default_conf=ROOT / "platform/orin_nano_super_default.conf",
        request=ROOT / "requests/text_4tok.json",
        stress=stress if stress.exists() else "")


def load(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise SystemExit("reading workload files needs Python >= 3.11 or `pip install tomli`")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    problems = [f"missing [{key}]" if kind is dict else f"missing {key}"
                for key, kind in REQUIRED.items() if not isinstance(config.get(key), kind)]
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    base = path.resolve().parent

    def resolve(section: str, key: str) -> None:
        value = config[section].get(key, "")
        if value:
            config[section][key] = str((base / value).resolve()) if not Path(value).is_absolute() else value

    for section, key in (("platform", "default_conf"), ("llm", "model"), ("llm", "plan"),
                         ("llm", "request"), ("llm", "mmproj"), ("profile", "stress_binary"),
                         ("profile", "stress_llm_request")):
        resolve(section, key)
    fg, llm, prof = config["foreground"], config["llm"], config["profile"]
    android = config["platform"].get("device", "local") == "android"
    if android:
        config["platform"].setdefault("device_root", "/data/local/tmp/edgepace")
        config["platform"].setdefault("adb", "adb")
        if not config["llm"].get("model", "").startswith("/"):
            problems.append("llm.model must be an absolute device path on a phone")
        if config["llm"].get("mmproj") and not config["llm"]["mmproj"].startswith("/"):
            problems.append("llm.mmproj must be an absolute device path on a phone")
    # {repo}: this repository (e.g. {repo}/media/test.mp4); expanded here, the other
    # placeholders per run
    command = fg.get("command", "").strip().replace("{repo}", str(ROOT))
    fg["command"] = command
    missing = [p for p in FOREGROUND_PLACEHOLDERS if p not in command]
    if missing:
        problems.append(f"foreground.command lacks {', '.join(missing)}")
    executable = shlex.split(command)[0] if command else ""
    if not android and "/" in executable and not Path(executable).is_absolute():
        executable = str((base / executable).resolve())
        fg["command"] = executable + " " + command.split(None, 1)[1] if " " in command else executable
    if not android and executable and not (Path(executable).exists() or shutil.which(executable)):
        problems.append(f"foreground executable not found: {executable}")
    for label, value in (("llm.model", None if android else llm.get("model")), ("llm.request", llm.get("request")),
                         ("llm.plan", llm.get("plan")), ("platform.default_conf", config["platform"].get("default_conf")),
                         # only the sweep runs the stressor (quick mode never does)
                         ("profile.stress_binary", prof.get("stress_binary") if prof.get("mode", "quick") == "sweep" else None),
                         ("profile.stress_llm_request", prof.get("stress_llm_request"))):
        if value and not Path(value).exists():
            problems.append(f"{label} not found: {value}")
    if not llm.get("model"):
        problems.append("llm.model is empty")
    if android and prof.get("stress_binary"):
        problems.append("profile.stress_binary: there is no phone GPU memory stressor; use stress_llm_request")
    if prof.get("stress_binary") and prof.get("stress_llm_request"):
        problems.append("profile: stress_binary and stress_llm_request are alternatives")
    if prof.get("mode", "quick") not in ("quick", "sweep"):
        problems.append("profile.mode must be quick or sweep")
    if prof.get("objective", "miss") not in ("miss", "throughput"):
        problems.append("profile.objective must be miss or throughput")
    if llm.get("align", "frame") not in ("frame", "none"):
        problems.append("llm.align must be frame or none")
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    return config


def profile_arguments(config: dict[str, Any], output_dir: Path, plan: Path, profile_dir: Path) -> list[str]:
    """edgepace_profile.py arguments for this workload."""
    fg, llm, prof = config["foreground"], config["llm"], config["profile"]
    command = fg["command"].replace("{progress_page}", "{phase_dir}/progress.page")
    args = [
        "--model", llm["model"], "--plan", str(plan), "--output-dir", str(output_dir),
        "--foreground", command,
        "--period-us", str(round(float(fg.get("period_ms", 33.333)) * 1000)),
        "--deadline-us", str(round(float(fg.get("deadline_ms", fg.get("period_ms", 33.333))) * 1000)),
        "--request", llm.get("request") or str(ROOT / "requests/text_4tok.json"),
        "--llm-period-ms", str(llm.get("period_ms", 1600)), "--llm-align", llm.get("align", "frame"),
        "--max-tokens", str(llm.get("max_tokens", 4)), "--freshness-ms", str(llm.get("freshness_ms", 1800)),
        "--parallel", str(llm.get("parallel", 1)), "--llm-extra-streams", str(llm.get("extra_streams", 0)),
        "--context-size", str(llm.get("context_size", 2048)),
        "--required-power-mode", config["platform"].get("power_mode", "MAXN_SUPER"),
        "--objective", prof.get("objective", "miss"), "--min-retention", str(prof.get("min_retention", 0.6)),
        "--miss-budget", str(prof.get("miss_budget", 0.02)),
        "--horizons", ",".join(f"{h:g}" for h in prof.get("horizons", [12, 16, 20, 24])),
        "--frames", str(prof.get("frames", 1800)), "--sweep-frames", str(prof.get("sweep_frames", 900)),
        "--profile-dir", str(profile_dir), "--profile-name", config["name"],
    ]
    if prof.get("stress_binary") and prof.get("mode", "quick") == "sweep":
        args += ["--stress-binary", prof["stress_binary"]]
    if prof.get("stress_llm_request"):
        args += ["--stress-llm-request", prof["stress_llm_request"],
                 "--stress-llm-streams", str(prof.get("stress_llm_streams", 4))]
    if "miss_tolerance" in prof:
        args += ["--miss-tolerance", str(prof["miss_tolerance"])]
    args += ["--mode", prof.get("mode", "quick"), "--quick-frames", str(prof.get("quick_frames", 600)),
             "--horizon-ms", str(prof.get("horizon_ms", 20)), *vision_arguments(config),
             *(["--interference"] if prof.get("interference") else [])]
    return args + device_arguments(config)


def device_arguments(config: dict[str, Any]) -> list[str]:
    """--device android and its adb settings; none for a local platform."""
    p = config["platform"]
    if p.get("device", "local") != "android":
        return []
    return ["--device", "android", "--adb", p["adb"], "--device-root", p["device_root"],
            *(["--adb-serial", p["adb_serial"]] if p.get("adb_serial") else [])]


def plan_arguments(config: dict[str, Any], output_dir: Path) -> list[str]:
    """edgepace_plan.py plan arguments: the Fine Plan depends on model, context,
    batch; its SLO check on deadline, freshness and output length."""
    fg, llm = config["foreground"], config["llm"]
    return ["plan", "--model", llm["model"], "--output-dir", str(output_dir),
            "--context-size", str(llm.get("context_size", 2048)),
            "--deadline-us", str(round(float(fg.get("deadline_ms", fg.get("period_ms", 33.333))) * 1000)),
            "--period-us", str(round(float(fg.get("period_ms", 33.333)) * 1000)),
            "--freshness-ms", str(llm.get("freshness_ms", 1800)), "--max-tokens", str(llm.get("max_tokens", 4)),
            "--required-power-mode", config["platform"].get("power_mode", "MAXN_SUPER"),
            # a VLM's plan is profiled with the workload's own (image) request
            *(["--request", llm["request"]] if llm.get("mmproj") and llm.get("request") else []),
            *(["--allow-uncuttable"] if llm.get("allow_uncuttable") else []),
            *vision_arguments(config), *device_arguments(config)]


def vision_arguments(config: dict[str, Any]) -> list[str]:
    """--mmproj / --vision-boundaries of a VLM workload; none for a text LLM."""
    llm = config["llm"]
    # batch sizes and text prefill cut points (any model)
    common = [*(["--batch-size", str(llm["batch_size"])] if "batch_size" in llm else []),
              *(["--ubatch-size", str(llm["ubatch_size"])] if "ubatch_size" in llm else []),
              *(["--prefill-boundaries", ",".join(llm["prefill_boundaries"])] if llm.get("prefill_boundaries") else [])]
    if not llm.get("mmproj"):
        return common
    return [*common, "--mmproj", llm["mmproj"],
            *(["--vision-boundaries", ",".join(llm["vision_boundaries"])] if llm.get("vision_boundaries") else []),
            *(["--vision-device", llm["vision_device"]] if llm.get("vision_device") else [])]
