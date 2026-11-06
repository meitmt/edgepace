#!/usr/bin/env python3
"""Fine Plan profile-first fine graph cutting on the EdgePace backend.

Fine Plan decides the cut geometry at startup instead of hard-coding it:

1. inspect the GGUF model (dense decoder-only only);
2. expose the finest backend-legal boundaries (prefill 2L+2, decode L+2);
3. run three real requests through the EdgePace backend with ``profile_static``;
4. keep every legal boundary, use the maximum observed submit-to-complete
   duration per boundary, and isolate the final output tail;
5. cache the plan by model/config identity (never by priority or SLO) and
   assess the external SLO against it without changing the geometry.  The key
 is the model's content (tensor layout and hyperparameters, not its
   path or file time), the graph shape (context, batch, ubatch), the cut points
   and the device; the runtime build is recorded and a mismatch is reported.
   ``tools/edgepace plans`` lists the cache.

The resulting maps drive the EdgePace backend at runtime.  Planning reuses
``edgepace_fineplan`` unchanged.

Subcommands::

    plan   profile (or reuse the cache) and write cut_plan.json
    env    print the EdgePace runtime environment for a cut_plan.json
    serve  plan, then exec llama-server with the runtime environment
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from edgepace_fineplan import (
    LEGACY_PREFILL_BOUNDARIES,
    SCHEMA_VERSION as CUT_SCHEMA_VERSION,
    assess_slo,
    cache_name,
    fine_map,
    inspect_gguf,
    inspect_mmproj,
    legacy_cache_name,
    make_manifest,
    profile_from_trace,
    session_manifest,
    validate_slo,
    vision_fine_map,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = ROOT / "build/llama/bin/llama-server"
DEFAULT_POLICY_LIBRARY = ROOT / "build/policy/libedgepace_policy.so"
DEFAULT_CUT_CACHE = ROOT / "build/plan-cache"
DEFAULT_REQUEST = Path(__file__).resolve().parents[1] / "requests/text_4tok.json"
PROFILE_REQUESTS = 3
CONTROL_PREFIXES = ("EDGEPACE_", "EDGEPACE_", "EDGEPACE_")


def resolved_priority(name: str, bias: float | None) -> dict[str, Any]:
    """Registered Fine Plan priority presets; priority never changes cut geometry."""
    if name == "miss":
        return {"name": name, "bias": 1.0, "stall_safe_max": 0.65,
                "fresh_rescue_threshold": 0.80}
    if name == "fresh":
        return {"name": name, "bias": 0.0, "stall_safe_max": 0.65,
                "fresh_rescue_threshold": 0.22}
    if name == "balanced":
        return {"name": name, "bias": 0.5, "stall_safe_max": 0.70,
                "fresh_rescue_threshold": 0.80}
    if bias is None or not 0.0 <= bias <= 1.0:
        raise ValueError("manual priority requires --priority-bias in [0, 1]")
    return {
        "name": name,
        "bias": bias,
        "stall_safe_max": 0.80 - 0.15 * bias,
        "fresh_rescue_threshold": 0.22 + 0.58 * bias,
    }


def server_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.server), "-m", str(args.model), "--alias", "local-model",
        "-ngl", "99", "-c", str(args.context_size),
        "-b", str(args.batch_size), "-ub", str(args.ubatch_size),
        "--parallel", str(getattr(args, "parallel", 1)), "--threads", "4", "--threads-batch", "4",
        "--host", args.host, "--port", str(args.port),
        "--no-webui", "--no-warmup",
    ]
    if not args.prompt_cache:
        command.extend(["--no-cache-prompt", "--cache-ram", "0"])
    if getattr(args, "mmproj", None):   # a VLM's vision encoder
        command.extend(["--mmproj", str(args.mmproj)])
    if getattr(args, "device", "local") == "android":
        # the LLM on the NPU only; the Adreno OpenCL backend is there for the
        # vision encoder (MTMD_BACKEND_DEVICE, edgepace_android.DEVICE_LIBRARIES)
        command.extend(["--device", "HTP0"])
    return command


def prefill_boundaries(args: argparse.Namespace) -> tuple[str, ...]:
    """Text prefill cut points per layer: --prefill-boundaries, else the
    historical pair, for a VLM too (its text cut stays the text LLM's; a smaller
    --ubatch-size keeps its image ubatches' quanta short instead)."""
    names = tuple(n for n in (getattr(args, "prefill_boundaries", "") or "").split(",") if n)
    return names or LEGACY_PREFILL_BOUNDARIES


def vision_info(args: argparse.Namespace) -> Any:
    """The --mmproj vision encoder with its cut points, or None."""
    if not getattr(args, "mmproj", None):
        return None
    names = tuple(n for n in (getattr(args, "vision_boundaries", "") or "").split(",") if n)
    if getattr(args, "device", "local") == "android":   # the phone's mmproj
        from edgepace_android import mmproj_info
        return mmproj_info(args, names)
    return inspect_mmproj(Path(args.mmproj), names)


def clean_environment() -> dict[str, str]:
    """Inherited control variables must not leak into a profile or a run."""
    return {
        name: value for name, value in os.environ.items()
        if not name.startswith(CONTROL_PREFIXES)
    }


def profile_environment(
    args: argparse.Namespace, layer_count: int, trace: Path,
) -> dict[str, str]:
    # EdgePace skips the policy entirely unless EDGEPACE_CONTROL_REQUIRED is set, and its
    # policy lives in an external DSO.  Without both, no quantum is timed and
    # the profile trace stays empty.  profile_static then admits one quantum
    # at a time (outstanding limit 1), so submit-to-complete is per quantum.
    vision = vision_info(args)
    prefill = prefill_boundaries(args)
    return {
        **({"EDGEPACE_VISION_BOUNDARIES": ",".join(vision.boundaries),
            "EDGEPACE_VISION_MAP": vision_fine_map(vision)} if vision is not None else {}),
        **({"EDGEPACE_PREFILL_BOUNDARIES": ",".join(prefill)} if prefill != LEGACY_PREFILL_BOUNDARIES else {}),
        "EDGEPACE_POLICY": "profile_static",
        "EDGEPACE_POLICY_LIBRARY": str(args.policy_library),
        "EDGEPACE_CONTROL_REQUIRED": "1",
        "EDGEPACE_MODEL_LAYER_COUNT": str(layer_count),
        "EDGEPACE_DECODE_LAYER_COUNT": str(layer_count),
        "EDGEPACE_PREFILL_SUBLAYER_BOUNDARIES": "1",
        "EDGEPACE_PREFILL_MAP": fine_map(layer_count, "prefill", prefill),
        "EDGEPACE_DECODE_MAP": fine_map(layer_count, "decode"),
        "EDGEPACE_LOG_PATH": str(trace),
        "EDGEPACE_LOG_FLUSH_ROWS": "1",
    }


def runtime_environment(
    args: argparse.Namespace, manifest: dict[str, Any],
) -> dict[str, str]:
    priority = resolved_priority(args.priority, args.priority_bias)
    layer_count = int(manifest["model"]["layer_count"])
    prefill_chunks = manifest["phases"]["prefill"]["chunks"]
    decode_chunks = manifest["phases"]["decode"]["chunks"]
    prefill_total = sum(float(value["estimated_runtime_ms"]) for value in prefill_chunks)
    decode_total = sum(float(value["estimated_runtime_ms"]) for value in decode_chunks)
    vision = manifest["phases"].get("vision")
    vision_total = sum(float(value["estimated_runtime_ms"]) for value in vision["chunks"]) if vision else 0.0
    prefill_names = manifest["phases"]["prefill"].get("boundaries")
    result = {
        **({"EDGEPACE_VISION_BOUNDARIES": ",".join(vision["boundaries"]),
            "EDGEPACE_VISION_MAP": vision["encoded_map"]} if vision else {}),
        **({"EDGEPACE_PREFILL_BOUNDARIES": ",".join(prefill_names)} if prefill_names else {}),
        "EDGEPACE_POLICY": args.policy,
        "EDGEPACE_POLICY_LIBRARY": str(args.policy_library),
        "EDGEPACE_CONTROL_REQUIRED": "1",
        "EDGEPACE_MODEL_LAYER_COUNT": str(layer_count),
        "EDGEPACE_DECODE_LAYER_COUNT": str(layer_count),
        "EDGEPACE_PREFILL_SUBLAYER_BOUNDARIES": "1",
        "EDGEPACE_PREFILL_MAP": manifest["phases"]["prefill"]["encoded_map"],
        "EDGEPACE_DECODE_MAP": manifest["phases"]["decode"]["encoded_map"],
        "EDGEPACE_TOTAL_SERVICE_MS": str(vision_total + prefill_total + args.max_tokens * decode_total),
        "EDGEPACE_FIRST_DECODE_MS": str(decode_total),
        "EDGEPACE_LATER_DECODE_1_MS": str(decode_total),
        "EDGEPACE_LATER_DECODE_2_MS": str(decode_total),
        "EDGEPACE_LATER_DECODE_3_MS": str(decode_total),
        "EDGEPACE_FOREGROUND_PERIOD_MS": str(args.period_us / 1000.0),
        "EDGEPACE_FOREGROUND_DEADLINE_MS": str(args.deadline_us / 1000.0),
        "EDGEPACE_FOREGROUND_RESPONSE_MS": str(min(args.deadline_us / 1250.0, 30.0)),
        "EDGEPACE_ONLINE_EFFECT_GATE": "1",
        "EDGEPACE_ONLINE_DEMAND_SAFE_MAX": "0.35",
        "EDGEPACE_ONLINE_STALL_SAFE_MAX": str(priority["stall_safe_max"]),
        "EDGEPACE_ONLINE_FRESH_STALL_SAFE_MAX": "0.80",
        "EDGEPACE_ONLINE_FRESH_RESCUE_THRESHOLD": str(priority["fresh_rescue_threshold"]),
        "EDGEPACE_ONLINE_DEADLINE_CHARGE_CAP_MS": "1.5",
        "EDGEPACE_ONLINE_RELEASE_CROSS_ALLOW_MS": "8.0",
        "EDGEPACE_ONLINE_MAX_HOLD_MS": "-1",
        "EDGEPACE_FRESHNESS_WARNING_MS": str(args.freshness_ms * 2.0 / 3.0),
        "EDGEPACE_PRIORITY_BIAS": str(priority["bias"]),
    }
    # Shared-memory pages are produced by the foreground/sampler side; the
    # policy treats an absent page as "no valid signal", never as safe.
    optional = {
        "EDGEPACE_TASK_PAGE": args.task_page,
        "EDGEPACE_REQUEST_PAGE": args.request_page,
        "EDGEPACE_SIGNAL_PAGE": args.signal_page,
        "EDGEPACE_ACTMON_PAGE": args.actmon_page,
        "EDGEPACE_LOG_PATH": args.trace,
    }
    result.update({name: str(value) for name, value in optional.items() if value})
    return result


def wait_server(host: str, port: int, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {timeout_s}s")


def post_request(host: str, port: int, payload: dict[str, Any], output: Path) -> None:
    endpoint = "/v1/completions" if "prompt" in payload else "/v1/chat/completions"
    request = urllib.request.Request(
        f"http://{host}:{port}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start_ns = time.monotonic_ns()
    status, body, error = 0, None, None
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status, error = exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        error = str(exc)
    output.write_text(json.dumps({
        "http_status": status,
        "latency_ms": (time.monotonic_ns() - start_ns) / 1e6,
        "error": error,
        "response": body,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not 200 <= status < 300:
        raise RuntimeError(f"profile request failed: status={status}, error={error}")


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def platform_snapshot() -> dict[str, Any]:
    """Unprivileged view of the Jetson power mode and CPU/GPU frequency pins.

    EMC lock state needs root (debugfs); scripts/locked_run.sh verifies it.
    """
    try:
        mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                              timeout=10.0).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        mode = []
    cpus = {}
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq")):
        cpus[cpu.parent.name] = {
            "min": read_text(cpu / "scaling_min_freq"),
            "max": read_text(cpu / "scaling_max_freq"),
        }
    gpu = {}
    for devfreq in Path("/sys/devices/platform").glob("bus@0/*.gpu/devfreq/*.gpu"):
        gpu = {"min": read_text(devfreq / "min_freq"),
               "max": read_text(devfreq / "max_freq")}
    pinned = (
        bool(cpus) and all(v["min"] is not None and v["min"] == v["max"] for v in cpus.values())
        and bool(gpu) and gpu["min"] is not None and gpu["min"] == gpu["max"]
    )
    return {
        "power_mode": mode[0].replace("NV Power Mode:", "").strip() if mode else None,
        "cpu": cpus,
        "gpu": gpu,
        "cpu_gpu_pinned": pinned,
    }


def require_locked_platform(args: argparse.Namespace) -> dict[str, Any]:
    # Profiled quantum runtimes become the cut geometry and the service
    # estimates; measuring them under DVFS would bake frequency noise in.
    if getattr(args, "device", "local") == "android":   # lock held on the phone
        from edgepace_android import require_locked_platform as require_device_lock
        return require_device_lock(args)
    snapshot = platform_snapshot()
    locked = snapshot["power_mode"] == args.required_power_mode and snapshot["cpu_gpu_pinned"]
    if not locked and not args.allow_unlocked_profile:
        raise RuntimeError(
            f"refusing to profile on an unlocked platform (power_mode="
            f"{snapshot['power_mode']}, cpu_gpu_pinned={snapshot['cpu_gpu_pinned']}); "
            "run under scripts/locked_run.sh or pass --allow-unlocked-profile"
        )
    snapshot["locked"] = locked
    return snapshot


def profile_plan(args: argparse.Namespace, info: Any, output_dir: Path) -> dict[str, Any]:
    if getattr(args, "device", "local") == "android":
        from edgepace_android import profile_plan_android
        return profile_plan_android(args, info, output_dir)
    platform = require_locked_platform(args)
    trace = output_dir / "cut_profile_trace.csv"
    trace.unlink(missing_ok=True)
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    environment = clean_environment()
    environment.update(profile_environment(args, info.layer_count, trace))
    command = server_command(args)
    (output_dir / "cut_profile_server.command.json").write_text(
        json.dumps({"command": command, "environment": profile_environment(
            args, info.layer_count, trace)}, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "cut_profile_server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=environment)
        try:
            wait_server(args.host, args.port, server, args.server_timeout)
            for index in range(PROFILE_REQUESTS):
                post_request(args.host, args.port, payload,
                             output_dir / f"cut_profile_request_{index + 1}.json")
        finally:
            terminate(server)
    vision = vision_info(args)
    prefill = prefill_boundaries(args)
    manifest = make_manifest(
        info, profile_from_trace(trace, info.layer_count, vision, prefill),
        args.context_size, args.batch_size, args.ubatch_size, vision, prefill,
    )
    manifest["profile_backend"] = {
        "runtime": "edgepace",
        "server": str(args.server.resolve()),
        "policy_library": str(args.policy_library.resolve()),
        "policy": "profile_static",
        "requests": PROFILE_REQUESTS,
        "request_payload": str(args.request.resolve()),
        "platform": platform,
    }
    return manifest


def platform_identity(args: argparse.Namespace) -> str:
    """The device a plan is timed on: a Jetson's device-tree model, a phone's
    product model and SoC, else the machine and CPU."""
    if getattr(args, "device", "local") == "android":
        from edgepace_android import device
        identity = "android:" + " ".join(device(args).run(
            "getprop ro.product.model; getprop ro.board.platform", check=False).split())
        # the vision encoder's timings are those of its backend
        return identity + (f" vision:{args.vision_device}" if getattr(args, "mmproj", None) else "")
    model = read_text(Path("/proc/device-tree/model"))
    if model:
        return model.replace("\x00", "").strip()
    cpu = next((line.split(":", 1)[1].strip() for line in
                (read_text(Path("/proc/cpuinfo")) or "").splitlines() if line.startswith("model name")), "")
    return f"{os.uname().machine} {cpu}".strip()


def runtime_fingerprint(args: argparse.Namespace) -> str:
    """The llama.cpp / policy build a plan is timed with: sha256 over the
    server's ggml/llama/mtmd libraries and the policy library."""
    import hashlib
    digest = hashlib.sha256()
    if getattr(args, "device", "local") == "android":
        from edgepace_android import device
        names = ("lib/libggml-base.so lib/libggml-hexagon.so lib/libggml-htp-v79.so lib/libedgepace_policy.so "
                 "bin/llama-server lib/libggml-opencl.so")   # the vision encoder's backend
        digest.update(device(args).run(f"cd {args.device_root} && sha256sum {names} 2>/dev/null", check=False).encode())
        return digest.hexdigest()[:16]
    lib_dir = Path(args.server).resolve().parent
    for name in ("libggml-base.so", "libggml-cuda.so", "libggml-cpu.so", "libllama.so", "libmtmd.so"):
        path = lib_dir / name
        if path.exists():
            digest.update(name.encode() + path.resolve().read_bytes())
    policy = Path(args.policy_library)
    if policy.exists():
        digest.update(policy.resolve().read_bytes())
    return digest.hexdigest()[:16]


def load_or_profile(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], bool]:
    if getattr(args, "device", "local") == "android":   # model and timings of the phone
        from edgepace_android import model_info
        info = model_info(args)
        if args.cut_plan_cache == DEFAULT_CUT_CACHE:
            args.cut_plan_cache = DEFAULT_CUT_CACHE.with_name("plan-cache-android")
    else:
        info = inspect_gguf(args.model)
    args.cut_plan_cache.mkdir(parents=True, exist_ok=True)
    # keyed by the model's content and the device; an entry under the old key
    # (model path and file time) is adopted once
    vision, prefill = vision_info(args), prefill_boundaries(args)
    shape = (args.context_size, args.batch_size, args.ubatch_size)
    cached = args.cut_plan_cache / cache_name(info, *shape, vision, prefill, platform_identity(args))
    legacy = args.cut_plan_cache / legacy_cache_name(info, *shape, vision, prefill)
    if not cached.is_file() and legacy.is_file() and not args.recalibrate_cut:
        adopted = json.loads(legacy.read_text(encoding="utf-8"))
        # the cache directory is this device's; the runtime it was timed with is unknown
        adopted.setdefault("profile_backend", {}).setdefault("device", platform_identity(args))
        adopted["model"]["fingerprint"] = info.fingerprint
        cached.write_text(json.dumps(adopted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"cut plan cache: adopted {legacy.name} as {cached.name}", file=sys.stderr)
    cache_hit = cached.is_file() and not args.recalibrate_cut
    runtime = runtime_fingerprint(args)
    if cache_hit:
        base_manifest = json.loads(cached.read_text(encoding="utf-8"))
        if base_manifest.get("schema_version") != CUT_SCHEMA_VERSION:
            raise RuntimeError(f"invalid cached Fine Plan schema: {cached}")
        timed_with = base_manifest.get("profile_backend", {}).get("runtime_fingerprint")
        if timed_with != runtime:
            print(f"cut plan cache: {cached.name} was timed with runtime "
                  f"{timed_with or 'unrecorded (before)'}, now {runtime}; reused — pass "
                  "--recalibrate-cut to time it again", file=sys.stderr)
    else:
        base_manifest = profile_plan(args, info, output_dir)
        base_manifest["profile_backend"].update(runtime_fingerprint=runtime, device=platform_identity(args))
        cached.write_text(json.dumps(base_manifest, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

    assessment = assess_slo(base_manifest, args.deadline_us, args.freshness_ms,
                            args.max_tokens, getattr(args, "allow_uncuttable", False))
    validate_slo(assessment)
    manifest = session_manifest(base_manifest, assessment)
    manifest["cache"] = {"hit": cache_hit, "path": str(cached)}
    (output_dir / "cut_plan.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, cache_hit


def describe(manifest: dict[str, Any]) -> str:
    prefill = manifest["phases"]["prefill"]
    decode = manifest["phases"]["decode"]
    vision = manifest["phases"].get("vision")
    vision_text = "" if not vision else (
        f"vision {len(vision['chunks'])} quanta from {vision['candidate_count']} boundaries "
        f"(body {vision['body_min_runtime_ms']:.2f}-{vision['body_max_runtime_ms']:.2f} ms, "
        f"{vision['body_quanta_in_range']}/{vision['body_quanta']} in "
        f"{vision['quantum_range_ms'][0]:g}-{vision['quantum_range_ms'][1]:g} ms, tail "
        f"{vision['terminal_runtime_ms']:.2f} ms), ")
    if "body_quanta_in_range" in prefill:   # merged prefill
        vision_text += (f"prefill {list(prefill['boundaries'])} -> {prefill['body_quanta_in_range']}/"
                        f"{prefill['body_quanta']} in 1-4 ms (min {prefill['body_min_runtime_ms']:.2f}), ")
    return (
        f"Fine Plan {manifest['model']['architecture']}/{manifest['model']['layer_count']}L "
        + vision_text +
        f"cache_hit={manifest['cache']['hit']}: "
        f"prefill {prefill['candidate_count']} quanta "
        f"(body max {prefill['body_max_runtime_ms']:.3f} ms, tail "
        f"{prefill['terminal_runtime_ms']:.3f} ms), "
        f"decode {decode['candidate_count']} quanta "
        f"(body max {decode['body_max_runtime_ms']:.3f} ms, tail "
        f"{decode['terminal_runtime_ms']:.3f} ms)"
        + "".join(f"\nadvisory: {text}" for text in manifest.get("slo_assessment", {}).get("advisories", []))
    )


def add_vision_arguments(parser: argparse.ArgumentParser) -> None:
    """a VLM's vision encoder and its cut points."""
    parser.add_argument("--mmproj", type=Path, help="VLM vision encoder (llama.cpp mmproj GGUF)")
    parser.add_argument("--prefill-boundaries", default="",
                        help="text prefill cut points per layer (llama callback names, graph "
                             "order); default ffn_inp,l_out, every boundary kept; finer names, e.g. "
                             "ffn_inp,ffn_gate,ffn_up,l_out, are merged into 1-4 ms quanta")
    parser.add_argument("--vision-boundaries", default="",
                        help="clip callback names (non-view nodes) cut inside each vision layer, in "
                             "graph order; default by projector type; EDGEPACE_VISION_DUMP=path lists a graph's")
    parser.add_argument("--vision-device", default="HTP0",
                        help="phone: the vision encoder's backend (MTMD_BACKEND_DEVICE), HTP0 "
                             "(Qwen3-VL-4B F16 mmproj: 0.50 s per 448x448 image, quanta p50 1.8 ms) or "
                             "GPUOpenCL (Adreno: 1.40 s, p50 9.6 ms)")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("plan_run"))
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--policy-library", type=Path, default=DEFAULT_POLICY_LIBRARY)
    parser.add_argument("--cut-plan-cache", type=Path, default=DEFAULT_CUT_CACHE)
    parser.add_argument("--recalibrate-cut", action="store_true")
    parser.add_argument("--allow-uncuttable", action="store_true",
                        help="quanta of a single legal span over their limits (a phone VLM's CPU "
                             "LM head and vision patch embedding) are advisories, not errors")
    parser.add_argument("--required-power-mode", default="MAXN_SUPER")
    parser.add_argument("--allow-unlocked-profile", action="store_true",
                        help="profile without MAXN and CPU/GPU pinning (data not comparable)")
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST,
                        help="JSON body of each startup profile request")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--server-timeout", type=float, default=180.0)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--prompt-cache", action="store_true")
    parser.add_argument("--deadline-us", type=int, default=33333)
    parser.add_argument("--period-us", type=int, default=33333)
    parser.add_argument("--freshness-ms", type=float, default=1800.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    add_vision_arguments(parser)
    from edgepace_android import add_device_arguments   # (imported late: it imports this module)
    add_device_arguments(parser)


def add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", default="adaptive_slo_online",
                        help="EDGEPACE_POLICY understood by the policy DSO")
    parser.add_argument("--priority", choices=("miss", "fresh", "balanced", "manual"),
                        default="balanced")
    parser.add_argument("--priority-bias", type=float)
    parser.add_argument("--task-page", type=Path)
    parser.add_argument("--request-page", type=Path)
    parser.add_argument("--signal-page", type=Path)
    parser.add_argument("--actmon-page", type=Path)
    parser.add_argument("--trace", type=Path, help="EDGEPACE_LOG_PATH for the run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="profile or reuse the Fine Plan")
    add_common(plan)

    env = commands.add_parser("env", help="print the EdgePace runtime environment")
    env.add_argument("--plan", type=Path, required=True)
    env.add_argument("--policy-library", type=Path, default=DEFAULT_POLICY_LIBRARY)
    env.add_argument("--deadline-us", type=int, default=33333)
    env.add_argument("--period-us", type=int, default=33333)
    env.add_argument("--freshness-ms", type=float, default=1800.0)
    env.add_argument("--max-tokens", type=int, default=4)
    env.add_argument("--format", choices=("shell", "json"), default="shell")
    add_runtime(env)

    serve = commands.add_parser("serve", help="plan, then exec llama-server")
    add_common(serve)
    add_runtime(serve)

    args = parser.parse_args()
    if args.command == "env":
        manifest = json.loads(args.plan.read_text(encoding="utf-8"))
        values = runtime_environment(args, manifest)
        if args.format == "json":
            print(json.dumps(values, indent=2, sort_keys=True))
        else:
            for name, value in values.items():
                print(f"export {name}='{value}'")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, _ = load_or_profile(args, args.output_dir)
    print(describe(manifest), file=sys.stderr)
    if args.command == "plan":
        return 0

    environment = clean_environment()
    runtime = runtime_environment(args, manifest)
    environment.update(runtime)
    command = server_command(args)
    (args.output_dir / "server.command.json").write_text(
        json.dumps({"command": command, "environment": runtime}, indent=2) + "\n",
        encoding="utf-8")
    os.execve(command[0], command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
