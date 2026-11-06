# Workload description reference

A workload file (TOML) names the platform, the real-time application, the LLM load
and how to profile them. `tools/edgepace init` writes a commented template;
`examples/workloads/` has complete ones. Relative paths are relative to the file;
on a phone (`platform.device = "android"`) the application's paths, `llm.model` and
`llm.mmproj` are device paths.

```toml
name = "my_app"           # profile name; also the run directory /dev/shm/edgepace/<name>
```

## `[platform]`

| Key | Default | Meaning |
|---|---|---|
| `power_mode` | `MAXN_SUPER` | Jetson nvpmodel mode the profile requires (and deployment should use) |
| `default_conf` | – | `jetson_clocks --store` reference the clocks are restored to (`platform/`) |
| `video_clocks` | `dynamic` | NVDEC/VIC: `dynamic` (their governor) or `locked` (pinned at max) |
| `device` | `local` | `android`: a phone over adb (`android/README.md`) |
| `adb`, `adb_serial` | `adb` | phone: the adb executable (from WSL: the Windows `adb.exe`) and serial |
| `device_root` | `/data/local/tmp/edgepace` | phone: where `android/install.sh` put the runtime |

## `[foreground]` — the real-time application

| Key | Meaning |
|---|---|
| `command` | Command line. Placeholders: `{frames}` (stop after that many jobs), `{task_page}`, `{progress_page}` (the pages it must publish), `{phase_dir}` (a per-run directory for its own outputs), `{repo}` (this repository) |
| `period_ms`, `deadline_ms` | Release period and relative deadline of a job (default 33.333) |

## `[llm]` — the LLM and its request pattern

| Key | Default | Meaning |
|---|---|---|
| `model` | – | GGUF model |
| `mmproj` | – | a VLM's vision encoder (llama.cpp mmproj GGUF) |
| `plan` | `""` | an existing `cut_plan.json`; empty: profile the Fine Plan (cached) |
| `request` | `requests/text_4tok.json` | JSON body of one request (`/v1/chat/completions`), as your client sends it; a VLM's carries its image |
| `period_ms` | 1600 | request period; 0 = back to back |
| `align` | `frame` | `frame`: send right after a job completes; `none` |
| `max_tokens` | 4 | output tokens per request |
| `freshness_ms` | 1800 | an answer older than this is stale (drives the freshness rescue) |
| `parallel`, `extra_streams` | 1, 0 | server slots; extra back-to-back clients beside the tracked one |
| `context_size`, `batch_size`, `ubatch_size` | 2048, 256, 256 | llama-server `-c`, `-b`, `-ub` (a smaller ubatch shortens prefill quanta) |
| `prefill_boundaries` | `["ffn_inp", "l_out"]` | text prefill cut points per layer; e.g. `["ffn_inp", "ffn_gate", "ffn_up", "l_out"]` for finer quanta |
| `vision_boundaries` | by projector | clip callback names cut inside each vision layer |
| `vision_device` | `HTP0` | phone: the vision encoder's backend, `HTP0` (NPU) or `GPUOpenCL` (Adreno) |
| `allow_uncuttable` | false | accept quanta of a single legal span over their limits (an LM head or patch embedding the accelerator cannot run) |

## `[profile]`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `quick` | `quick`: application alone + beside the unmanaged LLM, `quick_frames` each; `sweep`: also EdgePace at each H of `horizons` |
| `quick_frames` | 600 | frames per quick phase |
| `horizon_ms` | 20 | quick mode: the warning horizon H |
| `interference` | false | write the measured k as `EDGEPACE_RHO_INTERFERENCE` (it is always reported) |
| `objective` | `miss` | sweep: `miss` (fewest misses with retention ≥ `min_retention`) or `throughput` (most LLM throughput with misses ≤ `miss_budget`) |
| `min_retention`, `miss_budget` | 0.6, 0.02 | sweep objective bounds |
| `horizons` | `[12, 16, 20, 24]` | sweep: candidate H (ms) |
| `frames`, `sweep_frames` | 1800, 900 | sweep: frames of the solo and of each sweep phase |
| `miss_tolerance` | 2 frames | sweep: misses within this of the fewest count as equal |
| `stress_binary` | – | sweep, Jetson: GPU memory stressor for the S check (`build/stress/gpu_stress`) |
| `stress_llm_request`, `stress_llm_streams` | –, 4 | sweep, phone: the LLM itself as the memory load |
