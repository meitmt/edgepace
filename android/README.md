# EdgePace on Android (OnePlus 13, Snapdragon 8 Elite)

The phone runs the same EdgePace runtime as the Jetson — the scheduler, the policy
DSO, the PMU sampler with the progress predictor, the four-camera foreground and the
profile / deployment tools. The LLM runs on llama.cpp with geniex's prebuilt Hexagon
(NPU) backend; a VLM's vision encoder on the NPU or on the Adreno GPU (geniex's
OpenCL backend). The host tools run on a Linux machine (or WSL) with adb and drive
the phone; every measurement runs on the device and its CSVs come back to the host.

| Part | Jetson | Phone |
|---|---|---|
| LLM backend | `./llama.cpp` (EdgePace patches), CUDA | llama.cpp `be4a6a6` (`android/setup_base.sh`) with geniex-bench v0.3.14's `libggml-hexagon.so` / `libggml-htp-v79.so` / `libggml-opencl.so`, unchanged |
| Scheduler | `llama.cpp/ggml/src/ggml-backend.cpp` | `android/llama-overlay/ggml/src/ggml-backend.cpp`: the same logic in the tree whose backend ABI geniex's libraries expect |
| Fine Plan timing | submit → complete (CUDA launches asynchronously) | admit → complete (the NPU computes inside graph_compute) |
| Application video decode | GStreamer + NVDEC | NDK MediaCodec |
| PMU (S, D) | `stall_backend_mem`, `ll_cache_miss_rd` | `stall` (all-cause), `l2d_cache_lmiss_rd` (Oryon L2) |
| Clock lock | `scripts/locked_run.sh` | `android/locked_run_android.sh` (CPU 1.9968 / 2.6496 GHz, GPU 1.1 GHz, DDR/LLCC at hardware max), taken per measurement phase |

## Requirements

- **Phone**: Snapdragon 8 Elite (tested: OnePlus 13), rooted (`su` for adb shell,
  e.g. KernelSU or Magisk), with **geniex-bench v0.3.14** unpacked on it
  (default `/data/local/tmp/geniex/geniex-bench-android-arm64-v0.3.14`; only its
  `lib/llama_cpp/` backends are used). The phone's own `/vendor/lib64` OpenCV
  (4.5.1 on the OnePlus 13) is used by the example applications, for the NV12 → RGBA
  and RGBA → YUV conversions of each frame; EdgePace itself does not use OpenCV.
- **Build host**: x86_64 Linux with the **Android NDK r27c**
  (`ANDROID_NDK_HOME`), `cmake`, `ninja`, `git`, OpenCV 4 headers
  (`apt install libopencv-dev`, or `OPENCV_INCLUDE=...`). The applications import
  only `cv::Mat`, `cv::cvtColor` and `cv::setNumThreads`, so the header version need
  not match the phone's: 4.8 and 4.10 are tested against 4.5.1, with identical output. An aarch64 Linux host
  (e.g. a Jetson) works with its clang 14 (`ANDROID_HOST_CLANG=1`, chosen
  automatically there).
- **Host tools**: Python ≥ 3.10 (`pip install tomli` below 3.11) and adb. From WSL
  use the Windows `adb.exe`: `export ADB=/mnt/c/.../platform-tools/adb.exe` for the
  scripts below and set `adb = "..."` in the workload file.

## Build and install

One command does all of it (phone attached; re-running skips what is done):

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r27c     # WSL: also export ADB=/mnt/c/.../adb.exe
android/setup_android.sh
```

or step by step:

```bash
scripts/fetch_llama.sh                      # EdgePace's shared headers and policy sources
export ANDROID_NDK_HOME=/path/to/android-ndk-r27c
android/setup_base.sh                       # llama.cpp be4a6a6, configured in build/android-base/
android/pull_device_libs.sh                 # geniex backends + the phone's OpenCV/libc++/binder -> build/android-deps/
android/build_android.sh                    # -> build/android/{bin,lib}, SHA256SUMS
scripts/make_test_video.sh                  # media/test_1080p30_hevc.mp4 (needs ffmpeg with libx265)
scripts/fetch_models.sh qwen25-1.5b         # models/
android/install.sh --media media/test_1080p30_hevc.mp4 \
                   --model models/qwen2.5-1.5b-instruct-fp16.gguf   # -> /data/local/tmp/edgepace
```

`install.sh` sends only files whose sha256 differs, so re-running it after a
rebuild is cheap. `DEVICE_ROOT=...` installs elsewhere (then set `device_root` in
the workload file).

## Profile, deploy, measure

```bash
tools/edgepace profile examples/workloads/oneplus13_phone4cam_qwen25_1p5b_f16.toml
P=$(ls -d runs/profile_oneplus13_phone4cam_qwen25_1p5b_f16_*/profile | tail -1)
tools/edgepace evaluate $P --repeats 2      # unmanaged vs EdgePace (~25 min with cool-downs)
tools/edgepace check  $P                    # binaries and model on the phone vs the profile
tools/edgepace serve  $P                    # llama-server on the phone (detached)
tools/edgepace launch $P                    # sampler + application on the phone (detached)
tools/edgepace request $P --body requests/text_4tok.json --count 10 --period-ms 1600
tools/edgepace stop   $P
```

With the example (four-camera application on the CPU cores, Qwen2.5-1.5B F16 on the
NPU, a short request every 1.6 s) expect about 10 % misses unmanaged and 0.06–0.5 %
with EdgePace at ~85 % of the LLM throughput (`profiles/oneplus13/`).

## Measurement hygiene

- The tools lock the phone per phase and let it cool down unlocked between phases
  (shell ≤ 35 °C, CPU ≤ 50 °C): under continuous load the thermal engine caps the
  CPUs after about four minutes, and the GPU stays capped for many minutes after.
  Phase lines with `LOCK_CONTINUOUS=0` lost the lock to thermal capping.
- The phone has a slow state in which memory copies (MediaCodec output, model
  loading) and the NPU take about twice as long, while its clocks and thermal status
  read normal. An idle stretch under the lock, or a lock taken less than ~1 min after
  the previous one, puts it there. llama-server's model load time shows it (1.3 s vs
  2.7 s for Qwen2.5-1.5B F16): the tools warm up until the load time is steady and
  run a phase again when it loads 1.4× slower than the fastest seen (`device_lock` in
  each `phase.json`). Leave ≥ 2 min between separate tool invocations; do not leave
  a deployment locked and idle.
