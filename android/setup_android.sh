#!/usr/bin/env bash
# Everything the phone needs before `tools/edgepace profile`, in one go (on the build host,
# phone attached): llama.cpp with the EdgePace patch, the phone's base tree, the phone
# libraries, the build, the example model and test video, the install.
#
#   ANDROID_NDK_HOME=/path/to/android-ndk-r27c android/setup_android.sh
#   ... android/setup_android.sh qwen3-vl-2b      # other models (scripts/fetch_models.sh --list)
#   ADB=/mnt/c/.../adb.exe ...                     # WSL with the Windows adb
#
# The test video needs ffmpeg with libx265 on this host; without it, make it elsewhere
# (scripts/make_test_video.sh) and put it in media/ first.  Re-running skips what is done.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to an Android NDK (r27c tested)}"
models=("$@")
[ ${#models[@]} -gt 0 ] || models=(qwen25-1.5b)
step() { printf '\n== %s\n' "$*"; }

step "phone";                          "${ADB:-adb}" get-state > /dev/null || { echo "no phone on ${ADB:-adb}" >&2; exit 2; }
step "llama.cpp with EdgePace";        scripts/fetch_llama.sh
step "phone base tree";                android/setup_base.sh
step "phone libraries";                android/pull_device_libs.sh
step "build";                          android/build_android.sh
step "models: ${models[*]}";           scripts/fetch_models.sh "${models[@]}"
step "test video"
[ -f media/test_1080p30_hevc.mp4 ] || scripts/make_test_video.sh
dir=${MODELS_DIR:-$root/models}
args=(--media media/test_1080p30_hevc.mp4)
for name in "${models[@]}"; do
    while read -r file; do args+=(--model "$dir/$file"); done < <(scripts/fetch_models.sh --files "$name")
done
step "install";                        android/install.sh "${args[@]}"
cat <<EOF

ready. Next:
  tools/edgepace profile examples/workloads/oneplus13_phone4cam_qwen25_1p5b_f16.toml
  tools/edgepace evaluate runs/profile_oneplus13_phone4cam_qwen25_1p5b_f16_*/profile
EOF
