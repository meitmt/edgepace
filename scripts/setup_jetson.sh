#!/usr/bin/env bash
# Everything a Jetson needs before `tools/edgepace profile`, in one go: llama.cpp with the
# EdgePace patch, all parts built, the example model and the test video, the offline tests.
#
#   scripts/setup_jetson.sh                    # the light4cam example's model
#   scripts/setup_jetson.sh qwen25-vl-3b       # other models (scripts/fetch_models.sh --list)
#   MODELS_DIR=/data/models scripts/setup_jetson.sh   # models elsewhere (the examples expect models/)
#
# Re-running skips what is done (fetch, build and downloads are incremental).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
models=("$@")
[ ${#models[@]} -gt 0 ] || models=(qwen25-1.5b)
step() { printf '\n== %s\n' "$*"; }

step "python ($(python3 --version))"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null ||
    python3 -c 'import tomli' 2>/dev/null ||
    { echo "Python < 3.11 needs tomli: python3 -m pip install -r requirements.txt" >&2; exit 2; }
step "llama.cpp with EdgePace";       scripts/fetch_llama.sh
step "build (llama-server takes ~25 min the first time)"; scripts/build.sh
step "models: ${models[*]}";          scripts/fetch_models.sh "${models[@]}"
step "test video"
[ -f media/test_1080p30_hevc.mp4 ] && echo "have media/test_1080p30_hevc.mp4" || scripts/make_test_video.sh
step "offline tests";                 tests/run_offline.sh > build/offline_tests.log 2>&1 && echo "passed (build/offline_tests.log)"
cat <<EOF

ready. Next (locked clocks, sudo):
  tools/edgepace profile examples/workloads/light4cam_qwen25_1p5b_f16.toml
  tools/edgepace evaluate runs/profile_light4cam_qwen25_1p5b_f16_*/profile
EOF
