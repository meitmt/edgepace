#!/usr/bin/env bash
# Build the vendored light four-camera foreground (foreground/light) with the
# flags of the original light_workload/build_light_foreground.sh.
#   scripts/build_light_foreground.sh [output]   (default build/foreground/fourcam_light)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$root/build/foreground/fourcam_light}
mkdir -p "$(dirname "$out")"
read -r -a OPENCV_CFLAGS <<< "$(pkg-config --cflags opencv4)"
read -r -a OPENCV_LIBS <<< "$(pkg-config --libs opencv4)"
read -r -a GST_CFLAGS <<< "$(pkg-config --cflags gstreamer-app-1.0 gstreamer-video-1.0)"
read -r -a GST_LIBS <<< "$(pkg-config --libs gstreamer-app-1.0 gstreamer-video-1.0)"
g++ -O3 -mcpu=native -ffast-math -std=c++17 -Wall -Wextra -Wpedantic -pthread \
    -I"$root/foreground/light/include" -I"$root/monitor/third_party/v3_progress" \
    "${GST_CFLAGS[@]}" "${OPENCV_CFLAGS[@]}" \
    -o "$out" "$root/foreground/light/src/fourcam_light.cpp" \
    "${GST_LIBS[@]}" "${OPENCV_LIBS[@]}"
sha256sum "$out"
