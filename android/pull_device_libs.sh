#!/usr/bin/env bash
# Copy the phone libraries the Android build links against into build/android-deps:
#   geniex-bench v0.3.14 (unpacked on the phone, GENIEX_DIR):
#     lib/llama_cpp/libggml-hexagon.so, libggml-htp-v79.so   the Hexagon NPU backend
#     lib/llama_cpp/libggml-opencl.so                         the Adreno OpenCL backend
#   the phone's own libraries the foregrounds use:
#     /vendor/lib64/libopencv_core.so, libopencv_imgproc.so, libc++.so; /system/lib64/libbinder_ndk.so
#
#   android/pull_device_libs.sh
#   ADB=/mnt/c/.../adb.exe android/pull_device_libs.sh       # WSL with the Windows adb
#   ADB_SERIAL=... GENIEX_DIR=/data/local/tmp/geniex/geniex-bench-android-arm64-v0.3.14 ...
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEPS=${ANDROID_DEPS:-$ROOT/build/android-deps}
ADB=${ADB:-adb}
GENIEX_DIR=${GENIEX_DIR:-/data/local/tmp/geniex/geniex-bench-android-arm64-v0.3.14}
adb_=("$ADB" ${ADB_SERIAL:+-s "$ADB_SERIAL"})
mkdir -p "$DEPS"

pull() {   # device path -> $DEPS/<name> (as root: /vendor and the geniex dir need it on some phones)
    local src=$1 name=${2:-$(basename "$1")}
    "${adb_[@]}" exec-out "su -c 'cat $src'" > "$DEPS/$name.part"
    [ -s "$DEPS/$name.part" ] || { rm -f "$DEPS/$name.part"; echo "could not read $src on the phone" >&2; return 1; }
    mv "$DEPS/$name.part" "$DEPS/$name"
    echo "$(sha256sum "$DEPS/$name" | cut -c1-16)  $name  <- $src"
}
for lib in libggml-hexagon.so libggml-htp-v79.so; do
    pull "$GENIEX_DIR/lib/llama_cpp/$lib"
done
pull "$GENIEX_DIR/lib/llama_cpp/libggml-opencl.so" || echo "  (no OpenCL backend: a VLM's vision encoder stays off the GPU)"
for lib in libopencv_core.so libopencv_imgproc.so libc++.so; do
    pull "/vendor/lib64/$lib" || echo "  (no $lib: the foregrounds are not built)"
done
pull /system/lib64/libbinder_ndk.so
echo "PULL_OK $DEPS"
