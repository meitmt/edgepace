#!/usr/bin/env bash
# Android (Snapdragon 8 Elite, OnePlus 13) build of the EdgePace runtime -> build/android/{bin,lib}.
#
# The LLM runs on llama.cpp with geniex-bench v0.3.14's prebuilt Hexagon (NPU) and
# OpenCL (Adreno) backends.  libggml-base.so carries the EdgePace scheduler: it is
# built from the base tree of android/setup_base.sh with
#   android/llama-overlay/ggml/src/ggml-backend.cpp   the scheduler, as llama.cpp/'s
#   llama.cpp/ggml/src/{ggml-edgepace.h,edgepace_policy_abi.h,edgepace/abi.h}   shared as is
# Everything else is built from this repository's sources:
#   policy DSO   llama.cpp/external/edgepace_policy/edgepace_policy.cpp
#   sampler      monitor/src/pmu_sampler.cpp (+ progress predictor)
#   client       client/edgepace_client.cpp
#   foregrounds  foreground/{light,android} (NDK MediaCodec; the phone's own OpenCV)
#
# Before: scripts/fetch_llama.sh, android/setup_base.sh, android/pull_device_libs.sh.
#   ANDROID_NDK_HOME=/path/to/android-ndk-r27c android/build_android.sh
# Incremental: the base tree's ninja build is reused, overlay files are copied only
# when they differ, the other parts are rebuilt when a source is newer.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/android/toolchain.sh"
BASE=${ANDROID_LLAMA_SRC:-$ROOT/build/android-base/llama.cpp}
BUILD=${ANDROID_LLAMA_BUILD:-$ROOT/build/android-base/build}
DEPS=${ANDROID_DEPS:-$ROOT/build/android-deps}        # android/pull_device_libs.sh
OUT=${ANDROID_OUT:-$ROOT/build/android}
JOBS=${JOBS:-$(nproc)}
OPENCV_INCLUDE=${OPENCV_INCLUDE:-/usr/include/opencv4}   # headers matching the phone's OpenCV 4

for path in "$BASE/ggml/src/ggml-backend.cpp" "$BUILD/build.ninja" "$ROOT/llama.cpp/ggml/src/ggml-edgepace.h" \
            "$DEPS/libggml-hexagon.so" "$DEPS/libggml-htp-v79.so"; do
    [[ -e "$path" ]] || { echo "missing $path (see android/README.md: setup_base.sh, pull_device_libs.sh, scripts/fetch_llama.sh)" >&2; exit 2; }
done

sync_file() {   # copy only when different, so ninja rebuilds only what changed
    mkdir -p "$(dirname "$2")"
    cmp -s "$1" "$2" || { cp "$1" "$2"; echo "updated $2"; }
}
sync_file "$ROOT/android/llama-overlay/ggml/src/ggml-backend.cpp" "$BASE/ggml/src/ggml-backend.cpp"
for f in ggml-edgepace.h edgepace_policy_abi.h edgepace/abi.h; do
    sync_file "$ROOT/llama.cpp/ggml/src/$f" "$BASE/ggml/src/$f"
done

ninja -C "$BUILD" ggml-base llama-server -j "$JOBS" | grep -E 'Building|Linking|FAILED|error' || true
[ -f "$BUILD/bin/llama-server" ] || { echo "llama-server build failed (ninja -C $BUILD llama-server)" >&2; exit 1; }
mkdir -p "$OUT/bin" "$OUT/lib"
for lib in libggml-base.so libggml-cpu.so libggml.so libllama.so libllama-common.so libmtmd.so \
           libllama-server-impl.so; do
    [ -e "$BUILD/bin/$lib" ] && cp -L "$BUILD/bin/$lib" "$OUT/lib/$lib"
done
for lib in libggml.so libllama.so libllama-common.so libmtmd.so; do
    [ -e "$OUT/lib/$lib" ] && cp "$OUT/lib/$lib" "$OUT/lib/$lib.0"   # SONAMEs as real files: adb push drops symlinks
done
cp "$BUILD/bin/llama-server" "$OUT/bin/llama-server"
cp "$DEPS/libggml-hexagon.so" "$DEPS/libggml-htp-v79.so" "$OUT/lib/"

build_if_newer() {   # output, sources... -- compiler arguments
    local out=$1; shift
    local sources=() newer=0
    while [[ $1 != -- ]]; do sources+=("$1"); shift; done; shift
    for s in "${sources[@]}"; do [[ ! -f "$out" || "$s" -nt "$out" ]] && newer=1; done
    if (( newer )); then echo "building $out"; "${ANDROID_CXX[@]}" "$@" -o "$out"; fi
}
# geniex's Hexagon and OpenCL backends are regular libraries, not GGML_BACKEND_DL
# modules: shims give them the dynamic-backend entry points
build_if_newer "$OUT/lib/libggml-hexagon-dl.so" "$ROOT/android/ggml-hexagon-dl-shim.cpp" -- \
    -std=c++17 -O3 -fPIC -shared "$ROOT/android/ggml-hexagon-dl-shim.cpp" "${ANDROID_UNWIND[@]}" -ldl
build_if_newer "$OUT/lib/libggml-opencl-dl.so" "$ROOT/android/ggml-opencl-dl-shim.cpp" -- \
    -std=c++17 -O3 -fPIC -shared "$ROOT/android/ggml-opencl-dl-shim.cpp" "${ANDROID_UNWIND[@]}" -ldl
if [[ -f "$DEPS/libggml-opencl.so" ]]; then
    cp "$DEPS/libggml-opencl.so" "$OUT/lib/"
else
    echo "no OpenCL backend in $DEPS: a VLM's vision encoder then runs on the NPU/CPU only"
fi
P=$ROOT/llama.cpp/external/edgepace_policy
build_if_newer "$OUT/lib/libedgepace_policy.so" "$P/edgepace_policy.cpp" "$P/edgepace_policy_engine.h" \
        "$ROOT/llama.cpp/ggml/src/ggml-edgepace.h" "$ROOT/llama.cpp/ggml/src/edgepace_policy_abi.h" -- \
    -std=c++17 -O3 -fPIC -shared -pthread -I"$P" -I"$P/include" -I"$ROOT/llama.cpp/ggml/src" \
    "$P/edgepace_policy.cpp" "${ANDROID_UNWIND[@]}"
M=$ROOT/monitor
build_if_newer "$OUT/bin/pmu_sampler" "$M/src/pmu_sampler.cpp" "$M/src/progress_predictor.h" \
        "$M/third_party/full_sched/src/core.cpp" "$M/third_party/v3_progress/foreground_progress.h" -- \
    -std=c++17 -O2 -pthread -I"$M/third_party/full_sched/include" -I"$M/third_party/v3_progress" \
    "$M/src/pmu_sampler.cpp" "$M/third_party/full_sched/src/core.cpp" "${ANDROID_UNWIND[@]}"
C=$ROOT/client
build_if_newer "$OUT/bin/edgepace_client" "$C/edgepace_client.cpp" "$C/edgepace_request_page.h" \
        "$ROOT/llama.cpp/ggml/src/edgepace/abi.h" -- \
    -std=c++17 -O2 -pthread -I"$ROOT/llama.cpp/ggml/src" "$C/edgepace_client.cpp" "${ANDROID_UNWIND[@]}"
build_if_newer "$OUT/bin/cpu_mem_stress" "$ROOT/android/cpu_mem_stress.cpp" -- \
    -std=c++17 -O2 -pthread "$ROOT/android/cpu_mem_stress.cpp" "${ANDROID_UNWIND[@]}"

# The foregrounds link the phone's own OpenCV (/vendor/lib64), built against the
# platform libc++ (namespace std::__1, not the NDK's __ndk1): they use the platform
# libc++.so and its ABI configuration instead of the NDK's.
if [[ -f "$DEPS/libopencv_core.so" && -f "$DEPS/libopencv_imgproc.so" && -f "$DEPS/libc++.so" &&
      -f "$DEPS/libbinder_ndk.so" && -d "$OPENCV_INCLUDE/opencv2" ]]; then
    CFG=$ANDROID_TOOLCHAIN_DIR/platform-cxx-config
    mkdir -p "$CFG"
    cat > "$CFG/__config_site" <<'EOF'
// The Android platform libc++ ABI (the phone's vendor OpenCV is built against it).
#ifndef _LIBCPP___CONFIG_SITE
#define _LIBCPP___CONFIG_SITE
#define _LIBCPP_ABI_VERSION 1
#define _LIBCPP_ABI_NAMESPACE __1
#define _LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS
#define _LIBCPP_HAS_NO_STD_MODULES
#define _LIBCPP_HAS_NO_TIME_ZONE_DATABASE
#define _LIBCPP_PSTL_CPU_BACKEND_THREAD
#define _LIBCPP_HARDENING_MODE_DEFAULT 2
#endif
EOF
    fg_flags=(--target=aarch64-linux-android35 -isystem "$CFG" -isystem "$ANDROID_SYSROOT/usr/include/c++/v1"
              -isystem "$OPENCV_INCLUDE" -I"$ROOT/foreground/light/include" -I"$ROOT/monitor/third_party/v3_progress"
              -march=armv8.7a -O3 -DNDEBUG -std=c++17 -pthread
              -D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS -stdlib=libc++ -nostdlib++ -Wall -Wextra -Wpedantic)
    fg_links=(-L"$DEPS" ${ANDROID_FG_LINKS:+-L"$ANDROID_FG_LINKS"} -Wl,-rpath-link,"$DEPS" -Wl,--allow-shlib-undefined
              -lopencv_imgproc -lopencv_core -lmediandk "$DEPS/libbinder_ndk.so"
              -landroid -llog -lz -ldl -lm "$DEPS/libc++.so")
    F=$ROOT/foreground/light/src/fourcam_light.cpp
    A=$ROOT/foreground/android/src/fourcam_phone.cpp
    H=$ROOT/monitor/third_party/v3_progress/foreground_progress.h
    # the light foreground (foreground/light) on Android
    out=$OUT/bin/fourcam_light
    if [[ ! -f "$out" || "$F" -nt "$out" || "$H" -nt "$out" ]]; then
        echo "building $out"
        "${ANDROID_CLANGXX[@]}" "${fg_flags[@]}" -ffast-math "$F" "${fg_links[@]}" -o "$out"
    fi
    # the paper's phone foreground (foreground/android)
    out=$OUT/bin/fourcam_phone
    if [[ ! -f "$out" || "$A" -nt "$out" || "$H" -nt "$out" ]]; then
        echo "building $out"
        "${ANDROID_CLANGXX[@]}" "${fg_flags[@]}" "$A" "${fg_links[@]}" -o "$out"
    fi
else
    echo "skipping the foregrounds: needs $DEPS/{libopencv_core,libopencv_imgproc,libc++,libbinder_ndk}.so" \
         "(android/pull_device_libs.sh) and OpenCV 4 headers in $OPENCV_INCLUDE"
fi
# dynamically loaded backends are discovered next to the executable
cp "$OUT/lib/libggml-cpu.so" "$OUT/lib/libggml-hexagon-dl.so" "$OUT/lib/libggml-opencl-dl.so" "$OUT/bin/"
(cd "$OUT" && find bin lib -type f | sort | xargs sha256sum > SHA256SUMS)
echo "ANDROID_BUILD_OK $OUT"
