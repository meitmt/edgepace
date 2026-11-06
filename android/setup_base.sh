#!/usr/bin/env bash
# The phone's llama.cpp base tree: upstream llama.cpp be4a6a6 (the ggml backend ABI of
# geniex-bench v0.3.14's prebuilt Hexagon backend: GGML_BACKEND_API_VERSION 2,
# graph_optimize without params) + an unversioned libggml-base.so SONAME (as geniex's),
# configured for arm64 Android in build/android-base/build.  android/build_android.sh
# then puts the EdgePace scheduler into it (android/llama-overlay) and builds.
#
#   ANDROID_NDK_HOME=/path/to/android-ndk-r27c android/setup_base.sh
#   LLAMA_REPO=/path/to/llama.cpp ...      # a local clone / mirror instead of GitHub
#
# Done once; re-running reconfigures only when the build tree is missing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/android/toolchain.sh"
BASE_COMMIT=be4a6a63eb2b848e19c277bdcf2bd399e8af76d9
SRC=${ANDROID_LLAMA_SRC:-$ROOT/build/android-base/llama.cpp}
BUILD=${ANDROID_LLAMA_BUILD:-$ROOT/build/android-base/build}
repo=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}

if [ ! -d "$SRC/.git" ]; then
    git clone --filter=blob:none --no-checkout "$repo" "$SRC"
    git -C "$SRC" checkout -q -b edgepace-android "$BASE_COMMIT"
    git -C "$SRC" apply "$ROOT/android/patches/ggml-unversioned-soname.patch"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$BASE_COMMIT" ] || { echo "$SRC is not at $BASE_COMMIT" >&2; exit 2; }

if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    cmake -S "$SRC" -B "$BUILD" -G Ninja "${ANDROID_CMAKE_ARGS[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_UNVERSIONED_SONAME=ON \
        -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
        -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF \
        -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF
fi
echo "ANDROID_BASE_OK src=$SRC build=$BUILD (host clang: $ANDROID_HOST_CLANG)"
