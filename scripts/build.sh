#!/usr/bin/env bash
# Build EdgePace on a Jetson (CUDA):
#   llama    llama-server with the EdgePace scheduler (./llama.cpp, scripts/fetch_llama.sh)
#   policy   the external policy DSO it loads (build/policy/libedgepace_policy.so)
#   monitor  the foreground PMU sampler (monitor/build/pmu_sampler)
#   client   the C++ LLM client with the request-page protocol (client/build/edgepace_client)
#   foreground  the example four-camera foreground (build/foreground/...; GStreamer + OpenCV)
#   minimal  the minimal application template (build/examples/minimal_app)
#   stress   the GPU memory stressor of the sweep profile's S check (build/stress/gpu_stress)
#
#   scripts/build.sh                       # everything
#   scripts/build.sh llama policy          # some parts
#
# Environment: CUDA_HOME (default /usr/local/cuda), CUDA_ARCH (default: this GPU's, e.g. 87 on Orin),
# JOBS (default nproc), BUILD_DIR (default build/).  Incremental: an existing llama
# build tree is only rebuilt, never reconfigured.  Full output in $BUILD_DIR/build.log.
set -o pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src="$root/llama.cpp"
build=${BUILD_DIR:-"$root/build"}
jobs=${JOBS:-$(nproc)}
cuda=${CUDA_HOME:-/usr/local/cuda}

# CUDA architecture: CUDA_ARCH, else this machine's GPU (llama.cpp's own "native" needs
# CMake >= 3.24; JetPack 6 ships 3.22), else 87 (Jetson Orin)
detect_cuda_arch() {
    local dir
    dir=$(mktemp -d)
    cat > "$dir/arch.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
    std::printf("%d%d\n", p.major, p.minor);
    return 0;
}
EOF
    "$cuda/bin/nvcc" -o "$dir/arch" "$dir/arch.cu" > /dev/null 2>&1 && "$dir/arch" 2> /dev/null
    rm -rf "$dir"
}
arch=
resolve_arch() {   # once, for the parts that compile CUDA
    [ -n "$arch" ] && return
    if [ -n "${CUDA_ARCH:-}" ]; then
        arch=$CUDA_ARCH
    elif [ -x "$cuda/bin/nvcc" ] && arch=$(detect_cuda_arch) && [ -n "$arch" ]; then
        echo "CUDA architecture of this GPU: $arch (CUDA_ARCH= overrides)" | tee -a "$log"
    else
        arch=87
        echo "could not query the GPU; CUDA architecture 87 (Jetson Orin; set CUDA_ARCH= for another)" | tee -a "$log" >&2
    fi
}
parts=("$@")
[ ${#parts[@]} -gt 0 ] || parts=(llama policy monitor client foreground minimal stress)
mkdir -p "$build"
log="$build/build.log"
: > "$log"

run() {
    printf '+ %s\n' "$*" | tee -a "$log"
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        printf 'FAILED (exit %s): %s\n' "$rc" "$*" | tee -a "$log"
        exit "$rc"
    fi
}

need_llama() {
    [ -f "$src/ggml/src/ggml-edgepace.h" ] || { echo "no patched llama.cpp in $src: run scripts/fetch_llama.sh" >&2; exit 2; }
}

for part in "${parts[@]}"; do
    case "$part" in
    llama)
        need_llama
        if [ ! -f "$build/llama/CMakeCache.txt" ]; then
            resolve_arch
            run cmake -S "$src" -B "$build/llama" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
                -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" -DGGML_NATIVE=ON \
                -DBUILD_SHARED_LIBS=ON \
                -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
                -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
        fi
        run cmake --build "$build/llama" --target llama-server -j "$jobs"
        ;;
    policy)
        # Owns signal reading, slack/risk and the HOLD/ADMIT decisions, and the
        # profile_static mode of Fine Plan profiling.  Only a POD C ABI crosses the
        # boundary, so it is rebuilt without touching the llama binaries.
        need_llama
        mkdir -p "$build/policy"
        run g++ -O2 -std=c++17 -fPIC -shared -pthread \
            -I"$src/external/edgepace_policy" -I"$src/external/edgepace_policy/include" \
            -o "$build/policy/libedgepace_policy.so" \
            "$src/external/edgepace_policy/edgepace_policy.cpp"
        ;;
    monitor)
        run make -C "$root/monitor"
        ;;
    client)
        need_llama
        run make -C "$root/client"
        ;;
    foreground)
        run "$root/scripts/build_light_foreground.sh" "$build/foreground/fourcam_light"
        ;;
    minimal)
        # the minimal application of examples/minimal_app (a template for your own)
        mkdir -p "$build/examples"
        run g++ -O2 -std=c++17 -Wall -Wextra -pthread \
            -I"$root/foreground/light/include" -I"$root/monitor/third_party/v3_progress" \
            -o "$build/examples/minimal_app" "$root/examples/minimal_app/minimal_app.cpp"
        ;;
    stress)
        resolve_arch
        mkdir -p "$build/stress"
        run "$cuda/bin/nvcc" -O3 -std=c++17 -arch=sm_"$arch" -o "$build/stress/gpu_stress" "$root/stress/gpu_stress.cu"
        ;;
    *)
        echo "unknown part: $part (llama policy monitor client foreground minimal stress)" >&2
        exit 2
        ;;
    esac
done
printf 'BUILD_OK %s\n' "${parts[*]}" | tee -a "$log"
