#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace {

std::atomic<bool> stop_requested{false};

void stop_handler(int) { stop_requested.store(true, std::memory_order_relaxed); }

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure at %s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

double now_sec() {
    using Clock = std::chrono::steady_clock;
    static const auto origin = Clock::now();
    return std::chrono::duration<double>(Clock::now() - origin).count();
}

__global__ void initialize(float* data, size_t count, float value) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        data[i] = value + static_cast<float>(i & 255) * 0.00001f;
    }
}

// Three full-size arrays keep the stream well beyond the GPU caches.  This is
// deliberately a high-global-memory-traffic control, not a VLM approximation.
__global__ void memory_stream(float* a, const float* b, const float* c, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        a[i] = fmaf(b[i], 1.000001f, c[i] * 0.000001f);
    }
}

// The working set is one scalar per thread.  The repeated dependent FMAs make
// this a compute control with negligible global-memory traffic after launch.
__global__ void register_fma(float* out, int count, int inner) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = 0.000001f * static_cast<float>(i + 1);
    float y = 1.000001f + 0.0000001f * static_cast<float>((i & 255) + 1);
    float z = 0.999991f;
    for (int k = 0; k < inner; ++k) {
        x = __fmaf_rn(x, y, z);
        y = __fmaf_rn(y, 0.999989f, 0.000017f);
        z = __fmaf_rn(z, 1.000011f, -0.000013f);
    }
    out[i] = x + y + z;
}

struct Options {
    std::string mode = "memory";
    double duration_sec = 30.0;
    int memory_mb = 384;
    int compute_elems = 262144;
    int compute_inner = 8192;
    int streams = 1;
    std::string ready_file;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--mode") options.mode = value("--mode");
        else if (arg == "--duration-sec") options.duration_sec = std::atof(value("--duration-sec"));
        else if (arg == "--memory-mb") options.memory_mb = std::atoi(value("--memory-mb"));
        else if (arg == "--compute-elems") options.compute_elems = std::atoi(value("--compute-elems"));
        else if (arg == "--compute-inner") options.compute_inner = std::atoi(value("--compute-inner"));
        else if (arg == "--ready-file") options.ready_file = value("--ready-file");
        else if (arg == "--help") {
            std::puts("gpu_stress --mode memory|compute --duration-sec S [--memory-mb M] [--compute-elems N] [--compute-inner N] --ready-file PATH");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            std::exit(1);
        }
    }
    if ((options.mode != "memory" && options.mode != "compute") || options.duration_sec <= 0.0 ||
        options.memory_mb <= 0 || options.compute_elems <= 0 || options.compute_inner <= 0 ||
        options.ready_file.empty()) {
        std::fprintf(stderr, "invalid options\n");
        std::exit(1);
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
    const Options options = parse_options(argc, argv);
    check(cudaSetDevice(0), "cudaSetDevice");

    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, 0), "cudaGetDeviceProperties");
    const int block = 256;
    const int grid = std::max(1, property.multiProcessorCount * 8);
    uint64_t launches = 0;
    uint64_t bytes_touched = 0;

    if (options.mode == "memory") {
        const size_t count = static_cast<size_t>(options.memory_mb) * 1024ULL * 1024ULL / sizeof(float);
        float *a = nullptr, *b = nullptr, *c = nullptr;
        check(cudaMalloc(&a, count * sizeof(float)), "cudaMalloc(a)");
        check(cudaMalloc(&b, count * sizeof(float)), "cudaMalloc(b)");
        check(cudaMalloc(&c, count * sizeof(float)), "cudaMalloc(c)");
        initialize<<<grid, block>>>(a, count, 0.1f);
        initialize<<<grid, block>>>(b, count, 0.2f);
        initialize<<<grid, block>>>(c, count, 0.3f);
        check(cudaGetLastError(), "initialize");
        check(cudaDeviceSynchronize(), "initialize sync");
        std::ofstream(options.ready_file).put('\n');
        std::printf("event=active_start mode=memory buffer_mb_each=%d device=%s\n", options.memory_mb, property.name);
        std::fflush(stdout);
        const double start = now_sec();
        while (!stop_requested.load(std::memory_order_relaxed) && now_sec() - start < options.duration_sec) {
            memory_stream<<<grid, block>>>(a, b, c, count);
            memory_stream<<<grid, block>>>(b, c, a, count);
            memory_stream<<<grid, block>>>(c, a, b, count);
            check(cudaGetLastError(), "memory_stream");
            check(cudaDeviceSynchronize(), "memory stream sync");
            ++launches;
            bytes_touched += 3ULL * count * sizeof(float) * 3ULL;  // read b,c then write a for three kernels
        }
        const double elapsed = now_sec() - start;
        check(cudaFree(a), "cudaFree(a)");
        check(cudaFree(b), "cudaFree(b)");
        check(cudaFree(c), "cudaFree(c)");
        std::printf("event=active_end mode=memory launches=%llu bytes=%llu elapsed_sec=%.6f approx_gib_per_sec=%.3f\n",
                    static_cast<unsigned long long>(launches), static_cast<unsigned long long>(bytes_touched), elapsed,
                    bytes_touched / elapsed / 1073741824.0);
    } else {
        float* output = nullptr;
        check(cudaMalloc(&output, static_cast<size_t>(options.compute_elems) * sizeof(float)), "cudaMalloc(output)");
        register_fma<<<(options.compute_elems + block - 1) / block, block>>>(output, options.compute_elems, 32);
        check(cudaGetLastError(), "compute warmup");
        check(cudaDeviceSynchronize(), "compute warmup sync");
        std::ofstream(options.ready_file).put('\n');
        std::printf("event=active_start mode=compute elements=%d inner=%d device=%s\n", options.compute_elems,
                    options.compute_inner, property.name);
        std::fflush(stdout);
        const double start = now_sec();
        while (!stop_requested.load(std::memory_order_relaxed) && now_sec() - start < options.duration_sec) {
            register_fma<<<(options.compute_elems + block - 1) / block, block>>>(output, options.compute_elems,
                                                                                     options.compute_inner);
            check(cudaGetLastError(), "register_fma");
            check(cudaDeviceSynchronize(), "compute sync");
            ++launches;
            bytes_touched += static_cast<uint64_t>(options.compute_elems) * sizeof(float);
        }
        const double elapsed = now_sec() - start;
        check(cudaFree(output), "cudaFree(output)");
        std::printf("event=active_end mode=compute launches=%llu output_bytes=%llu elapsed_sec=%.6f estimated_gflop=%.3f\n",
                    static_cast<unsigned long long>(launches), static_cast<unsigned long long>(bytes_touched), elapsed,
                    6.0 * options.compute_elems * options.compute_inner * launches / elapsed / 1e9);
    }
    std::fflush(stdout);
    return 0;
}
