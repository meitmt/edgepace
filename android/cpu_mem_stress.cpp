// CPU memory-bandwidth stressor for the phone: the phone has no GPU
// load to contend with, its memory contention is CPU and NPU traffic.  Each
// thread is pinned to one CPU of --cpus and streams a private buffer
// (read-modify-write, 64 B stride) far larger than the caches, until
// --duration-sec or SIGTERM.  --ready-file appears once all threads run, as
// with the Jetson's gpu_stress.
//
//   cpu_mem_stress --cpus 1,2,3 --memory-mb 256 --duration-sec 120 --ready-file F
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }
}  // namespace

int main(int argc, char ** argv) {
    std::vector<int> cpus{1, 2, 3};
    size_t megabytes = 256;
    double duration_s = 60.0;
    std::string ready_file;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string { return ++i < argc ? argv[i] : ""; };
        if (arg == "--cpus") {
            cpus.clear();
            std::stringstream list(value());
            for (std::string item; std::getline(list, item, ',');) cpus.push_back(std::stoi(item));
        } else if (arg == "--memory-mb") megabytes = std::stoul(value());
        else if (arg == "--duration-sec") duration_s = std::stod(value());
        else if (arg == "--ready-file") ready_file = value();
        else { std::cerr << "usage: cpu_mem_stress --cpus 1,2,3 --memory-mb 256 --duration-sec S --ready-file F\n"; return 2; }
    }
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    std::atomic<int> running{0};
    std::atomic<uint64_t> lines{0};
    std::vector<std::thread> threads;
    for (int cpu : cpus) {
        threads.emplace_back([&, cpu] {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            sched_setaffinity(0, sizeof(set), &set);
            const size_t words = megabytes * 1024 * 1024 / sizeof(uint64_t);
            std::vector<uint64_t> buffer(words, 1);
            ++running;
            uint64_t local = 0;
            while (!g_stop) {
                for (size_t i = 0; i < words && !g_stop; i += 8) {   // one 64-byte line per step
                    buffer[i] = buffer[i] * 3 + 1;
                    ++local;
                }
            }
            lines += local;
        });
    }
    while (running < static_cast<int>(cpus.size())) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!ready_file.empty()) std::ofstream(ready_file) << "ready\n";
    const auto start = std::chrono::steady_clock::now();
    while (!g_stop && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < duration_s)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    g_stop = true;
    for (auto & t : threads) t.join();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"cpus\":" << cpus.size() << ",\"seconds\":" << seconds
              << ",\"gib_per_s_rmw\":" << lines.load() * 64.0 * 2 / seconds / (1 << 30) << "}\n";
    return 0;
}
