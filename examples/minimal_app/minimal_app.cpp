// Minimal real-time application for EdgePace: periodic jobs that stream over a buffer
// (memory-bandwidth bound, like image processing), publishing what EdgePace needs:
//   task page      release, deadline and completion of every job  (full_sched/task_writer.h)
//   progress page  the job's stages, for the sampler's progress predictor (foreground_progress.h)
// It stops after --frames jobs.  Replace run_job() with your own work.
//
//   minimal_app --frames 1800 --task-page P --progress-page P [--period-us 33333]
//               [--deadline-us 33333] [--work-mb 64] [--passes 4] [--threads 4]
#include "full_sched/task_writer.h"
#include "foreground_progress.h"

#include <time.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static uint64_t now_ns() {   // CLOCK_MONOTONIC, the clock every EdgePace page uses
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}

static void sleep_until(uint64_t t_ns) {
    timespec t{time_t(t_ns / 1000000000ULL), long(t_ns % 1000000000ULL)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, nullptr) != 0) {}
}

// The job: `passes` read-modify-write sweeps over each thread's slice of the buffer.
static uint64_t run_job(std::vector<uint64_t> & buffer, int passes, int threads) {
    std::vector<std::thread> workers;
    std::vector<uint64_t> sums(threads, 0);
    const size_t slice = buffer.size() / threads;
    for (int w = 0; w < threads; ++w) {
        workers.emplace_back([&, w] {
            uint64_t * p = buffer.data() + w * slice;
            uint64_t s = 0;
            for (int pass = 0; pass < passes; ++pass)
                for (size_t i = 0; i < slice; ++i) { p[i] = p[i] * 6364136223846793005ULL + 1; s += p[i]; }
            sums[w] = s;
        });
    }
    uint64_t total = 0;
    for (int w = 0; w < threads; ++w) { workers[w].join(); total += sums[w]; }
    return total;
}

int main(int argc, char ** argv) {
    int frames = 0, passes = 4, threads = 4;
    uint64_t period_ns = 33333000, deadline_ns = 33333000;
    size_t work_mb = 64;
    std::string task_page, progress_page;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--frames") frames = std::atoi(v.c_str());
        else if (a == "--task-page") task_page = v;
        else if (a == "--progress-page") progress_page = v;
        else if (a == "--period-us") period_ns = std::strtoull(v.c_str(), nullptr, 10) * 1000;
        else if (a == "--deadline-us") deadline_ns = std::strtoull(v.c_str(), nullptr, 10) * 1000;
        else if (a == "--work-mb") work_mb = std::strtoull(v.c_str(), nullptr, 10);
        else if (a == "--passes") passes = std::atoi(v.c_str());
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (frames <= 0) { std::fprintf(stderr, "usage: minimal_app --frames N --task-page P --progress-page P ...\n"); return 2; }

    std::vector<uint64_t> buffer(work_mb * 1024 * 1024 / sizeof(uint64_t), 1);
    full_sched::task_snapshot_writer task(task_page, -1, period_ns, 1);   // no eventfd, one job per period
    v3_progress::writer progress(progress_page);

    uint64_t release = now_ns() + 100000000ULL, misses = 0, checksum = 0, total_ns = 0, max_ns = 0;
    for (uint64_t id = 1; id <= uint64_t(frames); ++id) {
        sleep_until(release);
        const uint64_t start = now_ns(), deadline = release + deadline_ns;
        task.begin_frame(id, release, release, start, deadline, 0, 0, 0, 0);
        progress.begin(id, 0, release, start, deadline, release + period_ns, deadline + period_ns, 0);
        progress.pipeline_started(now_ns());          // the memory-heavy stage starts now
        checksum += run_job(buffer, passes, threads);
        const uint64_t done = now_ns();
        const bool missed = done > deadline;
        misses += missed;
        total_ns += done - release;
        max_ns = done - release > max_ns ? done - release : max_ns;
        task.complete(id, done, missed, 0, 0, 0, 0, done, done);
        progress.completed(done);
        release += period_ns;                          // a late job delays the next one's start
    }
    // response time = completion - release; size --work-mb / --passes so the job alone takes
    // well under the deadline (the example: ~half of it)
    std::printf("{\"frames\":%d,\"misses\":%llu,\"response_mean_ms\":%.2f,\"response_max_ms\":%.2f,"
                "\"checksum\":%llu}\n", frames, (unsigned long long) misses, total_ns / 1e6 / frames,
                max_ns / 1e6, (unsigned long long) checksum);
    return 0;
}
