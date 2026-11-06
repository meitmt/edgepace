// Synthetic check of the online progress predictor.
//
// A fake foreground publishes frames on a real progress page; fake sampler
// windows (2 ms, offset so that they straddle stage boundaries) carry
// instructions at a known rate inside the pipeline and a different rate
// outside it; every 7th read fails and its counts arrive with the next one.
// Frames 1-20 run the pipeline in 20 ms, frames 21-30 in 34 ms (just late),
// frames 31-40 in 20 ms again; every pipeline retires the same W instructions.
//
// Progress within a pipeline is front-loaded (see progress()).
//
// Expected: the work model equals W; pipeline counts exclude the outside
// work; the linear predictor never raises a false alarm but misses slow
// frames; the shape model flags at 8 ms exactly the frames that will miss
// (the slow ones and those still delayed by their backlog).

#include "progress_predictor.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kMs = 1000000ULL;
constexpr double kWork = 250e6;            // instructions per pipeline
constexpr double kOutsideRate = 5.0;       // instructions per ns outside the pipeline

struct Frame {
    uint64_t id, release, start, pipeline, complete, deadline;
};

// Front-loaded progress: work fraction done at time fraction u.  u^0.8 gives
// 27% of the work at 20% of the time, as measured on the 4-camera foreground.
double progress(double u) { return std::pow(std::clamp(u, 0.0, 1.0), 0.8); }

int failures = 0;
void check(bool condition, const std::string &what) {
    if (!condition) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s PAGE CSV\n", argv[0]);
        return 2;
    }
    std::remove(argv[1]);
    std::vector<Frame> frames;
    uint64_t previous_end = 0;
    for (uint64_t id = 1; id <= 40; ++id) {
        const uint64_t release = 1000 * kMs + (id - 1) * uint64_t{33333333};
        const uint64_t start = std::max<uint64_t>(release + 60000, previous_end + kMs);
        const uint64_t pipeline = start + 1800000;
        const uint64_t duration = (id > 20 && id <= 30) ? 34 * kMs : 20 * kMs;
        frames.push_back({id, release, start, pipeline, pipeline + duration, release + uint64_t{33333333}});
        previous_end = pipeline + duration;
    }

    v3_progress::writer writer(argv[1]);
    progress_predictor::Options options;
    options.page_path = argv[1];
    options.csv_path = argv[2];
    {
        progress_predictor::Predictor predictor(options);
        size_t next_event = 0;   // 3 events per frame: begin, pipeline, complete
        const uint64_t first = frames.front().release - 10 * kMs + 700000ULL;
        uint64_t begin = first;
        uint64_t tick = 0;
        for (uint64_t end = first + 2 * kMs; end < frames.back().complete + 10 * kMs; end += 2 * kMs) {
            while (next_event < 3 * frames.size()) {
                const Frame &f = frames[next_event / 3];
                const int kind = static_cast<int>(next_event % 3);
                const uint64_t stamp = kind == 0 ? f.start : kind == 1 ? f.pipeline : f.complete;
                if (stamp > end) break;
                if (kind == 0) writer.begin(f.id, 4, f.release, f.start, f.deadline, 0, 0, 4);
                else if (kind == 1) writer.pipeline_started(f.pipeline);
                else writer.completed(f.complete);
                ++next_event;
            }
            // every 7th group read fails: the sampler skips the tick and the
            // next read carries the counts of both windows
            if (++tick % 7 == 3) continue;
            double inst = 0;
            uint64_t inside = 0;
            for (const Frame &f : frames) {
                const uint64_t low = std::max(begin, f.pipeline), high = std::min(end, f.complete);
                if (high > low) {
                    inside += high - low;
                    const double span = static_cast<double>(f.complete - f.pipeline);
                    inst += kWork * (progress(static_cast<double>(high - f.pipeline) / span) -
                                     progress(static_cast<double>(low - f.pipeline) / span));
                }
            }
            // decode/semantic work outside the pipeline
            inst += kOutsideRate * static_cast<double>(end - begin - inside) * 0.1;
            predictor.on_window(begin, end, true, static_cast<uint64_t>(inst),
                                static_cast<uint64_t>((end - begin) * 1.7), 1000);
            begin = end;
        }
        check(predictor.frames_recorded() == 40, "40 frames recorded, got " +
              std::to_string(predictor.frames_recorded()));
        check(predictor.lost_frames() == 0, "no lost frames");
        check(std::fabs(predictor.work_model() / kWork - 1.0) < 0.12, "work model equals W");
    }

    std::ifstream csv(argv[2]);
    std::string line;
    std::getline(csv, line);
    std::vector<std::string> header;
    {
        std::stringstream fields(line);
        for (std::string field; std::getline(fields, field, ',');) header.push_back(field);
    }
    auto column = [&](const std::string &name) {
        for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return i;
        return header.size();
    };
    const size_t c_id = column("frame_id"), c_miss = column("miss"), c_work = column("work_inst"),
                 c_flag = column("t8_flag"), c_model = column("work_model_inst"),
                 c_shape = column("t8_shape_flag");
    check(c_flag < header.size() && c_shape < header.size(), "t8 flag columns present");
    int linear_false_negatives = 0, shape_decisions = 0;
    while (std::getline(csv, line)) {
        std::vector<std::string> v;
        std::stringstream fields(line);
        for (std::string field; std::getline(fields, field, ',');) v.push_back(field);
        if (line.back() == ',') v.push_back("");
        const int id = std::stoi(v[c_id]);
        // slow frames miss, and so do the frames delayed by their backlog
        const Frame &f = frames[id - 1];
        const bool slow = f.complete > f.deadline;
        // the two edge windows are credited pro rata, so the pipeline count is off
        // by at most one (possibly merged, 4 ms) window of the rate difference at
        // each end; the rates here differ 25x, and the steep start of u^0.8 adds to it
        const double work = std::stod(v[c_work]);
        check(std::fabs(work / kWork - 1.0) < 0.12, "frame " + v[c_id] + " work " + v[c_work]);
        check(std::stoi(v[c_miss]) == (slow ? 1 : 0), "frame " + v[c_id] + " miss");
        if (std::stoull(v[c_model]) == 0) {
            check(id <= 9, "model ready after 8 frames");
            check(v[c_flag] == "-1", "no decision without model");
            continue;
        }
        check(v[c_flag] != "1" || slow, "frame " + v[c_id] + " linear t8 false alarm");
        linear_false_negatives += slow && v[c_flag] == "0";
        if (v[c_shape] == "-1") {
            check(id <= 12, "frame " + v[c_id] + " shape model ready");
            continue;
        }
        ++shape_decisions;
        check(v[c_shape] == (slow ? "1" : "0"), "frame " + v[c_id] + " shape t8 flag " + v[c_shape]);
    }
    // the linear extrapolation underestimates front-loaded frames and misses
    // some of the slow ones; the shape model, learned on fast frames, does not
    check(linear_false_negatives > 0, "linear predictor shows its bias");
    check(shape_decisions >= 25, "shape decisions taken: " + std::to_string(shape_decisions));
    std::printf("linear t8 false negatives %d, shape decisions %d\n", linear_false_negatives,
                shape_decisions);
    std::printf(failures == 0 ? "progress_predictor_test: OK\n" : "progress_predictor_test: %d failures\n",
                failures);
    return failures == 0 ? 0 : 1;
}
