#pragma once

// Online frame-to-date progress predictor.
//
// The foreground publishes its frame stages on the v3 progress page
// (release, pipeline start, completion).  For the frame whose pipeline is
// running, the sampler accumulates the counter deltas of its windows from the
// pipeline start (a straddling window contributes the overlapping fraction)
// and predicts
//
//     predicted_pipeline = elapsed * max(1, W / I(elapsed))
//     predicted_response = (pipeline_start - release) + predicted_pipeline
//
// where I are instructions retired so far and W is the work model: the median
// pipeline instruction count of the last completed frames.  The instruction
// count of a frame does not change under contention ( check: solo and
// contended medians within 1%), so W needs no solo calibration.
//
// Progress is front-loaded (the four workers finish one after another, so the
// aggregate rate falls towards the end of the pipeline), and the linear
// extrapolation above underestimates: actual/predicted is 1.36 / 1.26 / 1.17
// at 4 / 8 / 12 ms, the same solo and contended.  The shape model corrects it
// with the foreground's own recent frames: for each work fraction g (20 bins)
// it keeps the ratio  pipeline time / time to reach g  of the last completed
// frames, and predicts
//
//     shape_pipeline = elapsed * median_ratio(bin of I / W)
//
// The ratio does not change when the whole frame slows down uniformly, so it
// is learned under any load and needs no calibration either.
//
// Decisions (prediction > deadline) are recorded, one CSV row per frame, for
// both predictors. adds the gate rule calibrated in (shape,
// margin, persistence, earliest); its per-frame state is what the sampler
// publishes on the signal page for the policy.

#include "foreground_progress.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace progress_predictor {

constexpr size_t kShapeBins = 20;

struct Options {
    std::string page_path;
    std::string csv_path;
    std::vector<uint64_t> decision_ns = {4000000ULL, 8000000ULL, 12000000ULL};
    // earliest elapsed time at which the continuous alarm may fire
    uint64_t alarm_min_ns = 4000000ULL;
    size_t work_window = 32;
    size_t work_min_frames = 8;
    // gate rule ( default): the shape prediction exceeds
    // deadline * (1 + margin) on `persistence` consecutive ticks at elapsed >= min
    double gate_margin = 0.05;
    uint32_t gate_persistence = 3;
    uint64_t gate_min_ns = 6000000ULL;
};

// What the sampler publishes on the signal page.
struct State {
    uint64_t frame_id = 0;
    bool tracking = false;
    bool alarm = false;
    bool model = false;
    // smoothed shape prediction of the response (from release) for the running
    // pipeline; 0 = none yet (min over the last gate_persistence ticks)
    uint64_t pred_response_ns = 0;
};

class Predictor final {
public:
    explicit Predictor(Options options)
        : options_(std::move(options)), reader_(options_.page_path), shape_(kShapeBins) {
        if (!enabled()) return;
        std::sort(options_.decision_ns.begin(), options_.decision_ns.end());
        csv_.open(options_.csv_path);
        if (!csv_) throw std::runtime_error("cannot open prediction csv " + options_.csv_path);
        csv_ << "frame_id,release_ns,pipeline_ns,complete_ns,deadline_ns,response_ns,miss,"
                "work_inst,work_cycles,work_stall,work_model_inst,windows,invalid_windows,"
                "alarm_elapsed_ns,alarm_pred_response_ns,"
                "shape_alarm_elapsed_ns,shape_alarm_pred_response_ns,"
                "gate_alarm_elapsed_ns,gate_alarm_pred_response_ns";
        for (uint64_t tau : options_.decision_ns) {
            const std::string p = "t" + std::to_string(tau / 1000000ULL) + "_";
            csv_ << ',' << p << "elapsed_ns," << p << "inst," << p << "cycles," << p << "stall,"
                 << p << "pred_response_ns," << p << "flag,"
                 << p << "shape_pred_response_ns," << p << "shape_flag";
        }
        csv_ << '\n';
    }

    bool enabled() const { return !options_.page_path.empty() && !options_.csv_path.empty(); }

    // Counter deltas of the sampler window (begin_ns, end_ns].  ``usable`` is
    // false when the deltas cannot be trusted (counter reset, no group read).
    void on_window(uint64_t begin_ns, uint64_t end_ns, bool usable,
                   uint64_t inst, uint64_t cycles, uint64_t stall) {
        if (!enabled() || end_ns <= begin_ns) return;
        v3_progress::page page{};
        if (!reader_.read(page)) {
            ++page_failures_;
            if (tracking_) ++frame_.invalid_windows;   // window not credited
            return;
        }
        const Delta delta{begin_ns, end_ns, usable, inst, cycles, stall};
        if (tracking_) {
            const v3_progress::frame *done = completed(page, frame_.id);
            if (done != nullptr) {
                add(delta, frame_.pipeline_ns, done->complete_ns);
                decide(std::min(end_ns, done->complete_ns), true);
                close(done->complete_ns);
            } else if (page.current.frame_id != frame_.id) {
                ++lost_frames_;   // completion fell out of the history ring
                tracking_ = false;
            } else {
                add(delta, frame_.pipeline_ns, end_ns);
                decide(end_ns, false);
            }
        }
        const v3_progress::frame &current = page.current;
        if (!tracking_ && current.frame_id > last_closed_id_ && current.pipeline_ns != 0 &&
            (current.current_stage == v3_progress::pipeline ||
             current.current_stage == v3_progress::complete)) {
            start(current);
            // pipeline began before this window: earlier windows were missed,
            // so the frame's counts are partial
            if (current.pipeline_ns < begin_ns) ++frame_.invalid_windows;
            const uint64_t stop = current.current_stage == v3_progress::complete
                ? current.complete_ns : end_ns;
            add(delta, frame_.pipeline_ns, stop);
            if (current.current_stage == v3_progress::complete) {
                decide(std::min(end_ns, current.complete_ns), true);
                close(current.complete_ns);
            } else {
                decide(end_ns, false);
            }
        }
    }

    uint64_t frames_recorded() const { return frames_recorded_; }
    uint64_t page_failures() const { return page_failures_; }
    uint64_t lost_frames() const { return lost_frames_; }
    uint64_t work_model() const { return model(); }
    uint64_t gate_alarms() const { return gate_alarms_; }

    State state() const {
        State s;
        s.frame_id = frame_.id;
        s.tracking = tracking_;
        s.alarm = frame_.gate_alarm_elapsed_ns != 0;
        s.model = frame_.model > 0;
        // publish the smoothed prediction (minimum over the last
        // gate_persistence ticks past gate_min), or 0 until there are that many
        if (tracking_ && frame_.recent_shape.size() >= std::max<uint32_t>(1, options_.gate_persistence))
            s.pred_response_ns = *std::min_element(frame_.recent_shape.begin(), frame_.recent_shape.end());
        return s;
    }

private:
    struct Delta {
        uint64_t begin_ns, end_ns;
        bool usable;
        uint64_t inst, cycles, stall;
    };
    struct Decision {
        bool taken = false;
        uint64_t elapsed_ns = 0, inst = 0, cycles = 0, stall = 0, pred_response_ns = 0;
        int flag = -1;   // -1: frame finished before tau, or no work model yet
        uint64_t shape_pred_ns = 0;
        int shape_flag = -1;   // -1 also while the shape bin has too few frames
    };
    struct Frame {
        uint64_t id = 0, release_ns = 0, pipeline_ns = 0, deadline_ns = 0;
        double inst = 0, cycles = 0, stall = 0;
        uint64_t windows = 0, invalid_windows = 0, model = 0;
        uint64_t alarm_elapsed_ns = 0, alarm_pred_ns = 0;
        uint64_t shape_alarm_elapsed_ns = 0, shape_alarm_pred_ns = 0;
        uint64_t gate_alarm_elapsed_ns = 0, gate_alarm_pred_ns = 0, last_shape_ns = 0;
        uint32_t gate_run = 0;   // consecutive ticks over the gate threshold
        // shape predictions of the last gate_persistence ticks at elapsed
        // >= gate_min; their minimum is the published (smoothed) prediction
        std::vector<uint64_t> recent_shape;
        std::vector<Decision> decisions;
        std::vector<std::pair<uint64_t, double>> trajectory;   // (elapsed, inst) per tick
    };

    static const v3_progress::frame *completed(const v3_progress::page &page, uint64_t id) {
        if (page.current.frame_id == id && page.current.current_stage == v3_progress::complete)
            return &page.current;
        for (uint64_t k = 0; k < page.history_count && k < page.total_completed; ++k) {
            const auto &entry = page.history[(page.total_completed - 1 - k) %
                                             v3_progress::history_capacity];
            if (entry.frame_id == id && entry.current_stage == v3_progress::complete) return &entry;
        }
        return nullptr;
    }

    void start(const v3_progress::frame &source) {
        frame_ = Frame{};
        frame_.id = source.frame_id;
        frame_.release_ns = source.release_ns;
        frame_.pipeline_ns = source.pipeline_ns;
        frame_.deadline_ns = source.deadline_ns;
        frame_.model = model();
        frame_.decisions.assign(options_.decision_ns.size(), Decision{});
        tracking_ = true;
    }

    // Credits the part of the window that overlaps [low, high].
    void add(const Delta &delta, uint64_t low, uint64_t high) {
        const uint64_t begin = std::max(delta.begin_ns, low);
        const uint64_t end = std::min(delta.end_ns, high);
        if (end <= begin) return;
        ++frame_.windows;
        if (!delta.usable) {
            ++frame_.invalid_windows;
            return;
        }
        const double share = static_cast<double>(end - begin) /
                             static_cast<double>(delta.end_ns - delta.begin_ns);
        frame_.inst += share * static_cast<double>(delta.inst);
        frame_.cycles += share * static_cast<double>(delta.cycles);
        frame_.stall += share * static_cast<double>(delta.stall);
    }

    uint64_t predicted_response(uint64_t elapsed) const {
        const double ratio = frame_.inst > 0 ? static_cast<double>(frame_.model) / frame_.inst : 1.0;
        return (frame_.pipeline_ns - frame_.release_ns) +
               static_cast<uint64_t>(static_cast<double>(elapsed) * std::max(1.0, ratio));
    }

    double shape_median(size_t bin) const {
        if (shape_[bin].size() < options_.work_min_frames) return 0;
        std::vector<double> sorted = shape_[bin];
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        return sorted[sorted.size() / 2];
    }

    // Ratio interpolated between the two nearest bin centres; 0 while either
    // has too few frames.
    uint64_t shape_predicted_response(uint64_t elapsed) const {
        if (frame_.model == 0 || frame_.inst <= 0) return 0;
        const double position = std::clamp(frame_.inst / static_cast<double>(frame_.model) * kShapeBins - 0.5,
                                           0.0, static_cast<double>(kShapeBins - 1));
        const size_t low = std::min(kShapeBins - 2, static_cast<size_t>(position));
        const double weight = position - static_cast<double>(low);
        const double a = shape_median(low), b = shape_median(low + 1);
        if (a <= 0 || b <= 0) return 0;
        const double ratio = (1.0 - weight) * a + weight * b;
        return (frame_.pipeline_ns - frame_.release_ns) +
               static_cast<uint64_t>(static_cast<double>(elapsed) * std::max(1.0, ratio));
    }

    // For each bin centre g: pipeline time / time at which the frame reached
    // g of its own instructions (linear between ticks).
    void learn_shape(uint64_t pipeline_time) {
        frame_.trajectory.emplace_back(pipeline_time, frame_.inst);
        for (size_t bin = 0; bin < kShapeBins; ++bin) {
            const double target = (static_cast<double>(bin) + 0.5) / kShapeBins * frame_.inst;
            uint64_t e0 = 0;
            double i0 = 0;
            for (const auto &[e1, i1] : frame_.trajectory) {
                if (i1 >= target) {
                    const double t = static_cast<double>(e0) + (i1 > i0 ? (target - i0) / (i1 - i0) : 1.0) *
                                     static_cast<double>(e1 - e0);
                    if (t > 0) {
                        shape_[bin].push_back(static_cast<double>(pipeline_time) / t);
                        if (shape_[bin].size() > options_.work_window) shape_[bin].erase(shape_[bin].begin());
                    }
                    break;
                }
                e0 = e1;
                i0 = i1;
            }
        }
    }

    // ``now`` is the end of the credited interval; ``finished`` once the
    // pipeline has completed, after which no decision can be taken.
    void decide(uint64_t now, bool finished) {
        const uint64_t elapsed = now > frame_.pipeline_ns ? now - frame_.pipeline_ns : 0;
        const bool ready = frame_.model > 0 && frame_.inst > 0;
        const uint64_t budget = frame_.deadline_ns - frame_.release_ns;
        for (size_t i = 0; i < options_.decision_ns.size(); ++i) {
            Decision &d = frame_.decisions[i];
            if (d.taken || elapsed < options_.decision_ns[i]) continue;
            d.taken = true;
            if (finished) continue;   // completed before tau: nothing left to decide
            d.elapsed_ns = elapsed;
            d.inst = static_cast<uint64_t>(frame_.inst);
            d.cycles = static_cast<uint64_t>(frame_.cycles);
            d.stall = static_cast<uint64_t>(frame_.stall);
            if (ready) {
                d.pred_response_ns = predicted_response(elapsed);
                d.flag = d.pred_response_ns > budget ? 1 : 0;
                d.shape_pred_ns = shape_predicted_response(elapsed);
                if (d.shape_pred_ns) d.shape_flag = d.shape_pred_ns > budget ? 1 : 0;
            }
        }
        if (finished) return;
        frame_.trajectory.emplace_back(elapsed, frame_.inst);
        if (ready) {
            const uint64_t shape = shape_predicted_response(elapsed);
            frame_.last_shape_ns = shape;
            if (shape > 0 && elapsed >= options_.gate_min_ns) {
                frame_.recent_shape.push_back(shape);
                if (frame_.recent_shape.size() > std::max<uint32_t>(1, options_.gate_persistence))
                    frame_.recent_shape.erase(frame_.recent_shape.begin());
            }
            const bool over = shape > 0 && elapsed >= options_.gate_min_ns &&
                static_cast<double>(shape) > static_cast<double>(budget) * (1.0 + options_.gate_margin);
            frame_.gate_run = over ? frame_.gate_run + 1 : 0;
            if (frame_.gate_alarm_elapsed_ns == 0 && frame_.gate_run >= options_.gate_persistence) {
                frame_.gate_alarm_elapsed_ns = elapsed;
                frame_.gate_alarm_pred_ns = shape;
                ++gate_alarms_;
            }
        }
        if (!ready || elapsed < options_.alarm_min_ns) return;
        if (frame_.alarm_elapsed_ns == 0) {
            const uint64_t predicted = predicted_response(elapsed);
            if (predicted > budget) {
                frame_.alarm_elapsed_ns = elapsed;
                frame_.alarm_pred_ns = predicted;
            }
        }
        if (frame_.shape_alarm_elapsed_ns == 0) {
            const uint64_t predicted = shape_predicted_response(elapsed);
            if (predicted > budget) {
                frame_.shape_alarm_elapsed_ns = elapsed;
                frame_.shape_alarm_pred_ns = predicted;
            }
        }
    }

    void close(uint64_t complete_ns) {
        tracking_ = false;
        last_closed_id_ = frame_.id;
        const uint64_t response = complete_ns - frame_.release_ns;
        csv_ << frame_.id << ',' << frame_.release_ns << ',' << frame_.pipeline_ns << ','
             << complete_ns << ',' << frame_.deadline_ns << ',' << response << ','
             << (complete_ns > frame_.deadline_ns ? 1 : 0) << ','
             << static_cast<uint64_t>(frame_.inst) << ',' << static_cast<uint64_t>(frame_.cycles) << ','
             << static_cast<uint64_t>(frame_.stall) << ',' << frame_.model << ','
             << frame_.windows << ',' << frame_.invalid_windows << ',';
        if (frame_.alarm_elapsed_ns) csv_ << frame_.alarm_elapsed_ns << ',' << frame_.alarm_pred_ns;
        else csv_ << ',';
        csv_ << ',';
        if (frame_.shape_alarm_elapsed_ns)
            csv_ << frame_.shape_alarm_elapsed_ns << ',' << frame_.shape_alarm_pred_ns;
        else csv_ << ',';
        csv_ << ',';
        if (frame_.gate_alarm_elapsed_ns)
            csv_ << frame_.gate_alarm_elapsed_ns << ',' << frame_.gate_alarm_pred_ns;
        else csv_ << ',';
        for (const Decision &d : frame_.decisions) {
            if (d.elapsed_ns == 0) csv_ << ",,,,,-1,,-1";
            else {
                csv_ << ',' << d.elapsed_ns << ',' << d.inst << ',' << d.cycles << ',' << d.stall << ',';
                if (d.flag >= 0) csv_ << d.pred_response_ns;
                csv_ << ',' << d.flag << ',';
                if (d.shape_flag >= 0) csv_ << d.shape_pred_ns;
                csv_ << ',' << d.shape_flag;
            }
        }
        csv_ << '\n';
        ++frames_recorded_;
        // only frames whose whole pipeline was counted feed the work and shape models
        if (frame_.invalid_windows == 0 && frame_.inst > 0) {
            work_.push_back(static_cast<uint64_t>(frame_.inst));
            if (work_.size() > options_.work_window) work_.erase(work_.begin());
            learn_shape(complete_ns - frame_.pipeline_ns);
        }
    }

    uint64_t model() const {
        if (work_.size() < options_.work_min_frames) return 0;
        std::vector<uint64_t> sorted = work_;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        return sorted[sorted.size() / 2];
    }

    Options options_;
    v3_progress::reader reader_;
    std::ofstream csv_;
    bool tracking_ = false;
    Frame frame_;
    uint64_t last_closed_id_ = 0;
    std::vector<uint64_t> work_;
    std::vector<std::vector<double>> shape_;   // per work-fraction bin, last frames
    uint64_t frames_recorded_ = 0, page_failures_ = 0, lost_frames_ = 0, gate_alarms_ = 0;
};

}  // namespace progress_predictor
