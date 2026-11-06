#pragma once

#include "full_sched/abi.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace full_sched {

uint64_t monotonic_ns();

template <typename Page>
class mapped_page final {
public:
    mapped_page() = default;
    mapped_page(const mapped_page &) = delete;
    mapped_page &operator=(const mapped_page &) = delete;

    void open(const std::filesystem::path &path, bool writable, bool truncate = true) {
        close();
        const int flags = writable ? (O_RDWR | O_CREAT | O_CLOEXEC) : (O_RDONLY | O_CLOEXEC);
        fd_ = ::open(path.c_str(), flags, 0600);
        if (fd_ < 0) throw std::runtime_error("open " + path.string() + ": " + std::strerror(errno));
        if (writable && truncate && ::ftruncate(fd_, static_cast<off_t>(sizeof(Page))) != 0)
            throw std::runtime_error("size " + path.string() + ": " + std::strerror(errno));
        struct stat status{};
        if (::fstat(fd_, &status) != 0 || status.st_size < static_cast<off_t>(sizeof(Page)))
            throw std::runtime_error("short mapped page " + path.string());
        const int protection = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
        void *mapping = ::mmap(nullptr, sizeof(Page), protection, MAP_SHARED, fd_, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap " + path.string());
        page_ = static_cast<Page *>(mapping);
        writable_ = writable;
    }

    void close() {
        if (page_ != nullptr) ::munmap(page_, sizeof(Page));
        if (fd_ >= 0) ::close(fd_);
        page_ = nullptr;
        fd_ = -1;
        writable_ = false;
    }

    ~mapped_page() { close(); }
    Page *get() { return page_; }
    const volatile Page *get() const { return page_; }
    bool valid() const { return page_ != nullptr; }

private:
    int fd_ = -1;
    Page *page_ = nullptr;
    bool writable_ = false;
};

struct TaskView {
    uint64_t update_ns = 0;
    uint64_t next_release_ns = 0;
    uint64_t next_deadline_ns = 0;
    uint64_t second_deadline_ns = 0;
    int64_t last_slack_ns = 0;
    int64_t min_recent_slack_ns = 0;
    uint64_t ewma_response_ns = 0;
    uint64_t q90_response_ns = 0;
    uint32_t consecutive_misses = 0;
    uint32_t callback_queue_depth = 0;
    uint32_t graph_ready = 0;
    uint32_t graph_inflight = 0;
    uint32_t graph_backlog = 0;
    uint64_t graph_next_due_ns = 0;
    uint32_t graph_cadence = 6;
    uint64_t sequence = 0;
};

struct SignalView {
    uint64_t update_ns = 0;
    uint64_t actual_interval_ns = 0;
    uint32_t demand_q16 = 0;
    uint32_t stall_q16 = 0;
    int32_t demand_trend_q16 = 0;
    int32_t stall_trend_q16 = 0;
    uint32_t pmu_high_edge_epoch = 0;
    uint32_t pmu_low_edge_epoch = 0;
    uint64_t mc_update_ns = 0;
    uint64_t mc_age_ns = 0;
    uint32_t mc_q16 = 0;
    uint32_t pmu_valid = 0;
    uint32_t mc_valid = 0;
    uint64_t sequence = 0;
};

struct VlmView {
    uint64_t request_id = 0;
    uint32_t phase = FULL_SCHED_PHASE_IDLE;
    uint32_t remaining_units = 0;
    uint32_t completed_units = 0;
    uint64_t last_boundary_ns = 0;
    bool running = false;
};

struct FeatureVector {
    std::array<double, 12> value{};
};

struct ActionPrediction {
    double full_slack_ms = 0.0;
    double gap_slack_ms = 0.0;
    double full_risk = 1.0;
    double gap_risk = 1.0;
    uint32_t full_risk_q16 = 0;
    uint32_t gap_risk_q16 = 0;
    uint32_t delta_risk_q16 = 0;
    FeatureVector features{};
};

struct PredictorProfile {
    double safe_margin_ms = 4.0;
    double hard_guard_ms = 0.0;
    double stale_signal_ms = 80.0;
    double stale_task_ms = 120.0;
    double action_margin_ms = 0.5;
    double pmu_scale = 1.0;
    double stall_scale = 1.0;
    double mc_scale = 1.0;
    uint64_t profile_epoch = 1;
};

class ActionPredictor final {
public:
    explicit ActionPredictor(PredictorProfile profile = {}): profile_(profile) {}

    ActionPrediction predict(const TaskView &task, const SignalView &signal,
                             const VlmView &vlm, uint64_t now_ns) const;
    const PredictorProfile &profile() const { return profile_; }

private:
    PredictorProfile profile_;
};

enum class scheduler_state : uint32_t {
    safe = 0,
    guarded = 1,
    protect = 2,
    cooldown = 3,
};

struct Decision {
    scheduler_state state = scheduler_state::safe;
    uint32_t mode = FULL_SCHED_MODE_FULL;
    uint32_t reasons = 0;
    int64_t predicted_slack_ns = 0;
    uint32_t risk_q16 = 0;
    uint32_t delta_risk_q16 = 0;
    bool task_stale = false;
    bool signal_stale = false;
    bool valid = false;
};

class BudgetStateMachine final {
public:
    explicit BudgetStateMachine(PredictorProfile profile = {}): profile_(profile) {}

    Decision decide(const TaskView &task, const SignalView &signal, const VlmView &vlm,
                    const ActionPrediction &prediction, uint64_t now_ns);
    scheduler_state state() const { return state_; }
    uint64_t epoch() const { return command_epoch_; }

private:
    PredictorProfile profile_;
    scheduler_state state_ = scheduler_state::safe;
    uint64_t command_epoch_ = 0;
    uint32_t low_streak_ = 0;
};

full_sched_budget_command_v1 make_command(const Decision &decision,
                                          const ActionPrediction &prediction,
                                          const TaskView &task, const SignalView &signal,
                                          uint64_t now_ns, uint64_t policy_epoch,
                                          uint64_t validity_ns = 20000000,
                                          uint32_t gap_quantum_us = 5000);

void initialize_task_page(full_sched_task_snapshot_v1 &page);
void initialize_signal_page(full_sched_signal_snapshot_v1 &page);
void initialize_command_page(full_sched_budget_command_v1 &page);
void initialize_feedback_page(full_sched_layer_feedback_v1 &page);

void publish_task(full_sched_task_snapshot_v1 &page,
                  const full_sched_task_snapshot_v1 &value);
void publish_signal(full_sched_signal_snapshot_v1 &page,
                    const full_sched_signal_snapshot_v1 &value);
void publish_command(full_sched_budget_command_v1 &page,
                     const full_sched_budget_command_v1 &value);
void publish_feedback(full_sched_layer_feedback_v1 &page,
                      const full_sched_layer_feedback_v1 &value);

TaskView task_view(const full_sched_task_snapshot_v1 &page);
SignalView signal_view(const full_sched_signal_snapshot_v1 &page);

}  // namespace full_sched
