#include "full_sched/core.h"

#include <algorithm>
#include <chrono>

namespace full_sched {

uint64_t monotonic_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

namespace {

double clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

uint32_t q16(double value) {
    const double bounded = clamp01(value);
    return static_cast<uint32_t>(std::lround(bounded * 65535.0));
}

double from_q16(uint32_t value) {
    return static_cast<double>(value) / 65535.0;
}

double logistic(double value) {
    if (value >= 20.0) return 1.0;
    if (value <= -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-value));
}

uint64_t response_estimate_ns(const TaskView &task) {
    if (task.q90_response_ns != 0) return task.q90_response_ns;
    if (task.ewma_response_ns != 0) return task.ewma_response_ns;
    return 20000000ULL;
}

}  // namespace

ActionPrediction ActionPredictor::predict(const TaskView &task,
                                           const SignalView &signal,
                                           const VlmView &vlm,
                                           uint64_t now_ns) const {
    ActionPrediction result;
    const double demand = from_q16(signal.demand_q16) * profile_.pmu_scale;
    const double stall = from_q16(signal.stall_q16) * profile_.stall_scale;
    // A missing MC source is not a measured zero.  Keep the raw page value
    // for audit, but remove its model contribution until a fresh valid sample
    // exists; the task/PMU terms remain usable.
    const bool mc_fresh = signal.mc_valid && signal.mc_update_ns != 0 &&
        now_ns >= signal.mc_update_ns &&
        now_ns - signal.mc_update_ns <= static_cast<uint64_t>(profile_.stale_signal_ms * 1e6);
    const double mc = mc_fresh ? from_q16(signal.mc_q16) * profile_.mc_scale : 0.0;
    const double trend = std::max(0.0, static_cast<double>(signal.demand_trend_q16) / 65535.0);
    const double stall_trend = std::max(0.0, static_cast<double>(signal.stall_trend_q16) / 65535.0);
    const double response_ms = static_cast<double>(response_estimate_ns(task)) / 1e6;
    const double deadline_ms = task.next_deadline_ns > now_ns
        ? static_cast<double>(task.next_deadline_ns - now_ns) / 1e6 : -1.0;
    const double second_deadline_ms = task.second_deadline_ns > now_ns
        ? static_cast<double>(task.second_deadline_ns - now_ns) / 1e6 : deadline_ms + 33.333;
    const double graph_pressure = (task.graph_ready || task.graph_backlog || task.graph_inflight)
        ? 1.0 : 0.0;
    const double vlm_pressure = vlm.running ? 1.0 : 0.0;

    // Fixed, interpretable action model.  It intentionally uses unclipped
    // margins for ranking; q16 values are reporting fields only.  The gap
    // action removes pressure contribution but consumes a fixed 5 ms window
    // at the VLM boundary, so it is not treated as free.
    const double shared_penalty_ms = vlm_pressure *
        (2.0 + 11.0 * demand + 9.0 * stall + 7.0 * mc +
         3.0 * trend + 2.0 * stall_trend + 1.5 * graph_pressure);
    const double base_guard_ms = response_ms +
        (task.consecutive_misses > 0 ? 2.0 : 0.0) +
        (task.callback_queue_depth > 0 ? 1.0 : 0.0);
    const double full_margin_ms = deadline_ms - base_guard_ms - shared_penalty_ms;
    const double gap_margin_ms = deadline_ms - base_guard_ms -
        (vlm_pressure ? 2.0 : 0.0) - 1.5;
    const double second_full_margin_ms = second_deadline_ms - base_guard_ms - shared_penalty_ms;
    const double second_gap_margin_ms = second_deadline_ms - base_guard_ms -
        (vlm_pressure ? 2.0 : 0.0) - 1.5;
    const double min_full = std::min(full_margin_ms, second_full_margin_ms);
    const double min_gap = std::min(gap_margin_ms, second_gap_margin_ms);

    const double risk_logit_full = -0.34 * min_full + 0.65 * demand +
        0.55 * stall + 0.40 * mc + (task.consecutive_misses > 0 ? 1.0 : 0.0);
    const double risk_logit_gap = -0.34 * min_gap + 0.20 * demand +
        0.18 * stall + 0.12 * mc + (task.consecutive_misses > 0 ? 0.45 : 0.0);
    result.full_slack_ms = min_full;
    result.gap_slack_ms = min_gap;
    result.full_risk = logistic(risk_logit_full);
    result.gap_risk = logistic(risk_logit_gap);
    result.full_risk_q16 = q16(result.full_risk);
    result.gap_risk_q16 = q16(result.gap_risk);
    result.delta_risk_q16 = q16(std::max(0.0, result.full_risk - result.gap_risk));
    result.features.value = {
        response_ms, deadline_ms, second_deadline_ms, demand, stall, mc,
        trend, stall_trend, graph_pressure, vlm_pressure, min_full, min_gap};
    return result;
}

Decision BudgetStateMachine::decide(const TaskView &task, const SignalView &signal,
                                    const VlmView &vlm,
                                    const ActionPrediction &prediction,
                                    uint64_t now_ns) {
    Decision result;
    result.task_stale = task.update_ns == 0 || now_ns < task.update_ns ||
        now_ns - task.update_ns > static_cast<uint64_t>(profile_.stale_task_ms * 1e6);
    result.signal_stale = signal.update_ns == 0 || now_ns < signal.update_ns ||
        now_ns - signal.update_ns > static_cast<uint64_t>(profile_.stale_signal_ms * 1e6);
    if (result.task_stale || task.next_deadline_ns == 0) {
        result.state = scheduler_state::safe;
        result.mode = FULL_SCHED_MODE_FALLBACK_FULL;
        result.reasons = FULL_SCHED_REASON_FALLBACK | FULL_SCHED_REASON_TASK_STALE;
        if (result.signal_stale) result.reasons |= FULL_SCHED_REASON_SIGNAL_STALE;
        result.valid = false;
        ++command_epoch_;
        return result;
    }

    const bool pressure_high = prediction.full_slack_ms < profile_.safe_margin_ms;
    const bool hard_risk = prediction.full_slack_ms <= profile_.hard_guard_ms;
    const bool useful_gap = prediction.gap_slack_ms > prediction.full_slack_ms +
        profile_.action_margin_ms && prediction.delta_risk_q16 > 512;
    const bool graph_due = task.graph_next_due_ns != 0 &&
        task.graph_next_due_ns <= now_ns + 5000000ULL;
    const bool odom_laxity = prediction.full_slack_ms < profile_.safe_margin_ms;
    uint32_t reasons = 0;
    if (odom_laxity) reasons |= FULL_SCHED_REASON_ODOM_LAXITY;
    if (graph_due) reasons |= FULL_SCHED_REASON_GRAPH_DUE;
    if (signal.demand_q16 > 49152) reasons |= FULL_SCHED_REASON_PMU_DEMAND;
    if (signal.stall_q16 > 49152) reasons |= FULL_SCHED_REASON_PMU_STALL;
    if (signal.mc_valid && signal.mc_q16 > 49152) reasons |= FULL_SCHED_REASON_MC_PRESSURE;
    if (useful_gap) reasons |= FULL_SCHED_REASON_ACTION_DELTA;
    if (vlm.running && vlm.phase != FULL_SCHED_PHASE_IDLE)
        reasons |= FULL_SCHED_REASON_VLM_PHASE;
    if (result.signal_stale) reasons |= FULL_SCHED_REASON_SIGNAL_STALE;

    if (state_ == scheduler_state::protect || state_ == scheduler_state::guarded) {
        if (!pressure_high && prediction.gap_slack_ms >= profile_.safe_margin_ms) {
            ++low_streak_;
        } else {
            low_streak_ = 0;
        }
        if (low_streak_ >= 2) {
            state_ = scheduler_state::cooldown;
            low_streak_ = 0;
        }
    }
    if (state_ == scheduler_state::cooldown) {
        reasons |= FULL_SCHED_REASON_COOLDOWN;
        ++low_streak_;
        if (low_streak_ >= 2) {
            state_ = scheduler_state::safe;
            low_streak_ = 0;
        }
    }

    if (state_ == scheduler_state::safe || state_ == scheduler_state::cooldown) {
        if (hard_risk && useful_gap) state_ = scheduler_state::protect;
        else if (pressure_high && useful_gap) state_ = scheduler_state::guarded;
    } else if (state_ == scheduler_state::guarded) {
        if (hard_risk && useful_gap) state_ = scheduler_state::protect;
    }

    result.state = state_;
    result.predicted_slack_ns = static_cast<int64_t>(std::llround(
        (state_ == scheduler_state::protect ? prediction.gap_slack_ms : prediction.full_slack_ms) * 1e6));
    result.risk_q16 = state_ == scheduler_state::protect
        ? prediction.gap_risk_q16 : prediction.full_risk_q16;
    result.delta_risk_q16 = prediction.delta_risk_q16;
    result.reasons = reasons;
    if (state_ == scheduler_state::protect) result.mode = FULL_SCHED_MODE_PROTECT;
    else if (state_ == scheduler_state::guarded) result.mode = FULL_SCHED_MODE_REDUCED;
    else result.mode = FULL_SCHED_MODE_FULL;
    result.valid = true;
    ++command_epoch_;
    return result;
}

full_sched_budget_command_v1 make_command(const Decision &decision,
                                          const ActionPrediction &prediction,
                                          const TaskView &task, const SignalView &signal,
                                          uint64_t now_ns, uint64_t policy_epoch,
                                          uint64_t validity_ns, uint32_t gap_quantum_us) {
    (void)prediction;
    full_sched_budget_command_v1 command{};
    init_header(command.header, FULL_SCHED_COMMAND_MAGIC, sizeof(command));
    command.policy_epoch = policy_epoch;
    command.command_epoch = policy_epoch == 0 ? 1 : policy_epoch;
    command.valid_from_ns = now_ns;
    command.valid_until_ns = now_ns + validity_ns;
    command.mode = decision.mode;
    command.actuator = 1;  // LAYER
    command.budget_units = decision.mode == FULL_SCHED_MODE_FULL ? 65535u :
        (decision.mode == FULL_SCHED_MODE_REDUCED ? 4u : 0u);
    command.max_contiguous_units = decision.mode == FULL_SCHED_MODE_FULL ? 65535u :
        (decision.mode == FULL_SCHED_MODE_REDUCED ? 4u : 0u);
    command.reason_bitmap = decision.reasons;
    command.predicted_slack_ns = decision.predicted_slack_ns;
    command.risk_score_q16 = decision.risk_q16;
    command.delta_risk_q16 = decision.delta_risk_q16;
    command.source_task_sequence = task.sequence;
    command.source_signal_sequence = signal.sequence;
    command.gap_quantum_us = gap_quantum_us;
    command.max_gaps_per_command = decision.mode == FULL_SCHED_MODE_PROTECT ? 1u : 0u;
    command.minimum_run_units = 1u;
    return command;
}

void initialize_task_page(full_sched_task_snapshot_v1 &page) {
    std::memset(&page, 0, sizeof(page));
    init_header(page.header, FULL_SCHED_TASK_MAGIC, sizeof(page));
}

void initialize_signal_page(full_sched_signal_snapshot_v1 &page) {
    std::memset(&page, 0, sizeof(page));
    init_header(page.header, FULL_SCHED_SIGNAL_MAGIC, sizeof(page));
}

void initialize_command_page(full_sched_budget_command_v1 &page) {
    std::memset(&page, 0, sizeof(page));
    init_header(page.header, FULL_SCHED_COMMAND_MAGIC, sizeof(page));
    page.mode = FULL_SCHED_MODE_FULL;
    page.gap_quantum_us = 5000;
}

void initialize_feedback_page(full_sched_layer_feedback_v1 &page) {
    std::memset(&page, 0, sizeof(page));
    init_header(page.header, FULL_SCHED_FEEDBACK_MAGIC, sizeof(page));
}

template <typename Page>
void publish(Page &page, const Page &value) {
    const uint64_t old_sequence = __atomic_load_n(&page.sequence, __ATOMIC_RELAXED);
    __atomic_store_n(&page.sequence, old_sequence | 1u, __ATOMIC_RELEASE);
    Page copy = value;
    copy.sequence = old_sequence | 1u;
    std::memcpy(&page, &copy, sizeof(Page));
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&page.sequence, (old_sequence | 1u) + 1u, __ATOMIC_RELEASE);
}

void publish_task(full_sched_task_snapshot_v1 &page,
                  const full_sched_task_snapshot_v1 &value) { publish(page, value); }
void publish_signal(full_sched_signal_snapshot_v1 &page,
                    const full_sched_signal_snapshot_v1 &value) { publish(page, value); }
void publish_command(full_sched_budget_command_v1 &page,
                     const full_sched_budget_command_v1 &value) { publish(page, value); }
void publish_feedback(full_sched_layer_feedback_v1 &page,
                      const full_sched_layer_feedback_v1 &value) { publish(page, value); }

TaskView task_view(const full_sched_task_snapshot_v1 &page) {
    TaskView view;
    view.update_ns = page.update_ns;
    view.next_release_ns = page.next_odom_release_ns;
    view.next_deadline_ns = page.next_odom_deadline_ns;
    view.second_deadline_ns = page.second_odom_deadline_ns;
    view.last_slack_ns = page.odom_last_slack_ns;
    view.min_recent_slack_ns = page.odom_min_recent_slack_ns;
    view.ewma_response_ns = page.odom_ewma_response_ns;
    view.q90_response_ns = page.odom_q90_response_ns;
    view.consecutive_misses = page.odom_consecutive_misses;
    view.callback_queue_depth = page.callback_queue_depth;
    view.graph_ready = page.graph_ready;
    view.graph_inflight = page.graph_inflight;
    view.graph_backlog = page.graph_backlog;
    view.graph_next_due_ns = page.graph_next_due_ns;
    view.graph_cadence = page.graph_cadence;
    view.sequence = page.sequence;
    return view;
}

SignalView signal_view(const full_sched_signal_snapshot_v1 &page) {
    SignalView view;
    view.update_ns = page.update_ns;
    view.actual_interval_ns = page.actual_interval_ns;
    view.demand_q16 = page.demand_ewma_q16 ? page.demand_ewma_q16 : page.demand_pressure_q16;
    view.stall_q16 = page.stall_ewma_q16 ? page.stall_ewma_q16 : page.stall_pressure_q16;
    view.demand_trend_q16 = page.demand_trend_q16;
    view.stall_trend_q16 = page.stall_trend_q16;
    view.pmu_high_edge_epoch = page.pmu_high_edge_epoch;
    view.pmu_low_edge_epoch = page.pmu_low_edge_epoch;
    view.mc_update_ns = page.mc_update_ns;
    view.mc_age_ns = page.mc_age_ns;
    view.mc_q16 = page.mc_activity_q16;
    view.pmu_valid = page.pmu_valid;
    view.mc_valid = page.mc_valid;
    view.sequence = page.sequence;
    return view;
}

}  // namespace full_sched
