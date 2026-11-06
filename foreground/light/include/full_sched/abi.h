#pragma once

// Fixed-layout, single-writer pages used by the full_sched online path.
// This header is intentionally C-compatible: the layer interposer and future
// platform adapters can include it without depending on the C++ coordinator.

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include <atomic>
#include <cstring>
#endif

#define FULL_SCHED_ABI_MAGIC 0x46534348u /* FSCH */
#define FULL_SCHED_ABI_VERSION 1u

#define FULL_SCHED_TASK_MAGIC 0x46535431u /* FST1 */
#define FULL_SCHED_SIGNAL_MAGIC 0x46535331u /* FSS1 */
#define FULL_SCHED_COMMAND_MAGIC 0x46534231u /* FSB1 */
#define FULL_SCHED_FEEDBACK_MAGIC 0x46534631u /* FSF1 */
#define FULL_SCHED_REQUEST_MAGIC 0x46535231u /* FSR1 */

#define FULL_SCHED_MODE_FULL 0u
#define FULL_SCHED_MODE_REDUCED 1u
#define FULL_SCHED_MODE_PROTECT 2u
#define FULL_SCHED_MODE_FALLBACK_FULL 3u

#define FULL_SCHED_PHASE_IDLE 0u
#define FULL_SCHED_PHASE_VISION 1u
#define FULL_SCHED_PHASE_PREFILL 2u
#define FULL_SCHED_PHASE_DECODE 3u
#define FULL_SCHED_PHASE_TERMINAL 4u

#define FULL_SCHED_DISPOSITION_NONE 0u
#define FULL_SCHED_DISPOSITION_GAP 1u
#define FULL_SCHED_DISPOSITION_FALLBACK 2u
#define FULL_SCHED_DISPOSITION_EXPIRED_DURING_QUIESCE 3u

#define FULL_SCHED_TERMINAL_NONE 0u
#define FULL_SCHED_TERMINAL_COMPLETED 1u
#define FULL_SCHED_TERMINAL_LATE 2u
#define FULL_SCHED_TERMINAL_FAILED 3u
#define FULL_SCHED_TERMINAL_CANCELED 4u

#define FULL_SCHED_REASON_ODOM_LAXITY (1u << 0)
#define FULL_SCHED_REASON_GRAPH_DUE (1u << 1)
#define FULL_SCHED_REASON_PMU_DEMAND (1u << 2)
#define FULL_SCHED_REASON_PMU_STALL (1u << 3)
#define FULL_SCHED_REASON_MC_PRESSURE (1u << 4)
#define FULL_SCHED_REASON_VLM_PHASE (1u << 5)
#define FULL_SCHED_REASON_ACTION_DELTA (1u << 6)
#define FULL_SCHED_REASON_SIGNAL_STALE (1u << 7)
#define FULL_SCHED_REASON_TASK_STALE (1u << 8)
#define FULL_SCHED_REASON_VLM_AGING (1u << 9)
#define FULL_SCHED_REASON_COOLDOWN (1u << 10)
#define FULL_SCHED_REASON_FAIRNESS_RECLAIM (1u << 11)
#define FULL_SCHED_REASON_LATE_ACTUATION (1u << 12)
#define FULL_SCHED_REASON_FALLBACK (1u << 13)

#define FULL_SCHED_TASK_LEDGER_CAPACITY 4u
#define FULL_SCHED_TASK_FLAG_FRAME_LEDGER (1u << 0)
#define FULL_SCHED_TASK_FRAME_VALID (1u << 0)
#define FULL_SCHED_TASK_FRAME_COMPLETE (1u << 1)
#define FULL_SCHED_TASK_FRAME_MISSED (1u << 2)

typedef struct full_sched_abi_header {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint32_t flags;
    uint32_t reserved;
} full_sched_abi_header;

// Fixed-size in-flight/completion ledger embedded in the task page.  The
// producer indexes completion by frame_id, which prevents a late completion
// from being paired with a newer frame's release/deadline.  Four entries cover
// more than three 30 Hz periods while keeping the page at 512 bytes.
typedef struct full_sched_task_frame_v1 {
    uint64_t frame_id;
    uint64_t source_ros_ns;
    uint64_t release_ns;
    uint64_t dequeue_ns;
    uint64_t complete_ns;
    uint64_t deadline_ns;
    uint32_t flags;
    uint32_t reserved0;
} full_sched_task_frame_v1;

typedef struct full_sched_task_snapshot_v1 {
    full_sched_abi_header header;
    uint64_t sequence;
    uint64_t update_ns;
    uint64_t measurement_epoch;
    uint64_t frame_id;
    uint64_t source_ros_ns;

    uint64_t odom_release_ns;
    uint64_t odom_dequeue_ns;
    uint64_t odom_complete_ns;
    uint64_t odom_deadline_ns;
    uint64_t odom_last_response_ns;
    uint64_t odom_ewma_response_ns;
    uint64_t odom_q90_response_ns;
    int64_t odom_last_slack_ns;
    int64_t odom_min_recent_slack_ns;
    int64_t odom_response_slope_q16;
    uint32_t odom_consecutive_misses;
    uint32_t callback_queue_depth;

    uint64_t next_odom_release_ns;
    uint64_t next_odom_deadline_ns;
    uint64_t second_odom_release_ns;
    uint64_t second_odom_deadline_ns;

    uint64_t graph_next_due_ns;
    uint64_t graph_last_ingress_ns;
    uint64_t graph_last_terminal_ns;
    uint64_t graph_last_commit_ns;
    uint32_t graph_ready;
    uint32_t graph_inflight;
    uint32_t graph_backlog;
    uint32_t graph_cadence;

    uint64_t critical_begin_ns;
    uint64_t critical_deadline_ns;
    uint32_t critical_kind;
    uint32_t flags;
    uint32_t frame_ledger_count;
    uint32_t frame_ledger_write_index;
    uint32_t frame_ledger_overwrites;
    uint32_t frame_ledger_unmatched_completions;
    full_sched_task_frame_v1 frame_ledger[FULL_SCHED_TASK_LEDGER_CAPACITY];
    uint8_t reserved[24];
} full_sched_task_snapshot_v1;

typedef struct full_sched_signal_snapshot_v1 {
    full_sched_abi_header header;
    uint64_t sequence;
    uint64_t update_ns;
    uint64_t sample_index;
    uint64_t actual_interval_ns;
    uint64_t read_duration_ns;

    uint64_t delta_cpu_cycles;
    uint64_t delta_inst_retired;
    uint64_t delta_stall_backend_mem;
    uint64_t delta_ll_cache_miss_rd;
    uint32_t active_tids;
    uint32_t opened_tids;
    uint32_t pmu_valid;
    uint32_t pmu_source_id;
    uint32_t pmu_time_running_q16;
    uint32_t pmu_counter_reset;

    uint32_t demand_pressure_q16;
    uint32_t stall_pressure_q16;
    uint32_t demand_ewma_q16;
    uint32_t stall_ewma_q16;
    int32_t demand_trend_q16;
    int32_t stall_trend_q16;
    uint32_t pmu_high_edge_epoch;
    uint32_t pmu_low_edge_epoch;

    uint64_t mc_update_ns;
    uint64_t mc_age_ns;
    uint64_t mc_raw_bits;
    uint32_t mc_activity_q16;
    int32_t mc_trend_q16;
    uint32_t mc_valid;
    uint32_t mc_source_id;

    uint64_t normalization_epoch;
    uint64_t dropped_samples;
    uint32_t flags;
    uint32_t reserved0;
    uint8_t reserved[304];
} full_sched_signal_snapshot_v1;

typedef struct full_sched_budget_command_v1 {
    full_sched_abi_header header;
    uint64_t sequence;
    uint64_t policy_epoch;
    uint64_t command_epoch;
    uint64_t valid_from_ns;
    uint64_t valid_until_ns;
    uint32_t mode;
    uint32_t actuator;
    uint32_t budget_units;
    uint32_t max_contiguous_units;
    uint32_t reason_bitmap;
    int64_t predicted_slack_ns;
    uint32_t risk_score_q16;
    uint32_t delta_risk_q16;
    uint64_t source_task_sequence;
    uint64_t source_signal_sequence;
    uint32_t gap_quantum_us;
    uint32_t max_gaps_per_command;
    uint32_t minimum_run_units;
    uint32_t reserved0;
} full_sched_budget_command_v1;

typedef struct full_sched_layer_feedback_v1 {
    full_sched_abi_header header;
    uint64_t sequence;
    uint64_t update_ns;
    uint64_t request_id;
    uint64_t observed_command_epoch;
    uint64_t boundary_ordinal;
    uint64_t boundary_detect_ns;
    uint64_t sync_start_ns;
    uint64_t sync_end_ns;
    uint64_t gap_end_ns;
    uint64_t resume_submit_ns;
    uint64_t cumulative_gap_ns;
    uint64_t cumulative_sync_ns;
    uint32_t phase;
    uint32_t completed_units;
    uint32_t gaps_for_command;
    uint32_t disposition;
    uint32_t fallback_flags;
    uint32_t ring_overflow;
    uint8_t reserved[120];
} full_sched_layer_feedback_v1;

// Frozen request/phase ledger.  The semantic client is the only writer;
// coordinator and the layer interposer only read it.  Keeping this as a
// separate page avoids making the layer hot path depend on the task page or
// on a request-registration socket.
typedef struct full_sched_request_state_v1 {
    full_sched_abi_header header;
    uint64_t sequence;
    uint64_t update_ns;
    uint64_t request_id;
    uint64_t enqueue_ns;
    uint64_t freshness_deadline_ns;
    uint64_t previous_source_ros_ns;
    uint64_t current_source_ros_ns;
    uint64_t pose_version;
    uint64_t map_version;
    uint64_t last_boundary_ns;
    uint64_t last_submit_ns;
    uint32_t phase;
    uint32_t running;
    uint32_t completed_units;
    uint32_t estimated_remaining_units;
    uint32_t terminal_disposition;
    uint32_t flags;
    uint8_t input_hash[32];
    uint8_t model_config_hash[32];
    uint8_t reserved[64];
} full_sched_request_state_v1;

#ifdef __cplusplus
static_assert(sizeof(full_sched_abi_header) == 16, "full_sched ABI header");
static_assert(sizeof(full_sched_task_frame_v1) == 56, "full_sched task frame");
static_assert(sizeof(full_sched_task_snapshot_v1) == 512, "full_sched task page");
static_assert(sizeof(full_sched_signal_snapshot_v1) == 512, "full_sched signal page");
static_assert(sizeof(full_sched_budget_command_v1) == 128, "full_sched command ABI");
static_assert(sizeof(full_sched_layer_feedback_v1) == 256, "full_sched feedback page");
static_assert(sizeof(full_sched_request_state_v1) == 256, "full_sched request page");
static_assert(alignof(full_sched_task_snapshot_v1) == 8, "full_sched task alignment");

namespace full_sched {

inline void init_header(full_sched_abi_header &header, uint32_t magic, uint16_t bytes) {
    header.magic = magic;
    header.version = FULL_SCHED_ABI_VERSION;
    header.bytes = bytes;
    header.flags = 0;
    header.reserved = 0;
}

template <typename Page>
inline void publish_begin(Page &page) {
    const uint64_t sequence = __atomic_load_n(&page.sequence, __ATOMIC_RELAXED);
    __atomic_store_n(&page.sequence, sequence | 1u, __ATOMIC_RELEASE);
}

template <typename Page>
inline void publish_end(Page &page, uint64_t sequence) {
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&page.sequence, (sequence | 1u) + 1u, __ATOMIC_RELEASE);
}

template <typename Page>
inline bool read_page(const volatile Page *page, Page &out, int attempts = 3) {
    if (page == nullptr) return false;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const uint64_t first = __atomic_load_n(&page->sequence, __ATOMIC_ACQUIRE);
        if (first & 1u) continue;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        std::memcpy(&out, const_cast<const Page *>(page), sizeof(Page));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        const uint64_t second = __atomic_load_n(&page->sequence, __ATOMIC_ACQUIRE);
        if (first == second && !(second & 1u)) return true;
    }
    return false;
}

inline bool command_valid(const full_sched_budget_command_v1 &command, uint64_t now_ns) {
    if (command.header.magic != FULL_SCHED_COMMAND_MAGIC ||
        command.header.version != FULL_SCHED_ABI_VERSION ||
        command.header.bytes != sizeof(command) || command.command_epoch == 0) {
        return false;
    }
    return now_ns >= command.valid_from_ns && now_ns <= command.valid_until_ns;
}

}  // namespace full_sched
#endif
