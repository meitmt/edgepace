#pragma once

#include "full_sched/abi.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>

namespace full_sched {

inline uint64_t task_writer_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Native foreground-side writer.  It deliberately publishes only from the foreground
// worker thread; the sensor thread hands release timestamps through the Frame
// object, so there is one writer and no callback-side cross-process lock.
class task_snapshot_writer final {
public:
    task_snapshot_writer() = default;
    task_snapshot_writer(const task_snapshot_writer &) = delete;
    task_snapshot_writer &operator=(const task_snapshot_writer &) = delete;

    task_snapshot_writer(const std::filesystem::path &path, int eventfd_fd,
                         uint64_t period_ns, uint32_t graph_cadence)
        : eventfd_fd_(eventfd_fd), period_ns_(period_ns), graph_cadence_(graph_cadence) {
        if (path.empty()) return;
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::runtime_error("full task page open failed: " + path.string());
        if (::ftruncate(fd_, sizeof(full_sched_task_snapshot_v1)) != 0)
            throw std::runtime_error("full task page size failed: " + path.string());
        void *mapped = ::mmap(nullptr, sizeof(full_sched_task_snapshot_v1), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) throw std::runtime_error("full task page mmap failed: " + path.string());
        page_ = static_cast<full_sched_task_snapshot_v1 *>(mapped);
        initialize(*page_);
        initialize(value_);
        publish(value_);
        enabled_ = true;
    }

    ~task_snapshot_writer() {
        if (page_ != nullptr) ::munmap(page_, sizeof(full_sched_task_snapshot_v1));
        if (fd_ >= 0) ::close(fd_);
    }

    bool enabled() const { return enabled_; }

    void begin_frame(uint64_t frame_id, uint64_t source_ros_ns, uint64_t release_ns,
                     uint64_t dequeue_ns, uint64_t deadline_ns,
                     uint32_t queue_depth, uint32_t graph_ready,
                     uint32_t graph_inflight, uint32_t graph_backlog) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        value_.update_ns = dequeue_ns;
        value_.measurement_epoch = 1;
        value_.frame_id = frame_id;
        value_.source_ros_ns = source_ros_ns;
        value_.odom_release_ns = release_ns;
        value_.odom_dequeue_ns = dequeue_ns;
        value_.odom_complete_ns = 0;
        value_.odom_deadline_ns = deadline_ns;
        value_.next_odom_release_ns = release_ns + period_ns_;
        value_.next_odom_deadline_ns = deadline_ns + period_ns_;
        value_.second_odom_release_ns = release_ns + 2 * period_ns_;
        value_.second_odom_deadline_ns = deadline_ns + 2 * period_ns_;
        value_.callback_queue_depth = queue_depth;
        value_.graph_ready = graph_ready;
        value_.graph_inflight = graph_inflight;
        value_.graph_backlog = graph_backlog;
        value_.graph_cadence = graph_cadence_;
        value_.graph_last_ingress_ns = dequeue_ns;
        value_.graph_next_due_ns = graph_ready ? dequeue_ns : dequeue_ns + period_ns_ * graph_cadence_;
        value_.critical_begin_ns = dequeue_ns;
        value_.critical_deadline_ns = deadline_ns;
        value_.critical_kind = 1;
        upsert_frame(frame_id, source_ros_ns, release_ns, dequeue_ns, deadline_ns);
        publish(value_);
        signal_edge();
    }

    void complete(uint64_t frame_id, uint64_t complete_ns, bool missed,
                  uint32_t queue_depth,
                  uint32_t graph_ready, uint32_t graph_inflight,
                  uint32_t graph_backlog, uint64_t graph_terminal_ns,
                  uint64_t graph_commit_ns) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        full_sched_task_frame_v1 *frame = find_frame(frame_id);
        if (frame == nullptr) {
            ++value_.frame_ledger_unmatched_completions;
            publish(value_);
            signal_edge();
            return;
        }
        frame->complete_ns = complete_ns;
        frame->flags |= FULL_SCHED_TASK_FRAME_COMPLETE;
        if (missed) frame->flags |= FULL_SCHED_TASK_FRAME_MISSED;
        else frame->flags &= ~FULL_SCHED_TASK_FRAME_MISSED;
        const uint64_t response = frame->release_ns == 0 || complete_ns < frame->release_ns
            ? 0 : complete_ns - frame->release_ns;
        value_.update_ns = complete_ns;
        // Preserve the legacy single-frame view only when the completion
        // belongs to that frame.  Consumers that understand the ledger do not
        // depend on these legacy fields.
        if (value_.frame_id == frame_id) value_.odom_complete_ns = complete_ns;
        value_.odom_last_response_ns = response;
        value_.odom_ewma_response_ns = value_.odom_ewma_response_ns == 0
            ? response : (4 * value_.odom_ewma_response_ns + response) / 5;
        value_.odom_q90_response_ns = std::max(value_.odom_q90_response_ns,
                                               value_.odom_ewma_response_ns);
        value_.odom_last_slack_ns = static_cast<int64_t>(frame->deadline_ns) -
            static_cast<int64_t>(complete_ns);
        value_.odom_min_recent_slack_ns = std::min(value_.odom_min_recent_slack_ns,
                                                   value_.odom_last_slack_ns);
        value_.odom_consecutive_misses = missed ? value_.odom_consecutive_misses + 1 : 0;
        value_.callback_queue_depth = queue_depth;
        value_.graph_ready = graph_ready;
        value_.graph_inflight = graph_inflight;
        value_.graph_backlog = graph_backlog;
        value_.graph_last_terminal_ns = graph_terminal_ns;
        value_.graph_last_commit_ns = graph_commit_ns;
        publish(value_);
        signal_edge();
    }

private:
    full_sched_task_frame_v1 *find_frame(uint64_t frame_id) {
        for (auto &frame : value_.frame_ledger) {
            if ((frame.flags & FULL_SCHED_TASK_FRAME_VALID) != 0 &&
                frame.frame_id == frame_id) {
                return &frame;
            }
        }
        return nullptr;
    }

    void upsert_frame(uint64_t frame_id, uint64_t source_ros_ns,
                      uint64_t release_ns, uint64_t dequeue_ns,
                      uint64_t deadline_ns) {
        full_sched_task_frame_v1 *frame = find_frame(frame_id);
        uint32_t selected = FULL_SCHED_TASK_LEDGER_CAPACITY;
        if (frame == nullptr) {
            const uint32_t start = value_.frame_ledger_write_index %
                FULL_SCHED_TASK_LEDGER_CAPACITY;
            for (uint32_t offset = 0; offset < FULL_SCHED_TASK_LEDGER_CAPACITY; ++offset) {
                const uint32_t index = (start + offset) % FULL_SCHED_TASK_LEDGER_CAPACITY;
                const uint32_t flags = value_.frame_ledger[index].flags;
                if ((flags & FULL_SCHED_TASK_FRAME_VALID) == 0 ||
                    (flags & FULL_SCHED_TASK_FRAME_COMPLETE) != 0) {
                    selected = index;
                    break;
                }
            }
            if (selected == FULL_SCHED_TASK_LEDGER_CAPACITY) {
                // Bounded fail-safe: retain constant memory and make the loss
                // auditable instead of corrupting an unrelated completion.
                selected = start;
                ++value_.frame_ledger_overwrites;
            }
            frame = &value_.frame_ledger[selected];
            if ((frame->flags & FULL_SCHED_TASK_FRAME_VALID) == 0)
                ++value_.frame_ledger_count;
            value_.frame_ledger_write_index =
                (selected + 1) % FULL_SCHED_TASK_LEDGER_CAPACITY;
        }
        *frame = full_sched_task_frame_v1{};
        frame->frame_id = frame_id;
        frame->source_ros_ns = source_ros_ns;
        frame->release_ns = release_ns;
        frame->dequeue_ns = dequeue_ns;
        frame->deadline_ns = deadline_ns;
        frame->flags = FULL_SCHED_TASK_FRAME_VALID;
    }

    static void initialize(full_sched_task_snapshot_v1 &page) {
        std::memset(&page, 0, sizeof(page));
        full_sched::init_header(page.header, FULL_SCHED_TASK_MAGIC, sizeof(page));
        page.flags = FULL_SCHED_TASK_FLAG_FRAME_LEDGER;
    }

    void publish(const full_sched_task_snapshot_v1 &value) const {
        const uint64_t old_sequence = __atomic_load_n(&page_->sequence, __ATOMIC_RELAXED);
        __atomic_store_n(&page_->sequence, old_sequence | 1u, __ATOMIC_RELEASE);
        full_sched_task_snapshot_v1 copy = value;
        copy.sequence = old_sequence | 1u;
        std::memcpy(page_, &copy, sizeof(copy));
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(&page_->sequence, (old_sequence | 1u) + 1u, __ATOMIC_RELEASE);
    }

    void signal_edge() const {
        if (eventfd_fd_ < 0) return;
        const uint64_t edge = 1;
        if (::write(eventfd_fd_, &edge, sizeof(edge)) < 0) {}
    }

    int fd_ = -1;
    int eventfd_fd_ = -1;
    full_sched_task_snapshot_v1 *page_ = nullptr;
    full_sched_task_snapshot_v1 value_{};
    uint64_t period_ns_ = 33333333ULL;
    uint32_t graph_cadence_ = 6;
    bool enabled_ = false;
    mutable std::mutex mutex_;
};

}  // namespace full_sched
