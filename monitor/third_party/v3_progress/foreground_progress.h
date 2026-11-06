#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace v3_progress {

constexpr uint64_t magic = 0x563350524f473031ULL;
constexpr uint32_t version = 1;
constexpr size_t history_capacity = 64;
enum stage : uint32_t { absent = 0, before_pipeline = 1, pipeline = 2, complete = 3 };

struct frame {
    uint64_t frame_id = 0;
    uint64_t release_ns = 0;
    uint64_t dequeue_ns = 0;
    uint64_t deadline_ns = 0;
    uint64_t pipeline_ns = 0;
    uint64_t complete_ns = 0;
    uint32_t service_class = 0;
    uint32_t current_stage = absent;
};

struct page {
    uint64_t sequence = 0;
    uint64_t magic_value = magic;
    uint32_t abi_version = version;
    uint32_t bytes = sizeof(page);
    uint64_t update_ns = 0;
    uint64_t valid_until_ns = 0;
    uint64_t next_release_ns = 0;
    uint64_t next_deadline_ns = 0;
    uint32_t next_class = 0;
    uint32_t history_count = 0;
    uint64_t total_completed = 0;
    frame current;
    frame history[history_capacity]{};
};
static_assert(sizeof(frame) == 56, "foreground progress frame ABI");
static_assert(sizeof(page) == 3712, "foreground progress page ABI");
static_assert(std::is_trivially_copyable<page>::value, "shared page must be POD");

class writer {
public:
    explicit writer(const std::string & path) {
        if (path.empty()) return;
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::runtime_error("open foreground progress page");
        if (::ftruncate(fd_, sizeof(page)) != 0) {
            ::close(fd_);
            throw std::runtime_error("size foreground progress page");
        }
        void * address = ::mmap(nullptr, sizeof(page), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (address == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("map foreground progress page");
        }
        mapped_ = static_cast<page *>(address);
        publish();
    }
    writer(const writer &) = delete;
    writer & operator=(const writer &) = delete;
    ~writer() {
        if (mapped_) ::munmap(mapped_, sizeof(page));
        if (fd_ >= 0) ::close(fd_);
    }
    void begin(uint64_t id, uint32_t service_class, uint64_t release,
               uint64_t dequeue, uint64_t deadline, uint64_t next_release,
               uint64_t next_deadline, uint32_t next_class) {
        if (!mapped_) return;
        value_.current = {id, release, dequeue, deadline, 0, 0,
                          service_class, before_pipeline};
        value_.update_ns = dequeue;
        value_.valid_until_ns = next_deadline;
        value_.next_release_ns = next_release;
        value_.next_deadline_ns = next_deadline;
        value_.next_class = next_class;
        publish();
    }
    void pipeline_started(uint64_t stamp) {
        if (!mapped_) return;
        value_.current.pipeline_ns = stamp;
        value_.current.current_stage = pipeline;
        value_.update_ns = stamp;
        publish();
    }
    void completed(uint64_t stamp) {
        if (!mapped_) return;
        value_.current.complete_ns = stamp;
        value_.current.current_stage = complete;
        value_.update_ns = stamp;
        value_.history[value_.total_completed % history_capacity] = value_.current;
        ++value_.total_completed;
        if (value_.history_count < history_capacity) ++value_.history_count;
        publish();
    }
private:
    void publish() {
        const uint64_t odd = ++sequence_;
        __atomic_store_n(&mapped_->sequence, odd, __ATOMIC_SEQ_CST);
        // Publish the odd sequence before any payload stores on weakly ordered CPUs.
        __atomic_thread_fence(__ATOMIC_RELEASE);
        std::memcpy(reinterpret_cast<char *>(mapped_) + sizeof(uint64_t),
                    reinterpret_cast<const char *>(&value_) + sizeof(uint64_t),
                    sizeof(page) - sizeof(uint64_t));
        __atomic_store_n(&mapped_->sequence, ++sequence_, __ATOMIC_RELEASE);
    }
    int fd_ = -1;
    page * mapped_ = nullptr;
    uint64_t sequence_ = 0;
    page value_{};
};

class reader {
public:
    explicit reader(std::string path) : path_(std::move(path)) {}
    reader(const reader &) = delete;
    reader & operator=(const reader &) = delete;
    ~reader() {
        if (mapped_) ::munmap(const_cast<page *>(mapped_), sizeof(page));
    }
    bool read(page & output) {
        if (!mapped_) {
            if (path_.empty()) return false;
            const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) return false;
            struct stat info{};
            if (::fstat(fd, &info) != 0 || info.st_size != sizeof(page)) {
                ::close(fd);
                return false;
            }
            void * address = ::mmap(nullptr, sizeof(page), PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            if (address == MAP_FAILED) return false;
            mapped_ = static_cast<const page *>(address);
        }
        for (int attempt = 0; attempt < 4; ++attempt) {
            const auto before = __atomic_load_n(&mapped_->sequence, __ATOMIC_ACQUIRE);
            if (before == 0 || (before & 1u)) continue;
            std::memcpy(&output, mapped_, sizeof(page));
            __atomic_thread_fence(__ATOMIC_ACQ_REL);
            const auto after = __atomic_load_n(&mapped_->sequence, __ATOMIC_ACQUIRE);
            if (before == after && !(after & 1u)) {
                return output.magic_value == magic && output.abi_version == version &&
                       output.bytes == sizeof(page) && output.history_count <= history_capacity;
            }
        }
        return false;
    }
private:
    std::string path_;
    const page * mapped_ = nullptr;
};

}  // namespace v3_progress
