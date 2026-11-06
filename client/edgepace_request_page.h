// Single-writer request page: how an LLM client tells the EdgePace backend
// that a request is running and when its answer goes stale.  The backend splits
// graphs only while a request runs, and the policy's freshness flag F uses the
// deadline.  Layout: edgepace_request_state_v1 (llama.cpp/ggml/src/edgepace/abi.h);
// same protocol as python/edgepace_phase.py RequestPage.
//
//   edgepace::request_page page("/dev/shm/edgepace/x/request.page");
//   page.publish(id, true, start_ns, start_ns + freshness_ns);   // request starts
//   ... send it, wait for the answer ...
//   page.publish(id, false, start_ns, deadline_ns, 1);           // done (terminal)
#pragma once

#include "edgepace/abi.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace edgepace {

inline uint64_t monotonic_ns() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<uint64_t>(now.tv_sec) * 1000000000ULL + static_cast<uint64_t>(now.tv_nsec);
}

class request_page {
public:
    explicit request_page(const std::string & path) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ < 0) throw std::runtime_error("open request page " + path);
        struct stat status{};
        // an existing page of the right size is reused, never truncated under a
        // server that may have it mapped
        if (::fstat(fd_, &status) != 0 ||
            (status.st_size != sizeof(edgepace_request_state_v1) &&
             ::ftruncate(fd_, sizeof(edgepace_request_state_v1)) != 0)) {
            ::close(fd_);
            throw std::runtime_error("size request page " + path);
        }
        void * address = ::mmap(nullptr, sizeof(edgepace_request_state_v1), PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd_, 0);
        if (address == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("map request page " + path);
        }
        page_ = static_cast<edgepace_request_state_v1 *>(address);
        publish(0, false, 0, 0);
    }
    request_page(const request_page &) = delete;
    request_page & operator=(const request_page &) = delete;
    ~request_page() {
        if (page_) ::munmap(page_, sizeof(edgepace_request_state_v1));
        if (fd_ >= 0) ::close(fd_);
    }

    void publish(uint64_t request_id, bool running, uint64_t enqueue_ns,
                 uint64_t freshness_deadline_ns, uint32_t terminal = 0) {
        const uint64_t writing = __atomic_load_n(&page_->sequence, __ATOMIC_RELAXED) | 1u;
        const uint64_t now = monotonic_ns();
        edgepace_request_state_v1 value{};
        value.header.magic = EDGEPACE_FULL_REQUEST_MAGIC;
        value.header.version = EDGEPACE_ABI_VERSION;
        value.header.bytes = sizeof(edgepace_request_state_v1);
        value.sequence = writing;
        value.update_ns = now;
        value.request_id = request_id;
        value.enqueue_ns = enqueue_ns;
        value.freshness_deadline_ns = freshness_deadline_ns;
        value.last_boundary_ns = now;
        value.last_submit_ns = now;
        value.running = running ? 1u : 0u;
        value.terminal_disposition = terminal;
        __atomic_store_n(&page_->sequence, writing, __ATOMIC_RELEASE);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        const size_t offset = offsetof(edgepace_request_state_v1, sequence) + sizeof(uint64_t);
        std::memcpy(reinterpret_cast<char *>(page_) + offset, reinterpret_cast<const char *>(&value) + offset,
                    sizeof(value) - offset);
        std::memcpy(page_, &value, offsetof(edgepace_request_state_v1, sequence));
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(&page_->sequence, writing + 1, __ATOMIC_RELEASE);
    }

private:
    int fd_ = -1;
    edgepace_request_state_v1 * page_ = nullptr;
};

}  // namespace edgepace
