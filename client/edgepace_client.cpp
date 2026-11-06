// edgepace_client: the LLM load of python/edgepace_phase.py
// LoadGenerator + BulkStream in C++, for devices without Python (the phone).
//
// The tracked stream issues the request body every --period-ms (0 =
// back-to-back), optionally right after the next foreground frame completion
// (--align-task-page, the paper-era client), and mirrors each request on the
// request page; --extra-streams add back-to-back clients that are not on the
// page.  It stops after the request in flight once --stop-file appears or on
// SIGTERM/SIGINT (a second signal exits at once).
//
// Output: --csv with the columns of requests.csv (request_id, start_ns,
// complete_ns, latency_ms, status, fresh, completion_tokens, prompt_tokens,
// aligned_frame_complete_ns), and CSV.streams for the extra streams (stream,
// completed, output_tokens, first_start_ns, last_complete_ns).  CLOCK_MONOTONIC.
#include "edgepace_request_page.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<int> g_signals{0};

void on_signal(int) {
    if (g_signals.fetch_add(1) >= 1) _exit(130);
}

struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string body_path, request_page, align_task_page, csv, stop_file;
    double period_ms = 1600.0, freshness_ms = 1800.0, timeout_s = 120.0;
    int extra_streams = 0;
    long count = 0;   // tracked requests to issue; 0 = until stopped
};

struct Reply {
    int status = 0;
    long completion_tokens = 0, prompt_tokens = 0;
};

long json_number(const std::string & text, const char * key) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t at = text.find(needle);
    if (at == std::string::npos) return 0;
    const size_t colon = text.find(':', at + needle.size());
    return colon == std::string::npos ? 0 : std::strtol(text.c_str() + colon + 1, nullptr, 10);
}

// One blocking HTTP/1.1 POST over a fresh connection (llama-server answers a
// non-streaming request with Content-Length; Connection: close ends it).
Reply post(const Options & options, const std::string & path, const std::string & body) {
    Reply reply;
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return reply;
    timeval timeout{static_cast<time_t>(options.timeout_s), 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(options.port));
    ::inet_pton(AF_INET, options.host.c_str(), &address.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return reply;
    }
    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\nHost: " << options.host << "\r\n"
            << "Content-Type: application/json\r\nConnection: close\r\nContent-Length: "
            << body.size() << "\r\n\r\n" << body;
    const std::string bytes = request.str();
    for (size_t sent = 0; sent < bytes.size();) {
        const ssize_t n = ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) { ::close(fd); return reply; }
        sent += static_cast<size_t>(n);
    }
    std::string response;
    char buffer[16384];
    for (ssize_t n; (n = ::recv(fd, buffer, sizeof(buffer), 0)) > 0;) response.append(buffer, static_cast<size_t>(n));
    ::close(fd);
    if (response.rfind("HTTP/1.", 0) == 0 && response.size() > 12) reply.status = std::atoi(response.c_str() + 9);
    reply.completion_tokens = json_number(response, "completion_tokens");
    reply.prompt_tokens = json_number(response, "prompt_tokens");
    return reply;
}

// Latest completion stamp in the task page's frame ledger (0 if unreadable).
uint64_t latest_completion(const edgepace_task_snapshot_v1 * page) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        const uint64_t before = __atomic_load_n(&page->sequence, __ATOMIC_ACQUIRE);
        if (before & 1u) continue;
        edgepace_task_snapshot_v1 copy;
        std::memcpy(&copy, page, sizeof(copy));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (__atomic_load_n(&page->sequence, __ATOMIC_ACQUIRE) != before) continue;
        if (copy.header.magic != EDGEPACE_FULL_TASK_MAGIC) return 0;
        uint64_t latest = 0;
        for (const auto & frame : copy.frame_ledger)
            if ((frame.flags & EDGEPACE_TASK_FRAME_VALID) && (frame.flags & EDGEPACE_TASK_FRAME_COMPLETE))
                latest = std::max(latest, frame.complete_ns);
        return latest;
    }
    return 0;
}

bool stopping(const Options & options) {
    return g_signals.load() > 0 || (!options.stop_file.empty() && ::access(options.stop_file.c_str(), F_OK) == 0);
}

void sleep_until(uint64_t when_ns, const Options & options) {
    while (!stopping(options)) {
        const uint64_t now = edgepace::monotonic_ns();
        if (now >= when_ns) return;
        const uint64_t step = std::min<uint64_t>(when_ns - now, 1000000ULL);
        timespec ts{0, static_cast<long>(step)};
        nanosleep(&ts, nullptr);
    }
}

Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--host") o.host = value();
        else if (arg == "--port") o.port = std::stoi(value());
        else if (arg == "--body") o.body_path = value();
        else if (arg == "--request-page") o.request_page = value();
        else if (arg == "--align-task-page") o.align_task_page = value();
        else if (arg == "--csv") o.csv = value();
        else if (arg == "--stop-file") o.stop_file = value();
        else if (arg == "--period-ms") o.period_ms = std::stod(value());
        else if (arg == "--freshness-ms") o.freshness_ms = std::stod(value());
        else if (arg == "--timeout-s") o.timeout_s = std::stod(value());
        else if (arg == "--extra-streams") o.extra_streams = std::stoi(value());
        else if (arg == "--count") o.count = std::stol(value());
        else if (arg == "--help") {
            std::cout << "edgepace_client --body JSON --request-page PAGE --csv OUT [--host H] [--port P]\n"
                         "  [--period-ms 1600] [--align-task-page TASK_PAGE] [--freshness-ms 1800]\n"
                         "  [--extra-streams N] [--count N] [--stop-file PATH] [--timeout-s 120]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (o.body_path.empty() || o.request_page.empty() || o.csv.empty())
        throw std::runtime_error("--body, --request-page and --csv are required");
    return o;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse(argc, argv);
        signal(SIGTERM, on_signal);
        signal(SIGINT, on_signal);
        std::ifstream body_file(options.body_path);
        const std::string body((std::istreambuf_iterator<char>(body_file)), std::istreambuf_iterator<char>());
        if (body.empty()) throw std::runtime_error("empty request body " + options.body_path);
        const std::string path = body.find("\"prompt\"") != std::string::npos ? "/v1/completions"
                                                                             : "/v1/chat/completions";
        edgepace::request_page page(options.request_page);
        const edgepace_task_snapshot_v1 * task = nullptr;
        if (!options.align_task_page.empty()) {
            const int fd = ::open(options.align_task_page.c_str(), O_RDONLY | O_CLOEXEC);
            void * address = fd < 0 ? MAP_FAILED
                : ::mmap(nullptr, sizeof(edgepace_task_snapshot_v1), PROT_READ, MAP_SHARED, fd, 0);
            if (fd >= 0) ::close(fd);
            if (address == MAP_FAILED) throw std::runtime_error("map task page " + options.align_task_page);
            task = static_cast<const edgepace_task_snapshot_v1 *>(address);
        }

        struct Stream { long completed = 0, tokens = 0; uint64_t first = 0, last = 0; };
        std::vector<Stream> streams(static_cast<size_t>(options.extra_streams));
        std::vector<std::thread> workers;
        for (int s = 0; s < options.extra_streams; ++s) {
            workers.emplace_back([&, s] {
                while (!stopping(options)) {
                    const uint64_t start = edgepace::monotonic_ns();
                    const Reply reply = post(options, path, body);
                    if (reply.status < 200 || reply.status >= 300) { usleep(100000); continue; }
                    Stream & stream = streams[static_cast<size_t>(s)];
                    stream.first = stream.first ? stream.first : start;
                    stream.last = edgepace::monotonic_ns();
                    ++stream.completed;
                    stream.tokens += reply.completion_tokens;
                }
            });
        }

        std::ofstream csv(options.csv);
        csv << "request_id,start_ns,complete_ns,latency_ms,status,fresh,completion_tokens,prompt_tokens,"
               "aligned_frame_complete_ns\n" << std::flush;
        const uint64_t period_ns = static_cast<uint64_t>(options.period_ms * 1e6);
        const uint64_t freshness_ns = static_cast<uint64_t>(options.freshness_ms * 1e6);
        uint64_t next_start = edgepace::monotonic_ns();
        for (uint64_t request_id = 1; !stopping(options) &&
                 (options.count <= 0 || request_id <= static_cast<uint64_t>(options.count)); ++request_id) {
            uint64_t frame_complete = 0;
            if (task != nullptr) {   // wait for a frame that completes after now
                const uint64_t seen = latest_completion(task);
                while (!stopping(options) && (frame_complete = latest_completion(task)) <= seen) usleep(1000);
                if (stopping(options)) break;
            }
            const uint64_t start = edgepace::monotonic_ns();
            const uint64_t deadline = start + freshness_ns;
            page.publish(request_id, true, start, deadline);
            const Reply reply = post(options, path, body);
            const uint64_t complete = edgepace::monotonic_ns();
            page.publish(request_id, false, start, deadline, 1);
            const bool ok = reply.status >= 200 && reply.status < 300;
            char latency[32];
            std::snprintf(latency, sizeof(latency), "%.3f", (complete - start) / 1e6);
            csv << request_id << ',' << start << ',' << complete << ',' << latency << ',' << reply.status << ','
                << (ok && complete <= deadline ? 1 : 0) << ',' << reply.completion_tokens << ','
                << reply.prompt_tokens << ',' << frame_complete << '\n' << std::flush;
            next_start = std::max(next_start + period_ns, complete);
            if (options.count <= 0 || request_id < static_cast<uint64_t>(options.count)) sleep_until(next_start, options);
        }
        if (options.count > 0) g_signals.store(1);   // let the extra streams stop after --count
        for (auto & worker : workers) worker.join();
        if (options.extra_streams > 0) {
            std::ofstream out(options.csv + ".streams");
            out << "stream,completed,output_tokens,first_start_ns,last_complete_ns\n";
            for (size_t s = 0; s < streams.size(); ++s)
                out << s << ',' << streams[s].completed << ',' << streams[s].tokens << ','
                    << streams[s].first << ',' << streams[s].last << '\n';
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "edgepace_client: " << error.what() << '\n';
        return 2;
    }
}
