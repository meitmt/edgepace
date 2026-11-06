#include "full_sched/core.h"
#include "progress_predictor.h"

#include <fcntl.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using full_sched::monotonic_ns;

namespace {

volatile sig_atomic_t stop_requested = 0;
void request_stop(int) { stop_requested = 1; }

constexpr uint64_t kNsPerMs = 1000000ULL;
constexpr uint64_t kNsPerSecond = 1000000000ULL;

struct Options {
    fs::path snapshot;
    fs::path csv;
    fs::path eventfd_path;
    int eventfd_fd = -1;
    fs::path pmu_dir = "/sys/bus/event_source/devices/armv8_cortex_a78";
    // last-level CPU cache read misses (paper §3.2 M_k); on cores
    // without ll_cache_miss_rd (Snapdragon 8 Elite Oryon: l2d_cache_lmiss_rd)
    std::string llc_event = "ll_cache_miss_rd";
    // memory-related backend stall cycles (paper §3.2 B_k); selectable,
    // the Oryon cores' stall_backend_mem barely follows memory contention
    std::string stall_event = "stall_backend_mem";
    std::vector<std::string> mc_paths = {
        "/sys/kernel/debug/cactmon/mc_all",
        "/sys/kernel/debug/bpmp/debug/actmon/mc_all_last_prd_activity",
        "/sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity"};
    pid_t pid = 0;
    uint64_t duration_ms = 0;
    uint64_t sample_interval_ms = 2;
    uint64_t tid_refresh_interval_ms = 100;
    uint64_t mc_interval_ms = 20;
    bool synthetic = false;
    bool inherit_threads = false;
    // Launch mode: everything after "--" is the foreground command.  The
    // sampler forks it, attaches the inherit group before exec, then releases
    // it, so every thread the foreground ever creates is counted.
    std::vector<std::string> exec_command;
    // online progress predictor, recorded only
    progress_predictor::Options prediction;
    double miss_rate_threshold = 87578405.875;
    double mpki_threshold = 32.651825;
    double stall_threshold = 0.288896292;
    double mc_low = 100000.0;
    double mc_high = 800000.0;
};

std::string read_text(const fs::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

uint64_t parse_u64(const std::string &value, const char *name) {
    uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    return result;
}

Options parse_options(int argc, char **argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--") {
            result.exec_command.assign(argv + index + 1, argv + argc);
            break;
        }
        auto value = [&](const char *name) {
            if (++index >= argc) throw std::runtime_error(std::string("missing ") + name);
            return std::string(argv[index]);
        };
        if (arg == "--snapshot") result.snapshot = value("--snapshot");
        else if (arg == "--csv") result.csv = value("--csv");
        else if (arg == "--eventfd") result.eventfd_path = value("--eventfd");
        else if (arg == "--eventfd-fd") result.eventfd_fd = static_cast<int>(parse_u64(value("--eventfd-fd"), "eventfd-fd"));
        else if (arg == "--pid") result.pid = static_cast<pid_t>(parse_u64(value("--pid"), "pid"));
        else if (arg == "--duration-ms") result.duration_ms = parse_u64(value("--duration-ms"), "duration-ms");
        else if (arg == "--sample-interval-ms") result.sample_interval_ms = parse_u64(value("--sample-interval-ms"), "sample-interval-ms");
        else if (arg == "--tid-refresh-ms") result.tid_refresh_interval_ms = parse_u64(value("--tid-refresh-ms"), "tid-refresh-ms");
        else if (arg == "--inherit") result.inherit_threads = true;
        else if (arg == "--mc-interval-ms") result.mc_interval_ms = parse_u64(value("--mc-interval-ms"), "mc-interval-ms");
        else if (arg == "--pmu-dir") result.pmu_dir = value("--pmu-dir");
        else if (arg == "--llc-event") result.llc_event = value("--llc-event");
        else if (arg == "--stall-event") result.stall_event = value("--stall-event");
        else if (arg == "--mc-path") result.mc_paths = {value("--mc-path")};
        else if (arg == "--no-mc") result.mc_paths.clear();
        else if (arg == "--synthetic") result.synthetic = true;
        else if (arg == "--progress-page") result.prediction.page_path = value("--progress-page");
        else if (arg == "--prediction-csv") result.prediction.csv_path = value("--prediction-csv");
        else if (arg == "--decision-ms") {
            result.prediction.decision_ns.clear();
            std::stringstream list(value("--decision-ms"));
            for (std::string item; std::getline(list, item, ',');)
                result.prediction.decision_ns.push_back(parse_u64(item, "decision-ms") * kNsPerMs);
        }
        else if (arg == "--alarm-min-ms") result.prediction.alarm_min_ns = parse_u64(value("--alarm-min-ms"), "alarm-min-ms") * kNsPerMs;
        else if (arg == "--work-window") result.prediction.work_window = parse_u64(value("--work-window"), "work-window");
        else if (arg == "--gate-margin") result.prediction.gate_margin = std::stod(value("--gate-margin"));
        else if (arg == "--gate-persistence") result.prediction.gate_persistence = static_cast<uint32_t>(parse_u64(value("--gate-persistence"), "gate-persistence"));
        else if (arg == "--gate-min-ms") result.prediction.gate_min_ns = parse_u64(value("--gate-min-ms"), "gate-min-ms") * kNsPerMs;
        else if (arg == "--miss-rate-threshold") result.miss_rate_threshold = std::stod(value("--miss-rate-threshold"));
        else if (arg == "--mpki-threshold") result.mpki_threshold = std::stod(value("--mpki-threshold"));
        else if (arg == "--stall-threshold") result.stall_threshold = std::stod(value("--stall-threshold"));
        else if (arg == "--mc-low") result.mc_low = std::stod(value("--mc-low"));
        else if (arg == "--mc-high") result.mc_high = std::stod(value("--mc-high"));
        else if (arg == "--help") {
            std::cout << "Usage: pmu_sampler --snapshot PATH [--pid PID|--synthetic] "
                         "[--duration-ms N] [--csv PATH] [--eventfd PATH|--eventfd-fd FD] "
                         "[--inherit] [--pmu-dir DIR] [--stall-event NAME] [--llc-event NAME] "
                         "[--progress-page PATH --prediction-csv PATH "
                         "[--decision-ms 4,8,12] [--alarm-min-ms 4] [--work-window 32] "
                         "[--gate-margin 0.05] [--gate-persistence 3] [--gate-min-ms 6]] "
                         "[-- COMMAND ARGS...]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (!result.exec_command.empty()) {
        if (result.pid > 0 || result.synthetic)
            throw std::runtime_error("-- COMMAND cannot be combined with --pid or --synthetic");
        result.inherit_threads = true;
    }
    if (result.snapshot.empty() ||
        (!result.synthetic && result.pid <= 0 && result.exec_command.empty()))
        throw std::runtime_error("--snapshot and --pid, --synthetic or -- COMMAND are required");
    if (result.sample_interval_ms == 0 || result.mc_interval_ms == 0)
        throw std::runtime_error("sampling intervals must be positive");
    if (!(result.mc_high > result.mc_low))
        throw std::runtime_error("--mc-high must be greater than --mc-low");
    if (result.prediction.page_path.empty() != result.prediction.csv_path.empty())
        throw std::runtime_error("--progress-page and --prediction-csv go together");
    if (result.prediction.decision_ns.empty() || result.prediction.work_window == 0)
        throw std::runtime_error("--decision-ms and --work-window must be non-empty");
    return result;
}

uint64_t parse_event_config(const fs::path &path) {
    const std::string text = read_text(path);
    const std::regex expression("event=0x([0-9a-fA-F]+)");
    std::smatch match;
    if (!std::regex_search(text, match, expression))
        throw std::runtime_error("cannot parse event " + path.string());
    return std::stoull(match[1].str(), nullptr, 16);
}

class PmuGroup final {
public:
    PmuGroup() = default;
    PmuGroup(const PmuGroup &) = delete;
    PmuGroup &operator=(const PmuGroup &) = delete;
    PmuGroup(PmuGroup &&other) noexcept : fds_(std::move(other.fds_)) {
        other.fds_.clear();
    }
    PmuGroup &operator=(PmuGroup &&other) noexcept {
        if (this != &other) {
            close_all();
            fds_ = std::move(other.fds_);
            other.fds_.clear();
        }
        return *this;
    }
    ~PmuGroup() { close_all(); }

    bool open_for(pid_t tid, const fs::path &pmu_dir, const std::string &stall_event,
                  const std::string &llc_event, std::string &error, bool inherit_threads = false) {
        close_all();
        try {
            const uint32_t type = static_cast<uint32_t>(std::stoul(read_text(pmu_dir / "type")));
            const std::array<std::string, 4> names = {
                "cpu_cycles", "inst_retired", stall_event, llc_event};
            for (const std::string &name : names) {
                perf_event_attr attr{};
                attr.type = type;
                attr.size = sizeof(attr);
                attr.config = parse_event_config(pmu_dir / "events" / name);
                attr.disabled = fds_.empty() ? 1 : 0;
                attr.exclude_kernel = 1;
                attr.exclude_hv = 1;
                attr.inherit = inherit_threads ? 1 : 0;
                if (fds_.empty()) attr.read_format = PERF_FORMAT_GROUP |
                    PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                const int fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, tid, -1,
                                                         fds_.empty() ? -1 : fds_.front(), 0));
                if (fd < 0) throw std::runtime_error(std::string("perf_event_open ") + name + ": " + std::strerror(errno));
                fds_.push_back(fd);
            }
            if (ioctl(fds_.front(), PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
                ioctl(fds_.front(), PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0)
                throw std::runtime_error(std::string("enable PMU: ") + std::strerror(errno));
            return true;
        } catch (const std::exception &exception) {
            error = exception.what();
            close_all();
            return false;
        }
    }

    bool read(std::array<uint64_t, 4> &values, uint64_t &duration_ns,
              double &time_running_ratio) const {
        if (fds_.empty()) return false;
        const uint64_t read_start_ns = monotonic_ns();
        std::array<uint64_t, 7> buffer{};
        const ssize_t expected = static_cast<ssize_t>(buffer.size() * sizeof(uint64_t));
        if (::read(fds_.front(), buffer.data(), buffer.size() * sizeof(uint64_t)) != expected)
            return false;
        if (buffer[0] != fds_.size()) return false;
        duration_ns = monotonic_ns() - read_start_ns;
        const uint64_t enabled = buffer[1];
        const uint64_t running = buffer[2];
        time_running_ratio = enabled == 0 ? 0.0
            : std::clamp(static_cast<double>(running) / static_cast<double>(enabled),
                         0.0, 1.0);
        for (size_t index = 0; index < values.size(); ++index) {
            if (running != 0 && enabled > running) {
                const long double scaled = static_cast<long double>(buffer[3 + index]) *
                    static_cast<long double>(enabled) / static_cast<long double>(running);
                values[index] = scaled >= static_cast<long double>(UINT64_MAX)
                    ? UINT64_MAX : static_cast<uint64_t>(scaled);
            } else {
                values[index] = buffer[3 + index];
            }
        }
        return true;
    }

private:
    void close_all() {
        for (int fd : fds_) if (fd >= 0) ::close(fd);
        fds_.clear();
    }
    std::vector<int> fds_;
};

struct ThreadState {
    PmuGroup group;
    std::array<uint64_t, 4> previous{};
    bool have_previous = false;
};

class McReader final {
public:
    explicit McReader(std::string path): path_(std::move(path)) {
        const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) error_ = std::strerror(errno);
        else ::close(fd);
    }
    const std::string &path() const { return path_; }
    bool opened() const { return error_.empty(); }
    const std::string &error() const { return error_; }
    std::optional<double> read_value() const {
        const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return std::nullopt;
        char buffer[128]{};
        ssize_t count;
        do {
            count = ::read(fd, buffer, sizeof(buffer) - 1);
        } while (count < 0 && errno == EINTR);
        ::close(fd);
        if (count <= 0) return std::nullopt;
        char *end = nullptr;
        errno = 0;
        const double value = std::strtod(buffer, &end);
        if (errno != 0 || end == buffer || !std::isfinite(value)) return std::nullopt;
        return value;
    }
private:
    std::string path_;
    std::string error_;
};

std::vector<pid_t> list_tids(pid_t pid) {
    std::vector<pid_t> tids;
    const fs::path task_dir = fs::path("/proc") / std::to_string(pid) / "task";
    std::error_code error;
    for (const auto &entry : fs::directory_iterator(task_dir, error)) {
        if (error) break;
        try { tids.push_back(static_cast<pid_t>(std::stol(entry.path().filename().string()))); }
        catch (...) {}
    }
    std::sort(tids.begin(), tids.end());
    return tids;
}

double pressure(double rate, double mpki, const Options &options) {
    if (rate <= 0.0 || mpki <= 0.0) return 0.0;
    return std::clamp(std::min(rate / options.miss_rate_threshold,
                                mpki / options.mpki_threshold), 0.0, 1.0);
}

void write_csv_header(std::ofstream &out) {
    out << "timestamp_ns,interval_ns,read_duration_ns,pmu_valid,active_tids,opened_tids,"
           "delta_cycles,delta_inst,delta_stall,delta_llc,demand_q16,stall_q16,"
           "demand_ewma_q16,stall_ewma_q16,pmu_time_running_q16,pmu_counter_reset,"
           "mc_valid,mc_q16,mc_trend_q16,mc_age_ns,high_epoch,low_epoch,window_valid\n";
}

// Fork the foreground stopped on a pipe; the caller attaches counters to the
// returned pid and then calls release_child() so the child execs.
pid_t spawn_held_child(const std::vector<std::string> &command, int &release_fd) {
    int fds[2];
    if (::pipe2(fds, O_CLOEXEC) != 0)
        throw std::runtime_error(std::string("pipe: ") + std::strerror(errno));
    const pid_t child = ::fork();
    if (child < 0) throw std::runtime_error(std::string("fork: ") + std::strerror(errno));
    if (child == 0) {
        ::close(fds[1]);
        char go = 0;
        if (::read(fds[0], &go, 1) != 1 || go != 'g') ::_exit(126);
        std::vector<char *> argv;
        for (const auto &word : command) argv.push_back(const_cast<char *>(word.c_str()));
        argv.push_back(nullptr);
        ::execvp(argv[0], argv.data());
        ::_exit(127);
    }
    ::close(fds[0]);
    release_fd = fds[1];
    return child;
}

void release_child(int &release_fd, bool run) {
    if (release_fd < 0) return;
    const char go = run ? 'g' : 'x';
    if (::write(release_fd, &go, 1) != 1) {}
    ::close(release_fd);
    release_fd = -1;
}

int exit_code(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    pid_t child = -1;
    int release_fd = -1;
    try {
        Options options = parse_options(argc, argv);
        std::signal(SIGINT, request_stop);
        std::signal(SIGTERM, request_stop);
        if (!options.exec_command.empty()) {
            child = spawn_held_child(options.exec_command, release_fd);
            options.pid = child;
        }
        fs::create_directories(options.snapshot.parent_path());
        full_sched::mapped_page<full_sched_signal_snapshot_v1> mapping;
        mapping.open(options.snapshot, true);
        full_sched_signal_snapshot_v1 initial{};
        full_sched::initialize_signal_page(initial);
        full_sched::publish_signal(*mapping.get(), initial);
        std::ofstream csv;
        if (!options.csv.empty()) {
            fs::create_directories(options.csv.parent_path());
            csv.open(options.csv);
            if (!csv) throw std::runtime_error("cannot open signal csv");
            write_csv_header(csv);
        }
        progress_predictor::Predictor predictor(options.prediction);
        int edge_fd = -1;
        if (options.eventfd_fd >= 0) {
            edge_fd = options.eventfd_fd;
        } else if (!options.eventfd_path.empty()) {
            edge_fd = ::open(options.eventfd_path.c_str(), O_WRONLY | O_CLOEXEC | O_NONBLOCK);
        }
        std::vector<McReader> mc_readers;
        for (const auto &path : options.mc_paths) mc_readers.emplace_back(path);
        uint64_t mc_open_failures = 0;
        std::string last_mc_error;
        for (const auto &reader : mc_readers) {
            if (reader.opened()) continue;
            ++mc_open_failures;
            last_mc_error = reader.path() + ": " + reader.error();
        }
        std::unordered_map<pid_t, ThreadState> threads;
        uint64_t last_refresh_ns = 0;
        uint64_t next_mc_ns = 0;
        uint64_t previous_sample_ns = 0;
        uint64_t last_counter_ns = 0;   // last tick with a successful group read
        uint64_t sample_index = 0;
        uint64_t normalization_epoch = 1;
        uint64_t high_epoch = 0;
        uint64_t low_epoch = 0;
        double demand_ewma = 0.0;
        double stall_ewma = 0.0;
        bool have_ewma = false;
        uint64_t valid_windows = 0;
        double mc_ewma = 0.0;
        double mc_trend = 0.0;
        bool high = false;
        uint64_t mc_update_ns = 0;
        uint64_t mc_age_ns = 0;
        uint64_t mc_raw_bits = 0;
        uint32_t mc_q16 = 0;
        bool mc_valid = false;
        std::string mc_source;
        uint64_t pmu_open_failures = 0;
        std::string last_pmu_error;
        const uint64_t start_ns = monotonic_ns();
        const uint64_t end_ns = options.duration_ms == 0 ? 0 : start_ns + options.duration_ms * kNsPerMs;
        uint64_t next_deadline = start_ns;

        // Inherit mode attaches one group to the process leader.  The kernel
        // accounts descendant threads in that group, so short-lived workers
        // do not require repeated /proc enumeration or per-TID opens.
        if (!options.synthetic && options.inherit_threads) {
            ThreadState state;
            std::string error;
            if (!state.group.open_for(options.pid, options.pmu_dir, options.stall_event, options.llc_event,
                                      error, true)) {
                ++pmu_open_failures;
                last_pmu_error = error;
                // In launch mode full coverage is the point; do not run the
                // foreground unobserved.
                if (child > 0) {
                    release_child(release_fd, false);
                    throw std::runtime_error("cannot attach inherit group before exec: " + error);
                }
            }
            threads.emplace(options.pid, std::move(state));
        }
        release_child(release_fd, true);
        int child_status = 0;
        bool child_exited = false;

        while (!stop_requested && (end_ns == 0 || monotonic_ns() < end_ns)) {
            if (child > 0 && ::waitpid(child, &child_status, WNOHANG) == child) {
                child_exited = true;
                break;
            }
            const uint64_t tick_start = monotonic_ns();
            if (!options.synthetic && !options.inherit_threads &&
                tick_start - last_refresh_ns >= options.tid_refresh_interval_ms * kNsPerMs) {
                last_refresh_ns = tick_start;
                const std::vector<pid_t> tids = list_tids(options.pid);
                for (pid_t tid : tids) {
                    if (threads.find(tid) != threads.end()) continue;
                    ThreadState state;
                    std::string error;
                    if (!state.group.open_for(tid, options.pmu_dir, options.stall_event, options.llc_event,
                                              error, false)) {
                        ++pmu_open_failures;
                        last_pmu_error = error;
                    }
                    threads.emplace(tid, std::move(state));
                }
                for (auto iterator = threads.begin(); iterator != threads.end();) {
                    if (std::find(tids.begin(), tids.end(), iterator->first) == tids.end())
                        iterator = threads.erase(iterator);
                    else ++iterator;
                }
            }

            if (tick_start >= next_mc_ns) {
                next_mc_ns = tick_start + options.mc_interval_ms * kNsPerMs;
                mc_valid = false;
                for (const McReader &reader : mc_readers) {
                    const auto value = reader.read_value();
                    if (!value) continue;
                    mc_valid = true;
                    mc_source = reader.path();
                    mc_update_ns = tick_start;
                    mc_age_ns = 0;
                    mc_raw_bits = 0;
                    std::memcpy(&mc_raw_bits, &*value, sizeof(mc_raw_bits));
                    const double normalized = std::clamp(
                        (*value - options.mc_low) / (options.mc_high - options.mc_low),
                        0.0, 1.0);
                    const double previous_mc_ewma = mc_ewma;
                    mc_ewma = mc_ewma == 0.0 ? normalized : 0.2 * normalized + 0.8 * mc_ewma;
                    mc_trend = mc_ewma - previous_mc_ewma;
                    mc_q16 = static_cast<uint32_t>(std::lround(mc_ewma * 65535.0));
                    break;
                }
                if (!mc_valid) {
                    mc_age_ns = mc_update_ns == 0 ? 0 : tick_start - mc_update_ns;
                }
            } else if (mc_update_ns != 0) {
                mc_age_ns = tick_start - mc_update_ns;
            }

            std::array<uint64_t, 4> deltas{};
            uint64_t read_duration_ns = 0;
            uint32_t active_tids = 0;
            uint32_t opened_tids = 0;
            bool pmu_valid = false;
            bool pmu_counter_reset = false;
            double min_running_ratio = 0.0;
            for (auto &[tid, state] : threads) {
                std::array<uint64_t, 4> values{};
                uint64_t duration_ns = 0;
                double running_ratio = 0.0;
                if (!state.group.read(values, duration_ns, running_ratio)) continue;
                ++opened_tids;
                // A newly opened or currently unscheduled perf group can
                // report time_running=0. It contributes no scaled counter
                // delta and must not invalidate every other active thread.
                if (running_ratio <= 0.0) continue;
                ++active_tids;
                pmu_valid = true;
                read_duration_ns += duration_ns;
                min_running_ratio = active_tids == 1 ? running_ratio
                    : std::min(min_running_ratio, running_ratio);
                if (state.have_previous) {
                    for (size_t i = 0; i < deltas.size(); ++i) {
                        if (values[i] < state.previous[i]) {
                            pmu_counter_reset = true;
                        } else {
                            deltas[i] += values[i] - state.previous[i];
                        }
                    }
                }
                state.previous = values;
                state.have_previous = true;
                (void)tid;
            }

            const uint64_t now = monotonic_ns();
            const uint64_t interval_ns = previous_sample_ns == 0 ? 0 : now - previous_sample_ns;
            previous_sample_ns = now;
            if (options.synthetic) {
                const double phase = static_cast<double>(sample_index % 300) / 300.0;
                const double wave = (phase > 0.35 && phase < 0.68) ? 0.85 : 0.18;
                deltas[0] = 1000000;
                deltas[1] = 600000;
                deltas[2] = static_cast<uint64_t>(deltas[0] * (0.06 + 0.30 * wave));
                deltas[3] = static_cast<uint64_t>(deltas[1] * (0.010 + 0.045 * wave));
                pmu_valid = true;
                active_tids = 1;
                opened_tids = 1;
                min_running_ratio = 1.0;
                pmu_counter_reset = false;
            }

            // Paper Eq. (1): a window with nonpositive duration, zero cycles or
            // zero instructions, or invalid/reset counters cannot supply the
            // ratios.  It neither updates the EWMA nor the published page, so
            // consumers keep the latest valid sample k and judge its age.
            const bool window_valid = pmu_valid && !pmu_counter_reset && interval_ns > 0 &&
                                      deltas[0] > 0 && deltas[1] > 0;
            // A failed group read leaves the previous values in place, so the
            // next successful read carries the counts since the last one: the
            // predictor gets that span, and skips ticks without a read.  A
            // zero-count window is still usable (idle); a reset one is not.
            if (pmu_valid) {
                if (last_counter_ns != 0)
                    predictor.on_window(last_counter_ns, now, !pmu_counter_reset,
                                        deltas[1], deltas[0], deltas[2]);
                last_counter_ns = now;
            }
            double demand = 0.0;
            double stall = 0.0;
            if (window_valid) {
                const double seconds = static_cast<double>(interval_ns) / kNsPerSecond;
                const double miss_rate = static_cast<double>(deltas[3]) / seconds;
                const double mpki = 1000.0 * static_cast<double>(deltas[3]) / deltas[1];
                const double stall_ratio = static_cast<double>(deltas[2]) / deltas[0];
                demand = pressure(miss_rate, mpki, options);
                stall = std::clamp(stall_ratio / options.stall_threshold, 0.0, 1.0);
                // Eq. (3), seeded by the first valid window
                demand_ewma = have_ewma ? 0.2 * demand + 0.8 * demand_ewma : demand;
                stall_ewma = have_ewma ? 0.2 * stall + 0.8 * stall_ewma : stall;
                have_ewma = true;
                ++valid_windows;
            }
            const bool next_high = !window_valid ? high :
                high ? demand_ewma >= 0.45 : demand_ewma >= 0.70;
            const bool edge_changed = next_high != high;
            if (next_high != high) {
                high = next_high;
                if (high) ++high_epoch;
                else ++low_epoch;
            }

            full_sched_signal_snapshot_v1 value{};
            full_sched::initialize_signal_page(value);
            value.update_ns = now;
            value.sample_index = sample_index++;
            value.actual_interval_ns = interval_ns;
            value.read_duration_ns = read_duration_ns;
            value.delta_cpu_cycles = deltas[0];
            value.delta_inst_retired = deltas[1];
            value.delta_stall_backend_mem = deltas[2];
            value.delta_ll_cache_miss_rd = deltas[3];
            value.active_tids = active_tids;
            value.opened_tids = opened_tids;
            value.pmu_valid = pmu_valid ? 1u : 0u;
            value.pmu_source_id = pmu_valid ? 1u : 0u;
            value.pmu_time_running_q16 = static_cast<uint32_t>(std::lround(
                std::clamp(min_running_ratio, 0.0, 1.0) * 65535.0));
            value.pmu_counter_reset = pmu_counter_reset ? 1u : 0u;
            value.demand_pressure_q16 = static_cast<uint32_t>(std::lround(demand * 65535.0));
            value.stall_pressure_q16 = static_cast<uint32_t>(std::lround(stall * 65535.0));
            value.demand_ewma_q16 = static_cast<uint32_t>(std::lround(demand_ewma * 65535.0));
            value.stall_ewma_q16 = static_cast<uint32_t>(std::lround(stall_ewma * 65535.0));
            value.demand_trend_q16 = static_cast<int32_t>(std::lround((demand - demand_ewma) * 65535.0));
            value.stall_trend_q16 = static_cast<int32_t>(std::lround((stall - stall_ewma) * 65535.0));
            value.pmu_high_edge_epoch = static_cast<uint32_t>(high_epoch);
            value.pmu_low_edge_epoch = static_cast<uint32_t>(low_epoch);
            value.mc_update_ns = mc_update_ns;
            value.mc_age_ns = mc_age_ns;
            value.mc_raw_bits = mc_raw_bits;
            value.mc_activity_q16 = mc_q16;
            value.mc_trend_q16 = static_cast<int32_t>(std::lround(
                std::clamp(mc_valid ? mc_trend : 0.0, -1.0, 1.0) * 65535.0));
            value.mc_valid = mc_valid ? 1u : 0u;
            value.mc_source_id = mc_valid ? (mc_source.find("cactmon") != std::string::npos ? 1u : 2u) : 0u;
            value.normalization_epoch = normalization_epoch;
            value.flags = (pmu_valid && !pmu_counter_reset) ? 0u : 1u;
            if (predictor.enabled()) {
                const progress_predictor::State progress = predictor.state();
                value.progress_frame_id = progress.frame_id;
                value.progress_update_ns = now;
                value.progress_pred_response_ns = progress.pred_response_ns;
                value.progress_state = (progress.tracking ? FULL_SCHED_PROGRESS_TRACKING : 0u) |
                                       (progress.alarm ? FULL_SCHED_PROGRESS_ALARM : 0u) |
                                       (progress.model ? FULL_SCHED_PROGRESS_MODEL : 0u);
            }
            if (window_valid) full_sched::publish_signal(*mapping.get(), value);
            if (csv) {
                csv << now << ',' << interval_ns << ',' << read_duration_ns << ',' << (pmu_valid ? 1 : 0)
                    << ',' << active_tids << ',' << opened_tids << ',' << deltas[0] << ',' << deltas[1]
                    << ',' << deltas[2] << ',' << deltas[3] << ',' << value.demand_pressure_q16 << ','
                    << value.stall_pressure_q16 << ',' << value.demand_ewma_q16 << ',' << value.stall_ewma_q16
                    << ',' << value.pmu_time_running_q16 << ',' << value.pmu_counter_reset
                    << ',' << (mc_valid ? 1 : 0) << ',' << value.mc_activity_q16 << ','
                    << value.mc_trend_q16 << ',' << mc_age_ns
                    << ',' << high_epoch << ',' << low_epoch << ',' << (window_valid ? 1 : 0) << '\n';
            }
            if (edge_fd >= 0 && edge_changed) {
                const uint64_t edge = high ? 1 : 2;
                if (::write(edge_fd, &edge, sizeof(edge)) < 0) {}
            }
            next_deadline += options.sample_interval_ms * kNsPerMs;
            timespec sleep_until{};
            sleep_until.tv_sec = static_cast<time_t>(next_deadline / kNsPerSecond);
            sleep_until.tv_nsec = static_cast<long>(next_deadline % kNsPerSecond);
            while (::clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &sleep_until, nullptr) == EINTR && !stop_requested) {}
        }
        if (edge_fd >= 0 && options.eventfd_fd < 0) ::close(edge_fd);
        if (child > 0 && !child_exited) {
            // stopped from outside or by --duration-ms: stop the foreground too
            ::kill(child, SIGTERM);
            for (int attempt = 0; attempt < 100 && !child_exited; ++attempt) {
                child_exited = ::waitpid(child, &child_status, WNOHANG) == child;
                if (!child_exited) ::usleep(100000);
            }
            if (!child_exited) {
                ::kill(child, SIGKILL);
                child_exited = ::waitpid(child, &child_status, 0) == child;
            }
        }
        std::ofstream status(options.snapshot.string() + ".status");
        if (child > 0) {
            status << "exec_pid=" << child << '\n'
                   << "exec_exit_code=" << (child_exited ? exit_code(child_status) : -1) << '\n';
        }
        status << "samples=" << sample_index << '\n'
               << "valid_windows=" << valid_windows << '\n'
               << "pmu_mode=" << (options.synthetic ? "synthetic" :
                   (options.inherit_threads ? "process_inherit_perf" : "per_tid_perf")) << '\n'
               << "normalization_epoch=" << normalization_epoch << '\n'
               << "pmu_open_failures=" << pmu_open_failures << '\n'
               << "last_pmu_error=" << last_pmu_error << '\n'
               << "mc_source=" << mc_source << '\n'
               << "mc_open_failures=" << mc_open_failures << '\n'
               << "last_mc_error=" << last_mc_error << '\n';
        if (predictor.enabled()) {
            status << "prediction_frames=" << predictor.frames_recorded() << '\n'
                   << "prediction_page_failures=" << predictor.page_failures() << '\n'
                   << "prediction_lost_frames=" << predictor.lost_frames() << '\n'
                   << "prediction_work_model_inst=" << predictor.work_model() << '\n'
                   << "prediction_gate_alarms=" << predictor.gate_alarms() << '\n';
        }
        // in launch mode the sampler reports the foreground's own outcome
        return child > 0 ? (child_exited ? exit_code(child_status) : 1) : 0;
    } catch (const std::exception &error) {
        std::cerr << "pmu_sampler: " << error.what() << '\n';
        if (child > 0) {
            release_child(release_fd, false);
            ::kill(child, SIGTERM);
            ::waitpid(child, nullptr, 0);
        }
        return 2;
    }
}
