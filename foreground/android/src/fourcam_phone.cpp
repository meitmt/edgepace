#include "fourcam_abi.h"
#include "foreground_progress.h"
#include "full_sched/task_writer.h"

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#include <opencv2/imgproc.hpp>

#include <arm_neon.h>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" bool ABinderProcess_setThreadPoolMaxThreadCount(std::uint32_t num_threads);
extern "C" void ABinderProcess_startThreadPool();

namespace {

constexpr int kCameraCount = 4;
constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
constexpr int kQuarterWidth = kWidth / 4;
constexpr int kQuarterHeight = kHeight / 4;
constexpr int32_t kColorFormatYuv420Flexible = 0x7F420888;
constexpr int32_t kColorFormatYuv420Vendor = 0x7FA30C04;
constexpr std::uint64_t kBaseLogicalBytesPerEpoch =
    static_cast<std::uint64_t>(kCameraCount) * kPixels * 139ULL / 2ULL;
constexpr std::uint64_t kSemanticExtraBytesPerEpoch =
    static_cast<std::uint64_t>(kCameraCount) * (kPixels / 64) *
    (sizeof(std::uint32_t) + sizeof(std::uint8_t));
constexpr std::uint64_t kLogicalBytesPerEpoch =
    kBaseLogicalBytesPerEpoch + kSemanticExtraBytesPerEpoch;

struct Options {
    int frames = 180;
    int warmup_frames = 12;
    std::int64_t period_us = 33333;
    std::int64_t deadline_us = 33333;
    std::filesystem::path video;
    std::filesystem::path output = "foreground.csv";
    std::filesystem::path semantic_output = "semantic_records.csv";
    std::filesystem::path task_page;
    std::filesystem::path progress_page;   // (as light)
    std::filesystem::path semantic_page;
    std::filesystem::path result_page;
    std::filesystem::path deadline_profile;
    std::string condition = "baseline";
    std::vector<int> cpu_list;
    int main_cpu = -1;
    std::vector<int> worker_cpus;
    int realtime_priority = 0;
    bool realtime_fifo = false;
    int worker_realtime_priority = 0;
    bool worker_realtime_fifo = false;
};

struct FrameRecord {
    std::uint64_t frame_id = 0;
    std::uint64_t scheduled_release_ns = 0;
    std::uint64_t actual_start_ns = 0;
    std::uint64_t deadline_ns = 0;
    double queue_wait_us = 0.0;
    double decode_copy_us = 0.0;
    double pipeline_us = 0.0;
    double semantic_aggregate_us = 0.0;
    double response_us = 0.0;
    std::int64_t deadline_us = 0;
    double slack_us = 0.0;
    bool miss = false;
    std::uint64_t checksum = 0;
    std::uint64_t result_request_id = 0;
    std::uint64_t result_source_frame_id = 0;
    std::uint64_t result_complete_ns = 0;
    double result_source_age_ms = std::numeric_limits<double>::quiet_NaN();
    std::uint32_t focus_camera = FOUR_CAMERA_COUNT;
    std::uint32_t result_command = FOUR_CAMERA_MODE_NONE;
    bool result_valid = false;
    bool result_fresh = false;
    bool result_applied = false;
    float detail_gain = 1.0F;
    std::array<four_camera_semantic_camera_v1, kCameraCount> semantics{};
};

std::uint64_t monotonic_ns() {
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        throw std::runtime_error(std::string("clock_gettime failed: ") + std::strerror(errno));
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}

std::uint64_t wall_time_ns() {
    timespec value{};
    if (::clock_gettime(CLOCK_REALTIME, &value) != 0) {
        throw std::runtime_error(std::string("clock_gettime failed: ") + std::strerror(errno));
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}

void sleep_until(std::uint64_t target_ns) {
    const timespec target{static_cast<time_t>(target_ns / 1000000000ULL),
                          static_cast<long>(target_ns % 1000000000ULL)};
    int result = 0;
    do {
        result = ::clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr);
    } while (result == EINTR);
    if (result != 0) {
        throw std::runtime_error(std::string("clock_nanosleep failed: ") +
                                 std::strerror(result));
    }
}

double us_between(std::uint64_t begin, std::uint64_t end) {
    return static_cast<double>(end - begin) / 1000.0;
}

std::vector<int> parse_cpu_list(const std::string& value) {
    std::vector<int> cpus;
    std::istringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            throw std::invalid_argument("--cpu-list contains an empty CPU ID");
        }
        std::size_t consumed = 0;
        const int cpu = std::stoi(token, &consumed);
        if (consumed != token.size() || cpu < 0 || cpu >= CPU_SETSIZE) {
            throw std::invalid_argument("invalid CPU ID in --cpu-list: " + token);
        }
        if (std::find(cpus.begin(), cpus.end(), cpu) == cpus.end()) {
            cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) {
        throw std::invalid_argument("--cpu-list must name at least one CPU");
    }
    return cpus;
}

void configure_foreground_thread(const Options& options) {
    if (!options.cpu_list.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (const int cpu : options.cpu_list) {
            CPU_SET(cpu, &set);
        }
        if (::sched_setaffinity(0, sizeof(set), &set) != 0) {
            throw std::runtime_error(std::string("sched_setaffinity failed: ") +
                                     std::strerror(errno));
        }
    }
    if (options.realtime_priority > 0) {
        const sched_param parameter{options.realtime_priority};
        const int policy = options.realtime_fifo ? SCHED_FIFO : SCHED_RR;
        if (::sched_setscheduler(0, policy, &parameter) != 0) {
            throw std::runtime_error(std::string("sched_setscheduler failed: ") +
                                     std::strerror(errno));
        }
    }
    if (options.main_cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(options.main_cpu, &set);
        if (::sched_setaffinity(0, sizeof(set), &set) != 0) {
            throw std::runtime_error(std::string("sched_setaffinity(main) failed: ") +
                                     std::strerror(errno));
        }
    }
}

void configure_worker_thread(const Options& options, std::size_t worker_index) {
    if (!options.worker_cpus.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(options.worker_cpus[worker_index % options.worker_cpus.size()], &set);
        if (::sched_setaffinity(0, sizeof(set), &set) != 0) {
            throw std::runtime_error(std::string("sched_setaffinity(worker) failed: ") +
                                     std::strerror(errno));
        }
    }
    if (options.worker_realtime_priority > 0) {
        const sched_param parameter{options.worker_realtime_priority};
        const int policy = options.worker_realtime_fifo ? SCHED_FIFO : SCHED_RR;
        if (::sched_setscheduler(0, policy, &parameter) != 0) {
            throw std::runtime_error(std::string("sched_setscheduler(worker) failed: ") +
                                     std::strerror(errno));
        }
    }
}

class InLoopResultReader {
public:
    explicit InLoopResultReader(std::filesystem::path path) : path_(std::move(path)) {}
    InLoopResultReader(const InLoopResultReader&) = delete;
    InLoopResultReader& operator=(const InLoopResultReader&) = delete;

    ~InLoopResultReader() {
        if (page_ != nullptr) {
            ::munmap(const_cast<four_camera_result_v1*>(page_), sizeof(*page_));
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    std::optional<four_camera_result_v1> read() {
        if (!ensure_mapped()) {
            return std::nullopt;
        }
        for (int attempt = 0; attempt < 4; ++attempt) {
            const std::uint64_t before = __atomic_load_n(&page_->sequence, __ATOMIC_ACQUIRE);
            if (before & 1U) {
                continue;
            }
            four_camera_result_v1 value{};
            std::memcpy(&value, page_, sizeof(value));
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const std::uint64_t after = __atomic_load_n(&page_->sequence, __ATOMIC_ACQUIRE);
            if (before == after && !(after & 1U) && value.magic == FOUR_CAMERA_RESULT_MAGIC &&
                value.version == FOUR_CAMERA_SEMANTIC_VERSION && value.bytes == sizeof(value)) {
                return value;
            }
        }
        return std::nullopt;
    }

private:
    bool ensure_mapped() {
        if (page_ != nullptr) {
            return true;
        }
        if (path_.empty()) {
            return false;
        }
        fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            return false;
        }
        struct stat status {};
        if (::fstat(fd_, &status) != 0 ||
            status.st_size < static_cast<off_t>(sizeof(four_camera_result_v1))) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        void* mapping = ::mmap(nullptr, sizeof(four_camera_result_v1), PROT_READ, MAP_SHARED, fd_, 0);
        if (mapping == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        page_ = static_cast<const four_camera_result_v1*>(mapping);
        return true;
    }

    std::filesystem::path path_;
    int fd_ = -1;
    const four_camera_result_v1* page_ = nullptr;
};

class SemanticWriter {
public:
    explicit SemanticWriter(const std::filesystem::path& path) {
        if (path.empty()) {
            return;
        }
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd_ < 0 || ::ftruncate(fd_, sizeof(*page_)) != 0) {
            throw std::runtime_error("cannot create semantic page: " + path.string());
        }
        void* mapping = ::mmap(nullptr, sizeof(*page_), PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd_, 0);
        if (mapping == MAP_FAILED) {
            throw std::runtime_error("cannot mmap semantic page: " + path.string());
        }
        page_ = static_cast<four_camera_semantic_snapshot_v1*>(mapping);
        four_camera_semantic_snapshot_v1 initial{};
        initial.magic = FOUR_CAMERA_SEMANTIC_MAGIC;
        initial.version = FOUR_CAMERA_SEMANTIC_VERSION;
        initial.bytes = sizeof(initial);
        initial.camera_count = FOUR_CAMERA_COUNT;
        publish(initial);
    }

    SemanticWriter(const SemanticWriter&) = delete;
    SemanticWriter& operator=(const SemanticWriter&) = delete;

    ~SemanticWriter() {
        if (page_ != nullptr) {
            ::munmap(page_, sizeof(*page_));
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    void write(std::uint64_t frame_id, std::uint64_t source_release_ns,
               std::uint64_t complete_ns,
               const std::array<four_camera_semantic_camera_v1, kCameraCount>& cameras) {
        if (page_ == nullptr) {
            return;
        }
        four_camera_semantic_snapshot_v1 value{};
        value.magic = FOUR_CAMERA_SEMANTIC_MAGIC;
        value.version = FOUR_CAMERA_SEMANTIC_VERSION;
        value.bytes = sizeof(value);
        value.update_ns = complete_ns;
        value.frame_id = frame_id;
        value.source_release_ns = source_release_ns;
        value.semantic_complete_ns = complete_ns;
        value.camera_count = FOUR_CAMERA_COUNT;
        value.flags = FOUR_CAMERA_SEMANTIC_VALID;
        std::copy(cameras.begin(), cameras.end(), value.cameras);
        publish(value);
    }

private:
    void publish(four_camera_semantic_snapshot_v1 value) {
        const std::uint64_t old = __atomic_load_n(&page_->sequence, __ATOMIC_RELAXED);
        value.sequence = old | 1U;
        __atomic_store_n(&page_->sequence, value.sequence, __ATOMIC_RELEASE);
        std::memcpy(page_, &value, sizeof(value));
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(&page_->sequence, value.sequence + 1U, __ATOMIC_RELEASE);
    }

    int fd_ = -1;
    four_camera_semantic_snapshot_v1* page_ = nullptr;
};

class Nv12Camera {
public:
    Nv12Camera(const std::filesystem::path& video, int offset_ms, int camera_id)
        : target_pts_us_(static_cast<std::int64_t>(offset_ms) * 1000) {
        static_cast<void>(camera_id);
        fd_ = ::open(video.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat status {};
        if (fd_ < 0 || ::fstat(fd_, &status) != 0) {
            throw std::runtime_error("cannot open camera video: " + video.string());
        }
        extractor_ = AMediaExtractor_new();
        if (extractor_ == nullptr || AMediaExtractor_setDataSourceFd(
                extractor_, fd_, 0, status.st_size) != AMEDIA_OK) {
            throw std::runtime_error("AMediaExtractor data source failed");
        }

        AMediaFormat* video_format = nullptr;
        const char* video_mime = nullptr;
        for (std::size_t track = 0; track < AMediaExtractor_getTrackCount(extractor_); ++track) {
            AMediaFormat* format = AMediaExtractor_getTrackFormat(extractor_, track);
            const char* mime = nullptr;
            if (format != nullptr && AMediaFormat_getString(
                    format, AMEDIAFORMAT_KEY_MIME, &mime) &&
                std::strncmp(mime, "video/", 6) == 0) {
                video_format = format;
                video_mime = mime;
                if (AMediaExtractor_selectTrack(extractor_, track) != AMEDIA_OK) {
                    AMediaFormat_delete(video_format);
                    throw std::runtime_error("AMediaExtractor track selection failed");
                }
                break;
            }
            if (format != nullptr) {
                AMediaFormat_delete(format);
            }
        }
        if (video_format == nullptr || video_mime == nullptr) {
            throw std::runtime_error("video track not found");
        }

        int32_t width = 0;
        int32_t height = 0;
        if (!AMediaFormat_getInt32(video_format, AMEDIAFORMAT_KEY_WIDTH, &width) ||
            !AMediaFormat_getInt32(video_format, AMEDIAFORMAT_KEY_HEIGHT, &height) ||
            width != kWidth || height != kHeight) {
            AMediaFormat_delete(video_format);
            throw std::runtime_error("video track is not 1920x1080");
        }
        codec_ = AMediaCodec_createDecoderByType(video_mime);
        if (codec_ == nullptr) {
            AMediaFormat_delete(video_format);
            throw std::runtime_error("AMediaCodec_createDecoderByType failed");
        }
        AMediaFormat_setInt32(
            video_format, AMEDIAFORMAT_KEY_COLOR_FORMAT, kColorFormatYuv420Flexible);
        // Request synchronous byte-buffer output. The device surface path returns
        // Qualcomm UBWC buffers that are not CPU-readable by the foreground ABI.
        const media_status_t configure_status = AMediaCodec_configure(
            codec_, video_format, nullptr, nullptr, 0);
        if (configure_status != AMEDIA_OK) {
            AMediaFormat_delete(video_format);
            AMediaCodec_delete(codec_);
            codec_ = nullptr;
            throw std::runtime_error("AMediaCodec_configure failed status=" +
                                     std::to_string(configure_status));
        }
        const media_status_t start_status = AMediaCodec_start(codec_);
        if (start_status != AMEDIA_OK) {
            AMediaFormat_delete(video_format);
            AMediaCodec_delete(codec_);
            codec_ = nullptr;
            throw std::runtime_error("AMediaCodec_start failed status=" +
                                     std::to_string(start_status));
        }
        started_ = true;
        AMediaFormat_delete(video_format);
        if (AMediaExtractor_seekTo(extractor_, target_pts_us_,
                                   AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC) != AMEDIA_OK) {
            throw std::runtime_error("AMediaExtractor camera offset seek failed");
        }
    }

    Nv12Camera(const Nv12Camera&) = delete;
    Nv12Camera& operator=(const Nv12Camera&) = delete;

    ~Nv12Camera() {
        if (codec_ != nullptr && started_) AMediaCodec_stop(codec_);
        if (codec_ != nullptr) AMediaCodec_delete(codec_);
        if (extractor_ != nullptr) AMediaExtractor_delete(extractor_);
        if (fd_ >= 0) ::close(fd_);
    }

    void pull(std::vector<std::uint8_t>& destination) {
        constexpr std::uint64_t kDecodeTimeoutNs = 10ULL * 1000000000ULL;
        const std::uint64_t deadline_ns = monotonic_ns() + kDecodeTimeoutNs;
        while (monotonic_ns() < deadline_ns) {
            feed_input();
            AMediaCodecBufferInfo info{};
            const ssize_t index = AMediaCodec_dequeueOutputBuffer(codec_, &info, 1000);
            if (index == AMEDIACODEC_INFO_TRY_AGAIN_LATER ||
                index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
                continue;
            }
            if (index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                AMediaFormat* output_format = AMediaCodec_getOutputFormat(codec_);
                if (output_format == nullptr) {
                    throw std::runtime_error("MediaCodec output format unavailable");
                }
                update_output_format(output_format);
                AMediaFormat_delete(output_format);
                continue;
            }
            if (index < 0) {
                throw std::runtime_error("AMediaCodec output dequeue failed");
            }
            const bool selected = info.presentationTimeUs >= target_pts_us_;
            if (selected) {
                std::size_t output_size = 0;
                std::uint8_t* output = AMediaCodec_getOutputBuffer(
                    codec_, static_cast<std::size_t>(index), &output_size);
                if (output == nullptr || info.offset < 0 ||
                    static_cast<std::size_t>(info.offset) + info.size > output_size) {
                    throw std::runtime_error("MediaCodec byte output buffer unavailable");
                }
                copy_byte_buffer(output + info.offset, static_cast<std::size_t>(info.size),
                                 destination);
            }
            if (AMediaCodec_releaseOutputBuffer(codec_, index, false) != AMEDIA_OK) {
                throw std::runtime_error("AMediaCodec output release failed");
            }
            if (!selected) {
                continue;
            }
            target_pts_us_ = info.presentationTimeUs + 1;
            return;
        }
        throw std::runtime_error("camera decode timed out before the requested frame");
    }

private:
    void feed_input() {
        if (input_eos_) return;
        const ssize_t index = AMediaCodec_dequeueInputBuffer(codec_, 0);
        if (index < 0) return;
        std::size_t capacity = 0;
        std::uint8_t* buffer = AMediaCodec_getInputBuffer(codec_, index, &capacity);
        if (buffer == nullptr) {
            throw std::runtime_error("AMediaCodec input buffer unavailable");
        }
        const ssize_t size = AMediaExtractor_readSampleData(extractor_, buffer, capacity);
        if (size < 0) {
            input_eos_ = true;
            if (AMediaCodec_queueInputBuffer(
                    codec_, index, 0, 0, 0, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) !=
                AMEDIA_OK) {
                throw std::runtime_error("AMediaCodec EOS queue failed");
            }
            return;
        }
        const std::int64_t pts_us = AMediaExtractor_getSampleTime(extractor_);
        std::uint32_t codec_flags = 0;
        if ((AMediaExtractor_getSampleFlags(extractor_) &
             AMEDIAEXTRACTOR_SAMPLE_FLAG_SYNC) != 0) {
            codec_flags |= AMEDIACODEC_BUFFER_FLAG_KEY_FRAME;
        }
        if (AMediaCodec_queueInputBuffer(
                codec_, index, 0, size, pts_us, codec_flags) != AMEDIA_OK) {
            throw std::runtime_error("AMediaCodec input queue failed");
        }
        AMediaExtractor_advance(extractor_);
    }

    void copy_byte_buffer(const std::uint8_t* source, std::size_t size,
                          std::vector<std::uint8_t>& destination) const {
        const std::size_t required = static_cast<std::size_t>(output_slice_height_) *
                                     static_cast<std::size_t>(output_stride_) * 3 / 2;
        if (output_stride_ < kWidth || output_slice_height_ < kHeight || size < required ||
            (output_color_format_ != kColorFormatYuv420Flexible &&
             output_color_format_ != kColorFormatYuv420Vendor)) {
            throw std::runtime_error("MediaCodec byte output format is not linear YUV420");
        }
        for (int row = 0; row < kHeight; ++row) {
            std::memcpy(destination.data() + static_cast<std::size_t>(row) * kWidth,
                        source + static_cast<std::size_t>(row) * output_stride_, kWidth);
        }
        const std::uint8_t* source_uv = source +
            static_cast<std::size_t>(output_slice_height_) * output_stride_;
        std::uint8_t* destination_uv = destination.data() + kPixels;
        for (int row = 0; row < kHeight / 2; ++row) {
            std::memcpy(destination_uv + static_cast<std::size_t>(row) * kWidth,
                        source_uv + static_cast<std::size_t>(row) * output_stride_, kWidth);
        }
    }

    void update_output_format(AMediaFormat* format) {
        if (!AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT,
                                   &output_color_format_) ||
            !AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_STRIDE, &output_stride_) ||
            !AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_SLICE_HEIGHT,
                                   &output_slice_height_)) {
            throw std::runtime_error("MediaCodec output format lacks YUV layout");
        }
    }

    int fd_ = -1;
    AMediaExtractor* extractor_ = nullptr;
    AMediaCodec* codec_ = nullptr;
    std::int64_t target_pts_us_ = 0;
    bool input_eos_ = false;
    bool started_ = false;
    int32_t output_color_format_ = 0;
    int32_t output_stride_ = 0;
    int32_t output_slice_height_ = 0;
};

struct PipelineResult {
    std::uint64_t checksum = 0;
    four_camera_semantic_camera_v1 semantic{};
};

std::uint32_t fraction_q16(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        return 0;
    }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
        65535, (numerator * 65535ULL + denominator / 2) / denominator));
}

class CameraPipeline {
public:
    CameraPipeline()
        : decoded_(kPixels * 3 / 2),
          rgba_{std::vector<std::uint8_t>(kPixels * 4),
                std::vector<std::uint8_t>(kPixels * 4)},
          denoised_{std::vector<std::uint8_t>(kPixels * 4),
                    std::vector<std::uint8_t>(kPixels * 4)},
          gradient_(kPixels), motion_(kPixels), mask_(kPixels),
          quarter_a_(kPixels / 16), quarter_b_(kPixels / 16),
          composite_(kPixels * 4), output_nv12_(kPixels * 3 / 2) {}

    std::vector<std::uint8_t>& decoded() { return decoded_; }

    PipelineResult run(float detail_gain) {
        const int current = current_;
        const int previous = 1 - current;
        auto& rgba = rgba_[current];
        auto& previous_rgba = rgba_[previous];
        auto& denoised = denoised_[current];
        auto& previous_denoised = denoised_[previous];
        cv::Mat nv12(kHeight * 3 / 2, kWidth, CV_8UC1, decoded_.data());
        cv::Mat rgba_mat(kHeight, kWidth, CV_8UC4, rgba.data());
        cv::cvtColor(nv12, rgba_mat, cv::COLOR_YUV2RGBA_NV12);

        for (std::size_t pixel = 0; pixel < kPixels; pixel += 16) {
            const uint8x16x4_t current_value = vld4q_u8(rgba.data() + pixel * 4);
            const uint8x16x4_t previous_value = vld4q_u8(previous_rgba.data() + pixel * 4);
            uint8x16x4_t temporal{};
            for (int channel = 0; channel < 4; ++channel) {
                temporal.val[channel] = vrhaddq_u8(
                    current_value.val[channel],
                    vrhaddq_u8(current_value.val[channel], previous_value.val[channel]));
            }
            vst4q_u8(denoised.data() + pixel * 4, temporal);

            const uint8x16_t rg = vabdq_u8(temporal.val[0], temporal.val[1]);
            const uint8x16_t gb = vabdq_u8(temporal.val[1], temporal.val[2]);
            vst1q_u16(gradient_.data() + pixel,
                      vaddl_u8(vget_low_u8(rg), vget_low_u8(gb)));
            vst1q_u16(gradient_.data() + pixel + 8,
                      vaddl_u8(vget_high_u8(rg), vget_high_u8(gb)));

            const uint8x16x4_t old_denoised =
                vld4q_u8(previous_denoised.data() + pixel * 4);
            uint16x8_t low = vaddl_u8(
                vget_low_u8(vabdq_u8(temporal.val[0], old_denoised.val[0])),
                vget_low_u8(vabdq_u8(temporal.val[1], old_denoised.val[1])));
            low = vaddw_u8(low,
                vget_low_u8(vabdq_u8(temporal.val[2], old_denoised.val[2])));
            low = vaddw_u8(low,
                vget_low_u8(vabdq_u8(temporal.val[3], old_denoised.val[3])));
            uint16x8_t high = vaddl_u8(
                vget_high_u8(vabdq_u8(temporal.val[0], old_denoised.val[0])),
                vget_high_u8(vabdq_u8(temporal.val[1], old_denoised.val[1])));
            high = vaddw_u8(high,
                vget_high_u8(vabdq_u8(temporal.val[2], old_denoised.val[2])));
            high = vaddw_u8(high,
                vget_high_u8(vabdq_u8(temporal.val[3], old_denoised.val[3])));
            vst1q_u32(motion_.data() + pixel, vmovl_u16(vget_low_u16(low)));
            vst1q_u32(motion_.data() + pixel + 4, vmovl_u16(vget_high_u16(low)));
            vst1q_u32(motion_.data() + pixel + 8, vmovl_u16(vget_low_u16(high)));
            vst1q_u32(motion_.data() + pixel + 12, vmovl_u16(vget_high_u16(high)));

            const uint16x8_t gradient_low = vld1q_u16(gradient_.data() + pixel);
            const uint16x8_t gradient_high = vld1q_u16(gradient_.data() + pixel + 8);
            const uint16x8_t motion_low = vcombine_u16(
                vqmovn_u32(vld1q_u32(motion_.data() + pixel)),
                vqmovn_u32(vld1q_u32(motion_.data() + pixel + 4)));
            const uint16x8_t motion_high = vcombine_u16(
                vqmovn_u32(vld1q_u32(motion_.data() + pixel + 8)),
                vqmovn_u32(vld1q_u32(motion_.data() + pixel + 12)));
            const uint16x8_t score_low = vqaddq_u16(gradient_low, motion_low);
            const uint16x8_t score_high = vqaddq_u16(gradient_high, motion_high);
            const uint8x16_t strong = vcombine_u8(
                vqmovn_u16(vcgtq_u16(score_low, vdupq_n_u16(72))),
                vqmovn_u16(vcgtq_u16(score_high, vdupq_n_u16(72))));
            const uint8x16_t weak = vcombine_u8(
                vqmovn_u16(vcgtq_u16(score_low, vdupq_n_u16(28))),
                vqmovn_u16(vcgtq_u16(score_high, vdupq_n_u16(28))));
            const uint8x16_t mask = vbslq_u8(strong, vdupq_n_u8(255),
                                              vbslq_u8(weak, vdupq_n_u8(128),
                                                       vdupq_n_u8(0)));
            vst1q_u8(mask_.data() + pixel, mask);
        }

        PipelineResult result;
        for (std::size_t index = 0; index < quarter_a_.size(); ++index) {
            quarter_a_[index] = mask_[(index * 16 + (index & 3U)) % kPixels];
        }
        for (int pass = 0; pass < 4; ++pass) {
            auto& source = (pass & 1) == 0 ? quarter_a_ : quarter_b_;
            auto& target = (pass & 1) == 0 ? quarter_b_ : quarter_a_;
            for (std::size_t index = 0; index < source.size(); ++index) {
                target[index] = source[index] >= (pass < 2 ? 96 : 128) ? 255 : 0;
            }
        }

        const bool fine = detail_gain > 1.05F;
        const bool coarse = detail_gain < 0.95F;
        for (std::size_t pixel = 0; pixel < kPixels; pixel += 16) {
            uint8x16x4_t values = vld4q_u8(denoised.data() + pixel * 4);
            const uint8x16_t active = vcgtq_u8(vld1q_u8(mask_.data() + pixel),
                                               vdupq_n_u8(0));
            for (int channel = 0; channel < 3; ++channel) {
                const uint8x16_t adjusted = fine
                    ? vqaddq_u8(values.val[channel], vdupq_n_u8(12))
                    : coarse ? vqsubq_u8(values.val[channel], vdupq_n_u8(12))
                             : values.val[channel];
                values.val[channel] = vbslq_u8(active, adjusted, values.val[channel]);
            }
            const uint8x16_t strong = vceqq_u8(vld1q_u8(mask_.data() + pixel),
                                               vdupq_n_u8(255));
            // Composition consumes the enhanced value immediately. Keeping
            // this in one pass removes a full-frame RGBA store/load pair.
            values.val[0] = vbslq_u8(
                strong, vqaddq_u8(values.val[0], vdupq_n_u8(12)), values.val[0]);
            vst4q_u8(composite_.data() + pixel * 4, values);
        }

        cv::Mat composite_mat(kHeight, kWidth, CV_8UC4, composite_.data());
        cv::Mat yuv420(kHeight * 3 / 2, kWidth, CV_8UC1, output_nv12_.data());
        cv::cvtColor(composite_mat, yuv420, cv::COLOR_RGBA2YUV_I420);

        current_ = previous;
        result.checksum = static_cast<std::uint64_t>(output_nv12_[17]) +
                          static_cast<std::uint64_t>(output_nv12_[kPixels / 2]) +
                          static_cast<std::uint64_t>(output_nv12_.back());
        return result;
    }

    four_camera_semantic_camera_v1 aggregate_semantic() {
        four_camera_semantic_camera_v1 semantic{};
        std::uint64_t active_count = 0;
        std::uint64_t strong_count = 0;
        std::uint64_t motion_count = 0;
        std::uint64_t centroid_x_sum = 0;
        std::uint64_t centroid_y_sum = 0;
        std::array<std::uint64_t, 4> zone_active{};
        std::array<std::uint64_t, 4> zone_samples{};
        for (int quarter_y = 0; quarter_y < kQuarterHeight; quarter_y += 2) {
            const int source_y = quarter_y * 4 + 2;
            for (int quarter_x = 0; quarter_x < kQuarterWidth; quarter_x += 2) {
                const int source_x = quarter_x * 4 + 2;
                const std::size_t source_index =
                    static_cast<std::size_t>(source_y) * kWidth + source_x;
                const std::uint8_t mask_value = mask_[source_index];
                const bool strong = mask_value == 255;
                const bool high_motion = motion_[source_index] > 192;
                const bool active = strong && high_motion;
                const std::size_t zone =
                    static_cast<std::size_t>(quarter_y >= kQuarterHeight / 2) * 2 +
                    static_cast<std::size_t>(quarter_x >= kQuarterWidth / 2);
                ++zone_samples[zone];
                if (active) {
                    ++active_count;
                    ++zone_active[zone];
                    centroid_x_sum += static_cast<std::uint64_t>(quarter_x);
                    centroid_y_sum += static_cast<std::uint64_t>(quarter_y);
                }
                strong_count += strong ? 1ULL : 0ULL;
                motion_count += high_motion ? 1ULL : 0ULL;
            }
        }
        const std::uint64_t sampled =
            static_cast<std::uint64_t>(kQuarterWidth / 2) * (kQuarterHeight / 2);
        semantic.active_q16 = fraction_q16(active_count, sampled);
        semantic.strong_q16 = fraction_q16(strong_count, sampled);
        semantic.motion_q16 = fraction_q16(motion_count, sampled);
        semantic.activity_trend_q16 = static_cast<std::int32_t>(semantic.active_q16) -
                                      static_cast<std::int32_t>(previous_active_q16_);
        previous_active_q16_ = semantic.active_q16;
        semantic.centroid_x_q16 = active_count == 0 ? 32768 : fraction_q16(
            centroid_x_sum, active_count * static_cast<std::uint64_t>(kQuarterWidth - 1));
        semantic.centroid_y_q16 = active_count == 0 ? 32768 : fraction_q16(
            centroid_y_sum, active_count * static_cast<std::uint64_t>(kQuarterHeight - 1));
        for (std::size_t zone = 0; zone < zone_active.size(); ++zone) {
            semantic.zone_q16[zone] = fraction_q16(zone_active[zone], zone_samples[zone]);
        }
        semantic.sampled_pixels = static_cast<std::uint32_t>(sampled);
        semantic.active_pixels = static_cast<std::uint32_t>(active_count);
        semantic.strong_pixels = static_cast<std::uint32_t>(strong_count);
        return semantic;
    }

private:
    std::vector<std::uint8_t> decoded_;
    std::array<std::vector<std::uint8_t>, 2> rgba_;
    std::array<std::vector<std::uint8_t>, 2> denoised_;
    std::vector<std::uint16_t> gradient_;
    std::vector<std::uint32_t> motion_;
    std::vector<std::uint8_t> mask_;
    std::vector<std::uint8_t> quarter_a_;
    std::vector<std::uint8_t> quarter_b_;
    std::vector<std::uint8_t> composite_;
    std::vector<std::uint8_t> output_nv12_;
    std::uint32_t previous_active_q16_ = 0;
    int current_ = 0;
};

class DecodeWorkerPool {
public:
    DecodeWorkerPool(std::array<std::unique_ptr<Nv12Camera>, kCameraCount>& cameras,
                     std::array<CameraPipeline, kCameraCount>& pipelines,
                     const Options& options)
        : cameras_(cameras), pipelines_(pipelines), options_(options) {
        for (int camera = 0; camera < kCameraCount; ++camera) {
            workers_[camera] = std::thread([this, camera] { worker_loop(camera); });
        }
    }

    DecodeWorkerPool(const DecodeWorkerPool&) = delete;
    DecodeWorkerPool& operator=(const DecodeWorkerPool&) = delete;

    ~DecodeWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        completed_ = 0;
        ++generation_;
        const std::uint64_t requested_generation = generation_;
        start_cv_.notify_all();
        done_cv_.wait(lock, [this, requested_generation] {
            return completed_ == kCameraCount || stopping_ || generation_ != requested_generation;
        });
        if (stopping_ || generation_ != requested_generation || completed_ != kCameraCount) {
            throw std::runtime_error("decode worker pool stopped unexpectedly");
        }
    }

private:
    void worker_loop(int camera) {
        configure_worker_thread(options_, static_cast<std::size_t>(camera));
        std::unique_lock<std::mutex> lock(mutex_);
        std::uint64_t consumed_generation = 0;
        while (true) {
            start_cv_.wait(lock, [this, consumed_generation] {
                return stopping_ || generation_ != consumed_generation;
            });
            if (stopping_) {
                return;
            }
            consumed_generation = generation_;
            lock.unlock();
            cameras_[camera]->pull(pipelines_[camera].decoded());
            lock.lock();
            ++completed_;
            if (completed_ == kCameraCount) {
                done_cv_.notify_one();
            }
        }
    }

    std::array<std::unique_ptr<Nv12Camera>, kCameraCount>& cameras_;
    std::array<CameraPipeline, kCameraCount>& pipelines_;
    const Options& options_;
    std::array<std::thread, kCameraCount> workers_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::uint64_t generation_ = 0;
    int completed_ = 0;
    bool stopping_ = false;
};

class PipelineWorkerPool {
public:
    PipelineWorkerPool(std::array<CameraPipeline, kCameraCount>& pipelines,
                       const Options& options)
        : pipelines_(pipelines), options_(options) {
        for (int camera = 0; camera < kCameraCount; ++camera) {
            workers_[camera] = std::thread([this, camera] { worker_loop(camera); });
        }
    }

    PipelineWorkerPool(const PipelineWorkerPool&) = delete;
    PipelineWorkerPool& operator=(const PipelineWorkerPool&) = delete;

    ~PipelineWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::array<PipelineResult, kCameraCount> run(
            const std::array<float, kCameraCount>& detail_gains) {
        std::unique_lock<std::mutex> lock(mutex_);
        gains_ = detail_gains;
        aggregate_mode_ = false;
        completed_ = 0;
        ++generation_;
        const std::uint64_t requested_generation = generation_;
        start_cv_.notify_all();
        done_cv_.wait(lock, [this, requested_generation] {
            return completed_ == kCameraCount || stopping_ || generation_ != requested_generation;
        });
        if (stopping_ || generation_ != requested_generation || completed_ != kCameraCount) {
            throw std::runtime_error("pipeline worker pool stopped unexpectedly");
        }
        return outputs_;
    }

    std::array<four_camera_semantic_camera_v1, kCameraCount> aggregate() {
        std::unique_lock<std::mutex> lock(mutex_);
        completed_ = 0;
        aggregate_mode_ = true;
        ++generation_;
        const std::uint64_t requested_generation = generation_;
        start_cv_.notify_all();
        done_cv_.wait(lock, [this, requested_generation] {
            return completed_ == kCameraCount || stopping_ || generation_ != requested_generation;
        });
        if (stopping_ || generation_ != requested_generation || completed_ != kCameraCount) {
            throw std::runtime_error("pipeline worker pool stopped unexpectedly");
        }
        return semantic_outputs_;
    }

private:
    void worker_loop(int camera) {
        configure_worker_thread(options_, static_cast<std::size_t>(camera + kCameraCount));
        std::unique_lock<std::mutex> lock(mutex_);
        std::uint64_t consumed_generation = 0;
        while (true) {
            start_cv_.wait(lock, [this, consumed_generation] {
                return stopping_ || generation_ != consumed_generation;
            });
            if (stopping_) {
                return;
            }
            consumed_generation = generation_;
            const auto gains = gains_;
            const bool aggregate_mode = aggregate_mode_;
            lock.unlock();
            PipelineResult output{};
            four_camera_semantic_camera_v1 semantic{};
            if (aggregate_mode) {
                semantic = pipelines_[camera].aggregate_semantic();
            } else {
                output = pipelines_[camera].run(gains[camera]);
            }
            lock.lock();
            if (aggregate_mode) {
                semantic_outputs_[camera] = semantic;
            } else {
                outputs_[camera] = output;
            }
            ++completed_;
            if (completed_ == kCameraCount) {
                done_cv_.notify_one();
            }
        }
    }

    std::array<CameraPipeline, kCameraCount>& pipelines_;
    const Options& options_;
    std::array<std::thread, kCameraCount> workers_;
    std::array<float, kCameraCount> gains_{};
    std::array<PipelineResult, kCameraCount> outputs_{};
    std::array<four_camera_semantic_camera_v1, kCameraCount> semantic_outputs_{};
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::uint64_t generation_ = 0;
    int completed_ = 0;
    bool stopping_ = false;
    bool aggregate_mode_ = false;
};

const char* command_name(std::uint32_t command) {
    switch (command) {
        case FOUR_CAMERA_MODE_FINE:
            return "fine";
        case FOUR_CAMERA_MODE_NORMAL:
            return "normal";
        case FOUR_CAMERA_MODE_COARSE:
            return "coarse";
        default:
            return "none";
    }
}

float command_gain(std::uint32_t command) {
    switch (command) {
        case FOUR_CAMERA_MODE_FINE:
            return 1.12F;
        case FOUR_CAMERA_MODE_COARSE:
            return 0.88F;
        default:
            return 1.0F;
    }
}

const char* next_value(int argc, char** argv, int& index, const char* option) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + option);
    }
    return argv[index];
}

struct DeadlineOverride {
    std::uint64_t start_frame = 0;
    std::uint64_t end_frame = 0;
    std::int64_t deadline_us = 0;
};

class DeadlineProfile {
public:
    explicit DeadlineProfile(const std::filesystem::path& path) {
        if (path.empty()) {
            return;
        }
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open deadline profile: " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            std::istringstream parser(line);
            DeadlineOverride value;
            char first_comma = 0;
            char second_comma = 0;
            if (!(parser >> value.start_frame >> first_comma >> value.end_frame >>
                  second_comma >> value.deadline_us) || first_comma != ',' ||
                second_comma != ',' || value.start_frame == 0 ||
                value.end_frame < value.start_frame || value.deadline_us <= 0) {
                throw std::runtime_error("invalid deadline profile row: " + line);
            }
            values_.push_back(value);
        }
    }

    std::int64_t resolve(std::uint64_t frame_id, std::int64_t fallback) const {
        for (const auto& value : values_) {
            if (value.start_frame <= frame_id && frame_id <= value.end_frame) {
                return value.deadline_us;
            }
        }
        return fallback;
    }

private:
    std::vector<DeadlineOverride> values_;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--frames") {
            options.frames = std::stoi(next_value(argc, argv, index, "--frames"));
        } else if (argument == "--warmup-frames") {
            options.warmup_frames = std::stoi(next_value(argc, argv, index, "--warmup-frames"));
        } else if (argument == "--period-us") {
            options.period_us = std::stoll(next_value(argc, argv, index, "--period-us"));
        } else if (argument == "--deadline-us") {
            options.deadline_us = std::stoll(next_value(argc, argv, index, "--deadline-us"));
        } else if (argument == "--video") {
            options.video = next_value(argc, argv, index, "--video");
        } else if (argument == "--output") {
            options.output = next_value(argc, argv, index, "--output");
        } else if (argument == "--semantic-output") {
            options.semantic_output = next_value(argc, argv, index, "--semantic-output");
        } else if (argument == "--task-page") {
            options.task_page = next_value(argc, argv, index, "--task-page");
        } else if (argument == "--progress-page") {
            options.progress_page = next_value(argc, argv, index, "--progress-page");
        } else if (argument == "--semantic-page") {
            options.semantic_page = next_value(argc, argv, index, "--semantic-page");
        } else if (argument == "--result-page") {
            options.result_page = next_value(argc, argv, index, "--result-page");
        } else if (argument == "--deadline-profile") {
            options.deadline_profile = next_value(argc, argv, index, "--deadline-profile");
        } else if (argument == "--condition") {
            options.condition = next_value(argc, argv, index, "--condition");
        } else if (argument == "--cpu-list") {
            options.cpu_list = parse_cpu_list(next_value(argc, argv, index, "--cpu-list"));
        } else if (argument == "--main-cpu") {
            options.main_cpu = std::stoi(next_value(argc, argv, index, "--main-cpu"));
            if (options.main_cpu < 0 || options.main_cpu >= CPU_SETSIZE) {
                throw std::invalid_argument("--main-cpu is outside CPU_SETSIZE");
            }
        } else if (argument == "--worker-cpus") {
            options.worker_cpus = parse_cpu_list(
                next_value(argc, argv, index, "--worker-cpus"));
        } else if (argument == "--realtime-priority") {
            options.realtime_priority = std::stoi(
                next_value(argc, argv, index, "--realtime-priority"));
            if (options.realtime_priority <= 0 || options.realtime_priority > 99) {
                throw std::invalid_argument("--realtime-priority must be in 1..99");
            }
        } else if (argument == "--realtime-fifo") {
            options.realtime_fifo = true;
        } else if (argument == "--worker-realtime-priority") {
            options.worker_realtime_priority = std::stoi(
                next_value(argc, argv, index, "--worker-realtime-priority"));
            if (options.worker_realtime_priority <= 0 ||
                options.worker_realtime_priority > 99) {
                throw std::invalid_argument("--worker-realtime-priority must be in 1..99");
            }
        } else if (argument == "--worker-realtime-fifo") {
            options.worker_realtime_fifo = true;
        } else if (argument == "--help") {
            std::cout << "fourcam --video FILE [--frames N] "
                         "[--warmup-frames N] [--period-us US] [--deadline-us US] "
                         "[--deadline-profile CSV] "
                         "[--cpu-list CPU[,CPU...]] [--main-cpu N] [--worker-cpus CPU[,CPU...]] "
                         "[--realtime-priority N] [--realtime-fifo] "
                         "[--worker-realtime-priority N] [--worker-realtime-fifo] "
                         "[--condition NAME] [--output CSV] [--task-page PAGE] "
                         "[--progress-page PAGE] "
                         "[--semantic-output CSV] [--semantic-page PAGE] [--result-page PAGE]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.frames <= 0 || options.warmup_frames < 0 || options.period_us <= 0 ||
        options.deadline_us <= 0 || options.video.empty()) {
        throw std::invalid_argument("frame counts, timing, and --video must be valid");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        configure_foreground_thread(options);
        // The foreground already has one persistent worker per camera. Prevent
        // OpenCV from creating a second nested pool inside each worker.
        cv::setNumThreads(1);
        static_cast<void>(ABinderProcess_setThreadPoolMaxThreadCount(4));
        ABinderProcess_startThreadPool();
        constexpr std::array<int, kCameraCount> offsets_ms{0, 533, 1067, 1600};
        std::array<std::unique_ptr<Nv12Camera>, kCameraCount> cameras;
        std::array<CameraPipeline, kCameraCount> pipelines;
        for (int camera = 0; camera < kCameraCount; ++camera) {
            cameras[camera] = std::make_unique<Nv12Camera>(
                options.video, offsets_ms[camera], camera);
        }
        DecodeWorkerPool decode_workers(cameras, pipelines, options);
        PipelineWorkerPool pipeline_workers(pipelines, options);

        auto run_epoch = [&](const std::array<float, kCameraCount>& detail_gains) {
            decode_workers.run();
            const auto outputs = pipeline_workers.run(detail_gains);
            std::uint64_t checksum = 0;
            for (const auto& output : outputs) {
                checksum += output.checksum;
            }
            return checksum;
        };

        auto aggregate_epoch = [&] {
            return pipeline_workers.aggregate();
        };

        for (int frame = 0; frame < options.warmup_frames; ++frame) {
            static_cast<void>(run_epoch({1.0F, 1.0F, 1.0F, 1.0F}));
            static_cast<void>(aggregate_epoch());
        }

        const std::uint64_t period_ns = static_cast<std::uint64_t>(options.period_us) * 1000ULL;
        const DeadlineProfile deadline_profile(options.deadline_profile);
        full_sched::task_snapshot_writer task_writer(options.task_page, -1, period_ns, 1);
        // frame stages for the sampler's progress predictor,
        // the light foreground's calls; inactive without --progress-page
        v3_progress::writer progress_writer(options.progress_page.string());
        SemanticWriter semantic_writer(options.semantic_page);
        InLoopResultReader result_reader(options.result_page);
        std::uint64_t next_release = monotonic_ns() + 100000000ULL;
        std::uint64_t last_applied_request_id = 0;
        std::uint64_t total_checksum = 0;
        std::uint64_t misses = 0;
        std::vector<FrameRecord> records;
        records.reserve(options.frames);

        std::cout << "{\"event\":\"active_start\",\"condition\":\"" << options.condition
                  << "\",\"monotonic_ns\":" << next_release << ",\"wall_time_ns\":"
                  << wall_time_ns() << ",\"cameras\":4,\"width\":1920,\"height\":1080"
                  << ",\"logical_touched_bytes_per_epoch\":" << kLogicalBytesPerEpoch
                  << "}\n";
        std::cout.flush();

        for (int frame_index = 0; frame_index < options.frames; ++frame_index) {
            sleep_until(next_release);
            const std::uint64_t start_ns = monotonic_ns();
            const std::uint64_t frame_id = static_cast<std::uint64_t>(frame_index) + 1;
            const std::int64_t frame_deadline_us =
                deadline_profile.resolve(frame_id, options.deadline_us);
            const std::uint64_t frame_deadline_ns =
                static_cast<std::uint64_t>(frame_deadline_us) * 1000ULL;
            task_writer.begin_frame(frame_id, next_release, next_release, start_ns,
                                    next_release + frame_deadline_ns, 0, 0, 0, 0);
            progress_writer.begin(frame_id, kCameraCount, next_release, start_ns,
                next_release + frame_deadline_ns, next_release + period_ns,
                next_release + period_ns + static_cast<std::uint64_t>(
                    deadline_profile.resolve(frame_id + 1, options.deadline_us)) * 1000ULL,
                kCameraCount);

            FrameRecord record;
            record.frame_id = frame_id;
            record.scheduled_release_ns = next_release;
            record.actual_start_ns = start_ns;
            record.deadline_ns = next_release + frame_deadline_ns;
            record.deadline_us = frame_deadline_us;
            record.queue_wait_us = us_between(next_release, start_ns);
            std::array<float, kCameraCount> detail_gains{1.0F, 1.0F, 1.0F, 1.0F};
            four_camera_result_v1 result{};
            if (const auto snapshot = result_reader.read()) {
                result = *snapshot;
                record.result_valid =
                    (result.flags & (FOUR_CAMERA_RESULT_VALID | FOUR_CAMERA_RESULT_PARSED |
                                     FOUR_CAMERA_RESULT_ACCEPTED)) ==
                        (FOUR_CAMERA_RESULT_VALID | FOUR_CAMERA_RESULT_PARSED |
                         FOUR_CAMERA_RESULT_ACCEPTED) &&
                    result.request_id > 0 && result.source_frame_id > 0 &&
                    result.source_release_ns <= result.enqueue_ns &&
                    result.enqueue_ns <= result.complete_ns && result.complete_ns <= start_ns &&
                    result.focus_camera < FOUR_CAMERA_COUNT &&
                    result.mode >= FOUR_CAMERA_MODE_FINE &&
                    result.mode <= FOUR_CAMERA_MODE_COARSE;
                record.result_fresh = record.result_valid &&
                                      result.complete_ns <= result.freshness_deadline_ns &&
                                      start_ns <= result.freshness_deadline_ns;
                if (record.result_fresh) {
                    detail_gains[result.focus_camera] = command_gain(result.mode);
                    if (result.request_id > last_applied_request_id) {
                        last_applied_request_id = result.request_id;
                        record.result_applied = true;
                    }
                }
            }
            record.result_request_id = result.request_id;
            record.result_source_frame_id = result.source_frame_id;
            record.result_complete_ns = result.complete_ns;
            record.focus_camera = result.focus_camera;
            record.result_command = result.mode;
            record.detail_gain = result.focus_camera < FOUR_CAMERA_COUNT
                ? detail_gains[result.focus_camera] : 1.0F;
            if (result.source_release_ns > 0 && start_ns >= result.source_release_ns) {
                record.result_source_age_ms = us_between(result.source_release_ns, start_ns) / 1000.0;
            }

            const std::uint64_t decode_begin_ns = monotonic_ns();
            decode_workers.run();
            const std::uint64_t pipeline_begin_ns = monotonic_ns();
            progress_writer.pipeline_started(pipeline_begin_ns);
            const auto outputs = pipeline_workers.run(detail_gains);
            const std::uint64_t complete_ns = monotonic_ns();
            for (int camera = 0; camera < kCameraCount; ++camera) {
                total_checksum += outputs[camera].checksum;
            }
            record.decode_copy_us = us_between(decode_begin_ns, pipeline_begin_ns);
            record.pipeline_us = us_between(pipeline_begin_ns, complete_ns);
            record.response_us = us_between(next_release, complete_ns);
            record.slack_us = static_cast<double>(frame_deadline_us) - record.response_us;
            record.miss = complete_ns > next_release + frame_deadline_ns;
            record.checksum = total_checksum;
            misses += record.miss ? 1ULL : 0ULL;
            task_writer.complete(frame_id, complete_ns, record.miss, 0, 0, 0, 0,
                                 complete_ns, complete_ns);
            progress_writer.completed(complete_ns);

            const std::uint64_t semantic_begin_ns = monotonic_ns();
            record.semantics = aggregate_epoch();
            const std::uint64_t semantic_complete_ns = monotonic_ns();
            record.semantic_aggregate_us = us_between(
                semantic_begin_ns, semantic_complete_ns);
            semantic_writer.write(
                frame_id, next_release, semantic_complete_ns, record.semantics);
            records.push_back(record);
            next_release += period_ns;
        }

        std::ofstream csv(options.output);
        if (!csv) {
            throw std::runtime_error("cannot open foreground CSV: " + options.output.string());
        }
        csv << "frame_id,scheduled_release_ns,actual_start_ns,deadline_ns,queue_wait_us,decode_copy_us,"
               "pipeline_us,semantic_aggregate_us,response_us,deadline_us,slack_us,miss,checksum,"
               "logical_touched_bytes,result_request_id,result_source_frame_id,"
               "result_complete_ns,result_source_age_ms,focus_camera,result_command,result_valid,"
               "result_fresh,result_applied,detail_gain\n";
        csv << std::fixed << std::setprecision(6);
        for (const auto& record : records) {
            csv << record.frame_id << ',' << record.scheduled_release_ns << ','
                << record.actual_start_ns << ',' << record.deadline_ns << ','
                << record.queue_wait_us << ','
                << record.decode_copy_us << ',' << record.pipeline_us << ','
                << record.semantic_aggregate_us << ',' << record.response_us << ','
                << record.deadline_us << ','
                << record.slack_us << ',' << (record.miss ? 1 : 0) << ','
                << record.checksum << ',' << kLogicalBytesPerEpoch << ','
                << record.result_request_id << ',' << record.result_source_frame_id << ','
                << record.result_complete_ns << ',';
            if (std::isfinite(record.result_source_age_ms)) {
                csv << record.result_source_age_ms;
            }
            csv << ',';
            if (record.focus_camera < FOUR_CAMERA_COUNT) {
                csv << record.focus_camera;
            }
            csv << ',' << command_name(record.result_command) << ','
                << (record.result_valid ? 1 : 0) << ','
                << (record.result_fresh ? 1 : 0) << ','
                << (record.result_applied ? 1 : 0) << ',' << record.detail_gain << '\n';
        }
        csv.flush();
        std::ofstream semantic_csv(options.semantic_output);
        if (!semantic_csv) {
            throw std::runtime_error(
                "cannot open semantic CSV: " + options.semantic_output.string());
        }
        semantic_csv << "frame_id,source_release_ns,camera,active,strong,motion,"
                        "activity_trend,centroid_x,centroid_y,zone_tl,zone_tr,zone_bl,"
                        "zone_br,sampled_pixels,active_pixels,strong_pixels\n";
        semantic_csv << std::fixed << std::setprecision(6);
        for (const auto& record : records) {
            for (int camera = 0; camera < kCameraCount; ++camera) {
                const auto& value = record.semantics[camera];
                semantic_csv << record.frame_id << ',' << record.scheduled_release_ns << ','
                             << camera << ','
                             << static_cast<double>(value.active_q16) / 65535.0 << ','
                             << static_cast<double>(value.strong_q16) / 65535.0 << ','
                             << static_cast<double>(value.motion_q16) / 65535.0 << ','
                             << static_cast<double>(value.activity_trend_q16) / 65535.0 << ','
                             << static_cast<double>(value.centroid_x_q16) / 65535.0 << ','
                             << static_cast<double>(value.centroid_y_q16) / 65535.0;
                for (std::uint32_t zone : value.zone_q16) {
                    semantic_csv << ',' << static_cast<double>(zone) / 65535.0;
                }
                semantic_csv << ',' << value.sampled_pixels << ',' << value.active_pixels
                             << ',' << value.strong_pixels << '\n';
            }
        }
        semantic_csv.flush();
        std::cout << "{\"event\":\"active_end\",\"condition\":\"" << options.condition
                  << "\",\"monotonic_ns\":" << monotonic_ns() << ",\"wall_time_ns\":"
                  << wall_time_ns() << ",\"frames\":" << options.frames
                  << ",\"misses\":" << misses << ",\"checksum\":" << total_checksum
                  << ",\"logical_touched_bytes_per_epoch\":" << kLogicalBytesPerEpoch
                  << "}\n";
        std::cout.flush();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fourcam: " << error.what() << '\n';
        return 2;
    }
}
