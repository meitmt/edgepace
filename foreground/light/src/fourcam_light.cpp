#include "fourcam_abi.h"
#include "foreground_progress.h"
#include "full_sched/task_writer.h"

// decoding is the only platform-specific part.  Jetson: GStreamer with
// NVDEC (nvv4l2decoder); Android: NDK MediaCodec byte-buffer output (as the
// 2026-09 phone foreground).  Both deliver 1920x1080 NV12 to the same pipeline.
#ifdef __ANDROID__
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#else
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#endif
#include <opencv2/imgproc.hpp>

#include <arm_neon.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int kCameraCount = 4;
constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
constexpr int kQuarterWidth = kWidth / 4;
constexpr int kQuarterHeight = kHeight / 4;
constexpr std::uint64_t kBaseLogicalBytesPerEpoch =
    static_cast<std::uint64_t>(kCameraCount) * kPixels * 139ULL / 2ULL;
constexpr std::uint64_t kSemanticExtraBytesPerEpoch =
    static_cast<std::uint64_t>(kCameraCount) * (kPixels / 64) *
    (sizeof(std::uint32_t) + sizeof(std::uint8_t));
constexpr std::uint64_t kLogicalBytesPerEpoch =
    kBaseLogicalBytesPerEpoch + kSemanticExtraBytesPerEpoch;
constexpr std::uint64_t kLogicalBytesPerCamera =
    kPixels * 139ULL / 2ULL + (kPixels / 64) *
    (sizeof(std::uint32_t) + sizeof(std::uint8_t));

std::uint64_t logical_bytes_for_cameras(int active_cameras) {
    return static_cast<std::uint64_t>(active_cameras) * kLogicalBytesPerCamera;
}

struct Options {
    int frames = 180;
    int warmup_frames = 12;
    std::int64_t period_us = 33333;
    std::int64_t deadline_us = 33333;
    std::filesystem::path video;
    std::filesystem::path output = "foreground.csv";
    std::filesystem::path semantic_output = "semantic_records.csv";
    std::filesystem::path task_page;
    std::filesystem::path progress_page;
    std::filesystem::path semantic_page;
    std::filesystem::path result_page;
    std::filesystem::path deadline_profile;
    std::string condition = "baseline";
    int active_cameras = kCameraCount;
    int partial_camera_period = 0;
};

std::uint64_t average_logical_bytes(const Options& options) {
    const std::uint64_t base = logical_bytes_for_cameras(options.active_cameras);
    if (options.partial_camera_period <= 0 || options.active_cameras >= kCameraCount) {
        return base;
    }
    return base + (kLogicalBytesPerCamera *
                   static_cast<std::uint64_t>(options.partial_camera_period - 1)) /
                      static_cast<std::uint64_t>(options.partial_camera_period);
}

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
    int processed_cameras = kCameraCount;
};

// Keep four decoder streams for the camera contract, but reduce average CPU
// pipeline work by omitting the fourth processing pipeline once per cycle.
int processed_camera_count(const Options& options, int frame_index) {
    if (options.partial_camera_period <= 0 || options.active_cameras >= kCameraCount) {
        return options.active_cameras;
    }
    const int phase = frame_index % options.partial_camera_period;
    return options.active_cameras + (phase < options.partial_camera_period - 1 ? 1 : 0);
}

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

#ifndef __ANDROID__
std::string shell_quote(const std::filesystem::path& path) {
    std::string quoted = "\"";
    for (char character : path.string()) {
        if (character == '\\' || character == '\"') {
            quoted.push_back('\\');
        }
        quoted.push_back(character);
    }
    quoted.push_back('\"');
    return quoted;
}
#endif

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

#ifdef __ANDROID__
}  // namespace
// Codec2 delivers decoded buffers through binder callbacks: without a binder
// thread pool in the process, dequeueOutputBuffer never sees output.
extern "C" bool ABinderProcess_setThreadPoolMaxThreadCount(std::uint32_t num_threads);
extern "C" void ABinderProcess_startThreadPool();
namespace {

constexpr int32_t kColorFormatYuv420Flexible = 0x7F420888;
constexpr int32_t kColorFormatYuv420Vendor = 0x7FA30C04;

// From llm_ros/oneplus/runtime/android_foreground (2026-09 phone foreground),
// unchanged: synchronous byte-buffer decode, frames at or after the camera offset.
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
#else
class Nv12Camera {
public:
    Nv12Camera(const std::filesystem::path& video, int offset_ms, int camera_id) {
        const std::string pipeline_text =
            "filesrc location=" + shell_quote(video) +
            " ! qtdemux ! h265parse ! nvv4l2decoder enable-max-performance=true" 
            " ! nvvidconv ! video/x-raw,format=NV12,width=1920,height=1080" 
            " ! appsink name=sink" + std::to_string(camera_id) +
            " sync=false max-buffers=2 drop=false";
        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_text.c_str(), &error);
        if (pipeline_ == nullptr) {
            const std::string message = error != nullptr ? error->message : "unknown error";
            if (error != nullptr) {
                g_error_free(error);
            }
            throw std::runtime_error("GStreamer pipeline creation failed: " + message);
        }
        const std::string sink_name = "sink" + std::to_string(camera_id);
        sink_ = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline_), sink_name.c_str()));
        if (sink_ == nullptr) {
            throw std::runtime_error("GStreamer appsink lookup failed");
        }
        gst_element_set_state(pipeline_, GST_STATE_PAUSED);
        gst_element_get_state(pipeline_, nullptr, nullptr, 10 * GST_SECOND);
        if (offset_ms > 0 && !gst_element_seek_simple(
                pipeline_, GST_FORMAT_TIME,
                static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE),
                static_cast<gint64>(offset_ms) * GST_MSECOND)) {
            throw std::runtime_error("GStreamer camera offset seek failed");
        }
        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            throw std::runtime_error("GStreamer camera failed to enter PLAYING state");
        }
    }

    Nv12Camera(const Nv12Camera&) = delete;
    Nv12Camera& operator=(const Nv12Camera&) = delete;

    ~Nv12Camera() {
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }
        if (sink_ != nullptr) {
            gst_object_unref(sink_);
        }
        if (pipeline_ != nullptr) {
            gst_object_unref(pipeline_);
        }
    }

    void pull(std::vector<std::uint8_t>& destination) {
        GstSample* sample = gst_app_sink_pull_sample(sink_);
        if (sample == nullptr) {
            throw std::runtime_error("camera stream ended before the requested foreground frames");
        }
        GstCaps* caps = gst_sample_get_caps(sample);
        GstVideoInfo info{};
        GstVideoFrame frame{};
        if (caps == nullptr || !gst_video_info_from_caps(&info, caps) ||
            GST_VIDEO_INFO_WIDTH(&info) != kWidth || GST_VIDEO_INFO_HEIGHT(&info) != kHeight ||
            GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12 ||
            !gst_video_frame_map(&frame, &info, gst_sample_get_buffer(sample), GST_MAP_READ)) {
            gst_sample_unref(sample);
            throw std::runtime_error("camera frame is not mapped 1920x1080 NV12");
        }
        const auto* y = static_cast<const std::uint8_t*>(
            GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
        const auto* uv = static_cast<const std::uint8_t*>(
            GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
        const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
        const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);
        for (int row = 0; row < kHeight; ++row) {
            std::memcpy(destination.data() + static_cast<std::size_t>(row) * kWidth,
                        y + static_cast<std::size_t>(row) * y_stride, kWidth);
        }
        std::uint8_t* uv_destination = destination.data() + kPixels;
        for (int row = 0; row < kHeight / 2; ++row) {
            std::memcpy(uv_destination + static_cast<std::size_t>(row) * kWidth,
                        uv + static_cast<std::size_t>(row) * uv_stride, kWidth);
        }
        gst_video_frame_unmap(&frame);
        gst_sample_unref(sample);
    }

private:
    GstElement* pipeline_ = nullptr;
    GstAppSink* sink_ = nullptr;
};
#endif

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
          enhanced_(kPixels * 4), composite_(kPixels * 4), output_nv12_(kPixels * 3 / 2) {}

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

        for (std::size_t index = 0; index < kPixels * 4; index += 16) {
            const uint8x16_t current_value = vld1q_u8(rgba.data() + index);
            const uint8x16_t previous_value = vld1q_u8(previous_rgba.data() + index);
            const uint8x16_t temporal = vrhaddq_u8(
                current_value, vrhaddq_u8(current_value, previous_value));
            vst1q_u8(denoised.data() + index, temporal);
        }

        for (std::size_t pixel = 0; pixel < kPixels; pixel += 16) {
            const uint8x16x4_t values = vld4q_u8(denoised.data() + pixel * 4);
            const uint8x16_t rg = vabdq_u8(values.val[0], values.val[1]);
            const uint8x16_t gb = vabdq_u8(values.val[1], values.val[2]);
            vst1q_u16(gradient_.data() + pixel,
                      vaddl_u8(vget_low_u8(rg), vget_low_u8(gb)));
            vst1q_u16(gradient_.data() + pixel + 8,
                      vaddl_u8(vget_high_u8(rg), vget_high_u8(gb)));
        }

        for (std::size_t pixel = 0; pixel < kPixels; pixel += 16) {
            const uint8x16x4_t current_value =
                vld4q_u8(denoised.data() + pixel * 4);
            const uint8x16x4_t previous_value =
                vld4q_u8(previous_denoised.data() + pixel * 4);
            uint16x8_t low = vaddl_u8(
                vget_low_u8(vabdq_u8(current_value.val[0], previous_value.val[0])),
                vget_low_u8(vabdq_u8(current_value.val[1], previous_value.val[1])));
            low = vaddw_u8(low,
                vget_low_u8(vabdq_u8(current_value.val[2], previous_value.val[2])));
            low = vaddw_u8(low,
                vget_low_u8(vabdq_u8(current_value.val[3], previous_value.val[3])));
            uint16x8_t high = vaddl_u8(
                vget_high_u8(vabdq_u8(current_value.val[0], previous_value.val[0])),
                vget_high_u8(vabdq_u8(current_value.val[1], previous_value.val[1])));
            high = vaddw_u8(high,
                vget_high_u8(vabdq_u8(current_value.val[2], previous_value.val[2])));
            high = vaddw_u8(high,
                vget_high_u8(vabdq_u8(current_value.val[3], previous_value.val[3])));
            vst1q_u32(motion_.data() + pixel, vmovl_u16(vget_low_u16(low)));
            vst1q_u32(motion_.data() + pixel + 4, vmovl_u16(vget_high_u16(low)));
            vst1q_u32(motion_.data() + pixel + 8, vmovl_u16(vget_low_u16(high)));
            vst1q_u32(motion_.data() + pixel + 12, vmovl_u16(vget_high_u16(high)));
        }

        for (std::size_t pixel = 0; pixel < kPixels; pixel += 16) {
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
            vst4q_u8(enhanced_.data() + pixel * 4, values);
        }

        for (std::size_t pixel = 0; pixel < kPixels; pixel += 16) {
            uint8x16x4_t values = vld4q_u8(enhanced_.data() + pixel * 4);
            const uint8x16_t strong = vceqq_u8(vld1q_u8(mask_.data() + pixel),
                                               vdupq_n_u8(255));
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
    std::vector<std::uint8_t> enhanced_;
    std::vector<std::uint8_t> composite_;
    std::vector<std::uint8_t> output_nv12_;
    std::uint32_t previous_active_q16_ = 0;
    int current_ = 0;
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
        } else if (argument == "--active-cameras") {
            options.active_cameras = std::stoi(
                next_value(argc, argv, index, "--active-cameras"));
        } else if (argument == "--partial-camera-period") {
            options.partial_camera_period = std::stoi(
                next_value(argc, argv, index, "--partial-camera-period"));
        } else if (argument == "--help") {
            std::cout << "fourcam --video FILE [--frames N] "
                         "[--warmup-frames N] [--period-us US] [--deadline-us US] "
                         "[--deadline-profile CSV] "
                         "[--condition NAME] [--output CSV] [--task-page PAGE] "
                         "[--progress-page PAGE] "
                         "[--semantic-output CSV] [--semantic-page PAGE] [--result-page PAGE] "
                         "[--active-cameras N] [--partial-camera-period N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.frames <= 0 || options.warmup_frames < 0 || options.period_us <= 0 ||
        options.deadline_us <= 0 || options.video.empty() ||
        options.active_cameras <= 0 || options.active_cameras > kCameraCount ||
        options.partial_camera_period < 0 ||
        (options.partial_camera_period > 0 && options.active_cameras >= kCameraCount)) {
        throw std::invalid_argument("frame counts, timing, and --video must be valid");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
#ifdef __ANDROID__
        static_cast<void>(ABinderProcess_setThreadPoolMaxThreadCount(4));
        ABinderProcess_startThreadPool();
#else
        gst_init(nullptr, nullptr);
#endif
        constexpr std::array<int, kCameraCount> offsets_ms{0, 533, 1067, 1600};
        std::array<std::unique_ptr<Nv12Camera>, kCameraCount> cameras;
        std::array<CameraPipeline, kCameraCount> pipelines;
        for (int camera = 0; camera < kCameraCount; ++camera) {
            cameras[camera] = std::make_unique<Nv12Camera>(
                options.video, offsets_ms[camera], camera);
        }

        auto run_epoch = [&](const std::array<float, kCameraCount>& detail_gains) {
            for (int camera = 0; camera < kCameraCount; ++camera) {
                cameras[camera]->pull(pipelines[camera].decoded());
            }
            std::array<PipelineResult, kCameraCount> outputs{};
            std::array<std::thread, kCameraCount> workers;
            for (int camera = 0; camera < options.active_cameras; ++camera) {
                workers[camera] = std::thread([&, camera] {
                    outputs[camera] = pipelines[camera].run(detail_gains[camera]);
                });
            }
            for (int camera = 0; camera < options.active_cameras; ++camera) {
                workers[camera].join();
            }
            std::uint64_t checksum = 0;
            for (int camera = 0; camera < options.active_cameras; ++camera) {
                checksum += outputs[camera].checksum;
            }
            return checksum;
        };

        auto aggregate_epoch = [&](int process_count) {
            std::array<four_camera_semantic_camera_v1, kCameraCount> semantics{};
            std::array<std::thread, kCameraCount> workers;
            for (int camera = 0; camera < process_count; ++camera) {
                workers[camera] = std::thread([&, camera] {
                    semantics[camera] = pipelines[camera].aggregate_semantic();
                });
            }
            for (int camera = 0; camera < process_count; ++camera) {
                workers[camera].join();
            }
            return semantics;
        };

        for (int frame = 0; frame < options.warmup_frames; ++frame) {
            static_cast<void>(run_epoch({1.0F, 1.0F, 1.0F, 1.0F}));
            static_cast<void>(aggregate_epoch(options.active_cameras));
        }

        const std::uint64_t period_ns = static_cast<std::uint64_t>(options.period_us) * 1000ULL;
        const DeadlineProfile deadline_profile(options.deadline_profile);
        full_sched::task_snapshot_writer task_writer(options.task_page, -1, period_ns, 1);
        // frame stages for the sampler's progress
        // predictor, as in gplus_load/foreground.cpp; inactive without --progress-page
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
                  << wall_time_ns() << ",\"cameras\":4,\"active_cameras\":"
                  << options.active_cameras << ",\"partial_camera_period\":"
                  << options.partial_camera_period << ",\"width\":1920,\"height\":1080"
                  << ",\"logical_touched_bytes_per_epoch\":"
                  << average_logical_bytes(options)
                  << "}\n";
        std::cout.flush();

        for (int frame_index = 0; frame_index < options.frames; ++frame_index) {
            sleep_until(next_release);
            const std::uint64_t start_ns = monotonic_ns();
            const std::uint64_t frame_id = static_cast<std::uint64_t>(frame_index) + 1;
            const int process_count = processed_camera_count(options, frame_index);
            const std::int64_t frame_deadline_us =
                deadline_profile.resolve(frame_id, options.deadline_us);
            const std::uint64_t frame_deadline_ns =
                static_cast<std::uint64_t>(frame_deadline_us) * 1000ULL;
            task_writer.begin_frame(frame_id, next_release, next_release, start_ns,
                                    next_release + frame_deadline_ns, 0, 0, 0, 0);
            progress_writer.begin(frame_id, process_count, next_release, start_ns,
                next_release + frame_deadline_ns, next_release + period_ns,
                next_release + period_ns + static_cast<std::uint64_t>(
                    deadline_profile.resolve(frame_id + 1, options.deadline_us)) * 1000ULL,
                processed_camera_count(options, frame_index + 1));

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
            for (int camera = 0; camera < kCameraCount; ++camera) {
                cameras[camera]->pull(pipelines[camera].decoded());
            }
            const std::uint64_t pipeline_begin_ns = monotonic_ns();
            progress_writer.pipeline_started(pipeline_begin_ns);
            std::array<PipelineResult, kCameraCount> outputs{};
            std::array<std::thread, kCameraCount> workers;
            for (int camera = 0; camera < process_count; ++camera) {
                workers[camera] = std::thread([&, camera] {
                    outputs[camera] = pipelines[camera].run(detail_gains[camera]);
                });
            }
            for (int camera = 0; camera < process_count; ++camera) {
                workers[camera].join();
            }
            const std::uint64_t complete_ns = monotonic_ns();
            for (int camera = 0; camera < process_count; ++camera) {
                total_checksum += outputs[camera].checksum;
            }
            record.decode_copy_us = us_between(decode_begin_ns, pipeline_begin_ns);
            record.pipeline_us = us_between(pipeline_begin_ns, complete_ns);
            record.response_us = us_between(next_release, complete_ns);
            record.slack_us = static_cast<double>(frame_deadline_us) - record.response_us;
            record.miss = complete_ns > next_release + frame_deadline_ns;
            record.checksum = total_checksum;
            record.processed_cameras = process_count;
            misses += record.miss ? 1ULL : 0ULL;
            task_writer.complete(frame_id, complete_ns, record.miss, 0, 0, 0, 0,
                                 complete_ns, complete_ns);
            progress_writer.completed(complete_ns);

            const std::uint64_t semantic_begin_ns = monotonic_ns();
            record.semantics = aggregate_epoch(process_count);
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
                << record.checksum << ','
                << logical_bytes_for_cameras(record.processed_cameras) << ','
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
                  << ",\"active_cameras\":" << options.active_cameras
                  << ",\"partial_camera_period\":" << options.partial_camera_period
                  << ",\"logical_touched_bytes_per_epoch\":"
                  << average_logical_bytes(options)
                  << "}\n";
        std::cout.flush();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fourcam: " << error.what() << '\n';
        return 2;
    }
}
