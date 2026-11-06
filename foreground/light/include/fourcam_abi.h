#pragma once

/* Binary-compatible G1 fork. The frozen four-camera v1 header remains untouched. */

#include <stddef.h>
#include <stdint.h>

#define FOUR_CAMERA_SEMANTIC_MAGIC 0x46435331u /* FCS1 */
#define FOUR_CAMERA_RESULT_MAGIC 0x46435231u   /* FCR1 */
#define FOUR_CAMERA_SEMANTIC_VERSION 1u
#define FOUR_CAMERA_COUNT 4u

#define FOUR_CAMERA_SEMANTIC_VALID (1u << 0)
#define FOUR_CAMERA_RESULT_VALID (1u << 0)
#define FOUR_CAMERA_RESULT_PARSED (1u << 1)
#define FOUR_CAMERA_RESULT_ACCEPTED (1u << 2)

#define FOUR_CAMERA_MODE_NONE 0u
#define FOUR_CAMERA_MODE_FINE 1u
#define FOUR_CAMERA_MODE_NORMAL 2u
#define FOUR_CAMERA_MODE_COARSE 3u

typedef struct four_camera_semantic_camera_v1 {
    uint32_t active_q16;
    uint32_t strong_q16;
    uint32_t motion_q16;
    int32_t activity_trend_q16;
    uint32_t centroid_x_q16;
    uint32_t centroid_y_q16;
    uint32_t zone_q16[4];
    uint32_t sampled_pixels;
    uint32_t active_pixels;
    uint32_t strong_pixels;
    uint32_t reserved0;
    uint32_t reserved1[2];
} four_camera_semantic_camera_v1;

typedef struct four_camera_semantic_snapshot_v1 {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint64_t sequence;
    uint64_t update_ns;
    uint64_t frame_id;
    uint64_t source_release_ns;
    uint64_t semantic_complete_ns;
    uint32_t camera_count;
    uint32_t flags;
    four_camera_semantic_camera_v1 cameras[FOUR_CAMERA_COUNT];
    uint8_t reserved[200];
} four_camera_semantic_snapshot_v1;

typedef struct four_camera_result_v1 {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint64_t sequence;
    uint64_t update_ns;
    uint64_t request_id;
    uint64_t source_frame_id;
    uint64_t source_release_ns;
    uint64_t enqueue_ns;
    uint64_t complete_ns;
    uint64_t freshness_deadline_ns;
    uint32_t focus_camera;
    uint32_t mode;
    uint32_t flags;
    uint32_t ttl_ms;
    float confidence;
    uint32_t reserved0;
    uint64_t reserved[4];
} four_camera_result_v1;

#ifdef __cplusplus
static_assert(sizeof(four_camera_semantic_camera_v1) == 64,
              "four-camera semantic camera ABI size");
static_assert(sizeof(four_camera_semantic_snapshot_v1) == 512,
              "four-camera semantic page ABI size");
static_assert(sizeof(four_camera_result_v1) == 128,
              "four-camera result ABI size");
#else
_Static_assert(sizeof(four_camera_semantic_camera_v1) == 64,
               "four-camera semantic camera ABI size");
_Static_assert(sizeof(four_camera_semantic_snapshot_v1) == 512,
               "four-camera semantic page ABI size");
_Static_assert(sizeof(four_camera_result_v1) == 128,
               "four-camera result ABI size");
#endif
