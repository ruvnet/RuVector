#ifndef RVECTOR_APPLE_CORE_H
#define RVECTOR_APPLE_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RVECTOR_APPLE_CORE_ABI_VERSION 1u

typedef struct RvAppleIndex RvAppleIndex;
typedef struct RvOwnedResults RvOwnedResults;
typedef struct RvOwnedBytes RvOwnedBytes;

typedef int32_t RvStatus;
#define RV_STATUS_OK ((RvStatus)0)
#define RV_STATUS_NULL_POINTER ((RvStatus)1)
#define RV_STATUS_INVALID_ARGUMENT ((RvStatus)2)
#define RV_STATUS_INVALID_METRIC ((RvStatus)3)
#define RV_STATUS_DIMENSION_MISMATCH ((RvStatus)4)
#define RV_STATUS_NON_FINITE_VALUE ((RvStatus)5)
#define RV_STATUS_ZERO_NORM_VECTOR ((RvStatus)6)
#define RV_STATUS_CAPACITY_EXCEEDED ((RvStatus)7)
#define RV_STATUS_CORRUPT_SNAPSHOT ((RvStatus)8)
#define RV_STATUS_RESOURCE_LIMIT ((RvStatus)9)
#define RV_STATUS_LOCK_POISONED ((RvStatus)10)
#define RV_STATUS_PANIC ((RvStatus)255)

typedef uint32_t RvDistanceMetric;
#define RV_DISTANCE_COSINE ((RvDistanceMetric)1u)
#define RV_DISTANCE_L2 ((RvDistanceMetric)2u)
#define RV_DISTANCE_DOT ((RvDistanceMetric)3u)

typedef struct RvSearchResult {
  uint64_t id;
  double score;
} RvSearchResult;

typedef struct RvIndexInfo {
  uint32_t abi_version;
  uint32_t dimensions;
  uint32_t capacity;
  uint32_t count;
  uint32_t metric;
} RvIndexInfo;

uint32_t ruvector_apple_core_abi_version(void);
RvStatus ruvector_apple_core_index_create(uint32_t dimensions,
                                          uint32_t capacity,
                                          RvDistanceMetric metric,
                                          RvAppleIndex **out_index);
RvStatus ruvector_apple_core_index_destroy(RvAppleIndex *index);
RvStatus ruvector_apple_core_index_upsert(const RvAppleIndex *index,
                                          uint64_t id,
                                          const float *values,
                                          uint32_t values_len);
RvStatus ruvector_apple_core_index_remove(const RvAppleIndex *index,
                                          uint64_t id,
                                          uint8_t *out_removed);
RvStatus ruvector_apple_core_index_info(const RvAppleIndex *index,
                                        RvIndexInfo *out_info);
RvStatus ruvector_apple_core_index_search(const RvAppleIndex *index,
                                          const float *query,
                                          uint32_t query_len,
                                          uint32_t limit,
                                          RvOwnedResults **out_results);
RvStatus ruvector_apple_core_results_data(const RvOwnedResults *results,
                                          const RvSearchResult **out_values,
                                          uint32_t *out_len);
RvStatus ruvector_apple_core_results_free(RvOwnedResults *results);
RvStatus ruvector_apple_core_index_snapshot(const RvAppleIndex *index,
                                            RvOwnedBytes **out_bytes);
RvStatus ruvector_apple_core_bytes_data(const RvOwnedBytes *bytes,
                                        const uint8_t **out_values,
                                        uint64_t *out_len);
RvStatus ruvector_apple_core_bytes_free(RvOwnedBytes *bytes);
RvStatus ruvector_apple_core_index_restore(const uint8_t *bytes,
                                           uint64_t bytes_len,
                                           RvAppleIndex **out_index);

/*
 * Handle contract: every non-null opaque pointer passed to this API must be a
 * live handle returned by this library, used only with its matching functions,
 * and freed exactly once after concurrent calls using it have completed.
 * Forged, stale, wrong-kind, or concurrently destroyed handles are invalid.
 * Returned data pointers are immutable borrows owned by their opaque handle.
 */

#ifdef __cplusplus
}
#endif

#endif
