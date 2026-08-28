#include "ruvector_apple_core.h"

#include <stdint.h>
#include <stdio.h>

static int require_status(RvStatus status, const char *operation) {
  if (status == RV_STATUS_OK) {
    return 1;
  }
  fprintf(stderr, "%s failed with status %d\n", operation, status);
  return 0;
}

int main(void) {
  RvAppleIndex *index = NULL;
  RvAppleIndex *restored = NULL;
  RvOwnedResults *results = NULL;
  RvOwnedBytes *snapshot = NULL;
  const RvSearchResult *result_values = NULL;
  const uint8_t *snapshot_values = NULL;
  uint32_t result_count = 0;
  uint64_t snapshot_count = 0;
  int success = 0;

  const float first[] = {1.0f, 0.0f};
  const float second[] = {0.0f, 1.0f};
  const float query[] = {1.0f, 0.0f};

  if (ruvector_apple_core_abi_version() != RVECTOR_APPLE_CORE_ABI_VERSION) {
    fprintf(stderr, "ABI version mismatch\n");
    goto cleanup;
  }
  if (!require_status(ruvector_apple_core_index_create(
                          2, 2, RV_DISTANCE_COSINE, &index),
                      "index_create")) {
    goto cleanup;
  }
  if (!require_status(
          ruvector_apple_core_index_upsert(index, 41, first, 2),
          "first upsert") ||
      !require_status(
          ruvector_apple_core_index_upsert(index, 42, second, 2),
          "second upsert")) {
    goto cleanup;
  }
  if (!require_status(
          ruvector_apple_core_index_search(index, query, 2, 2, &results),
          "index_search") ||
      !require_status(ruvector_apple_core_results_data(
                          results, &result_values, &result_count),
                      "results_data")) {
    goto cleanup;
  }
  if (result_count != 2 || result_values == NULL ||
      result_values[0].id != 41) {
    fprintf(stderr, "unexpected search ordering\n");
    goto cleanup;
  }
  if (!require_status(
          ruvector_apple_core_index_snapshot(index, &snapshot),
          "index_snapshot") ||
      !require_status(ruvector_apple_core_bytes_data(
                          snapshot, &snapshot_values, &snapshot_count),
                      "bytes_data") ||
      snapshot_count == 0 || snapshot_values == NULL) {
    goto cleanup;
  }
  if (!require_status(ruvector_apple_core_index_restore(
                          snapshot_values, snapshot_count, &restored),
                      "index_restore")) {
    goto cleanup;
  }

  RvIndexInfo info = {0};
  if (!require_status(ruvector_apple_core_index_info(restored, &info),
                      "index_info") ||
      info.abi_version != RVECTOR_APPLE_CORE_ABI_VERSION ||
      info.dimensions != 2 || info.count != 2) {
    fprintf(stderr, "restored index metadata mismatch\n");
    goto cleanup;
  }
  success = 1;

cleanup:
  if (snapshot != NULL &&
      !require_status(ruvector_apple_core_bytes_free(snapshot),
                      "bytes_free")) {
    success = 0;
  }
  if (results != NULL &&
      !require_status(ruvector_apple_core_results_free(results),
                      "results_free")) {
    success = 0;
  }
  if (restored != NULL &&
      !require_status(ruvector_apple_core_index_destroy(restored),
                      "restored index_destroy")) {
    success = 0;
  }
  if (index != NULL &&
      !require_status(ruvector_apple_core_index_destroy(index),
                      "index_destroy")) {
    success = 0;
  }
  return success ? 0 : 1;
}
