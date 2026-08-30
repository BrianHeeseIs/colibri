#ifndef COLI_V4_DECODE_TRACE_H
#define COLI_V4_DECODE_TRACE_H

#include <stdint.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint64_t elapsed_ns;
    uint64_t calls;
} ColiV4DecodeTraceStage;

typedef enum {
    COLI_V4_DT_WAIT_START_LOCK,
    COLI_V4_DT_WAIT_START_SCAN,
    COLI_V4_DT_WAIT_START_IDLE_BLOCK,
    COLI_V4_DT_WAIT_START_PUBLISH,
    COLI_V4_DT_WAIT_FINISH_LOCK,
    COLI_V4_DT_WAIT_FINISH_COMPLETE_BLOCK,
    COLI_V4_DT_WAIT_FINISH_RELEASE,
    COLI_V4_DT_START_CALLS,
    COLI_V4_DT_START_SLEPT_CALLS,
    COLI_V4_DT_FINISH_CALLS,
    COLI_V4_DT_FINISH_COMPLETED_AT_ENTRY,
    COLI_V4_DT_FINISH_SLEPT_CALLS,
    COLI_V4_DT_FINISH_WAKE_ITERATIONS,
    COLI_V4_DT_STORE_LOCK,
    COLI_V4_DT_STORE_HIT_SCAN,
    COLI_V4_DT_STORE_MISS_SELECT,
    COLI_V4_DT_STORE_SLAB_ALLOC,
    COLI_V4_DT_STORE_DISK_READ,
    COLI_V4_DT_STORE_PUBLISH,
    COLI_V4_DT_STORE_PACK,
    COLI_V4_DT_OMP_HC_PRE_WALL,
    COLI_V4_DT_OMP_HEAD_WALL,
    COLI_V4_DT_OMP_SPARSE_WALL,
    COLI_V4_DT_TENSOR_LOOKUP,
    COLI_V4_DT_DECODE_ALLOC,
    COLI_V4_DT_IO_CROSSCHECK,
    COLI_V4_DT_COUNT
} ColiV4DecodeTraceStageId;

typedef enum {
    COLI_V4_DT_NO_SLEEP,
    COLI_V4_DT_SLEPT,
    COLI_V4_DT_FALSE_WAKE
} ColiV4DecodeTraceWaitClass;

static ColiV4DecodeTraceStage coli_v4_decode_trace_stages[COLI_V4_DT_COUNT];

static const char *coli_v4_decode_trace_names[] = {
    "wait_start_lock",
    "wait_start_scan",
    "wait_start_idle_block",
    "wait_start_publish",
    "wait_finish_lock",
    "wait_finish_complete_block",
    "wait_finish_release",
    "start_calls",
    "start_slept_calls",
    "finish_calls",
    "finish_completed_at_entry",
    "finish_slept_calls",
    "finish_wake_iterations",
    "store_lock",
    "store_hit_scan",
    "store_miss_select",
    "store_slab_alloc",
    "store_disk_read",
    "store_publish",
    "store_pack",
    "omp_hc_pre_wall",
    "omp_head_wall",
    "omp_sparse_wall",
    "tensor_lookup",
    "decode_alloc",
    "io_crosscheck",
};

_Static_assert(sizeof(coli_v4_decode_trace_names) /
                   sizeof(coli_v4_decode_trace_names[0]) == COLI_V4_DT_COUNT,
               "decode trace names must match stage count");

static inline uint64_t coli_v4_decode_trace_now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

static inline void coli_v4_decode_trace_add(int stage, uint64_t ns) {
    if (stage < 0 || stage >= COLI_V4_DT_COUNT) return;
    __atomic_fetch_add(&coli_v4_decode_trace_stages[stage].elapsed_ns,
                       ns, __ATOMIC_RELAXED);
    __atomic_fetch_add(&coli_v4_decode_trace_stages[stage].calls,
                       UINT64_C(1), __ATOMIC_RELAXED);
}

static inline void coli_v4_decode_trace_reset(void) {
    memset(coli_v4_decode_trace_stages, 0, sizeof(coli_v4_decode_trace_stages));
}

static inline double coli_v4_decode_trace_pct_parent(uint64_t child_ns,
                                                      uint64_t parent_ns) {
    return parent_ns ? 100.0 * (double)child_ns / (double)parent_ns : 0.0;
}

static inline ColiV4DecodeTraceWaitClass
coli_v4_decode_trace_classify_wait(int completed_at_entry, int wake_iterations) {
    if (completed_at_entry) return COLI_V4_DT_NO_SLEEP;
    return wake_iterations == 1 ? COLI_V4_DT_SLEPT : COLI_V4_DT_FALSE_WAKE;
}

#endif /* COLI_V4_DECODE_TRACE_H */
