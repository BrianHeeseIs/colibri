#if defined(COLI_METAL)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend_metal_v4.h"
#include "build/metal-v4/deepseek_v4_source.h"

#include <stdatomic.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if defined(__cplusplus)
#define COLI_V4_METAL_EXTERN extern "C"
#else
#define COLI_V4_METAL_EXTERN extern
#endif

#ifndef COLI_V4_DEFAULT_METALLIB
#define COLI_V4_DEFAULT_METALLIB "build/metal-v4/deepseek_v4.metallib"
#endif

static id<MTLDevice> coli_v4_device;
static id<MTLCommandQueue> coli_v4_queue;
static id<MTLLibrary> coli_v4_library;
static id<MTLComputePipelineState> coli_v4_probe_pipeline;
static id<MTLComputePipelineState> coli_v4_qdq_pipeline;
static id<MTLComputePipelineState> coli_v4_matmul_pipeline;
static id<MTLComputePipelineState> coli_v4_matmul_hot_pipeline;
static id<MTLComputePipelineState> coli_v4_matmul_simd_exact_pipeline;
static id<MTLComputePipelineState> coli_v4_bf16_pipeline;
static id<MTLComputePipelineState> coli_v4_swiglu_pipeline;
static id<MTLComputePipelineState> coli_v4_weighted_pipeline;
static id<MTLComputePipelineState> coli_v4_expert_pipeline;
static const char *coli_v4_library_kind_value = "none";
static int coli_v4_metal_enabled_value;
static int coli_v4_metal_variant_value;
static int coli_v4_metal_profile_enabled_value;
static int coli_v4_metal_stats_enabled_value;
/* COLI_V4_METAL_ROWS16=1 -- EXPERIMENTAL, default OFF, KNOWN TO FAIL GOLDEN (experiments E97).
 * Lets rows16 (pinned hot) experts reach the GPU from the decode entry. They are ~45% of decode
 * expert calls. Retained as the instrument for the open E97 question; it must NOT be enabled in
 * any run whose output is expected to be bit-exact. */
static int coli_v4_metal_rows16_enabled_value;
static _Atomic unsigned long coli_v4_metal_dispatch_count;
/* Execution proof for the simd_exact matmul. AGENTS.md: a golden PASS with a flag ON proves
 * nothing about execution -- capture a counter that is zero without the feature. These are
 * printed by coli_v4_metal_stats_report under COLI_V4_METAL_STATS=1.
 *   simd_exact_matmuls  -- matmul encodes that used the simdgroup kernel (3 per expert call)
 *   simd_exact_rows16   -- expert calls that ASKED for simd_exact but hit the rows16 layout,
 *                          for which no simdgroup kernel exists yet, and so fell back to
 *                          ordered_hot_xcache. Silent fallback would corrupt an A/B. */
static _Atomic unsigned long coli_v4_metal_simd_exact_matmuls;
static _Atomic unsigned long coli_v4_metal_simd_exact_rows16_fallbacks;
/* Experts refused by the SINGLE-expert entry (the decode path) for not being block_rows==1,
 * and the last block_rows value seen there. Separated from the batch entry's layout
 * attribution because that refusal produces a layout reject with no per-tensor detail. */
static _Atomic unsigned long coli_v4_metal_single_rows_rejects;
static _Atomic unsigned long coli_v4_metal_single_rows_last;

typedef enum {
    COLI_V4_METAL_REJECT_VARIANT,
    COLI_V4_METAL_REJECT_INIT,
    COLI_V4_METAL_REJECT_QUEUE,
    COLI_V4_METAL_REJECT_DIMS,
    COLI_V4_METAL_REJECT_LAYOUT,
    COLI_V4_METAL_REJECT_PIPELINES,
    COLI_V4_METAL_REJECT_COUNT,
} ColiV4MetalReject;

typedef enum {
    COLI_V4_METAL_TENSOR_GATE,
    COLI_V4_METAL_TENSOR_UP,
    COLI_V4_METAL_TENSOR_DOWN,
    COLI_V4_METAL_TENSOR_COUNT,
} ColiV4MetalTensor;

typedef enum {
    COLI_V4_METAL_LAYOUT_FORMAT,
    COLI_V4_METAL_LAYOUT_POINTERS,
    COLI_V4_METAL_LAYOUT_ROWS,
    COLI_V4_METAL_LAYOUT_COLUMNS,
    COLI_V4_METAL_LAYOUT_ROW_BYTES,
    COLI_V4_METAL_LAYOUT_GROUPS,
    COLI_V4_METAL_LAYOUT_BLOCK_DIMS,
    COLI_V4_METAL_LAYOUT_FIELD_COUNT,
} ColiV4MetalLayoutField;

static _Atomic unsigned long coli_v4_metal_rejects[COLI_V4_METAL_REJECT_COUNT];
static _Atomic unsigned long coli_v4_metal_successes;
static _Atomic unsigned long coli_v4_metal_layout_tensors[COLI_V4_METAL_TENSOR_COUNT];
static _Atomic unsigned long coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_FIELD_COUNT];

static void coli_v4_metal_stats_report(void);

static void coli_v4_metal_reject(ColiV4MetalReject reject) {
    if (coli_v4_metal_stats_enabled_value)
        atomic_fetch_add_explicit(&coli_v4_metal_rejects[reject], 1,
                                  memory_order_relaxed);
}

typedef enum {
    COLI_V4_PROFILE_HOST_WEIGHT_MEMCPY,
    COLI_V4_PROFILE_FP8_QDQ_INPUT,
    COLI_V4_PROFILE_MATMUL_GATE,
    COLI_V4_PROFILE_MATMUL_UP,
    COLI_V4_PROFILE_BF16_GATE_UP,
    COLI_V4_PROFILE_SWIGLU,
    COLI_V4_PROFILE_WEIGHTED_BF16,
    COLI_V4_PROFILE_FP8_QDQ_DOWN_INPUT,
    COLI_V4_PROFILE_MATMUL_DOWN,
    COLI_V4_PROFILE_BF16_OUT,
    COLI_V4_PROFILE_SUBMIT_WAIT,
    COLI_V4_PROFILE_STAGE_COUNT,
} ColiV4ProfileStage;

static const char *const coli_v4_profile_stage_names[] = {
    "host weight memcpy",
    "fp8_qdq (input)",
    "matmul gate",
    "matmul up",
    "bf16 gate/up",
    "swiglu",
    "weighted_bf16",
    "fp8_qdq (down_input)",
    "matmul down",
    "bf16 out",
    "command-buffer submit+wait overhead",
};

typedef struct {
    double seconds[COLI_V4_PROFILE_STAGE_COUNT];
    unsigned long forwards;
} ColiV4MetalProfile;

static ColiV4MetalProfile coli_v4_metal_profile;
static pthread_mutex_t coli_v4_metal_profile_mutex = PTHREAD_MUTEX_INITIALIZER;

static double coli_v4_profile_now(void) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)mach_continuous_time() * (double)timebase.numer /
           (double)timebase.denom * 1.0e-9;
}

static void coli_v4_profile_add(ColiV4ProfileStage stage, double seconds) {
    pthread_mutex_lock(&coli_v4_metal_profile_mutex);
    coli_v4_metal_profile.seconds[stage] += seconds;
    pthread_mutex_unlock(&coli_v4_metal_profile_mutex);
}

static void coli_v4_profile_forward(void) {
    pthread_mutex_lock(&coli_v4_metal_profile_mutex);
    coli_v4_metal_profile.forwards++;
    pthread_mutex_unlock(&coli_v4_metal_profile_mutex);
}

enum {
    COLI_V4_MOE_MAX_HIDDEN = 4096,
    COLI_V4_MOE_MAX_INTERMEDIATE = 2048,
    COLI_V4_MOE_FP8_BLOCK = 128,
    COLI_V4_MOE_THREADS = 256,
    /* Apple GPUFamily 9 threadExecutionWidth. The simdgroup matmul geometry below maps one
     * simdgroup to one output row, so this must equal the device's actual execution width --
     * coli_v4_get_chain_pipelines verifies that against the pipeline before selecting it. */
    COLI_V4_MOE_SIMD_WIDTH = 32,
};

/* COLI_V4_METAL_VARIANT values. 1/2/3 are parsed but rejected by the expert path:
 * ordered_hot is superseded by the block_rows-driven selection, and simd_cold/simd_hot select
 * the tree-reduction kernel whose measured relative error is 1.06e-3 (see experiments E95). */
enum {
    COLI_V4_METAL_VARIANT_ORDERED_COLD = 0,
    COLI_V4_METAL_VARIANT_SIMD_EXACT_COLD = 4,
};

typedef struct { int S, I, O, rb, ng; } ColiV4MatmulDims;

/* The expert chain encodes three matmuls (gate, up, down) that differ only in their output
 * dimension. They previously carried three copies of the dispatch geometry; a kernel whose
 * grid shape differs would have needed that copied correctly three times, and a grid bug in
 * exactly one of them produces a plausible wrong answer rather than a crash. One helper. */
typedef enum {
    /* One thread per (s,o). Grid (ceil(O/256), S). Used by the ordered/xcache kernels, whose
     * threadgroup also stages the x vector, so all threads in a group must share one s. */
    COLI_V4_MATMUL_GEOM_ROWS = 0,
    /* One simdgroup per (s,o); lane L owns group gb+L. Grid (32, S*O). Used by simd_exact. */
    COLI_V4_MATMUL_GEOM_SIMDGROUP = 1,
} ColiV4MatmulGeometry;

static void coli_v4_encode_matmul(id<MTLComputeCommandEncoder> encoder,
                                  ColiV4MatmulGeometry geometry,
                                  NSUInteger out_dim, NSUInteger batch) {
    if (geometry == COLI_V4_MATMUL_GEOM_SIMDGROUP) {
        atomic_fetch_add_explicit(&coli_v4_metal_simd_exact_matmuls, 1, memory_order_relaxed);
        [encoder dispatchThreadgroups:MTLSizeMake(1, batch * out_dim, 1)
                threadsPerThreadgroup:MTLSizeMake(COLI_V4_MOE_SIMD_WIDTH, 1, 1)];
        return;
    }
    [encoder dispatchThreadgroups:MTLSizeMake(
                (out_dim + COLI_V4_MOE_THREADS - 1) / COLI_V4_MOE_THREADS, batch, 1)
            threadsPerThreadgroup:MTLSizeMake(COLI_V4_MOE_THREADS, 1, 1)];
}
typedef struct { int length, block_size; } ColiV4QdqParams;
typedef struct { int dimension; float limit; } ColiV4SwigluParams;
typedef struct { int n; float w; } ColiV4WeightedBf16Params;

static_assert(sizeof(ColiV4MatmulDims) == 5 * sizeof(uint32_t), "Metal Matmul ABI changed");
static_assert(sizeof(ColiV4QdqParams) == 2 * sizeof(uint32_t), "Metal QDQ ABI changed");
static_assert(sizeof(ColiV4SwigluParams) == 2 * sizeof(uint32_t), "Metal SwiGLU ABI changed");
static_assert(sizeof(ColiV4WeightedBf16Params) == 2 * sizeof(uint32_t), "Metal weighted ABI changed");

static id<MTLBuffer> coli_v4_input_scratch;
static id<MTLBuffer> coli_v4_qdq_scratch;
static id<MTLBuffer> coli_v4_input_scales_scratch;
static id<MTLBuffer> coli_v4_gate_q4_scratch;
static id<MTLBuffer> coli_v4_gate_scales_scratch;
static id<MTLBuffer> coli_v4_up_q4_scratch;
static id<MTLBuffer> coli_v4_up_scales_scratch;
static id<MTLBuffer> coli_v4_down_q4_scratch;
static id<MTLBuffer> coli_v4_down_scales_scratch;
static id<MTLBuffer> coli_v4_gate_scratch;
static id<MTLBuffer> coli_v4_up_scratch;
static id<MTLBuffer> coli_v4_activated_scratch;
static id<MTLBuffer> coli_v4_weighted_scratch;
static id<MTLBuffer> coli_v4_down_input_scratch;
static id<MTLBuffer> coli_v4_down_input_scales_scratch;
static id<MTLBuffer> coli_v4_output_scratch;
static size_t coli_v4_input_scratch_capacity;
static size_t coli_v4_qdq_scratch_capacity;
static size_t coli_v4_input_scales_scratch_capacity;
static size_t coli_v4_gate_q4_scratch_capacity;
static size_t coli_v4_gate_scales_scratch_capacity;
static size_t coli_v4_up_q4_scratch_capacity;
static size_t coli_v4_up_scales_scratch_capacity;
static size_t coli_v4_down_q4_scratch_capacity;
static size_t coli_v4_down_scales_scratch_capacity;
static size_t coli_v4_gate_scratch_capacity;
static size_t coli_v4_up_scratch_capacity;
static size_t coli_v4_activated_scratch_capacity;
static size_t coli_v4_weighted_scratch_capacity;
static size_t coli_v4_down_input_scratch_capacity;
static size_t coli_v4_down_input_scales_scratch_capacity;
static size_t coli_v4_output_scratch_capacity;

typedef struct ColiV4SlabBuffer {
    const void *base;
    size_t length;
    id<MTLBuffer> buffer;
    struct ColiV4SlabBuffer *next;
} ColiV4SlabBuffer;

static ColiV4SlabBuffer *coli_v4_slab_buffers;
static pthread_mutex_t coli_v4_slab_buffers_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Atomic unsigned long coli_v4_metal_zero_copy_tensors;
static _Atomic unsigned long coli_v4_metal_copy_fallback_tensors;

static id<MTLBuffer> coli_v4_resolve_slab(const void *pointer, size_t *offset) {
    pthread_mutex_lock(&coli_v4_slab_buffers_mutex);
    uintptr_t address = (uintptr_t)pointer;
    for (ColiV4SlabBuffer *entry = coli_v4_slab_buffers; entry; entry = entry->next) {
        uintptr_t base = (uintptr_t)entry->base;
        if (address >= base && address < base + entry->length) {
            if (!entry->buffer && coli_v4_device)
                entry->buffer = [coli_v4_device
                    newBufferWithBytesNoCopy:(void *)entry->base
                                        length:entry->length
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
            *offset = (size_t)(address - base);
            id<MTLBuffer> result = entry->buffer;
            pthread_mutex_unlock(&coli_v4_slab_buffers_mutex);
            return result;
        }
    }
    pthread_mutex_unlock(&coli_v4_slab_buffers_mutex);
    return nil;
}

static id<MTLBuffer> coli_v4_weight_buffer(const void *pointer, size_t bytes,
                                            id<MTLBuffer> scratch,
                                            size_t *offset) {
    id<MTLBuffer> slab = coli_v4_resolve_slab(pointer, offset);
    if (slab && *offset % 256 == 0 && *offset + bytes <= slab.length) {
        atomic_fetch_add_explicit(&coli_v4_metal_zero_copy_tensors, 1,
                                  memory_order_relaxed);
        return slab;
    }
    *offset = 0;
    atomic_fetch_add_explicit(&coli_v4_metal_copy_fallback_tensors, 1,
                              memory_order_relaxed);
    memcpy(scratch.contents, pointer, bytes);
    return scratch;
}

COLI_V4_METAL_EXTERN void coli_v4_metal_register_slab(void *base, size_t length) {
    if (!base || !length || (uintptr_t)base % 16384u || length % 16384u) return;
    pthread_mutex_lock(&coli_v4_slab_buffers_mutex);
    for (ColiV4SlabBuffer *entry = coli_v4_slab_buffers; entry; entry = entry->next) {
        if (entry->base == base) {
            entry->length = length;
            entry->buffer = nil;
            pthread_mutex_unlock(&coli_v4_slab_buffers_mutex);
            return;
        }
    }
    ColiV4SlabBuffer *entry = (ColiV4SlabBuffer *)calloc(1, sizeof(*entry));
    if (entry) {
        entry->base = base;
        entry->length = length;
        entry->next = coli_v4_slab_buffers;
        coli_v4_slab_buffers = entry;
    }
    pthread_mutex_unlock(&coli_v4_slab_buffers_mutex);
}

COLI_V4_METAL_EXTERN void coli_v4_metal_unregister_slab(void *base) {
    pthread_mutex_lock(&coli_v4_slab_buffers_mutex);
    ColiV4SlabBuffer **link = &coli_v4_slab_buffers;
    while (*link) {
        if ((*link)->base == base) {
            ColiV4SlabBuffer *entry = *link;
            *link = entry->next;
            entry->buffer = nil;
            free(entry);
            pthread_mutex_unlock(&coli_v4_slab_buffers_mutex);
            return;
        }
        link = &(*link)->next;
    }
    pthread_mutex_unlock(&coli_v4_slab_buffers_mutex);
}

static id<MTLBuffer> coli_v4_ensure_scratch(id<MTLBuffer> buffer,
                                             size_t *capacity, size_t need) {
    if (buffer && *capacity >= need) return buffer;
    id<MTLBuffer> replacement = [coli_v4_device
        newBufferWithLength:need options:MTLResourceStorageModeShared];
    if (!replacement) return nil;
    *capacity = need;
    return replacement;
}

static int coli_v4_ordered_tensor_valid(const ColiTensorView *tensor,
                                        int rows, int columns,
                                        size_t row_bytes, size_t groups,
                                        uint32_t block_rows) {
    return tensor && tensor->format == COLI_TENSOR_FP4_NATIVE_BLOCK &&
           tensor->scale_format == COLI_SCALE_UE8M0 && tensor->data &&
           tensor->scales && tensor->rows == rows &&
           tensor->columns == columns && tensor->block_rows == block_rows &&
           tensor->block_columns == 32 &&
           tensor->data_bytes == (size_t)rows * row_bytes &&
           tensor->scale_bytes == (size_t)rows * groups;
}

static void coli_v4_record_layout_mismatches(
    const ColiTensorView *tensor, int rows, int columns, size_t row_bytes,
    size_t groups, uint32_t block_rows, ColiV4MetalTensor tensor_kind) {
    if (coli_v4_ordered_tensor_valid(tensor, rows, columns, row_bytes, groups,
                                     block_rows)) return;
    atomic_fetch_add_explicit(&coli_v4_metal_layout_tensors[tensor_kind], 1,
                              memory_order_relaxed);
    if (!tensor) {
        atomic_fetch_add_explicit(
            &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_POINTERS], 1,
            memory_order_relaxed);
        return;
    }
#define COLI_V4_LAYOUT_MISMATCH(field, condition) do { \
        if (condition) \
            atomic_fetch_add_explicit(&coli_v4_metal_layout_fields[field], 1, \
                                      memory_order_relaxed); \
    } while (0)
    COLI_V4_LAYOUT_MISMATCH(
        COLI_V4_METAL_LAYOUT_FORMAT,
        tensor->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        tensor->scale_format != COLI_SCALE_UE8M0);
    COLI_V4_LAYOUT_MISMATCH(COLI_V4_METAL_LAYOUT_POINTERS,
                            !tensor->data || !tensor->scales);
    COLI_V4_LAYOUT_MISMATCH(COLI_V4_METAL_LAYOUT_ROWS,
                            tensor->rows != rows);
    COLI_V4_LAYOUT_MISMATCH(COLI_V4_METAL_LAYOUT_COLUMNS,
                            tensor->columns != columns);
    COLI_V4_LAYOUT_MISMATCH(COLI_V4_METAL_LAYOUT_ROW_BYTES,
                            tensor->data_bytes != (size_t)rows * row_bytes);
    COLI_V4_LAYOUT_MISMATCH(COLI_V4_METAL_LAYOUT_GROUPS,
                            tensor->scale_bytes != (size_t)rows * groups);
    COLI_V4_LAYOUT_MISMATCH(COLI_V4_METAL_LAYOUT_BLOCK_DIMS,
                            tensor->block_rows != block_rows ||
                            tensor->block_columns != 32);
#undef COLI_V4_LAYOUT_MISMATCH
}

static id<MTLComputePipelineState> __attribute__((unused)) coli_v4_get_expert_pipeline(void) {
    if (coli_v4_expert_pipeline) return coli_v4_expert_pipeline;
    if (!coli_v4_device || !coli_v4_library) return nil;
    NSError *error = nil;
    id<MTLFunction> function = [coli_v4_library
        newFunctionWithName:@"coli_v4_moe_expert_fp4_ordered_cold"];
    id<MTLComputePipelineState> pipeline = function
        ? [coli_v4_device newComputePipelineStateWithFunction:function
                                                         error:&error]
        : nil;
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup <
                         COLI_V4_MOE_THREADS ||
        pipeline.staticThreadgroupMemoryLength >
                         coli_v4_device.maxThreadgroupMemoryLength)
        return nil;
    coli_v4_expert_pipeline = pipeline;
    return pipeline;
}

static id<MTLComputePipelineState> coli_v4_get_named_pipeline(const char *name) {
    if (!coli_v4_device || !coli_v4_library) return nil;
    NSError *error = nil;
    id<MTLFunction> function = [coli_v4_library
        newFunctionWithName:[NSString stringWithUTF8String:name]];
    return function ? [coli_v4_device newComputePipelineStateWithFunction:function
                                                                      error:&error]
                    : nil;
}

static int coli_v4_get_chain_pipelines(uint32_t block_rows, int variant,
                                       id<MTLComputePipelineState> *matmul,
                                       ColiV4MatmulGeometry *geometry) {
    if (!coli_v4_qdq_pipeline) coli_v4_qdq_pipeline = coli_v4_get_named_pipeline("coli_v4_fp8_qdq");
    if (block_rows == 1 && !coli_v4_matmul_pipeline)
        coli_v4_matmul_pipeline = coli_v4_get_named_pipeline("coli_v4_matmul_mxfp4_ordered_xcache");
    if (block_rows == 16 && !coli_v4_matmul_hot_pipeline)
        coli_v4_matmul_hot_pipeline = coli_v4_get_named_pipeline("coli_v4_matmul_mxfp4_ordered_hot_xcache");
    if (!coli_v4_bf16_pipeline) coli_v4_bf16_pipeline = coli_v4_get_named_pipeline("coli_v4_bf16_round_array");
    if (!coli_v4_swiglu_pipeline) coli_v4_swiglu_pipeline = coli_v4_get_named_pipeline("coli_v4_swiglu");
    if (!coli_v4_weighted_pipeline) coli_v4_weighted_pipeline = coli_v4_get_named_pipeline("coli_v4_weighted_bf16");

    *geometry = COLI_V4_MATMUL_GEOM_ROWS;
    *matmul = block_rows == 16 ? coli_v4_matmul_hot_pipeline : coli_v4_matmul_pipeline;

    if (variant == COLI_V4_METAL_VARIANT_SIMD_EXACT_COLD) {
        if (block_rows == 1) {
            if (!coli_v4_matmul_simd_exact_pipeline)
                coli_v4_matmul_simd_exact_pipeline =
                    coli_v4_get_named_pipeline("coli_v4_matmul_mxfp4_simd_exact");
            /* The bit-exactness argument depends on one threadgroup being exactly one
             * simdgroup: lane L owns group gb+L and the outer reduction shuffles across all
             * 32 lanes. On a device whose execution width is not 32 the mapping is wrong, so
             * refuse the pipeline rather than silently computing something else. */
            if (coli_v4_matmul_simd_exact_pipeline &&
                coli_v4_matmul_simd_exact_pipeline.threadExecutionWidth == COLI_V4_MOE_SIMD_WIDTH) {
                *matmul = coli_v4_matmul_simd_exact_pipeline;
                *geometry = COLI_V4_MATMUL_GEOM_SIMDGROUP;
            } else {
                return 0;   /* -> COLI_V4_METAL_REJECT_PIPELINES */
            }
        } else {
            /* No rows16/hot simdgroup kernel exists yet. FALL BACK to ordered_hot_xcache
             * rather than rejecting: rejecting returns -1 and sends the expert to the CPU,
             * which would turn a kernel A/B into a GPU-vs-CPU swap and make the measurement
             * uninterpretable. Counted so the fallback is never silent. */
            atomic_fetch_add_explicit(&coli_v4_metal_simd_exact_rows16_fallbacks, 1,
                                      memory_order_relaxed);
        }
    }

    /* The 256-thread floor is a device-capability check on the ordered kernels, whose
     * threadgroup stages the x vector. The simdgroup kernel dispatches 32-wide threadgroups
     * and stages nothing, so that floor does not apply to it. */
    NSUInteger required_threads =
        (*geometry == COLI_V4_MATMUL_GEOM_SIMDGROUP) ? COLI_V4_MOE_SIMD_WIDTH : COLI_V4_MOE_THREADS;
    return coli_v4_qdq_pipeline && *matmul && coli_v4_bf16_pipeline &&
           coli_v4_swiglu_pipeline && coli_v4_weighted_pipeline &&
           (*matmul).maxTotalThreadsPerThreadgroup >= required_threads &&
           (*matmul).staticThreadgroupMemoryLength <= coli_v4_device.maxThreadgroupMemoryLength;
}

/* Mirrors coli_v4_baseline_mode() from deepseek_v4_internal.h, which this file does not include. */
static int coli_v4_metal_baseline_mode(void) {
    const char *v = getenv("COLI_V4_BASELINE");
    return v && *v && atoi(v) != 0;
}

__attribute__((constructor)) static void coli_v4_metal_read_environment(void) {
    /* NOTE: COLI_V4_METAL stays default-OFF. It gates SINGLE-TOKEN DECODE Metal, which measured
     * SLOWER here (E99); the champion runs it off. Prefill attention, batched MoE and the
     * whole-prompt dispatch are gated independently and default ON. */
    const char *enabled = getenv("COLI_V4_METAL");
    coli_v4_metal_enabled_value = enabled && !strcmp(enabled, "1");
    const char *profile = getenv("COLI_V4_METAL_PROFILE");
    coli_v4_metal_profile_enabled_value = profile && !strcmp(profile, "1");
    const char *rows16 = getenv("COLI_V4_METAL_ROWS16");
    coli_v4_metal_rows16_enabled_value = rows16 && !strcmp(rows16, "1");
    const char *stats = getenv("COLI_V4_METAL_STATS");
    coli_v4_metal_stats_enabled_value = stats && !strcmp(stats, "1");
    if (coli_v4_metal_stats_enabled_value) atexit(coli_v4_metal_stats_report);

    const char *variant = getenv("COLI_V4_METAL_VARIANT");
    if (!variant)
        coli_v4_metal_variant_value = coli_v4_metal_baseline_mode() ? 0 : 4;
    else if (!strcmp(variant, "ordered_cold"))
        coli_v4_metal_variant_value = 0;
    else if (!strcmp(variant, "ordered_hot"))
        coli_v4_metal_variant_value = 1;
    else if (!strcmp(variant, "simd_cold"))
        coli_v4_metal_variant_value = 2;
    else if (!strcmp(variant, "simd_hot"))
        coli_v4_metal_variant_value = 3;
    /* simd_exact_cold: the bit-exact simdgroup matmul (coli_v4_matmul_mxfp4_simd_exact).
     * Distinct from simd_cold=2, which selects the tree-reduction kernel that measures
     * 1.06e-3 relative error and is therefore still rejected by the expert path. */
    else if (!strcmp(variant, "simd_exact_cold"))
        coli_v4_metal_variant_value = 4;
    else
        coli_v4_metal_variant_value = 0;
}

COLI_V4_METAL_EXTERN __attribute__((used)) int coli_v4_metal_enabled(void) {
    return coli_v4_metal_enabled_value;
}

COLI_V4_METAL_EXTERN __attribute__((used)) int coli_v4_metal_variant(void) {
    return coli_v4_metal_variant_value;
}

COLI_V4_METAL_EXTERN __attribute__((used)) unsigned long
coli_v4_metal_dispatches(void) {
    return atomic_load_explicit(&coli_v4_metal_dispatch_count,
                                memory_order_relaxed);
}


/* ==========================================================================================
 * Bit-exact batched fp8 matmul for the attention projections (E68/E69).
 *
 * Attention dense weights are RESIDENT and IMMUTABLE for the life of the engine, so each
 * weight matrix is uploaded to the GPU AT MOST ONCE and cached by host pointer. wq_b alone is
 * 33.5 MB and is used on every layer-chunk call, so a per-call upload would erase the win.
 * Zero-copy (newBufferWithBytesNoCopy) is attempted first and only falls back to one memcpy.
 * ========================================================================================== */
static id<MTLComputePipelineState> coli_v4_fp8_pipeline;
static id<MTLBuffer> coli_v4_fp8_lut_buffer;
static id<MTLBuffer> coli_v4_fp8_in_scratch;
static id<MTLBuffer> coli_v4_fp8_out_scratch;
static int coli_v4_fp8_enabled_value = -1;
static _Atomic unsigned long coli_v4_fp8_dispatch_count;
static _Atomic unsigned long coli_v4_fp8_upload_bytes;
static _Atomic unsigned long coli_v4_fp8_cache_full;
/* A4 diagnosis: split the per-call cost so the 42% miss can be attributed, not guessed. */
static _Atomic unsigned long coli_v4_fp8_ns_memcpy_in;
static _Atomic unsigned long coli_v4_fp8_ns_dispatch;
static _Atomic unsigned long coli_v4_fp8_ns_memcpy_out;
static _Atomic unsigned long coli_v4_fp8_ns_total;
static _Atomic unsigned long coli_v4_fp8_rows_x_s;
static inline unsigned long coli_v4_fp8_now(void){
    static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
    return (unsigned long)((double)mach_absolute_time()*tb.numer/tb.denom);
}

/* 43 layers x (4 single projections + o_groups=8 per-group wo_a slices) x (data + scales)
 * = 1032 entries. 512 overflowed partway through and then silently returned nil for every
 * subsequent weight, permanently falling back to CPU - wo_a is registered last per layer so it
 * was starved first, which is why it measured as the most expensive projection. */
#define COLI_V4_FP8_CACHE 4096
typedef struct { const void *key; id<MTLBuffer> buf; size_t bytes; int nocopy; } ColiV4Fp8Entry;
static ColiV4Fp8Entry coli_v4_fp8_cache[COLI_V4_FP8_CACHE];
static int coli_v4_fp8_cache_count;

COLI_V4_METAL_EXTERN int coli_v4_metal_fp8_enabled(void) {
    if (coli_v4_fp8_enabled_value < 0) {
        /* Default ON: -21.0% TTFT and BIT-EXACT over six runs (E115), so unlike the other gates
         * this one costs nothing in reproducibility. Already prefill-only by construction. */
        const char *v = getenv("COLI_V4_METAL_ATTN");
        coli_v4_fp8_enabled_value = (v && *v) ? (atoi(v) != 0)
                                              : (coli_v4_metal_baseline_mode() ? 0 : 1);
    }
    return coli_v4_fp8_enabled_value;
}

COLI_V4_METAL_EXTERN void coli_v4_metal_fp8_register_lut(const float *lut256) {
    if (!lut256 || !coli_v4_device || coli_v4_fp8_lut_buffer) return;
    coli_v4_fp8_lut_buffer = [coli_v4_device newBufferWithBytes:lut256
                                                         length:256 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
}

/* Immutable weight -> MTLBuffer, uploaded at most once. */
static id<MTLBuffer> coli_v4_fp8_resident(const void *ptr, size_t bytes) {
    if (!ptr || !bytes || !coli_v4_device) return nil;
    for (int i = 0; i < coli_v4_fp8_cache_count; i++)
        if (coli_v4_fp8_cache[i].key == ptr && coli_v4_fp8_cache[i].bytes == bytes)
            return coli_v4_fp8_cache[i].buf;
    if (coli_v4_fp8_cache_count >= COLI_V4_FP8_CACHE) {
        atomic_fetch_add_explicit(&coli_v4_fp8_cache_full, 1, memory_order_relaxed);
        return nil;                      /* refuse, fall back to CPU - now COUNTED, not silent */
    }
    id<MTLBuffer> buf = nil;
    int nocopy = 0;
    if (((uintptr_t)ptr % 16384u) == 0 && (bytes % 16384u) == 0) {
        buf = [coli_v4_device newBufferWithBytesNoCopy:(void *)ptr
                                                length:bytes
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
        nocopy = buf ? 1 : 0;
    }
    if (!buf) {
        buf = [coli_v4_device newBufferWithBytes:ptr length:bytes
                                         options:MTLResourceStorageModeShared];
        if (buf) atomic_fetch_add_explicit(&coli_v4_fp8_upload_bytes,
                                          (unsigned long)bytes, memory_order_relaxed);
    }
    if (!buf) return nil;
    coli_v4_fp8_cache[coli_v4_fp8_cache_count].key    = ptr;
    coli_v4_fp8_cache[coli_v4_fp8_cache_count].buf    = buf;
    coli_v4_fp8_cache[coli_v4_fp8_cache_count].bytes  = bytes;
    coli_v4_fp8_cache[coli_v4_fp8_cache_count].nocopy = nocopy;
    coli_v4_fp8_cache_count++;
    return buf;
}

static id<MTLBuffer> coli_v4_fp8_grow(__strong id<MTLBuffer> *slot, size_t bytes) {
    if (*slot && (*slot).length >= bytes) return *slot;
    if (!coli_v4_device) return nil;
    *slot = [coli_v4_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return *slot;
}

COLI_V4_METAL_EXTERN int coli_v4_metal_fp8_matmul_batch(
        float *outputs, const void *weight_data, const float *weight_scales,
        const float *inputs, int batch, int rows, int columns) {
    if (!outputs || !weight_data || !weight_scales || !inputs) return -1;
    if (batch < 1 || rows < 1 || columns < 1 || (columns % 128) != 0) return -1;
    if (!coli_v4_metal_fp8_enabled()) return -1;
    /* Metal init is otherwise only triggered lazily by the MoE expert path, which is gated on
     * COLI_V4_METAL=1. Without this, setting COLI_V4_METAL_ATTN alone would leave the device nil,
     * make every call return -1, and produce a VACUOUS "bit-exact" pass from a path that never
     * ran (the failure mode recorded in RESULTS.md S12e). */
    if (!coli_v4_metal_available() && !coli_v4_metal_init(NULL)) return -1;
    if (!coli_v4_device || !coli_v4_queue) return -1;
    if (!coli_v4_fp8_lut_buffer) return -1;          /* LUT must be registered by the engine */

    if (!coli_v4_fp8_pipeline) {
        coli_v4_fp8_pipeline = coli_v4_get_named_pipeline("coli_v4_fp8_matmul_batch");
        if (!coli_v4_fp8_pipeline) return -1;
    }
    size_t O = (size_t)rows, I = (size_t)columns, S = (size_t)batch;
    size_t nblkI = (I + 127) / 128;
    size_t wbytes = O * I;
    size_t sbytes = ((O + 127) / 128) * nblkI * sizeof(float);

    id<MTLBuffer> W  = coli_v4_fp8_resident(weight_data, wbytes);
    id<MTLBuffer> SC = coli_v4_fp8_resident(weight_scales, sbytes);
    if (!W || !SC) return -1;
    id<MTLBuffer> X = coli_v4_fp8_grow(&coli_v4_fp8_in_scratch,  S * I * sizeof(float));
    id<MTLBuffer> Y = coli_v4_fp8_grow(&coli_v4_fp8_out_scratch, S * O * sizeof(float));
    if (!X || !Y) return -1;
    unsigned long t_all0 = coli_v4_fp8_now();
    unsigned long t0 = coli_v4_fp8_now();
    memcpy(X.contents, inputs, S * I * sizeof(float));
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_memcpy_in, coli_v4_fp8_now()-t0, memory_order_relaxed);

    struct { unsigned int O, I, S, nblkI; } dims = {
        (unsigned int)O, (unsigned int)I, (unsigned int)S, (unsigned int)nblkI };

    unsigned long t_d0 = coli_v4_fp8_now();
    id<MTLCommandBuffer> cb = [coli_v4_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
    if (!enc) return -1;
    [enc setComputePipelineState:coli_v4_fp8_pipeline];
    [enc setBuffer:W  offset:0 atIndex:0];
    [enc setBuffer:SC offset:0 atIndex:1];
    [enc setBuffer:X  offset:0 atIndex:2];
    [enc setBuffer:Y  offset:0 atIndex:3];
    [enc setBuffer:coli_v4_fp8_lut_buffer offset:0 atIndex:4];
    [enc setBytes:&dims length:sizeof(dims) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(O, S, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_dispatch, coli_v4_fp8_now()-t_d0, memory_order_relaxed);
    if (cb.status != MTLCommandBufferStatusCompleted) return -1;

    unsigned long t_o0 = coli_v4_fp8_now();
    memcpy(outputs, Y.contents, S * O * sizeof(float));
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_memcpy_out, coli_v4_fp8_now()-t_o0, memory_order_relaxed);
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_total, coli_v4_fp8_now()-t_all0, memory_order_relaxed);
    atomic_fetch_add_explicit(&coli_v4_fp8_rows_x_s, (unsigned long)(O*S), memory_order_relaxed);
    atomic_fetch_add_explicit(&coli_v4_fp8_dispatch_count, 1, memory_order_relaxed);
    return 0;
}


static id<MTLComputePipelineState> coli_v4_fp8_grouped_pipeline;

COLI_V4_METAL_EXTERN int coli_v4_metal_fp8_matmul_grouped(
        float *outputs, const void *weight_data, const float *weight_scales,
        const float *inputs, int batch, int rows, int columns, int groups) {
    if (!outputs || !weight_data || !weight_scales || !inputs) return -1;
    if (batch < 1 || rows < 1 || columns < 1 || groups < 1 || (columns % 128) != 0) return -1;
    if (!coli_v4_metal_fp8_enabled()) return -1;
    if (!coli_v4_metal_available() && !coli_v4_metal_init(NULL)) return -1;
    if (!coli_v4_device || !coli_v4_queue) return -1;
    if (!coli_v4_fp8_lut_buffer) return -1;
    if (!coli_v4_fp8_grouped_pipeline) {
        coli_v4_fp8_grouped_pipeline =
            coli_v4_get_named_pipeline("coli_v4_fp8_matmul_grouped");
        if (!coli_v4_fp8_grouped_pipeline) return -1;
    }
    size_t O = (size_t)rows, I = (size_t)columns, S = (size_t)batch, G = (size_t)groups;
    size_t nblkI = (I + 127) / 128;
    size_t wbytes = G * O * I;
    size_t sbytes = G * ((O + 127) / 128) * nblkI * sizeof(float);

    id<MTLBuffer> W  = coli_v4_fp8_resident(weight_data, wbytes);
    id<MTLBuffer> SC = coli_v4_fp8_resident(weight_scales, sbytes);
    if (!W || !SC) return -1;
    id<MTLBuffer> X = coli_v4_fp8_grow(&coli_v4_fp8_in_scratch,  G * S * I * sizeof(float));
    id<MTLBuffer> Y = coli_v4_fp8_grow(&coli_v4_fp8_out_scratch, G * S * O * sizeof(float));
    if (!X || !Y) return -1;

    unsigned long t_all0 = coli_v4_fp8_now();
    unsigned long t0 = coli_v4_fp8_now();
    memcpy(X.contents, inputs, G * S * I * sizeof(float));
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_memcpy_in, coli_v4_fp8_now()-t0, memory_order_relaxed);

    struct { unsigned int O, I, S, nblkI; } dims = {
        (unsigned int)O, (unsigned int)I, (unsigned int)S, (unsigned int)nblkI };
    unsigned int ng = (unsigned int)G;

    unsigned long t_d0 = coli_v4_fp8_now();
    id<MTLCommandBuffer> cb = [coli_v4_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
    if (!enc) return -1;
    [enc setComputePipelineState:coli_v4_fp8_grouped_pipeline];
    [enc setBuffer:W  offset:0 atIndex:0];
    [enc setBuffer:SC offset:0 atIndex:1];
    [enc setBuffer:X  offset:0 atIndex:2];
    [enc setBuffer:Y  offset:0 atIndex:3];
    [enc setBuffer:coli_v4_fp8_lut_buffer offset:0 atIndex:4];
    [enc setBytes:&dims length:sizeof(dims) atIndex:5];
    [enc setBytes:&ng   length:sizeof(ng)   atIndex:6];
    [enc dispatchThreads:MTLSizeMake(O, S, G)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_dispatch, coli_v4_fp8_now()-t_d0, memory_order_relaxed);
    if (cb.status != MTLCommandBufferStatusCompleted) return -1;

    unsigned long t_o0 = coli_v4_fp8_now();
    memcpy(outputs, Y.contents, G * S * O * sizeof(float));
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_memcpy_out, coli_v4_fp8_now()-t_o0, memory_order_relaxed);
    atomic_fetch_add_explicit(&coli_v4_fp8_ns_total, coli_v4_fp8_now()-t_all0, memory_order_relaxed);
    atomic_fetch_add_explicit(&coli_v4_fp8_dispatch_count, 1, memory_order_relaxed);
    return 0;
}

COLI_V4_METAL_EXTERN unsigned long coli_v4_metal_fp8_dispatches(void) {
    return atomic_load_explicit(&coli_v4_fp8_dispatch_count, memory_order_relaxed);
}
COLI_V4_METAL_EXTERN void coli_v4_metal_fp8_timing(unsigned long *in_ns, unsigned long *disp_ns,
                                                   unsigned long *out_ns, unsigned long *tot_ns,
                                                   unsigned long *rows_x_s) {
    if(in_ns)  *in_ns  = atomic_load_explicit(&coli_v4_fp8_ns_memcpy_in, memory_order_relaxed);
    if(disp_ns)*disp_ns= atomic_load_explicit(&coli_v4_fp8_ns_dispatch, memory_order_relaxed);
    if(out_ns) *out_ns = atomic_load_explicit(&coli_v4_fp8_ns_memcpy_out, memory_order_relaxed);
    if(tot_ns) *tot_ns = atomic_load_explicit(&coli_v4_fp8_ns_total, memory_order_relaxed);
    if(rows_x_s)*rows_x_s=atomic_load_explicit(&coli_v4_fp8_rows_x_s, memory_order_relaxed);
}
COLI_V4_METAL_EXTERN unsigned long coli_v4_metal_fp8_cache_full(void) {
    return atomic_load_explicit(&coli_v4_fp8_cache_full, memory_order_relaxed);
}
COLI_V4_METAL_EXTERN unsigned long coli_v4_metal_fp8_upload_bytes(void) {
    return atomic_load_explicit(&coli_v4_fp8_upload_bytes, memory_order_relaxed);
}

COLI_V4_METAL_EXTERN __attribute__((used)) int coli_v4_metal_expert_forward_batch(
    float *outs, const ColiExpertView *expert, const float *inputs_gathered,
    const float *route_weights, int batch, float swiglu_limit) {
    if (!outs || !expert || !inputs_gathered || !route_weights || batch < 1 ||
        swiglu_limit < 0.0f ||
        (coli_v4_metal_variant() != COLI_V4_METAL_VARIANT_ORDERED_COLD &&
         coli_v4_metal_variant() != COLI_V4_METAL_VARIANT_SIMD_EXACT_COLD)) {
        coli_v4_metal_reject(COLI_V4_METAL_REJECT_VARIANT);
        return -1;
    }
    if (!coli_v4_metal_available() && !coli_v4_metal_init(NULL)) {
        coli_v4_metal_reject(COLI_V4_METAL_REJECT_INIT);
        return -1;
    }
    if (!coli_v4_queue) {
        coli_v4_metal_reject(COLI_V4_METAL_REJECT_QUEUE);
        return -1;
    }

    const int64_t hidden64 = expert->down.rows;
    const int64_t intermediate64 = expert->gate.rows;
    if (hidden64 < 1 || hidden64 > COLI_V4_MOE_MAX_HIDDEN ||
        intermediate64 < 1 || intermediate64 > COLI_V4_MOE_MAX_INTERMEDIATE ||
        expert->gate.rows != expert->up.rows ||
        expert->gate.columns != expert->up.columns ||
        expert->gate.columns != hidden64 ||
        expert->down.rows != hidden64 ||
        expert->down.columns != intermediate64) {
        coli_v4_metal_reject(COLI_V4_METAL_REJECT_DIMS);
        return -1;
    }

    const int hidden = (int)hidden64;
    const int intermediate = (int)intermediate64;
    const size_t gate_row_bytes = ((size_t)hidden + 1) / 2;
    const size_t gate_groups = ((size_t)hidden + 31) / 32;
    const size_t down_row_bytes = ((size_t)intermediate + 1) / 2;
    const size_t down_groups = ((size_t)intermediate + 31) / 32;
    const uint32_t block_rows = expert->gate.block_rows;
    if ((block_rows != 1 && block_rows != 16) ||
        !coli_v4_ordered_tensor_valid(&expert->gate, intermediate, hidden,
                                      gate_row_bytes, gate_groups, block_rows) ||
        !coli_v4_ordered_tensor_valid(&expert->up, intermediate, hidden,
                                      gate_row_bytes, gate_groups, block_rows) ||
        !coli_v4_ordered_tensor_valid(&expert->down, hidden, intermediate,
                                      down_row_bytes, down_groups, block_rows)) {
        if (coli_v4_metal_stats_enabled_value) {
            coli_v4_record_layout_mismatches(
                &expert->gate, intermediate, hidden, gate_row_bytes,
                gate_groups, block_rows, COLI_V4_METAL_TENSOR_GATE);
            coli_v4_record_layout_mismatches(
                &expert->up, intermediate, hidden, gate_row_bytes,
                gate_groups, block_rows, COLI_V4_METAL_TENSOR_UP);
            coli_v4_record_layout_mismatches(
                &expert->down, hidden, intermediate, down_row_bytes,
                down_groups, block_rows, COLI_V4_METAL_TENSOR_DOWN);
        }
        coli_v4_metal_reject(COLI_V4_METAL_REJECT_LAYOUT);
        return -1;
    }

    id<MTLComputePipelineState> matmul_pipeline = nil;
    ColiV4MatmulGeometry matmul_geometry = COLI_V4_MATMUL_GEOM_ROWS;
    if (!coli_v4_get_chain_pipelines(block_rows, coli_v4_metal_variant(),
                                     &matmul_pipeline, &matmul_geometry)) {
        coli_v4_metal_reject(COLI_V4_METAL_REJECT_PIPELINES);
        return -1;
    }

    @autoreleasepool {
        const size_t input_bytes = (size_t)batch * hidden * sizeof(*inputs_gathered);
        const size_t output_bytes = (size_t)batch * hidden * sizeof(*outs);
        const size_t gate_q4_bytes = expert->gate.data_bytes;
        const size_t gate_scales_bytes = expert->gate.scale_bytes;
        const size_t up_q4_bytes = expert->up.data_bytes;
        const size_t up_scales_bytes = expert->up.scale_bytes;
        const size_t down_q4_bytes = expert->down.data_bytes;
        const size_t down_scales_bytes = expert->down.scale_bytes;
        const size_t intermediate_bytes =
            (size_t)batch * intermediate * sizeof(float);
        const size_t input_scale_bytes =
            (size_t)batch * (((size_t)hidden + COLI_V4_MOE_FP8_BLOCK - 1) /
                             COLI_V4_MOE_FP8_BLOCK);
        const size_t down_input_scale_bytes =
            (size_t)batch * (((size_t)intermediate + COLI_V4_MOE_FP8_BLOCK - 1) /
                             COLI_V4_MOE_FP8_BLOCK);

        coli_v4_input_scratch = coli_v4_ensure_scratch(
            coli_v4_input_scratch, &coli_v4_input_scratch_capacity,
            input_bytes);
        coli_v4_qdq_scratch = coli_v4_ensure_scratch(
            coli_v4_qdq_scratch, &coli_v4_qdq_scratch_capacity, input_bytes);
        coli_v4_input_scales_scratch = coli_v4_ensure_scratch(
            coli_v4_input_scales_scratch, &coli_v4_input_scales_scratch_capacity,
            input_scale_bytes);
        coli_v4_gate_q4_scratch = coli_v4_ensure_scratch(
            coli_v4_gate_q4_scratch, &coli_v4_gate_q4_scratch_capacity,
            gate_q4_bytes);
        coli_v4_gate_scales_scratch = coli_v4_ensure_scratch(
            coli_v4_gate_scales_scratch,
            &coli_v4_gate_scales_scratch_capacity, gate_scales_bytes);
        coli_v4_up_q4_scratch = coli_v4_ensure_scratch(
            coli_v4_up_q4_scratch, &coli_v4_up_q4_scratch_capacity,
            up_q4_bytes);
        coli_v4_up_scales_scratch = coli_v4_ensure_scratch(
            coli_v4_up_scales_scratch, &coli_v4_up_scales_scratch_capacity,
            up_scales_bytes);
        coli_v4_down_q4_scratch = coli_v4_ensure_scratch(
            coli_v4_down_q4_scratch, &coli_v4_down_q4_scratch_capacity,
            down_q4_bytes);
        coli_v4_down_scales_scratch = coli_v4_ensure_scratch(
            coli_v4_down_scales_scratch,
            &coli_v4_down_scales_scratch_capacity, down_scales_bytes);
        coli_v4_gate_scratch = coli_v4_ensure_scratch(
            coli_v4_gate_scratch, &coli_v4_gate_scratch_capacity, intermediate_bytes);
        coli_v4_up_scratch = coli_v4_ensure_scratch(
            coli_v4_up_scratch, &coli_v4_up_scratch_capacity, intermediate_bytes);
        coli_v4_activated_scratch = coli_v4_ensure_scratch(
            coli_v4_activated_scratch, &coli_v4_activated_scratch_capacity, intermediate_bytes);
        coli_v4_weighted_scratch = coli_v4_ensure_scratch(
            coli_v4_weighted_scratch, &coli_v4_weighted_scratch_capacity, intermediate_bytes);
        coli_v4_down_input_scratch = coli_v4_ensure_scratch(
            coli_v4_down_input_scratch, &coli_v4_down_input_scratch_capacity, intermediate_bytes);
        coli_v4_down_input_scales_scratch = coli_v4_ensure_scratch(
            coli_v4_down_input_scales_scratch,
            &coli_v4_down_input_scales_scratch_capacity, down_input_scale_bytes);
        coli_v4_output_scratch = coli_v4_ensure_scratch(
            coli_v4_output_scratch, &coli_v4_output_scratch_capacity,
            output_bytes);
        if (!coli_v4_input_scratch || !coli_v4_qdq_scratch ||
            !coli_v4_input_scales_scratch || !coli_v4_gate_q4_scratch ||
            !coli_v4_gate_scales_scratch || !coli_v4_up_q4_scratch ||
            !coli_v4_up_scales_scratch || !coli_v4_down_q4_scratch ||
            !coli_v4_down_scales_scratch || !coli_v4_gate_scratch ||
            !coli_v4_up_scratch || !coli_v4_activated_scratch ||
            !coli_v4_weighted_scratch || !coli_v4_down_input_scratch ||
            !coli_v4_down_input_scales_scratch || !coli_v4_output_scratch ||
            !coli_v4_input_scratch.contents ||
            !coli_v4_gate_q4_scratch.contents ||
            !coli_v4_gate_scales_scratch.contents ||
            !coli_v4_up_q4_scratch.contents ||
            !coli_v4_up_scales_scratch.contents ||
            !coli_v4_down_q4_scratch.contents ||
            !coli_v4_down_scales_scratch.contents || !coli_v4_output_scratch.contents) return -1;
        memcpy(coli_v4_input_scratch.contents, inputs_gathered, input_bytes);
        double copy_began = coli_v4_profile_now();
        size_t gate_q4_offset = 0, gate_scales_offset = 0;
        size_t up_q4_offset = 0, up_scales_offset = 0;
        size_t down_q4_offset = 0, down_scales_offset = 0;
        id<MTLBuffer> gate_q4_buffer = coli_v4_weight_buffer(
            expert->gate.data, gate_q4_bytes, coli_v4_gate_q4_scratch, &gate_q4_offset);
        id<MTLBuffer> gate_scales_buffer = coli_v4_weight_buffer(
            expert->gate.scales, gate_scales_bytes, coli_v4_gate_scales_scratch, &gate_scales_offset);
        id<MTLBuffer> up_q4_buffer = coli_v4_weight_buffer(
            expert->up.data, up_q4_bytes, coli_v4_up_q4_scratch, &up_q4_offset);
        id<MTLBuffer> up_scales_buffer = coli_v4_weight_buffer(
            expert->up.scales, up_scales_bytes, coli_v4_up_scales_scratch, &up_scales_offset);
        id<MTLBuffer> down_q4_buffer = coli_v4_weight_buffer(
            expert->down.data, down_q4_bytes, coli_v4_down_q4_scratch, &down_q4_offset);
        id<MTLBuffer> down_scales_buffer = coli_v4_weight_buffer(
            expert->down.scales, down_scales_bytes, coli_v4_down_scales_scratch, &down_scales_offset);
        if (!gate_q4_buffer || !gate_scales_buffer || !up_q4_buffer ||
            !up_scales_buffer || !down_q4_buffer || !down_scales_buffer) return -1;
        if (coli_v4_metal_profile_enabled_value)
            coli_v4_profile_add(COLI_V4_PROFILE_HOST_WEIGHT_MEMCPY,
                                coli_v4_profile_now() - copy_began);

        id<MTLCommandBuffer> command_buffer = [coli_v4_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command_buffer
            ? [command_buffer computeCommandEncoder] : nil;
        if (!encoder) return -1;
        const ColiV4QdqParams input_qdq_params = { batch * hidden, COLI_V4_MOE_FP8_BLOCK };
        const ColiV4QdqParams down_qdq_params = { batch * intermediate, COLI_V4_MOE_FP8_BLOCK };
        const ColiV4MatmulDims gate_dims = { batch, hidden, intermediate,
                                              (int)gate_row_bytes, (int)gate_groups };
        const ColiV4MatmulDims down_dims = { batch, intermediate, hidden,
                                              (int)down_row_bytes, (int)down_groups };
        const ColiV4SwigluParams swiglu_params = { batch * intermediate, swiglu_limit };
        const uint32_t intermediate_count = (uint32_t)(batch * intermediate);
        const uint32_t hidden_count = (uint32_t)(batch * hidden);
        const MTLSize qdq_threads = MTLSizeMake(COLI_V4_MOE_FP8_BLOCK, 1, 1);
        const MTLSize element_threads = MTLSizeMake(COLI_V4_MOE_THREADS, 1, 1);
#define COLI_V4_PROFILE_FLUSH(stage) do { \
        if (coli_v4_metal_profile_enabled_value) { \
            [encoder endEncoding]; \
            double began = coli_v4_profile_now(); \
            [command_buffer commit]; [command_buffer waitUntilCompleted]; \
            double ended = coli_v4_profile_now(); \
            if (command_buffer.status != MTLCommandBufferStatusCompleted || command_buffer.error) return -1; \
            double gpu = command_buffer.GPUEndTime - command_buffer.GPUStartTime; \
            if (gpu > 0.0) coli_v4_profile_add(stage, gpu); \
            coli_v4_profile_add(COLI_V4_PROFILE_SUBMIT_WAIT, (ended - began) - (gpu > 0.0 ? gpu : 0.0)); \
            command_buffer = [coli_v4_queue commandBuffer]; \
            encoder = command_buffer ? [command_buffer computeCommandEncoder] : nil; \
            if (!encoder) return -1; \
        } \
    } while (0)

        [encoder setComputePipelineState:coli_v4_qdq_pipeline];
        [encoder setBuffer:coli_v4_qdq_scratch offset:0 atIndex:0];
        [encoder setBuffer:coli_v4_input_scales_scratch offset:0 atIndex:1];
        [encoder setBuffer:coli_v4_input_scratch offset:0 atIndex:2];
        [encoder setBytes:&input_qdq_params length:sizeof(input_qdq_params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(input_scale_bytes, 1, 1)
               threadsPerThreadgroup:qdq_threads];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_FP8_QDQ_INPUT);

        [encoder setComputePipelineState:matmul_pipeline];
        [encoder setBuffer:coli_v4_qdq_scratch offset:0 atIndex:0];
        [encoder setBuffer:gate_q4_buffer offset:gate_q4_offset atIndex:1];
        [encoder setBuffer:gate_scales_buffer offset:gate_scales_offset atIndex:2];
        [encoder setBuffer:coli_v4_gate_scratch offset:0 atIndex:3];
        [encoder setBytes:&gate_dims length:sizeof(gate_dims) atIndex:4];
        coli_v4_encode_matmul(encoder, matmul_geometry, (NSUInteger)intermediate, (NSUInteger)batch);
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_MATMUL_GATE);
        [encoder setComputePipelineState:matmul_pipeline];
        [encoder setBuffer:coli_v4_qdq_scratch offset:0 atIndex:0];
        [encoder setBuffer:up_q4_buffer offset:up_q4_offset atIndex:1];
        [encoder setBuffer:up_scales_buffer offset:up_scales_offset atIndex:2];
        [encoder setBuffer:coli_v4_up_scratch offset:0 atIndex:3];
        coli_v4_encode_matmul(encoder, matmul_geometry, (NSUInteger)intermediate, (NSUInteger)batch);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_MATMUL_UP);

        [encoder setComputePipelineState:coli_v4_bf16_pipeline];
        [encoder setBuffer:coli_v4_gate_scratch offset:0 atIndex:0];
        [encoder setBytes:&intermediate_count length:sizeof(intermediate_count) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)batch * intermediate, 1, 1)
               threadsPerThreadgroup:element_threads];
        [encoder setBuffer:coli_v4_up_scratch offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)batch * intermediate, 1, 1)
               threadsPerThreadgroup:element_threads];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_WEIGHTED_BF16);
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_BF16_GATE_UP);

        [encoder setComputePipelineState:coli_v4_swiglu_pipeline];
        [encoder setBuffer:coli_v4_activated_scratch offset:0 atIndex:0];
        [encoder setBuffer:coli_v4_gate_scratch offset:0 atIndex:1];
        [encoder setBuffer:coli_v4_up_scratch offset:0 atIndex:2];
        [encoder setBytes:&swiglu_params length:sizeof(swiglu_params) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)batch * intermediate, 1, 1)
               threadsPerThreadgroup:element_threads];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_SWIGLU);

        [encoder setComputePipelineState:coli_v4_weighted_pipeline];
        for (int row = 0; row < batch; row++) {
            const NSUInteger offset =
                (NSUInteger)row * intermediate * sizeof(float);
            const ColiV4WeightedBf16Params weighted_params = {
                intermediate, route_weights[row] };
            [encoder setBuffer:coli_v4_weighted_scratch offset:offset atIndex:0];
            [encoder setBuffer:coli_v4_activated_scratch offset:offset atIndex:1];
            [encoder setBytes:&weighted_params length:sizeof(weighted_params) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(intermediate, 1, 1)
                   threadsPerThreadgroup:element_threads];
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:coli_v4_qdq_pipeline];
        [encoder setBuffer:coli_v4_down_input_scratch offset:0 atIndex:0];
        [encoder setBuffer:coli_v4_down_input_scales_scratch offset:0 atIndex:1];
        [encoder setBuffer:coli_v4_weighted_scratch offset:0 atIndex:2];
        [encoder setBytes:&down_qdq_params length:sizeof(down_qdq_params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(down_input_scale_bytes, 1, 1)
               threadsPerThreadgroup:qdq_threads];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_FP8_QDQ_DOWN_INPUT);

        [encoder setComputePipelineState:matmul_pipeline];
        [encoder setBuffer:coli_v4_down_input_scratch offset:0 atIndex:0];
        [encoder setBuffer:down_q4_buffer offset:down_q4_offset atIndex:1];
        [encoder setBuffer:down_scales_buffer offset:down_scales_offset atIndex:2];
        [encoder setBuffer:coli_v4_output_scratch offset:0 atIndex:3];
        [encoder setBytes:&down_dims length:sizeof(down_dims) atIndex:4];
        coli_v4_encode_matmul(encoder, matmul_geometry, (NSUInteger)hidden, (NSUInteger)batch);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        COLI_V4_PROFILE_FLUSH(COLI_V4_PROFILE_MATMUL_DOWN);

        [encoder setComputePipelineState:coli_v4_bf16_pipeline];
        [encoder setBuffer:coli_v4_output_scratch offset:0 atIndex:0];
        [encoder setBytes:&hidden_count length:sizeof(hidden_count) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)batch * hidden, 1, 1)
               threadsPerThreadgroup:element_threads];
        [encoder endEncoding];
        double submit_began = coli_v4_profile_now();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        double submit_ended = coli_v4_profile_now();
        if (command_buffer.status != MTLCommandBufferStatusCompleted ||
            command_buffer.error) return -1;
        if (coli_v4_metal_profile_enabled_value) {
            double gpu = command_buffer.GPUEndTime - command_buffer.GPUStartTime;
            if (gpu > 0.0) coli_v4_profile_add(COLI_V4_PROFILE_BF16_OUT, gpu);
            coli_v4_profile_add(COLI_V4_PROFILE_SUBMIT_WAIT,
                                (submit_ended - submit_began) - (gpu > 0.0 ? gpu : 0.0));
            coli_v4_profile_forward();
        }
        memcpy(outs, coli_v4_output_scratch.contents, output_bytes);
        atomic_fetch_add_explicit(&coli_v4_metal_dispatch_count, 1,
                                  memory_order_relaxed);
        if (coli_v4_metal_stats_enabled_value)
            atomic_fetch_add_explicit(&coli_v4_metal_successes, 1,
                                      memory_order_relaxed);
        return 0;
#undef COLI_V4_PROFILE_FLUSH
    }
}

COLI_V4_METAL_EXTERN __attribute__((used)) int coli_v4_metal_expert_forward(
    float *out, const ColiExpertView *expert, const float *input,
    float route_weight, float swiglu_limit) {
    /* rows16 (pinned hot) experts are refused here and run on the CPU. They are ~45% of decode
     * expert calls, so this is the single largest limit on the Metal expert path -- but widening
     * it was MEASURED and it BREAKS GOLDEN (experiments E97), and the cause is NOT the matmul:
     * validation/metal/probe_rows16_parity shows coli_v4_matmul_mxfp4_ordered_hot_xcache is
     * bit-exact to the rows16 NEON reference across normal, denormal-producing and wide UE8M0
     * scale bands. Do not re-widen this on the assumption that the kernel is the problem. */
    if (expert && !coli_v4_metal_rows16_enabled_value &&
                  (expert->gate.block_rows != 1 ||
                   expert->up.block_rows != 1 ||
                   expert->down.block_rows != 1)) {
        /* This refusal is INVISIBLE in the layout attribution: it bumps the layout counter but
         * never reaches coli_v4_record_layout_mismatches, which lives in the batch entry. A run
         * therefore reports "layout=N" with every per-tensor field zero, which reads like a
         * mystery. Count it separately, and record the block_rows actually seen, so the share of
         * experts excluded here is visible rather than inferred. */
        atomic_fetch_add_explicit(&coli_v4_metal_single_rows_rejects, 1,
                                  memory_order_relaxed);
        atomic_store_explicit(&coli_v4_metal_single_rows_last,
                              (unsigned long)expert->gate.block_rows,
                              memory_order_relaxed);
        coli_v4_metal_reject(COLI_V4_METAL_REJECT_LAYOUT);
        return -1;
    }
    return coli_v4_metal_expert_forward_batch(
        out, expert, input, &route_weight, 1, swiglu_limit);
}

static id<MTLLibrary> coli_v4_load_metallib(id<MTLDevice> device,
                                             const char *path) {
    if (!path || !path[0]) return nil;
    NSString *value = [NSString stringWithUTF8String:path];
    if (![[NSFileManager defaultManager] fileExistsAtPath:value]) return nil;
    NSError *error = nil;
    return [device newLibraryWithURL:[NSURL fileURLWithPath:value] error:&error];
}

static id<MTLLibrary> coli_v4_compile_source(id<MTLDevice> device) {
    NSString *source = [[NSString alloc]
        initWithBytes:coli_v4_metal_source
               length:coli_v4_metal_source_len
             encoding:NSUTF8StringEncoding];
    if (!source) return nil;
    NSError *error = nil;
    return [device newLibraryWithSource:source options:nil error:&error];
}

COLI_V4_METAL_EXTERN int coli_v4_metal_init(const char *metallib_path) {
    if (coli_v4_device) return 1;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 0;
        const char *path = metallib_path;
        if (!path || !path[0]) path = getenv("COLI_V4_METALLIB");
        if (!path || !path[0]) path = COLI_V4_DEFAULT_METALLIB;
        id<MTLLibrary> library = coli_v4_load_metallib(device, path);
        const char *kind = "metallib";
        if (!library) {
            library = coli_v4_compile_source(device);
            kind = "source";
        }
        if (!library) return 0;
        NSError *error = nil;
        id<MTLFunction> function =
            [library newFunctionWithName:@"coli_v4_probe_primitives"];
        id<MTLComputePipelineState> pipeline = function
            ? [device newComputePipelineStateWithFunction:function error:&error]
            : nil;
        if (!pipeline) return 0;
        coli_v4_device = device;
        coli_v4_queue = [device newCommandQueue];
        coli_v4_library = library;
        coli_v4_probe_pipeline = pipeline;
        coli_v4_library_kind_value = kind;
        return coli_v4_queue ? 1 : 0;
    }
}

COLI_V4_METAL_EXTERN void coli_v4_metal_shutdown(void) {
    pthread_mutex_lock(&coli_v4_slab_buffers_mutex);
    while (coli_v4_slab_buffers) {
        ColiV4SlabBuffer *entry = coli_v4_slab_buffers;
        coli_v4_slab_buffers = entry->next;
        entry->buffer = nil;
        free(entry);
    }
    pthread_mutex_unlock(&coli_v4_slab_buffers_mutex);
    coli_v4_output_scratch = nil;
    coli_v4_down_input_scales_scratch = nil;
    coli_v4_down_input_scratch = nil;
    coli_v4_weighted_scratch = nil;
    coli_v4_activated_scratch = nil;
    coli_v4_up_scratch = nil;
    coli_v4_gate_scratch = nil;
    coli_v4_down_scales_scratch = nil;
    coli_v4_down_q4_scratch = nil;
    coli_v4_up_scales_scratch = nil;
    coli_v4_up_q4_scratch = nil;
    coli_v4_gate_scales_scratch = nil;
    coli_v4_gate_q4_scratch = nil;
    coli_v4_input_scales_scratch = nil;
    coli_v4_qdq_scratch = nil;
    coli_v4_input_scratch = nil;
    coli_v4_output_scratch_capacity = 0;
    coli_v4_down_input_scales_scratch_capacity = 0;
    coli_v4_down_input_scratch_capacity = 0;
    coli_v4_weighted_scratch_capacity = 0;
    coli_v4_activated_scratch_capacity = 0;
    coli_v4_up_scratch_capacity = 0;
    coli_v4_gate_scratch_capacity = 0;
    coli_v4_down_scales_scratch_capacity = 0;
    coli_v4_down_q4_scratch_capacity = 0;
    coli_v4_up_scales_scratch_capacity = 0;
    coli_v4_up_q4_scratch_capacity = 0;
    coli_v4_gate_scales_scratch_capacity = 0;
    coli_v4_gate_q4_scratch_capacity = 0;
    coli_v4_input_scales_scratch_capacity = 0;
    coli_v4_qdq_scratch_capacity = 0;
    coli_v4_input_scratch_capacity = 0;
    coli_v4_expert_pipeline = nil;
    coli_v4_weighted_pipeline = nil;
    coli_v4_swiglu_pipeline = nil;
    coli_v4_bf16_pipeline = nil;
    coli_v4_matmul_pipeline = nil;
    coli_v4_qdq_pipeline = nil;
    coli_v4_probe_pipeline = nil;
    coli_v4_library = nil;
    coli_v4_queue = nil;
    coli_v4_device = nil;
    coli_v4_library_kind_value = "none";
}

COLI_V4_METAL_EXTERN int coli_v4_metal_available(void) {
    return coli_v4_device != nil;
}

COLI_V4_METAL_EXTERN const char *coli_v4_metal_library_kind(void) {
    return coli_v4_library_kind_value;
}

static void coli_v4_metal_stats_report(void) {
    fprintf(stderr,
            "v4_metal_reject variant=%lu init=%lu queue=%lu dims=%lu "
            "layout=%lu pipelines=%lu ok=%lu\n",
            atomic_load_explicit(
                &coli_v4_metal_rejects[COLI_V4_METAL_REJECT_VARIANT],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_rejects[COLI_V4_METAL_REJECT_INIT],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_rejects[COLI_V4_METAL_REJECT_QUEUE],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_rejects[COLI_V4_METAL_REJECT_DIMS],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_rejects[COLI_V4_METAL_REJECT_LAYOUT],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_rejects[COLI_V4_METAL_REJECT_PIPELINES],
                memory_order_relaxed),
            atomic_load_explicit(&coli_v4_metal_successes,
                                 memory_order_relaxed));
    fprintf(stderr,
            "v4_metal_simd_exact matmuls=%lu rows16_fallbacks=%lu\n",
            atomic_load_explicit(&coli_v4_metal_simd_exact_matmuls,
                                 memory_order_relaxed),
            atomic_load_explicit(&coli_v4_metal_simd_exact_rows16_fallbacks,
                                 memory_order_relaxed));
    fprintf(stderr,
            "v4_metal_single_entry rows_rejects=%lu last_block_rows=%lu\n",
            atomic_load_explicit(&coli_v4_metal_single_rows_rejects,
                                 memory_order_relaxed),
            atomic_load_explicit(&coli_v4_metal_single_rows_last,
                                 memory_order_relaxed));
    fprintf(stderr,
            "v4_metal_layout tensor_gate=%lu tensor_up=%lu tensor_down=%lu "
            "format=%lu pointers=%lu rows=%lu columns=%lu row_bytes=%lu "
            "groups=%lu block_dims=%lu\n",
            atomic_load_explicit(
                &coli_v4_metal_layout_tensors[COLI_V4_METAL_TENSOR_GATE],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_tensors[COLI_V4_METAL_TENSOR_UP],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_tensors[COLI_V4_METAL_TENSOR_DOWN],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_FORMAT],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_POINTERS],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_ROWS],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_COLUMNS],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_ROW_BYTES],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_GROUPS],
                memory_order_relaxed),
            atomic_load_explicit(
                &coli_v4_metal_layout_fields[COLI_V4_METAL_LAYOUT_BLOCK_DIMS],
                memory_order_relaxed));
    fprintf(stderr, "v4_metal_library kind=%s\n", coli_v4_library_kind_value);
}

COLI_V4_METAL_EXTERN void coli_v4_metal_profile_report(void) {
    pthread_mutex_lock(&coli_v4_metal_profile_mutex);
    if (!coli_v4_metal_profile.forwards) {
        pthread_mutex_unlock(&coli_v4_metal_profile_mutex);
        return;
    }
    double total = 0.0;
    for (int stage = 0; stage < COLI_V4_PROFILE_STAGE_COUNT; stage++)
        total += coli_v4_metal_profile.seconds[stage];
    fprintf(stderr, "metal_profile forwards=%lu zero_copy_tensors=%lu copy_fallback_tensors=%lu\n",
            coli_v4_metal_profile.forwards,
            atomic_load_explicit(&coli_v4_metal_zero_copy_tensors, memory_order_relaxed),
            atomic_load_explicit(&coli_v4_metal_copy_fallback_tensors, memory_order_relaxed));
    for (int stage = 0; stage < COLI_V4_PROFILE_STAGE_COUNT; stage++) {
        double ms = coli_v4_metal_profile.seconds[stage] * 1000.0 /
                    (double)coli_v4_metal_profile.forwards;
        double percent = total > 0.0 ? 100.0 * coli_v4_metal_profile.seconds[stage] / total : 0.0;
        fprintf(stderr, "metal_profile stage=%s ms_per_expert=%.6f percent=%.2f\n",
                coli_v4_profile_stage_names[stage], ms, percent);
    }
    pthread_mutex_unlock(&coli_v4_metal_profile_mutex);
}
#endif
