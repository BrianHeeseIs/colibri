#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../c/native_quant_fp4_rows16.h"

enum {
    WARMUP_ITERATIONS = 32,
    TIMED_ITERATIONS = 21,
    MAX_BATCH = 16,
    THREADGROUP_WIDTH = 256,
};

typedef struct {
    int S;
    int I;
    int O;
    int rb;
    int ng;
} ColiV4MatmulDims;

typedef enum {
    WORK_GATE_UP,
    WORK_DOWN,
} WorkKind;

typedef struct {
    const char *name;
    WorkKind kind;
    int outputs;
    int inputs;
} Shape;

typedef struct {
    double wall_seconds;
    double active_seconds;
} GpuTiming;

float coli_fp8_minprod = 3.4e38f;
int coli_fp8_minprod_enabled = 0;

static volatile double output_sink;

static double now_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer /
           (double)timebase.denom / 1e9;
}

static uint64_t random_next(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static void fill_weight_bytes(uint8_t *data, size_t count, uint64_t *state) {
    for (size_t index = 0; index < count; ++index)
        data[index] = (uint8_t)(random_next(state) >> 56);
}

static void fill_scale_bytes(uint8_t *scales, size_t count, uint64_t *state) {
    for (size_t index = 0; index < count; ++index)
        scales[index] = (uint8_t)(124u + random_next(state) % 7u);
}

static void fill_activations(float *values, size_t count, uint64_t *state) {
    static const int exponents[] = {-5, -2, 1, 4, 7};
    for (size_t index = 0; index < count; ++index) {
        uint64_t bits = random_next(state);
        float fraction = 0.5f + (float)((bits >> 16) & 0xffffu) / 131072.0f;
        float magnitude = ldexpf(fraction, exponents[bits % 5u]);
        values[index] = (bits & UINT64_C(0x8000000000000000))
                            ? -magnitude
                            : magnitude;
    }
}

static ColiTensorView make_rows16_view(const void *data, const void *scales,
                                       int outputs, int inputs) {
    ColiTensorView view = {
        .format = COLI_TENSOR_FP4_NATIVE_BLOCK,
        .scale_format = COLI_SCALE_UE8M0,
        .data = data,
        .scales = scales,
        .data_bytes = (size_t)outputs * (size_t)inputs / 2u,
        .scale_bytes = (size_t)outputs * (size_t)inputs / 32u,
        .rows = outputs,
        .columns = inputs,
        .block_rows = 16,
        .block_columns = 32,
    };
    return view;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(double samples[TIMED_ITERATIONS]) {
    double sorted[TIMED_ITERATIONS];
    memcpy(sorted, samples, sizeof(sorted));
    qsort(sorted, TIMED_ITERATIONS, sizeof(sorted[0]), compare_double);
    return sorted[TIMED_ITERATIONS / 2];
}

static void consume_outputs(const float *first, const float *second,
                            size_t count) {
    size_t stride = count / 17u + 1u;
    double sum = 0.0;
    for (size_t index = 0; index < count; index += stride) {
        sum += first[index];
        if (second) sum += second[index];
    }
    output_sink += sum;
}

static int cpu_dispatch(WorkKind kind, int batch, int inputs, int outputs,
                        const ColiTensorView *first_view,
                        const ColiTensorView *second_view,
                        const float *activations, float *first_output,
                        float *second_output, double *seconds) {
    double began = now_seconds();
    for (int item = 0; item < batch; ++item) {
        const float *input = activations + (size_t)item * (size_t)inputs;
        float *output_a = first_output + (size_t)item * (size_t)outputs;
        int result;
        if (kind == WORK_GATE_UP) {
            float *output_b = second_output + (size_t)item * (size_t)outputs;
            result = coli_fp4_dual_matvec_rows16_v10(
                output_a, output_b, first_view, second_view, input);
        } else {
            result = coli_fp4_matvec_rows16_v10(output_a, first_view, input);
        }
        if (result != 0) return -1;
    }
    *seconds = now_seconds() - began;
    return 0;
}

static int encode_matmul(id<MTLComputeCommandEncoder> encoder,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> activations, id<MTLBuffer> data,
                         id<MTLBuffer> scales, id<MTLBuffer> output,
                         const ColiV4MatmulDims *dims) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:activations offset:0 atIndex:0];
    [encoder setBuffer:data offset:0 atIndex:1];
    [encoder setBuffer:scales offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:dims length:sizeof(*dims) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)dims->O,
                                        (NSUInteger)dims->S, 1)
         threadsPerThreadgroup:MTLSizeMake(THREADGROUP_WIDTH, 1, 1)];
    return 0;
}

static int gpu_dispatch(id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> pipeline, WorkKind kind,
                        int batch, int inputs, int outputs,
                        id<MTLBuffer> activations, id<MTLBuffer> first_data,
                        id<MTLBuffer> first_scales, id<MTLBuffer> first_output,
                        id<MTLBuffer> second_data,
                        id<MTLBuffer> second_scales,
                        id<MTLBuffer> second_output, GpuTiming *timing) {
    ColiV4MatmulDims dims = {
        .S = batch,
        .I = inputs,
        .O = outputs,
        .rb = inputs / 2,
        .ng = inputs / 32,
    };
    double began = now_seconds();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        command_buffer ? [command_buffer computeCommandEncoder] : nil;
    if (!encoder) return -1;
    encode_matmul(encoder, pipeline, activations, first_data, first_scales,
                  first_output, &dims);
    if (kind == WORK_GATE_UP)
        encode_matmul(encoder, pipeline, activations, second_data,
                      second_scales, second_output, &dims);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    timing->wall_seconds = now_seconds() - began;
    timing->active_seconds = command_buffer.GPUEndTime -
                             command_buffer.GPUStartTime;
    if (command_buffer.status != MTLCommandBufferStatusCompleted ||
        command_buffer.error)
        return -1;
    return timing->active_seconds > 0.0 ? 0 : -1;
}

static int verify_rows16_guard(WorkKind kind, int inputs, int outputs,
                               const ColiTensorView *first_view,
                               const ColiTensorView *second_view,
                               const float *activations, float *first_output,
                               float *second_output) {
    ColiTensorView invalid_first = *first_view;
    ColiTensorView invalid_second;
    invalid_first.block_rows = 1;
    int rejected;
    if (kind == WORK_GATE_UP) {
        invalid_second = *second_view;
        invalid_second.block_rows = 1;
        rejected = coli_fp4_dual_matvec_rows16_v10(
            first_output, second_output, &invalid_first, &invalid_second,
            activations);
    } else {
        rejected = coli_fp4_matvec_rows16_v10(
            first_output, &invalid_first, activations);
    }
    if (rejected != -1) return -1;
    double elapsed;
    return cpu_dispatch(kind, 1, inputs, outputs, first_view, second_view,
                        activations, first_output, second_output, &elapsed);
}

static int run_shape(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline,
                     const Shape *shape) {
    int inputs = shape->inputs;
    int outputs = shape->outputs;
    size_t data_bytes = (size_t)outputs * (size_t)inputs / 2u;
    size_t scale_bytes = (size_t)outputs * (size_t)inputs / 32u;
    size_t activation_bytes = (size_t)MAX_BATCH * (size_t)inputs * sizeof(float);
    size_t output_bytes = (size_t)MAX_BATCH * (size_t)outputs * sizeof(float);

    id<MTLBuffer> first_data = [device newBufferWithLength:data_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> first_scales = [device newBufferWithLength:scale_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> second_data = shape->kind == WORK_GATE_UP
                                    ? [device newBufferWithLength:data_bytes
                                                         options:MTLResourceStorageModeShared]
                                    : nil;
    id<MTLBuffer> second_scales = shape->kind == WORK_GATE_UP
                                      ? [device newBufferWithLength:scale_bytes
                                                           options:MTLResourceStorageModeShared]
                                      : nil;
    id<MTLBuffer> activations = [device newBufferWithLength:activation_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> cpu_first = [device newBufferWithLength:output_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> cpu_second = shape->kind == WORK_GATE_UP
                                   ? [device newBufferWithLength:output_bytes
                                                        options:MTLResourceStorageModeShared]
                                   : nil;
    id<MTLBuffer> gpu_first = [device newBufferWithLength:output_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> gpu_second = shape->kind == WORK_GATE_UP
                                   ? [device newBufferWithLength:output_bytes
                                                        options:MTLResourceStorageModeShared]
                                   : nil;
    if (!first_data || !first_scales || !activations || !cpu_first ||
        !gpu_first || (shape->kind == WORK_GATE_UP &&
                       (!second_data || !second_scales || !cpu_second ||
                        !gpu_second))) {
        fprintf(stderr, "buffer allocation failed for %s\n", shape->name);
        return -1;
    }

    uint64_t state = UINT64_C(0x8b8b8b8b1234567) ^
                     ((uint64_t)inputs << 32) ^ (uint64_t)outputs;
    fill_weight_bytes(first_data.contents, data_bytes, &state);
    fill_scale_bytes(first_scales.contents, scale_bytes, &state);
    if (shape->kind == WORK_GATE_UP) {
        fill_weight_bytes(second_data.contents, data_bytes, &state);
        fill_scale_bytes(second_scales.contents, scale_bytes, &state);
    }
    fill_activations(activations.contents,
                     (size_t)MAX_BATCH * (size_t)inputs, &state);

    ColiTensorView first_view = make_rows16_view(
        first_data.contents, first_scales.contents, outputs, inputs);
    ColiTensorView second_view = make_rows16_view(
        second_data.contents, second_scales.contents, outputs, inputs);
    if (verify_rows16_guard(
            shape->kind, inputs, outputs, &first_view, &second_view,
            activations.contents, cpu_first.contents, cpu_second.contents)) {
        fprintf(stderr, "rows16 guard verification failed for %s\n", shape->name);
        return -1;
    }

    printf("\n%s: O=%d I=%d (%s)\n", shape->name, outputs, inputs,
           shape->kind == WORK_GATE_UP
               ? "CPU fused dual, GPU two dispatches"
               : "CPU single, GPU one dispatch");
    printf("  %-4s | %12s | %13s | %14s\n",
           "S", "CPU ms", "GPU ms (wall)", "ratio CPU/GPU");
    printf("  -----+--------------+---------------+----------------\n");

    static const int batches[] = {1, 2, 4, 8, 16};
    for (size_t batch_index = 0;
         batch_index < sizeof(batches) / sizeof(batches[0]);
         ++batch_index) {
        int batch = batches[batch_index];
        for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
            double cpu_seconds;
            if (cpu_dispatch(shape->kind, batch, inputs, outputs, &first_view,
                             &second_view, activations.contents,
                             cpu_first.contents, cpu_second.contents,
                             &cpu_seconds)) {
                fprintf(stderr, "CPU warmup failed for %s S=%d\n",
                        shape->name, batch);
                return -1;
            }
        }

        double cpu_samples[TIMED_ITERATIONS];
        double gpu_wall_samples[TIMED_ITERATIONS];
        double gpu_active_samples[TIMED_ITERATIONS];
        for (int iteration = 0; iteration < TIMED_ITERATIONS; ++iteration) {
            if (cpu_dispatch(shape->kind, batch, inputs, outputs, &first_view,
                             &second_view, activations.contents,
                             cpu_first.contents, cpu_second.contents,
                             &cpu_samples[iteration])) {
                fprintf(stderr, "timed CPU dispatch failed for %s S=%d\n",
                        shape->name, batch);
                return -1;
            }
        }
        usleep(300000);
        for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
            GpuTiming gpu_timing;
            if (gpu_dispatch(queue, pipeline, shape->kind, batch, inputs,
                             outputs, activations, first_data, first_scales,
                             gpu_first, second_data, second_scales, gpu_second,
                             &gpu_timing)) {
                fprintf(stderr, "GPU warmup failed for %s S=%d\n",
                        shape->name, batch);
                return -1;
            }
        }
        for (int iteration = 0; iteration < TIMED_ITERATIONS; ++iteration) {
            GpuTiming gpu_timing;
            if (gpu_dispatch(queue, pipeline, shape->kind, batch, inputs,
                             outputs, activations, first_data, first_scales,
                             gpu_first, second_data, second_scales, gpu_second,
                             &gpu_timing)) {
                fprintf(stderr, "timed GPU dispatch failed for %s S=%d\n",
                        shape->name, batch);
                return -1;
            }
            gpu_wall_samples[iteration] = gpu_timing.wall_seconds;
            gpu_active_samples[iteration] = gpu_timing.active_seconds;
        }
        consume_outputs(cpu_first.contents, cpu_second.contents,
                        (size_t)batch * (size_t)outputs);
        consume_outputs(gpu_first.contents, gpu_second.contents,
                        (size_t)batch * (size_t)outputs);

        double cpu_median_ms = median(cpu_samples) * 1e3;
        double gpu_wall_median_ms = median(gpu_wall_samples) * 1e3;
        double gpu_active_median_ms = median(gpu_active_samples) * 1e3;
        printf("  %-4d | %12.3f | %13.3f | %14.3fx",
               batch, cpu_median_ms, gpu_wall_median_ms,
               cpu_median_ms / gpu_wall_median_ms);
        printf("  (GPU active %.3f ms)\n", gpu_active_median_ms);
    }
    return 0;
}

static NSString *read_metal_source(NSString *path) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:path
                                                  encoding:NSUTF8StringEncoding
                                                     error:&error];
    if (!source)
        fprintf(stderr, "cannot read Metal source %s: %s\n",
                path.UTF8String, error.localizedDescription.UTF8String);
    return source;
}

static void print_usage(const char *program) {
    printf("usage: %s [c/metal/coli_v4_matmul.metal]\n", program);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc == 2 && strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (argc > 2) {
            print_usage(argv[0]);
            return 2;
        }
#if !defined(__aarch64__) || !defined(COLI_FP4_ROWS16_KERNEL)
#error "This probe requires the arm64 rows16 kernel"
#endif
        NSString *source_path = argc == 2
                                    ? [NSString stringWithUTF8String:argv[1]]
                                    : @"c/metal/coli_v4_matmul.metal";
        NSString *source = read_metal_source(source_path);
        if (!source) return 3;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device\n");
            return 3;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:options
                                                        error:&error];
        if (!library) {
            fprintf(stderr, "Metal compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 3;
        }
        id<MTLFunction> function = [library
            newFunctionWithName:@"coli_v4_matmul_mxfp4_ordered_xcache"];
        id<MTLComputePipelineState> pipeline = function
            ? [device newComputePipelineStateWithFunction:function error:&error]
            : nil;
        if (!pipeline ||
            pipeline.maxTotalThreadsPerThreadgroup < THREADGROUP_WIDTH) {
            fprintf(stderr, "ordered_xcache pipeline unavailable: %s\n",
                    error.localizedDescription.UTF8String);
            return 3;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "cannot create Metal command queue\n");
            return 3;
        }

        printf("MXFP4 live-kernel S-scaling gate\n");
        printf("device=%s cpu_threads=%d warmups=%d timed=%d statistic=median\n",
               device.name.UTF8String, omp_get_max_threads(),
               WARMUP_ITERATIONS, TIMED_ITERATIONS);
        printf("cpu=coli_fp4_{dual_,}matvec_rows16_v10 arm64 NEON block_rows=16\n");
        printf("cpu_guard=block_rows=1 rejected, block_rows=16 accepted\n");
        printf("gpu=coli_v4_matmul_mxfp4_ordered_xcache mathMode=Safe fp64=none\n");
        printf("buffers=MTLResourceStorageModeShared; CPU pointers are the same MTLBuffer.contents used by GPU\n");
        printf("weights=random e2m1 bytes; scales=UE8M0 codes 124..130; activations about 2^-6..2^7\n");
        printf("GPU wall includes command-buffer creation, encoding, commit, and waitUntilCompleted\n");
        printf("CPU and GPU phases are separated by 300 ms so OpenMP workers do not contend with Metal\n");

        static const Shape shapes[] = {
            {"gate/up shape", WORK_GATE_UP, 2048, 4096},
            {"down shape", WORK_DOWN, 4096, 2048},
        };
        for (size_t index = 0; index < sizeof(shapes) / sizeof(shapes[0]);
             ++index)
            if (run_shape(device, queue, pipeline, &shapes[index])) return 4;

        printf("\noutput_sink=%.9e\n", output_sink);
        return isfinite(output_sink) ? 0 : 5;
    }
}
