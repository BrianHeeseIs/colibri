// Fused DeepSeek-V4 routed-expert probe: CPU seven-step chain vs one Metal dispatch.
// Synthetic fixtures only. Production case uses hidden=4096, intermediate=2048.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOURCE_PATH "c/metal/coli_v4_moe.metal"
#define KERNEL_NAME "coli_v4_moe_expert_fp4_ordered_cold"
#define FP8_BLOCK 128

static const float mx4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float mx4_scale(uint8_t scale) {
    uint32_t bits = (uint32_t)scale << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int ceil_log2_positive(float value) {
    int exponent;
    float fraction = frexpf(value, &exponent);
    return fraction == 0.5f ? exponent - 1 : exponent;
}

static float e8m0_decode(uint8_t value) {
    return value == 0xff ? NAN : ldexpf(1.0f, (int)value - 127);
}

static float e4m3_decode(uint8_t value) {
    int sign = value >> 7;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float number = exponent == 0
        ? ldexpf((float)mantissa, -9)
        : ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}

static uint8_t e4m3_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabsf(value);
    if (magnitude == 0.0f) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f)
        return (uint8_t)((negative ? 0x80 : 0) | 0x7e);

    uint8_t best;
    if (magnitude < 0.015625f) {
        float scaled = magnitude * 512.0f;
        uint8_t rounded = (uint8_t)scaled;
        float fraction = scaled - (float)rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) rounded++;
        best = rounded;
    } else {
        uint32_t bits;
        memcpy(&bits, &magnitude, sizeof(bits));
        int exponent = (int)((bits >> 23) & 0xff) - 127;
        uint32_t significand = 0x800000u | (bits & 0x7fffffu);
        uint32_t rounded = significand >> 20;
        uint32_t remainder = significand & 0xfffffu;
        if (remainder > 0x80000u ||
            (remainder == 0x80000u && (rounded & 1u))) rounded++;
        if (rounded == 16u) {
            rounded = 8u;
            exponent++;
        }
        best = (uint8_t)((exponent + 7) * 8 + (int)rounded - 8);
    }
    return (uint8_t)(best | (negative ? 0x80 : 0));
}

static float bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint32_t tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float sigmoid_stable(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

static void cpu_qdq(float *output, uint8_t *scales, const float *input,
                    int length, int block_size) {
    for (int base = 0; base < length; base += block_size) {
        int count = length - base < block_size ? length - base : block_size;
        float maximum = 0.0f;
        for (int index = 0; index < count; index++)
            maximum = fmaxf(maximum, fabsf(input[base + index]));
        maximum = fmaxf(maximum, 1e-4f);
        int scale_exponent = ceil_log2_positive(maximum / 448.0f);
        if (scale_exponent < -127) scale_exponent = -127;
        if (scale_exponent > 127) scale_exponent = 127;
        uint8_t encoded_scale = (uint8_t)(scale_exponent + 127);
        float scale = e8m0_decode(encoded_scale);
        scales[base / block_size] = encoded_scale;
        for (int index = 0; index < count; index++) {
            float normalized = fmaxf(
                -448.0f, fminf(448.0f, input[base + index] / scale));
            output[base + index] =
                e4m3_decode(e4m3_encode(normalized)) * scale;
        }
    }
}

static void cpu_matmul(float *output, const float *input, const uint8_t *q4,
                       const uint8_t *e8s, int input_size, int output_size) {
    int row_bytes = (input_size + 1) / 2;
    int groups = (input_size + 31) / 32;
    for (int row = 0; row < output_size; row++) {
        const uint8_t *weights = q4 + (int64_t)row * row_bytes;
        const uint8_t *scales = e8s + (int64_t)row * groups;
        float accumulator = 0.0f;
        for (int group = 0; group < groups; group++) {
            int base = group * 32;
            int group_length = 32;
            if (base + group_length > input_size)
                group_length = input_size - base;
            float scale = mx4_scale(scales[group]);
            float group_accumulator = 0.0f;
            for (int column = base; column < base + group_length; column += 2) {
                uint8_t packed = weights[column >> 1];
                group_accumulator += input[column] * mx4_lut[packed & 0x0f];
                if (column + 1 < base + group_length)
                    group_accumulator += input[column + 1] * mx4_lut[packed >> 4];
            }
            accumulator += group_accumulator * scale;
        }
        output[row] = accumulator;
    }
}

static int cpu_expert(float *output, const float *input,
                      const uint8_t *gate_q4, const uint8_t *gate_scales,
                      const uint8_t *up_q4, const uint8_t *up_scales,
                      const uint8_t *down_q4, const uint8_t *down_scales,
                      int hidden, int intermediate, float route_weight,
                      float swiglu_limit) {
    float *qdq = malloc((size_t)hidden * sizeof(*qdq));
    float *gate = malloc((size_t)intermediate * sizeof(*gate));
    float *up = malloc((size_t)intermediate * sizeof(*up));
    float *activated = malloc((size_t)intermediate * sizeof(*activated));
    float *weighted = malloc((size_t)intermediate * sizeof(*weighted));
    float *down_input = malloc((size_t)intermediate * sizeof(*down_input));
    uint8_t *input_qdq_scales = malloc((size_t)(hidden + FP8_BLOCK - 1) / FP8_BLOCK);
    uint8_t *down_qdq_scales = malloc((size_t)(intermediate + FP8_BLOCK - 1) / FP8_BLOCK);
    if (!qdq || !gate || !up || !activated || !weighted || !down_input ||
        !input_qdq_scales || !down_qdq_scales) return -1;

    cpu_qdq(qdq, input_qdq_scales, input, hidden, FP8_BLOCK);
    cpu_matmul(gate, qdq, gate_q4, gate_scales, hidden, intermediate);
    cpu_matmul(up, qdq, up_q4, up_scales, hidden, intermediate);
    for (int index = 0; index < intermediate; index++) {
        gate[index] = bf16_round(gate[index]);
        up[index] = bf16_round(up[index]);
    }
    for (int index = 0; index < intermediate; index++) {
        float gate_value = gate[index];
        float up_value = up[index];
        if (swiglu_limit > 0.0f) {
            gate_value = fminf(gate_value, swiglu_limit);
            up_value = fmaxf(-swiglu_limit, fminf(up_value, swiglu_limit));
        }
        activated[index] =
            (gate_value * sigmoid_stable(gate_value)) * up_value;
        weighted[index] = bf16_round(activated[index] * route_weight);
    }
    cpu_qdq(down_input, down_qdq_scales, weighted, intermediate, FP8_BLOCK);
    cpu_matmul(output, down_input, down_q4, down_scales,
               intermediate, hidden);
    for (int index = 0; index < hidden; index++)
        output[index] = bf16_round(output[index]);

    free(down_qdq_scales);
    free(input_qdq_scales);
    free(down_input);
    free(weighted);
    free(activated);
    free(up);
    free(gate);
    free(qdq);
    return 0;
}

typedef struct {
    int hidden;
    int intermediate;
    int gate_row_bytes;
    int gate_groups;
    int down_row_bytes;
    int down_groups;
    int fp8_block;
    float route_weight;
    float swiglu_limit;
} MoeParams;

typedef struct {
    uint64_t state;
} FixtureRng;

static uint32_t rng_next(FixtureRng *rng) {
    rng->state = rng->state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return (uint32_t)(rng->state >> 32);
}

static float fixture_input(FixtureRng *rng) {
    return ((float)(rng_next(rng) % 4000) - 2000.0f) / 311.0f;
}

static void fill_weights(FixtureRng *rng, uint8_t *q4, size_t q4_count,
                         uint8_t *scales, size_t scale_count) {
    for (size_t index = 0; index < q4_count; index++)
        q4[index] = (uint8_t)rng_next(rng);
    for (size_t index = 0; index < scale_count; index++)
        scales[index] = (uint8_t)(112 + rng_next(rng) % 22);
}

static id<MTLBuffer> make_buffer(id<MTLDevice> device, const void *bytes,
                                 size_t length) {
    if (bytes)
        return [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
    return [device newBufferWithLength:length options:MTLResourceStorageModeShared];
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int compare_and_print(const char *label, uint64_t seed,
                             const float *cpu, const float *gpu, int hidden,
                             int intermediate) {
    int differences = 0;
    uint32_t max_ulp = 0;
    double max_absolute = 0.0;
    for (int index = 0; index < hidden; index++) {
        uint32_t cpu_bits = float_bits(cpu[index]);
        uint32_t gpu_bits = float_bits(gpu[index]);
        if (cpu_bits != gpu_bits) {
            differences++;
            uint32_t ulp = cpu_bits > gpu_bits ? cpu_bits - gpu_bits : gpu_bits - cpu_bits;
            if (ulp > max_ulp) max_ulp = ulp;
            double absolute = fabs((double)cpu[index] - (double)gpu[index]);
            if (absolute > max_absolute) max_absolute = absolute;
        }
    }

    printf("  HEX label=%s seed=%llu\n", label, (unsigned long long)seed);
    printf("    index  cpu_hex     gpu_hex     xor         cpu_value        gpu_value\n");
    int shown = hidden < 8 ? hidden : 8;
    for (int index = 0; index < shown; index++) {
        uint32_t cpu_bits = float_bits(cpu[index]);
        uint32_t gpu_bits = float_bits(gpu[index]);
        printf("    %5d  0x%08x  0x%08x  0x%08x  %+.8e  %+.8e\n",
               index, cpu_bits, gpu_bits, cpu_bits ^ gpu_bits,
               cpu[index], gpu[index]);
    }
    if (differences) {
        int mismatch_shown = 0;
        printf("    first mismatches beyond table:\n");
        for (int index = shown; index < hidden && mismatch_shown < 8; index++) {
            uint32_t cpu_bits = float_bits(cpu[index]);
            uint32_t gpu_bits = float_bits(gpu[index]);
            if (cpu_bits == gpu_bits) continue;
            printf("    %5d  0x%08x  0x%08x  0x%08x  %+.8e  %+.8e\n",
                   index, cpu_bits, gpu_bits, cpu_bits ^ gpu_bits,
                   cpu[index], gpu[index]);
            mismatch_shown++;
        }
    }
    printf("  RESULT label=%s seed=%llu reductions=gate/up:%d,down:%d : "
           "%s diff=%d/%d "
           "maxULP=%u maxAbs=%.3e\n",
           label, (unsigned long long)seed, hidden, intermediate,
           differences ? "MISMATCH" : "BIT-EXACT", differences, hidden,
           max_ulp, max_absolute);
    return differences != 0;
}

static int run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pipeline, const char *label,
                    int hidden, int intermediate, uint64_t seed) {
    int gate_row_bytes = (hidden + 1) / 2;
    int gate_groups = (hidden + 31) / 32;
    int down_row_bytes = (intermediate + 1) / 2;
    int down_groups = (intermediate + 31) / 32;
    size_t gate_q4_count = (size_t)intermediate * gate_row_bytes;
    size_t gate_scale_count = (size_t)intermediate * gate_groups;
    size_t down_q4_count = (size_t)hidden * down_row_bytes;
    size_t down_scale_count = (size_t)hidden * down_groups;

    float *input = malloc((size_t)hidden * sizeof(*input));
    uint8_t *gate_q4 = malloc(gate_q4_count);
    uint8_t *gate_scales = malloc(gate_scale_count);
    uint8_t *up_q4 = malloc(gate_q4_count);
    uint8_t *up_scales = malloc(gate_scale_count);
    uint8_t *down_q4 = malloc(down_q4_count);
    uint8_t *down_scales = malloc(down_scale_count);
    float *cpu_output = malloc((size_t)hidden * sizeof(*cpu_output));
    if (!input || !gate_q4 || !gate_scales || !up_q4 || !up_scales ||
        !down_q4 || !down_scales || !cpu_output) {
        fprintf(stderr, "fixture allocation failed for %s\n", label);
        return 1;
    }

    FixtureRng rng = { seed };
    for (int index = 0; index < hidden; index++) input[index] = fixture_input(&rng);
    fill_weights(&rng, gate_q4, gate_q4_count, gate_scales, gate_scale_count);
    fill_weights(&rng, up_q4, gate_q4_count, up_scales, gate_scale_count);
    fill_weights(&rng, down_q4, down_q4_count, down_scales, down_scale_count);
    float route_weight = 0.25f + (float)(rng_next(&rng) % 1000) / 1000.0f;
    float swiglu_limit = (rng_next(&rng) & 1u) ? 10.0f : 0.0f;

    if (cpu_expert(cpu_output, input, gate_q4, gate_scales, up_q4, up_scales,
                   down_q4, down_scales, hidden, intermediate,
                   route_weight, swiglu_limit) != 0) {
        fprintf(stderr, "CPU reference allocation failed for %s\n", label);
        return 1;
    }

    MoeParams params = {
        hidden, intermediate, gate_row_bytes, gate_groups,
        down_row_bytes, down_groups, FP8_BLOCK, route_weight, swiglu_limit,
    };
    id<MTLBuffer> input_buffer = make_buffer(device, input, (size_t)hidden * sizeof(*input));
    id<MTLBuffer> gate_q4_buffer = make_buffer(device, gate_q4, gate_q4_count);
    id<MTLBuffer> gate_scale_buffer = make_buffer(device, gate_scales, gate_scale_count);
    id<MTLBuffer> up_q4_buffer = make_buffer(device, up_q4, gate_q4_count);
    id<MTLBuffer> up_scale_buffer = make_buffer(device, up_scales, gate_scale_count);
    id<MTLBuffer> down_q4_buffer = make_buffer(device, down_q4, down_q4_count);
    id<MTLBuffer> down_scale_buffer = make_buffer(device, down_scales, down_scale_count);
    id<MTLBuffer> output_buffer = make_buffer(device, NULL, (size_t)hidden * sizeof(float));
    id<MTLBuffer> params_buffer = make_buffer(device, &params, sizeof(params));

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:gate_q4_buffer offset:0 atIndex:1];
    [encoder setBuffer:gate_scale_buffer offset:0 atIndex:2];
    [encoder setBuffer:up_q4_buffer offset:0 atIndex:3];
    [encoder setBuffer:up_scale_buffer offset:0 atIndex:4];
    [encoder setBuffer:down_q4_buffer offset:0 atIndex:5];
    [encoder setBuffer:down_scale_buffer offset:0 atIndex:6];
    [encoder setBuffer:output_buffer offset:0 atIndex:7];
    [encoder setBuffer:params_buffer offset:0 atIndex:8];
    [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "GPU dispatch failed for %s: %s\n", label,
                command_buffer.error.localizedDescription.UTF8String);
        return 1;
    }

    printf("CASE label=%s hidden=%d intermediate=%d gate_ng=%d down_ng=%d "
           "fp8_block=%d route_weight=%.3f swiglu_limit=%.1f\n",
           label, hidden, intermediate, gate_groups, down_groups,
           FP8_BLOCK, route_weight, swiglu_limit);
    int failed = compare_and_print(label, seed, cpu_output,
                                   (const float *)output_buffer.contents,
                                   hidden, intermediate);

    free(cpu_output);
    free(down_scales);
    free(down_q4);
    free(up_scales);
    free(up_q4);
    free(gate_scales);
    free(gate_q4);
    free(input);
    return failed;
}

static char *read_source(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long file_length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *source = malloc((size_t)file_length + 1);
    if (!source) {
        fclose(file);
        return NULL;
    }
    size_t read_length = fread(source, 1, (size_t)file_length, file);
    fclose(file);
    source[read_length] = '\0';
    *length = read_length;
    return source;
}

int main(void) {
    @autoreleasepool {
        size_t source_length = 0;
        char *source = read_source(SOURCE_PATH, &source_length);
        if (!source) {
            printf("RED: kernel source %s missing; entry %s unavailable\n",
                   SOURCE_PATH, KERNEL_NAME);
            return 2;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal device unavailable\n");
            free(source);
            return 3;
        }
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSString *source_string = [[NSString alloc]
            initWithBytes:source length:source_length encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [device newLibraryWithSource:source_string
                                                     options:options error:&error];
        free(source);
        if (!library) {
            printf("RED: Metal source compile failed:\n%s\n",
                   error.localizedDescription.UTF8String);
            return 4;
        }
        id<MTLFunction> function = [library newFunctionWithName:@KERNEL_NAME];
        if (!function) {
            printf("RED: kernel entry %s missing\n", KERNEL_NAME);
            return 5;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            printf("RED: pipeline creation failed for %s:\n%s\n",
                   KERNEL_NAME, error.localizedDescription.UTF8String);
            return 6;
        }
        printf("DEVICE name=%s threadWidth=%lu maxThreads=%lu maxThreadgroupMemory=%lu "
               "staticThreadgroupMemory=%lu\n",
               device.name.UTF8String,
               (unsigned long)pipeline.threadExecutionWidth,
               (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
               (unsigned long)device.maxThreadgroupMemoryLength,
               (unsigned long)pipeline.staticThreadgroupMemoryLength);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256 ||
            pipeline.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength) {
            fprintf(stderr, "pipeline cannot support required launch geometry\n");
            return 7;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        int failed = 0;
        failed |= run_case(device, queue, pipeline, "small", 64, 128,
                           UINT64_C(24601));
        failed |= run_case(device, queue, pipeline, "real-a", 4096, 2048,
                           UINT64_C(1));
        failed |= run_case(device, queue, pipeline, "real-b", 4096, 2048,
                           UINT64_C(24601));
        failed |= run_case(device, queue, pipeline, "real-c", 4096, 2048,
                           UINT64_C(0xc011b1));
        printf("SUMMARY %s cases=4 real_cases=3 real_reduction=4096/2048\n",
               failed ? "FAIL" : "PASS BIT-EXACT");
        return failed;
    }
}
