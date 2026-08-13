/* allow: SIZE_OK — single translation unit keeps pre-Metal safety gate auditable.
 * DeepSeek-V4 Metal benchmark harness.
 *
 * This file deliberately does not know how to open the model or encode the production
 * kernels.  A production benchmark adapter supplies the weak functions declared below.
 * Keeping that seam explicit lets this harness compile before backend wiring lands while
 * preserving one timing implementation for every shipping variant.
 *
 * Safety contract: no Metal object is created and no model/backend hook is called until
 * COLI_V4_METAL_BENCH_APPROVED has the exact approval phrase below.  Listing variants is
 * always safe.  Do not weaken this gate to a generic truthy environment value.
 *
 *   clang -fobjc-arc -O2 -Wall -Wextra -framework Metal -framework Foundation \
 *     validation/metal/bench_v4.m -o validation/metal/bench_v4
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define APPROVAL_ENV "COLI_V4_METAL_BENCH_APPROVED"
#define APPROVAL_PHRASE "USER_CONFIRMED_REAL_MODEL_BENCHMARK"

typedef NS_ENUM(unsigned, ColiV4BenchVariant) {
    COLI_V4_BENCH_CPU_BASELINE = 0,
    COLI_V4_BENCH_ORDERED_COLD,
    COLI_V4_BENCH_ORDERED_HOT,
    COLI_V4_BENCH_SIMD_COLD,
    COLI_V4_BENCH_SIMD_HOT,
    COLI_V4_BENCH_VARIANT_COUNT
};

typedef struct {
    ColiV4BenchVariant variant;
    const char *name;
    const char *reduction;
    const char *layout;
    const char *kernel_name;
    int uses_gpu;
} VariantDescriptor;

static const VariantDescriptor VARIANTS[] = {
    {COLI_V4_BENCH_CPU_BASELINE, "cpu-baseline", "CPU reference", "native", NULL, 0},
    {COLI_V4_BENCH_ORDERED_COLD, "ordered-cold", "ORDERED", "COLD row-major",
     "coli_v4_moe_expert_fp4_ordered_cold", 1},
    {COLI_V4_BENCH_ORDERED_HOT, "ordered-hot", "ORDERED", "HOT rows16",
     "coli_v4_moe_expert_fp4_ordered_hot", 1},
    {COLI_V4_BENCH_SIMD_COLD, "simd-cold", "SIMD", "COLD row-major",
     "coli_v4_moe_expert_fp4_simd_cold", 1},
    {COLI_V4_BENCH_SIMD_HOT, "simd-hot", "SIMD", "HOT rows16",
     "coli_v4_moe_expert_fp4_simd_hot", 1},
};

_Static_assert(sizeof(VARIANTS) / sizeof(VARIANTS[0]) == COLI_V4_BENCH_VARIANT_COUNT,
               "variant descriptor matrix must cover enum");

/* Adapter contract:
 * - open receives the real model path and benchmark device; it owns model/buffer setup.
 * - cpu_dispatch completes one CPU reference operation synchronously.
 * - encode_gpu receives the locked MSL entry name and returns an encoded, uncommitted
 *   command buffer. This harness commits and waits so wall and GPU timestamps always cover
 *   the same dispatch boundary.
 * - each dispatch must represent the same model operation and inputs across variants.
 */
__attribute__((weak)) unsigned coli_v4_bench_adapter_version(void) { return 0; }

__attribute__((weak))
int coli_v4_bench_model_open(const char *model_path, id<MTLDevice> device,
                             void **context, char *error, size_t error_size) {
    (void)model_path; (void)device; (void)context;
    snprintf(error, error_size, "benchmark adapter not linked");
    return -1;
}

__attribute__((weak))
void coli_v4_bench_model_close(void *context) { (void)context; }

__attribute__((weak))
int coli_v4_bench_cpu_dispatch(void *context, char *error, size_t error_size) {
    (void)context;
    snprintf(error, error_size, "benchmark adapter not linked");
    return -1;
}

__attribute__((weak))
int coli_v4_bench_encode_gpu(void *context, ColiV4BenchVariant variant,
                             const char *kernel_name,
                             id<MTLCommandQueue> queue,
                             id<MTLCommandBuffer> __autoreleasing *command_buffer,
                             char *error, size_t error_size) {
    (void)context; (void)variant; (void)kernel_name; (void)queue; (void)command_buffer;
    snprintf(error, error_size, "benchmark adapter not linked");
    return -1;
}

typedef struct {
    const char *model_path;
    unsigned warmup_count;
    unsigned iteration_count;
    int selected_variant;
} Options;

typedef struct {
    double wall_ms;
    double gpu_ms;
} Sample;

static unsigned long model_open_calls;
static unsigned long cpu_dispatches;
static unsigned long gpu_dispatches;
static unsigned long metal_device_requests;

static const VariantDescriptor *descriptor_for_id(unsigned id) {
    for (size_t index = 0; index < sizeof(VARIANTS) / sizeof(VARIANTS[0]); ++index) {
        if (VARIANTS[index].variant == id) return &VARIANTS[index];
    }
    return NULL;
}

static const VariantDescriptor *descriptor_for_name(const char *name) {
    for (size_t index = 0; index < sizeof(VARIANTS) / sizeof(VARIANTS[0]); ++index) {
        if (strcmp(VARIANTS[index].name, name) == 0) return &VARIANTS[index];
    }
    return NULL;
}

static void list_variants(void) {
    puts("DeepSeek-V4 benchmark variants (listing only; no model or Metal access):");
    puts("  name          reduction       layout             device      kernel");
    for (size_t index = 0; index < sizeof(VARIANTS) / sizeof(VARIANTS[0]); ++index) {
        const VariantDescriptor *item = &VARIANTS[index];
        printf("  %-13s %-15s %-18s %-11s %s\n", item->name, item->reduction,
               item->layout, item->uses_gpu ? "Metal GPU" : "CPU",
               item->kernel_name ? item->kernel_name : "-");
    }
}

static void refuse(void) {
    puts("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    puts("!! DEEPSEEK-V4 METAL BENCHMARK REFUSED                              !!");
    puts("!! This benchmark needs the REAL model and explicit user approval.   !!");
    puts("!! User has forbidden LLM load testing until they approve it.        !!");
    puts("!! No model was opened. No Metal device was requested.               !!");
    puts("!! No CPU or GPU benchmark work was dispatched.                      !!");
    puts("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    printf("safety-counters: model_open=%lu metal_device=%lu cpu_dispatch=%lu gpu_dispatch=%lu\n",
           model_open_calls, metal_device_requests, cpu_dispatches, gpu_dispatches);
    printf("Approval requires exact environment setting: %s=%s\n",
           APPROVAL_ENV, APPROVAL_PHRASE);
}

static int parse_unsigned(const char *text, int allow_zero, unsigned *output) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || end == text || *end != '\0' || value > UINT_MAX ||
        (!allow_zero && value == 0)) return -1;
    *output = (unsigned)value;
    return 0;
}

static int parse_options(int argc, const char *argv[], Options *options) {
    *options = (Options){
        .model_path = NULL,
        .warmup_count = 3,
        .iteration_count = 20,
        .selected_variant = -1,
    };

    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--model") == 0 && index + 1 < argc) {
            options->model_path = argv[++index];
        } else if (strcmp(argv[index], "--warmup") == 0 && index + 1 < argc) {
            if (parse_unsigned(argv[++index], 1, &options->warmup_count)) {
                fprintf(stderr, "invalid --warmup value: %s\n", argv[index]);
                return -1;
            }
        } else if (strcmp(argv[index], "--iterations") == 0 && index + 1 < argc) {
            if (parse_unsigned(argv[++index], 0, &options->iteration_count)) {
                fprintf(stderr, "invalid --iterations value: %s\n", argv[index]);
                return -1;
            }
        } else if (strcmp(argv[index], "--variant") == 0 && index + 1 < argc) {
            const VariantDescriptor *descriptor = descriptor_for_name(argv[++index]);
            if (!descriptor) {
                fprintf(stderr, "unknown --variant value: %s\n", argv[index]);
                return -1;
            }
            options->selected_variant = (int)descriptor->variant;
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[index]);
            return -1;
        }
    }

    if (!options->model_path || options->model_path[0] == '\0') {
        fputs("approved execution requires --model PATH for the real model\n", stderr);
        return -1;
    }
    return 0;
}

static double monotonic_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_continuous_time() * (double)timebase.numer /
           (double)timebase.denom * 1.0e-9;
}

static int compare_doubles(const void *left, const void *right) {
    double lhs = *(const double *)left;
    double rhs = *(const double *)right;
    return (lhs > rhs) - (lhs < rhs);
}

static double median_ms(const Sample *samples, unsigned count, int gpu) {
    double *values = malloc((size_t)count * sizeof(*values));
    if (!values) return -1.0;
    for (unsigned index = 0; index < count; ++index)
        values[index] = gpu ? samples[index].gpu_ms : samples[index].wall_ms;
    qsort(values, count, sizeof(*values), compare_doubles);
    double median = count & 1u ? values[count / 2]
                               : (values[count / 2 - 1] + values[count / 2]) * 0.5;
    free(values);
    return median;
}

static double standard_deviation_ms(const Sample *samples, unsigned count, int gpu,
                                    double mean) {
    double squared_deviation_sum = 0.0;
    for (unsigned index = 0; index < count; ++index) {
        double value = gpu ? samples[index].gpu_ms : samples[index].wall_ms;
        double deviation = value - mean;
        squared_deviation_sum += deviation * deviation;
    }
    return sqrt(squared_deviation_sum / count);
}

static int run_one_dispatch(void *context, id<MTLCommandQueue> queue,
                            const VariantDescriptor *descriptor, Sample *sample,
                            char *error, size_t error_size) {
    double began = monotonic_seconds();
    if (!descriptor->uses_gpu) {
        cpu_dispatches++;
        if (coli_v4_bench_cpu_dispatch(context, error, error_size)) return -1;
        sample->wall_ms = (monotonic_seconds() - began) * 1000.0;
        sample->gpu_ms = 0.0;
        return 0;
    }

    id<MTLCommandBuffer> command_buffer = nil;
    gpu_dispatches++;
    if (coli_v4_bench_encode_gpu(context, descriptor->variant,
                                 descriptor->kernel_name, queue,
                                 &command_buffer, error, error_size) || !command_buffer) {
        if (!error[0]) snprintf(error, error_size, "adapter returned no command buffer");
        return -1;
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    sample->wall_ms = (monotonic_seconds() - began) * 1000.0;
    if (command_buffer.status == MTLCommandBufferStatusError) {
        snprintf(error, error_size, "Metal command failed: %s",
                 command_buffer.error
                     ? command_buffer.error.localizedDescription.UTF8String
                     : "unknown error");
        return -1;
    }

    CFTimeInterval gpu_start = command_buffer.GPUStartTime;
    CFTimeInterval gpu_end = command_buffer.GPUEndTime;
    if (gpu_start <= 0.0 || gpu_end < gpu_start) {
        snprintf(error, error_size, "Metal GPU timestamps unavailable");
        return -1;
    }
    sample->gpu_ms = (gpu_end - gpu_start) * 1000.0;
    return 0;
}

static int run_variant(void *context, id<MTLCommandQueue> queue,
                       const VariantDescriptor *descriptor, const Options *options) {
    char error[512] = {0};
    Sample ignored = {0};
    for (unsigned index = 0; index < options->warmup_count; ++index) {
        @autoreleasepool {
            if (run_one_dispatch(context, queue, descriptor, &ignored,
                                 error, sizeof(error))) {
                printf("%-13s UNAVAILABLE stage=warmup iteration=%u reason=%s\n",
                       descriptor->name, index + 1, error[0] ? error : "unknown error");
                return 1;
            }
        }
    }

    Sample *samples = calloc(options->iteration_count, sizeof(*samples));
    if (!samples) {
        printf("%-13s UNAVAILABLE stage=allocate reason=sample-allocation-failed\n",
               descriptor->name);
        return 1;
    }

    double wall_total = 0.0;
    double gpu_total = 0.0;
    double wall_min = 1.0 / 0.0;
    double gpu_min = 1.0 / 0.0;
    for (unsigned index = 0; index < options->iteration_count; ++index) {
        @autoreleasepool {
            error[0] = '\0';
            if (run_one_dispatch(context, queue, descriptor, &samples[index],
                                 error, sizeof(error))) {
                printf("%-13s UNAVAILABLE stage=measure iteration=%u reason=%s\n",
                       descriptor->name, index + 1,
                       error[0] ? error : "unknown error");
                free(samples);
                return 1;
            }
        }
        wall_total += samples[index].wall_ms;
        if (samples[index].wall_ms < wall_min) wall_min = samples[index].wall_ms;
        if (descriptor->uses_gpu) {
            gpu_total += samples[index].gpu_ms;
            if (samples[index].gpu_ms < gpu_min) gpu_min = samples[index].gpu_ms;
        }
    }

    double wall_mean = wall_total / options->iteration_count;
    double gpu_mean = gpu_total / options->iteration_count;
    printf("%-13s n=%u warmup=%u wall_ms mean=%.3f median=%.3f min=%.3f sd=%.3f",
           descriptor->name, options->iteration_count, options->warmup_count,
           wall_mean, median_ms(samples, options->iteration_count, 0), wall_min,
           standard_deviation_ms(samples, options->iteration_count, 0, wall_mean));
    if (descriptor->uses_gpu) {
        printf(" gpu_ms mean=%.3f median=%.3f min=%.3f sd=%.3f",
               gpu_mean, median_ms(samples, options->iteration_count, 1), gpu_min,
               standard_deviation_ms(samples, options->iteration_count, 1, gpu_mean));
    }
    putchar('\n');
    free(samples);
    return 0;
}

static int adapter_available(void) {
    return coli_v4_bench_adapter_version() == 1;
}

static int run_approved(const Options *options) {
    if (!adapter_available()) {
        fputs("benchmark adapter not linked; expected coli_v4_bench_* hooks\n", stderr);
        return 1;
    }

    metal_device_requests++;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fputs("no Metal device available\n", stderr);
        return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        fputs("could not create Metal command queue\n", stderr);
        return 1;
    }

    char error[512] = {0};
    void *context = NULL;
    model_open_calls++;
    if (coli_v4_bench_model_open(options->model_path, device, &context,
                                 error, sizeof(error)) || !context) {
        fprintf(stderr, "model open failed: %s\n", error[0] ? error : "unknown error");
        return 1;
    }

    printf("DeepSeek-V4 Metal benchmark: device=%s warmup=%u iterations=%u\n",
           device.name.UTF8String, options->warmup_count, options->iteration_count);
    int result = 0;
    for (unsigned id = 0; id < COLI_V4_BENCH_VARIANT_COUNT; ++id) {
        if (options->selected_variant >= 0 && options->selected_variant != (int)id)
            continue;
        const VariantDescriptor *descriptor = descriptor_for_id(id);
        if (!descriptor) {
            result = 1;
            continue;
        }
        if (run_variant(context, queue, descriptor, options)) result = 1;
    }
    coli_v4_bench_model_close(context);
    return result;
}

int main(int argc, const char *argv[]) {
    if (argc == 2 && strcmp(argv[1], "--list-variants") == 0) {
        list_variants();
        return 0;
    }

    const char *approval = getenv(APPROVAL_ENV);
    if (!approval || strcmp(approval, APPROVAL_PHRASE) != 0) {
        refuse();
        return 0;
    }

    Options options;
    if (parse_options(argc, argv, &options)) return 2;
    @autoreleasepool {
        return run_approved(&options);
    }
}
