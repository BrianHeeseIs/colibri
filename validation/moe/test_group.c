#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    MAX_ITEMS = 32,
    MAX_TOPK = 6,
    MAX_EXPERTS = 64,
    MAX_ENTRIES = MAX_ITEMS * MAX_TOPK,
    VECTOR_WIDTH = 3,
};

typedef struct {
    int token;
    int rank;
} RoutePair;

typedef struct {
    int effective_reserve;
    int compute_wave_size;
    int wave_count;
    bool degenerate;
} WavePlan;

typedef struct {
    int expert_id;
    float values[VECTOR_WIDTH];
} ExpertOutput;

static int compare_ints(const void *left, const void *right) {
    const int left_value = *(const int *)left;
    const int right_value = *(const int *)right;
    return (left_value > right_value) - (left_value < right_value);
}

static void sort_unique_experts_ascending(int *unique_experts, int n_unique) {
    qsort(unique_experts, (size_t)n_unique, sizeof(*unique_experts),
          compare_ints);
}

static int find_sorted_expert(const int *unique_experts, int n_unique,
                              int expert_id) {
    int low = 0;
    int high = n_unique;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (unique_experts[middle] < expert_id) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low < n_unique && unique_experts[low] == expert_id ? low : -1;
}

static bool build_csr_grouping(const int *idx, int items, int topk,
                               const int *unique_experts, int n_unique,
                               int *counts, int *offsets, RoutePair *perm,
                               int *inv) {
    memset(counts, 0, (size_t)n_unique * sizeof(*counts));
    const int entries = items * topk;
    for (int source = 0; source < entries; ++source) {
        const int expert =
            find_sorted_expert(unique_experts, n_unique, idx[source]);
        if (expert < 0) {
            return false;
        }
        ++counts[expert];
    }

    offsets[0] = 0;
    for (int expert = 0; expert < n_unique; ++expert) {
        offsets[expert + 1] = offsets[expert] + counts[expert];
        inv[expert] = offsets[expert];
    }

    for (int source = 0; source < entries; ++source) {
        const int expert =
            find_sorted_expert(unique_experts, n_unique, idx[source]);
        const int grouped = inv[expert]++;
        perm[grouped] = (RoutePair){source / topk, source % topk};
    }
    for (int grouped = 0; grouped < entries; ++grouped) {
        const int source = perm[grouped].token * topk + perm[grouped].rank;
        inv[source] = grouped;
    }
    return true;
}

static WavePlan make_wave_plan(int capacity, int n_unique) {
    const int configured_reserve = 16;
    const int effective_reserve =
        configured_reserve < capacity - 1 ? configured_reserve : capacity - 1;
    const int compute_wave_size = capacity - effective_reserve;
    const bool degenerate = compute_wave_size < 2;
    return (WavePlan){
        .effective_reserve = effective_reserve,
        .compute_wave_size = compute_wave_size,
        .wave_count = degenerate
                          ? 0
                          : (n_unique + compute_wave_size - 1) / compute_wave_size,
        .degenerate = degenerate,
    };
}

static int compare_expert_outputs(const void *left, const void *right) {
    const ExpertOutput *left_expert = left;
    const ExpertOutput *right_expert = right;
    return (left_expert->expert_id > right_expert->expert_id) -
           (left_expert->expert_id < right_expert->expert_id);
}

static void combine_ascending_expert_id(float *output,
                                        const ExpertOutput *experts,
                                        int n_experts) {
    ExpertOutput ordered[n_experts > 0 ? n_experts : 1];
    memcpy(ordered, experts, (size_t)n_experts * sizeof(*ordered));
    qsort(ordered, (size_t)n_experts, sizeof(*ordered),
          compare_expert_outputs);

    memset(output, 0, VECTOR_WIDTH * sizeof(*output));
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int column = 0; column < VECTOR_WIDTH; ++column) {
            output[column] += ordered[expert].values[column];
        }
    }
}

static uint32_t random_u32(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static bool test_sort(char *detail, size_t detail_size) {
    int experts[] = {5, 2, 9, 1, 7, 3};
    const int expected[] = {1, 2, 3, 5, 7, 9};
    const int n_experts = (int)(sizeof(experts) / sizeof(experts[0]));

    sort_unique_experts_ascending(experts, n_experts);
    for (int index = 1; index < n_experts; ++index) {
        if (experts[index - 1] >= experts[index]) {
            snprintf(detail, detail_size, "order at %d: %d >= %d", index,
                     experts[index - 1], experts[index]);
            return false;
        }
    }
    if (memcmp(experts, expected, sizeof(experts)) != 0) {
        snprintf(detail, detail_size, "sorted output changed multiset");
        return false;
    }
    return true;
}

static bool token_has_expert(const int *idx, int token, int topk,
                             int candidate) {
    for (int rank = 0; rank < topk; ++rank) {
        if (idx[token * topk + rank] == candidate) {
            return true;
        }
    }
    return false;
}

static bool test_csr(char *detail, size_t detail_size) {
    uint32_t random_state = UINT32_C(0xc011b1a5);

    for (int routing = 0; routing < 1000; ++routing) {
        const int items = 1 + (int)(random_u32(&random_state) % MAX_ITEMS);
        const int topk = 1 + (int)(random_u32(&random_state) % MAX_TOPK);
        const int entries = items * topk;
        int idx[MAX_ENTRIES];
        int unique_experts[MAX_EXPERTS];
        bool expert_seen[MAX_EXPERTS] = {false};
        int n_unique = 0;

        for (int token = 0; token < items; ++token) {
            for (int rank = 0; rank < topk; ++rank) {
                int candidate;
                do {
                    candidate = (int)(random_u32(&random_state) % MAX_EXPERTS);
                } while (token_has_expert(idx, token, rank, candidate));
                idx[token * topk + rank] = candidate;
                if (!expert_seen[candidate]) {
                    expert_seen[candidate] = true;
                    unique_experts[n_unique++] = candidate;
                }
            }
        }
        sort_unique_experts_ascending(unique_experts, n_unique);

        int counts[MAX_EXPERTS];
        int offsets[MAX_EXPERTS + 1];
        RoutePair perm[MAX_ENTRIES];
        int inv[MAX_ENTRIES];
        if (!build_csr_grouping(idx, items, topk, unique_experts, n_unique,
                                counts, offsets, perm, inv)) {
            snprintf(detail, detail_size, "routing %d build rejected valid input",
                     routing);
            return false;
        }

        int count_sum = 0;
        for (int expert = 0; expert < n_unique; ++expert) {
            count_sum += counts[expert];
            if (offsets[expert + 1] != offsets[expert] + counts[expert]) {
                snprintf(detail, detail_size,
                         "routing %d invalid prefix at expert %d", routing,
                         expert);
                return false;
            }
        }
        if (offsets[0] != 0 || count_sum != entries ||
            offsets[n_unique] != entries) {
            snprintf(detail, detail_size, "routing %d count sum %d != %d",
                     routing, count_sum, entries);
            return false;
        }

        int source[MAX_ENTRIES];
        int grouped[MAX_ENTRIES];
        int restored[MAX_ENTRIES];
        int token_counts[MAX_ITEMS] = {0};
        bool source_seen[MAX_ENTRIES] = {false};
        for (int source_index = 0; source_index < entries; ++source_index) {
            source[source_index] = routing * MAX_ENTRIES + source_index;
        }
        for (int expert = 0; expert < n_unique; ++expert) {
            for (int grouped_index = offsets[expert];
                 grouped_index < offsets[expert + 1]; ++grouped_index) {
                const RoutePair pair = perm[grouped_index];
                if (pair.token < 0 || pair.token >= items || pair.rank < 0 ||
                    pair.rank >= topk) {
                    snprintf(detail, detail_size,
                             "routing %d invalid pair at grouped index %d", routing,
                             grouped_index);
                    return false;
                }
                const int source_index = pair.token * topk + pair.rank;
                if (source_seen[source_index] ||
                    idx[source_index] != unique_experts[expert] ||
                    inv[source_index] != grouped_index) {
                    snprintf(detail, detail_size,
                             "routing %d broken permutation at grouped index %d",
                             routing, grouped_index);
                    return false;
                }
                source_seen[source_index] = true;
                ++token_counts[pair.token];
                grouped[grouped_index] = source[source_index];
            }
        }
        for (int source_index = 0; source_index < entries; ++source_index) {
            restored[source_index] = grouped[inv[source_index]];
        }
        if (memcmp(source, restored, (size_t)entries * sizeof(*source)) != 0) {
            snprintf(detail, detail_size,
                     "routing %d permute/unpermute changed identity", routing);
            return false;
        }
        for (int token = 0; token < items; ++token) {
            if (token_counts[token] != topk) {
                snprintf(detail, detail_size,
                         "routing %d token %d appeared %d times, expected %d",
                         routing, token, token_counts[token], topk);
                return false;
            }
        }
    }
    return true;
}

static bool test_wave_sizing(char *detail, size_t detail_size) {
    const int capacities[] = {1, 2, 6, 17, 164};
    const int expected_reserve[] = {0, 1, 5, 16, 16};
    const int expected_size[] = {1, 1, 1, 1, 148};
    const int n_unique = 188;

    for (int index = 0; index < (int)(sizeof(capacities) / sizeof(capacities[0]));
         ++index) {
        const WavePlan plan = make_wave_plan(capacities[index], n_unique);
        if (plan.compute_wave_size < 1) {
            snprintf(detail, detail_size,
                     "capacity=%d compute_wave_size=%d", capacities[index],
                     plan.compute_wave_size);
            return false;
        }
        if (plan.effective_reserve != expected_reserve[index] ||
            plan.compute_wave_size != expected_size[index]) {
            snprintf(detail, detail_size,
                     "capacity=%d reserve/size=%d/%d expected=%d/%d",
                     capacities[index], plan.effective_reserve,
                     plan.compute_wave_size, expected_reserve[index],
                     expected_size[index]);
            return false;
        }
        if (plan.compute_wave_size < 2) {
            if (!plan.degenerate || plan.wave_count != 0) {
                snprintf(detail, detail_size,
                         "capacity=%d must report degenerate S=1 fallback",
                         capacities[index]);
                return false;
            }
        } else {
            const int expected_waves =
                (n_unique + plan.compute_wave_size - 1) / plan.compute_wave_size;
            if (plan.degenerate || plan.wave_count != expected_waves) {
                snprintf(detail, detail_size,
                         "capacity=%d wave_count=%d expected=%d",
                         capacities[index], plan.wave_count, expected_waves);
                return false;
            }
        }
    }
    return true;
}

static void combine_reference_ascending(float *output,
                                        const ExpertOutput *experts,
                                        int n_experts) {
    memset(output, 0, VECTOR_WIDTH * sizeof(*output));
    for (int expert_id = 1; expert_id <= n_experts; ++expert_id) {
        for (int index = 0; index < n_experts; ++index) {
            if (experts[index].expert_id != expert_id) {
                continue;
            }
            for (int column = 0; column < VECTOR_WIDTH; ++column) {
                output[column] += experts[index].values[column];
            }
            break;
        }
    }
}

static void combine_encounter_order(float *output, const ExpertOutput *experts,
                                    int n_experts) {
    memset(output, 0, VECTOR_WIDTH * sizeof(*output));
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int column = 0; column < VECTOR_WIDTH; ++column) {
            output[column] += experts[expert].values[column];
        }
    }
}

static bool test_combine_order(char *detail, size_t detail_size) {
    const ExpertOutput experts[] = {
        {1, {10000000.0f, 3.0f, -2.0f}},
        {3, {0.25f, 0.25f, 4.0f}},
        {2, {-9999999.0f, -2.5f, -2.0f}},
        {6, {0.0f, 2.0f, -1.0f}},
        {4, {0.0f, -4.0f, 0.5f}},
        {5, {0.0f, 1.5f, 0.5f}},
    };
    const int n_experts = (int)(sizeof(experts) / sizeof(experts[0]));
    float ascending[VECTOR_WIDTH];
    float reference[VECTOR_WIDTH];
    float encounter[VECTOR_WIDTH];

    combine_ascending_expert_id(ascending, experts, n_experts);
    combine_reference_ascending(reference, experts, n_experts);
    combine_encounter_order(encounter, experts, n_experts);

    if (encounter[0] == reference[0]) {
        snprintf(detail, detail_size,
                 "fixture insensitive: encounter and ascending both %.9g",
                 encounter[0]);
        return false;
    }
    if (memcmp(ascending, reference, sizeof(reference)) != 0) {
        snprintf(detail, detail_size,
                 "ascending[0]=%.9g reference[0]=%.9g encounter[0]=%.9g",
                 ascending[0], reference[0], encounter[0]);
        return false;
    }
    return true;
}

typedef bool (*TestFunction)(char *, size_t);

static int run_test(const char *name, TestFunction test) {
    char detail[160] = {0};
    if (test(detail, sizeof(detail))) {
        printf("PASS %s\n", name);
        return 0;
    }
    printf("FAIL %s %s\n", name, detail);
    return 1;
}

int main(void) {
    int failures = 0;
    failures += run_test("sort_unique_experts_ascending", test_sort);
    failures += run_test("csr_grouping_1000_random", test_csr);
    failures += run_test("wave_sizing_clamp", test_wave_sizing);
    failures += run_test("ascending_combine_near_cancellation",
                         test_combine_order);
    return failures == 0 ? 0 : 1;
}
