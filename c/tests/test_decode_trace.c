#include "../decode_trace.h"
#include <stdio.h>

static int check(int condition, const char *name) {
    if (condition) return 0;
    fprintf(stderr, "decode_trace: FAIL %s\n", name);
    return 1;
}

int main(void) {
    int bad = 0;
    const size_t names_count =
        sizeof(coli_v4_decode_trace_names) / sizeof(coli_v4_decode_trace_names[0]);

    bad += check(names_count == COLI_V4_DT_COUNT, "names length");

    coli_v4_decode_trace_reset();
    coli_v4_decode_trace_add(-1, UINT64_C(7));
    coli_v4_decode_trace_add(COLI_V4_DT_COUNT, UINT64_C(11));
    for (int stage = 0; stage < COLI_V4_DT_COUNT; stage++) {
        bad += check(coli_v4_decode_trace_stages[stage].elapsed_ns == 0,
                     "out-of-range elapsed no-op");
        bad += check(coli_v4_decode_trace_stages[stage].calls == 0,
                     "out-of-range calls no-op");
    }

    coli_v4_decode_trace_add(COLI_V4_DT_WAIT_START_LOCK, UINT64_C(7));
    coli_v4_decode_trace_add(COLI_V4_DT_WAIT_START_LOCK, UINT64_C(5));
    bad += check(
        coli_v4_decode_trace_stages[COLI_V4_DT_WAIT_START_LOCK].elapsed_ns == 12,
        "add accumulates elapsed_ns");
    bad += check(
        coli_v4_decode_trace_stages[COLI_V4_DT_WAIT_START_LOCK].calls == 2,
        "add accumulates calls");

    coli_v4_decode_trace_reset();
    for (int stage = 0; stage < COLI_V4_DT_COUNT; stage++) {
        bad += check(coli_v4_decode_trace_stages[stage].elapsed_ns == 0,
                     "reset elapsed_ns");
        bad += check(coli_v4_decode_trace_stages[stage].calls == 0,
                     "reset calls");
    }

    bad += check(coli_v4_decode_trace_pct_parent(UINT64_C(100), UINT64_C(100)) == 100.0,
                 "pct exact sum");
    bad += check(coli_v4_decode_trace_pct_parent(UINT64_C(97), UINT64_C(100)) < 98.0,
                 "pct below parent");
    bad += check(coli_v4_decode_trace_pct_parent(UINT64_C(103), UINT64_C(100)) > 102.0,
                 "pct above parent");
    bad += check(coli_v4_decode_trace_pct_parent(UINT64_C(1), UINT64_C(0)) == 0.0,
                 "pct zero parent");

    bad += check(coli_v4_decode_trace_classify_wait(1, 0) == COLI_V4_DT_NO_SLEEP,
                 "classify no sleep");
    bad += check(coli_v4_decode_trace_classify_wait(0, 1) == COLI_V4_DT_SLEPT,
                 "classify slept");
    bad += check(coli_v4_decode_trace_classify_wait(0, 2) == COLI_V4_DT_FALSE_WAKE,
                 "classify false wake");

    if (bad) {
        fprintf(stderr, "decode_trace: %d checks failed\n", bad);
        return 1;
    }
    printf("decode_trace: all checks passed\n");
    return 0;
}
