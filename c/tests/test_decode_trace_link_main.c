/* Guards the defect this whole trace design exists to avoid.
 *
 * deepseek_v4.c is an amalgamation compiled ~26 times, once per -DCOLI_V4_UNIT_*. If
 * decode_trace.h hands every object its own `static` counter array, writes made in a
 * consumer unit land in a private copy, and the report in the owner unit prints ALL
 * ZEROS. That output is indistinguishable from a genuinely idle path, and it would
 * silently satisfy the pre-registered "main-thread wait < 3% of decode wall" kill
 * criterion -- closing a live optimisation lever on a measurement artefact.
 *
 * This test reproduces the split across two translation units and fails if the owner
 * cannot observe the consumer's write. */
#include "../decode_trace.h"
#include <stdio.h>

extern void coli_v4_dt_link_consumer_write(void);
extern uint64_t coli_v4_dt_link_owner_elapsed(int stage);
extern uint64_t coli_v4_dt_link_owner_calls(int stage);
extern void coli_v4_dt_link_owner_reset(void);

static int check(int condition, const char *name) {
    if (condition) return 0;
    fprintf(stderr, "decode_trace_link: FAIL %s\n", name);
    return 1;
}

int main(void) {
    int bad = 0;

    coli_v4_dt_link_owner_reset();
    bad += check(coli_v4_dt_link_owner_calls(COLI_V4_DT_WAIT_START_LOCK) == 0,
                 "owner storage starts clean");

    coli_v4_dt_link_consumer_write();

    bad += check(coli_v4_dt_link_owner_elapsed(COLI_V4_DT_WAIT_START_LOCK) == 12,
                 "owner sees consumer elapsed_ns across translation units");
    bad += check(coli_v4_dt_link_owner_calls(COLI_V4_DT_WAIT_START_LOCK) == 2,
                 "owner sees consumer calls across translation units");
    bad += check(coli_v4_dt_link_owner_elapsed(COLI_V4_DT_STORE_LOCK) == 11,
                 "owner sees a second stage written by the consumer");
    bad += check(coli_v4_dt_link_owner_calls(COLI_V4_DT_STORE_LOCK) == 1,
                 "second stage call count");

    if (bad) {
        fprintf(stderr, "decode_trace_link: %d checks failed\n", bad);
        return 1;
    }
    printf("decode_trace_link: all checks passed\n");
    return 0;
}
