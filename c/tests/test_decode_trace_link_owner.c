/* Owner translation unit for the decode-trace link test.
 *
 * Includes decode_trace.h in owner mode (no COLI_V4_DECODE_TRACE_DECLS_ONLY), so this
 * object holds the ONE real copy of the counters -- exactly the role
 * COLI_V4_UNIT_GENERATE_STATS plays in the engine amalgamation. */
#include "../decode_trace.h"

int coli_v4_decode_trace_on = 1;

void coli_v4_decode_trace_note(int stage, uint64_t elapsed_ns) {
    coli_v4_decode_trace_add(stage, elapsed_ns);
}

int coli_v4_decode_trace_enabled(void) {
    return coli_v4_decode_trace_on;
}

uint64_t coli_v4_decode_trace_clock_ns(void) {
    return coli_v4_decode_trace_now_ns();
}

/* Test-only readers. The main TU observes the owner's storage through these rather
 * than including the header in owner mode, which would hand it a private copy and
 * defeat the point of the test. */
uint64_t coli_v4_dt_link_owner_elapsed(int stage) {
    return coli_v4_decode_trace_stages[stage].elapsed_ns;
}

uint64_t coli_v4_dt_link_owner_calls(int stage) {
    return coli_v4_decode_trace_stages[stage].calls;
}

void coli_v4_dt_link_owner_reset(void) {
    coli_v4_decode_trace_reset();
}
