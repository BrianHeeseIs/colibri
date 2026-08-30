/* Decode-trace bridge for tests that link a single amalgamation unit.
 *
 * tests/test_hc_omp and tests/test_sparse_omp compile deepseek_v4.c with one
 * -DCOLI_V4_UNIT_* and link only that object. Those units are consumers of the decode
 * trace, so they reference the bridge but never the owner (COLI_V4_UNIT_GENERATE_STATS),
 * and would otherwise fail to link with undefined symbols.
 *
 * This provides the owner side for those test binaries only, backed by a real counter
 * array so a test could assert on it if one ever needs to. The engine binary does NOT
 * link this file -- there the owner is COLI_V4_UNIT_GENERATE_STATS. */
#include "../decode_trace.h"

int coli_v4_decode_trace_on;

void coli_v4_decode_trace_note(int stage, uint64_t elapsed_ns) {
    coli_v4_decode_trace_add(stage, elapsed_ns);
}

int coli_v4_decode_trace_enabled(void) { return coli_v4_decode_trace_on; }

uint64_t coli_v4_decode_trace_clock_ns(void) {
    return coli_v4_decode_trace_now_ns();
}
