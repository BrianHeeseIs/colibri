/* Consumer translation unit for the decode-trace link test.
 *
 * Compiled with -DCOLI_V4_DECODE_TRACE_DECLS_ONLY, the same way every non-owner
 * COLI_V4_UNIT_* object is compiled in the engine. It must be able to record a sample
 * WITHOUT owning any counter storage: the only legal route is the extern bridge. */
#include "../decode_trace.h"

void coli_v4_dt_link_consumer_write(void) {
    coli_v4_decode_trace_note(COLI_V4_DT_WAIT_START_LOCK, UINT64_C(7));
    coli_v4_decode_trace_note(COLI_V4_DT_WAIT_START_LOCK, UINT64_C(5));
    coli_v4_decode_trace_note(COLI_V4_DT_STORE_LOCK, UINT64_C(11));
}
