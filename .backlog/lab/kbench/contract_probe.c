/* P1: does the engine's CFLAGS make `sum += bf16(w)*h` fuse into FMLA on aarch64?
 * head_bf16_dot is static+inlined so it has no symbol; this reproduces its inner loop verbatim
 * under the identical flags and is disassembled instead. Decides T1's bit-exactness model. */
#include <stdint.h>
static inline float bf16_decode(uint16_t v){ uint32_t b=(uint32_t)v<<16; float f; __builtin_memcpy(&f,&b,4); return f; }
float probe_head_dot(const uint16_t *w, const float *h, int n){
    float sum = 0.0f;
    for (int c = 0; c < n; c++) sum += bf16_decode(w[c]) * h[c];
    return sum;
}
