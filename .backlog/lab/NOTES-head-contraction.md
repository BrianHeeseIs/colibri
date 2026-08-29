# P1 — head_bf16_dot FP contraction

`head_bf16_dot` (c/deepseek_v4.c:9605) is `static` and INLINED: `nm c/deepseek_v4 | grep head_bf16`
returns nothing, and `objdump --disassemble='_head_argmax'` yields no instructions.

Standalone repro under the engine's exact CFLAGS
(-D_DARWIN_C_SOURCE -D_FILE_OFFSET_BITS=64 -O3 -Xclang -fopenmp -pthread -include pthread.h
 -Wall -Wextra), file .backlog/lab/kbench/contract_probe.c:

    fmla = 0    fmul = 9    fadd = 36     => UNFUSED

CAVEAT that makes this reading NON-BINDING: the probe inlines the bf16 decode via
__builtin_memcpy, whereas the engine calls `coli_bf16_decode` (declared c/native_quant.h:18,
defined in another translation unit) and the real build links with `-flto`, so the engine's
inlining — and therefore its contraction — may differ from the probe's.

## Consequence for T1 (this is the important part)
Do NOT try to match a disassembly. Make the multi-row path use the IDENTICAL per-row expression
as today (`sum += coli_bf16_decode(w[c]) * hidden[c]`), merely interleaving 4 independent rows in
one pass over `hidden`. Then whatever clang does — fused or not, vectorised or not — it applies the
SAME transformation to each row's chain, and each chain still accumulates in ascending `c`.
Bit-exactness then follows STRUCTURALLY and is robust to the contraction question, instead of
depending on an answer that the probe cannot settle authoritatively.

This is the `matmul_fp8` o+=4 idiom already in-tree (c/quant.h:506).
