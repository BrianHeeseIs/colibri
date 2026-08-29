# kbench — standalone kernel microbenchmarks

Built by hand; the binaries are ignored, the sources are the artefact.

```bash
clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
  -L/opt/homebrew/opt/libomp/lib -lomp -lm fp8bench.c -o fp8bench
./fp8bench <I> <O> <reps>        # e.g. ./fp8bench 8192 4096 21   (wo_b)
./bw 2048 5                      # host read-bandwidth ceiling
```

**Do NOT pipe the compiler through `head`** — it SIGPIPEs clang before it links and you silently run
a stale binary. That mistake cost two cycles in this project, once producing a bogus "the NaN guard
is unnecessary" result.

`fp8bench` reproduces `matmul_fp8` (c/quant.h:502) exactly on synthetic data and checks every variant
against it with `memcmp`, so a variant that is fast but reordered is reported as NOT bit-exact rather
than as a win. Measured on M3 Max: V0 scalar 30.6 GB/s, V3 (16-row interleave + f16-reinterpret
decode) 65.9 GB/s and bit-exact — the kernel shipped in E125. Host ceiling is ~105 GB/s (`bw`).
