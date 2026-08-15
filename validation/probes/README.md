# Measurement probes — evidence for the batched-MoE plan

Small standalone executables. Most do not link the engine; probes that audit a
live kernel link only the named engine units. These produced the numbers cited
in `docs/plans/metal-batched-moe-architecture.md`.

| probe | builds with | measures |
|---|---|---|
| `mtlinfo.m` | `clang -fobjc-arc -O2 -framework Metal -framework Foundation` | Metal device limits (threadgroup mem, max threads, GPUFamily) |
| `bwprobe.m` | same | CPU vs GPU streaming bandwidth on one unified buffer (first cut) |
| `bw2.m` | same | **hardened** bandwidth probe: unconditional writes + checksum verification, sweeps 512 MB–8 GB |
| `sweep_mxfp4.m` | same | GPU MXFP4 S-sweep, simple one-thread-per-row |
| `sweep2.m` | same | GPU MXFP4 S-sweep, tiled + `uchar4` + threadgroup-staged activations |
| `cpu_mxfp4.c` | `clang -O3` | CPU MXFP4 S-sweep, 12 threads via GCD |
| `mxfp4_s_scaling.m` | Objective-C, Metal, libomp, live V4 quant units | One-process production-shape comparison: live rows16 NEON versus safe-math `ordered_xcache`, on shared buffers |

```bash
cd validation/probes
clang -fobjc-arc -O2 -framework Metal -framework Foundation bw2.m -o /tmp/bw2 && /tmp/bw2
clang -O3 cpu_mxfp4.c -o /tmp/cpu_mxfp4 && /tmp/cpu_mxfp4
```

Build the live-kernel S-scaling gate from the repository root without changing
the shipped binary or any in-tree object:

```bash
OMP="$(brew --prefix libomp)"
COMMON=(-D_DARWIN_C_SOURCE -D_FILE_OFFSET_BITS=64 -O3 -flto -pthread \
  -Xclang -fopenmp -I"$OMP/include" -Ic)
clang "${COMMON[@]}" -DCOLI_V4_UNIT_NATIVE_QUANT \
  -c c/deepseek_v4.c -o /tmp/mxfp4_native_quant.o
clang "${COMMON[@]}" -DCOLI_V4_UNIT_NATIVE_QUANT_ROWS16 \
  -c c/deepseek_v4.c -o /tmp/mxfp4_rows16.o
clang "${COMMON[@]}" -fobjc-arc validation/probes/mxfp4_s_scaling.m \
  /tmp/mxfp4_native_quant.o /tmp/mxfp4_rows16.o \
  -L"$OMP/lib" -Wl,-rpath,"$OMP/lib" -lomp -lm \
  -framework Metal -framework Foundation -o /tmp/mxfp4_s_scaling
/tmp/mxfp4_s_scaling
```

The gate/up row compares one fused dual CPU call with two Metal dispatches;
the down row compares one call with one dispatch. `GPU ms` is host wall time
from command-buffer creation through `waitUntilCompleted`, while the trailing
GPU-active value comes from command-buffer timestamps. CPU and GPU phases are
separated by 300 ms so idle OpenMP workers cannot contend with Metal. The probe
rejects `block_rows=1` before accepting and timing `block_rows=16`.
