# Measurement probes — evidence for the batched-MoE plan

Standalone, no engine linkage. These produced the numbers cited in
`docs/plans/metal-batched-moe-architecture.md`.

| probe | builds with | measures |
|---|---|---|
| `mtlinfo.m` | `clang -fobjc-arc -O2 -framework Metal -framework Foundation` | Metal device limits (threadgroup mem, max threads, GPUFamily) |
| `bwprobe.m` | same | CPU vs GPU streaming bandwidth on one unified buffer (first cut) |
| `bw2.m` | same | **hardened** bandwidth probe: unconditional writes + checksum verification, sweeps 512 MB–8 GB |
| `sweep_mxfp4.m` | same | GPU MXFP4 S-sweep, simple one-thread-per-row |
| `sweep2.m` | same | GPU MXFP4 S-sweep, tiled + `uchar4` + threadgroup-staged activations |
| `cpu_mxfp4.c` | `clang -O3` | CPU MXFP4 S-sweep, 12 threads via GCD |

```bash
cd validation/probes
clang -fobjc-arc -O2 -framework Metal -framework Foundation bw2.m -o /tmp/bw2 && /tmp/bw2
clang -O3 cpu_mxfp4.c -o /tmp/cpu_mxfp4 && /tmp/cpu_mxfp4
```
