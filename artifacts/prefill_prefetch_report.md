# Prefill I/O campaign — final report

## Outcome
A correct, bit-exact, default-OFF prefill prefetch feature worth **6.20 % (p064) / 2.70 % (p256)**,
and — more valuable — the first **neutral in-engine attribution** of prefill, which killed four
successive mechanism hypotheses and identified the real target.

## Gates
| gate | criterion | measured | result |
|---|---|---|---|
| **A** | QD8 >= 2x QD1 | **1.34** (QD1 5.227 GB/s, saturates 7.028 at QD4) | **STOP** — deep-queue design killed |
| **B** | >= 15 % ttft on p256 | **2.70 %** | **FAIL** — feature kept anyway, by explicit user decision |

## The attribution that ended the guessing (E54)
p064, prefetch OFF, 43.544 s wall, sums to 100 % with 1.55 % residual:

| stage | % of wall |
|---|---|
| MoE | 60.05 |
| — of which: miss read + first touch | 43.47 |
| — of which: pack (hit+miss) | 5.12 |
| — of which: all mutex wait | 1.59 |
| **attention** | **33.18** |
| norms + HC post + head | 5.22 |
| residual | 1.55 |

**Expert compute is identical in prefill and decode** (0.6947 vs 0.7059 ms/op; 39.28 % vs 39.51 %
OpenMP efficiency at 16 threads). The "4.8x per-op gap" that motivated two lanes was an artefact of
arithmetic on aggregates.

## Four hypotheses, all killed by measurement
| # | claim | killed by | measured |
|---|---|---|---|
| 1 | Metal shaders are slow (~15 GFLOP/s) | E48 | first-touch artefact; shaders run at 2.35 TFLOP/s |
| 2 | 84 % of prefill is SSD wait | E52 | QD1 41.97 / QD4 39.57 / QD8 39.99 — queue depth moves nothing |
| 3 | prefill uses a slow unpacked kernel | E53 | ratio **1.099x**, not 4.8x; packing break-even 9.62 uses vs 4.4 available |
| 4 | prefill gets worse OpenMP scaling | E54 | identical to decode (39.28 % vs 39.51 %) |

Each time the *outcome* measurement held and the *explanation* did not.
**Rule earned: attribute inside the engine before proposing a mechanism.**

## Correctness canaries
Bit-exact **at both gate lengths**, not merely at golden's single prompt (E56):

| prompt | OFF vs ON | md5 |
|---|---|---|
| p064 | identical | `12f5fac018e335613d4018e595e70703` |
| p256 | identical | `4b146c969f31b97cefb5f5cfb251b207` |
| golden (60 tok) | identical | `5d04890413ff539e802985ce8c727814` |

Default path is fail-safe: unset / empty / `"0"` all resolve OFF; one `getenv` at init is the only
residual cost, and the OFF arm is faster than the pre-feature pinned baseline.

## Methodology note that changed a published number
The host drifted ~2 % faster between pinning the baseline and running GATE B. Diffing ON against the
**stale pinned baseline** would have reported p256 as -4.67 % instead of the true interleaved
-2.70 % — a **1.73x inflation**. Every number here is from paired OFF/ON runs for this reason.

## Measured feature stack (p064 ttft, cold)
| config | ttft |
|---|---|
| default | 42.780 s |
| `--fast-kernels` | 38.942 s |
| `COLI_V4_PREFILL_PREFETCH=1` | 40.452 s |
| **both** | **36.392 s (-14.9 %)** |

## What to do next, on evidence
**Attention is 33.18 % of prefill and untouched by any lane in this campaign.** It is O(n^2) in
context, which is why the prefetch gain decays 6.20 % -> 2.70 % as prompts grow. For long prompts
it is the only lever that scales the right way.

## Reproduction
```bash
./bench/build.sh                                   # both build traps encoded
env -u COLI_V4_PREFILL_PREFETCH ./bench/golden.sh ./c/deepseek_v4
N=3 ./bench/rebaseline.sh ./c/deepseek_v4          # pins baseline
N=3 ./bench/ab.sh "COLI_V4_PREFILL_PREFETCH=1" ./c/deepseek_v4
cc -O2 -o bench/qd_sweep bench/qd_sweep.c      # tools ship as source, not binaries
./bench/qd_sweep models/deepseek-v4-flash artifacts/layer_contig.json
cc -O2 -o bench/kernel_gap bench/kernel_gap.c  # E53 packed-vs-unpacked ratio
# attribution: rebuild with -DCOLI_V4_PREFILL_TRACE, run p064 --max-tokens 1
```
Artifacts: `artifacts/baseline.md`, `artifacts/qd_sweep.md`, `artifacts/layer_contig.json`,
`.backlog/results/`.
