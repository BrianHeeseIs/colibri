# Warm vs cold benchmarking — protocol and results

## Operational definitions
- **COLD** — one-shot process per measurement. Every run re-heats from the frozen `.coli_usage`
  snapshot. hit_pct ≈ 78 %. This is what all campaign benchmarks measure.
- **WARM** — persistent `SERVE=1` process, plateau = requests 5-16 (cache saturates after ~4).
  hit_pct ≈ 95.4 %.

## Protocol (both serve confounds neutralized)
1. **KV prefix reuse** — re-sending a prompt lets the engine skip prefill entirely.
2. **Workload comparability** — four *different* prompts are four different workloads, not a series.

**Cycle a fixed set of 4 distinct prompts, repeated 4 times.** Distinct within a cycle forces a full
attention reset (`kv_prefix_reuse()` is all-or-nothing, `c/kv_prefix.h:116-127`); the same prompt
recurring each cycle makes cycle *k* vs cycle 1 a controlled comparison of the **same** workload at
a hotter cache.

**Proof it worked:** the engine's `DONE` line carries a 10th field, `session->prefix_reused`
(`c/deepseek_v4.c:9213-9217`). It was **0 on all 16 requests in both runs** — every turn genuinely
did a full reset. Never compare two different prompts to each other.

**Never infer decode warmth from `hit_pct`:** it spans the whole generate call *including prefill*
(`:9174-9203`), while `tok_s` divides by `decode_sec`, which starts *after* prefill (`:9204-9217`).

## Results
| cell | hit_pct | tok/s | rel sd |
|---|---|---|---|
| COLD one-shot 60 tok, default | 77.97 | **1.531** | 0.3 % |
| COLD one-shot 60 tok, `--fast-kernels` | 78.51 | **1.700** | 0.7 % |
| MID one-shot 300 tok, default | 90.29 | 1.370 | 3.2 % |
| WARM serve plateau, default | 95.41 | 1.449 | 10.9 % |
| WARM serve plateau, `--fast-kernels` | 95.57 | 1.503 | 11.1 % |

Heating curve (hit_pct): **77.97 → 88.4 (E35, 220 tok) → 90.29 → 95.41**.

## Reading — CORRECTED after review

**The effect of cache heating on decode throughput is UNRESOLVED at this noise level.** An earlier
draft of this document claimed heating "does not speed up decode". That was overstated and is
retracted. Two reasons:

1. The comparison used was **invalid by this document's own rule** — it set request 1 (prompt A)
   against a plateau containing prompts A-D. Only same-prompt comparisons are legitimate.
2. The legitimate same-prompt comparison does not support any conclusion. Cycle 1 -> cycle 4 for
   each of the four prompts:

   | prompt | cyc1 | cyc2 | cyc3 | cyc4 | delta |
   |---|---|---|---|---|---|
   | 1 | 1.517 | 1.111 | 1.568 | 1.270 | **-16.3 %** |
   | 2 | 1.621 | 1.441 | 1.572 | 1.199 | **-26.0 %** |
   | 3 | 1.393 | 1.557 | 1.580 | 1.542 | **+10.7 %** |
   | 4 | 1.234 | 1.557 | 1.445 | 1.548 | **+25.4 %** |

   Deltas span **-26.0 % to +25.4 %** with no consistent sign. At ~10.9 % run-to-run noise and n=4
   this resolves nothing — neither a gain, nor a loss, nor a null.

**What IS supported:** hit_pct genuinely rises 77.97 % -> 95.41 % and saturates after ~4 requests
(`prefix_reused = 0` on all 16 requests proves each turn did a full reset). The *accounting* point
also stands on source: `hit_pct` spans prefill+decode (`:9174-9203`) while `tok_s` divides by
`decode_sec` (`:9204-9217`), so a rising hit_pct cannot by itself be read as warmer decode.
**But the hypothesis that the cold misses live in prefill was never measured — it is a plausible
explanation, not a finding.** Resolving it needs a prefill-vs-decode miss breakdown, not more
aggregate samples.

**Caveat on one-shot rows:** 60- and 300-token runs have different average context lengths, so they
are not a pure temperature series — MID is slower than COLD mostly because attention cost grows with
context. Only the serve rows isolate temperature at fixed token count.

**Use one-shot to measure kernel deltas.** `--fast-kernels` is 11.1 % faster cold (0.3-0.7 % sd,
decisive) but only 3.7 % ±11 % warm — unresolvable. Serve is the right tool for *observing warm
behaviour*, the wrong one for *measuring deltas*. Reaching true one-shot precision (0.27 % rel sd)
from serve's 10.9 % would need `(10.9/0.27)^2 ≈ 1630` samples per configuration, not the ~475 an
earlier draft quoted (that figure targeted 0.5 %, not parity).
