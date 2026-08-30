# E131 pin slots miss-rate pre-registration

Date 2026-08-30. Branch `ft-decode-expert-wait-applmtl`. Base commit `e743d9f`.

This document is written before the experiment runs. It fixes the hypothesis, arms, thresholds, attribution rules, and kill criteria for the Stage 1 mechanism probe. Do not edit the thresholds after any run exists.

## Hypothesis

Raising `COLI_V4_PIN_SLOTS` above its effective default of 16 reduces the decode expert cache MISS RATE, because `c/deepseek_v4.c:8573` excludes pinned experts from the LRU victim loop:

```c
!hot_is_pinned(policy, key.layer, slots[i].expert) &&
```

A third fallback loop at `c/deepseek_v4.c:8576-8578` can still evict a pin if nothing else is free, so over-pinning degrades gracefully rather than deadlocking. `hot_repin_locked` (`c/deepseek_v4.c:8424-8428`) also refreshes `slots[i].used` for pinned slots every `repin_interval=6` layer requests, reinforcing the shield.

## Why this is legitimately re-opened

`AGENTS.md` currently says: "Two things NOT to retry: raising `COLI_V4_PIN_SLOTS` (the scalar and NEON mxfp4 kernels are within 1.00-1.14x, measured head-to-head)."

That rejection measured KERNEL THROUGHPUT and its stated reason is about kernel throughput only. Pins ALSO gate LRU eviction, a property that rejection never mentions and could not have measured, because the miss-rate instrument (`COLI_V4_DECODE_TRACE`) did not exist until E130.

The rejection is sound for the property it tested and silent on the property E131 tests.

## E130 baseline block

All figures below are measured E130 values and are the baseline for this pre-registration:

- prompt p256, 40 tokens (39 decoded), `--memory-gb 96`
- TTFT 42.891 s, decode_sec 17.941, tok/s 2.1738
- decode_wall_ms 17885.966
- generated_text md5 d06053793d8f66d4f69a3f7c810441e1
- store_disk_read total_ms=2987.867 calls=765 (16.705% of decode wall)
- wait_finish_complete_block total_ms=1488.243 calls=10062 (8.321% of decode wall)
- finish_slept_calls=723, finish_completed_at_entry=9339, start_slept_calls=0
- expert_forward total_ms=7485.917 calls=10062
- v4_rows16 packed_slots=904 expert_calls_rows16=4484 expert_calls_scalar=10775
- v4_hot_policy pin_slots_per_layer=16 repin_interval=6 mode=resident-ram rows16=hot-pins
- 10062 lookups, 765 misses = 7.6% miss rate; loader already hides 50.2% of disk time (1488.243 of 2987.867 lands on the main thread)

## Ceiling of the whole avenue

If every main-thread park disappears, the upper bound is:

`decode_sec 17.941 - 1.488 = 16.453 s`

`39 / 16.453 = 2.3704 tok/s`

Versus baseline tok/s:

`(2.3704 / 2.1738 - 1) x 100 = 9.04%`

Eliminating EVERY park is +9.04%, which is inside the 5-13% decode noise floor. Therefore the COUNTERS are the primary evidence and tok/s can only ever corroborate. A plan that made tok/s the pass condition would be unfalsifiable.

## Loader-depth lever rejected without a run

Mean expert compute:

`7485.917 / 10062 = 0.7440 ms`

Mean disk read:

`2987.867 / 765 = 3.9057 ms`

Mean park:

`1488.243 / 723 = 2.0584 ms`

At depth 3 the lead time for expert i>=3 is `3 x 0.744 = 2.232 ms`, so unhidden time is:

`3.906 - 2.232 = 1.674 ms`

The measured mean park of 2.058 ms is consistent once positions 0-2, which have approximately 0 lead, are included. This validates the model.

At depth 6 all six issue at t=0, so leads by position are:

`0, 0.744, 1.488, 2.232, 2.976, 3.720`

Depth 3 leads are:

`0, 0.744, 1.488, 2.232, 2.232, 2.232`

Gain occurs ONLY at positions 4 (+0.744) and 5 (+1.488); positions 0-3 gain exactly zero.

Mean gain per miss:

`(0.744 + 1.488) / 6 = 2.232 / 6 = 0.372 ms`

Total wall reduction:

`765 x 0.372 = 284.6 ms`

Share of decode wall:

`284.6 / 17885.966 x 100 = 1.59% of decode wall`

Projected tok/s change:

`17.941 / (17.941 - 0.2846) - 1 = 0.0162 = +1.62% tok/s`

This independently corroborates the standing empirical record ("lanes 3/6/10 within ~1%") recorded in serena memory `deepseek_v4/dead_levers`. The REASONING half of that old rejection ("expert_wait is only 4.4%, no I/O stall exists") is void because E130 measured 8.47%; the EMPIRICAL half now has a mechanism behind it.

Loader depth stays dead. No run, no rebuild. E130's `start_slept_calls = 0` also proves pool growth is inert, so raising the compile-time constant would confound a dead variable with a ~1.6% one.

## Experiment: Stage 1 mechanism probe

Cost: 4 arms x ~62 s each ~= 6 minutes total.

Common setup for every arm:

- p256
- 40 tokens
- `--memory-gb 96`
- `COLI_V4_PROFILE=1 COLI_V4_DECODE_TRACE=1`

Arms:

- A0 baseline (pins=16), reproduces E130 counters as an instrument-drift control
- A1 `COLI_V4_PIN_SLOTS=48`
- A2 `COLI_V4_PIN_SLOTS=96`
- A3 baseline replicate, run LAST so drift across the block is bounded by |A3-A0|

The harness MUST assert the requested value was actually applied by parsing `v4_hot_policy pin_slots_per_layer=<N>`, because a silent clamp to `maximum_pins` would make two arms identical and manufacture a false null.

## Pre-registered criteria

These criteria are fixed before any run:

- VALIDITY PRECONDITION: |A3 - A0| on `wait_finish_complete_block total_ms` must be <= 3% (E130's replicate pair achieved 0.61%). If exceeded the probe is VOID: results discarded, not reinterpreted.
- GATE QUANTITY: `wait_finish_complete_block total_ms`, baseline 1488.243 ms. Chosen over the raw miss count because it is the only one of the two that lies on the critical path.
- Band 1: best arm > 1265 ms (less than a 15% cut) => KILL. Record, stop, do not escalate.
- Band 2: 744 - 1265 ms (15-50% cut) => MECHANISM CONFIRMED BUT SUB-RESOLUTION. Record the counters as the result; do NOT spend a tok/s run, it cannot resolve 1.25-4.2%.
- Band 3: <= 744 ms (>= 50% cut, >= 4.16% of wall, projected >= +4.3% tok/s) => ESCALATE to Stage 2 with a fresh approval request.
- REVERSAL GUARD: any arm whose `store_disk_read calls` exceeds 918 (+20% vs 765) is a documented negative - over-pinning is starving the LRU. Record it and stop that direction. A negative is a publishable result, not a failure.
- ATTRIBUTION RULE (binding on any positive verdict): a result may be attributed to the miss-rate/eviction effect ONLY if `store_disk_read calls` fell. If `wait_finish_complete_block` falls while `store_disk_read calls` is flat, the movement is the KERNEL-COVERAGE effect and the correct write-up is "the prior rejection is confirmed on its own terms". Size the confound honestly: expert_forward is 7485.917 ms, and if roughly 55% is scalar and all of it converted at the best-case 1.14x that is `7485.9 x 0.55 x (1 - 1/1.14) = 505.6 ms = 2.83% of wall` - real, and comparable to the miss effect, so attribution is mandatory not optional.

## Non-bit-exactness protocol

Pinning decides whether an expert is served by the rows16 packed kernel or the cold scalar kernel, and E86/E97/E123 recorded a rows16-vs-rows1 summation-order divergence, so output MAY NOT be token-identical.

Per `AGENTS.md`, a changed md5 is NEVER on its own grounds to reject. Required before the words "fails" or "breaks" may be used:

1. Extract `generated_text` from both arms and diff the WORDS, reporting exactly which tokens changed.
2. Run the determinism control: `tokps.sh` at N=5 runs each arm five times and reports deterministic/NONDETERMINISTIC, and if an arm cannot reproduce its own md5 then an ON-vs-OFF difference proves nothing.
3. `.backlog/lab/taskcheck.sh` must score 5/5.

## Why no rebuild and no golden run are needed for Stage 1

E131 makes no code change. It is a pure env-var sweep.

The binary on disk, md5 `7d454de9ba229e0ea083845a7a9e594b`, is the one that PASSED both goldens: `5d04890413ff539e802985ce8c727814` and `cc09015d089d9a25d10d75753f9e849a`.

Record the binary md5 before and after the probe. If unchanged, both golden verdicts still hold by construction.

Caveat: this is strong evidence from mtime ordering, not proof. A golden re-run is available as an optional ~10 minute item if certainty is wanted.

## Results

Stage 1 ran 2026-08-30 22:38-22:43. Binary md5 `7d454de9ba229e0ea083845a7a9e594b` before AND after,
so both golden verdicts still hold by construction. All four arms rc=0 and every arm applied the
pins it requested (no silent clamp): A0/A3 16, A1 48, A2 96.

Logs: `.backlog/lab/decode_trace_A{0,1,2,3}_20260830-22*.log`,
console `.backlog/lab/e131_pinsweep_console.log`.

### Validity precondition: PASS
|A3 - A0| on `wait_finish_complete_block` = |1481.836 - 1520.264| = 38.428 ms = **2.528%**, inside
the pre-registered 3%. The probe is valid. (Wider than E130's 0.61% replicate agreement, so treat
sub-3% differences between arms as unresolved noise.)

### Measured

| arm | pins | misses | disk_ms | wait_block_ms | pack_ms | lock_ms | hit_scan_ms | expert_fwd_ms | rows16 | scalar | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 16 | 765 | 3008.291 | **1520.264** | 697.363 | 471.319 | 44.877 | 7462.544 | 4490 | 10769 | **2.1810** |
| A1 | 48 | 740 | 2761.625 | **1772.257** | 1603.053 | 1043.051 | 232.958 | 7759.977 | 8222 | 7037 | **2.1122** |
| A2 | 96 | 726 | 2670.776 | **1789.314** | 1760.313 | 981.905 | 757.773 | 8007.405 | 11825 | 3434 | **2.0890** |
| A3 | 16 | 765 | 2978.500 | **1481.836** | 686.618 | 462.718 | 44.615 | 7477.785 | 4483 | 10776 | **2.1829** |

Both axes, as required: decode_wall_ms 17881.8 / 18463.5 / 18669.2 / 17866.2; tok/s = 39/decode_sec.

### VERDICT: KILL — and stronger than the gate required, this is a measured NEGATIVE

Best intervention arm is A1 at **1772.257 ms** against the KILL band of >1265 ms. Both pin arms are
WORSE than baseline, not merely insufficient: wait rose **+16.57%** (A1) and **+17.70%** (A2), and
tok/s fell **-3.15%** and **-4.22%**. A3 reproduces A0 to 0.09% on tok/s, so the direction is not noise.

### The hypothesis was half right and still lost

The eviction shield genuinely works. Misses fell 765 -> 740 -> 726 (**-3.27%**, **-5.10%**) and disk
time fell **-8.20%** / **-11.22%**. The attribution rule is therefore satisfied: `store_disk_read
calls` DID fall, so the eviction effect is real and measured, not inferred.

It is swamped by a cost this pre-registration did not anticipate. Pinning forces rows16 packing at
lookup (`should_pack = hot_is_pinned`), and that work runs while `state->mutex` is held:

- `store_pack` **+129.9%** / **+152.4%**
- `store_lock` **+121.3%** / **+108.3%**
- `store_hit_scan` **+419.1%** / **+1588.5%**

The loader thread needs the same mutex to publish its result. So the main thread's own packing
delays the very loads it then blocks on, and `wait_finish_complete_block` rises even though there
are fewer misses to wait for. Fewer, slower-to-satisfy misses beat more, faster-to-satisfy ones.

### Second finding, not part of the hypothesis: more rows16 coverage made expert_forward SLOWER

rows16 share went 29.4% -> 53.9% -> 77.5% of expert calls, and `expert_forward` rose
7462.5 -> 7760.0 -> 8007.4 ms (**+3.99%**, **+7.30%**). The packed kernel is supposed to be the fast
one. This is a direct, in-situ contradiction of the assumption behind the standing "45% ceiling"
lever in `mem:deepseek_v4/dead_levers`, which wants to WIDEN rows16/GPU expert coverage. That lever
should not be pursued on the assumption that coverage alone is a win, at least not on this host.
Note the measurement is confounded: raising pins changes both coverage and lock pressure at once,
so this is a flag for a dedicated experiment, not a finished result.

### What this closes

`COLI_V4_PIN_SLOTS` is now dead on BOTH axes, each for its own measured reason: as a kernel-coverage
lever (the pre-existing 1.00-1.14x head-to-head) and now as a miss-rate lever (this experiment).
Raising it is actively harmful here. AGENTS.md is corrected to say which property each result
retired, so the next session does not re-open it a third time on a fresh rationale.

### Stage 2 deliberately NOT run

The gate returned KILL, so the ~19 minutes budgeted for `tokps.sh` N=5 plus `taskcheck.sh` were not
spent. The non-bit-exactness protocol was therefore not needed for a verdict, but the observation is
recorded: A1 and A2 both produced text md5 `14675beac7a40ba9249639aac58ca58d` against baseline
`d06053793d8f66d4f69a3f7c810441e1`. That divergence is expected from the rows16 summation-order
change and is NOT evidence of breakage; no capability claim is made either way, because a losing
arm does not need one.

### Where the avenue stands after E131

The two attacks named in E130 are now split. Attack (a), cutting the miss rate via pinning, is
closed. What remains of (a) is cache SIZE rather than cache POLICY (`--memory-gb`, untested at this
granularity). Attack (b), hiding more of each miss, still has no cheap lever: loader depth is
rejected on arithmetic above (+1.62%), and this engine has no cross-layer decode lookahead.

The binding constraint remains the ceiling: eliminating every park is **+9.04%**, inside the decode
noise floor. Any future work here must either raise that ceiling or accept that counters, not tok/s,
are the only usable evidence.

