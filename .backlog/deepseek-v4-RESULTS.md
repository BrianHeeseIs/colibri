# DeepSeek-V4 Flash on macOS arm64 — investigation results

Host: Apple Silicon, 137.4 GB RAM, 16 logical / 12 physical cores, macOS 26.6.1.
Model: `deepseek-ai/DeepSeek-V4-Flash-0731`, 166.9 GB, internal NVMe (APFS).
Date: 2026-08-12. All figures measured on this host; nothing extrapolated.

## 1. macOS port — COMPLETE AND VALIDATED

Upstream ships V4 as x86-64/aarch64 Linux + Windows only. Port is **62 lines across 3 files**.

The blocker documented in `Makefile.deepseek-v4:2` — *"Uses GNU ld -Wl,--wrap"* — **does not
exist**. `grep -rn wrap Makefile.deepseek-v4*` returns nothing; actual link flags are
`-lm -fopenmp -pthread`. The comment is stale and was the sole reason the port looked hard.

Real changes:
- `Makefile.deepseek-v4`: Darwin branch — `clang`, `-mcpu=` (not `-march=`), Homebrew libomp,
  `-D_DARWIN_C_SOURCE` (not `-D_GNU_SOURCE`).
- `c/Makefile`: `COLI_V4_SUPPORTED` extended to `AARCH64 + DARWIN`.
- `c/deepseek_v4.c`: ONE portability fix — `_SC_AVPHYS_PAGES` is glibc-only; added an
  `__APPLE__` branch using `host_statistics64`, mirroring `colibri.c:8270-8276`.

Verification (zero model download required):
```
deepseek_v4: Mach-O 64-bit executable arm64, libomp linked
tests/test_deepseek_v4       10 suites   ALL ok
tests/test_v4_ownership       5 tests    ALL ok
make test-c                   FULL suite exit 0 — no regressions
make deepseek-v4-tiny-check   exit 0 — TOKEN-EXACT
coli info                     "DeepSeek V4 Flash · 284B MoE" · engine ready
```
Oracle-validity check: the regenerated fixture differs from the committed one only in
`transformers_version` / `torch_version` strings; `prompt_ids`, `cases`, `quantization_format`
and `seed` are byte-identical. The token-exact result is therefore NOT circular.

## 2. Speculation — n-gram wins, MTP does not

Protocol: 10 fixed prompts on one topic, 24 tokens, `--temp 0`, persistent `coli serve`,
15-min cap, paired per-prompt comparison. Prompt #1 is a cold-start outlier and is excluded
from the headline figures.

| config | vs baseline (paired) | excl. cold start |
|---|---|---|
| `V4_DRAFT=4 V4_NGRAM=1` | +23.5%, wins 7/7 | **+18.5%, wins 6/6** |
| `V4_MTP=1 V4_DRAFT=4` | +9.4%, wins 4/7 | **+3.3%, wins 3/6** |

Upstream defaults both to `0`, citing MTP accepting "10 in 24" (~42%) and one 14-token answer
taking 495 s. On this host MTP reached **70-89% acceptance** and `adaptive_disabled=0` — the
auto-disable at `deepseek_v4.c:7982` never fired. Yet throughput barely moved.

**The upstream default is correct, but for a different reason than stated.** Acceptance is not
the problem here; rejection cost is. Each rejected suffix forces a recurrent-attention replay
(`deepseek_v4.c:7973-7979`), and that tax cancels the saved forward passes even at 83%
acceptance. Faster storage improved acceptance, not the economics.

**Recommended for this host:** `V4_DRAFT=4 V4_NGRAM=1`, leave `V4_MTP=0`.
Prompt-lookup drafting costs no extra RAM, no drafter forward pass, and no DSpark cache.

## 3. Metal — DEFER (measured, not assumed)

Full analysis in `deepseek-v4-metal-design.md`. Summary:

V4 has **zero** Metal support (`deepseek_v4.c`: 0 `COLI_METAL` refs, vs `colibri.c` 70,
`inkling.c` 25; `Makefile.deepseek-v4`: 0 METAL). It cannot be "activated" — it does not exist.

Three reviewers gated a build on profiling (kill <10%, justify >15-20% offloadable share).
Profiled with `/usr/bin/sample`, main-thread critical path, 13.5k samples @1ms:

| | expert-compute | of which OpenMP join-barrier | truly offloadable |
|---|---|---|---|
| cold | 20.0% | 8.8% | ~11.2% |
| **warm** | **14.4%** | **6.4%** | **~8.1%** |

**8.1% < 10% kill threshold.** Most time inside `matmul_mxfp4` is the main thread parked at
`__kmp_join_barrier` while 12 OpenMP workers compute — already parallel, so a GPU would
displace working CPU capacity rather than unlock idle capacity.

Independent correctness blockers (never resolved, from review round 1):
- the existing Metal MoE pipeline computes a DIFFERENT function than V4's CPU path (BF16
  rounding boundaries, `swiglu_limit` clamping, route-weight ordering) — could not be token-exact;
- bit-exactness may be structurally impossible: rows16 CPU accumulates sequentially per row,
  `moe_gemv` tree-reduces across 32 lanes via `simd_sum`;
- `moe_gemv` accepts fmt {1,2,5,6} only — fmt=4 grouped-int4 is explicitly unsupported for
  batched MoE, so weight conversion would not have reused the kernel either.

For scale: **n-gram gives +18.5% for zero code — more than double Metal's theoretical ceiling.**

## 4. RAM — prior negative results RETRACTED as pressure-contaminated

`--ram 96` doubles the planner's cache (80.31 GiB / 150 slots vs 39.62 GiB / 74 slots).

**Every previous `--ram` figure in this document was invalid.** The `-16.8% (0/7 wins)` and
`-10.2% (n=3, 1/3)` results were measured while system memory pressure was uncontrolled, and
they were interpreted using `ps` RSS — which on macOS **excludes compressed pages** and
therefore cannot see this engine's memory. Both the measurements and the reasoning built on
them are withdrawn.

The earlier "swap thrashing retraction" recorded in this section is **itself retracted**: it
rested on "engine RSS was ~47 GB at BOTH configs", which is exactly the unreliable number.
A nested error — a wrong retraction of a possibly-correct explanation, both from the same
measurement trap.

### The measurement trap (record this before anything else)

| number | what it means | trustworthy for |
|---|---|---|
| `ps` RSS | pages currently resident, **excludes compressed** | nothing, alone |
| `footprint -p PID` → `phys_footprint` | dirty-memory charge, **includes compressed** | *retention* |
| `phys_footprint − RSS` | the compression penalty | *residency / speed* |

Measured directly on pid 93003, same process, same `--ram 96`, no restart:

| time | footprint | RSS | compressor (physical) | swap used | note |
|---|---|---|---|---|---|
| 15:14 | 87 GB | 37.4 GB | 48.0 GB | 13.0 GB | apps open, thrashing |
| 15:34 | 92 GB | 90.9 GB | 1.2 GB | 4.6 GB | apps closed |

Footprint tracked the planner target (80.31 GiB cache + 6.27 GiB dense ≈ 87 GB) the entire
time. **The cache was never evicted.** RSS rising 37 → 90.9 GB with footprint flat is
decompression, not allocation.

### Coding-agent workload, `--ram 96`, 128 tok/prompt

| # | tok/s | note |
|---|---|---|
| 1 | 0.1557 | ran under 48 GB compressor, swap 91% full |
| 2 | 0.2471 | pressure easing |
| 3 | 0.3441 | compressor ~0 |
| 4 | 0.3478 | plateau (within 1% of #3) |

Mean excluding the contaminated #1: **0.3130 tok/s**.

**This is NOT a claim that `--ram 96` wins.** Two variables moved together across the run —
memory pressure fell *and* the expert cache warmed — and this workload (4 coding prompts) is
not the 10-prompt chat protocol that produced the original figures. Both comparisons are
unresolved.

### Source audit — the engine is not at fault (5 parallel agents + Oracle)

- **No runtime pressure governor.** `coli_v4_os_available_memory()` (`deepseek_v4.c:674`) is
  called exactly ONCE, from `build_runtime_plan()` at line 987, during engine open. Never in
  the serving or decode loop. No `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`, no
  `os_proc_available_memory`, no `sysctl vm.*`. The budget is fixed at startup.
- **The engine cannot release expert memory mid-run.** There is no `mmap` for weights anywhere
  (only Linux-only io_uring rings in `uring.h`). Experts are `pread()` into `posix_memalign`
  anonymous slabs, freed ONLY in `destroy()`/`destroy_hot()` at shutdown; a miss overwrites the
  victim slot in place. No `madvise`, no `MADV_FREE*`, no `mlock`. This independently
  corroborates the footprint measurement.
- **The macOS port is correct.** The Darwin probe returns `free + inactive + purgeable`,
  mirroring Linux `MemAvailable`. A proposed patch to add `speculative_count` was **discarded**:
  XNU `osfmk/mach/vm_statistics.h` shows it is already inside `free_count`, so it would
  double-count.
- **Linear-scan cost dismissed with arithmetic.** The hot store is a per-layer linear array
  (up to 4 O(`slots_per_layer`) passes per miss) and `slots_per_layer` scales with `--ram`.
  But 76 extra slots × ~301 lookups/token at a generous 50 cycles/check is 0.2–1.5 ms/token,
  against decode measured in *seconds* per token. Not a candidate.
- **Minor defects, not perf-relevant.** `stats.resident_bytes` adds `record_bytes` while the
  allocation is `record_bytes + 8192` (undercount/slot); `policy->packed_slots` increments but
  never decrements on victim overwrite (monotonic telemetry drift).

**Independently re-verified 2026-08-12, not taken on agent trust.** `deepseek_v4.c` contains
TWO expert-store implementations — the hot rows16 store at ~5039-6070 (inside
`#ifdef COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16`, which renames the inlined base copy to
`..._open_base` at :5043 and re-exports the wrapper at :5995) and a second plain copy whose
`destroy`/`open` live at :9406/:9425. Only the first is shipped: `nm` on
`libexec/colibri/deepseek_v4` shows `_coli_deepseek_v4_expert_store_open.hot_operations`
(the static declared inside the rows16 wrapper) and `..._open_base.operations`, while the
:9425 copy's static is named `operations` and appears nowhere in the binary. The audited code
is therefore the code that runs.

All seven slab-free sites (:5363, :5365, :5369, :5447, :9412, :9419, :9497) are inside
`destroy()` or open-failure cleanup. The frees present in the serve/decode range (:7921-:8144)
are `spec_attention_free` snapshots and per-token activation scratch (`batch_hidden`, `state`,
`next`, `hidden`) — **no expert slab is released during serving**.

### Verdict and open questions

Root cause of the original slowdown: **macOS compressing and swapping the engine's retained
anonymous cache** once the working set exceeded what the machine could hold. Engine
self-eviction is **rejected** on both code and measurement evidence.

Still unproven, needs controlled runs:
1. Does `--ram 96` beat `--ram 48` at steady state under a passing pressure gate? It may not —
   if the hot set already fits in 39.62 GiB, the extra cache buys only reclaim risk.
   **STILL OPEN.**
2. ~~How much of the climb was pressure relief vs cache warming?~~ **RESOLVED — see §4b.**

## 4b. Cold vs warm at `--ram 96` — confound RESOLVED (2026-08-12, 15:56-16:45)

`validation/dsv4/coldwarm.sh`: fresh cold engine, 4 prompts, then the SAME 4 prompts again on
the SAME process without restart. Pressure held low throughout. Only variable = cache warmth.
**All 8 rows passed the gate** (compressor 0.8-16.8 GiB, limit 20), so no row is discarded.

| # | cold tok/s | warm tok/s | delta | cold gap | warm gap |
|---|---|---|---|---|---|
| 1 | 0.2555 | 0.3404 | **+33.2%** | 0.3 GB | 1.1 GB |
| 2 | 0.2896 | 0.3616 | **+24.9%** | 0.2 GB | 6.8 GB |
| 3 | 0.3798 | 0.4571 | **+20.4%** | 15.2 GB | 2.1 GB |
| 4 | 0.4252 | 0.4281 | +0.7% | 2.3 GB | 11.0 GB |

**cold mean 0.3375 → warm mean 0.3968 tok/s, +17.6%, 4/4 paired wins.**

### What this settles

**Cache warming alone produces the climb.** The cold pass rose 0.2555 → 0.4252 (+66%) with the
compressor at 1.1/1.1/0.8/16.8 GiB — i.e. essentially no pressure change. The earlier run's
0.1557 → 0.3478 climb therefore did NOT require pressure relief to explain it; warming is
sufficient. Pressure relief was a second, separate effect.

**Both effects are real and now separable.** Comparing prompt-for-prompt against the
pressure-contaminated run of the same workload:

| # | contaminated | clean cold | compression tax |
|---|---|---|---|
| 1 | 0.1557 | 0.2555 | **-39%** |
| 2 | 0.2471 | 0.2896 | -15% |
| 3 | 0.3441 | 0.3798 | -9% |
| 4 | 0.3478 | 0.4252 | -18% |

**The warm advantage decays to zero by prompt 4** (+33% → +25% → +20% → +0.7%). By its fourth
prompt the cold pass has populated the working set and caught up. So "warm" is worth ~25-33%
on the first few prompts of a session and nothing thereafter — the hot set for this workload
saturates within roughly 3 prompts.

**Steady-state throughput at `--ram 96` on the coding workload is ~0.42-0.46 tok/s.** This is
the first figure on this host measured with the pressure gate green for every row.

### Still not answered

This says nothing about `--ram 48`. The comparison needs the same protocol on the other arm.
Do NOT compare 0.3968 against the 10-prompt chat figures (`spec_ngram` 0.2295) — different
workload, different token count, different protocol.

Rejected fixes (Oracle): broad `mlock` of the cache is **dangerous** — ~92 GB approaches the
~109 GiB (85% of RAM) wire limit, and wired pages deny macOS reclaim entirely, forcing
everything else to compress harder. A `mmap` rewrite is **unproven** on macOS: file-backed
clean pages would degrade more gracefully in principle, but llama.cpp issue #9244 measured
mmap at 730 MB/s vs 6.4 GB/s with `--no-mmap` on M3 Max, and this engine's `pread + F_NOCACHE`
design is deliberate. A runtime pressure governor is **pointless without a shrink path**, which
the engine does not have.

Prior art confirms the inversion is real elsewhere: `moe-l2` measured whole-pin 84 GB → 30.9 t/s
vs selective-pin 26.8 GB → 34.67 t/s vs on-demand 17.5 GB → 35.96 t/s.

### Minimum valid protocol for any future `--ram` A/B

1. Fresh engine process per arm (the plan is fixed at open).
2. Identical workload, token limit, sampling, speculation, thread count.
3. Declare cold-start or steady-state warm — never mix.
4. **Pressure gate before each timed run**: `footprint − RSS` ≈ 0, compressor low, swap stable.
5. Paired, ABBA or randomized order — never all-A-then-all-B.
6. Report `phys_footprint`, RSS, the gap, compressor size, swap used, swapout delta *with*
   every throughput number.
7. **Discard any pair where an arm entered compression** — it became a pressure test.

Caveat: this host carries 63.2 GB of cumulative swapouts. Swap history can poison later runs;
a reboot may be required before a genuinely clean A/B.

## 4c. `--ram 48` vs `--ram 96` — ANSWERED (2026-08-12, 16 rows, 0 gate failures)

Same `coldwarm.sh` protocol as §4b: fresh engine per arm, identical 4 coding prompts, 128 tok,
`V4_DRAFT=4 V4_NGRAM=1`, pressure gate checked before every prompt.

### Cold pass — paired, both arms gate-clean

| # | `--ram 96` | `--ram 48` | delta |
|---|---|---|---|
| 1 | 0.2555 | 0.2211 | −13.5% |
| 2 | 0.2896 | 0.2006 | −30.7% |
| 3 | 0.3798 | 0.2963 | −22.0% |
| 4 | 0.4252 | 0.2909 | −31.6% |
| **mean** | **0.3375** | **0.2522** | **−25.3%** |

**4/4 paired losses for `--ram 48`. Zero gate failures in either arm.** Every row in both arms
showed a footprint-minus-RSS gap of ~0.2–0.5 GB and a compressor under 1.1 GB, so **no row was
touched by compression**. This is a pure cache-capacity effect, not the memory-pressure
artifact that invalidated the original −16.8%.

### What it answers

The original question — *does more RAM help?* — gets its first clean answer, and it is **yes**.

It also **contradicts Oracle's specific counter-hypothesis** that the hot set might already fit
in `--ram 48`'s 39.62 GiB, making the extra cache pure reclaim risk. On this 4-prompt coding
workload it does not fit: the smaller cache is 25.3% slower with the machine quiet in both arms.

### Caveat that must travel with this number

**The arms were not interleaved.** `--ram 96` ran 15:56–16:45; `--ram 48` ran 17:09–17:45.
Between them a user re-login drove the compressor to 78 GB (§9). Protocol step 5 above calls for
ABBA or randomised paired order and explicitly warns against all-A-then-all-B — which is exactly
what this is. The comparison is far stronger than the original (per-row gating, zero
compression, paired prompts) but it is **not fully protocol-compliant**. Re-running one arm
interleaved would close that gap.

Also unchanged: n=4 per arm, one workload, one host. Directional and well-controlled, not
definitive.

### Warm pass — the gap WIDENS

| # | `--ram 96` | `--ram 48` | delta |
|---|---|---|---|
| 1 | 0.3404 | 0.2203 | −35.3% |
| 2 | 0.3616 | 0.2379 | −34.2% |
| 3 | 0.4571 | 0.3012 | −34.1% |
| 4 | 0.4281 | 0.2813 | −34.3% |
| **mean** | **0.3968** | **0.2602** | **−34.4%** |

I framed this as a binary — gap *narrows* or *holds*. **It did neither: it widened.**

| | `--ram 96` | `--ram 48` | 96 advantage |
|---|---|---|---|
| cold mean | 0.3375 | 0.2522 | **+33.8%** |
| warm mean | 0.3968 | 0.2602 | **+52.5%** |

**8/8 paired losses for `--ram 48` across both passes. 16 rows, 0 gate failures, every row with
a 0.2–0.5 GB footprint-RSS gap and compressor under 1.1 GB.**

The warm deltas are also far tighter than the cold ones: −35.3 / −34.2 / −34.1 / −34.3 (spread
1.2pp) versus cold's −13.5 / −30.7 / −22.0 / −31.6 (spread 18pp). Steady state converges hard
on a **~34% penalty**.

### Mechanism: the small cache cannot retain, so it barely warms

Within-arm warm benefit (same prompt, same process, cold → warm):

| | #1 | #2 | #3 | #4 | mean |
|---|---|---|---|---|---|
| `--ram 96` | +33.2% | +24.9% | +20.4% | +0.7% | **+17.6%** |
| `--ram 48` | −0.4% | +18.6% | +1.7% | −3.3% | **+3.2%** |

`--ram 96` gains 17.6% from warming; `--ram 48` gains 3.2% — barely anything. Prompt #1 at
`--ram 48` replays at **exactly** its cold speed (0.2211 → 0.2203), meaning its experts were
already evicted while the cold pass worked through prompts 2–4. 39.62 GiB cannot hold this
workload's hot set, so the second pass is not meaningfully warm. That is *why* the gap widens:
the larger cache compounds its advantage as the session continues, while the smaller one keeps
paying for re-reads.

### Verdict

**More RAM helps, and helps more the longer the session runs.** Oracle's counter-hypothesis —
that the hot set might already fit in 39.62 GiB, making extra cache pure reclaim risk — is
**contradicted on this workload**. The original `-16.8%` finding was not merely
pressure-contaminated; it had the sign backwards.

Standing caveats: n=4 per arm, one workload, one host, and the arms were **not interleaved**
(protocol step 5 violated — see the caveat above). The robustness point from §9 still stands
independently: `--ram 96`'s 96 GB working set gets compressed the moment the machine gets busy,
while `--ram 48`'s 47 GB does not. Fastest when quiet ≠ safest when shared.

### Third point: `--ram 64` (incidental, from the prefetch A/B control arm)

The prefetch A/B (§4d) runs its control arm at `--ram 64` with prefetch OFF — the same 4 prompts
and same harness, so its cold pass doubles as a third point on the RAM curve.

| # | `--ram 48` | `--ram 64` | `--ram 96` |
|---|---|---|---|
| 1 | 0.2211 | 0.2306 | 0.2555 |
| 2 | 0.2006 | 0.2490 | 0.2896 |
| 3 | 0.2963 | 0.3153 | 0.3798 |
| 4 | 0.2909 | 0.2977 | 0.4252 |
| **mean** | **0.2522** | **0.2732** | **0.3375** |
| vs 48 | — | **+8.3%** | **+33.8%** |

**Monotonic at every one of the 4 prompt positions. 12 rows, 0 gate failures across all three
budgets.** This confirms §4c is not a two-point artifact.

The scaling is **non-linear**: +33% more cache (48→64) buys +8.3%, but +100% (48→96) buys
+33.8%. The return accelerates, which is what you expect when the hot set is substantially
larger than 64 GB can hold — the cache only starts paying off properly once it can retain a
meaningful fraction of it.

Caveat: the three arms ran at different times (96 at 15:56, 48 at 17:09, 64 at 18:37) so ambient
state differed. Every row is gate-clean, evidenced by the compressor column (`vm_stat`) reading
0.6–1.1 GB throughout. **Do NOT cite the footprint-RSS gap for the `--ram 64` rows** — 2 of its 8
rows recorded a failed measurement as `0.0` (see the measurement-bug note in §4d). The gap
evidence is valid for `--ram 48` and `--ram 96`, which have zero affected rows.

### Retention is a THRESHOLD, not a gradient

Cold throughput scales smoothly with cache size, but the *warm* benefit does not:

| budget | cold mean | warm mean | warm benefit |
|---|---|---|---|
| `--ram 48` | 0.2522 | 0.2602 | **+3.2%** |
| `--ram 64` | 0.2732 | 0.2719 | **−0.4%** |
| `--ram 96` | 0.3375 | 0.3968 | **+17.6%** |

48 and 64 are both at noise level — and not even monotonic with each other, which is itself
evidence they are measuring nothing. The `--ram 64` per-prompt deltas were +0.7 / +0.8 / −1.0 /
−1.8%, scattered around zero. Only `--ram 96` shows a real second-pass gain.

**Two different effects, often conflated:**

1. **Throughput per prompt** scales continuously with cache size (0.2522 → 0.2732 → 0.3375
   cold). More cache always helps somewhat, because more of each prompt's experts are already
   resident.
2. **Retention across prompts** is threshold-like. Below ~64 GB the working set of prompts 2–4
   evicts prompt 1's experts before it can be replayed, so a session never "warms up". Somewhere
   between 64 and 96 GB the cache becomes large enough to hold the whole rotation.

Practical consequence: an intermediate budget makes each prompt faster but will **not** make a
long session get faster as it runs. Only crossing the retention threshold does that — and on
this host that costs ~94 GB resident, which §4d shows cannot be sustained under normal desktop
load. The honest recommendation therefore depends on the machine's other duties, not on
throughput alone.

Caveat: n=4 per arm; the ±1–3% figures at 48/64 are within run-to-run noise, so the correct
claim is "no measurable warm benefit below 96 GB", not "exactly zero".

## 4d. `COLI_V4_EXPERT_PREFETCH=1` — arm ABORTED, contaminated (2026-08-12 18:19–18:34)

V4's analogue of GLM's `PILOT` (router-lookahead prefetch), off by default (`deepseek_v4.c:3270`),
documented nowhere (§6b). Ran at `--ram 96` to compare against the gate-clean baseline in §4b.

**Env reached the server correctly** — verified in the process environment, not assumed:
`V4_DRAFT=4 V4_NGRAM=1 COLI_V4_EXPERT_PREFETCH=1`. The server launches into a separate tmux
pane, so an exported variable would NOT have propagated; it must be inline in the `send-keys`
string. Getting this wrong would have produced a null result that looked real.

### One clean row before contamination

| | wall | tok/s | gap | comp |
|---|---|---|---|---|
| baseline cold #1 (prefetch OFF) | 501s | 0.2555 | 0.3 GB | 1.1 GB |
| prefetch cold #1 (ON) | 555s | 0.2306 | 0.3 GB | 0.6 GB |

−9.7%, **n=1, inconclusive**. Prompt #1 is the noisiest row measured today (0.2211–0.2555 across
three arms). Preserved as `coldwarm_cold_ram96_pf1.ABORTED.csv`.

### Why it was aborted

Mid-run the compressor went **0.6 GB → 44.7 GB** in ~3.5 minutes:

```
engine       fp 94 GB, rss 52.5 GB   -> 41.5 GB of the engine compressed
free         0.1 GB
non-engine   18.0 GB across 663 procs   (Chrome + QEMU + opencode returned)
```

A 94 GB engine plus ~18 GB of desktop load exceeds what 137 GB holds uncompressed. Comparing a
compressed run against an uncompressed baseline measures **memory pressure, not prefetch** — the
exact error that produced the bogus −16.8%. Teardown restored free 0.1 → 100.7 GB and compressor
44.7 → 0.8 GB, confirming for the third time that the compressor held the engine's own pages.

### Gate flaw found and FIXED

`coldwarm.sh` sampled the compressor **only before** each prompt and stamped the row `ok`.
Cold #2 started at 0.6 GB and then ran its entire duration under 40+ GB of compression — it would
have been recorded as clean. `coldwarm2.sh` now samples **before AND after** and fails the row if
either end breaches the limit; the CSV carries `comp_start_gb` and `comp_end_gb`.

**Any earlier row stamped `ok` is only guaranteed clean at its start.** The §4b/§4c rows survive
this doubt because their footprint-minus-RSS gaps stayed at 0.2–0.5 GB, which is an
end-of-prompt measurement and independently proves no compression occurred.

### Robustness consequence

`--ram 96` **cannot reliably run gate-clean on this host under normal desktop load.** The §4b/§4c
arms succeeded because the machine was quiet. This makes the §9 tension concrete: `--ram 96` is
**+52.5% faster warm** but needs ~94 GB resident; `--ram 48` needs ~47 GB and never compressed in
any run today.

### Re-run: controlled A/B at `--ram 64` — VERDICT: prefetch is a NULL

`prefetch_ab.sh` ran both arms **back-to-back** at `--ram 64` (fits alongside desktop load:
~82 GB of 137 GB), fresh engine per arm, identical prompts, with the fixed before-and-after gate.
This also removes the non-interleaved objection that caveats §4c.

| pass | # | OFF | ON | delta |
|---|---|---|---|---|
| cold | 1 | 0.2306 | 0.2302 | −0.17% |
| cold | 2 | 0.2490 | 0.2500 | +0.40% |
| cold | 3 | 0.3153 | 0.3107 | −1.46% |
| cold | 4 | 0.2977 | 0.2956 | −0.71% |
| **cold mean** | | **0.2732** | **0.2716** | **−0.6%** |
| warm | 1 | 0.2323 | 0.2319 | −0.2% |
| warm | 2 | 0.2510 | 0.2500 | −0.4% |
| warm | 3 | 0.3122 | 0.3099 | −0.7% |
| warm | 4 | 0.2922 | 0.2984 | +2.1% |
| **warm mean** | | **0.2719** | **0.2726** | **+0.2%** |

**16 rows, 0 gate failures, compressor 0.8→0.8 GB on every single row.**

`COLI_V4_EXPERT_PREFETCH=1` produces **no measurable effect**: −0.6% cold, +0.2% warm, deltas
scattering both directions with no trend. Wall times differ by 1–6 s on 7–9 minute prompts.
**Leave it at its default of 0.** Consistent with the README's warning that GLM's `PILOT`
equivalent "can be net negative on disk-saturated hosts — measure on yours".

**This also vindicates discarding the first attempt.** That arm showed −9.7% on prompt #1 and
would have made prefetch look actively harmful; under the controlled design the same prompt
shows −0.17%. The −9.7% was entirely baseline drift, not prefetch.

### Measurement bug found in the harness (affects gap evidence only)

`fp_gb()`/`rss_gb()` silently returned **0** when `footprint`/`ps` momentarily failed, and 0 is
indistinguishable from a real reading. Six rows across the `--ram 64` arms recorded
`fp=0 rss=0.0 gap=0.0`, and that 0.0 was initially cited as evidence of no compression when it
was a failed measurement.

- Affected: `cold_ram64_pf0` 1/4, `warm_ram64_pf0` 1/4, `cold_ram64_pf1` 3/4, `warm_ram64_pf1` 1/4.
- **Unaffected: all `--ram 48` and `--ram 96` CSVs — 0/4 each.** The §4b/§4c gap evidence stands.
- **The conclusion is unchanged** because the compressor column comes from `vm_stat`, an
  independent source, and reads 0.8→0.8 GB on all 16 rows.

Fixed in `coldwarm2.sh`: retry up to 3×, then emit the sentinel `NA` — never `0` — so a failed
reading is visible in the data instead of masquerading as a clean one. `GAP` is `NA` whenever
either input is `NA`.


## 5. Learning cache — design approved, NOT implemented

V4 writes `.coli_usage` from exactly ONE site (`deepseek_v4.c:5923`, inside `destroy_hot`);
GLM writes from FIVE (`colibri.c:6742, 7059, 7421, 7589, 7594`), with the comment
*"la cache che impara non deve aspettare l'uscita"* — "the learning cache must not wait for exit".
V4 does precisely what that warns against: nothing persists until graceful engine destroy, so
SIGINT/crash/OOM discard the session, and the `history_total >= 5000` seed threshold can never
be reached across restarts.

V4 is also the ONLY engine with a private cache implementation — `inkling.c`, `kimi_k3.c` and
`olmoe.c` all use shared `route_trace.h`. The README names this defect class explicitly.

Design at `deepseek-v4-cache-design.md`: **unanimously approved** after 5 review rounds
(Oracle PASS, Momus OKAY). Scope: 1 exported helper, 1 static epilogue helper, 1 test-only
wrapper, 1 production call site, 1 fault-injection flag, 1 test.

## 6. Upstream-worthy defects found

1. **ExFAT `._*` scanner bug.** macOS AppleDouble sidecars are globbed as safetensors shards
   (`st.h:414-437`) — `bad safetensors header length 2199142139136 (file 4096 bytes)`. Hit
   with GLM-5.2 on an ExFAT volume; 150 such files. Any macOS user on ExFAT/FAT will hit it.
2. **Stale `-Wl,--wrap` comment** (`Makefile.deepseek-v4:2`) — blocked the macOS port for a
   dependency that does not exist in the file.
3. **Stubbed profiler.** `COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE` has 5 call sites and 2
   declarations (`deepseek_v4.c:3550-3551`) but **zero definitions**, and appears in no
   Makefile. Enabling it fails to link — proven: compiling `COLI_V4_UNIT_BLOCK_HYBRID` with
   the flag yields undefined `_coli_v4_block_profile_add` / `_coli_v4_block_profile_now`.
4. **`resource_plan.py` mis-reports V4.** `EXPERT_RE` matches only GLM's
   `model.layers.N.mlp.experts.M.` naming; V4 uses `layers.N.ffn.experts.M.`, so all 1542
   expert tensors per shard are counted as dense -> "166.9 GB dense, cap 0/layer" -> spurious
   `coli doctor` FAIL. Platform-independent; affects Linux too.
5. **`docs/tuning.md` is a GLM document that never says so, and V4's own knobs are documented
   nowhere.** Verified by grepping both engines. Of the 25 knobs in `tuning.md`, exactly ONE
   (`RAM_GB`) exists in `deepseek_v4.c`. `PILOT`/`PILOT_REAL`/`PILOT_TWO`, `DIRECT`, `PIPE`,
   `PIPE_WORKERS`, `URING`, `PIN`, `PIN_GB`, `AUTOPIN`, `CAP_RAISE`, `REPIN`, `DRAFT`,
   `GRAMMAR`, `SPEC_PIN`, `THINK`, `TF`, `KVSAVE`, `CACHE_ROUTE`, `COLI_NUMA`, `IDOT`,
   `COLI_CUDA`, `COLI_NO_OMP_TUNE`, `COLI_OMP_TUNED` all have **0 references** in V4 — they
   silently no-op. Nor are `--temp`, `--topp`, `--ngen`, `--repin`, `--policy` or `--auto-tier`
   V4 CLI flags. Conversely every V4 knob greps to **0 hits** in `tuning.md`,
   `deepseek-v4.md` AND `ENVIRONMENT.md`. Most damaging: a V4 user has no documented route to
   `V4_NGRAM`, measured here at **+18.5%** for zero code. This is the same defect class the
   README names — a mechanism landing in one engine and never reaching its siblings' docs.

## 6b. V4 runtime knob inventory (from source, since no doc covers it)

Extracted from every `getenv()` in `deepseek_v4.c`. Defaults matter: several are the OPPOSITE
of the GLM knob with the similar name.

| knob | default | effect |
|---|---|---|
| `RAM_GB` / `--memory-gb` / `--ram` | auto | expert cache budget; the single most valuable knob |
| `COLI_V4_DIRECT` | **ON** | O_DIRECT/F_NOCACHE expert reads. GLM's `DIRECT` is opt-in — V4 inverts it. Disable with `=0` |
| `COLI_V4_AUTOPIN` | **ON** | load `.coli_usage` and pin hot experts. Disable with `=0` |
| `COLI_V4_SAVE_USAGE` | **ON** | persist `.coli_usage`. Disable with `=0` |
| `COLI_V4_EXPERT_PREFETCH` | **OFF** | V4's analogue of GLM `PILOT`. **Untested on this host** |
| `COLI_V4_PREWARM` | **OFF** | prewarm cache from history at startup — see the dead-chain note below |
| `COLI_V4_MARKOV_SPEC` / `_BLOCK` / `_KEEP` | OFF | DSpark markov drafter, opt-in pending acceptance data |
| `V4_DRAFT` | 0 | draft depth; **gates both speculation paths** |
| `V4_NGRAM` | ON *but inert unless `V4_DRAFT>0`* | prompt-lookup drafting — **+18.5% measured** |
| `V4_MTP` (+ `_DRAFT`/`_GB`/`_MIN`/`_PARTIAL_KEEP`) | OFF, needs `V4_DRAFT>0` | native MTP; measured +3.3%, not recommended |
| `V4_PREFIX_LOG` | OFF | logs prompt-token reuse |
| `CTX` / `NGEN` | 4096 / 1024 | context and max new tokens |
| `SERVE`, `SNAP` | — | serve mode / snapshot |

### `COLI_V4_PREWARM` is dead in practice — and §5 is the unlock

`COLI_V4_PREWARM=1` exists to prewarm the expert cache from usage history at startup, which is
precisely the cold-start penalty quantified in §4b (**+33% on prompt 1, +17.6% overall**). It
cannot currently fire:

```
prewarm requires   policy->history_seeded            (deepseek_v4.c:6068)
history_seeded  =  history_total >= 5000             (deepseek_v4.c:6054)
.coli_usage written from ONE site, inside destroy_hot (deepseek_v4.c:5923)
.coli_usage on models/deepseek-v4-flash              ABSENT (confirmed)
```

History is written only on graceful engine destroy, and every harness run `pkill`s the engine,
so `history_total` never leaves 0 and the 5000 threshold is unreachable across restarts —
exactly the defect §5 describes. **Implementing the approved cache-flush design turns
`COLI_V4_PREWARM` from dead code into the fix for the measured cold-start penalty.** That
raises its value above "correctness tidy-up".

Two knobs worth measuring once the current arms finish: `COLI_V4_EXPERT_PREFETCH=1` (untested
prefetch, off by default) and `COLI_V4_PREWARM=1` (only meaningful after history persists).

## 7. Status

| item | state |
|---|---|
| macOS port | complete, token-exact, no regressions |
| n-gram speculation | **+18.5%**, recommended |
| MTP speculation | measured, not recommended |
| Metal | DEFER — measured ceiling below kill threshold |
| High RAM | **ANSWERED: `--ram 96` beats `--ram 48` by +33.8% cold, +52.5% warm** (8/8 paired, 16 rows, 0 gate failures). Prior negatives retracted — sign was backwards |
| Cache flush | design approved, **implementation not started** |
| Commits | **none** — 47 changed/untracked, 0 staged |

## 8. Measurement discipline (added 2026-08-12 after two misreads)

`ps` RSS was trusted twice in this investigation and produced two wrong conclusions in
sequence — first "experts are not being retained" (they were; the pages were compressed), then
a retraction of the swap explanation that was itself built on the same blind number.

Standing rule for this host: **never report a memory conclusion from RSS alone.** Always pair
`phys_footprint` (retention) with RSS (residency) and treat the gap as the compression penalty.
`validation/dsv4/memmon.sh` enforces this — it prints footprint first, labels RSS
*"excludes compressed, do not trust"*, and warns when the compressor exceeds 20 GB.

## 9. Operational: an idle engine gets compressed, and a restart reclaims it

Observed 2026-08-12 between runs. After the cold/warm run finished at 16:45 the engine sat
idle. A user re-login (session restart: Dock, Finder, Chrome, agents) created modest pressure,
and macOS compressed the idle process:

| | before | after re-login | after engine teardown |
|---|---|---|---|
| compressor (physical) | 1.2 GB | **78.0 GB** | **0.7 GB** |
| free | ~12 GB | 0.3 GB | 56.2 GB |
| engine footprint / RSS | 92 / 90.9 GB | 96 / 22.2 GB | — |

The 78 GB was **the engine's own cache**, not other applications: total non-engine RSS was only
**15.0 GB across 613 processes**, while the engine's footprint-minus-RSS gap was ~74 GB.
The compressor did not drain on its own (78.0 GB flat across three samples 10 s apart) because
nothing was touching those pages — decompression happens on access.

Consequences for running experiments on this host:

1. **A long-idle engine is not a warm engine.** Its cache may be compressed, so the first
   prompt after an idle gap pays decompression and looks like a cold start.
2. **Killing the engine is the fastest way to clear the gate.** `coldwarm.sh` tears down before
   loading, which took the compressor 78.0 → 0.7 GB and free 0.3 → 56.2 GB in under three
   minutes. No other application had to be closed.
3. **Always gate-check immediately before a timed run**, never on state observed minutes
   earlier. `coldwarm.sh` logs `pre-start compressor:` for exactly this reason.

## 10. HAZARD: `bin/coli` and `c/coli` resolve to DIFFERENT engine binaries

Discovered 2026-08-12 21:15 while verifying the cache-flush implementation.

```
bin/coli  ->  libexec/colibri/deepseek_v4   (built 00:57)   STALE
c/coli    ->  c/deepseek_v4                 (built 20:53)   current
```

`engine_for()` (`c/coli:249`) returns `HERE/deepseek_v4` when it exists, else the `libexec`
copy. `HERE` is the launcher's own directory, so the two launchers pick different binaries
whenever `libexec/` has not been refreshed by `make install`.

**Every experiment harness uses `bin/coli`** (line numbers current as of the guard insertion
below, which shifted them by ~10):

| harness | launches | binary |
|---|---|---|
| `coding_agent.sh:25` | `$BASE/bin/coli serve` | libexec, **stale** |
| `coldwarm.sh:107` | `$BASE/bin/coli serve` | libexec, **stale** |
| `coldwarm2.sh:136` | `$BASE/bin/coli serve` | libexec, **stale** |
| `run_experiment.sh:25` | `$BASE/bin/coli serve` | libexec, **stale** |
| `t2_serve_flush_qa.sh:27` | `python3 c/coli serve` | current ✓ |
| `t4_loop_closes.sh:28` | `python3 c/coli serve` | current ✓ |

The two QA scripts use `c/coli` per the design plan's rule that QA must use the in-repo
launcher — which is the only reason those runs exercised the new code.

Determined by reading the actual invocation lines. A first pass with `grep -l` put five files
in *both* columns, because it matched a comment naming both launchers and the
`pkill -f 'coli serve'` cleanup lines. Pattern-matching was not sufficient here.

### Why this matters for the outstanding T4 benefit gate

Running T4 through `coldwarm*.sh` as-is would launch the **00:57 binary, which contains no
per-turn flush at all**. It would show no hit-rate improvement, and that null would read as a
genuine kill verdict for the persistence feature while actually measuring code that never
contained it.

This is the third instance of the same failure shape today: the stale `.o` that made a fixed
`main()` collision look unfixed, the `libexec`-vs-`c` pgrep pattern that reported a live engine
as 0 procs, and now this. **The common cause is asserting which artifact is under test instead
of measuring it.**

**Before any further V4 experiment, assert which binary is under test.** Either refresh the
install artifact (`make -C c install`, or copy `c/deepseek_v4` to `libexec/colibri/`), point
the harness at `c/coli`, or set `COLI_ENGINE` explicitly. A one-line guard in each harness —
print the resolved engine path and its mtime at startup — turns this from silent into obvious.

## 11. Budget ceiling: ~100 GB total, not what the naive formula says

Established 2026-08-12 by getting it wrong twice in one night.

I sized `--ram` with `engine_footprint + non_engine_RSS + 8 GB OS <= 137`. That formula said
`--ram 80` would fit at 124 GB. It did not: the engine filled to its full 80 GB budget and
macOS immediately compressed **38 GB of it** (rss collapsed 80 → 41.6 GB) while also squeezing
other applications from 29.9 → 16.8 GB. Compressor went 0.7 → 47.2 GB mid-run and the arm had
to be discarded.

| budget | engine fp | + non-engine | total | outcome |
|---|---|---|---|---|
| `--ram 64` | ~64 GB | ~31 GB | **~95 GB** | ran clean, gap 0.4 GB at full budget |
| `--ram 64` | ~64 GB | ~18 GB | **~86 GB** | ran clean, 4/4 rows gate-green |
| `--ram 80` | ~80 GB | ~30 GB | **~110 GB** | **compressed 38 GB of the engine** |
| `--ram 96` | ~94 GB | ~18 GB | ~112 GB | compressed (aborted prefetch arm, §4d) |

**Working rule: keep `engine + non-engine` at or under ~100 GB on this 137 GB host.** The
missing term in the naive formula is that macOS needs far more than 8 GB of slack — wired
memory, the compressor's own working set, and file-backed pages it is unwilling to drop all
compete before it will leave a large anonymous allocation resident.

Consequence: `--ram 96` is only reachable when the machine is genuinely quiet (non-engine
under ~15 GB), which is how §4b/§4c were measured. At a normal working desktop load of 18-31 GB,
**`--ram 64` is the largest budget that reliably stays uncompressed.**

### Absolute throughput is not comparable across sessions

The same `--ram 64` cold prompt #1 measured 0.2306 tok/s at 19:51 and 0.1798 tok/s at 01:24 —
**22% apart with no configuration change**, both gate-green with 0.4 GB gaps. Ambient load
alone moved it. Only paired within-session deltas are valid; any cross-session comparison of
absolute tok/s in this document should be treated as indicative, not measured.

---

## 12. Metal GPU backend — built, proven correct, and MEASURABLY SLOWER (2026-08-13)

A Metal backend for the V4 routed-expert path was built end-to-end and measured on the real
model. **Verdict: it does not make V4 faster on this hardware through an expert-level seam.**
Correctness is perfect; performance regresses. The reasons are structural, and the honest fix is
an engine change rather than a backend one.

All numbers below are **paired within-session** measurements per §8/§11 discipline: same binary,
`.coli_usage` restored from a frozen 29127 B / 14190-selection snapshot before every arm, and
`COLI_V4_SAVE_USAGE=0` so neither arm mutates history. Absolute values are not comparable across
sessions; only the within-pair deltas are.

### 12a. End-to-end results

Decode, 8 tokens, `--ram 96`, prompt "The capital of France is":

| arm | ttft | generation | dispatches | output |
|---|---|---|---|---|
| Metal OFF | 9.850–10.376 s | **6.560–6.980 s** | 0 | `**Paris**.` |
| Metal ON (final) | 11.241–11.454 s | 7.805–7.992 s | 2391 | `**Paris**.` |

**Best achieved: 1.118x SLOWER on generation.** Output byte-identical in every paired run.

Prefill, 799-token prompt:

| arm | ttft | dispatches | GPU coverage |
|---|---|---|---|
| Metal OFF | **708.143 s** | 0 | — |
| Metal ON | 753.030 s | 36740 | 36740 / 206916 = **17.8%** |

**1.063x slower on prefill as well.**

Progression across three architectural fixes: **2.30x slower → 1.17x → 1.118x.** Diminishing
returns are obvious; the trend does not extrapolate to a win.

### 12b. Why — four causes that stack against the GPU

1. **Every expert call is S=1.** The seam is
   `coli_v4_metal_expert_forward(float *out, const ColiExpertView*, const float *input, ...)` —
   one input vector, one output vector. There is **no token dimension to batch over**.
2. **Prefill does not help.** 799 tokens x 43 layers x 6 top-k = 206,142 ≈ the 206,916 observed
   `expert_requests`. The engine walks prefill **token by token through the same S=1 seam**, so
   prefill is merely *more* batch-1 calls. Arithmetic intensity never rises.
3. **Unified memory gives the GPU no bandwidth advantage.** Both processors read the same DRAM.
   This workload is bandwidth-bound, not compute-bound.
4. **~Zero weight reuse.** Top-6 of 256 experts, ~8 MB of weights per ~0.5 MFLOP of work. Plus the
   GPU pays per-call dispatch and a synchronous wait that the CPU path does not.

The isolated matmul microbenchmarks *did* show real GPU wins at real dims — **7.26x at batch-8**
(gate/up 4096→2048, n=20, mean±sd, bit-exact). That result is genuine but **structurally
unreachable through this seam**, because the seam never presents more than one token.

### 12c. What would actually be needed

Batch the token dimension **inside** the expert call: gather all tokens routed to expert E within a
layer and issue one S=N matmul. That is a change to the MoE scheduler (`moe_token_pipeline`,
`deepseek_v4.c:3579`), **not** to the backend — and it is precisely the shape the 7.26x
microbenchmark measured. Until that exists, no backend tuning changes the sign.

Backlog items that were deliberately **not** pursued because they move the margin and not the sign:
GPU coverage is only 17.8% (rows16-packed pinned experts are rejected by cold-layout validation —
`v4_rows16 packed_slots=491`, `pin_slots_per_layer=16`); a `simd_sum` variant; and two
order-preserving occupancy levers (multiple output rows per thread, `uchar4` vectorized loads).
Each would make the GPU faster and more-used; none removes S=1 or unified-memory bandwidth.

### 12d. What was established, and is reusable

Correctness, all verified at **production reduction length** (gate/up I=4096 with ng=128; down
I=2048 with ng=64) — per §8, every parity claim states the reduction length it was measured at:

| component | result |
|---|---|
| `mx4_scale`, `e4m3fn` decode | **bit-exact, exhaustive 256/256** |
| `e4m3fn` encode | bit-exact across both arms (subnormal <0.015625, and normal with round-to-nearest-even incl. the `rounded==16` exponent carry) |
| `bf16_round` | bit-exact incl. NaN, ±Inf, denormals, round-to-even ties |
| `fp8_activation_qdq` | bit-exact incl. subnormal-encode, near-448 clamp, 1e-4 floor |
| `matmul_mxfp4` ordered / xcache / hot_xcache | **bit-exact at I=4096, ng=128** (real reduction length, O=64). Tail (I=48) and ragged-tile (O=17) shapes verified bit-exact **separately at small dims** — the two results should not be conflated: no ragged-O case was run at I=4096 |
| fused single-dispatch expert kernel | **bit-exact, 12,288 outputs / 3 real seeds / 0 differing bits** |
| host glue vs `coli_v4_expert_forward_ref` | **bit-exact 0/4096** at real dims, with cap-fallback returning -1 |
| `sigmoid` | **NOT** bit-exact — 18.87% differ, ≤2 ULP, 1.611e-07 rel (`precise::exp` ≠ libm `expf`, plus GPU denormal-result flush) |

The sigmoid divergence is fully **absorbed** by the FP8 requantize before the down projection (the
error sits far below E4M3 granularity), which is why the fused kernel is bit-exact end-to-end
despite containing a non-exact primitive.

Infrastructure: seam wired at all three call sites (`:3173/3180`, `:3744/3752`, `:3786/3794`) with
the lease invariant verified — blocking `waitUntilCompleted` precedes every `coli_expert_release`,
so a refill cannot race an in-flight GPU read. Default build integrity preserved throughout: 26
objects with `METAL=0`, 27 with `METAL=1`, manifests byte-identical, tiny oracle token-exact with
Metal both ON and OFF.

### 12e. Two measurement traps worth carrying forward

**A dispatch counter is mandatory.** The first working seam appeared to pass every test while
silently falling back to CPU on every call — Metal was never initialized at runtime. Without
`metal_dispatches`, "token-exact and fast enough" would have been indistinguishable from "Metal
never ran", and every subsequent number would have been a lie.

**Profile-mode attribution is not production attribution.** A per-stage profile reported
`matmul gate` at 1.1597 ms against `matmul up` at 0.0452 ms — **25.7x apart on identical shapes** —
and a `submit+wait` share of 53.87%. Neither survives arithmetic: the stages sum to 1.922 ms, below
the CPU's 2.919 ms, so a literal reading would mean Metal already wins; and the gate "excess" alone
(2.66 s/run) exceeds the entire measured gap (0.825 s). Profile mode issues ~10 command buffers
instead of 1, inflating whichever dispatch lands first. The host-memcpy figure from that profile
*is* reliable (CPU timer, granularity-independent) and showed copying was only **0.52%** — refuting
a prior estimate of ~0.84 ms by roughly 40x, and with it the assumption that weight upload was the
bottleneck.

---

## 13. `COLI_V4_PREWARM` A/B — VERDICT: KILL (2026-08-13, `--ram 64`)

First clean, complete, gate-green A/B of `COLI_V4_PREWARM=0` vs `=1`. Both arms: same binary
(`sha f84813a9`), same frozen seed history (29127 B / 14190 selections) restored before each arm,
`COLI_V4_SAVE_USAGE=0` so neither arm mutates it, 4 prompts x 128 tokens, `target_slots=104`.
**Zero gate failures across all 8 rows** (compressor 2.2–2.6 GB against a 20 GB limit,
`gap_gb=0.4` throughout — no engine compression).

### 13a. Result: prewarm is actively harmful

| prompt | OFF tok/s | ON tok/s | delta |
|---|---|---|---|
| 1 (cold) | 0.1614 | 0.1321 | **−18.2%** |
| 2 | 0.1753 | 0.1275 | **−27.3%** |
| 3 | 0.2058 | 0.1385 | **−32.7%** |
| 4 | 0.1836 | 0.1125 | **−38.7%** |
| **mean** | **0.1815** | **0.1277** | **−29.7%** (paired mean −29.2%, sd 8.7pp) |

The delta **worsens monotonically**, so this is not a one-off startup cost being amortised — it
degrades steady state as well.

### 13b. The kill condition fired, but not the way §5/§6b predicted

The stated condition was: *if hit rate does not improve, the limit is pin policy, not persistence.*
Measured directly (the harness does not capture hit rate; this was measured separately at `--ram 64`,
8 tokens, same frozen history):

| | hits | misses | hit_rate |
|---|---|---|---|
| PREWARM=0 | 2383 | 1745 | 57.728% |
| PREWARM=1 | 2795 | 1333 | **67.708%** |

**Hit rate IMPROVED by +9.98pp.** So persistence is genuinely working, and the warmed experts are
genuinely useful — 412 fewer misses. The anticipated failure mode (warming the *wrong* experts)
did **not** occur.

### 13c. Why a better hit rate still loses — I/O accounting

An expert record at real dims is 4.19 (gate) + 4.19 (up) + 4.19 (down) + 0.79 (scales) = **13.37 MB**.

- misses avoided: 412 x 13.37 MB = **~5.51 GB of on-demand reads saved**
- eager warming cost: `v4_autopin warmed=688 requested=688 bytes=8.566GiB` = **~9.20 GB read up front**
- **net: ~+3.69 GB MORE I/O with prewarm enabled**

Prewarm spends ~9.2 GB eagerly to avoid ~5.5 GB on demand. It is a **net I/O loss**, and the eager
reads land at the worst possible moment — competing with prefill for SSD bandwidth and page cache.
The improved hit rate is real, it is simply **bought for more than it saves**.

### 13d. Verdict

**KILL `COLI_V4_PREWARM`.** Leave the default at 0 and treat the branch as dead weight.

The limitation is **neither** pin policy **nor** persistence, which is why the original
justify/kill framing did not anticipate this outcome:
- persistence works — history survives restarts and seeds correctly (14190 selections)
- pin selection works — the warmed experts really are the ones subsequently wanted (+9.98pp)
- **the economics do not work** — eager bulk reading costs more I/O than the misses it prevents,
  because prediction only needs to be *wrong 40% of the time* to make 688 speculative
  13.37 MB reads a losing trade against on-demand loading

§5's learning cache is therefore **not** unlocked by prewarm and should not be justified on its
basis. If eager warming is revisited, the requirement is not better prediction — it is warming
**far fewer** experts (only the highest-confidence pins) so that saved misses exceed bytes read.
A useful next experiment would sweep the warm count (e.g. 64/128/256 instead of 688) and find the
break-even point, rather than warming everything the history suggests.

### 13e. Two false leads on the way here, recorded

**A partial arm is worse than no arm.** An earlier partial run (different binary, `sha 7406b1fd`)
showed arm1 prompt #1 at **+14.1%** — the exact opposite sign of the final result. It was a single
row of a 4-row arm, measured under compression contamination. Had it been reported as a result,
it would have justified keeping a feature that costs 29.7%.

**Ambient load silently invalidated an entire run.** A `--ram 64` attempt gate-failed at prompt #2
with the compressor at **25.0 GB** and `gap_gb=20.1` (20 GB of the engine compressed). Cause was
non-engine RSS at **43–44 GB** — outside the 18–31 GB band §11 says keeps `--ram 64` uncompressed.
The same config then ran clean at 21.9 GB non-engine: prompt #1 went 901s → 793s (**+13.6%**) with
`gap_gb` 20.1 → 0.4. Before quoting any V4 throughput number, check `gap_gb` and the compressor;
the budget headroom is part of the measurement, not context around it.
