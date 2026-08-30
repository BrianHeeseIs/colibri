# AGENTS.md — working rules for this repository

## NEVER USE /tmp FOR ANYTHING YOU NEED

**`/tmp` is cleared on reboot, standby, and at the OS's discretion. Treat it as write-only
scratch that can vanish between two commands.**

This is not hypothetical. Both of these were lost mid-session on this machine:

| lost | consequence |
|---|---|
| `/tmp/coli_usage.snapshot` | the sacred benchmark seed. A gate run started with a broken seed protocol — every `cp` failed silently while the benchmark carried on, producing numbers not comparable to anything previously recorded |
| `/tmp/REF_MD5` | the reference binary hash for the `a20c7aa` binary-identity proof |

### The rule
- Anything you will need again — seeds, reference hashes, logs, measurements, notes, plans,
  recovered files — goes in the **repository**, under `.backlog/` (or another tracked path).
- `/tmp` is acceptable ONLY for a value consumed within the same command, that you would not
  mind losing instantly, and whose loss cannot silently corrupt a result.
- If a script reads a file it needs, it must **verify** the file exists AND hashes as expected,
  and **abort loudly** if not. A silent `cp` failure that lets a benchmark continue is worse
  than a crash, because it produces confident wrong numbers.
- Scratch BUILD trees (e.g. instrumented variants) may live outside the repo since they are
  rebuildable — but never the artefact you intend to compare against.

### Canonical durable locations in this repo
- benchmark seed: `.backlog/lab/coli_usage.snapshot` (md5 `599f3d12e9347ef30541bd6f9ba18bde`)
- reference binary hash: `.backlog/lab/REF_MD5`
- benchmark logs / runners: `.backlog/lab/`
- durable notes and resumable state: `.backlog/*.md`

## DEFAULTS CHANGED 2026-08-29 — the engine now ships the champion stack
As of E114-E119 the performance gates default **ON**: `KERNELS=all`, `MOE_GROUPED`, `MOE_BATCHED`,
`MOE_BATCHED_ROWS16`, `MOE_WHOLE_PROMPT`, `METAL_ATTN`, and `METAL_VARIANT=simd_exact_cold`.
`COLI_V4_METAL` still defaults **OFF** — it gates single-token DECODE Metal, which is slower here
(E99). Any individual flag still overrides the default (`COLI_V4_MOE_BATCHED=0` disables just that).

**`COLI_V4_BASELINE=1` restores every historical default in one move.** Use it whenever you need the
old deterministic reference: bit-exactness differentials, regression triage, or bisecting.

Consequence you must internalise: **the engine's default output is no longer token-identical to the
historical CPU arm.** It is capability-equivalent (`taskcheck` 5/5 at every step), not byte-equal.

- `bench/golden.sh` pins `COLI_V4_BASELINE=1`, so its sacred md5 still guards the reference path.
- `bench/golden_default.sh` guards the SHIPPING path against `bench/GOLDEN_DEFAULT_MD5`. That value
  is NOT sacred: re-record it deliberately when a default changes, and write down why.

## Benchmark integrity (earned the hard way)
- Golden output md5 `5d04890413ff539e802985ce8c727814` is SACRED. Never edit the expected
  value to make something pass; fix the code. It is now reached via `COLI_V4_BASELINE=1`, which
  `bench/golden.sh` sets for you — if you invoke the engine by hand and see a different hash, check
  that you set it before concluding anything broke.
- `bench/ab.sh` prints `delta = 100*(on-off)/off` — **FASTER IS NEGATIVE**.
- `bench/ab.sh` runs `--max-tokens 1` and parses only `time_to_first_token`; the metric is
  TTFT (load + prefill) and **decode is excluded by construction**. `p064`/`p256` are prompt
  FILES, not token counts.
- `bench/ab.sh` is `set -euo pipefail`, sends engine output to a per-run log inside a
  `mktemp` dir, and deletes that dir via `trap cleanup EXIT`. A non-zero engine exit
  therefore kills it **silently and destroys the error**. Pre-flight the engine yourself.
- Build with `make -C c -f Makefile.deepseek-v4 METAL=1 deepseek-v4 -j8`. `METAL ?= 0`, so a
  plain `make` silently compiles the Metal seam out with zero errors and zero warnings, and
  every Metal env flag becomes a no-op. Verify:
  `nm c/deepseek_v4 | grep -c coli_v4_metal_expert_forward_batch` must be > 0.
- One engine at a time: `pgrep -f '[d]eepseek_v4'` must be empty before starting one.
- Before dividing any counter by any wall, open the harness and read the exact invocation and
  the field it parses. Never infer a metric's definition from a sibling script.

## `bench/golden.sh` is a NECESSARY gate, not a SUFFICIENT one
Its prompt is ~20 tokens and fits a **single** 64-token prefill chunk. Any feature conditioned on
batch / group / chunk size may therefore **never execute during it**, and golden passes trivially.

This already produced one false result here: a rows16 change PASSED golden and was written up as
"bit-exact", then **diverged on p256** (184 tokens, 3 chunks). The entry had to be retracted.

- Before writing "bit-exact", run a differential on a **multi-chunk** prompt (p256 or larger),
  60 tokens, comparing the extracted `generated_text` md5 with the feature ON vs OFF.
- Always run the **determinism control** first: the same arm twice. If an arm does not reproduce
  its own md5, an ON-vs-OFF difference proves nothing.
- A golden **FAIL is informative**: a changed md5 proves the path executed. A golden PASS with a
  flag ON proves nothing about execution — capture a counter that is zero without the feature.

## Two performance axes. Never conflate them
- **TTFT** (load + prefill): `bench/ab.sh`. Decode excluded by construction.
- **tok/s** (decode): `.backlog/lab/tokps.sh`. `tok/s = (max_tokens-1) / after_first`, where
  `after_first` is `gen_stats.decode_sec` (`c/deepseek_v4.c:11708`).

They do not move together. `COLI_V4_METAL_ATTN=1` is -15.9 % TTFT but trends *negative* on decode.
`COLI_V4_KERNELS=all` grows on TTFT with prompt length while its tok/s gain is non-monotonic.

**Noise floors:** TTFT spread 0.6-0.8 % (n=3 is fine). **Decode spread 5-13 % — use n>=5.** A decode
delta under ~10 % cannot be resolved at n=3, and several apparent effects here reversed when a
third or fifth point was added.

### MANDATORY: no background agents in flight while a timing run executes
**This is a measurement-integrity rule, not a rule about machine noise.** A `COLI_V4_PROFILE=1`
profile taken while two explore agents were running reported `decode_wall` 28500 ms and
`expert_forward` 13290 ms — **a fake 76 % jump**. `ps -Ao pcpu -r` showed the agent host at 235 %
CPU with load average 5.72, against an engine that takes 16 OpenMP threads. The E125 numbers either
side of it were trustworthy precisely because nothing else was running
(`.backlog/m3-max-decode-research-2026-08-29.md:582`,
`.backlog/ulw-decode-next-avenues-20260830-004040.md:69`).

- Before ANY timing run — `tokps.sh`, `ab.sh`, `golden.sh`, a profile, a kbench arm — confirm no
  agents, builds or other jobs are in flight. Let background tasks finish first.
- This binds **timing** work only. Pure authoring, exploration, reading and building may still run
  fully parallel; parallelise those freely.
- A number captured under load is not merely noisy, it is **wrong by up to 76 %**, and it is not
  comparable to anything else in the ledger. If load was present, the run is void — rerun it.
- ONE engine at a time (`pgrep -f '[d]eepseek_v4'`): two engines contend for the same weights and
  the same ~100 GB budget. GPU probes must not run against a live engine — same device.

### Run long engine work through tmux, never directly
`golden.sh`, `golden_default.sh`, `tokps.sh`, `taskcheck.sh` and any trace run take minutes. Invoked
directly they hit the tool timeout, get killed mid-flight, and **silently leave the work undone while
looking like they ran** — this burned several turns on 2026-08-30. Send them to the tmux session
`colibri-lab`, pane `colibri-lab:0.0`, writing to a durable log, and `touch` a completion-marker file
as the last action. Poll for the marker in a LATER turn; never `sleep`/poll inside bash.
`.backlog/lab/run_goldens.sh` is a working example.

### A nonzero rc from golden.sh is often a REFUSAL, not a hash failure
`golden.sh` and `golden_default.sh` abort with rc=2 and
`another deepseek_v4 process is already running` / `FATAL: engine already running` when any engine is
live. That is the one-engine guard doing its job, not a broken build. **Read the log body before
concluding anything.** A concurrent session shares this host, and `pgrep` can clear a moment before
their process actually exits — so a launch that passed your own guard can still be refused.

### The stderr prefix strip-list is load-bearing. Adding a new prefix corrupts md5 comparisons
`.backlog/lab/tokps.sh` and `taskcheck.sh` extract `generated_text` by stripping ONLY lines matching
`^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal) `. Any stderr line with a
different prefix **leaks into the extracted text, changes its md5, and manufactures a false
"not bit-exact" verdict.** New counters must ride an existing prefix — extend the `v4_profile ` or
`v4_rows16 ` line rather than inventing one.
**Latent bug, unfixed:** `COLI_V4_PREFILL_TRACE` prints under `v4_prefill_trace `, which is NOT in
that list. Enabling it during a `tokps.sh` run today would corrupt the comparison.

### Verify from disk. Do not trust a self-report — including your own
On 2026-08-30 one delegated task reported detailed line numbers and a clean build for work it had
never performed, and a separate turn narrated a tool result that was never returned. Both looked
entirely plausible. Before building on any claimed change, confirm it exists:
`grep -c <symbol> <file>`, `git status`, `git diff --cached --name-only`. A claim with no observed
tool output behind it is not evidence.

### MANDATORY: do not run a long benchmark without asking. Size it to the question first.
Operator rule. Benchmark wall-clock is the scarcest resource in this project; spending it badly is
worse than not measuring, because it also delays everything behind it.

1. **Size the run to the claim.** If 3-5 runs settle it, NEVER plan 15. A lever predicted to be
   flat on an axis needs a *flatness check* (2 points), not a resolution-grade measurement (n>=5).
   The n>=5 decode rule exists for resolving deltas under ~10% — it is not a default for every arm.
2. **Ask before starting anything long.** Say what it costs, what it decides, and why it must run
   NOW rather than going into the backlog. Default is the backlog:
   `.backlog/simd-exact-remaining-measurements.md`, run in a batch when the operator says so.
3. **Cut arms that are already known-losing.** Re-measuring a recorded dead setting is pure cost.
4. **Check the signal-to-load ratio before choosing a prompt.** Model load is ~35 s of EVERY fresh
   run. At p064 TTFT is ~46 s, so load is ~75% of the metric and a prefill effect is diluted into
   the noise. Choose the prompt so the thing being measured dominates the wall — or accept that a
   null result at that length means nothing.
5. **A short prompt can make the effect structurally impossible.** p064 is 64 tokens = exactly ONE
   64-token chunk, so any chunk-width or grouping lever is inert there BY CONSTRUCTION. Confirm the
   mechanism can even fire at the chosen length before spending runs on it.

Learned by wasting two sweeps: a 4-arm x N=3 MIN_N sweep at p064 (12 runs) whose first round already
showed a 0.6% spread — the TTFT noise floor — with identical md5s across all arms, and a p512 chunk
sweep launched unasked that would have taken 30+ minutes at ~5 min/run.

### MANDATORY: if an experiment generates tokens, it MUST report tok/s — not TTFT alone
Operator rule, binding on every experiment and every ledger entry.

- **Report BOTH axes whenever tokens are produced.** A TTFT-only result is incomplete and will be
  read as a general "speedup" by the next session. State tok/s with its N, or state explicitly why
  no tokens were generated.
- **`bench/ab.sh` CANNOT satisfy this on its own.** It runs `--max-tokens 1` and parses only
  `time_to_first_token`, so decode is excluded by construction. Using it alone for anything that
  can touch decode is a measurement error, not a shortcut.
- **Default to `.backlog/lab/tokps.sh`**, which reports TTFT *and* tok/s *and* the per-arm md5 *and*
  a determinism verdict from the same runs, and verifies the seed hash. Reach for `ab.sh` only when
  the change is prefill-only, and say so.
- Quote the prompt length and N next to every number. TTFT and tok/s do not move together, and on
  this host they have moved in OPPOSITE directions (`COLI_V4_METAL_ATTN=1`).

Why this is a rule: roughly 97 of the ~100 entries in `experiments_results.md` optimise TTFT
because `ab.sh` was the only harness, and tok/s had never once been A/B'd before E87. Several of
those entries read like general speedups and are not.

## Seed: the harnesses check existence, not correctness
`bench/ab.sh` and `bench/golden.sh` only test that `/tmp/coli_usage.snapshot` EXISTS. A
present-but-wrong seed corrupts results silently. Sync from the durable copy before every run:
`cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot`.
`.backlog/lab/tokps.sh` and `.backlog/lab/taskcheck.sh` verify the md5 and abort loudly.

## Fastest known setting
**SUPERSEDED 2026-08-29 (E114-E119).** The fastest known setting is now the GPU-prefill stack:
```bash
COLI_V4_METAL=0 COLI_V4_KERNELS=all COLI_V4_MOE_GROUPED=1 COLI_V4_MOE_BATCHED=1 \
COLI_V4_METAL_VARIANT=simd_exact_cold COLI_V4_METAL_ATTN=1 COLI_V4_MOE_BATCHED_ROWS16=1 \
COLI_V4_MOE_WHOLE_PROMPT=1 COLI_V4_FP8_ROWS16=1
```
(All of these are now DEFAULTS; the list is written out only so the stack is explicit.)
**Measured head to head in E128** (p256, N=3, both arms deterministic): `COLI_V4_BASELINE=1` gives
1.3948 tok/s, shipping defaults give 2.1596 -- **+54.83% tok/s, TTFT -62.4%, net wall @40 tokens
-57.1%**. Quote THAT as "versus the historical engine".

**Do not quote +54.83% as the decode work.** `COLI_V4_BASELINE=1` also switches off `KERNELS=all`
(`c/deepseek_v4.c:10693`), so it sits BELOW the CPU arm the decode experiments were measured from.
The decomposition: **+29.7%** (2.1596/1.6655) is this session's decode work in E125-E127, matching
the recorded +28.1%; the rest is pre-existing `KERNELS=all` plus the E114-E119 GPU-prefill stack.
That split is INFERRED -- 1.6655 comes from an earlier session and was not re-measured in E128.
The TTFT -62.4% is the prefill stack, not decode.

**Decode is +28.1% since E125** (1.6655 -> 2.1341 tok/s at p256, N=5, all bit-exact; endpoint
reproduced as 2.1596 in E128). After the fp8 kernel below, four more landed in E126/E127 --
`COLI_V4_HEAD_ILP`, `COLI_V4_HC_OMP`, `COLI_V4_FP8_DUAL_ROWS16`, `COLI_V4_SPARSE_OMP` -- all
default ON.
**They were all the same defect: an `#ifdef __AVX2__` fast path with no `__aarch64__` sibling, or a
hot loop with no `#pragma omp`.** That sweep is now exhausted for decode. Two things NOT to retry:
raising `COLI_V4_PIN_SLOTS` (the scalar and NEON mxfp4 kernels are within 1.00-1.14x, measured
head-to-head), and `OMP_NUM_THREADS=12` (null at N=1, 2.5% spread, inside the noise floor).

**Decode moved for the first time in E125**: `COLI_V4_FP8_ROWS16` is +10.18% tok/s, bit-exact, and
free of memory cost. Note what did NOT work, so it is not retried: a drop-in NEON port is neutral
(E124), arithmetic e4m3 decode is 0.70x, and keeping the packed weights as a SECOND copy costs +4 GB
and reverses the win — permute in place.
Against the `KERNELS=all` CPU arm below: **TTFT -48.2%, net wall @40 tokens -38.5%, tok/s -1.75%**
at p256 (N=3-5, non-overlapping ranges). `COLI_V4_METAL=0` still disables single-token decode Metal;
prefill attention, batched MoE and the whole-prompt dispatch are gated independently. Do NOT combine
with `COLI_V4_PREFILL_PREFETCH=1` (deadlocks, E113). Output is not token-identical to the CPU arm;
capability was gated with `taskcheck` (5/5) at every step. The section below is retained because it
remains the fastest DETERMINISTIC-ish reference and the basis of every comparison.

## Fastest known CPU-only setting (historical baseline)
**`COLI_V4_KERNELS=all`, with Metal OFF** — measured -10.4 / -18.0 / -20.2 % TTFT and
+16.7 / +12.1 / +20.1 % tok/s at p064 / p256 / p512. Keep it **OFF** for golden, bit-exactness
differentials, and regression triage: its output is nondeterministic at short prompts (two variants
over eight runs) though bit-exact at p512.

**Confirmed in combination (E105).** At p064 it is +17.11 % tok/s and -8.8 % TTFT over the CPU arm
(1.42395 -> 1.66755), and it composes cleanly with the Metal path too (same ~16 % multiplier).
**The CPU arm still wins**: fully stacked, CPU+KERNELS=all beats Metal+simd_exact+KERNELS=all by
8.6 %. Do not assume the Metal expert path is faster here — it is not (E99), and the only reason
to enable it is a configuration that differs from this host.

## Non-bit-exact changes are gated on TASK-LEVEL correctness
Operator-approved bar: capability must be identical (verifiable arithmetic / factual / code / logic
answers stay correct); token identity is not required. Harness `.backlog/lab/taskcheck.sh`.
Reproducibility is a SEPARATE property this bar does not cover — state it explicitly when it fails.

### NEVER conclude an approach is wrong from a CHANGED MD5. READ THE TEXT FIRST.
A golden FAIL means "the output is not token-identical". It does **not** mean the output is wrong,
and it is **not** grounds to reject, revert, or write off an approach. **Slight variations with the
meaning retained are fully acceptable for this project's goal.**

Before calling any approach broken, you MUST:
1. Extract the generated text from BOTH arms and **diff it**. Report what actually changed.
2. Run `.backlog/lab/taskcheck.sh` — the capability gate.
3. Run the arm against itself once to separate a *difference* from *nondeterminism*.

Only after those three may you use the words "fails" or "breaks".

This was learned the expensive way in E97. Widening the Metal expert path to rows16 was reported as
"BREAKS GOLDEN" and reverted on the strength of an md5 alone. The text was then compared and the
ENTIRE divergence was one token in sixty — `FFN layers.` versus `FFN layer.` — deterministic, with
`taskcheck` scoring 5/5 on both arms and byte-identical output on its prompt. The approach was
sound; a 45%-of-expert-calls speedup had been discarded because nobody looked at the words.
`bench/golden.sh` deletes its run directory on exit (`trap cleanup EXIT`), so capture the text
yourself — run the engine directly, or use `.backlog/lab/tokps.sh`/`taskcheck.sh`, which keep
transcripts.

Corollary for the ledger: "fails golden" is never a sufficient experiment write-up. Record WHICH
tokens changed and whether capability survived, or the entry will mislead the next session.

## Re-measure every upstream claim before adopting it
Three upstream performance claims in a row failed to reproduce on this host: #1097 loader lanes
("1.41x decode"), `V4_NGRAM` ("+18.5 %"), and `e36a1c7` hot-pack ("~6.9 s TTFT"). Upstream numbers
come from different hardware and workloads. Measure first; pay the adoption cost only if it
survives. Details in `experiments_results.md` E87-E94 and `mem:deepseek_v4/dead_levers`.
