# Maximum-performance candidates for this MacBook Pro M3 Max

Date: 2026-08-28  
Repository state reviewed: `simd-apple-metal` at `a942b99d727b5a64eea765d8fb682121b5eee7c1`  
Host: Mac15,9, Apple M3 Max, 12P+4E CPU cores, 40 GPU cores, 128 GB unified memory  
Scope: all local performance progress through E108, relevant backlog/history/remote branches, current runtime code, and mechanisms that fit Apple Silicon. No model run or long benchmark was performed for this report.

## Executive answer

The fastest configuration actually measured on this host is still:

```bash
COLI_V4_KERNELS=all COLI_V4_METAL=0 \
  ./c/coli serve --model /path/to/model --ram 64
```

Use `--ram 64` for a comfortable daily desktop profile. On a quiet, freshly booted machine, older directional evidence suggests 96 GB can help heterogeneous conversations whose expert working set misses the seeded cache, but those arms were not interleaved and it is not a generic speedup: E108 found 48/72/96 GB equal within 1.2% on the frozen p064 working set, N=2. Keep the machine on AC, use High Power mode, let the launcher keep 16 OpenMP threads, and run one engine.

The decisive composed result is E105, p064, 60 generated tokens, N=2:

| arm | decode | TTFT | total wall |
|---|---:|---:|---:|
| CPU baseline | 1.42395 tok/s | 43.42 s | 84.86 s |
| **CPU + `COLI_V4_KERNELS=all`** | **1.66755 tok/s** | **39.58 s** | **74.96 s** |
| Metal experts `simd_exact_cold` + fast kernels | 1.53590 tok/s | 42.40 s | 80.81 s |

The CPU fast-kernel arm is 17.11% faster in decode and 8.8% faster in TTFT than the CPU baseline; it also beats the fully stacked Metal-expert arm by 8.6% in decode. E88 provides the stronger p064 decode sample, N=5: +17.08% tok/s and -9.6% TTFT, taskcheck 5/5. The price is short-prompt nondeterminism: the fast-kernel arm produced two stable output MD5 variants. Capability survived; exact reproducibility did not.

The first thing to fix is not a kernel. The internal APFS container has only **7.5 GB free out of 994.7 GB (0.8%)**, while a roughly 155 GB model streams expert shards from that storage and macOS also needs VM/swap headroom. Reclaim substantial internal space before expecting comfortable sustained operation. No trustworthy local measurement converts that constraint into a tok/s estimate, so none is claimed here.

After that, the strongest engineering candidates are:

1. **Re-evaluate the existing batched Metal MoE prefill path with the strongest Metal kernel:** E84 measured `COLI_V4_MOE_BATCHED=1` at -10.79%/-11.70% incremental TTFT on p064/p256, N=5, against an older grouped+Metal-attention stack. E96 later measured `simd_exact_cold` at p256, 60 tokens, N=2, +33.90% tok/s and -16.6% TTFT within the Metal lane, with deterministic identical multi-chunk output. The current CPU+fast-kernel composition remains unmeasured.
2. **Revalidate the live prefill-prefetch path on the current winner:** `COLI_V4_PREFILL_PREFETCH=1` is present and default-off. Historical interleaved TTFT-only N=3 evidence was -6.20% at p064 and -2.70% at p256; E56 found ON/OFF outputs identical at both lengths with `--max-tokens 32`, but its per-arm N and same-arm repeatability were not stated. Its composition with the current CPU+fast-kernel arm and its decode effect have never been measured under the current both-axes gate.
3. **Retune `COLI_V4_MOE_BATCHED_MIN_N` with all prerequisite flags and engagement proof:** E101's p064, 60-token sweep recorded `COLI_V4_MOE_BATCHED=1` but not the required `COLI_V4_MOE_GROUPED=1`, and captured no grouped-engagement counters. Its null is therefore inactive or engagement-unproven, not a valid threshold result. An older engaged p064 TTFT probe, N=3, found MIN_N=3 0.81% slower than 4 with the ordered kernel; `simd_exact_cold` makes that crossover stale.
4. **Measure the two rows16 lanes separately:** prefill `COLI_V4_MOE_BATCHED_ROWS16=1` measured -4.64%/-5.49% TTFT at p064/p256, N=3, but diverged on a deterministic p256 differential; decode `COLI_V4_METAL_ROWS16=1` raised Metal expert coverage from 2684/4902 to 4902/4902 calls and passed taskcheck 5/5 at N=1, but its speed is unmeasured. Neither is the vague “build a rows16 kernel” problem.
5. **Phase-dependent attention:** Metal for prefill, CPU for decode. Metal attention measured -15.9% TTFT at p064, 24 generated tokens, N=3, while its decode direction was negative/unresolved. Do not move single-token decode routed experts to Metal by default.
6. **Decouple whole-prompt MoE batching from the 64-token attention chunks:** E62 measured mean same-expert group size rising from 4.14 to 7.99 on one p256, 184-token route dump. Interpolating the existing kernel curve implies 1.48x lower expert-kernel cost, not a 1.48x end-to-end engine gain. This is a strong architecture candidate, but it remains unimplemented and lacks an end-to-end A/B.
7. **Direct expert-slot indexing plus bounded LRU miss selection:** portable, arithmetic-neutral upstream work that removes full slot scans. Profile its existing counters first; port only if slot lookup/selection is material at p256/p512.
8. **Reuse the quantized/dequantized attention input across `wq_a` and `wkv`:** a small, upstream-parity-proven CPU candidate. Its share must be measured in the actual CPU+fast-kernel arm.
9. **Batch the shared expert during prefill:** a low/medium-risk TTFT candidate with no host-local end-to-end percentage yet.
10. **Reusable long-prefix checkpoints:** valuable for repeated long system/tool prefixes and restarts, but this is a large feature and current same-session strict prefix reuse already exists.
11. **Non-bit-exact 16-column FP8 projection:** the remaining plausible raw decode-kernel experiment. Amdahl bounds are only about +7.8% to +17.1% whole-decode for 1.33x to 2x speedup on its verified 29.2% share. The exact NEON route measured 1.00-1.01x in a standalone real-shape microprobe, not an engine prompt/N run, and is dead.
12. **Multiple KV slots plus continuous batching:** the largest aggregate-serving-throughput opportunity, but it does not make a single chat decode faster and is a large architecture project.

## What “fast” means here

This repository has two independent axes:

- **TTFT:** load plus prefill. `bench/ab.sh` runs `--max-tokens 1`; it contains no decode measurement.
- **Decode tok/s:** `.backlog/lab/tokps.sh` computes `(max_tokens - 1) / decode_sec` and reports TTFT, output hashes, and determinism from the same arms.

They can move in opposite directions. A candidate is not called a general speedup unless both are measured. This report also separates:

- token identity from task-level correctness;
- same-arm determinism from ON/OFF equality;
- an isolated phase gain from an end-to-end gain;
- a measured host-local result from an Amdahl bound or an unmeasured mechanism.

## Current host and bottleneck model

### Decode at the winning CPU backend

E102's single p064 profile run (24 generated tokens, N=1) measured 1.41 tok/s and accounted for 99.4% of decode:

| phase | decode share | implication |
|---|---:|---|
| attention | **38.7%** | largest phase; `attn_out` alone is 17.9% |
| expert forward | **27.0%** | CPU remains faster than the optimized Metal expert lane |
| expert wait | **9.7%** | disk/cache misses matter in heterogeneous workloads |
| shared expert | 5.8% | possible batch-prefill target |
| LM head | 5.1% | secondary |
| router | 4.7% | included in `COLI_V4_KERNELS=all` |
| hc norm | 4.0% | secondary |
| indexer | 2.3% | secondary |
| compressor | 2.0% | secondary |

`COLI_V4_KERNELS=all` enables reassociated `attn_sparse` plus router, together 16.4% of this profile. This matches its measured ~17% decode gain unusually well. The output-order change explains why it belongs in the fastest profile but not in token-identity or regression baselines.

### Prefill/TTFT

Load is typically about 35 seconds across the fresh-process harnesses; this is a descriptive cross-run value, not one prompt/N estimate. Consequently p064 TTFT is dominated by fixed load and structurally hides some prefill effects. p064 is also exactly one 64-token prefill chunk; chunk/group levers cannot be inferred from it. Longer prompts are required for prefill scheduling work, but they are expensive and must be sized before running.

The prefill width cannot currently exceed 64. Four batched APIs enforce `batch <= 64`; E108's p256 sweep failed before producing a valid chunk128 arm/N. The runtime knob is now clamped. Lifting the limit is a cross-engine-buffer audit, not a tuning flag.

The important existing Metal exception is `COLI_V4_MOE_BATCHED=1`: during prefill, grouping can give one expert S>=4 rows in a single dispatch. E84 measured -10.79%/-11.70% incremental TTFT at p064/p256, N=5, against its older grouped+Metal-attention baseline. It does not make S=1 decode experts faster, and it has not yet been composed with `COLI_V4_KERNELS=all` under the current both-axes and multi-chunk correctness rules.

### Memory, storage, and thermal comfort

- **Internal storage:** 7.5 GB free is the urgent operational issue. Review large unrelated data and unused Xcode Simulator runtimes; do not delete blindly. Keep the live model on the internal SSD unless a real Thunderbolt/NVMe device is attached and measured. The mounted Time Machine/network backup volume is not a credible live-shard mirror.
- **RAM:** E108 says 48 GB already holds the seeded p064 working set at N=2. Older, non-interleaved heterogeneous-chat evidence found 64 GB the largest budget that reliably stayed uncompressed under an 18-31 GB desktop load and directionally favored 96 GB when the rest of the machine was quiet. Start at 64 GB; raise only when compression and swap growth stay quiet.
- **Current state:** roughly 0.94 GB was occupied by the compressor and about 10 GB swap remained allocated after 17 days uptime. Historical swap allocation is not proof of current pressure, but a reboot and closing multi-GB apps gives a cleaner high-RAM run.
- **Power/thermal:** stay on AC and High Power mode for sustained inference. Apple states that High Power mode permits higher fan speeds and may improve performance in intensive workloads. The external display also consumes some GPU/memory/thermal budget; disconnecting it is a testable last-mile option, not a guaranteed gain.
- **Threads:** keep the launcher default of 16 physical cores. E107's p064, 60-generated-token, N=2-per-arm comparison put the 12-thread arm 1.05% slower, inside noise. Do not widen pinned experts: at p064, 60 generated tokens, N=2 per arm, pins158 lost 7.79% decode and added 18% TTFT.
- **Process model:** one engine at a time. Use persistent `serve` plus `chat --attach` for actual use so loading, warmed experts, and session state can be reused. Use fresh one-shot processes only when the benchmark requires isolation.

## Evidence quality

| claim | evidence | confidence | limitation |
|---|---|---|---|
| Fast kernels are the fastest current backend | E88 p064 N=5; E105 composed N=2; E91 length sweep N=2 | **High** for direction | exact long-prompt decode deltas are not resolution-grade |
| CPU experts beat stacked Metal experts | E105 p064, 60 tokens, N=2, non-overlapping direction | **High** for current host/config | not every RAM/prompt regime measured |
| Grouped/batched Metal MoE improves historical prefill | E84 p064/p256, TTFT-only N=5 | **High** against that older baseline | not composed with fast kernels/simd_exact; no tok/s |
| `simd_exact_cold` improves the Metal lane | E96 p256, 60 tokens, N=2, deterministic identical multi-chunk output | **High** for direction, provisional magnitude | below N>=5; CPU/current full composition absent |
| Live prefill prefetch improves historical TTFT | E55 p064/p256, TTFT-only N=3; E56 max-tokens 32 equality check, per-arm N unavailable | **High** against that older baseline | E56 same-arm repeatability not reported; only -2.70% at p256; current fast-kernel composition and tok/s unknown |
| Batched prefill rows16 improves historical TTFT | E86 p064/p256, N=3, -4.64%/-5.49% | **High** for TTFT direction | deterministic p256 divergence; task capability/tok/s incomplete |
| Single-token rows16 admission improves speed | no performance A/B; E97 p064 profile and taskcheck N=1 only | **Unknown** | 45.2% extra call coverage is not a speed estimate |
| MIN_N tuning is closed | E101 p064, 60 tokens, one complete round plus a second baseline point | **False** | grouped prerequisite not recorded and engagement not proved; older engaged MIN_N=3 vs 4 result used the pre-simd kernel |
| Metal attention improves TTFT | E87 p064, 24 tokens, N=3, -15.9% | **Medium-high** | not composed with fast kernels; decode delta unresolved |
| 48/72/96 GB are equal | E108 p064 N=2, spread 1.2%, identical hashes | **High only for seeded p064** | explicitly not heterogeneous chat |
| Reducing OpenMP threads from 16 to 12 does not help | E107 p064, 60 generated tokens, N=2 per arm, 12-thread arm 1.05% slower | **Medium** | difference is noise; a flatness check, not a broad scheduler study |
| Exact arm64 FP8 SIMD has no gain | E104 standalone real-shape microprobe (no prompt/N), 1.00-1.01x | **High** for tested design | does not test non-bit-exact across-column designs |
| Whole-prompt MoE batching can nearly double same-expert group size | E62 one p256, 184-token route dump; mean N 4.14 to 7.99 | **Medium-high** for route shape | 1.48x is interpolated expert-kernel cost only; no implementation or end-to-end A/B |
| Direct index/LRU, QDQ reuse, shared prefill batch may help | upstream implementations and mechanism | **Low until local A/B** | no end-to-end M3 result |
| Low disk space hurts this exact workload by X% | no valid local A/B | **Unquantified** | operational risk only; do not invent X |

## Ranked candidate matrix

### P0: do now, no engine change

#### 1. Reclaim internal SSD headroom

- **Why it fits:** this runtime streams routed experts and the same device backs normal system VM/swap work.
- **Expected gain:** unquantified; primarily avoids stalls, failed writes, and uncomfortable system behavior.
- **Work/risk:** low work if moving unrelated files; destructive cleanup risk requires explicit inspection/approval.
- **Smallest decisive experiment:** after cleanup, capture APFS free space and compare a real heterogeneous replay, not just seeded p064. If a before/after comparison can be preserved, freeze the prompt/seed and include same-state controls. If it generates a continuation, report TTFT and tok/s from the same runs plus expert-wait/storage counters.
- **Kill criterion:** none as a reliability action; reject only claims of a specific speed percentage without an A/B.

#### 2. Run CPU experts with fast attention/router kernels

- **Configuration:** `COLI_V4_KERNELS=all`, `COLI_V4_METAL=0`, RAM 64 GB daily or up to 96 GB when the desktop is quiet.
- **Host fit:** directly accelerates `attn_sparse` plus router, 16.4% of the corrected M3 Max decode profile, without paying the S=1 Metal-expert dispatch cost.
- **Expected gain:** measured +17.08% decode and -9.6% TTFT at p064 N=5 versus CPU baseline; E105 measured -11.7% total wall at p064 N=2.
- **Work/risk:** no implementation work; low operational work. Reassociated arithmetic is the correctness/reproducibility risk.
- **Smallest decisive experiment:** none needed to adopt it for interactive use; if a new binary/configuration is compared, run same-arm determinism first, then a multi-chunk text differential, taskcheck, TTFT, and tok/s.
- **Kill criterion:** task-level capability regression, unacceptable nondeterminism for the target workload, or a newer composed arm beating both axes outside their noise floors.

#### 3. Keep a persistent server/session

- **Why it fits:** avoids repeated ~35-second loads and preserves warmed expert/session state.
- **Host fit:** model load and expert streaming are large fixed costs on this storage-bound 128 GB laptop; persistence removes repeated work without changing numerical kernels.
- **Current capability:** strict same-session prefix reuse exists. It records prompt plus generated token ids and reuses state only when the next prompt strictly extends the entire recorded sequence.
- **Expected gain:** workload-dependent avoided load/prefill, not higher steady-state tok/s.
- **Work/risk:** no implementation work; low operational risk. One-slot queueing and stale-session assumptions are the main limits.
- **Smallest decisive experiment:** two equivalent multi-turn sessions, one persistent and one fresh-process, using the same prompt sequence. Record load, prefix engagement, TTFT, tok/s, total wall, and same-arm output stability; if outputs differ, inspect the text and run taskcheck before judging capability.
- **Kill criterion:** no load/prefix reuse occurs, persistent state changes capability, or queueing latency outweighs avoided work for the actual use pattern.

### P1: strongest current-stack and measured-architecture candidates

#### 4. Compose grouped/batched Metal prefill with `simd_exact_cold`

- **Mechanism:** current `COLI_V4_MOE_GROUPED=1 COLI_V4_MOE_BATCHED=1` gathers rows routed to the same expert and uses one Metal dispatch only when group size clears `COLI_V4_MOE_BATCHED_MIN_N` (default 4). `COLI_V4_METAL_VARIANT=simd_exact_cold` selects the strongest current cold-layout Metal kernel. Keep the independent master `COLI_V4_METAL=0` so S=1 decode experts remain on CPU; the batched prefill seam is feature-gated separately.
- **Host fit:** the older ordered Metal kernel lost at S=1/2 and won at S>=4, which motivated grouping. `simd_exact_cold` later made the isolated Metal matmul much faster even at S=1, but E105 shows the whole single-token Metal expert path still loses after orchestration and rows16 exclusions. Prefill grouping is the existing seam that amortizes those costs with larger same-expert batches.
- **Evidence:** E84, TTFT-only historical harness, N=5: CPU grouping alone was -4.43%/-3.58% TTFT at p064/p256 against its older adjacent baseline. Adding ordered-kernel batched MoE to grouped+Metal-attention reduced p064 TTFT 35.842 to 31.975 s (-10.79%, 1.121x) and p256 94.251 to 83.221 s (-11.70%, 1.133x); engagement was 1009 Metal expert dispatches at p064. E96 then isolated `simd_exact_cold` versus ordered Metal at p256, 60 tokens, N=2: +33.90% tok/s and -16.6% TTFT, with both arms deterministic and the same multi-chunk MD5. Its p512 differential was interrupted and has no data. E105 composed simd_exact with fast kernels but did not enable grouped/batched MoE. No experiment has composed all three on the current winner.
- **Expected gain:** the historical ~11% incremental TTFT and E96's within-Metal gains are measured separately; their current composition is unknown and must not be multiplied.
- **Work/risk:** small measurement/configuration work because code exists; medium correctness/configuration risk due to build seam, grouping gates, and stale historical baseline.
- **Smallest decisive experiment:** build with `METAL=1`, verify the seam symbol and engagement counters, then compare three p256 arms: CPU+fast-kernels; plus CPU grouping; plus grouped+batched-MoE+`simd_exact_cold`, with `COLI_V4_METAL=0` in all arms so decode remains CPU. E96 already isolates ordered versus simd_exact, so the scarce run should test the strongest current composition rather than remeasure the known-weaker ordered kernel. Run each arm against itself first, generate enough continuation tokens to report TTFT and tok/s, diff the multi-chunk generated text, and run taskcheck before any capability verdict. Add rows16 and phase-dependent attention only in later steps so causality remains clear.
- **Kill criterion:** no engagement, TTFT improvement within the 0.6-0.8% noise floor, any material decode regression, failed taskcheck, or unacceptable same-arm nondeterminism. A changed MD5 alone triggers text diff/taskcheck; it does not kill the path.

#### 5. Revalidate live prefill prefetch on CPU+fast-kernels

- **Mechanism:** `COLI_V4_PREFILL_PREFETCH=1` enables the existing default-off route-ahead/overlap loader, which looks ahead across routed experts and overlaps expert miss reads with useful work. It changes I/O scheduling, not model arithmetic.
- **Host fit:** E54 attributed 43.47% of its historical p064 prefill wall to expert miss read plus first touch, exactly the work prefetch targets; its N is unavailable in the ledger. This M3 Max still streams a model larger than the daily RAM budget from the nearly full internal SSD, although attention's growing share dilutes the benefit on longer prompts.
- **Evidence:** E55 used `bench/ab.sh`, interleaved OFF/ON, N=3 per prompt: p064 TTFT 43.065 to 40.393 seconds (-6.20%) and p256 TTFT 111.442 to 108.429 seconds (-2.70%). It was TTFT-only, so no decode result exists. E56 then used `--max-tokens 32` and found identical OFF/ON generated-text MD5s at p064 and p256; its per-arm N is unavailable and it did not report same-arm repeatability. E52's older p064 cold table put fast-kernels alone at 39.352 seconds and prefetch+fast-kernels at 35.597 seconds (-9.54% incremental), but the entry does not state an N for that table. E54's separate p064 stack, also with N unavailable, measured default 42.780, fast-kernels 38.942, prefetch 40.452, and the combined arm 36.392 seconds; its -14.9% is the combined change from default, not prefetch's incremental contribution.
- **Expected gain:** historical TTFT direction is measured, with a 2.7% p256 effect and a 6.2% p064 effect. The old ~9.5% incremental short-prompt composition is not transferable to the current binary/baseline without remeasurement; steady decode tok/s is expected to be mostly flat but is unknown.
- **Work/risk:** no implementation work; small measurement work. Numerical risk is low because prior generated text matched, but I/O ordering, contention, and route-ahead overhead can change with the current cache and kernel stack.
- **Smallest decisive experiment:** after restoring SSD headroom, compare CPU+`COLI_V4_KERNELS=all` with and without prefetch on p256, N=2 per arm for direction. Record a positive engagement counter, same-arm repeatability, a multi-chunk text diff, taskcheck, TTFT, tok/s, expert-wait time, and route-ahead overhead from the same token-generating runs. Ask before any larger or resolution-grade run.
- **Kill criterion:** no engagement, TTFT change within noise or below the overhead justified by the workload, material decode regression, taskcheck regression, or unacceptable same-arm nondeterminism. A changed MD5 alone requires text inspection and taskcheck; it is not a kill signal.

#### 6. Retune the grouped-MoE `MIN_N` crossover with `simd_exact_cold`

- **Mechanism:** `COLI_V4_MOE_BATCHED_MIN_N` decides the smallest same-expert token group sent to Metal. The default 4 was chosen before `simd_exact_cold` changed the GPU side of the crossover.
- **Host fit:** a 64-token top-6/256-expert chunk averages only 1.5 rows per expert; lowering a now-stale gate could expose substantially more prefill groups to the 40-core GPU without any serving-architecture work.
- **Evidence:** E101 planned N=3 for MIN_N 4/2/1/8 at p064, 60 tokens, but stopped after one complete round plus a second default point. TTFT spread was 0.6%, tok/s 1.1%, and MD5s matched. Its documented baseline flags omit `COLI_V4_MOE_GROUPED=1`, which current source requires before the grouped/MIN_N path runs; no engagement counters were captured. Classify it as inactive or engagement-unproven. A separate earlier p064 `--max-tokens 1` probe, interleaved N=3, did prove engagement: MIN_N=3 increased Metal row share 49.6% to 56.9% but was 0.81% slower than 4, inside TTFT noise, on the older kernel.
- **Expected gain:** unknown with simd_exact; neither the invalid E101 sweep nor the older ordered-kernel flatness check closes it.
- **Work/risk:** no implementation work; low configuration/numerical risk because batching preserves the current kernel and fallback, but an over-low threshold can send small groups into an end-to-end dispatch regime whose crossover is not established for simd_exact.
- **Smallest decisive experiment:** first extend the standalone production-shape S-scaling probe to S=3. Use that result to choose one challenger, likely MIN_N=3 versus default 4, then run a two-arm p256 N=2 directional check with explicit `COLI_V4_MOE_GROUPED=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold COLI_V4_METAL=0`. Require engagement histogram, same-arm controls, multi-chunk text diff, taskcheck, TTFT, and tok/s. Escalate to p512 or N>=5 only if the signal is plausible but unresolved, and ask before that long run.
- **Kill criterion:** the S=3 microprobe shows insufficient headroom to pay dispatch/wait, engagement barely changes, both axes remain inside noise at a multi-chunk prompt, taskcheck regresses, or same-arm nondeterminism is unacceptable. A changed MD5 alone is not a failure.

#### 7. Validate batched prefill rows16 on the current stack

- **Mechanism:** `COLI_V4_MOE_BATCHED_ROWS16=1` lets hot-pinned rows16 experts whose groups already clear MIN_N enter the batched Metal prefill path; it is distinct from single-token decode admission.
- **Host fit:** E85 found 251 groups were excluded for layout rather than insufficient S, so this captures already-large prefill groups without lowering the efficiency gate.
- **Evidence:** E86, interleaved N=3: p064 TTFT 38.374 to 36.595 s (-4.64%) and p256 102.061 to 96.453 s (-5.49%), with non-overlapping ranges. A p256, 60-token, N=2 differential was deterministic within both arms but produced different MD5s. Meaning retention was established only on the short prompt; p256 text/capability and decode tok/s were not assessed.
- **Expected gain:** measured ~5% TTFT against the older ordered-Metal stack; current incremental gain on fast-kernels+simd_exact is unknown.
- **Work/risk:** code already exists, so measurement work is small; numerical/capability risk is medium because the multi-chunk output changes deterministically.
- **Smallest decisive experiment:** after candidate 4 establishes the cold-layout composition, compare that arm with/without `COLI_V4_MOE_BATCHED_ROWS16=1` at p256. Run same-arm controls first, capture engagement, diff the generated text, run taskcheck, and report TTFT and tok/s from the same token-generating runs.
- **Kill criterion:** no incremental TTFT gain outside noise, material decode regression, taskcheck failure, or unacceptable same-arm nondeterminism. Deterministic wording changes with capability retained are admissible under the project's task-level bar.

#### 8. Phase-dependent Metal attention

- **Mechanism:** use `COLI_V4_METAL_ATTN=1` for batched prefill, then dispatch decode attention to the faster/safer CPU path. Leave routed experts on CPU.
- **Host fit:** prefill exposes enough parallel rows for the 40-core GPU; single-token decode is dominated by dispatch/synchronization and small matrix shapes.
- **Evidence:** current all-Metal-attention arm measured -15.9% TTFT at p064 N=3 and produced the same p064 output MD5. That one-chunk equality is not a multi-chunk bit-exactness proof. Decode trended -5.25%, but ranges overlapped and N=3 is below the repository's threshold for resolving a sub-10% decode delta.
- **Expected gain:** retain a meaningful fraction of the measured TTFT win while intentionally leaving decode unchanged. The exact composed number is unknown.
- **Work/risk:** small/medium; numerical risk low if phase routing keeps existing kernels unchanged.
- **Smallest decisive experiment:** first add/verify engagement counters, then p256 or p512, CPU+fast-kernels versus CPU+fast-kernels+prefill-only Metal attention. Run each arm against itself first; diff multi-chunk generated text and run taskcheck before any correctness verdict. Two points can reject a flat/bad arm; use N>=5 only if decode differs by under 10% and needs resolution. Report TTFT and tok/s from the same token-generating runs.
- **Build integrity:** compile the experimental binary with `make -C c -f Makefile.deepseek-v4 METAL=1 deepseek-v4 -j8`; the Makefile defaults Metal to 0. Before measuring, verify `nm c/deepseek_v4 | grep -c coli_v4_metal_expert_forward_batch` is nonzero even though routed experts remain disabled at runtime.
- **Kill criterion:** no TTFT improvement beyond noise at a prompt where multiple chunks execute, any >3% decode regression, taskcheck failure, or unacceptable same-arm nondeterminism. A changed MD5 alone is not a failure.

#### 9. Measure single-token Metal rows16 admission

- **Mechanism:** `COLI_V4_METAL_ROWS16=1` lets the existing single-expert decode entry accept hot-pinned `block_rows=16` experts. This is distinct from `COLI_V4_MOE_BATCHED_ROWS16`, which affects batched prefill groups.
- **Host fit:** E97 counted 2218 of 4902 expert calls (45.2%) rejected solely because they were rows16. Enabling the flag raised Metal expert coverage from 2684/4902 to 4902/4902 on this M3 Max.
- **Evidence:** E97's p064 24-token profile was a single run (N=1). The widened arm passed taskcheck 5/5 at 120 tokens, N=1, with byte-identical taskcheck text. On the 60-token golden prompt it deterministically changed only `FFN layers.` to `FFN layer.` Performance was not measured. E105's 8.6% CPU-over-Metal result did **not** enable this flag, so it does not settle the rows16-admitted lane.
- **Expected gain:** unknown. Doubling GPU call coverage is a population bound, not a speed forecast; the rows16 hot kernel differs from `simd_exact_cold` and could still lose to CPU.
- **Work/risk:** existing opt-in code makes measurement work small; numerical risk is medium and synchronization risk remains plausible, although deterministic output weakens the original race hypothesis.
- **Smallest decisive experiment:** with operator approval for the multi-chunk cost, run p256 N=2 arms for CPU+fast-kernels, Metal+simd_exact+fast-kernels, and the same Metal arm plus rows16 admission. Capture coverage counters, run each arm against itself, diff text, run taskcheck at N>=3 if the directional result is promising, and report TTFT and tok/s.
- **Kill criterion:** rows16 admission does not improve the Metal arm outside noise, the composed arm remains materially behind CPU+fast-kernels on this host, taskcheck regresses, or same-arm nondeterminism is unacceptable. A one-token wording difference alone is not a failure.

#### 10. Decouple whole-prompt MoE batching from attention chunks

- **Mechanism:** keep attention in causal 64-token chunks, but for each layer buffer the post-attention/FFN-normalized rows across all prompt chunks and run one grouped MoE schedule over the whole prompt. This increases same-expert group size without changing the seven engine paths that assume `batch <= 64`; the existing wave scheduler handles more unique experts than cache capacity.
- **Host fit:** the 40-core GPU gains efficiency when more rows share an expert dispatch, while the current 64-token attention chunk pins mean group size near the flat part of the Metal kernel curve. The proposed buffer is about 13.1 MB at 799 tokens (`799 * 4096 * 4`), small relative to 128 GB unified memory, though expert weights and waves still dominate traffic.
- **Evidence:** E62 ran one p256 route dump covering 184 prompt tokens, three chunks, and 43 layers. Offline union by layer raised measured mean same-expert group size from 4.14 per chunk to 7.99 at whole-prompt scope, 1.93x higher. Interpolating E60's existing kernel curve changes expert cost from 203.2 to 137.2 microseconds/token, a **1.48x expert-kernel-cost estimate only**, not an end-to-end speedup. Early layers touched 226-230 unique experts while later sampled layers touched 126-151, implying two waves in the widest layers. No whole-prompt engine implementation or performance A/B exists.
- **Expected gain:** meaningful prefill-MoE headroom if the measured route distribution repeats, bounded below the 1.48x expert-kernel estimate once attention, route construction, buffering, waves, and I/O are included. It should not improve S=1 decode tok/s directly.
- **Work/risk:** medium/large scheduling change. Risks are residual/order preservation, layer/chunk state lifetime, buffer memory, wave/cache churn, cancellation, and accidentally violating causal attention order.
- **Smallest decisive experiment:** before implementation, repeat the route dump on two representative multi-chunk prompts, preferably p256 and p512, to verify N, per-layer unique counts, and wave demand. If stable, add the smallest opt-in layer-local buffer/schedule while leaving attention chunking untouched. Then require engagement counters, same-arm controls, multi-chunk text diff, taskcheck, TTFT and tok/s from token-generating runs, and memory/expert-wait/wave counters. Ask before the long route or engine runs.
- **Kill criterion:** whole-prompt N fails to improve materially on representative prompts, projected expert-phase gain cannot clear total scheduling/I/O overhead, memory/compression worsens, taskcheck regresses, or the end-to-end TTFT gain is within noise. Do not kill it merely because outputs are token-different if meaning and task capability survive.

#### 11. O(1) expert hit index plus bounded exact-LRU miss selection

- **Mechanism:** port `3a217ef2` (`slot_by_expert`) and `45fb8b2` (per-layer LRU/next-empty selection) from `origin/dev`; remove full resident-slot scans on hits and misses.
- **Host fit:** arithmetic-neutral CPU bookkeeping in the disk/cache path; portable to Apple Silicon.
- **Evidence:** source-level mechanism only. Current prefill trace already labels `hit_slot_scan` and `miss_slot_select`; no host-local end-to-end number was found.
- **Expected gain:** unknown and bounded by measured scan/selection time. Do not assume upstream improvement.
- **Work/risk:** small/medium; low model-correctness risk, moderate cache-invariant/concurrency risk.
- **Smallest decisive experiment:** profile existing counters at p256/p512 with CPU+fast-kernels. Port only if the combined share is material; then run same-arm determinism, multi-chunk output differential, taskcheck, and both speed axes.
- **Kill criterion:** scan/selection under ~1% of relevant wall, no A/B improvement outside the 0.6-0.8% TTFT noise floor, taskcheck regression, or unacceptable same-arm nondeterminism. A changed MD5 alone is not a failure.

#### 12. QDQ the attention input once for `wq_a` and `wkv`

- **Mechanism:** port `7c6cdc8`; the projections consume the same raw input, so share the quantize/dequantize preparation.
- **Host fit:** reduces CPU work in attention without changing projection math. Upstream tiny-oracle and real-text checks were byte-identical.
- **Evidence:** upstream correctness, no host-local performance number. E77's 0.35%-of-wall QDQ result came from a single p064 on-lane Metal profile (N unavailable in the entry) and is not transferable to the winning CPU arm.
- **Expected gain:** unknown; likely small unless counters show repeated QDQ is material.
- **Work/risk:** small; low numerical risk, moderate buffer-lifetime integration risk.
- **Smallest decisive experiment:** add or read QDQ timing in CPU+fast-kernels p064/p256; port only above a meaningful share, then N=2 directional A/B before paying for N>=5. Token-generating A/Bs must report TTFT and tok/s together, with same-arm determinism, text diff, and taskcheck.
- **Kill criterion:** QDQ share under ~1%, improvement within noise, taskcheck regression, or unacceptable same-arm nondeterminism. A changed MD5 alone is not a failure.

#### 13. Batch shared experts during prefill

- **Mechanism:** port the order-preserving batched FP8 matmul from `a2f9e3f` for `batch > 1`.
- **Host fit:** prefill has token-level batch; the current scalar/per-item route leaves CPU vector throughput unused.
- **Evidence:** upstream code and E102's p064, 24-token, N=1 decode profile show shared expert at 5.8% of decode, but that decode share does **not** quantify prefill benefit.
- **Expected gain:** TTFT-only hypothesis; no valid percentage.
- **Work/risk:** small/medium; low if accumulation order and fallback remain identical.
- **Smallest decisive experiment:** profile shared-expert prefill share on p256/p512, then port. A proven prefill-only `--max-tokens 1` run may use the TTFT harness and must explicitly say no decode tokens were generated; any token-generating run must report TTFT and tok/s. Run same-arm determinism, diff multi-chunk text, and run taskcheck before judging correctness.
- **Kill criterion:** prefill share immaterial, improvement within noise, taskcheck regression, or unacceptable same-arm nondeterminism. A changed output hash alone is not a failure.

### P2: high-value conditional projects

#### 14. Reusable system-prefix checkpoints

- **Mechanism:** selectively port behavior from `51638e8`: in-memory LRU checkpoints, optional persistence, cancellable/resumable prefill segments, and safe fallback.
- **What already exists:** exact same-session strict-extension reuse. The missing value is reuse across sessions/restarts and reuse of a stable system/tool prefix when later conversation tokens differ.
- **Host fit:** avoids recomputing long fixed prefixes and repeated disk/compute work; especially relevant to agent/tool schemas.
- **Expected gain:** potentially most of the repeated prefix's prefill time, minus checkpoint restore; no decode tok/s gain.
- **Work/risk:** large. The upstream commit is ~1,600 changed lines and must not be cherry-picked blindly into this divergent branch.
- **Smallest decisive experiment:** first record real prompt prefix lengths and reuse frequency. If material, port only the minimal in-memory checkpoint behavior, maintain the 64-token batch contract, and compare the same long multi-chunk request with/without restore. Report engagement, TTFT, tok/s (expected flat), total wall, same-arm determinism, text diff, and taskcheck.
- **Kill criterion:** common prefixes are short/infrequent, checkpoint restore approaches recompute cost, or memory pressure harms expert residency.

#### 15. Non-bit-exact 16-column FP8 projection kernel

- **Mechanism:** vectorize across 16 columns so NEON amortizes FP8 decode; this changes floating-point summation order and must use the task-level correctness gate.
- **Evidence:** in the standalone four-real-shape microprobe (no prompt/N), the bit-exact lane-per-row design measured 1.00-1.01x and arithmetic four-wide decode measured 0.57x. At least 29.2% of decode passes through the relevant FP8 projections in E102's p064, 24-token, N=1 profile.
- **Amdahl bounds:** with phase share `p=0.292`, whole-decode speedup is `1 / ((1-p) + p/s)`. At `s=1.33`, 1.078x (+7.8%); at 1.5, 1.108x (+10.8%); at an optimistic 2x, 1.171x (+17.1%). These are ceilings conditional on achieving `s`, not forecasts.
- **Expected gain:** unknown until a new microkernel clears the numeric gate; the defensible range is the conditional Amdahl ceiling above, not a prediction.
- **Work/risk:** medium/high; highest numerical risk among plausible single-user decode candidates because the shared quant header serves several engines.
- **Smallest decisive experiment:** stage 1 is a standalone real-shape numeric microprobe only: speed, error distribution, repeatability, and real tensor shapes; no taskcheck is possible there. Only if it exceeds the microkernel threshold should stage 2 integrate an opt-in engine arm and run same-arm determinism, multi-chunk text diff, taskcheck, TTFT, and tok/s.
- **Kill criterion:** microkernel below ~1.33x, taskcheck regression, unstable output, or composition under ~5% end-to-end decode gain.

#### 16. Multi-KV-slot continuous batching

- **Mechanism:** add independent KV/session slots, a token scheduler, and batched decode kernels so several requests share each expert/attention dispatch.
- **Host fit:** raises arithmetic intensity and can feed the 40-core GPU with larger groups; aligns with PagedAttention/continuous-batching methods.
- **Current blocker:** the launcher explicitly supports exactly one KV slot for DeepSeek V4.
- **Expected gain:** potentially large aggregate requests/second at concurrency; approximately no benefit to a lone request and possibly worse latency under contention.
- **Work/risk:** large/extra-large; KV isolation, cancellation, fairness, memory budgeting, and reproducibility all become first-class concerns.
- **Smallest decisive experiment:** only after the MIN_N gate is retuned and a real concurrent-serving requirement exists, implement two slots plus a synthetic two-client scheduler. Report aggregate tok/s, **single-request/per-request tok/s separately**, TTFT, p50/p95 latency, and memory/swap. For every stream, run same-arm repeatability, a multi-chunk ON/OFF text differential, output-isolation checks, and taskcheck.
- **Kill criterion:** single-user is the only workload, or two-slot memory/latency cost dominates aggregate gain.

### P3: investigate only after attribution

- **Shared expert cache across prefill layers (`a0ba8582`):** may reduce reads in true CPU batches, but integration touches current hot-cache/rows16 behavior. Only pursue after a prefill trace shows the misses.
- **Duplicate load coalescing (`95e1aa34`):** useful only after concurrency/prefetch can request the same expert simultaneously.
- **Long-context KV quantization/paging:** improves context capacity and concurrent comfort, not short-prompt decode. It needs accuracy and memory-pressure gates.
- **Dimension-specific/fused MLA attention:** potentially fits the GPU, but the custom MLA/DSA shapes and current dispatch need attribution first. Generic MLX fused-attention gains are not transferable.
- **Dedicated hot-layout `simd_exact` kernel:** only after candidate 9 measures the existing ordered-hot rows16 admission. E97's current flag is already a distinct live measurement lane; building a new hot kernel is the lower-priority follow-on if admission improves coverage but the ordered-hot kernel remains the limiter.
- **Lift the batch-64 contract:** only if prefill profiling proves larger batches can materially improve enough phases to pay for a broad buffer/API audit.
- **Real external SSD mirror:** current code supports model mirroring, but test only with an actual fast physical Thunderbolt/NVMe device. The mounted backup volume is not it.

## Approaches not worth more time now

| lever | local result/reason |
|---|---|
| Current S=1 Metal routed experts | p064, 60 tokens, N=2: stacked arm is 8.6% slower in decode than CPU+fast-kernels. This does not kill S>=4 batched prefill MoE. |
| E90 six-expert argument-buffer fusion | p064, 24 tokens, N=3: -8.18% tok/s and three output MD5s; slower and racy |
| E100 one-command-buffer top-k fan-out | p064, 60 tokens, N=3, on simd_exact: -7.33% tok/s, +19% TTFT; deterministic, meaning retained, reverted |
| Eager expert prewarm | four-prompt paired workload, per-prompt N unavailable in E26 summary: -29.7%, worsening by prompt |
| Loader-lane expansion | upstream headline did not reproduce locally |
| MTP/ngram speculation | acceptance/replay cost loses on real chat |
| OMP 12 threads | p064, 60 generated tokens, N=2 per arm: 1.05% slower, noise |
| Wider pinned rows16 coverage | p064, N=2: -7.79% tok/s, +18% TTFT |
| Exact NEON FP8 | standalone four-real-shape microprobe, not a prompt/N run: 1.00-1.01x; compiler already exploits the four-row structure |
| Arithmetic four-wide FP8 decode | same standalone microprobe: 0.57x |
| More RAM for frozen p064 | p064, N=2: 48/72/96 GB flat within 1.2% |
| `COLI_V4_PREFILL_CHUNK=128` | p256 sweep failed before a valid arm/N; engine-wide batch contract caps at 64 |
| Unlocked hot packing | p064, `--max-tokens 1`, N=3: -0.05% TTFT with fully overlapping ranges |
| Dual-device striping to Time Machine volume | storage is not a measured low-latency physical mirror |
| Full MLX-model runtime rewrite now | custom DeepSeek V4 quantization, expert streaming, MLA/DSA, and cache semantics are not a drop-in MLX graph; large work with no local gain evidence |

## Ordered measurement backlog

No long run should start without operator approval. The lowest-cost decision sequence is:

1. **Operational replay after disk cleanup:** one representative heterogeneous chat trace, before/after only if a clean seed and environment can be preserved. Purpose: determine whether storage pressure was contributing to expert wait/comfort.
2. **Profile-only attribution:** existing CPU+fast-kernels, p256 or p512, one run with counters for hit-slot scan, miss selection, QDQ, shared-expert prefill, attention phase, and expert wait. Purpose: eliminate remote candidates below ~1% before porting anything.
3. **Existing grouped/batched-MoE composition gate:** three p256 arms, N=2 initially: current CPU+fast-kernels; plus CPU grouping; plus grouped+batched Metal MoE+`simd_exact_cold`. Keep `COLI_V4_METAL=0` so single-token decode experts stay on CPU. Include engagement, same-arm determinism, multi-chunk text diff, taskcheck, TTFT, and tok/s. This tests the strongest current prefill-Metal composition and decides whether E84's historical CPU-grouping/~11% increment survives the current baseline.
4. **Live prefetch composition gate:** p256, N=2 per arm initially, CPU+fast-kernels with/without `COLI_V4_PREFILL_PREFETCH=1`. Purpose: determine whether E55's historical -2.70% p256 TTFT survives the current winner and whether tok/s remains flat. Require engagement, same-arm controls, multi-chunk text diff, taskcheck, expert-wait/route-ahead counters, TTFT, and tok/s.
5. **MIN_N crossover:** add S=3 to the production-shape microprobe, then use the result for a two-arm p256 N=2 threshold check with the grouped, batched, simd_exact, and CPU-decode flags written explicitly. This must happen before multi-slot work and avoids repeating E101's engagement-unproven sweep.
6. **Batched prefill rows16:** only after step 3, compare the winning cold-layout arm with/without `COLI_V4_MOE_BATCHED_ROWS16=1`; run the full same-arm/text/task/both-axes gate.
7. **Phase-dependent attention directional A/B:** two arms, N=2 initially, long enough to execute multiple chunks, with same-arm controls, text diff, taskcheck, TTFT, and tok/s from the same token-generating runs. Expand to N>=5 only if a sub-10% decode delta must be resolved.
8. **Single-token rows16 admission:** p256 N=2 three-arm CPU/Metal/Metal+rows16 check with coverage counters and the full capability/determinism gate. This is separate from step 6 and E105 did not settle it.
9. **Whole-prompt MoE route validation:** before changing scheduling, repeat E62's route union on two representative multi-chunk prompts and calculate N, unique experts, and waves per layer. Only implement the opt-in buffer/schedule if the route-shape gain repeats; then run the full capability/determinism/both-axes gate. Ask before either long run.
10. **Direct index + LRU port:** only if step 2 shows material slot-management wall. Validate same-arm determinism first, then multi-chunk text differential, taskcheck, TTFT, and tok/s.
11. **QDQ reuse or shared prefill batch:** choose whichever has the larger attributed wall; do not run both and lose causal clarity. Any generated continuation requires both axes.
12. **Non-bit-exact FP8 probe:** numeric microbenchmark only; it generates no tokens. Do not touch the shared engine header unless it exceeds 1.33x. Taskcheck and both performance axes apply only after an opt-in engine arm exists.
13. **Prefix checkpoint workload audit:** measure repeated stable prefix tokens before any large feature port.

## External mechanism fit

- Apple's [Metal command-buffer guidance](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html) recommends the fewest command buffers that still keep the GPU utilized because frequent submissions can introduce CPU stalls. That matches the local failure of tiny per-expert Metal dispatch and supports phase-level prefill work.
- Apple's [High Power Mode guidance](https://support.apple.com/en-us/101613) says higher fan speeds may allow higher performance in intensive workloads; this is a comfort/sustained-power setting, not an engine percentage.
- [PagedAttention](https://arxiv.org/abs/2309.06180) is relevant to multi-slot serving and KV memory efficiency, not the next single-request decode fix.
- [KVQuant](https://arxiv.org/abs/2401.18079) is relevant when long-context KV capacity becomes the constraint.
- MLX exposes [quantized matrix multiplication](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul), but Colibri's model format and custom sparse/disk-streaming runtime make whole-model MLX support a backend/conversion project, not a low-effort kernel switch. MLX percentages from other models/hardware are not evidence for this host.

## Local evidence map

- Repository rules and metric/correctness gates: `AGENTS.md`
- Full experiment ledger: `experiments_results.md`, especially E52-E56, E58-E62, E84, E86-E88, E91, E96-E102, E104-E105, E107-E108
- Decode/TTFT harness: `.backlog/lab/tokps.sh`
- Task correctness harness: `.backlog/lab/taskcheck.sh`
- Current runtime/flags/prefix reuse: `c/deepseek_v4.c`
- Launcher memory/thread/one-slot behavior: `c/coli`
- Prefix reuse tests: `c/tests/test_deepseek_v4_prefix.py`
- Remote candidates: commits `3a217ef2`, `45fb8b2`, `7c6cdc8`, `a2f9e3f`, `a0ba8582`, `95e1aa34`, `51638e8`

## Bottom line

For this M3 Max, indiscriminate “more Metal” is not the answer. The measured best decode path is CPU-routed experts plus the fast sparse-attention/router kernels. First restore internal SSD headroom and use a persistent 64 GB server profile. Then compose the existing S>=4 batched-MoE prefill path with `simd_exact_cold`, revalidate the live prefill-prefetch flag, retune MIN_N, and validate the prefill/decode rows16 flags as separate lanes before adding phase-dependent attention. Whole-prompt MoE decoupling is the strongest measured scheduling-architecture lead, but its 1.48x number applies only to interpolated expert-kernel cost and needs route-shape validation plus an end-to-end implementation. In parallel, profile and selectively port the arithmetic-neutral upstream cache-index/QDQ/shared-prefill improvements. Treat non-bit-exact FP8 and multi-slot continuous batching as separate, higher-risk bets for single-user decode and aggregate throughput respectively.
