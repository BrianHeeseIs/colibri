# Ultrawork Notepad — rank maximum-performance candidates for this M3 Max
Started: 2026-08-28T21:03:00+02:00

## Plan (exhaustively detailed)

1. Inventory the current branch, hardware, tracked and untracked performance artifacts, and recent Git history without altering shared work.
2. Read all progress-bearing experiment ledgers, durable notepads, benchmark harnesses/logs, current code flags, and recent performance commits through the latest entry.
3. Reconcile evidence quality: metric definition, prompt, N, cache/seed state, engagement, determinism, capability, noise floor, stale baselines, retractions, and structurally inert arms.
4. Build a current bottleneck/configuration model for Mac15,9 M3 Max: CPU, GPU, unified-memory pressure, SSD streaming, attention, expert compute/wait, TTFT, and decode.
5. Classify every meaningful lever as shipped/default, opt-in verified win, conditional, pending high-value candidate, low-value, killed, or superseded.
6. Compare current local evidence with primary Apple/MLX/llama.cpp/sparse-inference methods, excluding percentages from other hardware and Apple10/M5-only acceleration.
7. Rank the smallest high-value candidates by expected end-to-end effect, confidence, engineering work, correctness risk, measurement cost, and comfort impact.
8. Write `.backlog/m3-max-performance-candidates-2026-08-28.md` as the durable recommendation and produce a concise user-facing answer.
9. QA the report by complete read, source-anchor spot checks, arithmetic/metric checks, and current Git-safety inspection; no long model benchmark will run without operator approval.
10. Obtain unconditional HEAVY reviewer approval, reconcile any criterion-cited blockers, confirm all child agents are terminal, update the ledger, and stop.

## Success criteria + QA scenarios

- Deliverable/tier: a durable, source-backed candidate ranking plus concise answer; **HEAVY** because the user requested maximum-performance review and the corpus contains conflicting axes, retractions, noisy decode data, and concurrent work.
- Criterion 1 — corpus completeness/happy path: the report covers current progress through the latest experiment/commit and names the fastest verified configuration. Scenario: `git log -80 --date=short --pretty=format:'%h %ad %s'` plus full reads of `experiments_results.md`, `.backlog/*.md`, and cited logs; PASS iff every post-`7f93efe` performance lane is classified and the report includes E105/E107-era evidence. Evidence: this notepad and final report source matrix. Pure-prose target: no machine-consumed RED seam; the pre-existing `claude.md` is stale context, not a test assertion.
- Criterion 2 — metric integrity/edge cases: TTFT and tok/s are never conflated; every headline local number includes prompt and N or explicitly says unavailable; deterministic/token-exact/capability distinctions and structurally inert short-prompt tests are preserved. Scenario: read `AGENTS.md`, `.backlog/lab/tokps.sh`, `bench/ab.sh`, and cited logs; PASS iff a manual claim audit finds zero mislabeled metric, sign, prompt, N, or correctness claims. Evidence: report evidence-quality table. No RED seam exists for prose; QA-by-read is binding.
- Criterion 3 — host-specific ranking/adversarial risk: each recommended candidate states the current bottleneck, why it fits M3/Apple9/128GB, expected gain as measured/bound/hypothesis, risk, smallest falsifiable experiment, and what local negative evidence could kill it. Scenario: complete-read the ranking against current code/ledger and independently recompute any Amdahl/ratio math; PASS iff every P0/P1 candidate has all fields and no external-hardware percentage is projected. Evidence: report candidate table and calculation notes.
- Criterion 4 — comfort/regression: recommendations include a fastest-now configuration, memory/compression margin, power/thermal/OMP guidance, determinism/capability caveats, one-engine rule, and benchmark-cost discipline. Scenario: compare report with `AGENTS.md`, hardware profile, resource planner/telemetry, and current logs; PASS iff it gives an actionable daily configuration and names tradeoffs. Evidence: report operational section.
- Criterion 5 — repository safety/adjacent surface: no tracked source change, model run, long benchmark, commit, push, or unrelated-file mutation. Scenario: final `git status --short`, `git diff --stat`, `git rev-parse HEAD`, child-agent status; PASS iff only task documentation/notepad changes are attributable here and no child/runtime remains active. Evidence: final notepad receipt.
- WHEN TO STOP: I'll stop right away when the report passes all five criteria, the HEAVY reviewer approves unconditionally, all children are terminal, and Git/resource safety is recorded.

## Skills used

- `omo:ultrawork` — binding evidence/review/notepad workflow explicitly requested by `ulw`.
- `omo:git-master` — read-only HISTORY/STATUS investigation is required to cover recent progress.
- `data-analytics:analyze-data-quality` — benchmark evidence must be reconciled across metric definitions, noise, stale baselines, retractions, and correctness gates.
- `omo:programming` — read because the launcher surface `c/coli` is Python; investigation remained read-only and no source implementation was made.
- Skipped report-generation plugins: the user requested a direct technical recommendation, and a repository Markdown artifact is the matching durable surface.

## Now

- Final gate: obtain unconditional HEAVY approval of the revised report, then record repository/agent safety and close the goal.

## Todo

- [x] Integrate all experiment-ledger evidence through E108.
- [x] Integrate backlog/log candidates and operational constraints.
- [x] Integrate current code/flag/hot-path map.
- [x] Integrate recent Git-history classifications.
- [x] Integrate primary-source external candidates.
- [x] Build evidence-quality and bottleneck model.
- [x] Rank candidates and write durable report.
- [x] Run complete-read/source/arithmetic/Git QA.
- [x] Run HEAVY reviewer loop.
- [x] Record cleanup/safety and final handoff.

## Findings

- Current branch is `simd-apple-metal`, HEAD `a942b99d727b5a64eea765d8fb682121b5eee7c1`, with no configured upstream. Task-start untracked files were `.backlog/ulw-colibri-metal-mlx-research.md`, `claude.md`, and `validation/fp8_neon_probe`; preserve them.
- Local hardware revalidated: Mac15,9 M3 Max, 16 CPU cores (12P+4E), 40-core GPU, 128GB unified memory, Metal 4, macOS 26.6.1. The host also drives an external 3840x1080 display, relevant to comfortable thermal/memory use.
- Repository progress has advanced from the previous report's E84 endpoint through at least E107. Recent commits identify `COLI_V4_KERNELS=all` with Metal off as the fastest configuration, while Metal rows16/simd_exact improved its own lane but still lost the fully stacked CPU comparison.
- No long model benchmark is authorized in this task. Repository rules require sizing and operator approval first; research will use existing captured measurements only.

## Learnings

- A result labeled “speedup” is unsafe until its axis is identified: `bench/ab.sh` is TTFT-only, while `.backlog/lab/tokps.sh` reports decode tok/s and determinism.
- A passing short golden can be inert for batch/group/chunk gates; engagement counters and multi-chunk differentials are mandatory before “bit-exact” claims.
- The fastest comfortable configuration may differ from the fastest isolated kernel or fastest Metal-only arm; compare composed end-to-end arms.

## Discovery receipt — corpus through E108

- `experiments_results.md` now ends at E108. E105 is the decisive composed measurement: p064, 60 generated tokens, N=2, deterministic within each arm. CPU base is 1.42395 tok/s / 43.42 s TTFT; CPU + `COLI_V4_KERNELS=all` is 1.66755 tok/s (+17.11%) / 39.58 s TTFT (-8.8%) / 74.96 s total (-11.7%); Metal `simd_exact_cold` + `KERNELS=all` is 1.5359 tok/s / 42.40 s TTFT. The CPU fast-kernel arm beats the fully stacked Metal expert arm by 8.6%.
- E88 is the stronger decode sample for `KERNELS=all`: p064, 24 generated tokens, N=5, +17.08% tok/s and -9.6% TTFT, taskcheck 5/5. Its short-prompt ON arm produced two stable MD5 variants; capability was retained but reproducibility was not.
- E91 gives length coverage at N=2: TTFT -10.44/-17.99/-20.23% and tok/s +16.66/+12.08/+20.12% at p064/p256/p512. Decode is non-monotonic and N=2, so the exact intermediate deltas are directional, not resolution-grade.
- E102's corrected p064 profile (24 tokens, accounted 99.4%, CPU 1.41 tok/s) is the current bottleneck map: attention 38.7%, expert_forward 27.0%, expert_wait 9.7%, shared expert 5.8%, head 5.1%, router 4.7%, norm 4.0%, indexer 2.3%, compressor 2.0%. `KERNELS=all` accelerates `attn_sparse` + router, 16.4% of that measured wall.
- E104 killed the easy exact NEON FP8 hypothesis: real projection shapes measured 1.00-1.01x for the bit-exact lane-per-row design and 0.57x for arithmetic decode. The only remaining SIMD route changes reduction order by vectorising 16 columns; it is non-bit-exact and covers at least 29.2% of decode.
- E107 killed 12-thread OMP tuning (-1.05%, noise) and wider pinned rows16 coverage: pins158 was -7.79% tok/s and +18% TTFT versus the existing cap16. E108 found 48/72/96 GB flat within 1.2% on the frozen seeded p064 working set, but explicitly does not cover heterogeneous chat where 3-11 s expert disk time was observed.
- E108 also closed `COLI_V4_PREFILL_CHUNK>64` as a knob. Four batched APIs reject `batch>64`; widening means lifting an engine-wide contract and auditing all backing buffers, not changing one loop constant.
- The Git-history audit covers 58 commits after `7f93efeb`, ending at `a942b99d727b5a64eea765d8fb682121b5eee7c1`. It contains no unclassified win after E105; later commits correct telemetry, refute OMP/pin changes, and close residency/chunk work.

## Live architecture receipt

- `c/Makefile.deepseek-v4` defaults `METAL ?= 0`. A plain build omits the seam. The fastest current expert path is therefore the default CPU backend.
- `COLI_V4_KERNELS=all` enables exactly two runtime kernels: reassociated `attn_sparse` and `router`. The supported launcher alias is `--fast-kernels`; serve mode now honors the environment variable too.
- Metal attention is independent of Metal experts: `COLI_V4_METAL_ATTN=1` lazily initializes the Metal backend and leaves `COLI_V4_METAL=0`, so the known-slower routed-expert seam can remain off. E87 measured it bit-exact at p064, N=3: TTFT -15.9%; decode -5.25% directionally but unresolved because ranges overlap. This makes phase-dependent attention the strongest small single-user implementation candidate.
- The Metal expert path's `simd_exact_cold` kernel is a real 20-34% within-lane improvement but the lane remains slower end-to-end. About 45% of single-token expert calls are rows16 and rejected; `COLI_V4_METAL_ROWS16=1` admits them, preserves task capability 5/5, but changes one of 60 golden-prompt tokens and remains opt-in.
- Native serve has one active KV slot. Raising decode S through concurrent requests therefore requires KV-slot management plus a batching scheduler; it can improve aggregate service throughput, not single-chat latency.
- Current machine state is AC power, high-power mode (`powermode 2`), ~0.94 GB occupied compressor, but 10.0 GB swap still allocated after a 17-day uptime. The old heterogeneous-workload ledger found `--ram 96` materially faster when the desktop was quiet and uncompressed, while `--ram 64` was the largest budget that reliably stayed uncompressed under 18-31 GB of normal desktop load. E108's flat seeded sweep does not supersede that different-workload result.

## Candidate filters established

- Highest-value small candidate: phase-dependent Metal attention, GPU for prefill and CPU for decode. Local bound is the already measured -15.9% TTFT with no intentional decode change; the exact composed result with `KERNELS=all` still needs a sized A/B.
- Highest-value numerical-risk candidate: non-bit-exact 16-column FP8 decode. Amdahl bounds from the verified 29.2% share are 1.08x whole-decode if the phase reaches 1.33x and 1.17x even at an optimistic 2x phase speedup. The existing arithmetic-decode prototype is slower, so a new packed/LUT-friendly design is required; do not present the bound as an expectation.
- Highest-value structural throughput candidate: multi-request continuous batching to raise expert S. It fits the 40-core GPU, but is a large serving architecture project and benefits aggregate throughput rather than one user's latency.
- Conditional comfort candidate: adaptive residency/history for heterogeneous prompts. The seeded benchmark says no generic speed gain; only a replay of real mismatched chat traces can justify cache-policy work.
- Lower-priority large projects: fused/dimension-specific MLA attention after shape/fallback attribution, lifting the batch-64 contract for prefill, expert-selective mixed precision, and KV-cache quantization/paging for long-context concurrency.
- Closed: Metal routed experts as currently composed, top-k command-buffer fusion, eager prewarm, loader-lane expansion, MTP/ngram speculation, wider pin coverage, 12-thread OMP, exact FP8 NEON, hot-pack unlocked, RAM increases for the frozen p064 set, and chunk widths above 64 as a runtime knob.

## Final synthesis inputs

- The internal APFS container is the host's most urgent comfort/performance constraint: 994.7 GB total, 987.1 GB used, only 7.5 GB (0.8%) unallocated. The model is roughly 155 GB and streams experts from this same storage while macOS also needs room for VM/swap activity. This establishes a strong operational recommendation to reclaim substantial internal storage, but it does **not** justify inventing a tok/s percentage or a universal free-space threshold.
- The mounted 16 TB “Backups of rptr9521” surface is a Time Machine/network disk-image path, not a proven low-latency Thunderbolt/NVMe model mirror. Do not recommend it for live expert shards. Current dual-device/mirror support only becomes relevant if the operator attaches and measures a genuinely fast physical SSD.
- Current same-session exact prefix reuse already exists in `c/deepseek_v4.c`: it records prompt plus generated ids, reuses only when the next prompt strictly extends the full recorded token sequence, and exposes `prefix_reused`. The large remote checkpoint commit is therefore a **conditional extension** for reusable system prefixes, restarts, and segmented long prefill, not a replacement for a missing feature.
- Remote `origin/dev` contains several portable, unmeasured candidates absent from this HEAD: direct resident-expert indexing (`3a217ef2`), bounded exact-LRU miss selection (`45fb8b2`), one attention-input QDQ shared by `wq_a`/`wkv` (`7c6cdc8`), batched shared experts during prefill (`a2f9e3f`), and broad prefix checkpoints (`51638e8`). None has a host-local end-to-end percentage, so each must remain a hypothesis until ported and measured against the CPU+`KERNELS=all` baseline.
- The direct-index plus bounded-LRU pair is especially clean numerically because it changes lookup/bookkeeping rather than model arithmetic. Existing trace labels (`hit_slot_scan`, `miss_slot_select`) provide the correct attribution seam; profile p256/p512 first and kill the port if those counters are immaterial.
- Attention-input QDQ reuse has upstream oracle/text parity evidence but no end-to-end timing. E77's 0.35%-of-wall QDQ observation belonged to a different Metal configuration, so it cannot kill or validate the CPU+fast-kernel opportunity. Measure its share in the actual winning arm.
- Apple's published Metal guidance favors using the fewest command buffers that still keep the GPU busy because excessive submissions create CPU stalls. That mechanism matches this repository's evidence that per-expert Metal orchestration/fusion attempts lose; it supports phase-level prefill dispatch, not blindly moving more tiny expert calls to Metal.
- PagedAttention/continuous batching and KV-cache quantization fit long-context or multi-client serving, but native V4 currently exposes exactly one KV slot. They are aggregate-throughput/comfort projects, not the next way to lower single-user decode latency.

## HEAVY review loop

- First independent verdict: **BLOCKERS**. The reviewer correctly found that the initial report omitted E84's live `COLI_V4_MOE_BATCHED=1` prefill path, which measured -10.79%/-11.70% incremental TTFT at p064/p256 N=5 against an older grouped+Metal-attention stack. The broad “do not move routed experts to Metal” statement was overbroad: S=1 decode experts lose, while prefill groups can reach S>=4 and measured a historical win.
- Corrections made: classified E84 as the first current-baseline composition candidate; required `METAL=1` build/symbol/engagement checks; required same-arm controls, multi-chunk text diff, taskcheck, TTFT and tok/s; explicitly retained decode on CPU.
- The reviewer also found missing prompt/N metadata, incomplete P0 fields, incomplete correctness gates, missing both-axis requirements in token-generating experiments, and an impossible request to run engine taskcheck before integrating the FP8 arm. All were corrected: benchmark numbers now name prompt/N or say unavailable/not applicable; every P0/P1 has host fit, evidence/gain, work/risk, smallest experiment, and kill criterion; changed MD5 alone never kills a candidate; the FP8 microprobe and later engine capability gate are separate stages.
- A fresh unconditional review is required after the complete-read QA of the revised report.
- Second independent verdict: **BLOCKERS**. It found five additional corpus/ranking gaps: E86 batched-prefill rows16 omitted; E97 single-token rows16 conflated with that lane and incorrectly treated as settled by E105; E96 `simd_exact_cold` missing from the E84 composition; E90 and E100 fusion attempts conflated; E101's short-prompt MIN_N null incorrectly omitted rather than classified as still open at useful lengths; and continuous-batching validation lacked per-request tok/s plus multi-chunk ON/OFF text checks.
- Corrections made: candidate 4 now composes grouped/batched prefill with `simd_exact_cold` and explicitly avoids multiplying separate gains; candidate 5 retunes MIN_N via an S=3 microprobe and a bounded p256 check before concurrency; candidate 6 separately validates `COLI_V4_MOE_BATCHED_ROWS16`; candidate 8 separately measures `COLI_V4_METAL_ROWS16`, noting E105 did not enable it and performance is unmeasured; E90/E100 have distinct dead-lever rows; continuous batching now requires aggregate and per-request tok/s, TTFT, per-stream multi-chunk differential, isolation, repeatability, and taskcheck.
- The measurement order now uses the strongest current Metal variant and isolates CPU grouping, cold batched Metal, prefill rows16, phase-dependent attention, and decode rows16 instead of folding them into one causal tangle. A third unconditional review is required.
- Third independent verdict: **BLOCKERS**. It found that E101 had been treated as an engaged MIN_N sweep even though its recorded flags omit the current source's required `COLI_V4_MOE_GROUPED=1`; the report also omitted the older positively engaged MIN_N=3 versus 4 probe. Corrected classification: E101 is inactive or engagement-unproven, while the older ordered-kernel probe increased Metal row share 49.6% to 56.9% and put MIN_N=3 0.81% slower, inside TTFT noise. The current `simd_exact_cold` crossover remains open and now requires the grouped prerequisite plus engagement counters.
- The same review found two omitted performance lanes. First, the live default-off `COLI_V4_PREFILL_PREFETCH` path: E55's interleaved TTFT-only N=3 evidence was -6.20% at p064 and -2.70% at p256; E56 generated 32 tokens and found identical ON/OFF text at both lengths; older fast-kernel composition evidence lacked a stated N and is now explicitly provisional. It is restored as a bounded current-baseline P1 revalidation candidate with both axes and route-ahead/expert-wait counters.
- Second, F19 whole-prompt MoE decoupling: E62's one p256, 184-token route dump measured mean N 4.14 to 7.99 and interpolated expert-kernel cost 203.2 to 137.2 microseconds/token (1.48x). The report now labels that as a route-shape/kernel-cost estimate, not an end-to-end gain, and requires representative route validation before an opt-in scheduling implementation.
- E107 metadata was corrected everywhere to p064, 60 generated tokens, N=2 per arm. The 12-thread arm was 1.05% slower, within noise; “four total runs” is no longer used as the experimental N.
- A fourth unconditional HEAVY review is required after a full report claim audit.

## Final QA receipt before reviewer pass 4

- Complete-read audit covered all 352 report lines after the prefetch/F19 revisions. Candidate numbering is contiguous 1-16, and the executive ranking now matches the detailed P1 ordering for the two existing-code composition gates.
- Every candidate has an explicit expected-gain classification, work/risk, smallest decisive experiment, and kill criterion. Token-generating proposals require TTFT and tok/s; TTFT-only historical evidence is labeled; non-engine microprobes explicitly defer taskcheck until integration.
- Arithmetic was independently recomputed: E105 CPU over stacked Metal = 8.572%; E52 prefetch incremental to its old fast-kernel arm = -9.542%; E62 kernel-curve ratio = 1.4810x; FP8 Amdahl bounds are 1.0781x/1.1078x/1.1710x at phase speedups 1.33x/1.5x/2x.
- Current source re-check confirms prefetch is live and default-off; grouped MoE is the outer gate for the batched/MIN_N path; prefill batch remains capped at 64. Report classifications match those source conditions.
- Safety check: HEAD remains `a942b99d727b5a64eea765d8fb682121b5eee7c1`; `git diff --stat` is empty because only untracked documentation artifacts exist; no `deepseek_v4` process is running. Task-owned files are the report and this notepad. Pre-existing untracked `.backlog/ulw-colibri-metal-mlx-research.md`, `claude.md`, and `validation/fp8_neon_probe` remain untouched. No model run, long benchmark, commit, or push occurred.

## HEAVY review pass 4 correction receipt

- Pass 4 returned **BLOCKERS** for two report-contract issues. First, E54's attribution/table and E56's max-tokens-32 equality result lacked explicit N-availability labels, and E56 did not state that same-arm repeatability was absent. The executive, evidence table, and detailed prefetch candidate now say E54/E56 per-arm N is unavailable where applicable and E56 did not report same-arm repeatability.
- Second, the executive and ordered backlog ranked whole-prompt MoE ahead of the small remote ports while the numbered matrix placed it after them. Whole-prompt MoE is now candidate 10, immediately before direct-index/LRU 11, QDQ 12, and shared-expert prefill 13; the P1 heading now covers both current-stack and measured-architecture candidates. Executive, detailed matrix, ordered backlog, and bottom line use the same order.
- Post-correction structural audit: candidate numbering remains contiguous 1-16 and every candidate retains expected gain, work/risk, smallest decisive experiment, and kill criterion. A fresh unconditional HEAVY review is required.

## HEAVY approval

- Fresh pass 5 verdict: **APPROVE**, unconditional. The reviewer re-read the current report and explicitly rechecked the E54/E56 metadata corrections plus the consistent whole-prompt-MoE ranking across the executive, matrix, backlog, and bottom line.

## Final safety and artifact receipt

- Final report: `.backlog/m3-max-performance-candidates-2026-08-28.md`, 352 lines, MD5 `00a8771a0a445968ee6789e87c7f7fd4`.
- Final HEAD remains `a942b99d727b5a64eea765d8fb682121b5eee7c1`; no commit or push occurred. `git diff --stat` is empty; the task added only the untracked report and this untracked notepad.
- Pre-existing untracked `.backlog/ulw-colibri-metal-mlx-research.md`, `claude.md`, and `validation/fp8_neon_probe` remain present and were not modified by this task.
- No `deepseek_v4` process is running. No model run or long benchmark was started, honoring the operator-approval rule.
- Final agent inventory shows every child research/review agent terminal; HEAVY pass 5 is recorded as unconditional `APPROVE`.
