# Ultrawork Notepad — Re-analyze E109+ and maximize M3 Max decode tok/s

Started: 2026-08-29T13:58:43+0200

## Plan (exhaustively detailed)

1. Resolve the final-material format before the first research wave: offer PDF+DOCX, Markdown, and standalone HTML with a technical template and record the user's binding choice.
2. Snapshot repository safety state: branch, HEAD, dirty/untracked files, latest experiment number, recent commits, host hardware, disk/RAM/power state, and active engine processes.
3. Launch the full first research wave across orthogonal axes: E109+ experiment audit; current decode call graph/profile; shipping-default composition; FP8/NEON memory behavior; Metal decode scheduling; CPU/cache/prefetch; MLX/llama.cpp/ggml implementations; Apple hardware/Metal guidance; quantization; concurrency/serving; comfort/thermal/storage; and a skeptical evidence lane.
4. Journal every worker return, source, observation, claim candidate, contradiction, and EXPAND lead immediately under `.omo/ulw-research/20260829-135843/`.
5. Run at least two expansion waves, opening each actionable lead and closing every excursion by the skill's EXIT rules.
6. Reconcile all repository experiments through the latest entry, separating TTFT, decode tok/s, total wall, task capability, token identity, determinism, prompt, N, and current default status.
7. Attribute current decode on this exact M3 Max from existing profiles and source, then produce an Amdahl-ranked opportunity map without treating a phase estimate as an engine result.
8. Deep-research host-compatible decode methods in primary papers, official documentation, and SHA-pinned open-source implementations; counter-search every high-risk claim.
9. Verify contested performance/compatibility claims with bounded code or source-level checks where feasible; do not start a model benchmark without operator approval.
10. Build the claim graph, intent diff, observation manifest, verification-economics ledger, and cause-disappearance ledger to convergence.
11. Write the approved technical deliverable with measured/derived/assumed labels, source citations, negative results, candidate ranking, smallest decisive experiments, and explicit kill criteria.
12. Render the approved deliverable, verify its asset manifest, and visually inspect every page/surface; correct only evidenced defects.
13. Run a complete claim/source/metric audit and repository-safety check.
14. Obtain unconditional HEAVY reviewer approval, fixing criterion-cited blockers within the allowed review loop.
15. Confirm every child agent is terminal, record cleanup and elapsed/source counts, close the goal, and hand off without committing or pushing.

## Success criteria + QA scenarios

- Deliverable/tier: an exhaustive, durable technical research report that reconciles all new Colibri improvements and ranks the strongest host-specific decode/tok-s and comfort opportunities. **HEAVY** because the user demanded deep research, the evidence includes concurrency/cache/default-path changes, and wrong performance advice would waste long benchmark time.
- Criterion 1 — current-repository truth: cover every performance experiment and shipping-default change after the prior E108 report through the latest entry. Scenario: `git log --all --since=2026-08-28 --oneline`, latest `experiments_results.md` sections, current `AGENTS.md`, and current flag source are cross-read. PASS iff the final report names the latest experiment, champion stack, default/baseline behavior, and every new negative result with prompt/N/axis limitations. Evidence: wave repo audits, intent diff, and cited report sections. RED: the prior report says CPU+KERNELS is fastest while current AGENTS states E114-E119 superseded it. GREEN: revised synthesis reflects the current champion and retains the CPU reference correctly.
- Criterion 2 — decode/tok-s diagnosis: identify the current decode bottleneck on this M3 Max and rank opportunities by measured share, plausible phase speedup, Amdahl end-to-end bound, engineering cost, numerical risk, and falsifiable experiment. Scenario: cross-check source call paths, latest profile logs, and experiment ledger; independently recompute every derived bound. PASS iff no TTFT result is described as decode gain, no phase-local number is presented as end-to-end, and every recommended decode candidate has evidence/gain/risk/test/kill fields. Evidence: claim graph, calculation notes, and report matrix. RED: the supplied scalar/NEON diagnosis has no next-step ranking beyond the neutral kernel. GREEN: report gives a complete host-feasible decode opportunity map.
- Criterion 3 — external research saturation: cover Apple CPU/AMX/Accelerate/Metal/unified-memory methods, quantized matvec/GEMV layouts, MoE decode batching, cache/I/O, MLX/llama.cpp/ggml implementations, KV/attention, speculative methods, and multi-request throughput. Scenario: dedicated workers plus at least two expansion waves; every final external claim has a primary or SHA-pinned source and a counter-search record. PASS iff every axis is represented and unchecked EXPAND leads are zero or explicitly closed. Evidence: wave digests, observation manifest, expansion log, source ledger, and claim graph.
- Criterion 4 — exact-host actionability and comfort: recommendations must fit Mac15,9 M3 Max, 12P+4E CPU, 40 GPU cores, 128 GB unified memory, current storage/power state, and Colibri's actual one-slot/custom-quant architecture. Scenario: current host commands plus code inspection. PASS iff the report gives the fastest shipping configuration, safe daily RAM/process/power/storage guidance, separates single-user tok/s from aggregate throughput, and invents no host-local percentage. Evidence: host snapshot and P0/P1 candidate sections.
- Criterion 5 — benchmark/correctness integrity: proposed tests obey seed, build-seam, prompt-length, both-axis, same-arm determinism, multi-chunk text-diff, and taskcheck rules. Scenario: compare every proposed token-generating experiment against `AGENTS.md`, `.backlog/lab/tokps.sh`, `.backlog/lab/taskcheck.sh`, and current golden harnesses. PASS iff every experiment states cost/purpose/prompt/N/metrics/gates and no long benchmark is run without approval. Evidence: experiment backlog section and final audit.
- Criterion 6 — adjacent-surface safety: no source mutation, model run, long benchmark, commit, push, or unrelated-file modification. Scenario: final `git status --short`, `git diff --stat`, `git rev-parse HEAD`, `pgrep -f '[d]eepseek_v4'`, and agent inventory. PASS iff only task-owned research/report artifacts changed and all spawned resources are terminal. Evidence: final notepad receipt.
- Criterion 7 — delivery quality: the user-selected materials follow the technical-report structure, all quantitative claims have visible lineage, and the rendered surface passes visual QA. Scenario: render every final page/surface and inspect for clipping, broken figures, unreadable tables, citation gaps, and inconsistent summary/body claims. PASS iff every page is clean and the HEAVY reviewer returns unconditional APPROVE. Evidence: render/visual-QA artifacts and reviewer verdict.
- Pure-prose/research target: there is no honest unit-test RED seam. Failing-first evidence is the stale prior report versus the now-superseding E114-E119 repository truth; GREEN is QA-by-read plus source/claim/audit and rendered-surface verification.
- WHEN TO STOP: I'll stop right away when all seven criteria pass, every research lead is closed, the selected deliverables pass visual QA, the HEAVY reviewer approves unconditionally, every child is terminal, and Git/runtime safety is recorded.

## Skills used

- `omo:ultrawork` — binding goal/notepad/evidence/reviewer workflow explicitly requested by `ulw`.
- `omo:ulw-research` — explicit deep-research request requires saturation waves, claim instrumentation, format gate, and cited materials.
- `omo:git-master` — current history and status are required; mode is read-only HISTORY/STATUS and no Git writes are authorized.
- `data-analytics:analyze-data-quality` — benchmark claims must be reconciled across axes, noise, prompt/N, corrections, and changed defaults.
- `data-analytics:build-report` — the result needs a durable technical report rather than chat-only prose.
- `omo:programming` — the launcher surface includes Python; inspection is read-only and no Python code will be written unless its language references are loaded first.

## Now

- Complete. Report and `claude.md` are synchronized with E109-E124 and the post-E124 real-shape standalone FP8 microprobe. Rendered QA is clean, the HEAVY reviewer returned unconditional APPROVE, all task-spawned research agents are terminal, preview resources are closed, and the final Git/process receipt is recorded below.

## Todo

- [x] Record final format/template choice.
- [x] Capture repository and host safety snapshot.
- [x] Launch and integrate research wave 1.
- [x] Execute and integrate expansion wave 2+ until convergence.
- [x] Reconcile E109+ repository evidence and current defaults.
- [x] Build the decode bottleneck/Amdahl opportunity model.
- [x] Verify contested claims with bounded evidence.
- [x] Complete claim graph and epistemic ledgers.
- [x] Author and render the selected technical deliverables.
- [x] Run visual QA and complete claim/metric/source audit.
- [x] Obtain unconditional HEAVY approval.
- [x] Record cleanup/safety/source counts/elapsed time and hand off.

## Findings

- The previous 2026-08-28 report is already RED against the new repository contract: current AGENTS states E114-E119 ship the GPU-prefill champion stack by default, while the prior report names CPU+KERNELS as fastest.
- Current binding rules say `COLI_V4_BASELINE=1` restores historical defaults and `bench/golden.sh` pins it; shipping output is capability-equivalent rather than token-identical to the old CPU arm.
- Current binding rules also state `COLI_V4_PREFILL_PREFETCH=1` deadlocks with the champion stack (E113) and must not be combined.
- Post-E124 standalone microbenchmark evidence now sharpens P0: rows16 interleave plus exact E4M3-to-FP16 reinterpretation measured 1.61-2.28x across all four real attention shapes at N=21, reducing the weighted kernel sum from 4.798 to 2.200 ms/layer. Root independently reproduced 1.28-2.20x across those shapes at N=7. This is not engine tok/s; the roughly 2.05 tok/s/+23% figure is derived and retains NaN, compiler, packing, integration, and model-gate risks.
- The same microbenchmark shows locality alone is modest (rows4 + LUT 1.04-1.21x) and a heavier arithmetic decoder is slower (0.70-0.78x), so the winner is the specific rows16 plus FP16 reinterpretation combination, not generic interleaving or LUT removal.

## Learnings

- A decode-only request still requires total-wall and comfort context, but TTFT-only improvements must remain visibly separate from tok/s.
- The neutral NEON kernel refutes arithmetic-width as the bottleneck for the current scalar FP8 loop; it does not close data-layout, LUT elimination, cache-line locality, fusion, mixed precision, or concurrency approaches.
- A durable N=21 synthetic microkernel plus an independent N=7 rerun can promote an integration candidate, but it cannot satisfy the project's token-generating evidence rule; TTFT/tok/s remain explicitly unavailable until an operator-approved engine A/B.

## Transition — format gate resolved

- 2026-08-29: user selected **Markdown**. Binding delivery: one repository technical report at `.backlog/m3-max-decode-research-2026-08-29.md`, with a rendered Markdown preview used only for visual QA. No PDF, DOCX, Sites, or parallel HTML deliverable.
- Technical structure: summary; E109+ changes; metric definitions; current decode profile; host/hardware constraints; research findings; ranked candidates; falsifiable experiment backlog; negative/dead methods; uncertainty; further questions; inline source references.
- Now: capture current safety/host/repository state, then launch all first-wave research lanes in parallel.

## Current-state snapshot — 2026-08-29 14:02 CEST

- Branch `simd-apple-metal`, HEAD `d5864b00e4a56f9b1f5f9d274b13d387698e3e38`. Only task-owned untracked files are currently visible: `.backlog/m3-max-decode-research-2026-08-29.md` and `.backlog/ulw-decode-m3max-20260829.md`; `git diff --stat` is empty.
- Experiment ledger now ends at **E124**, not E119. New range E109-E124 includes retraction of E101, grouped/batched Metal prefill, prefetch deadlock, attention/rows16/whole-prompt champion composition, shipping defaults, bounded deferred tiles, decode profile, and neutral NEON FP8 matvec.
- Hardware revalidated: Mac15,9 M3 Max, 12P+4E CPU, 40-core GPU, 128 GB unified memory, Metal 4, internal 3456x2234 plus external 3840x1080 display.
- Current power is materially different from the prior report: attached to the 140 W adapter, but `system_profiler` reports **High Power Mode: No** and **Low Power Mode: Yes** on AC. This is an operational candidate that needs authoritative confirmation and no invented speed percentage.
- Internal root now reports 112 GiB available, so the previous 7.5 GB emergency has been resolved. Repository/model tree is about 163 GB.
- Current swap is 5.54/7.17 GB used after 18 days uptime; compressor occupies about 720 MiB (`46111 * 16 KiB`). Historical allocation is not current pressure, but a reboot/quiet-state measurement remains relevant.
- No `deepseek_v4` process is running. No benchmark or model run has started.

## Final receipt — 2026-08-29 14:44 CEST

- Deliverables: `.backlog/m3-max-decode-research-2026-08-29.md` (436 lines, 4,994 words) and `claude.md` (192 lines, 1,530 words).
- Evidence coverage: 15-agent first wave, six-agent expansion wave, four-agent final expansion wave, plus one HEAVY gate reviewer; every task-spawned agent is terminal.
- External citation inventory: 20 unique URLs and 27 rendered external-link instances, led by Apple primary documentation and SHA-pinned MLX/MLX-LM/llama.cpp source.
- Visual QA: full rendered report inspected; final DOM audit found zero overflow elements, zero broken images, zero bad fragment links, and no browser warnings/errors.
- HEAVY gate: first pass correctly rejected stale microbenchmark lineage; corrected pass returned unconditional `APPROVE` for all seven criteria. Durable gate artifact: `.omo/evidence/m3-max-decode-research-gate-review.md`.
- Safety: HEAD remains `d5864b00e4a56f9b1f5f9d274b13d387698e3e38`; `git diff --check` is clean; no `deepseek_v4` or report HTTP server process is running; no model run, long benchmark, power change, source edit, commit, or push occurred in this research task.
- Repository status: intentional tracked change is `claude.md`; intentional report/notepad artifacts are untracked. Concurrent task artifacts `.backlog/.ulw_note`, `.backlog/lab/kbench/`, and `.backlog/ulw-decode-fp8-20260829-142423.md` were preserved and incorporated without attribution as engine evidence.
- Cleanup: the abandoned 204 MB research clone was moved recoverably to `/Users/cptn/.Trash/colibri-librarian-llama-20260829`; the in-app browser tab and local preview server were closed.
- Elapsed research time: about 45 minutes from 13:58:43 to 14:44:07 CEST, excluding earlier work already present in the repository.
