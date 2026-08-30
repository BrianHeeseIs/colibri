HANDOFF CONTEXT
===============

USER REQUESTS (AS-IS)
---------------------
First request of the session (verbatim, from session_read):
- "ulw investigate and develop \"simd\" to boost apple metal performance ttft and tok/s.

  1. create a new branch called simd-apple-metal and switch to it before starting work.

  Pick up where you left off in the other session. Use serena memories and agents.md for all relevant knowledge of the project. Read experiment-results.md for insight into what experiments have been done and which results we have already recorded. Read .backlog directory for future experiments to pull from when you're done.

  Right now focus on:

   Branch perf-upstream-adopt at f4110ca, pushed, tree clean.
  The single most valuable line now recorded in all three places: run with COLI_V4_KERNELS=all - and the biggest open lead, the simd variant at 5.5-6.3x CPU at S=1, is captured in dead_levers under \"still open\" so the next session starts there rather than re-deriving it."

Later top-level requests (verbatim):
- "ulw 1. document, commit and push all work done on this branch. 2. create a new branch called ft-mxfp4-expkern-applmtl and switch to it 3. start work on \"P0 - mxfp4 expert kernel.\""
- "the quiet rule is abolished by the way, push the machine to the max whenever you can. record this in agents/md and serena memory if it's not already in there"
- "sorry, i thought the quiet rule was for the noise level of my machine. reinstate the no background agents rule as it was"
- "merge the findings of this branch back into the previous branch and close this one"
- "ulw continue until all merge work is verified"
- "if you see any remaining changes in the repo directory, that's only the research currently running for the next phase. keep those changes locally"
- "ulw read all new / updated .md files in the .backlog directory and present the opportunities for performance optimization they contain in order of most gain to be had"
- "ulw finish the merge and branch deletion and then move on to: \"The next exact action is one current-shipping decode critical-path trace\" (roadmap:19-22) - ~5 minutes, no direct gain, very high decision value."
- "ulw continue working and run the golden benchmark in the named tmux session colibri-lab"
- "why do you keep halting suddenly?"
- "the current session to handoff.md"
- "after which, make sure all critical learnings this session have been documented in .md files + written to serena memory"

GOAL
----
Execute Wave 2 of the decode critical-path trace: wire c/decode_trace.h into c/deepseek_v4.c (task T4), then add the timers (T5-T8), build, and run the trace to decide the next optimisation target.

WORK COMPLETED
--------------
- Ran B4 and recorded it as E128: measured the shipping stack directly against COLI_V4_BASELINE=1 at p256, N=3, both arms deterministic. tok/s +54.83 percent (1.3948 to 2.1596), TTFT -62.4 percent, net wall -57.1 percent. Established that the previously quoted +28.1 percent is the decode work only, and that BASELINE=1 also disables KERNELS=all, which is why the totals differ. Propagated the decomposition into AGENTS.md, CHANGELOG.md and docs/deepseek-v4.md.
- Found and fixed a stale claim in CHANGELOG.md that still said "Decode is unimproved" and named the weight-layout change as the outstanding lever; that change is the E125 +10.18 percent two entries above it.
- Worked P0 (mxfp4 expert kernel) on branch ft-mxfp4-expkern-applmtl and CLOSED it as E129 without writing a kernel, because two measurements taken first showed it cannot clear the noise floor. Three findings: the recorded "~6 percent of decode expert calls reach NEON" was wrong and is actually 22.05 percent; the accumulator-recurrence hypothesis is refuted (chain sweep 4/8/12/16 gives 1.00/1.09/0.95/0.99, non-monotone); a ceiling arm abandoning Metal bit-parity reaches only 1.154x/1.201x, worth +4.88 percent decode against a 5-13 percent noise floor.
- Landed a permanent kernel-split counter in c/deepseek_v4.c that retires E123's standing request to instrument which kernel each expert call takes. Both golden gates pass with it compiled in.
- Merged ft-mxfp4-expkern-applmtl into ft-decode-apple-metal (merge commit 46a38ad, --no-ff, matching the repo's own "merge <branch>: <desc>" precedent), verified it fully, and deleted the branch locally and on fork. 21c0cfa stays reachable as a merge parent, which matters because claude.md cites that SHA.
- Verified the merge with evidence: forced rebuild exit 0 with the Metal seam linked, golden.sh = 5d04890413ff539e802985ce8c727814, golden_default.sh = cc09015d089d9a25d10d75753f9e849a.
- Read the three new .backlog research documents and produced a ranked list of remaining performance opportunities.
- Started the decode critical-path trace. Wave 1 is complete and verified by me directly: c/decode_trace.h plus c/tests/test_decode_trace.c (26 stages, atomic adder, RED transcript captured, now GREEN - "decode_trace: all checks passed", exit 0), and .backlog/lab/decodetrace.sh.

CURRENT STATE
-------------
- On branch ft-decode-apple-metal at e87d18f, pushed to fork, in sync.
- Both golden gates pass. Build is clean with METAL=1 and the Metal seam linked.
- make -C c check has ONE pre-existing failure at tests/test_fp8_passthrough (undefined _coli_fp8_minprod_enabled). I reproduced it at HEAD with my own change stashed. It is not mine and was not fixed.
- Working tree is deliberately dirty. c/Makefile plus the three new untracked files under c/ and .backlog/lab/ are MY uncommitted Wave 1 work. claude.md, experiments_results.md, .backlog/m3-max-decode-research-2026-08-29.md and the three .backlog/m3-max-* / ulw-expert-wait-* files belong to a CONCURRENT research session and must be left alone.
- c/deepseek_v4.c has NOT been touched by the trace work (grep -c 'decode_trace' returns 0).

PENDING TASKS
-------------
- T4: wire decode_trace.h into c/deepseek_v4.c - include near line 78, globals near 9611-9618, extend coli_v4_profile_reset_decode at 9658-9665 to read COLI_V4_DECODE_TRACE, add a report function printing all tables at zero, call it after coli_v4_profile_report at 11424-11425. This task was drafted but NEVER RAN.
- T5: table=wait, seven timers plus six counters into dual_expert_load_start (4482-4512) and dual_expert_load_finish (4515-4528).
- T6: table=store_nested plus table=io in coli_expert_lookup (8310-8557); stage 18 must reuse the existing disk_t0/disk_t1 pair at 8491-8495, adding zero new clock reads.
- T7: table=omp master-side wall timing at 1492, 3620, 9857/9910 plus a fork/join calibration probe. No region body is modified in this phase.
- T8: table=control - the three fp8_view calls near 5088 and the decode allocation sites from T2.
- T9: build with METAL=1, unit tests, and the static prefix gate.
- T10-T12 (serial, engine, exclusive): goldens plus prefix contract, a zero-cost flatness check at N=2, then the trace run itself - p256, 40 tokens, one OFF control and two ON arms.
- T13-T15: analysis, decision gate, ledger as E130, commits.
- Todo list state: the twelve-item mxfp4 list is fully resolved - seven completed, four cancelled by the E129 kill gate, one completed as the ledger entry. No live todo list exists for the trace work yet.

KEY FILES
---------
- c/decode_trace.h - new, Wave 1, 26 stages with an atomic adder; the trace core. Uncommitted.
- c/tests/test_decode_trace.c - new, its unit test; currently passing. Uncommitted.
- c/Makefile - modified, holds the new test rule after tests/test_head_ilp. Uncommitted.
- .backlog/lab/decodetrace.sh - new runner: seed hash guard, pgrep guard, three arms, S6 prefix verdict, no cleanup trap. Uncommitted.
- c/deepseek_v4.c - the target of T4-T8; currently untouched by this work.
- .backlog/m3-max-whole-engine-performance-roadmap-2026-08-30.md - the roadmap driving this work. Concurrent session's file, do not commit.
- .backlog/m3-max-expert-wait-research-2026-08-30.md - refutes the old expert_wait story. Concurrent session's file, do not commit.
- .backlog/ulw-mxfp4-expkern-20260830-133441.md - my notepad, path stored in .backlog/.ulw_note4.
- .backlog/benchmark-backlog.md - P0 marked CLOSED; B1/B2/B3 still parked.
- experiments_results.md - E128 and E129 committed; currently being edited by the concurrent session.

IMPORTANT DECISIONS
-------------------
- The trace uses a SEPARATE accumulator array, not new members of the COLI_V4_PROFILE enum, so COLI_V4_PROFILE_COUNT stays 16 and the existing accounted_pct arithmetic is provably unchanged and stays diffable against .backlog/lab/profile_post_e125.txt.
- A separate env var COLI_V4_DECODE_TRACE gates the new counters, so the default profile output keeps its historical shape and an execution-proof scenario can compare ON and OFF on the same binary.
- All new counter output must print under the existing "v4_profile " prefix.
- The trace core lives in a standalone header so its arithmetic is unit-testable without linking the 15k-line translation unit; precedent is c/Makefile's tests/test_head_ilp rule.
- OpenMP is instrumented master-side only in phase one, plus a fork/join calibration probe, because per-thread accounting would require restructuring parallel-for regions in numerics-adjacent hot loops - the highest bit-exactness risk in the work and not needed to answer "is OpenMP overhead >= 2 percent".
- Wave 2 is deliberately SERIAL. T4-T8 all edit one 15k-line file; parallel agents there trade minutes for merge corruption.
- P0 was closed without writing a kernel because the gate was defined before implementing and it fired. The two RED ILP tests were deleted afterwards since matmul_mxfp4_ilp will never exist and they would break make check.

EXPLICIT CONSTRAINTS
--------------------
Verbatim from the user this session:
- "the quiet rule is abolished by the way, push the machine to the max whenever you can. record this in agents/md and serena memory if it's not already in there"
- "sorry, i thought the quiet rule was for the noise level of my machine. reinstate the no background agents rule as it was"
- "if you see any remaining changes in the repo directory, that's only the research currently running for the next phase. keep those changes locally"
- "continue working and run the golden benchmark in the named tmux session colibri-lab"

Verbatim from AGENTS.md:
- "NEVER USE /tmp FOR ANYTHING YOU NEED"
- "Golden output md5 5d04890413ff539e802985ce8c727814 is SACRED. Never edit the expected value to make something pass; fix the code."
- "NEVER conclude an approach is wrong from a CHANGED MD5. READ THE TEXT FIRST."
- "MANDATORY: no background agents in flight while a timing run executes"
- "MANDATORY: do not run a long benchmark without asking. Size it to the question first."
- "MANDATORY: if an experiment generates tokens, it MUST report tok/s - not TTFT alone"

CONTEXT FOR CONTINUATION
------------------------
- Run long engine work through tmux session colibri-lab, pane colibri-lab:0.0, with a completion-marker file, then poll the marker in a LATER turn. Direct bash invocation of golden.sh or a trace run times out and silently leaves the work undone while appearing to have run. This cost several turns.
- golden.sh and golden_default.sh return rc=2 with "another deepseek_v4 process is already running" when an engine is live. That is a REFUSAL, not a hash failure. Read the log body before concluding anything. A concurrent research session shares this host, so pgrep can clear a moment before their process actually exits.
- LATENT BUG, unfixed: .backlog/lab/tokps.sh and taskcheck.sh strip only lines matching ^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal) followed by a space. The existing COLI_V4_PREFILL_TRACE prints under "v4_prefill_trace " which is NOT in that list, so enabling it during a tokps.sh run would corrupt generated_text and manufacture a false non-bit-exact verdict.
- COLI_V4_PREFILL_TRACE at c/deepseek_v4.c:25-56 and 105-288 already decomposes coli_expert_lookup for PREFILL. It is the design precedent for the decode trace: copy its atomic adder and its non-additivity note, but not its compile-time gating and not its output prefix.
- A T2 discovery contradicts the roadmap: decode allocation is NOT near-zero. There are heap allocations per token per layer at block scratch 5857-5863, HC norm 3833 (twice per layer), attention/RoPE/QDQ/sparse 7069-7181, and MoE scratch 4964-4969, plus per-token embedding and head scores. At 43 layers that is hundreds of malloc/free pairs per token, so the prebind-and-scratch lever is more promising than its "probably low single digit" rating.
- fp8_view at c/deepseek_v4.c:5088 binds to the definition at 3796; both are inside COLI_V4_UNIT_BLOCK_HYBRID (3715-6243).
- The kill criterion for the trace is set in advance: if total main-thread wait is under 3 percent of decode wall, expert_wait is declared dead and recorded as closed, the same discipline that closed E129.
- Unresolved correctness question spun out of E129: mx4_scale at c/quant.h:1437 gives s=255 to +inf, while coli_e8m0_decode at c/deepseek_v4.c:13503 returns NaN for 0xff. Same byte, same engine, different value. No gate exercises it.
- Do not trust an agent's self-report. Verify from disk. This session produced one fabricated tool result, and one agent's claimed work had not happened at all.
