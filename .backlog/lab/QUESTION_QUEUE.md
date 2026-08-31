# Question queue — ask via question tool when the operator says "good morning"/"good afternoon"
Opened: 2026-08-30T20:50:24Z

## Queued 2026-08-30 (E132/E133 planning)

**Q1 — scope.** Is `ft-decode-expert-wait-applmtl` decode-only until further notice, or should the
prefill/TTFT backlog be scheduled? B2 (p1024 whole-prompt scaling) was dropped as out of scope and
is the most expensive prompt class in the repo.

**Q2 — E130 measurement gap.** `omp_head_wall` recorded 0 calls because the instrumented head site
is the non-resident bf16 path while the shipping build takes resident `head_ilp`. Wiring the
resident path needs a rebuild plus BOTH goldens (~15 min) and has NO lever attached (OMP master-side
is 2.90% against disk's 16.7%). Leave parked, or close the gap for completeness?

**Q3 — memory ceiling.** Is `--memory-gb 96` a deliberate operational ceiling? The planner reports
`projected=95.95GiB` of `available=96.00GiB` on a 128 GiB host, so the budget is 99.95% consumed.
Whether 104-108 is acceptable decides if an upward cache arm is even runnable. (Currently moot:
ram_sweep shows halving the cache cost only 0.22%, so the upward arm is dropped on that evidence.)

**Q4 — PREWARM observability.** `COLI_V4_PREWARM` emits no stderr confirmation on success, so a
silent no-op is indistinguishable from a null result. May I add one under the existing
`v4_hot_policy ` prefix? That is a code change, so it needs a rebuild + goldens - hence queued.

**Q5 — memory correction.** If the probe shows `HOT_PACK_UNLOCKED` changes rows16 coverage (via the
`slot->references != 1` guard that the locked path lacks), should the "bit-exact both ways" claim in
serena `deepseek_v4/dead_levers` be corrected in the same commit?

**Q6 — latent md5 bug.** `COLI_V4_PREFILL_TRACE` prints under `v4_prefill_trace `, which is NOT in
the 7-prefix strip list used by tokps.sh/taskcheck.sh, so enabling it during a tokps run silently
corrupts generated_text and manufactures a false "not bit-exact" verdict. AGENTS.md records it as a
known latent bug. Fix now, or park?
