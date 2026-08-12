# DeepSeek-V4 learning-cache per-turn flush — work plan (v2)

> **SCOPE UNCHANGED, VALUE RAISED (added 2026-08-12 after the tuning.md audit).**
>
> This plan was approved as a *correctness* fix: `.coli_usage` is written from exactly one site
> (`deepseek_v4.c:5923`, inside `destroy_hot`), so SIGINT/crash/OOM discard the session and the
> `history_total >= 5000` seed threshold can never be reached across restarts.
>
> It is now also the **unlock for a dormant performance feature**. `COLI_V4_PREWARM=1` prewarms
> the expert cache from usage history at startup — exactly the cold-start penalty measured in
> RESULTS.md §4b: **+33.2% on prompt 1, +17.6% across a 4-prompt pass** (`--ram 96`, cold 0.3375
> → warm 0.3968 tok/s, 4/4 paired wins, every row gate-green).
>
> `COLI_V4_PREWARM` cannot fire today:
>
> ```
> prewarm requires   policy->history_seeded        (deepseek_v4.c:6068)
> history_seeded  =  history_total >= 5000         (deepseek_v4.c:6054)
> .coli_usage written ONLY in destroy_hot          (deepseek_v4.c:5923)
> .coli_usage on models/deepseek-v4-flash          ABSENT (confirmed 2026-08-12)
> ```
>
> Every harness run `pkill`s the engine, so history never persists, `history_total` stays 0, and
> the prewarm branch is unreachable. Landing this plan makes the history durable, which makes
> `COLI_V4_PREWARM` testable — the only in-tree mechanism that addresses the measured
> cold-start cost.
>
> **The approved scope does not change**: 1 exported helper, 1 static epilogue, 1 test-only
> wrapper, 1 production call site, 1 fault-injection flag, 1 test. Only the priority argument
> changes — this is no longer just a robustness fix.
>
> Follow-up once landed: run a session long enough to cross 5000, confirm `.coli_usage` exists
> and `history_seeded` flips true, then A/B `COLI_V4_PREWARM=0` vs `=1` on a cold start using
> the `coldwarm.sh` protocol and pressure gate.

Status: REVISED after review round 1. All decisions RESOLVED. No open questions.
Round 1: Oracle = PASS WITH CHANGES (3 blockers); Momus = REJECT (1 blocker).
Every blocker is addressed below and marked [R1-FIX].

## IMPLEMENTED 2026-08-12 — unit + T2 + T4-mechanism GREEN; T4 BENEFIT gate outstanding

134 insertions across 3 files + 1 new test. Scope matched the plan exactly: 1 exported
helper, 1 static epilogue, 1 test-only wrapper, 1 production call site, 1 fault-injection
flag, 1 test.

| task | what landed | gate |
|---|---|---|
| T1 | `coli_v4_expert_store_flush_usage()` in the hot rows16 unit (`deepseek_v4.c:5632`), declared `deepseek_v4_internal.h:715` | build exit 0 |
| T2 | static `v4_flush_usage_epilogue()` + **one** call site in `v4_serve_one`, placed BEFORE the `if (result)` branch so a single site covers the error early-return and the success path | build exit 0 |
| T3 | `coli_v4_test_fail_generate_after_prefill` (RUNTIME unit), checked after prefill AND after the session takes ownership of `state`/`next` so the injected failure cannot leak the prefill buffers; `coli_v4_test_flush_usage_epilogue()` forwarder in the SAME TU as the epilogue; `tests/test_v4_generate_error_flush.c` | **RED/GREEN verified** |
| T5 | regression | `make test-c` exit 0, `make deepseek-v4` exit 0 |

**RED/GREEN evidence** (the test is sensitive, not vacuous):
```
GREEN (default)                error-path flush: ok       exit 0   .coli_usage present
RED   (COLI_V4_SAVE_USAGE=0)   FAILED -- absent           exit 1   .coli_usage absent
```

### Implementation notes not anticipated by the plan

- **TU placement was load-bearing.** `hot_find`/`hot_usage_save` live in
  `COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16`; `v4_serve_one` and the prefill site live in
  `COLI_V4_UNIT_GENERATE_STATS`; the existing hook block lives in `COLI_V4_UNIT_RUNTIME`.
  The flag is therefore defined in RUNTIME and read in GENERATE_STATS — fine, it is a
  non-static global resolved at link time. The epilogue and its test forwarder had to be
  co-located in GENERATE_STATS (D8's whole point).
- **The test target needs flags the ownership target does not.** `V4_OWN_CFLAGS` suffices for
  4 units but the full set needs `-include pthread.h` and `-D_FILE_OFFSET_BITS=64`, plus
  `-DCOLI_V4_SKIP_GENERATE_MAIN` — otherwise the engine's own `main()` collides with the
  test's. New `V4_FLUSH_*` vars in `c/Makefile`; objects land in `build/flush/`.
- **The rule must depend on `Makefile`.** Without it, changing CFLAGS leaves stale objects and
  the link fails against a `main()` that the current flags would have removed. Cost me one
  false diagnosis before the preprocessor settled it.
- **The test cleans up after itself**, unlinking `.coli_usage` AFTER the destroys (teardown
  rewrites it). `.coli_usage` is not gitignored, so otherwise every `make test-c` would leave
  an untracked artifact in `c/deepseek_v4_tiny/`.

### Honest limitation of the T3 test

The tiny fixture is 3 layers / 4 routed experts / top-2 and reports
`pin_slots_per_layer=0`; both the epilogue save AND the later `destroy_hot` save print
`selections=0 distinct=0`. Because the *teardown* save also reports 0, the flag is firing in
the right place — the fixture simply never accumulates usage.

So the test proves the epilogue **executes on the error path and writes `.coli_usage` before
any destroy** (the ordering point Oracle raised in round 3), but it does **not** prove the
content is non-vacuous. Change 1's stated rationale is therefore only partly satisfied on
this fixture.

### T2 serve-level QA — GREEN on the real 167 GB model (2026-08-12 21:05)

`validation/dsv4/t2_serve_flush_qa.sh`, `--ram 48`, one 4-token request:

```
reply            prompt_tokens=5 completion_tokens=4
.coli_usage      9505 bytes, written 21:05:12
v4_autopin saved=.../.coli_usage selections=2064 distinct=1143
```

**This closes the non-vacuity gap the tiny fixture left open.** 2064 selections across 1143
distinct experts, versus `selections=0` on the fixture. The magnitude is coherent: §5 predicts
~258 selections/token (43 layers x top-6), and 2064/258 ≈ 8 against 9 total tokens.

Evidence that the **per-turn epilogue** wrote it, not teardown:
1. Exactly **one** `saved=` line in the serve log, timestamped with the request.
2. The existence check found the file **before** the script's `cleanup` ran.
3. Teardown was `pkill` (SIGTERM), which bypasses the graceful destroy path — that bypass is
   the very defect this design exists to fix — so `destroy_hot()` could not have written it.

Correction to that run: the script's own liveness counter printed
`engine still running (0 proc)`, which was **wrong**. It used
`pgrep -f 'libexec/colibri/deepseek_v4'`, but `c/coli`'s `engine_for()` (`coli:249`) prefers
`c/deepseek_v4` when present, so the pattern matched nothing. The engine was alive; the counter
was broken. Pattern fixed to `[d]eepseek_v4`. The GREEN verdict rests on points 1–3 above, not
on that counter.

Note for T4: the model now carries a real history of **2064 selections**. The
`history_seeded` threshold is 5000, so roughly one more comparable request crosses it and
`COLI_V4_PREWARM` becomes testable for the first time.

### T4 primary evidence — GREEN: the loop closes across a NON-GRACEFUL kill (21:13)

`validation/dsv4/t4_loop_closes.sh`, `--ram 48`, two 16-token requests, teardown by SIGTERM:

| stage | history | selections |
|---|---|---|
| start (left by T2) | 9505 B | 2064 |
| during run 1 | — | 8256 → **14190** |
| after **SIGTERM** kill | 29127 B | — |
| run 2 startup | — | **`v4_autopin history=... selections=14190`** |

**+12126 selections persisted across a kill that never runs `destroy_hot()`.** That is the
defect this design was written to fix, demonstrated end to end: before the change a SIGTERM
discarded everything since the last graceful destroy — which in practice meant everything,
because every harness `pkill`s the engine.

Magnitude checks out: 12126 / 258 selections-per-token ≈ 47 tokens, consistent with 2 prompts
plus 2×16 completions.

**`history_seeded` is now TRUE for the first time** (14190 ≫ 5000), which makes
`COLI_V4_PREWARM` testable — the dormant feature this plan was identified as unlocking.

Caveat on the script's own framing: it printed "run 1 had 1 such line, run 2 has 1", expecting
run 1 to be virgin. It was not — T2 had already left 2064 selections on disk. The intended
"none → one" contrast was therefore unavailable. The actual proof is the **count growth across
the kill** (2064 → 14190, reloaded intact), which is the stronger claim anyway.

### NOT done — the T4 BENEFIT gate

What is proven: the mechanism. History accumulates per turn, survives a non-graceful kill, and
is reloaded and seeded on the next start.

What is NOT proven: that persisting it **improves** anything. The full protocol — 2×10 prompts
comparing hit rate / `bytes_read`, plus the `COLI_V4_SAVE_USAGE=0` vs `=1` confound control
under an identically warmed page cache — is multiple hours at ~0.25 tok/s and was not run.

**The justify/kill decision in §7 therefore remains open.** §7's kill condition is explicit and
still live: if hit rate does not improve despite a valid history file, the limiting factor is
pin policy (`COLI_V4_PREWARM`, `pin_slots_per_layer`), not persistence.

Scope: make V4's `.coli_usage` learning cache durable per turn, matching GLM.
Non-goal: Metal, weight conversion, speculation, engine unification.

## 1. Problem (measured)

A 15-minute `coli serve` session with DeepSeek-V4 never created `.coli_usage`. Three
servings of an identical prompt: 0.1616 -> 0.2308 -> 0.1529 tok/s — no repetition benefit.

Root cause by call-site count:

| engine | save function | call sites |
|---|---|---|
| GLM (`colibri.c`) | `usage_save(m)` | **5** — 6742, 7059, 7421, 7589, 7594 |
| DeepSeek-V4 (`deepseek_v4.c`) | `hot_usage_save()` | **1** — 5923, inside `destroy_hot` |

[R1-FIX, Oracle #1] Precise wording: persistence happens **only on graceful engine destroy**
(`coli_v4_engine_destroy` at 6482-6485 reaching `destroy_hot` at 5915-5941). SIGINT, crash,
OOM and power loss all bypass that path entirely. It is not "clean teardown" in general —
it is specifically the graceful destroy path.

GLM's own comments state the intent:
- `colibri.c:7059` — "la cache che impara non deve aspettare l'uscita" ("must not wait for exit")
- `colibri.c:7589` — "storia aggiornata a ogni turno" ("history updated every turn")

V4 is also the only engine with a private cache implementation; `inkling.c`, `kimi_k3.c` and
`olmoe.c` all use shared `route_trace.h`.

## 2. Reachability constraint

`hot_usage_save` is `static` inside `#ifdef COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16`
(`deepseek_v4.c:5039-6073`). V4 compiles as an amalgamation of translation units
(`Makefile.deepseek-v4.units`), so the serve loop cannot see it, and `V4HotPolicy` is private
to that unit.

## 3. DECISIONS (all resolved)

- **D1** [R1-FIX, Oracle #2 BLOCKER] Export a **V4-private helper**, not a generic vtable entry:
  `void coli_v4_expert_store_flush_usage(ColiExpertStore *store);`
  defined in the hot-store unit, declared in the V4 internal header.
  *Rationale (Oracle):* widening the shared `expert_store.h` ops contract for one
  engine-private need adds surface to every store implementation for no benefit.
  The vtable approach from v1 is WITHDRAWN.
- **D2** Helper body is lock-only + save-only: resolve policy, lock `state->mutex`, call the
  existing `hot_usage_save(policy, state)`, unlock. **No repin, no prewarm, no load, no
  prefetch** — it must not alter placement behaviour mid-run.
- **D3** [R1-FIX, Oracle #3 BLOCKER][R3-FIX] Factor the epilogue into a tiny static helper
  `static void v4_flush_usage_epilogue(ColiV4Engine *engine)` which calls
  `coli_v4_expert_store_flush_usage(engine->experts)` when non-NULL.
  Call it from **exactly ONE production site**: `v4_serve_one`, after
  `coli_v4_session_generate()` returns — on **both success and generate-error paths**,
  not only after the `DONE` line.
  *Rationale for the helper (Oracle round 3):* the error-path test must execute the SAME
  code the serve path executes. Without a shared helper the test can only call
  `coli_v4_session_generate()` directly, which never traverses the epilogue and therefore
  proves nothing.
- **D8** [R4-FIX, Oracle + Momus consensus] `v4_flush_usage_epilogue` stays `static` (production
  encapsulation preserved). Add a **test-only forwarding wrapper in the SAME translation unit**:
  ```c
  #ifdef COLI_V4_TEST_HOOKS
  void coli_v4_test_flush_usage_epilogue(ColiV4Engine *engine) {
      v4_flush_usage_epilogue(engine);   /* forwards to the production body */
  }
  #endif
  ```
  declared in `deepseek_v4_internal.h` alongside the existing hook declarations (~line 716).
  *Why:* round 4 introduced a genuine compile error — a separate test file linked against the
  V4 unit objects cannot reference a `static` symbol. A same-unit wrapper fixes linkage while
  keeping the production helper private.
  *Explicitly forbidden (Oracle):* the test must NOT call `coli_v4_expert_store_flush_usage()`
  directly — that bypasses the epilogue body and re-opens the round-3 path-equivalence gap.
  *Rationale (Oracle):* a success-only flush loses counts accumulated during prefill when a
  request fails mid-generation.
  The v1 proposal to also flush in `v4_generate_cleanup` is **WITHDRAWN as redundant** —
  that path already calls `coli_v4_engine_destroy` -> `destroy_hot` -> `hot_usage_save`.
- **D4** Respect the existing `COLI_V4_SAVE_USAGE` gate (`deepseek_v4.c:5565`). No new env var.
- **D5** Failure is non-fatal and must never abort a generation (`hot_usage_save` is already void).
- **D6** Do NOT migrate V4 to `route_trace.h` in this change. Recorded as follow-up (§7).
- **D7** [R1-FIX] Lock order: resolve policy first, then take `state->mutex`. Flush is called
  outside any locked region (after generate returns), so no deadlock with
  `lookup_hot`/`release`/`prefetch`/`stats`, which all take the same mutex.

## 4. Explicit scope limits (honesty)

[R1-FIX, Oracle #7] This change restores durability of **placement history only**. It does NOT:
- protect against loss of an in-flight turn on abnormal termination before the first flush;
- add signal handling for Ctrl-C mid-turn (that is separate work, deliberately out of scope);
- change any router, math, or token-selection behaviour.

## 5. Corrected assumptions

[R1-FIX, Oracle #5] My v1 draft worried the `history_total >= 5000` seed threshold
(`deepseek_v4.c:6054`) might be unreachable. **It is not.** V4 routes top-6 across 43 layers
≈ **258 selections/token**, so 5000 ≈ **20 generated tokens** — reachable in a single short
session. `COLI_V4_PREWARM=0` (`6067`) does not block learning either; it only blunts the
immediate post-restart speedup, because pins are ranked and eviction-protected rather than
eagerly loaded.

## 6. Tasks — each with executable QA

[R1-FIX, Momus] All QA uses the in-repo launcher `c/coli`, NOT `bin/coli`.
`bin/` is a local `make install` artifact and is not present in a fresh checkout.

### T1. Declare + implement the exported helper
- `c/deepseek_v4.c`: add `void coli_v4_expert_store_flush_usage(ColiExpertStore *store)` in the
  hot-store unit; declare it in the V4 internal header.
- **QA**: `make -C c deepseek-v4 ARCH=native` exits 0; `make -C c test-c` exits 0.

### T2. Call it from the `v4_serve_one` common epilogue
- Insert after `coli_v4_session_generate()` returns, covering success AND error paths.
- **QA (RED->GREEN, the decisive observable)**:
  ```
  rm -f models/deepseek-v4-flash/.coli_usage
  COLI_MODEL=models/deepseek-v4-flash python3 c/coli serve --ram 48 --port 8090 &
  curl -s -X POST localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-colibri","messages":[{"role":"user","content":"hi"}],
         "max_tokens":4,"temperature":0}' >/dev/null
  ls -l models/deepseek-v4-flash/.coli_usage
  ```
  RED (today): file ABSENT after the request.
  GREEN (after fix): file EXISTS, server still running.

### T3. Error-path coverage [R2-FIX + R3-FIX]

Round-2 defect (Oracle): `oversized max_tokens` fails at the `CONTEXT_EXCEEDED` precheck
BEFORE `coli_v4_session_generate()`; a client-side kill maps to callback cancellation. Neither
reaches the epilogue.

Round-3 defect (Oracle AND Momus, independently): calling `coli_v4_session_generate()`
directly does not traverse the `v4_serve_one` epilogue, so the test would not exercise the
flush site at all. Oracle additionally flagged that asserting AFTER destroy yields a FALSE
POSITIVE, because `destroy_hot()` writes `.coli_usage` regardless.

Both are fixed by D3's shared helper plus strict assertion ordering.

- Change 1: add `int coli_v4_test_fail_generate_after_prefill = 0;` to the existing hook block
  (`deepseek_v4.c:6289`, declared alongside the others in `deepseek_v4_internal.h:716`),
  checked immediately after prefill `target_batch(...)` returns 0 — i.e. after expert lookups
  have already incremented `policy->usage` in `lookup_hot` (`deepseek_v4.c:5839`), so the
  history written is non-vacuous.
- Change 2: `tests/test_v4_generate_error_flush.c`, built with `-DCOLI_V4_TEST_HOOKS`
  (mirror the FLAGS of `V4_OWN_CFLAGS` at `Makefile:880`, but NOT the ownership target's
  minimal link set — this test needs the full generate/flush units).

- **QA**:
  ```
  make -C c tests/test_v4_generate_error_flush && ./c/tests/test_v4_generate_error_flush
  ```
  Test body, in this exact order:
  1. open engine on the tiny fixture `c/deepseek_v4_tiny`, FRESH session (no prefix reuse,
     so prefill definitely performs expert lookups);
  2. `unlink()` any existing `.coli_usage`; assert it is absent;
  3. set `coli_v4_test_fail_generate_after_prefill = 1`;
  4. run generation; assert it returned **non-zero**;
  5. call `coli_v4_test_flush_usage_epilogue(engine)` — the D8 test-only wrapper, which
     forwards to the SAME static `v4_flush_usage_epilogue` body that `v4_serve_one` calls.
     (Calling `coli_v4_expert_store_flush_usage()` directly here is forbidden — see D8.)
  6. **assert `.coli_usage` EXISTS — BEFORE any session or engine destroy**;
  7. only then destroy.
  Expected: exit 0, `error-path flush: ok`.

  Step 6's ordering is the whole point: asserting after destroy would pass even with the
  feature absent, because `destroy_hot()` saves on teardown.

### T4. Prove the learning loop closes across restarts
[R1-FIX, Oracle #6 BLOCKER] Primary evidence is **hit rate / bytes_read / autopin line**,
NOT tok/s — OS page-cache warming can inflate a second run's throughput regardless of
`.coli_usage`.
- **QA**:
  1. `rm -f .coli_usage`; run 10 prompts; record `hits`/`misses`/`bytes_read` from stats.
  2. Stop server gracefully. Assert `.coli_usage` exists.
  3. Restart, same `--ram 48`, replay the SAME 10 prompts.
  - **PRIMARY**: run 2 prints `v4_autopin history=... selections=N` at startup
    (`deepseek_v4.c:5555-5557`); run 1 does not. Expert hit rate run2 > run1; bytes_read run2 < run1.
  - **SECONDARY**: mean tok/s run2 > run1.
  - **CONFOUND CONTROL**: additionally run `COLI_V4_SAVE_USAGE=0` vs `=1` back-to-back under
    identically warmed page cache, so any delta is attributable to the history file rather
    than to the model file being cached.

### T5. Regression: numerics unchanged
- **QA**: `make -C c test-c` exits 0; `make -C c deepseek-v4-tiny-check PYTHON=<venv>` exits 0.
  Plus: same prompt, `--temp 0`, `COLI_V4_SAVE_USAGE=0` vs `=1` must yield **byte-identical**
  token streams.

## 7. Justify / kill
- **Justify**: T4 shows the `v4_autopin history=` line on restart AND hit-rate improvement.
- **Kill/defer**: if hit rate does not improve despite a valid history file, the limiting factor
  is pin policy (`COLI_V4_PREWARM`, `pin_slots_per_layer`), not persistence — redirect effort there.

## 8. Follow-up (recorded, not dropped)
- Unify V4 onto shared `route_trace.h` so fixes reach all engines (D6).
- Optional signal handling for mid-turn Ctrl-C durability (§4).
- Revisit whether `COLI_V4_PREWARM` should default on once persistence works.
