# todo.md — colibri performance work order

Durable task list. Survives reboots and context loss. Update status inline as work lands.
Branch: `rows16-metal-bitexact` (HOLD — do not push until a lever lands).

## Acceptance bar for non-bit-exact gains (DECIDED)

**Task-level check** is the gate. A non-bit-exact change is acceptable only if it produces
**identical correctness** on prompts with verifiable answers (arithmetic, code, factual recall)
compared to the bit-exact arm. Token-level text may differ; correctness may not.

Harness: `.backlog/lab/taskcheck.sh`. Bit-exact changes still use `bench/golden.sh` as before.

**Deferred QA (do NOT run until the user says they are ready):** multi-prompt read-through —
p064/p256/p512/p1024 + the golden prompt, 60 tokens each, diff OFF vs ON, report every semantic
divergence verbatim for human judgment. This is a human-gated QA pass, not an automated gate.

## Work order (user-specified sequence)

### 1. Evaluate `--fast-kernels` for tok/s  — IN PROGRESS
Historically measured **11.01% faster decode** (38554.8 -> 34311.2 ms, n=3 interleaved), already
implemented, gated behind a flag. Non-bit-exact: md5 `7155bab905cbfa70aa06afa08f757cee` vs golden
`5d04890413ff539e802985ce8c727814`. Zero implementation cost — needs only the task-level check
plus a tok/s A/B. Fastest available path to the tok/s pain.
- [ ] confirm invocation (`--fast-kernels` flag vs `COLI_V4_KERNELS` env) and what it changes
- [ ] task-level check OFF vs ON — identical correctness required
- [ ] tok/s A/B, n>=5 (decode spread is 5-13%, so n=3 cannot resolve <10%)
- [ ] record as an experiments_results.md entry

### 2. Amortise expert dispatch
Attack the 52.5% of decode spent in `expert_forward` with the GPU idle. Fuse the 8-expert fan-out
and/or cross-layer work into fewer dispatches. Naive `MIN_N` lowering does NOT work — the GPU is
0.40x at S=1, so the win must come from amortising per-dispatch overhead, not from more dispatches.
Largest prize, largest risk.

### 3. Batch speculative verify through the MoE
`V4_DRAFT=4` currently yields ZERO extra batched groups (E87 Finding 2): the verify path does not
batch its draft tokens through the expert path, so it never clears `MIN_N=4`. Making it batch turns
S=1 into S=draft. Would make speculation pay for the first time.

### 4. Harvest cheap TTFT wins
- [ ] port upstream `e36a1c7` (#900/#1023) hot-pack outside `state->mutex` — conflicts, hand-port.
      Current tree still has `hot_pack_slot_locked()` under `state->mutex` at :7935, :8061, :8241
- [ ] `COLI_V4_MOE_BATCHED_MIN_N=3` probe (S=3 is the unmeasured crossover; upward is known-losing)
- [ ] length sweep p064/p256/p512/p1024 interleaved — does the gain grow with prompt length?
- [ ] phase-dependent `COLI_V4_METAL_ATTN` (ON prefill / OFF decode) — needs the decode penalty
      confirmed at n>=5 first; currently unestablished (ranges overlap)

## Shelved
- rows16 residual p256 divergence (user: SHELVE). Flag `COLI_V4_MOE_BATCHED_ROWS16` stays OFF.
  Perf is real (-4.64% p064 / -5.49% p256 TTFT); correctness is not bit-exact.

## Dead — do not revisit without new evidence
- **#1097 loader lanes 3->9.** `expert_wait` is 4.4% of decode; there is no I/O stall to recover.
  Second independent confirmation of E34's cold-cache rejection.
- **`V4_NGRAM` +18.5%.** Re-measured at +1.40% inside a 7.8% spread. Bit-exact but worthless here.
- **MoE batching as a decode lever.** Fires zero times during decode (prefill-only by construction).

## Standing measurement facts
- tok/s = (max_tokens-1) / `after_first`; harness `.backlog/lab/tokps.sh`.
- Decode spread 5-13% vs TTFT 0.6-0.8% — decode deltas under ~10% need n>=5.
- `bench/ab.sh` excludes decode by construction (`--max-tokens 1`); it cannot measure tok/s.
- `bench/golden.sh` is a ~20-token single-chunk prompt: a necessary gate, NOT sufficient for any
  feature conditioned on batch/group/chunk size (this is how the rows16 claim went wrong).
- Baseline: **0.90-0.97 tok/s**, ~1046 ms/token. Decode = expert_forward 52.5%, attention 26.3%.
