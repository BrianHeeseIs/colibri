# E136 — `COLI_V4_HOT_PACK_UNLOCKED` becomes a default

2026-08-31. Branch `ft-decode-generalization`. Binary `7083f237a0df3d14130ad284395b23a2`,
`METAL=1`, Metal seam linked.

## What changed
Two commits, deliberately split so the test could go red between them:
1. `feat: report hot pack policy as pack= on the v4_hot_policy line` — adds the observable.
2. `feat: default COLI_V4_HOT_PACK_UNLOCKED on` — flips the default.

`hot_pack_unlocked_init` now splits "env present" from "env value", which the previous single
expression `enabled && *enabled && atoi(enabled) != 0` could not express, and falls back to
`!coli_v4_baseline_mode()`. That shared inline is visible in `COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16`
(`deepseek_v4_internal.h` is included at the top of the unit block), which was PROVEN before the
edit rather than assumed: the preprocessor count of `coli_v4_baseline_mode` in that unit went
1 -> 2, i.e. from "defined but unused" to "defined and used".

## Why
E132: `store_lock` 454.331 -> 36.711 ms = **-91.92%** of main-thread lock time at p256, with every
validity precondition passing (V1 drift 0.398%, V5 0.016%), output byte-identical across all seven
arms of that batch, and both reversal-guard quantities moving DOWN (`expert_forward` -4.64%,
`decode_wall` -1.58%).

The end-to-end payoff is small and is stated as such: **decode wall -1.58%, tok/s +1.60%**, the
latter INSIDE the 5-13% decode noise floor. `wait_finish_complete_block` actually ROSE 6.24%,
because freeing the lock lets the main thread reach the finish barrier sooner and then wait longer
for a disk read that has not landed. **Disk, not the lock, is the binding constraint.** The
lock-time collapse is the measured result; the throughput figure is corroboration only.

The prior rejection in `mem:deepseek_v4/dead_levers` ("-0.05%, ranges fully overlapping, that
contention does not exist in this configuration") was not wrong on its own terms. Its provenance is
`.backlog/lab/hotpack_ab_20260826-205025.log`: a **p064 TTFT** A/B, interleaved n=3, dated
2026-08-26, reporting `ttft=` only — the PREFILL axis, four days before any decode instrument
existed, at the one prompt length AGENTS.md warns cannot exercise chunk-conditioned behaviour.

## The observable
`pack=locked|unlocked` is appended to the existing `v4_hot_policy` line. Before this, the pack
policy was invisible and its only evidence was `store_lock` collapsing under
`COLI_V4_DECODE_TRACE` — the same silent-state class that forced an INDETERMINATE verdict on
`COLI_V4_PREWARM` (E133b) before that got a confirmation line.

**Why it is md5-safe, and why the obvious reason is wrong.** `v4_hot_policy` is NOT in any strip
list — not in `bench/golden*.sh`, `tokps.sh`, `taskcheck.sh`, `decodetrace.sh`, `pinsweep.sh` or
`envsweep.sh`. The field is safe because of LINE POSITION: `ext()` is
`awk '/^generated_text=/{f=1} f&&!/^(...)/{print} /^timing /{f=0}'`, so capture begins ON the
`generated_text=` marker, and this line is emitted at store open roughly 57 lines earlier. Do not
move this `fprintf` after generation on the belief that it is stripped, and do not give it a new
prefix.

## TDD — a default flip with a real RED
A static grep gate would only assert that the source contains a string just typed, and the goldens
are EXPECTED to be unchanged by this flip, so neither can go red. The test surface is therefore the
engine's own reported state, four arms, each failing for a distinct real reason:

| arm | env | expect | fails if |
|---|---|---|---|
| 1 | (none) | `unlocked` | the default flip did not happen |
| 2 | `COLI_V4_BASELINE=1` | `locked` | **the baseline guard is missing — sacred md5 at risk** |
| 3 | `COLI_V4_HOT_PACK_UNLOCKED=0` | `locked` | explicit override stopped beating the default |
| 4 | `BASELINE=1 HOT_PACK_UNLOCKED=1` | `unlocked` | explicit flag stopped beating baseline |

Observed, in order, each captured to disk:
- **RED** (`E136_RED.log`, pre-edit binary): 4/4 fail on `<no-pack-field>`, engine rc=0 — the
  failure is the assertion, not the run.
- **Stage A** (`E136_GREEN_stage_a.log`, observable added, default still OFF): exactly 3/4 pass,
  arm 1 fails with `pack=locked`. Precisely the predicted intermediate state.
- **Full GREEN** (`E136_GREEN.log`, default flipped): 4/4 pass, rc=0.

Exactly one assertion flipped, driven by four changed lines. Arm 2 is the important one: it is a
cheap, repeatable, direct test of the single property protecting
`5d04890413ff539e802985ce8c727814`.

## Golden gates — both PASS, no re-record
```
PASS golden         md5=5d04890413ff539e802985ce8c727814   (SACRED, unchanged)
PASS golden_default md5=cc09015d089d9a25d10d75753f9e849a   (shipping, unchanged)
```
`bench/GOLDEN_DEFAULT_MD5` is untouched. `golden.sh` sets `COLI_V4_BASELINE=1`, so its PASS is
independent confirmation that the baseline guard works — the same property arm 2 asserts.
`golden_default.sh` exercised the shipping path with the flag ON for the first time and produced
the identical hash, confirming E132's seven-arm p256 byte-identity result at the golden prompt.

## Not done, deliberately
`v4_hot_policy` was NOT added to the strip lists. It touches nine files, changes nothing
observable, and would dilute two otherwise clean commits. The line-position argument above is what
makes the field safe; the strip-list gap is a separate documentation issue.
