# Remaining measurements for `COLI_V4_METAL_VARIANT=simd_exact_cold`

Deferred from the session that built and landed the kernel (branch `simd-apple-metal`).
**Everything here is measurement only — no code change is pending.** All correctness gates that
were run PASSED; see `experiments_results.md` E96 for what is already established and for the
pre-registered decision rule in E95 that these runs are meant to settle.

## Why these were deferred
Each engine run at p256 is ~200 s (TTFT ~120-145 s + 60 s decode). A 3-arm N=5 sweep is 15 runs,
so roughly 50 minutes per prompt length. The session had already established the result was
positive and the correctness gates were green; the remaining runs tighten the statistics and add
the one comparison that the ship decision actually turns on — **versus CPU**, not versus the
other Metal arm.

## Run hygiene (mandatory, from AGENTS.md)
```bash
cd /Users/cptn/workbench/ai/colibri
pgrep -f '[d]eepseek_v4' && { echo "ABORT: engine running"; exit 2; }
ls -d /tmp/colibri-prefill-bench.lock 2>/dev/null && { echo "ABORT: stale lock"; exit 2; }
md5 -q .backlog/lab/coli_usage.snapshot     # must be 599f3d12e9347ef30541bd6f9ba18bde
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot
```
Run these **strictly one at a time**. `tokps.sh` and `taskcheck.sh` do NOT take the
`/tmp/colibri-prefill-bench.lock` that `ab.sh`/`golden.sh` use — they only `pgrep` — so two
concurrent harnesses will corrupt each other's numbers silently.
Keep `COLI_V4_KERNELS` **unset** throughout: it is nondeterministic at short prompts.

---

## 1. p512 multi-chunk differential (CORRECTNESS — highest priority)
Extends the bit-exactness evidence past p256. p256 (3 chunks) already passed with identical md5
and both arms deterministic. This was started and interrupted; no data.
```bash
N=2 TOKENS=60 PROMPT_FILE=.backlog/prefill_prompts/p512.txt ./.backlog/lab/tokps.sh \
  'metal_ord=@=COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1' \
  'metal_simd=@=COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold' \
  2>&1 | tee .backlog/lab/differential_p512_$(date +%Y%m%d-%H%M%S).log
```
**PASS** = both arms `deterministic`, both arms the SAME md5, 4 run lines present.
A divergence here is a REJECT of the wiring under E95's decision rule — do not fall back to
`taskcheck.sh` to rescue it, because the claim being made is bit-exactness and `taskcheck.sh`
answers a weaker question.

## 2. tok/s 3-arm, p064, N=5 (THE DECISION RUN)
The CPU arm is the one that matters: E95's rule distinguishes "beats CPU by >10%" (default-on
candidate) from "beats metal_ordered but not CPU" (opt-in, default OFF). Only the latter is
currently evidenced. `tokps.sh` baselines on the FIRST arm, so `cpu` goes first.
```bash
N=5 TOKENS=60 PROMPT_FILE=.backlog/prefill_prompts/p064.txt ./.backlog/lab/tokps.sh \
  'cpu=@=COLI_V4_METAL=0' \
  'metal_ord=@=COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1' \
  'metal_simd=@=COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold' \
  2>&1 | tee .backlog/lab/tokps_simdexact_p064_$(date +%Y%m%d-%H%M%S).log
grep -c '^  run' .backlog/lab/tokps_simdexact_p064_*.log      # MUST be 15
```
N=5 because the recorded decode spread on this host is 5-13% and effects here have reversed when
a fifth point was added. **Fewer than 15 run lines means an arm failed silently** — `tokps.sh` is
`set -uo pipefail` WITHOUT `-e` and continues past a failed engine run.

## 3. tok/s 3-arm, p256, N=5
Same command with `PROMPT_FILE=.backlog/prefill_prompts/p256.txt`. Needed because E95's
default-on rule requires the win at BOTH lengths — length-dependent, non-monotonic effects are
the recorded norm here (`COLI_V4_KERNELS=all` is the precedent).
Existing partial data at this length: N=2, `metal_ord` 0.9724 vs `metal_simd` 1.30205 tok/s.

## 4. TTFT A/B via ab.sh, N=3
**This one is now much more interesting than planned.** E95 pre-registered a prediction of
"TTFT ~0", and the N=2 p256 run contradicted it: TTFT fell 143.7 s -> 119.9 s (-16.6%). That was
measured by `tokps.sh`, not by the TTFT harness, so it needs confirming on `ab.sh`.

**THE ISOLATION TRAP IS THE WHOLE DIFFICULTY.** `bench/ab.sh:105-112` builds the OFF arm as
`env -u VAR` for every assignment in `ON_ENV`. So baseline flags must be `export`ed in the PARENT
shell and ONLY the variant passed in `ON_ENV`. Passing `COLI_V4_METAL=1` inside `ON_ENV` would
make the OFF arm unset it, conflating "Metal vs CPU" with "simd_exact vs ordered".
```bash
# preflight BOTH arms by hand first: ab.sh is set -euo pipefail with a trap that deletes its
# log dir, so a non-zero engine exit kills it silently and destroys the error.
cp .backlog/lab/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1 COLI_V4_SAVE_USAGE=0 ./c/deepseek_v4 \
  models/deepseek-v4-flash "$(cat .backlog/prefill_prompts/p064.txt)" \
  --max-tokens 1 --memory-gb 96 >/dev/null 2>&1; echo "off_arm_rc=$?"   # expect 0
COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold \
  COLI_V4_SAVE_USAGE=0 ./c/deepseek_v4 \
  models/deepseek-v4-flash "$(cat .backlog/prefill_prompts/p064.txt)" \
  --max-tokens 1 --memory-gb 96 >/dev/null 2>&1; echo "on_arm_rc=$?"    # expect 0
cp .backlog/lab/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage

export COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1
N=3 ./bench/ab.sh "COLI_V4_METAL_VARIANT=simd_exact_cold" ./c/deepseek_v4 \
  2>&1 | tee .backlog/lab/ab_simdexact_$(date +%Y%m%d-%H%M%S).log
unset COLI_V4_METAL COLI_V4_MOE_BATCHED
grep '^AB ' .backlog/lab/ab_simdexact_*.log
```
**FASTER IS NEGATIVE.** Noise floor 0.6-0.8%, so anything within +-1.5% is no effect.

---

## Follow-on work these measurements would unlock
- **A rows16/hot `simd_exact` kernel.** Today `simd_exact` covers only the cold (`block_rows==1`)
  layout; rows16 experts fall back to `ordered_hot_xcache`, counted as
  `v4_metal_simd_exact rows16_fallbacks=`. In the measured decode configuration that counter was
  **0**, so nothing was lost there — but `v4_metal_reject layout=3104` in the same run shows
  ~3.1k expert calls being refused by the seam on layout grounds and sent to the CPU. Covering
  rows16 would bring those onto the GPU too. Note the hot kernel applies its scale PER COLUMN to
  match the rows16 NEON loop, a DIFFERENT reference than the cold scalar path, so a hot
  `simd_exact` must reproduce THAT form, not this one.
- **Dispatch fusion across the top-6 fan-out.** E90 buried this on a premise E94 withdrew. With a
  matmul that is now ~5x cheaper at S=1, per-call submit+wait is a correspondingly larger share of
  the expert chain, so fusion is worth re-sizing against the new baseline.
- **`COLI_V4_KERNELS=all` combined with `simd_exact_cold`** — both are wins on decode; they have
  never been measured together, and `KERNELS=all` is nondeterministic at short prompts so it needs
  its own arm rather than being folded into the baseline.
