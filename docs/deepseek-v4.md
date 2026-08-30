# DeepSeek V4 target engine (colibri CPU)

[简体中文](deepseek-v4.zh-CN.md)

This is the target-only DeepSeek V4 Flash engine for the first PR of the V4
split. DSpark speculative decoding is intentionally excluded and belongs in a
separate stacked follow-up.

## Scope

- Production code is in `c/deepseek_v4.c`; the experimental public engine and
  session API is in `c/deepseek_v4.h`.
- Official sharded safetensors checkpoints load through shared `st.h`.
- Standard MXFP4 matrix multiplication uses shared `quant.h`.
- Unified `c/coli` routes `run`, `chat`, `serve`, and `web` to V4. Serving keeps
  the engine and caches warm across requests.
- `--no-dspark` is a compatibility no-op. This PR has no DSpark model, memory
  tier, or speculative loop.
- Build targets are x86-64/aarch64 Linux and Windows/MSYS2.

Destroy every session before destroying its engine.

## Shared migration status

| Checkpoint path | Current implementation | Follow-up |
|---|---|---|
| Safetensors index/range reads | shared `st.h` | done |
| fmt7 standard MXFP4 matmul | shared `quant.h` | done |
| fmt7 resident rows16 expert cache | temporary V4-private layout | **TODO:** migrate after upstream exposes a resident rows16 API |
| fmt8 E4M3 + UE8M0 128x128 scales | shared `st_read_scale_f32` + `quant.h` `matmul_fp8` | done |

Only the rows16 resident-cache layout remains V4-private. Its
`TODO(upstream-fmt7-rows16)` marker names the shared API still needed before
that specialized cache layout can be removed.

## Memory policy

A typical checkpoint has 43 transformer layers, hidden size 4096, and 256
routed experts per sparse layer with top-k 6. Dense weights occupy about
6.27 GiB and a resident BF16 output head about 1.06 GiB. Routed-expert weights
are streamed and cached according to the RAM budget.

The planner reserves workspace and a minimum expert working set, then enables
dense/head residency and grows the expert cache when memory permits. Dense
residency is independent of DSpark and works in this target-only build,
including with the legacy `--no-dspark` option.

`--ram GiB` is a planner budget, not an OS-enforced limit. Without it, the
budget is derived from currently available OS memory.

## Download

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

A download can finish with a truncated shard even when the client reports
success. If `st.h` rejects a shard as out of bounds, compare every local shard
size with the Hugging Face repository before treating it as an engine failure.

## Build and use

```bash
cd c
make deepseek-v4
python ./coli run --model /path/to/DeepSeek-V4-Flash --ram 32 \
  "What is the capital of France?"
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli serve --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli web --model /path/to/DeepSeek-V4-Flash --ram 32
```

Generation length: `--ngen` is a ceiling, not a target — answers end at EOS.
If the ceiling exceeds what the context window can hold, the engine clamps it
and says so on stderr; raise `CTX` for genuinely longer answers.

V4 chat uses native model markers. Native serving currently supports greedy
generation and one active KV slot; tools and grammar are rejected. Requests
re-prefill their context, while the process, weights, dense tensors, head, and
expert cache stay warm.

### Apple Silicon: the Metal expert path (`METAL=1` builds)

Build with `make -f Makefile.deepseek-v4 METAL=1 deepseek-v4`. `METAL` defaults to `0`
and a plain `make` compiles the seam out with no error and no warning, which makes every
Metal environment variable a silent no-op — check with
`nm c/deepseek_v4 | grep -c coli_v4_metal_expert_forward_batch`.

> **Defaults changed 2026-08-29.** The gates below now default **ON**; the engine ships the fast
> path. Set `COLI_V4_BASELINE=1` to restore every historical default at once. See
> `experiments_results.md` E114-E122 and the note under [Performance defaults](#performance-defaults).

| variable | default | effect |
|---|---|---|
| `COLI_V4_KERNELS` | `all` | enable the tuned `attn_sparse` and `router` kernels. `none` disables. |
| `COLI_V4_MOE_GROUPED` | **on** | group routed rows by expert during prefill |
| `COLI_V4_MOE_BATCHED` | **on** | batch expert groups into one GPU dispatch (gated by `COLI_V4_MOE_BATCHED_MIN_N`, default 4) |
| `COLI_V4_MOE_BATCHED_ROWS16` | **on** | also admit pinned hot (rows16) experts to the *batched prefill* dispatch — takes the Metal row share from 52.9% to 71.0% (E116) |
| `COLI_V4_MOE_WHOLE_PROMPT` | **on** | run the expert dispatch once per prompt instead of once per 64-token chunk (E119) |
| `COLI_V4_MOE_TILE` | `1024` | upper bound in tokens on one deferred dispatch; caps its buffers at ~176 MB. `0` = unbounded (E122) |
| `COLI_V4_METAL_ATTN` | **on** | run *prefill* attention on the GPU. Bit-exact, -21% TTFT (E115) |
| `COLI_V4_METAL_VARIANT` | `simd_exact_cold` | simdgroup expert matmul (was `ordered_cold`) |
| `COLI_V4_METAL` | **off** | run the *single-token decode* expert forward on the GPU. Measured **slower** on M3 Max — see E99. Not part of the recommended stack. |
| `COLI_V4_METAL_ROWS16=1` | off | also send pinned hot experts to the GPU on the *single-expert decode* path — see the warning below |
| `COLI_V4_METAL_STATS=1` | off | print dispatch, reject and `simd_exact` counters at exit |
| `COLI_V4_MOE_GROUPED_STATS=1` | off | print group counts, `metal_row_share` and the group-size histogram |
| `COLI_V4_FP8_ROWS16` | **on** | NEON rows16 fp8 matvec for DECODE attention projections. Bit-exact, +10.18% tok/s (E125) |
| `COLI_V4_FP8_ROWS16_MAXMB` | 8192 | legacy cap from the rejected shadow-copy design; the in-place permute uses no extra memory |
| `COLI_V4_HEAD_ILP` | **on** | four vocabulary rows in flight in the LM head. Bit-exact, head phase -62% (E126) |
| `COLI_V4_HC_OMP` | **on** | parallelise the hyper-connection mix matvec. Bit-exact, phase -63% (E126) |
| `COLI_V4_FP8_DUAL_ROWS16` | **on** | rows16 kernel for the fp8 dual matvec (shared-expert gate/up). Bit-exact (E126) |
| `COLI_V4_SPARSE_OMP` | **on** | parallelise the 64-head sparse-attention loop. Bit-exact, phase -79% (E127) |
| `COLI_V4_BASELINE=1` | off | restore every historical default in one move |

**Two different `ROWS16` knobs — do not confuse them.** `COLI_V4_METAL_ROWS16` gates the
*single-token decode* Metal path (warning below, still off). `COLI_V4_MOE_BATCHED_ROWS16` gates the
*batched prefill* dispatch and is now on by default.

**`COLI_V4_PREFILL_PREFETCH=1` deadlocks** when combined with the default GPU prefill stack (E113).
Do not enable it.

**`COLI_V4_METAL_ROWS16` is not bit-exact and is off for that reason.** Without it the decode path
refuses every pinned hot expert, which on a 24-token run is 2218 of 4902 expert calls — 45% — so
the GPU only ever sees the cold-layout remainder (`COLI_V4_METAL_STATS=1` prints
`v4_metal_single_entry rows_rejects=`). Enabling it takes coverage to 4902/4902.

The cost is that `bench/golden.sh` no longer reproduces its md5. The output is not corrupted:
the task-level gate `.backlog/lab/taskcheck.sh` scores 5/5 on both arms with byte-identical text,
and on golden's own prompt the whole difference is one token in sixty (`FFN layers.` becomes
`FFN layer.`), reproduced deterministically. It is a near-tie logit flip of the same family as
`COLI_METAL_GEMM_MIN`/#622 elsewhere in this engine. Use it only where token-exact parity with the
CPU is not required, and see `experiments_results.md` E97.

`simd_exact_cold` maps one simdgroup to each output row instead of one thread. At decode the
expert matmul is a single row against 2048 outputs, so the default kernel launches only 2048
threads and leaves the GPU thread-starved; the simdgroup form is a 32x occupancy increase and
is **bit-identical** to the CPU path — `bench/golden.sh` is unchanged and a multi-chunk p256
differential produces an identical generated-text md5.

(The earlier "+33.9% tok/s vs `COLI_V4_METAL=1` alone" figure from E95/E96 has been superseded:
that comparison was against another Metal arm, and the pure-CPU comparison it lacked was later run.
`COLI_V4_METAL=1` is **slower** than CPU decode on this host — E99.)

## Performance defaults

Since 2026-08-29 the engine ships the measured-fastest stack, so the fast path needs no flags:

```bash
./c/deepseek_v4 /path/to/DeepSeek-V4-Flash "your prompt" --max-tokens 60 --memory-gb 96
```

Against the previous `KERNELS=all` CPU arm at p256 (N=3-5, non-overlapping ranges):
**time to first token -48.2%, net wall at 40 tokens -38.5%, tok/s -1.75%.** The gain grows with
prompt length — whole-prompt dispatch alone is -17.4% TTFT at p256 and -25.9% at p512 (E120), because
a longer prompt spans more chunks and so the union of routed experts is larger.

**The default output is no longer token-identical to the historical CPU arm.** It is
capability-equivalent, not byte-equal: `.backlog/lab/taskcheck.sh` scores 5/5 on every arm, and the
differences are wording-level. Two gates enforce this split:

- `bench/golden.sh` pins `COLI_V4_BASELINE=1` and still guards the deterministic reference with its
  original md5.
- `bench/golden_default.sh` guards the shipping path against `bench/GOLDEN_DEFAULT_MD5`. That value is
  **not** sacred — re-record it deliberately when a default changes, and say why.

For bit-exactness differentials, regression triage or bisecting, set `COLI_V4_BASELINE=1` first.

### Decode: +28% across E125-E127
Decode went from 1.6655 to 2.1341 tok/s at p256 (N=5, non-overlapping ranges), and TTFT improved as
a side effect because several of these kernels are also on the prefill per-item path. **Every step is
bit-exact**: both golden hashes were reproduced after each landing and no expected value was edited.

| stage | tok/s | vs start |
|---|---|---|
| before E125 | 1.6655 | — |
| E125 fp8 rows16 | 1.8350 | +10.18% |
| E126/E127 head ILP, hc_norm, fp8 dual, sparse attention | **2.1341** | **+28.1%** |

**Versus the historical engine the figure is larger, and it is a different claim.** E128 measured
`COLI_V4_BASELINE=1` against shipping defaults directly (p256, N=3, both arms deterministic):
1.3948 -> 2.1596 tok/s = **+54.83%**, with TTFT -62.4% and net wall at 40 tokens -57.1%. The extra
over +28.1% is not decode work from these experiments — `COLI_V4_BASELINE=1` also disables
`KERNELS=all`, so it starts below the 1.6655 CPU arm the table above is measured from, and the TTFT
share belongs to the E114-E119 prefill stack. Use +54.83% for "what do I get today", +28.1% for
"what did the decode work buy".

**All four E126/E127 wins were the same defect class**: a fast path compiled only for x86, or a hot
loop left serial on a 16-thread machine. None needed new numerics. The systematic move for the next
reader is to grep the decode path for `#ifdef __AVX2__` with no `__aarch64__` sibling, and for hot
loops with no `#pragma omp`. That sweep is now **exhausted for decode** — the remaining AVX2-only
sites are an `immintrin.h` include guard and the prefill batch path.

Largest remaining phase is `expert_forward` at ~41.7%, and **the whole expert-arithmetic lever is now
closed (E129)**. It is not a coverage problem — the scalar and NEON mxfp4 kernels measure within
1.00-1.14x of each other, so raising `COLI_V4_PIN_SLOTS` stays closed (E107, vindicated by direct
measurement in E126). It is not a kernel-design problem either:

- A ceiling arm that abandons Metal bit-parity entirely — block scale hoisted, FMA, reassociation,
  16 accumulator chains — measures only **1.154x / 1.201x** on the two real expert shapes.
- Widening accumulator chains, the mechanism that gave the LM head 2.7x in E126a, is **flat** here:
  4/8/12/16 chains give 1.00/1.09/0.95/0.99, non-monotone.
- Against a **measured** kernel split of 22.05% NEON / 77.95% scalar — not the ~6% previously
  inferred from pin counts — that ceiling is worth **+4.88% decode**, or +6.26% even if both kernels
  were rewritten. The decode noise floor is 5-13%.

The remaining decode levers are elsewhere: `expert_wait` (~1395 ms, structural — two condvar waits
per expert call) and the streaming/residency path, not the expert arithmetic.

### Decode: +10.18% from a NEON fp8 kernel (E125)
Decode profiling (E123) accounts for 99.3% of it: `attn_out` 21.6%, `expert_forward` 32.2%,
`attn_qkv` 10.8%. The attention projections both run `coli_fp8_matvec_ref`, whose 8-wide SIMD path is
`#ifdef __AVX2__` — x86 only — so Apple silicon ran a scalar loop.

`COLI_V4_FP8_ROWS16` fixes that: a 16-row interleaved layout so one column's 16 weights are a single
`vld1q_u8`, and an e4m3 decode done by **reinterpreting the byte as f16** (`h = ((b & 0x80) << 8) |
((b & 0x7F) << 7)`, then `vcvt_f32_f16`), which is exact for every value including subnormals, with
the resulting 2⁸ folded into the block scale. The e4m3 NaN codes are the one case needing an explicit
select. **+10.18% tok/s at N=5 with non-overlapping ranges, bit-exact** (both golden md5s unchanged),
and no extra memory — the weights are permuted in place, and the layout-unaware prefill path restores
row-major order on demand.

Three things that do **not** work, all measured, so nobody repeats them:
- A drop-in NEON port of the existing loop is **neutral** — it is load-bound, not compute-bound (E124).
- Arithmetic bit-twiddle e4m3 decode (the PyTorch/MLX approach) is **0.70×**. The 1 KB LUT is
  L1-resident and retires at ~1/cycle; only a nearly-free replacement beats it.
- Keeping the packed weights as a **second copy** costs +4 GB and the memory pressure reverses the
  win entirely (E125). Permute in place.
- Raising `COLI_V4_PIN_SLOTS` does **not** help (E123).

Still open on decode: `expert_forward` (32.2%) uses the mxfp4 path, untouched here; and DeepSeek-V4
never adopted `c/omp_tune.h`, so it runs 16 threads including the 4 efficiency cores against that
header's own P-cores-only policy.

Note: an unrecognised `COLI_V4_METAL_VARIANT` value silently falls back to `ordered_cold`, so a
typo produces the default rather than an error. `COLI_V4_METAL_STATS=1` prints
`v4_metal_simd_exact matmuls=N`, which is zero unless the simdgroup kernel actually ran.

## Validation

The tiny safetensors fixture is generated locally, ignored, and not committed:

```bash
python -m pip install -r tools/requirements-deepseek-v4-tiny.txt
make deepseek-v4-tiny-check
```

This covers loading, teacher forcing, greedy decode, long/repeated sessions,
`--no-dspark` compatibility, and two requests through the persistent
`SUBMIT`/`DATA`/`DONE` protocol.

For a real checkpoint:

```bash
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash \
  MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

The oracle is target-only. DSpark on/off speed, acceptance, and token identity
evidence belong to the stacked DSpark PR.

## Follow-ups

- Add non-greedy sampling and more serving slots.
- Add shared replacements for the two temporary private quant paths above.
- In the stacked PR, restore DSpark without changing target tokens and report
  DSpark on/off performance and acceptance data.
