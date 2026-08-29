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

### Decode is unimproved, and here is where it goes
All of the above is prefill. Decode profiling (E123, `COLI_V4_PROFILE=1`) accounts for 99.3% of it:
`attn_out` 21.6%, `expert_forward` 32.2%, `attn_qkv` 10.8%. The attention projections both run
`coli_fp8_matvec_ref`, whose 8-wide SIMD path is `#ifdef __AVX2__` — x86 only — so Apple silicon runs
a scalar loop. A drop-in NEON port was written, proved bit-exact, and measured **neutral**: the loop
is load-bound, not compute-bound (E124). The real lever is repacking fp8 weights row-interleaved, or
decoding e4m3 in-register. Raising `COLI_V4_PIN_SLOTS` does **not** help (E123).

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
