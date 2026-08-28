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

| variable | default | effect |
|---|---|---|
| `COLI_V4_METAL=1` | off | run the routed-expert forward on the GPU |
| `COLI_V4_MOE_BATCHED=1` | off | batch expert groups during prefill (gated by `COLI_V4_MOE_BATCHED_MIN_N`, default 4) |
| `COLI_V4_METAL_VARIANT=simd_exact_cold` | `ordered_cold` | use the simdgroup expert matmul |
| `COLI_V4_METAL_ROWS16=1` | off | also send pinned hot (rows16) experts to the GPU — see the warning below |
| `COLI_V4_METAL_STATS=1` | off | print dispatch, reject and `simd_exact` counters at exit |

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

Measured at p256 (N=2, 60 tokens) against `COLI_V4_METAL=1` alone: **+33.9% tok/s and -16.6%
TTFT**. It is **off by default**: that sample is below the n>=5 this project requires on the
decode axis, and the comparison against the pure-CPU path has not been run yet. See
`experiments_results.md` E95/E96.

```bash
COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold \
  ./c/deepseek_v4 /path/to/DeepSeek-V4-Flash "your prompt" --max-tokens 60 --memory-gb 96
```

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
