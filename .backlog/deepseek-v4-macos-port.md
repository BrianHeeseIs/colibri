# DeepSeek V4 Flash — macOS arm64 port

Status: **PORT COMPLETE AND VALIDATED ON macOS arm64.**
`make deepseek-v4-tiny-check` passes end-to-end — token-exact, with ZERO model download.
Date: 2026-08-12. Host: Apple Silicon arm64, macOS 26.6.1, clang + Homebrew libomp.

## Outcome

`make -f Makefile.deepseek-v4 ARCH=native deepseek-v4` now produces:

```
deepseek_v4: Mach-O 64-bit executable arm64
  /usr/lib/libSystem.B.dylib
  /opt/homebrew/opt/libomp/lib/libomp.dylib     <- multithreaded, not single-thread fallback
```

Test results (all green, exit 0):

```
tests/test_deepseek_v4     attention cache · config · expert · ExpertStore · KV cache
                           layer · math · prompt · resource plan · sparse attention
tests/test_v4_ownership    index-close-after-failure · model path copy
                           session lifetime accounting · tokenizer free across create/destroy
make test-c                FULL suite exit 0 — zero regressions
```

## The original HIGH-risk blocker did not exist

The previous version of this document listed `-Wl,--wrap` as the one high-risk unknown,
because `Makefile.deepseek-v4` line 2 stated:

> `# Uses GNU ld -Wl,--wrap and gcc -march; do not invoke from macOS/PowerPC`

**There is no `--wrap` anywhere in that file.** Actual link flags were:

- Windows: `LDFLAGS = -lm -fopenmp -static -pthread`
- Linux:   `LDFLAGS = -lm -fopenmp -pthread`

`grep -rn wrap Makefile.deepseek-v4 Makefile.deepseek-v4.units` returns nothing. The comment
is stale. Apple's `ld64`/`ld-prime` genuinely lacks `--wrap`, but nothing here needed it.
**Lesson: the comment was load-bearing in planning and wrong in fact.**

## Changes made (3 files)

### 1. `c/Makefile.deepseek-v4` — Darwin branch
- Platform gate extended: `UNAME_S = Darwin` + `arm64|x86_64` sets `COLI_V4_OK` and `IS_DARWIN`.
- New Darwin compile branch, mirroring `c/Makefile`'s proven Apple-clang handling:
  - `CC = clang`
  - **`-mcpu=$(ARCH)` on arm64** (clang rejects `-march` there); `-march` retained for Intel Macs
  - **`-D_DARWIN_C_SOURCE` replaces `-D_GNU_SOURCE`** (glibc-only feature-test macro)
  - OpenMP via Homebrew libomp, detected exactly as `c/Makefile` does — verifies BOTH
    `include/omp.h` and `lib/libomp.*` exist before adding flags, because
    `brew --prefix libomp` prints a prospective path even when not installed
  - falls back to a single-threaded build with a warning if libomp is absent
- Stale `--wrap` comment corrected.

### 2. `c/Makefile` — gate relaxed
`COLI_V4_SUPPORTED` now also set for `AARCH64 + DARWIN`. This additionally un-excludes
`tests/test_deepseek_v4` and `tests/test_v4_ownership` from `make test-c` on macOS.

### 3. `c/deepseek_v4.c` — one portability fix
`_SC_AVPHYS_PAGES` (glibc-only) is undefined on macOS, and there is no `/proc/meminfo`.
Added an `__APPLE__` branch to the available-memory helper (~line 687) that mirrors
`colibri.c:8270-8276`'s `mem_available_gb()`:

```c
host_statistics64(mach_host_self(), HOST_VM_INFO64, ...)
-> (free_count + inactive_count + purgeable_count) * sysconf(_SC_PAGESIZE)
```

Same semantics as Linux `MemAvailable`: pages reclaimable without swapping.
Guarded `#include <mach/mach.h>` added. **This is the only C change in the whole port.**
Correctness is exercised by `DeepSeek V4 resource plan tests: ok`.

## Token-exact oracle: PASSED (zero download)

`transformers 5.15.0` **does** expose `DeepseekV4ForCausalLM`, so the fixture generator works:

```
uv pip install --python .venv-convert/bin/python transformers   # 5.15.0
python tools/make_deepseek_v4_tiny.py --output ./deepseek_v4_tiny --force
  -> wrote deepseek_v4_tiny (928658 bytes, transformers=5.15.0)
make deepseek-v4-tiny-check PYTHON=<venv python>   -> EXIT 0
```

Full result:

```
PASS target short:      teacher forcing and greedy token-exact
PASS target greedy:     truncated prefix rejected
PASS target compressed: teacher forcing and greedy token-exact
PASS target long:       teacher forcing and greedy token-exact
PASS target session short/long: exact IDs and exact length
PASS target CLI:        prompt beyond the old 512-token cap
PASS target serve:      persistent SUBMIT/DATA/DONE protocol is token-exact
PASS prefix reuse:      11 tokens reused, output identical to a cold prefill
PASS prefix repeat/reset
```

Runtime self-report during the oracle run confirms the storage path works on macOS/ExFAT:

```
v4_ssd_io mode=direct-aligned fallback=buffered-pread
v4_autopin history=.../deepseek_v4_tiny/.coli_usage selections=6138
v4_hot_policy pin_slots_per_layer=0 repin_interval=4 mode=resident-ram rows16=hot-pins
```

**This is the go/no-go gate from the original plan, and it is a GO** — correctness proven on
real hardware before spending 167 GB of bandwidth.

## Oracle validity: the PASS is NOT circular (verified)

Regenerating the fixture with `--force` modified two repo-tracked files. Diffing them to make
sure the token-exact result was not validated against a reference my own toolchain had just
produced:

```
config.json:  transformers_version 5.14.1 -> 5.15.0        (only line changed)
ref.json:     transformers_version 5.14.1 -> 5.15.0
              torch_version        2.13.0+cpu -> 2.13.0     (only lines changed)
```

**Payload is byte-identical.** Unchanged: `prompt_ids_short`, `prompt_ids_compressed`,
`prompt_ids_long`, `cases`, `quantization_format`, `config_summary`, `seed: 1234`, and every
architecture field (`model_type: deepseek_v4`, `expert_dtype: fp4`,
`scoring_func: sqrtsoftplus`). Only toolchain provenance strings differ.

So the engine was checked against the reference data the repository ships. If the payload had
been regenerated, the "token-exact" claim would have been self-referential and meaningless.

## Working-tree state (nothing committed, per operator instruction)

```
M .gitignore                      new: ignores models/, build artifacts, runtime logs
M c/Makefile                      COLI_V4_SUPPORTED gate + Darwin
M c/Makefile.deepseek-v4          Darwin build branch
M c/deepseek_v4.c                 __APPLE__ available-memory branch
M c/deepseek_v4_tiny/config.json  provenance string only (see above)
M c/deepseek_v4_tiny/ref.json     provenance strings only (see above)
?? validation/glm52-metal/        harnesses + evidence
?? .backlog/                      this file
```

Nothing is staged and nothing is committed — the operator reserves commit approval.
`c/Makefile.orig` and `c/Makefile.deepseek-v4.orig` hold pre-port copies.

**Decision needed before the port can be exercised end to end:** the checkpoint is ~167 GB and
`/Volumes/Extreme SSD` has ~126 GB free after GLM-5.2. Options: `ken_disk` (500 GB free),
free space on the Extreme SSD, or split shards across drives with `COLI_MODEL_DIRS`
(primary must hold config/tokenizer/sidecars; st.h:414-473).

## Next steps

1. ~~Resolve the `DeepseekV4ForCausalLM` dependency; run `make deepseek-v4-tiny-check`.~~ DONE — PASSES.
2. Token-exactness has passed. Next: download `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB).
   Disk note: `/Volumes/Extreme SSD` currently has ~125 GB free after GLM-5.2 — **167 GB will
   not fit**. Needs `ken_disk` (500 GB free) or space freed.
3. Run `python ./coli chat --model <dir> --ram 22`.
4. Upstream PR: the repo already anticipated macOS (`__APPLE__` branch present at
   deepseek_v4.c:6655) but never wired the build. The diff is small and self-contained.

## Upstream notes (unchanged)

- Greedy decode only, one KV slot; tools and grammar not wired for V4.
- `V4_DRAFT` / `V4_MTP` default `0` upstream — measured to cost more than they saved.
- `--ram` is the highest-value knob (sets expert cache hit rate); changes speed only, never output.

## Backups

`c/Makefile.deepseek-v4.orig` and `c/Makefile.orig` hold the pre-port originals.
