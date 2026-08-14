# Cache preservation between benchmark runs — decision

**Verdict: no *score-neutral* IO-layer preservation exists on the default direct-I/O path. A
persistent process is the only lever, and it is not worth using for benchmarking. Ship docs +
harness, change nothing in the IO path.**

*Precision note (review):* the page cache is bypassed **because `COLI_V4_DIRECT` defaults on**, and
`COLI_V4_DIRECT=0` explicitly restores buffered `pread` (`c/deepseek_v4.c:144-150`). So page-cache
warming is not impossible in the absolute — it is unavailable *without changing the I/O path being
measured*, which would make the benchmark non-representative of the shipping configuration. Partial
residency / smaller-working-set schemes were **not measured** and are not ruled out by evidence here;
they are simply out of scope for a benchmarking-speed exercise.

## Why RAM disk / page-cache warming are structurally dead
| approach | why it fails here |
|---|---|
| RAM disk for the model | Model is **155 GB on disk vs 128 GiB RAM** - it does not fit. `hdiutil` RAM disks also use **wired** memory, and swapping the storage medium invalidates the benchmark against production APFS/SSD. |
| Warming the OS page cache | Expert reads set **`F_NOCACHE`** (`COLI_V4_DIRECT` defaults on; `c/compat.h:44-52`, `c/st.h:160-161`), so the unified buffer cache is **deliberately bypassed**. Warming it cannot help expert I/O. |
| `posix_fadvise` | Does not exist on macOS. |
| `madvise(MADV_WILLNEED)` | Applies to *mapped* ranges; the engine uses `pread`, not mmap (`c/st.h:1-6`). |
| `F_RDADVISE` | Advisory only - forwarded to `VNOP_IOCTL`, filesystem decides. |
| `vmtouch` / `mlock` | Bounded by `RLIMIT_MEMLOCK`, and changes eviction behaviour => benchmark-invalid unless production also pins. |

## The only lever: keep the process alive
`SERVE=1` holds one engine + one session and serves requests forever (`c/deepseek_v4.c:9221-9279`),
READY in ~0.33 s. The in-process expert cache survives across requests: **hit_pct 75.2 % -> 95.4 %**.

## Why we do NOT use it for benchmarking
| | relative sd |
|---|---|
| one-shot `decode_wall` | **0.27 %** |
| serve plateau `tok_s` | **10.9 %** |

**~40x noisier.** Matching one-shot precision would need ~472 samples per config (~6.5 h). Probable
cause: the hot policy repins on an interval (`repin_interval=6`) and repacks rows16 -
`packed_slots=1894` in serve versus 722-902 one-shot.

**Score-neutrality rule:** the harness is measurement-only. It drives the documented serve protocol,
runs with `COLI_V4_SAVE_USAGE=0` so `.coli_usage` is frozen, and touches no engine path used for
scoring. Golden md5 `5d04890413ff539e802985ce8c727814` is re-verified after every engine change. If
the harness ever changes that md5 it does not ship on any scoring path.

## What shipped
- `.backlog/serve/ab_harness.py` (+`ab_protocol.py`, `ab_runtime.py`, 21 tests) - serve-mode driver.
- One real engine fix found by it: `COLI_V4_KERNELS` silently ignored in serve mode (587c309).
- Nothing in the IO path. There is nothing there to preserve.
