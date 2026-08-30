# E130 — decode critical-path trace: analysis and verdict

Date 2026-08-30. Branch `ft-decode-apple-metal`. Binary md5 `7d454de9ba229e0ea083845a7a9e594b`,
built `METAL=1`, Metal seam linked (`nm | grep -c coli_v4_metal_expert_forward_batch` = 1).
Prompt p256, 40 tokens (39 decoded), `--memory-gb 96`, seed md5 `599f3d12e9347ef30541bd6f9ba18bde`.
Arms: one trace-off control, two trace-on replicates, one invocation of
`.backlog/lab/decodetrace.sh`.

Logs (every figure below is quoted from one of these):
- off  `.backlog/lab/decode_trace_off_20260830-192058.log`
- on_1 `.backlog/lab/decode_trace_on_1_20260830-192200.log`
- on_2 `.backlog/lab/decode_trace_on_2_20260830-192301.log`

## Trust conditions — evaluated BEFORE any verdict

| # | condition | required | observed | verdict |
|---|---|---|---|---|
| TC1 | execution proof | absent off, >0 on | off 0 lines; on_1 `decode_trace_total_calls=138510`, on_2 `=138526`, both `stages_nonzero=23/26` | PASS |
| TC2 | named-subset liveness | 5 stages calls>0 | wait_start_lock 10062, finish_calls 10062, store_lock 10062, tensor_lookup 1677, decode_alloc 5070 — both arms | PASS |
| TC3 | reconciliation R | 0.80–1.20 | on_1 1510.414/1511.391 = **0.9994**; on_2 1519.598/1520.605 = **0.9993** | PASS |
| TC4 | on_1 vs on_2 spread | <= 25% | max 13.26% (`store_publish`); wait total 0.61%; 16 of 21 stages under 5% | PASS |
| TC5 | zero cost | <= 5% tok/s | off 2.1738, on mean 2.1794 tok/s → **0.26%** | PASS |
| TC6 | md5 identity | all arms equal | all three `d06053793d8f66d4f69a3f7c810441e1`; `S6_PREFIX_CONTRACT: PASS` | PASS |

Two independent structural confirmations that the wiring is real, not coincidental:
- `calls=10062` for every wait stage = 39 tokens x 43 layers x 6 routed experts, exactly.
- `io_crosscheck` equals `store_disk_read` to the microsecond in both arms (2987.867 / 2973.455),
  and both were recorded at different points in the function from the same timespec pair.

## Both axes (mandatory)

| arm | TTFT s | after_first s | tok/s |
|---|---|---|---|
| off | 42.891 | 17.941 | 2.1738 |
| on_1 | 42.497 | 17.886 | 2.1805 |
| on_2 | 42.521 | 17.904 | 2.1783 |

The trace costs nothing measurable on either axis; the on arms are marginally faster, which is
inside the run-to-run noise floor rather than a real speedup.

## VERDICT — the `expert_wait` lever is LIVE

**S = 8.47% of decode wall** (on_1 8.445, on_2 8.488), against a pre-registered kill threshold of
3.0%. The criterion does NOT fire. `expert_wait` is not dead and is not closed.

> The `expert_wait` lever is LIVE at 8.47%. The dominant sub-stage is
> `wait_finish_complete_block` at 8.32% of decode wall, which is 98.5% of all main-thread wait.
> E130 records the decomposition and opens the next experiment against that sub-stage.

## The wait is one stage, and it is disk-miss-driven

| wait stage | on_1 ms | % decode wall | share of W |
|---|---|---|---|
| **wait_finish_complete_block** | **1488.243** | **8.321** | **98.5%** |
| wait_start_publish | 18.862 | 0.105 | 1.2% |
| wait_start_lock | 1.351 | 0.008 | 0.09% |
| wait_start_scan | 0.922 | 0.005 | 0.06% |
| wait_finish_release | 0.534 | 0.003 | 0.04% |
| wait_finish_lock | 0.501 | 0.003 | 0.03% |
| wait_start_idle_block | 0.000 | 0.000 | 0% |
| total W | 1510.414 | 8.445 | 100% |

The counters explain the mechanism completely:

- `start_slept_calls = 0`. The main thread NEVER blocked acquiring a loader slot. Loader-pool
  contention is not a problem and needs no work.
- `finish_completed_at_entry = 9339 / 10062` = **92.8%** of finishes found the loader already done
  and cost nothing.
- `finish_slept_calls = 723` = **7.2%** of finishes actually blocked, and those 723 sleeps account
  for essentially the whole 1488 ms — a mean of **2.06 ms per real sleep**.
- `store_disk_read` fired **765** times: a 7.6% miss rate on 10062 lookups.

723 sleeps against 765 disk reads is very nearly one to one. **The main thread blocks when, and
only when, an expert misses the cache and has to be read from disk.** Slot contention, lock
contention and publish costs are all noise by comparison.

## The largest stage in the whole trace is disk read, and half of it is already hidden

`store_disk_read` = **2987.867 ms = 16.705% of decode wall**, 765 calls, mean **3.906 ms**.
That is nearly twice the entire main-thread wait.

Main thread absorbed 1488.243 ms of that 2987.867 ms, so **the async loader is already hiding
50.2% of disk time**. The remaining half is the 8.3% the main thread pays.

This reframes the lever. There are two independent ways to attack it:
1. **Fewer misses** — 7.6% miss rate over 10062 lookups. Cache/pinning work reduces the 765.
2. **Hide more of each miss** — the loader already conceals half; deeper lookahead conceals more.

## Ranked remaining stages

| stage | on_1 ms | % decode wall | note |
|---|---|---|---|
| store_disk_read | 2987.867 | 16.705 | loader thread; 50.2% already hidden |
| store_pack | 699.763 | 3.912 | 3908 calls, mean 179 us — second-largest controllable cost |
| store_lock | 466.225 | 2.607 | 10062 calls, mean 46 us |
| omp_hc_pre_wall | 287.603 | 1.608 | 3354 regions |
| omp_sparse_wall | 230.380 | 1.288 | 1677 regions |
| store_hit_scan | 44.767 | 0.250 | |
| store_publish | 38.178 | 0.213 | |
| store_slab_alloc | 7.202 | 0.040 | 502 calls |
| decode_alloc | 5.075 | **0.028** | see below |
| store_miss_select | 4.843 | 0.027 | |
| tensor_lookup | 2.978 | **0.017** | see below |

**OpenMP answer:** hc 1.608% + sparse 1.288% = **2.90% combined**, master-side. That is above the
2% line the roadmap asked about, so OpenMP overhead is real but second-order next to disk.

## Two levers closed cheaply by this run

- **decode allocation is DEAD at 0.028%.** The T2 reading that decode allocation might matter —
  hundreds of malloc/free pairs per token across the hc pair, router scratch, block scratch and
  head scores — is REFUTED. 5070 allocation groups cost 5.075 ms total, a mean of 1.0 us. The
  prebind-and-scratch lever should not be pursued.
- **tensor_lookup is DEAD at 0.017%.** By-name shared-expert view resolution, 1677 calls at a mean
  of 1.8 us, is not worth caching.

Both were rated plausible before this run. Retiring them costs nothing further.

## Known gap — report this, do not paper over it

`omp_head_wall` recorded **0 calls**. The instrumented site is the bf16 non-resident head path;
the shipping configuration takes the resident `head_ilp` path instead, which was not instrumented.
The head OMP region is therefore UNMEASURED, not measured-as-zero. The 2.90% OpenMP figure above
is a floor, missing the head contribution. Wiring the resident head path is a small follow-up.

`stages_nonzero=23/26`: the three zero stages are `wait_start_idle_block` (genuinely never taken,
`start_slept_calls=0` corroborates), `omp_head_wall` (the gap above), and the third is accounted
for by the same idle path. No stage is zero because of a wiring fault.

## Next experiment

Against `wait_finish_complete_block` / `store_disk_read`, in that order. The measurement to beat:
**765 misses, 2987.9 ms of disk read, 50.2% hidden, 1488.2 ms landing on the main thread = 8.32%
of decode wall.** Any candidate must move one of those four numbers.
