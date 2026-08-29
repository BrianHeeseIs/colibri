# P3 / P4 probe results

## P4 — omp reduction on the decode path: NONE
`grep -n 'reduction(' c/deepseek_v4.c c/quant.h` returns exactly one hit:
  c/deepseek_v4.c:8464   #pragma omp parallel for schedule(dynamic, 1) reduction(+:warmed)
That is `hot_prewarm_history`, an INT counter, at load time, not on the decode path.
=> Zero live FLOAT reductions during decode. Thread count cannot change float summation order,
   so T5 (OMP_NUM_THREADS) is a pure scheduling change and `bench/golden.sh` is a valid gate for it.

## P3 — is coli_v4_hc_pre nested inside an OpenMP parallel region? NO (TOP-LEVEL)
`normalized_hc_pre` decode callers are :5709 and :5725 inside `block_token_pipeline`.
`awk` over lines 5600-5800 finds NO `#pragma omp` at all.
=> hc_pre executes at top level, so adding `#pragma omp parallel for` inside it will actually take
   effect (it would be inert under the default OMP_NESTED=false if it were nested).
