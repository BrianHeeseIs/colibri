# QD sweep — GATE A result (2026-08-14)

Real record offsets (artifacts/layer_contig.json), engine-identical F_NOCACHE aligned reads,
256-record shuffled list, 3 timed rounds median.

| QD | GB/s | p50 | p99 |
|---|---|---|---|
| 1 | 5.227 | 2.501 ms | 2.970 ms |
| 4 | **7.028** | 7.377 ms | 13.203 ms |
| 8 | 7.029 | 13.534 ms | 26.520 ms |
| 16 | 6.657 | 28.089 ms | 56.449 ms |
| 32 | 6.628 | 58.710 ms | 109.101 ms |

**GATE_A: STOP ratio_qd8_qd1=1.34** — deep-queue design killed. SSD saturates at QD4.
