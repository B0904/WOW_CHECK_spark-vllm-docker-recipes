# Kimi-K3 measured results — 16x GB10, 2026-08-23/24

Effective tok/s throughout = `completion_tokens / wall_clock` (all speculation overhead included).
Median of 5 samples after an 8-request JIT warmup unless stated.

## Speculative-token sweep (short prompts)

| `num_speculative_tokens` | code | prose |
|---|---|---|
| 3 | 14.88 | 12.98 |
| 6 *(upstream default)* | 17.59 | 13.01 |
| **8 — chosen** | **20.05** | **15.06** |
| 12 | 19.92 | 14.15 |

Deeper wins because the verify pass dominates. Plateaus at 8 — by position 12 acceptance is
9% of position 1. Final config re-measured at **21.04 code / 15.16 prose**.

Per-position accepts at spec=8 (~1162 draft cycles):

    p1=1127  p2=941  p3=790  p4=631  p5=507  p6=428  p7=353  p8=268
    -> position 1 accepted ~97% of the time; position 8 ~23%

## Other levers

| change | decode (code) | decode (prose) | prefill @75K |
|---|---|---|---|
| B12X PCIe paths off | 20.03 | 15.19 | — (neutral) |
| NCCL channels 2 -> 4 | 19.39 | 13.85 | 876 -> **1316 t/s (+50%)** |

Channels are a **decode-vs-prefill trade**: more lanes help big bandwidth-bound prefill
all-reduces, hurt the tiny 14 KB latency-bound decode ones. Same physics as a 100->200 Gb/s
fabric upgrade, which also only helps prefill.

## Decode vs context depth (3 samples/point, prefix-cache assisted)

| depth | decode | DSpark acceptance |
|---|---|---|
| 4K | 15.62 | 39.3% |
| 50K | 13.64 | 34.8% |
| 100K | 12.26 | 29.8% |
| 200K | 11.14 | 31.5% |
| 300K | 9.31 | 34.3% |
| **460K** | **7.96** | 28.4% |

**-49% from 4K to 460K.** Decode falls because acceptance falls — longer context gives the
drafter more to condition on and it predicts less well. NB: synthetic repetitive filler drafts
poorly (39% at 4K vs **65% on real coding prompts**), so these are a floor, not typical.

**Cold TTFT at 460K: 28.1 minutes.** The same depth warm (prefix cache hit) was 18 seconds.

## Prefill vs depth (`tg=1500`, comparable to ciprianveg's published table)

| depth | ours pp t/s | his pp t/s |
|---|---|---|
| 4K | 663.0 | 675.1 |
| 50K | 731.0 | 748.4 |
| 100K | 693.6 | 712.8 |
| 200K | 624.5 | 657.6 |
| 300K | 569.3 | 607.7 |

Within 2-6% of his at every depth.

Prefill has **~6.5 s of fixed per-request overhead** (a 4.7K and an 18.7K prompt both took ~6.5 s),
so use >=64K prompts to measure prefill meaningfully.

## Correctness

7/7 at spec=8, including `8347 * 291 = 2428977` (multi-step arithmetic, where corrupted
speculation would show).

Multi-image vision, 16-image limit: 1, 3 and 6 distinct labelled images all read correctly and
**in order** (3.2s / 15.7s / 17.0s).

## Cluster behaviour under load

All 16 nodes at 96% GPU util but only **12-15 W**, CPUs 72-84°C — **latency-bound on collectives,
not compute-bound**. Whole cabinet (all gear, main PDU) measures **2.2 kW**.

Memory: **111.8 GiB free per node** after weights; KV reserved 4.66 GiB. `gmu 0.881` was never at
risk. NB if weights ever don't fit, *lowering* `gpu-memory-utilization` makes it worse — it is the
ceiling weights load inside.
