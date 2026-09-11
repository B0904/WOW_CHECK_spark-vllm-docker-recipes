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

---

# v4-prd image on STOCK MXFP4 weights (2026-08-30)

Image `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd` (`873542172ea6`, 31.9 GB),
pointed at stock `moonshotai/Kimi-K3` — **not** the QSRT checkpoint it ships for.
Same command as v3 otherwise: TP=16, DCP=8, B12X_MLA, spec=8, 500K, channels=4,
`--limit-mm-per-prompt image:64` (verified live, vision tower loaded).

## Two things had to change to make it start

1. **Removed 4 `VLLM_DSPARK_*` env vars** inherited from v3's script:
   `SHARD_MARKOV_HEAD`, `CAPTURE_SHARDED_MARKOV`, `REPLICATE_MARKOV_W1`,
   `COMPACT_ROPE`. v4's own script sets **none** of them. Forcing
   `VLLM_DSPARK_SHARD_MARKOV_HEAD=1` trips a hard guard in
   `vllm/models/kimi_k3/nvidia/dspark_mla.py`:
   ```
   RuntimeError: A TP-sharded DSpark Markov head requires matching
                 rank-local target and Markov vocabulary projections.
   ```
   The guard needs `model.supports_local_draft_argmax()`, which is **False for
   stock MXFP4** (true for QSRT). `VLLM_DSPARK_DRAFT_KV_WINDOW` was kept.
2. Nothing else — the v3 command block worked verbatim.

**v4's `/usr/local/bin/serve-kimi-k3-*` wrappers are BROKEN**: they exec
`serve-kimi-k3-full-mxfp4-dcp16-1m-dspark.sh`, which no longer exists. The K3
script is now `serve-kimi-k3-qsrt.sh` and defaults to a different checkpoint,
TP=12, 262144 ctx, gmu 0.9711. Harmless here (run-recipe supplies its own
command; do NOT pass --keep-entrypoint) but it means the image is unusable
as-shipped for stock weights.

## Depth sweep — v4 vs v3

| depth | v3 decode | v4 decode | v4 prefill t/s | v4 TTFT |
|---|---|---|---|---|
| 4K | 15.62 | 13.36 | 607.56 | 6.6 s |
| 50K | 13.64 | 11.38 | 734.77 | 68 s |
| 100K | 12.26 | 12.09 | 688.71 | 2.4 min |
| 200K | 11.14 | 11.05 | 618.01 | 5.4 min |
| 300K | 9.31 | 8.93 | 558.51 | 9.0 min |
| **460-480K** | **7.96** | **8.11** | **271.27** | **29.5 min** |

**Decode: worse shallow (-14% at 4K), parity from 100K, marginally BETTER at
480K.** The thing that hurts in practice (decay past 200K) is unchanged — v4
does not fix it, but does not make it worse either.
NB v3's column was measured at tg=1500 and v4's at tg=300, so compare the
*shape*, not the absolute values.

**Prefill: better in the 50K-300K band, then COLLAPSES at 480K** — 558 -> 271
t/s over a 1.6x depth step, when every prior step was gentle. TTFT 29.5 min vs
v3's ~15 min in the needle test. This is the one real regression.

Shallow decode (median of 5, short prompts): code 15.18 (v3 20.03, -24%),
prose 14.54 (v3 15.19, -4%), acceptance ~36%, tokens/cycle 3.9.
Prefill @75K: 1304 t/s vs v3's 1174 at channels=4 (+11%).
KV pool 1,952,620 tokens (v3 ~4.66 GiB) — memory laid out very differently.

## Why the shallow decode regressed, probably

Dropping `VLLM_DSPARK_SHARD_MARKOV_HEAD` means the Markov head is no longer
TP-sharded, so every rank computes the full vocabulary projection instead of
1/16th. v3's script comments say replicating W1 "avoids a graph-captured TP
all-reduce" — these settings were doing real work. Unverified alternatives:
the 1.95M-token KV layout, or torch 2.13 vs 2.11.

**RE-TEST ON EACH NEW IMAGE:** v4 kept both the env var and the guard, so if a
later build makes stock MXFP4 satisfy `supports_local_draft_argmax()`, flipping
`VLLM_DSPARK_SHARD_MARKOV_HEAD=1` back on should recover the shallow decode.

## Multimodal sanity check on v4 (2026-08-30) — PASS 4/4

`--limit-mm-per-prompt '{"image":64,"video":0}'` verified live in the engine's
parsed args; vision tower loaded (no `--language-model-only`).

| test | result |
|---|---|
| single 64x64 solid image | `Red` — correct |
| three images, asked in order | `red, green, blue` — **order preserved** |
| white cross on black, shape only | `White plus sign` — real geometry, not colour averaging |
| batch of 8 images, count them | `8` — correct |

prompt_tokens scaled 118 -> 159 -> 245 with image count, so each image is
genuinely encoded. The ordered test matters most: it proves images bind to their
correct sequence positions rather than being pooled, which is the failure mode
that would break UI-trace / multi-screenshot workflows.

Worth re-running after any image change — v4 is built around a `-mm-` (multimodal)
QSRT checkpoint, so its vision path is not guaranteed to behave the same on stock
MXFP4 weights just because the flag is accepted.

## Production config change 2026-08-30: 750K / seqs=2

K3 is now Roger's **hard-background-work** model (GLM-5.3 handles interactive).
Raised `--max-model-len` 500000 -> **750000** and dropped `--max-num-seqs` 4 -> **2**.

```
KV pool 2,158,618 tokens  (2.88x the 750K window)
kv_cache_max_concurrency 2.88  -> TWO full-depth 750K jobs run concurrently
shallow decode 14.82 tok/s (was 14.54 at 500K/seqs=4 — no cost)
correctness 3/3
```

The pool GREW when max_model_len rose (1.95M -> 2.16M tokens), so the wider
window was effectively free. K3 supports 1M natively (v3's own script used
`--max-model-len 1048576`), so 750K is well inside model limits.

**UNMEASURED above 480K.** Deepest real datapoint is 480K: decode 8.11 t/s,
prefill 271 t/s, TTFT 29.5 min. Extrapolating the curve suggests ~6-7 t/s decode
and >45 min cold prefill at 750K, but the prefill curve was already bending
sharply at 480K so that estimate is soft. Prefix caching hides it on repeat
visits, and for background work the amortised cost is acceptable.

## Image limit removed 2026-08-30

`--limit-mm-per-prompt` is now **absent** from the recipe. Roger's agentic image
work was erroring at 64 images in one prompt.

**vLLM's default is 999 per modality**, not 1 —
`vllm/config/multimodal.py: count: int = Field(999, ge=0)` (and
`"Unspecified modality is set to 999 by default"`). So the old
`'{"image":64,"video":0}'` was **capping** the limit; deleting the flag RAISES it
to 999. Docs: https://docs.vllm.ai/en/stable/configuration/engine_args/#-limit-mm-per-prompt

No memory cost: KV pool 2,158,618 tokens and concurrency 2.88 — **identical**
before and after. The encoder-budget concern did not materialise.

Note `video` is no longer pinned to 0 either; it now defaults to 999 as well.
Pin it back explicitly if video input should stay disabled.

## B12X PCIe all-reduce on v4 — TESTED, REVERTED (2026-08-30)

The recipe's old comment (*"GB10 has 48 SMs, B12X PCIe path needs 128 -> always
falls back"*) is **v3-era and stale**. In v4:
- `b12x.comm.pcie.pcie_allreduce`: `HIERARCHICAL_WORLD_SIZES = (12, 16)`,
  `_algorithm_for_world_size(16)` returns `"hierarchical"` — TP=16 is supported
- `_DENSE_SPARK_MAX_SMS = 64`, so GB10's 48 SMs take a **Spark-specific** policy
- no cooperative-launch / 128-SM gate in the Python layer
- the recipe already carries 8 `B12X_PCIE_HIERARCHICAL_*` tuning knobs

So it starts cleanly with `VLLM_ENABLE_PCIE_ALLREDUCE=1`. **It just doesn't help:**

| | PCIe OFF | PCIe ON |
|---|---|---|
| decode prose | 14.82 | 14.37 (-3.0%) |
| decode code | 15.18 | 13.14 (-13.4%, noisy: 12.5-16.6) |
| prefill @75K | 1304 | 1316 (+0.9%) |
| acceptance | ~36% | ~36% |

**Probable reason:** "PCIe all-reduce" is for GPUs sharing a PCIe fabric inside
one host. Sparks are separate machines over 100G RoCE — there is no PCIe path
between them to exploit. Reverted to 0.

`VLLM_USE_B12X_DCP_A2A` was left at 0 and is still untested.

**So the gap to Veghes's 25.38 tok/s remains unexplained.** Roofline (104B active,
MXFP4, TP=16) puts him at ~5 us/hop and this fleet at ~8 us/hop; collectives are
~69% of decode time, so that per-hop delta is the whole story — but PCIe
all-reduce is not the cause.

## Decode-speed levers tested 2026-08-30 — ALL FOUR NULL

Baseline: prose 14.82 / code 15.18 tok/s (v4, TP=16, eager, NCCL auto).

Roofline first, so the results make sense. K3 is **104B active** (not the 50B
often quoted), MXFP4 at ~0.53 B/param:

```
active weights   55.1 GB -> 3.44 GB/node at TP=16
weight read      12.6 ms/token  (31% of budget)
TP all-reduce    93 layers x 2 x 30 hops = 186 collectives
                 ~27.9 ms at 5 us/hop    (69% of budget)
=> ~25 tok/s at 5 us/hop, ~17.5 at 8 us/hop.  Measured 14.8.
```
**Collective latency is ~69% of decode time, and hop count sets it.**

| lever | result | why it missed |
|---|---|---|
| dual rail | 20.45 vs 20.61 (K2.7) | adds *bandwidth*; decode all-reduces are ~14 KB, latency-bound |
| `VLLM_ENABLE_PCIE_ALLREDUCE=1` | prose -3.0%, code -13.4% | starts fine on v4 (world_size 16 = "hierarchical"), but PCIe all-reduce targets GPUs sharing a PCIe fabric in ONE host; Sparks are separate machines over RoCE |
| CUDA graphs (drop `--enforce-eager`) | prose -6.2%, code -3.6% | graphs captured cleanly (FULL_AND_PIECEWISE, dspark 2/2, no downgrade — B12X_MLA's UNIFORM_BATCH holds). But graphs remove kernel-LAUNCH overhead, which lives in the non-network 31%. Contrast K2.7 at TP=8 (14 hops), where graphs were worth **2.05x** |
| `NCCL_ALGO=Tree` | **BREAKS STARTUP** | right idea (Ring 30 hops -> Tree ~8) but NCCL_ALGO is GLOBAL and Tree has no AllGather. InstantTensor's loader calls `ncclAllGather` -> `5:invalid usage`, worker dies before weights load |

**Conclusion: every remaining knob targets something other than hop count, and
TP=16 is forced by the weights.** Tree would need scoping to the all-reduce path
only, which is not exposed as a flag.

**Unresolved:** Veghes reports 25.38 tok/s single-stream — right on the 5 us/hop
line, vs this fleet's effective ~8 us. Best remaining hypothesis is that his
number comes from the **QSRT checkpoint (~3 bpw)** rather than MXFP4 (~4.5 bpw):
fewer bytes/token read puts him higher on the same curve with nothing
misconfigured here. Roger has declined QSRT on quality grounds.

## DCP 8 -> 16 (2026-09-01) — ADOPTED

Per ciprianveg, forum post ~128: *"change dcp 8 to dcp 16... at 700k the speed of
350k with a slight decrease in speed at < 50k context"* and *"almost double your
max context to cca 3M"*.

**His own v3 serve script hard-requires `DCP_SIZE=16` and exits otherwise** — the
`8` here was a divergence carried from the original adaptation, so every earlier
benchmark on this fleet ran on a config he would have refused to start.

### Capacity

| | DCP=8 | DCP=16 |
|---|---|---|
| KV pool | 2,158,618 tok | **3,517,554 tok (+63%)** |
| concurrency @750K | 2.88 | **4.69** |

Four concurrent 750K jobs instead of two, from the same pinned 5 GB
(`--kv-cache-memory-bytes 5000000000`). Matches his "cca 3M" claim.

### Speed — the gain grows with depth, exactly as the mechanism predicts

| depth | DCP=8 | DCP=16 | delta |
|---|---|---|---|
| shallow prose | 14.82 | 13.54 | **-8.6%** |
| shallow code | 15.18 | 13.74 | **-9.5%** |
| 100K | 12.09 | 12.15 | +0.5% |
| 300K | 8.93 | 9.19 | +2.9% |
| **480K** | **8.11** | **8.91** | **+9.9%** |

DCP shards KV by sequence position, so the deeper the context the more each rank
saves on KV reads. **Crossover is ~100K** — earlier than expected. K3 is the
deep-context batch model here, so the shallow regression is the right trade.
Prefill essentially unchanged (698 vs 689 t/s @100K).

## NCCL ring order: CONTIGUOUS (settles the cabling question)

`NCCL_DEBUG=INFO` on this launch shows:
```
Channel 00/04 :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
```
Ranks in order, so only **2 of 16 ring links** cross between mars-01 and mars-02.
No interleaving. Confirms the earlier estimate that re-cabling onto a single
CRS804 would buy **~2%** — not worth doing for performance.

## NCCL_ALGO is pinned to Ring fleet-wide — and cannot be overridden here

`.env` line 13 sets `CONTAINER_NCCL_ALGO=Ring` (from GLM-5.2 tuning), injected
into every container. NCCL's own init log confirms it:
`NCCL INFO NCCL_ALGO set by environment to Ring`.

So **every** collective uses Ring's 2(N-1)=30 sequential hops at TP=16, including
the ~14 KB decode all-reduces where Tree would use ~8. That is the dominant term
in the roofline (~69% of decode time).

**Three attempts to change it, all fail identically:**

| value | result |
|---|---|
| `Tree` | `ncclAllGather(...) -> 5:invalid usage` |
| `allreduce:tree` | same |
| `allreduce:tree,ring;allgather:ring;reducescatter:ring` | same |

NCCL logs `NCCL_ALGO set by environment to <value>` every time, so it reads the
string — but this custom `/opt/local-inference` NCCL 2.31.2 build does not honour
the per-collective prefix syntax. Anything non-Ring breaks InstantTensor's
`ncclAllGather` during weight load, before a single weight is read.

**Untried, and needs Roger's call:** removing `NCCL_ALGO` from `.env` entirely so
NCCL's tuner selects per message size. That is fleet-wide and would change GLM
too, so it is not an autonomous change.

## THE BIG ONE: removed `CONTAINER_NCCL_ALGO=Ring` from .env (2026-09-01)

**+24-30% decode from deleting one stale debugging line.**

`.env` line 13 set `CONTAINER_NCCL_ALGO=Ring`, injected into every container on
every model. It dates from the sparse-MLA wedge investigation, and the GLM-5.2
recipe's own comment admits what it was:

```
# sparse-MLA wedge (raw NCCL swept clean without them) but kept as
# known-good; candidates to drop when tuning.
  NCCL_PROTO: Simple
  NCCL_ALGO: Ring
  NCCL_NCHANNELS: "1"      <- a DIAGNOSTIC value; nobody tunes TO one channel
```

Defensive, never measured to help. It pinned every collective to Ring's
2(N-1)=**30 sequential hops** at TP=16, where NCCL would otherwise pick Tree
(~8 hops) for the small latency-bound decode all-reduces — ~69% of decode time
per the roofline.

| | Ring-pinned | NCCL auto |
|---|---|---|
| shallow prose | 13.54 | **16.82  (+24.2%)** |
| shallow code | 13.74 | **17.87  (+30.1%)** |
| 100K | 12.15 | **14.76  (+21.5%)** |
| prefill @75K | ~1304 | **1372** |

**The gain holds at depth**, so it is not a shallow artifact. 17.87 is the best
decode figure ever recorded on this fleet — the previous best was 15.18.

Confirms the roofline exactly: hop count was the dominant term, and every earlier
null (dual rail, PCIe all-reduce, CUDA graphs) failed because it targeted
something else. The three failed `NCCL_ALGO=Tree` override attempts were also
chasing this — they failed on syntax, but the *diagnosis* was right; the fix was
to stop forcing Ring rather than to force Tree.

**Backup:** `.env.bak-nccl-algo-20260901`. Restore the line if anything regresses.

**UNTESTED ON GLM.** This is a fleet-wide `.env` change and GLM-5.2/5.3 have not
been run since. Ring was originally pinned during the GLM sparse-MLA
investigation, so benchmark GLM before trusting it there.

## Needle test @480K with DCP=16 — 5/5 EXACT, synthesis YES

Clears the DCP16 NaN/generation-loop history (capped at 270K until
local-inference-lab/vllm PR #418). Accuracy at depth is unchanged from DCP=8.

**But cold prefill roughly doubled: retrieval TTFT 1899 s (31.7 min) vs DCP=8's
882 s.** The depth sweep had suggested only ~5% worse deep prefill, so this is
worse than expected on a real long prompt. Warm synthesis on the same context was
225 s. Matters for cold deep loads; prefix caching hides it on repeat visits.

### Full depth sweep with NCCL auto — the 480K exception

| depth | DCP=8+Ring | DCP=16+Ring | DCP=16+auto | vs Ring |
|---|---|---|---|---|
| shallow prose | 14.82 | 13.54 | **16.82** | +24% |
| shallow code | 15.18 | 13.74 | **17.87** | +30% |
| 100K | 12.09 | 12.15 | **14.76** | +21% |
| 300K | 8.93 | 9.19 | **13.52** | **+47%** |
| 480K | 8.11 | 8.91 | 8.64 | **-3%** |

**The gain grows to +47% at 300K then COLLAPSES at 480K.** Prefill at 480K also
dips (255 vs 271 t/s). No confident explanation: most likely NCCL's tuner
switches algorithm by message size and the 480K working set pushes it back toward
Ring, or into a mode that suits this shape less. Single sample per point, so some
is noise, but 13.52 -> 8.64 across one depth step is too large to dismiss.

**Practical read:** big win shallow through ~300K, roughly neutral at 480K
(8.64 vs 8.11 before any of today's changes). Nothing regressed materially.
Worth re-measuring 480K with more samples before treating the dip as structural.

## Still untested, both aimed at the collective term

1. **`--compilation-config` is absent from this recipe entirely.** Veghes's v3
   script uses
   `{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"pass_config":{"fuse_allreduce_rms":true}}`.
   **`fuse_allreduce_rms`** fuses the all-reduce with the following RMSNorm — it
   does not cut hop count but removes a kernel boundary and overlaps norm with
   communication. Note it also implies FULL_AND_PIECEWISE, which conflicts with
   `--enforce-eager` (graphs measured neutral here), though `pass_config` may
   still apply under eager.
2. **`VLLM_USE_B12X_DCP_A2A: "0"`** with a stale *"cannot initialise on 48 SMs"*
   comment from v3-era testing. The sibling claim about PCIe all-reduce was
   already proven wrong for v4. DCP all-to-all sits on the attention path, which
   matters more now that DCP=16.

## Two more levers tested 2026-09-01 — both reverted

| lever | decode | prefill | verdict |
|---|---|---|---|
| `pass_config.fuse_allreduce_rms=true` | prose 16.80 vs 16.82, code 17.44 vs 17.87 | neutral | **NEUTRAL — reverted.** Applies cleanly even under `--enforce-eager` (log shows `fuse_allreduce_rms': True`), just does not measurably help. It IS in Veghes's v3 script. |
| `VLLM_USE_B12X_DCP_A2A=1` | prose +3.7%, code -1.4% (noise) | **-11%** | **NET NEGATIVE — reverted.** It DOES start on v4; the old *"cannot initialise on 48 SMs"* comment was v3-era and stale, exactly like the PCIe one. |

## METHODOLOGY CORRECTION: the ad-hoc "prefill @75K" numbers are unreliable

Those measurements reused the SAME filler document across samples and took a
median of 2 — so one cold run got averaged with one **prefix-cache hit**. A cache
hit on that doc measures **11,627 tok/s**, which swamps everything. The figures
1304 / 1372 / 1362 / 1216 t/s are therefore contaminated and should not be used
to compare configs. A single cold run on the restored config gives **723 tok/s**.

**Unaffected:** all decode numbers (fresh generations with `ignore_eos`, no cache
effect), and the depth_sweep prefill column (unique content, one cold run per
depth). Those remain sound.

**For future prefill tests:** vary the document per sample, or restart between
runs. Do not median across a cold and a warm run.
