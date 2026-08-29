# Kimi-K3 on 16x DGX Spark (GB10)

Full [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (1.56 TB, mxfp4) served across all
16 GB10 nodes with TP16 + DCP8 + RedHat DSpark speculative decoding, using
[ciprianveg/gb10-vllm](https://github.com/ciprianveg/gb10-vllm)'s `v3-sm121` image.

    ./launch-kimi-k3-v3.sh      # start (16 nodes)
    ./stop-kimi-k3.sh           # stop

Serves as **`KIMI-K3`** on **`localhost:8000`**, `max_model_len 500000`, up to **16 images** per prompt.

## Prerequisites

| | |
|---|---|
| Image | `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121` (**47 GB**) — pull on all 16 nodes |
| Weights | `moonshotai/Kimi-K3` — 1.56 TB, **local HF cache on every node** (caches are not shared) |
| Drafter | `RedHatAI/Kimi-K3-speculator.dspark` — 8.9 GB, also on every node (`draft_tensor_parallel_size: 1`, so it is not sharded) |

`docker pull` is cancelled if the ssh client disconnects — run it detached
(`setsid nohup docker pull ... &`) or it will silently abort.

## Two fixes needed to run Veghes's recipe on eugr's scripts

**1. Do NOT pass `--keep-entrypoint`.** The image entrypoint `/usr/local/bin/serve-kimi-k3-dspark`
execs a *hardcoded* serve script and exits. `launch-cluster.sh` needs a container that idles so it
can `docker exec` the recipe command into it. With the entrypoint kept, the head container dies
instantly and you get `No such container: vllm_node` -> `Failed to create /workspace`.
Veghes's `raw_entrypoint: true` is a key **his own patched scripts** understand — it is not in
eugr upstream (verified at `origin/HEAD`).

**2. Bypassing that entrypoint skips 38 env vars it would have set.** They are baked into the
recipe here (85 env vars total). Without them the B12X/DSpark paths silently misconfigure — the
likely cause of the garbled output other users have reported. The critical one is
`PYTHONPATH=/opt/kimi-k3/vllm:/opt/kimi-k3/b12x`; also every `VLLM_USE_B12X_MOE` / `B12X_*` /
`VLLM_KIMI_*` / `VLLM_DSPARK_*`, plus `INSTANTTENSOR_BACKEND=AIO` and
`VLLM_WORKER_MULTIPROC_METHOD=spawn`.

To re-extract them from a newer image:

    cid=$(docker create <image>)
    docker cp "$cid":/opt/kimi-k3/vllm/serve-kimi-k3-full-mxfp4-dcp16-1m-dspark.sh -

## Tuning applied (differs from upstream)

| setting | upstream | here | why |
|---|---|---|---|
| `num_speculative_tokens` | 6 | **8** | **+14% code / +16% prose.** Verify pass (97.6 GB/rank) dominates, drafter (8.9 GB) is cheap, so deep drafts nearly free. Plateaus at 8. |
| `NCCL_M{AX,IN}_NCHANNELS` | 2 | **2** | Kept. 4 gives **+50% prefill** but **-3..-9% decode** — flip only if cold long-context TTFT matters more than decode. |
| `VLLM_ENABLE_PCIE_ALLREDUCE`, `VLLM_USE_B12X_DCP_A2A` | 1 | **0** | GB10 exposes **48 SMs**; the B12X PCIe path needs 128, so it always falls back to NCCL. Throughput-neutral; set to 0 only to stop constant warning spam. |
| `--limit-mm-per-prompt` | `image:5` | **`image:16`** | Costs no memory (encoder cache is lazy: 111.78 GiB free vs 111.62 at 5). |
| port | 5002 | **8000** | Matches the rest of this fleet. |
| paths | `/root/models/...` (NFS) | local HF cache | `shared_weights_nfs` is not supported by eugr's scripts. |

`recipes/kimi-k3-dcp-tp16-prd.yaml.untuned` is the pre-tuning version (path fixes only).

## Gotchas

- **Verify the running config before trusting any measurement.** A relaunch that collides with a
  live server fails with `Address already in use` and the *old* engine keeps serving — an A/B can
  silently compare a config against itself. Check
  `grep "'num_speculative_tokens':" <launchlog>` and count the
  `spec_decode_num_accepted_tokens_per_pos_total` series.
- **`launch-cluster.sh stop` only covers `CLUSTER_NODES` from `.env`** (8 nodes). Use `./stop-kimi-k3.sh`.
- **Do not tune on acceptance *rate*** — it is depth-dependent and misleading (spec=3 shows 83%
  while being slower than spec=8 at 54%). Optimise `accepted/draft + 1` = tokens per verify cycle.
- **K3 streams reasoning in a delta field named `reasoning`**, not `reasoning_content`. Parsers
  that only check the latter silently see zero tokens.
- **Not bitwise deterministic at `temperature=0`** — normal for TP inference (reduction order,
  batch composition). Validate with correctness, not output identity.
- Expect a slow first ~10-20 requests while CuTe DSL / Triton JIT-compile per batch shape.

See `results/benchmarks.md` for measured numbers.
