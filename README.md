# Spark vLLM Docker Recipes

Recipes, vLLM patches ("mods") and launch scripts for serving very large LLMs with
[vLLM](https://github.com/vllm-project/vllm) on **clusters of NVIDIA DGX Spark (GB10, sm_121) nodes**.
The reference setup is 8 Sparks with tensor parallelism 8 (Kimi-K3 uses 16 nodes).

This repo is an **overlay** for eugr's [`spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker).
It does not contain `run-recipe.sh`, `launch-cluster.sh`, `hf-download.sh` or the Dockerfile; those come
from that project, which you must check out separately (the scripts here assume `~/code/spark-vllm-docker`).

## Layout

| Path | Purpose |
|---|---|
| `recipes/*.yaml` | Model recipes: container, mods, defaults, env vars and the `vllm serve` command (GLM-5.2/5.3, DeepSeek-V4-Flash, Kimi-K2.7-code, MiniMax-M3, Qwen3.5/3.8, Step-3.7-flash, Gemma-4). |
| `mods/` | Patch scripts (`run.sh` + README) applied inside the container to work around sm_121 / vLLM issues. |
| `launch-*.sh` | Thin wrappers that call upstream `run-recipe.sh <recipe> --ib-if … --eth-if … -n <hosts>`. |
| `download-model.sh` | Downloads a Hugging Face model and copies it to all nodes via upstream `hf-download.sh`. |
| `stop-cluster.sh` | Stops the serving cluster on all nodes (asks for confirmation). |
| `check-glm-mods.sh`, `check-glm-indexer-mod.sh` | Check a built image and report whether each GLM mod is still NEEDED or OBSOLETE. |
| `management/netplan.sh` | Writes the netplan config (static fabric IP, MTU 9000) on a node: `./netplan.sh <ip-suffix>`. |
| `kimi-k3/` | Self-contained Kimi-K3 on 16 Sparks: README, recipes, launchers, benchmarks. |
| `documentation/cluster-8-spark-setup.md` | Hardware, switch, network and NCCL setup guide for an 8-Spark cluster. |

## Typical workflow

1. Cable and configure the cluster (switch, MTU 9000, static IPs `192.168.100.11-.18`, NCCL interfaces) — see
   [`documentation/cluster-8-spark-setup.md`](documentation/cluster-8-spark-setup.md).
2. Check out `spark-vllm-docker` and build/copy the `vllm-node` image to every node.
3. Copy `mods/` and `recipes/` into the `spark-vllm-docker` checkout.
4. Download weights to all nodes: `./download-model.sh nvidia/GLM-5.2-NVFP4`
5. Launch: `./launch-glm-5.2-nvfp4.sh`
6. Stop: `./stop-cluster.sh`

## How a recipe works

A recipe YAML (e.g. [`recipes/glm-5.2-nvfp4.yaml`](recipes/glm-5.2-nvfp4.yaml)) declares the model, container,
`mods:` to apply, `defaults:` (TP size, memory utilization, context length…), `env:` and a `command:` template.
Mods edit the installed vLLM with exact-string replacements and refuse to patch if the code has changed.
The recipe comments document the root causes, workarounds and benchmark results.

## Caveats

- Scripts hard-code `$HOME/code/spark-vllm-docker`, host IPs `192.168.100.11-.18` and interface names
  (`rocep1s0f0`, `enp1s0f0np0`); edit them for your cluster.
- Some launchers (e.g. `launch-deepseek-v4-flash.sh`, `launch-step-3.7-flash-fp8.sh`) reference recipes that are
  not in this repo, and the root `launch-kimi-k3-v3.sh` differs from `kimi-k3/launch-kimi-k3-v3.sh`.
- Mods target specific vLLM/FlashInfer builds and may become obsolete; use the `check-glm-*` scripts.
- Env vars needed on every Ray rank must also be set in `.env` as `CONTAINER_<VAR>` (see the GLM-5.2 recipe).
- No license file is included.

## Credits

Built on eugr's `spark-vllm-docker`; Kimi-K3 uses [ciprianveg/gb10-vllm](https://github.com/ciprianveg/gb10-vllm).
