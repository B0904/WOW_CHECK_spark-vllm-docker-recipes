#!/bin/bash
# Kimi-K3 v4 (ciprianveg image) on 16x GB10 — TP16 + DCP8 + RedHat DSpark.
# Tuned: num_speculative_tokens=8, NCCL channels=4 (prefill-favoured), B12X PCIe paths off, 16 images.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$HOME/code/spark-vllm-docker"
RECIPE="$HERE/recipes/kimi-k3-v4.yaml"

NODES=192.168.100.11,192.168.100.12,192.168.100.13,192.168.100.14,\
192.168.100.15,192.168.100.16,192.168.100.17,192.168.100.18,\
192.168.100.21,192.168.100.22,192.168.100.23,192.168.100.24,\
192.168.100.25,192.168.100.26,192.168.100.27,192.168.100.28

# NOTE: do NOT add --keep-entrypoint. See README ("Two fixes").
exec "$BASE_PATH/run-recipe.sh" "$RECIPE" \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $NODES
