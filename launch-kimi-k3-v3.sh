#!/bin/bash
# Kimi-K3 v3 (ciprianveg image) — TP16 + DCP8 + RedHat DSpark, all 16 GB10s.
BASE_PATH="$HOME/code/spark-vllm-docker"

NODES=192.168.100.11,192.168.100.12,192.168.100.13,192.168.100.14,\
192.168.100.15,192.168.100.16,192.168.100.17,192.168.100.18,\
192.168.100.21,192.168.100.22,192.168.100.23,192.168.100.24,\
192.168.100.25,192.168.100.26,192.168.100.27,192.168.100.28

"$BASE_PATH/run-recipe.sh" kimi-k3-dcp-tp16-prd \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $NODES
