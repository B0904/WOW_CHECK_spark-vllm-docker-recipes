#!/bin/bash
# Kimi-K2.7-Code on spark-09..16, DUAL RAIL (rocep1s0f0 + roceP2p1s0f0).
#
# FALLBACK: if the 2nd rail is ever disturbed, drop ",roceP2p1s0f0" from
# --ib-if below. You lose NOTHING — dual rail measured 20.45 vs 20.61 tok/s
# single rail (2026-08-24), i.e. no decode benefit. Decode all-reduces one
# token (~14 KB) and are ~99% latency-bound, so link width does nothing;
# only prefill (~29 MB all-reduces) is bandwidth-bound.
#
# The 2nd rail needs 192.168.101.21-.28/24 on enP2p1s0f0np0. spark-09..16
# have it; spark-01..08 do NOT (they were link-local 169.254, cleared
# 2026-08-25 to ipv4.method=disabled), which is why there is no dual-rail
# script for the 8-ring.
BASE_PATH="$HOME/code/spark-vllm-docker"
NODES=192.168.100.21,192.168.100.22,192.168.100.23,192.168.100.24,192.168.100.25,192.168.100.26,192.168.100.27,192.168.100.28

"$BASE_PATH/run-recipe.sh" kimi-k2.7-code \
    --ib-if rocep1s0f0,roceP2p1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $NODES
