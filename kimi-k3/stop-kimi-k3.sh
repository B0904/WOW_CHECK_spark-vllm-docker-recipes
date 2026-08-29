#!/bin/bash
# launch-cluster.sh stop only covers CLUSTER_NODES from .env (8 nodes).
# The K3 cluster is 16, so clear containers explicitly.
for i in 11 12 13 14 15 16 17 18 21 22 23 24 25 26 27 28; do
  ( ssh -o ConnectTimeout=8 -o BatchMode=yes 192.168.100.$i \
      'docker rm -f $(docker ps -aq --filter name=vllm) 2>/dev/null' >/dev/null 2>&1 ) &
done
wait
left=$(for i in 11 12 13 14 15 16 17 18 21 22 23 24 25 26 27 28; do
  ssh -o ConnectTimeout=5 -o BatchMode=yes 192.168.100.$i 'docker ps -q --filter name=vllm' 2>/dev/null
done | grep -c .)
echo "containers remaining: $left"
