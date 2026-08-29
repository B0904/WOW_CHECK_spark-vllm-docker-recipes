#!/bin/bash
# Post-rebuild helper: decide whether mods/glm-indexer-mtp-overhang is still needed.
#
# vLLM PR #50302 (issue #46074) fixed the DSA sparse indexer off-by-one that this
# mod patches, and refactored expanded_block_table_buffer so the mod's anchor
# ("get_kv_cache_shard_count(),") no longer exists in mla/indexer.py.
#
# On a post-#50302 build the mod's run.sh exits 1 and TAKES THE LAUNCH WITH IT.
# This script checks the built image and comments the mod out of the GLM recipes.
#
# Usage:  ./check-indexer-mod.sh            # dry run, just report
#         ./check-indexer-mod.sh --apply    # make the edits
set -uo pipefail

IMAGE="${IMAGE:-vllm-node:latest}"
APPLY=false
[ "${1:-}" = "--apply" ] && APPLY=true

A="$HOME/code/spark-vllm-docker"
B="$HOME/code/spark-vllm-docker-recipes"
RECIPES=(
  "$A/recipes/glm-5.2-nvfp4.yaml"
  "$A/recipes/glm-5.3-nvfp4.yaml"
  "$B/recipes/glm-5.2-nvfp4.yaml"
  "$B/recipes/glm-5.3-nvfp4.yaml"
)
TARGET=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py

echo "=== checking $IMAGE ==="
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  ERROR: image $IMAGE not found. Build first." >&2; exit 1
fi
VER=$(docker run --rm --entrypoint bash "$IMAGE" -lc 'python3 -c "import vllm;print(vllm.__version__)"' 2>/dev/null)
N=$(docker run --rm --entrypoint bash "$IMAGE" -lc "grep -c 'get_kv_cache_shard_count(),' $TARGET 2>/dev/null; true" 2>/dev/null | tr -d '[:space:]')
echo "  vllm version : ${VER:-unknown}"
echo "  anchor count : ${N:-?}   (0 = PR #50302 landed, mod is obsolete)"
echo

if [ "${N:-1}" != "0" ]; then
  echo "  VERDICT: anchor still present -> build PREDATES #50302."
  echo "           KEEP mods/glm-indexer-mtp-overhang. No changes needed."
  exit 0
fi

echo "  VERDICT: anchor gone -> #50302 is in. The mod would exit 1 and abort the launch."
echo "           mods/glm-indexer-mtp-overhang should be commented out."
echo

CHANGED=0
for R in "${RECIPES[@]}"; do
  [ -f "$R" ] || { printf "  %-60s (missing)\n" "$R"; continue; }
  if ! grep -qE '^\s*-\s*mods/glm-indexer-mtp-overhang\s*$' "$R"; then
    printf "  %-60s already done / not present\n" "${R/#$HOME/\~}"; continue
  fi
  if [ "$APPLY" = true ]; then
    cp "$R" "$R.bak-indexermod"
    sed -i 's|^\(\s*\)-\s*mods/glm-indexer-mtp-overhang\s*$|\1# - mods/glm-indexer-mtp-overhang   # OBSOLETE: fixed upstream by vLLM PR #50302 (issue #46074); anchor removed, mod would exit 1|' "$R"
    if python3 -c "import yaml,sys;yaml.safe_load(open('$R'))" 2>/dev/null; then
      printf "  %-60s COMMENTED OUT (backup .bak-indexermod)\n" "${R/#$HOME/\~}"; CHANGED=$((CHANGED+1))
    else
      cp "$R.bak-indexermod" "$R"
      printf "  %-60s YAML BROKE — reverted\n" "${R/#$HOME/\~}"
    fi
  else
    printf "  %-60s WOULD comment out\n" "${R/#$HOME/\~}"
  fi
done

echo
if [ "$APPLY" = true ]; then
  echo "  $CHANGED recipe(s) edited. Remaining mods: glm-5.2, sparse-mla-chunked, glm-topk-per-row (all still required)."
else
  echo "  Dry run. Re-run with --apply to make these edits."
fi
