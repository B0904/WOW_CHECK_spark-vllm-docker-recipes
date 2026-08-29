#!/bin/bash
# Check every GLM mod against a built image and report NEEDED / OBSOLETE.
# Usage: ./check-mods.sh [image]   (default vllm-node:latest)
IMAGE="${1:-vllm-node:latest}"
P=/usr/local/lib/python3.12/dist-packages

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found"; exit 1; }
echo "=== $IMAGE ==="
docker run --rm --entrypoint bash "$IMAGE" -lc "
  printf '  vllm       : %s\n' \"\$(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null)\"
  printf '  flashinfer : %s\n' \"\$(python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null)\"
"
echo
docker run --rm --entrypoint bash "$IMAGE" -lc "
P=$P
chk() { printf '  %-28s %s\n' \"\$1\" \"\$2\"; }

# 1. glm-5.2 : Triton MLA num_stages threshold for sm_121 shared memory
F=\$P/vllm/v1/attention/ops/triton_decode_attention.py
if   ! [ -f \"\$F\" ];                          then chk glm-5.2 'ANCHOR GONE - file moved, re-verify'
elif grep -q 'BLOCK_DMODEL >= 512' \"\$F\";     then chk glm-5.2 'OBSOLETE - upstream already lowered it'
elif ! grep -q 'Lk == 576' \"\$F\";             then chk glm-5.2 'ANCHOR GONE - MLA tile alignment missing'
elif grep -q 'BLOCK_DMODEL >= 1024' \"\$F\";    then chk glm-5.2 'NEEDED - still >= 1024'
else chk glm-5.2 'UNKNOWN - inspect manually'; fi

# 2. glm-topk-per-row : persistent_topk CTA budget vs GB10's 48 SMs
F=\$P/vllm/model_executor/layers/sparse_attn_indexer.py
if   ! [ -f \"\$F\" ];                                        then chk glm-topk-per-row 'ANCHOR GONE - file moved'
elif grep -qE 'multi_processor_count|sm_count|cta' \"\$F\" && ! grep -q 'use_persistent_topk = current_platform.is_cuda() and topk_tokens in' \"\$F\"; then chk glm-topk-per-row 'MAYBE OBSOLETE - SM-aware check appeared'
elif grep -q 'use_persistent_topk = current_platform.is_cuda() and topk_tokens in' \"\$F\"; then chk glm-topk-per-row 'NEEDED - still no SM/CTA check'
else chk glm-topk-per-row 'UNKNOWN - inspect manually'; fi

# 3. glm-indexer-mtp-overhang : fixed by vLLM PR #50302
F=\$P/vllm/v1/attention/backends/mla/indexer.py
if   ! [ -f \"\$F\" ];                                   then chk glm-indexer-mtp-overhang 'ANCHOR GONE - file moved'
elif grep -q 'get_kv_cache_shard_count(),' \"\$F\";      then chk glm-indexer-mtp-overhang 'NEEDED - predates PR #50302'
else chk glm-indexer-mtp-overhang 'OBSOLETE - PR #50302 landed, REMOVE (mod exits 1)'; fi

# 4. sparse-mla-chunked : FlashInfer sm120 >64-token path. PR #4732 may fix.
F=\$P/flashinfer/mla/_sparse_mla_sm120.py
B=\$P/flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/arch/barrier.cuh
if   ! [ -f \"\$F\" ];                                   then chk sparse-mla-chunked 'ANCHOR GONE - file moved'
elif ! grep -q '_decode_dsv3_2_dispatchable' \"\$F\";    then chk sparse-mla-chunked 'ANCHOR GONE - dispatch changed, re-verify'
elif [ -f \"\$B\" ] && grep -qiE 'alternat|phase|two arrivals' \"\$B\"; then chk sparse-mla-chunked 'MAYBE OBSOLETE - PR #4732 barrier fix appears present; TEST before dropping'
else chk sparse-mla-chunked 'NEEDED - dispatch intact, no #4732 barrier fix seen'; fi

# 5. deepgemm-sm121-headers : DeepGEMM ships sm120_* only
D=\$P/vllm/third_party/deep_gemm/include/deep_gemm/impls
if   ! [ -d \"\$D\" ];                                   then chk deepgemm-sm121-headers 'N/A - deep_gemm not vendored'
elif ls \$D/sm121_*.cuh >/dev/null 2>&1;                then chk deepgemm-sm121-headers 'OBSOLETE - sm121_* now shipped'
elif ls \$D/sm120_*.cuh >/dev/null 2>&1;                then chk deepgemm-sm121-headers 'NEEDED - only sm120_* present'
else chk deepgemm-sm121-headers 'UNKNOWN - no sm12x headers at all'; fi
"
echo
echo "  NEEDED   = keep in recipe.  OBSOLETE = remove.  MAYBE = test before dropping."
