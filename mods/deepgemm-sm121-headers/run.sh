#!/bin/bash
# DeepGEMM ships sm120_*.cuh kernels but vLLM builds the JIT include path AND
# the called symbol name from the live device's compute capability. On GB10
# (sm_121) that yields sm121_fp8_mqa_logits.cuh / sm121_fp8_mqa_logits(), and
# DeepGEMM vendors neither, so the DSA sparse indexer dies at startup:
#
#   Failed to open: .../deep_gemm/impls/sm121_fp8_mqa_logits.cuh
#   -> after aliasing the file only: "NVCC compilation failed"
#      (header still defined sm120_fp8_mqa_logits, JIT called sm121_...)
#
# So each sm120_X.cuh is copied to sm121_X.cuh AND its entry symbol sm120_X is
# renamed to sm121_X. The sm120:: / mma::sm120 helper namespaces and
# sm120_utils.cuh includes are deliberately NOT renamed - those are real
# sm_120 helpers that exist and are shared.
#
# The kernel body is guarded by `__CUDA_ARCH__ >= 1200`, which sm_121 (1210)
# satisfies, so the real implementation compiles rather than the
# "This kernel only supports sm_120a" fallback assert.
#
# CAVEAT: sm_120a and sm_121a are architecture-SPECIFIC targets and are not
# formally forward-compatible. This works because the guard is >= 1200 and the
# MMA paths used here assemble for sm_121a. If a future DeepGEMM adds real sm_121
# kernels, drop this mod.
set -euo pipefail
PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
IMPLS="$PYTHON_ROOT/vllm/third_party/deep_gemm/include/deep_gemm/impls"
MARKER="// spark-vllm-docker: generated from sm120 header for GB10 sm_121"

[ -d "$IMPLS" ] || { echo "[deepgemm-sm121-headers] impls dir not found at $IMPLS" >&2; exit 1; }

shopt -s nullglob
SRC=("$IMPLS"/sm120_*.cuh)
[ ${#SRC[@]} -gt 0 ] || { echo "[deepgemm-sm121-headers] no sm120_*.cuh - DeepGEMM layout changed, refusing" >&2; exit 1; }

# remove only previously-generated copies, never a real vendored sm121 header
for f in "$IMPLS"/sm121_*.cuh; do
  head -1 "$f" 2>/dev/null | grep -qF "$MARKER" && rm -f "$f"
done

MADE=0; KEPT=0; RENAMED=0
for f in "${SRC[@]}"; do
  base=$(basename "$f" .cuh)          # sm120_fp8_mqa_logits
  stem="${base#sm120_}"               # fp8_mqa_logits
  dst="$IMPLS/sm121_${stem}.cuh"
  if [ -e "$dst" ]; then KEPT=$((KEPT+1)); continue; fi   # vendored sm121 exists: leave it
  { echo "$MARKER"; cat "$f"; } > "$dst"
  if grep -qE "\bsm120_${stem}\b" "$dst"; then
    sed -i "s/\bsm120_${stem}\b/sm121_${stem}/g" "$dst"
    RENAMED=$((RENAMED+1))
  fi
  MADE=$((MADE+1))
done

echo "[deepgemm-sm121-headers] generated $MADE sm121_*.cuh (entry symbol renamed in $RENAMED, $KEPT vendored kept)."
echo "=====> DeepGEMM sm121 kernel headers aliased for GB10"
