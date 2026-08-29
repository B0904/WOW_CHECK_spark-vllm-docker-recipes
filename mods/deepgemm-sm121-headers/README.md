# deepgemm-sm121-headers

Aliases DeepGEMM's `sm120_*.cuh` JIT kernel headers to `sm121_*` names so the
DSA sparse indexer can find them on DGX Spark (GB10, sm_121).

**Symptom without it** (vLLM 0.26.1rc1.dev1231+g7a9993878, 2026-08-26 build):
engine core dies during startup at `profile_cudagraph_memory` ->
`_fp8_fp4_mqa_logits_impl` with

    RuntimeError: Assertion error (include_parser.hpp:69): Failed to open:
      .../deep_gemm/include/deep_gemm/impls/sm121_fp8_mqa_logits.cuh

**Cause:** DeepGEMM vendors `sm100_*` and `sm120_*` kernels only; the compiled
`_C.so` contains no `sm121_*` symbols. vLLM builds the JIT include path from
the live device's compute capability, so 12.1 -> `sm121_`. On real sm_120
hardware this resolves; on GB10 it does not.

**Why aliasing is safe:** GB10 is `sm_121a`, a superset of sm_120 for these
kernels, and the rest of the stack already runs sm120 code on it (the
`FlashInferMLASparseSM120` attention backend, `CUTE_DSL_ARCH=sm_121a`). The
header is JIT-compiled for the real target, so this only fixes name lookup.

Affects any DSA model (GLM-5.2, GLM-5.3). Refuses to patch if no `sm120_*.cuh`
exist, which would mean the DeepGEMM layout changed.
