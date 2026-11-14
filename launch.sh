#!/usr/bin/env bash
# launch.sh — start GLM-5.3-Flash on one 8-GPU H200 node with a measured recipe.
#   TIER=f1  (default) <=64 concurrent long agent sessions   (2,500 out tok/s @64 sessions; 2,224 with MOE=split)
#   TIER=f2            ~128 concurrent long sessions, FP8 KV  (2,480 out tok/s @128 sessions; 2,087 with MOE=split)
#   TIER=p1            a few concurrent requests, ~5K-token prompts, short answers: f1 without the host-RAM KV tier
#   TIER=a1            coding agents, up to 64 concurrent sessions (19K-token median prompts, mostly cached):
#                      p1 + running cap 64 + 16K prefill chunks  (3,355 out tok/s @64, first token p50 0.17 s)
# Env: MODEL=/model  PORT=30000  TP=8  SERVED_NAME=zai-org/GLM-5.3-Flash
#      MOE=mega|split  mega (default since 2026-11-10): fused MoE kernel for the main model, draft head on the light
#                      Triton path, block verification. split = the 2026-11-06 behaviour (separate dispatch / GEMM / combine).
#      HICACHE=1|0     host-RAM KV tier (needs ~1.2 TB free RAM). Default: 1 for f1/f2, 0 for p1.
# Anything after the script name is appended to the sglang args (later flags win).
set -euo pipefail
TIER=${TIER:-f1}; MODEL=${MODEL:-/model}; PORT=${PORT:-30000}; TP=${TP:-8}
SERVED_NAME=${SERVED_NAME:-zai-org/GLM-5.3-Flash}; HICACHE=${HICACHE:-auto}; MOE=${MOE:-mega}
export SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=${SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION:-1}
COMMON=(--ep "$TP" --moe-runner-backend deep_gemm --dsa-decode-backend tilelang --trust-remote-code
  --mem-fraction-static 0.80          # 0.84 (default) ran out of memory mid-run on one of our nodes
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --enable-cache-report --reasoning-parser glm45 --tool-call-parser glm47
  --chat-template /opt/glm-flash/glm_flash_chat_template.jinja)
HIC=(--enable-hierarchical-cache --hicache-ratio 2 --hicache-write-policy write_through
  --hicache-mem-layout page_first_direct --hicache-io-backend direct)   # needs ~1.2 TB free host RAM
# MegaMoE = dispatch + GEMM + combine across the 8 GPUs in one kernel (faster whenever prompts are being read); the MTP
# draft head stays on the lighter Triton path; block verification accepts slightly longer drafts under sampling.
case "$MOE" in
  mega)  M=(--moe-a2a-backend megamoe --speculative-moe-a2a-backend none --speculative-moe-runner-backend triton
            --speculative-use-block-verification) ;;
  split) M=() ;;
  *) echo "unknown MOE=$MOE (mega|split)" >&2; exit 2 ;;
esac
case "$TIER" in
  f1) T=(--dsa-prefill-backend flashmla_sparse_q8 --kv-cache-dtype bfloat16) ;;
  f2) T=(--dsa-prefill-backend tilelang --kv-cache-dtype fp8_e4m3 --max-running-requests 96 --cuda-graph-max-bs-decode 96) ;;
  p1) T=(--dsa-prefill-backend flashmla_sparse_q8 --kv-cache-dtype bfloat16); [ "$HICACHE" = auto ] && HICACHE=0 ;;
  # a1: with MTP on, SGLang caps running requests at 48 by itself, so agent #49..64 wait in the queue (TTFT 1.7 s at 64).
  #     Raising the cap to 64 (and capturing decode CUDA graphs up to 64) removes that wait; KV fits in GPU memory, so no
  #     host-RAM tier (it cost 4-5% here).
  #     Prefill chunks stay at 8192 tokens with MOE=mega: MegaMoE silently returns wrong results when one forward pass
  #     carries more than 8192 tokens (prompts >=10K tokens answered garbage at 16384; fixed 2026-11-12, see README).
  a1) T=(--dsa-prefill-backend flashmla_sparse_q8 --kv-cache-dtype bfloat16 --max-running-requests 64
         --cuda-graph-max-bs-decode 64 --chunked-prefill-size 8192); [ "$HICACHE" = auto ] && HICACHE=0 ;;
  *) echo "unknown TIER=$TIER (f1|f2|p1|a1)" >&2; exit 2 ;;
esac
# Guard for extra args: MegaMoE + prefill chunks above 8192 tokens gives wrong answers on long prompts.
if [ "$MOE" = mega ]; then
  for ((i=1; i<=$#; i++)); do
    if [ "${!i}" = --chunked-prefill-size ]; then j=$((i+1))
      [ "${!j:-0}" -gt 8192 ] && { echo "refusing --chunked-prefill-size ${!j} with MOE=mega (wrong outputs above 8192); use <=8192 or MOE=split" >&2; exit 2; }
    fi
  done
fi
[ "$HICACHE" = auto ] && HICACHE=1
[ "$HICACHE" = 1 ] || HIC=()
set -x
exec python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" "${COMMON[@]}" "${T[@]}" "${M[@]}" "${HIC[@]}" "$@"
