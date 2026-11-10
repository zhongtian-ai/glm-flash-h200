#!/usr/bin/env bash
# launch.sh — start GLM-5.3-Flash on one 8-GPU H200 node with a measured recipe.
#   TIER=f1  (default) <=64 concurrent sessions, best latency      (measured 2,183 out tok/s @64 agent sessions)
#   TIER=f2            ~128 concurrent long sessions, FP8 KV cache  (measured 2,113 out tok/s @128, TTFT p50 2.5 s)
# Env: MODEL=/model  PORT=30000  TP=8  SERVED_NAME=zai-org/GLM-5.3-Flash  HICACHE=1 (0 = no host-RAM KV tier)
# Anything after the script name is appended to the sglang args (later flags win).
set -euo pipefail
TIER=${TIER:-f1}; MODEL=${MODEL:-/model}; PORT=${PORT:-30000}; TP=${TP:-8}
SERVED_NAME=${SERVED_NAME:-zai-org/GLM-5.3-Flash}; HICACHE=${HICACHE:-1}
export SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=${SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION:-1}
COMMON=(--ep "$TP" --moe-runner-backend deep_gemm --dsa-decode-backend tilelang --trust-remote-code
  --mem-fraction-static 0.80          # 0.84 (default) ran out of memory mid-run on one of our nodes
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --enable-cache-report --reasoning-parser glm45 --tool-call-parser glm47
  --chat-template /opt/glm-flash/glm_flash_chat_template.jinja)
HIC=(--enable-hierarchical-cache --hicache-ratio 2 --hicache-write-policy write_through
  --hicache-mem-layout page_first_direct --hicache-io-backend direct)   # needs ~1.2 TB free host RAM
case "$TIER" in
  f1) T=(--dsa-prefill-backend flashmla_sparse_q8 --kv-cache-dtype bfloat16) ;;
  f2) T=(--dsa-prefill-backend tilelang --kv-cache-dtype fp8_e4m3 --max-running-requests 96 --cuda-graph-max-bs-decode 96) ;;
  *) echo "unknown TIER=$TIER (f1|f2)" >&2; exit 2 ;;
esac
[ "$HICACHE" = 1 ] || HIC=()
set -x
exec python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" "${COMMON[@]}" "${T[@]}" "${HIC[@]}" "$@"
