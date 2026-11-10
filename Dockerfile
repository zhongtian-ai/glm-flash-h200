# GLM-5.3-Flash on 8×H200: official SGLang nightly (pinned by digest) + our patch overlay + launcher.
# Prebuilt: ghcr.io/zhongtian-ai/glm-flash-sglang:h200-20261106   (identical to building this file)
FROM docker.io/lmsysorg/sglang@sha256:034f4c6122b50785ef939f26488a696d823048925176bafcf090b0de0ff97a5a
# sglang is an editable install under /sgl-workspace/sglang/python; only changed .py/.cuh files are copied over,
# the image's compiled extensions stay in place.
COPY overlay/python/sglang/ /sgl-workspace/sglang/python/sglang/
COPY launch.sh check.sh glm_flash_chat_template.jinja /opt/glm-flash/
LABEL org.opencontainers.image.source=https://github.com/zhongtian-ai/glm-flash-h200 \
      org.opencontainers.image.description="GLM-5.3-Flash on 8xH200 (SGLang nightly-dev-20261106-858073e4 + patches), launcher /opt/glm-flash/launch.sh" \
      org.opencontainers.image.licenses=Apache-2.0
