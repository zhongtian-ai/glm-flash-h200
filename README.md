# GLM-5.3-Flash on one 8×H200 node — reproducible image

Prebuilt image (public):

```bash
docker pull ghcr.io/zhongtian-ai/glm-flash-sglang:h200-20261106
```

It is the official SGLang nightly image `lmsysorg/sglang:nightly-dev-20261106-858073e4`
(pinned by digest `sha256:034f4c61…`) plus three things: a small patch overlay (35 source files), the
production chat template, and a launcher with two measured recipes. The [`Dockerfile`](Dockerfile) rebuilds the
same image from public parts. Weights are **not** in the image: download `zai-org/GLM-5.3-Flash` (FP8, ~328 GB).

## Quick start

```bash
# 1. weights on local NVMe
huggingface-cli download zai-org/GLM-5.3-Flash --local-dir /data/GLM-5.3-Flash

# 2. serve (TIER=f1 default; see below)
docker run -d --name glm --gpus all --network host --ipc host --ulimit memlock=-1 \
  -v /data/GLM-5.3-Flash:/model:ro -v /data/sglang-cache:/root/.cache \
  -e TIER=f1 ghcr.io/zhongtian-ai/glm-flash-sglang:h200-20261106 /opt/glm-flash/launch.sh

# 3. wait for "The server is fired up" (first start ~10–15 min: DeepGEMM JIT + CUDA graphs;
#    restarts ~3–6 min when /root/.cache is kept), then
docker exec glm /opt/glm-flash/check.sh       # health, one answer, 1- and 16-stream speed
```

OpenAI-compatible API on port 30000 (`/v1/chat/completions`), model name `zai-org/GLM-5.3-Flash`.

## Which recipe

| `TIER` | for | what it changes | measured on 8×H200 (agent benchmark) |
|---|---|---|---|
| `f1` (default) | up to ~64 concurrent sessions | MTP speculative decoding 3/1/4, FP8 sparse prefill, host-RAM KV tier (HiCache ×2), mem fraction 0.80, runtime optimizations on | 2,183 output tok/s at 64 sessions; warm TTFT p50 0.87 s |
| `f2` | ~128 concurrent long sessions | f1 + FP8 KV cache (2× capacity) + running cap 96 | 2,113 output tok/s at 128 sessions; warm TTFT p50 2.5 s |

Short chat (ShareGPT, ~2K in / 1K out, same MTP setting): 271 tok/s per stream at 1 user, 202 at 4, 134 at 16.
Full GSM8K (1,319 questions, thinking off): 0.961–0.964 for every recipe, same as without our changes.

Options: `HICACHE=0` drops the host-RAM KV tier (use it if the host has < ~1.3 TB free RAM, or for low
concurrency with short context where the GPU KV is enough — not yet benchmarked by us). `PORT`, `TP`, `MODEL`,
`SERVED_NAME` are env vars; any extra arguments after `launch.sh` go straight to `sglang.launch_server`.

## Requirements

- 8× H200 141 GB with NVLink (H100 80 GB fits the weights but leaves little KV room; recipes not tuned for it).
  **Not for Blackwell (B200 / RTX 6000-class)**: `flashmla_sparse_q8` is Hopper-only; Blackwell needs an NVFP4 recipe.
- NVIDIA driver supporting CUDA 13.0; Docker or Podman with the NVIDIA container toolkit.
- Host RAM ≥ 1.5 TB for `f1`/`f2` with HiCache (it pins ~1.17 TB); ~500 GB NVMe for weights + cache.

## What the patch overlay contains

[`overlay/`](overlay) holds only the files that differ from the base image (the image's compiled extensions are untouched):

1. **Our fix** ([`recipes/0001-…patch`](recipes)): two `torch.repeat_interleave` calls in `dsa_backend.py` get
   `output_size`, removing a GPU→CPU sync on every prefill step. +6% on our agent benchmark (3 repeats).
2. Ports of upstream SGLang PRs for hybrid-model cache handling (#48667, #47339, #43484, #42443) and kernels
   (#44612, #48743, #46784, #47893, #45274). Individually within noise on our load; kept because the fix above sits on them.

Code is from SGLang (Apache-2.0) with the changes above.

## Known pitfalls

- Don't raise `--mem-fraction-static` back to 0.84: one of our nodes ran out of memory mid-run and dropped to 1/7 speed.
- Don't bind-mount a sglang source tree over `/sgl-workspace/sglang/python/sglang` — it hides the compiled Rust extension.
- Running cap above 96 was slower in every test (sessions evict each other's cache).
- Speed numbers depend on the workload; please measure with your own traffic and send us the result.
