# GLM-5.3-Flash on one 8×H200 node — reproducible image

Prebuilt image (public):

```bash
docker pull ghcr.io/zhongtian-ai/glm-flash-sglang:h200-20261110
```

Or build the identical image yourself from public parts only (no login needed):

```bash
docker build -t ghcr.io/zhongtian-ai/glm-flash-sglang:h200-20261110 https://github.com/zhongtian-ai/glm-flash-h200.git#main
```

The image is the official SGLang nightly image `lmsysorg/sglang:nightly-dev-20261106-858073e4`
(pinned by digest `sha256:034f4c61…`) plus three things: a small patch overlay (35 source files), the
production chat template, and a launcher with three measured recipes. The [`Dockerfile`](Dockerfile) rebuilds the
same image from public parts. Weights are **not** in the image: download `zai-org/GLM-5.3-Flash` (FP8, ~328 GB).

## Quick start

```bash
# 1. weights on local NVMe
huggingface-cli download zai-org/GLM-5.3-Flash --local-dir /data/GLM-5.3-Flash

# 2. serve (TIER=f1 default; f2 for ~128 long sessions, p1 for a handful of medium requests; see below)
docker run -d --name glm --gpus all --network host --ipc host --ulimit memlock=-1 \
  -v /data/GLM-5.3-Flash:/model:ro -v /data/sglang-cache:/root/.cache \
  -e TIER=f1 ghcr.io/zhongtian-ai/glm-flash-sglang:h200-20261110 /opt/glm-flash/launch.sh

# 3. wait for "The server is fired up" (first start ~10–15 min: DeepGEMM JIT + CUDA graphs;
#    restarts ~3–6 min when /root/.cache is kept), then
docker exec glm /opt/glm-flash/check.sh       # health, one answer, 1- and 16-stream speed
```

OpenAI-compatible API on port 30000 (`/v1/chat/completions`), model name `zai-org/GLM-5.3-Flash`.

## Which recipe

| `TIER` | for | what it sets | measured on 8×H200 |
|---|---|---|---|
| `f1` (default) | up to ~64 concurrent long sessions | MTP speculative decoding 3/1/4, FP8 sparse prefill, host-RAM KV tier (HiCache ×2), mem fraction 0.80, runtime optimizations on | agent benchmark, 64 sessions: 2,500 output tok/s, warm first token p50 0.75 s (2,224 tok/s with `MOE=split`) |
| `f2` | ~128 concurrent long sessions | f1 + FP8 KV cache (2× capacity) + running cap 96 | agent benchmark, 128 sessions: 2,480–2,499 output tok/s, warm first token p50 2.3 s (2,086 tok/s with `MOE=split`) |
| `p1` | a handful of concurrent requests, prompts of a few thousand tokens | f1 without the host-RAM KV tier (no 1.2 TB RAM requirement) | 5 in flight, 5.4K in / 160 out: 605–846 output tok/s depending on cache hits; see the next section |
| `a1` | **coding agents**, up to 64 concurrent sessions with long, mostly cached prompts | p1 + running cap 64 (SGLang's automatic cap is 48 with MTP on); prefill chunks 8192 (see the 2026-11-12 fix below) | [coding-agent-serving-bench](https://github.com/zhongtian-ai/coding-agent-serving-bench), 16 / 32 / 64 agents: 1,779 / 2,563 / 3,390 output tok/s, first token p50 126 / 141 / 172 ms (p1: 1,780 / 2,532 / 2,946 tok/s, 1,676 ms at 64) |

**Added 2026-11-11 (`TIER=a1`):** tuned on the coding-agent workload of
[coding-agent-serving-bench](https://github.com/zhongtian-ai/coding-agent-serving-bench) (19K-token median prompt, 93%
served from the prefix cache, closed-loop agents). Of eight configurations measured, the one large change is the
running-request cap: with speculative decoding on, SGLang limits the server to 48 running requests, so at 64 agents a
quarter of them wait in the queue (first token 1.7 s). `a1` raises the cap to 64: at 64 agents +15% output tokens/s,
first token p50 1.7 s → 0.17 s, end-to-end p50 5.1 → 3.9 s (per-token time 15.3 → 17.4 ms, since 64 instead of 48
requests now decode together); unchanged at 16 and 32. Did not help here (vs cap 64 alone): the host-RAM KV tier
(−4–5%, the KV fits in GPU memory), FP8 KV + cap 96 (`f2`, −5–10%), 2 speculative steps instead of 3 (−5–8%), 4 steps
(within noise). All runs with per-request data:
[results/2026-11-11-h200-tuning](https://github.com/zhongtian-ai/coding-agent-serving-bench/tree/main/results/2026-11-11-h200-tuning).
Launcher-only change: on an older image the same thing is `TIER=p1` plus
`--max-running-requests 64 --cuda-graph-max-bs-decode 64 --chunked-prefill-size 8192` after `launch.sh`.

**Fixed 2026-11-12 (`TIER=a1` prefill chunks 16384 → 8192):** with `MOE=mega`, any forward pass carrying more than 8192
tokens returns wrong outputs, silently. Every prompt of ~10K tokens or more that was not served from the cache came back
wrong: a retrieval question over a 10K–56K-token document got the document echoed, `0`, or `0.0.0.0…`, while the
official recipe answered correctly. Short prompts (GSM8K, canaries) were unaffected, which is why the a1 tuning runs did
not catch it. Bisected on 8x L20X: prefill kernel `tilelang` instead of `flashmla_sparse_q8` → still wrong; chunks 8192 → correct;
`MOE=split` with 16384 → correct. With 8192, long-context retrieval is 8/8 (official 8/8), GSM8K 95.8% (official 95.8%), and the
coding-agent test at 64 agents gives 3,154 output tok/s vs 2,888 for `MOE=split` + 16384 on the same L20X node.
`launch.sh` now refuses `--chunked-prefill-size` above 8192 with `MOE=mega`. The H200 numbers in the table above were measured
with 16384 and have not been re-measured yet.

**Changed 2026-11-10 (`MOE=mega`, now the default in every tier):** the expert layers of the main model run as one fused
kernel (`--moe-a2a-backend megamoe`), the MTP draft head stays on the lighter Triton path, and block verification is on.
Launch flags only, no code change. Measured gain: +12% at 64 long sessions, +19% at 128, +12% on the low-concurrency
load when prompts are mostly uncached, no loss when they are cached. `MOE=split` restores the 2026-11-06 behaviour.
The one case that got slower: `f1` pushed to 128 sessions (1,853 → 1,526 tok/s), because the fused kernel leaves ~4% less
GPU memory for the prompt cache; use `f2` there.

Short chat (ShareGPT, ~2K in / 1K out, measured 2026-11-06 with `MOE=split`): 271 tok/s per stream at 1 user, 202 at 4, 134 at 16.
Full GSM8K (1,319 questions, thinking off): 0.961–0.964 for every recipe (0.964 for `f2` with `MOE=mega`), same as without our changes.

## Low concurrency: 5 requests in flight, ~5.4K tokens in / ~160 out

A second load shape, measured 2026-11-10 on one 8×H200 node: a 200-prompt corpus (tool-using agent turns, retrieval
questions, code reading, document summaries) cycled closed-loop at concurrency 5 for 1,000 s per row, thinking off.
"Cache state" is how much of each prompt the server has already seen: every prompt (97% of input tokens hit), only the
shared system prompt and tool list (20%), or almost nothing (15%).

| settings | cache state | output tok/s | first token p50 / p95 / p99 (ms) | per token p50 / p99 (ms) | whole request p50 (ms) |
|---|---|---|---|---|---|
| `f1`, `MOE=split` (2026-11-06 default) | every prompt cached | 842 | 67 / 117 / 217 | 5.4 / 8.8 | 613 |
| `p1` (new) | every prompt cached | 846 | 68 / 120 / 211 | 5.3 / 8.2 | 606 |
| `f1`, `MOE=split` (2026-11-06 default) | system prompt + tools cached | 552 | 175 / 391 / 566 | 7.3 / 19.5 | 954 |
| `p1` (new) | system prompt + tools cached | 619 | 149 / 306 / 423 | 6.6 / 15.5 | 842 |
| `f1`, `MOE=split` (2026-11-06 default) | almost nothing cached | 542 | 183 / 429 / 620 | 7.3 / 21.1 | 982 |
| `p1` (new) | almost nothing cached | 605 | 156 / 320 / 440 | 6.7 / 15.6 | 852 |

For reference, the same node with speculative decoding switched off gives 499 / 382 tok/s (all / nothing cached) and
9.4 ms per token: MTP is worth about 1.4–1.7× on this load. GSM8K (200 questions) is 0.980 (`MOE=split`) vs 0.985
(`MOE=mega`), i.e. one question apart; a cached-vs-uncached output comparison passes. The host-RAM tier, which the
`f1` rows have and `p1` does not, changes these numbers by 2% or less at this concurrency.

With the earlier image tag `h200-20261106` (whose launcher has neither `p1` nor `MOE`) the equivalent of `p1` is
`-e TIER=f1 -e HICACHE=0` plus, after `launch.sh`:
`--moe-a2a-backend megamoe --speculative-moe-a2a-backend none --speculative-moe-runner-backend triton --speculative-use-block-verification`.

To replay this load against any OpenAI-compatible endpoint (yours or ours), see [`loadgen/`](loadgen):

```bash
pip install aiohttp && gunzip -k loadgen/corpus.jsonl.gz
python3 loadgen/replay.py --corpus loadgen/corpus.jsonl --out run_fresh --url http://localhost:30000 \
  --concurrency 5 --duration 1800 --mode fresh        # modes: replay (add --warm-pass) | sysshared | fresh
```

It prints and writes (`summary.json`) throughput, first-token time, time per output token, gap between streamed
chunks and whole-request time at p50 / p95 / p99. The corpus is synthetic: passages from Wikipedia (CC BY-SA 4.0)
and SGLang source (Apache-2.0) wrapped in generated task prompts; `loadgen/gen_corpus.py` rebuilds it.

Options: `HICACHE=0` drops the host-RAM KV tier (use it if the host has < ~1.3 TB free RAM, or for low
concurrency with short context where the GPU KV is enough; `p1` sets it by default). `PORT`, `TP`, `MODEL`,
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
- Once in about 30 server starts the base SGLang build exited about a minute into serving with `CUDA error: invalid argument`
  while capturing a CUDA graph at run time (`eagle_post_verify_graphs.py`). Restarting fixes it; run under a restart policy.
- Speed numbers depend on the workload; please measure with your own traffic and send us the result.
