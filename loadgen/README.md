# 压测工具与三版本对比方法 / Load test and three-way comparison

> **English summary.** `replay.py` is a closed-loop load generator for any OpenAI-compatible `/v1/chat/completions`
> endpoint: 5 requests in flight, a fixed 200-prompt corpus (avg 5.4K tokens in / 160 out) cycled for a fixed time,
> under three cache states. `run_bench.sh` runs all three and `summarize.py` prints one table. On two identical
> 8×H200 nodes, same weights, same corpus, same chat template, thinking off: our recipe gives 610–842 output tok/s,
> the unmodified official SGLang recipe 370–483 tok/s (1.65–1.74×); our recipe with only speculative decoding turned
> off falls back to 378–455 tok/s, so the gap is configuration, not hardware. Details below (in Chinese).

这个目录让任何人用同一份负载、同一套指标，在自己的机器上复现我们的对比结论。**只测速度，不测答案质量**（质量见主 README 里的 GSM8K）。

---

## 1. 一句话：这是什么压力测试

**5 个请求同时在跑（一个结束才发下一个），把 200 条固定的提示词循环发，每次测 400–1000 秒，记录整台机器每秒输出多少 token，以及每个请求的首字延迟和出字节奏。**

这是**低并发、中等长度输入、短输出**的形状：每条请求约 5.4K token 输入，约 160 token 输出。它像带工具的 agent、检索问答、读代码、长文摘要这类线上流量，不是「几百个用户同时聊天」的高并发。选这个形状，是因为对方合作方给我们的参考结果表就是这个形状（5 并发、平均 5,313 入 / 161 出、200 条语料循环、30 分钟），要比就得同形状比。

> 对方的语料我们拿不到，所以 `gen_corpus.py` 合成了一份：正文取自维基百科（CC BY-SA）和 SGLang 源码（Apache-2.0），套上生成的任务提示词。200 条里 80 条 agent 多轮（带工具定义，回答多为工具调用）、70 条检索问答、30 条读代码、20 条长文摘要，中英混合。输出长度上限（`max_tokens`）服从按对方延迟分位数反推出的对数正态分布。**任务构成是我们猜的**，长度分布是按对方数据对齐的。

## 2. 三种「缓存状态」：这是最容易测歪的地方

服务端有前缀缓存（prefix cache：开头相同的提示词，已算过的部分不用重算）。同一个服务，缓存命中多少，速度能差一倍，所以**必须说清缓存状态，并且把几种情况都测**。对方的结果表只写了缓存是「warm」，没给命中率，我们就把它夹在中间：

| `--mode` | 做法 | 输入 token 命中率 | 含义 |
|---|---|---|---|
| `replay`（加 `--warm-pass`） | 200 条原样循环，先跑一遍预热 | ≈97% | 上限：每条提示词都见过 |
| `sysshared` | 每条首个用户消息开头插一个唯一标记 | ≈20% | 只有系统提示词和工具列表能命中（同一个应用的多个请求共享这部分） |
| `fresh` | 提示词最开头插唯一标记 | ≈15% | 下限：几乎全要重算（对话模板把工具定义排在标记之前，所以还有一点命中） |

真实流量大概率落在后两者之间到第一种之间，三档都报，读者自己挑。

## 3. 指标定义（和对方的结果表同一定义）

| 指标 | 含义 |
|---|---|
| 输出 tok/s | 整台机器，5 个请求加起来**每秒输出的 token 数**（总输出 token ÷ 墙钟时间）。这是主指标 |
| 首字延迟 TTFT | 发出请求到收到第一段回答（含思考内容、工具调用）的时间 |
| 每 token 耗时 TPOT | （整个请求时间 − 首字延迟）÷（输出 token 数 − 1） |
| 流式间隔 ITL | 相邻两段流式输出的间隔。**注意**：开了推测解码后一段里带约 2.6 个 token，所以中位数会比 TPOT 大，只拿它的 p99 看「卡顿」 |
| p50 / p95 / p99 | 中位数 / 最慢 5% / 最慢 1% |

输出 token 数取服务端返回的 `usage.completion_tokens`，不是客户端数的块数。

## 4. 怎么跑

```bash
pip install aiohttp
# 对任意 OpenAI 兼容服务（自己的、我们的都行）；三档各 400 秒，约 22 分钟
URL=http://localhost:30000 NAME=my-setup DUR=400 bash loadgen/run_bench.sh

# 对比多个配置：给每个配置一个 NAME，最后一起出表
python3 loadgen/summarize.py results/*
```

常用环境变量：`URL`、`NAME`（必填，结果落到 `results/$NAME/`）、`DUR`（每档秒数，发表用的表我们用 1000）、`CONC`（并发，默认 5）、`MODEL`（默认 `zai-org/GLM-5.3-Flash`）、`API_KEY`（需要带 `Authorization: Bearer` 时）、`NO_TEMPLATE_KWARGS=1`（服务端不认 `chat_template_kwargs` 时）、`RARGS`（透传给 `replay.py` 的额外参数，如 `--temperature 0`）。

单独跑一档：`python3 replay.py --corpus corpus.jsonl --out out --mode fresh --duration 400`。每档输出 `requests.jsonl`（逐请求数据）和 `summary.json`。

## 5. 我们是怎么得出「更快」的结论的

三个版本，各自用同一份语料、同一个脚本、每档 400 秒、5 并发、思考关闭、0 失败：

| 版本 | 硬件 | 部署 |
|---|---|---|
| ① 他们的版本 | 8×H200 | **官方 SGLang 镜像，原样不改**（`lmsysorg/sglang:nightly-dev-20261106-858073e4`）+ SGLang cookbook 里 GLM-5.3-Flash 的配方：TP8 + EP8、tilelang 稀疏注意力、bf16 KV、`deep_gemm` MoE，**不开推测解码、不开 MegaMoE、无补丁** |
| ② 我们优化的版本 | 8×H200（和 ① 同型号、同权重） | 本仓库镜像，`TIER=p1`：MTP 推测解码 + MegaMoE 融合内核 + 我们的补丁 |
| ③ 芯华算版本 | 6000D 集群 | 我们的生产入口，开着思考（见下面的注意） |
| 对照 | 同 ② 那台机器 | ② 只把推测解码关掉，其他一模一样 |

结果（输出 tok/s；括号里是缓存命中率。完整数据在 [`results/`](results)）：

| 版本 | 全命中 (97%) | 只命中系统提示词 (20%) | 几乎不命中 (15%) | 每 token 耗时 p50（不命中，ms） |
|---|---|---|---|---|
| 对方结果表 | 317（只写「warm」） | | | 13.2 |
| ① 官方原版（H200） | 483 | 375 | 370 | 11.7 |
| **② 我们的配置（H200）** | **842** | **619** | **610** | **6.8** |
| 对照：② 只关推测解码 | 455 | 379 | 378 | 11.8 |
| ③ 6000D 生产（开着思考，不可直接比） | 772 | 422 | 402 | 5.8 |

**② 比 ① 快 1.65–1.74 倍，而且这个差距来自配置，不是硬件**，依据有三条：

1. **同机型、同权重、同语料、同对话模板，只有部署配置不同**（上表 ① 和 ②）。
2. **对照实验**：在 ② 的机器上只关掉推测解码，吞吐掉到 378–455，和 ① 同一量级。也就是说大头来自推测解码（模型先草拟几个 token，主模型一步验证，保留它本来也会输出的那些，输出不变），MegaMoE 和补丁是在此之上的增益。
3. **① 确实复现了对方**：平均输出 163 token（对方 161）；相邻两次输出的间隔中位数 9.24 ms，**和对方结果表的 9.24 ms 一致**，这个节奏是「没开推测解码」的特征。

## 6. 怎么读这些数字，哪些地方不能说

- **缓存命中少时，尾部延迟才是看点**：首字延迟 p99 和流式间隔 p99 在不命中时明显变差（正在输出的请求被别人的提示词预填充打断）。对方的 ITL p99 = 186 ms 就是这个特征。
- **①、② 在全命中档首字延迟差不多，① 还略快（56 对 68 ms）**：推测解码让整条回答更快结束，但不降低「读完提示词」的时间。我们没有挑数字，请按完整表读。
- **③ 不能和前两个直接比**：生产入口的对话模板没有关思考的开关，输出里大半是思考过程（平均 208 token，85% 的请求写到了 `max_tokens` 才停，① ② 约 59%），首字延迟也包含思考。`results/3-zt6000d-prod/*.json` 里的 `"thinking": false` 只是客户端发出的请求参数，不代表服务端没思考。
- **对话模板会毁掉对比**：我们踩过一次。① 第一次用模型自带模板，它没有关思考的开关，99% 的请求在思考里写到长度上限、一次工具调用也没有，数据和对方对不上。**对比时双方必须用同一份对话模板**（模板只影响请求格式，不是性能优化）。
- **噪声**：同一配置重复跑，吞吐和中位数差 ±1%；p99 这类尾部指标 5 分钟的短跑会差到 ±25%。发表的表请用 `DUR=1000`。
- **本测试没有覆盖**：高并发（几十上百个会话，见主 README 的 agent 基准）、更长输入、答案质量。对方 317 来自他们自己的结果表，不是在同一台机器上同时测的；①② 是同型号的两台机器，不是同一台。
- **隐含前提**：请求不带 `temperature`，用服务端默认；`replay.py` 固定随机种子 `--seed 1`，每次请求顺序相同。

## 7. 文件

| 文件 | 作用 |
|---|---|
| `replay.py` | 压测主程序：闭环回放、三种缓存模式、统计 TTFT / TPOT / ITL / 端到端的 p50/p95/p99 |
| `run_bench.sh` | 一键跑三档并出表 |
| `summarize.py` | 把多个配置的 `summary.json` 汇成一张 Markdown 表 |
| `corpus.jsonl.gz` | 合成语料（200 条，已随仓库提供，直接用） |
| `gen_corpus.py` | 重新生成语料（需要联网取维基百科；固定种子，但维基页面会更新，所以请优先用现成的 `corpus.jsonl.gz`） |
| `results/` | 上面三个版本和对照的原始 `summary.json`（每档一个） |

直观版：[三版本报告](https://reports.aitist.ai/zhongtian/glm-flash-three-versions-20261111/) · [同一问题三版本并排的录屏](https://reports.aitist.ai/zhongtian/glm-flash-speed-video-20261111/)
