#!/usr/bin/env python3
"""replay.py — closed-loop corpus replay, 5 requests in flight, a 200-prompt corpus cycled (the shape of a reference run we were given).

  replay.py --corpus corpus.jsonl --out DIR [--concurrency 5] [--duration 1800] [--mode replay|sysshared|fresh]
            [--warm-pass] [--temperature T] [--thinking]
Modes (what a "warm cache" can mean — the reference sheet does not say which):
  replay     the 200 requests are cycled unchanged, so after the first pass every prompt is a full prefix-cache hit
  sysshared  a unique marker is put at the start of the first user turn: only system prompt + tool schemas can hit
  fresh      a unique marker at the very start: no cache hit at all (lower bound)
Metrics follow the reference sheet: TTFT = first streamed delta; TPOT = (e2e - TTFT) / (output tokens - 1);
ITL = gap between consecutive streamed chunks (a chunk can carry several tokens under speculative decoding).
"""
import argparse, asyncio, json, os, random, time, uuid
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument('--corpus', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--url', default='http://localhost:30000'); ap.add_argument('--model', default='zai-org/GLM-5.3-Flash')
ap.add_argument('--concurrency', type=int, default=5); ap.add_argument('--duration', type=float, default=1800)
ap.add_argument('--mode', default='replay', choices=['replay', 'sysshared', 'fresh']); ap.add_argument('--warm-pass', action='store_true')
ap.add_argument('--temperature', type=float, default=None); ap.add_argument('--thinking', action='store_true'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--passes', type=int, default=None, help='record exactly N passes over the corpus instead of --duration')
A = ap.parse_args()
rows = [json.loads(l) for l in open(A.corpus)]; order = list(range(len(rows))); random.Random(A.seed).shuffle(order)
os.makedirs(A.out, exist_ok=True)

def body_for(r):
    msgs = [dict(m) for m in r['messages']]
    if A.mode != 'replay':
        i = 0 if A.mode == 'fresh' else next(j for j, m in enumerate(msgs) if m['role'] == 'user')
        msgs[i]['content'] = f'[req {uuid.uuid4().hex}]\n' + msgs[i]['content']
    b = {'model': A.model, 'messages': msgs, 'max_tokens': r['max_tokens'], 'max_completion_tokens': r['max_tokens'],
         'stream': True, 'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': bool(A.thinking)}}
    if r.get('tools'): b['tools'] = r['tools']
    if A.temperature is not None: b['temperature'] = A.temperature
    return b

async def one(sess, r):
    t0 = time.perf_counter(); stamps, usage, fin, err = [], {}, None, None
    try:
        async with sess.post(A.url + '/v1/chat/completions', json=body_for(r)) as resp:
            if resp.status != 200: return dict(id=r['id'], err=f'http {resp.status}: {(await resp.text())[:200]}')
            async for raw in resp.content:
                if not raw.startswith(b'data: {'): continue
                d = json.loads(raw[6:])
                if d.get('usage'): usage = d['usage']
                for c in d.get('choices') or []:
                    dl = c.get('delta') or {}
                    if dl.get('content') or dl.get('reasoning_content') or dl.get('tool_calls'): stamps.append(time.perf_counter())
                    fin = c.get('finish_reason') or fin
    except Exception as e: err = repr(e)[:200]
    t1 = time.perf_counter()
    if err or not stamps: return dict(id=r['id'], err=err or f'no content (finish={fin}, usage={usage})')
    n = usage.get('completion_tokens') or 0
    return dict(id=r['id'], kind=r['kind'], t=t0, ttft=stamps[0] - t0, e2e=t1 - t0, out=n, inp=usage.get('prompt_tokens') or 0,
                cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0, fin=fin,
                tpot=(t1 - stamps[0]) / (n - 1) if n > 1 else None, itl=[b - a for a, b in zip(stamps, stamps[1:])])

async def run(duration, record, passes=None):
    cur = {'i': 0}; res = []; tend = time.perf_counter() + duration if duration else None
    async def worker():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900), headers={'User-Agent': 'zt-replay/1.0'}) as sess:
            while True:
                i = cur['i']; cur['i'] += 1
                if (tend and time.perf_counter() >= tend) or (passes and i >= passes * len(rows)): return
                x = await one(sess, rows[order[i % len(rows)]])
                if record: res.append(x)
    t0 = time.perf_counter(); await asyncio.gather(*[worker() for _ in range(A.concurrency)]); return res, time.perf_counter() - t0

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float('nan')
async def main():
    if A.warm_pass:
        _, tw = await run(None, False, passes=1); print(f'warm pass done in {tw:.0f}s', flush=True)
    res, wall = await run(None if A.passes else A.duration, True, passes=A.passes)
    ok = [x for x in res if 'err' not in x]; bad = [x for x in res if 'err' in x]
    with open(f'{A.out}/requests.jsonl', 'w') as f:
        for x in res: f.write(json.dumps({k: v for k, v in x.items() if k != 'itl'}) + '\n')
    itl = [g for x in ok for g in x['itl']]; ms = lambda v: round(v * 1000, 2)
    S = dict(scope=f'{A.mode}@c{A.concurrency}' + ('/warm' if A.warm_pass else ''), wall_s=round(wall, 1), temperature=A.temperature, thinking=A.thinking,
        requests=dict(total=len(res), completed=len(ok), failed=len(bad), error_rate=round(len(bad) / max(1, len(res)), 4)),
        tokens=dict(input_total=sum(x['inp'] for x in ok), output_total=sum(x['out'] for x in ok), cached_total=sum(x['cached'] for x in ok),
                    avg_input_per_request=round(sum(x['inp'] for x in ok) / max(1, len(ok))), avg_output_per_request=round(sum(x['out'] for x in ok) / max(1, len(ok))),
                    cached_frac=round(sum(x['cached'] for x in ok) / max(1, sum(x['inp'] for x in ok)), 4),
                    hit_length_cap_frac=round(sum(x['fin'] == 'length' for x in ok) / max(1, len(ok)), 3)),
        throughput=dict(requests_per_s=round(len(ok) / wall, 2), input_tokens_per_s=round(sum(x['inp'] for x in ok) / wall, 2), output_tokens_per_s=round(sum(x['out'] for x in ok) / wall, 2)),
        latency_ms={k: {f'p{p}': ms(pct(v, p)) for p in (50, 95, 99)} for k, v in dict(ttft=[x['ttft'] for x in ok], tpot=[x['tpot'] for x in ok if x['tpot']], itl=itl, e2e=[x['e2e'] for x in ok]).items()},
        errors=[x['err'] for x in bad[:5]])
    json.dump(S, open(f'{A.out}/summary.json', 'w'), indent=1)
    L = S['latency_ms']; print(f"SUM {S['scope']} n={len(ok)} fail={len(bad)} in/out={S['tokens']['avg_input_per_request']}/{S['tokens']['avg_output_per_request']} cached={S['tokens']['cached_frac']:.0%} "
          f"out_tok/s={S['throughput']['output_tokens_per_s']} ttft50/95/99={L['ttft']['p50']}/{L['ttft']['p95']}/{L['ttft']['p99']} tpot50/95/99={L['tpot']['p50']}/{L['tpot']['p95']}/{L['tpot']['p99']} "
          f"itl50/99={L['itl']['p50']}/{L['itl']['p99']} e2e50/95/99={L['e2e']['p50']}/{L['e2e']['p95']}/{L['e2e']['p99']}", flush=True)
asyncio.run(main())
