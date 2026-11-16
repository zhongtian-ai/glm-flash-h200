#!/usr/bin/env python3
"""summarize.py — turn the summary.json files written by replay.py into one comparison table.

  summarize.py results/*            # each argument is a directory holding replay.json / sysshared.json / fresh.json
                                    # (or the replay/ sysshared/ fresh/ sub-directories that replay.py --out creates)
Prints a Markdown table: one row per (setup, cache state).
"""
import json, os, sys

MODES = [('replay', 'every prompt cached (~97%)'), ('sysshared', 'system prompt + tools cached (~20%)'), ('fresh', 'almost nothing cached (~15%)')]

def load(d, m):
    for p in (f'{d}/{m}.json', f'{d}/{m}/summary.json'):
        if os.path.exists(p): return json.load(open(p))

rows = []
for d in sys.argv[1:]:
    d = d.rstrip('/')
    for m, label in MODES:
        s = load(d, m)
        if not s: continue
        L, T = s['latency_ms'], s['tokens']
        rows.append((os.path.basename(d), label, s['throughput']['output_tokens_per_s'], f"{T['cached_frac']:.0%}",
                     f"{L['ttft']['p50']:.0f} / {L['ttft']['p99']:.0f}", f"{L['tpot']['p50']:.1f}", f"{L['itl']['p50']:.1f}",
                     T['avg_output_per_request'], f"{T['hit_length_cap_frac']:.0%}", s['requests']['failed']))
print('| setup | cache state | output tok/s | cache hit | first token p50 / p99 (ms) | per token p50 (ms) | stream gap p50 (ms) | avg out tokens | hit max_tokens | failed |')
print('|---|---|---|---|---|---|---|---|---|---|')
for r in rows: print('| ' + ' | '.join(str(x) for x in r) + ' |')
