#!/usr/bin/env bash
# check.sh — run on the host after launch: health, one chat answer, short speed probe (1 and 16 parallel streams).
set -euo pipefail
URL=${URL:-http://localhost:30000}; NAME=${SERVED_NAME:-zai-org/GLM-5.3-Flash}
curl -sf "$URL/health" >/dev/null && echo "health OK"
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":32,\"temperature\":0}" \
  | python3 -c 'import json,sys;m=json.load(sys.stdin)["choices"][0]["message"];print("answer:",(m.get("content") or m.get("reasoning_content") or "").strip()[:80])'
python3 - "$URL" "$NAME" <<'PY'
import json,sys,time,threading,urllib.request
url,name=sys.argv[1],sys.argv[2]
def one(out):
    body=json.dumps({"model":name,"messages":[{"role":"user","content":"Write a 600-word story about a lighthouse keeper."}],
                     "max_tokens":600,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(url+"/v1/chat/completions",body,{"Content-Type":"application/json","User-Agent":"glm-flash-check/1"})
    t0=time.time();first=None;n=0
    for line in urllib.request.urlopen(req,timeout=600):
        if line.startswith(b"data: {"):
            d=json.loads(line[6:])
            dl=d["choices"][0]["delta"] if d.get("choices") else {}
            if first is None and (dl.get("content") or dl.get("reasoning_content")): first=time.time()   # thinking is on by default
            if d.get("usage"): n=d["usage"]["completion_tokens"]
    if first is None or not n: raise SystemExit("no streamed tokens; check the server log")
    out.append((first-t0,n/(time.time()-first)))
for c in (1,16):
    out=[];ts=[threading.Thread(target=one,args=(out,)) for _ in range(c)];[t.start() for t in ts];[t.join() for t in ts]
    print(f"{c:>2} streams: per-stream {sum(o[1] for o in out)/c:6.1f} tok/s, TTFT {sum(o[0] for o in out)/c*1000:6.0f} ms")
PY
