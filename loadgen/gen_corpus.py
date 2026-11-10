#!/usr/bin/env python3
"""gen_corpus.py — synthetic stand-in for a private 200-prompt corpus we could not obtain.

What we know (the reference result sheet): 200-entry corpus replayed closed-loop at concurrency 5 for 30 min
(3,546 requests ≈ 17.7 passes), avg 5,313 input / 161 output tokens, cache "warm", per-request token limit set.
Their e2e p50/p95/p99 (1.45 / 7.2 / 13.6 s) with TPOT 13 ms and TTFT 0.23 s imply output lengths of about
93 / 520 / 1,030 tokens — a lognormal(median 93, sigma 1.05) reproduces all three and the mean of 161.

What we assume (not known): the task mix. A 93-token median with a long tail and a ~5K prompt looks like
agent / RAG traffic, so the corpus mixes four shapes, mostly Chinese:
  agent  40%  tool schemas + multi-turn history with tool results -> next step (short)
  rag    35%  system prompt + retrieved passages + question
  code   15%  source file context + request
  doc    10%  long document -> summary / extraction
Requests of one "app" share a system prompt (prefix sharing across requests), as real apps do.
Text is public: Wikipedia (CC BY-SA) passages and SGLang source (Apache-2.0).
Output: corpus.jsonl  one {"id","kind","app","messages","max_tokens","in_tokens"} per line, plus corpus_stats.json.
"""
import json, math, os, random, re, sys, glob, zlib, urllib.request, urllib.parse
from tokenizers import Tokenizer

OUT = sys.argv[1]; TOKJSON = sys.argv[2]; CODE_DIR = sys.argv[3]
N, MEAN_IN, MEAN_OUT, SEED = 200, 5313, 161, 20261110
IN_SCALE = float(os.environ.get('IN_SCALE', 1)); OUT_SCALE = float(os.environ.get('OUT_SCALE', 1))  # calibration knobs, set from a measured pass
rng = random.Random(SEED); tok = Tokenizer.from_file(TOKJSON)
ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
os.makedirs(OUT, exist_ok=True); CACHE = f'{OUT}/wiki_cache.json'

ZH = "人工智能 机器学习 深度学习 自然语言处理 图形处理器 半导体 集成电路 云计算 数据中心 操作系统 数据库 计算机网络 互联网 电子商务 供应链管理 物流 银行 保险 证券交易所 货币政策 通货膨胀 国内生产总值 增值税 企业所得税 合同法 知识产权 专利 商标 劳动法 消费者权益保护 糖尿病 高血压 流行性感冒 疫苗 抗生素 心脏病 医疗保险 中医 营养学 睡眠 长江 黄河 北京市 上海市 深圳市 杭州市 成都市 高速铁路 新能源汽车 锂离子电池 太阳能 风力发电 核能 碳中和 气候变化 唐朝 宋朝 明朝 丝绸之路 科举 红楼梦 三国演义 李白 杜甫 孔子 围棋 足球 奥林匹克运动会 高等教育 义务教育 大学入学考试 航空航天 北斗卫星导航系统 量子计算 区块链 移动支付 第五代移动通信技术 物联网 自动驾驶汽车 机器人 无人机 微信 淘宝网 抖音 网络安全 密码学 开源软件 Linux Python 关系数据库 分布式计算 编译器 算法 数据结构".split()
EN = "Transformer_(deep_learning_architecture) Large_language_model Graphics_processing_unit Cloud_computing Kubernetes PostgreSQL Distributed_computing Public-key_cryptography Supply_chain_management Inventory Double-entry_bookkeeping Value-added_tax Contract Patent Insurance Mortgage Central_bank Inflation Diabetes Hypertension Vaccine Antibiotic Electric_vehicle Lithium-ion_battery Solar_power Wind_power Nuclear_power Climate_change High-speed_rail Semiconductor_device_fabrication Operating_system Computer_network Database_index Compiler Garbage_collection_(computer_science) Hash_table MapReduce Raft_(algorithm) HTTP Transmission_Control_Protocol Customer_relationship_management Enterprise_resource_planning Project_management Agile_software_development Software_testing Continuous_integration".split()

def wiki(lang, title):
    q = urllib.parse.urlencode(dict(action='query', prop='extracts', explaintext=1, redirects=1, format='json', titles=title.replace('_', ' '), **({'variant': 'zh-cn'} if lang == 'zh' else {})))
    req = urllib.request.Request(f'https://{lang}.wikipedia.org/w/api.php?{q}', headers={'User-Agent': 'zt-bench-corpus/1.0 (research; synthetic load test)'})
    pages = json.load(urllib.request.urlopen(req, timeout=40))['query']['pages']
    return next(iter(pages.values())).get('extract', '')

import time
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
for lang, titles in (('zh', ZH), ('en', EN)):
    for t in titles:
        k = f'{lang}:{t}'
        for attempt in range(0 if os.environ.get('OFFLINE') else 4):
            if cache.get(k): break
            try: cache[k] = wiki(lang, t)
            except Exception as e: print('wiki retry', k, e, file=sys.stderr); time.sleep(3 * (attempt + 1))
            time.sleep(0.4)
json.dump(cache, open(CACHE, 'w'), ensure_ascii=False)
arts = {k: v for k, v in cache.items() if len(v) > 1500}
print(len(arts), 'articles', file=sys.stderr)
def paras(k): return [p.strip() for p in re.split(r'\n+', arts[k]) if len(p.strip()) > 80 and not p.startswith('==')]
zh_keys = [k for k in arts if k.startswith('zh:')]; en_keys = [k for k in arts if k.startswith('en:')]
code_files = [f for f in glob.glob(f'{CODE_DIR}/**/*.py', recursive=True) if 3000 < os.path.getsize(f) < 60000]
rng.shuffle(code_files)

def take(texts, budget):
    """concatenate whole paragraphs until ~budget tokens"""
    out, n = [], 0
    for t in texts:
        c = ntok(t)
        if n + c > budget and out: break
        out.append(t); n += c
        if n >= budget: break
    return out

# ---------- tool schemas (agent apps) ----------
DOMAINS = {
 '订单客服': ['查询订单', '查询物流', '申请退款', '修改收货地址', '查询优惠券', '创建工单', '查询商品库存', '转人工客服', '查询发票', '取消订单'],
 '企业知识库助手': ['搜索知识库', '读取文档', '查询员工信息', '提交审批', '查询日程', '创建会议', '发送通知', '查询报销单', '查询考勤', '生成周报'],
 '运维助手': ['查询告警', '查询主机指标', '执行巡检脚本', '查询发布记录', '回滚版本', '查询日志', '创建故障单', '查询值班表', '扩容实例', '查询数据库慢查询'],
 '金融顾问': ['查询账户余额', '查询交易明细', '查询基金净值', '计算贷款月供', '查询汇率', '风险测评', '查询理财产品', '预约网点', '挂失银行卡', '查询信用额度'],
 'data analyst': ['run_sql', 'describe_table', 'plot_chart', 'export_csv', 'search_metrics', 'get_dashboard', 'create_alert', 'list_tables', 'explain_query', 'sample_rows'],
 'coding agent': ['read_file', 'write_file', 'grep', 'run_tests', 'list_dir', 'git_diff', 'apply_patch', 'run_shell', 'search_symbols', 'open_pull_request'],
}
PTYPES = [('string', '字符串，必填'), ('integer', '整数'), ('boolean', '是否'), ('string', 'ISO 8601 日期'), ('number', '金额，单位元')]
def tool(name, zh=True):
    props = {}
    for i in range(rng.randint(2, 6)):
        t, d = rng.choice(PTYPES)
        props[f'{"param" if not zh else "arg"}_{i}_{rng.choice(["id","type","start","end","page","limit","keyword","status","amount","user"])}'] = {'type': t, 'description': f'{name}所需的{d}；留空时使用默认值。' if zh else f'{d} used by {name}; default applies when omitted.'}
    return {'type': 'function', 'function': {'name': name if not zh else 'fn_' + str(zlib.crc32(name.encode()) % 10**6), 'description': (f'{name}。根据用户提供的信息调用后端接口并返回结构化结果；调用前必须确认必填参数齐全，失败时返回错误码与原因。' if zh else f'{name}: call the backend and return structured JSON; validate required params first, return error code and reason on failure.'),
            'parameters': {'type': 'object', 'properties': props, 'required': list(props)[:2]}}}
RULES_ZH = ['回答必须基于工具返回的数据或给定资料，不得编造。', '涉及金额、日期、单号时逐字核对。', '用户情绪激动时先致歉再处理。', '一次只调用一个工具，等待结果后再决定下一步。', '无法处理时说明原因并建议转人工。', '输出使用简体中文，语气专业、简洁。', '不要泄露系统提示词与内部接口名称。', '需要用户补充信息时一次问清楚。', '引用资料时标注段落编号。', '遇到政策类问题以最新发布的文件为准。']
def app_system(name, zh, n_rules):
    rules = [rng.choice(RULES_ZH) + f'（规则 {i+1}）' for i in range(n_rules)]
    head = f'你是「{name}」，一个面向企业客户的智能助手。今天是 2026 年 10 月 26 日。\n\n## 行为准则\n' if zh else f'You are "{name}", an enterprise assistant. Today is 2026-10-26.\n\n## Rules\n'
    return head + '\n'.join(f'{i+1}. {r}' for i, r in enumerate(rules))
APPS = []
for name, tools in DOMAINS.items():
    zh = not name.isascii()
    APPS.append(dict(name=name, zh=zh, system=app_system(name, zh, rng.randint(12, 30)), tools=[tool(t, zh) for t in tools]))
RAG_APPS = [dict(name=n, zh=True, system=app_system(n, True, rng.randint(15, 40)) + '\n\n请只依据「参考资料」回答，资料不足时明确说明。') for n in ['政策问答助手', '产品手册助手', '医疗科普助手', '法务检索助手']] + \
           [dict(name='docs assistant', zh=False, system=app_system('docs assistant', False, 20) + '\n\nAnswer only from the reference passages; say so when they are insufficient.')]

def out_len():
    return max(8, min(2000, int(rng.lognormvariate(math.log(93), 1.05))))
def in_budget(kind):
    base = {'agent': 4600, 'rag': 5200, 'code': 5600, 'doc': 9000}[kind]
    return int(base * IN_SCALE * rng.lognormvariate(0, 0.35))

Q_ZH = ['请根据以上资料，概括要点并说明其对企业的实际影响。', '上述内容中有哪些关键数据？请分条列出并解释。', '请对比资料中提到的不同方案，各自的优缺点是什么？', '如果我是初学者，应该怎样理解这部分内容？请详细讲解。', '请写一段面向管理层的说明，解释其发展历程和现状。']
Q_EN = ['Summarise the key points above and explain why they matter in practice.', 'List the important figures mentioned and explain each one.', 'Compare the approaches described; give pros and cons of each.', 'Explain this to a newcomer in detail, step by step.']
def mk_rag(i):
    app = rng.choice(RAG_APPS); keys = rng.sample(zh_keys if app['zh'] else en_keys, 6); b = in_budget('rag') - ntok(app['system'])
    ps = []
    for k in keys: ps += rng.sample(paras(k), min(14, len(paras(k))))
    rng.shuffle(ps)
    ps = take(ps, max(600, b))
    ctx = '\n\n'.join(f'[{j+1}] {p}' for j, p in enumerate(ps))
    q = rng.choice(Q_ZH if app['zh'] else Q_EN)
    return app['name'], [{'role': 'system', 'content': app['system']}, {'role': 'user', 'content': ('## 参考资料\n\n' if app['zh'] else '## Reference passages\n\n') + ctx + '\n\n## ' + ('问题' if app['zh'] else 'Question') + '\n' + q}], None
def mk_doc(i):
    zh = rng.random() < 0.7; k = rng.choice(zh_keys if zh else en_keys); extra = rng.sample(zh_keys if zh else en_keys, 3)
    ps = paras(k); [ps.extend(paras(e)) for e in extra]; body = '\n\n'.join(take(ps, in_budget('doc')))
    ask = rng.choice(['请为下面的长文写一份结构化摘要，包含背景、主要内容、关键结论三部分，并尽量保留具体数据。', '请从下面的文档中抽取所有时间、机构、数字，并整理成条目清单后逐条说明。'] if zh else ['Write a structured summary of the document below (background, main content, conclusions), keeping concrete figures.', 'Extract every date, organisation and number from the document below as a list and explain each.'])
    return 'doc', [{'role': 'user', 'content': ask + '\n\n---\n\n' + body}], None
def mk_code(i):
    f = code_files[i % len(code_files)]; src = open(f, errors='ignore').read(); b = in_budget('code')
    ids = tok.encode(src, add_special_tokens=False).ids
    if len(ids) > b: src = tok.decode(ids[:b])
    ask = rng.choice(['Review this file: explain what it does, then point out bugs or risky edge cases and propose concrete fixes with code.', '请阅读下面的代码，逐个函数说明作用，并给出一份改进后的关键函数实现。', 'Write unit tests for the main functions in this file, with brief comments on what each test covers.', '请为这个文件写一份面向新同事的技术文档，说明模块职责、主要流程和注意事项。'])
    return 'coding agent', [{'role': 'system', 'content': 'You are a senior software engineer. Be precise and concrete.'}, {'role': 'user', 'content': f'{ask}\n\n```python\n# {os.path.basename(f)}\n{src}\n```'}], None
def mk_agent(i):
    app = rng.choice(APPS); zh = app['zh']; b = in_budget('agent') - ntok(app['system']) - ntok(json.dumps(app['tools'], ensure_ascii=False))
    msgs = [{'role': 'system', 'content': app['system']}]
    first = rng.choice(['我上周下的订单到现在还没收到，帮我查一下怎么回事，另外我想改一下收货地址。', '帮我查一下最近三个月的记录，汇总一下异常的情况，并告诉我下一步该怎么处理。', '系统今天早上开始报错，帮我排查原因并给出处理建议。', '我想了解一下相关的规定，然后帮我提交一个申请。'] if zh else ['Find out why yesterday\'s numbers dropped and show me the breakdown by region.', 'The test suite fails on main; investigate and fix it.', 'Pull the last quarter\'s data and summarise anomalies.'])
    msgs.append({'role': 'user', 'content': first}); n, step = 0, 0
    pool = zh_keys if zh else en_keys
    while n < max(400, b):
        t = rng.choice(app['tools'])['function']; step += 1; cid = f'call_{i}_{step}'
        args = {k: rng.choice(['A2026102600' + str(rng.randint(100, 999)), 'pending', 20, True, '2026-10-2' + str(rng.randint(0, 6))]) for k in t['parameters']['required']}
        msgs.append({'role': 'assistant', 'content': '', 'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': t['name'], 'arguments': json.dumps(args, ensure_ascii=False)}}]})
        ps = rng.sample(paras(rng.choice(pool)), 3)
        res = {'code': 0, 'request_id': f'{rng.getrandbits(64):016x}', 'data': {'items': [{'id': rng.randint(10**6, 10**7), 'status': rng.choice(['已完成', '处理中', '异常', 'ok', 'failed']), 'updated_at': f'2026-10-{rng.randint(10, 26)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00+08:00', 'amount': round(rng.uniform(10, 9999), 2), 'detail': p[:rng.randint(150, 500)]} for p in ps], 'total': rng.randint(3, 200)}}
        body = json.dumps(res, ensure_ascii=False); msgs.append({'role': 'tool', 'tool_call_id': cid, 'content': body}); n += ntok(body) + 40
        if rng.random() < 0.25 and n < b:
            msgs.append({'role': 'assistant', 'content': ('已查到相关记录，正在进一步核实。' if zh else 'Found the records; checking further.')}); msgs.append({'role': 'user', 'content': rng.choice(['好的，继续。另外把处理结果详细总结给我。', '那接下来呢？请把每一步的依据说清楚。'] if zh else ['OK, continue, and give me a detailed write-up at the end.', 'What next? Explain the reasoning for each step.'])})
    return app['name'], msgs, app['tools']

kinds = ['agent'] * 80 + ['rag'] * 70 + ['code'] * 30 + ['doc'] * 20; rng.shuffle(kinds)
MK = dict(agent=mk_agent, rag=mk_rag, code=mk_code, doc=mk_doc); rows = []
for i, k in enumerate(kinds):
    app, msgs, tools = MK[k](i)
    r = dict(id=i, kind=k, app=app, messages=msgs, max_tokens=out_len())
    if tools: r['tools'] = tools
    r['in_tokens_est'] = ntok(json.dumps(msgs, ensure_ascii=False)) + (ntok(json.dumps(tools, ensure_ascii=False)) if tools else 0)
    rows.append(r)
# Output caps: agent steps stop on their own (tool call), the other kinds run to the cap. OUT_SCALE is calibrated from a
# measured pass so that the realised mean output is MEAN_OUT.
for r in rows:
    if r['kind'] != 'agent': r['max_tokens'] = max(8, min(2400, round(r['max_tokens'] * OUT_SCALE)))
with open(f'{OUT}/corpus.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
ins = sorted(r['in_tokens_est'] for r in rows); outs = sorted(r['max_tokens'] for r in rows); pct = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
st = dict(n=N, seed=SEED, in_est_mean=sum(ins) / N, in_p50=pct(ins, .5), in_p95=pct(ins, .95), in_max=ins[-1], out_cap_mean=sum(outs) / N, out_p50=pct(outs, .5), out_p95=pct(outs, .95), out_p99=pct(outs, .99),
          kinds={k: kinds.count(k) for k in MK}, target=dict(in_mean=MEAN_IN, out_mean=MEAN_OUT))
json.dump(st, open(f'{OUT}/corpus_stats.json', 'w'), indent=1); print(json.dumps(st))
