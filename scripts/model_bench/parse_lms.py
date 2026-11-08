"""Parse LM Studio server logs into one row per POST request (CSV + JSON)."""
import json, re, sys, csv
from datetime import datetime

TS = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]')
REQ = re.compile(r'Received request: POST to (\S+) with body \{')
CACHE = re.compile(r'Prompt cache restore: cached_tokens=(\d+) uncached_tokens=(\d+)')

def ts(line):
    m = TS.match(line)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None

def read_json(lines, i):
    """lines[i] ends with '{'; collect until a line that is exactly '}'."""
    buf = ['{']
    j = i + 1
    while j < len(lines) and lines[j].rstrip('\n') != '}':
        buf.append(lines[j]); j += 1
    buf.append('}')
    try:
        return json.loads('\n'.join(buf)), j
    except Exception:
        return None, j

rows = []
for path in sys.argv[1:-1]:
    lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
    i = 0
    cur = None
    while i < len(lines):
        ln = lines[i]
        m = REQ.search(ln)
        if m:
            body, j = read_json(lines, i)
            cur = {'t_req': ts(ln), 'endpoint': m.group(1), 'body': body or {}}
            rows.append(cur)
            i = j + 1; continue
        if cur is not None:
            c = CACHE.search(ln)
            if c:
                cur['cached'], cur['uncached'] = int(c.group(1)), int(c.group(2))
            if 'Prompt processing progress: 100.0%' in ln or ('Prompt processing progress' in ln):
                cur['t_prefill_end'] = ts(ln)
            if 'error' in ln.lower() and 'Caused By' in ln:
                cur['error'] = ln[:200]
            if 'Generated prediction: {' in ln or ln.rstrip().endswith('Finished streaming response') :
                cur['t_done'] = ts(ln)
                if ln.rstrip().endswith('{'):
                    pred, j = read_json(lines, i)
                    cur['pred'] = pred or {}
                    i = j + 1; continue
        i += 1

out = []
for r in rows:
    b, p = r['body'], r.get('pred', {})
    msgs = b.get('messages') or ([{'role': 'user', 'content': b.get('input')}] if b.get('input') else [])
    sysmsg = next((m.get('content') for m in msgs if m.get('role') == 'system'), '') or ''
    usermsg = next((m.get('content') for m in reversed(msgs) if m.get('role') == 'user'), '') or ''
    if not isinstance(sysmsg, str): sysmsg = json.dumps(sysmsg)[:200]
    if not isinstance(usermsg, str): usermsg = json.dumps(usermsg)[:200]
    u = p.get('usage', {}) or {}
    ch = (p.get('choices') or [{}])[0]
    msg = ch.get('message', {}) or {}
    t0, t1, t2 = r.get('t_req'), r.get('t_prefill_end'), r.get('t_done')
    out.append({
        'time': t0.isoformat(sep=' ') if t0 else '',
        'endpoint': r['endpoint'],
        'model': b.get('model', ''),
        'n_msgs': len(msgs),
        'system': sysmsg[:140].replace('\n', ' '),
        'user': usermsg[:160].replace('\n', ' '),
        'max_tokens': b.get('max_tokens', ''),
        'reasoning_effort': b.get('reasoning_effort', ''),
        'temperature': b.get('temperature', ''),
        'stream': b.get('stream', ''),
        'tools': len(b.get('tools') or []),
        'resp_format': (b.get('response_format') or {}).get('type', ''),
        'cached': r.get('cached', ''), 'uncached': r.get('uncached', ''),
        'prompt_tokens': u.get('prompt_tokens', ''),
        'completion_tokens': u.get('completion_tokens', ''),
        'reasoning_tokens': (u.get('completion_tokens_details') or {}).get('reasoning_tokens', ''),
        'finish': ch.get('finish_reason', ''),
        'content_len': len(msg.get('content') or ''),
        'prefill_s': (t1 - t0).total_seconds() if t0 and t1 else '',
        'total_s': (t2 - t0).total_seconds() if t0 and t2 else '',
        'error': r.get('error', ''),
    })

with open(sys.argv[-1], 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(len(out), 'rows')
