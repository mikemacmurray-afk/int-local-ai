"""Compare bench result files: python compare.py results/a.jsonl results/b.jsonl ..."""
import collections
import json
import statistics as st
import sys


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


files = sys.argv[1:]
data = {p.split("/")[-1].split("\\")[-1].replace(".jsonl", ""): load(p) for p in files}


def key(r):
    s = r["suite"]
    if s in ("chat", "hard", "code"):
        return f"{s}:{r.get('mode')}"
    return s


groups = collections.OrderedDict()
for label, rows in data.items():
    for r in rows:
        if r["suite"] in ("conc", "perf", "longctx"):
            continue
        groups.setdefault(key(r), {}).setdefault(label, []).append(r)

labels = list(data)
print(f"{'suite/mode':22}" + "".join(f"{l:>34}" for l in labels))
print(f"{'':22}" + "".join(f"{'right':>10}{'time s':>8}{'out tok':>9}{'trunc':>7}" for _ in labels))
tot = {l: [0, 0, 0.0, 0] for l in labels}
for g, by in groups.items():
    line = f"{g:22}"
    for l in labels:
        rs = by.get(l, [])
        if not rs:
            line += f"{'-':>34}"
            continue
        ok = sum(1 for r in rs if r.get("ok"))
        secs = sum(r.get("secs", 0) for r in rs)
        out = sum(r.get("out_tokens", 0) for r in rs)
        tr = sum(1 for r in rs if r.get("finish") == "length")
        tot[l][0] += ok; tot[l][1] += len(rs); tot[l][2] += secs; tot[l][3] += out
        line += f"{f'{ok}/{len(rs)}':>10}{secs:8.0f}{out:9}{tr:7}"
    print(line)
print(f"{'ALL (quality suites)':22}" + "".join(
    f"{f'{tot[l][0]}/{tot[l][1]}':>10}{tot[l][2]:8.0f}{tot[l][3]:9}{'':7}" for l in labels))

print("\nTasks that differ (right count per label):")
per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for l, rows in data.items():
    for r in rows:
        if r["suite"] in ("conc", "perf", "longctx"):
            continue
        k = f"{key(r)}/{r.get('task')}"
        per[k][l][0] += 1 if r.get("ok") else 0
        per[k][l][1] += 1
for k, by in sorted(per.items()):
    vals = [tuple(by[l]) for l in labels]
    if len({v[0] for v in vals}) > 1:
        print(f"  {k:55}" + "".join(f"{f'{v[0]}/{v[1]}':>8}" for v in vals))

print("\nSpeed:")
for l, rows in data.items():
    perf = [r for r in rows if r["suite"] == "perf"]
    tps = [r.get("tps") for r in perf if r.get("tps")]
    allt = [r.get("tps") for r in rows if r.get("tps") and r.get("out_tokens", 0) > 100]
    print(f"  {l}: decode (perf) {[round(x, 1) for x in tps]}; median decode over all >100-token replies "
          f"{st.median(allt):.1f} tok/s (n={len(allt)})" if allt else f"  {l}: no tps")
    for r in rows:
        if r["suite"] == "longctx":
            print(f"    longctx {r['size'] // 1000:>3}K ok={r['ok']} prompt={r['prompt_tokens']} ttft={r.get('ttft')} "
                  f"prefill={r.get('prefill_tps') and round(r['prefill_tps'])} tok/s decode={r.get('tps') and round(r['tps'])}")
        if r["suite"] == "conc" and "n" in r:
            print(f"    conc {r['n']}: aggregate {r['agg_tps']:.1f} tok/s, per-request {[round(x or 0) for x in r['per_tps']]}")
        if r["suite"] == "conc" and r.get("task") == "head-of-line":
            print(f"    head-of-line: checks {r['check_secs']} s during big prefill ({r['big_secs']:.1f}s)")
