import csv,collections,statistics as st,sys
r=list(csv.DictReader(open(sys.argv[1],encoding='utf-8')))
def src(x):
    s,u,e=x['system'],x['user'],x['endpoint']
    if e=='/v1/embeddings': return 'QDR-embed'
    if 'curator' in s or 'curator' in u: return 'QDR-curator'
    if 'strict text classifier' in s: return 'SGC-classifier'
    if x['max_tokens'] in ('1024','8192') or 'ticket log below' in s or 'ticket log' in u.lower(): return 'int-local-ai-bench'
    return 'probe/manual'
def f(v):
    try: return float(v)
    except: return None
g=collections.defaultdict(list)
for x in r:
    x['src']=src(x); g[x['src']].append(x)
for k,xs in sorted(g.items()):
    tt=[f(x['total_s']) for x in xs if f(x['total_s']) is not None]
    pt=[f(x['prompt_tokens']) for x in xs if f(x['prompt_tokens'])]
    ct=[f(x['completion_tokens']) for x in xs if f(x['completion_tokens']) is not None]
    ca=[f(x['cached']) for x in xs if f(x['cached']) is not None]; un=[f(x['uncached']) for x in xs if f(x['uncached']) is not None]
    pf=[f(x['prefill_s']) for x in xs if f(x['prefill_s']) is not None]
    dec=[]
    for x in xs:
        T,P,C=f(x['total_s']),f(x['prefill_s']),f(x['completion_tokens'])
        if T and P is not None and C and C>50 and T-P>1: dec.append(C/(T-P))
    fin=collections.Counter(x['finish'] for x in xs)
    empty=sum(1 for x in xs if x['endpoint']!='/v1/embeddings' and x['content_len']=='0' and x['finish'])
    times=sorted(x['time'] for x in xs)
    print(f"\n== {k}: n={len(xs)}  {times[0]} -> {times[-1]}")
    if tt: print(f"  latency s: median {st.median(tt):.1f}  p90 {sorted(tt)[int(.9*len(tt))-1]:.1f}  max {max(tt):.0f}  sum {sum(tt)/60:.1f} min")
    if pt: print(f"  prompt tok: median {st.median(pt):.0f} max {max(pt):.0f}")
    if ca: print(f"  cache: cached {sum(ca):.0f} / total {sum(ca)+sum(un):.0f} = {100*sum(ca)/max(1,sum(ca)+sum(un)):.0f}%  (hits {sum(1 for c in ca if c>0)}/{len(ca)})")
    if pf: print(f"  prefill s median {st.median(pf):.1f} max {max(pf):.0f}")
    if ct: print(f"  completion tok median {st.median(ct):.0f} max {max(ct):.0f}")
    if dec: print(f"  est decode tok/s median {st.median(dec):.0f} (n={len(dec)})")
    print("  finish:",dict(fin)," empty content:",empty, " errors:",sum(1 for x in xs if x['error']))
    print("  params:",dict(collections.Counter((x['max_tokens'],x['reasoning_effort'],x['temperature']) for x in xs).most_common(5)))
