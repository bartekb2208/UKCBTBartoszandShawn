"""Learned scenario reconstruction (trained on the scenario_id annotations in labels.csv; no labels used at test time).
Candidate links: two one-off payments touching the same wallet within W seconds (next K events at that wallet).
A LightGBM link model (trained on scenario_id annotations) scores each candidate; union-find over confident links
gives reconstructed scenarios, which are then described as a whole (rhythm, flow conservation, shape)."""
import numpy as np, pandas as pd, lightgbm as lgb
from collections import defaultdict

def candidates(t, s, d, usd, pair_n, conv, asset, deg, W=4*3600, K=4, pmax=3):
    n=len(t); cand=pair_n<=pmax
    ev=defaultdict(list)
    for i in np.where(cand)[0]:
        ev[s[i]].append((i,0)); ev[d[i]].append((i,1))   # role 0 = sender at w, 1 = receiver at w
    rows=[]
    for w,lst in ev.items():
        # lst is in time order because i is time-sorted
        m=len(lst)
        for a in range(m):
            i,ri=lst[a]
            for b in range(a+1,min(m,a+1+K)):
                j,rj=lst[b]
                if t[j]-t[i]>W: break
                rows.append((i,j,w,ri,rj,b-a))
    P=pd.DataFrame(rows,columns=['i','j','w','ri','rj','k']).drop_duplicates(['i','j'])
    i,j=P.i.values,P.j.values
    X=pd.DataFrame({
        'dt':t[j]-t[i], 'logdt':np.log1p(t[j]-t[i]), 'k':P.k.values,
        'flow':((P.ri==1)&(P.rj==0)).astype(int).values,      # money arrives at w, then leaves w
        'fanout':((P.ri==0)&(P.rj==0)).astype(int).values, 'fanin':((P.ri==1)&(P.rj==1)).astype(int).values,
        'back':((P.ri==0)&(P.rj==1)).astype(int).values,
        'amt_ratio':np.log((usd[j]+1e-9)/(usd[i]+1e-9)),'abs_amt_ratio':np.abs(np.log((usd[j]+1e-9)/(usd[i]+1e-9))),
        'usd_i':np.log1p(usd[i]),'usd_j':np.log1p(usd[j]),
        'pn_i':pair_n[i],'pn_j':pair_n[j],'conv_i':conv[i],'conv_j':conv[j],
        'same_asset':(asset[i]==asset[j]).astype(int),'deg_w':deg[P.w.values],
        'deg_other_i':deg[np.where(P.ri.values==0,d[i],s[i])],'deg_other_j':deg[np.where(P.rj.values==0,d[j],s[j])],
        'same_other':(np.where(P.ri.values==0,d[i],s[i])==np.where(P.rj.values==0,d[j],s[j])).astype(int),
    })
    return P[['i','j']].reset_index(drop=True), X

def union(n, i, j):
    par=np.arange(n)
    def find(x):
        r=x
        while par[r]!=r: r=par[r]
        while par[x]!=r: par[x],x=r,par[x]
        return r
    for a,b in zip(i,j):
        ra,rb=find(a),find(b)
        if ra!=rb: par[ra]=rb
    return np.array([find(x) for x in range(n)])

def scenario_stats(root, t, s, d, usd, conv, prefix, linkp=None):
    c=pd.DataFrame({'r':root,'t':t,'u':usd,'s':s,'d':d,'c':conv}).sort_values(['r','t'])
    c['gap']=c.groupby('r').t.diff()
    g=c.groupby('r'); gg=g.gap
    out=pd.DataFrame({'size':g.size(),'dur':g.t.max()-g.t.min(),'gap_med':gg.median(),'gap_max':gg.max(),'gap_min':gg.min(),
        'gap_cv':gg.std()/gg.mean().clip(lower=1),'usd_sum':np.log1p(g.u.sum()),'usd_cv':g.u.std()/g.u.mean(),'conv':g.c.mean(),
        'srcs':g.s.nunique(),'dsts':g.d.nunique()})
    ins=c.groupby(['r','d']).u.sum(); outs=c.groupby(['r','s']).u.sum()
    ins.index.names=['r','w']; outs.index.names=['r','w']
    j=pd.concat([ins.rename('i'),outs.rename('o')],axis=1).dropna()
    j['lr']=np.abs(np.log(j.o/j.i)); j['ratio']=j.o/j.i
    out['n_inter']=j.groupby('r').size(); out['cons_dev']=j.groupby('r').lr.median(); out['cons_ratio']=j.groupby('r').ratio.median()
    w_in=c.groupby('r').d.apply(set); w_out=c.groupby('r').s.apply(set)
    out['n_source']=[len(a-b) for a,b in zip(w_out,w_in)]; out['n_sink']=[len(b-a) for a,b in zip(w_out,w_in)]
    out['max_out']=c.groupby(['r','s']).size().groupby('r').max(); out['max_in']=c.groupby(['r','d']).size().groupby('r').max()
    wal=pd.concat([c[['r','s']].rename(columns={'s':'w'}),c[['r','d']].rename(columns={'d':'w'})]).groupby('r').w.nunique()
    out['wal']=wal; out['ew']=out['size']/wal
    if linkp is not None: out['linkp']=pd.Series(linkp).groupby(root).mean()
    out.loc[out['size']==1]=np.nan; out.loc[out['size']==1,'size']=1
    return pd.DataFrame({'r':root}).join(out.add_prefix(prefix),on='r').drop(columns='r')


LINK_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1, n_jobs=2)


def _node_and_scenario_feats(n, P, pl, t, s, d, usd, conv, thresholds=(0.7, 0.8, 0.9)):
    i, j = P.i.values, P.j.values
    L = pd.DataFrame({"a": np.r_[i, j], "p": np.r_[pl, pl]})
    g = L.groupby("a").p
    NF = pd.DataFrame(index=range(n))
    NF["lk_max"] = g.max(); NF["lk_sum"] = g.sum()
    NF["lk_n50"] = L[L.p > 0.5].groupby("a").size(); NF["lk_n80"] = L[L.p > 0.8].groupby("a").size()
    NF = NF.fillna(0)
    S = [NF]
    for th in thresholds:
        k = pl > th
        root = union(n, i[k], j[k])
        lp = pd.Series(np.r_[pl[k], pl[k]]).groupby(np.r_[i[k], j[k]]).mean().reindex(range(n)).values
        S.append(scenario_stats(root, t, s, d, usd, conv, f"sc{int(th * 100)}_", linkp=lp))
    return pd.concat(S, axis=1)


def link_features(tr, ctx_tr, F_tr, scenario_id, te, ctx_te, F_te, seed=42):
    """Train the link model on train (day-blocked out-of-fold for train rows), apply to test."""
    out = []
    deg_tr = np.bincount(np.r_[ctx_tr["s"], ctx_tr["d"]])
    deg_te = np.bincount(np.r_[ctx_te["s"], ctx_te["d"]])
    Ptr, Xtr = candidates(ctx_tr["t"], ctx_tr["s"], ctx_tr["d"], tr.usd.values, F_tr.pair_n.values,
                          tr.is_conv.values, tr.asset_out.values, deg_tr)
    Pte, Xte = candidates(ctx_te["t"], ctx_te["s"], ctx_te["d"], te.usd.values, F_te.pair_n.values,
                          te.is_conv.values, te.asset_out.values, deg_te)
    sc = np.asarray(scenario_id, dtype=object)
    yl = np.array([isinstance(sc[a], str) and sc[a] == sc[b] for a, b in zip(Ptr.i, Ptr.j)]).astype(int)
    fold = (ctx_tr["t"][Ptr.i.values] // 86400) % 5
    pl_tr = np.zeros(len(Ptr))
    for f in range(5):
        m = lgb.LGBMClassifier(**LINK_PARAMS, random_state=seed).fit(Xtr[fold != f], yl[fold != f])
        pl_tr[fold == f] = m.predict_proba(Xtr[fold == f])[:, 1]
    m = lgb.LGBMClassifier(**LINK_PARAMS, random_state=seed).fit(Xtr, yl)
    pl_te = m.predict_proba(Xte)[:, 1]
    print(f"[link] {len(Ptr):,} train candidate links ({yl.mean():.1%} same-scenario), {len(Pte):,} test")
    Str = _node_and_scenario_feats(len(tr), Ptr, pl_tr, ctx_tr["t"], ctx_tr["s"], ctx_tr["d"], tr.usd.values, tr.is_conv.values)
    Ste = _node_and_scenario_feats(len(te), Pte, pl_te, ctx_te["t"], ctx_te["s"], ctx_te["d"], te.usd.values, te.is_conv.values)
    return Str.astype(float), Ste.astype(float)
