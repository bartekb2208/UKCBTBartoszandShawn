"""Features, computed on one split's own 7-day graph. Never uses wallet IDs as features.

What the training data showed (see README "Findings"):
- 700 busy wallets; planted scenarios are bursts of ONE-OFF pairs inside normal recurring commerce
- every shape has legitimate decoys; easy cases run ~60 s between steps, decoys ~10 min, medium ~18 min
- so the core idea is to stitch one-off payments into time clusters and describe each cluster's rhythm
"""
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C

B = np.int64(10**10)


def _codes(e):
    W = pd.Index(pd.unique(np.r_[e["src"].values, e["dst"].values]))
    return W.get_indexer(e["src"]), W.get_indexer(e["dst"])


def clusters(t, s, d, rare, D):
    """Union one-off payments that touch the same wallet within D seconds (input sorted by time)."""
    n = len(t)
    par = np.arange(n)

    def find(x):
        r = x
        while par[r] != r:
            r = par[r]
        while par[x] != r:
            par[x], x = r, par[x]
        return r

    last = {}
    for i in np.where(rare)[0]:
        for w in (s[i], d[i]):
            j = last.get(w)
            if j is not None and t[i] - t[j] <= D:
                a, b = find(i), find(j)
                if a != b:
                    par[a] = b
            last[w] = i
    root = np.array([find(i) for i in range(n)])
    root[~rare] = -1 - np.where(~rare)[0]   # recurring payments are singletons
    return root


def _cluster_stats(root, t, s, d, usd, asset, amt_rel, prefix):
    c = pd.DataFrame({"r": root, "t": t, "u": usd, "s": s, "d": d, "a": asset, "ar": amt_rel}).sort_values(["r", "t"])
    c["gap"] = c.groupby("r")["t"].diff()
    g, gg = c.groupby("r"), c.groupby("r")["gap"]
    out = pd.DataFrame({
        "size": g.size(), "dur": g["t"].max() - g["t"].min(),
        "gap_med": gg.median(), "gap_max": gg.max(), "gap_min": gg.min(),
        "gap_cv": gg.std() / gg.mean().clip(lower=1),
        "usd_sum": np.log1p(g["u"].sum()), "usd_med": np.log1p(g["u"].median()), "usd_cv": g["u"].std() / g["u"].mean(),
        "n_assets": g["a"].nunique(), "srcs": g["s"].nunique(), "dsts": g["d"].nunique(), "amt_rel": g["ar"].median(),
    })
    out["wal"] = pd.concat([c[["r", "s"]].rename(columns={"s": "w"}),
                            c[["r", "d"]].rename(columns={"d": "w"})]).groupby("r")["w"].nunique()
    out["ew"] = out["size"] / out["wal"]
    out["max_out"] = c.groupby(["r", "s"]).size().groupby("r").max()
    out["max_in"] = c.groupby(["r", "d"]).size().groupby("r").max()
    out["decay"] = g["u"].apply(lambda x: np.mean(np.diff(x.values) < 0) if len(x) > 1 else 0.5)
    return pd.DataFrame({"r": root}).join(out.add_prefix(prefix), on="r").drop(columns="r")


def _flow(s, d, t, a, mask, lab):
    """Was the sender funded just before? Does the receiver pass money on just after? (whole week)"""
    F = {}
    order_i = np.argsort(d[mask].astype(np.int64) * B + t[mask], kind="stable")
    ki = (d[mask].astype(np.int64) * B + t[mask])[order_i]; ai = a[mask][order_i]; cs_i = np.r_[0, np.cumsum(ai)]
    order_o = np.argsort(s[mask].astype(np.int64) * B + t[mask], kind="stable")
    ko = (s[mask].astype(np.int64) * B + t[mask])[order_o]; ao = a[mask][order_o]; cs_o = np.r_[0, np.cumsum(ao)]
    p = np.searchsorted(ki, s.astype(np.int64) * B + t, "left") - 1
    ok = (p >= 0) & ((ki[np.maximum(p, 0)] // B) == s)
    F[f"{lab}_back_dt"] = np.where(ok, t - ki[np.maximum(p, 0)] % B, 1e6)
    F[f"{lab}_back_amt_ratio"] = np.where(ok, a / np.maximum(ai[np.maximum(p, 0)], 1e-9), 0)
    p = np.searchsorted(ko, d.astype(np.int64) * B + t, "right")
    ok = (p < len(ko)) & ((ko[np.minimum(p, len(ko) - 1)] // B) == d)
    F[f"{lab}_fwd_dt"] = np.where(ok, ko[np.minimum(p, len(ko) - 1)] % B - t, 1e6)
    F[f"{lab}_fwd_amt_ratio"] = np.where(ok, ao[np.minimum(p, len(ko) - 1)] / np.maximum(a, 1e-9), 0)
    for H in C.FLOW_WINDOWS:
        lo = np.searchsorted(ki, s.astype(np.int64) * B + t - H, "left"); hi = np.searchsorted(ki, s.astype(np.int64) * B + t, "left")
        F[f"{lab}_in_before_{H // 3600}h"] = (cs_i[hi] - cs_i[lo]) / np.maximum(a, 1e-9)
        lo = np.searchsorted(ko, d.astype(np.int64) * B + t, "right"); hi = np.searchsorted(ko, d.astype(np.int64) * B + t + H, "right")
        F[f"{lab}_out_after_{H // 3600}h"] = (cs_o[hi] - cs_o[lo]) / np.maximum(a, 1e-9)
    return F


def _pair_match(s, d, t, pair_n, H=C.RELAY_HANDOFF_S):
    """Relay signature: relationships w->u and u->x that repeat, with u passing money on within H each round."""
    pair = s.astype(np.int64) * 100000 + d
    ok = (pair_n >= 2) & (pair_n <= 6)
    times = defaultdict(list)
    for i in np.where(ok)[0]:
        times[pair[i]].append(t[i])
    times = {k: np.sort(np.array(v)) for k, v in times.items()}
    ins, outs = defaultdict(list), defaultdict(list)
    for p in times:
        outs[p // 100000].append(p); ins[p % 100000].append(p)
    up, upl, dn, dnl = defaultdict(int), {}, defaultdict(int), {}
    for u in set(ins) & set(outs):
        for p1 in ins[u]:
            T1 = times[p1]
            for p2 in outs[u]:
                if p1 // 100000 == p2 % 100000:
                    continue
                T2 = times[p2]
                j = np.searchsorted(T1, T2, "left") - 1
                lag = np.where(j >= 0, T2 - T1[np.maximum(j, 0)], np.inf)
                m = lag <= H
                k = int(m.sum())
                if k == 0:
                    continue
                if k > up[p2]:
                    up[p2], upl[p2] = k, float(np.median(lag[m]))
                if k > dn[p1]:
                    dn[p1], dnl[p1] = k, float(np.median(lag[m]))
    ps = pd.Series(pair)
    return {"pm_up_rounds": ps.map(up).fillna(0).values, "pm_up_lag": ps.map(upl).fillna(1e6).values,
            "pm_dn_rounds": ps.map(dn).fillna(0).values, "pm_dn_lag": ps.map(dnl).fillna(1e6).values}


def _ego_features(t, s, d, usd, rare, windows=(180,600,1800,3600), asset=None):
    """For each payment: the one-off payments within 2 wallet-hops and +-D seconds (no transitive chaining).
    Describes the local burst: size, wallets, rhythm, value, shape."""
    n=len(t)
    byw=defaultdict(list)
    for i in np.where(rare)[0]:
        byw[s[i]].append(i); byw[d[i]].append(i)
    byw={w:np.array(v) for w,v in byw.items()}   # indices already in time order (input sorted by t)
    tw={w:t[v] for w,v in byw.items()}
    out={}
    for D in windows:
        size=np.zeros(n); wal=np.zeros(n); gap_med=np.full(n,np.nan); gap_cv=np.full(n,np.nan); span=np.zeros(n)
        usd_cv=np.full(n,np.nan); maxdeg=np.zeros(n); hop1=np.zeros(n); amt_ratio=np.full(n,np.nan)
        for i in range(n):
            ti=t[i]
            h1=set()
            for w in (s[i],d[i]):
                v=byw.get(w)
                if v is None: continue
                tt=tw[w]; lo,hi=np.searchsorted(tt,ti-D,'left'),np.searchsorted(tt,ti+D,'right')
                h1.update(v[lo:hi].tolist())
            h1.discard(i)
            hop1[i]=len(h1)
            S=set(h1); S.add(i)
            for j in list(h1):
                for w in (s[j],d[j]):
                    if w==s[i] or w==d[i]: continue
                    v=byw.get(w)
                    if v is None: continue
                    tt=tw[w]; lo,hi=np.searchsorted(tt,max(t[j]-D,ti-2*D),'left'),np.searchsorted(tt,min(t[j]+D,ti+2*D),'right')
                    S.update(v[lo:hi].tolist())
            idx=np.fromiter(S,int)
            size[i]=len(idx)
            if len(idx)>1:
                ts=np.sort(t[idx]); g=np.diff(ts)
                gap_med[i]=np.median(g); gap_cv[i]=g.std()/max(g.mean(),1); span[i]=ts[-1]-ts[0]
                u=usd[idx]; usd_cv[i]=u.std()/u.mean(); amt_ratio[i]=usd[i]/np.median(u)
                ws=np.r_[s[idx],d[idx]]; wal[i]=len(np.unique(ws)); maxdeg[i]=np.bincount(np.unique(ws,return_inverse=True)[1]).max()
            else:
                wal[i]=2
        p=f'e{D}_'
        out.update({p+'size':size,p+'hop1':hop1,p+'wal':wal,p+'gap_med':gap_med,p+'gap_cv':gap_cv,p+'span':span,
                    p+'usd_cv':usd_cv,p+'maxdeg':maxdeg,p+'amt_ratio':amt_ratio,p+'density':size/np.maximum(span,60),p+'ew':size/np.maximum(wal,1)})
    return pd.DataFrame(out)



def build_features(e: pd.DataFrame):
    """e: output of load.load_split (sorted by time). Returns (F, ctx); ctx feeds propagation."""
    n = len(e)
    s, d = _codes(e)
    t = e["t"].values.astype(np.int64)
    usd = e["usd"].values
    F = {}

    # ---- the payment
    F["log_usd"] = np.log1p(usd)
    F["log_usd_paid"] = np.log1p(e["usd_paid"].values)
    F["spread"] = np.log((e["usd_paid"].values + 1e-9) / (usd + 1e-9))
    for k in ("XRP", "GBP", "EUR", "USD"):
        F[f"out_{k}"] = e["asset_out"].str.endswith(k).astype(int).values
        F[f"in_{k}"] = e["asset_in"].str.endswith(k).astype(int).values
    for c in ("is_conv", "n_legs", "n_assets", "is_path", "has_sendmax", "has_paths"):
        F[c] = e[c].values
    F["hour"] = (t % 86400) // 3600
    pair = pd.Series(s.astype(np.int64) * 100000 + d)
    pc = pair.value_counts()
    pair_n = pair.map(pc).values
    F["pair_n"] = pair_n
    F["pair_rank"] = pair.groupby(pair).cumcount().values
    F["rev_n"] = pd.Series(d.astype(np.int64) * 100000 + s).map(pc).fillna(0).values

    # ---- the wallets over the week (out of character?)
    wl = pd.DataFrame({"w": np.r_[s, d], "u": np.r_[usd, usd], "p": np.r_[pair_n, pair_n]})
    wg = wl.groupby("w")
    wmed, wdeg, wrare = wg["u"].median(), wg.size(), wg["p"].apply(lambda x: (x <= C.RARE_PAIR_MAX).mean())
    for side, w in (("s", s), ("d", d)):
        F[f"{side}_amt_rel"] = np.log(usd / wmed.values[w])
        F[f"{side}_deg"] = wdeg.values[w]
        F[f"{side}_rare_share"] = wrare.values[w]

    # ---- local timing around each wallet: all payments and one-off payments
    own = np.r_[s, d]; tt = np.r_[t, t]; idx = np.r_[np.arange(n), np.arange(n)]
    o = np.lexsort((tt, own)); key = own[o].astype(np.int64) * B + tt[o]
    rare = pair_n <= C.RARE_PAIR_MAX
    rk = key[rare[idx[o]]]

    def cnt(k, w, lo, hi):
        return np.searchsorted(k, w.astype(np.int64) * B + hi, "right") - np.searchsorted(k, w.astype(np.int64) * B + lo, "left")
    for side, w in (("s", s), ("d", d)):
        for win in C.LOCAL_WINDOWS:
            F[f"{side}_any_{win}"] = cnt(key, w, t - win, t + win) - 1
            F[f"{side}_rare_{win}"] = cnt(rk, w, t - win, t + win) - rare

    # ---- time clusters of one-off payments (scenario reconstruction)
    F = pd.DataFrame(F)
    amt_rel = np.maximum(F["s_amt_rel"].values, F["d_amt_rel"].values)
    roots = {D: clusters(t, s, d, rare, D) for D in C.PROP_WINDOWS}
    for D in C.CLUSTER_WINDOWS:
        F = pd.concat([F, _cluster_stats(roots[D], t, s, d, usd, e["asset_out"].values, amt_rel, f"c{D}_")], axis=1)
    F = pd.concat([F, _ego_features(t, s, d, usd, rare, C.EGO_WINDOWS)], axis=1)

    # ---- money flow in and out of each wallet (layering / pass-through)
    extra = {}
    extra.update(_flow(s, d, t, usd, np.ones(n, bool), "all"))
    extra.update(_flow(s, d, t, usd, pair_n <= 4, "rare"))
    extra.update(_pair_match(s, d, t, pair_n))
    F = pd.concat([F, pd.DataFrame(extra)], axis=1)
    ctx = {"s": s, "d": d, "t": t, "roots": roots}
    return F.astype(float), ctx


def propagation_features(p: np.ndarray, ctx: dict) -> pd.DataFrame:
    """Neighbours' scores: same time cluster, and payments touching the same wallet within 10 min / 1 h."""
    s, d, t = ctx["s"], ctx["d"], ctx["t"]
    P = {}
    for D, r in ctx["roots"].items():
        g = pd.Series(p).groupby(r)
        sm, ct, mx = g.transform("sum").values, g.transform("count").values, g.transform("max").values
        P[f"p{D}_mean_oth"] = np.where(ct > 1, (sm - p) / np.maximum(ct - 1, 1), np.nan)
        P[f"p{D}_max"] = mx
    own = np.r_[s, d]; tt = np.r_[t, t]; pp = np.r_[p, p]
    o = np.lexsort((tt, own)); key = own[o].astype(np.int64) * B + tt[o]; cs = np.r_[0, np.cumsum(pp[o])]
    for side, w in (("s", s), ("d", d)):
        for H in (600, 3600):
            lo = np.searchsorted(key, w.astype(np.int64) * B + t - H, "left")
            hi = np.searchsorted(key, w.astype(np.int64) * B + t + H, "right")
            P[f"p_{side}_{H}_mean"] = (cs[hi] - cs[lo] - p) / np.maximum(hi - lo - 1, 1)
    return pd.DataFrame(P)
