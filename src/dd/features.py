"""Behavioural features per transaction, computed once over a whole split.

No wallet address, issuer address or split identity enters the features.
"""
import re

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

WEEK = 7 * 86400
# The organiser README: "Asset and conversion choices do not inspect suspicious labels", and
# test/ converts more often than train/. Features built only from assets or conversions are dropped.
ASSET_FEATURES = r"legs|converts|spread|cur_.*|cost_cur_same|fee|.*_conv|.*_ncur|.*samecur.*|.*currun.*|units_whole"
MAX_CLUSTER = 60
WINDOWS = {"5m": 300, "30m": 1800, "2h": 7200, "12h": 43200}


class Wallets:
    """Per-wallet sorted send and receive events."""

    def __init__(self, df):
        self.out, self.inn = {}, {}
        for side, key, cp in ((self.out, "src", "dst"), (self.inn, "dst", "src")):
            for w, g in df.groupby(key, sort=False):
                g = g.sort_values("ts", kind="stable")
                side[w] = dict(t=g.ts.to_numpy(), v=g.usd.to_numpy(), cp=g[cp].to_numpy(),
                               cur=g.currency.to_numpy(), i=g.index.to_numpy())

    def get(self, side, w):
        return (self.out if side == "out" else self.inn).get(w)


def _slice(ev, lo, hi, lo_inc=True, hi_inc=True):
    if ev is None:
        return slice(0, 0)
    a = np.searchsorted(ev["t"], lo, "left" if lo_inc else "right")
    b = np.searchsorted(ev["t"], hi, "right" if hi_inc else "left")
    return slice(a, b)


def _cv(x):
    return np.std(x) / np.mean(x) if len(x) > 1 and np.mean(x) > 0 else -1.0


def window_features(df, W):
    n = len(df)
    feats = {}
    t, v, s, r, cur = df.ts.to_numpy(), df.usd.to_numpy(), df.src.to_numpy(), df.dst.to_numpy(), df.currency.to_numpy()
    idx = df.index.to_numpy()

    def put(name, i, val):
        feats.setdefault(name, np.full(n, np.nan))[i] = val

    for i in range(n):
        ti, vi = t[i], v[i]
        so, si, ro, ri = W.get("out", s[i]), W.get("in", s[i]), W.get("out", r[i]), W.get("in", r[i])
        for wn, w in WINDOWS.items():
            # Fan-out around this payment: the sender's other sends within ±w.
            sl = _slice(so, ti - w, ti + w)
            vals = so["v"][sl]
            put(f"s_out_{wn}", i, len(vals) - 1)
            put(f"s_out_cp_{wn}", i, len(set(so["cp"][sl])))
            put(f"s_out_cv_{wn}", i, _cv(vals))
            put(f"s_out_sum_{wn}", i, vals.sum() / vi)
            put(f"s_out_samecur_{wn}", i, np.mean(so["cur"][sl] == cur[i]))
            # Fan-in: the receiver's other receipts within ±w.
            sl = _slice(ri, ti - w, ti + w)
            vals = ri["v"][sl]
            put(f"r_in_{wn}", i, len(vals) - 1)
            put(f"r_in_cp_{wn}", i, len(set(ri["cp"][sl])))
            put(f"r_in_cv_{wn}", i, _cv(vals))
            put(f"r_in_sum_{wn}", i, vals.sum() / vi)
            put(f"r_in_samecur_{wn}", i, np.mean(ri["cur"][sl] == cur[i]))
            # Pass-through: receiver sends on after; sender was funded before.
            sl = _slice(ro, ti, ti + w, lo_inc=True)
            vals = ro["v"][sl] if ro is not None else np.empty(0)
            put(f"r_fwd_n_{wn}", i, len(vals))
            put(f"r_fwd_sum_{wn}", i, vals.sum() / vi)
            put(f"r_fwd_best_{wn}", i, np.min(np.abs(vals / vi - 1)) if len(vals) else 9.0)
            sl = _slice(si, ti - w, ti, hi_inc=True)
            vals = si["v"][sl] if si is not None else np.empty(0)
            put(f"s_fund_n_{wn}", i, len(vals))
            put(f"s_fund_sum_{wn}", i, vals.sum() / vi)
            put(f"s_fund_best_{wn}", i, np.min(np.abs(vals / vi - 1)) if len(vals) else 9.0)
            # Receiver's receipts and sender's sends on the other side of the payment.
            sl = _slice(ri, ti - w, ti, hi_inc=False)
            put(f"r_in_before_{wn}", i, sl.stop - sl.start)
            sl = _slice(so, ti, ti + w, lo_inc=False)
            put(f"s_out_after_{wn}", i, sl.stop - sl.start)

        # Gaps to the nearest neighbouring events (log seconds).
        for name, ev, before in (("r_next_out", ro, False), ("s_prev_in", si, True), ("s_prev_out", so, True),
                                 ("s_next_out", so, False), ("r_prev_in", ri, True), ("r_next_in", ri, False)):
            if ev is None:
                gap = WEEK
            else:
                tt = ev["t"]
                if before:
                    k = np.searchsorted(tt, ti, "left") - 1
                    gap = ti - tt[k] if k >= 0 else WEEK
                else:
                    k = np.searchsorted(tt, ti, "right")
                    gap = tt[k] - ti if k < len(tt) else WEEK
            put(f"gap_{name}", i, np.log1p(gap))
    return pd.DataFrame(feats, index=df.index)


def wallet_week_features(df):
    """Whole-week activity of each endpoint (behaviour, not identity)."""
    f = pd.DataFrame(index=df.index)
    out_n = df.src.value_counts(); in_n = df.dst.value_counts()
    out_v = df.groupby("src").usd.sum(); in_v = df.groupby("dst").usd.sum()
    out_cp = df.groupby("src").dst.nunique(); in_cp = df.groupby("dst").src.nunique()
    for role, col in (("s", "src"), ("r", "dst")):
        w = df[col]
        f[f"{role}_wk_out_n"] = w.map(out_n).fillna(0).to_numpy()
        f[f"{role}_wk_in_n"] = w.map(in_n).fillna(0).to_numpy()
        f[f"{role}_wk_out_cp"] = w.map(out_cp).fillna(0).to_numpy()
        f[f"{role}_wk_in_cp"] = w.map(in_cp).fillna(0).to_numpy()
        f[f"{role}_wk_pass"] = (w.map(out_v).fillna(0) / (w.map(in_v).fillna(0) + 1)).clip(0, 10).to_numpy()
    pair = df.groupby(["src", "dst"]).size()
    f["pair_n"] = pd.MultiIndex.from_arrays([df.src, df.dst]).map(pair).to_numpy()
    rev = pd.MultiIndex.from_arrays([df.dst, df.src]).map(pair)
    f["pair_rev_n"] = pd.Series(rev).fillna(0).to_numpy()
    return f


def tx_features(df):
    f = pd.DataFrame(index=df.index)
    f["log_usd"] = np.log10(df.usd)
    f["legs"] = df.legs
    f["converts"] = df.converts.astype(float)
    f["spread"] = df.spread
    for c in ("XRP", "USD", "GBP", "EUR"):
        f[f"cur_{c}"] = (df.currency == c).astype(float)
    f["cost_cur_same"] = (df.cost_currency == df.currency).astype(float)
    f["fee"] = df.fee_drops
    frac = np.round(df.units * 1e6).astype(np.int64) % 1_000_000
    f["units_whole"] = (frac == 0).astype(float)
    f["usd_round10"] = (np.abs(df.usd - np.round(df.usd, -1)) < 0.5).astype(float)
    f["hour_of_day"] = (df.ts % 86400) / 3600
    return f


def motif_clusters(df, gap):
    """Link payments that touch the same wallet within `gap` seconds; return component id."""
    ev = pd.concat([pd.DataFrame({"w": df.src, "t": df.ts, "i": np.arange(len(df))}),
                    pd.DataFrame({"w": df.dst, "t": df.ts, "i": np.arange(len(df))})]).sort_values(["w", "t"])
    same = (ev.w.to_numpy()[1:] == ev.w.to_numpy()[:-1]) & (np.diff(ev.t.to_numpy()) <= gap)
    a, b = ev.i.to_numpy()[:-1][same], ev.i.to_numpy()[1:][same]
    n = len(df)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def cluster_features(df):
    f = pd.DataFrame(index=df.index)
    for name, gap in (("c5m", 300), ("c20m", 1200)):
        cid = motif_clusters(df, gap)
        g = df.assign(cid=cid)
        grp = g.groupby("cid")
        size = grp.size()
        wallets = pd.concat([g[["cid", "src"]].rename(columns={"src": "w"}), g[["cid", "dst"]].rename(columns={"dst": "w"})]).groupby("cid").w.nunique()
        span = grp.ts.max() - grp.ts.min()
        vcv = grp.usd.std() / grp.usd.mean()
        ncur = grp.currency.nunique()
        conv = grp.converts.mean()
        srcs = grp.src.nunique(); dsts = grp.dst.nunique()
        f[f"{name}_size"] = g.cid.map(size).to_numpy()
        f[f"{name}_wallets"] = g.cid.map(wallets).to_numpy()
        f[f"{name}_span"] = np.log1p(g.cid.map(span).to_numpy())
        f[f"{name}_rate"] = f[f"{name}_size"] / (g.cid.map(span).to_numpy() / 60 + 1)
        f[f"{name}_vcv"] = g.cid.map(vcv).fillna(-1).to_numpy()
        f[f"{name}_ncur"] = g.cid.map(ncur).to_numpy()
        f[f"{name}_conv"] = g.cid.map(conv).to_numpy()
        f[f"{name}_src_ratio"] = (g.cid.map(srcs) / g.cid.map(size)).to_numpy()
        f[f"{name}_dst_ratio"] = (g.cid.map(dsts) / g.cid.map(size)).to_numpy()
        f[f"{name}_w_per_tx"] = f[f"{name}_wallets"] / f[f"{name}_size"]
        f[f"{name}_pos"] = ((g.ts - g.cid.map(grp.ts.min())) / (g.cid.map(span) + 1)).to_numpy()
        # A giant cluster's statistics describe the whole split, not a motif.
        giant = f[f"{name}_size"].to_numpy() > MAX_CLUSTER
        cols = [c for c in f.columns if c.startswith(name + "_")]
        f.loc[giant, cols] = np.nan
        f[f"{name}_giant"] = giant.astype(float)
    return f


def chain_features(df, gap=1800, tol=0.25):
    """Follow similar-value hops forwards and backwards from each payment."""
    W = Wallets(df)
    t, v, s, r = df.ts.to_numpy(), df.usd.to_numpy(), df.src.to_numpy(), df.dst.to_numpy()
    n = len(df)
    nxt = np.full(n, -1); prv = np.full(n, -1)
    for i in range(n):
        ro = W.get("out", r[i])
        if ro is not None:
            sl = _slice(ro, t[i], t[i] + gap, lo_inc=False)
            vals = ro["v"][sl]
            if len(vals):
                k = np.argmin(np.abs(vals / v[i] - 1))
                if abs(vals[k] / v[i] - 1) <= tol:
                    nxt[i] = ro["i"][sl][k]
    for i in range(n):
        if nxt[i] >= 0 and prv[nxt[i]] < 0:
            prv[nxt[i]] = i
    down = np.zeros(n, int); up = np.zeros(n, int)
    for i in np.argsort(-t):
        if nxt[i] >= 0:
            down[i] = down[nxt[i]] + 1
    for i in np.argsort(t):
        if prv[i] >= 0:
            up[i] = up[prv[i]] + 1
    f = pd.DataFrame(index=df.index)
    f["chain_down"] = down; f["chain_up"] = up; f["chain_len"] = down + up
    # Value kept along the forward chain and whether it returns to the start.
    ret = np.zeros(n); kept = np.ones(n)
    for i in range(n):
        j, hops, seen = nxt[i], 0, {s[i]}
        while j >= 0 and hops < 12:
            hops += 1
            if r[j] == s[i]:
                ret[i] = hops + 1
                break
            j = nxt[j]
        last = i
        while nxt[last] >= 0 and last != nxt[last]:
            last = nxt[last]
            if hops > 12:
                break
        kept[i] = v[last] / v[i]
    f["chain_return"] = ret
    f["chain_kept"] = kept
    return f


def cycle_features(df, w=6 * 3600):
    """Money returns to the sender within w via 2 or 3 hops."""
    W = Wallets(df)
    t, s, r = df.ts.to_numpy(), df.src.to_numpy(), df.dst.to_numpy()
    n = len(df)
    c2 = np.zeros(n); c3 = np.zeros(n)
    for i in range(n):
        ro = W.get("out", r[i])
        if ro is None:
            continue
        sl = _slice(ro, t[i], t[i] + w)
        cps, ts_ = ro["cp"][sl], ro["t"][sl]
        if np.any(cps == s[i]):
            c2[i] = 1
        for x, tx in zip(cps, ts_):
            xo = W.get("out", x)
            if xo is None:
                continue
            sl2 = _slice(xo, tx, t[i] + w)
            if np.any(xo["cp"][sl2] == s[i]):
                c3[i] = 1
                break
    return pd.DataFrame({"cyc2": c2, "cyc3": c3}, index=df.index)


def shape_features(df, W):
    """Regularity, novelty and same-asset runs of the burst around a payment, at longer windows."""
    n = len(df)
    t, v, s, r, cur = df.ts.to_numpy(), df.usd.to_numpy(), df.src.to_numpy(), df.dst.to_numpy(), df.currency.to_numpy()
    pair = df.groupby(["src", "dst"]).size()
    pair_n = {k: c for k, c in pair.items()}
    med_out = df.groupby("src").usd.median(); med_in = df.groupby("dst").usd.median()
    feats = {}

    def put(name, i, val):
        feats.setdefault(name, np.full(n, np.nan))[i] = val

    for i in range(n):
        ti = t[i]
        for side, wallet, role in (("in", r[i], "r_in"), ("out", s[i], "s_out")):
            ev = W.get(side, wallet)
            for wn, w in (("30m", 1800), ("2h", 7200), ("6h", 21600)):
                sl = _slice(ev, ti - w, ti + w)
                tt, cps, cc = ev["t"][sl], ev["cp"][sl], ev["cur"][sl]
                gaps = np.diff(tt)
                put(f"{role}_gapcv_{wn}", i, _cv(gaps) if len(gaps) > 1 else -1.0)
                put(f"{role}_gapmed_{wn}", i, np.log1p(np.median(gaps)) if len(gaps) else -1.0)
                if side == "in":
                    nov = [pair_n[(c, wallet)] == 1 for c in cps]
                else:
                    nov = [pair_n[(wallet, c)] == 1 for c in cps]
                put(f"{role}_novel_{wn}", i, np.mean(nov))
                # Run of same-currency events around this payment.
                k = np.searchsorted(tt, ti, "left")
                a = k
                while a > 0 and cc[a - 1] == cur[i]:
                    a -= 1
                b = k
                while b < len(cc) and cc[b] == cur[i]:
                    b += 1
                put(f"{role}_currun_{wn}", i, b - a)
                put(f"{role}_frac_{wn}", i, len(tt) / len(ev["t"]))
    f = pd.DataFrame(feats, index=df.index)
    f["v_vs_s_med"] = np.log(v / df.src.map(med_out).to_numpy())
    f["v_vs_r_med"] = np.log(v / df.dst.map(med_in).fillna(v.mean()).to_numpy())
    f["pair_first"] = (df.groupby(["src", "dst"]).ts.rank(method="first") == 1).astype(float).to_numpy()
    return f


def novel_features(df):
    """Planted motifs are built from first-time pairs; describe each endpoint's first-time-pair
    activity around the payment, and cluster first-time-pair payments at longer gaps."""
    pair = df.groupby(["src", "dst"]).size()
    novel = np.asarray(pd.MultiIndex.from_arrays([df.src, df.dst]).map(pair)) == 1
    sub = df[novel]
    n = len(df)
    ev = {}
    for w, g in pd.concat([sub[["src", "ts", "usd", "currency"]].rename(columns={"src": "w"}),
                           sub[["dst", "ts", "usd", "currency"]].rename(columns={"dst": "w"})]).sort_values("ts").groupby("w"):
        ev[w] = (g.ts.to_numpy(), g.usd.to_numpy(), g.currency.to_numpy())
    t, s, r, cur = df.ts.to_numpy(), df.src.to_numpy(), df.dst.to_numpy(), df.currency.to_numpy()
    feats = {}

    def put(name, i, val):
        feats.setdefault(name, np.full(n, np.nan))[i] = val

    for i in range(n):
        for role, w in (("s", s[i]), ("r", r[i])):
            e = ev.get(w)
            for wn, win in (("30m", 1800), ("1h", 3600), ("3h", 10800)):
                if e is None:
                    put(f"nv_{role}_n_{wn}", i, 0)
                    continue
                a, b = np.searchsorted(e[0], t[i] - win, "left"), np.searchsorted(e[0], t[i] + win, "right")
                tt, vv, cc = e[0][a:b], e[1][a:b], e[2][a:b]
                gaps = np.diff(tt)
                put(f"nv_{role}_n_{wn}", i, len(tt))
                if len(gaps):
                    put(f"nv_{role}_gapmed_{wn}", i, np.log1p(np.median(gaps)))
                    put(f"nv_{role}_gapmin_{wn}", i, np.log1p(gaps.min()))
                    put(f"nv_{role}_span_{wn}", i, np.log1p(tt[-1] - tt[0]))
                    put(f"nv_{role}_gapcv_{wn}", i, _cv(gaps))
                    put(f"nv_{role}_vcv_{wn}", i, _cv(vv))
                    put(f"nv_{role}_samecur_{wn}", i, np.mean(cc == cur[i]))
    f = pd.DataFrame(feats, index=df.index)
    f["novel"] = novel.astype(float)

    # Clusters of first-time-pair payments only.
    sidx = np.flatnonzero(novel)
    for name, gap in (("nc20m", 1200), ("nc40m", 2400)):
        cid = motif_clusters(sub.reset_index(drop=True), gap)
        g = sub.reset_index(drop=True).assign(cid=cid)
        grp = g.groupby("cid")
        size = g.cid.map(grp.size()).to_numpy()
        span = g.cid.map(grp.ts.max() - grp.ts.min()).to_numpy()
        wallets = pd.concat([g[["cid", "src"]].rename(columns={"src": "w"}),
                             g[["cid", "dst"]].rename(columns={"dst": "w"})]).groupby("cid").w.nunique()
        gapmed = grp.ts.apply(lambda x: np.median(np.diff(np.sort(x.to_numpy()))) if len(x) > 1 else np.nan)
        cols = {
            f"{name}_size": size,
            f"{name}_span": np.log1p(span),
            f"{name}_gapmed": np.log1p(g.cid.map(gapmed).to_numpy()),
            f"{name}_w_per_tx": g.cid.map(wallets).to_numpy() / size,
            f"{name}_src_ratio": g.cid.map(grp.src.nunique()).to_numpy() / size,
            f"{name}_dst_ratio": g.cid.map(grp.dst.nunique()).to_numpy() / size,
            f"{name}_vcv": g.cid.map(grp.usd.std() / grp.usd.mean()).to_numpy(),
            f"{name}_ncur": g.cid.map(grp.currency.nunique()).to_numpy(),
            f"{name}_conv": g.cid.map(grp.converts.mean()).to_numpy(),
        }
        giant = size > MAX_CLUSTER
        for c, vals in cols.items():
            col = np.full(n, np.nan)
            col[sidx] = np.where(giant, np.nan, vals)
            f[c] = col
        col = np.zeros(n); col[sidx] = giant; f[f"{name}_giant"] = col
    return f


def prepare(df):
    df = df.copy()
    df["currency"] = df.asset.str.split(":").str[-1]
    df["cost_currency"] = df.cost_asset.str.split(":").str[-1]
    return df


def build_features(df):
    df = prepare(df)
    W = Wallets(df)
    parts = [tx_features(df), wallet_week_features(df), window_features(df, W),
             cluster_features(df), chain_features(df), chain_features(df, 7200, 0.4).add_suffix("_slow"), cycle_features(df), shape_features(df, W), novel_features(df)]
    X = pd.concat(parts, axis=1).copy()
    # Burstiness relative to the wallet's own weekly rate.
    for wn, w in WINDOWS.items():
        X[f"s_out_burst_{wn}"] = X[f"s_out_{wn}"] / (X.s_wk_out_n * 2 * w / WEEK + 0.1)
        X[f"r_in_burst_{wn}"] = X[f"r_in_{wn}"] / (X.r_wk_in_n * 2 * w / WEEK + 0.1)
    # Volume-dependent features become within-split percentiles, so a busier or
    # quieter week does not shift them (validation/ has 15% fewer payments than train/).
    vol = [c for c in X.columns if re.match(r"[sr]_wk_|c20m_(size|wallets|span|rate)XX", c)]
    X[vol] = X[vol].rank(pct=True)
    # Column creation order depends on the data, so fix it for train/test alignment.
    X = X.drop(columns=[c for c in X.columns if re.fullmatch(ASSET_FEATURES, c)])
    return X[sorted(X.columns)].astype(float)
