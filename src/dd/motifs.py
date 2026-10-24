"""Observable-hard relay motifs: two label-free temporal-motif detectors.

Nothing here uses labels, scenario or cohort ids, or hashes; only time, USD value and who paid whom, over the
whole seven-day window (offline task, so earlier and later payments are both allowed).

Aggregate relay (per hub H): a compact inbound burst to H, a 3.5-5.5 h delay, then a compact outbound burst from
H, whose total value is roughly conserved (out/in within a tolerant band), with at least three counterparties and
at least one payment on each side; the same structure repeated, with consistent delay and value ratio, on several
separate occasions in the week.

Distributed relay (per directed pair A -> B that repeats): a feeder burst into A ending 1.5-3.5 h before a bridge
payment A -> B and a payout burst from B starting 1.5-3.0 h after it, with feeder -> bridge and bridge -> payout
values both inside the tolerant band; the same structure at several of the pair's bridge payments.

Wallets take part in about twenty relay scenarios at once, so a wallet's bursts are contaminated by other
scenarios. Two physical constraints cut that down: a single payout cannot exceed what the burst took in (and a
feeder or payout cannot exceed the bridge it serves, scaled by the band), and repeats must agree with each other.
Every payment that takes part in a detected motif gets the motif's features (NaN elsewhere).
"""
import numpy as np
import pandas as pd

HOUR = 3600.0
DAY = 86400.0

# defaults follow the design brief; all are plain arguments so they can be re-calibrated on train
AGG = dict(win=90 * 60.0, dmin=3.5 * HOUR, dmax=5.5 * HOUR, rlo=0.5, rhi=1.5, min_cp=3,
           d_tol=0.75, r_tol=0.35, bucket=12 * HOUR, intake_scale=3.0, max_pair_n=4)
DIST = dict(win=90 * 60.0, fmin=1.5 * HOUR, fmax=3.5 * HOUR, pmin=1.5 * HOUR, pmax=3.0 * HOUR, rlo=0.5, rhi=1.5,
            min_pair=3, member_cap=True, max_pair_n=4)


def _group(codes, n_wallets):
    order = np.argsort(codes, kind="stable")  # time order is kept inside each wallet
    bounds = np.searchsorted(codes[order], np.arange(n_wallets + 1))
    return order, bounds


def _prep(df):
    wallets, uniq = pd.factorize(pd.concat([df.src, df.dst], ignore_index=True))
    n = len(df)
    t = df.ts.to_numpy(dtype=float)
    assert (np.diff(t) >= 0).all(), "payments must be sorted by time"
    s_, d_ = wallets[:n], wallets[n:]
    pn = pd.Series(1, index=range(n)).groupby([s_, d_]).transform("size").to_numpy()   # how often this pair pays in the week
    return s_, d_, t, df.usd.to_numpy(dtype=float), len(uniq), pn


def _confidence(rep, delay_sd, logdev):
    c_rep = min(rep, 3) / 3.0
    c_time = 0.5 if (delay_sd is None or np.isnan(delay_sd)) else float(np.exp(-delay_sd / 0.5))
    c_val = float(np.exp(-logdev / 0.3))
    return c_rep * c_time * c_val


def _chain(cands, p):
    """Best set of mutually consistent, separate-occasion batches for one hub: seed on each candidate, add the best
    compatible candidate (similar delay and value ratio) from every other 12-hour bucket."""
    n = len(cands)
    delay = np.array([c["delay"] for c in cands])
    lr = np.array([c["logratio"] for c in cands])
    q = np.array([c["quality"] for c in cands])
    bucket = (np.array([c["t_ref"] for c in cands]) // p["bucket"]).astype(int)
    best, best_key = None, None
    for s in range(n):
        comp = np.flatnonzero((np.abs(delay - delay[s]) <= p["d_tol"]) & (np.abs(lr - lr[s]) <= p["r_tol"]) & (bucket != bucket[s]))
        chosen, used = [s], set(cands[s]["in_idx"] + cands[s]["out_idx"])
        for b in np.unique(bucket[comp]):
            j = comp[bucket[comp] == b]
            j = j[np.argmin(q[j])]
            members = set(cands[j]["in_idx"] + cands[j]["out_idx"])
            if used & members:
                continue
            chosen.append(j)
            used |= members
        key = (len(chosen), -float(np.mean(q[chosen])))
        if best_key is None or key > best_key:
            best, best_key = chosen, key
    return [cands[i] for i in best]


def aggregate_motifs(df, **kw):
    p = {**AGG, **kw}
    s, d, t, v, n_w, pn = _prep(df)
    in_order, in_b = _group(d, n_w)
    out_order, out_b = _group(s, n_w)
    tags = []
    for h in range(n_w):
        ii, oi = in_order[in_b[h]:in_b[h + 1]], out_order[out_b[h]:out_b[h + 1]]
        ii, oi = ii[pn[ii] <= p['max_pair_n']], oi[pn[oi] <= p['max_pair_n']]   # relay payments sit on rare pairs
        if len(ii) == 0 or len(oi) == 0:
            continue
        ti, vi, ci = t[ii], v[ii], s[ii]
        to_, vo, co = t[oi], v[oi], d[oi]
        cands, seen = [], set()
        for k in range(len(ii)):
            tk = ti[k]
            lo = np.searchsorted(ti, tk - p["win"], "left")
            keep = np.flatnonzero((vi[lo:k + 1] <= vi[k] * p["intake_scale"]) & (vi[lo:k + 1] >= vi[k] / p["intake_scale"])) + lo
            sin = vi[keep].sum()
            a0, a1 = np.searchsorted(to_, tk + p["dmin"], "left"), np.searchsorted(to_, tk + p["dmax"], "right")
            for a in range(a0, a1):
                b = np.searchsorted(to_, to_[a] + p["win"], "right")
                okc = np.flatnonzero(vo[a:b] <= sin * p["rhi"]) + a            # no single payout above what came in
                if len(okc) == 0:
                    continue
                delay = (to_[okc[0]] - tk) / HOUR                                 # from the first payout actually kept
                if not (p["dmin"] / HOUR <= delay <= p["dmax"] / HOUR):
                    continue
                sig = (tuple(keep), tuple(okc))
                if sig in seen:
                    continue
                seen.add(sig)
                sout = vo[okc].sum()
                ratio = sout / sin
                if not (p["rlo"] <= ratio <= p["rhi"]):
                    continue
                if len(set(ci[keep]) | set(co[okc])) < p["min_cp"]:
                    continue
                cands.append(dict(in_idx=ii[keep].tolist(), out_idx=oi[okc].tolist(), t_ref=tk, delay=delay,
                                  ratio=ratio, logratio=float(np.log(ratio)), quality=abs(float(np.log(ratio))),
                                  n_in=len(keep), n_out=len(okc), usd_in=sin, usd_out=sout))
        if not cands:
            continue
        chosen = _chain(cands, p)
        rep = len(chosen)
        days = len({int(c["t_ref"] // DAY) for c in chosen})
        dsd = float(np.std([c["delay"] for c in chosen])) if rep > 1 else np.nan
        logdev = float(np.mean([c["quality"] for c in chosen]))
        conf = _confidence(rep, dsd, logdev)
        for c in chosen:
            for role, members in ((1, c["in_idx"]), (2, c["out_idx"])):
                for m in members:
                    tags.append(dict(i=m, rep=rep, days=days, n_in=c["n_in"], n_out=c["n_out"], usd_in=c["usd_in"],
                                     usd_out=c["usd_out"], ratio=c["ratio"], delay=c["delay"], dsd=dsd, role=role,
                                     conf=conf, logdev=logdev))
    return tags


def distributed_motifs(df, **kw):
    p = {**DIST, **kw}
    s, d, t, v, n_w, pn = _prep(df)
    in_order, in_b = _group(d, n_w)
    out_order, out_b = _group(s, n_w)
    pair = pd.Series(np.arange(len(df))).groupby([s, d]).apply(list)
    tags = []
    for (a_w, b_w), events in pair.items():
        if len(events) < p["min_pair"]:
            continue
        ia, ob = in_order[in_b[a_w]:in_b[a_w + 1]], out_order[out_b[b_w]:out_b[b_w + 1]]
        ia, ob = ia[pn[ia] <= p['max_pair_n']], ob[pn[ob] <= p['max_pair_n']]
        if len(ia) == 0 or len(ob) == 0:
            continue
        tA, vA, tB, vB = t[ia], v[ia], t[ob], v[ob]
        cands = []
        for e in events:
            te, ve = t[e], v[e]
            if ve <= 0:
                continue
            cap = ve * p["rhi"] if p["member_cap"] else np.inf
            f0, f1 = np.searchsorted(tA, te - p["fmax"], "left"), np.searchsorted(tA, te - p["fmin"], "right")
            q0, q1 = np.searchsorted(tB, te + p["pmin"], "left"), np.searchsorted(tB, te + p["pmax"], "right")
            best = None
            for f in range(f0, f1):
                flo = np.searchsorted(tA, tA[f] - p["win"], "left")
                fk = np.flatnonzero(vA[flo:f + 1] <= cap) + flo
                if len(fk) == 0:
                    continue
                rf = vA[fk].sum() / ve
                if not (p["rlo"] <= rf <= p["rhi"]):
                    continue
                for q in range(q0, q1):
                    qhi = np.searchsorted(tB, tB[q] + p["win"], "right")
                    qk = np.flatnonzero(vB[q:qhi] <= cap) + q
                    if len(qk) == 0:
                        continue
                    rp = vB[qk].sum() / ve
                    if not (p["rlo"] <= rp <= p["rhi"]):
                        continue
                    qual = abs(np.log(rf)) + abs(np.log(rp))
                    if best is None or qual < best["quality"]:
                        best = dict(bridge=e, in_idx=ia[fk].tolist(), out_idx=ob[qk].tolist(), quality=qual, t_ref=te,
                                    d_feed=(te - tA[f]) / HOUR, d_pay=(tB[q] - te) / HOUR, rf=rf, rp=rp, n_f=len(fk), n_p=len(qk))
            if best:
                cands.append(best)
        if not cands:
            continue
        rep = len(cands)
        days = len({int(c["t_ref"] // DAY) for c in cands})
        delays = [c["d_feed"] for c in cands] + [c["d_pay"] for c in cands]
        dsd = float(np.std(delays)) if rep > 1 else np.nan
        logdev = float(np.mean([c["quality"] for c in cands])) / 2.0
        frac = rep / len(events)
        conf = _confidence(rep, dsd, logdev) * frac
        for c in cands:
            for role, members in ((1, c["in_idx"]), (2, [c["bridge"]]), (3, c["out_idx"])):
                for m in members:
                    tags.append(dict(i=m, rep=rep, days=days, d_feed=c["d_feed"], d_pay=c["d_pay"], rf=c["rf"], rp=c["rp"],
                                     n_f=c["n_f"], n_p=c["n_p"], role=role, conf=conf, dsd=dsd, frac=frac, logdev=logdev))
    return tags


def _attach(n, tags, rename):
    """Each payment keeps its highest-confidence tag; NaN for untagged payments."""
    cols = list(rename)
    out = pd.DataFrame(np.nan, index=range(n), columns=[rename[c] for c in cols])
    if not tags:
        return out
    T = pd.DataFrame(tags).sort_values("conf", ascending=False).drop_duplicates("i")
    out.loc[T.i.to_numpy(), [rename[c] for c in cols]] = T[cols].to_numpy()
    return out


def motif_features(df):
    """df: output of dd.load.load_split (time-sorted). Returns one row per payment, named as in the design brief."""
    n = len(df)
    A = _attach(n, aggregate_motifs(df), {
        "rep": "aggregate_relay_repeat_count", "n_in": "aggregate_in_count", "n_out": "aggregate_out_count",
        "usd_in": "aggregate_in_usd", "usd_out": "aggregate_out_usd", "ratio": "aggregate_value_ratio",
        "delay": "aggregate_delay_hours", "dsd": "aggregate_delay_consistency", "role": "aggregate_role",
        "conf": "aggregate_motif_confidence", "days": "aggregate_distinct_days", "logdev": "aggregate_value_deviation"})
    D = _attach(n, distributed_motifs(df), {
        "rep": "distributed_bridge_repeat_count", "d_feed": "feed_to_bridge_delay_hours", "d_pay": "bridge_to_payout_delay_hours",
        "rf": "feed_value_ratio", "rp": "payout_value_ratio", "n_f": "feeder_count", "n_p": "payout_count",
        "role": "distributed_role", "conf": "distributed_motif_confidence", "dsd": "distributed_delay_consistency",
        "days": "distributed_distinct_days", "frac": "distributed_pass_fraction", "logdev": "distributed_value_deviation"})
    F = pd.concat([A, D], axis=1)
    F["motif_confidence"] = F[["aggregate_motif_confidence", "distributed_motif_confidence"]].max(axis=1)
    return F
