"""The starter notebook's 12 rules and logistic regression, ported to our loader."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HOUR, TEN_MINUTES = 3600, 600


def starter_features(df):
    t = df.ts.to_numpy()
    value = df.usd.to_numpy()
    sends = {a: g.ts.to_numpy() for a, g in df.groupby("src")}
    gets = {a: g.ts.to_numpy() for a, g in df.groupby("dst")}
    sent_values = {a: g.usd.to_numpy() for a, g in df.groupby("src")}
    got_values = {a: g.usd.to_numpy() for a, g in df.groupby("dst")}

    def count(times_by, a, ti, before, after):
        x = times_by.get(a)
        if x is None:
            return 0
        return int(np.searchsorted(x, ti + after, "right") - np.searchsorted(x, ti - before, "left"))

    def forwards_similar(a, ti, v):
        x = sends.get(a)
        if x is None:
            return False
        lo, hi = np.searchsorted(x, ti, "right"), np.searchsorted(x, ti + HOUR, "right")
        return bool(np.any(np.abs(sent_values[a][lo:hi] - v) <= 0.1 * v))

    def received_similar(a, ti, v):
        x = gets.get(a)
        if x is None:
            return False
        lo, hi = np.searchsorted(x, ti - HOUR, "left"), np.searchsorted(x, ti, "left")
        return bool(np.any(np.abs(got_values[a][lo:hi] - v) <= 0.1 * v))

    def minutes_to_next_send(a, ti):
        x = sends.get(a)
        if x is None:
            return 60.0
        i = np.searchsorted(x, ti, "right")
        return min(60.0, (x[i] - ti) / 60) if i < len(x) else 60.0

    f = pd.DataFrame(index=df.index)
    f["log_value"] = np.log1p(value)
    f["sender_payments"] = df.src.map(df.src.value_counts())
    f["receiver_payments"] = df.dst.map(df.dst.value_counts())
    f["receiver_forwards_1h"] = [count(sends, a, ti, 0, HOUR) > 0 for a, ti in zip(df.dst, t)]
    f["sender_received_1h_before"] = [count(gets, a, ti, HOUR, 0) > 0 for a, ti in zip(df.src, t)]
    f["sender_burst_10m"] = [count(sends, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.src, t)]
    f["receiver_burst_10m"] = [count(gets, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.dst, t)]
    f["receiver_forwards_similar_1h"] = [forwards_similar(a, ti, v) for a, ti, v in zip(df.dst, t, value)]
    f["sender_received_similar_1h"] = [received_similar(a, ti, v) for a, ti, v in zip(df.src, t, value)]
    f["receiver_minutes_to_forward"] = [minutes_to_next_send(a, ti) for a, ti in zip(df.dst, t)]
    sent_total = df.groupby("src").usd.sum()
    got_total = df.groupby("dst").usd.sum()
    pt = (sent_total / got_total.reindex(sent_total.index)).clip(0, 3)
    f["sender_pass_through"] = df.src.map(pt).fillna(0)
    f["receiver_pass_through"] = df.dst.map(pt).fillna(0)
    return f.astype(float)


def starter_model():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
