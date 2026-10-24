"""Load one split into a per-transaction table valued in USD."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

XRP_TOKEN = "0x1111222211112222111122221111222211112222"


def load_rates(directory):
    r = pd.read_csv(Path(directory) / "market_rates.csv")
    return {(int(t), a): p / 1e6 for t, a, p in r.itertuples(index=False)}


def _asset_of(wire):
    if isinstance(wire, str):
        return "XRP", int(wire) / 1e6
    return f"{wire['issuer']}:{wire['currency']}", float(wire["value"])


def _leg_asset(token, symbol):
    return "XRP" if token == XRP_TOKEN else f"{token}:{symbol}"


def load_split(directory):
    """One row per transaction: endpoints, time, delivered value, actual cost, legs."""
    directory = Path(directory)
    rates = load_rates(directory)
    tx = pd.read_parquet(directory / "transactions.parquet")
    tr = pd.read_parquet(directory / "transfers.parquet")

    ts = tx.TIMESTAMP.astype(np.int64).to_numpy()
    hour = ts // 3600 * 3600
    fields = tx.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    delivered = tx.DELIVERED_AMOUNT.map(lambda s: _asset_of(json.loads(s)))
    d_asset = delivered.map(lambda x: x[0]).to_numpy()
    d_units = delivered.map(lambda x: x[1]).to_numpy()
    d_usd = np.array([u * rates[(h, a)] for u, h, a in zip(d_units, hour, d_asset)])

    out = pd.DataFrame({
        "hash": tx.TRANSACTION_HASH.to_numpy(),
        "ts": ts,
        "ledger": tx.LEDGER_INDEX.astype(np.int64).to_numpy(),
        "tx_index": tx.TRANSACTION_INDEX.astype(np.int64).to_numpy(),
        "src": tx.ACCOUNT.to_numpy(),
        "dst": fields.map(lambda f: f["Destination"]).to_numpy(),
        "seq": tx.SEQUENCE.astype(np.int64).to_numpy(),
        "fee_drops": tx.SOURCE_FEE.astype(np.int64).to_numpy(),
        "asset": d_asset,
        "units": d_units,
        "usd": d_usd,
        "has_sendmax": fields.map(lambda f: "SendMax" in f).to_numpy(),
        "has_paths": fields.map(lambda f: "Paths" in f).to_numpy(),
    })

    # Actual cost: the first executed transfer leg (SendMax is only a ceiling).
    tr = tr.assign(TRANSFER_INDEX=tr.TRANSFER_INDEX.astype(np.int64)).sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    legs = tr.groupby("TRANSACTION_HASH").size()
    first = tr.groupby("TRANSACTION_HASH").first()
    out["legs"] = out.hash.map(legs).fillna(1).astype(int).to_numpy()
    f = first.reindex(out.hash)
    cost_asset = [_leg_asset(t, s) for t, s in zip(f.TOKEN_ADDRESS, f.SYMBOL)]
    out["cost_asset"] = cost_asset
    out["cost_usd"] = [v * rates[(h, a)] for v, h, a in zip(f.VALUE.to_numpy(), hour, cost_asset)]
    out["spread"] = out.cost_usd / out.usd - 1
    out["converts"] = out.cost_asset != out.asset
    return out.sort_values(["ts", "ledger", "tx_index"]).reset_index(drop=True)
