"""Raw organiser files -> one row per transaction.

Follows the data README exactly:
- one row per TRANSACTION_HASH (never per transfer)
- receiver = Destination in EXTRA_TRANSACTION_FIELDS, delivered amount = DELIVERED_AMOUNT JSON
- XRP amounts are drops; an asset is issuer + currency ("issuer:USD"), XRP is "XRP"
- USD value uses market_rates.csv at the hour containing TIMESTAMP (PRICE_MICRO_USD / 1e6)
- SendMax is a ceiling, so the amount actually paid is the first executed transfer leg
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def _rates(directory: Path) -> dict:
    r = pd.read_csv(directory / "market_rates.csv")
    return {(int(t), a): p / 1e6 for t, a, p in r[["TIMESTAMP", "ASSET", "PRICE_MICRO_USD"]].itertuples(index=False)}


def _decode(wire):
    """DELIVERED_AMOUNT: string of drops for XRP, else {currency, issuer, value}."""
    if isinstance(wire, str):
        return C.XRP_ASSET, int(wire) / C.DROPS_PER_XRP
    return f"{wire['issuer']}:{wire['currency']}", float(wire["value"])


def _price(rates, t, asset):
    h = int(t) // 3600 * 3600
    p = rates.get((h, asset))
    if p is None:  # nearest available hour for that asset (should not happen with organiser data)
        hs = [k[0] for k in rates if k[1] == asset]
        p = rates[(min(hs, key=lambda x: abs(x - h)), asset)]
    return p


def load_split(directory) -> pd.DataFrame:
    directory = Path(directory)
    tx = pd.read_parquet(directory / "transactions.parquet")
    tr = pd.read_parquet(directory / "transfers.parquet")
    rates = _rates(directory)

    ex = tx["EXTRA_TRANSACTION_FIELDS"].map(json.loads)
    delivered = tx["DELIVERED_AMOUNT"].map(json.loads).map(_decode)
    t = tx["TIMESTAMP"].astype(np.int64).values
    asset_out = delivered.map(lambda x: x[0]).values
    amt_out = delivered.map(lambda x: x[1]).values.astype(float)
    usd = np.array([a * _price(rates, ti, s) for a, ti, s in zip(amt_out, t, asset_out)])

    out = pd.DataFrame({
        "tx": tx["TRANSACTION_HASH"].astype(str).values,
        "src": tx["ACCOUNT"].astype(str).values,
        "dst": ex.map(lambda x: x.get("Destination")).astype(str).values,
        "t": t,
        "tix": pd.to_numeric(tx["TRANSACTION_INDEX"]).astype(int).values,
        "usd": usd,
        "amt_out": amt_out,
        "asset_out": asset_out,
        "has_sendmax": ex.map(lambda x: "SendMax" in x).astype(int).values,
        "has_paths": ex.map(lambda x: "Paths" in x).astype(int).values,
    })

    # executed legs: what was really paid, how many hops, how many assets
    tr = tr.assign(
        asset=np.where(tr["SYMBOL"] == "XRP", C.XRP_ASSET, tr["TOKEN_ADDRESS"].astype(str) + ":" + tr["SYMBOL"].astype(str)),
        idx=pd.to_numeric(tr["TRANSFER_INDEX"]).astype(int),
        units=tr["VALUE"].astype(float),            # VALUE is in asset units for XRP and tokens
    ).sort_values(["TRANSACTION_HASH", "idx"])
    g = tr.groupby("TRANSACTION_HASH", sort=False)
    first = g.head(1).set_index("TRANSACTION_HASH")
    legs = pd.DataFrame({
        "n_legs": g.size(),
        "n_assets": g["asset"].nunique(),
        "asset_in": first["asset"],
        "amt_in": first["units"],
        "is_path": g["TRANSFER_TYPE"].agg(lambda s: int((s == "path_payment").any())),
    })
    out = out.join(legs, on="tx")
    out["asset_in"] = out["asset_in"].fillna(out["asset_out"])
    out["amt_in"] = out["amt_in"].fillna(out["amt_out"])
    out["usd_paid"] = [a * _price(rates, ti, s) for a, ti, s in zip(out["amt_in"], out["t"], out["asset_in"])]
    out["is_conv"] = (out["asset_in"] != out["asset_out"]).astype(int)
    out[["n_legs", "n_assets", "is_path"]] = out[["n_legs", "n_assets", "is_path"]].fillna(1)

    assert out["tx"].is_unique and len(out) == tx["TRANSACTION_HASH"].nunique()
    print(f"[load] {directory}: {len(out):,} transactions, {len(tr):,} transfer legs, "
          f"{out['is_conv'].mean():.1%} conversions, {pd.concat([out.src, out.dst]).nunique()} wallets")
    return out.sort_values(["t", "tix"]).reset_index(drop=True)


def load_labels(directory) -> pd.DataFrame:
    """is_suspicious plus annotation metadata (family, difficulty, scenario_id, cohort_id).
    The metadata is used ONLY for validation grouping and per-slice reporting, never as features."""
    lb = pd.read_csv(Path(directory) / "labels.csv")
    return lb.rename(columns={"TRANSACTION_HASH": "tx", "is_suspicious": "y"})
