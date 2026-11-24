"""Cleaning + feature engineering for the freight-rate task."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT = ["equipment"]
RAMP_DAYS = 25


def haversine(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, [lat1, lon1, lat2, lon2])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))


def clean(df: pd.DataFrame, weight_medians: pd.Series | None = None) -> pd.DataFrame:
    """Fix data-quality issues. Only uses features (never labels), so it is
    safe to apply identically to train, validation and December inputs."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    # 1) negative weights are sign flips (|w| distribution matches the positives)
    out["weight_was_neg"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    # 2) missing weight -> flag + equipment median
    out["weight_missing"] = out["weight"].isna().astype(int)
    if weight_medians is not None:
        out["weight"] = out["weight"].fillna(out["equipment"].map(weight_medians))
    return out


def add_daily_signals(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Attach per-date market_index / quote_signal aggregates.
    market_index is essentially one value per day (+/-0.025 noise), so a load with
    a missing value is imputed with its own date's mean from all other loads."""
    out = df.merge(daily, on="date", how="left")
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["mi_day"])
    return out


def daily_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    allx = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    g = allx.groupby("date").agg(mi_day=("market_index", "mean"), qs_day=("quote_signal", "mean")).reset_index()
    return g


def build_features(df: pd.DataFrame, route_dist: pd.Series | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    X["weight"] = df["weight"]
    X["weight_missing"] = df["weight_missing"]
    X["weight_was_neg"] = df["weight_was_neg"]
    X["market_index"] = df["market_index"]
    X["market_index_missing"] = df["market_index_missing"]
    X["mi_day"] = df["mi_day"]
    X["quote_signal"] = df["quote_signal"]
    X["qs_dev"] = (df["quote_signal"] - 2.05).abs()          # U-shaped effect
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    X["hav"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["dist_over_hav"] = df["distance"] / X["hav"].clip(lower=1)
    X["dlat"] = df.delivery_lat - df.pickup_lat
    X["dlon"] = df.delivery_lon - df.pickup_lon
    X["bearing_ew"] = np.sign(X["dlon"])
    X["equipment"] = df["equipment"].map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})
    dt = df["date"]
    X["dow"] = dt.dt.dayofweek
    X["dom"] = dt.dt.day
    X["month"] = dt.dt.month
    X["doy"] = dt.dt.dayofyear
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    # Quarter-end effect: the rate level ramps up over the last ~25 days of every quarter (+~3.5% on the
    # final day) and resets at the start of the next one. Found in EDA; repeats identically in Q1-Q3.
    qend = dt.dt.to_period("Q").dt.end_time.dt.normalize()
    X["dtq"] = (qend - dt).dt.days                          # days until quarter end (0 on the last day)
    X["ramp"] = np.clip((RAMP_DAYS - X["dtq"]) / RAMP_DAYS, 0, 1)
    return X
