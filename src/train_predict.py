"""Train the final model on ALL labelled data (Jan-Oct 2025) and predict
   (1) the 12,000 validation loads  -> validation_predictions.csv
   (2) the 31 fixed December chart inputs -> data/december_chart_inputs.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from data import load_all
from features import clean, daily_table, add_daily_signals, build_features
from model import TwoStage, flag_corrupt, FINAL_CFG, T0

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df); X["date"] = df["date"]; X["t_idx"] = (df["date"] - T0).dt.days
    return X


def main():
    tr, va, dec, tmpl = load_all()
    wmed = clean(tr).groupby("equipment")["weight"].median()
    tr, va = clean(tr, wmed), clean(va, wmed)
    # label-free per-date aggregate of market_index (used to impute the ~1-2% missing values and for December)
    daily = daily_table([tr, va])
    tr, va = add_daily_signals(tr, daily), add_daily_signals(va, daily)

    # ---- fit ----
    Xtr, ytr = featurize(tr), tr["posted_rate"].values
    ok = flag_corrupt(np.log(ytr), Xtr)
    print(f"training rows: {len(Xtr):,}; dropped {int((~ok).sum()):,} ({(~ok).mean()*100:.2f}%) corrupted-rate rows from fitting")
    # Selected by mean all-row MAE on the four early rolling windows.
    model = TwoStage(**dict(FINAL_CFG, gbm_loss="squared_error")).fit(Xtr[ok], ytr[ok])

    # ---- 1) validation predictions ----
    pred = model.predict(featurize(va))
    out = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": va["load_id"], "predicted_rate": np.round(pred, 2)}), on="load_id", how="left")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv", out.shape, "| mean rate", out.predicted_rate.mean().round(1))

    # ---- 2) December chart inputs: only the date varies ----
    coords = pd.concat([tr[["pickup", "pickup_lat", "pickup_lon"]].rename(columns=lambda c: c.replace("pickup", "city")),
                        tr[["delivery", "delivery_lat", "delivery_lon"]].rename(columns=lambda c: c.replace("delivery", "city"))]).drop_duplicates("city").set_index("city")
    d = dec.copy(); d["date"] = pd.to_datetime(d["date"])
    for side in ("pickup", "delivery"):
        d[f"{side}_lat"] = d[side].map(coords["city_lat"]); d[f"{side}_lon"] = d[side].map(coords["city_lon"])
    d = clean(d.assign(weight=d["weight"].astype(float)), wmed)
    d = d.merge(daily, on="date", how="left")           # market_index = that date's mean across Nov-Dec loads
    d["market_index"] = d["mi_day"]; d["market_index_missing"] = 0
    d["quote_signal"] = d["qs_day"]                      # not used by the final model
    dec_pred = model.predict(featurize(d))
    result = dec.copy(); result["predicted_rate"] = np.round(dec_pred, 2)
    result.to_csv(ROOT / "data" / "december_chart_inputs.csv", index=False)
    print("wrote data/december_chart_inputs.csv"); print(result.assign(mi=d["market_index"].round(3).values).to_string(index=False))


if __name__ == "__main__":
    main()
