"""Chronological validation with locked final two-month evaluation."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_all
from features import add_daily_signals, build_features, clean, daily_table
from model import FINAL_CFG, T0, TwoStage, flag_corrupt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TUNING_CUTS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01"]
FINAL_CUT = "2025-09-01"
CANDIDATES = {
    "two_stage_l1": FINAL_CFG,
    "two_stage_squared": dict(FINAL_CFG, gbm_loss="squared_error"),
    "linear_only": dict(FINAL_CFG, gbm=False),
}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error) / actual) * 100),
        "MedAPE": float(np.median(np.abs(error) / actual) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    development, validation, _, _ = load_all()
    medians = clean(development).groupby("equipment")["weight"].median()
    development = clean(development, medians).sort_values("date").reset_index(drop=True)
    validation = clean(validation, medians)
    # Feature-only same-date aggregates available in the prediction batch.
    development = add_daily_signals(development, daily_table([development, validation]))
    features = build_features(development)
    features["date"] = development["date"]
    features["t_idx"] = (development["date"] - T0).dt.days
    return development, features, development["posted_rate"].to_numpy()


def window(frame: pd.DataFrame, cut: str) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    start = pd.Timestamp(cut)
    end = start + pd.DateOffset(months=2)
    return (frame.date < start).to_numpy(), ((frame.date >= start) & (frame.date < end)).to_numpy(), end


def baseline(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    _, edges = pd.qcut(train_x["distance"], 12, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    table = pd.DataFrame({"band": pd.cut(train_x.distance, edges, labels=False), "equipment": train_x.equipment, "rpm": train_y / train_x.distance}).groupby(["band", "equipment"]).rpm.median()
    key = pd.MultiIndex.from_arrays([pd.cut(test_x.distance, edges, labels=False), test_x.equipment])
    rate = table.reindex(key).to_numpy(dtype=float)
    rate = np.where(np.isfinite(rate), rate, float(np.median(train_y / train_x.distance)))
    return rate * test_x.distance.to_numpy()


def evaluate(frame, features, target, cut, name, config):
    train_mask, test_mask, end = window(frame, cut)
    train_x, test_x = features.loc[train_mask], features.loc[test_mask]
    train_y, test_y = target[train_mask], target[test_mask]
    keep_train = flag_corrupt(np.log(train_y), train_x)
    model = TwoStage(**config).fit(train_x.loc[keep_train], train_y[keep_train])
    prediction = model.predict(test_x)
    diagnostic_keep = flag_corrupt(np.log(test_y), test_x)
    row = {"cut": cut, "test_end": str((end - pd.Timedelta(days=1)).date()), "model": name,
           "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
           "training_rows_retained": int(keep_train.sum()), "training_rows_flagged": int((~keep_train).sum()),
           "diagnostic_test_rows_flagged": int((~diagnostic_keep).sum())}
    row.update({f"all_{k}": v for k, v in metrics(test_y, prediction).items()})
    row.update({f"diagnostic_clean_{k}": v for k, v in metrics(test_y[diagnostic_keep], prediction[diagnostic_keep]).items()})
    return row, test_y, prediction


def main() -> None:
    frame, features, target = prepare()
    tuning_rows = [evaluate(frame, features, target, cut, name, config)[0] for cut in TUNING_CUTS for name, config in CANDIDATES.items()]
    tuning = pd.DataFrame(tuning_rows)
    selected = tuning.groupby("model").all_MAE.mean().sort_values(kind="stable").index[0]
    final, actual, prediction = evaluate(frame, features, target, FINAL_CUT, selected, CANDIDATES[selected])
    train_mask, test_mask, _ = window(frame, FINAL_CUT)
    base_prediction = baseline(features.loc[train_mask], target[train_mask], features.loc[test_mask])
    final.update({f"baseline_all_{k}": v for k, v in metrics(target[test_mask], base_prediction).items()})

    RESULTS.mkdir(exist_ok=True)
    tuning.to_csv(RESULTS / "tuning_windows.csv", index=False)
    pd.DataFrame([final]).to_csv(RESULTS / "final_evaluation.csv", index=False)
    summary = {"protocol": {"selection_windows": TUNING_CUTS, "final_evaluation_window": "2025-09-01 through 2025-10-31", "selection_metric": "mean all-row MAE", "selected_model": selected, "diagnostic_note": "Clean-label metrics use held-out targets and are diagnostic only."}, "tuning_mean_all_row_metrics": tuning.groupby("model")[["all_MAE", "all_RMSE", "all_MAPE", "all_MedAPE", "all_R2"]].mean().round(4).reset_index().to_dict("records"), "final_evaluation": final}
    (RESULTS / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
