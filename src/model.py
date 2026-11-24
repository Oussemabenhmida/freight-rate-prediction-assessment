"""Two-stage model: (1) robust linear model in log space with a smooth time trend,
(2) gradient boosting on the stage-1 residual using non-time features."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

T0 = pd.Timestamp("2025-01-01")
GBM_EXCLUDE = {"month", "doy", "t_idx"}   # raw time can not extrapolate in trees; trend handled in stage 1


def flag_corrupt(y_log: np.ndarray, X: pd.DataFrame, k: float = 4.0) -> np.ndarray:
    """True = trustworthy label. Robust z of log(rate/mile) within (distance-bin x equipment) cells."""
    d = pd.DataFrame({"l": y_log - np.log(X["distance"].values), "b": pd.qcut(X["distance"], 12, labels=False).values,
                      "e": X["equipment"].values})
    g = d.groupby(["b", "e"])["l"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    return ((d["l"] - med) / mad).abs().values < k


class TwoStage:
    def __init__(self, trend="linear", gbm_loss="absolute_error", use_time_in_gbm=False, n_iter=500, lr=0.05,
                 leaf=40, max_leaf_nodes=31, local_level_days=28, seed=0, gbm=True, qs_cols=True, lin_qs=False, ramp=True):
        self.trend, self.gbm_loss, self.use_time_in_gbm = trend, gbm_loss, use_time_in_gbm
        self.n_iter, self.lr, self.leaf, self.mln = n_iter, lr, leaf, max_leaf_nodes
        self.local_level_days, self.seed, self.gbm, self.qs_cols = local_level_days, seed, gbm, qs_cols
        self.lin_qs, self.ramp = lin_qs, ramp

    # ---- stage 1 design matrix (smooth, extrapolates) ----
    def _lin(self, X, fit=False):
        ld = X[["log_distance"]].values
        if fit:
            self.spl = SplineTransformer(n_knots=6, degree=3, extrapolation="linear").fit(ld)
        S = self.spl.transform(ld)
        cols = [S, np.eye(3)[X["equipment"].astype(int).values][:, 1:],
                X[["weight", "market_index"]].values / np.array([1e4, 1]),
                (X["weight"].values / 1e4 * X["equipment"].values)[:, None]]
        if self.lin_qs:
            cols.append(X[["qs_dev"]].values)
        if self.ramp:
            cols.append(X[["ramp"]].values)
        if self.trend == "linear":
            cols.append(X[["t_idx"]].values / 100.0)
        return np.hstack(cols)

    def fit(self, X, y_rate):
        X = X.copy(); X["t_idx"] = (X["date"] - T0).dt.days if "date" in X else X["t_idx"]
        yl = np.log(y_rate)
        self.lin = LinearRegression().fit(self._lin(X, True), yl)
        r = yl - self.lin.predict(self._lin(X))
        if self.trend == "local":   # level = mean residual over the last N days of training data
            last = X["t_idx"] >= X["t_idx"].max() - self.local_level_days
            self.level = float(r[last.values].mean())
        self.cols = [c for c in X.columns if c not in GBM_EXCLUDE and c not in ("date",)] if not self.use_time_in_gbm else \
                    [c for c in X.columns if c not in ("month", "doy", "date")]
        if not self.qs_cols:
            self.cols = [c for c in self.cols if not (c.startswith("qs") or c.startswith("quote"))]
        if not self.ramp:
            self.cols = [c for c in self.cols if c not in ("ramp", "dtq")]
        if self.gbm:
            self.m = HistGradientBoostingRegressor(loss=self.gbm_loss, learning_rate=self.lr, max_iter=self.n_iter,
                                                   max_leaf_nodes=self.mln, min_samples_leaf=self.leaf, l2_regularization=1.0,
                                                   early_stopping=False, random_state=self.seed)
            self.m.fit(X[self.cols], r - (self.level if self.trend == "local" else 0.0))
        return self

    def predict(self, X):
        X = X.copy(); X["t_idx"] = (X["date"] - T0).dt.days if "date" in X else X["t_idx"]
        p = self.lin.predict(self._lin(X))
        if self.trend == "local":
            p = p + self.level
        if self.gbm:
            p = p + self.m.predict(X[self.cols])
        return np.exp(p)


def level_forecast(model: "TwoStage", Xa: pd.DataFrame, ya: np.ndarray, t_future: np.ndarray, recent_days: int = 14):
    """Forecast the slowly-moving rate level (an unexplained, highly autocorrelated daily component).

    Tree models cannot extrapolate in time, and validation/December lie after the training window, so the
    level is forecast explicitly: 50% linear trend of the daily mean residual, 50% mean of the last
    `recent_days` days. Blending the two hypotheses (steady drift vs. random-walk level) was the most
    robust choice in rolling time-ordered CV."""
    r = np.log(ya) - np.log(model.predict(Xa))
    t = Xa["t_idx"].values if "t_idx" in Xa else (Xa["date"] - T0).dt.days.values
    dl = pd.DataFrame({"t": t, "r": r}).groupby("t").r.mean()
    slope, icpt = np.polyfit(dl.index.values, dl.values, 1)
    trend = icpt + slope * t_future
    local = dl[dl.index > dl.index.max() - recent_days].mean()
    return 0.5 * trend + 0.5 * local, dict(trend=trend, local=float(local), slope_per_30d=float(slope * 30))


# Final configuration chosen by time-ordered CV (see experiments/ and the report)
FINAL_CFG = dict(trend="linear", gbm_loss="absolute_error", qs_cols=False, ramp=True,
                 n_iter=300, lr=0.05, leaf=100, max_leaf_nodes=15)
