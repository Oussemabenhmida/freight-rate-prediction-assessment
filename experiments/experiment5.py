"""Re-run time-ordered CV with the quarter-end ramp + linear trend features."""
import sys, warnings; sys.path.insert(0, "../src")
import numpy as np, pandas as pd
from validate import prepare, metrics
from model import TwoStage, flag_corrupt, level_forecast
warnings.filterwarnings("ignore")
tr, X, y = prepare()
cuts = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
cfg = {
 "old: no ramp, no trend (prev. final)": dict(trend="none", ramp=False, lin_qs=True, qs_cols=True),
 "S1 linear only (mi+trend+ramp)":       dict(trend="linear", gbm=False),
 "S2 trend+ramp+GBM, no qs":             dict(trend="linear", qs_cols=False),
 "S3 trend+ramp+GBM, with qs":           dict(trend="linear", qs_cols=True, lin_qs=True),
 "S4 S2 with MSE loss":                  dict(trend="linear", qs_cols=False, gbm_loss="squared_error"),
 "S5 no trend, ramp+GBM":                dict(trend="none", qs_cols=False),
}
rows = []
for c in cuts:
    s = pd.Timestamp(c); e = s + pd.DateOffset(months=2)
    itr = (tr.date < s).values; ite = ((tr.date >= s) & (tr.date < e)).values
    Xa, ya, Xb, yb = X[itr], y[itr], X[ite], y[ite]
    ok = flag_corrupt(np.log(ya), Xa); ok_te = flag_corrupt(np.log(yb), Xb)
    for k, kw in cfg.items():
        m = TwoStage(gbm_loss=kw.pop("gbm_loss", "absolute_error") if False else kw.get("gbm_loss", "absolute_error"), **{a: b for a, b in kw.items() if a != "gbm_loss"}).fit(Xa[ok], ya[ok])
        p = m.predict(Xb)
        for scope, mk in (("all", np.ones(len(yb), bool)), ("clean", ok_te)):
            rows.append(dict(fold=c, model=k, scope=scope, **metrics(yb[mk], p[mk])))
    print(c, "done", flush=True)
df = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(df.groupby(["model", "scope"])[["MAE", "RMSE", "MAPE", "MedAPE", "R2"]].mean().round(3))
print("\nclean MAPE by fold:\n", df[df.scope == "clean"].pivot(index="model", columns="fold", values="MAPE").round(2))
