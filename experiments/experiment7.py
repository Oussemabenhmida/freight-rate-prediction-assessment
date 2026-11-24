import sys, warnings; sys.path.insert(0, "../src")
import numpy as np, pandas as pd
from validate import prepare, metrics
from model import TwoStage, flag_corrupt, FINAL_CFG
warnings.filterwarnings("ignore")
tr, X, y = prepare()
cuts = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
cfg = {"FINAL (no quote_signal at all)": dict(FINAL_CFG),
       "FINAL + quote_signal in GBM":     dict(FINAL_CFG, qs_cols=True),
       "FINAL + quote_signal everywhere": dict(FINAL_CFG, qs_cols=True, lin_qs=True)}
rows = []
for c in cuts:
    s = pd.Timestamp(c); e = s + pd.DateOffset(months=2)
    itr = (tr.date < s).values; ite = ((tr.date >= s) & (tr.date < e)).values
    Xa, ya, Xb, yb = X[itr], y[itr], X[ite], y[ite]
    ok = flag_corrupt(np.log(ya), Xa); ok_te = flag_corrupt(np.log(yb), Xb)
    for k, kw in cfg.items():
        m = TwoStage(**kw).fit(Xa[ok], ya[ok]); p = m.predict(Xb)
        rows.append(dict(fold=c, model=k, **metrics(yb[ok_te], p[ok_te])))
        if k.startswith("FINAL (") and c == cuts[-1]:
            print("GBM features used:", m.cols)
df = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(df.groupby("model")[["MAE", "RMSE", "MAPE", "MedAPE", "R2"]].mean().round(3))
print(df.pivot(index="model", columns="fold", values="MAPE").round(2))
