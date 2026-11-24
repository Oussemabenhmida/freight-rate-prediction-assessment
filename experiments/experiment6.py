import sys, warnings, time; sys.path.insert(0, "../src")
import numpy as np, pandas as pd
from validate import prepare, metrics
from model import TwoStage, flag_corrupt, level_forecast
warnings.filterwarnings("ignore")
tr, X, y = prepare()
cuts = ["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
grid = {
 "base 500/.05/leaf40/31":  dict(n_iter=500, lr=0.05, leaf=40, max_leaf_nodes=31),
 "slow 900/.03/leaf40/31":  dict(n_iter=900, lr=0.03, leaf=40, max_leaf_nodes=31),
 "small-leaf 500/.05/20/63": dict(n_iter=500, lr=0.05, leaf=20, max_leaf_nodes=63),
 "regular 500/.05/100/15":  dict(n_iter=500, lr=0.05, leaf=100, max_leaf_nodes=15),
 "short 250/.05/40/31":     dict(n_iter=250, lr=0.05, leaf=40, max_leaf_nodes=31),
}
rows = []
for c in cuts:
    s = pd.Timestamp(c); e = s + pd.DateOffset(months=2)
    itr = (tr.date < s).values; ite = ((tr.date >= s) & (tr.date < e)).values
    Xa, ya, Xb, yb = X[itr], y[itr], X[ite], y[ite]
    ok = flag_corrupt(np.log(ya), Xa); ok_te = flag_corrupt(np.log(yb), Xb)
    for k, kw in grid.items():
        t0 = time.time(); m = TwoStage(trend="linear", qs_cols=False, **kw).fit(Xa[ok], ya[ok]); p = m.predict(Xb)
        rows.append(dict(fold=c, model=k, **metrics(yb[ok_te], p[ok_te])))
        if k.startswith("base"):
            lev, info = level_forecast(TwoStage(trend="linear", qs_cols=False, **kw).fit(Xa[ok], ya[ok]), Xa[ok], ya[ok], Xb["t_idx"].values)
            p2 = p * np.exp(lev); rows.append(dict(fold=c, model="base + extra local/trend level", **metrics(yb[ok_te], p2[ok_te])))
    print(c, "done", round(time.time() - t0, 1), "s/fit", flush=True)
df = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(df.groupby("model")[["MAE", "RMSE", "MAPE", "MedAPE", "R2"]].mean().round(3))
