"""Unseen-city generalisation (12% of validation rows touch a city absent from train) + corrupted-label size."""
import sys, json, warnings; sys.path.insert(0, "../src")
import numpy as np, pandas as pd
from validate import prepare, metrics
from model import TwoStage, flag_corrupt, FINAL_CFG
warnings.filterwarnings("ignore")
tr, X, y = prepare()
s, e = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-11-01")
itr = (tr.date < s).values; ite = ((tr.date >= s) & (tr.date < e)).values
cities = sorted(set(tr.pickup) | set(tr.delivery)); rng = np.random.RandomState(0); out = []
Xa2, ya2 = X[itr], y[itr]; ok2 = flag_corrupt(np.log(ya2), Xa2)
m_all = TwoStage(**FINAL_CFG).fit(Xa2[ok2], ya2[ok2])
for rep in range(5):
    held = set(rng.choice(cities, 8, replace=False))
    touch = (tr.pickup.isin(held) | tr.delivery.isin(held)).values
    a, b = itr & ~touch, ite & touch
    ok = flag_corrupt(np.log(y[a]), X[a]); okt = flag_corrupt(np.log(y[b]), X[b])
    m = TwoStage(**FINAL_CFG).fit(X[a][ok], y[a][ok])
    out.append(dict(rep=rep, n_test=int(b.sum()), mape_unseen=metrics(y[b][okt], m.predict(X[b])[okt])["MAPE"], mape_allcities=metrics(y[b][okt], m_all.predict(X[b])[okt])["MAPE"]))
df = pd.DataFrame(out); print(df.round(3)); print(df[["mape_unseen", "mape_allcities"]].mean().round(3))
# size of the corrupted labels (ratio of observed rate to model prediction), in-sample on all data
ok = flag_corrupt(np.log(y), X); m = TwoStage(**FINAL_CFG).fit(X[ok], y[ok]); ratio = y[~ok] / m.predict(X[~ok])
print("corrupted rows:", (~ok).sum(), "| ratio quantiles", np.quantile(ratio, [.05, .25, .5, .75, .95]).round(2), "| share >1:", (ratio > 1).mean().round(2),
      "| share outside [0.6,1.6]:", ((ratio < .6) | (ratio > 1.6)).mean().round(3))
json.dump(dict(unseen=df.to_dict("records"), mape_unseen=float(df.mape_unseen.mean()), mape_all=float(df.mape_allcities.mean()),
               corrupt_ratio_q=np.quantile(ratio, [.05, .5, .95]).tolist(), corrupt_n=int((~ok).sum())), open("../results/unseen_and_corrupt.json", "w"), indent=1)
