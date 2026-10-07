"""
Decompose 2026 moves in the 10y gilt yield into a global part and a UK-specific residual.
Free data replaces Bloomberg:
  gilt.csv  : BoE Statistical Database (IADB), 10y nominal par yield, daily
  ust.csv   : FRED DGS10
  bund.csv  : Bundesbank 10y listed federal securities yield, daily (or investing.com export)
  brent.csv : FRED DCOILBRENTEU
Each CSV needs two columns: date, value. Edit the loader if your headers differ.
pip install pandas numpy statsmodels matplotlib
"""
import pandas as pd, numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load(path, name):
    df = pd.read_csv(path)
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    df[name] = pd.to_numeric(df[name], errors="coerce")   # FRED uses "." for missing
    return df.dropna().set_index("date")[name]

gilt, ust = load("gilt.csv", "gilt"), load("ust.csv", "ust")
bund, brent = load("bund.csv", "bund"), load("brent.csv", "brent")

# 1) Align on common business days, forward-fill at most 1 day for holidays
df = pd.concat([gilt, ust, bund, brent], axis=1).sort_index().ffill(limit=1).dropna()

# 2) Changes: yields in bp, Brent in % (log change x100)
d = pd.DataFrame(index=df.index)
d["gilt"] = df["gilt"].diff() * 100
d["ust"] = df["ust"].diff() * 100
d["bund"] = df["bund"].diff() * 100
d["brent"] = np.log(df["brent"]).diff() * 100
# Timing fix: gilts close ~4:30pm London, US closes later, so US news after
# the London close hits gilts next day. Add lagged UST as a control.
d["ust_lag"] = d["ust"].shift(1)
d = d.dropna()

# 3) Winsorise extreme outliers (data errors, not signal)
for c in d.columns:
    lo, hi = d[c].quantile([0.005, 0.995])
    d[c] = d[c].clip(lo, hi)

# 4) Estimate betas on a PRE-2026 window, apply to 2026 (out-of-sample residual)
est = d.loc["2024-01-01":"2025-12-31"]
test = d.loc["2026-01-01":]
X = ["ust", "ust_lag", "bund", "brent"]
model = sm.OLS(est["gilt"], sm.add_constant(est[X])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(model.summary())

# 5) 2026 residual = actual gilt change - predicted by global factors (no constant)
pred = test[X] @ model.params[X]
resid = test["gilt"] - pred
cum_resid = resid.cumsum()
cum_actual = test["gilt"].cumsum()
cum_global = pred.cumsum()
print(f"2026 gilt move: {cum_actual.iloc[-1]:.0f}bp | global: {cum_global.iloc[-1]:.0f}bp | UK residual: {cum_resid.iloc[-1]:.0f}bp")

# 6) Robustness: full-sample regression incl. 2026, and rolling 120d beta (run separately and report)

# 7) Chart with event markers
events = {"2026-07-20": "Burnham PM /\nHealey Chancellor",
          "2026-08-15": "'Any flexibility'\n(set exact date)",
          "2026-09-01": "10y gilt 5.21%",
          "2026-10-07": "Budget run-up"}
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cum_resid.index, cum_resid, lw=2, label="Cumulative UK-specific residual (bp)")
ax.plot(cum_global.index, cum_global, lw=1.2, ls="--", label="Global-explained (bp)")
ax.axhline(0, color="grey", lw=0.6)
for dt, lab in events.items():
    ax.axvline(pd.Timestamp(dt), color="red", alpha=0.5, lw=1)
    ax.text(pd.Timestamp(dt), ax.get_ylim()[1]*0.95, lab, fontsize=7, rotation=90, va="top")
ax.set_ylabel("bp, cumulative from 1 Jan 2026")
ax.set_title("10y gilt: global vs UK-specific component, 2026")
ax.legend(loc="upper left"); plt.tight_layout()
plt.savefig("exhibit1_residual.png", dpi=200)

# 8) Event-window table: residual sum over [-1,+1] days around each date
for dt, lab in events.items():
    t = pd.Timestamp(dt)
    w = resid.loc[t - pd.Timedelta(days=1): t + pd.Timedelta(days=1)]
    print(dt, lab.replace("\n", " "), f"{w.sum():.1f}bp")
