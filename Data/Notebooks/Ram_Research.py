import pandas as pd, numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- Load ----------
# Gilts now come from the Bank of England fitted spot curve (2y/10y/30y) instead of the
# single 10y par-yield CSV. Don't mix the two sources in one series: spot (zero-coupon)
# and par yields differ slightly in level.
BOE_FILES = ["glc_2024_2025.xlsx"]   # file actually runs to Oct 2026 despite its name
SHEET, HEADER_ROW = "4. spot curve", 3

def load_boe():
    frames = []
    for f in BOE_FILES:
        df = pd.read_excel(f, sheet_name=SHEET, header=HEADER_ROW, index_col=0)
        # Excel headers may read 2, "2", "2.0": normalise to "2.0" so selection is robust
        df.columns = [str(c).strip() for c in df.columns]
        df.columns = [str(float(c)) if c.replace(".", "", 1).isdigit() else c for c in df.columns]
        frames.append(df[["2.0", "10.0", "30.0"]])
    g = pd.concat(frames)
    g.index = pd.to_datetime(g.index, errors="coerce")
    g = g[g.index.notna()].apply(pd.to_numeric, errors="coerce")   # drops the blank spacer row
    g = g[~g.index.duplicated()].sort_index().dropna()   # if files overlap, first file wins
    g.columns = ["g2", "g10", "g30"]
    return g

def load_csv(path, name, skiprows=0):
    # first two columns only (bund has an extra flags column); utf-8-sig strips bund's BOM
    df = pd.read_csv(path, skiprows=skiprows, usecols=[0, 1], encoding="utf-8-sig")
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[name] = pd.to_numeric(df[name], errors="coerce")   # FRED uses "." for holidays
    return df.dropna().set_index("date")[name]

g = load_boe()
ust2, ust10, ust30 = load_csv("ust2.csv", "ust2"), load_csv("ust.csv", "ust10"), load_csv("ust30.csv", "ust30")
bund, brent = load_csv("bund.csv", "bund"  , skiprows=7), load_csv("brent.csv", "brent")

# 1) Align on common business days, forward-fill at most 1 day for holidays
df = pd.concat([g, ust2, ust10, ust30, bund, brent], axis=1).sort_index().ffill(limit=1).dropna()

print(df.shape, df.index.min().date(), df.index.max().date())
print(df.describe().round(2))
print(df.isna().sum().sum(), "NaNs")

# 2) Convert levels to daily changes.
# WHY changes, not levels: yields and oil prices are near-random-walks (non-stationary).
# Regressing one level on another produces "spurious regressions" with high R^2 and
# meaningless betas. Daily changes are roughly stationary, so the betas are interpretable.
# UNITS: yields are in % so diff()*100 gives basis points (bp). Brent is a price, so we
# use log changes x100 = approx. % change (a $2 move means more at $60 than at $120).
d = pd.DataFrame(index=df.index)
for c in ["g2", "g10", "g30"]:
    d[c] = df[c].diff() * 100
# 2s30s slope = 30y minus 2y, in bp (positive = steeper). Its daily change is simply
# the 30y change minus the 2y change, so it is exactly d["g30"] - d["g2"].
df["s2s30s"] = (df["g30"] - df["g2"]) * 100
d["d2s30s"] = df["s2s30s"].diff()
d["ust2"] = df["ust2"].diff() * 100
d["ust10"] = df["ust10"].diff() * 100
d["ust30"] = df["ust30"].diff() * 100
d["bund"] = df["bund"].diff() * 100
d["brent"] = np.log(df["brent"]).diff() * 100
# Timing fix (non-synchronous closes): gilts fix ~4:30pm London, but the US market keeps
# trading for hours after. US news arriving after the London close can only hit gilts
# the NEXT day. Adding yesterday's UST change as a control captures that delayed effect;
# without it, same-day "ust" beta would be understated and the leftover would leak into
# the "UK residual". TRADEOFF: one extra regressor, slightly less precision.
# Each US maturity gets its own lag.
d["ust2_lag"] = d["ust2"].shift(1)
d["ust10_lag"] = d["ust10"].shift(1)
d["ust30_lag"] = d["ust30"].shift(1)
d = d.dropna()  # drops the first row(s) lost to diff()/shift()

# 3) Winsorise ONLY the estimation window (done after the split in step 4 below).
# WHY: a handful of extreme days (data glitches) can dominate an OLS fit, since OLS squares
# errors; clipping them limits their leverage when ESTIMATING betas.
# FIX (earlier version clipped the whole sample): clipping the 2026 test window too made the
# "actual" 2026 moves wrong. Clipped daily changes no longer add up to the true level change
# (e.g. the 2s30s slope showed -14bp when the true move was -35bp), because the big 2y and
# 30y days were shaved. It also used 2026 data to set the clip points (look-ahead). Now the
# clip points come from 2024-25 only and the 2026 test data is left raw, so "actual" equals
# the true level change and the test is cleaner.
# TRADEOFF: a real 2026 shock now enters the residual in full (more honest, noisier).

# 4) Out-of-sample design: fit betas on 2024-2025 only, then apply them to 2026.
# WHY: if we fit on 2026 itself, the model would "explain away" 2026 and the residual
# would be ~zero by construction. Fixing the betas beforehand asks a cleaner question:
# "given how gilts normally reacted to global factors, what did 2026 look like?"
# TRADEOFF: assumes betas are stable; if the relationship shifted in 2026 (regime
# change), part of the "UK residual" is really a beta shift, not UK-specific news.
est = d.loc["2024-01-01":"2025-12-31"].copy()
test = d.loc["2026-01-01":]
for c in est.columns:                      # winsorise estimation window only (see step 3)
    lo, hi = est[c].quantile([0.005, 0.995])
    est[c] = est[c].clip(lo, hi)
# Match each gilt maturity to its own US counterpart: 2y~UST2, 10y~UST10, 30y~UST30. The
# 2y pair matters most: the front end is driven by policy-rate expectations, which UST10
# captures poorly. Bund is the 10y Bund for all (no other German maturities loaded).
FACTORS = {
    "g2":  ["ust2", "ust2_lag", "bund", "brent"],
    "g10": ["ust10", "ust10_lag", "bund", "brent"],
    "g30": ["ust30", "ust30_lag", "bund", "brent"],
}
# HAC (Newey-West) standard errors, 5 lags: daily data show autocorrelation and
# changing volatility, which make plain OLS standard errors too small (overconfident
# t-stats). HAC fixes the SEs without changing the betas. Note that the betas
# themselves are unaffected, only the significance tests.
# Same model for each point on the curve. The betas show how each maturity responds to
# global factors; the 2s30s slope is derived from the residuals' difference, not re-fitted.
results = {}
for y, X in FACTORS.items():
    model = sm.OLS(est[y], sm.add_constant(est[X])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    if y == "g10":
        print(model.summary())   # full output for the benchmark 10y only
    # 5) Decompose the 2026 move into "global" and "UK-specific".
    # pred uses model.params[X] only, deliberately EXCLUDING the constant: the constant is
    # the average daily drift in 2024-25; carrying it into 2026 would add a made-up trend
    # to the residual. resid = actual - predicted = the part global factors do not explain,
    # read as UK-specific (fiscal news, BoE, ...). It also holds noise and omitted global
    # factors, so it is an upper bound on "UK-only". cumsum() = running total since 1 Jan.
    pred = test[X] @ model.params[X]
    resid = test[y] - pred
    results[y] = (test[y].cumsum(), pred.cumsum(), resid.cumsum())
    print(f"{y:>3} beta us={model.params[X[0]]:.2f} bund={model.params['bund']:.2f} R2={model.rsquared:.2f} | "
          f"2026 move: {results[y][0].iloc[-1]:.0f}bp | global: {results[y][1].iloc[-1]:.0f}bp | UK residual: {results[y][2].iloc[-1]:.0f}bp")

# 6) Slope (2s30s) decomposition: was the steepening global or UK-specific?
# The slope is a DIFFERENCE of two yields, so a global factor only moves it if it hits
# the 30y and 2y differently. Both US legs are therefore included (ust30 AND ust2): with
# ust30 alone, a pure US front-end move would show up as a false "UK" slope residual.
# TRADEOFF: more regressors, so noisier betas; the slope is also a noisier target.
Xs = ["ust30", "ust30_lag", "ust2", "ust2_lag", "bund", "brent"]
ms = sm.OLS(est["d2s30s"], sm.add_constant(est[Xs])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(ms.summary().tables[1], f"\nR2 = {ms.rsquared:.2f}")
pred_s = test[Xs] @ ms.params[Xs]       # no constant, as before
resid_s = test["d2s30s"] - pred_s
print(f"2s30s 2026 change: {test['d2s30s'].sum():.0f}bp | global: {pred_s.sum():.0f}bp | UK residual: {resid_s.sum():.0f}bp")

# 7) Bootstrap: is each cumulative "UK residual" distinguishable from zero?
# Two sources of uncertainty feed the residual, and we simulate both:
#   (i)  BETA ERROR: betas were estimated from only ~500 days, so a different sample
#        would have given different betas and a different predicted 2026 move.
#   (ii) DAILY NOISE: even with perfect betas, ~190 days of unexplained daily noise add
#        up to a nonzero cumulative total by chance (roughly sigma*sqrt(n) if independent).
# NULL HYPOTHESIS: "no UK-specific effect, the model holds in 2026". We build the
# distribution of cumulative residual under that null, then ask how extreme the observed
# value is (two-sided p-value) and report a 95% band of what chance alone would produce.
# MOVING-BLOCK resampling (blocks of L consecutive days) rather than single days:
# daily yield changes show volatility clustering/autocorrelation (see the DW stats), and
# resampling single days would destroy that, making the bands too narrow.
# TRADEOFF: the block length is a judgement call (L=10 ~ two trading weeks); more
# blocks-length = more dependence preserved but fewer distinct blocks.
rng = np.random.default_rng(42)   # fixed seed so results are reproducible
B, L = 2000, 10

def block_idx(n, L, rng):
    # concatenate randomly chosen blocks of L consecutive row positions, trimmed to n
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([np.arange(s, s + L) for s in starts])[:n]

def boot_resid(y, X, label):
    Xe = sm.add_constant(est[X]).to_numpy(); ye = est[y].to_numpy()
    Xt = test[X].to_numpy(); yt = test[y].to_numpy()
    n, nt = len(ye), len(yt)
    b_hat = np.linalg.lstsq(Xe, ye, rcond=None)[0]
    observed = (yt - Xt @ b_hat[1:]).sum()          # same definition as before: no constant
    null = np.empty(B)
    null_path = np.empty((B, nt))                   # cumulative null path, for the plot bands
    for i in range(B):
        idx = block_idx(n, L, rng)                  # (i) re-draw the estimation sample
        b_star = np.linalg.lstsq(Xe[idx], ye[idx], rcond=None)[0]
        beta_err = Xt @ (b_hat[1:] - b_star[1:])           # daily shift in predicted 2026 move
        e_star = ye[idx] - Xe[idx] @ b_star
        noise = e_star[block_idx(n, L, rng)[:nt]]          # (ii) daily noise draws
        null_path[i] = np.cumsum(beta_err + noise)         # running total under the null
        null[i] = null_path[i, -1]
    lo, hi = np.percentile(null, [2.5, 97.5])
    p = (np.abs(null) >= abs(observed)).mean()
    print(f"{label:>7}: UK residual {observed:6.1f}bp | null 95% band [{lo:6.1f}, {hi:6.1f}] | "
          f"bootstrap SE {null.std():5.1f}bp | p = {p:.3f}")
    boot_stats[label] = dict(observed=float(observed), lo=float(lo), hi=float(hi), se=float(null.std()), p=float(p))
    # return the 95% band at EVERY date (it widens roughly with sqrt(days), since noise accumulates)
    return np.percentile(null_path, [2.5, 97.5], axis=0), np.cumsum(yt - Xt @ b_hat[1:])

print("\nBootstrap test of cumulative 2026 UK residuals (null = global factors explain everything)")
bands = {}
boot_stats = {}   # filled by boot_resid, used by the exports in step 10
for y, X in FACTORS.items():
    bands[y] = boot_resid(y, X, y)
bands["d2s30s"] = boot_resid("d2s30s", Xs, "2s30s")

# 8) Plot: one panel per series (a separate panel each, never a second y-axis).
# Lines: actual cumulative move, the part global factors explain, and the UK residual.
# The shaded band is the 95% range the residual would stay inside by chance alone if
# global factors explained everything (from the bootstrap above). A residual line that
# stays inside its band = no evidence of a UK-specific effect.
C_ACT, C_GLOB, C_RES = "#2a78d6", "#eb6834", "#1baf7a"   # palette slots 1-3 (blue, orange, aqua)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FACT = {**FACTORS, "d2s30s": Xs}
TITLES = {"g2": "2y gilt", "g10": "10y gilt", "g30": "30y gilt", "d2s30s": "2s30s slope (30y minus 2y)"}
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for ax, (y, X) in zip(axes.ravel(), FACT.items()):
    (lo, hi), _ = bands[y]
    m = sm.OLS(est[y], sm.add_constant(est[X])).fit()
    pr = test[X] @ m.params[X]
    ca, cg, cr = test[y].cumsum(), pr.cumsum(), (test[y] - pr).cumsum()
    ax.fill_between(test.index, lo, hi, color=C_RES, alpha=0.15, linewidth=0)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.plot(test.index, ca, color=C_ACT, linewidth=2)
    ax.plot(test.index, cg, color=C_GLOB, linewidth=2)
    ax.plot(test.index, cr, color=C_RES, linewidth=2)
    # direct end-labels; where two lines finish close together, push the labels apart
    # vertically (in points) so the text stays readable
    ends = sorted([(ca.iloc[-1], "actual"), (cg.iloc[-1], "global"), (cr.iloc[-1], "UK residual")])
    ypos = [v for v, _ in ends]
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for k in range(1, 3):
        ypos[k] = max(ypos[k], ypos[k - 1] + 0.075 * span)
    for (v, lab), yl in zip(ends, ypos):
        ax.annotate(f"{lab} {v:+.0f}bp", (test.index[-1], v), xytext=(test.index[-1] + pd.Timedelta(days=6), yl),
                    textcoords="data", va="center", fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
    ax.set_title(TITLES[y], loc="left", fontsize=11, color=INK)
    ax.set_ylabel("cumulative change since 1 Jan 2026 (bp)", fontsize=8, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.margins(x=0.02); ax.set_xlim(right=test.index[-1] + pd.Timedelta(days=60))
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(labelsize=8, colors=INK2)
fig.suptitle("2026 gilt moves: global factors vs UK residual (shaded = 95% chance band)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig("gilt_decomposition.png", dpi=150)
print("saved gilt_decomposition.png")

plt.show()

# ---------- 9. Event charts, event windows, trade-box helpers ----------
# Residual/global series per target, from the same 2024-25 betas as above (no constant).
def decompose(y, X):
    m = sm.OLS(est[y], sm.add_constant(est[X])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    pred = test[X] @ m.params[X]
    return test[y] - pred, pred          # (UK residual, global-explained), daily bp

res10, glob10 = decompose("g10", FACTORS["g10"])
res_s, glob_s = decompose("d2s30s", Xs)  # uses BOTH US legs (see step 6), not ust30 alone
res30, _ = decompose("g30", FACTORS["g30"])
res2, _ = decompose("g2", FACTORS["g2"])

# Event dates, checked against news reports (Bloomberg/Reuters-style coverage via web search):
#  - 2026-07-20: Burnham became PM, said he would seek "any flexibility" within the fiscal
#    rules, and named Healey Chancellor, all the SAME day (so the earlier separate
#    "any flexibility" placeholder of 2026-08-15 was wrong and is merged into this event).
#  - 2026-07-31: Healey confirmed the Budget date, 28 Oct 2026 (after our data ends).
#  - 2026-09-28: the 10y gilt peak in OUR data (5.41% spot); the earlier 2026-09-01 label
#    was wrong. It coincides with the "Budget run-up" label, so they are one marker.
events = {"2026-07-20": "Burnham PM / Healey Chancellor /\n'any flexibility'",
          "2026-07-31": "Budget date set\n(28 Oct)",
          "2026-09-28": "10y gilt peak /\nBudget run-up"}
peak = df["g10"].loc["2026-01-01":].idxmax()
print(f"\nCheck: actual 2026 10y gilt peak in the data is {peak.date()} ({df['g10'][peak]:.2f}%) "
      f"vs the '10y gilt peak' label on 2026-09-28")
print(f"Check: 10y gilt change on 2026-07-20 in the data: {d.loc['2026-07-20','g10']:.1f}bp "
      f"(press reports ~+8bp on par yields; ours is the fitted spot curve)")

def chart(resid, glob, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(resid.cumsum(), lw=2, label="Cumulative UK-specific residual (bp)")
    ax.plot(glob.cumsum(), lw=1.2, ls="--", label="Global-explained (bp)")
    ax.axhline(0, color="grey", lw=0.6)
    for dt, lab in events.items():
        ax.axvline(pd.Timestamp(dt), color="red", alpha=0.5, lw=1)
        ax.text(pd.Timestamp(dt), ax.get_ylim()[1] * 0.95, lab, fontsize=7, rotation=90, va="top")
    ax.set_title(title); ax.set_ylabel("bp, cumulative from 1 Jan 2026"); ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

chart(res10, glob10, "10y gilt: global vs UK-specific", "exhibit1_10y_residual.png")
chart(res_s, glob_s, "2s30s slope: global vs UK-specific", "exhibit1_2s30s_residual.png")

# Slope LEVEL chart (the traded quantity), in bp
fig, ax = plt.subplots(figsize=(10, 4))
df.loc["2026-01-01":, "s2s30s"].plot(ax=ax, lw=2)
ax.set_title("UK 2s30s spot slope, 2026 (bp)"); ax.set_ylabel("bp")
plt.tight_layout(); plt.savefig("exhibit_2s30s_level.png", dpi=200); plt.close()

# Event windows: sum of the daily residual over [-1, +1] TRADING days around each event.
# Trading days (index positions), not calendar days: a calendar slice of t-1..t+1 around a
# Monday would skip Friday and a weekend date would catch only one day. An event on a
# non-trading day is anchored to the next trading day. TRADEOFF: a 3-day sum is a small
# sample of noisy daily residuals (daily sigma ~3bp), so one window is weak evidence on its
# own; compare to roughly +/- 2*sigma*sqrt(3) before reading anything into a number.
print("\nEvent windows: 2s30s residual / 10y residual / 30y residual / 2y residual (bp)")
for dt, lab in events.items():
    pos = res_s.index.searchsorted(pd.Timestamp(dt))      # first trading day on/after the date
    sl = slice(max(pos - 1, 0), pos + 2)                  # positions pos-1, pos, pos+1
    print(dt, lab.replace("\n", " "),
          f"{res_s.iloc[sl].sum():.1f} / {res10.iloc[sl].sum():.1f} / {res30.iloc[sl].sum():.1f} / {res2.iloc[sl].sum():.1f}")
print(f"(for scale: 3-day residual 1-sigma ~ {res_s.std()*np.sqrt(3):.1f}bp slope, {res10.std()*np.sqrt(3):.1f}bp 10y)")

# Trade-box helpers: daily slope volatility scaled to a 10-day horizon with sqrt(time),
# which assumes independent days (fine as a rough stop-sizing guide, but it understates
# risk if changes trend or cluster).
vol = d["d2s30s"].loc["2025-01-01":].std()
print(f"\nDaily 2s30s vol (2025-26): {vol:.1f}bp | 10-day 1-sigma ~ {vol*np.sqrt(10):.1f}bp (use for stop sizing)")
print("Latest 2s30s level:", round(df['s2s30s'].iloc[-1], 1), "bp on", df.index[-1].date())

# 10) Exports for the thesis note: regression tables, 2026 decomposition, bootstrap stats,
# and the two-panel exhibit (left: 2s30s decomposition with chance band; right: CPI breakdown).
import json
rows, summ = [], {}
for y, X in FACT.items():
    m = sm.OLS(est[y], sm.add_constant(est[X])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    for k in m.params.index:
        rows.append(dict(model=y, var=k, coef=m.params[k], se=m.bse[k], z=m.tvalues[k], p=m.pvalues[k],
                         r2=m.rsquared, n=int(m.nobs)))
    pr = test[X] @ m.params[X]
    summ[y] = dict(actual=float(test[y].sum()), glob=float(pr.sum()), resid=float((test[y] - pr).sum()),
                   r2=float(m.rsquared), n=int(m.nobs), **boot_stats[{"d2s30s": "2s30s"}.get(y, y)])
pd.DataFrame(rows).to_csv("regression_results.csv", index=False)
json.dump(summ, open("decomposition_summary.json", "w"), indent=2)

# Exhibit: left = slope decomposition (cumulative bp since 1 Jan 2026), right = CPI Aug 2026
# CPI figures are annual rates from the ONS bulletin (Aug 2026, released 16 Sep 2026).
fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
(lo_b, hi_b), _ = bands["d2s30s"]
ms_ = sm.OLS(est["d2s30s"], sm.add_constant(est[Xs])).fit()
pr_ = test[Xs] @ ms_.params[Xs]
ca_, cg_, cr_ = test["d2s30s"].cumsum(), pr_.cumsum(), (test["d2s30s"] - pr_).cumsum()
axl.fill_between(test.index, lo_b, hi_b, color=C_RES, alpha=0.15, linewidth=0)
axl.axhline(0, color=GRID, linewidth=1)
for ser, col in [(ca_, C_ACT), (cg_, C_GLOB), (cr_, C_RES)]:
    axl.plot(test.index, ser, color=col, linewidth=2)
ends = sorted([(ca_.iloc[-1], "actual", C_ACT), (cg_.iloc[-1], "global", C_GLOB), (cr_.iloc[-1], "UK residual", C_RES)])
ypos = [e[0] for e in ends]
for k in range(1, 3):
    ypos[k] = max(ypos[k], ypos[k - 1] + 9)
for (v, lab, col), yl in zip(ends, ypos):
    axl.annotate(f"{lab} {v:+.0f}bp", (test.index[-1], v), xytext=(test.index[-1] + pd.Timedelta(days=8), yl),
                 va="center", fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
axl.set_title("Exhibit 1: 2s30s change in 2026, actual vs global vs UK residual (bp)\nShaded: 95% range if global factors explained everything",
              loc="left", fontsize=9, color=INK)
axl.set_xlim(right=test.index[-1] + pd.Timedelta(days=75))
axl.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); axl.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
cpi = [("Motor fuels", 23.0, "energy"), ("Electricity, gas & fuels", 6.0, "energy"),
       ("Restaurants & hotels", 4.1, "dom"), ("Services", 3.4, "dom"), ("Core CPI", 2.6, "dom")]
cols = {"energy": "#e87ba4", "dom": "#4a3aa7"}
axr.barh([c[0] for c in cpi][::-1], [c[1] for c in cpi][::-1], color=[cols[c[2]] for c in cpi][::-1], height=0.6)
for i, c in enumerate(cpi[::-1]):
    axr.text(c[1] + 0.5, i, f"{c[1]:.1f}%", va="center", fontsize=8.5, color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=1))
axr.axvline(3.1, color=INK2, linestyle="--", linewidth=1)
axr.text(3.3, -0.62, "headline 3.1%", fontsize=8, color=INK2, va="center"); axr.set_ylim(-0.9, 4.5)
axr.set_xlim(0, 27); axr.set_title("Exhibit 2: UK CPI, annual rate, Aug 2026 (%)\nPink = energy-linked, violet = domestic / core", loc="left", fontsize=9, color=INK)
for ax in (axl, axr):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK2)
axl.grid(axis="y", color=GRID, linewidth=0.8); axl.set_axisbelow(True)
fig.tight_layout(); fig.savefig("exhibits_note.png", dpi=200)
print("saved regression_results.csv, decomposition_summary.json, exhibits_note.png")
