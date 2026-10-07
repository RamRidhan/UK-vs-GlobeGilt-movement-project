<div align="center">

# 🇬🇧 UK Curve: A Budget Steepener

**Is the UK gilt curve pricing fiscal risk ahead of the 28 October Budget?**
A global-vs-UK decomposition of 2026 gilt moves, and the trade it suggests.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Method](https://img.shields.io/badge/Method-OLS%20%2B%20HAC%20%2B%20block%20bootstrap-1F3864)
![Status](https://img.shields.io/badge/Note-dated%208%20Oct%202026-success)
![Type](https://img.shields.io/badge/Personal%20research-not%20investment%20advice-lightgrey)

**By Ram Ridhan** · 8 October 2026

[📄 Read the note](Research%20reports/UK_Curve_Thesis_Note.docx) · [🧮 Code](Data/Notebooks/Ram_Research.py) · [📊 Regression output](Data/regression_results.csv)

</div>

---

## 💡 The thesis in one paragraph

Despite fiscal headroom roughly halving (£23.6bn → ~£13bn) and a new Chancellor's first Budget, the **30y gilt carries no measurable UK premium** (+3bp residual in 2026). The UK-specific move sits in the **2y (+28bp)**, driven by BoE hike risk on UK energy inflation. A **2s30s steepener** (long 2y / short 30y, DV01-neutral) is cheap insurance against fiscal slippage on 28 October.

<div align="center">

![Exhibits](Research%20reports/exhibits_note.png)

</div>

## 🎯 The trade

| | |
|:--|:--|
| **Position** | Long 2s30s steepener, DV01-neutral |
| **Entry** | ~135bp (6 Oct, BoE spot curve) |
| **Target** | 155bp (+20bp, about 1σ of horizon vol) |
| **Stop** | 115bp (−20bp) |
| **Horizon** | Exit 30 Oct, after the Budget (no view on the 5 Nov MPC) |
| **Reward / risk** | 1 : 1, small size |

## 🔬 What the data say

| 2026 cumulative move | Actual | Explained by global factors | **UK residual** | p-value |
|:--|--:|--:|--:|--:|
| 2y gilt | +103bp | +75bp | **+28bp** | 0.51 |
| 10y gilt | +83bp | +84bp | **−1bp** | 0.97 |
| 30y gilt | +68bp | +66bp | **+3bp** | 0.94 |
| 2s30s slope | −35bp | −15bp | **−19bp** | 0.66 |

> ⚠️ **Honest caveat:** no residual is statistically significant, and the test cannot detect a UK effect smaller than roughly 70–90bp. The point estimates are suggestive, not proof. This is an event view, not a measured mispricing.

## 🧠 Method

1. **Estimate** each gilt maturity's daily change (bp) on the matching US Treasury (plus a 1-day lag, since the US closes after London), the 10y Bund and Brent.
   OLS over **2024-01-02 to 2025-12-31** (n = 513) with **HAC / Newey-West** standard errors.
2. **Apply** the fitted betas to 2026 to get the part of each move that global factors explain.
3. **UK residual** = actual − predicted, cumulated over the year.
4. **Test** with a 2,000-draw **block bootstrap** (10-day blocks) to build a 95% "chance band" for the residual.

## 🗂️ Repository

```
├── Research reports/
│   ├── UK_Curve_Thesis_Note.docx     ← the published note (start here)
│   └── *.png                         ← exhibits
├── Data/
│   ├── Notebooks/Ram_Research.py     ← full analysis and chart code
│   ├── regression_results.csv        ← model coefficients and z-stats
│   ├── gilt.csv, ust*.csv, bund.csv, brent.csv
│   └── glc_2024_2025.xlsx            ← BoE gilt spot curve
└── Gilt_Decomposition_INTERVIEW_Report.docx
```

## ▶️ Run it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy statsmodels matplotlib openpyxl scipy
python Data/Notebooks/Ram_Research.py
```

> The script reads the CSVs by file name. Run it from the folder containing the data, or update the paths at the top of the file.

## 📚 Sources

[ONS CPI, Aug 2026](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/august2026) ·
[BoE Monetary Policy Summary, Sep 2026](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/september-2026) ·
[BoE gilt-sales market notice, 17 Sep 2026](https://www.bankofengland.co.uk/markets/market-notices/2026/asset-purchase-facility-gilt-sales-market-notice-17-september-2026) ·
[DMO revised remit](https://www.dmo.gov.uk/media/ajmifgdv/pr230426_2.pdf) ·
[HM Treasury Budget-date letter](https://gov.uk/government/publications/chancellor-letter-to-the-treasury-select-committee-tsc-budget-2026-date)

## ⚖️ Disclaimer

Personal views, not investment advice. Levels come from the Bank of England fitted spot curve and are not tradable prices.
