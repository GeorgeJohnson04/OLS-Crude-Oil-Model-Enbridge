# WTI & WCS Oil Price Forecasting Model

A reproducible monthly forecast pipeline for light (WTI) and heavy (WCS) crude oil prices, with a focus on Enbridge mainline egress economics. Targeted at the Tidal Energy US trading desk: outputs a 4-month forecast with confidence bands, scenario analysis, an Excel diagnostics workbook, a 16-slide presentation deck, and an 8-page equation reference PDF.

## What this codebase does

- **Forecasts WTI and WCS** monthly through August 2026 using a long-conflict regression regime
- **Models the WCS–WTI differential** (heavy–light spread) directly — the Enbridge-relevant object
- **Time-varying volatility** via GJR-GARCH(1,1) — feeds realistic Monte Carlo confidence bands
- **Validates out-of-sample** against random-walk, AR(1), and seasonal-naive baselines (walk-forward, 36 months)
- **Incorporates pipeline tariffs** for Edmonton → US Gulf Coast egress; corrects the EIA heavy proxy to a wellhead-equivalent WCS via netback subtraction
- **Enforces a physical constraint** that Heavy ≤ Light − $3/bbl (quality discount floor)

## Run the full pipeline

```bash
python main.py
```

This executes, in order:

1. `fetch_data.py` — pull from FRED, EIA, CBOE, CFTC, Iacoviello GPR
2. `build_features.py` — CPI deflation, log transforms, regime dummies, lagged terms
3. `run_models.py` — fit Levels + Returns + Differential families with Newey-West HAC SEs; runs Ljung-Box, ADF/Engle-Granger cointegration, and Lasso pruning
4. `vol_model.py` — fit GJR-GARCH(1,1) per crude
5. `backtest.py` — walk-forward out-of-sample backtest
6. `forecast.py` — Monte Carlo forecast (5,000 paths) with scenarios and the physical floor
7. `plot_results.py` — all chart artifacts
8. `tariff_chart.py` — pipeline tariff visualization
9. `build_presentation.py` — 16-slide PowerPoint deck
10. `make_equations_pdf.py` — 8-page equation reference PDF

## File map

| File | Purpose |
|---|---|
| `config.py` | All knobs: data sources, war windows, regressor lists, pipeline tariffs, netback constants |
| `fetch_data.py` | Data ingestion + WCS netback adjustment (delivered RAC → wellhead WCS) |
| `build_features.py` | Feature engineering pipeline |
| `run_models.py` | OLS fits for 3 model families + diagnostic sheets |
| `vol_model.py` | GJR-GARCH(1,1) volatility model |
| `forecast.py` | Monte Carlo forecast with GARCH vol + scenarios + Heavy ≤ Light constraint |
| `backtest.py` | Walk-forward backtest vs naive baselines |
| `build_presentation.py` | `python-pptx` deck builder |
| `make_equations_pdf.py` | Equation-spec PDF |
| `tariff_chart.py` | Pipeline tariff bar-chart generator |
| `data/` | `panel_raw.csv`, `panel_features.csv`, optional user CSVs |
| `output/` | All generated artifacts (xlsx, pptx, pdf, csv, charts) |

## The six equations

| # | Equation | What it produces |
|---|---|---|
| 1 | `log(WTI_real)_t = β_0 + β_1·log(WTI_real)_{t-1} + Σ β_i X_i + ε_t` | WTI price forecast |
| 2 | `(WCS-WTI)_t = γ_0 + γ_1·(WCS-WTI)_{t-1} + γ_2·egress_{t-1} + Σ γ_j Z_j + η_t` | Heavy-light spread forecast |
| 3 | `WCS_t = WTI_t + (WCS-WTI)_t` subject to `(WCS-WTI)_t ≤ -$3` | Heavy crude (derived) |
| 4 | `σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·1[ε_{t-1}<0] + β·σ²_{t-1}` | Monthly volatility |
| 5 | `ln_P_{k,t} = X_t·β̂ + σ_t·z_{k,t}, z~N(0,1)`, k = 1..5000 | Monte Carlo paths |
| 6 | `egress_t = $12.50 if t < May 2024, else $11.00` | Pipeline netback cost |

Full annotated equations live in `output/Model_Equations.pdf`.

## Data sources

| Source | Series | Used for |
|---|---|---|
| FRED | MCOILWTICO, MCOILBRENTEU, CPIAUCSL, MCRFPUS2, MCESTUS1, WPULEUS3, DTWEXBGS | Prices, CPI, production, inventory, refinery util, USD |
| EIA | R0000____3 (Imported RAC heavy/sour blend) | Heavy crude proxy (netback to wellhead) |
| CBOE | OVX | Oil volatility index |
| CFTC | COT managed-money positioning | Speculator flow |
| Iacoviello/Caldara | Geopolitical Risk Index | Geopolitical context |
| Hand-coded (in `config.py`) | OPEC+ events, war windows, structural breaks, pipeline tariffs | Regime and event features |

## Output artifacts

| Path | What it is |
|---|---|
| `output/forecast.csv` | Base-case forecast with Monte Carlo bands |
| `output/forecast_scenarios.csv` | Base / bull / bear scenarios |
| `output/forecast_differential.csv` | WCS-WTI spread forecast |
| `output/backtest_results.csv` | RMSE / MAE / MAPE / DA / Theil's U vs naive baselines |
| `output/vol_params.csv`, `vol_forecast.csv` | GJR-GARCH parameters and forecast |
| `output/OLS_Model_Results.xlsx` | All model summaries, diagnostics, Lasso, cointegration |
| `output/charts/*.png` | 12 chart artifacts |
| `output/Oil_OLS_Model_Presentation.pptx` | 16-slide deck |
| `output/Model_Equations.pdf` | 8-page equation reference |

## Common tasks

**Refit everything from fresh data**
```bash
python main.py
```

**Regenerate just the deck (no refit)**
```bash
python build_presentation.py
```

**Regenerate the equation PDF**
```bash
python make_equations_pdf.py
```

**Add a new pipeline tariff route**
Edit `PIPELINE_TARIFFS` in `config.py`, then:
```bash
python tariff_chart.py && python build_presentation.py
```

**Change forecast horizon**
Edit `FORECAST_END` in both `forecast.py` and `vol_model.py`.

**Override the heavy crude proxy with a clean WCS series**
Drop `data/wcs_prices.csv` (columns: `Date,Price`) — `fetch_data.py` will use it instead of the EIA RAC netback.

## Architecture notes

- **Real prices**: all prices CPI-deflated to a fixed base year (`config.REAL_PRICE_BASE_YEAR = 2025`)
- **Regime segmentation**: models fit separately on short-conflict and long-conflict subsets (B2 segmenter, threshold `config.LONG_WAR_MONTHS = 6`)
- **HAC standard errors**: Newey-West, maxlags = 6, on every OLS fit
- **WCS netback adjustment**: EIA Imported RAC is delivered into US refineries — subtract `HEAVY_NETBACK_TRANSPORT` ($11/bbl post-TMX, $12.50 pre-TMX) for wellhead-equivalent WCS at Hardisty
- **Physical constraint**: `Heavy ≤ Light - MIN_QUALITY_DISCOUNT` ($3/bbl) enforced in `forecast.py` (binds <1% of Monte Carlo cells)
- **CPI handling**: latest month's CPI typically NaN at publication time; forward-filled in `build_features.py` so real-price series stay continuous

## Known limitations

- **Level forecasts don't beat random walk on RMSE.** Use the Returns model for direction (~57% accuracy) and the Differential model for the spread — the latter is the only model that beats its naive baseline out-of-sample.
- **Heavy crude proxy uses netback-adjusted EIA RAC**, not pure WCS. A clean Hardisty daily series would sharpen the model.
- **Forecast assumes today's macro conditions hold** through the horizon. A regime shift (escalation, deep recession, pipeline outage) would shift the picture meaningfully.
- **Bull/bear scenarios are tight** (~$1 spread for Light). The real uncertainty lives in the GARCH-driven Monte Carlo fan, not in the named scenarios.

## Dependencies

See `requirements.txt`. Key packages:

| Package | Use |
|---|---|
| `pandas`, `numpy` | Data handling |
| `statsmodels` | OLS, HAC SEs, Ljung-Box, ADF, cointegration |
| `arch` | GJR-GARCH(1,1) |
| `scikit-learn` | LassoCV feature pruning |
| `matplotlib` | All charts + equation PDF |
| `python-pptx` | Deck generation |
| `pymupdf` | PDF rendering / QA |
| `openpyxl` | Excel workbook |

## Conventions for working in this repo

- **Don't refit by hand.** Always run `python main.py` after editing a regressor list or data source — partial refits leave the deck and equation PDF stale.
- **Edit `config.py` first.** Almost every change (regressors, tariffs, war windows, structural breaks) is a one-line config edit before any pipeline code touches it.
- **CSVs in `data/` are user-overrides.** Drop `wcs_prices.csv`, `rig_count.csv`, `opec_production.csv`, or `apportionment.csv` to override scraped or proxied values.
- **Charts are numbered.** `01_*` through `12_*` correspond to slide positions; renaming breaks the deck builder.
- **Date format throughout: `YYYY-MM-01`.** Monthly observations indexed to the first of the month.

---

*Last refresh: May 2026 — forecast horizon through August 2026.*
*Reproducibility: every artifact in `output/` can be regenerated by running `python main.py`.*
