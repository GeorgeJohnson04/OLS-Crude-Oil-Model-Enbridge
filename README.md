# Oil Price Model — three families, walk-forward validated, with strategy backtest

This project models real (CPI-deflated) crude oil prices for both Light (WTI)
and Heavy (WCS) crude, plus the WCS-WTI differential — the spread that
matters most for Enbridge mainline economics. It also tests whether the
model's signals translate into actual PnL.

## What this model can and can't tell you

| Question | Verdict |
|---|---|
| ❌ Should we buy oil and hold for 5 years? | **The model has no view** — no fundamentals, decarbonization, EV penetration. Buy-and-hold is a fundamentals bet this model isn't built for. |
| ❌ What will WTI be in Aug 2026? | The model gives a number with bands, but the random-walk benchmark beats it on RMSE. Don't commit capital to the point estimate. |
| 🟡 Should we hedge oil exposure this month? | **Maybe** — the model has 55–59% directional accuracy. Useful for sizing a hedge, not for deciding whether to hedge. |
| ✅ Will the WCS-WTI spread widen or narrow? | **Yes, with measurable edge** — Theil U=0.93, 71% directional accuracy. Beats persistence and rolling-mean baselines. |
| ✅ Is there a profitable monthly strategy? | **Yes, on confidence-thresholded entries.** Backtest shows ~30 pp cumulative outperformance vs buy-and-hold over the last 36 months, with 1/8 the drawdown. (Caveat: thin sample of trades — see `output/strategy_results.csv`.) |

## Three model families

| Family | Target variable | What it answers |
|---|---|---|
| **Levels** | log(real price) | What's the equilibrium level given macro factors? |
| **Returns** | Δlog(real price) | What moves prices month-to-month? |
| **Differential** | WCS − WTI ($/bbl) | What drives the heavy/light spread? |

The **levels** family is the original spec — but its R² (0.94–0.99) is mostly
the lagged price explaining itself. The **returns** family is the honest one:
much lower R² (0.55–0.61), but it directly tests whether anything beyond
inertia moves prices. The **differential** model is the Enbridge-relevant
one: how much WCS trades below WTI is largely a pipeline-egress story.

All three families use Newey-West HAC standard errors (`maxlags=6`) to
correct for residual autocorrelation.

## Out-of-sample validation (the punchline)

The model is back-tested **walk-forward** over the most recent 36 months —
refit each month on data strictly before the test date. Compared against:

- **random_walk** — `price_t = price_{t-1}`  (the null hypothesis)
- **ar1** — AR(1) on log price
- **ar1_returns** — AR(1) on log returns
- **seasonal_naive** — `price_t = price_{t-12}`

Honest result on the most recent backtest:

| Crude | Best by RMSE | Best by directional accuracy |
|---|---|---|
| Light (WTI) | random_walk (~$4.6/bbl) | ar1_returns @ 58% |
| Heavy (WCS) | random_walk (~$4.0/bbl) | levels_model @ 59% |
| **WCS-WTI** | **differential_model (Theil U=0.93)** | **differential_model @ 71%** |

**Translation:** for absolute price level forecasting, no model meaningfully
beats "tomorrow looks like today" — this is a well-known property of oil
prices. But the **differential model adds real value** over naive baselines,
and the levels model has useful **directional** signal even when its RMSE is
worse. The boss is right that absolute-price forecasts are weak; the
differential / direction story is the defensible one.

## Data sources

Pulled automatically (free, no API keys):

| Source | Series | Cadence |
|---|---|---|
| FRED | WTI, Brent, CPI, DXY broad | monthly / daily |
| EIA  | US production, inventory, refinery util, imports/exports, **SPR ending stocks** | monthly |
| EIA RAC | Imported Refiner Acquisition Cost (heavy/sour blend, WCS proxy) | monthly |
| Caldara-Iacoviello | GPR (Geopolitical Risk Index) | monthly |
| **CBOE** | **OVX (oil volatility index)** | daily → monthly mean |
| **CFTC** | **Disaggregated COT — managed money WTI net % of OI** | weekly → monthly mean |

User-CSV fallbacks (drop in `data/` to enable; pipeline falls back gracefully):

- `data/wcs_prices.csv` — true WCS instead of EIA RAC proxy  (Date, Price)
- `data/rig_count.csv`  — US oil + gas rig count             (Date, RigCount)
- `data/opec_production.csv` — OPEC crude production         (Date, Production)
- `data/apportionment.csv`   — Enbridge mainline apportionment (Date, Pct)

For **OPEC supply policy** without a paid feed, hand-coded `OPEC_EVENTS` in
`config.py` provide a +/− mbd shock series — covers all major OPEC+
decisions 2008–2025.

## Structural breaks

Sample 2003–2026 spans real regime shifts. Each is a binary dummy:

- `regime_shale_era`     — 2010+
- `regime_covid`         — Mar–Dec 2020
- `regime_russia_war`    — Feb 2022 onward
- `regime_tmx_inservice` — May 2024 onward (especially relevant for differential)

## Forecast methodology

Through August 2026, with **Monte Carlo** (5000 paths) and three scenarios
(base / bull / bear). Each path independently perturbs:

1. **Each exogenous** by its AR(1) residual sd (replaces the prior version's
   "freeze at last value" approach)
2. **The price equation** by its residual sd

Bull/bear scenarios add persistent monthly shocks to production (±2%), DXY
(±2%), and refinery utilization (±1pp).

Outputs: 80% Monte Carlo bands (p10/p90), 95% CI (p2.5/p97.5), and
deterministic point forecast for each scenario.

## Strategy backtest — does the model make money?

`strategy_backtest.py` translates the model's monthly predictions into
positions, applies 5 bps transaction cost, and computes Sharpe / drawdown /
hit rate vs buy-and-hold. Strategies tested:

| Strategy | Position rule |
|---|---|
| `buy_and_hold` | always 100% long. Baseline. |
| `directional_sign` | long when model predicts UP, flat when DOWN. |
| `confidence_threshold` | long only when \|predicted return\| > 0.5σ. |
| `hedge_overlay_residual` | producer naturally long oil, hedges 25–75% of exposure based on directional confidence. |
| `spread_pair_trade` | long WCS / short WTI when diff narrows, opposite when widens. |

Output: `output/strategy_results.csv`, `output/strategy_equity_curves.csv`,
`output/charts/12_strategies.png`.

## Live signals — `signals.py`

Reads the latest data and produces a one-page next-month signal for each
asset, with confidence in residual-sigma units:

```
Light (WTI)         BEARISH (HIGH CONF)   z = -2.99   ($89 -> $77)
Heavy (RAC proxy)   BULLISH (HIGH CONF)   z = +1.21   ($64 -> $68)
WCS - WTI spread    NEUTRAL               z = +0.09
```

Saved to `output/signals.csv`. Confidence guide: |z|<0.5 = NEUTRAL,
|z|<1.0 = low confidence, |z|≥1.0 = high conviction.

## Files

```
config.py               # All parameters: dates, war windows, OPEC events,
                        # structural breaks, regressor lists per family
fetch_data.py           # FRED + EIA + CBOE + CFTC + GPR pulls -> data/panel_raw.csv
build_features.py       # logs, returns, differential, regime dummies,
                        # OPEC events, COT/OVX features -> data/panel_features.csv
run_models.py           # 3 families, 9 model fits -> output/OLS_Model_Results.xlsx
backtest.py             # Walk-forward + baselines + RMSE/MAPE/DA/Theil U
                        # -> output/backtest_results.csv + 08_backtest.png
strategy_backtest.py    # Model-derived strategies vs buy-and-hold w/ tx cost
                        # -> output/strategy_results.csv + 12_strategies.png
signals.py              # Next-month directional + spread signals
                        # -> output/signals.csv
plot_results.py         # 11 evaluation charts -> output/charts/
forecast.py             # Monte Carlo + scenarios + WCS-WTI differential
                        # -> output/forecast.csv, forecast_scenarios.csv,
                        #    07_forecast.png, 09_diff_forecast.png
main.py                 # End-to-end orchestrator (8 steps)
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Honest caveats

1. **Random walk is unbeatable on absolute price RMSE** — and that's a
   well-known finding in commodity econometrics. Anyone claiming a
   structural model crushes random walk on monthly oil prices is either
   over-fitting or measuring wrong.
2. **Levels-model R² is misleading.** It's mostly the AR(1) lag. The
   returns model is the honest one.
3. **WCS proxy is EIA RAC**, not actual WCS. The "differential" we model
   is RAC − WTI, which can sit on either side of zero. Drop a real WCS
   series in `data/wcs_prices.csv` to fix.
4. **Apportionment**, **rig count**, and **OPEC production** would all
   improve the model further; user-CSV slots are wired in.
5. **Forecast scenarios are mild** because the AR(1) on price dominates.
   That's a structural property of the model, not a bug — but if you want
   bigger fans, the returns model (which has lower persistence) is
   probably the better engine for scenario stress-testing in a follow-up.
