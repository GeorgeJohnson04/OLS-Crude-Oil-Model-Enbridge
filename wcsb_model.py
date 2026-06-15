"""
WCSB Tracker - production forecast + egress utilization model.

The existing price models take WCSB production as a regressor (an INPUT).
This model is the supply-side complement: it forecasts WCSB production
itself, tracks total egress capacity over time, and outputs an apportionment-
risk signal.

Three objects produced:

  1. WCSB production forecast (AR(1) with price + regime regressors)
  2. Aggregate egress capacity as a step-function over time (sum of active
     pipelines + rail)
  3. Utilization rate and "basis-pressure" signal = max(0, utilization - 0.90)

Why this matters:
  - When utilization is below ~85-90%, basis trades close to its structural
    transport floor (~-$11/bbl).
  - As utilization rises through 90-95%, marginal barrels start needing rail
    (premium-priced) - basis widens.
  - Above ~95%, apportionment kicks in on Enbridge Mainline - basis can blow
    out to -$25 or worse (e.g. 2018 glut).

Outputs:
  output/wcsb_tracker.csv             monthly production / capacity / utilization
  output/charts/14_wcsb_balance.png   production vs capacity over time + forecast
  output/charts/15_basis_pressure.png utilization and basis-pressure signal
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import config

warnings.filterwarnings("ignore", category=Warning)

FORECAST_END = pd.Timestamp("2026-08-01")
N_SIM = 2000

# Palette
INK   = "#1A1A2E"
AMBER = "#E8893E"
TEAL  = "#16697A"
RED   = "#A8324E"
GRAY  = "#555560"

# Utilization thresholds for the basis-pressure regime classification
UTIL_NORMAL    = 0.85   # below: structural floor, basis ~ -$11
UTIL_TIGHT     = 0.95   # 85-95%: rail needed, basis widens
UTIL_CRITICAL  = 1.00   # 95-100%: apportionment kicks in, basis can blow out


def _aggregate_egress_capacity(index: pd.DatetimeIndex) -> pd.Series:
    """Sum of all pipelines + rail that are in service at each month."""
    out = pd.Series(0.0, index=index, name="egress_capacity_kbpd")
    for name, kbpd, in_serv, retired in config.WCSB_EGRESS_CAPACITIES:
        in_ts  = pd.Timestamp(in_serv)
        ret_ts = pd.Timestamp(retired) if retired else pd.Timestamp("2099-12-31")
        active = (index >= in_ts) & (index < ret_ts)
        out.loc[active] += kbpd
    return out


def _fit_production_model(df: pd.DataFrame) -> tuple[sm.regression.linear_model.RegressionResults, list[str]]:
    """Fit a small regression for log Canadian production.

    Spec: log_can_prod_t ~ log_can_prod_{t-1} + log_wcs_real_{t-1}
                          + regime_tmx + regime_shale + seasonality
    """
    df = df.copy()
    df["log_can_prod_lag1"]  = df["log_can_production"].shift(1)
    df["log_wcs_real_lag1"]  = df["log_wcs_real"].shift(1)

    regs = [
        "log_can_prod_lag1",
        "log_wcs_real_lag1",     # higher price last month -> drilling/production response
        "regime_tmx_inservice",  # TMX unlocked oil-sands projects waiting for egress
        "regime_shale_era",
        "month_sin",
        "month_cos",
    ]
    regs = [r for r in regs if r in df.columns]
    sub = df[["log_can_production"] + regs].dropna()
    X = sm.add_constant(sub[regs])
    res = sm.OLS(sub["log_can_production"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return res, regs


def _forecast_production(df: pd.DataFrame, res, regs: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Recursive AR(1) forecast of Canadian production through FORECAST_END.
    Returns the base path (Series) and a percentile DataFrame from Monte Carlo."""
    last_log = float(df["log_can_production"].dropna().iloc[-1])
    last_wcs = float(df["log_wcs_real"].dropna().iloc[-1])
    last_date = df["log_can_production"].dropna().index.max()
    horizon = pd.date_range(last_date + pd.DateOffset(months=1), FORECAST_END, freq="MS")
    if len(horizon) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    sigma = float(np.sqrt(res.mse_resid))
    rng = np.random.default_rng(42)
    coef = res.params

    paths = np.empty((N_SIM, len(horizon)))
    tmx_start = pd.Timestamp(config.TMX_INSERVICE_DATE)

    for s in range(N_SIM):
        prev = last_log
        for t, d in enumerate(horizon):
            x = {
                "log_can_prod_lag1":     prev,
                "log_wcs_real_lag1":     last_wcs,        # held flat (sensitivity in the price model)
                "regime_tmx_inservice":  1.0 if d >= tmx_start else 0.0,
                "regime_shale_era":      1.0,
                "month_sin":             np.sin(2 * np.pi * d.month / 12),
                "month_cos":             np.cos(2 * np.pi * d.month / 12),
            }
            mean_log = coef.get("const", 0.0) + sum(coef.get(r, 0.0) * x[r] for r in regs)
            mean_log += rng.normal(0, sigma)
            paths[s, t] = mean_log
            prev = mean_log

    levels = np.exp(paths)
    pct = pd.DataFrame({
        "date":  horizon,
        "prod_pred": np.exp(paths.mean(axis=0)),
        "prod_p10":  np.percentile(levels, 10, axis=0),
        "prod_p90":  np.percentile(levels, 90, axis=0),
    }).set_index("date")
    return pct["prod_pred"], pct


def _classify_pressure(util: float) -> str:
    if util < UTIL_NORMAL:    return "NORMAL (basis near floor)"
    if util < UTIL_TIGHT:     return "TIGHT (rail needed, basis widens)"
    if util < UTIL_CRITICAL:  return "CRITICAL (apportionment risk)"
    return "OVERSUPPLIED (basis blows out)"


def _chart_balance(hist: pd.DataFrame, fcst: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    cutoff = FORECAST_END - pd.DateOffset(years=8)
    h = hist.loc[hist.index >= cutoff]
    f = fcst.loc[fcst.index >= cutoff]

    # Production (history)
    ax.plot(h.index, h["production_kbpd"], color=TEAL, lw=2.0, label="WCSB production (historical)")
    # Production forecast band + line
    last_date = h.index.max()
    xs = [last_date] + list(f.index)
    ys = [float(h["production_kbpd"].iloc[-1])] + list(f["production_kbpd"].values)
    ax.plot(xs, ys, color=TEAL, lw=2.0, ls="--", label="WCSB production (forecast)")
    ax.fill_between(f.index, f["prod_p10"], f["prod_p90"],
                    color=TEAL, alpha=0.15, lw=0, label="80% Monte Carlo band")

    # Total egress capacity (step-function)
    ax.step(h.index, h["egress_capacity_kbpd"], color=AMBER, lw=2.0, where="post",
            label="Total egress capacity")
    ax.step(f.index, f["egress_capacity_kbpd"], color=AMBER, lw=2.0, where="post", ls="--")

    # Annotate TMX in-service
    tmx = pd.Timestamp(config.TMX_INSERVICE_DATE)
    if tmx >= cutoff:
        ax.axvline(tmx, color=GRAY, lw=0.8, ls=":", alpha=0.7)
        ax.text(tmx, ax.get_ylim()[1] * 0.96, "  TMX in-service",
                fontsize=9, color=GRAY, va="top", ha="left")

    ax.set_title("WCSB supply-egress balance — production vs total available egress capacity",
                 fontsize=13)
    ax.set_ylabel("kbbl / day")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _chart_pressure(hist: pd.DataFrame, fcst: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    cutoff = FORECAST_END - pd.DateOffset(years=8)
    h = hist.loc[hist.index >= cutoff]
    f = fcst.loc[fcst.index >= cutoff]

    # Utilization line, history
    ax.plot(h.index, h["utilization"] * 100, color=TEAL, lw=2.0, label="Utilization (history)")
    # Forecast
    last_date = h.index.max()
    xs = [last_date] + list(f.index)
    ys = [float(h["utilization"].iloc[-1]) * 100] + list(f["utilization"].values * 100)
    ax.plot(xs, ys, color=TEAL, lw=2.0, ls="--", label="Utilization (forecast)")

    # Threshold bands
    ax.axhspan(0, UTIL_NORMAL * 100, color="#C8E6C9", alpha=0.30,  label=f"<{UTIL_NORMAL*100:.0f}% — normal")
    ax.axhspan(UTIL_NORMAL * 100, UTIL_TIGHT * 100, color="#FFF59D", alpha=0.30,
               label=f"{UTIL_NORMAL*100:.0f}-{UTIL_TIGHT*100:.0f}% — tight (rail needed)")
    ax.axhspan(UTIL_TIGHT * 100, UTIL_CRITICAL * 100, color="#FFCDD2", alpha=0.40,
               label=f"{UTIL_TIGHT*100:.0f}-{UTIL_CRITICAL*100:.0f}% — apportionment risk")
    ax.axhspan(UTIL_CRITICAL * 100, max(140, (h["utilization"].max() * 100) + 10),
               color="#EF9A9A", alpha=0.45, label=f">{UTIL_CRITICAL*100:.0f}% — oversupplied (basis blows out)")

    ax.set_ylim(40, max(140, h["utilization"].max() * 100 + 5))
    ax.set_title("WCSB egress utilization — apportionment / basis-pressure regime",
                 fontsize=13)
    ax.set_ylabel("Utilization (%)")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _chart_stack(df: pd.DataFrame, out_path):
    """Stacked-area chart of the WCSB production stack: in-situ bitumen,
    mining bitumen, SCO, and conventional, all in pipeline-marketable units."""
    fig, ax = plt.subplots(figsize=(13, 6))
    cutoff = FORECAST_END - pd.DateOffset(years=10)
    d = df.loc[df.index >= cutoff].copy()

    # Build the stack in marketable terms (dilbit for raw bitumen, level for the rest)
    if "bitumen_for_dilbit_kbpd" in d.columns:
        insitu_share = (d.get("aer_insitu_bitumen_kbpd", 0) /
                        d["raw_bitumen_kbpd"].replace(0, np.nan)).fillna(0.5)
        mining_share = 1 - insitu_share
        d["insitu_dilbit_kbpd"] = d["dilbit_volume_kbpd"] * insitu_share
        d["mining_dilbit_kbpd"] = d["dilbit_volume_kbpd"] * mining_share
        sco = d.get("sco_kbpd", 0)
        conv = d.get("aer_conventional_kbpd", 0)
        cond = d.get("aer_condensate_kbpd", 0)
        stack_components = [
            ("In-situ (as dilbit)", d["insitu_dilbit_kbpd"], "#B83A4B"),
            ("Mining (as dilbit)",  d["mining_dilbit_kbpd"], "#E8893E"),
            ("SCO (synthetic)",     sco,                     "#16697A"),
            ("Conventional",        conv,                    "#7A8C68"),
            ("Condensate",          cond,                    "#CADCFC"),
        ]
        stack = [c[1] if hasattr(c[1], "fillna") else pd.Series(c[1], index=d.index) for c in stack_components]
        labels = [c[0] for c in stack_components]
        colors = [c[2] for c in stack_components]
        ax.stackplot(d.index, *[s.fillna(0).values for s in stack],
                     labels=labels, colors=colors, alpha=0.85,
                     edgecolor="white", linewidth=0.4)
        ax.plot(d.index, d["marketable_wcsb_kbpd"], color="black", lw=1.3,
                label="Marketable WCSB total", ls="--")
    else:
        # Fallback: just the aggregate
        ax.plot(d.index, d["canadian_production"], color=TEAL, lw=2,
                label="Canadian production (aggregate)")

    ax.set_title("WCSB production stack — marketable volume by stream (after dilbit blending)",
                 fontsize=13)
    ax.set_ylabel("kbbl / day")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)

    # Prefer marketable_wcsb_kbpd (AER blending applied); fall back to aggregate
    if "marketable_wcsb_kbpd" in df.columns and df["marketable_wcsb_kbpd"].notna().sum() > 24:
        prod_col = "marketable_wcsb_kbpd"
        prod_source = "AER ST39+ST53 with dilbit blending"
    elif "canadian_production" in df.columns and df["canadian_production"].notna().sum() > 24:
        prod_col = "canadian_production"
        prod_source = "Stats Canada aggregate (no breakdown)"
    else:
        print("WCSB model needs canadian_production or AER breakdown. Drop required CSVs into data/.")
        return
    print(f"Production source: {prod_source}  (col = {prod_col})\n")

    # Mirror to canonical names used downstream. ALWAYS rebuild log_can_production
    # from the chosen source so the OLS forecast is in the same units as history.
    df["_wcsb_prod_kbpd"] = df[prod_col]
    df["log_can_production"] = np.log(df["_wcsb_prod_kbpd"].replace(0, np.nan))

    # ---- 1. Egress capacity over the whole timeline (history + forecast) ----
    full_index = pd.date_range(df.index.min(), FORECAST_END, freq="MS")
    cap = _aggregate_egress_capacity(full_index)

    # ---- 2. Fit production model and forecast ----
    res, regs = _fit_production_model(df)
    n_obs = int(res.nobs)
    print(f"\nWCSB production model fit: n={n_obs}, R^2={res.rsquared:.3f}, "
          f"adj R^2={res.rsquared_adj:.3f}")
    print("Coefficients (with HAC SEs):")
    for var, beta, se, p in zip(res.params.index, res.params.values,
                                 res.bse.values, res.pvalues.values):
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {var:<26s}  coef={beta:>+8.4f}   SE={se:.4f}   p={p:.4f} {sig}")

    fcst_series, fcst_pct = _forecast_production(df, res, regs)

    # ---- 3. Build the unified tracker dataframe (history + forecast) ----
    hist = pd.DataFrame(index=df.index)
    hist["production_kbpd"]      = df["_wcsb_prod_kbpd"]
    hist["egress_capacity_kbpd"] = cap.reindex(df.index)
    hist["utilization"]          = hist["production_kbpd"] / hist["egress_capacity_kbpd"]
    hist["basis_pressure"]       = (hist["utilization"] - UTIL_NORMAL).clip(lower=0)
    hist["regime"]               = hist["utilization"].apply(_classify_pressure)

    fcst = pd.DataFrame(index=fcst_pct.index)
    fcst["production_kbpd"]      = fcst_pct["prod_pred"]
    fcst["prod_p10"]             = fcst_pct["prod_p10"]
    fcst["prod_p90"]             = fcst_pct["prod_p90"]
    fcst["egress_capacity_kbpd"] = cap.reindex(fcst.index)
    fcst["utilization"]          = fcst["production_kbpd"] / fcst["egress_capacity_kbpd"]
    fcst["basis_pressure"]       = (fcst["utilization"] - UTIL_NORMAL).clip(lower=0)
    fcst["regime"]               = fcst["utilization"].apply(_classify_pressure)

    # ---- 4. Save and print summary ----
    combined = pd.concat([
        hist.assign(period="historical"),
        fcst.assign(period="forecast"),
    ])
    combined.index.name = "date"
    out_csv = config.OUTPUT_DIR / "wcsb_tracker.csv"
    combined.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    # Latest history vs end of forecast
    hist_valid = hist.dropna(subset=["production_kbpd"])
    last_h = hist_valid.iloc[-1]
    last_f = fcst.iloc[-1]
    print(f"\nCurrent state ({hist_valid.index.max().date()}):")
    print(f"  production: {last_h['production_kbpd']:>5.0f} kbpd  "
          f"capacity: {last_h['egress_capacity_kbpd']:>5.0f} kbpd  "
          f"utilization: {last_h['utilization']*100:>5.1f}%  "
          f"-> regime: {last_h['regime']}")
    print(f"\nForecast end ({fcst.index.max().date()}):")
    print(f"  production: {last_f['production_kbpd']:>5.0f} kbpd  "
          f"capacity: {last_f['egress_capacity_kbpd']:>5.0f} kbpd  "
          f"utilization: {last_f['utilization']*100:>5.1f}%  "
          f"-> regime: {last_f['regime']}")

    # ---- 5. Charts ----
    out1 = config.CHARTS_DIR / "14_wcsb_balance.png"
    _chart_balance(hist, fcst, out1)
    print(f"Saved: {out1}")

    out2 = config.CHARTS_DIR / "15_basis_pressure.png"
    _chart_pressure(hist, fcst, out2)
    print(f"Saved: {out2}")

    # Stacked-area chart of the WCSB production stack (only meaningful with AER breakdown)
    if "dilbit_volume_kbpd" in df.columns:
        out3 = config.CHARTS_DIR / "16_wcsb_stack.png"
        _chart_stack(df, out3)
        print(f"Saved: {out3}")
        # Print decomposition snapshot
        last = df.dropna(subset=["raw_bitumen_kbpd"]).iloc[-1]
        print(f"\nLast available decomposition ({df.dropna(subset=['raw_bitumen_kbpd']).index.max().date()}):")
        print(f"  Mining bitumen (raw):    {last.get('aer_mining_bitumen_kbpd', 0):>6.0f} kbpd")
        print(f"  In-situ bitumen (raw):   {last.get('aer_insitu_bitumen_kbpd', 0):>6.0f} kbpd")
        print(f"  Raw bitumen total:       {last['raw_bitumen_kbpd']:>6.0f} kbpd")
        print(f"  Bitumen -> upgrader:     {last['bitumen_to_upgrader_kbpd']:>6.0f} kbpd  -> SCO {last.get('sco_kbpd', 0):>4.0f}")
        print(f"  Bitumen -> dilbit:       {last['bitumen_for_dilbit_kbpd']:>6.0f} kbpd  x{config.DILBIT_VOLUME_FACTOR:.3f}")
        print(f"  Dilbit volume:           {last['dilbit_volume_kbpd']:>6.0f} kbpd")
        print(f"  Conventional + condensate: {last.get('aer_conventional_kbpd',0) + last.get('aer_condensate_kbpd',0):>4.0f} kbpd")
        print(f"  MARKETABLE WCSB:         {last['marketable_wcsb_kbpd']:>6.0f} kbpd")
        print(f"  (Diluent demand to make all the dilbit: {last['diluent_demand_kbpd']:.0f} kbpd)")


if __name__ == "__main__":
    main()
