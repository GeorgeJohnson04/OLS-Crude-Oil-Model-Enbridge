"""
Monte Carlo price forecast through August 2026.

Design notes:
  1. Residual noise uses the GJR-GARCH conditional vol path (vol_forecast.csv
     from vol_model.py) rather than a constant sigma, so the fan widens and
     narrows with realized volatility.
  2. Heavy is not fit on its own. We forecast Light with the Levels model and
     the spread with the Differential model, then set Heavy = Light + spread
     per path. That keeps Heavy consistent with both upstream models instead
     of letting a separate Heavy regression disagree with the spread.
  3. Bull/bear scenarios apply +/- 2 sigma shocks sized from the historical
     first-difference std of each driver (log_production, log_dxy,
     refinery_util), scaled by SCENARIO_SIGMA_MULT, not flat percentage moves.

Outputs:
  output/forecast.csv                base scenario point + bands
  output/forecast_scenarios.csv      all three scenarios x Light/Heavy
  output/forecast_differential.csv   differential model output
  output/charts/07_forecast.png      Light + Heavy scenarios with fan
  output/charts/09_diff_forecast.png WCS-WTI spread forecast
"""
from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
from run_models import _resolve_levels, _drop_constant_cols

warnings.filterwarnings("ignore", category=Warning)

FORECAST_END = pd.Timestamp("2026-08-01")
N_SIM = 5000           # Monte Carlo paths
SCENARIO_SIGMA_MULT = 2.0  # bull/bear shock = MULT * sigma(historical first diff)

# Map scenario name -> sign of shock applied to each driver.
# (+ for bear-side direction of that driver, then bull = negated)
SCENARIO_SIGNS = {
    "bull": {"log_production": -1.0, "log_dxy": -1.0, "refinery_util": +1.0},
    "bear": {"log_production": +1.0, "log_dxy": +1.0, "refinery_util": -1.0},
}


def _build_scenarios(df: pd.DataFrame) -> dict:
    """Construct +/-2 sigma scenario shocks from each driver's first-diff std."""
    sigmas = {}
    for col in ["log_production", "log_dxy", "refinery_util"]:
        if col not in df.columns:
            sigmas[col] = 0.0; continue
        d = df[col].dropna().diff().dropna()
        sigmas[col] = float(d.std()) if len(d) > 12 else 0.0

    scenarios = {"base": {}}
    for name, signs in SCENARIO_SIGNS.items():
        scenarios[name] = {
            k: signs[k] * SCENARIO_SIGMA_MULT * sigmas.get(k, 0.0)
            for k in signs
        }
    return scenarios, sigmas


def _load_vol_forecast() -> dict:
    """Load GARCH volatility forecast as {crude: array of sigma-per-month (decimal)}.
    Returns empty dict if vol_forecast.csv missing (caller falls back to OLS-residual sigma)."""
    p = config.OUTPUT_DIR / "vol_forecast.csv"
    if not p.exists():
        print("  [WARN] vol_forecast.csv not found - falling back to OLS-residual sigma")
        return {}
    vf = pd.read_csv(p, parse_dates=["date"])
    out = {}
    for crude in ["Light", "Heavy"]:
        sub = vf[vf["crude"] == crude].sort_values("date")
        # vol_model saves sigma in PERCENT; convert to decimal log-return std
        out[crude] = sub["sigma_forecast"].values / 100.0
    return out


def _fit_ar1(s: pd.Series) -> tuple[float, float, float]:
    s = s.dropna()
    y = s.iloc[1:].values; x = s.iloc[:-1].values
    if len(y) < 6:
        return 0.0, 1.0, float(s.std() if len(s) > 2 else 0.0)
    res = sm.OLS(y, sm.add_constant(x)).fit()
    sd = float(np.sqrt(res.mse_resid))
    return float(res.params[0]), float(res.params[1]), sd


def _project_exogenous(df: pd.DataFrame, regressors: list[str],
                       horizon: pd.DatetimeIndex, scenario_shocks: dict) -> dict:
    """AR(1)-project each non-dummy regressor; deterministic for dummies/seasonality.
    Returns {regressor: {'mean': array, 'sd': float}}.
    """
    out = {}
    for r in regressors:
        if r in ("month_sin", "month_cos"):
            m = horizon.month
            out[r] = {
                "mean": (np.sin if r == "month_sin" else np.cos)(2 * np.pi * m / 12),
                "sd":   0.0,
            }
            continue
        if r.startswith("regime_") or r in ("opec_shock", "spr_drawdown"):
            last = float(df[r].dropna().iloc[-1]) if df[r].notna().any() else 0.0
            out[r] = {"mean": np.full(len(horizon), last), "sd": 0.0}
            continue
        s = df[r].dropna()
        if len(s) == 0:
            out[r] = {"mean": np.zeros(len(horizon)), "sd": 0.0}
            continue
        intercept, slope, sd = _fit_ar1(s)
        last_val = float(s.iloc[-1])
        shock = scenario_shocks.get(r, 0.0)  # one-time level shift to starting state
        path = np.empty(len(horizon))
        prev = last_val + shock              # shock applied ONCE; AR(1) then evolves
        for t in range(len(horizon)):
            prev = intercept + slope * prev
            path[t] = prev
        out[r] = {"mean": path, "sd": sd}
    return out


def _forecast_paths_levels(df: pd.DataFrame, dep: str, lag_var: str,
                            scenario_shocks: dict,
                            vol_path: np.ndarray | None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return raw simulated log-price paths (N_SIM, horizon)."""
    regressors = _resolve_levels(lag_var)
    # Filter to columns that exist (Canadian production is optional)
    regressors = [r for r in regressors if r in df.columns]
    sub = df[df["war_long"] == 1][[dep] + regressors].dropna()
    regressors = _drop_constant_cols(sub, regressors)
    X = sm.add_constant(sub[regressors])
    res = sm.OLS(sub[dep], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    sigma_price_ols = float(np.sqrt(res.mse_resid))

    last_log_price = float(df[dep].dropna().iloc[-1])
    last_date = df[dep].dropna().index.max()
    horizon = pd.date_range(last_date + pd.DateOffset(months=1), FORECAST_END, freq="MS")
    if len(horizon) == 0:
        return np.empty((N_SIM, 0)), horizon

    # If no GARCH forecast provided, fall back to constant OLS residual sigma
    if vol_path is None or len(vol_path) < len(horizon):
        sigma_t = np.full(len(horizon), sigma_price_ols)
    else:
        sigma_t = vol_path[:len(horizon)]

    exo = _project_exogenous(df, [r for r in regressors if r != lag_var],
                             horizon, scenario_shocks)

    paths = np.empty((N_SIM, len(horizon)))
    rng = np.random.default_rng(42)
    coef = res.params

    for s_i in range(N_SIM):
        prev_log = last_log_price
        for t in range(len(horizon)):
            x = {lag_var: prev_log}
            for r in regressors:
                if r == lag_var: continue
                e = exo[r]
                x[r] = e["mean"][t] + (rng.normal(0, e["sd"]) if e["sd"] > 0 else 0.0)
            mean_log = coef.get("const", 0.0) + sum(coef[r] * x[r] for r in regressors)
            mean_log += rng.normal(0, sigma_t[t])  # TIME-VARYING vol shock
            paths[s_i, t] = mean_log
            prev_log = mean_log
    return paths, horizon


def _forecast_paths_differential(df: pd.DataFrame,
                                  scenario_shocks: dict) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return raw simulated differential paths (N_SIM, horizon) in $/bbl."""
    regs = list(config.REGRESSORS_DIFFERENTIAL)
    if "apportionment_lag1" in df.columns and df["apportionment_lag1"].notna().any():
        regs.append("apportionment_lag1")
    regs = [r for r in regs if r in df.columns]
    sub = df[["wcs_wti_diff"] + regs].dropna()
    regs = _drop_constant_cols(sub, regs)
    X = sm.add_constant(sub[regs])
    res = sm.OLS(sub["wcs_wti_diff"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    sigma = float(np.sqrt(res.mse_resid))
    last_diff = float(df["wcs_wti_diff"].dropna().iloc[-1])
    last_date = df["wcs_wti_diff"].dropna().index.max()
    horizon = pd.date_range(last_date + pd.DateOffset(months=1), FORECAST_END, freq="MS")
    exo = _project_exogenous(df, [r for r in regs if r != "wcs_wti_diff_lag1"],
                             horizon, scenario_shocks)
    paths = np.empty((N_SIM, len(horizon)))
    rng = np.random.default_rng(43)  # different seed than Light to keep them independent shocks
    coef = res.params
    for s_i in range(N_SIM):
        prev = last_diff
        for t in range(len(horizon)):
            x = {"wcs_wti_diff_lag1": prev}
            for r in regs:
                if r == "wcs_wti_diff_lag1": continue
                e = exo[r]
                x[r] = e["mean"][t] + (rng.normal(0, e["sd"]) if e["sd"] > 0 else 0.0)
            val = coef.get("const", 0.0) + sum(coef[r] * x[r] for r in regs)
            val += rng.normal(0, sigma)
            paths[s_i, t] = val
            prev = val
    return paths, horizon


def _agg_levels(paths_log: np.ndarray, horizon: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate raw log-price paths to percentile DataFrame in level ($/bbl)."""
    levels = np.exp(paths_log)
    return pd.DataFrame({
        "date":         horizon,
        "price_pred":   np.exp(paths_log.mean(axis=0)),
        "price_p10":    np.percentile(levels, 10, axis=0),
        "price_p25":    np.percentile(levels, 25, axis=0),
        "price_p50":    np.percentile(levels, 50, axis=0),
        "price_p75":    np.percentile(levels, 75, axis=0),
        "price_p90":    np.percentile(levels, 90, axis=0),
        "price_lo_95":  np.percentile(levels, 2.5, axis=0),
        "price_hi_95":  np.percentile(levels, 97.5, axis=0),
    }).set_index("date")


def _agg_levels_from_array(level_arr: np.ndarray, horizon: pd.DatetimeIndex) -> pd.DataFrame:
    """Same aggregation but starting from level paths (no exp)."""
    return pd.DataFrame({
        "date":         horizon,
        "price_pred":   level_arr.mean(axis=0),
        "price_p10":    np.percentile(level_arr, 10, axis=0),
        "price_p25":    np.percentile(level_arr, 25, axis=0),
        "price_p50":    np.percentile(level_arr, 50, axis=0),
        "price_p75":    np.percentile(level_arr, 75, axis=0),
        "price_p90":    np.percentile(level_arr, 90, axis=0),
        "price_lo_95":  np.percentile(level_arr, 2.5, axis=0),
        "price_hi_95":  np.percentile(level_arr, 97.5, axis=0),
    }).set_index("date")


def _agg_diff(paths: np.ndarray, horizon: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "date":      horizon,
        "diff_pred": paths.mean(axis=0),
        "diff_p10":  np.percentile(paths, 10, axis=0),
        "diff_p90":  np.percentile(paths, 90, axis=0),
    }).set_index("date")


def _chart_scenarios(df: pd.DataFrame, scenarios: dict, last_dates: dict, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    cutoff = FORECAST_END - pd.DateOffset(years=6)
    hist = df.loc[df.index >= cutoff]
    for ax_i, (crude, color, lvl) in enumerate([("Light", "#16697A", "wti_real"),
                                                 ("Heavy", "#E8893E", "wcs_real")]):
        ax = axes[ax_i]
        actual = hist[lvl].dropna()
        ax.plot(actual.index, actual.values, color=color, lw=1.8, label=f"{crude} - actual")
        last_date = last_dates[crude]
        last_val = float(df[lvl].dropna().iloc[-1])
        for sc, color_sc, ls in [("base", color, "--"),
                                  ("bull", "#2A8B5E", ":"),
                                  ("bear", "#B83A4B", ":")]:
            f = scenarios[crude][sc]
            xs = [last_date] + list(f.index)
            ys = [last_val] + list(f["price_pred"].values)
            ax.plot(xs, ys, color=color_sc, lw=1.6, ls=ls, label=f"{crude} - {sc}")
        f_base = scenarios[crude]["base"]
        ax.fill_between(f_base.index, f_base["price_p10"], f_base["price_p90"],
                        color=color, alpha=0.15, lw=0, label=f"{crude} - 80% MC band")
        ax.fill_between(f_base.index, f_base["price_p25"], f_base["price_p75"],
                        color=color, alpha=0.20, lw=0)
        ax.axvline(last_date, color="#555560", lw=0.9, ls=":", alpha=0.7)
        ax.set_title(f"{crude} oil - base/bull/bear through {FORECAST_END.strftime('%b %Y')}")
        ax.set_ylabel("Price ($/bbl, real 2025 USD)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Oil price forecast - GARCH-vol fan, Heavy = Light + Differential, +/-2sigma scenarios",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _chart_differential(df: pd.DataFrame, fcst: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cutoff = FORECAST_END - pd.DateOffset(years=6)
    hist = df.loc[df.index >= cutoff, "wcs_wti_diff"].dropna()
    ax.plot(hist.index, hist.values, color="#16697A", lw=1.6, label="WCS - WTI (actual)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    last_date = hist.index.max()
    last_val = float(hist.iloc[-1])
    xs = [last_date] + list(fcst.index)
    ys = [last_val] + list(fcst["diff_pred"].values)
    ax.plot(xs, ys, color="#E8893E", lw=2.2, ls="--", label="Differential model forecast")
    ax.fill_between(fcst.index, fcst["diff_p10"], fcst["diff_p90"],
                    color="#E8893E", alpha=0.15, lw=0, label="80% MC band")
    ax.set_title(f"WCS - WTI differential - forecast through {FORECAST_END.strftime('%b %Y')}\n"
                 "Drives Heavy = Light + Diff jointly; eliminates the prior cross-model inconsistency")
    ax.set_ylabel("WCS - WTI ($/bbl, real)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    print(f"Forecast end: {FORECAST_END.date()}  |  Monte Carlo paths: {N_SIM}")

    # Build scenarios from historical first-diff sigma * SCENARIO_SIGMA_MULT
    scenarios_cfg, fd_sigmas = _build_scenarios(df)
    print(f"\nScenario shocks (+/-{SCENARIO_SIGMA_MULT:.0f} sigma of historical monthly first-diff):")
    for k, v in fd_sigmas.items():
        print(f"  {k:18s}  sigma={v:.4f}  -> shock=+/-{SCENARIO_SIGMA_MULT*v:.4f}")

    vol_forecasts = _load_vol_forecast()
    if vol_forecasts:
        vL = vol_forecasts.get("Light")
        print(f"\nGARCH vol forecast loaded: Light first/last sigma = "
              f"{vL[0]*100:.2f}% / {vL[-1]*100:.2f}% (monthly)")

    # -------- LIGHT (Levels model) --------
    last_light = float(df["wti_real"].dropna().iloc[-1])
    last_light_date = df["wti_real"].dropna().index.max()
    light_paths = {}
    light_horizon = None
    for sc in ["base", "bull", "bear"]:
        paths_log, light_horizon = _forecast_paths_levels(
            df, "log_wti_real", "log_wti_lag1",
            scenarios_cfg[sc], vol_forecasts.get("Light"),
        )
        light_paths[sc] = paths_log  # log-price paths

    # -------- DIFFERENTIAL --------
    diff_paths = {}
    diff_horizon = None
    for sc in ["base", "bull", "bear"]:
        p, diff_horizon = _forecast_paths_differential(df, scenarios_cfg[sc])
        diff_paths[sc] = p

    # Align horizons: Light/Diff may end on different last-actual dates.
    # Take the intersection (by date), trim both to that common horizon.
    common = light_horizon.intersection(diff_horizon)
    horizon = common
    li = [list(light_horizon).index(d) for d in common]
    di = [list(diff_horizon).index(d) for d in common]
    light_paths = {sc: light_paths[sc][:, li] for sc in light_paths}
    diff_paths  = {sc: diff_paths[sc][:, di]  for sc in diff_paths}

    # -------- HEAVY = LIGHT + DIFF (joint paths, level scale) --------
    # Enforce the PHYSICAL CONSTRAINT: WCS must always trade below WTI by at
    # least MIN_QUALITY_DISCOUNT ($/bbl, real). Heavy crude has lower API gravity
    # (~21 vs ~40) and higher sulfur (~3.5% vs ~0.4%) -- it CANNOT trade at parity
    # or premium to WTI in a normal market. We clip the differential to be at
    # most -MIN_QUALITY_DISCOUNT, then derive Heavy = Light + clipped_Diff.
    min_qd = config.MIN_QUALITY_DISCOUNT
    heavy_paths_lvl = {}
    clipped_paths_count = 0
    total_paths_count = 0
    for sc in ["base", "bull", "bear"]:
        light_lvl = np.exp(light_paths[sc])
        # Clip differential: it can never exceed -MIN_QUALITY_DISCOUNT
        diff_clipped = np.minimum(diff_paths[sc], -min_qd)
        clipped_paths_count += int((diff_paths[sc] > -min_qd).sum())
        total_paths_count += diff_paths[sc].size
        diff_paths[sc] = diff_clipped  # persist clip so saved CSV reflects it
        heavy_paths_lvl[sc] = light_lvl + diff_clipped
    pct_clipped = 100 * clipped_paths_count / max(total_paths_count, 1)
    print(f"\nQuality-discount floor enforced (Heavy <= Light - ${min_qd:.0f}/bbl): "
          f"{pct_clipped:.1f}% of Monte Carlo cells clipped to the floor")

    # Aggregate
    scenarios = {"Light": {}, "Heavy": {}}
    for sc in ["base", "bull", "bear"]:
        scenarios["Light"][sc] = _agg_levels(light_paths[sc], horizon)
        scenarios["Heavy"][sc] = _agg_levels_from_array(heavy_paths_lvl[sc], horizon)

    last_dates = {"Light": last_light_date, "Heavy": df["wcs_real"].dropna().index.max()}
    last_vals  = {"Light": last_light, "Heavy": float(df["wcs_real"].dropna().iloc[-1])}

    print(f"\nResults at {FORECAST_END.strftime('%b %Y')}:")
    for crude in ["Light", "Heavy"]:
        end = scenarios[crude]["base"].iloc[-1]
        print(f"  {crude:5s}  last {last_dates[crude].date()} ${last_vals[crude]:6.2f}  ->  "
              f"base ${end['price_pred']:.2f}  (p10-p90 ${end['price_p10']:.2f} - ${end['price_p90']:.2f})")
        for sc in ("bull", "bear"):
            v = scenarios[crude][sc].iloc[-1]["price_pred"]
            print(f"           {sc:4s}: ${v:.2f}")

    # -------- SAVES --------
    pieces = []
    for crude in ["Light", "Heavy"]:
        f = scenarios[crude]["base"].copy()
        f.columns = [f"{crude.lower()}_{c}" for c in f.columns]
        pieces.append(f)
    pd.concat(pieces, axis=1).to_csv(config.OUTPUT_DIR / "forecast.csv")
    print(f"\nSaved: {config.OUTPUT_DIR / 'forecast.csv'}  (base scenario)")

    rows = []
    for crude in ["Light", "Heavy"]:
        for sc in ["base", "bull", "bear"]:
            f = scenarios[crude][sc].copy()
            f["crude"] = crude; f["scenario"] = sc
            rows.append(f.reset_index())
    pd.concat(rows, ignore_index=True).to_csv(
        config.OUTPUT_DIR / "forecast_scenarios.csv", index=False)
    print(f"Saved: {config.OUTPUT_DIR / 'forecast_scenarios.csv'}")

    out = config.CHARTS_DIR / "07_forecast.png"
    _chart_scenarios(df, scenarios, last_dates, out)
    print(f"Saved: {out}")

    # Differential (base) - save and chart
    diff_base_df = _agg_diff(diff_paths["base"], horizon)
    diff_base_df.to_csv(config.OUTPUT_DIR / "forecast_differential.csv")
    end_d = diff_base_df.iloc[-1]
    last_d = float(df["wcs_wti_diff"].dropna().iloc[-1])
    print(f"\nDifferential: last ${last_d:+.2f}  ->  {FORECAST_END.strftime('%b %Y')}: "
          f"${end_d['diff_pred']:+.2f}  (p10-p90 ${end_d['diff_p10']:+.2f} to ${end_d['diff_p90']:+.2f})")
    print(f"Saved: {config.OUTPUT_DIR / 'forecast_differential.csv'}")
    out2 = config.CHARTS_DIR / "09_diff_forecast.png"
    _chart_differential(df, diff_base_df, out2)
    print(f"Saved: {out2}")

    return scenarios, diff_base_df


if __name__ == "__main__":
    main()
