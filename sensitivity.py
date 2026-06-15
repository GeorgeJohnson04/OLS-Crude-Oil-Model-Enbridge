"""
Calibrated perturbation / sensitivity analysis.

Answers: "If driver X has a typical one-month adverse move, how much does the
August forecast shift?"

Calibration choices (every shock magnitude is empirical, not arbitrary):

  1. Shock size = 1sigma of the historical FIRST-DIFFERENCE of each regressor.
     A 1sigma shock represents a "typical bad month" for that variable, not an
     arbitrary "10% move".

  2. Output reported in $/bbl on the LEVEL forecast at Aug 2026, not log
     units or coefficient values, so it is directly interpretable.

  3. Bootstrap confidence intervals on the sensitivity itself, computed by
     residual-bootstrapping the OLS coefficient vector 200 times. So we
     report 'driver X moves the forecast by $A/bbl, 95% CI $B-$C/bbl'.

  4. +/-1sigma shocks are applied symmetrically. The reported sensitivity is the
     average of |+ impact| and |- impact| so non-linearities (asymmetric
     coefficient effects) are caught.

Outputs:
  output/sensitivity_results.csv  ranked drivers with $/bbl impacts and CIs
  output/charts/13_tornado.png    tornado chart, sorted by absolute impact

Run:
  python sensitivity.py
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import config
from run_models import _resolve_levels, _drop_constant_cols

warnings.filterwarnings("ignore", category=Warning)

# Tunables
FORECAST_END = pd.Timestamp("2026-08-01")
N_SIM = 500          # Paths per sensitivity run (smaller than forecast.py to keep total time reasonable)
N_BOOT = 200         # Bootstrap iterations for CI on the sensitivity
RNG_SEED = 42

# Color palette
INK    = "#1A1A2E"
AMBER  = "#E8893E"
TEAL   = "#16697A"
RED    = "#A8324E"
GRAY   = "#555560"


def _fit_ar1(s: pd.Series) -> tuple[float, float, float]:
    """Fit AR(1) on a stationary or unit-root series. Returns (intercept, slope, residual_sd)."""
    s = s.dropna()
    if len(s) < 6:
        return 0.0, 1.0, float(s.std() if len(s) > 2 else 0.0)
    y = s.iloc[1:].values
    x = s.iloc[:-1].values
    res = sm.OLS(y, sm.add_constant(x)).fit()
    return float(res.params[0]), float(res.params[1]), float(np.sqrt(res.mse_resid))


def _historical_sigmas(df: pd.DataFrame, regressors: list[str]) -> dict[str, float]:
    """1sigma historical first-difference for each regressor. These are the CALIBRATED shock sizes."""
    sigmas = {}
    for r in regressors:
        if r not in df.columns:
            continue
        s = df[r].dropna()
        if len(s) < 12:
            sigmas[r] = 0.0
            continue
        sigmas[r] = float(s.diff().dropna().std())
    return sigmas


def _project_exogenous(df: pd.DataFrame, regressors: list[str],
                        horizon: pd.DatetimeIndex, shock_var: str | None = None,
                        shock_size: float = 0.0) -> dict:
    """AR(1) projection of each exogenous regressor over the horizon.
    Optionally apply a ONE-TIME level shock to `shock_var` at t=0."""
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
        if r == shock_var:
            last_val = last_val + shock_size  # one-time impulse at t=0
        path = np.empty(len(horizon))
        prev = last_val
        for t in range(len(horizon)):
            prev = intercept + slope * prev
            path[t] = prev
        out[r] = {"mean": path, "sd": sd}
    return out


def _run_forecast(df: pd.DataFrame, dep: str, lag_var: str,
                  coefs: pd.Series, regressors: list[str], sigma_price: float,
                  shock_var: str = None, shock_size: float = 0.0,
                  n_sim: int = N_SIM, rng: np.random.Generator = None) -> float:
    """Run Monte Carlo from the fitted OLS coefficients and return mean forecast
    at FORECAST_END (in $/bbl, not log)."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    last_log_price = float(df[dep].dropna().iloc[-1])
    last_date = df[dep].dropna().index.max()
    horizon = pd.date_range(last_date + pd.DateOffset(months=1), FORECAST_END, freq="MS")
    if len(horizon) == 0:
        return float(np.exp(last_log_price))

    exo = _project_exogenous(
        df, [r for r in regressors if r != lag_var],
        horizon, shock_var=shock_var, shock_size=shock_size,
    )

    paths = np.empty((n_sim, len(horizon)))
    for s_i in range(n_sim):
        prev_log = last_log_price
        for t in range(len(horizon)):
            x = {lag_var: prev_log}
            for r in regressors:
                if r == lag_var: continue
                e = exo[r]
                shock = rng.normal(0, e["sd"]) if e["sd"] > 0 else 0.0
                x[r] = e["mean"][t] + shock
            mean_log = coefs.get("const", 0.0) + sum(coefs[r] * x[r] for r in regressors)
            mean_log += rng.normal(0, sigma_price)
            paths[s_i, t] = mean_log
            prev_log = mean_log

    # Mean LEVEL at the final month
    return float(np.exp(paths.mean(axis=0)[-1]))


def _bootstrap_coefs(X: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                      rng: np.random.Generator = None) -> list[pd.Series]:
    """Residual-bootstrap the OLS coefficients. Returns a list of n_boot coefficient Series."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED + 1)
    base = sm.OLS(y, X).fit()
    resid = base.resid
    fitted = base.fittedvalues
    boots = []
    for _ in range(n_boot):
        # Resample residuals with replacement, build synthetic y, refit
        resid_boot = resid[rng.integers(0, len(resid), size=len(resid))]
        y_boot = fitted + resid_boot
        b = sm.OLS(y_boot, X).fit()
        boots.append(b.params)
    return boots


def _run_sensitivity(df: pd.DataFrame, dep: str, lag_var: str,
                     boot_coefs: list, regressors: list[str], sigma_price: float,
                     shock_var: str, shock_size: float) -> dict:
    """For one driver: run +1sigma and -1sigma forecast for each bootstrap coefficient
    vector; return the mean and 95% CI on the symmetric sensitivity."""
    rng = np.random.default_rng(RNG_SEED + 2)
    pos = np.empty(len(boot_coefs))
    neg = np.empty(len(boot_coefs))
    for i, coefs in enumerate(boot_coefs):
        pos[i] = _run_forecast(df, dep, lag_var, coefs, regressors, sigma_price,
                                shock_var=shock_var, shock_size=+shock_size,
                                n_sim=100, rng=rng)
        neg[i] = _run_forecast(df, dep, lag_var, coefs, regressors, sigma_price,
                                shock_var=shock_var, shock_size=-shock_size,
                                n_sim=100, rng=rng)
    sym = (pos - neg) / 2.0  # signed symmetric sensitivity in $/bbl
    return {
        "mean_pos":   float(pos.mean()),
        "mean_neg":   float(neg.mean()),
        "sensitivity": float(np.median(sym)),
        "ci_low":     float(np.percentile(sym, 2.5)),
        "ci_high":    float(np.percentile(sym, 97.5)),
    }


def _tornado_chart(results: pd.DataFrame, baseline: float, out_path):
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(results) + 1.5))
    # Sort by |sensitivity| descending; plot largest at top
    results = results.sort_values("abs_sens", ascending=True)
    y = np.arange(len(results))
    sens = results["sensitivity"].values
    ci_low = results["ci_low"].values
    ci_high = results["ci_high"].values
    err_low = sens - ci_low
    err_high = ci_high - sens
    colors = ["#16697A" if s >= 0 else "#A8324E" for s in sens]
    ax.barh(y, sens, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(sens, y, xerr=[err_low, err_high], fmt="none",
                ecolor="#555560", capsize=3, lw=0.8)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(results["driver"].values, fontsize=10)
    ax.set_xlabel(f"Impact on Aug 2026 forecast ($/bbl)  —  baseline = ${baseline:.2f}", fontsize=11)
    ax.set_title("Tornado chart — calibrated sensitivity (±1sigma historical shock per driver)\n"
                 "Bars = signed sensitivity; whiskers = 95% bootstrap CI",
                 fontsize=12)
    # Number labels at bar ends
    for i, (s, lo, hi) in enumerate(zip(sens, ci_low, ci_high)):
        x_text = s + (0.05 if s >= 0 else -0.05)
        ha = "left" if s >= 0 else "right"
        ax.text(x_text, i, f"${s:+.2f}", va="center", ha=ha, fontsize=9, color=INK)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    dep, lag_var = "log_wti_real", "log_wti_lag1"
    print(f"Calibrated sensitivity analysis -- target: {dep}\n")

    # Fit the base OLS
    regressors = _resolve_levels(lag_var)
    regressors = [r for r in regressors if r in df.columns]
    sub = df[df["war_long"] == 1][[dep] + regressors].dropna()
    regressors = _drop_constant_cols(sub, regressors)
    X = sm.add_constant(sub[regressors])
    y = sub[dep].values
    base = sm.OLS(y, X).fit()
    sigma_price = float(np.sqrt(base.mse_resid))
    print(f"Base OLS fit: n={int(base.nobs)}, R2={base.rsquared:.3f}, sigma={sigma_price:.4f}\n")

    # Baseline forecast (no shock)
    rng_base = np.random.default_rng(RNG_SEED)
    baseline = _run_forecast(df, dep, lag_var, base.params, regressors,
                              sigma_price, n_sim=N_SIM, rng=rng_base)
    print(f"Baseline Aug-2026 forecast: ${baseline:.2f}/bbl\n")

    # Calibration step: compute 1sigma historical first-differences
    sigmas = _historical_sigmas(df, regressors)
    print("Calibrated shock sizes (1sigma of historical monthly first-differences):")
    for r in regressors:
        if r == lag_var: continue  # skip the lag - it's not a free input
        s = sigmas.get(r, 0.0)
        if s > 0:
            print(f"  {r:24s}  1sigma = {s:.4f}")
    print()

    # Bootstrap OLS coefficients for CI on sensitivity
    print(f"Residual-bootstrapping coefficients ({N_BOOT} iterations)...")
    boot_coefs = _bootstrap_coefs(X.values, y, n_boot=N_BOOT)
    # Convert numpy arrays back to Series with proper labels
    col_names = list(X.columns)
    boot_coefs = [pd.Series(b, index=col_names) for b in boot_coefs]

    # Sensitivities per driver
    print("\nDriver sensitivities (median + 95% CI on Aug-2026 forecast impact):")
    print(f"  {'driver':<24}  {'shock(1sigma)':>11}  {'+1sigma impact':>11}  {'-1sigma impact':>11}  {'|sens|':>8}  {'95% CI':>20}")
    results = []
    for r in regressors:
        if r == lag_var or sigmas.get(r, 0.0) == 0.0:
            continue
        shk = sigmas[r]
        out = _run_sensitivity(df, dep, lag_var, boot_coefs, regressors,
                                sigma_price, shock_var=r, shock_size=shk)
        sens = out["sensitivity"]
        ci_low, ci_high = out["ci_low"], out["ci_high"]
        results.append({
            "driver": r,
            "shock_1sigma": shk,
            "plus_impact_$/bbl": out["mean_pos"] - baseline,
            "minus_impact_$/bbl": out["mean_neg"] - baseline,
            "sensitivity": sens,
            "abs_sens": abs(sens),
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
        print(f"  {r:<24}  {shk:>11.4f}  {out['mean_pos']-baseline:>+10.2f}  "
              f"{out['mean_neg']-baseline:>+10.2f}  {abs(sens):>7.2f}  "
              f"[{ci_low:+.2f}, {ci_high:+.2f}]")

    results = pd.DataFrame(results).sort_values("abs_sens", ascending=False)
    out_csv = config.OUTPUT_DIR / "sensitivity_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    out_chart = config.CHARTS_DIR / "13_tornado.png"
    _tornado_chart(results, baseline, out_chart)
    print(f"Saved: {out_chart}")


if __name__ == "__main__":
    main()
