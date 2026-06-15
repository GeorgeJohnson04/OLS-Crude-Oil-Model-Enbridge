"""
Volatility model for oil log-returns.

Fits GARCH(1,1) and asymmetric GJR-GARCH(1,1) to monthly log-returns of WTI
and WCS (real-price, CPI-deflated). Picks the better-fitting model by BIC.

Outputs:
  output/vol_params.csv               estimated parameters per crude
  output/vol_history.csv              fitted conditional volatility (history)
  output/vol_forecast.csv             forward conditional volatility path
  output/charts/10_vol_history.png    realized vs GARCH vol over time
  output/charts/11_vol_forecast.png   forecasted vol path with confidence band

The forecast.py module imports vol_forecast.csv to replace the constant
residual-std assumption with a time-varying volatility path.

Why GARCH?
  Plain OLS assumes homoscedastic residuals (constant sigma). Oil violates
  this aggressively - quiet months at sigma ~3-4% sit next to crisis months
  at sigma ~15-20%. Using constant sigma in Monte Carlo produces fan charts
  that are simultaneously too wide in calm regimes and too narrow in
  volatile ones. GARCH captures the volatility clustering that's observably
  present in oil returns since 2003.

Why also GJR-GARCH?
  Oil exhibits the "leverage effect" - negative shocks raise volatility more
  than positive shocks of equal size (a $5 down-move spooks the market more
  than a $5 up-move calms it). GJR-GARCH adds an asymmetric term and
  almost always beats vanilla GARCH on oil data. We fit both and pick by BIC.
"""
from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

import config

warnings.filterwarnings("ignore", category=Warning)

FORECAST_END = pd.Timestamp("2026-08-01")

# Returns are scaled up by SCALE so the optimizer works in a sensible
# numerical range (raw monthly log-returns are ~0.05, GARCH likes ~1-5).
SCALE = 100.0

CRUDE_SPECS = [
    # (label, return_col, price_col, color)
    ("Light", "dlog_wti_real", "wti_real",  "#16697A"),
    ("Heavy", "dlog_wcs_real", "wcs_real",  "#E8893E"),
]


def _fit_garch_family(returns: pd.Series) -> tuple[object, str, pd.DataFrame]:
    """Fit GARCH(1,1) and GJR-GARCH(1,1); return the better one by BIC.

    Returns: (fitted_result, model_name, comparison_df)
    """
    r = returns.dropna() * SCALE  # scale returns to ~percent

    # GARCH(1,1)
    g11 = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(
        disp="off", show_warning=False
    )
    # GJR-GARCH(1,1) - asymmetric (leverage) term
    gjr = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(
        disp="off", show_warning=False
    )

    comparison = pd.DataFrame({
        "model":     ["GARCH(1,1)", "GJR-GARCH(1,1)"],
        "LL":        [g11.loglikelihood, gjr.loglikelihood],
        "AIC":       [g11.aic,           gjr.aic],
        "BIC":       [g11.bic,           gjr.bic],
        "persistence": [
            float(g11.params.get("alpha[1]", 0.0) + g11.params.get("beta[1]", 0.0)),
            float(gjr.params.get("alpha[1]", 0.0)
                  + gjr.params.get("beta[1]", 0.0)
                  + 0.5 * gjr.params.get("gamma[1]", 0.0)),
        ],
    }).round(4)

    # Pick by BIC (lower is better)
    if gjr.bic < g11.bic:
        return gjr, "GJR-GARCH(1,1)", comparison
    return g11, "GARCH(1,1)", comparison


def _forecast_vol(res, horizon_months: int) -> np.ndarray:
    """Return the forecast conditional sigma path (in scaled units) of length
    `horizon_months`."""
    fcst = res.forecast(horizon=horizon_months, reindex=False)
    # variance is a DataFrame of shape (1, horizon); each entry = sigma^2
    var_row = fcst.variance.iloc[-1].values
    sigma_path = np.sqrt(var_row)
    return sigma_path


def _chart_history(panel: pd.DataFrame, out_path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, (crude, color) in zip(axes, [("Light", "#16697A"), ("Heavy", "#E8893E")]):
        sig_col = f"sigma_{crude.lower()}"
        ret_col = f"return_{crude.lower()}"
        if sig_col not in panel.columns:
            continue
        # Realized absolute return as eyeball reference
        ax.plot(panel.index, np.abs(panel[ret_col]) * 100, color="#999999", lw=0.7,
                alpha=0.7, label="|monthly return| (%)")
        ax.plot(panel.index, panel[sig_col], color=color, lw=1.8,
                label=f"GARCH conditional σ ({crude})")
        ax.set_title(f"{crude} crude — conditional volatility (monthly %)")
        ax.set_ylabel("Sigma (% per month)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Estimated conditional volatility — GARCH family", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _chart_forecast(panel: pd.DataFrame, fcst_df: pd.DataFrame, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cutoff = FORECAST_END - pd.DateOffset(years=4)
    hist = panel.loc[panel.index >= cutoff]
    for ax, (crude, color) in zip(axes, [("Light", "#16697A"), ("Heavy", "#E8893E")]):
        sig_col = f"sigma_{crude.lower()}"
        last_date = hist.index.max()
        last_sigma = float(hist[sig_col].dropna().iloc[-1])
        # History
        ax.plot(hist.index, hist[sig_col], color=color, lw=1.6,
                label=f"{crude} — GARCH σ (history)")
        # Forecast
        f = fcst_df[fcst_df["crude"] == crude]
        xs = [last_date] + list(f["date"].values)
        ys = [last_sigma] + list(f["sigma_forecast"].values)
        ax.plot(xs, ys, color=color, lw=2.2, ls="--",
                label=f"{crude} — σ forecast")
        ax.axvline(last_date, color="#555560", lw=0.9, ls=":", alpha=0.7)
        ax.set_title(f"{crude} crude — forecast conditional σ through {FORECAST_END.strftime('%b %Y')}")
        ax.set_ylabel("Sigma (% per month)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("GARCH volatility forecast — time-varying σ for Monte Carlo", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    last_date = df.index.max()
    horizon = pd.date_range(last_date + pd.DateOffset(months=1), FORECAST_END, freq="MS")
    n_h = len(horizon)
    print(f"Vol model - fitting GARCH family. Last obs: {last_date.date()}, "
          f"horizon: {n_h} months to {FORECAST_END.date()}")

    panel = pd.DataFrame(index=df.index)
    params_rows = []
    comparison_rows = []
    fcst_rows = []

    for crude, ret_col, _price_col, _color in CRUDE_SPECS:
        if ret_col not in df.columns:
            print(f"  [SKIP] {crude}: {ret_col} not found")
            continue
        r = df[ret_col].dropna()
        if len(r) < 50:
            print(f"  [SKIP] {crude}: too few obs ({len(r)})")
            continue
        res, chosen, comparison = _fit_garch_family(r)
        comparison["crude"] = crude
        comparison_rows.append(comparison)

        # Backfit conditional sigma over history (in scaled units -> back to %)
        sigma_history = res.conditional_volatility  # in scaled (percent) units
        panel.loc[r.index, f"return_{crude.lower()}"] = r
        panel.loc[sigma_history.index, f"sigma_{crude.lower()}"] = sigma_history.values

        # Forecast forward
        sigma_path = _forecast_vol(res, horizon_months=n_h)
        for d, s in zip(horizon, sigma_path):
            fcst_rows.append({"date": d, "crude": crude, "sigma_forecast": float(s)})

        # Persistence and unconditional vol for reporting
        a = float(res.params.get("alpha[1]", 0.0))
        b = float(res.params.get("beta[1]", 0.0))
        g = float(res.params.get("gamma[1]", 0.0))
        omega = float(res.params.get("omega", 0.0))
        persistence = a + b + 0.5 * g
        uncond_var = omega / max(1e-9, 1 - persistence) if persistence < 1 else float("nan")
        uncond_sigma = float(np.sqrt(uncond_var)) if uncond_var > 0 else float("nan")

        print(f"  {crude:5s}  chosen: {chosen}")
        print(f"         omega={omega:.4f}  alpha={a:.4f}  beta={b:.4f}  gamma={g:.4f}")
        print(f"         persistence={persistence:.3f}  unconditional sigma={uncond_sigma:.2f}% per month "
              f"(~{uncond_sigma * np.sqrt(12):.0f}% annualized)")
        print(f"         end-of-history sigma={float(sigma_history.iloc[-1]):.2f}%  ->  "
              f"{FORECAST_END.strftime('%b-%Y')} forecast sigma={float(sigma_path[-1]):.2f}%")

        params_rows.append({
            "crude": crude, "model": chosen,
            "omega": round(omega, 5), "alpha": round(a, 4),
            "beta": round(b, 4), "gamma": round(g, 4),
            "persistence": round(persistence, 4),
            "uncond_sigma_monthly_pct": round(uncond_sigma, 3),
            "uncond_sigma_ann_pct":     round(uncond_sigma * np.sqrt(12), 2),
            "end_history_sigma_pct":    round(float(sigma_history.iloc[-1]), 3),
            "forecast_end_sigma_pct":   round(float(sigma_path[-1]), 3),
            "LL": round(res.loglikelihood, 2),
            "BIC": round(res.bic, 2),
            "n_obs": int(res.nobs),
        })

    # Save
    pd.DataFrame(params_rows).to_csv(config.OUTPUT_DIR / "vol_params.csv", index=False)
    pd.concat(comparison_rows, ignore_index=True).to_csv(
        config.OUTPUT_DIR / "vol_model_comparison.csv", index=False)
    panel.to_csv(config.OUTPUT_DIR / "vol_history.csv")
    pd.DataFrame(fcst_rows).to_csv(config.OUTPUT_DIR / "vol_forecast.csv", index=False)

    print(f"\nSaved: {config.OUTPUT_DIR / 'vol_params.csv'}")
    print(f"Saved: {config.OUTPUT_DIR / 'vol_model_comparison.csv'}")
    print(f"Saved: {config.OUTPUT_DIR / 'vol_history.csv'}")
    print(f"Saved: {config.OUTPUT_DIR / 'vol_forecast.csv'}")

    out1 = config.CHARTS_DIR / "10_vol_history.png"
    _chart_history(panel, out1)
    print(f"Saved: {out1}")

    fcst_df = pd.DataFrame(fcst_rows)
    out2 = config.CHARTS_DIR / "11_vol_forecast.png"
    _chart_forecast(panel, fcst_df, out2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
