"""
One-step-ahead trading signals from the structural models.

For each tradable target (Light, Heavy, WCS-WTI spread):
  1. Refit on all available data.
  2. Predict next month using current values of exogenous regressors.
  3. Convert the prediction into a directional signal + confidence band.

Confidence is the predicted log return (or differential change) divided by
that fit's residual sigma. |z| < 0.5 -> NEUTRAL; >= 0.5 -> directional;
>= 1.0 -> high conviction.

Outputs:
  output/signals.csv   one row per asset for the upcoming month
  Console: human-readable summary the friend can paste into a deck
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
from run_models import _resolve_levels, _drop_constant_cols

warnings.filterwarnings("ignore", category=Warning)

LOW_CONF  = 0.5   # in residual-sigma units
HIGH_CONF = 1.0


def _label(z: float, up_word: str, down_word: str) -> str:
    if abs(z) < LOW_CONF:
        return "NEUTRAL"
    if abs(z) < HIGH_CONF:
        return f"{up_word} (low conf)" if z > 0 else f"{down_word} (low conf)"
    return f"{up_word} (HIGH CONF)" if z > 0 else f"{down_word} (HIGH CONF)"


def _signal_for_crude(df: pd.DataFrame, dep: str, lag_var: str, level_col: str) -> dict:
    regressors = _resolve_levels(lag_var)
    sub = df[df["war_long"] == 1][[dep] + regressors].dropna()
    regs = _drop_constant_cols(sub, regressors)
    X = sm.add_constant(sub[regs])
    res = sm.OLS(sub[dep], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    sigma = float(np.sqrt(res.mse_resid))

    last_log = float(df[dep].dropna().iloc[-1])
    last_price = float(df[level_col].dropna().iloc[-1])
    last_date = df[dep].dropna().index.max()

    # Build the "next month" feature vector by carrying the latest known
    # exogenous values forward (one-step ahead, no AR projection needed).
    x_next = {lag_var: last_log}
    for r in regs:
        if r == lag_var:
            continue
        v = df[r].dropna()
        x_next[r] = float(v.iloc[-1]) if len(v) else 0.0
    x_pred = pd.DataFrame([[x_next[r] for r in regs]], columns=regs)
    x_pred = sm.add_constant(x_pred, has_constant="add")
    pred_log = float(res.predict(x_pred)[0])
    pred_price = float(np.exp(pred_log))
    log_ret = pred_log - last_log
    z = log_ret / sigma if sigma > 0 else 0.0
    direction = _label(z, "BULLISH", "BEARISH")

    return {
        "asset":             "Light (WTI)" if level_col == "wti_real" else "Heavy (RAC proxy)",
        "as_of":             str(last_date.date()),
        "current_price":     round(last_price, 2),
        "predicted_price":   round(pred_price, 2),
        "predicted_return_%": round(log_ret * 100, 2),
        "z_score":           round(z, 2),
        "signal":            direction,
        "residual_sigma":    round(sigma, 4),
    }


def _signal_for_diff(df: pd.DataFrame) -> dict:
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

    x_next = {"wcs_wti_diff_lag1": last_diff}
    for r in regs:
        if r == "wcs_wti_diff_lag1":
            continue
        v = df[r].dropna()
        x_next[r] = float(v.iloc[-1]) if len(v) else 0.0
    x_pred = pd.DataFrame([[x_next[r] for r in regs]], columns=regs)
    x_pred = sm.add_constant(x_pred, has_constant="add")
    pred = float(res.predict(x_pred)[0])
    delta = pred - last_diff
    z = delta / sigma if sigma > 0 else 0.0
    direction = _label(z, "WIDENING", "NARROWING")

    return {
        "asset":            "WCS - WTI spread",
        "as_of":            str(last_date.date()),
        "current_price":    round(last_diff, 2),
        "predicted_price":  round(pred, 2),
        "predicted_return_%": round(delta, 2),  # in $/bbl, not %
        "z_score":          round(z, 2),
        "signal":           direction,
        "residual_sigma":   round(sigma, 4),
    }


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    rows = [
        _signal_for_crude(df, "log_wti_real", "log_wti_lag1", "wti_real"),
        _signal_for_crude(df, "log_wcs_real", "log_wcs_lag1", "wcs_real"),
        _signal_for_diff(df),
    ]
    out = pd.DataFrame(rows)
    out_path = config.OUTPUT_DIR / "signals.csv"
    out.to_csv(out_path, index=False)

    # Human-readable summary
    print("=" * 72)
    print(f"NEXT-MONTH SIGNALS  (as of {rows[0]['as_of']})")
    print("=" * 72)
    print()
    for r in rows:
        unit = "$/bbl change" if r["asset"] == "WCS - WTI spread" else "% predicted return"
        print(f"  {r['asset']:22s}  {r['signal']}")
        print(f"  {'':22s}  current ${r['current_price']:>7.2f}  ->  forecast ${r['predicted_price']:>7.2f}")
        print(f"  {'':22s}  {unit:24s} {r['predicted_return_%']:+.2f}    z = {r['z_score']:+.2f}")
        print()

    print("Confidence guide:")
    print(f"  |z| < {LOW_CONF}     NEUTRAL    (no actionable edge)")
    print(f"  |z| < {HIGH_CONF}     low confidence directional")
    print(f"  |z| >= {HIGH_CONF}    HIGH conviction directional")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
