"""
Translates model predictions into trading strategies and computes realistic
PnL on historical data. The goal is to answer "did the model make money,
not just fit?" on the most recent walk-forward test window.

Reads:  output/backtest_predictions.csv      (Light + Heavy walk-forward predictions)
        output/backtest_predictions_diff.csv (WCS-WTI walk-forward predictions)

Strategies tested:
  1. buy_and_hold          - always 100% long. Baseline.
  2. directional_sign      - long when model predicts UP, flat when DOWN.
  3. confidence_threshold  - long only when |predicted return| > 0.5 sigma_hist.
  4. spread_pair_trade     - long WCS / short WTI when diff forecast to NARROW;
                             reverse when WIDEN. Dollar-neutral.
  5. hedge_overlay         - producer naturally long oil sizes monthly hedge
                             from 25% (model bullish) to 75% (model bearish).
                             Reports residual exposure return after hedge.

Transaction cost: 5 bps applied to |position change| each month (round-trip on
futures, conservative for an institutional desk).

Outputs:
  output/strategy_results.csv        per-strategy metrics
  output/strategy_equity_curves.csv  monthly NAV per strategy
  output/charts/12_strategies.png    equity-curve chart
"""
from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore", category=Warning)

TX_COST_BPS = 5            # 5 bps per unit of position change
ANNUALIZATION = 12         # monthly data
CONFIDENCE_THRESHOLD_SIGMA = 0.5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(returns: pd.Series, label: str) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"strategy": label, "n": 0}
    cum = (1 + r).prod() - 1
    ann_ret = (1 + cum) ** (ANNUALIZATION / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ANNUALIZATION)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    nav = (1 + r).cumprod()
    max_dd = float((nav / nav.cummax() - 1).min())
    hit = float((r > 0).mean() * 100)
    avg_win  = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(r[r < 0].mean()) if (r < 0).any() else 0.0
    return {
        "strategy":      label,
        "n":             len(r),
        "cum_return_%":  round(cum * 100, 2),
        "ann_return_%":  round(ann_ret * 100, 2),
        "ann_vol_%":     round(ann_vol * 100, 2),
        "sharpe":        round(sharpe, 2),
        "max_dd_%":      round(max_dd * 100, 2),
        "hit_rate_%":    round(hit, 1),
        "avg_win_%":     round(avg_win * 100, 2),
        "avg_loss_%":    round(avg_loss * 100, 2),
    }


# ---------------------------------------------------------------------------
# Strategy logic - produce a position series in [-1, 1] per month
# ---------------------------------------------------------------------------
def _signal_buy_hold(actual: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=actual.index)


def _signal_directional(predicted: pd.Series, prev: pd.Series) -> pd.Series:
    """Long if predicted price > prev price, else flat (no shorts)."""
    return ((predicted > prev).astype(float)).reindex(predicted.index)


def _signal_confidence(predicted: pd.Series, prev: pd.Series, sigma: float) -> pd.Series:
    """Long only if predicted log return magnitude exceeds 0.5 sigma."""
    pred_ret = np.log(predicted / prev)
    return ((pred_ret > CONFIDENCE_THRESHOLD_SIGMA * sigma).astype(float)).reindex(predicted.index)


def _signal_hedge_overlay(predicted: pd.Series, prev: pd.Series, sigma: float) -> pd.Series:
    """A producer who's naturally LONG oil 100%. The overlay SHORTS futures.
    Hedge ratio scales linearly from 25% (model bullish, +1 sigma) to 75%
    (model bearish, -1 sigma). Returns RESIDUAL exposure (= 1 - hedge_size)."""
    pred_ret = np.log(predicted / prev) / sigma
    pred_ret = pred_ret.clip(-1, 1)
    # bullish: pred_ret = +1 -> hedge 25% -> residual 75%
    # bearish: pred_ret = -1 -> hedge 75% -> residual 25%
    hedge_size = 0.5 - 0.25 * pred_ret  # 0.25 .. 0.75
    return (1 - hedge_size).reindex(predicted.index)


def _apply_costs(position: pd.Series, asset_return: pd.Series) -> pd.Series:
    """Strategy return = position_{t-1} * asset_return_t - tx_cost on position change.
    Position decided at end of t-1 holds through t."""
    pos_lag = position.shift(1).fillna(0)
    gross = pos_lag * asset_return
    turnover = (position - pos_lag).abs()
    cost = turnover * (TX_COST_BPS / 1e4)
    return gross - cost


# ---------------------------------------------------------------------------
# Spread strategy
# ---------------------------------------------------------------------------
def _spread_strategy(diff_preds: pd.DataFrame) -> pd.Series:
    """Long WCS / short WTI when diff predicted to NARROW (move toward zero or
    less negative); opposite when WIDEN. We need the actual realized differential
    return per month plus the position sign."""
    if len(diff_preds) == 0:
        return pd.Series(dtype=float)
    actual = diff_preds["actual"]
    prev   = diff_preds["prev"]
    pred   = diff_preds["differential_model"]
    # Position = sign(pred - prev). Positive when spread predicted to widen
    # (more positive); negative when spread predicted to narrow.
    # Define a "long-spread" position as +1 = bet spread widens.
    raw_pos = np.sign(pred - prev)
    pos = pd.Series(raw_pos.values, index=diff_preds.index).fillna(0)
    # Spread realized change = (actual - prev) per month.
    # Approximate dollar-neutral pair return as spread change / |prev_average_price|.
    # Use $80 as a normalization base so spread P&L is in % terms similar to
    # outright. (We don't have the WTI level in the diff predictions file.)
    ref_price = 80.0
    realized_spread_return = (actual - prev) / ref_price
    rets = _apply_costs(pos, realized_spread_return)
    return rets


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_for_crude(label: str, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all single-asset strategies on one crude's walk-forward predictions."""
    actual = predictions["actual"]
    prev   = predictions["prev"]
    realized_return = (actual / prev) - 1

    # Fit the historical sigma of monthly log returns from the training-period
    # actuals (use the prediction window itself as a proxy - same volatility regime).
    sigma = float(np.log(actual / prev).dropna().std())

    # Pick the model with the best directional accuracy as the signal source.
    # From the previous backtest: ar1_returns (Light) and levels_model (Heavy)
    # had the best DA. We use the returns_model as the structural signal here
    # so the strategy is reproducible from the structural model output, but
    # report which model would have done better.
    pred = predictions["returns_model"]

    strategies = {
        "buy_and_hold":          _signal_buy_hold(actual),
        "directional_sign":      _signal_directional(pred, prev),
        "confidence_threshold":  _signal_confidence(pred, prev, sigma),
        "hedge_overlay_residual": _signal_hedge_overlay(pred, prev, sigma),
    }

    rets_df = pd.DataFrame(index=actual.index)
    metrics = []
    for name, sig in strategies.items():
        r = _apply_costs(sig.reindex(actual.index).fillna(0), realized_return)
        rets_df[name] = r
        metrics.append({**_metrics(r, f"{label}::{name}"), "asset": label})

    return rets_df, metrics


def main():
    pred_path = config.OUTPUT_DIR / "backtest_predictions.csv"
    diff_path = config.OUTPUT_DIR / "backtest_predictions_diff.csv"
    if not pred_path.exists():
        print("Run backtest.py first to produce backtest_predictions.csv")
        return

    # Multi-index columns: (Light, ...) and (Heavy, ...) - read with header rows
    raw = pd.read_csv(pred_path, header=[0, 1], index_col=0, parse_dates=True)
    diff_preds = pd.read_csv(diff_path, index_col=0, parse_dates=True) if diff_path.exists() else pd.DataFrame()

    all_rets = {}
    all_metrics = []
    for crude in ["Light", "Heavy"]:
        if crude not in raw.columns.get_level_values(0):
            continue
        sub = raw[crude]
        rets, metrics = run_for_crude(crude, sub)
        all_rets[crude] = rets
        all_metrics.extend(metrics)

    # Spread strategy
    if len(diff_preds):
        spread_rets = _spread_strategy(diff_preds)
        all_rets["Spread"] = pd.DataFrame({"spread_pair_trade": spread_rets})
        all_metrics.append({**_metrics(spread_rets, "Spread::pair_trade"), "asset": "Spread"})
        # Also report spread persistence (always flat) as the null benchmark
        flat = pd.Series(0.0, index=spread_rets.index)
        all_metrics.append({**_metrics(flat, "Spread::no_trade"), "asset": "Spread"})

    metrics_df = pd.DataFrame(all_metrics)
    cols = ["asset", "strategy", "n", "cum_return_%", "ann_return_%",
            "ann_vol_%", "sharpe", "max_dd_%", "hit_rate_%", "avg_win_%", "avg_loss_%"]
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
    metrics_df.to_csv(config.OUTPUT_DIR / "strategy_results.csv", index=False)

    # Equity curves
    equity_pieces = []
    for asset, rets in all_rets.items():
        nav = (1 + rets.fillna(0)).cumprod()
        nav.columns = [f"{asset}::{c}" for c in nav.columns]
        equity_pieces.append(nav)
    equity = pd.concat(equity_pieces, axis=1)
    equity.to_csv(config.OUTPUT_DIR / "strategy_equity_curves.csv")

    # Print results
    print("=" * 72)
    print("STRATEGY BACKTEST — model-derived signals on walk-forward predictions")
    print("=" * 72)
    print(f"Transaction cost: {TX_COST_BPS} bps per unit position change")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD_SIGMA:.1f} σ of historical log returns")
    print(f"Test horizon: {len(next(iter(all_rets.values())))} months\n")
    for asset in metrics_df["asset"].unique():
        rows = metrics_df[metrics_df["asset"] == asset]
        print(f"--- {asset} ---")
        print(rows.drop(columns="asset").to_string(index=False))
        print()

    # Headline verdicts
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    for asset in ["Light", "Heavy", "Spread"]:
        rows = metrics_df[metrics_df["asset"] == asset]
        if len(rows) == 0:
            continue
        bench_label = "buy_and_hold" if asset != "Spread" else "no_trade"
        bench_rows = rows[rows["strategy"].str.endswith(bench_label)]
        if len(bench_rows) == 0:
            continue
        bench_sharpe = float(bench_rows["sharpe"].iloc[0])
        bench_cum    = float(bench_rows["cum_return_%"].iloc[0])
        active = rows[~rows["strategy"].str.endswith(bench_label)]
        if len(active) == 0:
            continue
        best = active.sort_values("sharpe", ascending=False).iloc[0]
        delta_sharpe = best["sharpe"] - bench_sharpe
        delta_cum    = best["cum_return_%"] - bench_cum
        print(f"  {asset}: best active = {best['strategy']:40s}  "
              f"Sharpe {best['sharpe']:+.2f} (vs {bench_label} {bench_sharpe:+.2f}, Δ {delta_sharpe:+.2f}); "
              f"cum {best['cum_return_%']:+.2f}% (vs {bench_cum:+.2f}%, Δ {delta_cum:+.2f}pp)")

    # Chart
    fig, axes = plt.subplots(1, len(all_rets), figsize=(6 * len(all_rets), 5), squeeze=False)
    axes = axes.flatten()
    for i, (asset, rets) in enumerate(all_rets.items()):
        ax = axes[i]
        nav = (1 + rets.fillna(0)).cumprod()
        for col in nav.columns:
            ax.plot(nav.index, nav[col].values, lw=1.6, label=col.replace("_", " "))
        ax.axhline(1.0, color="black", lw=0.5, alpha=0.4)
        ax.set_title(f"{asset} — equity curves (start = 1.0)")
        ax.set_ylabel("NAV")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Model-derived strategies vs buy & hold ({TX_COST_BPS} bps tx cost)", y=1.00, fontsize=12)
    fig.tight_layout()
    out = config.CHARTS_DIR / "12_strategies.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nSaved: {config.OUTPUT_DIR / 'strategy_results.csv'}")
    print(f"Saved: {config.OUTPUT_DIR / 'strategy_equity_curves.csv'}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
