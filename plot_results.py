"""
Generate evaluation charts for the four OLS oil price models.

Outputs PNGs to output/charts/:
    01_price_history_with_regimes.png  - Real prices over time, war regimes shaded
    02_actual_vs_fitted.png            - 2x2 grid, one panel per model
    03_residuals_timeseries.png        - 2x2 grid, residuals over time per model
    04_residual_diagnostics.png        - 2x2 grid: hist, Q-Q, resid vs fitted, ACF
    05_coefficients_comparison.png     - All four models side-by-side
    06_correlation_heatmap.png         - Regressor multicollinearity check
"""
from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

CRUDE_SPECS = [
    ("Light", 1, "log_wti_real", "log_wti_lag1", "wti_real",  "tab:blue"),
    ("Heavy", 0, "log_wcs_real", "log_wcs_lag1", "wcs_real",  "tab:orange"),
]
WAR_SPECS = [("Short-term", 0), ("Long-term", 1)]


def _drop_constant_cols(sub, regressors):
    return [r for r in regressors if sub[r].nunique(dropna=True) > 1]


def _resolve_regressors(lag_var: str) -> list[str]:
    return [lag_var if x == "log_price_lag1" else x for x in config.REGRESSORS]


def _fit(df: pd.DataFrame, dep: str, regressors: list[str]):
    cols = [dep] + regressors
    sub = df[cols].dropna()
    regressors = _drop_constant_cols(sub, regressors)
    X = sm.add_constant(sub[regressors])
    y = sub[dep]
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return res, sub.index


def _shade_war_regimes(ax, df):
    long_mask = df["war_long"] == 1
    short_war_mask = (df["in_war"] == 1) & (df["war_long"] == 0)
    if long_mask.any():
        for start, end in _contig_runs(df.index, long_mask):
            ax.axvspan(start, end, color="red", alpha=0.12, lw=0)
    if short_war_mask.any():
        for start, end in _contig_runs(df.index, short_war_mask):
            ax.axvspan(start, end, color="orange", alpha=0.15, lw=0)


def _contig_runs(idx, mask):
    out, in_run, start = [], False, None
    arr = mask.values
    for i, v in enumerate(arr):
        if v and not in_run:
            in_run, start = True, idx[i]
        elif not v and in_run:
            in_run = False
            out.append((start, idx[i - 1]))
    if in_run:
        out.append((start, idx[-1]))
    return out


# ---------------------------------------------------------------------------
def chart_price_history(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df.index, df["wti_real"], label="Light oil (WTI)", color="#16697A", lw=1.6)
    ax.plot(df.index, df["wcs_real"], label="Heavy oil", color="#E8893E", lw=1.6)
    _shade_war_regimes(ax, df)

    # Annotate war labels - stagger y-positions to prevent collision when wars overlap or sit close
    y_max = df["wti_real"].max()
    levels = [0.97, 0.90, 0.83, 0.76]   # cycle through to stagger labels vertically
    last_end_per_level = {i: pd.Timestamp("1900-01-01") for i in range(len(levels))}
    min_gap_days = 365  # ~1yr between labels at the same level

    for start, end, label, _iran in config.WAR_WINDOWS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else df.index.max()
        if s >= df.index.max() or e <= df.index.min():
            continue
        mid = s + (e - s) / 2
        # Choose lowest level whose last label ended >= min_gap_days ago
        chosen = 0
        for i in range(len(levels)):
            if (mid - last_end_per_level[i]).days >= min_gap_days:
                chosen = i
                break
        y = y_max * levels[chosen]
        ax.annotate(label, xy=(mid, y), ha="center", va="bottom",
                    fontsize=8, alpha=0.85, fontweight="bold", color="#1A1A2E",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=0.6, alpha=0.85))
        last_end_per_level[chosen] = mid

    ax.set_title(f"Oil prices over time (in {config.REAL_PRICE_BASE_YEAR} dollars, adjusted for inflation)\n"
                 "Red shading = long conflict periods   |   Orange = short conflict periods")
    ax.set_ylabel("Price per barrel (USD, 2025 dollars)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.set_ylim(0, y_max * 1.08)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = config.CHARTS_DIR / "01_price_history_with_regimes.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
def chart_actual_vs_fitted(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    for i, (crude, b1, dep, lag, lvl, color) in enumerate(CRUDE_SPECS):
        for j, (war_label, b2) in enumerate(WAR_SPECS):
            ax = axes[i][j]
            sub = df[df["war_long"] == b2]
            res, idx = _fit(sub, dep, _resolve_regressors(lag))
            actual = res.fittedvalues + res.resid
            fitted = res.fittedvalues
            # Convert log -> level for readability
            base_cpi = df.loc[df.index.year == config.REAL_PRICE_BASE_YEAR, "cpi"].mean()
            actual_lvl = np.exp(actual)
            fitted_lvl = np.exp(fitted)
            ax.scatter(actual_lvl, fitted_lvl, s=12, alpha=0.6, color=color)
            lim_lo = min(actual_lvl.min(), fitted_lvl.min()) * 0.95
            lim_hi = max(actual_lvl.max(), fitted_lvl.max()) * 1.05
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.6)
            ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
            ax.set_xlabel("Actual price (USD/barrel)")
            ax.set_ylabel("Predicted price (USD/barrel)")
            war_friendly = "short conflict" if b2 == 0 else "long conflict"
            crude_friendly = "Light oil" if crude == "Light" else "Heavy oil"
            ax.set_title(f"{crude_friendly}, {war_friendly}   ({int(res.nobs)} months, {res.rsquared*100:.0f}% fit)")
    fig.suptitle("How well predictions match reality (closer to the line = better)", y=1.00)
    fig.tight_layout()
    out = config.CHARTS_DIR / "02_actual_vs_fitted.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
def chart_residuals_timeseries(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, (crude, b1, dep, lag, lvl, color) in enumerate(CRUDE_SPECS):
        for j, (war_label, b2) in enumerate(WAR_SPECS):
            ax = axes[i][j]
            sub = df[df["war_long"] == b2]
            res, idx = _fit(sub, dep, _resolve_regressors(lag))
            ax.axhline(0, color="black", lw=0.6)
            ax.plot(idx, res.resid, color=color, lw=0.9, marker="o", ms=2.5)
            ax.set_title(f"{crude} | {war_label}  residuals (log scale)")
            ax.xaxis.set_major_locator(mdates.YearLocator(3))
            for label in ax.get_xticklabels():
                label.set_rotation(0)
    fig.suptitle("Residuals over time — look for patterns (autocorrelation, regime drift)", y=1.00)
    fig.tight_layout()
    out = config.CHARTS_DIR / "03_residuals_timeseries.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
def chart_residual_diagnostics(df: pd.DataFrame):
    """For each model: hist, Q-Q, residuals vs fitted, ACF. One figure per model."""
    for crude, b1, dep, lag, lvl, color in CRUDE_SPECS:
        for war_label, b2 in WAR_SPECS:
            sub = df[df["war_long"] == b2]
            res, _ = _fit(sub, dep, _resolve_regressors(lag))
            r = res.resid

            fig, axes = plt.subplots(2, 2, figsize=(9, 7))

            ax = axes[0][0]
            ax.hist(r, bins=20, color=color, alpha=0.75, edgecolor="white")
            ax.set_title("Residual histogram"); ax.set_xlabel("residual")

            ax = axes[0][1]
            stats.probplot(r, dist="norm", plot=ax)
            ax.set_title("Q-Q vs normal")
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[0].set_alpha(0.6)

            ax = axes[1][0]
            ax.scatter(res.fittedvalues, r, s=10, alpha=0.6, color=color)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title("Residuals vs fitted")
            ax.set_xlabel("fitted log(real_price)"); ax.set_ylabel("residual")

            ax = axes[1][1]
            sm.graphics.tsa.plot_acf(r, lags=min(24, len(r) // 4), ax=ax)
            ax.set_title("Residual ACF (autocorrelation)")

            fig.suptitle(f"{crude} | {war_label}  diagnostics  (n={int(res.nobs)}, DW={sm.stats.stattools.durbin_watson(r):.2f})", y=1.00)
            fig.tight_layout()
            tag = f"{crude.lower()}_{war_label.lower().replace('-', '')}"
            out = config.CHARTS_DIR / f"04_diagnostics_{tag}.png"
            fig.savefig(out)
            plt.close(fig)
            print(f"  {out.name}")


# ---------------------------------------------------------------------------
def chart_coefficients_comparison(df: pd.DataFrame):
    """Side-by-side coefficient plot with 95% CIs, all 4 models."""
    friendly_labels = {
        "log_production":  "US production",
        "hormuz_threat":   "Hormuz threat",
        "log_inventory":   "Storage levels",
        "log_wti_lag1":    "Last month's price (light)",
        "log_wcs_lag1":    "Last month's price (heavy)",
        "net_exports":     "Net exports",
        "refinery_util":   "Refinery activity",
        "log_dxy":         "US dollar strength",
        "gpr":             "Risk index",
        "crack_spread":    "Refining margin",
        "month_sin":       "Seasonality (sine)",
        "month_cos":       "Seasonality (cosine)",
    }
    friendly_models = {
        "Light | Short-term": "Light oil  |  Short conflict",
        "Light | Long-term":  "Light oil  |  Long conflict",
        "Heavy | Short-term": "Heavy oil  |  Short conflict",
        "Heavy | Long-term":  "Heavy oil  |  Long conflict",
    }

    rows = []
    for crude, b1, dep, lag, lvl, color in CRUDE_SPECS:
        for war_label, b2 in WAR_SPECS:
            sub = df[df["war_long"] == b2]
            res, _ = _fit(sub, dep, _resolve_regressors(lag))
            ci = res.conf_int()
            for var in res.params.index:
                if var == "const":
                    continue
                rows.append({
                    "model": friendly_models.get(f"{crude} | {war_label}", f"{crude} | {war_label}"),
                    "var":   friendly_labels.get(var, var),
                    "coef":  res.params[var],
                    "lo":    ci.loc[var, 0],
                    "hi":    ci.loc[var, 1],
                    "p":     res.pvalues[var],
                })
    coef_df = pd.DataFrame(rows)
    variables = list(coef_df["var"].unique())
    models    = list(coef_df["model"].unique())
    n_models  = len(models)

    fig, ax = plt.subplots(figsize=(11, max(6, len(variables) * 0.55)))
    width = 0.18
    y = np.arange(len(variables))
    palette = ["tab:blue", "tab:cyan", "tab:orange", "tab:red"]
    for i, m in enumerate(models):
        sub = coef_df[coef_df["model"] == m].set_index("var").reindex(variables)
        offset = (i - (n_models - 1) / 2) * width
        ax.errorbar(sub["coef"], y + offset,
                    xerr=[sub["coef"] - sub["lo"], sub["hi"] - sub["coef"]],
                    fmt="o", color=palette[i], label=m, capsize=2.5, ms=4)
    ax.axvline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel("Effect on price  (negative = pushes price down, positive = up)")
    ax.set_title("How much each factor moves oil prices\n"
                 "Bars show the range of uncertainty; bars crossing zero mean the effect is unclear")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    out = config.CHARTS_DIR / "05_coefficients_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
def chart_correlation_heatmap(df: pd.DataFrame):
    cols = ["log_wti_real", "log_wcs_real"] + [c for c in config.REGRESSORS if c != "log_price_lag1"]
    cols += ["log_wti_lag1"]
    sub = df[cols].dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson corr")
    ax.set_title("Regressor correlation matrix\nHigh off-diagonal values indicate multicollinearity")
    fig.tight_layout()
    out = config.CHARTS_DIR / "06_correlation_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
def chart_differential_history(df: pd.DataFrame):
    """WCS-WTI spread over time with regime overlay (TMX in-service, COVID,
    Russia war). Shows the structural breaks the differential model exploits."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    diff = df["wcs_wti_diff"].dropna()
    ax.plot(diff.index, diff.values, color="#16697A", lw=1.6, label="WCS - WTI ($/bbl, real)")
    ax.axhline(0, color="black", lw=0.6)
    palette = {"shale_era": "#2A8B5E", "covid": "#B83A4B",
               "russia_war": "#D4A55A", "tmx_inservice": "#7A4FB8"}
    for key, start, end, desc in config.STRUCTURAL_BREAKS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else df.index.max()
        if s > df.index.max() or e < df.index.min():
            continue
        ax.axvspan(s, e, color=palette.get(key, "gray"), alpha=0.10, lw=0)
        # Label at top
        mid = s + (e - s) / 2
        ax.annotate(desc.split('(')[0].strip(), xy=(mid, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 0),
                    xytext=(0, -8), textcoords="offset points",
                    ha="center", fontsize=7, color=palette.get(key, "gray"),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.set_title("WCS - WTI heavy/light spread with structural-break regimes\n"
                 "Negative = WCS at discount (typical); narrowing = better Enbridge mainline economics")
    ax.set_ylabel("Spread ($/bbl)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    out = config.CHARTS_DIR / "10_differential_history.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"  {out.name}")


def chart_opec_events(df: pd.DataFrame):
    """Stem chart of OPEC+ supply policy events with cumulative line."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    shocks = df["opec_shock"].copy()
    shocks_only = shocks[shocks != 0]
    cum = df["opec_cumulative"]
    colors = ["#2A8B5E" if v > 0 else "#B83A4B" for v in shocks_only.values]
    ax.bar(shocks_only.index, shocks_only.values, width=60, color=colors, alpha=0.85,
           label="Single-event delta (mbd)")
    ax2 = ax.twinx()
    ax2.plot(cum.index, cum.values, color="#1F1F2E", lw=1.6, label="Cumulative supply policy stance (mbd)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("OPEC+ supply policy events (green = cuts, red = increases)")
    ax.set_ylabel("Per-event Δ production (mbd)", color="#1F1F2E")
    ax2.set_ylabel("Cumulative (mbd)", color="#1F1F2E")
    ax.grid(alpha=0.3, axis="y")
    out = config.CHARTS_DIR / "11_opec_events.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"  {out.name}")


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    print(f"Generating charts -> {config.CHARTS_DIR}/")
    chart_price_history(df)
    chart_actual_vs_fitted(df)
    chart_residuals_timeseries(df)
    chart_residual_diagnostics(df)
    chart_coefficients_comparison(df)
    chart_correlation_heatmap(df)
    chart_differential_history(df)
    chart_opec_events(df)
    print("Done.")


if __name__ == "__main__":
    main()
