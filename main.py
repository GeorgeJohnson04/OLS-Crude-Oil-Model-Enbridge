"""
End-to-end runner: fetch -> features -> models -> backtest -> strategies ->
charts -> forecast -> signals.
"""
from fetch_data import fetch_panel
from build_features import build_features
from run_models import run_all
from plot_results import main as plot_main
from backtest import main as backtest_main
from strategy_backtest import main as strategy_main
from forecast import main as forecast_main
from signals import main as signals_main


STEPS = [
    ("Fetch raw panel from FRED + EIA + CBOE + CFTC + GPR", fetch_panel),
    ("Build feature matrix",                                 build_features),
    ("Fit OLS models (Levels + Returns + Differential)",     run_all),
    ("Walk-forward backtest vs naive baselines",             backtest_main),
    ("Strategy backtest — model signals to PnL",             strategy_main),
    ("Generate evaluation charts",                           plot_main),
    ("Forecast (Monte Carlo + scenarios) + differential",    forecast_main),
    ("Next-month trading signals",                           signals_main),
]


def main():
    n = len(STEPS)
    for i, (desc, fn) in enumerate(STEPS, 1):
        print()
        print("=" * 72)
        print(f"STEP {i}/{n}  {desc}")
        print("=" * 72)
        fn()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
