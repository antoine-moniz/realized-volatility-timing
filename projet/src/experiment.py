from __future__ import annotations

from datetime import datetime

import pandas as pd

from investment_lab.backtest import BacktesterBidAskFromData, StrategyBacktester
from investment_lab.metrics.performance import calmar_ratio, max_drawdown, sharpe_ratio
from investment_lab.option_trade import OptionTrade

from .config import RollingMLEConfig, UKFConfig
from .data_pipeline import build_market_dataset
from .estimation import run_rolling_heston_filter
from .signal_allocation import (
    apply_dynamic_allocation,
    build_spread_signal,
    spread_to_multiplier,
    spread_to_multiplier_regime_based,
)


def _compute_perf_table(nav: pd.Series) -> pd.DataFrame:
    rets = nav.pct_change().dropna()
    table = pd.DataFrame(
        {
            "sharpe": [sharpe_ratio(rets)],
            "max_drawdown": [max_drawdown(rets)],
            "calmar": [calmar_ratio(rets)],
            "nav_final": [float(nav.dropna().iloc[-1])],
        }
    )
    return table


def run_dynamic_carry_experiment(
    start_date: datetime,
    end_date: datetime,
    ticker: str,
    legs: list[dict],
    day_to_expiry_target_iv: int = 30,
    with_bid_ask_cost: bool = True,
    ukf_cfg: UKFConfig | None = None,
    mle_cfg: RollingMLEConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Pipeline end-to-end: baseline carry vs carry dynamique pilote par spread IV-RV."""
    ukf_cfg = ukf_cfg or UKFConfig()
    mle_cfg = mle_cfg or RollingMLEConfig()

    # 1) Donnees pour signal UKF
    df_market = build_market_dataset(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        day_to_expiry_target=day_to_expiry_target_iv,
    )

    df_est = run_rolling_heston_filter(df_market=df_market, ukf_cfg=ukf_cfg, mle_cfg=mle_cfg)
    df_signal = build_spread_signal(df_estimation=df_est)
    # AMÉLIORATION: utiliser z_cap=3.0 pour clipper les outliers extrêmes
    # et stabiliser l'allocation dynamique (au lieu de z_cap=2.0 par défaut)
    df_mult = spread_to_multiplier(df_signal, z_cap=3.0)

    # 2) Positions baseline
    df_trades_base = OptionTrade.generate_trades(
        start_date=start_date,
        end_date=end_date,
        tickers=ticker,
        legs=legs,
        cost_neutral=False,
    )

    # 3) Positions dynamiques
    df_trades_dyn = apply_dynamic_allocation(df_trades=df_trades_base.copy(), df_multiplier=df_mult)

    # 4) Backtests
    backtester_cls = BacktesterBidAskFromData if with_bid_ask_cost else StrategyBacktester
    bt_base = backtester_cls(df_trades_base).compute_backtest()
    bt_dyn = backtester_cls(df_trades_dyn).compute_backtest()

    perf_base = _compute_perf_table(bt_base.nav["NAV"])
    perf_base["mode"] = "baseline"
    perf_dyn = _compute_perf_table(bt_dyn.nav["NAV"])
    perf_dyn["mode"] = "dynamic"

    return {
        "market": df_market,
        "estimation": df_est,
        "signal": df_signal,
        "multiplier": df_mult,
        "trades_baseline": df_trades_base,
        "trades_dynamic": df_trades_dyn,
        "pnl_baseline": bt_base.pnl.reset_index().rename(columns={"index": "date"}),
        "pnl_dynamic": bt_dyn.pnl.reset_index().rename(columns={"index": "date"}),
        "nav_baseline": bt_base.nav.reset_index().rename(columns={"index": "date"}),
        "nav_dynamic": bt_dyn.nav.reset_index().rename(columns={"index": "date"}),
        "perf": pd.concat([perf_base, perf_dyn], ignore_index=True),
    }


def run_dynamic_carry_experiment_regime_based(
    start_date: datetime,
    end_date: datetime,
    ticker: str,
    legs: list[dict],
    day_to_expiry_target_iv: int = 30,
    with_bid_ask_cost: bool = True,
    ukf_cfg: UKFConfig | None = None,
    mle_cfg: RollingMLEConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Pipeline with REGIME-BASED allocation (improved non-linear allocation strategy)."""
    ukf_cfg = ukf_cfg or UKFConfig()
    mle_cfg = mle_cfg or RollingMLEConfig()

    df_market = build_market_dataset(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        day_to_expiry_target=day_to_expiry_target_iv,
    )

    df_est = run_rolling_heston_filter(df_market=df_market, ukf_cfg=ukf_cfg, mle_cfg=mle_cfg)
    df_signal = build_spread_signal(df_estimation=df_est)
    # AMÉLIORATION #2: Allocation non-linéaire basée sur 3 régimes
    # Capture mieux l'asymétrie du signal IV-RV
    df_mult = spread_to_multiplier_regime_based(df_signal)

    df_trades_base = OptionTrade.generate_trades(
        start_date=start_date,
        end_date=end_date,
        tickers=ticker,
        legs=legs,
        cost_neutral=False,
    )

    df_trades_dyn = apply_dynamic_allocation(df_trades=df_trades_base.copy(), df_multiplier=df_mult)

    backtester_cls = BacktesterBidAskFromData if with_bid_ask_cost else StrategyBacktester
    bt_base = backtester_cls(df_trades_base).compute_backtest()
    bt_dyn = backtester_cls(df_trades_dyn).compute_backtest()

    perf_base = _compute_perf_table(bt_base.nav["NAV"])
    perf_base["mode"] = "baseline"
    perf_dyn = _compute_perf_table(bt_dyn.nav["NAV"])
    perf_dyn["mode"] = "dynamic (regime-based)"

    return {
        "market": df_market,
        "estimation": df_est,
        "signal": df_signal,
        "multiplier": df_mult,
        "trades_baseline": df_trades_base,
        "trades_dynamic": df_trades_dyn,
        "pnl_baseline": bt_base.pnl.reset_index().rename(columns={"index": "date"}),
        "pnl_dynamic": bt_dyn.pnl.reset_index().rename(columns={"index": "date"}),
        "nav_baseline": bt_base.nav.reset_index().rename(columns={"index": "date"}),
        "nav_dynamic": bt_dyn.nav.reset_index().rename(columns={"index": "date"}),
        "perf": pd.concat([perf_base, perf_dyn], ignore_index=True),
    }
