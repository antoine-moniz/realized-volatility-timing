"""Projet final - Realized Volatility Timing.

Ce package contient le pipeline complet:
- preparation des donnees spot/options
- estimation Heston via UKF + MLE glissant
- construction du signal IV-RV
- allocation dynamique des strategies carry
"""

from .config import HestonParams, RollingMLEConfig, UKFConfig
from .data_pipeline import build_market_dataset
from .estimation import run_rolling_heston_filter
from .signal_allocation import (
    build_spread_signal,
    spread_to_multiplier,
    apply_dynamic_allocation,
)
from .experiment import run_dynamic_carry_experiment

__all__ = [
    "HestonParams",
    "RollingMLEConfig",
    "UKFConfig",
    "build_market_dataset",
    "run_rolling_heston_filter",
    "build_spread_signal",
    "spread_to_multiplier",
    "apply_dynamic_allocation",
    "run_dynamic_carry_experiment",
]
