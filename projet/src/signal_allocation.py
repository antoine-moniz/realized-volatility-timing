from __future__ import annotations

import numpy as np
import pandas as pd


def build_spread_signal(
    df_estimation: pd.DataFrame,
    z_window: int = 63,
) -> pd.DataFrame:
    """Construit le spread s_t = sigma_IV,t - sigma_hat,t et sa version z-score."""
    required_cols = {"date", "ticker", "sigma_iv", "sigma_hat"}
    missing = required_cols.difference(df_estimation.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df_estimation: {missing}")

    df = df_estimation[["date", "ticker", "sigma_iv", "sigma_hat"]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["spread"] = df["sigma_iv"] - df["sigma_hat"]

    grouped = df.groupby("ticker")["spread"]
    mean_rolling = grouped.transform(lambda s: s.rolling(z_window, min_periods=max(10, z_window // 3)).mean())
    std_rolling = grouped.transform(lambda s: s.rolling(z_window, min_periods=max(10, z_window // 3)).std())

    df["spread_z"] = (df["spread"] - mean_rolling) / std_rolling.replace(0.0, np.nan)
    df["spread_z"] = df["spread_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def spread_to_multiplier(
    df_signal: pd.DataFrame,
    z_cap: float = 2.0,
    min_mult: float = 0.0,
    max_mult: float = 2.0,
) -> pd.DataFrame:
    """Traduit le spread normalise en multiplicateur d'exposition.

    Intuition short-vol carry:
    - spread positif (IV > RV estimee): on augmente l'exposition
    - spread negatif: on reduit l'exposition
    """
    required_cols = {"date", "ticker", "spread_z"}
    missing = required_cols.difference(df_signal.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df_signal: {missing}")

    df = df_signal[["date", "ticker", "spread_z"]].copy()
    clipped = np.clip(df["spread_z"].to_numpy(dtype=float), -z_cap, z_cap)
    raw_mult = 1.0 + clipped / z_cap

    df["allocation_multiplier"] = np.clip(raw_mult, min_mult, max_mult)
    return df


def spread_to_multiplier_regime_based(
    df_signal: pd.DataFrame,
    regime_threshold_low: float = -2.0,
    regime_threshold_high: float = 0.0,
    mult_extreme_neg: float = 0.2,  # Régime 1: spread très négatif → réduction importante
    mult_neutral: float = 1.0,       # Régime 2: spread neutre → position nominale
    mult_positive: float = 1.5,      # Régime 3: spread positif → augmentation modérée
) -> pd.DataFrame:
    """Allocation basée sur 3 régimes (non-linéaire) pour mieux exploiter l'asymétrie du spread.
    
    Régime 1 (z < -2): Spread très négatif → IV vendue bon marché → réduire carry
    Régime 2 (-2 < z < 0): Spread neutre → carry standard
    Régime 3 (z > 0): Spread positif → IV chère → augmenter carry modérément
    
    Cette approche est plus robuste que l'allocation linéaire pour les spreads asymétriques.
    """
    required_cols = {"date", "ticker", "spread_z"}
    missing = required_cols.difference(df_signal.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df_signal: {missing}")

    df = df_signal[["date", "ticker", "spread_z"]].copy()
    z = df["spread_z"].to_numpy(dtype=float)
    
    # Créer les multiplicateurs selon les régimes
    mult = np.ones_like(z, dtype=float)
    mult[z < regime_threshold_low] = mult_extreme_neg  # Régime 1: très négatif
    mult[(z >= regime_threshold_low) & (z < regime_threshold_high)] = mult_neutral  # Régime 2: neutre
    mult[z >= regime_threshold_high] = mult_positive  # Régime 3: positif
    
    df["allocation_multiplier"] = mult
    return df


def apply_dynamic_allocation(
    df_trades: pd.DataFrame,
    df_multiplier: pd.DataFrame,
) -> pd.DataFrame:
    """Applique un multiplicateur journalier d'allocation sur les poids de trades."""
    required_trades = {"date", "ticker", "weight"}
    missing_trades = required_trades.difference(df_trades.columns)
    if missing_trades:
        raise ValueError(f"Colonnes manquantes dans df_trades: {missing_trades}")

    required_mult = {"date", "ticker", "allocation_multiplier"}
    missing_mult = required_mult.difference(df_multiplier.columns)
    if missing_mult:
        raise ValueError(f"Colonnes manquantes dans df_multiplier: {missing_mult}")

    df = df_trades.merge(
        df_multiplier[["date", "ticker", "allocation_multiplier"]],
        on=["date", "ticker"],
        how="left",
    )
    df["allocation_multiplier"] = df["allocation_multiplier"].fillna(1.0)
    df["weight"] = df["weight"] * df["allocation_multiplier"]
    return df
