from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from investment_lab.data.option_db import OptionLoader, extract_spot_from_options
from investment_lab.option_selection import select_options


def _build_iv_proxy(df_options: pd.DataFrame, day_to_expiry_target: int = 30) -> pd.DataFrame:
    """Construit un proxy de volatilite implicite journaliere.

    On selectionne les options ATM (moneyness=1) calls et puts autour d'une maturite cible,
    puis on prend la moyenne call/put par date.
    """
    df_call = select_options(
        df_options,
        call_or_put="C",
        strike_col="moneyness",
        strike_target=1.0,
        day_to_expiry_target=day_to_expiry_target,
    )[["date", "ticker", "implied_volatility"]].rename(columns={"implied_volatility": "iv_call"})

    df_put = select_options(
        df_options,
        call_or_put="P",
        strike_col="moneyness",
        strike_target=1.0,
        day_to_expiry_target=day_to_expiry_target,
    )[["date", "ticker", "implied_volatility"]].rename(columns={"implied_volatility": "iv_put"})

    df_iv = df_call.merge(df_put, on=["date", "ticker"], how="outer")
    df_iv["sigma_iv"] = df_iv[["iv_call", "iv_put"]].mean(axis=1)
    return df_iv[["date", "ticker", "sigma_iv"]].dropna().sort_values(["ticker", "date"])


def build_market_dataset(
    start_date: datetime,
    end_date: datetime,
    ticker: str,
    day_to_expiry_target: int = 30,
) -> pd.DataFrame:
    """Assemble les donnees necessaires a l'estimation Heston-UKF.

    Sortie:
    - date, ticker
    - spot, log_return
    - sigma_iv (proxy implicite ATM maturite cible)
    """
    df_options = OptionLoader.load_data(
        start_date,
        end_date,
        process_kwargs={"ticker": ticker},
    )
    df_spot = extract_spot_from_options(df_options)
    df_spot["ticker"] = ticker
    df_spot = df_spot.sort_values("date")
    ratio = (df_spot["spot"] / df_spot["spot"].shift(1)).astype(float)
    ratio = ratio.where(ratio > 0)
    df_spot["log_return"] = np.log(ratio)

    df_iv = _build_iv_proxy(df_options=df_options, day_to_expiry_target=day_to_expiry_target)

    df = df_spot.merge(df_iv, on=["date", "ticker"], how="inner")
    return df.dropna(subset=["log_return", "sigma_iv"]).reset_index(drop=True)
