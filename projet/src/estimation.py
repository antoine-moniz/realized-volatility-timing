from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import HestonParams, RollingMLEConfig, UKFConfig
from .ukf_heston import HestonUKF, UKFState


def _pack_params(p: HestonParams) -> np.ndarray:
    return np.array([p.kappa, p.theta, p.xi, p.rho, p.mu], dtype=float)


def _unpack_params(x: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(max(x[0], 1e-6)),
        theta=float(max(x[1], 1e-8)),
        xi=float(max(x[2], 1e-6)),
        rho=float(np.clip(x[3], -0.999, 0.999)),
        mu=float(x[4]),
    )


def _initial_params(cfg: RollingMLEConfig) -> HestonParams:
    return HestonParams(
        kappa=cfg.init_kappa,
        theta=cfg.init_theta,
        xi=cfg.init_xi,
        rho=cfg.init_rho,
        mu=cfg.init_mu,
    )


def _window_neg_loglik(returns: np.ndarray, p: HestonParams, ukf_cfg: UKFConfig) -> float:
    # Initialisation robuste sur variance empirique annualisee
    v0 = max(float(np.var(returns) * 252.0), ukf_cfg.var_floor)
    state = UKFState(variance=v0, covariance=max(v0 * 0.25, ukf_cfg.cov_floor))
    ukf = HestonUKF(params=p, cfg=ukf_cfg)

    total_ll = 0.0
    eps_prev = 0.0
    dt = ukf_cfg.dt

    for r_t in returns:
        state, innovation, ll = ukf.step(state=state, observation=float(r_t), eps_prev=eps_prev)
        total_ll += ll
        eps_prev = innovation / max(np.sqrt(max(state.variance * dt, ukf_cfg.obs_var_floor)), 1e-8)

    return float(-total_ll)


def _fit_window_params(
    returns_window: np.ndarray,
    start_guess: HestonParams,
    ukf_cfg: UKFConfig,
    mle_cfg: RollingMLEConfig,
) -> HestonParams:
    x0 = _pack_params(start_guess)

    bounds = [
        (1e-6, 25.0),
        (1e-8, 2.0),
        (1e-6, 5.0),
        (-0.999, 0.999),
        (-1.0, 1.0),
    ]

    def objective(x: np.ndarray) -> float:
        return _window_neg_loglik(returns_window, _unpack_params(x), ukf_cfg)

    res = minimize(
        objective,
        x0,
        method=mle_cfg.optimizer_method,
        bounds=bounds,
        options={"maxiter": mle_cfg.maxiter, "disp": False},
    )

    if not res.success:
        return start_guess

    return _unpack_params(res.x)


def run_rolling_heston_filter(
    df_market: pd.DataFrame,
    ukf_cfg: UKFConfig,
    mle_cfg: RollingMLEConfig,
) -> pd.DataFrame:
    """Estime la variance latente v_t via UKF avec recalibrage rolling MLE."""
    required_cols = {"date", "ticker", "log_return"}
    missing = required_cols.difference(df_market.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df_market: {missing}")

    df = df_market.sort_values("date").reset_index(drop=True).copy()
    returns = df["log_return"].to_numpy(dtype=float)

    if len(returns) < mle_cfg.min_obs:
        raise ValueError("Pas assez d'observations pour lancer l'estimation rolling.")

    current_params = _initial_params(mle_cfg)
    v0 = max(float(np.var(returns[: mle_cfg.min_obs]) * 252.0), ukf_cfg.var_floor)
    state = UKFState(variance=v0, covariance=max(v0 * 0.25, ukf_cfg.cov_floor))

    var_hat = np.full(len(df), np.nan, dtype=float)
    innovation_arr = np.full(len(df), np.nan, dtype=float)
    loglik_arr = np.full(len(df), np.nan, dtype=float)

    kappa_arr = np.full(len(df), np.nan, dtype=float)
    theta_arr = np.full(len(df), np.nan, dtype=float)
    xi_arr = np.full(len(df), np.nan, dtype=float)
    rho_arr = np.full(len(df), np.nan, dtype=float)
    mu_arr = np.full(len(df), np.nan, dtype=float)

    eps_prev = 0.0
    dt = ukf_cfg.dt

    for i in range(len(df)):
        if i >= mle_cfg.window_size and ((i - mle_cfg.window_size) % mle_cfg.recalibrate_every == 0):
            window = returns[i - mle_cfg.window_size : i]
            current_params = _fit_window_params(
                returns_window=window,
                start_guess=current_params,
                ukf_cfg=ukf_cfg,
                mle_cfg=mle_cfg,
            )

        ukf = HestonUKF(params=current_params, cfg=ukf_cfg)
        state, innovation, ll = ukf.step(state=state, observation=returns[i], eps_prev=eps_prev)

        var_hat[i] = state.variance
        innovation_arr[i] = innovation
        loglik_arr[i] = ll

        kappa_arr[i] = current_params.kappa
        theta_arr[i] = current_params.theta
        xi_arr[i] = current_params.xi
        rho_arr[i] = current_params.rho
        mu_arr[i] = current_params.mu

        eps_prev = innovation / max(np.sqrt(max(state.variance * dt, ukf_cfg.obs_var_floor)), 1e-8)

    out = df.copy()
    out["v_hat"] = var_hat
    out["sigma_hat"] = np.sqrt(np.maximum(out["v_hat"], ukf_cfg.var_floor))
    out["innovation"] = innovation_arr
    out["loglik"] = loglik_arr

    out["kappa_hat"] = kappa_arr
    out["theta_hat"] = theta_arr
    out["xi_hat"] = xi_arr
    out["rho_hat"] = rho_arr
    out["mu_hat"] = mu_arr

    return out
