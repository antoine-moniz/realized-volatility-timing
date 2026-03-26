from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import HestonParams, UKFConfig


@dataclass
class UKFState:
    variance: float
    covariance: float


class HestonUKF:
    """UKF scalaire pour la variance latente v_t dans une discretisation Heston.

    Etat: v_t (variance annualisee)
    Observation: r_t (log-return journalier)
    """

    def __init__(self, params: HestonParams, cfg: UKFConfig):
        self.params = params
        self.cfg = cfg
        self.n = 1

        lam = (cfg.alpha**2) * (self.n + cfg.kappa_sigma) - self.n
        self.lambda_ = lam
        scale = self.n + lam

        self.wm = np.array([lam / scale, 1.0 / (2.0 * scale), 1.0 / (2.0 * scale)])
        self.wc = np.array(
            [lam / scale + (1.0 - cfg.alpha**2 + cfg.beta), 1.0 / (2.0 * scale), 1.0 / (2.0 * scale)]
        )

    def _ensure_positive(self, x: float) -> float:
        return float(max(x, self.cfg.var_floor))

    def _sigma_points(self, mean: float, cov: float) -> np.ndarray:
        cov = max(cov, self.cfg.cov_floor)
        spread = math.sqrt((self.n + self.lambda_) * cov)
        return np.array([mean, mean + spread, mean - spread], dtype=float)

    def _state_transition(self, v: np.ndarray, eps_prev: float) -> np.ndarray:
        p = self.params
        dt = self.cfg.dt
        v_pos = np.maximum(v, self.cfg.var_floor)

        drift = v_pos + p.kappa * (p.theta - v_pos) * dt
        leverage = p.xi * np.sqrt(v_pos * dt) * p.rho * eps_prev
        v_next = drift + leverage
        return np.maximum(v_next, self.cfg.var_floor)

    def _measurement(self, v: np.ndarray) -> np.ndarray:
        dt = self.cfg.dt
        return (self.params.mu - 0.5 * np.maximum(v, self.cfg.var_floor)) * dt

    def _process_variance(self, v_mean: float) -> float:
        dt = self.cfg.dt
        p = self.params
        v_pos = max(v_mean, self.cfg.var_floor)
        orthogonal = max(1.0 - p.rho**2, 0.0)
        return max((p.xi**2) * v_pos * dt * orthogonal, self.cfg.cov_floor)

    def _observation_variance(self, v_mean: float) -> float:
        dt = self.cfg.dt
        return max(v_mean * dt, self.cfg.obs_var_floor)

    def step(self, state: UKFState, observation: float, eps_prev: float) -> tuple[UKFState, float, float]:
        """Execute une etape predict/update et renvoie (state, innovation, loglik)."""
        sigma = self._sigma_points(state.variance, state.covariance)

        sigma_pred = self._state_transition(sigma, eps_prev=eps_prev)
        v_pred = float(np.dot(self.wm, sigma_pred))
        q_t = self._process_variance(v_pred)
        p_pred = float(np.dot(self.wc, (sigma_pred - v_pred) ** 2) + q_t)

        y_sigma = self._measurement(sigma_pred)
        y_pred = float(np.dot(self.wm, y_sigma))
        r_t = self._observation_variance(v_pred)

        p_yy = float(np.dot(self.wc, (y_sigma - y_pred) ** 2) + r_t)
        p_xy = float(np.dot(self.wc, (sigma_pred - v_pred) * (y_sigma - y_pred)))

        if p_yy <= 0:
            p_yy = self.cfg.obs_var_floor

        innovation = float(observation - y_pred)
        k_gain = p_xy / p_yy

        v_upd = self._ensure_positive(v_pred + k_gain * innovation)
        p_upd = max(p_pred - k_gain * p_yy * k_gain, self.cfg.cov_floor)

        loglik = -0.5 * (math.log(2.0 * math.pi) + math.log(p_yy) + (innovation**2) / p_yy)

        return UKFState(v_upd, p_upd), innovation, float(loglik)
