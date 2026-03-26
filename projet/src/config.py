from dataclasses import dataclass


@dataclass(frozen=True)
class HestonParams:
    """Parametres du modele de Heston (version journaliere discretisee)."""

    kappa: float
    theta: float
    xi: float
    rho: float
    mu: float


@dataclass(frozen=True)
class UKFConfig:
    """Configuration numerique de l'Unscented Kalman Filter."""

    dt: float = 1.0 / 252.0
    alpha: float = 1e-2
    beta: float = 2.0
    kappa_sigma: float = 0.0
    var_floor: float = 1e-8
    cov_floor: float = 1e-10
    obs_var_floor: float = 1e-8


@dataclass(frozen=True)
class RollingMLEConfig:
    """Configuration de l'optimisation en fenetre glissante."""

    window_size: int = 126
    recalibrate_every: int = 5
    maxiter: int = 200
    min_obs: int = 80
    optimizer_method: str = "L-BFGS-B"

    init_kappa: float = 2.0
    init_theta: float = 0.04
    init_xi: float = 0.35
    init_rho: float = -0.5
    init_mu: float = 0.0
