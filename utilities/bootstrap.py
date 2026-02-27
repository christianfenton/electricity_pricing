from typing import Tuple

import numpy as np
import statsmodels.api as sm
from numba import njit, prange


def cond_sieve_bootstrap(
    residuals: np.ndarray,
    point_forecast: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0
) -> np.ndarray:
    """
    Conditional sieve bootstrapping on a point forecast.

    Args:
        residuals (Array): Residuals from fitting the model to the training data
        point_forecast (Array): A point (mean) forecast to horizon from
        n_bootstrap (int): Number of paths generated
        seed (int): Seed for pseudo-random number generator

    Returns:
        bootstrap_paths: Matrix with shape (n_bootstrap, len(point_forecast))
    
    The bootstrapping procedure goes as follows:
        1. Get residuals X_k from the fitted model
        2. Fit a sieve / autoregressive model AR(p) to the residuals X_k
        3. Centre the residuals e_k from the sieve
        4. Bootstrap (with resampling) the centered residuals
        5. Generate new model residuals using the bootstrapped sieve residuals
        6. Generate forecast paths using the bootstrapped model residuals

    Reference: "Sieve bootstrap for time series", P. Bühlmann.
    """
    # Set PRNG seed
    np.random.seed(seed)

    # Fit AR(p) sieve to residuals
    n = len(residuals)
    p_sieve = int((n / np.log(n)) ** (1/4)) + 1
    sieve_model = sm.tsa.AutoReg(residuals, lags=p_sieve, old_names=False)
    sieve_fitted = sieve_model.fit()

    # Get innovations (epsilon_k) from sieve and center them
    sieve_resid = sieve_fitted.resid
    eps_centered = sieve_resid - sieve_resid.mean()

    # Get AR coefficients from sieve model
    ar_params = sieve_fitted.params[1:]  # Exclude intercept

    # Get last p_sieve residuals from training set to initialise bootstrapping
    last_resid = residuals[-p_sieve:]

    return _bootstrap_paths_numba(
        np.asarray(eps_centered), np.asarray(point_forecast),
        np.asarray(ar_params), np.asarray(last_resid),
        p_sieve, n_bootstrap, seed
    )


@njit(parallel=True, cache=True)
def _bootstrap_paths_numba(
    eps_centered: np.ndarray,
    point_forecast: np.ndarray,
    ar_params: np.ndarray,
    last_resid: np.ndarray,
    p_sieve: int,
    n_bootstrap: int,
    seed: int
):
    horizon = len(point_forecast)
    n_eps = len(eps_centered)
    bootstrap_paths = np.zeros((n_bootstrap, horizon))

    for b in prange(n_bootstrap):
        np.random.seed(seed + b)  # Each thread gets a deterministic seed

        # Resample the centered innovations
        eps_boot = np.empty(horizon)
        for i in range(horizon):
            eps_boot[i] = eps_centered[np.random.randint(0, n_eps)]

        # Generate residual path via AR process
        # AR process: X_{k+1} = sum(phi_j * X_{k+1-j}) + epsilon_{k+1}
        X_boot = np.zeros(p_sieve + horizon)
        X_boot[:p_sieve] = last_resid[:p_sieve]
        for k in range(horizon):
            ar_contribution = 0.0
            for j in range(p_sieve):
                ar_contribution += ar_params[j] * X_boot[p_sieve + k - 1 - j]
            X_boot[p_sieve + k] = ar_contribution + eps_boot[k]

        # Generate forecast path
        for i in range(horizon):
            bootstrap_paths[b, i] = point_forecast[i] + X_boot[p_sieve + i]

    return bootstrap_paths


def get_bootstrap_percentiles(
    bootstrap_paths: np.ndarray,
    lower: float = 2.5,
    upper: float = 97.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute confidence intervals from Monte Carlo simulations."""
    ci_lower = np.percentile(bootstrap_paths, lower, axis=0)
    ci_upper = np.percentile(bootstrap_paths, upper, axis=0)
    return ci_lower, ci_upper