from typing import Tuple

import numpy as np
import statsmodels.api as sm


def cond_sieve_bootstrap(
    residuals: np.ndarray,
    point_forecast: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0
) -> np.ndarray:
    """
    Conditional sieve bootstrapping on a point forecast.

    Reference: "Sieve bootstrap for time series", P. Bühlmann.
    
    The bootstrapping procedure goes as follows:
        1. Get residuals X_k from the fitted model
        2. Fit a sieve / autoregressive model AR(p) to the residuals X_k
        3. Centre the residuals e_k from the sieve
        4. Bootstrap (with resampling) the centered residuals
        5. Generate new model residuals using the bootstrapped sieve residuals
        6. Generate forecast paths using the bootstrapped model residuals

    Args:
        residuals (Array): Residuals from fitting the model to the training data
        point_forecast (Array): A point (mean) forecast to horizon from
        n_bootstrap (int): Number of paths generated
        seed (int): Seed for pseudo-random number generator

    Returns:
        bootstrap_paths: Matrix with shape (n_bootstrap, len(point_forecast))
    """
    # Convert to numpy arrays to avoid pandas index issues
    residuals = np.asarray(residuals)
    point_forecast = np.asarray(point_forecast)

    # Forecast horizon
    horizon = len(point_forecast)

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

    bootstrap_paths = np.zeros((n_bootstrap, horizon))

    # Get AR coefficients from sieve model
    ar_params = sieve_fitted.params[1:]  # Exclude intercept

    # Get last p_sieve residuals from training set to initialise bootstrap
    last_resid = residuals[-p_sieve:]

    # TODO: Use this for multiprocessing
    # def map(args):
    #     eps_centered, horizon, p_sieve, ar_params, point_forecast = args
    #     # Bootstrap (resample) centered innovations
    #     eps_boot = np.random.choice(eps_centered, size=horizon, replace=True)
    #     # Generate residual path
    #     X_boot = np.zeros(horizon + p_sieve)
    #     X_boot[:p_sieve] = last_resid  # Init with residuals from training
    #     for k in range(horizon):
    #         # AR process: X_{k+1} = sum(phi_j * X_{k+1-j}) + epsilon_{k+1}
    #         ar_contribution = np.sum(ar_params * X_boot[k : k + p_sieve][::-1])
    #         X_boot[k + p_sieve] = ar_contribution + eps_boot[k]
    #     # Generate forecast path
    #     bootstrap_path = point_forecast + X_boot[p_sieve:]
    #     return  bootstrap_path

    for b in range(n_bootstrap):
        # Bootstrap (resample) centered innovations
        eps_boot = np.random.choice(eps_centered, size=horizon, replace=True)

        # Generate residual path
        X_boot = np.zeros(horizon + p_sieve)
        X_boot[:p_sieve] = last_resid  # Init with residuals from training

        for k in range(horizon):
            # AR process: X_{k+1} = sum(phi_j * X_{k+1-j}) + epsilon_{k+1}
            ar_contribution = np.sum(ar_params * X_boot[k : k + p_sieve][::-1])
            X_boot[k + p_sieve] = ar_contribution + eps_boot[k]

        # Generate forecast path
        bootstrap_paths[b, :] = point_forecast + X_boot[p_sieve:]

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