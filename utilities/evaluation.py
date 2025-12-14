"""Evaluation metrics for forecasting models."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the root mean square error (RMSE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        rmse: Root mean square error
    """
    return np.sqrt(np.mean((y_pred - y_true)**2))


def relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate relative root mean square error.

    RRMSE = sqrt(mean((y_pred - y_true)^2 / y_true^2))

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        rrmse: Relative root mean square error
    """
    err2 = (y_pred - y_true)**2
    denom = y_true**2

    # Protect against divide-by-zero errors
    inds = np.where(np.isclose(denom, 0))
    denom = np.delete(denom, inds)
    err2 = np.delete(err2, inds)

    return np.sqrt(np.mean(err2 / denom))


def mae(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate mean absolute error (MAE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        mae: Mean absolute error
    """
    return np.mean(np.abs(y_pred - y_true))


def mape(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate mean absolute percentage error (MAPE).

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        mape: Mean absolute percentage error (in percent)
    """
    err = y_true - y_pred
    denom = y_true

    # Protect against divide-by-zero errors
    inds = np.where(np.isclose(denom, 0))
    denom = np.delete(denom, inds)
    err = np.delete(err, inds)

    return np.mean(np.abs(err / denom)) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    The correlation coefficient between a set of predictions and ground truths.

    Args:
        y_true: True values
        y_pred: Predicted values
    """
    sum_e = np.sum((y_true - y_pred)**2)
    sum_s = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - sum_e / sum_s
