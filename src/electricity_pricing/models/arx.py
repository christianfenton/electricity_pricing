"""Autoregressive model with exogenous variables (ARX)."""

import numpy as np
from collections import deque
from .base import ForecastModel
from ..regressors import LinearRegression
from typing import List


class ARXModel(ForecastModel):
    """
    Autoregressive model with exogenous variables (ARX).

    Attributes:
        lags: List of autoregressive lag indices (e.g., [1] for an order-1 model)
        regressor: Underlying regression model (default: LinearRegression)

    Example:
        >>> from electricity_pricing.models import ARXModel
        >>> from electricity_pricing.regressors import LinearRegression
        >>>
        >>> # Create model with 3 autoregressive lags
        >>> model = ARXModel(lags=[1, 2, 3], regressor=LinearRegression())
        >>>
        >>> # Fit on endogenous variable history and exogenous features
        >>> model.fit(endog, exog)
        >>>
        >>> # Forecasting:
        >>> predictions = model.predict(
        >>>     endog_history,  # endogenous variable history used to initialise AR terms
        >>>     exog_future,    # exogenous features for forecast horizon
        >>>     n_steps=48      # number of forecast steps
        >>> )
        >>>
        >>> # Non-standard use cases:
        >>> model.fit_raw(features, targets)  # features includes autoregressive and exogenous features
        >>> predictions = model.predict_raw(features_test)
    """

    def __init__(self, lags: List[int] = [1], regressor=LinearRegression()):
        """
        Create an `ARXModel` instance.

        Args:
            lag: List of autoregressive lag indices. Default: `[1]`.
            regressor: Regression model to use. Must implement fit(X, y) and 
                predict(X) methods. Default: LinearRegression().
        """
        self.lags = lags
        self.regressor = regressor
        self._fitted = False  # indicate whether parameters have been fitted yet

    def _build_ar_features(self, endog: np.ndarray):
        """
        Build autoregressive features from the endogenous variable history.

        Args:
            endog: Endogenous variables. Array-like with shape (n_samples,).

        Returns:
            ar: Autoregressive terms. Shape: (n_samples - max(lags), len(lags))
        """
        n_samples = len(endog)
        max_lag = max(self.lags)

        if n_samples <= max_lag:
            raise ValueError(
                f"Endogenous variable must have more than max(lag)={max_lag} samples, got {n_samples}."
            )

        # Collect autoregressive terms
        ar_terms = np.zeros((n_samples - max_lag, len(self.lags)))
        for i, lag in enumerate(self.lags):
            ar_terms[:, i] = endog[max_lag - lag : n_samples - lag]

        return ar_terms

    def fit(self, endog: np.ndarray, exog: np.ndarray | None = None):
        """
        Fit the model parameters to observed data.

        Args:
            endog: Endogenous variable history. Array-like with shape (n_samples,).
            exog: Exogenous variables. Array-like with shape (n_samples, n_features). Default: None.
        """
        assert endog.ndim == 1, "Expected `endog` to be a vector."

        n_samples = len(endog)
        max_lag = max(self.lags)

        if exog is not None:
            if exog.ndim == 1:
                exog = exog.reshape((-1, 1))
            assert exog.shape[0] == n_samples, "Dimensions in `endog` and `exog` do not match."

        # create target vector
        targets = endog[max_lag:]  # skip forward so every target has correct number of AR terms

        # Build feature matrix
        features = self._build_ar_features(endog)  # autoregressive (AR) features
        if exog is not None:
            # add exogenous terms to feature matrix
            exog_skipforward = exog[max_lag:]
            features = np.column_stack([features, exog_skipforward])

        # fit the regressor
        self.regressor.fit(features, targets)
        self._fitted = True

        return self
    
    def fit_raw(self, features: np.ndarray, target: np.ndarray):
        """
        Fit the model on observed data with a manually constructed feature 
        matrix containing autoregressive and exogenous features.

        Use this for full control over feature construction.

        Args:
            features: Array-like with shape (n_samples, n_features)
            target: Target vector with shape (n_samples,)

        Example:
            >>> # Build features manually
            >>> features = np.column_stack([lag1, lag2, exog1, exog2])
            >>> model.fit_raw(features, target)
        """
        self.regressor.fit(features, target)
        self._fitted = True
        return self

    def predict(self, endog_history: np.ndarray, exog_future: np.ndarray | None = None, n_steps: int = 1):
        """
        Generate future predictions for the endogenous variable from historical observations and exogenous features.

        Args:
            endog_history: Endogenous variable history. 
                Should contain at least max(lags) values. 
                Array-like with shape (n_history,).
            exog_future: Exogenous features for the forecast horizon. 
                Array-like with shape (n_steps, n_features_exog).
                Default: None.
            n_steps: Number of steps to forecast. Default: 1.

        Returns:
            predictions: Forecasted prices. Shape: (n_steps,).
        """
        if not self._fitted:
            raise ValueError("Model parameters have not been fitted yet.")
        
        assert endog_history.ndim == 1, "Expected `endog` to be a vector."

        if exog_future is not None:
            if exog_future.ndim == 1:
                exog_future = exog_future.reshape(-1, 1)

        max_lag = max(self.lags)

        if len(endog_history) < max_lag:
            raise ValueError(
                f"Endogenous variable history must contain at least max(lags)={max_lag} values, got {len(endog_history)}."
            )

        if exog_future is not None and exog_future.shape[0] < n_steps:
            raise ValueError(
                f"exog_future must have {n_steps} rows, got {exog_future.shape[0]}"
            )

        # initialise buffer with recent history
        endog_buffer = deque(endog_history, maxlen=max_lag)  # deque for fast append/pop operations

        predictions = np.zeros(n_steps)

        for i in range(n_steps):
            # get autoregressive terms from buffer
            ar_terms = np.array([endog_buffer[-lag] for lag in self.lags])

            # combine with exogenous features
            if exog_future is not None:
                features = np.concatenate([ar_terms, exog_future[i, :]])
            else:
                features = ar_terms

            # make prediction
            pred = self.regressor.predict(features.reshape(1, -1))[0]
            predictions[i] = pred

            # update buffer
            endog_buffer.append(pred)

        return predictions
    
    def predict_raw(self, endog_history: np.ndarray, features_future: np.ndarray):
        raise Exception("This method has not been implemented yet.")

    def get_params(self):
        """
        Get model parameters from the underlying regressor.

        Format depends on the regressor used.
        """
        if hasattr(self.regressor, 'coef_'):
            return self.regressor.coef_  # sklearn-style regressor
        elif hasattr(self.regressor, 'get_params'):
            return self.regressor.get_params()
        else:
            return {}

    def __repr__(self):
        return f"ARXModel(lags={self.lags}, regressor={self.regressor})"
