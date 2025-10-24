"""Autoregressive model with exogenous variables (ARX)."""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from collections import deque
from .base import ForecastModel
from sklearn.linear_model import LinearRegression
from typing import List, Optional


class ARXModel(ForecastModel):
    """
    Autoregressive model with exogenous variables (ARX).

    Attributes:
        lags: List of autoregressive lag indices (e.g., [1] for an order-1 model)
        regressor: Underlying regression model (default: LinearRegression)
    """

    def __init__(self, lags: List[int] = [1], regressor=None):
        """
        Create an `ARXModel` instance.

        Args:
            lag: List of autoregressive lag indices. Default: `[1]`.
            regressor: Scikit-learn compatible regression model. Must implement
                fit(X, y) and predict(X) methods with coef_ and intercept_
                attributes. Default: LinearRegression(fit_intercept=True).
        """
        self.lags = lags
        self.regressor = regressor if regressor is not None else LinearRegression(fit_intercept=True)
        self._fitted = False  # indicate whether parameters have been fitted yet

    def _build_ar_features(self, endog: pd.Series) -> pd.DataFrame:
        """
        Build autoregressive features from the endogenous variable history.
        Returns a `pandas.DataFrame` with shape (n_samples - max(lags), len(lags)).
        """
        max_lag = max(self.lags)

        if len(endog) <= max_lag:
            raise ValueError(
                f"Endogenous variable must have more than max(lag)={max_lag} samples, got {len(endog)}."
            )

        # Shift columns for each lag
        ar_df = pd.DataFrame({f'lag_{lag}': endog.shift(lag) for lag in self.lags})

        return ar_df.dropna()

    def fit(self, endog: pd.Series, exog: Optional[pd.DataFrame] = None):
        """
        Fit model parameters to observed data.
        Returns the fitted model.

        Args:
            endog: Endogenous variable. `pandas.Series` with datetime index.
            exog: Exogenous variables.
                If provided, should be a `pandas.DataFrame` with datetime index.
                Default: None.
        """
        assert np.all(ptypes.is_datetime64_any_dtype(endog.index)), "endog must have datetime index."
        if exog is not None:
            assert np.all(ptypes.is_datetime64_any_dtype(exog.index)), "exog must have datetime index."

        max_lag = max(self.lags)

        if len(endog) <= max_lag:
            raise ValueError(
                f"Endogenous variable must have more than max(lag)={max_lag} samples, got {len(endog)}."
            )

        if exog is not None and len(exog) != len(endog):
            raise ValueError(
                f"Dimensions in endog ({len(endog)}) and exog ({len(exog)}) do not match."
            )

        # Store lag values from end of training data
        self.lag_values = endog.iloc[-np.array(self.lags)]

        # Build autoregressive features
        ar_features = self._build_ar_features(endog)

        # Create target vector (skip first max_lag rows to align with autoregressive features)
        targets = endog.iloc[max_lag:]

        # Combine AR features with exogenous variables
        if exog is not None:
            # Skip first max_lag rows of exog to align with AR features
            exog_aligned = exog.iloc[max_lag:].reset_index(drop=True)
            features = pd.concat([ar_features.reset_index(drop=True), exog_aligned], axis=1)
        else:
            features = ar_features

        self.regressor.fit(features.values, targets.values)
        self._fitted = True

        return self

    def forecast(
        self,
        steps: int,
        exog: Optional[pd.DataFrame] = None,
        endog_history: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Make a forecast.

        Args:
            steps: Number of steps to forecast ahead
            exog: Exogenous variables for forecast horizon.
                  If provided, should be a `pandas.DataFrame` with shape (steps, n_features).
                  Default: None.
            endog_history: Historical values of endogenous variable for AR lags used for forecasting.
                If None, uses last lag values stored during fit().
                If provided, should be a `pandas.Series` with shape (max(lag), ).
                Default: None.

        Returns:
            predictions: Forecasted values of shape (steps,)

        Example:
            # Forecast from end of training data
            model.fit(y_train, X_train)
            forecast = model.forecast(steps=48, exog=X_future)

            # Forecast from a different starting point
            forecast = model.forecast(steps=48, exog=X_future, endog_history=y_recent)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before calling forecast(). Call fit() first.")

        if exog is not None:
            assert np.all(ptypes.is_datetime64_any_dtype(exog.index)), "exog must have datetime index."
        if endog_history is not None:
            assert np.all(ptypes.is_datetime64_any_dtype(endog_history.index)), "endog_history must have datetime index."

        max_lag = max(self.lags)

        if endog_history is None:
            if not hasattr(self, 'lag_values'):
                raise ValueError(
                    "No lag values stored from fit(). Either fit the model or provide `endog_history`."
                )
            endog_buffer_init = self.lag_values.copy()
        else:
            if len(endog_history) < max_lag:
                raise ValueError(
                    f"endog_history must contain at least max(lags)={max_lag} values, "
                    f"got {len(endog_history)}."
                )
            endog_buffer_init = endog_history.iloc[-np.array(self.lags)].values

        exog_values = exog.values if exog is not None else None

        if exog_values is not None:
            if exog_values.ndim == 1:
                exog_values = exog_values.reshape(-1, 1)
            if exog_values.shape[0] < steps:
                raise ValueError(
                    f"exog must have at least {steps} rows, got {exog_values.shape[0]}"
                )

        # Make predictions
        endog_buffer = deque(endog_buffer_init, maxlen=max_lag)
        predictions = np.zeros(steps)
        for i in range(steps):
            ar_terms = np.array([endog_buffer[-lag] for lag in self.lags])

            if exog_values is not None:
                features = np.concatenate([ar_terms, exog_values[i, :]])
            else:
                features = ar_terms

            pred = self.regressor.predict(features.reshape(1, -1))[0]
            predictions[i] = pred
            endog_buffer.appendleft(pred)

        return predictions

    def get_params(self):
        """
        Get model parameters from the underlying regressor.

        Returns:
            coef_: Regression coefficients (numpy array)
            intercept_: Intercept term (float or numpy array)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before calling get_params(). Call fit() first.")
        return self.regressor.coef_, self.regressor.intercept_

    def __repr__(self):
        return f"ARXModel(lags={self.lags}, regressor={self.regressor})"
