"""Base interface for forecasting models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional


class ForecastModel(ABC):
    """
    Base class for forecasting models.
    """

    @abstractmethod
    def fit(self, endog: pd.Series, exog: Optional[pd.DataFrame] = None) -> "ForecastModel":
        """
        Fit model parameters to observed data.

        Args:
            endog: Endogenous variable (target). `pandas.Series` with datetime index.
            exog: Exogenous variables (features). `pandas.DataFrame` with datetime index.
                  Default: None.

        Returns:
            self: The fitted model
        """
        pass

    @abstractmethod
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
                  `pandas.DataFrame` with shape (steps, n_features).
                  Default: None.
            endog_history: Historical values of endogenous variable used for autoregressive lags.
                          If None, uses lag values stored during fit().
                          If provided, forecasts from this history instead.
                          Array with shape (max_lag,) or longer.
                          Default: None.

        Returns:
            predictions: Forecasted values of shape (steps,)

        Example:
            # Forecast from end of training data
            model.fit(y_train, X_train)
            forecast = model.predict(steps=48, exog=X_future)

            # Forecast from a different starting point
            forecast = model.predict(steps=48, exog=X_future, endog_history=y_recent)
        """
        pass