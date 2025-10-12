"""Autoregressive model with exogenous variables (ARX)."""

import numpy as np
from .base import ForecastModel
from ..regressors import LinearRegression


class ARXModel(ForecastModel):
    """
    Autoregressive model with exogenous variables (ARX).

    Models electricity prices as a linear combination of:
    - Lagged prices (autoregressive terms)
    - Exogenous variables (demand, generation, weather, etc.)

    Attributes:
        regressor: Underlying regression model (default: LinearRegression)

    Example:
        >>> from electricity_pricing.models import ARXModel
        >>> # Linear ARX model (default)
        >>> model = ARXModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)

        >>> # ARX with Ridge regression
        >>> from sklearn.linear_model import Ridge
        >>> model = ARXModel(regressor=Ridge(alpha=1.0))
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, regressor=LinearRegression()):
        """
        Args:
            regressor: Regression model to use for fitting. Must implement
                fit(X, y) and predict(X) methods. If None, uses scikit-learn's
                LinearRegression with fit_intercept=True.
        """
        self.regressor = regressor

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ARXModel":
        """
        Fit the ARX model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
               Should contain lagged prices and exogenous variables
            y: Target vector of shape (n_samples,)
               Electricity prices to predict

        Returns:
            self: The fitted model
        """
        self.regressor.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate price predictions using the fitted ARX model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
               Should contain the same features used during training

        Returns:
            predictions: Predicted prices of shape (n_samples,)
        """
        return self.regressor.predict(X)

    def get_params(self) -> dict:
        """
        Get model parameters from the underlying regressor.

        Returns:
            params: Dictionary of model parameters. Contents depend on
                the regressor used.
        """
        if hasattr(self.regressor, 'get_params'):
            return self.regressor.get_params()
        else:
            return {}
