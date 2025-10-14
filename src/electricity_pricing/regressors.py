"""Regression methods for model fitting."""

import numpy as np


class LinearRegression():
    """
    Ordinary Least Squares (OLS) linear regression.

    Fits a linear model by minimizing the squared residuals:
        minimize ||y - Xθ||²

    The solution is obtained via the normal equations:
        θ = (X^T X)^{-1} X^T y

    Attributes:
        params: Fitted parameters (coefficients) of shape (n_features,)

    Example:
        >>> regressor = LinearRegression()
        >>> regressor.fit(X_train, y_train)
        >>> predictions = regressor.predict(X_test)
        >>> coefficients = regressor.get_params()
    """

    def __init__(self):
        """Initialize the linear regression model."""
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear regression model using ordinary least squares.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            self: The fitted model
        """
        self.params = np.linalg.solve(X.T @ X,  X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the fitted model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            predictions: Predicted values of shape (n_samples,)
        """
        return np.einsum('i,ji->j', self.params, X)

    def get_params(self) -> np.ndarray:
        """
        Get the fitted parameters.

        Returns:
            params: Coefficient array of shape (n_features,)
        """
        return self.params
