"""Base interface for forecasting models."""

from abc import ABC, abstractmethod
import numpy as np


class ForecastModel(ABC):
    """
    Thin wrapper providing consistent interface for forecasting models.

    This class provides a minimal abstraction to wrap models from different
    libraries (scikit-learn, statsmodels, custom implementations) with a
    consistent interface.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForecastModel":
        """
        Fit model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            self: The fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            predictions: Predicted values of shape (n_samples,)
        """
        pass
