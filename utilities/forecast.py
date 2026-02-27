"""Forecasting tools."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .bootstrap import cond_sieve_bootstrap
from .evaluation import rmse, mae


class Forecaster(ABC):
    """Base class for day-ahead forecasting."""

    def __init__(self, *, endog_col="ELECTRICITY_PRICE"):
        self.endog_col = endog_col

    @abstractmethod
    def _forecast_steps(self, steps, exog=None) -> np.ndarray:
        """Generate point forecast for `steps` ahead."""
        ...

    @abstractmethod
    def _fit(self, endog, exog=None) -> None:
        """Fit or refit the model on data."""
        ...

    @abstractmethod
    def _get_residuals(self) -> np.ndarray:
        """Return residuals from the most recent fit."""
        ...

    def forecast_dayahead(
        self,
        df_observed: pd.DataFrame,
        df_forecasts: pd.DataFrame,
        issue_time: pd.Timestamp,
        *,
        percentiles: list[float] | None = None,
        n_bootstrap: int = 500,
        seed: int = 0,
    ) -> pd.DataFrame:
        """
        Generate a day-ahead forecast.

        Args:
            df_observed: DataFrame with DatetimeIndex containing observed data
            df_forecasts: DataFrame with DatetimeIndex containing 
                forecasted exogenous variables for the target day
            issue_time: Time at which the forecast is issued

        Kwargs:
            percentiles: Percentiles for bootstrapped confidence intervals
            n_bootstrap: Number of bootstrapped runs
            seed: Random seed for resampling

        Returns:
            DataFrame with DatetimeIndex and columns:
                forecast, actual, issue_time, plus p{X} percentile columns
        """
        target_day = issue_time.normalize() + pd.Timedelta(days=1)
        target_end = target_day + pd.Timedelta(days=1)
        target_dates = pd.date_range(
            target_day, target_end, freq="30min", inclusive="left"
        )

        forecast_exog = df_forecasts.loc[target_dates]
        steps = len(forecast_exog)

        if len(forecast_exog) == 0:
            raise ValueError(
                f"No forecast data for target day {target_day.date()}"
            )

        point_forecast = self._forecast_steps(
            steps=steps,
            exog=forecast_exog if len(forecast_exog.columns) > 0 else None,
        )

        actuals = df_observed.loc[target_dates, self.endog_col].values

        result = pd.DataFrame(
            {
                "forecast": point_forecast,
                "actual": actuals,
                "issue_time": issue_time,
            },
            index=target_dates,
        )

        if percentiles is not None:
            bootstrap_paths = cond_sieve_bootstrap(
                residuals=self._get_residuals(),
                point_forecast=np.array(point_forecast),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            for p in percentiles:
                result[f"p{p}"] = np.percentile(bootstrap_paths, p, axis=0)

        return result

    def rolling_forecast_dayahead(
        self,
        df_observed: pd.DataFrame,
        df_forecasts: pd.DataFrame,
        issue_times: pd.DatetimeIndex,
        n_train_samples: int,
        *,
        percentiles: list[float] | None = None,
        n_bootstrap: int = 500,
        seed: int = 0,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Perform rolling day-ahead forecasts.

        Fits the model at the start of each iteration.

        Args:
            df_observed: DataFrame with DatetimeIndex containing observed data
            df_forecasts: DataFrame with DatetimeIndex containing forecasted 
                exogenous variables
            issue_times: Times at which each day-ahead forecast is issued
            n_train_samples: Number of training samples for the rolling window

        Kwargs:
            percentiles: Percentiles for bootstrapped confidence intervals
            n_bootstrap: Number of bootstrapped runs
            seed: Pseudo-random seed
            verbose: Print progress information

        Returns:
            DataFrame with DatetimeIndex and columns: forecast, actual,
            issue_time, plus p{X} columns for each requested percentile
        """
        results = []

        for i, issue_time in enumerate(issue_times):
            if verbose:
                print(
                    f"[{i + 1}/{len(issue_times)}] Forecasting from {issue_time}"
                )

            # Training window: n_train_samples periods ending before target day
            target_day = issue_time.normalize() + pd.Timedelta(days=1)
            train_end = target_day - pd.Timedelta(minutes=30)
            train_start = train_end - pd.Timedelta(
                minutes=30 * (n_train_samples - 1)
            )
            train_df = df_observed.loc[train_start:train_end]

            endog = train_df[self.endog_col]
            exog = train_df.drop(columns=[self.endog_col])
            self._fit(endog, exog if len(exog.columns) > 0 else None)

            # Forecast
            df_dayahead = self.forecast_dayahead(
                df_observed,
                df_forecasts,
                issue_time,
                percentiles=percentiles,
                n_bootstrap=n_bootstrap,
                seed=seed + i,
            )
            results.append(df_dayahead)

            if verbose:
                y_pred = np.asarray(df_dayahead["forecast"])
                y_true = np.asarray(df_dayahead["actual"])
                _rmse = rmse(y_pred, y_true)
                _mae = mae(y_pred, y_true)
                print(f"  RMSE: {_rmse:.2f}, MAE: {_mae:.2f}")

        return pd.concat(results)
