"""
Rolling day-ahead forecasts with optionally bootstrapped confidence intervals.
"""

from typing import Protocol, List, Tuple, Optional

import numpy as np
import pandas as pd

from .bootstrap import cond_sieve_bootstrap
from .datatools import periods_remaining, expected_periods
from .evaluation import rmse, mae


class ForecastModel(Protocol):
    """Protocol defining the interface for forecast models."""
    
    @property
    def resid(self) -> np.ndarray:
        """Model residuals from fitting."""
        ...
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate point forecast for given number of steps."""
        ...
    
    def apply(
        self,
        endog: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        refit: bool = True,
        copy_initialization: bool = True,
        **kwargs
    ) -> "ForecastModel":
        """Apply model to new data, returning updated fitted model."""
        ...


def _find_issue_index(
    df: pd.DataFrame,
    issue_date: pd.Timestamp,
    issue_period: int,
    date_col: str,
    period_col: str
) -> int:
    """Find the first index in df corresponding to the issue date/period."""
    mask = (df[date_col] == issue_date) & (df[period_col] == issue_period)
    if not mask.any():
        raise ValueError(f"No data for {issue_date} period {issue_period}")
    return df[mask].index[0]


def _get_forecast_horizon(
    df: pd.DataFrame,
    issue_date: pd.Timestamp,
    issue_period: int,
    date_col: str,
    period_col: str
) -> Tuple[int, int, pd.Timestamp, int]:
    """
    Calculate indices to forecast to end of day-ahead from issue date/period.

    Args:
        df: DataFrame with sequential integer index
        issue_date: Date when forecast is issued
        issue_period: Period when forecast is issued
        cols: Column name specification

    Returns:
        (start_idx, end_idx, target_date, n_periods_tomorrow)
    """
    issue_idx = _find_issue_index(
        df, issue_date, issue_period, date_col, period_col
    )

    periods_today = periods_remaining(issue_date, issue_period)
    target_date = issue_date + pd.Timedelta(days=1)
    periods_tomorrow = expected_periods(target_date)
    horizon = periods_today + periods_tomorrow

    end_idx = issue_idx + horizon

    if end_idx > len(df):
        raise ValueError(
            f"""Insufficient future data: need {horizon} periods, 
            have {len(df) - issue_idx}"""
        )

    return issue_idx, end_idx, target_date, periods_tomorrow


def forecast_dayahead(
    fitted_model: ForecastModel,
    df: pd.DataFrame,
    issue_date: pd.Timestamp,
    issue_period: int,
    *,
    endog_col: str = "ELECTRICITY_PRICE",
    date_col: str = "SETTLEMENT_DATE",
    period_col: str = "SETTLEMENT_PERIOD",
    exog_cols: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a day-ahead forecast.

    Args:
        fitted_model: Pre-fitted model satisfying ForecastModel protocol
        df: DataFrame with all required columns
        issue_date: Date when forecast is issued
        issue_period: Period when forecast is issued (e.g., 18 for 09:00)

    Kwargs:
        endog_col: Column containing endogenous variable
            Default: "ELECTRICITY_PRICE"
        date_col: Date column (default: "SETTLEMENT_DATE")
        period_col: Period column (default: "SETTLEMENT_PERIOD")
        exog_cols: Columns containing exogenous variables (default: None)
        percentiles: If provided, compute bootstrapped confidence intervals
            at these percentiles
        n_bootstrap: Number of bootstrapped samples (default: 500)
            Ignored if percentiles are not provided
        seed: Random seed for bootstrapping

    Returns:
        DataFrame with columns:
            - date_col
            - period_col
            - "forecast"
            - "actual"
            - "issue_date"
            - "issue_period"
            - Percentile columns p{X}, e.g. "p2.5" for the 2.5th percentile
    """
    fcast_start, fcast_end, target_date, n_tomorrow = _get_forecast_horizon(
        df, issue_date, issue_period, date_col, period_col
    )

    # Point forecast
    if exog_cols is not None and len(exog_cols) > 0:
        forecast_exog = df.iloc[fcast_start:fcast_end][exog_cols]
        steps_full = len(forecast_exog)
        full_forecast = fitted_model.forecast(steps=steps_full, exog=forecast_exog)
    else:
        steps_full = fcast_end - fcast_start
        full_forecast = fitted_model.forecast(steps=steps_full, exog=None)
    tomorrow_forecast = full_forecast[-n_tomorrow:]

    # Get actual values for target date
    tomorrow_df = df[df[date_col] == target_date]
    actuals = tomorrow_df[endog_col].values
    periods = tomorrow_df[period_col].values

    # Build result DataFrame
    result = pd.DataFrame({
        date_col: target_date,
        period_col: periods,
        "forecast": tomorrow_forecast,
        "actual": actuals,
        "issue_date": issue_date,
        "issue_period": issue_period,
    })

    # Bootstrap confidence intervals if requested
    if percentiles is not None:
        bootstrap_paths = cond_sieve_bootstrap(
            residuals=fitted_model.resid,
            point_forecast=full_forecast,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        bootstrap_tomorrow = bootstrap_paths[:, -n_tomorrow:]

        for p in percentiles:
            result[f"p{p}"] = np.percentile(bootstrap_tomorrow, p, axis=0)

    return result


def rolling_dayahead_forecast(
    fitted_model: ForecastModel,
    df: pd.DataFrame,
    issue_dates: pd.DatetimeIndex,
    issue_period: int,
    n_train_samples: int,
    *,
    endog_col: str = "ELECTRICITY_PRICE",
    date_col: str = "SETTLEMENT_DATE",
    period_col: str = "SETTLEMENT_PERIOD",
    exog_cols_fit: Optional[List[str]] = None,
    exog_cols_predict: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None,
    n_bootstrap: int = 500,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform rolling day-ahead forecasts.

    Args:
        fitted_model: Pre-fitted model satisfying ForecastModel protocol
        df: DataFrame with dates, periods, target, and feature columns
        issue_dates: Dates on which day-ahead forecasts are issued
        issue_period: Period at which forecast is issued (e.g., 18 for 09:00)
        n_train_samples: Number of training samples for rolling window

    Kwargs:
        endog_col: Column containing endogenous variable
            Default: "ELECTRICITY_PRICE"
        date_col: Date column (default: "SETTLEMENT_DATE")
        period_col: Period column (default: "SETTLEMENT_PERIOD")
        exog_cols_fit: Columns containing exogenous variables for fitting
            Default: None
        exog_cols_predict: Columns containing exogenous variables for prediction
            Default: None
        percentiles: Compute bootstrap CIs at these percentiles (default: None)
        n_bootstrap: Number of bootstrap samples (default: 500)
        seed: Random seed incremented by 1 for each forecast date (default: 0)
        verbose: Whether to print progress (default: True)

    Returns:
        DataFrame with columns: date_col, period_col, forecast, actual,
        issue_date, plus p{X} columns for each requested percentile
    """
    results = []
    current_model = fitted_model

    for i, issue_date in enumerate(issue_dates):
        if verbose:
            print(f"[{i + 1}/{len(issue_dates)}] Forecasting from {issue_date}")

        df_dayahead = forecast_dayahead(
            fitted_model=current_model,
            df=df,
            issue_date=issue_date,
            issue_period=issue_period,
            exog_cols=exog_cols_predict,
            date_col=date_col,
            period_col=period_col,
            endog_col=endog_col,
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

        # Update model for next iteration
        if i < len(issue_dates) - 1:

            issue_idx = _find_issue_index(
                df,
                issue_dates[i + 1], issue_period,
                date_col, period_col
            )
            train_start = issue_idx - n_train_samples
            train_end = issue_idx

            train_df = df.iloc[train_start:train_end]

            # Handle exogenous variables
            if exog_cols_fit is not None and len(exog_cols_fit) > 0:
                train_exog = train_df[exog_cols_fit]
            else:
                train_exog = None

            current_model = current_model.apply(
                endog=train_df[endog_col],
                exog=train_exog,
                refit=True,
                copy_initialization=True,
            )

    return pd.concat(results, ignore_index=True)