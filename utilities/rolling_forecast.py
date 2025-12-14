"""Rolling forecast utilities for day-ahead electricity price forecasting."""

from typing import Optional

import pandas as pd

from .datatools import (
    periods_in_date_range, periods_remaining, expected_periods
)


def rolling_day_ahead_forecast(
    df: pd.DataFrame,
    model,
    endog_col: str,
    exog_cols: Optional[list[str]],
    forecast_days: int,
    window_days: int = 90,
    issue_period: int = 18,  # Period 18 = 09:00 AM
) -> pd.DataFrame:
    """
    Perform a rolling day-ahead forecast using settlement periods.

    For each forecast day:
    1. Fits the model on a rolling window of historical data
    2. Generates a forecast for the remainder of today + all of tomorrow
    3. Extracts and stores tomorrow's forecast
    4. Advances to the next day

    Args:
        df: DataFrame with SETTLEMENT_DATE, SETTLEMENT_PERIOD columns and data
            Should have a sequential integer index (0, 1, 2, ...)
        model: Model instance with fit() and forecast() methods
            - fit(endog, exog=None) should fit the model
            - forecast(steps, exog=None) should return forecast array
        endog_col: Column name for endogenous variable (e.g., 'ELECTRICITY_PRICE')
        exog_cols: List of column names for exogenous variables (or None)
        forecast_days: Number of days to forecast
        window_days: Number of past days to use for training (default: 90)
        issue_period: Settlement period at which forecast is issued (default: 18 = 09:00)

    Returns:
        DataFrame with columns: SETTLEMENT_DATE, SETTLEMENT_PERIOD, forecast
        Contains day-ahead forecasts for each requested day
    """
    # Validate inputs
    required_cols = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", endog_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    if exog_cols:
        for col in exog_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing exog column: {col}")

    # Ensure proper sorting
    df = df.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]).reset_index(
        drop=True
    )

    # Get date range
    start_date = df["SETTLEMENT_DATE"].min()
    end_date = df["SETTLEMENT_DATE"].max()

    # First forecast date is after window_days
    first_forecast_date = start_date + pd.Timedelta(days=window_days)

    # Generate forecast dates
    forecast_dates = pd.date_range(
        first_forecast_date,
        first_forecast_date + pd.Timedelta(days=forecast_days - 1),
        freq="D",
    )

    results = []

    for issue_date in forecast_dates:
        # Find the row index for the issue time (issue_date, issue_period)
        issue_mask = (df["SETTLEMENT_DATE"] == issue_date) & (
            df["SETTLEMENT_PERIOD"] == issue_period
        )

        if not issue_mask.any():
            print(
                f"""Warning: No data for settlement date {issue_date}
                and period {issue_period}, skipping"""
            )
            continue

        issue_idx = df[issue_mask].index[0]

        # Calculate training window: last window_days before issue time
        train_start_date = issue_date - pd.Timedelta(days=window_days)
        train_periods = periods_in_date_range(
            train_start_date.date(), issue_date.date()
        )

        # Subtract periods remaining in issue day (we only go up to issue_period)
        train_periods -= periods_remaining(issue_date.date(), issue_period)

        # Get training data
        train_start_idx = issue_idx - train_periods
        if train_start_idx < 0:
            print(f"Warning: Not enough history for {issue_date}, skipping")
            continue

        train_df = df.iloc[train_start_idx:issue_idx]

        endog_train = train_df[endog_col].values
        exog_train = train_df[exog_cols].values if exog_cols else None

        # Fit model
        model.fit(endog_train, exog=exog_train)

        # Calculate forecast horizon
        periods_remaining_today = periods_remaining(
            issue_date.date(), issue_period
        )
        target_date = issue_date + pd.Timedelta(days=1)
        periods_tomorrow = expected_periods(target_date.date()) # type: ignore[attr-defined]
        horizon = periods_remaining_today + periods_tomorrow

        # Get future exogenous variables
        forecast_end_idx = issue_idx + horizon
        if forecast_end_idx > len(df):
            print(
                f"Warning: Not enough future data for {issue_date}, skipping"
            )
            continue

        future_df = df.iloc[issue_idx:forecast_end_idx]
        exog_future = future_df[exog_cols].values if exog_cols else None

        # Generate forecast
        forecast = model.forecast(steps=horizon, exog=exog_future)

        # Extract tomorrow's forecast (last periods_tomorrow values)
        day_ahead_forecast = forecast[-periods_tomorrow:]

        # Get tomorrow's settlement dates and periods
        tomorrow_df = df[df["SETTLEMENT_DATE"] == target_date][
            ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]
        ].copy()

        if len(tomorrow_df) != periods_tomorrow:
            print(
                f"""Warning: Period count mismatch for {target_date}, 
                expected {periods_tomorrow}, got {len(tomorrow_df)}"""
            )
            continue

        tomorrow_df["forecast"] = day_ahead_forecast
        results.append(tomorrow_df)

    # Combine all forecasts
    if not results:
        raise ValueError(
            "No forecasts generated. Check date range and data availability."
        )

    forecast_df = pd.concat(results, ignore_index=True)

    return forecast_df
