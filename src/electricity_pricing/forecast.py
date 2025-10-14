import numpy as np
import pandas as pd
from .models.base import ForecastModel


def forecast_day_ahead(model: ForecastModel, endog: pd.Series, exog: pd.Series, skip_days: int = 2):
    """
    Generate day-ahead forecasts.
    
    Forecasting setup:
    - Forecasts for day T are made at 09:00 on day T-1
    - At 09:00 on T-1, we have actuals up to 09:00 on T-1
    - We generate an intraday forecast for the remainder of T-1
    - We use history from T-2 (full day) + T-1 (actuals + forecast) to forecast T
    
    Args:
        model: Trained forecasting model with predict() method
        endog: Endogenous variable (pandas.Series with datetime index)
        exog: Exogenous variables (pandas.DataFrame with datetime index)
        skip_days: Number of initial days to skip to get the required variable history. Default: 2
        
    Returns:
        Tuple of (forecast, actuals) as pandas.Series with datetime indices (only forecasted days)
    """
    endog = endog.sort_index()
    exog = exog.sort_index()
    dates = exog.index.date
    unique_dates = pd.unique(dates)
    
    forecast_list = []
    actuals_list = []
    forecast_indices = []
    
    for T in range(skip_days, len(unique_dates)):
        date = unique_dates[T]
        
        # indices for day T
        day_mask = dates == date
        day_indices = np.where(day_mask)[0]
        settlement_periods = len(day_indices)  # 48 periods except on DST switches
        
        # Collect history from day T-2 (full day) and day T-1 (before 09:00)
        # Day T-2: get all data
        day_T2_mask = dates == unique_dates[T-2]
        day_T2_indices = np.where(day_T2_mask)[0]
        actuals_T2 = endog.iloc[day_T2_indices].values
        
        # Day T-1: get data before 09:00
        day_T1_mask = dates == unique_dates[T-1]
        day_T1_indices = np.where(day_T1_mask)[0]
        day_T1_times = exog.index[day_T1_indices]
        cutoff_mask = day_T1_times.hour < 9
        indices_before_0900 = day_T1_indices[cutoff_mask]
        indices_from_0900 = day_T1_indices[~cutoff_mask]
        
        actuals_before_0900 = endog.iloc[indices_before_0900].values
        exog_intraday = exog.iloc[indices_from_0900].values
        
        # Combine T-2 and T-1 (before 09:00) for history
        history_for_intraday = np.concatenate([actuals_T2, actuals_before_0900])
        
        # Generate intraday forecast for remainder of day T-1
        intraday_forecast = model.predict(history_for_intraday, exog_intraday, n_steps=len(indices_from_0900))
        
        # Combine all history: T-2 (full) + T-1 actual (before 09:00) + T-1 forecast (from 09:00)
        endog_history = np.concatenate([actuals_T2, actuals_before_0900, intraday_forecast])
        
        # make forecast for day T
        exog_future = exog.iloc[day_indices].values  # forecasted exogenous variables
        forecast_list.append(model.predict(endog_history, exog_future, n_steps=settlement_periods))
        actuals_list.append(endog.iloc[day_indices].values)
        forecast_indices.extend(endog.index[day_indices])
    
    # concatenate forecasts and actuals for all days
    forecast_array = np.concatenate(forecast_list)
    actuals_array = np.concatenate(actuals_list)

    # Create Series with datetime indices (only for forecasted days, no NaN padding)
    forecast_series = pd.Series(data=forecast_array, index=forecast_indices)
    actuals_series = pd.Series(data=actuals_array, index=forecast_indices)

    return forecast_series, actuals_series