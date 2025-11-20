import numpy as np
import pandas as pd
from electricity_pricing.models import ARXModel
from sklearn.linear_model import LinearRegression


class TestARXModel:
    def test_fit_no_exog(self):
        """Fit to observed data without exogenous variables."""
        # Create some observed data with datetime index
        n_obs = 10
        dates = pd.date_range(
            "2024-01-01", periods=n_obs, freq="30min", tz="Europe/London"
        )
        endog_obs = pd.Series(np.arange(1, n_obs + 1), index=dates)

        # Create AR model
        lags = [1, 2]
        max_lag = np.max(lags)
        model = ARXModel(
            lags=lags, regressor=LinearRegression(fit_intercept=True)
        )
        model.fit(endog_obs, exog=None)
        coef, intercept = model.get_params()

        # Manually construct autoregressive terms and fit regressor
        features = pd.DataFrame(
            {f"lag_{lag}": endog_obs.shift(lag) for lag in lags}
        ).dropna()
        targets = endog_obs.iloc[max_lag:]
        regressor = LinearRegression(fit_intercept=True)
        regressor.fit(features.values, targets.values)
        coef_expected = regressor.coef_
        intercept_expected = regressor.intercept_

        np.testing.assert_allclose(coef, coef_expected, rtol=1e-10)
        np.testing.assert_allclose(intercept, intercept_expected, rtol=1e-10)

    def test_fit_with_exog(self):
        """Fit to observed data with exogenous variables."""
        n_obs = 10
        dates = pd.date_range(
            "2024-01-01", periods=n_obs, freq="30min", tz="Europe/London"
        )
        endog_obs = pd.Series(np.arange(1, n_obs + 1), index=dates)
        exog_obs = pd.DataFrame(
            {
                "feature1": np.random.default_rng(seed=1).uniform(size=n_obs),
                "feature2": np.random.default_rng(seed=2).uniform(size=n_obs),
            },
            index=dates,
        )

        # Create ARX model
        lags = [1, 2]
        max_lag = np.max(lags)
        model = ARXModel(
            lags=lags, regressor=LinearRegression(fit_intercept=True)
        )
        model.fit(endog_obs, exog=exog_obs)
        coef, intercep = model.get_params()

        # Manually construct features and fit regressor
        features = pd.DataFrame(
            {f"lag_{lag}": endog_obs.shift(lag) for lag in lags}
        ).dropna()
        targets = endog_obs.iloc[max_lag:]
        exog_aligned = exog_obs.iloc[max_lag:].reset_index(drop=True)
        features = pd.concat(
            [features.reset_index(drop=True), exog_aligned], axis=1
        )
        regressor = LinearRegression(fit_intercept=True)
        regressor.fit(features.values, targets.values)
        coef_expected = regressor.coef_
        intercept_expected = regressor.intercept_

        np.testing.assert_allclose(coef, coef_expected, rtol=1e-10)
        np.testing.assert_allclose(intercep, intercept_expected, rtol=1e-10)

    def test_forecast_no_exog(self):
        # Create some observed data with datetime index
        n_obs = 10
        dates = pd.date_range(
            "2024-01-01", periods=n_obs, freq="30min", tz="Europe/London"
        )
        endog_obs = pd.Series(np.arange(1.0, n_obs + 1.0), index=dates)

        # Create a model instance
        lags = [1, 2]
        model = ARXModel(
            lags=lags, regressor=LinearRegression(fit_intercept=True)
        )

        # Test 1: single-step forecast with no exogenous variables
        model.fit(endog_obs, exog=None)
        forecast = model.forecast(steps=1, exog=None)
        coef, intercept = model.get_params()

        # Manual calculation
        history = endog_obs.iloc[-np.array(lags)]
        forecast_expected = coef @ history + intercept
        np.testing.assert_allclose(forecast[0], forecast_expected, rtol=1e-10)

        # Test 2: multi-step forecast with no exogenous variables
        n_steps = 5
        forecast = model.forecast(steps=n_steps, exog=None)

        # Manual calculation of iterative forecast
        forecasts_expected = []
        buffer = history
        for i in range(n_steps):
            pred = coef @ buffer + intercept
            forecasts_expected.append(pred)
            # update buffer
            buffer.iloc[1:] = buffer.iloc[:-1]
            buffer.iloc[0] = pred

        np.testing.assert_allclose(forecast, forecasts_expected, rtol=1e-10)

    def test_forecast_with_exog(self):
        """Test forecasting with exogenous variables."""
        # Create observed data with datetime index
        n_obs = 10
        dates = pd.date_range(
            "2024-01-01", periods=n_obs, freq="30min", tz="Europe/London"
        )
        endog_obs = pd.Series(
            np.arange(
                1.0,
                n_obs + 1.0,
            ),
            index=dates,
        )

        # Create exogenous training data
        exog_train = pd.DataFrame(
            {
                "feature1": np.random.default_rng(seed=1).uniform(size=n_obs),
                "feature2": np.random.default_rng(seed=2).uniform(size=n_obs),
            },
            index=dates,
        )

        # Create a model instance
        lags = [1, 2]
        model = ARXModel(
            lags=lags, regressor=LinearRegression(fit_intercept=True)
        )
        model.fit(endog_obs, exog=exog_train)
        coef, intercept = model.get_params()

        # Test 1: single-step forecast with exogenous variables
        dates_future = pd.date_range(
            dates[-1] + pd.Timedelta("30min"),
            periods=1,
            freq="30min",
            tz="Europe/London",
        )
        exog_future = pd.DataFrame(
            {"feature1": [0.5], "feature2": [0.6]}, index=dates_future
        )

        forecast = model.forecast(steps=1, exog=exog_future)

        # Manual calculation
        history = endog_obs.iloc[-np.array(lags)]
        features = np.concatenate([history.values, exog_future.iloc[0].values])
        forecast_expected = coef @ features + intercept
        np.testing.assert_allclose(forecast[0], forecast_expected, rtol=1e-10)

        # Test 2: multi-step forecast with exogenous variables
        n_steps = 5
        dates_future_multi = pd.date_range(
            dates[-1] + pd.Timedelta("30min"),
            periods=n_steps,
            freq="30min",
            tz="Europe/London",
        )
        exog_future_multi = pd.DataFrame(
            {
                "feature1": np.random.default_rng(seed=3).uniform(
                    size=n_steps
                ),
                "feature2": np.random.default_rng(seed=4).uniform(
                    size=n_steps
                ),
            },
            index=dates_future_multi,
        )

        forecast_multi = model.forecast(steps=n_steps, exog=exog_future_multi)

        # Manual calculation of iterative forecast
        forecasts_expected = []
        buffer = history.copy()
        for i in range(n_steps):
            features = np.concatenate(
                [buffer.values, exog_future_multi.iloc[i].values]
            )
            pred = coef @ features + intercept
            forecasts_expected.append(pred)
            # Update buffer: shift values and add new prediction
            buffer.iloc[1:] = buffer.iloc[:-1].values
            buffer.iloc[0] = pred

        np.testing.assert_allclose(
            forecast_multi, forecasts_expected, rtol=1e-10
        )
