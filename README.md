# Electricity Price Forecasting

The aim of this project is to explore methods of forecasting electricity prices in Great Britain (GB) on the day-ahead power markets.

Day-ahead electricity prices in Great Britain are determined through auctions where buyers and sellers agree on prices for specific delivery periods. Most short-term energy trades in GB occur for hourly and half-hourly settlement periods on day-ahead auctions.

## Installation

Clone the repository and install dependencies using Poetry by running

```bash
cd electricity_pricing
poetry install
```

## Overview

This project provides:

- **Data collection and processing:** GB electricity prices, demand forecasts, generation data, and weather
- **Exploratory analysis:** Investigating trends, seasonal patterns, and correlations
- **Forecasting models:** Currently only ARX (autoregressive with exogenous variables) time series models are available

## Notebooks

- **`example_ARX.ipynb`**: Demonstration of electricity price forecasting models with ARX-style time series models.
- **`exploratory_analysis.ipynb`**: Exploratory data analysis investigating trends, seasonal patterns, and correlations between electricity prices, weather variables, and generation sources.
- **`model_development.ipynb`**: Development and discussion of models for electricity price forecasting, including feature engineering and model selection.

## Data

Processed weather and electricity price data is already available in this repository's `data/processed/` directory.

Users that want to process raw data themselves can download the relevant datasets, update the paths in the data collection scripts, and run the data pipeline.

### Data Sources

Actual historical data and forecasted data on electricity price, demand and generation are sourced from the [Elexon BMRS API](https://bmrs.elexon.co.uk) and the [National Energy System Operator's data portal](https://www.neso.energy/data-portal).

Natural gas prices are sourced from the [Office for National Statistics](https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas).

Weather data is sourced from the UK Met Office (see references below). The locations used were Heathrow in Greater London, Crosby in Merseyside and Dyce in Aberdeenshire, since these are all near major population centres and wind farms.

*References:*

Met Office (2025): MIDAS Open: UK hourly weather observation data, v202507. NERC EDS Centre for Environmental Data Analysis, 18 July 2025. doi:10.5285/99173f6a802147aeba430d96d2bb3099.

Met Office (2025): MIDAS Open: UK hourly solar radiation data, v202507. NERC EDS Centre for Environmental Data Analysis, 18 July 2025. doi:10.5285/76e54f87291c4cd98c793e37524dc98e.

### Units

- Power: megawatts (MW)
- Electricity and natural gas prices: GBP per megawatt hour (£ / MWh)
- Wind speed: metres per second (m/s)
- Wind direction: degrees
- Temperature: degrees Celsius
- Solar irradiation: kilojoules per metre squared (KJ/m²)

### Glossary of terms

- MIP: Market index price of electricity
- AGPT: Actual generation data per settlement period aggregrated by power system resource type
- FUELHH: Half-hourly generation outturn aggregrated by fuel type
- CCGT: Combined cycle gas turbine
- OCGT: Open cycle gas turbine
- NPSHYD: Non-pumped storage hydropower
- PS: Pumped storage
- INTER: Imports/exports from/to other grids via interconnectors
- INDO: Initial national demand outturn
- ITSDO: Initial transmission system demand outturn

The AGPT data does not include flows from interconnects, while the FUELHH data does not include energy generation from solar or embedded generation. The AGPT and FUELHH data are merged to get an accurate breakdown of the different energy generation sources.

## Models

### ARX (Autoregressive with Exogenous Variables)

The ARX model forecasts electricity prices as a linear combination of:
- **Autoregressive terms**: Lagged prices from previous time periods
- **Exogenous variables**: Demand forecasts, generation forecasts, temporal features (holidays, weekends), etc.

See `notebooks/model_development.ipynb` for example usage.

## API Reference

### Models
- `ARXModel`: Autoregressive model with exogenous variables

### Regressors
- `LinearRegression`: Ordinary least squares regression

### Features
- `is_holiday()`: Create holiday indicator features
- `is_weekend()`: Create weekend indicator features
- `add_intercept()`: Add intercept column for linear models

### Evaluation Metrics
- `rmse()`: Root mean square error
- `mae()`: Mean absolute error
- `mape()`: Mean absolute percentage error
- `r2_score()`: Coefficient of determination
- `relative_rmse()`: Relative root mean square error

### Utilities
- `timeshift()`: Shift time-indexed `pandas.DataFrame`s
- `train_test_split()`: Split time-indexed data into train/test sets