# Electricity Price Forecasting

The aim of this project is to explore methods of forecasting electricity prices in Great Britain (GB) on the day-ahead power markets.

Day-ahead electricity prices in Great Britain are decided on auctions where buyers are matched with bidders for an agreed price for a specific period. Most of the energy trades on the short-term markets in Great Britain are made for hourly and half-hourly settlement periods on day-ahead auctions.

## Project Plan

### 1. Select data

To create a simple model that is capable of accurately predicting daily average prices, it is critical to include the most relevant data. This section uses the basic idea of supply and demand to determine which data to include in this project.

In the region of interest, the main sources of energy are
- Gas
- Coal
- Wind
- Solar
- Nuclear
- Biomass
- Interconnectors (energy from abroad)

It is reasonable to assume that the cost of supplying one unit of energy from coal and nuclear is constant on short enough time scales. For simplicity, let us also assume that supply from biomass and interconnectors is negligible in Great Britain. With these assumptions, we make the following hypothesis:

**Supply hypothesis**: The main variables affecting electricity supply in Great Britain are (i) gas prices (daily average global gas price), (ii) wind (daily average wind speed and direction), and (iii) sunshine (daily average solar radiation). All other variables may be neglected.

It is reasonable to assume that heating and industrial activity are the main consumers of electricity in GB. It is also reasonable to assume that the both the demand for heat energy and the amount of industrial activity is tied to the temperature and the seasons. With these assumptions, we make the following hypothesis:

**Demand hypothesis**: The main variables affecting electricity demand in Great Britain are (i) temperature (daily average temperature in GB), and (ii) time (time of day, day of the week, holidays, etc.). All other variables may be neglected.

Finally, we want to connect supply and demand to the buy/sell price of electricity on the wholesale electricity market. To do this, we can use historical data on:
- Energy supply: Amount of energy generated broken down into each generation type
- Energy demand: Total amount of energy bought by retailers per day
- Market index price (MIP): Index that reflects the daily price of wholesale electricity in the short term markets

Relevant data for all of the above variables can be found on the following websites:
- [Elexon BMRS](https://bmrs.elexon.co.uk/market-index-prices): UK electricity markets data
- [Met Office](): UK weather data

### 2. Build historical data pipeline

- Clearly specify units
- Load data from open API endpoints
- Unpack data into dataframes
- Clean NaNs, Infs and other abnormalities
- Unpack data into a standardised pandas.DataFrame, checking that the timestamps and units are correct for all entries

### 3. Explore and analyse data

- Plot daily, weekly, seasonal and annual patterns
- Analyse supply/demand behaviours and test hypotheses
- Idea: Analyse frequency spectrum

### 4. Create models

- Simple regression models
- Deep learning models
- Stochastic differential equation model

### 5. Backtest models

- Backtest models on historical data and select the top performers

## 6. Operational forecasting

- Establish data pipeline using streaming data
- Use models to forecast prices from streaming data

