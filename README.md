# Electricity Price Forecasting

The aim of this project is to explore methods of forecasting electricity 
prices in Great Britain (GB) on the day-ahead power markets.

## About GB electricity markets

Day-ahead electricity prices in Great Britain are determined through 
auctions where buyers and sellers agree on prices for specific delivery 
periods. Most short-term energy trades in GB occur for hourly and 
half-hourly settlement periods on day-ahead auctions.

## Project overview

- `analysis/`: Exploratory data analysis
- `models/`: Forecasting models
- `utilities/`: Utilities for data processing, feature creation, etc.
- `data/`: Raw and processed data, along with data processing scripts

## Installation

This project's dependencies can be installed by running
```bash
pip install -r requirements.txt
```

## Data

### ETL pipeline

The data is collected and processed with the following scripts:
- `data/scripts/collect_bmrs_data.py`
- `data/scripts/collect_gas_prices.py`
- `data/scripts/collect_neso_forecasts.py`
- `data/scripts/collect_weather_data.py`

The data resulting from running these scripts is stored in `data/processed/`

Processed weather and electricity price data is already available in 
this repository's `data/processed/` directory.

Users that want to process raw data themselves can download the relevant data, 
update the paths in the data collection scripts and run the required scripts.

### Citations

Electricity price, demand and generation data are sourced from the 
[Elexon BMRS API](https://bmrs.elexon.co.uk) and the 
[National Energy System Operator's data portal](https://www.neso.energy/data-portal).

Natural gas price data are sourced from the [Office for National Statistics](https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas).

Weather data are sourced from the UK Met Office (see references below). 
The locations used were Heathrow in Greater London, Crosby in Merseyside 
and Dyce in Aberdeenshire, since these are all near major population 
centres and wind farms.

Met Office (2025): MIDAS Open: UK hourly weather observation data, v202507. 
NERC EDS Centre for Environmental Data Analysis, 18 July 2025. 
doi:10.5285/99173f6a802147aeba430d96d2bb3099.

Met Office (2025): MIDAS Open: UK hourly solar radiation data, v202507. 
NERC EDS Centre for Environmental Data Analysis, 18 July 2025. 
doi:10.5285/76e54f87291c4cd98c793e37524dc98e.

### Units

- Power: megawatts (MW)
- Electricity and natural gas prices: GBP per megawatt hour (£ / MWh)
- Wind speed: metres per second (m/s)
- Wind direction: degrees
- Temperature: degrees Celsius
- Solar irradiation: kilojoules per metre squared (KJ/m²)

### Glossary of terms

- MIP: Market index price of electricity
- AGPT: Actual generation data per settlement period aggregrated by power 
    system resource type
- FUELHH: Half-hourly generation outturn aggregrated by fuel type
- CCGT: Combined cycle gas turbine
- OCGT: Open cycle gas turbine
- NPSHYD: Non-pumped storage hydropower
- PS: Pumped storage
- INTER: Imports/exports from/to other grids via interconnectors
- INDO: Initial national demand outturn
- ITSDO: Initial transmission system demand outturn

The AGPT data does not include flows from interconnects, while the FUELHH 
data does not include energy generation from solar or embedded generation. 
The AGPT and FUELHH data are merged to get an accurate breakdown of the 
different energy generation sources.