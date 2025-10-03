"""
Collect weather and electricity data to form the datasets used in the electricity price forecasting models.

# Raw data sources:

- Elexon BMRS API: https://bmrs.elexon.co.uk/api-documentation/introduction)
- Weather data from the UK Met Office: https://catalogue.ceda.ac.uk/uuid/99173f6a802147aeba430d96d2bb3099/
- Solar irradiation data from the UK Met Office: https://catalogue.ceda.ac.uk/uuid/76e54f87291c4cd98c793e37524dc98e/

Citations:
- Met Office (2025): MIDAS Open: UK hourly weather observation data, v202507. NERC EDS Centre for Environmental Data Analysis, 18 July 2025. doi:10.5285/99173f6a802147aeba430d96d2bb3099.
- Met Office (2025): MIDAS Open: UK hourly solar radiation data, v202507. NERC EDS Centre for Environmental Data Analysis, 18 July 2025. doi:10.5285/76e54f87291c4cd98c793e37524dc98e.

# Instructions

Users need to download the relevant raw weather data from the Met Office themselves and update the load paths below accordingly. 

The weather station data can be found at the following links:
- Heathrow: https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202507/greater-london/00708_heathrow)
- Crosby: https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202507/merseyside/17309_crosby)
- Dyce: https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202507/aberdeenshire/00161_dyce)

# Glossary of terms

- CCGT: Combined cycle gas turbine
- OCGT: Open cycle gas turbine
- NPSHYD: Non-pumped storage hydropower
- PS: Pumped storage
- INTER:  Imports/exports from/to other grids via interconnectors
- AGPT: Actual generation data per settlement period aggregrated by power system resource type
- FUELHH: Half-hourly generation outturn aggregrated by fuel type
"""

import os
import json
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd


def collect_agpt_data(start_date, end_date, verbose=False):
    """
    Collect AGPT (Actual Generation Per Type) data from BMRS API.
    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)

    Returns:
        DataFrame with generation data by fuel type
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/AGPT/"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        # Calculate chunk end (4 days maximum)
        current_end = min(current_start + timedelta(days=3), end)

        params = {
            "publishDateTimeFrom": current_start.strftime("%Y-%m-%d 00:00"),
            "publishDateTimeTo": current_end.strftime("%Y-%m-%d 01:00"),
            "format": "json"
        }

        if verbose:
            print(f"Fetching AGPT data from {params['publishDateTimeFrom']} to {params['publishDateTimeTo']}...")

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data['data'])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    # Process data
    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

    # Create pivot table
    df_AGPT = df.pivot_table(
        index=['settlementDate', 'settlementPeriod', 'startTime'],
        columns='psrType',
        values='quantity',
        aggfunc='first'
    ).reset_index()

    df_AGPT = df_AGPT.sort_values(['settlementDate', 'settlementPeriod']).reset_index(drop=True)

    # Group wind types
    df_AGPT['WIND'] = df_AGPT[['Wind Offshore', 'Wind Onshore']].sum(axis=1)
    df_AGPT = df_AGPT.drop(columns=['Wind Offshore', 'Wind Onshore'])

    # Group hydro and other
    df_AGPT['OTHER'] = df_AGPT[['Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Other']].sum(axis=1)
    df_AGPT = df_AGPT.drop(columns=['Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Other'])

    # Rename columns
    df_AGPT = df_AGPT.rename(columns={
        "Nuclear": "NUCLEAR",
        "Biomass": "BIOMASS",
        "Fossil Gas": "GAS",
        "Fossil Oil": "OIL",
        "Fossil Hard coal": "COAL",
        "Solar": "SOLAR"
    })

    return df_AGPT


def collect_fuelhh_data(start_date, end_date, verbose=False):
    """
    Collect FUELHH (Fuel Type Half-Hourly) data from BMRS API.
    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)

    Returns:
        DataFrame with generation data including interconnectors
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        # Calculate chunk end (4 days maximum)
        current_end = min(current_start + timedelta(days=3), end)

        params = {
            "publishDateTimeFrom": current_start.strftime("%Y-%m-%d 00:00"),
            "publishDateTimeTo": current_end.strftime("%Y-%m-%d 01:00"),
            "format": "json"
        }

        if verbose:
            print(f"Fetching FUELHH data from {params['publishDateTimeFrom']} to {params['publishDateTimeTo']}...")

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data['data'])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    # Process data
    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

    # Create pivot table
    df_FUELHH = df.pivot_table(
        index=['settlementDate', 'settlementPeriod', 'startTime'],
        columns='fuelType',
        values='generation',
        aggfunc='first'
    ).reset_index()

    df_FUELHH = df_FUELHH.sort_values(['settlementDate', 'settlementPeriod']).reset_index(drop=True)

    # Group gas categories
    df_FUELHH['GAS'] = df_FUELHH[['OCGT', 'CCGT']].sum(axis=1)
    df_FUELHH = df_FUELHH.drop(columns=['OCGT', 'CCGT'])

    # Group interconnectors
    interconnects = df_FUELHH.filter(regex='^INT')
    total_interconnect = interconnects.sum(axis=1)
    df_FUELHH = df_FUELHH.filter(regex='^(?!INT)')
    df_FUELHH['INTER'] = total_interconnect

    # Group pumped storage, non-pumped hydro and other
    df_FUELHH['OTHER'] = df_FUELHH[['OTHER', 'PS', 'NPSHYD']].sum(axis=1)
    df_FUELHH = df_FUELHH.drop(columns=['PS', 'NPSHYD'])

    return df_FUELHH


def collect_demand_data(start_date, end_date, verbose=False):
    """
    Collect demand outturn data from BMRS API.
    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)

    Returns:
        DataFrame with INDO and ITSO demand data
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        # Calculate chunk end (4 days maximum)
        current_end = min(current_start + timedelta(days=3), end)

        params = {
            "settlementDateFrom": current_start.strftime("%Y-%m-%d"),
            "settlementDateTo": current_end.strftime("%Y-%m-%d"),
            "settlementPeriod": np.arange(1, 50).tolist(),
            "format": "json"
        }

        if verbose:
            print(f"Fetching demand data from {params['settlementDateFrom']} to {params['settlementDateTo']}...")

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data['data'])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    # Process data
    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

    # Create pivot table
    df_demand = df.pivot_table(
        index=['settlementDate', 'settlementPeriod', 'startTime'],
        values=['initialDemandOutturn', 'initialTransmissionSystemDemandOutturn']
    ).reset_index()

    df_demand = df_demand.sort_values(['settlementDate', 'settlementPeriod']).reset_index(drop=True)

    df_demand = df_demand.rename(columns={
        "initialDemandOutturn": "INDO",
        "initialTransmissionSystemDemandOutturn": "ITSO"
    })

    return df_demand


def collect_price_data(start_date, end_date, verbose=False):
    """
    Collect market price index data from BMRS API.
    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)

    Returns:
        DataFrame with price and volume data
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        # Calculate chunk end (4 days maximum)
        current_end = min(current_start + timedelta(days=4), end)

        params = {
            "from": current_start.strftime("%Y-%m-%d 00:00"),
            "to": current_end.strftime("%Y-%m-%d 00:00"),
            "dataProviders": ["APX"],
            "format": "json"
        }

        if verbose:
            print(f"Fetching price data from {params['from']} to {params['to']}...")

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data['data'])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    # Process data
    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

    # Create pivot table
    df_price = df.pivot_table(
        index=['settlementDate', 'settlementPeriod', 'startTime'],
        values=['price', 'volume']
    ).reset_index()

    df_price = df_price.sort_values(['settlementDate', 'settlementPeriod']).reset_index(drop=True)

    return df_price


def create_bmrs_dataset(start_date, end_date, output_path="electricity_data.csv", verbose=False):
    """
    Create merged BMRS dataset combining generation, demand, and pricing data.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        output_path: Path to save the output CSV file

    Returns:
        DataFrame with merged BMRS data
    """
    print("\n=== Collecting electricity Data ===")

    # Collect all datasets
    df_agpt = collect_agpt_data(start_date, end_date, verbose=verbose)
    df_fuelhh = collect_fuelhh_data(start_date, end_date, verbose=verbose)
    df_demand = collect_demand_data(start_date, end_date, verbose=verbose)
    df_price = collect_price_data(start_date, end_date, verbose=verbose)

    # Merge generation datasets (AGPT with interconnector from FUELHH)
    print("\nMerging generation datasets...")
    df_generation = df_agpt.merge(
        df_fuelhh[['settlementDate', 'settlementPeriod', 'INTER']],
        on=['settlementDate', 'settlementPeriod'],
        how='inner'
    )

    # Merge with demand data
    print("Merging with demand data...")
    df_merged = df_generation.merge(
        df_demand[['settlementDate', 'settlementPeriod', 'INDO', 'ITSO']],
        on=['settlementDate', 'settlementPeriod'],
        how='inner'
    )

    # Merge with price data
    print("Merging with price data...")
    df_merged = df_merged.merge(
        df_price[['settlementDate', 'settlementPeriod', 'price', 'volume']],
        on=['settlementDate', 'settlementPeriod'],
        how='inner'
    )

    # Save to disk
    print(f"\nSaving merged BMRS data to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print(f"Saved {len(df_merged)} records")

    return df_merged


def create_weather_dataset(data_dir="raw_data_met_office", output_path="weather_data.csv"):
    """
    Create weather dataset from Met Office data files.

    Args:
        data_dir: Directory containing Met Office data
        output_path: Path to save the output CSV file

    Returns:
        DataFrame with weather data
    """
    print("\n=== Processing Weather Data ===")

    locations = ["heathrow", "crosby", "dyce"]
    years = [2021, 2022, 2023, 2024]
    KNOT_TO_MS = 0.514  # 1 knot = 0.514 m/s

    dfs = []

    for location in locations:
        base_dir = os.path.join(data_dir, location)
        weather_dir = os.path.join(base_dir, "weather")
        radiation_dir = os.path.join(base_dir, "solar_radiation")
        dfs_location = []

        for year in years:
            # Load hourly weather data
            weather_path = os.path.join(weather_dir, f"hourly_weather_{location}_{year}.csv")

            if not os.path.exists(weather_path):
                print(f"Warning: Weather data not found for {location} {year}")
                continue

            print(f"Processing {location} {year}...")
            df_hw = pd.read_csv(weather_path, header=283, low_memory=False)
            df_hw = df_hw[['ob_time', 'wind_speed_unit_id', 'wind_speed', 'wind_direction',
                           'visibility', 'air_temperature']]

            # Convert wind speed to m/s
            is_mps = df_hw['wind_speed_unit_id'].isin([0, 1])
            df_hw['wind_speed'] = df_hw['wind_speed'] * (is_mps + ~is_mps * KNOT_TO_MS)
            df_hw.drop(columns='wind_speed_unit_id', inplace=True)

            # Clean up data
            df_hw = df_hw.iloc[:-1]  # Drop last row ('end of data' line)
            df_hw['ob_time'] = pd.to_datetime(df_hw['ob_time'])

            # Load solar radiation data if available
            radiation_path = os.path.join(radiation_dir, f"solar_radiation_{location}_{year}.csv")
            if os.path.exists(radiation_path):
                df_rad = pd.read_csv(radiation_path, header=78, low_memory=False)
                df_rad = df_rad[['ob_end_time', 'glbl_irad_amt']].iloc[:-1]
                df_rad['ob_time'] = pd.to_datetime(df_rad['ob_end_time'])
                df_rad.drop(columns='ob_end_time', inplace=True)
                df_merged = df_hw.merge(df_rad, on='ob_time', how='left')
            else:
                print(f"Warning: No radiation data found for {location} {year}")
                df_hw['glbl_irad_amt'] = np.nan
                df_merged = df_hw

            dfs_location.append(df_merged)

        if dfs_location:
            # Combine all years for this location
            df_location = pd.concat(dfs_location, ignore_index=True)
            df_location['location'] = location.capitalize()
            dfs.append(df_location)

    if not dfs:
        print("Error: No weather data found")
        return None

    # Combine all locations
    df_weather = pd.concat(dfs, ignore_index=True)

    # Save to disk
    print(f"\nSaving weather data to {output_path}...")
    df_weather.to_csv(output_path, index=False)
    print(f"Saved {len(df_weather)} records")

    return df_weather


def main():
    """
    Main function to create both electricity and weather datasets.
    """
    # electricity data parameters
    elec_start_date = "2021-01-01"
    elec_end_date = "2024-12-31"
    elec_output_path = "electricity_data.csv"

    # Weather data parameters
    weather_data_dir = "raw_data_met_office"
    weather_output_path = "weather_data.csv"

    # Create BMRS dataset
    df_electricity = create_bmrs_dataset(elec_start_date, elec_end_date, elec_output_path)

    # Create weather dataset
    df_weather = create_weather_dataset(weather_data_dir, weather_output_path)

    print("\n=== Dataset Creation Complete ===")
    if df_electricity is not None:
        print(f"BMRS dataset: {len(df_electricity)} records saved to {elec_output_path}")
    if df_weather is not None:
        print(f"Weather dataset: {len(df_weather)} records saved to {weather_output_path}")


if __name__ == "__main__":
    main()