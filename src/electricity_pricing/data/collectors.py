"""
Data collection functions for BMRS API and weather data.

This module provides functions to collect:
- Electricity generation data by fuel type (AGPT)
- Interconnector flows (FUELHH)
- Demand data (INDO/ITSO)
- Market price index data
- Weather observations from locally stored Met Office files
"""

import os
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
        verbose: If True, print progress messages

    Returns:
        DataFrame with generation data by fuel type
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/AGPT/"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
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

    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

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
        verbose: If True, print progress messages

    Returns:
        DataFrame with generation data including interconnectors
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
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

    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

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
        verbose: If True, print progress messages

    Returns:
        DataFrame with INDO and ITSO demand data
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
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

    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

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


def collect_mip_data(start_date, end_date, verbose=False):
    """
    Collect market price index data from BMRS API.
    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        verbose: If True, print progress messages

    Returns:
        DataFrame with price and volume data
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
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

    df = pd.DataFrame(all_data)
    df['startTime'] = pd.to_datetime(df['startTime'])

    df_price = df.pivot_table(
        index=['settlementDate', 'settlementPeriod', 'startTime'],
        values=['price', 'volume']
    ).reset_index()

    df_price = df_price.sort_values(['settlementDate', 'settlementPeriod']).reset_index(drop=True)

    df_price.rename(columns={'price': 'marketIndexPrice', 'volume': 'marketIndexTradingVolume'}, inplace=True)

    return df_price


def collect_gas_price_data(start_date, end_date, path_to_file):
    """
    Collect gas price data from locally stored Excel file.

    Returns daily gas prices that can be merged with settlement period data.
    The daily price will be broadcast to all settlement periods for that day.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        path_to_file: Path to Excel file containing gas price data

    Returns:
        DataFrame with settlementDate and naturalGasPrice columns
    """
    df = pd.read_excel(path_to_file, sheet_name="Table 1 Daily SAP of Gas", header=5)

    # Filter date range and drop unnecessary columns
    df.drop(columns='SAP seven-day rolling average', inplace=True)
    df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Convert price units from p/kWh to £/MWh
    PENCE_TO_POUNDS = 1 / 100
    KWH_TO_MWH = 1000
    df['naturalGasPrice'] = df['SAP actual day'] * PENCE_TO_POUNDS * KWH_TO_MWH

    # Convert Date to settlementDate format (string YYYY-MM-DD)
    df['settlementDate'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    return df[['settlementDate', 'naturalGasPrice']]


def collect_weather_data(data_dir, locations=None, years=None):
    """
    Load and process weather data from Met Office CSV files.

    Args:
        data_dir: Directory containing Met Office data
        locations: List of location names (default: ["heathrow", "crosby", "dyce"])
        years: List of years to process (default: [2021, 2022, 2023, 2024])

    Returns:
        DataFrame with processed weather data from all locations and years

    Helpful Information:
        - missing_value: NA
        - ob_time: Date and time (YYYY-MM-DD HH:MM:SS) of the observation
        - wind_speed_unit_id: Code to describe the origin of the wind speed units.
        - wind_speed_unit_id=0: wind speed estimated (metres per second)
        - wind_speed_unit_id=1: wind speed from anemometer (metres per second)
        - wind_speed_unit_id=3: wind speed estimated (knots)
        - wind_speed_unit_id=4: wind speed from anemometer (knots)
        - wind_direction: wind direction in true degrees
        - wind_speed: wind speed (knots)
        - visibility: visibility (decametres)
        - air_temperature: Air temperature (degrees celsius)
        - glbl_irad_amt: Global solar irradiation amount (KJ/m^2)
    """
    if locations is None:
        locations = ["heathrow", "crosby", "dyce"]
    if years is None:
        years = [2021, 2022, 2023, 2024]

    KNOT_TO_MS = 0.514  # 1 knot = 0.514 m/s
    dfs = []

    for location in locations:
        base_dir = os.path.join(data_dir, location)
        weather_dir = os.path.join(base_dir, "weather")
        irradiation_dir = os.path.join(base_dir, "solar_irradiation")
        dfs_location = []

        for year in years:
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

            # Load solar irradiation data if available
            irradiation_path = os.path.join(irradiation_dir, f"solar_irradiation_{location}_{year}.csv")
            if os.path.exists(irradiation_path):
                df_rad = pd.read_csv(irradiation_path, header=78, low_memory=False)
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
            df_location = pd.concat(dfs_location, ignore_index=True)
            df_location['location'] = location.capitalize()
            dfs.append(df_location)

    if not dfs:
        raise ValueError("No weather data found")

    return pd.concat(dfs, ignore_index=True)
