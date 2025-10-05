"""
Data pipeline for creating processed datasets.

This module orchestrates the collection and merging of electricity
generation, demand, price, and weather data.
"""

from .collectors import (
    collect_agpt_data,
    collect_fuelhh_data,
    collect_demand_data,
    collect_mip_data,
    collect_gas_price_data,
    collect_weather_data
)


def create_electricity_dataset(start_date, end_date, gas_price_path, output_path=None, verbose=False):
    """
    Create merged electricity dataset combining generation, demand, and pricing data.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        gas_price_path: Path to gas price file
        output_path: Path to save the output CSV file (optional)
        verbose: If True, print progress messages

    Returns:
        DataFrame with merged electricity data
    """
    print("\n=== Collecting Electricity Data ===") if verbose else None

    # Collect all datasets
    df_demand = collect_demand_data(start_date, end_date, verbose=verbose)
    df_mip = collect_mip_data(start_date, end_date, verbose=verbose)
    df_agpt = collect_agpt_data(start_date, end_date, verbose=verbose)
    df_fuelhh = collect_fuelhh_data(start_date, end_date, verbose=verbose)
    df_gas_price = collect_gas_price_data(start_date, end_date, gas_price_path)

    # Merge generation datasets (AGPT with interconnector from FUELHH)
    print("\nMerging generation datasets...") if verbose else None
    df_generation = df_agpt.merge(
        df_fuelhh[['settlementDate', 'settlementPeriod', 'INTER']],
        on=['settlementDate', 'settlementPeriod'],
        how='inner'
    )

    # Merge with demand data
    print("Merging with demand data...") if verbose else None
    df_merged = df_generation.merge(
        df_demand[['settlementDate', 'settlementPeriod', 'INDO', 'ITSO']],
        on=['settlementDate', 'settlementPeriod'],
        how='inner'
    )

    # Merge with MIP data
    print("Merging with market index price data...") if verbose else None
    df_merged = df_merged.merge(
        df_mip[['settlementDate', 'settlementPeriod', 'marketIndexPrice', 'marketIndexTradingVolume']],
        on=['settlementDate', 'settlementPeriod'],
        how='inner'
    )

    # Merge with gas price data (broadcasts daily price to all settlement periods)
    print("Merging with gas price data...") if verbose else None
    df_merged = df_merged.merge(
        df_gas_price[['settlementDate', 'naturalGasPrice']],
        on='settlementDate',
        how='left'
    )

    # Save to disk if path provided
    if output_path:
        print(f"\nSaving merged electricity data to {output_path}...") if verbose else None
        df_merged.to_csv(output_path, index=False)
        print(f"Saved {len(df_merged)} records") if verbose else None

    return df_merged


def create_weather_dataset(data_dir, output_path=None, locations=None, years=None, verbose=False):
    """
    Create weather dataset from Met Office data files.

    Args:
        data_dir: Directory containing Met Office data
        output_path: Path to save the output CSV file (optional)
        locations: List of location names (optional)
        years: List of years to process (optional)
        verbose: If True, print progress messages

    Returns:
        DataFrame with weather data
    """
    print("\n=== Processing Weather Data ===") if verbose else None

    df_weather = collect_weather_data(data_dir, locations=locations, years=years)

    # Save to disk if path provided
    if output_path:
        print(f"\nSaving weather data to {output_path}...") if verbose else None
        df_weather.to_csv(output_path, index=False)
        print(f"Saved {len(df_weather)} records") if verbose else None

    return df_weather
