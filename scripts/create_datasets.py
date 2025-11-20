"""
Script to create electricity and weather datasets.

This script uses the `electricity_pricing` package to collect and process
data from the BMRS API and Met Office files.

Requests are chunked when requesting data from the BMRS API for time periods
longer than 3 days, which can be slow.

See `docs/data.md` for usage instructions.
"""

from electricity_pricing.data.pipeline import (
    create_electricity_dataset,
    create_weather_dataset,
)


def main():
    """Create both electricity and weather datasets."""
    # Electricity data parameters
    elec_start_date = "2021-01-01"
    elec_end_date = "2024-12-31"
    gas_price_path = "data/raw/ons_gas_prices.xlsx"
    elec_output_path = "data/processed/electricity_data.csv"

    # Weather data parameters
    weather_data_dir = "data/raw/met_office"  # path to raw data directory
    weather_output_path = "data/processed/weather_data.csv"

    # Create electricity dataset
    print("Creating electricity dataset...")
    df_electricity = create_electricity_dataset(
        elec_start_date, elec_end_date, elec_output_path, verbose=True
    )

    # Create weather dataset
    print("\nCreating weather dataset...")
    try:
        df_weather = create_weather_dataset(
            weather_data_dir, weather_output_path, gas_price_path, verbose=True
        )
    except ValueError as e:
        print(f"Warning: {e}")
        df_weather = None

    # Summary
    print("\n=== Dataset Creation Complete ===")
    if df_electricity is not None:
        print(
            f"Electricity dataset: {len(df_electricity)} "
            +
            f"records saved to {elec_output_path}"
        )
    if df_weather is not None:
        print(
            f"Weather dataset: {len(df_weather)} records "
            +
            f"saved to {weather_output_path}"
        )


if __name__ == "__main__":
    main()
