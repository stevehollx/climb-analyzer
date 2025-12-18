#!/usr/bin/env python3
"""
Convenience script to download DEM data for pre-configured countries.

This script combines country_dem_config.py with setup_wizard.py to provide
an easy way to download DEMs for specific countries.

Usage:
    python download_country_dem.py switzerland
    python download_country_dem.py iceland --output-dir /path/to/data
    python download_country_dem.py --list
"""

import argparse
import sys
from pathlib import Path

from country_dem_config import get_country_config, list_countries, print_country_info
from climb_analyzer.data.dem_downloaders import download_dem_for_country


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download DEM data for pre-configured countries"
    )

    parser.add_argument(
        "country",
        nargs="?",
        help="Country code (e.g., switzerland, iceland, japan)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available countries"
    )

    parser.add_argument(
        "--info",
        metavar="COUNTRY",
        help="Show information for a specific country"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/elevation_data"),
        help="Output directory for DEM tiles (default: elevation_data)"
    )

    args = parser.parse_args()

    # List countries
    if args.list:
        print("=" * 70)
        print("Available Countries")
        print("=" * 70)
        for country in list_countries():
            try:
                config = get_country_config(country)
                print(f"\n{country:20s} - {config['description']}")
                print(f"{'':20s}   Datasets: {', '.join(config['datasets']).upper()}")
            except KeyError:
                pass
        print("\n" + "=" * 70)
        return 0

    # Show country info
    if args.info:
        print_country_info(args.info)
        return 0

    # Download DEM for country
    if not args.country:
        parser.print_help()
        return 1

    try:
        config = get_country_config(args.country)

        print("=" * 70)
        print(f"Downloading DEM data for: {config['description']}")
        print("=" * 70)

        bbox = config['bbox']
        datasets = config['datasets']

        print(f"\nBounding Box: {bbox}")
        print(f"Datasets: {', '.join(datasets).upper()}")
        print(f"Output Directory: {args.output_dir}")

        # Download the DEM data
        success = download_dem_for_country(
            country_name=config['description'],
            bbox=bbox,
            output_dir=args.output_dir,
            datasets=datasets
        )

        if success:
            print("\n" + "=" * 70)
            print("Download Complete!")
            print("=" * 70)
            print(f"\nDEM data saved to: {args.output_dir.absolute()}")
            return 0
        else:
            print("\n" + "=" * 70)
            print("Download Failed")
            print("=" * 70)
            return 1

    except KeyError as e:
        print(f"Error: {e}")
        print(f"\nUse --list to see available countries")
        return 1


if __name__ == "__main__":
    sys.exit(main())
