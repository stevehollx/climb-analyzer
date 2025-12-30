#!/usr/bin/env python3
"""
CLI tool for downloading elevation data.

Can be run standalone or via Docker wrapper.

Usage:
    python cli_download_elevation.py --state Vermont --datasets srtm ned10m
    python cli_download_elevation.py --country Switzerland --datasets srtm aw3d30
    python cli_download_elevation.py --bbox 43.5 -73.5 45.0 -71.5 --datasets srtm
    python cli_download_elevation.py --interactive
"""

import argparse
import sys
from pathlib import Path

from climb_analyzer.data.manager import DataManager
from elevation_dataset_selector import get_elevation_datasets_for_region, explain_dataset_selection
from climb_analyzer.data.geo_lookup import find_region, get_region_bounds, is_us_state
from utils.geographic_menu import select_countries, select_us_states


def main():
    parser = argparse.ArgumentParser(
        description="Download elevation data for regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Datasets:
  ned10m      - NED 10m (US only, AWS S3 public)
  srtm        - SRTM 30m (global 60°N-56°S, OpenTopography S3 public)
  aster       - ASTER GDEM 30m (DEPRECATED - data source retired Dec 2025)
  aw3d30      - AW3D30 30m (global, JAXA FTP public)
  arcticdem   - ArcticDEM 32m (Arctic regions, AWS S3 public)
  rema        - REMA 32m (Antarctica, AWS S3 public)

Note: All datasets now use public sources - no credentials required.

Examples:
  # Interactive mode - auto-selects datasets based on region
  python cli_download_elevation.py --interactive

  # Download for a specific US state with auto-selected datasets
  python cli_download_elevation.py --state "Vermont"

  # Download for a state with specific datasets
  python cli_download_elevation.py --state "Alaska" --datasets arcticdem aw3d30

  # Download for a country
  python cli_download_elevation.py --country "Switzerland" --datasets srtm aw3d30

  # Download for a custom bounding box
  python cli_download_elevation.py --bbox 43.5 -73.5 45.0 -71.5 --datasets srtm

  # Docker wrapper (from host machine)
  docker compose run --rm climb-analyzer python cli_download_elevation.py --state Vermont
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--state',
        type=str,
        help='US state name (e.g., "Vermont", "California")'
    )
    group.add_argument(
        '--country',
        type=str,
        help='Country name (e.g., "Switzerland", "Japan")'
    )
    group.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('LAT_MIN', 'LON_MIN', 'LAT_MAX', 'LON_MAX'),
        help='Bounding box: lat_min lon_min lat_max lon_max'
    )
    group.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode - select from menus'
    )

    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['ned10m', 'srtm', 'aster', 'aw3d30', 'arcticdem', 'rema'],
        help='Datasets to download (default: auto-select based on region)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='elevation_data',
        help='Output directory for elevation files (default: elevation_data)'
    )

    parser.add_argument(
        '--deployment-mode',
        type=str,
        choices=['local', 'cloud'],
        default='local',
        help='Deployment mode for auto-selection (default: local)'
    )

    args = parser.parse_args()

    # Initialize data manager
    manager = DataManager()

    region_name = None
    bounds = None
    is_state = False
    datasets = args.datasets

    if args.interactive:
        # Interactive menu
        print("\n" + "="*80)
        print("  Elevation Data Downloader")
        print("="*80 + "\n")

        print("Select region type:")
        print("  1. US State")
        print("  2. Country")
        print("  3. Custom bounding box")
        print()

        choice = input("Enter choice [1/2/3]: ").strip()

        if choice == "1":
            # US States
            _, selected_states, _ = select_us_states()
            if not selected_states:
                print("No state selected")
                return 1
            region_name = selected_states[0]  # Take first selection
            is_state = True
            bounds = manager.get_region_bounds(region_name, is_state=True)

        elif choice == "2":
            # Countries
            _, selected_countries, _ = select_countries()
            if not selected_countries:
                print("No country selected")
                return 1
            region_name = selected_countries[0]  # Take first selection
            is_state = False
            bounds = manager.get_region_bounds(region_name, is_state=False)

        elif choice == "3":
            # Custom bbox
            print("\nEnter bounding box coordinates:")
            try:
                lat_min = float(input("  Latitude min:  ").strip())
                lon_min = float(input("  Longitude min: ").strip())
                lat_max = float(input("  Latitude max:  ").strip())
                lon_max = float(input("  Longitude max: ").strip())
                bounds = (lat_min, lon_min, lat_max, lon_max)
                region_name = f"Custom ({lat_min:.2f}, {lon_min:.2f}) to ({lat_max:.2f}, {lon_max:.2f})"
            except ValueError:
                print("Invalid coordinates")
                return 1
        else:
            print("Invalid choice")
            return 1

    elif args.state:
        # Single state
        if not is_us_state(args.state):
            print(f"Error: State '{args.state}' not found")
            return 1

        region_name = args.state
        is_state = True
        bounds = manager.get_region_bounds(region_name, is_state=True)

    elif args.country:
        # Single country - check if it exists in geo_lookup
        region_info = find_region(args.country)
        if not region_info:
            print(f"Error: Country '{args.country}' not found")
            return 1

        region_name = args.country
        is_state = False
        bounds = manager.get_region_bounds(region_name, is_state=False)

    elif args.bbox:
        # Custom bounding box
        lat_min, lon_min, lat_max, lon_max = args.bbox
        bounds = (lat_min, lon_min, lat_max, lon_max)
        region_name = f"Custom ({lat_min:.2f}, {lon_min:.2f}) to ({lat_max:.2f}, {lon_max:.2f})"

    if not bounds:
        print(f"Error: Could not determine bounds for {region_name}")
        return 1

    # Auto-select datasets if not specified
    if not datasets:
        datasets = get_elevation_datasets_for_region(bounds, args.deployment_mode)
        print(f"\n{'='*80}")
        print(f"  Auto-selected datasets for {region_name}")
        print(f"{'='*80}")
        explain_dataset_selection(datasets, args.deployment_mode)
        print()

    # Download elevation data (no credentials needed - all sources are public)
    print(f"\n{'='*80}")
    print(f"  Downloading Elevation Data")
    print(f"{'='*80}\n")

    success = manager.download_elevation_data(
        region_name,
        bounds,
        datasets,
        credentials=None  # All datasets now use public sources
    )

    if success:
        print(f"\n{'='*80}")
        print(f"  ✓ Download Complete")
        print(f"{'='*80}\n")
        print(f"  Region: {region_name}")
        print(f"  Datasets: {', '.join(datasets)}")
        print(f"  Location: {manager.elevation_dir}")
        print()
        return 0
    else:
        print(f"\n{'='*80}")
        print(f"  ⚠️  Download Completed with Issues")
        print(f"{'='*80}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
