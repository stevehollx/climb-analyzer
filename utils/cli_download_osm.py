#!/usr/bin/env python3
"""
CLI tool for downloading OSM planet files.

Can be run standalone or via Docker wrapper.

Usage:
    python cli_download_osm.py --state Vermont
    python cli_download_osm.py --country Switzerland
    python cli_download_osm.py --interactive
"""

import argparse
import sys
from pathlib import Path

from climb_analyzer.data.manager import DataManager
from climb_analyzer.data.geo_lookup import find_region, is_us_state
from utils.geographic_menu import select_countries, select_us_states


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenStreetMap planet files for regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - select from menus
  python cli_download_osm.py --interactive

  # Download for a specific US state
  python cli_download_osm.py --state "Vermont"
  python cli_download_osm.py --state "California"

  # Download for a specific country
  python cli_download_osm.py --country "Switzerland"
  python cli_download_osm.py --country "New Zealand"

  # Docker wrapper (from host machine)
  docker compose run --rm climb-analyzer python cli_download_osm.py --state Vermont
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
        '--interactive',
        action='store_true',
        help='Interactive mode - select from menus'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/planet_osm_data',
        help='Output directory for .pbf files (default: data/planet_osm_data)'
    )

    parser.add_argument(
        '--build-index',
        action='store_true',
        help='Build spatial index after download'
    )

    args = parser.parse_args()

    # Initialize data manager
    manager = DataManager()

    regions_to_download = []
    is_state = False

    if args.interactive:
        # Interactive menu
        print("\n" + "="*80)
        print("  OpenStreetMap Data Downloader")
        print("="*80 + "\n")

        print("Select data source:")
        print("  1. US State")
        print("  2. Country")
        print()

        choice = input("Enter choice [1/2]: ").strip()

        if choice == "1":
            # US States
            _, selected_states, _ = select_us_states()
            if selected_states:
                regions_to_download = selected_states
                is_state = True
        elif choice == "2":
            # Countries
            _, selected_countries, _ = select_countries()
            if selected_countries:
                regions_to_download = selected_countries
                is_state = False
        else:
            print("Invalid choice")
            return 1

    elif args.state:
        # Single state - validate using geo_lookup
        if not is_us_state(args.state):
            print(f"Error: State '{args.state}' not found")
            print("\nUse --interactive mode to see available states")
            return 1

        regions_to_download = [args.state]
        is_state = True

    elif args.country:
        # Single country - validate using geo_lookup
        region_info = find_region(args.country)
        if not region_info:
            print(f"Error: Country '{args.country}' not found")
            print("\nUse --interactive mode to see available countries")
            return 1

        regions_to_download = [args.country]
        is_state = False

    if not regions_to_download:
        print("No regions selected")
        return 1

    # Download each region
    success_count = 0
    fail_count = 0

    for region in regions_to_download:
        print(f"\n{'='*80}")
        print(f"  Processing: {region}")
        print(f"{'='*80}\n")

        # Download OSM data
        pbf_path = manager.download_osm_data(region, is_state=is_state)

        if pbf_path:
            success_count += 1
            print(f"  ✓ Downloaded: {pbf_path}")

            # Build index if requested
            if args.build_index:
                print(f"\n  Building spatial index...")
                if manager.build_osm_index(pbf_path):
                    print(f"  ✓ Index built successfully")
                else:
                    print(f"  ⚠️  Index build failed")
        else:
            fail_count += 1
            print(f"  ❌ Download failed for {region}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  Download Summary")
    print(f"{'='*80}\n")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(regions_to_download)}")
    print()

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
