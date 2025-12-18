#!/usr/bin/env python3
"""
Check which Geofabrik regions are missing bounds in osm_pbf_urls.

With the new architecture, all bounds come from Geofabrik .poly files.
This script validates that all regions have bounds data.
"""

from climb_analyzer.data.geo_definitions import osm_pbf_urls


def count_regions_with_bounds(data, path=""):
    """Recursively count regions with and without bounds."""
    with_bounds = 0
    without_bounds = 0
    missing = []

    for key, value in data.items():
        if not isinstance(value, dict):
            continue

        current_path = f"{path}/{key}" if path else key

        # Check if this level has a pbf_url (it's a downloadable region)
        if 'pbf_url' in value:
            if 'bounds' in value and value['bounds']:
                with_bounds += 1
            else:
                without_bounds += 1
                missing.append(current_path)

        # Recursively check subregions
        if 'subregions' in value:
            sub_with, sub_without, sub_missing = count_regions_with_bounds(
                value['subregions'], current_path
            )
            with_bounds += sub_with
            without_bounds += sub_without
            missing.extend(sub_missing)

    return with_bounds, without_bounds, missing


def validate_bounds():
    """Validate that all OSM regions have bounds data."""
    print("=" * 80)
    print("VALIDATING BOUNDS DATA IN osm_pbf_urls")
    print("=" * 80)
    print()

    with_bounds, without_bounds, missing = count_regions_with_bounds(osm_pbf_urls)

    total = with_bounds + without_bounds
    print(f"Total regions: {total}")
    print(f"With bounds:   {with_bounds} ({100*with_bounds/total:.1f}%)")
    print(f"Without bounds: {without_bounds} ({100*without_bounds/total:.1f}%)")
    print()

    if missing:
        print(f"{'=' * 80}")
        print(f"MISSING BOUNDS: {len(missing)}")
        print(f"{'=' * 80}\n")

        for region_path in missing[:50]:  # Show first 50
            print(f"  • {region_path}")

        if len(missing) > 50:
            print(f"\n  ... and {len(missing) - 50} more")
    else:
        print("✓ All regions have bounds data!")

    return missing


if __name__ == "__main__":
    missing = validate_bounds()

    if missing:
        print(f"\n{'=' * 80}")
        print("RECOMMENDATION")
        print(f"{'=' * 80}\n")
        print("Regions with missing bounds need to have their .poly files fetched.")
        print("Run: python utils/update_geo_definitions.py")
        print()
