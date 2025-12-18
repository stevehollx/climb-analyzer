#!/usr/bin/env python3
"""
Programmatically fetch bounding boxes for countries missing from Natural Earth 110m.

Uses multiple data sources:
1. Natural Earth 10m dataset (more detailed)
2. Overpass API (OpenStreetMap boundary queries)
3. Nominatim API (geocoding service)
4. Hardcoded bounds for known micro-states
"""

import requests
import time
import json
import geopandas as gpd
from pathlib import Path
from typing import Optional, Tuple, Dict
import os


# Hardcoded bounds for known micro-states (too small for even 10m dataset)
HARDCODED_BOUNDS = {
    'monaco': (43.7247, 7.4090, 43.7519, 7.4398),
    'vatican': (41.9002, 12.4457, 41.9073, 12.4584),
    'san marino': (43.8937, 12.4035, 43.9920, 12.5158),
    'liechtenstein': (47.0484, 9.4716, 47.2706, 9.6357),
}


def get_bounds_from_nominatim(country_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box from Nominatim geocoding API.

    Args:
        country_name: Name of the country/region

    Returns:
        (lat_min, lon_min, lat_max, lon_max) or None
    """
    print(f"  Trying Nominatim for: {country_name}")

    try:
        # Nominatim API endpoint
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': country_name,
            'format': 'json',
            'limit': 1,
            'polygon_geojson': 0
        }
        headers = {
            'User-Agent': 'ClimbAnalyzer/1.0 (boundary lookup)'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        results = response.json()

        if results:
            result = results[0]
            boundingbox = result.get('boundingbox')

            if boundingbox and len(boundingbox) == 4:
                # Nominatim returns: [min_lat, max_lat, min_lon, max_lon]
                lat_min = float(boundingbox[0])
                lat_max = float(boundingbox[1])
                lon_min = float(boundingbox[2])
                lon_max = float(boundingbox[3])

                print(f"    ✓ Found via Nominatim: {lat_min:.4f}, {lon_min:.4f}, {lat_max:.4f}, {lon_max:.4f}")
                return (lat_min, lon_min, lat_max, lon_max)

        print(f"    ✗ Not found in Nominatim")
        return None

    except Exception as e:
        print(f"    ✗ Nominatim error: {e}")
        return None


def get_bounds_from_overpass(country_name: str, iso_code: Optional[str] = None) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box from Overpass API (OpenStreetMap).

    Args:
        country_name: Name of the country/region
        iso_code: Optional ISO country code

    Returns:
        (lat_min, lon_min, lat_max, lon_max) or None
    """
    print(f"  Trying Overpass API for: {country_name}")

    try:
        # Overpass API query for country boundary
        query = f"""
        [out:json][timeout:30];
        (
          relation["name:en"="{country_name}"]["boundary"="administrative"]["admin_level"="2"];
          relation["name"="{country_name}"]["boundary"="administrative"]["admin_level"="2"];
        );
        out bb;
        """

        url = "https://overpass-api.de/api/interpreter"
        response = requests.post(url, data={'data': query}, timeout=35)
        response.raise_for_status()

        data = response.json()

        if 'elements' in data and data['elements']:
            element = data['elements'][0]
            bounds = element.get('bounds')

            if bounds:
                lat_min = float(bounds['minlat'])
                lon_min = float(bounds['minlon'])
                lat_max = float(bounds['maxlat'])
                lon_max = float(bounds['maxlon'])

                print(f"    ✓ Found via Overpass: {lat_min:.4f}, {lon_min:.4f}, {lat_max:.4f}, {lon_max:.4f}")
                return (lat_min, lon_min, lat_max, lon_max)

        print(f"    ✗ Not found in Overpass")
        return None

    except Exception as e:
        print(f"    ✗ Overpass error: {e}")
        return None


def get_bounds_from_ne_10m(country_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box from Natural Earth 10m dataset.

    Args:
        country_name: Name of the country/region

    Returns:
        (lat_min, lon_min, lat_max, lon_max) or None
    """
    print(f"  Trying Natural Earth 10m for: {country_name}")

    try:
        # Check if we have the 10m dataset
        ne_10m_dir = Path("./data/ne_10m_admin_0_countries")

        if not ne_10m_dir.exists():
            print("    ✗ Natural Earth 10m dataset not downloaded")
            print("      Run: python3 download_ne_10m.py")
            return None

        shp_file = ne_10m_dir / "ne_10m_admin_0_countries.shp"

        if not shp_file.exists():
            print("    ✗ Shapefile not found")
            return None

        # Read the shapefile
        gdf = gpd.read_file(shp_file)

        # Try various name matching strategies
        matches = []

        # Try exact match on various name fields
        for name_field in ['NAME', 'NAME_LONG', 'ADMIN', 'SOVEREIGNT', 'NAME_EN']:
            if name_field in gdf.columns:
                match = gdf[gdf[name_field].str.lower() == country_name.lower()]
                if not match.empty:
                    matches.append(match.iloc[0])
                    break

        # Try partial match if no exact match
        if not matches:
            for name_field in ['NAME', 'NAME_LONG', 'ADMIN']:
                if name_field in gdf.columns:
                    match = gdf[gdf[name_field].str.lower().str.contains(country_name.lower(), na=False)]
                    if not match.empty:
                        matches.append(match.iloc[0])
                        break

        if matches:
            bounds = matches[0].geometry.bounds
            lat_min, lon_min, lat_max, lon_max = bounds[1], bounds[0], bounds[3], bounds[2]

            print(f"    ✓ Found via Natural Earth 10m: {lat_min:.4f}, {lon_min:.4f}, {lat_max:.4f}, {lon_max:.4f}")
            return (lat_min, lon_min, lat_max, lon_max)

        print(f"    ✗ Not found in Natural Earth 10m")
        return None

    except Exception as e:
        print(f"    ✗ Natural Earth 10m error: {e}")
        return None


def get_country_bounds(country_name: str, osm_path: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box for a country using multiple fallback methods.

    Args:
        country_name: Name of the country/region
        osm_path: OSM path (e.g., "europe/monaco")

    Returns:
        (lat_min, lon_min, lat_max, lon_max) or None
    """
    print(f"\nLooking up: {country_name} ({osm_path})")

    # Method 1: Check hardcoded bounds
    if country_name.lower() in HARDCODED_BOUNDS:
        bounds = HARDCODED_BOUNDS[country_name.lower()]
        print(f"  ✓ Using hardcoded bounds: {bounds}")
        return bounds

    # Method 2: Try Natural Earth 10m (most reliable for countries)
    bounds = get_bounds_from_ne_10m(country_name)
    if bounds:
        return bounds

    # Wait a bit before API calls
    time.sleep(1)

    # Method 3: Try Nominatim (good coverage, rate limited)
    bounds = get_bounds_from_nominatim(country_name)
    if bounds:
        return bounds

    # Wait before next API call
    time.sleep(2)

    # Method 4: Try Overpass API (most accurate, but slower)
    bounds = get_bounds_from_overpass(country_name)
    if bounds:
        return bounds

    print(f"  ✗ Could not find bounds for {country_name}")
    return None


def download_ne_10m_dataset():
    """Download Natural Earth 10m dataset if not present."""
    print("=" * 80)
    print("DOWNLOADING NATURAL EARTH 10M DATASET")
    print("=" * 80)
    print()

    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    extract_to = "./data/boundary_data"
    zip_name = "ne_10m_admin_0_countries.zip"

    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, zip_name)

    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name}...")
        resp = requests.get(url)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        print("✓ Downloaded")

    shp_dir = os.path.join(extract_to, zip_name.replace(".zip", ""))
    if not os.path.exists(shp_dir):
        print(f"Extracting {zip_name}...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(shp_dir)
        print("✓ Extracted")

    print()
    return shp_dir


def fetch_all_missing_bounds(missing_countries):
    """Fetch bounds for all missing countries."""

    print("=" * 80)
    print("FETCHING BOUNDS FOR MISSING COUNTRIES")
    print("=" * 80)

    # Try to download Natural Earth 10m first
    try:
        download_ne_10m_dataset()
    except Exception as e:
        print(f"⚠️  Could not download Natural Earth 10m: {e}")
        print("    Will use API fallbacks")
        print()

    results = {}
    failed = []

    for continent, osm_path, country_name in missing_countries:
        bounds = get_country_bounds(country_name, osm_path)

        if bounds:
            results[(continent, osm_path)] = bounds
        else:
            failed.append((continent, osm_path, country_name))

        # Rate limiting
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print(f"RESULTS: {len(results)} found, {len(failed)} failed")
    print("=" * 80)

    if failed:
        print("\nFailed to find bounds for:")
        for continent, osm_path, country_name in failed:
            print(f"  • {country_name} ({osm_path})")

    return results, failed


def generate_bounds_dict_code(results: Dict):
    """Generate Python code for the bounds dictionary."""

    print("\n" + "=" * 80)
    print("GENERATED PYTHON CODE")
    print("=" * 80)
    print()
    print("# Add these entries to geo_definitions.py region_bounds dictionary:")
    print()

    for (continent, osm_path), (lat_min, lon_min, lat_max, lon_max) in sorted(results.items()):
        # Convert path to tuple format
        path_parts = [osm_path]
        path_tuple = tuple(path_parts)

        print(f'    ("{continent}", {path_tuple!r}): ({lat_min:.6f}, {lon_min:.6f}, {lat_max:.6f}, {lon_max:.6f}),')


if __name__ == "__main__":
    # Import the missing countries list
    import sys
    sys.path.insert(0, '/Volumes/usb1-drive/ca8')

    from check_missing_countries import find_missing_countries

    print("Step 1: Finding missing countries...")
    missing = find_missing_countries()

    if not missing:
        print("\n✓ No missing countries to fetch!")
        sys.exit(0)

    print(f"\nStep 2: Fetching bounds for {len(missing)} missing countries...")
    print("This may take 5-10 minutes due to API rate limiting...")
    print()

    results, failed = fetch_all_missing_bounds(missing)

    if results:
        generate_bounds_dict_code(results)

        # Save to file
        output_file = "missing_country_bounds.py"
        with open(output_file, 'w') as f:
            f.write("# Missing country bounds\n")
            f.write("# Generated automatically - add to geo_definitions.py region_bounds\n\n")
            f.write("missing_bounds = {\n")
            for (continent, osm_path), (lat_min, lon_min, lat_max, lon_max) in sorted(results.items()):
                path_parts = [osm_path]
                path_tuple = tuple(path_parts)
                f.write(f'    ("{continent}", {path_tuple!r}): ({lat_min:.6f}, {lon_min:.6f}, {lat_max:.6f}, {lon_max:.6f}),\n')
            f.write("}\n")

        print(f"\n✓ Saved to {output_file}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review missing_country_bounds.py")
    print("2. Add entries to geo_definitions.py region_bounds")
    print("3. Or re-run update_geo_definitions.py to regenerate everything")
