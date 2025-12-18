#!/usr/bin/env python3
"""
Country-specific DEM configuration.

This module provides bounding boxes and recommended DEM datasets for countries
that require custom elevation data support.

Use this with setup_wizard.py to download appropriate DEM tiles for specific countries.
"""

from typing import Dict, List, Tuple

# Type alias for bounding box: (min_lat, min_lon, max_lat, max_lon)
BoundingBox = Tuple[float, float, float, float]


# Country configurations with bounding boxes and recommended datasets
# Datasets are listed in priority order: primary, secondary, tertiary
COUNTRY_DEM_CONFIG: Dict[str, Dict] = {
    # North America - United States
    "us_continental": {
        "bbox": (25.0, -125.0, 49.0, -66.0),
        "datasets": ["ned10m", "srtm30m"],
        "description": "United States (Continental) - North America",
    },
    "us_west": {
        "bbox": (31.0, -125.0, 49.0, -102.0),
        "datasets": ["ned10m", "srtm30m"],
        "description": "United States (Western) - North America",
    },
    "us_east": {
        "bbox": (25.0, -102.0, 49.0, -66.0),
        "datasets": ["ned10m", "srtm30m"],
        "description": "United States (Eastern) - North America",
    },
    "alaska": {
        "bbox": (51.0, -180.0, 71.0, -130.0),
        "datasets": ["arcticdem", "aw3d30", "aster"],
        "description": "Alaska, USA - Arctic/Subarctic",
    },
    "hawaii": {
        "bbox": (18.9, -160.3, 22.3, -154.8),
        "datasets": ["ned10m", "srtm30m"],
        "description": "Hawaii, USA - Pacific Ocean",
    },

    # North America - Canada
    "canada_south": {
        "bbox": (42.0, -141.0, 60.0, -52.0),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Canada (Southern <60°N) - North America",
    },
    "canada_north": {
        "bbox": (60.0, -141.0, 84.0, -52.0),
        "datasets": ["arcticdem", "aw3d30", "aster"],
        "description": "Canada (Northern >60°N) - North America",
    },
    "canada_bc": {
        "bbox": (48.3, -139.0, 60.0, -114.0),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "British Columbia, Canada - North America",
    },
    "canada_alberta": {
        "bbox": (49.0, -120.0, 60.0, -110.0),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Alberta, Canada - North America",
    },
    "canada_rockies": {
        "bbox": (49.0, -125.0, 55.0, -110.0),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Canadian Rockies - North America",
    },
    "greenland": {
        "bbox": (60.0, -73.0, 84.0, -12.0),
        "datasets": ["arcticdem"],
        "description": "Greenland - Arctic region",
    },

    # European Countries
    "switzerland": {
        "bbox": (45.8, 5.9, 47.8, 10.5),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Switzerland - Central Europe",
    },
    "iceland": {
        "bbox": (63.0, -25.0, 66.5, -13.0),
        "datasets": ["arcticdem", "aw3d30", "aster"],
        "description": "Iceland - North Atlantic",
    },
    "norway": {
        "bbox": (58.0, 4.5, 71.0, 31.0),
        "datasets": ["arcticdem", "aw3d30", "aster"],
        "description": "Norway - Northern Europe (includes Arctic regions)",
    },
    "sweden": {
        "bbox": (55.3, 11.0, 69.1, 24.2),
        "datasets": ["arcticdem", "aw3d30", "aster"],
        "description": "Sweden - Northern Europe",
    },
    "finland": {
        "bbox": (59.8, 20.5, 70.1, 31.6),
        "datasets": ["arcticdem", "aw3d30", "aster"],
        "description": "Finland - Northern Europe",
    },
    "austria": {
        "bbox": (46.4, 9.5, 49.0, 17.2),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Austria - Central Europe",
    },

    # Asian Countries
    "japan": {
        "bbox": (24.0, 123.0, 46.0, 146.0),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Japan - East Asia",
    },
    "nepal": {
        "bbox": (26.3, 80.0, 30.4, 88.2),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Nepal - South Asia (Himalayas)",
    },
    "bhutan": {
        "bbox": (26.7, 88.8, 28.3, 92.1),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Bhutan - South Asia (Himalayas)",
    },

    # Russia
    "russia_south": {
        "bbox": (41.0, 19.0, 60.0, 180.0),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Russia (Southern <60°N)",
    },
    "russia_north": {
        "bbox": (60.0, 19.0, 82.0, 180.0),
        "datasets": ["arcticdem", "aw3d30", "aster"],
        "description": "Russia (Northern >60°N)",
    },

    # Oceania
    "new_zealand": {
        "bbox": (-47.3, 166.0, -34.4, 178.6),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "New Zealand - Oceania",
    },

    # South America
    "chile": {
        "bbox": (-56.0, -75.6, -17.5, -66.4),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Chile - South America (Andes)",
    },
    "argentina": {
        "bbox": (-55.0, -73.6, -21.8, -53.6),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Argentina - South America",
    },
    "colombia": {
        "bbox": (-4.2, -79.0, 12.5, -66.9),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Colombia - South America (Andes)",
    },

    # Polar Regions
    "antarctica": {
        "bbox": (-90.0, -180.0, -60.0, 180.0),
        "datasets": ["rema"],
        "description": "Antarctica - Antarctic region",
    },
    "antarctica_peninsula": {
        "bbox": (-75.0, -80.0, -63.0, -53.0),
        "datasets": ["rema"],
        "description": "Antarctic Peninsula - Antarctic region",
    },

    # Island Nations / Territories
    "canary_islands": {
        "bbox": (27.6, -18.2, 29.5, -13.4),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Canary Islands, Spain - Atlantic Ocean",
    },
    "reunion": {
        "bbox": (-21.4, 55.2, -20.9, 55.8),
        "datasets": ["srtm30m", "aw3d30", "aster"],
        "description": "Réunion, France - Indian Ocean",
    },
}


def get_country_config(country_code: str) -> Dict:
    """
    Get DEM configuration for a country.

    Args:
        country_code: Country code (lowercase, e.g., 'switzerland', 'iceland')

    Returns:
        Dictionary with bbox, datasets, and description

    Raises:
        KeyError: If country not found
    """
    country_code = country_code.lower()
    if country_code not in COUNTRY_DEM_CONFIG:
        available = ", ".join(sorted(COUNTRY_DEM_CONFIG.keys()))
        raise KeyError(
            f"Country '{country_code}' not found. Available countries: {available}"
        )

    return COUNTRY_DEM_CONFIG[country_code]


def list_countries() -> List[str]:
    """
    Get list of all configured countries.

    Returns:
        List of country codes
    """
    return sorted(COUNTRY_DEM_CONFIG.keys())


def print_country_info(country_code: str):
    """
    Print DEM configuration info for a country.

    Args:
        country_code: Country code
    """
    try:
        config = get_country_config(country_code)
        print(f"\nCountry: {config['description']}")
        print(f"Bounding Box: {config['bbox']}")
        print(f"Recommended Datasets: {', '.join(config['datasets']).upper()}")

        min_lat, min_lon, max_lat, max_lon = config['bbox']
        print(f"\nArea Coverage:")
        print(f"  Latitude:  {min_lat}° to {max_lat}° ({max_lat - min_lat:.1f}° span)")
        print(f"  Longitude: {min_lon}° to {max_lon}° ({max_lon - min_lon:.1f}° span)")

    except KeyError as e:
        print(f"Error: {e}")


def main():
    """Demo: Print configurations for all countries."""
    print("=" * 70)
    print("Available Country DEM Configurations")
    print("=" * 70)

    for country_code in list_countries():
        print_country_info(country_code)
        print("-" * 70)

    print("\nUsage example:")
    print("  from country_dem_config import get_country_config")
    print("  config = get_country_config('switzerland')")
    print("  bbox = config['bbox']")
    print("  datasets = config['datasets']")
    print("\nOr use with setup_wizard.py:")
    print("  python setup_wizard.py --bbox 45.8 5.9 47.8 10.5 --datasets aw3d30 aster")


if __name__ == "__main__":
    main()
