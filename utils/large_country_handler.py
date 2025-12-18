#!/usr/bin/env python3
"""
Large country handler for climb analyzer.

Automatically splits very large countries into manageable regions to avoid memory exhaustion.
"""

from typing import List, Tuple, Dict
from pathlib import Path


# Countries that need to be split due to size
LARGE_COUNTRIES = {
    "France": {
        "split_method": "latitude",
        "split_at": 46.0,  # Split at approximately 46°N (near Lyon)
        "regions": [
            ("France North", (46.0, -5.0, 51.2, 9.6)),  # North of 46°N
            ("France South", (41.3, -5.0, 46.0, 9.6))   # South of 46°N
        ]
    },
    "Germany": {
        "split_method": "latitude",
        "split_at": 51.0,  # Split at approximately 51°N (near Cologne)
        "regions": [
            ("Germany North", (51.0, 5.9, 55.1, 15.0)),  # North of 51°N
            ("Germany South", (47.3, 5.9, 51.0, 15.0))   # South of 51°N
        ]
    },
    "United States": {
        "split_method": "custom",
        "regions": [
            # Split into major regions
            ("USA Northeast", (38.0, -83.0, 48.0, -67.0)),     # NY, PA, New England
            ("USA Southeast", (24.0, -90.0, 38.0, -75.0)),     # FL to VA
            ("USA Midwest", (36.0, -105.0, 49.0, -83.0)),      # Great Lakes to Plains
            ("USA Southwest", (28.0, -125.0, 42.0, -102.0)),   # CA, AZ, NV, NM
            ("USA Northwest", (42.0, -125.0, 49.0, -102.0)),   # WA, OR, ID, MT
            ("USA Alaska", (51.0, -180.0, 72.0, -130.0)),      # Alaska
            ("USA Hawaii", (18.0, -161.0, 23.0, -154.0))       # Hawaii
        ]
    },
    "Canada": {
        "split_method": "custom",
        "regions": [
            ("Canada West", (48.0, -141.0, 70.0, -110.0)),     # BC, Yukon, NW Territories west
            ("Canada Prairies", (48.0, -110.0, 70.0, -95.0)),  # Alberta, Saskatchewan, Manitoba
            ("Canada Central", (41.0, -95.0, 55.0, -74.0)),    # Ontario
            ("Canada East", (44.0, -80.0, 55.0, -52.0)),       # Quebec, Maritimes
            ("Canada North", (55.0, -141.0, 83.0, -52.0))      # Northern territories
        ]
    },
    "Russia": {
        "split_method": "custom",
        "regions": [
            ("Russia West", (41.0, 19.0, 82.0, 60.0)),         # European Russia
            ("Russia Ural", (50.0, 60.0, 70.0, 70.0)),         # Ural region
            ("Russia Siberia", (50.0, 70.0, 75.0, 110.0)),     # Western Siberia
            ("Russia Far East", (42.0, 110.0, 75.0, 180.0))    # Far East
        ]
    },
    "China": {
        "split_method": "custom",
        "regions": [
            ("China North", (35.0, 73.0, 54.0, 135.0)),        # North of Yangtze
            ("China South", (18.0, 73.0, 35.0, 135.0))         # South of Yangtze
        ]
    },
    "Brazil": {
        "split_method": "latitude",
        "split_at": -15.0,  # Split at approximately 15°S (near Brasília)
        "regions": [
            ("Brazil North", (-34.0, -74.0, -15.0, -34.0)),    # Amazon and North
            ("Brazil South", (-15.0, -74.0, 5.0, -34.0))       # South and Southeast
        ]
    },
    "Australia": {
        "split_method": "custom",
        "regions": [
            ("Australia West", (-35.0, 112.0, -13.0, 129.0)),   # Western Australia
            ("Australia Central", (-35.0, 129.0, -13.0, 141.0)), # NT and SA
            ("Australia East", (-44.0, 141.0, -10.0, 154.0))    # QLD, NSW, VIC, TAS
        ]
    },
    "Argentina": {
        "split_method": "latitude",
        "split_at": -35.0,  # Split at approximately 35°S (near Buenos Aires)
        "regions": [
            ("Argentina North", (-55.0, -74.0, -35.0, -53.0)),  # North and Central
            ("Argentina South", (-35.0, -74.0, -22.0, -53.0))   # Patagonia
        ]
    },
    "India": {
        "split_method": "latitude",
        "split_at": 23.0,  # Split at Tropic of Cancer
        "regions": [
            ("India North", (23.0, 68.0, 36.0, 97.0)),         # North India
            ("India South", (8.0, 68.0, 23.0, 97.0))           # South India
        ]
    }
}


def should_split_country(country_name: str) -> bool:
    """
    Check if a country should be split into regions.

    Args:
        country_name: Name of the country

    Returns:
        True if country should be split
    """
    return country_name in LARGE_COUNTRIES


def get_country_regions(country_name: str) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """
    Get the split regions for a large country.

    Args:
        country_name: Name of the country

    Returns:
        List of (region_name, bbox) tuples, or single entry if not a large country
    """
    if country_name not in LARGE_COUNTRIES:
        return [(country_name, None)]  # Return as-is, bbox will be calculated normally

    country_config = LARGE_COUNTRIES[country_name]
    return country_config["regions"]


def estimate_segment_count(osm_file_path: Path) -> int:
    """
    Estimate the number of road segments in an OSM file.

    This is a rough estimate based on file size.

    Args:
        osm_file_path: Path to OSM .pbf file

    Returns:
        Estimated number of road segments
    """
    if not osm_file_path.exists():
        return 0

    # Rough estimate: ~2000 segments per MB for typical OSM data
    file_size_mb = osm_file_path.stat().st_size / (1024 * 1024)

    # Adjust estimate based on known country densities
    country_name = osm_file_path.stem.replace("-latest", "").replace("_", " ").title()

    density_multipliers = {
        "france": 1.5,      # Dense road network
        "germany": 1.4,     # Dense road network
        "united-states": 1.2,  # Mixed density
        "canada": 0.6,      # Sparse outside cities
        "russia": 0.5,      # Very sparse
        "australia": 0.7,   # Sparse outside coast
        "brazil": 0.8,      # Mixed density
        "china": 1.3,       # Dense in east
        "india": 1.3,       # Dense overall
    }

    base_estimate = file_size_mb * 2000
    multiplier = density_multipliers.get(country_name.lower(), 1.0)

    return int(base_estimate * multiplier)


def should_use_streaming_mode(country_name: str, osm_file_path: Path = None) -> bool:
    """
    Determine if streaming mode should be used for a country.

    Args:
        country_name: Name of the country
        osm_file_path: Optional path to OSM file for size estimation

    Returns:
        True if streaming mode should be used
    """
    # Always use streaming for known large countries
    if country_name in LARGE_COUNTRIES:
        return True

    # If we have the OSM file, estimate based on size
    if osm_file_path and osm_file_path.exists():
        estimated_segments = estimate_segment_count(osm_file_path)
        # Use streaming if estimated > 5 million segments
        return estimated_segments > 5_000_000

    return False


def get_memory_limit_mb() -> int:
    """
    Get the recommended memory limit for processing.

    Returns:
        Memory limit in MB
    """
    import psutil

    # Get available memory
    available_mb = psutil.virtual_memory().available / (1024 * 1024)

    # Use 70% of available memory, max 8GB
    recommended = min(int(available_mb * 0.7), 8192)

    # Minimum 2GB
    return max(recommended, 2048)


def format_region_info(region_name: str, bbox: Tuple[float, float, float, float]) -> str:
    """
    Format region information for display.

    Args:
        region_name: Name of the region
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

    Returns:
        Formatted string
    """
    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        return (f"{region_name}\n"
                f"  Bounding box: ({min_lat:.1f}°, {min_lon:.1f}°) to ({max_lat:.1f}°, {max_lon:.1f}°)")
    else:
        return region_name


def suggest_split_strategy(segment_count: int, memory_mb: int) -> str:
    """
    Suggest a processing strategy based on segment count and available memory.

    Args:
        segment_count: Estimated number of segments
        memory_mb: Available memory in MB

    Returns:
        Suggested strategy description
    """
    segments_per_gb = 1_000_000  # Rough estimate
    required_gb = segment_count / segments_per_gb
    available_gb = memory_mb / 1024

    if required_gb <= available_gb * 0.5:
        return "Standard processing (sufficient memory)"
    elif required_gb <= available_gb:
        return "Streaming mode recommended (near memory limit)"
    else:
        regions_needed = int(required_gb / (available_gb * 0.7)) + 1
        return f"Split into {regions_needed} regions (exceeds memory limit)"


# Export functions
__all__ = [
    'LARGE_COUNTRIES',
    'should_split_country',
    'get_country_regions',
    'estimate_segment_count',
    'should_use_streaming_mode',
    'get_memory_limit_mb',
    'format_region_info',
    'suggest_split_strategy'
]