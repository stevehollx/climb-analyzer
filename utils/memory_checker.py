#!/usr/bin/env python3
"""
Memory checker for climb analyzer.

Automatically checks if there's enough memory to analyze a region and provides
split recommendations if needed. Also provides merge instructions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import psutil
import math


# Memory estimation constants based on empirical data
SEGMENTS_PER_MB_OSM = 2000  # ~2000 road segments per MB of OSM file
MEMORY_PER_MILLION_SEGMENTS_GB = 1.5  # ~1.5GB RAM per million segments
SAFETY_FACTOR = 1.3  # Add 30% safety margin
MIN_FREE_MEMORY_GB = 2.0  # Always keep 2GB free for system
CONTAINER_OVERHEAD_GB = 1.0  # OpenTopoData + system overhead


# Density multipliers for different countries/regions
DENSITY_MULTIPLIERS = {
    # Dense road networks
    "france": 1.5,
    "germany": 1.4,
    "united-kingdom": 1.3,
    "netherlands": 1.6,
    "belgium": 1.5,
    "japan": 1.4,
    "south-korea": 1.4,
    "taiwan": 1.3,

    # Medium density
    "united-states": 1.2,
    "italy": 1.2,
    "spain": 1.1,
    "poland": 1.1,
    "china": 1.3,
    "india": 1.3,
    "mexico": 1.0,

    # Sparse networks
    "canada": 0.6,
    "russia": 0.5,
    "australia": 0.7,
    "brazil": 0.8,
    "argentina": 0.7,
    "sweden": 0.8,
    "norway": 0.7,
    "finland": 0.7,
    "iceland": 0.5,

    # Default
    "_default": 1.0
}


# Pre-defined splits for large countries
COUNTRY_SPLITS = {
    "France": [
        ("France-North", (46.0, -5.0, 51.2, 9.6), "North of Lyon (Paris, Normandy, Brittany)"),
        ("France-South", (41.3, -5.0, 46.0, 9.6), "South of Lyon (Provence, Alps, Pyrenees)")
    ],
    "Germany": [
        ("Germany-North", (51.0, 5.9, 55.1, 15.0), "North of Cologne (Hamburg, Berlin)"),
        ("Germany-South", (47.3, 5.9, 51.0, 15.0), "South of Cologne (Munich, Stuttgart)")
    ],
    "United States": [
        ("USA-Northeast", (38.0, -83.0, 48.0, -67.0), "NY, PA, New England"),
        ("USA-Southeast", (24.0, -90.0, 38.0, -75.0), "FL to VA"),
        ("USA-Midwest", (36.0, -105.0, 49.0, -83.0), "Great Lakes to Plains"),
        ("USA-Southwest", (28.0, -125.0, 42.0, -102.0), "CA, AZ, NV, NM"),
        ("USA-Northwest", (42.0, -125.0, 49.0, -102.0), "WA, OR, ID, MT"),
        ("USA-Alaska", (51.0, -180.0, 72.0, -130.0), "Alaska"),
        ("USA-Hawaii", (18.0, -161.0, 23.0, -154.0), "Hawaii")
    ],
    "Canada": [
        ("Canada-West", (48.0, -141.0, 70.0, -110.0), "BC, Yukon, NW Territories"),
        ("Canada-Prairies", (48.0, -110.0, 70.0, -95.0), "Alberta, Saskatchewan, Manitoba"),
        ("Canada-Central", (41.0, -95.0, 55.0, -74.0), "Ontario"),
        ("Canada-East", (44.0, -80.0, 55.0, -52.0), "Quebec, Maritimes"),
        ("Canada-North", (55.0, -141.0, 83.0, -52.0), "Northern territories")
    ],
    "Russia": [
        ("Russia-West", (41.0, 19.0, 82.0, 60.0), "European Russia"),
        ("Russia-Ural", (50.0, 60.0, 70.0, 70.0), "Ural region"),
        ("Russia-Siberia", (50.0, 70.0, 75.0, 110.0), "Western Siberia"),
        ("Russia-FarEast", (42.0, 110.0, 75.0, 180.0), "Far East")
    ],
    "China": [
        ("China-North", (35.0, 73.0, 54.0, 135.0), "Beijing, Shanghai, Xi'an"),
        ("China-South", (18.0, 73.0, 35.0, 135.0), "Guangzhou, Shenzhen, Hong Kong")
    ],
    "Brazil": [
        ("Brazil-North", (-34.0, -74.0, -15.0, -34.0), "Amazon and Northeast"),
        ("Brazil-South", (-15.0, -74.0, 5.0, -34.0), "São Paulo, Rio, South")
    ],
    "India": [
        ("India-North", (23.0, 68.0, 36.0, 97.0), "Delhi, Mumbai, Himalayas"),
        ("India-South", (8.0, 68.0, 23.0, 97.0), "Bangalore, Chennai, Kerala")
    ],
    "Australia": [
        ("Australia-West", (-35.0, 112.0, -13.0, 129.0), "Western Australia"),
        ("Australia-Central", (-35.0, 129.0, -13.0, 141.0), "NT and SA"),
        ("Australia-East", (-44.0, 141.0, -10.0, 154.0), "Sydney, Melbourne, Brisbane")
    ]
}


def get_available_memory_gb() -> float:
    """
    Get available system memory in GB.

    Returns:
        Available memory in GB
    """
    return psutil.virtual_memory().available / (1024 ** 3)


def get_total_memory_gb() -> float:
    """
    Get total system memory in GB.

    Returns:
        Total memory in GB
    """
    return psutil.virtual_memory().total / (1024 ** 3)


def estimate_segments_from_osm_file(osm_file_path: Path) -> int:
    """
    Estimate number of road segments from OSM file size.

    Args:
        osm_file_path: Path to OSM .pbf file

    Returns:
        Estimated number of segments
    """
    if not osm_file_path.exists():
        return 0

    # Get file size in MB
    file_size_mb = osm_file_path.stat().st_size / (1024 * 1024)

    # Extract country name from filename
    country = osm_file_path.stem.replace("-latest", "").replace("_", " ").replace("-", " ")

    # Get density multiplier
    multiplier = DENSITY_MULTIPLIERS.get(country.lower(), DENSITY_MULTIPLIERS["_default"])

    # Estimate segments
    segments = int(file_size_mb * SEGMENTS_PER_MB_OSM * multiplier)

    return segments


def estimate_segments_from_radius(center_lat: float, radius_km: float,
                                 country: str = None) -> int:
    """
    Estimate number of segments from search radius.

    Args:
        center_lat: Center latitude
        radius_km: Search radius in kilometers
        country: Optional country name for density adjustment

    Returns:
        Estimated number of segments
    """
    # Approximate area
    area_km2 = math.pi * (radius_km ** 2)

    # Base segments per km2 (varies by latitude and urbanization)
    # Higher latitudes typically have lower road density
    lat_factor = 1.0 - (abs(center_lat) / 90.0) * 0.5  # Reduce by up to 50% at poles

    # Base density: ~50 segments per km2 in moderate areas
    base_density = 50 * lat_factor

    # Apply country-specific multiplier
    if country:
        multiplier = DENSITY_MULTIPLIERS.get(country.lower(), 1.0)
    else:
        multiplier = 1.0

    segments = int(area_km2 * base_density * multiplier)

    # Apply radius-based adjustments (larger areas tend to include more rural areas)
    if radius_km <= 25:
        segments *= 1.2  # Small radius - likely urban
    elif radius_km <= 50:
        segments *= 1.0  # Medium radius - mixed
    elif radius_km <= 100:
        segments *= 0.8  # Large radius - includes rural
    else:
        segments *= 0.6  # Very large - mostly rural

    return segments


def estimate_memory_required_gb(segments: int) -> float:
    """
    Estimate memory required for processing segments.

    Args:
        segments: Number of road segments

    Returns:
        Estimated memory requirement in GB
    """
    memory_gb = (segments / 1_000_000) * MEMORY_PER_MILLION_SEGMENTS_GB
    memory_gb *= SAFETY_FACTOR  # Add safety margin
    memory_gb += CONTAINER_OVERHEAD_GB  # Add container overhead

    return memory_gb


def check_memory_sufficient(required_gb: float, available_gb: float) -> Tuple[bool, str]:
    """
    Check if available memory is sufficient.

    Args:
        required_gb: Required memory in GB
        available_gb: Available memory in GB

    Returns:
        Tuple of (is_sufficient, message)
    """
    usable_gb = available_gb - MIN_FREE_MEMORY_GB

    if usable_gb <= 0:
        return False, f"Insufficient memory: Only {available_gb:.1f}GB available, need {MIN_FREE_MEMORY_GB:.1f}GB minimum"

    if required_gb <= usable_gb:
        margin_pct = ((usable_gb - required_gb) / required_gb) * 100
        return True, f"Sufficient memory: {required_gb:.1f}GB required, {usable_gb:.1f}GB available ({margin_pct:.0f}% margin)"
    else:
        deficit_gb = required_gb - usable_gb
        return False, f"Insufficient memory: {required_gb:.1f}GB required, only {usable_gb:.1f}GB available (need {deficit_gb:.1f}GB more)"


def get_radius_category(radius_km: float) -> str:
    """
    Categorize search radius.

    Args:
        radius_km: Radius in kilometers

    Returns:
        Category string
    """
    if radius_km <= 25:
        return "small"
    elif radius_km <= 50:
        return "medium"
    elif radius_km <= 100:
        return "large"
    else:
        return "very_large"


def recommend_splits(country: str, required_gb: float, available_gb: float) -> List[Dict]:
    """
    Recommend region splits for a country.

    Args:
        country: Country name
        required_gb: Required memory in GB
        available_gb: Available memory in GB

    Returns:
        List of recommended splits with commands
    """
    usable_gb = available_gb - MIN_FREE_MEMORY_GB

    # Check if country has pre-defined splits
    if country in COUNTRY_SPLITS:
        splits = []
        for name, bbox, description in COUNTRY_SPLITS[country]:
            min_lat, min_lon, max_lat, max_lon = bbox
            command = f"./climb-analyzer -b {min_lat},{min_lon},{max_lat},{max_lon} --name \"{name}\""
            splits.append({
                "name": name,
                "description": description,
                "command": command,
                "bbox": bbox
            })
        return splits

    # Generic split recommendation
    num_splits = math.ceil(required_gb / (usable_gb * 0.7))  # Use 70% of available

    if num_splits == 2:
        return [{
            "name": f"{country}-North",
            "description": "Northern region",
            "command": f"# Split {country} at middle latitude",
            "bbox": None
        }, {
            "name": f"{country}-South",
            "description": "Southern region",
            "command": f"# Split {country} at middle latitude",
            "bbox": None
        }]
    else:
        splits = []
        for i in range(num_splits):
            splits.append({
                "name": f"{country}-Region{i+1}",
                "description": f"Region {i+1} of {num_splits}",
                "command": f"# Divide {country} into {num_splits} regions",
                "bbox": None
            })
        return splits


def generate_merge_instructions(country: str, splits: List[Dict]) -> str:
    """
    Generate instructions for merging split analyses.

    NOTE: This function is deprecated. Use offer_automatic_merge() instead,
    which provides automatic merging after subregion analysis completes.

    Args:
        country: Country name
        splits: List of split regions

    Returns:
        Merge instructions as string
    """
    if len(splits) < 2:
        return ""

    try:
        from climb_analyzer.utils.formatting import print_header, print_info, Colors

        # Build instructions in memory for return
        instructions = []
        instructions.append("\n")

        # Use modern formatter for header
        header_lines = []
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        print_header("Merging Instructions", spacing_before=0)
        header_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        instructions.append(header_output)
        instructions.append("After analyzing all regions, they will be automatically merged.")
        instructions.append("")

        # Generate merge info
        instructions.append("The system will:")
        instructions.append(f"  {Colors.CYAN}•{Colors.RESET} Identify climbs that cross region boundaries")
        instructions.append(f"  {Colors.CYAN}•{Colors.RESET} Combine their metrics (length, elevation gain)")
        instructions.append(f"  {Colors.CYAN}•{Colors.RESET} Update files with complete climb data")
        instructions.append(f"  {Colors.CYAN}•{Colors.RESET} Create backup files before making changes")

        instructions.append("")
        if len(splits) == 2:
            instructions.append(f"Regions to merge: {splits[0]['name']} + {splits[1]['name']}")
        else:
            instructions.append(f"Regions to merge: {len(splits)} subregions in sequence")

        instructions.append("")
        instructions.append(f"{Colors.GREEN}✓{Colors.RESET} Regions will be automatically merged after analysis completes")

        return "\n".join(instructions)

    except ImportError:
        # Fallback to simple formatting if import fails
        instructions = []
        instructions.append("\n" + "="*80)
        instructions.append("MERGING INSTRUCTIONS")
        instructions.append("="*80)
        instructions.append("")
        instructions.append("After analyzing all regions, they will be automatically merged.")
        instructions.append("")
        instructions.append("The system will:")
        instructions.append("  • Identify climbs that cross region boundaries")
        instructions.append("  • Combine their metrics (length, elevation gain)")
        instructions.append("  • Update files with complete climb data")
        instructions.append("  • Create backup files before making changes")
        instructions.append("")

        if len(splits) == 2:
            instructions.append(f"Regions to merge: {splits[0]['name']} + {splits[1]['name']}")
        else:
            instructions.append(f"Regions to merge: {len(splits)} subregions in sequence")

        instructions.append("")
        instructions.append("✓ Regions will be automatically merged after analysis completes")

        return "\n".join(instructions)


def print_memory_report(segments: int, required_gb: float, available_gb: float,
                        is_sufficient: bool, message: str):
    """
    Print formatted memory report.

    Args:
        segments: Estimated number of segments
        required_gb: Required memory in GB
        available_gb: Available memory in GB
        is_sufficient: Whether memory is sufficient
        message: Status message
    """
    try:
        from climb_analyzer.utils.formatting import print_header, print_key_value

        print_header("Memory Analysis", spacing_before=1)
        print_key_value("Estimated segments", f"{segments:,}", indent=2)
        print_key_value("Memory required", f"{required_gb:.1f} GB", indent=2)
        print_key_value("Memory available", f"{available_gb:.1f} GB", indent=2)
        print_key_value("System total", f"{get_total_memory_gb():.1f} GB", indent=2)
        print()

        if is_sufficient:
            print("  ✅", message)
        else:
            print("  ⚠️ ", message)
    except ImportError:
        # Fallback to simple formatting if import fails
        print("\n" + "="*80)
        print("MEMORY ANALYSIS")
        print("="*80)
        print()
        print(f"Estimated segments:     {segments:,}")
        print(f"Memory required:        {required_gb:.1f} GB")
        print(f"Memory available:       {available_gb:.1f} GB")
        print(f"System total:          {get_total_memory_gb():.1f} GB")
        print()

        if is_sufficient:
            print("✅", message)
        else:
            print("⚠️ ", message)


def check_memory_for_region(country: str, osm_file_path: Optional[Path] = None) -> Dict:
    """
    Check memory requirements for analyzing a country/region.

    Args:
        country: Country/region name
        osm_file_path: Optional path to OSM file

    Returns:
        Dictionary with memory check results
    """
    # Find OSM file if not provided
    if not osm_file_path:
        planet_dir = Path("data/planet_osm_data")
        if planet_dir.exists():
            pattern = f"{country.lower().replace(' ', '-')}*.pbf"
            files = list(planet_dir.glob(pattern))
            if files:
                osm_file_path = files[0]

    # Estimate segments
    if osm_file_path and osm_file_path.exists():
        segments = estimate_segments_from_osm_file(osm_file_path)
    else:
        # Fallback estimates for known countries
        fallback_estimates = {
            "france": 8_000_000,
            "germany": 7_000_000,
            "united states": 15_000_000,
            "canada": 5_000_000,
            "russia": 8_000_000,
            "china": 10_000_000,
            "brazil": 6_000_000,
            "india": 8_000_000,
            "australia": 4_000_000,
            "united kingdom": 5_000_000,
            "italy": 4_500_000,
            "spain": 4_000_000,
            "japan": 5_000_000
        }
        segments = fallback_estimates.get(country.lower(), 2_000_000)

    # Calculate memory requirements
    required_gb = estimate_memory_required_gb(segments)
    available_gb = get_available_memory_gb()

    # Check if sufficient
    is_sufficient, message = check_memory_sufficient(required_gb, available_gb)

    # Generate recommendations if needed
    splits = []
    merge_instructions = ""

    if not is_sufficient:
        splits = recommend_splits(country, required_gb, available_gb)
        merge_instructions = generate_merge_instructions(country, splits)

    return {
        "country": country,
        "segments": segments,
        "required_gb": required_gb,
        "available_gb": available_gb,
        "is_sufficient": is_sufficient,
        "message": message,
        "splits": splits,
        "merge_instructions": merge_instructions,
        "osm_file": str(osm_file_path) if osm_file_path else None
    }


def check_memory_for_radius(center_lat: float, center_lon: float, radius_km: float,
                           country: str = None) -> Dict:
    """
    Check memory requirements for radius-based analysis.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Search radius in kilometers
        country: Optional country name

    Returns:
        Dictionary with memory check results
    """
    # Estimate segments
    segments = estimate_segments_from_radius(center_lat, radius_km, country)

    # Calculate memory requirements
    required_gb = estimate_memory_required_gb(segments)
    available_gb = get_available_memory_gb()

    # Check if sufficient
    is_sufficient, message = check_memory_sufficient(required_gb, available_gb)

    # Get radius category
    category = get_radius_category(radius_km)

    # Generate recommendations if needed
    recommendations = []

    if not is_sufficient:
        if radius_km > 100:
            recommendations.append("Consider reducing radius to 100km or less")
        elif radius_km > 50:
            recommendations.append("Consider reducing radius to 50km")
        elif radius_km > 25:
            recommendations.append("Consider reducing radius to 25km")
        else:
            recommendations.append("This area is very dense. Consider analyzing a specific neighborhood")

        # Suggest splitting into quadrants for large radius
        if radius_km > 50:
            recommendations.append("Or split into 4 quadrants (NE, NW, SE, SW)")

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": radius_km,
        "radius_category": category,
        "segments": segments,
        "required_gb": required_gb,
        "available_gb": available_gb,
        "is_sufficient": is_sufficient,
        "message": message,
        "recommendations": recommendations
    }


def print_split_guidance(result: Dict):
    """
    Print split guidance for insufficient memory.

    Args:
        result: Memory check result dictionary
    """
    if result.get("is_sufficient", False):
        return

    try:
        from climb_analyzer.utils.formatting import print_header

        print_header("Recommended Splits", spacing_before=1)

        if result.get("splits"):
            print(f"  Split {result['country']} into {len(result['splits'])} regions:")
            print()

            for i, split in enumerate(result["splits"], 1):
                print(f"  {i}. {split['name']}")
                print(f"     Description: {split['description']}")
                print(f"     Command: {split['command']}")
                print()

        if result.get("merge_instructions"):
            print(result["merge_instructions"])
    except ImportError:
        # Fallback to simple formatting
        print("\n" + "="*80)
        print("RECOMMENDED SPLITS")
        print("="*80)
        print()

        if result.get("splits"):
            print(f"Split {result['country']} into {len(result['splits'])} regions:")
            print()

            for i, split in enumerate(result["splits"], 1):
                print(f"{i}. {split['name']}")
                print(f"   Description: {split['description']}")
                print(f"   Command: {split['command']}")
                print()

        if result.get("merge_instructions"):
            print(result["merge_instructions"])


# Export functions
__all__ = [
    'check_memory_for_region',
    'check_memory_for_radius',
    'get_available_memory_gb',
    'get_total_memory_gb',
    'estimate_segments_from_osm_file',
    'estimate_segments_from_radius',
    'estimate_memory_required_gb',
    'print_memory_report',
    'print_split_guidance'
]


if __name__ == "__main__":
    # Test with command line arguments
    if len(sys.argv) > 1:
        country = sys.argv[1]
        result = check_memory_for_region(country)
        print_memory_report(
            result["segments"],
            result["required_gb"],
            result["available_gb"],
            result["is_sufficient"],
            result["message"]
        )
        if not result["is_sufficient"]:
            print_split_guidance(result)
    else:
        print("Usage: python memory_checker.py <country>")
        print("Example: python memory_checker.py France")