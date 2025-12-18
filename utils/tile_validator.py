#!/usr/bin/env python3
"""
Tile-level elevation data validator.

Checks if specific elevation tiles exist for a given bounding box,
not just if dataset directories exist.
"""

import math
from pathlib import Path
from typing import List, Tuple


def get_required_ned_tiles(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[str]:
    """
    Get list of required NED tile filenames for bounding box.

    NED tiles are 1-degree squares named by their lower-left corner (SRTM format).
    After renaming, NED files use the same convention as SRTM.
    Example: USGS_13_n20w157.tif covers 20°N-21°N, 156°W-157°W

    Note: Original NED tiles use upper-left corner naming, but we rename them
    to lower-left corner format (subtracting 1 from latitude) for OpenTopoData.

    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box

    Returns:
        List of required tile filenames (in SRTM-compatible format)
    """
    tiles = []

    # NED tiles are named by lower-left corner after renaming (floor of coordinates)
    lat_start = int(math.floor(min_lat))
    lat_end = int(math.floor(max_lat))
    lon_start = int(math.floor(min_lon))
    lon_end = int(math.floor(max_lon))

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            # NED naming after conversion: n##w### or s##e### (lowercase, SRTM-compatible)
            lat_prefix = 'n' if lat >= 0 else 's'
            lon_prefix = 'w' if lon < 0 else 'e'

            lat_abs = abs(lat)
            lon_abs = abs(lon)

            # Standard filename format (SRTM-compatible after renaming)
            filename = f"USGS_13_{lat_prefix}{lat_abs:02d}{lon_prefix}{lon_abs:03d}.tif"
            tiles.append(filename)

    return tiles


def get_required_srtm_tiles(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[str]:
    """
    Get list of required SRTM tile filenames for bounding box.

    SRTM tiles are 1-degree squares named by their northwest corner.
    Example: N20W157.tif covers 20°N-21°N, 156°W-157°W

    Note: This project stores SRTM tiles as .tif files, not .hgt files.

    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box

    Returns:
        List of required tile filenames
    """
    tiles = []

    # SRTM tiles are named by northwest corner
    # For southern latitudes, use floor; for northern, use floor
    lat_start = int(math.floor(min_lat))
    lat_end = int(math.floor(max_lat))
    lon_start = int(math.floor(min_lon))
    lon_end = int(math.floor(max_lon))

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            # SRTM naming: N##W### or S##E###
            lat_prefix = 'N' if lat >= 0 else 'S'
            lon_prefix = 'W' if lon < 0 else 'E'

            lat_abs = abs(lat)
            lon_abs = abs(lon)

            # Use .tif extension (project stores SRTM as GeoTIFF, not raw .hgt)
            filename = f"{lat_prefix}{lat_abs:02d}{lon_prefix}{lon_abs:03d}.tif"
            tiles.append(filename)

    return tiles


def get_required_aster_tiles(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[str]:
    """
    Get list of required ASTER tile filenames for bounding box.

    ASTER tiles are 1-degree squares.
    Example: ASTGTMV003_N20W157_dem.tif

    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box

    Returns:
        List of required tile filenames
    """
    tiles = []

    lat_start = int(math.floor(min_lat))
    lat_end = int(math.floor(max_lat))
    lon_start = int(math.floor(min_lon))
    lon_end = int(math.floor(max_lon))

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            lat_prefix = 'N' if lat >= 0 else 'S'
            lon_prefix = 'W' if lon < 0 else 'E'

            lat_abs = abs(lat)
            lon_abs = abs(lon)

            filename = f"ASTGTMV003_{lat_prefix}{lat_abs:02d}{lon_prefix}{lon_abs:03d}_dem.tif"
            tiles.append(filename)

    return tiles


def get_required_aw3d30_tiles(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[str]:
    """
    Get list of required AW3D30 tile filenames for bounding box.

    AW3D30 tiles are 1-degree squares.
    Example: N020W157.tif

    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box

    Returns:
        List of required tile filenames
    """
    tiles = []

    lat_start = int(math.floor(min_lat))
    lat_end = int(math.floor(max_lat))
    lon_start = int(math.floor(min_lon))
    lon_end = int(math.floor(max_lon))

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            lat_prefix = 'N' if lat >= 0 else 'S'
            lon_prefix = 'W' if lon < 0 else 'E'

            lat_abs = abs(lat)
            lon_abs = abs(lon)

            filename = f"{lat_prefix}{lat_abs:03d}{lon_prefix}{lon_abs:03d}.tif"
            tiles.append(filename)

    return tiles


def check_tiles_exist(dataset_dir: Path, required_tiles: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Check which required tiles exist in the dataset directory.

    Excludes tiles listed in .unavailable file (water/ocean areas with no data).

    Args:
        dataset_dir: Path to dataset directory
        required_tiles: List of required tile filenames

    Returns:
        Tuple of (existing_tiles, missing_tiles, unavailable_tiles)
    """
    import re

    # Load unavailable tiles (water/ocean areas) from .unavailable file
    unavailable_tile_set = set()
    unavailable_file = dataset_dir / ".unavailable"
    if unavailable_file.exists():
        with open(unavailable_file, 'r') as f:
            # File contains tile names without extension (e.g., "N18W155" or "n18w155")
            for line in f:
                tile_name = line.strip()
                if tile_name:
                    unavailable_tile_set.add(tile_name.upper())

    existing = []
    missing = []
    unavailable = []

    for tile in required_tiles:
        # Extract tile coordinates from filename (without extension)
        # Handle formats: USGS_13_n20w157.tif, N20W157.hgt, etc.
        tile_base = tile.rsplit('.', 1)[0]  # Remove extension
        tile_ext = tile.rsplit('.', 1)[1] if '.' in tile else ''

        # Extract coordinate part (N##W### or n##w###)
        coords_match = re.search(r'([NS]\d{2,3}[EW]\d{2,3})', tile_base, re.IGNORECASE)
        if coords_match:
            coords = coords_match.group(1).upper()
            if coords in unavailable_tile_set:
                # This tile is known to be unavailable (water/ocean)
                unavailable.append(tile)
                continue

        # Check for tile existence - try both USGS-prefixed and simplified formats
        tile_path = dataset_dir / tile
        if tile_path.exists():
            existing.append(tile)
        elif coords_match and tile_ext:
            # Try simplified format (e.g., n20w157.tif instead of USGS_13_n20w157.tif)
            simplified_name = f"{coords_match.group(1).lower()}.{tile_ext}"
            simplified_path = dataset_dir / simplified_name
            if simplified_path.exists():
                existing.append(tile)
            else:
                missing.append(tile)
        else:
            missing.append(tile)

    return existing, missing, unavailable


def validate_dataset_tiles(
    dataset_name: str,
    elevation_dir: Path,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Validate that all required tiles exist for a dataset and region.

    Args:
        dataset_name: Name of dataset (ned10m, srtm30m, aster30m, aw3d30)
        elevation_dir: Base elevation_data directory
        min_lat, max_lat, min_lon, max_lon: Bounding box

    Returns:
        Tuple of (has_all_tiles, existing_tiles, missing_tiles, unavailable_tiles)
    """
    dataset_dir = elevation_dir / dataset_name

    if not dataset_dir.exists():
        return False, [], [], []

    # Get required tiles based on dataset type
    if dataset_name == "ned10m":
        required_tiles = get_required_ned_tiles(min_lat, max_lat, min_lon, max_lon)
    elif dataset_name == "srtm30m":
        required_tiles = get_required_srtm_tiles(min_lat, max_lat, min_lon, max_lon)
    elif dataset_name == "aster30m":
        required_tiles = get_required_aster_tiles(min_lat, max_lat, min_lon, max_lon)
    elif dataset_name == "aw3d30":
        required_tiles = get_required_aw3d30_tiles(min_lat, max_lat, min_lon, max_lon)
    else:
        # Unknown dataset type
        return False, [], [], []

    existing, missing, unavailable = check_tiles_exist(dataset_dir, required_tiles)

    has_all = len(missing) == 0
    return has_all, existing, missing, unavailable


def check_region_elevation_coverage(
    elevation_dir: Path,
    datasets: List[str],
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> Tuple[bool, dict]:
    """
    Check elevation coverage for all datasets in a region.

    Args:
        elevation_dir: Path to elevation_data directory
        datasets: List of dataset names to check
        min_lat, max_lat, min_lon, max_lon: Bounding box

    Returns:
        Tuple of (has_complete_coverage, coverage_report)
        coverage_report = {
            'ned10m': {'has_all': True, 'existing': [...], 'missing': [], 'unavailable': [...], 'total_required': N},
            ...
        }
    """
    report = {}
    has_any_complete = False

    for dataset in datasets:
        has_all, existing, missing, unavailable = validate_dataset_tiles(
            dataset, elevation_dir, min_lat, max_lat, min_lon, max_lon
        )

        report[dataset] = {
            'has_all': has_all,
            'existing': existing,
            'missing': missing,
            'unavailable': unavailable,
            'total_required': len(existing) + len(missing) + len(unavailable)
        }

        if has_all:
            has_any_complete = True

    return has_any_complete, report


if __name__ == "__main__":
    # Test with Hawaii (Maui area)
    import sys

    elevation_dir = Path("data/elevation_data")

    # Maui bounding box
    min_lat, max_lat = 20.5, 21.0
    min_lon, max_lon = -156.7, -155.9

    print("Testing tile validation for Maui, Hawaii")
    print(f"Bounding box: {min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E\n")

    # Test NED
    print("=== NED10M ===")
    ned_tiles = get_required_ned_tiles(min_lat, max_lat, min_lon, max_lon)
    print(f"Required tiles: {ned_tiles}")

    has_all, existing, missing, unavailable = validate_dataset_tiles(
        "ned10m", elevation_dir, min_lat, max_lat, min_lon, max_lon
    )
    print(f"Has all tiles: {has_all}")
    print(f"Existing: {existing}")
    print(f"Missing: {missing}")
    print(f"Unavailable: {unavailable}\n")

    # Test SRTM
    print("=== SRTM30M ===")
    srtm_tiles = get_required_srtm_tiles(min_lat, max_lat, min_lon, max_lon)
    print(f"Required tiles: {srtm_tiles}")

    has_all, existing, missing, unavailable = validate_dataset_tiles(
        "srtm30m", elevation_dir, min_lat, max_lat, min_lon, max_lon
    )
    print(f"Has all tiles: {has_all}")
    print(f"Existing: {existing}")
    print(f"Missing: {missing}")
    print(f"Unavailable: {unavailable}")
