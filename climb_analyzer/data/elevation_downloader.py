#!/usr/bin/env python3
"""
Elevation dataset downloader for OpenTopoData.

Automatically downloads required elevation tiles for a region before analysis starts.
Supports SRTM 30m (global coverage, free) as the primary dataset.
"""

import os
import math
import subprocess
from pathlib import Path
from typing import List, Tuple, Set
from tqdm import tqdm

from climb_analyzer.utils.formatting import (
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_key_value
)


def get_required_srtm_tiles(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> List[str]:
    """
    Calculate which SRTM 1° tiles are needed for a bounding box.

    SRTM tiles are named like: N37W123.hgt (1° x 1° tiles)

    Args:
        min_lat: Minimum latitude
        min_lon: Minimum longitude
        max_lat: Maximum latitude
        max_lon: Maximum longitude

    Returns:
        List of tile names (e.g., ['N37W123', 'N37W122', ...])
    """
    tiles = []

    # SRTM only covers -60° to 60° latitude
    min_lat_clipped = max(-60, min_lat)
    max_lat_clipped = min(60, max_lat)

    # Calculate tile coordinates (floor for lat/lon to get 1° tile boundaries)
    lat_start = int(math.floor(min_lat_clipped))
    lat_end = int(math.floor(max_lat_clipped))
    lon_start = int(math.floor(min_lon))
    lon_end = int(math.floor(max_lon))

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            # Format tile name
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tile_name = f"{lat_str}{lon_str}"
            tiles.append(tile_name)

    return tiles


def check_existing_tiles(tiles: List[str], dataset_dir: Path) -> Tuple[Set[str], Set[str]]:
    """
    Check which tiles already exist in the dataset directory.

    Args:
        tiles: List of tile names to check
        dataset_dir: Path to dataset directory

    Returns:
        Tuple of (existing_tiles, missing_tiles)
    """
    existing = set()
    missing = set()

    for tile in tiles:
        tile_file = dataset_dir / f"{tile}.hgt"
        if tile_file.exists():
            existing.add(tile)
        else:
            missing.add(tile)

    return existing, missing


def download_srtm_tiles(tiles: List[str], dataset_dir: Path, dry_run: bool = False) -> bool:
    """
    Download SRTM 30m tiles using OpenTopoData's eio CLI tool.

    The eio tool (elevation.io) can download SRTM data from public sources.

    Args:
        tiles: List of tile names to download
        dataset_dir: Path to dataset directory
        dry_run: If True, only show what would be downloaded

    Returns:
        True if successful, False otherwise
    """
    if not tiles:
        return True

    # Ensure dataset directory exists
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print_info("DRY RUN - would download:")
        for tile in sorted(tiles):
            print(f"  • {tile}.hgt")
        return True

    print_info(f"Downloading {len(tiles)} SRTM tiles using eio CLI...")
    print_info("(This may take several minutes depending on tile count)")
    print()

    # Download using eio CLI
    # eio clip -o output.tif --bounds lat_min lon_min lat_max lon_max
    # For individual tiles, we'll use the tile boundaries

    success_count = 0
    failed_tiles = []

    with tqdm(total=len(tiles), desc="Downloading tiles", unit="tile") as pbar:
        for tile in tiles:
            try:
                # Parse tile name to get bounds
                # Format: N37W123 → lat=37, lon=-123
                lat_str = tile[:3]  # N37 or S37
                lon_str = tile[3:]  # W123 or E123

                lat = int(lat_str[1:])
                if lat_str[0] == 'S':
                    lat = -lat

                lon = int(lon_str[1:])
                if lon_str[0] == 'W':
                    lon = -lon

                # Tile bounds (1° x 1° tile)
                output_file = dataset_dir / f"{tile}.hgt"

                # Use eio to download the tile
                # eio clip -o output.tif --bounds lat_min lon_min lat_max lon_max
                cmd = [
                    'eio',
                    'clip',
                    '-o', str(output_file),
                    '--bounds', str(lat), str(lon), str(lat + 1), str(lon + 1),
                    '--product', 'srtm'
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                success_count += 1
                pbar.update(1)

            except subprocess.CalledProcessError as e:
                failed_tiles.append(tile)
                pbar.write(f"⚠️  Failed to download {tile}: {e.stderr[:100]}")
                pbar.update(1)
            except Exception as e:
                failed_tiles.append(tile)
                pbar.write(f"⚠️  Error downloading {tile}: {e}")
                pbar.update(1)

    print()
    if success_count > 0:
        print_success(f"Successfully downloaded {success_count}/{len(tiles)} tiles")

    if failed_tiles:
        print_warning(f"Failed to download {len(failed_tiles)} tiles:")
        for tile in failed_tiles[:10]:
            print(f"  • {tile}")
        if len(failed_tiles) > 10:
            print(f"  ... and {len(failed_tiles) - 10} more")
        return False

    return True


def ensure_elevation_coverage(
    region_name: str,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    dataset: str = "srtm30m",
    auto_download: bool = True
) -> bool:
    """
    Ensure elevation data coverage exists for a region before analysis.

    This function:
    1. Calculates which tiles are needed for the region
    2. Checks which tiles already exist
    3. Downloads missing tiles (if auto_download=True)
    4. Reports coverage status

    Args:
        region_name: Name of the region (for display)
        min_lat: Minimum latitude
        min_lon: Minimum longitude
        max_lat: Maximum latitude
        max_lon: Maximum longitude
        dataset: Dataset name ('srtm30m', 'ned10m', etc.)
        auto_download: If True, automatically download missing tiles

    Returns:
        True if coverage is complete, False otherwise
    """
    print_header(f"Checking Elevation Data Coverage for {region_name}")

    # Determine dataset directory
    # Default to opentopodata/data/{dataset} directory relative to project root
    base_dir = Path(__file__).parent.parent.parent / "opentopodata" / "data" / dataset

    print_key_value("Dataset", dataset)
    print_key_value("Dataset directory", str(base_dir))
    print_key_value("Region bounds", f"lat: [{min_lat:.3f}, {max_lat:.3f}], lon: [{min_lon:.3f}, {max_lon:.3f}]")
    print()

    # Check if dataset is SRTM (only one we can auto-download currently)
    if dataset not in ['srtm30m', 'srtm']:
        print_warning(f"Auto-download not supported for {dataset}")
        print_info("Assuming data is already available")
        return True

    # Get required tiles
    required_tiles = get_required_srtm_tiles(min_lat, min_lon, max_lat, max_lon)
    print_info(f"Required tiles: {len(required_tiles)}")

    # Check which tiles exist
    existing_tiles, missing_tiles = check_existing_tiles(required_tiles, base_dir)

    print_key_value("  Existing", f"{len(existing_tiles)} tiles")
    print_key_value("  Missing", f"{len(missing_tiles)} tiles")
    print()

    if not missing_tiles:
        print_success("All required elevation tiles are available")
        return True

    # Handle missing tiles
    if not auto_download:
        print_error(f"Missing {len(missing_tiles)} elevation tiles and auto-download is disabled")
        print_info("Missing tiles:")
        for tile in sorted(list(missing_tiles)[:20]):
            print(f"  • {tile}")
        if len(missing_tiles) > 20:
            print(f"  ... and {len(missing_tiles) - 20} more")
        return False

    # Download missing tiles
    print_info(f"Downloading {len(missing_tiles)} missing tiles...")
    print()

    success = download_srtm_tiles(list(missing_tiles), base_dir, dry_run=False)

    if success:
        print()
        print_success("Elevation data coverage is complete")
        return True
    else:
        print()
        print_error("Some tiles failed to download")
        print_warning("Analysis may have incomplete elevation data")
        return False


def check_elevation_tools() -> bool:
    """
    Check if required elevation tools (eio CLI) are installed.

    Returns:
        True if tools are available, False otherwise
    """
    try:
        result = subprocess.run(
            ['eio', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
