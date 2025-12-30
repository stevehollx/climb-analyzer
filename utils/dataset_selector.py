#!/usr/bin/env python3
"""
Smart dataset selection for different geographic regions.

Auto-detects appropriate DEM datasets based on latitude/longitude.
"""

from typing import List, Tuple


def get_required_datasets(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    deployment_type: str = 'local'
) -> List[str]:
    """
    Auto-detect required DEM datasets for a geographic region.

    Dataset coverage:
    - NED 10m: US only (18°N to 72°N, -180° to -60°W)
    - SRTM 30m: 60°N to 56°S (OpenTopography S3, no auth needed)
    - AW3D30: Global 84°N to 84°S (JAXA FTP, no auth needed)
    - ArcticDEM 32m: Arctic regions (>60°N) - local only
    - REMA 32m: Antarctica (<-60°S) - local only

    Note: ASTER is deprecated as of December 2025 (NASA LP DAAC retired).
    AW3D30 provides better coverage (84°N to 84°S) and better accuracy.

    Args:
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        deployment_type: 'cloud' or 'local' (default: 'local')

    Returns:
        List of dataset names in priority order
    """
    datasets = []

    # Antarctica (<= -60°S)
    if lat_max <= -60:
        if deployment_type == 'local':
            datasets.append('rema32m')
            return datasets  # REMA is sufficient for Antarctica in local mode
        else:
            # Cloud mode doesn't support Antarctica
            return []

    # Detect region type
    is_arctic = lat_min > 60 or lat_max > 60  # Full or partial Arctic coverage

    # US bounds: roughly 18°N to 72°N, -180°W to -60°W
    is_us_region = (lat_min >= 18 and lat_max <= 72 and
                    lon_min >= -180 and lon_max <= -60)

    # Alaska detection: US region with high latitude
    is_alaska = is_us_region and lat_max > 60

    # Greenland detection (approximate bounds: 59°N-84°N, 73°W-11°W)
    # Greenland is unique: very high latitude, western hemisphere, large area
    is_greenland = (lat_min >= 59 and lat_max >= 70 and
                   lon_min >= -74 and lon_max <= -11 and
                   lat_max - lat_min > 15)  # Greenland spans >15 degrees latitude

    # US (excluding Alaska)
    if is_us_region and not is_alaska:
        datasets.append('ned10m')
        datasets.append('srtm30m')
        return datasets

    # Alaska - US region with Arctic coverage
    if is_alaska:
        if deployment_type == 'local':
            datasets.append('ned10m')      # US high-res (complete AK coverage)
            datasets.append('arctic32m')   # Arctic coverage (>60°N)
            datasets.append('srtm30m')     # Mid-latitude backup (<60°N)
            # No aw3d30 - NED covers all US at higher resolution
        else:
            # Cloud mode: SRTM for parts within coverage
            datasets.append('srtm30m')
        return datasets

    # Greenland (special case)
    if is_greenland:
        if deployment_type == 'local':
            datasets.append('arctic32m')
            return datasets  # Only ArcticDEM for Greenland in local mode
        else:
            # Cloud mode doesn't support Greenland well
            return []

    # Arctic regions (Iceland, Scandinavia, northern Russia, northern Canada)
    if is_arctic:
        if deployment_type == 'local':
            datasets.append('arctic32m')
            # Add additional coverage for regions that extend below 60°N
            if lat_min < 60:
                # SRTM for parts below 60°N
                if lat_min >= -56:
                    datasets.append('srtm30m')
            # AW3D30 provides global coverage including high latitudes
            datasets.append('aw3d30')
        else:
            # Cloud mode: SRTM for parts within coverage
            datasets.append('srtm30m')
        return datasets

    # Mid-latitude regions (SRTM coverage: 60°N to 56°S)
    if lat_min >= -56 and lat_max <= 60:
        datasets.append('srtm30m')
        if deployment_type == 'local':
            datasets.append('aw3d30')
        return datasets

    # Other regions (below SRTM coverage or mixed)
    if deployment_type == 'local':
        datasets.append('aw3d30')
    datasets.append('srtm30m')

    return datasets


def get_dataset_descriptions() -> dict:
    """
    Get human-readable descriptions of each dataset.

    Returns:
        Dictionary mapping dataset name to description
    """
    return {
        'ned10m': 'NED 10m - US only, highest quality (10m resolution)',
        'srtm30m': 'SRTM 30m - Mid-latitudes (60°N to 56°S), OpenTopography S3',
        'aw3d30': 'AW3D30 - Global coverage 84°N to 84°S (30m resolution)',
        'arctic32m': 'ArcticDEM - Arctic regions, high quality (>60°N, 32m)',
        'rema32m': 'REMA - Antarctica, high quality (<-60°S, 32m)'
    }


def explain_dataset_selection(
    datasets: List[str],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> None:
    """
    Print explanation of why datasets were selected.

    Args:
        datasets: List of selected dataset names
        lat_min, lat_max, lon_min, lon_max: Region bounds
    """
    descriptions = get_dataset_descriptions()

    print("\nRequired datasets (auto-detected):")

    for dataset in datasets:
        reason = get_selection_reason(dataset, lat_min, lat_max, lon_min, lon_max)
        desc = descriptions.get(dataset, dataset)
        print(f"  ✓ {desc}")
        if reason:
            print(f"    → {reason}")


def get_selection_reason(
    dataset: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> str:
    """
    Get human-readable reason for dataset selection.

    Args:
        dataset: Dataset name
        lat_min, lat_max, lon_min, lon_max: Region bounds

    Returns:
        Explanation string
    """
    if dataset == 'rema32m':
        return "Antarctic region - highest quality for Antarctica"

    elif dataset == 'arctic32m':
        if lat_min > 60:
            return "Arctic region - highest quality for high latitudes (>60°N)"
        else:
            return "Region extends into Arctic - coverage for northern areas"

    elif dataset == 'ned10m':
        return "US region - highest quality available (10m resolution)"

    elif dataset == 'srtm30m':
        return "Within SRTM coverage area (60°N to 56°S)"

    elif dataset == 'aw3d30':
        return "Global coverage dataset (84°N to 84°S, 30m resolution)"

    return ""


# Example usage and testing
if __name__ == "__main__":
    print("Dataset Selection Examples:\n")

    test_regions = [
        ("Vermont, USA", 42.7, 45.0, -73.4, -71.5),
        ("Greenland", 60.0, 83.6, -73.3, -12.2),
        ("Canada", 41.7, 83.2, -141.0, -52.6),
        ("Antarctica", -85.0, -60.1, -180, 180),
        ("Switzerland", 45.8, 47.8, 5.9, 10.5),
        ("Australia", -43.6, -10.4, 113.3, 153.6),
    ]

    for name, lat_min, lat_max, lon_min, lon_max in test_regions:
        print(f"{'=' * 60}")
        print(f"{name}")
        print(f"Bounds: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")

        datasets = get_required_datasets(lat_min, lat_max, lon_min, lon_max)
        explain_dataset_selection(datasets, lat_min, lat_max, lon_min, lon_max)
        print()
