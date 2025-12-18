#!/usr/bin/env python3
"""
Cross-Region Climb Merger

This script analyzes two climb analysis output files (CSV or XLSX) and identifies
climbs that span across both regions. When such climbs are found, it merges them
into a single comprehensive climb with combined metrics.

CONTAINER-AWARE WRAPPER:
    This script automatically detects if it's running inside or outside the
    climb-analyzer Docker container:

    - **Inside container**: Executes directly with full access to dependencies
    - **Outside container**: Automatically re-executes via 'docker exec'

    This means you can run it from your host machine without installing pandas,
    openpyxl, or other Python dependencies - it will automatically run inside
    the container where everything is already installed.

Usage:
    # From host machine (auto-wraps to container)
    python scripts/merge_cross_region_climbs.py output/region1.xlsx output/region2.xlsx

    # From inside container (runs directly)
    python /app/scripts/merge_cross_region_climbs.py output/region1.xlsx output/region2.xlsx

    # Explicit docker exec (manual)
    docker exec climb-analyzer python /app/scripts/merge_cross_region_climbs.py file1.xlsx file2.xlsx

Examples:
    python scripts/merge_cross_region_climbs.py Vermont.xlsx New_Hampshire.xlsx
    python scripts/merge_cross_region_climbs.py France_North.xlsx France_South.xlsx

Requirements:
    - Docker container must be running (docker-compose up -d)
    - Files must be accessible from container (in mounted volume)
"""

import argparse
import os
import pandas as pd
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

# Import formatting utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from climb_analyzer.utils.formatting import (
    print_banner, print_header, print_success, print_warning,
    print_info, print_list_item, print_separator, STANDARD_WIDTH
)

# Try to import scipy for spatial indexing (optional optimization)
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth (in km).

    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point

    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r


def load_climb_file(filepath: str) -> pd.DataFrame:
    """
    Load a climb analysis file (CSV or XLSX).

    Args:
        filepath: Path to the file

    Returns:
        DataFrame with climb data
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() == '.csv':
        df = pd.read_csv(filepath)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .xlsx")

    # Validate required columns
    required_cols = ['Street Name', 'Latitude', 'Longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def extract_way_ids(way_id_str: str) -> List[int]:
    """
    Extract OSM Way IDs from string.

    Args:
        way_id_str: String containing way IDs (e.g., "12345, 67890")

    Returns:
        List of way IDs as integers
    """
    if pd.isna(way_id_str) or way_id_str == '':
        return []

    try:
        # Handle both comma-separated and space-separated formats
        way_ids = str(way_id_str).replace(' ', ',').split(',')
        return [int(wid.strip()) for wid in way_ids if wid.strip().isdigit()]
    except Exception:
        return []


def calculate_region_bbox(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from climb coordinates.

    Args:
        df: DataFrame with Latitude and Longitude columns

    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    return (
        df['Latitude'].min(),
        df['Longitude'].min(),
        df['Latitude'].max(),
        df['Longitude'].max()
    )


def bbox_distance_km(bbox1: Tuple[float, float, float, float],
                     bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate minimum distance between two bounding boxes.

    Args:
        bbox1: (min_lat, min_lon, max_lat, max_lon) for first region
        bbox2: (min_lat, min_lon, max_lat, max_lon) for second region

    Returns:
        Minimum distance in kilometers between the two bboxes
    """
    min_lat1, min_lon1, max_lat1, max_lon1 = bbox1
    min_lat2, min_lon2, max_lat2, max_lon2 = bbox2

    # Check if bboxes overlap
    if not (max_lat1 < min_lat2 or max_lat2 < min_lat1 or
            max_lon1 < min_lon2 or max_lon2 < min_lon1):
        return 0.0  # Overlapping regions

    # Find closest points between the two bboxes
    # Clamp bbox2 center to bbox1 bounds to find nearest point
    closest_lat1 = max(min_lat1, min(max_lat1, (min_lat2 + max_lat2) / 2))
    closest_lon1 = max(min_lon1, min(max_lon1, (min_lon2 + max_lon2) / 2))

    closest_lat2 = max(min_lat2, min(max_lat2, (min_lat1 + max_lat1) / 2))
    closest_lon2 = max(min_lon2, min(max_lon2, (min_lon1 + max_lon1) / 2))

    return haversine_distance(closest_lat1, closest_lon1, closest_lat2, closest_lon2)


def regions_are_adjacent(bbox1: Tuple[float, float, float, float],
                         bbox2: Tuple[float, float, float, float],
                         buffer_km: float = 10.0) -> bool:
    """
    Check if two bounding boxes are adjacent (within buffer distance).

    Args:
        bbox1: (min_lat, min_lon, max_lat, max_lon) for first region
        bbox2: (min_lat, min_lon, max_lat, max_lon) for second region
        buffer_km: Maximum distance in km to consider regions adjacent

    Returns:
        True if regions overlap or are within buffer distance
    """
    distance = bbox_distance_km(bbox1, bbox2)
    return distance <= buffer_km


def check_climbs_connected(climb1: pd.Series, climb2: pd.Series,
                          distance_threshold_km: float = 0.5,
                          length_tolerance_pct: float = 5.0) -> Tuple[bool, float]:
    """
    Check if two climbs are connected (same road or very close endpoints).

    IMPORTANT: Only considers climbs as merge candidates if their lengths differ,
    indicating they were truncated at region boundaries. Climbs with identical
    lengths are considered duplicates, not cross-boundary segments.

    Args:
        climb1: First climb data
        climb2: Second climb data
        distance_threshold_km: Maximum distance in km to consider climbs connected
        length_tolerance_pct: Percentage tolerance for considering lengths "same" (default 5%)

    Returns:
        Tuple of (is_connected, distance_in_km)
    """
    # Check if same street name
    if climb1['Street Name'] != climb2['Street Name']:
        return False, float('inf')

    # Find length column
    length_col = None
    for col in climb1.index:
        if 'Length' in col:
            length_col = col
            break

    if not length_col:
        # No length column found, can't determine if duplicate
        return False, float('inf')

    length1 = climb1.get(length_col, 0)
    length2 = climb2.get(length_col, 0)

    # Check if lengths are essentially the same (within tolerance)
    # If so, these are duplicates, not cross-boundary segments
    if length1 > 0 and length2 > 0:
        length_diff_pct = abs(length1 - length2) / max(length1, length2) * 100
        if length_diff_pct < length_tolerance_pct:
            # Lengths are too similar - this is a duplicate, not a merge candidate
            return False, float('inf')

    # Calculate distance between start points
    distance = haversine_distance(
        climb1['Latitude'], climb1['Longitude'],
        climb2['Latitude'], climb2['Longitude']
    )

    # If they're close and have different lengths, they might be the same climb
    # truncated at region boundary
    if distance < distance_threshold_km:
        return True, distance

    # Check if they share any OSM Way IDs (but still require different lengths)
    way_ids1 = extract_way_ids(climb1.get('Way ID', ''))
    way_ids2 = extract_way_ids(climb2.get('Way ID', ''))

    if way_ids1 and way_ids2:
        common_ways = set(way_ids1) & set(way_ids2)
        if common_ways and distance < distance_threshold_km:
            return True, distance

    return False, float('inf')


def merge_climbs(climb1: pd.Series, climb2: pd.Series, file1_name: str,
                 file2_name: str) -> Dict:
    """
    Merge two connected climbs into a single comprehensive climb.

    Args:
        climb1: First climb data
        climb2: Second climb data
        file1_name: Name of first file (for tracking)
        file2_name: Name of second file (for tracking)

    Returns:
        Dictionary with merged climb data
    """
    # Determine which climb is longer or has more elevation gain
    length_col = None
    elev_gain_col = None

    # Find the length column (could be in different units)
    for col in climb1.index:
        if 'Length' in col:
            length_col = col
        if 'Elev Gain' in col:
            elev_gain_col = col

    length1 = climb1.get(length_col, 0) if length_col else 0
    length2 = climb2.get(length_col, 0) if length_col else 0

    elev_gain1 = climb1.get(elev_gain_col, 0) if elev_gain_col else 0
    elev_gain2 = climb2.get(elev_gain_col, 0) if elev_gain_col else 0

    # Create merged climb with combined metrics
    merged = {
        'Street Name': climb1['Street Name'],
        'Source Files': f"{file1_name} + {file2_name}",
        'Original Climb 1': f"{climb1['Street Name']} ({file1_name})",
        'Original Climb 2': f"{climb2['Street Name']} ({file2_name})",
    }

    # Copy all columns from climb1 as base
    for col in climb1.index:
        if col not in merged:
            merged[col] = climb1[col]

    # Add combined/max metrics
    if length_col:
        merged[f'{length_col} (Combined)'] = length1 + length2
        merged[f'{length_col} (File 1)'] = length1
        merged[f'{length_col} (File 2)'] = length2

    if elev_gain_col:
        merged[f'{elev_gain_col} (Combined)'] = elev_gain1 + elev_gain2
        merged[f'{elev_gain_col} (File 1)'] = elev_gain1
        merged[f'{elev_gain_col} (File 2)'] = elev_gain2

    # Take max of gradient values
    if 'Max Grade (%)' in climb1.index:
        merged['Max Grade (%)'] = max(
            climb1.get('Max Grade (%)', 0),
            climb2.get('Max Grade (%)', 0)
        )

    # Recalculate average grade if possible
    if length_col and elev_gain_col:
        total_length_m = (length1 + length2) * 1000  # Assuming length is in km
        total_gain_m = elev_gain1 + elev_gain2  # Assuming gain is in meters
        if total_length_m > 0:
            merged['Avg Grade (%) (Recalculated)'] = (total_gain_m / total_length_m) * 100

    # Combine Way IDs
    way_ids1 = extract_way_ids(climb1.get('Way ID', ''))
    way_ids2 = extract_way_ids(climb2.get('Way ID', ''))
    all_way_ids = sorted(set(way_ids1 + way_ids2))
    merged['Way ID (Combined)'] = ', '.join(map(str, all_way_ids))

    # Add location info from both
    merged['City (File 1)'] = climb1.get('City', 'Unknown')
    merged['City (File 2)'] = climb2.get('City', 'Unknown')
    merged['State (File 1)'] = climb1.get('State', 'Unknown')
    merged['State (File 2)'] = climb2.get('State', 'Unknown')

    return merged


def find_candidates_with_spatial_index(climbs1: pd.DataFrame, climbs2: pd.DataFrame,
                                        distance_threshold_km: float) -> List[Tuple[int, int]]:
    """
    Use spatial indexing (KDTree) to find candidate climb pairs within distance threshold.

    Args:
        climbs1: DataFrame of climbs from first region
        climbs2: DataFrame of climbs from second region
        distance_threshold_km: Maximum distance in km

    Returns:
        List of (index1, index2) tuples for candidate pairs
    """
    # Convert lat/lon to approximate cartesian coordinates (works for small areas)
    # 1 degree latitude ≈ 111 km, longitude varies by latitude
    coords1 = np.column_stack([
        climbs1['Latitude'].values,
        climbs1['Longitude'].values
    ])
    coords2 = np.column_stack([
        climbs2['Latitude'].values,
        climbs2['Longitude'].values
    ])

    # Build KDTree for climbs2
    tree = cKDTree(coords2)

    # Convert km threshold to approximate degrees (conservative estimate)
    # At equator: 1 degree ≈ 111 km, but we use 110 to be conservative
    threshold_degrees = distance_threshold_km / 110.0

    # Query tree for all climbs1 points
    candidate_pairs = []
    for i, coord in enumerate(coords1):
        # Find all points in climbs2 within threshold
        indices = tree.query_ball_point(coord, threshold_degrees)
        for j in indices:
            # Get actual indices from dataframe
            idx1 = climbs1.index[i]
            idx2 = climbs2.index[j]
            candidate_pairs.append((idx1, idx2))

    return candidate_pairs


def find_and_merge_cross_region_climbs(file1_path: str, file2_path: str,
                                       distance_threshold_km: float = 0.5) -> List[Dict]:
    """
    Find and merge climbs that span across two region files.

    Optimized algorithm:
    1. Group climbs by street name (O(n) + O(m))
    2. Only compare climbs with matching street names (reduces comparisons dramatically)
    3. Use coordinate proximity checks only within same-name groups

    Args:
        file1_path: Path to first climb file
        file2_path: Path to second climb file
        distance_threshold_km: Distance threshold for considering climbs connected

    Returns:
        List of merged climb dictionaries
    """
    df1 = load_climb_file(file1_path)
    df2 = load_climb_file(file2_path)

    # Check if regions are adjacent
    bbox1 = calculate_region_bbox(df1)
    bbox2 = calculate_region_bbox(df2)

    bbox_distance = bbox_distance_km(bbox1, bbox2)
    are_adjacent = regions_are_adjacent(bbox1, bbox2, buffer_km=10.0)

    if not are_adjacent:
        print_warning(f"Regions not adjacent ({bbox_distance:.1f}km apart) - skipping")
        return []

    print_info(f"Analyzing {len(df1)} vs {len(df2)} climbs for cross-boundary segments...")

    merged_climbs = []
    processed_indices = {'file1': set(), 'file2': set()}
    duplicate_count = 0  # Track duplicates that were filtered out

    # Group by street name (O(n) + O(m) instead of O(n*m))
    df1_grouped = df1.groupby('Street Name')
    df2_grouped = df2.groupby('Street Name')

    # Find common street names
    common_streets = set(df1_grouped.groups.keys()) & set(df2_grouped.groups.keys())

    comparison_count = 0

    # Only compare climbs on streets that exist in both regions
    for street_name in sorted(common_streets):
        climbs1 = df1_grouped.get_group(street_name)
        climbs2 = df2_grouped.get_group(street_name)

        # Use spatial indexing if available and beneficial (many climbs on same street)
        use_spatial_index = HAS_SCIPY and len(climbs1) * len(climbs2) > 100

        if use_spatial_index:
            # Use KDTree for spatial filtering
            candidate_pairs = find_candidates_with_spatial_index(
                climbs1, climbs2, distance_threshold_km
            )
            pairs_to_check = candidate_pairs
        else:
            # Use brute force for small groups
            pairs_to_check = [(i1, i2) for i1 in climbs1.index for i2 in climbs2.index]

        # Check candidate pairs
        for i1, i2 in pairs_to_check:
            if i1 in processed_indices['file1'] or i2 in processed_indices['file2']:
                continue

            climb1 = df1.loc[i1]
            climb2 = df2.loc[i2]

            comparison_count += 1

            # Check if same street name and nearby coordinates first
            if climb1['Street Name'] == climb2['Street Name']:
                distance = haversine_distance(
                    climb1['Latitude'], climb1['Longitude'],
                    climb2['Latitude'], climb2['Longitude']
                )

                if distance < distance_threshold_km:
                    # Check if lengths are similar (duplicate) or different (merge candidate)
                    length_col = None
                    for col in climb1.index:
                        if 'Length' in col:
                            length_col = col
                            break

                    if length_col:
                        length1 = climb1.get(length_col, 0)
                        length2 = climb2.get(length_col, 0)

                        if length1 > 0 and length2 > 0:
                            length_diff_pct = abs(length1 - length2) / max(length1, length2) * 100

                            if length_diff_pct < 5.0:
                                # Duplicate - skip silently
                                duplicate_count += 1
                                continue

            # Now check with full logic (will filter out duplicates)
            is_connected, distance = check_climbs_connected(
                climb1, climb2, distance_threshold_km
            )

            if is_connected:
                # Get lengths for display
                length_col = None
                for col in climb1.index:
                    if 'Length' in col:
                        length_col = col
                        break

                length1 = climb1.get(length_col, 0) if length_col else 0
                length2 = climb2.get(length_col, 0) if length_col else 0

                merged = merge_climbs(
                    climb1, climb2,
                    Path(file1_path).name,
                    Path(file2_path).name
                )
                merged['Row in File 1'] = i1
                merged['Row in File 2'] = i2
                merged['Distance Between (km)'] = round(distance, 2)

                merged_climbs.append(merged)
                processed_indices['file1'].add(i1)
                processed_indices['file2'].add(i2)

    # Print summary
    if merged_climbs:
        print()
        print_header("Cross-Boundary Climbs Found")
        for climb in merged_climbs:
            # Get length info
            length_col = None
            for key in climb.keys():
                if 'Length' in key and 'Combined' in key:
                    length_col = key
                    break

            if length_col:
                combined_length = climb[length_col]
                print_list_item(f"{climb['Street Name']}: {combined_length:.2f} km combined")
            else:
                print_list_item(f"{climb['Street Name']}")

        print()
        print_info(f"Filtered {duplicate_count} duplicate(s), found {len(merged_climbs)} true cross-boundary climb(s)")

    return merged_climbs


def display_merged_climbs(merged_climbs: List[Dict]):
    """
    Display merged climbs in a readable format.

    Args:
        merged_climbs: List of merged climb dictionaries
    """
    if not merged_climbs:
        print_info("No cross-boundary climbs found")
        return

    # Already printed in find_and_merge_cross_region_climbs()
    # This function is now a no-op
    pass


def prompt_user_for_replacement(merged_climbs: List[Dict], file1_path: str,
                                file2_path: str) -> bool:
    """
    Prompt user whether to replace original climbs with merged versions.

    Args:
        merged_climbs: List of merged climb dictionaries
        file1_path: Path to first file
        file2_path: Path to second file

    Returns:
        True if user wants to proceed with replacement
    """
    if not merged_climbs:
        return False

    print()
    print_separator(STANDARD_WIDTH, '=')
    print("REPLACEMENT OPTIONS")
    print_separator(STANDARD_WIDTH, '=')
    print()
    print("The script can replace the original climbs in each file with the merged")
    print("cross-region climb data. This will:")
    print_list_item(f"Update the climb in {Path(file1_path).name}")
    print_list_item(f"Update the climb in {Path(file2_path).name}")
    print_list_item("Create backup files (*.backup) before making changes")
    print()
    print("Would you like to proceed with the replacement? (Y/n): ", end='')

    response = input().strip().lower()
    # Default to yes if user just presses Enter
    if response == '':
        return True
    return response in ['yes', 'y']


def replace_climbs_in_files(merged_climbs: List[Dict], file1_path: str,
                            file2_path: str):
    """
    Replace original climbs with merged versions in both files.

    Args:
        merged_climbs: List of merged climb dictionaries
        file1_path: Path to first file
        file2_path: Path to second file
    """
    if not merged_climbs:
        return

    # Create backups
    file1_backup = Path(file1_path).with_suffix(Path(file1_path).suffix + '.backup')
    file2_backup = Path(file2_path).with_suffix(Path(file2_path).suffix + '.backup')

    import shutil
    shutil.copy2(file1_path, file1_backup)
    shutil.copy2(file2_path, file2_backup)

    print_success(f"Created backups ({file1_backup.name}, {file2_backup.name})")

    # Load files
    df1 = load_climb_file(file1_path)
    df2 = load_climb_file(file2_path)

    # Update climbs
    for climb in merged_climbs:
        row1 = climb['Row in File 1']
        row2 = climb['Row in File 2']

        # Update rows with merged data (only update columns that exist)
        # Convert dtypes to match original columns to avoid pandas warnings
        for col in df1.columns:
            if col in climb:
                value = climb[col]
                # Convert to match original column dtype
                if col in df1.columns:
                    original_dtype = df1[col].dtype
                    try:
                        if pd.api.types.is_integer_dtype(original_dtype):
                            value = int(float(value)) if pd.notna(value) else value
                        elif pd.api.types.is_float_dtype(original_dtype):
                            value = float(value) if pd.notna(value) else value
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                df1.at[row1, col] = value

        for col in df2.columns:
            if col in climb:
                value = climb[col]
                # Convert to match original column dtype
                if col in df2.columns:
                    original_dtype = df2[col].dtype
                    try:
                        if pd.api.types.is_integer_dtype(original_dtype):
                            value = int(float(value)) if pd.notna(value) else value
                        elif pd.api.types.is_float_dtype(original_dtype):
                            value = float(value) if pd.notna(value) else value
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                df2.at[row2, col] = value

    # Save updated files
    if Path(file1_path).suffix.lower() == '.csv':
        df1.to_csv(file1_path, index=False)
    else:
        df1.to_excel(file1_path, index=False)

    if Path(file2_path).suffix.lower() == '.csv':
        df2.to_csv(file2_path, index=False)
    else:
        df2.to_excel(file2_path, index=False)

    print_success(f"Updated {len(merged_climbs)} climb(s) in both files")
    print_info("Original files backed up with .backup extension")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Merge cross-region climbs from two climb analysis files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_cross_region_climbs.py region1/output.csv region2/output.csv
  python merge_cross_region_climbs.py file1.xlsx file2.xlsx --threshold 1.0

The script will:
  1. Load both climb files
  2. Find climbs that appear in both files (same name and close proximity)
  3. Merge the climbs with combined metrics
  4. Display the merged results
  5. Optionally update the original files with merged data
        """
    )

    parser.add_argument('file1', help='First climb analysis file (CSV or XLSX)')
    parser.add_argument('file2', help='Second climb analysis file (CSV or XLSX)')
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Distance threshold in km for matching climbs (default: 0.5)'
    )
    parser.add_argument(
        '--auto-replace',
        action='store_true',
        help='Automatically replace climbs without prompting'
    )

    args = parser.parse_args()

    try:
        # Find and merge cross-region climbs
        merged_climbs = find_and_merge_cross_region_climbs(
            args.file1,
            args.file2,
            args.threshold
        )

        # Display results
        display_merged_climbs(merged_climbs)

        # Ask user about replacement
        if merged_climbs:
            if args.auto_replace:
                should_replace = True
                print("\n(Auto-replace mode enabled)")
            else:
                should_replace = prompt_user_for_replacement(
                    merged_climbs,
                    args.file1,
                    args.file2
                )

            if should_replace:
                replace_climbs_in_files(merged_climbs, args.file1, args.file2)
            else:
                print_info("No changes made to original files")

    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)


def is_running_in_container() -> bool:
    """
    Detect if script is running inside a Docker container.

    Returns:
        True if running inside container, False otherwise
    """
    # Check for .dockerenv file (standard Docker indicator)
    if Path('/.dockerenv').exists():
        return True

    # Check for container-specific environment variable
    if os.environ.get('CONTAINER_ENV') == 'docker':
        return True

    # Check if running as PID 1 with container-like cgroup
    try:
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read():
                return True
    except (FileNotFoundError, PermissionError):
        pass

    return False


def run_in_container(args: List[str]) -> int:
    """
    Execute this script inside the climb-analyzer Docker container.

    Args:
        args: Command line arguments to pass to the script

    Returns:
        Exit code from Docker execution
    """
    import subprocess
    import os

    print("=" * 80)
    print("  CONTAINER WRAPPER")
    print("=" * 80)
    print("\nDetected execution outside container.")
    print("Re-executing inside climb-analyzer container...\n")

    # Build docker exec command
    # Convert host paths to container paths if needed
    container_args = []
    for arg in args[1:]:  # Skip script name
        arg_path = Path(arg)
        if arg_path.exists() and arg_path.is_file():
            # Convert to absolute path for container
            abs_path = arg_path.absolute()
            # Assume volume mounted at /app or current directory
            container_args.append(str(abs_path))
        else:
            container_args.append(arg)

    # Check if container is running
    check_cmd = ['docker', 'ps', '--filter', 'name=climb-analyzer', '--format', '{{.Names}}']
    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
        if 'climb-analyzer' not in result.stdout:
            print("❌ Error: climb-analyzer container is not running")
            print("\nStart the container with:")
            print("   docker-compose up -d")
            print("\nOr if not using docker-compose:")
            print("   docker run -d --name climb-analyzer ...")
            return 1
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Error checking Docker status: {e}")
        print("\nMake sure Docker is installed and running.")
        return 1

    # Execute inside container
    docker_cmd = [
        'docker', 'exec',
        '-it',  # Interactive + TTY for colored output
        'climb-analyzer',
        'python3',
        '/app/scripts/merge_cross_region_climbs.py'
    ] + container_args

    print(f"Running: {' '.join(docker_cmd)}\n")
    print("=" * 80)
    print()

    try:
        # Use os.system to preserve colors and interactive features
        cmd_str = ' '.join(docker_cmd)
        return os.system(cmd_str) >> 8  # Extract exit code
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130


if __name__ == '__main__':
    import os

    # Check if running inside container
    if not is_running_in_container():
        # Running outside container - re-execute inside
        exit_code = run_in_container(sys.argv)
        sys.exit(exit_code)
    else:
        # Running inside container - execute normally
        main()
