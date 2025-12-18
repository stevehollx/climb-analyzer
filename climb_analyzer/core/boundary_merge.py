#!/usr/bin/env python3
"""
Boundary Climb Merger for DataFrames

This module provides functionality to merge climbs that were split at region boundaries
during analysis. It works on a single DataFrame of climbs (in-memory post-processing)
rather than operating on segments (which would require 16GB+ RAM for large regions).

Memory Efficiency:
- Operates on ~50K climbs (~500MB) instead of ~7.9M segments (~16GB)
- Uses the same proven algorithm as cross-region merge
- Safe for large regions like France, Germany, California

Algorithm:
1. Group climbs by street name (O(n) instead of O(n²))
2. Within each street, find climbs with nearby endpoints (<500m)
3. Filter duplicates (similar lengths = duplicate, skip)
4. Merge splits (different lengths = boundary split, merge)

Author: Climb Analyzer Team
"""

import pandas as pd
import math
import sys
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Try to import scipy for spatial indexing (optional optimization)
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Configure tqdm for GUI compatibility
FORCE_TQDM_OUTPUT = os.environ.get('FORCE_TQDM', '0') == '1'

TQDM_DEFAULTS = {
    'dynamic_ncols': True,
    'disable': False if FORCE_TQDM_OUTPUT else None,
    'file': sys.stderr if FORCE_TQDM_OUTPUT else None,
    'mininterval': 0.5 if FORCE_TQDM_OUTPUT else 0.1,
    'ascii': " ▏▎▍▌▋▊▉█",   # Gradient block characters for smooth progress bars (9 chars needed for tqdm)
    'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
}


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


def find_length_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the column name containing climb length data.

    Args:
        df: DataFrame with climb data

    Returns:
        Column name or None if not found
    """
    for col in df.columns:
        if 'Length' in col and 'Combined' not in col:
            return col
    return None


def find_elevation_gain_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the column name containing elevation gain data.

    Args:
        df: DataFrame with climb data

    Returns:
        Column name or None if not found
    """
    for col in df.columns:
        if 'Elevation Gain' in col or 'Elev Gain' in col:
            return col
    return None


def check_climbs_connected(climb1: pd.Series, climb2: pd.Series,
                          distance_threshold_km: float = 0.5) -> Tuple[bool, float]:
    """
    Check if two climbs are connected (share endpoints within threshold).

    Args:
        climb1: First climb data
        climb2: Second climb data
        distance_threshold_km: Maximum distance for connection (km)

    Returns:
        Tuple of (is_connected, distance_km)
    """
    # Get coordinates
    lat1_start = climb1.get('Latitude', climb1.get('Start Latitude'))
    lon1_start = climb1.get('Longitude', climb1.get('Start Longitude'))
    lat1_end = climb1.get('End Latitude', lat1_start)
    lon1_end = climb1.get('End Longitude', lon1_start)

    lat2_start = climb2.get('Latitude', climb2.get('Start Latitude'))
    lon2_start = climb2.get('Longitude', climb2.get('Start Longitude'))
    lat2_end = climb2.get('End Latitude', lat2_start)
    lon2_end = climb2.get('End Longitude', lon2_start)

    # Check all possible endpoint connections
    distances = [
        haversine_distance(lat1_end, lon1_end, lat2_start, lon2_start),  # climb1 end -> climb2 start
        haversine_distance(lat1_start, lon1_start, lat2_end, lon2_end),  # climb1 start -> climb2 end
        haversine_distance(lat1_end, lon1_end, lat2_end, lon2_end),      # both ends connect
        haversine_distance(lat1_start, lon1_start, lat2_start, lon2_start),  # both starts connect
    ]

    min_distance = min(distances)
    is_connected = min_distance < distance_threshold_km

    return is_connected, min_distance


def merge_two_climbs(climb1: pd.Series, climb2: pd.Series) -> pd.Series:
    """
    Merge two climbs into a single combined climb.

    Args:
        climb1: First climb data
        climb2: Second climb data

    Returns:
        Merged climb as pandas Series
    """
    merged = climb1.copy()

    # Find length and elevation gain columns
    length_col = None
    elev_col = None
    for col in climb1.index:
        if 'Length' in col and 'Combined' not in col:
            length_col = col
        if 'Elevation Gain' in col or 'Elev Gain' in col:
            elev_col = col

    # Combine lengths
    if length_col:
        length1 = climb1.get(length_col, 0)
        length2 = climb2.get(length_col, 0)
        merged[length_col] = length1 + length2

        # Add original lengths for tracking
        merged['Length 1 (Original)'] = length1
        merged['Length 2 (Original)'] = length2

    # Combine elevation gains
    if elev_col:
        gain1 = climb1.get(elev_col, 0)
        gain2 = climb2.get(elev_col, 0)
        merged[elev_col] = gain1 + gain2

        # Add original gains for tracking
        merged['Elev Gain 1 (Original)'] = gain1
        merged['Elev Gain 2 (Original)'] = gain2

    # Calculate new average gradient if both length and elevation exist
    if length_col and elev_col:
        total_length = merged[length_col]
        total_gain = merged[elev_col]
        if total_length > 0:
            # Convert to meters for gradient calculation
            length_m = total_length * 1000
            merged['Average Gradient (%)'] = (total_gain / length_m) * 100

    # Add merge metadata
    merged['Merged'] = 'Yes'
    merged['Merge Type'] = 'Boundary'

    return merged


def find_candidates_with_spatial_index(climbs1: pd.DataFrame, climbs2: pd.DataFrame,
                                      distance_threshold_km: float) -> List[Tuple[int, int]]:
    """
    Use spatial indexing to find candidate climb pairs within distance threshold.

    Args:
        climbs1: First set of climbs
        climbs2: Second set of climbs
        distance_threshold_km: Maximum distance for candidates

    Returns:
        List of (index1, index2) tuples for candidate pairs
    """
    if not HAS_SCIPY:
        # Fallback to brute force
        return [(i1, i2) for i1 in climbs1.index for i2 in climbs2.index]

    # Extract coordinates from both dataframes
    coords1 = []
    for idx, climb in climbs1.iterrows():
        lat = climb.get('Latitude', climb.get('Start Latitude', 0))
        lon = climb.get('Longitude', climb.get('Start Longitude', 0))
        coords1.append([lat, lon])

    coords2 = []
    for idx, climb in climbs2.iterrows():
        lat = climb.get('Latitude', climb.get('Start Latitude', 0))
        lon = climb.get('Longitude', climb.get('Start Longitude', 0))
        coords2.append([lat, lon])

    # Build KD-tree for coords2
    tree = cKDTree(coords2)

    # Query for candidates within threshold
    # Convert km to degrees (approximate: 1 degree ≈ 111 km)
    degree_threshold = distance_threshold_km / 111.0

    candidates = []
    for i1, coord1 in enumerate(coords1):
        indices = tree.query_ball_point(coord1, degree_threshold)
        for i2 in indices:
            candidates.append((climbs1.index[i1], climbs2.index[i2]))

    return candidates


def merge_boundary_climbs_in_dataframe(
    climbs_df: pd.DataFrame,
    distance_threshold_km: float = 0.5,
    length_diff_threshold_pct: float = 5.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Find and merge climbs within a single DataFrame that were split at region boundaries.

    This function uses the same proven algorithm as cross-region merge but operates
    on a single DataFrame instead of two separate files. It's designed to be memory-efficient
    for large regions (operates on ~50K climbs instead of ~7.9M segments).

    Algorithm:
    1. Group climbs by street name (O(n) instead of O(n²))
    2. Within each street, find climbs with endpoints within distance_threshold_km
    3. Filter duplicates: if lengths are similar (within length_diff_threshold_pct), skip as duplicate
    4. Merge splits: if lengths are different, merge as boundary split

    Memory Usage:
    - France: ~50K climbs × ~10KB each = ~500MB (vs 16GB for segments)
    - California: ~80K climbs × ~10KB each = ~800MB

    Args:
        climbs_df: DataFrame containing climb data
        distance_threshold_km: Max distance between climb endpoints to consider them connected (default: 0.5)
        length_diff_threshold_pct: Length difference threshold to distinguish duplicates from splits (default: 5%)
        verbose: Print progress and statistics

    Returns:
        Tuple of (merged_dataframe, statistics_dict)

    Statistics dict contains:
        - original_count: Number of climbs before merge
        - merged_count: Number of climbs after merge
        - splits_merged: Number of split climb pairs that were merged
        - duplicates_skipped: Number of duplicate climbs that were skipped
        - streets_processed: Number of street names processed
    """
    if verbose:
        print(f"\n🔄 Searching for boundary splits in {len(climbs_df):,} climbs...")

    # Initialize statistics
    stats = {
        'original_count': len(climbs_df),
        'merged_count': 0,
        'splits_merged': 0,
        'duplicates_skipped': 0,
        'streets_processed': 0
    }

    # Find column names
    length_col = find_length_column(climbs_df)
    if not length_col:
        if verbose:
            print("⚠️  Warning: Could not find length column, skipping merge")
        stats['merged_count'] = len(climbs_df)
        return climbs_df, stats

    # Group by street name
    grouped = climbs_df.groupby('Street Name')
    stats['streets_processed'] = len(grouped)

    if verbose:
        print(f"   Grouped into {stats['streets_processed']:,} unique street names")

    # Track which rows to remove (merged into other rows)
    rows_to_remove = set()
    rows_to_update = {}  # Maps index -> updated Series

    # Process each street name group with progress bar
    merge_candidates = []

    # Create progress bar for street processing
    street_groups = list(grouped)

    # Randomize order to distribute heavy streets (like "service", "track") throughout
    import random
    random.shuffle(street_groups)

    pbar = tqdm(
        total=len(street_groups),
        desc="Finding splits",
        unit="streets",
        **TQDM_DEFAULTS
    )

    for street_name, street_climbs in street_groups:
        pbar.update(1)

        if len(street_climbs) < 2:
            continue  # Single climb on this street, can't be split

        # Use spatial indexing if available and beneficial
        use_spatial_index = HAS_SCIPY and len(street_climbs) > 10

        # Find candidate pairs (same street, nearby endpoints)
        if use_spatial_index:
            candidate_pairs = find_candidates_with_spatial_index(
                street_climbs, street_climbs, distance_threshold_km
            )
            # Remove self-pairs
            candidate_pairs = [(i1, i2) for i1, i2 in candidate_pairs if i1 != i2]
        else:
            # Brute force for small groups
            candidate_pairs = []
            indices = list(street_climbs.index)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    candidate_pairs.append((indices[i], indices[j]))

        # Check each candidate pair
        for idx1, idx2 in candidate_pairs:
            # Skip if either climb already processed
            if idx1 in rows_to_remove or idx2 in rows_to_remove:
                continue
            if idx1 in rows_to_update or idx2 in rows_to_update:
                continue

            climb1 = climbs_df.loc[idx1]
            climb2 = climbs_df.loc[idx2]

            # Check if climbs are connected (nearby endpoints)
            is_connected, distance = check_climbs_connected(
                climb1, climb2, distance_threshold_km
            )

            if not is_connected:
                continue

            # Check if lengths are similar (duplicate) or different (split)
            length1 = climb1.get(length_col, 0)
            length2 = climb2.get(length_col, 0)

            if length1 <= 0 or length2 <= 0:
                continue

            length_diff_pct = abs(length1 - length2) / max(length1, length2) * 100

            if length_diff_pct < length_diff_threshold_pct:
                # Duplicate - skip
                stats['duplicates_skipped'] += 1
                continue

            # This is a split - merge them
            merged_climb = merge_two_climbs(climb1, climb2)
            merge_candidates.append({
                'street_name': street_name,
                'idx1': idx1,
                'idx2': idx2,
                'length1': length1,
                'length2': length2,
                'combined_length': length1 + length2,
                'distance': distance,
                'merged_climb': merged_climb
            })

    # Close progress bar
    pbar.close()

    # Apply merges with progress bar
    if verbose and len(merge_candidates) > 0:
        print(f"\n   Applying {len(merge_candidates):,} candidate merges...")

    merge_pbar = tqdm(
        total=len(merge_candidates),
        desc="Applying merges",
        unit="merges",
        **TQDM_DEFAULTS
    )

    for candidate in merge_candidates:
        merge_pbar.update(1)
        idx1 = candidate['idx1']
        idx2 = candidate['idx2']

        # Skip if either already processed
        if idx1 in rows_to_remove or idx2 in rows_to_remove:
            continue
        if idx1 in rows_to_update or idx2 in rows_to_update:
            continue

        # Update first climb with merged data
        rows_to_update[idx1] = candidate['merged_climb']

        # Mark second climb for removal
        rows_to_remove.add(idx2)

        stats['splits_merged'] += 1

        if verbose and stats['splits_merged'] <= 10:  # Show first 10 merges
            print(f"   • {candidate['street_name']}: "
                  f"{candidate['length1']:.2f} + {candidate['length2']:.2f} = "
                  f"{candidate['combined_length']:.2f} km (merged)")

    # Close merge progress bar
    merge_pbar.close()

    if verbose and stats['splits_merged'] > 10:
        print(f"   ... and {stats['splits_merged'] - 10} more")

    # Apply updates
    result_df = climbs_df.copy()

    # Update merged climbs
    for idx, updated_climb in rows_to_update.items():
        result_df.loc[idx] = updated_climb

    # Remove climbs that were merged into others
    result_df = result_df.drop(index=list(rows_to_remove))

    # Reset index
    result_df = result_df.reset_index(drop=True)

    stats['merged_count'] = len(result_df)

    if verbose:
        print(f"\n✅ Post-processing complete:")
        print(f"   Original climbs: {stats['original_count']:,}")
        print(f"   Split pairs merged: {stats['splits_merged']:,}")
        print(f"   Duplicates skipped: {stats['duplicates_skipped']:,}")
        print(f"   Final climbs: {stats['merged_count']:,}")

    return result_df, stats


# Backward compatibility alias (deprecated)
merge_cross_chunk_climbs_in_dataframe = merge_boundary_climbs_in_dataframe
