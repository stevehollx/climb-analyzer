#!/usr/bin/env python3
"""
Unified Climb Merger

This module provides comprehensive climb merging functionality that handles:
1. Within-region chunk boundary merges
2. Cross-region merges (e.g., N/S France subregions)
3. Cross-state merges (e.g., Oregon/Idaho)
4. Country boundary enforcement (with Schengen zone exceptions)

The merger operates on DataFrames and uses streaming for memory efficiency.

Author: Climb Analyzer Team
"""

import pandas as pd
import math
import sys
import os
import yaml
from typing import Dict, List, Tuple, Optional, Set
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
    'ascii': " ▏▎▍▌▋▊▉█",
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
    """Find the column name containing climb length data."""
    for col in df.columns:
        if 'Length' in col and 'Combined' not in col and 'Original' not in col:
            return col
    return None


def find_elevation_gain_column(df: pd.DataFrame) -> Optional[str]:
    """Find the column name containing elevation gain data."""
    for col in df.columns:
        if 'Elevation Gain' in col or 'Elev Gain' in col:
            if 'Original' not in col:
                return col
    return None


class UnifiedClimbMerger:
    """
    Unified climb merger that handles all types of climb merging:
    - Within-region chunk boundaries
    - Cross-region (same country)
    - Cross-country (with Schengen exceptions)
    """

    def __init__(self, merge_rules: Dict, region_info: Optional[Dict] = None, verbose: bool = True):
        """
        Initialize the unified climb merger.

        Args:
            merge_rules: Configuration dict with merge rules
            region_info: Optional dict with region metadata
            verbose: Whether to print progress
        """
        self.merge_rules = merge_rules
        self.region_info = region_info or {}
        self.verbose = verbose

        # Extract merge parameters
        self.max_merge_distance_km = merge_rules.get('max_merge_distance_km', 0.5)
        self.length_tolerance_percent = merge_rules.get('length_tolerance_percent', 0)
        self.allow_cross_country_merge = merge_rules.get('allow_cross_country_merge', False)
        self.allowed_cross_country_list = set(merge_rules.get('allowed_cross_country_merging', []))

    def is_duplicate_or_merge_candidate(self, climb1: pd.Series, climb2: pd.Series,
                                       length_col: str) -> str:
        """
        Determine if two climbs are duplicates, merge candidates, or different.

        Args:
            climb1: First climb
            climb2: Second climb
            length_col: Name of the length column

        Returns:
            'duplicate', 'merge', or 'different'
        """
        # Check street name
        if climb1.get('Street Name') != climb2.get('Street Name'):
            return 'different'

        # Check endpoint proximity
        is_connected, distance = self._check_climbs_connected(climb1, climb2)
        if not is_connected:
            return 'different'

        # Check length difference
        length1 = climb1.get(length_col, 0)
        length2 = climb2.get(length_col, 0)

        if length1 <= 0 or length2 <= 0:
            return 'different'

        # 0% tolerance: exact length match = duplicate
        if self.length_tolerance_percent == 0:
            if length1 == length2:
                return 'duplicate'
            else:
                # Different lengths but connected = merge candidate
                return 'merge'
        else:
            # With tolerance
            length_diff_pct = abs(length1 - length2) / max(length1, length2) * 100
            if length_diff_pct < self.length_tolerance_percent:
                return 'duplicate'
            else:
                return 'merge'

    def _check_climbs_connected(self, climb1: pd.Series, climb2: pd.Series) -> Tuple[bool, float]:
        """
        Check if two climbs are connected (share endpoints within threshold).

        Returns:
            Tuple of (is_connected, min_distance_km)
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
        is_connected = min_distance < self.max_merge_distance_km

        return is_connected, min_distance

    def is_allowed_cross_boundary(self, country1: str, country2: str) -> bool:
        """
        Check if merging is allowed between two countries.

        Args:
            country1: First country name
            country2: Second country name

        Returns:
            True if merge is allowed, False otherwise
        """
        # Same country always allowed
        if country1 == country2:
            return True

        # If global cross-country merge is enabled, allow
        if self.allow_cross_country_merge:
            return True

        # Check if both countries in allowed list (e.g., Schengen)
        if country1 in self.allowed_cross_country_list and country2 in self.allowed_cross_country_list:
            return True

        return False

    def merge_two_climbs(self, climb1: pd.Series, climb2: pd.Series,
                        length_col: str, elev_col: str) -> pd.Series:
        """
        Merge two climbs into a single combined climb.

        Args:
            climb1: First climb
            climb2: Second climb
            length_col: Name of length column
            elev_col: Name of elevation gain column

        Returns:
            Merged climb as pandas Series
        """
        merged = climb1.copy()

        # Combine lengths
        if length_col:
            length1 = climb1.get(length_col, 0)
            length2 = climb2.get(length_col, 0)
            merged[length_col] = length1 + length2

        # Combine elevation gains
        if elev_col:
            gain1 = climb1.get(elev_col, 0)
            gain2 = climb2.get(elev_col, 0)
            merged[elev_col] = gain1 + gain2

        # Recalculate average gradient
        if length_col and elev_col:
            total_length = merged[length_col]
            total_gain = merged[elev_col]
            if total_length > 0:
                length_m = total_length * 1000
                avg_grade_col = None
                for col in merged.index:
                    if 'Avg Grade' in col or 'Average Gradient' in col:
                        avg_grade_col = col
                        break
                if avg_grade_col:
                    merged[avg_grade_col] = (total_gain / length_m) * 100

        return merged

    def merge_all_climbs(self, climbs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find and merge all eligible climbs in the DataFrame.

        This handles:
        - Exact duplicates (removed)
        - Within-region chunk splits (merged)
        - Cross-region splits if allowed by country rules (merged)

        Args:
            climbs_df: DataFrame containing climb data

        Returns:
            Merged DataFrame with duplicates removed and splits combined
        """
        if self.verbose:
            print(f"\n🔄 Merging climbs in {len(climbs_df):,} climb DataFrame...")

        # Find column names
        length_col = find_length_column(climbs_df)
        elev_col = find_elevation_gain_column(climbs_df)

        if not length_col:
            if self.verbose:
                print("⚠️  Warning: Could not find length column, skipping merge")
            return climbs_df

        # Check if we have country information
        has_country = 'Country' in climbs_df.columns

        # Group by street name for efficient processing
        grouped = climbs_df.groupby('Street Name')

        if self.verbose:
            print(f"   Grouped into {len(grouped):,} unique street names")

        # Track operations
        rows_to_remove = set()  # Duplicates and merged climbs
        rows_to_update = {}  # Index -> updated Series

        # Statistics
        duplicates_removed = 0
        merges_performed = 0
        cross_country_blocked = 0

        # Process each street name group
        street_groups = list(grouped)

        # Randomize to distribute heavy streets
        import random
        random.shuffle(street_groups)

        pbar = tqdm(
            total=len(street_groups),
            desc="Processing streets",
            unit="streets",
            **TQDM_DEFAULTS
        )

        for street_name, street_climbs in street_groups:
            pbar.update(1)

            if len(street_climbs) < 2:
                continue  # Single climb on this street

            # Find candidate pairs
            indices = list(street_climbs.index)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]

                    # Skip if already processed
                    if idx1 in rows_to_remove or idx2 in rows_to_remove:
                        continue
                    if idx1 in rows_to_update or idx2 in rows_to_update:
                        continue

                    climb1 = climbs_df.loc[idx1]
                    climb2 = climbs_df.loc[idx2]

                    # Check if duplicate or merge candidate
                    classification = self.is_duplicate_or_merge_candidate(
                        climb1, climb2, length_col
                    )

                    if classification == 'different':
                        continue
                    elif classification == 'duplicate':
                        # Remove the duplicate (keep first one)
                        rows_to_remove.add(idx2)
                        duplicates_removed += 1
                    elif classification == 'merge':
                        # Check country boundary rules if country info available
                        if has_country:
                            country1 = climb1.get('Country', 'Unknown')
                            country2 = climb2.get('Country', 'Unknown')

                            if not self.is_allowed_cross_boundary(country1, country2):
                                cross_country_blocked += 1
                                continue

                        # Merge the climbs
                        merged_climb = self.merge_two_climbs(climb1, climb2, length_col, elev_col)
                        rows_to_update[idx1] = merged_climb
                        rows_to_remove.add(idx2)
                        merges_performed += 1

        pbar.close()

        # Apply updates
        result_df = climbs_df.copy()

        # Update merged climbs
        for idx, updated_climb in rows_to_update.items():
            result_df.loc[idx] = updated_climb

        # Remove duplicates and merged climbs
        result_df = result_df.drop(index=list(rows_to_remove))

        # Reset index
        result_df = result_df.reset_index(drop=True)

        if self.verbose:
            print(f"\n✅ Merge complete:")
            print(f"   Original climbs: {len(climbs_df):,}")
            print(f"   Duplicates removed: {duplicates_removed:,}")
            print(f"   Splits merged: {merges_performed:,}")
            if has_country:
                print(f"   Cross-country blocked: {cross_country_blocked:,}")
            print(f"   Final climbs: {len(result_df):,}")

        return result_df
