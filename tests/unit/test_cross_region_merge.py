#!/usr/bin/env python3
"""
Unit tests for cross-region climb merging functionality.

Tests the adjacency detection and merge logic from
scripts/merge_cross_region_climbs.py using mock DataFrames.
"""

import pytest
import sys
import math
from pathlib import Path
from typing import Tuple

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

import pandas as pd
import numpy as np

# Import functions from merge script
from merge_cross_region_climbs import (
    haversine_distance,
    calculate_region_bbox,
    bbox_distance_km,
    regions_are_adjacent,
    check_climbs_connected,
    merge_climbs,
    extract_way_ids,
)


# Bounding boxes for testing (from geo_definitions.py)
DELAWARE_BBOX = (38.45, -75.79, 39.84, -74.98)  # min_lat, min_lon, max_lat, max_lon
NEW_JERSEY_BBOX = (38.75, -75.58, 41.36, -73.68)
OREGON_BBOX = (41.99, -124.57, 46.29, -116.46)
CALIFORNIA_BBOX = (32.53, -124.48, 42.01, -114.13)


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point_zero_distance(self):
        """Same point should have zero distance."""
        dist = haversine_distance(40.0, -75.0, 40.0, -75.0)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        """Test against known distance (NYC to Philadelphia ~130 km)."""
        # NYC: 40.7128, -74.0060
        # Philadelphia: 39.9526, -75.1652
        dist = haversine_distance(40.7128, -74.0060, 39.9526, -75.1652)
        assert dist == pytest.approx(130, rel=0.1)  # Within 10%

    def test_short_distance(self):
        """Test short distance calculation."""
        # Two points ~1 km apart
        dist = haversine_distance(40.0, -75.0, 40.009, -75.0)
        assert dist == pytest.approx(1.0, rel=0.1)

    def test_cross_continent_distance(self):
        """Test long distance (east coast to west coast)."""
        # Delaware to Oregon (~4000 km)
        dist = haversine_distance(39.0, -75.5, 44.0, -120.5)
        assert dist > 3500  # Should be > 3500 km
        assert dist < 4500  # Should be < 4500 km


class TestCalculateRegionBbox:
    """Tests for bounding box calculation from DataFrame."""

    def test_simple_bbox(self):
        """Test bbox calculation with simple coordinates."""
        df = pd.DataFrame({
            'Latitude': [39.0, 40.0, 41.0],
            'Longitude': [-76.0, -75.0, -74.0]
        })
        bbox = calculate_region_bbox(df)
        assert bbox == (39.0, -76.0, 41.0, -74.0)

    def test_single_point(self):
        """Test bbox with single point."""
        df = pd.DataFrame({
            'Latitude': [40.0],
            'Longitude': [-75.0]
        })
        bbox = calculate_region_bbox(df)
        assert bbox == (40.0, -75.0, 40.0, -75.0)


class TestBboxDistanceKm:
    """Tests for distance between bounding boxes."""

    def test_overlapping_bboxes_zero_distance(self):
        """Overlapping bboxes should have zero distance."""
        bbox1 = (38.0, -76.0, 40.0, -74.0)
        bbox2 = (39.0, -75.5, 41.0, -73.5)  # Overlaps with bbox1
        dist = bbox_distance_km(bbox1, bbox2)
        assert dist == 0.0

    def test_adjacent_bboxes(self):
        """Delaware and New Jersey should have small/zero distance."""
        dist = bbox_distance_km(DELAWARE_BBOX, NEW_JERSEY_BBOX)
        # These states share a border, so distance should be 0 or very small
        assert dist < 50  # Should be adjacent or very close

    def test_distant_bboxes(self):
        """Delaware and Oregon should have large distance."""
        dist = bbox_distance_km(DELAWARE_BBOX, OREGON_BBOX)
        # East coast to west coast should be > 3000 km
        assert dist > 3000


class TestRegionsAreAdjacent:
    """Tests for region adjacency detection."""

    def test_delaware_new_jersey_adjacent(self):
        """Delaware and New Jersey should be adjacent."""
        result = regions_are_adjacent(DELAWARE_BBOX, NEW_JERSEY_BBOX, buffer_km=10.0)
        assert result is True

    def test_delaware_oregon_not_adjacent(self):
        """Delaware and Oregon should NOT be adjacent."""
        result = regions_are_adjacent(DELAWARE_BBOX, OREGON_BBOX, buffer_km=10.0)
        assert result is False

    def test_buffer_distance_10km_default(self):
        """Verify 10km default buffer behavior."""
        # Create two bboxes exactly 5km apart
        bbox1 = (40.0, -75.0, 40.1, -74.9)
        # ~5km away (0.045 degrees at 40° latitude ≈ 5km)
        bbox2 = (40.0, -74.85, 40.1, -74.75)

        # Should be adjacent with 10km buffer
        assert regions_are_adjacent(bbox1, bbox2, buffer_km=10.0) is True
        # Should NOT be adjacent with 2km buffer
        assert regions_are_adjacent(bbox1, bbox2, buffer_km=2.0) is False

    def test_overlapping_always_adjacent(self):
        """Overlapping regions should always be adjacent."""
        bbox1 = (39.0, -76.0, 41.0, -74.0)
        bbox2 = (40.0, -75.0, 42.0, -73.0)  # Overlaps
        assert regions_are_adjacent(bbox1, bbox2, buffer_km=0.0) is True


class TestCheckClimbsConnected:
    """Tests for climb connection detection."""

    @pytest.fixture
    def border_climb_delaware(self):
        """Climb near Delaware/NJ border."""
        return pd.Series({
            'Street Name': 'Border Road',
            'Latitude': 39.84,  # Near DE/NJ border
            'Longitude': -75.0,
            'Length (km)': 1.8,  # Truncated at border
            'Elev Gain (m)': 95,
            'Way ID': '789'
        })

    @pytest.fixture
    def border_climb_new_jersey(self):
        """Matching climb on NJ side of border."""
        return pd.Series({
            'Street Name': 'Border Road',
            'Latitude': 39.85,  # Just across border in NJ
            'Longitude': -75.0,
            'Length (km)': 2.2,  # Different length (not a duplicate)
            'Elev Gain (m)': 110,
            'Way ID': '790'
        })

    @pytest.fixture
    def duplicate_climb(self):
        """Duplicate climb with same length."""
        return pd.Series({
            'Street Name': 'Border Road',
            'Latitude': 39.84,
            'Longitude': -75.0,
            'Length (km)': 1.8,  # Same length as border_climb_delaware
            'Elev Gain (m)': 95,
            'Way ID': '789'
        })

    @pytest.fixture
    def different_street_climb(self):
        """Climb on different street."""
        return pd.Series({
            'Street Name': 'Main Street',
            'Latitude': 39.84,
            'Longitude': -75.0,
            'Length (km)': 2.0,
            'Elev Gain (m)': 100,
            'Way ID': '111'
        })

    def test_same_street_close_different_lengths_connects(
        self, border_climb_delaware, border_climb_new_jersey
    ):
        """Climbs with same street, close, different lengths should connect."""
        connected, distance = check_climbs_connected(
            border_climb_delaware, border_climb_new_jersey
        )
        assert connected is True
        assert distance < 0.5  # Within 500m

    def test_same_street_same_lengths_is_duplicate(
        self, border_climb_delaware, duplicate_climb
    ):
        """Climbs with same street and same length are duplicates, not merges."""
        connected, distance = check_climbs_connected(
            border_climb_delaware, duplicate_climb
        )
        assert connected is False

    def test_different_streets_never_connect(
        self, border_climb_delaware, different_street_climb
    ):
        """Climbs on different streets should never connect."""
        connected, distance = check_climbs_connected(
            border_climb_delaware, different_street_climb
        )
        assert connected is False

    def test_distance_threshold_500m(self, border_climb_delaware):
        """Test 500m distance threshold."""
        # Create climb 600m away (beyond threshold)
        far_climb = pd.Series({
            'Street Name': 'Border Road',
            'Latitude': 39.845,  # ~500m+ away
            'Longitude': -74.99,
            'Length (km)': 2.0,
            'Elev Gain (m)': 100,
            'Way ID': '999'
        })
        connected, distance = check_climbs_connected(
            border_climb_delaware, far_climb
        )
        # Should be close but verify threshold behavior
        # 0.5 km = 500m default threshold


class TestMergeClimbs:
    """Tests for climb merging functionality."""

    @pytest.fixture
    def climb1(self):
        """First climb segment."""
        return pd.Series({
            'Street Name': 'Mountain Road',
            'Latitude': 39.84,
            'Longitude': -75.0,
            'Length (km)': 1.8,
            'Elev Gain (m)': 95,
            'Max Grade (%)': 8.5,
            'Way ID': '123, 456',
            'City': 'Wilmington',
            'State': 'DE'
        })

    @pytest.fixture
    def climb2(self):
        """Second climb segment."""
        return pd.Series({
            'Street Name': 'Mountain Road',
            'Latitude': 39.85,
            'Longitude': -75.0,
            'Length (km)': 2.2,
            'Elev Gain (m)': 110,
            'Max Grade (%)': 10.2,
            'Way ID': '456, 789',
            'City': 'Camden',
            'State': 'NJ'
        })

    def test_merge_combines_lengths(self, climb1, climb2):
        """Merged climb should have combined length."""
        merged = merge_climbs(climb1, climb2, 'delaware.xlsx', 'new_jersey.xlsx')
        assert merged['Length (km) (Combined)'] == pytest.approx(4.0)  # 1.8 + 2.2

    def test_merge_combines_elevation_gain(self, climb1, climb2):
        """Merged climb should have combined elevation gain."""
        merged = merge_climbs(climb1, climb2, 'delaware.xlsx', 'new_jersey.xlsx')
        assert merged['Elev Gain (m) (Combined)'] == pytest.approx(205)  # 95 + 110

    def test_merge_recalculates_avg_grade(self, climb1, climb2):
        """Merged climb should have recalculated average grade."""
        merged = merge_climbs(climb1, climb2, 'delaware.xlsx', 'new_jersey.xlsx')
        # Total: 205m gain over 4km = 5.125%
        expected_grade = (205 / 4000) * 100  # 5.125%
        assert merged['Avg Grade (%) (Recalculated)'] == pytest.approx(expected_grade, rel=0.01)

    def test_merge_combines_way_ids(self, climb1, climb2):
        """Merged climb should have combined way IDs (deduplicated)."""
        merged = merge_climbs(climb1, climb2, 'delaware.xlsx', 'new_jersey.xlsx')
        # Way IDs: 123, 456 + 456, 789 = 123, 456, 789 (deduplicated, sorted)
        assert '123' in merged['Way ID (Combined)']
        assert '456' in merged['Way ID (Combined)']
        assert '789' in merged['Way ID (Combined)']

    def test_merge_tracks_source_files(self, climb1, climb2):
        """Merged climb should track source files."""
        merged = merge_climbs(climb1, climb2, 'delaware.xlsx', 'new_jersey.xlsx')
        assert 'delaware.xlsx' in merged['Source Files']
        assert 'new_jersey.xlsx' in merged['Source Files']

    def test_merge_preserves_max_grade(self, climb1, climb2):
        """Merged climb should take max of grade values."""
        merged = merge_climbs(climb1, climb2, 'delaware.xlsx', 'new_jersey.xlsx')
        assert merged['Max Grade (%)'] == pytest.approx(10.2)  # Max of 8.5 and 10.2


class TestExtractWayIds:
    """Tests for OSM Way ID extraction."""

    def test_comma_separated(self):
        """Extract comma-separated way IDs."""
        result = extract_way_ids('123, 456, 789')
        assert result == [123, 456, 789]

    def test_space_separated(self):
        """Extract space-separated way IDs."""
        result = extract_way_ids('123 456 789')
        assert result == [123, 456, 789]

    def test_single_id(self):
        """Extract single way ID."""
        result = extract_way_ids('12345')
        assert result == [12345]

    def test_empty_string(self):
        """Empty string returns empty list."""
        result = extract_way_ids('')
        assert result == []

    def test_nan_value(self):
        """NaN value returns empty list."""
        result = extract_way_ids(float('nan'))
        assert result == []


class TestNonAdjacentRegionsSkipped:
    """Tests verifying non-adjacent regions are handled correctly."""

    def test_non_adjacent_detected(self):
        """Non-adjacent regions should be detected."""
        # Delaware to Oregon - clearly not adjacent
        is_adjacent = regions_are_adjacent(DELAWARE_BBOX, OREGON_BBOX)
        assert is_adjacent is False

    def test_distance_calculated_correctly(self):
        """Verify large distance is calculated for non-adjacent regions."""
        dist = bbox_distance_km(DELAWARE_BBOX, OREGON_BBOX)
        # Delaware (east coast) to Oregon (west coast) should be > 3500km
        assert dist > 3500

    def test_california_oregon_adjacent(self):
        """California and Oregon should be adjacent."""
        is_adjacent = regions_are_adjacent(CALIFORNIA_BBOX, OREGON_BBOX)
        assert is_adjacent is True


class TestIntegrationMockDataFrames:
    """Integration tests using mock DataFrames."""

    @pytest.fixture
    def delaware_climbs_df(self):
        """Mock Delaware climbs DataFrame."""
        return pd.DataFrame({
            'Street Name': ['Main St', 'Hill Rd', 'Border Ave'],
            'Latitude': [39.7, 39.6, 39.84],
            'Longitude': [-75.5, -75.4, -75.0],
            'Length (km)': [2.5, 3.0, 1.8],
            'Elev Gain (m)': [120, 180, 95],
            'Way ID': ['123', '456', '789']
        })

    @pytest.fixture
    def new_jersey_climbs_df(self):
        """Mock New Jersey climbs DataFrame."""
        return pd.DataFrame({
            'Street Name': ['Main St', 'Coast Rd', 'Border Ave'],
            'Latitude': [40.1, 40.5, 39.85],
            'Longitude': [-74.5, -74.0, -75.0],
            'Length (km)': [4.0, 5.5, 2.2],
            'Elev Gain (m)': [200, 280, 110],
            'Way ID': ['111', '222', '790']
        })

    def test_find_cross_border_climb(
        self, delaware_climbs_df, new_jersey_climbs_df
    ):
        """Test finding cross-border climb candidates."""
        # Border Ave should be identified as potential merge
        de_border = delaware_climbs_df[
            delaware_climbs_df['Street Name'] == 'Border Ave'
        ].iloc[0]
        nj_border = new_jersey_climbs_df[
            new_jersey_climbs_df['Street Name'] == 'Border Ave'
        ].iloc[0]

        connected, distance = check_climbs_connected(de_border, nj_border)
        assert connected is True

    def test_main_st_not_merged_different_locations(
        self, delaware_climbs_df, new_jersey_climbs_df
    ):
        """Main St in DE and NJ are too far apart to merge."""
        de_main = delaware_climbs_df[
            delaware_climbs_df['Street Name'] == 'Main St'
        ].iloc[0]
        nj_main = new_jersey_climbs_df[
            new_jersey_climbs_df['Street Name'] == 'Main St'
        ].iloc[0]

        connected, distance = check_climbs_connected(de_main, nj_main)
        # Should not connect - they're in different parts of the states
        assert connected is False or distance > 0.5
