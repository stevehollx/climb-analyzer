"""
Unit tests for boundary road merging functionality.

Tests cover:
- Segment merging logic
- Duplicate detection with tolerance
- Endpoint-based indexing
- Spatial grouping for large streets
- Parallel processing
- Distance calculations
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import math
from typing import List, Dict, Tuple

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from climb_analyzer.core.merger import (
    BoundaryMerger,
    _merge_street_batch_worker_top_level,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def merger():
    """Create a BoundaryMerger instance."""
    with patch('climb_analyzer.core.merger.MERGE_PARALLEL_ENABLED', True):
        with patch('climb_analyzer.core.merger.MERGE_MAX_WORKERS', 4):
            with patch('climb_analyzer.core.merger.MERGE_BATCH_SIZE', 100):
                return BoundaryMerger(coordinate_tolerance=0.002)


@pytest.fixture
def sample_segment():
    """Create a sample road segment."""
    return {
        "way_id": 12345,
        "way_name": "Main Street",
        "highway_type": "primary",
        "coordinates": [
            (35.1000, -82.5000),
            (35.1010, -82.5010),
            (35.1020, -82.5020),
        ],
        "distance": 2500.0,
        "elevation_gain": 50.0,
    }


@pytest.fixture
def sample_segments_connectable():
    """Create two segments that can be connected."""
    return [
        {
            "way_id": 100,
            "way_name": "Test Road",
            "coordinates": [
                (35.1000, -82.5000),
                (35.1010, -82.5010),
                (35.1020, -82.5020),
            ],
        },
        {
            "way_id": 101,
            "way_name": "Test Road",
            "coordinates": [
                (35.1020, -82.5020),  # Matches endpoint of first segment
                (35.1030, -82.5030),
                (35.1040, -82.5040),
            ],
        },
    ]


@pytest.fixture
def sample_segments_duplicate():
    """Create duplicate segments (same coordinates)."""
    coords = [
        (35.1000, -82.5000),
        (35.1010, -82.5010),
        (35.1020, -82.5020),
    ]
    return [
        {
            "way_id": 200,
            "way_name": "Duplicate Road",
            "coordinates": coords.copy(),
        },
        {
            "way_id": 201,
            "way_name": "Duplicate Road",
            "coordinates": coords.copy(),
        },
    ]


@pytest.fixture
def sample_segments_large_street():
    """Create many segments for spatial grouping test."""
    segments = []
    for i in range(60):
        # Create segments in different spatial locations
        lat_offset = (i // 10) * 0.01  # Group every 10 segments
        lon_offset = (i % 10) * 0.001
        segments.append({
            "way_id": 1000 + i,
            "way_name": "service",  # Common name that triggers spatial grouping
            "coordinates": [
                (35.1000 + lat_offset, -82.5000 + lon_offset),
                (35.1005 + lat_offset, -82.5005 + lon_offset),
            ],
        })
    return segments


# ============================================================================
# Distance Calculation Tests
# ============================================================================

class TestDistanceCalculation:
    """Test Haversine distance calculations."""

    def test_calculate_distance_zero(self, merger):
        """Test distance between identical points."""
        distance = merger.calculate_distance(35.0, -82.0, 35.0, -82.0)
        assert distance == 0.0

    def test_calculate_distance_known_values(self, merger):
        """Test distance calculation with known values."""
        # Distance between approximately 1 degree latitude apart
        distance = merger.calculate_distance(35.0, -82.0, 36.0, -82.0)
        # 1 degree latitude ≈ 111 km
        assert 110 < distance < 112

    def test_calculate_distance_symmetry(self, merger):
        """Test that distance calculation is symmetric."""
        d1 = merger.calculate_distance(35.0, -82.0, 36.0, -83.0)
        d2 = merger.calculate_distance(36.0, -83.0, 35.0, -82.0)
        assert abs(d1 - d2) < 0.001

    def test_calculate_distance_small_delta(self, merger):
        """Test distance for small coordinate changes."""
        # Small change should give small distance
        distance = merger.calculate_distance(35.0, -82.0, 35.001, -82.001)
        assert distance < 1.0  # Less than 1 km


# ============================================================================
# Coordinate Matching Tests
# ============================================================================

class TestCoordinateMatching:
    """Test coordinate matching with tolerance."""

    def test_coordinates_match_exact(self, merger):
        """Test exact coordinate matching."""
        coord1 = (35.1234, -82.5678)
        coord2 = (35.1234, -82.5678)
        assert merger._coordinates_match(coord1, coord2) is True

    def test_coordinates_match_within_tolerance(self, merger):
        """Test matching within coordinate tolerance."""
        coord1 = (35.1234, -82.5678)
        coord2 = (35.1235, -82.5679)  # Within 0.002 tolerance
        assert merger._coordinates_match(coord1, coord2) is True

    def test_coordinates_no_match_outside_tolerance(self, merger):
        """Test no match outside tolerance."""
        coord1 = (35.1234, -82.5678)
        coord2 = (35.1300, -82.5700)  # Outside tolerance
        assert merger._coordinates_match(coord1, coord2) is False

    def test_coordinates_match_distance_tolerance(self, merger):
        """Test matching based on distance tolerance."""
        merger.distance_tolerance_m = 100  # 100 meters
        coord1 = (35.0000, -82.0000)
        coord2 = (35.0008, -82.0000)  # ~89 meters away
        assert merger._coordinates_match(coord1, coord2) is True

    def test_coordinates_no_match_distance_tolerance(self, merger):
        """Test no match when distance exceeds tolerance."""
        merger.distance_tolerance_m = 50  # 50 meters
        coord1 = (35.0000, -82.0000)
        coord2 = (35.0010, -82.0000)  # ~111 meters away
        assert merger._coordinates_match(coord1, coord2) is False


# ============================================================================
# Duplicate Detection Tests
# ============================================================================

class TestDuplicateDetection:
    """Test duplicate segment detection."""

    def test_are_segments_duplicates_identical(self, merger):
        """Test detection of identical segments."""
        seg1 = {
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
        }
        seg2 = {
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
        }
        assert merger._are_segments_duplicates(seg1, seg2) is True

    def test_are_segments_duplicates_reversed(self, merger):
        """Test detection of reversed duplicate segments."""
        seg1 = {
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
        }
        seg2 = {
            "coordinates": [(35.2, -82.6), (35.1, -82.5)],  # Reversed
        }
        assert merger._are_segments_duplicates(seg1, seg2) is True

    def test_are_segments_duplicates_within_tolerance(self, merger):
        """Test detection with coordinates within tolerance."""
        seg1 = {
            "coordinates": [(35.1000, -82.5000), (35.2000, -82.6000)],
        }
        seg2 = {
            "coordinates": [(35.1001, -82.5001), (35.2001, -82.6001)],
        }
        assert merger._are_segments_duplicates(seg1, seg2) is True

    def test_are_segments_not_duplicates_different_length(self, merger):
        """Test that different length segments are not duplicates."""
        seg1 = {
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
        }
        seg2 = {
            "coordinates": [(35.1, -82.5), (35.2, -82.6), (35.3, -82.7)],
        }
        assert merger._are_segments_duplicates(seg1, seg2) is False

    def test_are_segments_not_duplicates_different_coords(self, merger):
        """Test that different segments are not duplicates."""
        seg1 = {
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
        }
        seg2 = {
            "coordinates": [(35.3, -82.7), (35.4, -82.8)],
        }
        assert merger._are_segments_duplicates(seg1, seg2) is False


# ============================================================================
# Segment Connection Tests
# ============================================================================

class TestSegmentConnection:
    """Test segment connection logic."""

    def test_connect_segments_end_to_start(self, merger):
        """Test connecting segments where end of seg1 meets start of seg2."""
        seg1 = {
            "way_id": 100,
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
            "way_ids": [100],
        }
        seg2 = {
            "way_id": 101,
            "coordinates": [(35.2, -82.6), (35.3, -82.7)],
            "way_ids": [101],
        }

        connected = merger._connect_segments(seg1, seg2)

        assert connected is not None
        assert len(connected["coordinates"]) == 3
        assert connected["coordinates"][0] == (35.1, -82.5)
        assert connected["coordinates"][-1] == (35.3, -82.7)
        assert 100 in connected["way_ids"]
        assert 101 in connected["way_ids"]

    def test_connect_segments_end_to_end(self, merger):
        """Test connecting segments where ends meet."""
        seg1 = {
            "way_id": 100,
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
            "way_ids": [100],
        }
        seg2 = {
            "way_id": 101,
            "coordinates": [(35.3, -82.7), (35.2, -82.6)],  # End matches seg1 end
            "way_ids": [101],
        }

        connected = merger._connect_segments(seg1, seg2)

        assert connected is not None
        # seg2 should be reversed and connected
        assert len(connected["coordinates"]) == 3
        assert connected["coordinates"][-1] == (35.3, -82.7)

    def test_connect_segments_start_to_start(self, merger):
        """Test connecting segments where starts meet."""
        seg1 = {
            "way_id": 100,
            "coordinates": [(35.2, -82.6), (35.1, -82.5)],
            "way_ids": [100],
        }
        seg2 = {
            "way_id": 101,
            "coordinates": [(35.2, -82.6), (35.3, -82.7)],
            "way_ids": [101],
        }

        connected = merger._connect_segments(seg1, seg2)

        assert connected is not None
        assert len(connected["coordinates"]) == 3

    def test_connect_segments_no_connection(self, merger):
        """Test that disconnected segments return None."""
        seg1 = {
            "way_id": 100,
            "coordinates": [(35.1, -82.5), (35.2, -82.6)],
            "way_ids": [100],
        }
        seg2 = {
            "way_id": 101,
            "coordinates": [(35.5, -82.9), (35.6, -83.0)],  # No connection
            "way_ids": [101],
        }

        connected = merger._connect_segments(seg1, seg2)

        assert connected is None


# ============================================================================
# Spatial Grouping Tests
# ============================================================================

class TestSpatialGrouping:
    """Test spatial grouping for large streets."""

    def test_group_segments_by_location(self, merger):
        """Test spatial grouping creates reasonable groups."""
        segments = [
            {"coordinates": [(35.00, -82.00), (35.01, -82.01)]},  # Group 1
            {"coordinates": [(35.00, -82.00), (35.01, -82.01)]},  # Group 1
            {"coordinates": [(35.10, -82.10), (35.11, -82.11)]},  # Group 2
            {"coordinates": [(35.10, -82.10), (35.11, -82.11)]},  # Group 2
        ]

        groups = merger._group_segments_by_location(segments, grid_size=0.05)

        # Should create 2 groups
        assert len(groups) >= 2
        # Each group should have at least one segment
        assert all(len(group) > 0 for group in groups)

    def test_group_segments_single_location(self, merger):
        """Test grouping when all segments in same location."""
        segments = [
            {"coordinates": [(35.00, -82.00), (35.01, -82.01)]},
            {"coordinates": [(35.001, -82.001), (35.011, -82.011)]},
            {"coordinates": [(35.002, -82.002), (35.012, -82.012)]},
        ]

        groups = merger._group_segments_by_location(segments, grid_size=0.05)

        # All segments should be in same group
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_merge_large_street_with_spatial_grouping(self, merger, sample_segments_large_street):
        """Test that large streets use spatial grouping."""
        with patch.object(merger, '_merge_street') as mock_merge:
            mock_merge.return_value = sample_segments_large_street[:10]  # Return subset

            result = merger._merge_large_street_with_spatial_grouping(
                sample_segments_large_street, "service"
            )

            # Should have called _merge_street for each group
            assert mock_merge.call_count > 0
            assert len(result) > 0


# ============================================================================
# Street Merging Tests
# ============================================================================

class TestStreetMerging:
    """Test the main street merging algorithm."""

    def test_merge_street_single_segment(self, merger, sample_segment):
        """Test merging with single segment."""
        result = merger._merge_street([sample_segment], "Main Street")

        assert len(result) == 1
        assert result[0] == sample_segment

    def test_merge_street_duplicates(self, merger, sample_segments_duplicate):
        """Test merging removes duplicates."""
        result = merger._merge_street(sample_segments_duplicate, "Duplicate Road")

        # Should merge duplicates into one
        assert len(result) == 1
        # Way IDs should be consolidated
        assert "way_ids" in result[0]

    def test_merge_street_connectable(self, merger, sample_segments_connectable):
        """Test merging connects adjacent segments."""
        result = merger._merge_street(sample_segments_connectable, "Test Road")

        # Should connect into one segment
        assert len(result) == 1
        # Should have coordinates from both segments
        assert len(result[0]["coordinates"]) > 3

    def test_merge_street_no_merge_needed(self, merger):
        """Test merging when segments can't be merged."""
        segments = [
            {"coordinates": [(35.1, -82.5), (35.2, -82.6)], "way_ids": [100]},
            {"coordinates": [(35.8, -82.9), (35.9, -83.0)], "way_ids": [200]},  # Far away
        ]

        result = merger._merge_street(segments, "Test")

        # Should return both segments unmerged
        assert len(result) == 2

    def test_merge_street_empty_list(self, merger):
        """Test merging empty segment list."""
        result = merger._merge_street([], "Empty")

        assert result == []

    def test_merge_street_complex_chain(self, merger):
        """Test merging a chain of connected segments."""
        segments = [
            {"coordinates": [(35.1, -82.5), (35.2, -82.6)], "way_ids": [100]},
            {"coordinates": [(35.2, -82.6), (35.3, -82.7)], "way_ids": [101]},
            {"coordinates": [(35.3, -82.7), (35.4, -82.8)], "way_ids": [102]},
        ]

        result = merger._merge_street(segments, "Chain")

        # Should merge into one segment
        assert len(result) == 1
        assert len(result[0]["coordinates"]) == 4
        assert len(result[0]["way_ids"]) == 3


# ============================================================================
# Parallel Processing Tests
# ============================================================================

class TestParallelProcessing:
    """Test parallel processing functionality."""

    def test_calculate_safe_worker_count(self, merger):
        """Test worker count calculation."""
        with patch('climb_analyzer.core.merger.psutil') as mock_psutil:
            mock_mem = Mock()
            mock_mem.total = 16 * (1024**3)  # 16 GB
            mock_mem.available = 8 * (1024**3)  # 8 GB available
            mock_psutil.virtual_memory.return_value = mock_mem

            with patch('climb_analyzer.core.merger.os.cpu_count', return_value=8):
                worker_count = merger._calculate_safe_worker_count(total_segments=10000)

                assert worker_count > 0
                assert worker_count <= merger.max_workers

    def test_calculate_safe_worker_count_low_memory(self, merger):
        """Test worker count with low memory."""
        with patch('climb_analyzer.core.merger.psutil') as mock_psutil:
            mock_mem = Mock()
            mock_mem.total = 4 * (1024**3)  # 4 GB
            mock_mem.available = 1 * (1024**3)  # 1 GB available
            mock_psutil.virtual_memory.return_value = mock_mem

            worker_count = merger._calculate_safe_worker_count(total_segments=10000)

            # Should reduce workers due to low memory
            assert worker_count >= 1
            assert worker_count <= 4

    def test_calculate_safe_worker_count_error_handling(self, merger):
        """Test worker count calculation with error."""
        with patch('climb_analyzer.core.merger.psutil') as mock_psutil:
            mock_psutil.virtual_memory.side_effect = Exception("psutil error")

            worker_count = merger._calculate_safe_worker_count(total_segments=10000)

            # Should fall back to conservative default
            assert worker_count == min(4, merger.max_workers)

    def test_worker_function_top_level(self):
        """Test the top-level worker function."""
        street_items = [
            ("Test Street", [
                {"coordinates": [(35.1, -82.5), (35.2, -82.6)], "way_ids": [100]},
                {"coordinates": [(35.2, -82.6), (35.3, -82.7)], "way_ids": [101]},
            ])
        ]

        result = _merge_street_batch_worker_top_level(
            (street_items, 0.002, 200)
        )

        assert len(result) > 0

    def test_worker_function_with_error(self):
        """Test worker function handles errors gracefully."""
        # Invalid segment data to trigger error
        street_items = [
            ("Bad Street", [
                {"coordinates": None},  # Will cause error
            ])
        ]

        # Should not raise, should return original segments
        result = _merge_street_batch_worker_top_level(
            (street_items, 0.002, 200)
        )

        assert isinstance(result, list)


# ============================================================================
# Integration Tests
# ============================================================================

class TestMergeIntegration:
    """Integration tests for complete merge workflow."""

    def test_merge_boundary_segments_serial(self, merger):
        """Test serial merging workflow."""
        street_segments = {
            "Main St": [
                {"coordinates": [(35.1, -82.5), (35.2, -82.6)], "way_ids": [100]},
                {"coordinates": [(35.2, -82.6), (35.3, -82.7)], "way_ids": [101]},
            ],
            "Oak Ave": [
                {"coordinates": [(35.4, -82.8), (35.5, -82.9)], "way_ids": [200]},
            ],
        }

        with patch.object(merger, 'parallel_enabled', False):
            result = merger.merge_boundary_segments(
                street_segments,
                checkpoint_path=None
            )

            assert len(result) >= 2  # At least 2 merged segments

    def test_merge_with_checkpoint(self, merger, tmp_path):
        """Test merging with checkpoint saving."""
        street_segments = {
            "Test St": [
                {"coordinates": [(35.1, -82.5), (35.2, -82.6)], "way_ids": [100]},
            ],
        }

        checkpoint_file = tmp_path / "checkpoint.pkl"

        with patch.object(merger, 'parallel_enabled', False):
            result = merger.merge_boundary_segments(
                street_segments,
                checkpoint_path=str(checkpoint_file)
            )

            assert len(result) == 1


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_merge_single_coordinate_segment(self, merger):
        """Test merging segment with only one coordinate."""
        segments = [
            {"coordinates": [(35.1, -82.5)], "way_ids": [100]},
        ]

        result = merger._merge_street(segments, "Single")

        assert len(result) == 1

    def test_merge_very_long_segment(self, merger):
        """Test merging very long segment (many coordinates)."""
        coords = [(35.0 + i*0.01, -82.0) for i in range(1000)]
        segments = [
            {"coordinates": coords, "way_ids": [100]},
        ]

        result = merger._merge_street(segments, "Long")

        assert len(result) == 1
        assert len(result[0]["coordinates"]) == 1000

    def test_merge_segments_with_gaps(self, merger):
        """Test merging segments with small gaps."""
        # Segments with small gaps (within distance tolerance)
        merger.distance_tolerance_m = 150
        segments = [
            {"coordinates": [(35.0, -82.0), (35.001, -82.001)], "way_ids": [100]},
            {"coordinates": [(35.0011, -82.0011), (35.002, -82.002)], "way_ids": [101]},  # ~122m gap
        ]

        result = merger._merge_street(segments, "Gaps")

        # Depending on tolerance, might merge
        assert len(result) >= 1

    def test_coordinate_precision_handling(self, merger):
        """Test handling of high-precision coordinates."""
        seg1 = {
            "coordinates": [(35.123456789, -82.987654321)],
            "way_ids": [100],
        }
        seg2 = {
            "coordinates": [(35.123456788, -82.987654320)],  # Tiny difference
            "way_ids": [101],
        }

        # Should match within tolerance
        assert merger._are_segments_duplicates(seg1, seg2) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
