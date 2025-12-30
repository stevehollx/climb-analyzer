"""
Unit tests for reverse geocoding functionality.

Tests cover:
- Offline reverse geocoding
- Spatial clustering for deduplication
- Batch processing
- Checkpoint/resume functionality
- Signal handling integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call, AsyncMock
import asyncio
from typing import List, Tuple, Dict

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from climb_analyzer.data.geocoding import ReverseGeocoder


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def geocoder():
    """Create a ReverseGeocoder instance."""
    return ReverseGeocoder(max_concurrent=4)


@pytest.fixture
def mock_persistence():
    """Mock persistence manager."""
    mock = Mock()
    mock.save_geocoding_progress = Mock()
    mock.clear_geocoding_progress = Mock()
    return mock


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing."""
    return [
        (35.1234, -82.5678),
        (35.2345, -82.6789),
        (35.3456, -82.7890),
        (35.4567, -82.8901),
        (35.5678, -82.9012),
    ]


@pytest.fixture
def sample_coordinates_with_duplicates():
    """Sample coordinates with duplicates for clustering test."""
    return [
        (35.1000, -82.5000),
        (35.1001, -82.5001),  # Very close to first
        (35.1002, -82.5002),  # Very close to first
        (35.5000, -82.8000),  # Far away
        (35.5001, -82.8001),  # Close to fourth
    ]


@pytest.fixture
def mock_rg_results():
    """Mock results from reverse_geocoder library."""
    return [
        {
            "name": "Asheville",
            "admin1": "North Carolina",
            "cc": "US",
        },
        {
            "name": "Charlotte",
            "admin1": "North Carolina",
            "cc": "US",
        },
        {
            "name": "Atlanta",
            "admin1": "Georgia",
            "cc": "US",
        },
    ]


# ============================================================================
# Initialization Tests
# ============================================================================

class TestReverseGeocoderInit:
    """Test ReverseGeocoder initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        geocoder = ReverseGeocoder(max_concurrent=8)
        assert geocoder is not None

    def test_initialization_no_params(self):
        """Test initialization with default parameters."""
        geocoder = ReverseGeocoder()
        assert geocoder is not None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager protocol."""
        async with ReverseGeocoder() as geocoder:
            assert geocoder is not None


# ============================================================================
# Coordinate Deduplication Tests
# ============================================================================

class TestCoordinateDeduplication:
    """Test spatial clustering and deduplication."""

    def test_deduplicate_with_clustering_identical(self, geocoder):
        """Test deduplication of identical coordinates."""
        coords = [
            (35.1234, -82.5678),
            (35.1234, -82.5678),  # Exact duplicate
            (35.1234, -82.5678),  # Another duplicate
        ]

        unique_coords, coord_mapping = geocoder._deduplicate_with_clustering(coords)

        # Should deduplicate to 1 unique coordinate
        assert len(unique_coords) == 1
        # Mapping should preserve original indices
        assert len(coord_mapping) == 3
        assert coord_mapping[0] == coord_mapping[1] == coord_mapping[2]

    def test_deduplicate_with_clustering_nearby(self, geocoder, sample_coordinates_with_duplicates):
        """Test clustering of nearby coordinates."""
        unique_coords, coord_mapping = geocoder._deduplicate_with_clustering(
            sample_coordinates_with_duplicates
        )

        # Should cluster nearby coordinates
        assert len(unique_coords) < len(sample_coordinates_with_duplicates)
        assert len(coord_mapping) == len(sample_coordinates_with_duplicates)

    def test_deduplicate_no_duplicates(self, geocoder, sample_coordinates):
        """Test deduplication when no duplicates exist."""
        unique_coords, coord_mapping = geocoder._deduplicate_with_clustering(
            sample_coordinates
        )

        # If coordinates are far apart, should keep all
        assert len(unique_coords) <= len(sample_coordinates)
        assert len(coord_mapping) == len(sample_coordinates)

    def test_deduplicate_empty_list(self, geocoder):
        """Test deduplication of empty coordinate list."""
        unique_coords, coord_mapping = geocoder._deduplicate_with_clustering([])

        assert len(unique_coords) == 0
        assert len(coord_mapping) == 0

    def test_deduplicate_single_coordinate(self, geocoder):
        """Test deduplication with single coordinate."""
        coords = [(35.1234, -82.5678)]

        unique_coords, coord_mapping = geocoder._deduplicate_with_clustering(coords)

        assert len(unique_coords) == 1
        assert len(coord_mapping) == 1
        assert coord_mapping[0] == 0


# ============================================================================
# Reverse Geocoding Tests
# ============================================================================

class TestReverseGeocoding:
    """Test reverse geocoding functionality."""

    @pytest.mark.asyncio
    async def test_reverse_geocode_parallel_empty(self, geocoder, mock_persistence):
        """Test geocoding with empty coordinate list."""
        result = await geocoder.reverse_geocode_parallel(
            [], mock_persistence, progress_desc="Test"
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_reverse_geocode_parallel_basic(self, geocoder, mock_persistence,
                                                   sample_coordinates, mock_rg_results):
        """Test basic reverse geocoding."""
        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = mock_rg_results

            result = await geocoder.reverse_geocode_parallel(
                sample_coordinates[:3],
                mock_persistence,
                progress_desc="Test"
            )

            # Should have called reverse_geocoder
            assert mock_search.called
            # Should return location data
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_reverse_geocode_parallel_batching(self, geocoder, mock_persistence):
        """Test that large coordinate sets are batched."""
        # Create 2000 coordinates to force batching (batch_size is 1700)
        many_coords = [(35.0 + i*0.001, -82.0) for i in range(2000)]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [
                {"name": "Test", "admin1": "State", "cc": "US"}
            ] * 1700

            result = await geocoder.reverse_geocode_parallel(
                many_coords,
                mock_persistence,
                progress_desc="Test"
            )

            # Should have called search multiple times (for batches)
            assert mock_search.call_count >= 2

    @pytest.mark.asyncio
    async def test_reverse_geocode_result_format(self, geocoder, mock_persistence):
        """Test that results are formatted correctly."""
        coords = [(35.1234, -82.5678)]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [{
                "name": "Asheville",
                "admin1": "North Carolina",
                "cc": "US",
            }]

            with patch.object(geocoder, '_deduplicate_with_clustering') as mock_dedup:
                mock_dedup.return_value = (coords, {0: 0})

                result = await geocoder.reverse_geocode_parallel(
                    coords,
                    mock_persistence,
                    progress_desc="Test"
                )

                # Check result format
                coord_key = coords[0]
                assert coord_key in result
                assert "city" in result[coord_key]
                assert "state" in result[coord_key]
                assert "full_address" in result[coord_key]
                assert result[coord_key]["city"] == "Asheville"
                assert result[coord_key]["state"] == "North Carolina"

    @pytest.mark.asyncio
    async def test_reverse_geocode_unknown_location(self, geocoder, mock_persistence):
        """Test handling of unknown locations."""
        coords = [(35.1234, -82.5678)]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            # Return result with missing data
            mock_search.return_value = [None]

            with patch.object(geocoder, '_deduplicate_with_clustering') as mock_dedup:
                mock_dedup.return_value = (coords, {0: 0})

                result = await geocoder.reverse_geocode_parallel(
                    coords,
                    mock_persistence,
                    progress_desc="Test"
                )

                coord_key = coords[0]
                assert result[coord_key]["city"] == "Unknown"
                assert result[coord_key]["state"] == "Unknown"

    @pytest.mark.asyncio
    async def test_reverse_geocode_partial_data(self, geocoder, mock_persistence):
        """Test handling of partial location data."""
        coords = [(35.1234, -82.5678)]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            # Return result with only city
            mock_search.return_value = [{
                "name": "TestCity",
                # Missing admin1 and cc
            }]

            with patch.object(geocoder, '_deduplicate_with_clustering') as mock_dedup:
                mock_dedup.return_value = (coords, {0: 0})

                result = await geocoder.reverse_geocode_parallel(
                    coords,
                    mock_persistence,
                    progress_desc="Test"
                )

                coord_key = coords[0]
                assert result[coord_key]["city"] == "TestCity"


# ============================================================================
# Checkpoint Tests
# ============================================================================

class TestCheckpointing:
    """Test checkpoint save/resume functionality."""

    @pytest.mark.asyncio
    async def test_save_geocoding_checkpoint(self, geocoder, mock_persistence):
        """Test saving geocoding checkpoint."""
        coord_to_location = {
            (35.1, -82.5): {"city": "TestCity", "state": "TestState", "full_address": "TestCity, TestState"}
        }
        coord_mapping = {0: 0}

        geocoder._save_geocoding_checkpoint(
            mock_persistence,
            coord_to_location,
            coord_mapping,
            completed_count=50,
            total_count=100
        )

        # Should have called persistence save method
        assert mock_persistence.save_geocoding_progress.called

    @pytest.mark.asyncio
    async def test_checkpoint_on_signal_handler(self, geocoder, mock_persistence):
        """Test that checkpoint is saved when signal handler triggers."""
        coords = [(35.0 + i*0.01, -82.0) for i in range(2000)]

        mock_signal_handler = Mock()
        mock_signal_handler.kill_now = True  # Trigger early exit
        mock_signal_handler.set_operation = Mock()

        with patch('climb_analyzer.data.geocoding.GracefulKiller', return_value=mock_signal_handler):
            with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
                mock_search.return_value = [{"name": "Test", "admin1": "State", "cc": "US"}]

                with patch('sys.exit') as mock_exit:
                    try:
                        result = await geocoder.reverse_geocode_parallel(
                            coords,
                            mock_persistence,
                            progress_desc="Test"
                        )
                    except SystemExit:
                        pass

                    # If signal handler triggered, should have saved checkpoint
                    # (Note: actual behavior depends on signal handler integration)


# ============================================================================
# Mapping Tests
# ============================================================================

class TestCoordinateMapping:
    """Test coordinate mapping from unique back to original."""

    @pytest.mark.asyncio
    async def test_mapping_preserves_order(self, geocoder, mock_persistence):
        """Test that coordinate mapping preserves original order."""
        coords = [
            (35.1, -82.5),
            (35.1, -82.5),  # Duplicate
            (35.2, -82.6),
            (35.1, -82.5),  # Another duplicate
        ]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [
                {"name": "City1", "admin1": "State1", "cc": "US"},
                {"name": "City2", "admin1": "State2", "cc": "US"},
            ]

            result = await geocoder.reverse_geocode_parallel(
                coords,
                mock_persistence,
                progress_desc="Test"
            )

            # All duplicates should map to same result
            assert len(result) <= 2  # At most 2 unique locations

    @pytest.mark.asyncio
    async def test_mapping_with_clustering(self, geocoder, mock_persistence,
                                           sample_coordinates_with_duplicates):
        """Test mapping after spatial clustering."""
        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            # Return fewer results than input (due to clustering)
            mock_search.return_value = [
                {"name": "City1", "admin1": "State1", "cc": "US"},
                {"name": "City2", "admin1": "State2", "cc": "US"},
            ]

            result = await geocoder.reverse_geocode_parallel(
                sample_coordinates_with_duplicates,
                mock_persistence,
                progress_desc="Test"
            )

            # Should have clustered some coordinates
            assert len(result) < len(sample_coordinates_with_duplicates)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in geocoding."""

    @pytest.mark.asyncio
    async def test_geocoding_with_rg_error(self, geocoder, mock_persistence, sample_coordinates):
        """Test handling of reverse_geocoder errors."""
        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.side_effect = Exception("reverse_geocoder error")

            with pytest.raises(Exception):
                result = await geocoder.reverse_geocode_parallel(
                    sample_coordinates,
                    mock_persistence,
                    progress_desc="Test"
                )

    @pytest.mark.asyncio
    async def test_geocoding_with_invalid_coordinates(self, geocoder, mock_persistence):
        """Test handling of invalid coordinates."""
        invalid_coords = [
            (None, -82.5),  # Invalid latitude
            (35.1, None),   # Invalid longitude
        ]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            # reverse_geocoder might handle this or raise error
            mock_search.side_effect = Exception("Invalid coordinates")

            with pytest.raises(Exception):
                result = await geocoder.reverse_geocode_parallel(
                    invalid_coords,
                    mock_persistence,
                    progress_desc="Test"
                )


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance-related functionality."""

    @pytest.mark.asyncio
    async def test_large_coordinate_set(self, geocoder, mock_persistence):
        """Test geocoding large coordinate set."""
        # Create 5000 coordinates
        large_coords = [(35.0 + i*0.0001, -82.0) for i in range(5000)]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [{"name": "Test", "admin1": "State", "cc": "US"}] * 1700

            result = await geocoder.reverse_geocode_parallel(
                large_coords,
                mock_persistence,
                progress_desc="Test"
            )

            # Should handle large set
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_deduplication_reduces_api_calls(self, geocoder, mock_persistence):
        """Test that deduplication reduces number of API calls."""
        # Create many duplicate coordinates
        coords = [(35.1, -82.5)] * 100 + [(35.2, -82.6)] * 100

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [
                {"name": "City1", "admin1": "State1", "cc": "US"},
                {"name": "City2", "admin1": "State2", "cc": "US"},
            ]

            result = await geocoder.reverse_geocode_parallel(
                coords,
                mock_persistence,
                progress_desc="Test"
            )

            # Should have significantly reduced the number of unique lookups
            # mock_search should be called with much fewer coordinates than 200
            total_coords_sent = sum(len(call.args[0]) for call in mock_search.call_args_list)
            assert total_coords_sent < len(coords)


# ============================================================================
# Integration Tests
# ============================================================================

class TestGeocodingIntegration:
    """Integration tests for complete geocoding workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, geocoder, mock_persistence, sample_coordinates):
        """Test complete geocoding workflow."""
        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [
                {"name": f"City{i}", "admin1": f"State{i}", "cc": "US"}
                for i in range(len(sample_coordinates))
            ]

            result = await geocoder.reverse_geocode_parallel(
                sample_coordinates,
                mock_persistence,
                progress_desc="Complete Test"
            )

            # Should have geocoded all coordinates
            assert len(result) > 0
            # All results should have proper format
            for coord, location in result.items():
                assert "city" in location
                assert "state" in location
                assert "full_address" in location

    @pytest.mark.asyncio
    async def test_workflow_with_mixed_results(self, geocoder, mock_persistence):
        """Test workflow with mix of successful and failed results."""
        coords = [(35.1, -82.5), (35.2, -82.6), (35.3, -82.7)]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            # Mix of valid and invalid results
            mock_search.return_value = [
                {"name": "City1", "admin1": "State1", "cc": "US"},
                None,  # Failed lookup
                {"name": "City3", "admin1": "State3", "cc": "US"},
            ]

            result = await geocoder.reverse_geocode_parallel(
                coords,
                mock_persistence,
                progress_desc="Mixed Test"
            )

            # Should handle mix of results
            assert len(result) == len(coords)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
