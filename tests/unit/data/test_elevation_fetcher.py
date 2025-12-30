"""
Unit tests for elevation data fetching functionality.

Tests cover:
- Batch processing and parallel execution
- Fallback logic between datasets
- Rate limiting and retry logic
- Adaptive batch sizing
- Coordinate deduplication
- Error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import requests
from typing import List, Tuple, Optional
import time

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from climb_analyzer.data.elevation import (
    FastElevationFetcher,
    ElevationFetchLog,
    build_elevation_url,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_persistence():
    """Mock persistence manager for checkpointing."""
    mock = Mock()
    mock.save_elevation_progress = Mock()
    mock.clear_elevation_progress = Mock()
    return mock


@pytest.fixture
def sample_coords():
    """Sample coordinate list for testing."""
    return [
        (35.1234, -82.5678),
        (35.2345, -82.6789),
        (35.3456, -82.7890),
        (35.4567, -82.8901),
    ]


@pytest.fixture
def sample_coord_metadata():
    """Sample coordinate metadata for testing."""
    return {
        (35.1234, -82.5678): {"way_id": "123", "street_name": "Main St"},
        (35.2345, -82.6789): {"way_id": "456", "street_name": "Oak Ave"},
        (35.3456, -82.7890): {"way_id": "789", "street_name": "Hill Rd"},
        (35.4567, -82.8901): {"way_id": "101", "street_name": "Mountain Way"},
    }


@pytest.fixture
def elevation_fetcher():
    """Create a FastElevationFetcher instance."""
    with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
        mock_build.return_value = "http://localhost:5000/v1/srtm30m"
        fetcher = FastElevationFetcher(primary_dataset="srtm30m")
        return fetcher


@pytest.fixture
def fetch_log():
    """Create an ElevationFetchLog instance."""
    return ElevationFetchLog()


# ============================================================================
# ElevationFetchLog Tests
# ============================================================================

class TestElevationFetchLog:
    """Test the ElevationFetchLog tracking class."""

    def test_initialization(self, fetch_log):
        """Test that log is properly initialized."""
        assert fetch_log.total_requested == 0
        assert fetch_log.total_successful == 0
        assert fetch_log.total_failed == 0
        assert fetch_log.total_fallback_success == 0
        assert fetch_log.failed_fetches == []
        assert fetch_log.fallback_successes == []

    def test_add_failed_fetch(self, fetch_log):
        """Test recording failed fetches."""
        coord = (35.1234, -82.5678)
        fetch_log.add_failed_fetch(
            coord=coord,
            way_id="12345",
            street_name="Test Street",
            datasets_tried=["srtm30m", "ned10m"]
        )

        assert fetch_log.total_failed == 1
        assert len(fetch_log.failed_fetches) == 1
        assert fetch_log.failed_fetches[0]["coordinate"] == coord
        assert fetch_log.failed_fetches[0]["way_id"] == "12345"
        assert fetch_log.failed_fetches[0]["street_name"] == "Test Street"

    def test_add_successful_fetch_primary(self, fetch_log):
        """Test recording successful fetch from primary dataset."""
        fetch_log.add_successful_fetch(used_fallback=False, dataset="srtm30m")

        assert fetch_log.total_successful == 1
        assert fetch_log.total_fallback_success == 0

    def test_add_successful_fetch_fallback(self, fetch_log):
        """Test recording successful fetch from fallback dataset."""
        fetch_log.add_successful_fetch(used_fallback=True, dataset="ned10m")

        assert fetch_log.total_successful == 1
        assert fetch_log.total_fallback_success == 1

    def test_print_summary(self, fetch_log, capsys):
        """Test that summary prints without errors."""
        fetch_log.total_requested = 100
        fetch_log.total_successful = 95
        fetch_log.total_failed = 5
        fetch_log.total_fallback_success = 10

        fetch_log.print_summary()
        captured = capsys.readouterr()

        assert "ELEVATION FETCH SUMMARY" in captured.out
        assert "100" in captured.out
        assert "95" in captured.out


# ============================================================================
# build_elevation_url Tests
# ============================================================================

class TestBuildElevationUrl:
    """Test URL building functionality."""

    def test_build_url_success(self):
        """Test building a valid URL."""
        with patch('climb_analyzer.data.elevation.TOPO_API_BASE_URL', 'http://localhost:5000/v1'):
            url = build_elevation_url("srtm30m")
            assert url == "http://localhost:5000/v1/srtm30m"

    def test_build_url_with_trailing_slash(self):
        """Test that trailing slashes are handled correctly."""
        with patch('climb_analyzer.data.elevation.TOPO_API_BASE_URL', 'http://localhost:5000/v1/'):
            url = build_elevation_url("ned10m")
            assert url == "http://localhost:5000/v1/ned10m"

    def test_build_url_no_base_url(self):
        """Test behavior when base URL is not configured."""
        with patch('climb_analyzer.data.elevation.TOPO_API_BASE_URL', None):
            url = build_elevation_url("srtm30m")
            assert url is None


# ============================================================================
# FastElevationFetcher Tests
# ============================================================================

class TestFastElevationFetcherInit:
    """Test FastElevationFetcher initialization."""

    def test_initialization_default_dataset(self):
        """Test initialization with default dataset."""
        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/srtm30m"
            fetcher = FastElevationFetcher()

            assert fetcher.primary_dataset == "srtm30m"
            assert fetcher.primary_url == "http://localhost:5000/v1/srtm30m"
            assert fetcher.optimal_batch_size == 500  # Default from config
            assert fetcher.batch_size_adapted is False

    def test_initialization_custom_dataset(self):
        """Test initialization with custom dataset."""
        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/ned10m"
            fetcher = FastElevationFetcher(primary_dataset="ned10m")

            assert fetcher.primary_dataset == "ned10m"
            assert fetcher.primary_url == "http://localhost:5000/v1/ned10m"

    def test_configure_fallback_endpoints(self):
        """Test that fallback endpoints are configured correctly."""
        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            # Mock different URLs for different datasets
            def build_url_side_effect(dataset):
                return f"http://localhost:5000/v1/{dataset}"

            mock_build.side_effect = build_url_side_effect
            fetcher = FastElevationFetcher(primary_dataset="srtm30m")

            # Should have ned10m and aster30m as fallbacks (not srtm30m since it's primary)
            assert "http://localhost:5000/v1/ned10m" in fetcher.fallback_urls
            assert "http://localhost:5000/v1/aster30m" in fetcher.fallback_urls
            assert "http://localhost:5000/v1/srtm30m" not in fetcher.fallback_urls


class TestCoordinateMatching:
    """Test coordinate matching and deduplication logic."""

    def test_coordinate_deduplication(self, elevation_fetcher, mock_persistence, sample_coords):
        """Test that duplicate coordinates are deduplicated."""
        # Add duplicate coordinates
        coords_with_dupes = sample_coords + [sample_coords[0], sample_coords[1]]

        with patch.object(elevation_fetcher, '_fetch_single_batch') as mock_fetch:
            mock_fetch.return_value = ([100.0, 200.0, 300.0, 400.0], [], {})

            results = elevation_fetcher.fetch_elevations_for_coordinates(
                coords_with_dupes, mock_persistence, progress_desc=None
            )

            # Should only fetch unique coordinates
            assert len(results) == len(coords_with_dupes)  # Original length maintained
            # First and last two results should match (duplicates)
            assert results[0] == results[4]
            assert results[1] == results[5]

    def test_coordinate_rounding(self, elevation_fetcher, mock_persistence):
        """Test that coordinates are rounded to 6 decimal places."""
        # Coordinates that round to the same value
        coords = [
            (35.1234561, -82.5678901),
            (35.1234562, -82.5678902),  # Should be treated as duplicate
        ]

        with patch.object(elevation_fetcher, '_fetch_single_batch') as mock_fetch:
            mock_fetch.return_value = ([100.0], [], {})

            results = elevation_fetcher.fetch_elevations_for_coordinates(
                coords, mock_persistence, progress_desc=None
            )

            # Both should have same elevation (deduplicated)
            assert results[0] == results[1]
            assert len(results) == 2


class TestBatchFetching:
    """Test batch fetching and API interaction."""

    def test_successful_batch_fetch(self, elevation_fetcher):
        """Test successful batch fetching from primary endpoint."""
        coords = [(35.1, -82.5), (35.2, -82.6)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "OK",
                "results": [
                    {"elevation": 100.0},
                    {"elevation": 200.0}
                ]
            }
            mock_get.return_value = mock_response

            result, urls_tried, sources = elevation_fetcher._fetch_single_batch(
                coords, max_retries=3, base_delay=0.1, silent_mode=True
            )

            assert result == [100.0, 200.0]
            assert len(urls_tried) == 1

    def test_batch_fetch_with_nulls(self, elevation_fetcher):
        """Test batch fetching with some null elevations."""
        coords = [(35.1, -82.5), (35.2, -82.6), (35.3, -82.7)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "OK",
                "results": [
                    {"elevation": 100.0},
                    {"elevation": None},  # Missing data
                    {"elevation": 300.0}
                ]
            }
            mock_get.return_value = mock_response

            result, urls_tried, sources = elevation_fetcher._fetch_single_batch(
                coords, max_retries=3, base_delay=0.1, silent_mode=True
            )

            assert result[0] == 100.0
            assert result[1] is None
            assert result[2] == 300.0


class TestFallbackLogic:
    """Test fallback logic between datasets."""

    def test_fallback_on_primary_failure(self, elevation_fetcher):
        """Test that fallback is used when primary fails."""
        coords = [(35.1, -82.5)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            # First call (primary) fails, second call (fallback) succeeds
            primary_response = Mock()
            primary_response.ok = False
            primary_response.status_code = 500

            fallback_response = Mock()
            fallback_response.ok = True
            fallback_response.status_code = 200
            fallback_response.json.return_value = {
                "status": "OK",
                "results": [{"elevation": 150.0}]
            }

            mock_get.side_effect = [primary_response] + [fallback_response] * 10

            result, urls_tried, sources = elevation_fetcher._fetch_single_batch(
                coords, max_retries=1, base_delay=0.01, silent_mode=True
            )

            assert result == [150.0]
            assert len(urls_tried) >= 2  # Primary + at least one fallback

    def test_fallback_fills_missing_data(self, elevation_fetcher):
        """Test that fallback fills in missing data from primary."""
        coords = [(35.1, -82.5), (35.2, -82.6)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            # Primary returns partial data
            primary_response = Mock()
            primary_response.ok = True
            primary_response.status_code = 200
            primary_response.json.return_value = {
                "status": "OK",
                "results": [
                    {"elevation": 100.0},
                    {"elevation": None}  # Missing
                ]
            }

            # Fallback provides missing data
            fallback_response = Mock()
            fallback_response.ok = True
            fallback_response.status_code = 200
            fallback_response.json.return_value = {
                "status": "OK",
                "results": [
                    {"elevation": 100.0},
                    {"elevation": 200.0}  # Fills missing
                ]
            }

            mock_get.side_effect = [primary_response, fallback_response]

            result, urls_tried, sources = elevation_fetcher._fetch_single_batch(
                coords, max_retries=1, base_delay=0.01, silent_mode=True
            )

            assert result == [100.0, 200.0]
            assert 0 in sources  # Primary provided first coord
            assert 1 in sources  # Fallback provided second coord


class TestRateLimiting:
    """Test rate limiting and retry logic."""

    def test_retry_on_429(self, elevation_fetcher):
        """Test retry logic on HTTP 429 rate limit."""
        coords = [(35.1, -82.5)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            with patch('time.sleep'):  # Speed up test
                # Fail with 429 twice, then succeed
                rate_limit_response = Mock()
                rate_limit_response.ok = False
                rate_limit_response.status_code = 429

                success_response = Mock()
                success_response.ok = True
                success_response.status_code = 200
                success_response.json.return_value = {
                    "status": "OK",
                    "results": [{"elevation": 100.0}]
                }

                mock_get.side_effect = [
                    rate_limit_response,
                    rate_limit_response,
                    success_response
                ]

                result, urls_tried, sources = elevation_fetcher._fetch_single_batch(
                    coords, max_retries=3, base_delay=0.01, silent_mode=True
                )

                assert result == [100.0]
                assert mock_get.call_count == 3

    def test_retry_on_504_timeout(self, elevation_fetcher):
        """Test retry logic on HTTP 504 gateway timeout."""
        coords = [(35.1, -82.5)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            with patch('time.sleep'):  # Speed up test
                timeout_response = Mock()
                timeout_response.ok = False
                timeout_response.status_code = 504

                success_response = Mock()
                success_response.ok = True
                success_response.status_code = 200
                success_response.json.return_value = {
                    "status": "OK",
                    "results": [{"elevation": 100.0}]
                }

                mock_get.side_effect = [timeout_response, success_response]

                result, urls_tried, sources = elevation_fetcher._fetch_single_batch(
                    coords, max_retries=2, base_delay=0.01, silent_mode=True
                )

                assert result == [100.0]

    def test_max_retries_exceeded(self, elevation_fetcher):
        """Test behavior when max retries are exceeded."""
        coords = [(35.1, -82.5)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            with patch('time.sleep'):  # Speed up test
                error_response = Mock()
                error_response.ok = False
                error_response.status_code = 500
                mock_get.return_value = error_response

                result, urls_tried, sources = elevation_fetcher._fetch_single_batch_from_endpoint(
                    coords,
                    "http://localhost:5000/v1/srtm30m",
                    max_retries=2,
                    base_delay=0.01,
                    silent_mode=True
                )

                assert result is None
                assert mock_get.call_count == 3  # Initial + 2 retries


class TestAdaptiveBatchSizing:
    """Test adaptive batch sizing logic."""

    def test_reduce_batch_size(self, elevation_fetcher):
        """Test batch size reduction."""
        original_size = elevation_fetcher.optimal_batch_size

        reduced = elevation_fetcher._reduce_batch_size(reason="Test")

        assert reduced is True
        assert elevation_fetcher.optimal_batch_size < original_size
        assert elevation_fetcher.optimal_batch_size == int(original_size * 0.9)
        assert elevation_fetcher.batch_size_adapted is True

    def test_reduce_batch_size_minimum(self, elevation_fetcher):
        """Test that batch size doesn't go below minimum."""
        elevation_fetcher.optimal_batch_size = 99  # At minimum

        reduced = elevation_fetcher._reduce_batch_size(reason="Test")

        assert reduced is False
        assert elevation_fetcher.optimal_batch_size == 99

    def test_batch_size_reduction_progressive(self, elevation_fetcher):
        """Test progressive batch size reduction."""
        sizes = [elevation_fetcher.optimal_batch_size]

        for i in range(5):
            elevation_fetcher._reduce_batch_size(reason="Test")
            sizes.append(elevation_fetcher.optimal_batch_size)

        # Ensure sizes are decreasing
        assert all(sizes[i] > sizes[i+1] for i in range(len(sizes)-1))


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_connection_error(self, elevation_fetcher):
        """Test handling of connection errors."""
        coords = [(35.1, -82.5)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            with patch('time.sleep'):  # Speed up test
                mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

                result, urls_tried, sources = elevation_fetcher._fetch_single_batch_from_endpoint(
                    coords,
                    "http://localhost:5000/v1/srtm30m",
                    max_retries=1,
                    base_delay=0.01,
                    silent_mode=True
                )

                assert result is None

    def test_timeout_error(self, elevation_fetcher):
        """Test handling of timeout errors."""
        coords = [(35.1, -82.5)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            with patch('time.sleep'):
                mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

                result, urls_tried, sources = elevation_fetcher._fetch_single_batch_from_endpoint(
                    coords,
                    "http://localhost:5000/v1/srtm30m",
                    max_retries=1,
                    base_delay=0.01,
                    silent_mode=True
                )

                assert result is None

    def test_invalid_json_response(self, elevation_fetcher):
        """Test handling of invalid JSON responses."""
        coords = [(35.1, -82.5)]

        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = mock_response

            result, urls_tried, sources = elevation_fetcher._fetch_single_batch_from_endpoint(
                coords,
                "http://localhost:5000/v1/srtm30m",
                max_retries=0,
                base_delay=0.01,
                silent_mode=True
            )

            assert result == [None]

    def test_empty_coordinates(self, elevation_fetcher, mock_persistence):
        """Test handling of empty coordinate list."""
        with patch.object(elevation_fetcher, '_fetch_single_batch') as mock_fetch:
            result = elevation_fetcher.fetch_elevations_for_coordinates(
                [], mock_persistence, progress_desc=None
            )

            assert result == []
            mock_fetch.assert_not_called()


class TestParallelProcessing:
    """Test parallel processing functionality."""

    @patch('climb_analyzer.data.elevation.ThreadPoolExecutor')
    @patch('climb_analyzer.data.elevation.SmartCheckpointer')
    def test_parallel_execution(self, mock_checkpointer, mock_executor, elevation_fetcher, mock_persistence):
        """Test that parallel execution uses ThreadPoolExecutor."""
        coords = [(35.1 + i*0.1, -82.5) for i in range(10)]

        with patch.object(elevation_fetcher, '_fetch_single_batch') as mock_fetch:
            mock_fetch.return_value = ([100.0], [], {})

            # Mock executor behavior
            mock_executor_instance = Mock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance

            # This will fail since we're mocking, but we can verify the setup
            try:
                elevation_fetcher._fetch_elevations_parallel(
                    coords, mock_persistence, "Test", silent_mode=True
                )
            except:
                pass

            # Verify ThreadPoolExecutor was used
            mock_executor.assert_called()


# ============================================================================
# Integration-like Tests
# ============================================================================

class TestEndToEndFetching:
    """End-to-end tests for complete fetching workflow."""

    def test_complete_fetch_workflow(self, elevation_fetcher, mock_persistence, sample_coords):
        """Test complete workflow from coordinates to elevations."""
        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "OK",
                "results": [
                    {"elevation": 100.0 + i * 50} for i in range(len(sample_coords))
                ]
            }
            mock_get.return_value = mock_response

            result = elevation_fetcher.fetch_elevations_for_coordinates(
                sample_coords, mock_persistence, progress_desc=None
            )

            assert len(result) == len(sample_coords)
            assert all(e is not None for e in result)
            assert result[0] == 100.0
            assert result[1] == 150.0

    def test_fetch_with_metadata_and_log(self, elevation_fetcher, mock_persistence,
                                         sample_coords, sample_coord_metadata, fetch_log):
        """Test fetching with metadata and logging."""
        with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "OK",
                "results": [
                    {"elevation": 100.0 + i * 50} for i in range(len(sample_coords))
                ]
            }
            mock_get.return_value = mock_response

            result = elevation_fetcher.fetch_elevations_for_coordinates(
                sample_coords,
                mock_persistence,
                progress_desc=None,
                coord_metadata=sample_coord_metadata,
                fetch_log=fetch_log
            )

            assert fetch_log.total_successful == len(sample_coords)
            assert fetch_log.total_failed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
