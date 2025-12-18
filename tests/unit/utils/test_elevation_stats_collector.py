"""
Unit tests for elevation_stats_collector.py.

Tests the elevation statistics collection functionality including:
- Singleton pattern
- Statistics recording and retrieval
- Per-way failure tracking
- Success/failure rate calculations
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from utils.elevation_stats_collector import (
    ElevationStatsCollector,
    get_stats_collector,
    reset_stats,
    record_elevation_fetch,
    get_elevation_stats,
    has_elevation_stats,
    record_way_failure,
    get_way_failures
)


# ============================================================================
# Singleton Pattern Tests
# ============================================================================

@pytest.mark.unit
class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, mock_elevation_stats_collector):
        """Test that ElevationStatsCollector is a singleton."""
        collector1 = ElevationStatsCollector()
        collector2 = ElevationStatsCollector()

        # Both should be the same instance
        assert collector1 is collector2

    def test_get_stats_collector(self):
        """Test get_stats_collector returns singleton instance."""
        collector1 = get_stats_collector()
        collector2 = get_stats_collector()

        assert collector1 is collector2

    def test_singleton_state_persistence(self, mock_elevation_stats_collector):
        """Test that singleton maintains state across instances."""
        collector1 = ElevationStatsCollector()
        collector1.record_fetch(100, 5, 95)

        collector2 = ElevationStatsCollector()

        # Second instance should have same stats
        assert collector2.total_coords_requested == 100
        assert collector2.total_coords_failed == 5
        assert collector2.total_unique_coords == 95


# ============================================================================
# Basic Statistics Recording Tests
# ============================================================================

@pytest.mark.unit
class TestStatisticsRecording:
    """Test basic statistics recording functionality."""

    def test_initial_state(self, mock_elevation_stats_collector):
        """Test initial state of collector."""
        collector = ElevationStatsCollector()

        assert collector.total_coords_requested == 0
        assert collector.total_coords_failed == 0
        assert collector.total_unique_coords == 0
        assert collector.runs_count == 0
        assert collector.way_failures == {}

    def test_record_single_fetch(self, mock_elevation_stats_collector):
        """Test recording a single fetch operation."""
        collector = ElevationStatsCollector()
        collector.record_fetch(total_requested=100, failed=5, unique=95)

        assert collector.total_coords_requested == 100
        assert collector.total_coords_failed == 5
        assert collector.total_unique_coords == 95
        assert collector.runs_count == 1

    def test_record_multiple_fetches(self, mock_elevation_stats_collector):
        """Test recording multiple fetch operations."""
        collector = ElevationStatsCollector()

        collector.record_fetch(100, 5, 95)
        collector.record_fetch(200, 10, 190)
        collector.record_fetch(150, 3, 147)

        assert collector.total_coords_requested == 450
        assert collector.total_coords_failed == 18
        assert collector.total_unique_coords == 432
        assert collector.runs_count == 3

    def test_record_perfect_fetch(self, mock_elevation_stats_collector):
        """Test recording fetch with zero failures."""
        collector = ElevationStatsCollector()
        collector.record_fetch(total_requested=1000, failed=0, unique=1000)

        assert collector.total_coords_requested == 1000
        assert collector.total_coords_failed == 0
        assert collector.total_unique_coords == 1000

    def test_record_complete_failure(self, mock_elevation_stats_collector):
        """Test recording fetch where all coordinates failed."""
        collector = ElevationStatsCollector()
        collector.record_fetch(total_requested=100, failed=100, unique=100)

        assert collector.total_coords_requested == 100
        assert collector.total_coords_failed == 100


# ============================================================================
# Reset Functionality Tests
# ============================================================================

@pytest.mark.unit
class TestResetFunctionality:
    """Test statistics reset functionality."""

    def test_reset_after_recording(self, mock_elevation_stats_collector):
        """Test that reset clears all statistics."""
        collector = ElevationStatsCollector()

        # Record some data
        collector.record_fetch(100, 5, 95)
        collector.record_way_failure("way123", "Test Road", 0, 50)

        # Reset
        collector.reset()

        # All stats should be zero
        assert collector.total_coords_requested == 0
        assert collector.total_coords_failed == 0
        assert collector.total_unique_coords == 0
        assert collector.runs_count == 0
        assert collector.way_failures == {}

    def test_global_reset(self, mock_elevation_stats_collector):
        """Test global reset_stats function."""
        collector = ElevationStatsCollector()
        collector.record_fetch(100, 5, 95)

        reset_stats()

        # Stats should be reset
        assert collector.total_coords_requested == 0
        assert collector.total_coords_failed == 0


# ============================================================================
# Success/Failure Rate Tests
# ============================================================================

@pytest.mark.unit
class TestSuccessFailureRates:
    """Test success and failure rate calculations."""

    def test_success_rate_perfect(self, mock_elevation_stats_collector):
        """Test success rate with zero failures."""
        collector = ElevationStatsCollector()
        collector.record_fetch(1000, 0, 1000)

        stats = collector.get_stats()
        assert stats['success_rate'] == 100.0
        assert stats['failure_rate'] == 0.0

    def test_success_rate_partial(self, mock_elevation_stats_collector):
        """Test success rate with some failures."""
        collector = ElevationStatsCollector()
        collector.record_fetch(1000, 50, 950)

        stats = collector.get_stats()
        assert stats['success_rate'] == 95.0
        assert stats['failure_rate'] == 5.0

    def test_success_rate_complete_failure(self, mock_elevation_stats_collector):
        """Test success rate with all failures."""
        collector = ElevationStatsCollector()
        collector.record_fetch(100, 100, 100)

        stats = collector.get_stats()
        assert stats['success_rate'] == 0.0
        assert stats['failure_rate'] == 100.0

    def test_success_rate_no_data(self, mock_elevation_stats_collector):
        """Test success rate with no data."""
        collector = ElevationStatsCollector()

        stats = collector.get_stats()
        assert stats['success_rate'] == 100.0  # Default to 100% when no data
        assert stats['failure_rate'] == 0.0

    def test_success_rate_multiple_runs(self, mock_elevation_stats_collector):
        """Test success rate across multiple runs."""
        collector = ElevationStatsCollector()

        # 90% success
        collector.record_fetch(1000, 100, 900)
        # 95% success
        collector.record_fetch(1000, 50, 950)

        # Overall: 1850/2000 = 92.5% success
        stats = collector.get_stats()
        assert abs(stats['success_rate'] - 92.5) < 0.01
        assert abs(stats['failure_rate'] - 7.5) < 0.01


# ============================================================================
# Per-Way Failure Tracking Tests
# ============================================================================

@pytest.mark.unit
class TestWayFailureTracking:
    """Test per-way failure tracking functionality."""

    def test_record_single_way_failure(self, mock_elevation_stats_collector):
        """Test recording failure for a single way."""
        collector = ElevationStatsCollector()

        collector.record_way_failure(
            way_id="way123",
            way_name="Mountain Road",
            coord_index=5,
            total_coords=100
        )

        way_failures = collector.get_way_failures()

        assert "way123" in way_failures
        assert way_failures["way123"]["name"] == "Mountain Road"
        assert way_failures["way123"]["total_coords"] == 100
        assert way_failures["way123"]["failed_coords"] == 1
        assert way_failures["way123"]["failed_indices"] == [5]

    def test_record_multiple_failures_same_way(self, mock_elevation_stats_collector):
        """Test recording multiple failures for same way."""
        collector = ElevationStatsCollector()

        collector.record_way_failure("way123", "Mountain Road", 5, 100)
        collector.record_way_failure("way123", "Mountain Road", 15, 100)
        collector.record_way_failure("way123", "Mountain Road", 25, 100)

        way_failures = collector.get_way_failures()

        assert way_failures["way123"]["failed_coords"] == 3
        assert way_failures["way123"]["failed_indices"] == [5, 15, 25]
        assert abs(way_failures["way123"]["failure_percentage"] - 3.0) < 0.01

    def test_record_failures_multiple_ways(self, mock_elevation_stats_collector):
        """Test recording failures for multiple ways."""
        collector = ElevationStatsCollector()

        collector.record_way_failure("way123", "Mountain Road", 5, 100)
        collector.record_way_failure("way456", "Valley Road", 10, 200)
        collector.record_way_failure("way789", "Hill Road", 3, 50)

        way_failures = collector.get_way_failures()

        assert len(way_failures) == 3
        assert "way123" in way_failures
        assert "way456" in way_failures
        assert "way789" in way_failures

    def test_way_failure_percentage(self, mock_elevation_stats_collector):
        """Test failure percentage calculation per way."""
        collector = ElevationStatsCollector()

        # 10 failures out of 100 = 10%
        for i in range(10):
            collector.record_way_failure("way123", "Test Road", i, 100)

        way_failures = collector.get_way_failures()
        assert abs(way_failures["way123"]["failure_percentage"] - 10.0) < 0.01

    def test_way_failures_sorting(self, mock_elevation_stats_collector):
        """Test that way failures are sorted by failure count."""
        collector = ElevationStatsCollector()

        # Way 1: 2 failures
        collector.record_way_failure("way1", "Road 1", 0, 100)
        collector.record_way_failure("way1", "Road 1", 1, 100)

        # Way 2: 5 failures (should be first)
        for i in range(5):
            collector.record_way_failure("way2", "Road 2", i, 100)

        # Way 3: 1 failure
        collector.record_way_failure("way3", "Road 3", 0, 100)

        way_failures = collector.get_way_failures()
        way_ids = list(way_failures.keys())

        # Should be sorted by failure count (highest first)
        assert way_ids[0] == "way2"  # 5 failures
        assert way_ids[1] == "way1"  # 2 failures
        assert way_ids[2] == "way3"  # 1 failure

    def test_way_total_coords_update(self, mock_elevation_stats_collector):
        """Test that total_coords is updated to maximum seen."""
        collector = ElevationStatsCollector()

        # First record with 100 total coords
        collector.record_way_failure("way123", "Test Road", 5, 100)

        # Later record with 150 total coords (e.g., after merging)
        collector.record_way_failure("way123", "Test Road", 120, 150)

        way_failures = collector.get_way_failures()

        # Should use the maximum (150)
        assert way_failures["way123"]["total_coords"] == 150
        assert way_failures["way123"]["failed_coords"] == 2


# ============================================================================
# Get Stats Tests
# ============================================================================

@pytest.mark.unit
class TestGetStats:
    """Test statistics retrieval functionality."""

    def test_get_stats_structure(self, mock_elevation_stats_collector):
        """Test that get_stats returns correct structure."""
        collector = ElevationStatsCollector()
        collector.record_fetch(1000, 50, 950)

        stats = collector.get_stats()

        assert 'total_coords_requested' in stats
        assert 'total_coords_failed' in stats
        assert 'total_unique_coords' in stats
        assert 'runs_count' in stats
        assert 'success_rate' in stats
        assert 'failure_rate' in stats

    def test_get_stats_values(self, mock_elevation_stats_collector):
        """Test that get_stats returns correct values."""
        collector = ElevationStatsCollector()
        collector.record_fetch(1000, 50, 950)

        stats = collector.get_stats()

        assert stats['total_coords_requested'] == 1000
        assert stats['total_coords_failed'] == 50
        assert stats['total_unique_coords'] == 950
        assert stats['runs_count'] == 1
        assert stats['success_rate'] == 95.0
        assert stats['failure_rate'] == 5.0


# ============================================================================
# Has Data Tests
# ============================================================================

@pytest.mark.unit
class TestHasData:
    """Test has_data functionality."""

    def test_has_data_initially_false(self, mock_elevation_stats_collector):
        """Test that has_data is False initially."""
        collector = ElevationStatsCollector()
        assert collector.has_data() is False

    def test_has_data_after_recording(self, mock_elevation_stats_collector):
        """Test that has_data is True after recording."""
        collector = ElevationStatsCollector()
        collector.record_fetch(100, 5, 95)

        assert collector.has_data() is True

    def test_has_data_after_reset(self, mock_elevation_stats_collector):
        """Test that has_data is False after reset."""
        collector = ElevationStatsCollector()
        collector.record_fetch(100, 5, 95)
        collector.reset()

        assert collector.has_data() is False


# ============================================================================
# Summary Text Tests
# ============================================================================

@pytest.mark.unit
class TestSummaryText:
    """Test summary text generation."""

    def test_summary_text_no_data(self, mock_elevation_stats_collector):
        """Test summary text when no data is available."""
        collector = ElevationStatsCollector()
        summary = collector.get_summary_text()

        assert "No elevation fetch statistics available" in summary

    def test_summary_text_with_data(self, mock_elevation_stats_collector):
        """Test summary text with data."""
        collector = ElevationStatsCollector()
        collector.record_fetch(10000, 500, 9500)

        summary = collector.get_summary_text()

        assert "Elevation Fetch Statistics:" in summary
        assert "10,000" in summary  # Formatted number
        assert "500" in summary
        assert "9,500" in summary
        assert "95.0%" in summary  # Success rate
        assert "5.0%" in summary   # Failure rate

    def test_summary_text_formatting(self, mock_elevation_stats_collector):
        """Test that summary text is properly formatted."""
        collector = ElevationStatsCollector()
        collector.record_fetch(1234567, 1234, 1233333)

        summary = collector.get_summary_text()

        # Should have comma-separated numbers
        assert "1,234,567" in summary


# ============================================================================
# Convenience Function Tests
# ============================================================================

@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_record_elevation_fetch(self, mock_elevation_stats_collector):
        """Test record_elevation_fetch convenience function."""
        record_elevation_fetch(100, 5, 95)

        stats = get_elevation_stats()
        assert stats['total_coords_requested'] == 100
        assert stats['total_coords_failed'] == 5

    def test_has_elevation_stats(self, mock_elevation_stats_collector):
        """Test has_elevation_stats convenience function."""
        assert has_elevation_stats() is False

        record_elevation_fetch(100, 5, 95)

        assert has_elevation_stats() is True

    def test_record_way_failure_convenience(self, mock_elevation_stats_collector):
        """Test record_way_failure convenience function."""
        record_way_failure("way123", "Test Road", 5, 100)

        way_failures = get_way_failures()
        assert "way123" in way_failures
        assert way_failures["way123"]["name"] == "Test Road"

    def test_get_way_failures_convenience(self, mock_elevation_stats_collector):
        """Test get_way_failures convenience function."""
        record_way_failure("way123", "Road 1", 0, 100)
        record_way_failure("way456", "Road 2", 0, 200)

        way_failures = get_way_failures()
        assert len(way_failures) == 2


# ============================================================================
# Edge Cases Tests
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_total_coords(self, mock_elevation_stats_collector):
        """Test with zero total coordinates."""
        collector = ElevationStatsCollector()
        collector.record_fetch(0, 0, 0)

        stats = collector.get_stats()
        assert stats['success_rate'] == 100.0  # Default to 100%
        assert stats['failure_rate'] == 0.0

    def test_large_numbers(self, mock_elevation_stats_collector):
        """Test with very large numbers."""
        collector = ElevationStatsCollector()
        collector.record_fetch(10_000_000, 100_000, 9_900_000)

        stats = collector.get_stats()
        assert stats['total_coords_requested'] == 10_000_000
        assert stats['success_rate'] == 99.0

    def test_multiple_runs_accumulation(self, mock_elevation_stats_collector):
        """Test that multiple runs accumulate correctly."""
        collector = ElevationStatsCollector()

        for i in range(10):
            collector.record_fetch(100, 5, 95)

        assert collector.total_coords_requested == 1000
        assert collector.total_coords_failed == 50
        assert collector.runs_count == 10

    def test_way_failure_with_zero_total(self, mock_elevation_stats_collector):
        """Test way failure with zero total coords."""
        collector = ElevationStatsCollector()
        collector.record_way_failure("way123", "Empty Road", 0, 0)

        way_failures = collector.get_way_failures()
        # Failure percentage should handle division by zero
        assert way_failures["way123"]["failure_percentage"] == 0.0

    def test_way_failure_all_coords_failed(self, mock_elevation_stats_collector):
        """Test way where all coordinates failed."""
        collector = ElevationStatsCollector()

        for i in range(100):
            collector.record_way_failure("way123", "Bad Road", i, 100)

        way_failures = collector.get_way_failures()
        assert way_failures["way123"]["failed_coords"] == 100
        assert way_failures["way123"]["failure_percentage"] == 100.0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.unit
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, mock_elevation_stats_collector):
        """Test complete workflow from start to finish."""
        collector = ElevationStatsCollector()

        # Simulate multiple elevation fetch operations
        collector.record_fetch(5000, 100, 4900)
        collector.record_fetch(3000, 50, 2950)
        collector.record_fetch(2000, 25, 1975)

        # Record way failures
        collector.record_way_failure("way1", "Mountain Road", 10, 500)
        collector.record_way_failure("way1", "Mountain Road", 20, 500)
        collector.record_way_failure("way2", "Valley Road", 5, 300)

        # Get statistics
        stats = collector.get_stats()
        way_failures = collector.get_way_failures()
        summary = collector.get_summary_text()

        # Verify overall stats
        assert stats['total_coords_requested'] == 10000
        assert stats['total_coords_failed'] == 175
        assert stats['runs_count'] == 3
        assert abs(stats['success_rate'] - 98.25) < 0.01

        # Verify way failures
        assert len(way_failures) == 2
        assert way_failures["way1"]["failed_coords"] == 2
        assert way_failures["way2"]["failed_coords"] == 1

        # Verify summary text
        assert "10,000" in summary
        assert "175" in summary

        # Reset and verify clean state
        collector.reset()
        assert collector.has_data() is False
        assert len(collector.way_failures) == 0
