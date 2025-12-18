"""
Unit tests for error_logger.py.

Tests error logging functionality including:
- Coordinate failure logging
- Log file rotation
- CSV streaming
- Error statistics tracking
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from utils.error_logger import ErrorLogger


# ============================================================================
# Basic Error Logger Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.error_handling
class TestErrorLoggerBasics:
    """Test basic ErrorLogger initialization and configuration."""

    def test_initialization_default(self, temp_dir):
        """Test ErrorLogger initialization with defaults."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log"
        )

        assert logger.error_log_path == temp_dir / "error.log"
        assert logger.elevation_error_log == temp_dir / "elevation_errors.log"
        assert logger.max_runs == 100
        assert logger.streaming is True
        assert logger.error_count == 0
        assert logger.elevation_errors == []

    def test_initialization_custom_params(self, temp_dir):
        """Test ErrorLogger initialization with custom parameters."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "custom_error.log",
            elevation_error_log=temp_dir / "custom_elevation.log",
            max_runs=50,
            streaming=False,
            region_name="Test Region"
        )

        assert logger.max_runs == 50
        assert logger.streaming is False
        assert logger.region_name == "Test Region"


# ============================================================================
# Coordinate Failure Logging Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.error_handling
class TestCoordinateFailureLogging:
    """Test coordinate failure logging functionality."""

    def test_log_single_coordinate_failure(self, temp_dir):
        """Test logging a single coordinate failure."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            streaming=False  # Don't write to file yet
        )

        logger.log_coordinate_failure(
            coordinate=(35.1234, -82.5678),
            street_name="Test Mountain Road",
            osm_way_id="123456789",
            datasets_tried=["ned10m", "ned30m"],
            primary_dataset="ned10m",
            successful_dataset=None,
            level="ERROR"
        )

        assert len(logger.elevation_errors) == 1
        error = logger.elevation_errors[0]

        assert error["coordinate"] == (35.1234, -82.5678)
        assert error["street_name"] == "Test Mountain Road"
        assert error["osm_way_id"] == "123456789"
        assert error["datasets_tried"] == ["ned10m", "ned30m"]
        assert error["primary_dataset"] == "ned10m"
        assert error["successful_dataset"] is None
        assert error["level"] == "ERROR"

    def test_log_fallback_success(self, temp_dir):
        """Test logging a fallback success (primary failed, secondary succeeded)."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            streaming=False
        )

        logger.log_coordinate_failure(
            coordinate=(35.2345, -82.6789),
            street_name="Backup Road",
            osm_way_id="987654321",
            datasets_tried=["ned10m", "ned30m"],
            primary_dataset="ned10m",
            successful_dataset="ned30m",
            level="INFO"
        )

        assert len(logger.elevation_errors) == 1
        error = logger.elevation_errors[0]

        assert error["level"] == "INFO"
        assert error["successful_dataset"] == "ned30m"

    def test_log_multiple_coordinate_failures(self, temp_dir):
        """Test logging multiple coordinate failures."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            streaming=False
        )

        for i in range(10):
            logger.log_coordinate_failure(
                coordinate=(35.0 + i * 0.1, -82.0 + i * 0.1),
                street_name=f"Road {i}",
                osm_way_id=str(100000 + i),
                datasets_tried=["ned10m"],
                primary_dataset="ned10m",
                level="ERROR"
            )

        assert len(logger.elevation_errors) == 10


# ============================================================================
# CSV Streaming Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.error_handling
class TestCSVStreaming:
    """Test CSV streaming functionality."""

    def test_start_elevation_logging(self, temp_dir):
        """Test starting elevation logging (CSV header creation)."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.csv",
            streaming=True
        )

        logger.start_elevation_logging()

        # Check that file was created
        assert logger.elevation_error_log.exists()
        assert logger.elevation_log_file is not None

        # Read and verify CSV header
        with open(logger.elevation_error_log, 'r') as f:
            content = f.read()
            assert "# Elevation Fetch Error Log" in content
            assert "level,timestamp,latitude,longitude" in content

        logger.stop_elevation_logging()

    def test_stop_elevation_logging(self, temp_dir):
        """Test stopping elevation logging and file closure."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.csv",
            streaming=True
        )

        logger.start_elevation_logging()
        logger.stop_elevation_logging()

        # File should be closed
        assert logger.elevation_log_file is None

        # File should exist with completion metadata
        with open(logger.elevation_error_log, 'r') as f:
            content = f.read()
            assert "# Analysis Completed:" in content

    def test_streaming_coordinate_failures(self, temp_dir):
        """Test streaming coordinate failures to CSV."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.csv",
            streaming=True,
            region_name="Test Region"
        )

        logger.start_elevation_logging()

        # Log several failures
        for i in range(5):
            logger.log_coordinate_failure(
                coordinate=(35.0 + i * 0.1, -82.0 - i * 0.1),
                street_name=f"Street {i}",
                osm_way_id=f"WAY{i}",
                datasets_tried=["ned10m", "ned30m"],
                primary_dataset="ned10m",
                level="ERROR"
            )

        logger.stop_elevation_logging()

        # Read CSV and verify entries
        with open(logger.elevation_error_log, 'r') as f:
            lines = f.readlines()

            # Filter out comment lines and header
            data_lines = [line for line in lines if not line.startswith('#') and 'level,timestamp' not in line]

            # Should have 5 data rows
            assert len(data_lines) == 5

            # Check that coordinates are properly formatted
            assert "35.000000" in data_lines[0]
            assert "35.400000" in data_lines[4]

    def test_csv_escaping(self, temp_dir):
        """Test that CSV special characters are properly escaped."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.csv",
            streaming=True
        )

        logger.start_elevation_logging()

        # Log failure with special characters in street name
        logger.log_coordinate_failure(
            coordinate=(35.0, -82.0),
            street_name='Street with "quotes" and, commas',
            osm_way_id="123",
            datasets_tried=["ned10m"],
            primary_dataset="ned10m",
            level="ERROR"
        )

        logger.stop_elevation_logging()

        # Read and verify CSV can be parsed
        with open(logger.elevation_error_log, 'r') as f:
            content = f.read()
            # Should have escaped the quotes and commas
            assert 'Street with ""quotes"" and, commas' in content or '"Street with' in content


# ============================================================================
# Log Rotation Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.error_handling
class TestLogRotation:
    """Test log file rotation functionality."""

    def test_log_rotation_under_limit(self, temp_dir):
        """Test that logs under max_runs are not rotated."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            max_runs=10
        )

        # Log 5 runs (under limit of 10)
        for i in range(5):
            logger.log_elevation_errors(
                report_filename=f"report_{i}.xlsx",
                total_coords_in_ways=1000,
                failed_coords=10,
                total_coords_in_country=100000,
                way_percentage=1.0,
                country_percentage=0.01
            )

        # Read entries
        entries = logger._read_log_entries()
        assert len(entries) == 5

    def test_log_rotation_over_limit(self, temp_dir):
        """Test that old logs are rotated out when exceeding max_runs."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            max_runs=10
        )

        # Log 15 runs (over limit of 10)
        for i in range(15):
            logger.log_elevation_errors(
                report_filename=f"report_{i}.xlsx",
                total_coords_in_ways=1000,
                failed_coords=10,
                total_coords_in_country=100000,
                way_percentage=1.0,
                country_percentage=0.01,
                timestamp=f"2024-01-{i+1:02d} 12:00:00"
            )

        # Read entries - should only have last 10
        entries = logger._read_log_entries()
        assert len(entries) == 10

        # Verify oldest entries were removed
        assert entries[0]["report_filename"] == "report_5.xlsx"
        assert entries[-1]["report_filename"] == "report_14.xlsx"

    def test_log_rotation_exactly_at_limit(self, temp_dir):
        """Test log rotation when exactly at max_runs limit."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            max_runs=5
        )

        # Log exactly 5 runs
        for i in range(5):
            logger.log_elevation_errors(
                report_filename=f"report_{i}.xlsx",
                total_coords_in_ways=1000,
                failed_coords=10,
                total_coords_in_country=100000,
                way_percentage=1.0,
                country_percentage=0.01
            )

        entries = logger._read_log_entries()
        assert len(entries) == 5


# ============================================================================
# Error Statistics Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.error_handling
class TestErrorStatistics:
    """Test error statistics tracking."""

    def test_elevation_error_logging(self, temp_dir):
        """Test logging elevation errors with statistics."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log"
        )

        logger.log_elevation_errors(
            report_filename="test_report.xlsx",
            total_coords_in_ways=10000,
            failed_coords=150,
            total_coords_in_country=5000000,
            way_percentage=1.5,
            country_percentage=0.003
        )

        # Verify log file was created
        assert logger.error_log_path.exists()

        # Read and verify content
        with open(logger.error_log_path, 'r') as f:
            content = f.read()
            assert "test_report.xlsx" in content
            assert "10,000" in content  # total_coords_in_ways formatted
            assert "150" in content  # failed_coords
            assert "1.50%" in content  # way_percentage
            assert "0.0030%" in content  # country_percentage

    def test_way_failures_tracking(self, temp_dir):
        """Test per-way failure tracking."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log"
        )

        way_failures = {
            "123456": {
                "name": "Mountain Road",
                "total_coords": 100,
                "failed_coords": 15,
                "failure_percentage": 15.0
            },
            "789012": {
                "name": "Valley Road",
                "total_coords": 200,
                "failed_coords": 5,
                "failure_percentage": 2.5
            }
        }

        logger.log_elevation_errors(
            report_filename="test_report.xlsx",
            total_coords_in_ways=300,
            failed_coords=20,
            total_coords_in_country=5000000,
            way_percentage=6.67,
            country_percentage=0.0004,
            way_failures=way_failures
        )

        # Read and verify way failures are logged
        with open(logger.error_log_path, 'r') as f:
            content = f.read()
            assert "Mountain Road" in content
            assert "Valley Road" in content
            assert "15/100" in content
            assert "5/200" in content


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.unit
@pytest.mark.error_handling
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_log_file(self, temp_dir):
        """Test reading from non-existent log file."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "nonexistent.log",
            elevation_error_log=temp_dir / "elevation_errors.log"
        )

        entries = logger._read_log_entries()
        assert entries == []

    def test_no_elevation_errors(self, temp_dir):
        """Test with no elevation errors to log."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            streaming=False
        )

        # Don't log any errors
        logger.write_elevation_errors_to_file()

        # File should not be created if no errors
        assert len(logger.elevation_errors) == 0

    def test_logging_with_none_values(self, temp_dir):
        """Test logging with None values for optional parameters."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.csv",
            streaming=True
        )

        logger.start_elevation_logging()

        logger.log_coordinate_failure(
            coordinate=(35.0, -82.0),
            street_name="Unknown Road",
            osm_way_id="N/A",
            datasets_tried=None,
            primary_dataset="ned10m",
            successful_dataset=None,
            level="ERROR"
        )

        logger.stop_elevation_logging()

        # Should handle None gracefully
        assert logger.elevation_error_log.exists()

    def test_zero_failures(self, temp_dir):
        """Test logging with zero failures."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log"
        )

        logger.log_elevation_errors(
            report_filename="perfect_run.xlsx",
            total_coords_in_ways=10000,
            failed_coords=0,
            total_coords_in_country=5000000,
            way_percentage=0.0,
            country_percentage=0.0
        )

        with open(logger.error_log_path, 'r') as f:
            content = f.read()
            assert "Failed Coordinates: 0" in content
            assert "0.00%" in content

    def test_large_number_of_failures(self, temp_dir):
        """Test handling large number of failures."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.log",
            streaming=False
        )

        # Log 2000 failures (should only write first 1000 to file)
        for i in range(2000):
            logger.log_coordinate_failure(
                coordinate=(35.0, -82.0),
                street_name=f"Road {i}",
                osm_way_id=f"{i}",
                datasets_tried=["ned10m"],
                primary_dataset="ned10m",
                level="ERROR"
            )

        assert len(logger.elevation_errors) == 2000

        logger.write_elevation_errors_to_file()

        # Should indicate truncation
        with open(logger.elevation_error_log, 'r') as f:
            content = f.read()
            assert "and 1,000 more failures" in content


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.error_handling
class TestErrorLoggerIntegration:
    """Integration tests for complete error logging workflows."""

    def test_complete_logging_workflow(self, temp_dir):
        """Test complete workflow from start to finish."""
        logger = ErrorLogger(
            error_log_path=temp_dir / "error.log",
            elevation_error_log=temp_dir / "elevation_errors.csv",
            streaming=True,
            region_name="Integration Test"
        )

        # Start logging
        logger.start_elevation_logging()

        # Log various types of failures
        logger.log_coordinate_failure(
            coordinate=(35.1, -82.1),
            street_name="Failed Road 1",
            osm_way_id="111",
            datasets_tried=["ned10m", "ned30m"],
            primary_dataset="ned10m",
            level="ERROR"
        )

        logger.log_coordinate_failure(
            coordinate=(35.2, -82.2),
            street_name="Fallback Road",
            osm_way_id="222",
            datasets_tried=["ned10m", "ned30m"],
            primary_dataset="ned10m",
            successful_dataset="ned30m",
            level="INFO"
        )

        # Stop logging
        logger.stop_elevation_logging()

        # Log summary statistics
        logger.log_elevation_errors(
            report_filename="integration_test.xlsx",
            total_coords_in_ways=1000,
            failed_coords=2,
            total_coords_in_country=1000000,
            way_percentage=0.2,
            country_percentage=0.0002
        )

        # Verify both files exist
        assert logger.elevation_error_log.exists()
        assert logger.error_log_path.exists()

        # Verify CSV has correct entries
        with open(logger.elevation_error_log, 'r') as f:
            content = f.read()
            assert "Failed Road 1" in content
            assert "Fallback Road" in content
            assert "ERROR" in content
            assert "INFO" in content

        # Verify error log has summary
        with open(logger.error_log_path, 'r') as f:
            content = f.read()
            assert "integration_test.xlsx" in content
