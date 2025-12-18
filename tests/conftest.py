"""
Pytest configuration and shared fixtures for climb_analyzer tests.

This file provides centralized test configuration and reusable fixtures
for all test modules in the project.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_coordinates():
    """Provide sample GPS coordinates for testing."""
    return [
        {"lat": 35.1234, "lon": -82.5678},
        {"lat": 35.2345, "lon": -82.6789},
        {"lat": 35.3456, "lon": -82.7890},
    ]


@pytest.fixture
def sample_elevation_data():
    """Provide sample elevation data for testing."""
    return [
        {"lat": 35.1234, "lon": -82.5678, "elevation": 500.0},
        {"lat": 35.2345, "lon": -82.6789, "elevation": 750.0},
        {"lat": 35.3456, "lon": -82.7890, "elevation": 1000.0},
        {"lat": 35.4567, "lon": -82.8901, "elevation": 850.0},
        {"lat": 35.5678, "lon": -82.9012, "elevation": 600.0},
    ]


@pytest.fixture
def sample_climb_profile():
    """Provide a sample climb elevation profile."""
    # Simulates a climb with 250m gain over 5km (5% average grade)
    return {
        "distances": [0, 1000, 2000, 3000, 4000, 5000],  # meters
        "elevations": [500, 550, 650, 800, 900, 750],    # meters
        "lats": [35.1, 35.11, 35.12, 35.13, 35.14, 35.15],
        "lons": [-82.5, -82.51, -82.52, -82.53, -82.54, -82.55],
    }


@pytest.fixture
def sample_steep_climb_profile():
    """Provide a sample steep climb elevation profile."""
    # Simulates a steep climb with 300m gain over 3km (10% average grade)
    return {
        "distances": [0, 500, 1000, 1500, 2000, 2500, 3000],
        "elevations": [400, 450, 550, 700, 850, 950, 700],
        "lats": [35.2, 35.21, 35.22, 35.23, 35.24, 35.25, 35.26],
        "lons": [-82.6, -82.61, -82.62, -82.63, -82.64, -82.65, -82.66],
    }


@pytest.fixture
def sample_way_data():
    """Provide sample OSM way data for testing."""
    return {
        "way_id": 123456789,
        "name": "Test Mountain Road",
        "highway_type": "primary",
        "nodes": [
            {"id": 1, "lat": 35.1234, "lon": -82.5678},
            {"id": 2, "lat": 35.2345, "lon": -82.6789},
            {"id": 3, "lat": 35.3456, "lon": -82.7890},
        ],
        "tags": {
            "name": "Test Mountain Road",
            "highway": "primary",
            "surface": "asphalt",
        }
    }


@pytest.fixture
def sample_climb_segment():
    """Provide a sample ClimbSegment for testing."""
    from climb_analyzer.core.segment import ClimbSegment, ClimbMetrics

    metrics = ClimbMetrics(
        climb_score=45.5,
        fiets_score=234.5,
        pdi_score=15.2,
        elevation_gain=250.0,
        distance=5000.0,
        avg_grade=5.0,
        max_grade=12.0,
        prominence=200.0,
        way_id=123456789,
        way_name="Test Mountain Road",
        start_lat=35.1234,
        start_lon=-82.5678,
        end_lat=35.5678,
        end_lon=-82.9012,
    )

    return ClimbSegment(
        way_id=123456789,
        way_name="Test Mountain Road",
        metrics=metrics,
        elevation_profile=[500, 550, 650, 800, 900, 750],
        distance_profile=[0, 1000, 2000, 3000, 4000, 5000],
    )


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_osm_file(temp_dir):
    """Create a temporary OSM file for testing."""
    osm_path = temp_dir / "test.osm.pbf"
    osm_path.touch()
    return osm_path


@pytest.fixture
def temp_elevation_file(temp_dir):
    """Create a temporary elevation data file for testing."""
    elevation_path = temp_dir / "test_elevation.tif"
    elevation_path.touch()
    return elevation_path


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create a temporary output directory for test results."""
    output_path = temp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# ============================================================================
# Mock and Stub Fixtures
# ============================================================================

@pytest.fixture
def mock_elevation_fetcher(monkeypatch):
    """Mock elevation data fetcher that returns predictable data."""
    class MockElevationFetcher:
        def __init__(self):
            self.call_count = 0
            self.called_with = []

        def fetch_elevation(self, lat, lon):
            self.call_count += 1
            self.called_with.append((lat, lon))
            # Return elevation based on latitude (simple test pattern)
            return {"elevation": 500 + (lat - 35.0) * 1000}

        def fetch_elevations_batch(self, coords):
            results = []
            for coord in coords:
                results.append(self.fetch_elevation(coord["lat"], coord["lon"]))
            return results

    return MockElevationFetcher()


@pytest.fixture
def mock_error_logger(temp_dir):
    """Mock error logger for testing error handling."""
    from utils.error_logger import ErrorLogger

    log_dir = temp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = ErrorLogger(log_dir=str(log_dir))
    return logger


@pytest.fixture
def mock_elevation_stats_collector():
    """Mock elevation stats collector for testing."""
    from utils.elevation_stats_collector import ElevationStatsCollector

    # Reset singleton instance
    ElevationStatsCollector._instance = None
    collector = ElevationStatsCollector()

    yield collector

    # Cleanup singleton
    ElevationStatsCollector._instance = None


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Provide a test configuration."""
    return {
        "min_climb_gain": 30,  # meters
        "min_climb_distance": 500,  # meters
        "min_avg_grade": 3.0,  # percent
        "max_gap_distance": 200,  # meters
        "max_grade": 30.0,  # percent
        "smoothing_window": 5,
    }


@pytest.fixture
def test_checkpoint_config(temp_dir):
    """Provide a checkpoint configuration for testing."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_interval": 100,  # Save every 100 items
        "enable_checkpointing": True,
    }


# ============================================================================
# Parametrized Test Data
# ============================================================================

@pytest.fixture(params=[
    {"gain": 50, "distance": 1000, "expected_grade": 5.0},
    {"gain": 100, "distance": 2000, "expected_grade": 5.0},
    {"gain": 300, "distance": 3000, "expected_grade": 10.0},
])
def climb_grade_test_data(request):
    """Parametrized test data for grade calculations."""
    return request.param


@pytest.fixture(params=[
    "primary", "secondary", "tertiary", "unclassified", "residential"
])
def highway_types(request):
    """Parametrized highway types for testing."""
    return request.param


# ============================================================================
# Marker Utilities
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring external data"
    )


# ============================================================================
# Test Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add markers based on test location
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
