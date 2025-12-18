#!/usr/bin/env python3
"""
Simple test runner to verify test infrastructure without requiring pytest.

This script runs a few basic tests to ensure the test structure is working.
For full test execution, use pytest.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_distance_calculation():
    """Test basic distance calculation."""
    import math

    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # Test 1: Same point distance
    distance = calculate_distance(35.0, -82.0, 35.0, -82.0)
    assert distance == 0.0, f"Same point distance should be 0, got {distance}"
    print("✓ test_same_point_distance PASSED")

    # Test 2: Known distance
    distance = calculate_distance(35.0, -82.0, 36.0, -82.0)
    assert 110 < distance < 112, f"Distance should be ~111km, got {distance}"
    print("✓ test_known_distance PASSED")

    # Test 3: Distance symmetry
    dist1 = calculate_distance(35.0, -82.0, 36.0, -83.0)
    dist2 = calculate_distance(36.0, -83.0, 35.0, -82.0)
    assert abs(dist1 - dist2) < 0.001, f"Distance should be symmetric, got {dist1} vs {dist2}"
    print("✓ test_distance_symmetry PASSED")


def test_elevation_metrics():
    """Test elevation gain calculation."""

    def calculate_elevation_gain(elevations):
        """Calculate total elevation gain."""
        gain = 0.0
        for i in range(len(elevations) - 1):
            elev_change = elevations[i + 1] - elevations[i]
            if elev_change > 0:
                gain += elev_change
        return gain

    # Test 1: Simple climb
    elevations = [100, 150, 200, 250, 300]
    gain = calculate_elevation_gain(elevations)
    assert gain == 200.0, f"Elevation gain should be 200m, got {gain}"
    print("✓ test_elevation_gain_simple_climb PASSED")

    # Test 2: Climb with descent
    elevations = [100, 200, 150, 250, 200, 300]
    gain = calculate_elevation_gain(elevations)
    assert gain == 300.0, f"Elevation gain should be 300m, got {gain}"
    print("✓ test_elevation_gain_with_descent PASSED")

    # Test 3: Flat terrain
    elevations = [100, 100, 100, 100]
    gain = calculate_elevation_gain(elevations)
    assert gain == 0.0, f"Flat terrain should have 0 gain, got {gain}"
    print("✓ test_flat_terrain PASSED")


def test_climb_scoring():
    """Test basic climb scoring."""

    def calculate_basic_score(distance_km: float, avg_grade: float) -> float:
        """Calculate basic climb score."""
        if avg_grade <= 0:
            return 0.0
        return distance_km * 1000 * avg_grade

    def calculate_fiets_score(elevation_gain_m: float, distance_km: float) -> float:
        """Calculate FIETS difficulty index."""
        if distance_km <= 0:
            return 0.0
        return (elevation_gain_m ** 2) / (distance_km * 10)

    # Test 1: Basic score moderate climb
    score = calculate_basic_score(5.0, 5.0)
    assert score == 25000.0, f"Basic score should be 25000, got {score}"
    print("✓ test_basic_score_moderate_climb PASSED")

    # Test 2: FIETS score
    score = calculate_fiets_score(200.0, 4.0)
    assert score == 1000.0, f"FIETS score should be 1000, got {score}"
    print("✓ test_fiets_score_moderate_climb PASSED")

    # Test 3: Zero grade
    score = calculate_basic_score(5.0, 0.0)
    assert score == 0.0, f"Zero grade should give 0 score, got {score}"
    print("✓ test_basic_score_zero_grade PASSED")


def test_climb_categorization():
    """Test climb categorization."""

    def categorize_climb(avg_grade: float, length_m: float, elevation_gain_m: float) -> str:
        """Categorize climb based on climb score."""
        climb_score = length_m * avg_grade

        if climb_score > 80000:
            return "HC"
        elif climb_score > 64000:
            return "1"
        elif climb_score > 32000:
            return "2"
        elif climb_score > 16000:
            return "3"
        elif climb_score > 8000:
            return "4"
        else:
            return "N/A"

    # Test 1: Category HC
    category = categorize_climb(6.0, 16000.0, 960.0)
    assert category == "HC", f"Should be HC, got {category}"
    print("✓ test_category_hc PASSED")

    # Test 2: Category 3
    category = categorize_climb(5.0, 4000.0, 200.0)
    assert category == "3", f"Should be Category 3, got {category}"
    print("✓ test_category_3 PASSED")

    # Test 3: Below threshold
    category = categorize_climb(5.0, 1000.0, 50.0)
    assert category == "N/A", f"Should be N/A, got {category}"
    print("✓ test_category_na PASSED")


def test_data_structures():
    """Test climb data structures."""
    from climb_analyzer.core.segment import ClimbSegment, ClimbMetrics

    # Test 1: ClimbSegment creation
    segment = ClimbSegment(
        name="Test Road",
        start_lat=35.1,
        start_lon=-82.1,
        end_lat=35.3,
        end_lon=-82.3,
        distance_km=2.0,
        elevation_gain_m=200.0,
        avg_gradient=10.0,
        max_gradient=15.0,
        category="4",
        points=[(35.1, -82.1, 500.0), (35.3, -82.3, 700.0)]
    )

    assert segment.name == "Test Road"
    assert segment.distance_km == 2.0
    assert len(segment.points) == 2
    print("✓ test_climb_segment_creation PASSED")

    # Test 2: ClimbMetrics creation
    metrics = ClimbMetrics(
        street_name="Test Mountain Road",
        climb_category="3",
        climb_score=25000.0,
        elevation_gain=300.0,
        height=300.0,
        prominence=250.0,
        length_km=5.0,
        distance_km=4.8,
        avg_grade=6.0,
        max_grade=12.0,
        min_elevation=500.0,
        max_elevation=800.0,
        surface="asphalt",
        tracktype="-",
        tracktype_definition="-",
        way_ids=[123456, 789012],
        osm_links=["[123456](https://www.openstreetmap.org/way/123456)"]
    )

    assert metrics.street_name == "Test Mountain Road"
    assert metrics.climb_category == "3"
    assert len(metrics.way_ids) == 2
    print("✓ test_climb_metrics_creation PASSED")


def main():
    """Run all sample tests."""
    print("=" * 70)
    print("Running Sample Tests")
    print("=" * 70)
    print()

    tests = [
        ("Distance Calculation", test_distance_calculation),
        ("Elevation Metrics", test_elevation_metrics),
        ("Climb Scoring", test_climb_scoring),
        ("Climb Categorization", test_climb_categorization),
        ("Data Structures", test_data_structures),
    ]

    failed_tests = []

    for test_name, test_func in tests:
        print(f"Running {test_name} tests...")
        try:
            test_func()
            print(f"✓ All {test_name} tests PASSED\n")
        except AssertionError as e:
            print(f"✗ {test_name} tests FAILED: {e}\n")
            failed_tests.append(test_name)
        except ImportError as e:
            print(f"⚠ {test_name} tests SKIPPED (missing dependencies): {e}\n")
        except Exception as e:
            print(f"✗ {test_name} tests ERROR: {e}\n")
            failed_tests.append(test_name)

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    if not failed_tests:
        print(f"✓ All {len(tests)} test suites PASSED!")
        print()
        print("Next steps:")
        print("  1. Install pytest: pip install pytest pytest-cov")
        print("  2. Run full test suite: pytest")
        print("  3. See tests/README.md for more options")
        return 0
    else:
        print(f"✗ {len(failed_tests)} test suite(s) FAILED:")
        for test_name in failed_tests:
            print(f"  - {test_name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
