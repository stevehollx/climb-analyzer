"""
Unit tests for climb data structures.

Tests the ClimbSegment and ClimbMetrics dataclasses to ensure they
properly store and validate climb data.
"""

import pytest
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from climb_analyzer.core.segment import ClimbSegment, ClimbMetrics


# ============================================================================
# ClimbSegment Tests
# ============================================================================

@pytest.mark.unit
class TestClimbSegment:
    """Test ClimbSegment dataclass."""

    def test_climb_segment_creation(self):
        """Test basic ClimbSegment creation."""
        segment = ClimbSegment(
            name="Mount Mitchell Road",
            start_lat=35.7645,
            start_lon=-82.2652,
            end_lat=35.7651,
            end_lon=-82.2688,
            distance_km=5.5,
            elevation_gain_m=300.0,
            avg_gradient=5.45,
            max_gradient=12.0,
            category="3",
            points=[(35.7645, -82.2652, 500.0), (35.7651, -82.2688, 800.0)]
        )

        assert segment.name == "Mount Mitchell Road"
        assert segment.distance_km == 5.5
        assert segment.elevation_gain_m == 300.0
        assert segment.category == "3"
        assert len(segment.points) == 2

    def test_climb_segment_point_structure(self):
        """Test that points are properly structured as (lat, lon, elevation)."""
        points = [
            (35.1, -82.1, 500.0),
            (35.2, -82.2, 600.0),
            (35.3, -82.3, 700.0),
        ]

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
            points=points
        )

        # Verify points structure
        for i, point in enumerate(segment.points):
            assert len(point) == 3  # (lat, lon, elev)
            assert isinstance(point[0], float)  # lat
            assert isinstance(point[1], float)  # lon
            assert isinstance(point[2], float)  # elevation

    def test_climb_segment_coordinates(self):
        """Test that start/end coordinates are accessible."""
        segment = ClimbSegment(
            name="Test",
            start_lat=35.1234,
            start_lon=-82.5678,
            end_lat=35.9876,
            end_lon=-82.1234,
            distance_km=10.0,
            elevation_gain_m=500.0,
            avg_gradient=5.0,
            max_gradient=10.0,
            category="2",
            points=[]
        )

        assert segment.start_lat == 35.1234
        assert segment.start_lon == -82.5678
        assert segment.end_lat == 35.9876
        assert segment.end_lon == -82.1234


# ============================================================================
# ClimbMetrics Tests
# ============================================================================

@pytest.mark.unit
class TestClimbMetrics:
    """Test ClimbMetrics dataclass."""

    def test_climb_metrics_creation_minimal(self):
        """Test ClimbMetrics with minimal required fields."""
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
        assert metrics.climb_score == 25000.0
        assert metrics.elevation_gain == 300.0
        assert len(metrics.way_ids) == 2

    def test_climb_metrics_with_optional_fields(self):
        """Test ClimbMetrics with optional fields."""
        metrics = ClimbMetrics(
            street_name="Scenic Byway",
            climb_category="2",
            climb_score=35000.0,
            elevation_gain=400.0,
            height=400.0,
            prominence=350.0,
            length_km=7.0,
            distance_km=6.5,
            avg_grade=5.7,
            max_grade=10.0,
            min_elevation=600.0,
            max_elevation=1000.0,
            surface="paved",
            tracktype="-",
            tracktype_definition="-",
            way_ids=[111111],
            osm_links=["[111111](https://www.openstreetmap.org/way/111111)"],
            city_state="Asheville, NC",
            distance_from_center_km=12.5,
            mid_lat=35.5,
            mid_lon=-82.5,
            start_lat=35.4,
            start_lon=-82.6,
            fiets_score=1500.0,
            pdi_score=250.0,
            cycling_access="Yes",
            highway_type="primary"
        )

        assert metrics.city_state == "Asheville, NC"
        assert metrics.distance_from_center_km == 12.5
        assert metrics.fiets_score == 1500.0
        assert metrics.pdi_score == 250.0
        assert metrics.cycling_access == "Yes"
        assert metrics.highway_type == "primary"

    def test_climb_metrics_default_values(self):
        """Test that optional fields have correct defaults."""
        metrics = ClimbMetrics(
            street_name="Test",
            climb_category="N/A",
            climb_score=0.0,
            elevation_gain=100.0,
            height=100.0,
            prominence=80.0,
            length_km=2.0,
            distance_km=2.0,
            avg_grade=5.0,
            max_grade=8.0,
            min_elevation=400.0,
            max_elevation=500.0,
            surface="unknown",
            tracktype="-",
            tracktype_definition="-",
            way_ids=[],
            osm_links=[]
        )

        # Check defaults
        assert metrics.city_state == "Unknown"
        assert metrics.distance_from_center_km == 0.0
        assert metrics.mid_lat == 0.0
        assert metrics.mid_lon == 0.0
        assert metrics.start_lat == 0.0
        assert metrics.start_lon == 0.0
        assert metrics.fiets_score == 0.0
        assert metrics.pdi_score == 0.0
        assert metrics.cycling_access == "Unknown"
        assert metrics.highway_type == "unknown"
        assert metrics.nodes is None
        assert metrics.connected_climbs is None

    def test_climb_metrics_multiple_way_ids(self):
        """Test ClimbMetrics with multiple OSM way IDs (merged ways)."""
        way_ids = [123456, 789012, 345678, 901234]
        osm_links = [
            f"[{way_id}](https://www.openstreetmap.org/way/{way_id})"
            for way_id in way_ids
        ]

        metrics = ClimbMetrics(
            street_name="Long Mountain Road",
            climb_category="1",
            climb_score=65000.0,
            elevation_gain=800.0,
            height=800.0,
            prominence=700.0,
            length_km=13.0,
            distance_km=12.5,
            avg_grade=6.2,
            max_grade=15.0,
            min_elevation=500.0,
            max_elevation=1300.0,
            surface="asphalt",
            tracktype="-",
            tracktype_definition="-",
            way_ids=way_ids,
            osm_links=osm_links
        )

        assert len(metrics.way_ids) == 4
        assert len(metrics.osm_links) == 4
        assert all(isinstance(way_id, int) for way_id in metrics.way_ids)

    def test_climb_metrics_surface_types(self):
        """Test various surface types."""
        surfaces = ["asphalt", "gravel", "dirt", "paved", "concrete", "unknown"]

        for surface in surfaces:
            metrics = ClimbMetrics(
                street_name=f"Test Road ({surface})",
                climb_category="4",
                climb_score=10000.0,
                elevation_gain=100.0,
                height=100.0,
                prominence=80.0,
                length_km=2.0,
                distance_km=2.0,
                avg_grade=5.0,
                max_grade=8.0,
                min_elevation=400.0,
                max_elevation=500.0,
                surface=surface,
                tracktype="-",
                tracktype_definition="-",
                way_ids=[],
                osm_links=[]
            )

            assert metrics.surface == surface

    def test_climb_metrics_all_categories(self):
        """Test all climb categories."""
        categories = ["HC", "1", "2", "3", "4", "N/A"]

        for category in categories:
            metrics = ClimbMetrics(
                street_name=f"Cat {category} Climb",
                climb_category=category,
                climb_score=10000.0,
                elevation_gain=100.0,
                height=100.0,
                prominence=80.0,
                length_km=2.0,
                distance_km=2.0,
                avg_grade=5.0,
                max_grade=8.0,
                min_elevation=400.0,
                max_elevation=500.0,
                surface="asphalt",
                tracktype="-",
                tracktype_definition="-",
                way_ids=[],
                osm_links=[]
            )

            assert metrics.climb_category == category


# ============================================================================
# Data Validation Tests
# ============================================================================

@pytest.mark.unit
class TestDataValidation:
    """Test data validation for climb structures."""

    def test_elevation_gain_positive(self):
        """Test that elevation gain is positive."""
        metrics = ClimbMetrics(
            street_name="Test",
            climb_category="3",
            climb_score=20000.0,
            elevation_gain=250.0,
            height=250.0,
            prominence=200.0,
            length_km=4.0,
            distance_km=3.8,
            avg_grade=6.25,
            max_grade=12.0,
            min_elevation=500.0,
            max_elevation=750.0,
            surface="asphalt",
            tracktype="-",
            tracktype_definition="-",
            way_ids=[],
            osm_links=[]
        )

        assert metrics.elevation_gain > 0

    def test_height_calculation_consistency(self):
        """Test that height equals max - min elevation."""
        min_elev = 500.0
        max_elev = 800.0
        expected_height = max_elev - min_elev

        metrics = ClimbMetrics(
            street_name="Test",
            climb_category="3",
            climb_score=20000.0,
            elevation_gain=300.0,
            height=expected_height,
            prominence=250.0,
            length_km=5.0,
            distance_km=4.8,
            avg_grade=6.0,
            max_grade=12.0,
            min_elevation=min_elev,
            max_elevation=max_elev,
            surface="asphalt",
            tracktype="-",
            tracktype_definition="-",
            way_ids=[],
            osm_links=[]
        )

        # Height should match max - min
        assert metrics.height == expected_height
        assert metrics.max_elevation - metrics.min_elevation == metrics.height

    def test_avg_grade_vs_max_grade(self):
        """Test that max grade is >= average grade."""
        metrics = ClimbMetrics(
            street_name="Test",
            climb_category="3",
            climb_score=20000.0,
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
            way_ids=[],
            osm_links=[]
        )

        # Max grade should be >= avg grade
        assert metrics.max_grade >= metrics.avg_grade

    def test_distance_vs_length(self):
        """Test relationship between distance and length."""
        # Distance (straight line) should be <= length (actual road distance)
        metrics = ClimbMetrics(
            street_name="Winding Road",
            climb_category="3",
            climb_score=25000.0,
            elevation_gain=300.0,
            height=300.0,
            prominence=250.0,
            length_km=6.0,      # Actual road distance
            distance_km=5.0,    # Straight-line distance
            avg_grade=5.0,
            max_grade=10.0,
            min_elevation=500.0,
            max_elevation=800.0,
            surface="asphalt",
            tracktype="-",
            tracktype_definition="-",
            way_ids=[],
            osm_links=[]
        )

        # In most cases, straight-line distance <= road distance
        # (though they can be equal for very straight roads)
        assert metrics.distance_km <= metrics.length_km


# ============================================================================
# OSM Link Tests
# ============================================================================

@pytest.mark.unit
class TestOSMLinks:
    """Test OSM link generation and formatting."""

    def test_osm_link_format(self):
        """Test that OSM links are properly formatted."""
        way_id = 123456789
        expected_link = f"[{way_id}](https://www.openstreetmap.org/way/{way_id})"

        metrics = ClimbMetrics(
            street_name="Test",
            climb_category="3",
            climb_score=20000.0,
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
            way_ids=[way_id],
            osm_links=[expected_link]
        )

        assert metrics.osm_links[0] == expected_link
        assert str(way_id) in metrics.osm_links[0]
        assert "openstreetmap.org/way/" in metrics.osm_links[0]

    def test_multiple_osm_links(self):
        """Test multiple OSM links for merged ways."""
        way_ids = [111111, 222222, 333333]
        osm_links = [
            f"[{way_id}](https://www.openstreetmap.org/way/{way_id})"
            for way_id in way_ids
        ]

        metrics = ClimbMetrics(
            street_name="Merged Road",
            climb_category="2",
            climb_score=35000.0,
            elevation_gain=400.0,
            height=400.0,
            prominence=350.0,
            length_km=7.0,
            distance_km=6.5,
            avg_grade=5.7,
            max_grade=11.0,
            min_elevation=600.0,
            max_elevation=1000.0,
            surface="asphalt",
            tracktype="-",
            tracktype_definition="-",
            way_ids=way_ids,
            osm_links=osm_links
        )

        assert len(metrics.osm_links) == 3
        assert all("openstreetmap.org/way/" in link for link in metrics.osm_links)
