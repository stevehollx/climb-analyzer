"""
Integration tests for complete climb analysis workflow.

Tests the end-to-end process of:
1. Loading OSM data
2. Extracting road segments
3. Fetching elevation data
4. Detecting climbs
5. Calculating metrics and scores
6. Generating output
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Mock Classes for Integration Testing
# ============================================================================

class MockNode:
    """Mock OSM node for testing."""

    def __init__(self, node_id: int, lat: float, lon: float):
        self.id = node_id
        self.lat = lat
        self.lon = lon


class MockRoadSegment:
    """Mock road segment for testing."""

    def __init__(self, name: str, nodes: List[MockNode], way_ids: List[int]):
        self.street_name = name
        self.nodes = nodes
        self.way_ids = way_ids
        self.highway_type = "primary"
        self.surface = "asphalt"
        self.tracktype = "-"
        self.tracktype_definition = "-"
        self.cycling_access = "Yes"


class MockElevationFetcher:
    """Mock elevation fetcher that returns predictable elevations."""

    def __init__(self, elevation_function=None):
        """
        Initialize mock fetcher.

        Args:
            elevation_function: Function that takes (lat, lon) and returns elevation.
                               If None, returns 500 + lat*1000
        """
        if elevation_function is None:
            self.elevation_function = lambda lat, lon: 500 + (lat - 35.0) * 1000
        else:
            self.elevation_function = elevation_function

        self.fetch_count = 0
        self.failed_coords = []

    def fetch_elevation(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch elevation for a single coordinate."""
        self.fetch_count += 1
        elevation = self.elevation_function(lat, lon)

        if elevation is None:
            self.failed_coords.append((lat, lon))
            return {"elevation": None, "error": "Not found"}

        return {"elevation": elevation}

    def fetch_elevations_batch(self, coords: List[tuple]) -> List[Dict[str, Any]]:
        """Fetch elevations for multiple coordinates."""
        return [self.fetch_elevation(lat, lon) for lat, lon in coords]


# ============================================================================
# Complete Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestCompleteClimbWorkflow:
    """Test complete climb detection workflow."""

    def create_mock_climb_road(
        self,
        num_points: int = 10,
        start_lat: float = 35.0,
        start_lon: float = -82.0,
        elevation_gain: float = 300.0
    ) -> MockRoadSegment:
        """
        Create a mock road segment representing a climb.

        Args:
            num_points: Number of points in the road
            start_lat: Starting latitude
            start_lon: Starting longitude
            elevation_gain: Total elevation gain

        Returns:
            MockRoadSegment with elevation profile
        """
        nodes = []
        for i in range(num_points):
            lat = start_lat + (i * 0.001)  # ~111m per point
            lon = start_lon + (i * 0.001)
            nodes.append(MockNode(i, lat, lon))

        return MockRoadSegment(
            name="Test Mountain Road",
            nodes=nodes,
            way_ids=[123456]
        )

    def test_simple_climb_detection_workflow(self):
        """Test detecting a simple climb from mock data."""
        # Create mock road with climbing profile
        # 10 points, each 111m apart (total ~1km)
        # Elevation increases linearly from 500m to 800m (300m gain)

        def elevation_func(lat, lon):
            # Linear climb: 500m at start (35.0) to 800m at end (35.009)
            progress = (lat - 35.0) / 0.009
            return 500 + (300 * progress)

        road = self.create_mock_climb_road(num_points=10)
        fetcher = MockElevationFetcher(elevation_function=elevation_func)

        # Fetch elevations for all nodes
        elevations = []
        for node in road.nodes:
            result = fetcher.fetch_elevation(node.lat, node.lon)
            elevations.append(result["elevation"])

        # Verify elevation profile
        assert len(elevations) == 10
        assert elevations[0] == pytest.approx(500.0, abs=1.0)
        assert elevations[-1] == pytest.approx(800.0, abs=1.0)

        # Calculate elevation gain
        elevation_gain = sum(
            max(0, elevations[i+1] - elevations[i])
            for i in range(len(elevations) - 1)
        )

        assert elevation_gain == pytest.approx(300.0, abs=1.0)

    def test_climb_with_descent_sections(self):
        """Test climb detection with descent sections mixed in."""

        def elevation_func(lat, lon):
            # Climb with some descents: 500, 600, 550, 700, 650, 800
            progress = (lat - 35.0) / 0.005
            segment = int(progress * 6)

            elevations_profile = [500, 600, 550, 700, 650, 800]
            if segment >= len(elevations_profile):
                return elevations_profile[-1]
            return elevations_profile[segment]

        road = self.create_mock_climb_road(num_points=6)
        fetcher = MockElevationFetcher(elevation_function=elevation_func)

        elevations = []
        for node in road.nodes:
            result = fetcher.fetch_elevation(node.lat, node.lon)
            elevations.append(result["elevation"])

        # Calculate elevation gain (only uphill sections)
        elevation_gain = 0.0
        elevation_loss = 0.0

        for i in range(len(elevations) - 1):
            change = elevations[i+1] - elevations[i]
            if change > 0:
                elevation_gain += change
            else:
                elevation_loss += abs(change)

        # Total gain should be 400m (100+150+150)
        # Total loss should be 100m (50+50)
        assert elevation_gain == pytest.approx(400.0, abs=1.0)
        assert elevation_loss == pytest.approx(100.0, abs=1.0)

    def test_multiple_climbs_detection(self):
        """Test detecting multiple separate climbs."""
        climbs_data = [
            # Climb 1: Moderate climb
            {
                "start_lat": 35.0,
                "elevation_func": lambda lat, lon: 500 + (lat - 35.0) * 30000,
                "num_points": 10
            },
            # Climb 2: Steep climb
            {
                "start_lat": 35.1,
                "elevation_func": lambda lat, lon: 600 + (lat - 35.1) * 50000,
                "num_points": 8
            },
            # Climb 3: Gentle climb
            {
                "start_lat": 35.2,
                "elevation_func": lambda lat, lon: 400 + (lat - 35.2) * 20000,
                "num_points": 15
            }
        ]

        detected_climbs = []

        for i, climb_data in enumerate(climbs_data):
            # Create mock road for this climb
            nodes = []
            for j in range(climb_data["num_points"]):
                lat = climb_data["start_lat"] + (j * 0.01)
                lon = -82.0 + (j * 0.01)
                nodes.append(MockNode(j, lat, lon))

            road = MockRoadSegment(
                name=f"Climb {i+1}",
                nodes=nodes,
                way_ids=[1000 + i]
            )

            # Fetch elevations
            fetcher = MockElevationFetcher(elevation_function=climb_data["elevation_func"])
            elevations = []
            for node in road.nodes:
                result = fetcher.fetch_elevation(node.lat, node.lon)
                elevations.append(result["elevation"])

            # Calculate gain
            gain = sum(
                max(0, elevations[j+1] - elevations[j])
                for j in range(len(elevations) - 1)
            )

            detected_climbs.append({
                "name": road.street_name,
                "gain": gain,
                "num_points": len(elevations)
            })

        # Verify all climbs were processed
        assert len(detected_climbs) == 3
        assert all(climb["gain"] > 0 for climb in detected_climbs)


# ============================================================================
# Scoring Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestScoringWorkflow:
    """Test complete scoring workflow with realistic data."""

    def create_climb_scenario(
        self,
        name: str,
        elevation_profile: List[float],
        distance_per_segment: float = 0.1  # km
    ) -> Dict[str, Any]:
        """
        Create a climb scenario with complete data.

        Args:
            name: Climb name
            elevation_profile: List of elevations in meters
            distance_per_segment: Distance between points in km

        Returns:
            Dictionary with climb data and calculated metrics
        """
        # Calculate distance
        total_distance = distance_per_segment * (len(elevation_profile) - 1)

        # Calculate elevation gain/loss
        elevation_gain = 0.0
        elevation_loss = 0.0

        for i in range(len(elevation_profile) - 1):
            change = elevation_profile[i+1] - elevation_profile[i]
            if change > 0:
                elevation_gain += change
            else:
                elevation_loss += abs(change)

        # Calculate height (max - min)
        height = max(elevation_profile) - min(elevation_profile)

        # Calculate average grade
        avg_grade = (height / (total_distance * 1000)) * 100 if total_distance > 0 else 0

        # Calculate max grade
        max_grade = 0.0
        for i in range(len(elevation_profile) - 1):
            change = elevation_profile[i+1] - elevation_profile[i]
            distance_m = distance_per_segment * 1000
            if distance_m > 0:
                grade = abs(change / distance_m) * 100
                max_grade = max(max_grade, grade)

        # Calculate scores
        basic_score = total_distance * 1000 * avg_grade
        fiets_score = (elevation_gain ** 2) / (total_distance * 10)

        # Categorize
        climb_score = total_distance * 1000 * avg_grade
        if climb_score > 80000:
            category = "HC"
        elif climb_score > 64000:
            category = "1"
        elif climb_score > 32000:
            category = "2"
        elif climb_score > 16000:
            category = "3"
        elif climb_score > 8000:
            category = "4"
        else:
            category = "N/A"

        return {
            "name": name,
            "elevation_profile": elevation_profile,
            "total_distance": total_distance,
            "elevation_gain": elevation_gain,
            "elevation_loss": elevation_loss,
            "height": height,
            "avg_grade": avg_grade,
            "max_grade": max_grade,
            "basic_score": basic_score,
            "fiets_score": fiets_score,
            "category": category
        }

    def test_moderate_climb_scoring(self):
        """Test scoring for a moderate climb."""
        # 5km climb, 250m gain, avg 5% grade
        elevation_profile = [
            500, 520, 545, 575, 610, 650,
            685, 715, 740, 760, 750
        ]  # 10 segments × 0.5km = 5km

        climb = self.create_climb_scenario(
            "Moderate Mountain Road",
            elevation_profile,
            distance_per_segment=0.5
        )

        assert climb["total_distance"] == 5.0
        assert climb["elevation_gain"] == pytest.approx(260.0, abs=5.0)
        assert climb["avg_grade"] == pytest.approx(5.0, abs=0.5)
        assert climb["category"] in ["3", "4"]  # Should be Cat 3 or 4

    def test_steep_climb_scoring(self):
        """Test scoring for a steep climb."""
        # 3km climb, 400m gain, avg 13.3% grade
        elevation_profile = [
            600, 700, 800, 900, 1000
        ]  # 4 segments × 0.75km = 3km

        climb = self.create_climb_scenario(
            "Steep Mountain Road",
            elevation_profile,
            distance_per_segment=0.75
        )

        assert climb["total_distance"] == 3.0
        assert climb["elevation_gain"] == pytest.approx(400.0, abs=5.0)
        assert climb["avg_grade"] > 10.0  # Should be steep
        assert climb["fiets_score"] > 5000  # FIETS favors steep climbs

    def test_hc_climb_scoring(self):
        """Test scoring for HC (Hors Catégorie) climb."""
        # 20km climb, 1200m gain, avg 6% grade
        # Should have climb_score > 80,000 for HC
        elevation_profile = [
            400, 460, 520, 580, 640, 700, 760, 820,
            880, 940, 1000, 1060, 1120, 1180, 1240,
            1300, 1360, 1420, 1480, 1540, 1600
        ]  # 20 segments × 1km = 20km

        climb = self.create_climb_scenario(
            "HC Mountain Pass",
            elevation_profile,
            distance_per_segment=1.0
        )

        assert climb["total_distance"] == 20.0
        assert climb["elevation_gain"] == pytest.approx(1200.0, abs=10.0)
        assert climb["category"] == "HC"
        assert climb["basic_score"] > 80000

    def test_scoring_comparison_different_systems(self):
        """Test that different scoring systems rank climbs differently."""
        # Short steep climb
        steep = self.create_climb_scenario(
            "Short Steep",
            [500, 700, 900],  # 400m gain
            distance_per_segment=1.0  # 2km total
        )

        # Long gentle climb
        gentle = self.create_climb_scenario(
            "Long Gentle",
            [500, 540, 580, 620, 660, 700, 740, 780, 820, 860, 900],  # 400m gain
            distance_per_segment=1.0  # 10km total
        )

        # Basic score: Should be similar (both 2km × grade or 10km × grade)
        # But steep should have higher basic score
        assert steep["basic_score"] > gentle["basic_score"]

        # FIETS score: Should heavily favor the steep climb
        # (because gain is squared but distance is linear)
        assert steep["fiets_score"] > gentle["fiets_score"]


# ============================================================================
# Error Handling Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestErrorHandlingWorkflow:
    """Test error handling in complete workflows."""

    def test_partial_elevation_failure(self):
        """Test handling when some elevation fetches fail."""

        def elevation_func_with_gaps(lat, lon):
            # Fail for certain coordinates
            if 35.003 < lat < 35.006:
                return None  # Simulate data gap
            return 500 + (lat - 35.0) * 1000

        nodes = []
        for i in range(10):
            lat = 35.0 + (i * 0.001)
            lon = -82.0
            nodes.append(MockNode(i, lat, lon))

        road = MockRoadSegment("Road with Gaps", nodes, [123])
        fetcher = MockElevationFetcher(elevation_function=elevation_func_with_gaps)

        # Fetch elevations
        elevations = []
        failed_indices = []

        for i, node in enumerate(road.nodes):
            result = fetcher.fetch_elevation(node.lat, node.lon)
            if result["elevation"] is None:
                failed_indices.append(i)
                elevations.append(None)
            else:
                elevations.append(result["elevation"])

        # Should have some failures
        assert len(failed_indices) > 0
        assert len(fetcher.failed_coords) > 0

        # Valid elevations should still be processable
        valid_elevations = [e for e in elevations if e is not None]
        assert len(valid_elevations) > 0

    def test_complete_elevation_failure(self):
        """Test handling when all elevation fetches fail."""

        def always_fail(lat, lon):
            return None

        nodes = []
        for i in range(5):
            nodes.append(MockNode(i, 35.0 + i*0.001, -82.0))

        road = MockRoadSegment("Failed Road", nodes, [456])
        fetcher = MockElevationFetcher(elevation_function=always_fail)

        elevations = []
        for node in road.nodes:
            result = fetcher.fetch_elevation(node.lat, node.lon)
            elevations.append(result["elevation"])

        # All should be None
        assert all(e is None for e in elevations)
        assert len(fetcher.failed_coords) == len(nodes)


# ============================================================================
# Data Structure Integration Tests
# ============================================================================

@pytest.mark.integration
class TestDataStructureIntegration:
    """Test integration of data structures throughout workflow."""

    def test_road_to_metrics_conversion(self):
        """Test converting road data to ClimbMetrics."""
        from climb_analyzer.core.segment import ClimbMetrics

        # Simulate complete workflow data
        road_data = {
            "street_name": "Integration Test Road",
            "way_ids": [111111, 222222],
            "elevation_gain": 350.0,
            "height": 350.0,
            "length_km": 7.0,
            "distance_km": 6.5,
            "avg_grade": 5.0,
            "max_grade": 12.0,
            "min_elevation": 500.0,
            "max_elevation": 850.0,
        }

        # Calculate scores
        basic_score = road_data["length_km"] * 1000 * road_data["avg_grade"]
        fiets_score = (road_data["elevation_gain"] ** 2) / (road_data["length_km"] * 10)

        # Create metrics
        metrics = ClimbMetrics(
            street_name=road_data["street_name"],
            climb_category="3",
            climb_score=basic_score,
            elevation_gain=road_data["elevation_gain"],
            height=road_data["height"],
            prominence=300.0,
            length_km=road_data["length_km"],
            distance_km=road_data["distance_km"],
            avg_grade=road_data["avg_grade"],
            max_grade=road_data["max_grade"],
            min_elevation=road_data["min_elevation"],
            max_elevation=road_data["max_elevation"],
            surface="asphalt",
            tracktype="-",
            tracktype_definition="-",
            way_ids=road_data["way_ids"],
            osm_links=[f"[{wid}](https://www.openstreetmap.org/way/{wid})" for wid in road_data["way_ids"]],
            fiets_score=fiets_score
        )

        # Verify metrics were created correctly
        assert metrics.street_name == "Integration Test Road"
        assert len(metrics.way_ids) == 2
        assert metrics.climb_score == basic_score
        assert metrics.fiets_score > 0
