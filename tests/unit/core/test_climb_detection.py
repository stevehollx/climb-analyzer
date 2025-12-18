"""
Unit tests for climb detection logic.

Tests the core functionality of detecting climbs from elevation data,
including segment detection, filtering, and validation.
"""

import pytest
import math
from typing import List, Tuple


# ============================================================================
# Distance Calculation Tests
# ============================================================================

class TestDistanceCalculation:
    """Test Haversine distance calculation."""

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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

    def test_same_point_distance(self):
        """Distance between same point should be zero."""
        distance = self.calculate_distance(35.0, -82.0, 35.0, -82.0)
        assert distance == 0.0

    def test_known_distance(self):
        """Test with known distance between two points."""
        # Roughly 1 degree latitude = ~111km
        distance = self.calculate_distance(35.0, -82.0, 36.0, -82.0)
        assert 110 < distance < 112  # Should be approximately 111km

    def test_distance_symmetry(self):
        """Distance should be same regardless of order."""
        dist1 = self.calculate_distance(35.0, -82.0, 36.0, -83.0)
        dist2 = self.calculate_distance(36.0, -83.0, 35.0, -82.0)
        assert abs(dist1 - dist2) < 0.001

    def test_equator_distance(self):
        """Test distance calculation at equator."""
        # 1 degree longitude at equator ≈ 111km
        distance = self.calculate_distance(0.0, 0.0, 0.0, 1.0)
        assert 110 < distance < 112

    def test_pole_distance(self):
        """Test distance calculation near pole."""
        # Longitude differences matter less near poles
        distance = self.calculate_distance(89.0, 0.0, 89.0, 1.0)
        assert distance < 2  # Should be very small


# ============================================================================
# Elevation Metrics Tests
# ============================================================================

class TestElevationMetrics:
    """Test elevation gain, loss, and grade calculations."""

    def calculate_elevation_gain(self, elevations: List[float]) -> float:
        """Calculate total elevation gain."""
        gain = 0.0
        for i in range(len(elevations) - 1):
            elev_change = elevations[i + 1] - elevations[i]
            if elev_change > 0:
                gain += elev_change
        return gain

    def calculate_elevation_loss(self, elevations: List[float]) -> float:
        """Calculate total elevation loss."""
        loss = 0.0
        for i in range(len(elevations) - 1):
            elev_change = elevations[i + 1] - elevations[i]
            if elev_change < 0:
                loss += abs(elev_change)
        return loss

    def calculate_grade(self, elev_change: float, distance_m: float) -> float:
        """Calculate grade as percentage."""
        if distance_m == 0:
            return 0.0
        return (elev_change / distance_m) * 100

    def test_elevation_gain_simple_climb(self):
        """Test elevation gain for simple upward climb."""
        elevations = [100, 150, 200, 250, 300]
        gain = self.calculate_elevation_gain(elevations)
        assert gain == 200.0

    def test_elevation_gain_with_descent(self):
        """Test elevation gain with some descents."""
        elevations = [100, 200, 150, 250, 200, 300]
        gain = self.calculate_elevation_gain(elevations)
        # Gains: 100 + 100 + 100 = 300
        assert gain == 300.0

    def test_elevation_loss_simple_descent(self):
        """Test elevation loss for simple descent."""
        elevations = [300, 250, 200, 150, 100]
        loss = self.calculate_elevation_loss(elevations)
        assert loss == 200.0

    def test_elevation_loss_with_climbs(self):
        """Test elevation loss with some climbs."""
        elevations = [300, 200, 250, 150, 200, 100]
        loss = self.calculate_elevation_loss(elevations)
        # Losses: 100 + 100 + 100 = 300
        assert loss == 300.0

    def test_flat_terrain(self):
        """Test flat terrain has no gain or loss."""
        elevations = [100, 100, 100, 100]
        gain = self.calculate_elevation_gain(elevations)
        loss = self.calculate_elevation_loss(elevations)
        assert gain == 0.0
        assert loss == 0.0

    def test_grade_calculation(self):
        """Test grade percentage calculation."""
        # 10m rise over 100m distance = 10% grade
        grade = self.calculate_grade(10.0, 100.0)
        assert grade == 10.0

    def test_grade_steep(self):
        """Test steep grade calculation."""
        # 20m rise over 100m = 20% grade
        grade = self.calculate_grade(20.0, 100.0)
        assert grade == 20.0

    def test_grade_gentle(self):
        """Test gentle grade calculation."""
        # 3m rise over 100m = 3% grade
        grade = self.calculate_grade(3.0, 100.0)
        assert grade == 3.0

    def test_grade_zero_distance(self):
        """Test grade with zero distance."""
        grade = self.calculate_grade(10.0, 0.0)
        assert grade == 0.0


# ============================================================================
# Climb Detection Tests
# ============================================================================

class TestClimbDetection:
    """Test climb segment detection logic."""

    def is_valid_climb(
        self,
        elevation_gain: float,
        distance_m: float,
        avg_grade: float,
        min_gain: float = 30.0,
        min_distance: float = 500.0,
        min_grade: float = 3.0
    ) -> bool:
        """
        Determine if a segment qualifies as a climb.

        Args:
            elevation_gain: Total elevation gain in meters
            distance_m: Total distance in meters
            avg_grade: Average grade as percentage
            min_gain: Minimum elevation gain threshold
            min_distance: Minimum distance threshold
            min_grade: Minimum average grade threshold

        Returns:
            True if segment qualifies as a climb
        """
        return (
            elevation_gain >= min_gain
            and distance_m >= min_distance
            and avg_grade >= min_grade
        )

    @pytest.mark.unit
    @pytest.mark.climb_detection
    def test_valid_climb(self):
        """Test that valid climb is detected."""
        # 50m gain, 1000m distance, 5% grade
        assert self.is_valid_climb(50.0, 1000.0, 5.0) is True

    @pytest.mark.unit
    @pytest.mark.climb_detection
    def test_insufficient_gain(self):
        """Test that climb with insufficient gain is rejected."""
        # Only 20m gain (below 30m threshold)
        assert self.is_valid_climb(20.0, 1000.0, 5.0) is False

    @pytest.mark.unit
    @pytest.mark.climb_detection
    def test_insufficient_distance(self):
        """Test that climb with insufficient distance is rejected."""
        # Only 400m distance (below 500m threshold)
        assert self.is_valid_climb(50.0, 400.0, 5.0) is False

    @pytest.mark.unit
    @pytest.mark.climb_detection
    def test_insufficient_grade(self):
        """Test that climb with insufficient grade is rejected."""
        # Only 2% grade (below 3% threshold)
        assert self.is_valid_climb(50.0, 1000.0, 2.0) is False

    @pytest.mark.unit
    @pytest.mark.climb_detection
    def test_edge_case_exactly_at_threshold(self):
        """Test climb exactly at threshold values."""
        # Exactly at thresholds: 30m gain, 500m distance, 3% grade
        assert self.is_valid_climb(30.0, 500.0, 3.0) is True

    @pytest.mark.unit
    @pytest.mark.climb_detection
    def test_custom_thresholds(self):
        """Test with custom threshold values."""
        # Custom thresholds for harder climbs
        assert self.is_valid_climb(
            elevation_gain=100.0,
            distance_m=2000.0,
            avg_grade=5.0,
            min_gain=100.0,
            min_distance=2000.0,
            min_grade=5.0
        ) is True

        # Should fail with harder thresholds
        assert self.is_valid_climb(
            elevation_gain=50.0,
            distance_m=1000.0,
            avg_grade=3.0,
            min_gain=100.0,
            min_distance=2000.0,
            min_grade=5.0
        ) is False


# ============================================================================
# Max Grade Tests
# ============================================================================

class TestMaxGrade:
    """Test maximum grade calculation across segments."""

    def calculate_max_grade(self, elevations: List[float], distances: List[float]) -> float:
        """
        Calculate maximum grade across all segments.

        Args:
            elevations: List of elevation values
            distances: List of distance values between consecutive points (in km)

        Returns:
            Maximum grade as percentage
        """
        max_grade = 0.0

        for i in range(len(elevations) - 1):
            elev_change = elevations[i + 1] - elevations[i]
            if distances[i] > 0:
                grade = abs(elev_change / (distances[i] * 1000)) * 100
                max_grade = max(max_grade, grade)

        return max_grade

    @pytest.mark.unit
    def test_uniform_grade(self):
        """Test climb with uniform grade."""
        elevations = [100, 105, 110, 115, 120]
        distances = [0.1, 0.1, 0.1, 0.1]  # 100m each
        max_grade = self.calculate_max_grade(elevations, distances)
        assert abs(max_grade - 5.0) < 0.01  # 5m / 100m = 5%

    @pytest.mark.unit
    def test_varying_grade(self):
        """Test climb with varying grades."""
        elevations = [100, 105, 120, 130, 135]
        # Grades: 5%, 15%, 10%, 5%
        distances = [0.1, 0.1, 0.1, 0.1]
        max_grade = self.calculate_max_grade(elevations, distances)
        assert abs(max_grade - 15.0) < 0.01

    @pytest.mark.unit
    def test_steep_section(self):
        """Test climb with one very steep section."""
        elevations = [100, 105, 125, 130, 135]
        # Grades: 5%, 20%, 5%, 5% - max should be 20%
        distances = [0.1, 0.1, 0.1, 0.1]
        max_grade = self.calculate_max_grade(elevations, distances)
        assert abs(max_grade - 20.0) < 0.01

    @pytest.mark.unit
    def test_descent_absolute_value(self):
        """Test that descent is counted as absolute grade."""
        elevations = [100, 80, 100]
        # -20m / 100m = -20%, but we take absolute value
        distances = [0.1, 0.1]
        max_grade = self.calculate_max_grade(elevations, distances)
        assert abs(max_grade - 20.0) < 0.01


# ============================================================================
# Average Grade Tests
# ============================================================================

class TestAverageGrade:
    """Test average grade calculation."""

    def calculate_avg_grade(self, height: float, total_distance_km: float) -> float:
        """
        Calculate average grade.

        Args:
            height: Height difference (max - min elevation)
            total_distance_km: Total distance in kilometers

        Returns:
            Average grade as percentage
        """
        if total_distance_km == 0:
            return 0.0
        return (height / (total_distance_km * 1000)) * 100

    @pytest.mark.unit
    def test_simple_avg_grade(self):
        """Test simple average grade calculation."""
        # 100m height over 2km = 5%
        avg_grade = self.calculate_avg_grade(100.0, 2.0)
        assert avg_grade == 5.0

    @pytest.mark.unit
    def test_steep_avg_grade(self):
        """Test steep average grade."""
        # 200m over 1km = 20%
        avg_grade = self.calculate_avg_grade(200.0, 1.0)
        assert avg_grade == 20.0

    @pytest.mark.unit
    def test_gentle_avg_grade(self):
        """Test gentle average grade."""
        # 30m over 1km = 3%
        avg_grade = self.calculate_avg_grade(30.0, 1.0)
        assert avg_grade == 3.0

    @pytest.mark.unit
    def test_zero_distance(self):
        """Test with zero distance."""
        avg_grade = self.calculate_avg_grade(100.0, 0.0)
        assert avg_grade == 0.0


# ============================================================================
# Integration Tests for Complete Metrics
# ============================================================================

@pytest.mark.unit
class TestCompleteClimbMetrics:
    """Test complete climb metrics calculation with realistic data."""

    def test_realistic_moderate_climb(self, sample_climb_profile):
        """Test metrics for a realistic moderate climb."""
        distances = sample_climb_profile["distances"]
        elevations = sample_climb_profile["elevations"]

        # Calculate total distance
        total_distance_km = distances[-1] / 1000.0

        # Calculate height
        height = max(elevations) - min(elevations)

        # Calculate elevation gain
        gain = 0.0
        for i in range(len(elevations) - 1):
            change = elevations[i + 1] - elevations[i]
            if change > 0:
                gain += change

        # Calculate average grade
        avg_grade = (height / (total_distance_km * 1000)) * 100

        # Assertions
        assert total_distance_km == 5.0
        assert height == 400.0  # 900 - 500
        assert gain == 400.0  # All uphill sections
        assert abs(avg_grade - 8.0) < 0.1  # 400m / 5000m

    def test_realistic_steep_climb(self, sample_steep_climb_profile):
        """Test metrics for a realistic steep climb."""
        distances = sample_steep_climb_profile["distances"]
        elevations = sample_steep_climb_profile["elevations"]

        total_distance_km = distances[-1] / 1000.0
        height = max(elevations) - min(elevations)

        gain = 0.0
        for i in range(len(elevations) - 1):
            change = elevations[i + 1] - elevations[i]
            if change > 0:
                gain += change

        avg_grade = (height / (total_distance_km * 1000)) * 100

        # Assertions
        assert total_distance_km == 3.0
        assert height == 550.0  # 950 - 400
        assert gain == 550.0
        assert abs(avg_grade - 18.33) < 0.1  # 550m / 3000m


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling in climb detection."""

    def test_empty_elevation_list(self):
        """Test handling of empty elevation list."""
        elevations = []
        gain = 0.0
        for i in range(len(elevations) - 1):
            change = elevations[i + 1] - elevations[i]
            if change > 0:
                gain += change
        assert gain == 0.0

    def test_single_point(self):
        """Test handling of single elevation point."""
        elevations = [100]
        gain = 0.0
        for i in range(len(elevations) - 1):
            change = elevations[i + 1] - elevations[i]
            if change > 0:
                gain += change
        assert gain == 0.0

    def test_two_points_minimum(self):
        """Test with minimum two points."""
        elevations = [100, 150]
        gain = 0.0
        for i in range(len(elevations) - 1):
            change = elevations[i + 1] - elevations[i]
            if change > 0:
                gain += change
        assert gain == 50.0

    def test_negative_elevations(self):
        """Test with negative elevation values (below sea level)."""
        elevations = [-50, 0, 50, 100]
        gain = 0.0
        for i in range(len(elevations) - 1):
            change = elevations[i + 1] - elevations[i]
            if change > 0:
                gain += change
        assert gain == 150.0

    def test_very_small_changes(self):
        """Test with very small elevation changes."""
        elevations = [100.0, 100.1, 100.2, 100.15, 100.3]
        gain = 0.0
        for i in range(len(elevations) - 1):
            change = elevations[i + 1] - elevations[i]
            if change > 0:
                gain += change
        assert abs(gain - 0.3) < 0.001
