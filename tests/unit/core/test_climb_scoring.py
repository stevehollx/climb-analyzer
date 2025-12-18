"""
Unit tests for climb scoring algorithms.

Tests the three scoring systems used in climb analysis:
1. Basic score (distance × grade)
2. FIETS score (elevation_gain² / (distance × 10))
3. PDI score (complex formula with surface and elevation factors)
"""

import pytest
import math


# ============================================================================
# Basic Score Tests
# ============================================================================

class TestBasicScore:
    """Test basic climb scoring (distance × grade)."""

    def calculate_basic_score(self, distance_km: float, avg_grade: float) -> float:
        """
        Calculate basic climb score.

        Args:
            distance_km: Distance in kilometers
            avg_grade: Average grade as percentage

        Returns:
            Basic score (distance_m × avg_grade)
        """
        if avg_grade <= 0:
            return 0.0
        return distance_km * 1000 * avg_grade

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_basic_score_moderate_climb(self):
        """Test basic score for moderate climb."""
        # 5km at 5% = 25,000
        score = self.calculate_basic_score(5.0, 5.0)
        assert score == 25000.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_basic_score_steep_short_climb(self):
        """Test basic score for steep but short climb."""
        # 2km at 10% = 20,000
        score = self.calculate_basic_score(2.0, 10.0)
        assert score == 20000.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_basic_score_long_gentle_climb(self):
        """Test basic score for long gentle climb."""
        # 10km at 3% = 30,000
        score = self.calculate_basic_score(10.0, 3.0)
        assert score == 30000.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_basic_score_zero_grade(self):
        """Test basic score with zero grade."""
        score = self.calculate_basic_score(5.0, 0.0)
        assert score == 0.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_basic_score_negative_grade(self):
        """Test basic score with negative grade (descent)."""
        score = self.calculate_basic_score(5.0, -5.0)
        assert score == 0.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_basic_score_category_thresholds(self):
        """Test scores at category boundaries."""
        # Category 4: > 8,000
        score_cat4 = self.calculate_basic_score(2.0, 5.0)  # 10,000
        assert score_cat4 > 8000

        # Category 3: > 16,000
        score_cat3 = self.calculate_basic_score(4.0, 5.0)  # 20,000
        assert score_cat3 > 16000

        # Category 2: > 32,000
        score_cat2 = self.calculate_basic_score(8.0, 5.0)  # 40,000
        assert score_cat2 > 32000

        # Category 1: > 64,000
        score_cat1 = self.calculate_basic_score(16.0, 5.0)  # 80,000
        assert score_cat1 > 64000

        # HC: > 80,000
        score_hc = self.calculate_basic_score(20.0, 5.0)  # 100,000
        assert score_hc > 80000


# ============================================================================
# FIETS Score Tests
# ============================================================================

class TestFietsScore:
    """Test FIETS difficulty index scoring."""

    def calculate_fiets_score(self, elevation_gain_m: float, distance_km: float) -> float:
        """
        Calculate FIETS difficulty index.

        Formula: elevation_gain² / (distance × 10)

        Args:
            elevation_gain_m: Elevation gain in meters
            distance_km: Distance in kilometers

        Returns:
            FIETS score
        """
        if distance_km <= 0:
            return 0.0
        return (elevation_gain_m ** 2) / (distance_km * 10)

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_fiets_score_moderate_climb(self):
        """Test FIETS score for moderate climb."""
        # 200m gain over 4km
        # (200²) / (4 × 10) = 40,000 / 40 = 1,000
        score = self.calculate_fiets_score(200.0, 4.0)
        assert score == 1000.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_fiets_score_steep_climb(self):
        """Test FIETS score for steep climb."""
        # 500m gain over 5km
        # (500²) / (5 × 10) = 250,000 / 50 = 5,000
        score = self.calculate_fiets_score(500.0, 5.0)
        assert score == 5000.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_fiets_score_gentle_climb(self):
        """Test FIETS score for gentle climb."""
        # 100m gain over 10km
        # (100²) / (10 × 10) = 10,000 / 100 = 100
        score = self.calculate_fiets_score(100.0, 10.0)
        assert score == 100.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_fiets_score_zero_distance(self):
        """Test FIETS score with zero distance."""
        score = self.calculate_fiets_score(100.0, 0.0)
        assert score == 0.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_fiets_score_zero_gain(self):
        """Test FIETS score with zero elevation gain."""
        score = self.calculate_fiets_score(0.0, 5.0)
        assert score == 0.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_fiets_score_doubling_gain(self):
        """Test that doubling gain quadruples FIETS score."""
        score1 = self.calculate_fiets_score(100.0, 5.0)
        score2 = self.calculate_fiets_score(200.0, 5.0)
        # Should be 4x because gain is squared
        assert abs(score2 - (score1 * 4)) < 0.01

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_fiets_score_doubling_distance(self):
        """Test that doubling distance halves FIETS score."""
        score1 = self.calculate_fiets_score(200.0, 5.0)
        score2 = self.calculate_fiets_score(200.0, 10.0)
        # Should be 0.5x because distance is in denominator
        assert abs(score2 - (score1 * 0.5)) < 0.01


# ============================================================================
# PDI Score Tests
# ============================================================================

class TestPDIScore:
    """Test PDI (Plan de Inclinación) difficulty index scoring."""

    def map_surface_to_pdi_scale(self, surface: str) -> float:
        """Map surface types to PDI's 0-5 scale."""
        surface_mapping = {
            "asphalt": 0.0,
            "paved": 0.0,
            "concrete": 0.0,
            "gravel": 1.5,
            "compacted": 1.0,
            "dirt": 2.5,
            "unpaved": 2.0,
            "track": 2.0,
            "unknown": 1.0,
        }
        return surface_mapping.get(surface.lower(), 1.0)

    def calculate_pdi_score(
        self,
        elevation_gain_m: float,
        elevation_loss_m: float,
        distance_km: float,
        min_elev_m: float,
        max_elev_m: float,
        surface: str = "asphalt"
    ) -> float:
        """
        Calculate PDI difficulty index.

        Formula:
        - work = (8.6 × distance_m + 735 × (gain - 0.25 × loss)) / 1600
        - elevation_factor = 1 + (min_elev² + max_elev²) / 7.2e6
        - surface_factor = 1 + 0.2 × surface_index
        - PDI = elevation_factor × surface_factor × (work² / distance_m)

        Args:
            elevation_gain_m: Elevation gain in meters
            elevation_loss_m: Elevation loss in meters
            distance_km: Distance in kilometers
            min_elev_m: Minimum elevation in meters
            max_elev_m: Maximum elevation in meters
            surface: Surface type

        Returns:
            PDI score
        """
        total_distance_m = distance_km * 1000

        if total_distance_m == 0:
            return 0.0

        # Work calculation
        work = (8.6 * total_distance_m + 735 * (elevation_gain_m - 0.25 * elevation_loss_m)) / 1600

        # Elevation factor
        elevation_factor = 1 + (min_elev_m ** 2 + max_elev_m ** 2) / (7.2e6)

        # Surface factor
        surface_index = self.map_surface_to_pdi_scale(surface)
        surface_factor = 1 + 0.2 * surface_index

        # Final PDI
        pdi_score = elevation_factor * surface_factor * (work ** 2 / total_distance_m)

        return pdi_score

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_pdi_score_moderate_climb_paved(self):
        """Test PDI score for moderate paved climb."""
        score = self.calculate_pdi_score(
            elevation_gain_m=200.0,
            elevation_loss_m=20.0,
            distance_km=4.0,
            min_elev_m=500.0,
            max_elev_m=700.0,
            surface="asphalt"
        )
        assert score > 0  # Should be positive
        assert score < 1000  # Reasonable range for moderate climb

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_pdi_score_steep_climb(self):
        """Test PDI score for steep climb."""
        steep_score = self.calculate_pdi_score(
            elevation_gain_m=500.0,
            elevation_loss_m=50.0,
            distance_km=5.0,
            min_elev_m=500.0,
            max_elev_m=1000.0,
            surface="asphalt"
        )

        moderate_score = self.calculate_pdi_score(
            elevation_gain_m=200.0,
            elevation_loss_m=20.0,
            distance_km=5.0,
            min_elev_m=500.0,
            max_elev_m=700.0,
            surface="asphalt"
        )

        # Steeper climb should have higher score
        assert steep_score > moderate_score

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_pdi_score_surface_impact(self):
        """Test that surface type impacts PDI score."""
        paved_score = self.calculate_pdi_score(
            elevation_gain_m=200.0,
            elevation_loss_m=20.0,
            distance_km=4.0,
            min_elev_m=500.0,
            max_elev_m=700.0,
            surface="asphalt"
        )

        dirt_score = self.calculate_pdi_score(
            elevation_gain_m=200.0,
            elevation_loss_m=20.0,
            distance_km=4.0,
            min_elev_m=500.0,
            max_elev_m=700.0,
            surface="dirt"
        )

        # Dirt surface should increase difficulty
        assert dirt_score > paved_score

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_pdi_score_elevation_factor(self):
        """Test that higher elevations increase PDI score."""
        low_elev_score = self.calculate_pdi_score(
            elevation_gain_m=200.0,
            elevation_loss_m=20.0,
            distance_km=4.0,
            min_elev_m=100.0,
            max_elev_m=300.0,
            surface="asphalt"
        )

        high_elev_score = self.calculate_pdi_score(
            elevation_gain_m=200.0,
            elevation_loss_m=20.0,
            distance_km=4.0,
            min_elev_m=2000.0,
            max_elev_m=2200.0,
            surface="asphalt"
        )

        # Higher elevation should increase difficulty (altitude effect)
        assert high_elev_score > low_elev_score

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_pdi_score_zero_distance(self):
        """Test PDI score with zero distance."""
        score = self.calculate_pdi_score(
            elevation_gain_m=200.0,
            elevation_loss_m=20.0,
            distance_km=0.0,
            min_elev_m=500.0,
            max_elev_m=700.0,
            surface="asphalt"
        )
        assert score == 0.0

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_pdi_surface_mapping(self):
        """Test surface to PDI scale mapping."""
        assert self.map_surface_to_pdi_scale("asphalt") == 0.0
        assert self.map_surface_to_pdi_scale("paved") == 0.0
        assert self.map_surface_to_pdi_scale("gravel") == 1.5
        assert self.map_surface_to_pdi_scale("dirt") == 2.5
        assert self.map_surface_to_pdi_scale("unknown") == 1.0


# ============================================================================
# Climb Categorization Tests
# ============================================================================

class TestClimbCategorization:
    """Test climb categorization based on score."""

    def categorize_climb(self, avg_grade: float, length_m: float, elevation_gain_m: float) -> str:
        """
        Categorize climb based on climb score (cycling categorization system).

        Args:
            avg_grade: Average grade as percentage
            length_m: Length in meters
            elevation_gain_m: Elevation gain in meters (not used in basic categorization)

        Returns:
            Climb category (HC, 1, 2, 3, 4, or N/A)
        """
        # Calculate climb score for categorization
        climb_score = length_m * avg_grade

        if climb_score > 80000:
            return "HC"  # Hors Catégorie (beyond categorization)
        elif climb_score > 64000:
            return "1"
        elif climb_score > 32000:
            return "2"
        elif climb_score > 16000:
            return "3"
        elif climb_score > 8000:
            return "4"
        else:
            return "N/A"  # Below category 4 threshold

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_hc(self):
        """Test Hors Catégorie classification."""
        # 16km at 6% = 96,000 > 80,000
        category = self.categorize_climb(6.0, 16000.0, 960.0)
        assert category == "HC"

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_1(self):
        """Test Category 1 classification."""
        # 13km at 5% = 65,000 > 64,000
        category = self.categorize_climb(5.0, 13000.0, 650.0)
        assert category == "1"

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_2(self):
        """Test Category 2 classification."""
        # 8km at 5% = 40,000 > 32,000
        category = self.categorize_climb(5.0, 8000.0, 400.0)
        assert category == "2"

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_3(self):
        """Test Category 3 classification."""
        # 4km at 5% = 20,000 > 16,000
        category = self.categorize_climb(5.0, 4000.0, 200.0)
        assert category == "3"

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_4(self):
        """Test Category 4 classification."""
        # 2km at 5% = 10,000 > 8,000
        category = self.categorize_climb(5.0, 2000.0, 100.0)
        assert category == "4"

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_na(self):
        """Test N/A (below threshold) classification."""
        # 1km at 5% = 5,000 < 8,000
        category = self.categorize_climb(5.0, 1000.0, 50.0)
        assert category == "N/A"

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_boundary_values(self):
        """Test categorization at exact boundary values."""
        # Exactly at HC threshold
        category_hc = self.categorize_climb(5.0, 16000.0, 800.0)  # 80,000
        assert category_hc == "N/A"  # Not > 80,000

        # Just above HC threshold
        category_hc_above = self.categorize_climb(5.01, 16000.0, 801.6)  # 80,160
        assert category_hc_above == "HC"

    @pytest.mark.unit
    @pytest.mark.climb_scoring
    def test_category_steep_short_vs_long_gentle(self):
        """Test that steep short and long gentle can be same category."""
        # 4km at 10% = 40,000 (Category 2)
        steep_short = self.categorize_climb(10.0, 4000.0, 400.0)

        # 8km at 5% = 40,000 (Category 2)
        long_gentle = self.categorize_climb(5.0, 8000.0, 400.0)

        assert steep_short == long_gentle == "2"


# ============================================================================
# Score Comparison Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.climb_scoring
class TestScoreComparison:
    """Test comparison between different scoring systems."""

    def test_same_climb_different_scores(self):
        """Test that same climb produces different scores with different systems."""
        # Test climb: 300m gain, 5km distance, avg 6% grade
        elevation_gain = 300.0
        distance_km = 5.0
        avg_grade = 6.0

        # Basic score
        basic_score = distance_km * 1000 * avg_grade  # 30,000

        # FIETS score
        fiets_score = (elevation_gain ** 2) / (distance_km * 10)  # 1,800

        # Scores should be very different
        assert basic_score != fiets_score
        assert basic_score > fiets_score  # Basic typically higher for moderate climbs

    def test_scoring_system_rankings(self):
        """Test that different systems may rank climbs differently."""
        # Climb A: Long, gentle (10km, 200m gain, 2% avg)
        climb_a_basic = 10.0 * 1000 * 2.0  # 20,000
        climb_a_fiets = (200.0 ** 2) / (10.0 * 10)  # 400

        # Climb B: Short, steep (2km, 200m gain, 10% avg)
        climb_b_basic = 2.0 * 1000 * 10.0  # 20,000
        climb_b_fiets = (200.0 ** 2) / (2.0 * 10)  # 2,000

        # Basic scores are equal
        assert climb_a_basic == climb_b_basic

        # But FIETS heavily favors the steeper climb
        assert climb_b_fiets > climb_a_fiets
