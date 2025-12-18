#!/usr/bin/env python3
"""
Unit tests for strava_scraper.py - parsing and validation logic.

Tests the parsing functions without making actual HTTP requests.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strava_scraper import (
    StravaScraper,
    StravaSegmentData,
    validate_against_strava,
    format_validation_result,
)


class TestStravaSegmentData:
    """Tests for StravaSegmentData dataclass."""

    def test_is_valid_with_good_data(self):
        """Valid segment has no error and positive distance."""
        data = StravaSegmentData(
            segment_id="123",
            name="Test Climb",
            distance_km=5.0,
            elev_gain_m=300,
            avg_grade=6.0
        )
        assert data.is_valid() is True

    def test_is_valid_with_fetch_error(self):
        """Segment with fetch error is invalid."""
        data = StravaSegmentData(
            segment_id="123",
            name="Unknown",
            distance_km=0,
            elev_gain_m=0,
            avg_grade=0,
            fetch_error="Network error"
        )
        assert data.is_valid() is False

    def test_is_valid_with_zero_distance(self):
        """Segment with zero distance is invalid."""
        data = StravaSegmentData(
            segment_id="123",
            name="Test",
            distance_km=0,
            elev_gain_m=100,
            avg_grade=5.0
        )
        assert data.is_valid() is False


class TestDistanceParsing:
    """Tests for _parse_distance() method."""

    @pytest.fixture
    def scraper(self):
        """Create scraper instance with mocked dependencies."""
        with patch('strava_scraper.HAS_DEPENDENCIES', True):
            with patch('strava_scraper.requests'):
                scraper = StravaScraper.__new__(StravaScraper)
                scraper._cache = {}
                return scraper

    def test_parse_km(self, scraper):
        """Parse kilometers."""
        assert scraper._parse_distance("5.2km") == pytest.approx(5.2)
        assert scraper._parse_distance("5.2 km") == pytest.approx(5.2)
        assert scraper._parse_distance("10 kilometers") == pytest.approx(10.0)

    def test_parse_miles_to_km(self, scraper):
        """Parse miles and convert to km."""
        # 1 mile = 1.60934 km
        assert scraper._parse_distance("1mi") == pytest.approx(1.60934, rel=0.001)
        assert scraper._parse_distance("1 mile") == pytest.approx(1.60934, rel=0.001)
        assert scraper._parse_distance("3.2 miles") == pytest.approx(5.15, rel=0.01)

    def test_parse_invalid_returns_none(self, scraper):
        """Invalid distance strings return None."""
        assert scraper._parse_distance("no distance here") is None
        assert scraper._parse_distance("") is None
        assert scraper._parse_distance("5") is None  # No unit


class TestElevationParsing:
    """Tests for _parse_elevation() method."""

    @pytest.fixture
    def scraper(self):
        """Create scraper instance with mocked dependencies."""
        with patch('strava_scraper.HAS_DEPENDENCIES', True):
            with patch('strava_scraper.requests'):
                scraper = StravaScraper.__new__(StravaScraper)
                scraper._cache = {}
                return scraper

    def test_parse_meters(self, scraper):
        """Parse meters."""
        assert scraper._parse_elevation("500m") == pytest.approx(500.0)
        assert scraper._parse_elevation("500 m") == pytest.approx(500.0)
        assert scraper._parse_elevation("1234 meters") == pytest.approx(1234.0)

    def test_parse_feet_to_meters(self, scraper):
        """Parse feet and convert to meters."""
        # 1 foot = 0.3048 meters
        assert scraper._parse_elevation("1000ft") == pytest.approx(304.8, rel=0.001)
        assert scraper._parse_elevation("1000 feet") == pytest.approx(304.8, rel=0.001)

    def test_parse_with_commas(self, scraper):
        """Parse elevation with thousands separator."""
        assert scraper._parse_elevation("1,640ft") == pytest.approx(499.87, rel=0.01)
        assert scraper._parse_elevation("2,000 m") == pytest.approx(2000.0)

    def test_parse_invalid_returns_none(self, scraper):
        """Invalid elevation strings return None."""
        assert scraper._parse_elevation("no elevation") is None
        assert scraper._parse_elevation("") is None
        assert scraper._parse_elevation("500") is None  # No unit


class TestGradeParsing:
    """Tests for _parse_grade() method."""

    @pytest.fixture
    def scraper(self):
        """Create scraper instance with mocked dependencies."""
        with patch('strava_scraper.HAS_DEPENDENCIES', True):
            with patch('strava_scraper.requests'):
                scraper = StravaScraper.__new__(StravaScraper)
                scraper._cache = {}
                return scraper

    def test_parse_percentage(self, scraper):
        """Parse grade percentage."""
        assert scraper._parse_grade("5.2%") == pytest.approx(5.2)
        assert scraper._parse_grade("5.2 %") == pytest.approx(5.2)
        assert scraper._parse_grade("10%") == pytest.approx(10.0)

    def test_parse_invalid_returns_none(self, scraper):
        """Invalid grade strings return None."""
        assert scraper._parse_grade("no grade") is None
        assert scraper._parse_grade("") is None
        assert scraper._parse_grade("5.2") is None  # No percent sign


class TestStravaValidationComparison:
    """Tests for validation comparison logic."""

    def test_within_tolerance(self):
        """Test metrics within tolerance."""
        climb_data = {
            'distance_km': 5.0,
            'elev_gain_m': 300,
            'avg_grade': 6.0
        }
        strava_data = StravaSegmentData(
            segment_id="123",
            name="Test",
            distance_km=5.2,    # 4% diff
            elev_gain_m=310,    # 3.2% diff
            avg_grade=6.3       # 4.8% diff
        )

        # Create mock scraper
        with patch('strava_scraper.StravaScraper') as MockScraper:
            mock_instance = MockScraper.return_value
            mock_instance.get_segment.return_value = strava_data

            result = validate_against_strava(climb_data, "123", tolerance_percent=15.0)

        assert result['valid'] is True
        assert result['comparisons']['distance']['within_tolerance'] is True
        assert result['comparisons']['elevation']['within_tolerance'] is True
        assert result['comparisons']['grade']['within_tolerance'] is True

    def test_outside_tolerance(self):
        """Test metrics outside tolerance."""
        climb_data = {
            'distance_km': 5.0,
            'elev_gain_m': 300,
            'avg_grade': 6.0
        }
        strava_data = StravaSegmentData(
            segment_id="123",
            name="Test",
            distance_km=7.0,    # 40% diff - outside 15% tolerance
            elev_gain_m=310,
            avg_grade=6.3
        )

        with patch('strava_scraper.StravaScraper') as MockScraper:
            mock_instance = MockScraper.return_value
            mock_instance.get_segment.return_value = strava_data

            result = validate_against_strava(climb_data, "123", tolerance_percent=15.0)

        assert result['valid'] is False
        assert result['comparisons']['distance']['within_tolerance'] is False
        assert result['comparisons']['distance']['diff_percent'] == pytest.approx(28.57, rel=0.1)

    def test_missing_strava_data(self):
        """Test handling of failed Strava fetch."""
        climb_data = {'distance_km': 5.0}
        strava_data = StravaSegmentData(
            segment_id="123",
            name="Unknown",
            distance_km=0,
            elev_gain_m=0,
            avg_grade=0,
            fetch_error="Segment not found"
        )

        with patch('strava_scraper.StravaScraper') as MockScraper:
            mock_instance = MockScraper.return_value
            mock_instance.get_segment.return_value = strava_data

            result = validate_against_strava(climb_data, "123")

        assert result['valid'] is False
        assert result['error'] == "Segment not found"
        assert result['comparisons'] == {}

    def test_custom_tolerance(self):
        """Test with custom tolerance percentage."""
        climb_data = {
            'distance_km': 5.0,
            'elev_gain_m': 300,
            'avg_grade': 6.0
        }
        strava_data = StravaSegmentData(
            segment_id="123",
            name="Test",
            distance_km=5.5,    # 10% diff
            elev_gain_m=330,    # 10% diff
            avg_grade=6.6       # 10% diff
        )

        with patch('strava_scraper.StravaScraper') as MockScraper:
            mock_instance = MockScraper.return_value
            mock_instance.get_segment.return_value = strava_data

            # Should fail with 5% tolerance
            result = validate_against_strava(climb_data, "123", tolerance_percent=5.0)
            assert result['valid'] is False

            # Should pass with 15% tolerance
            result = validate_against_strava(climb_data, "123", tolerance_percent=15.0)
            assert result['valid'] is True


class TestFormatValidationResult:
    """Tests for format_validation_result() function."""

    def test_format_error_result(self):
        """Format result with error."""
        result = {
            'valid': False,
            'error': 'Network timeout',
            'comparisons': {}
        }
        output = format_validation_result(result)
        assert "Strava validation failed" in output
        assert "Network timeout" in output

    def test_format_success_result(self):
        """Format successful validation result."""
        result = {
            'valid': True,
            'strava_data': {
                'name': 'Test Climb',
                'distance_km': 5.0,
                'elev_gain_m': 300,
                'avg_grade': 6.0,
                'location': 'Test Location'
            },
            'comparisons': {
                'distance': {
                    'climb_analyzer': 5.0,
                    'strava': 5.2,
                    'diff_percent': 3.8,
                    'within_tolerance': True
                }
            },
            'tolerance_percent': 15
        }
        output = format_validation_result(result)
        assert "Test Climb" in output
        assert "Test Location" in output
        assert "PASS" in output
        assert "[OK]" in output

    def test_format_failed_result(self):
        """Format failed validation result."""
        result = {
            'valid': False,
            'strava_data': {
                'name': 'Test Climb',
                'distance_km': 5.0,
                'elev_gain_m': 300,
                'avg_grade': 6.0
            },
            'comparisons': {
                'distance': {
                    'climb_analyzer': 5.0,
                    'strava': 7.0,
                    'diff_percent': 40.0,
                    'within_tolerance': False
                }
            },
            'tolerance_percent': 15
        }
        output = format_validation_result(result)
        assert "FAIL" in output
        assert "[DIFF]" in output


class TestScraperCaching:
    """Tests for scraper caching behavior."""

    def test_cache_hit(self):
        """Test that cached results are returned."""
        with patch('strava_scraper.HAS_DEPENDENCIES', True):
            with patch('strava_scraper.requests'):
                scraper = StravaScraper.__new__(StravaScraper)
                scraper._cache = {}
                scraper.session = Mock()

                # Pre-populate cache
                cached_data = StravaSegmentData(
                    segment_id="123",
                    name="Cached Climb",
                    distance_km=5.0,
                    elev_gain_m=300,
                    avg_grade=6.0
                )
                scraper._cache["123"] = cached_data

                # Should return cached data without fetching
                result = scraper.get_segment("123")
                assert result.name == "Cached Climb"
                assert scraper.session.get.call_count == 0

    def test_clear_cache(self):
        """Test cache clearing."""
        with patch('strava_scraper.HAS_DEPENDENCIES', True):
            with patch('strava_scraper.requests'):
                scraper = StravaScraper.__new__(StravaScraper)
                scraper._cache = {"123": Mock()}

                scraper.clear_cache()
                assert len(scraper._cache) == 0
