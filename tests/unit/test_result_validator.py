#!/usr/bin/env python3
"""
Unit tests for result_validator.py - validation and search logic.

Tests validation functions using mock DataFrames without actual file I/O.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from result_validator import (
    ResultValidator,
    ValidationResult,
    ValidationError,
    ClimbMatch,
    REQUIRED_COLUMNS,
    CATEGORY_THRESHOLDS,
    format_validation_summary,
)


class TestScoreToCategory:
    """Tests for _score_to_category() method."""

    @pytest.fixture
    def validator(self):
        """Create validator instance with mocked pandas check."""
        with patch('result_validator.HAS_PANDAS', True):
            return ResultValidator.__new__(ResultValidator)

    def test_hc_category(self, validator):
        """Test Hors Catégorie (HC) threshold."""
        assert validator._score_to_category(80000) == 'HC'
        assert validator._score_to_category(100000) == 'HC'
        assert validator._score_to_category(500000) == 'HC'

    def test_category_1(self, validator):
        """Test Category 1 threshold."""
        assert validator._score_to_category(64000) == '1'
        assert validator._score_to_category(79999) == '1'

    def test_category_2(self, validator):
        """Test Category 2 threshold."""
        assert validator._score_to_category(32000) == '2'
        assert validator._score_to_category(63999) == '2'

    def test_category_3(self, validator):
        """Test Category 3 threshold."""
        assert validator._score_to_category(16000) == '3'
        assert validator._score_to_category(31999) == '3'

    def test_category_4(self, validator):
        """Test Category 4 threshold."""
        assert validator._score_to_category(8000) == '4'
        assert validator._score_to_category(15999) == '4'

    def test_category_5(self, validator):
        """Test Category 5 (uncategorized)."""
        assert validator._score_to_category(0) == '5'
        assert validator._score_to_category(7999) == '5'
        assert validator._score_to_category(100) == '5'

    def test_boundary_values(self, validator):
        """Test exact boundary values."""
        # Just below HC
        assert validator._score_to_category(79999) == '1'
        # Exactly HC
        assert validator._score_to_category(80000) == 'HC'
        # Just below Cat 4
        assert validator._score_to_category(7999) == '5'
        # Exactly Cat 4
        assert validator._score_to_category(8000) == '4'


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_add_error(self):
        """Test adding errors invalidates result."""
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        assert result.valid is True

        result.add_error('test_field', 'Test error message')
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'test_field'
        assert result.errors[0].message == 'Test error message'

    def test_add_warning(self):
        """Test adding warnings doesn't invalidate result."""
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)

        result.add_warning('test_field', 'Test warning')
        assert result.valid is True  # Still valid
        assert len(result.warnings) == 1

    def test_add_error_with_row(self):
        """Test adding error with row index."""
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        result.add_error('field', 'Error at row', row=42)

        assert result.errors[0].row == 42


class TestClimbSearch:
    """Tests for find_climb() method."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        with patch('result_validator.HAS_PANDAS', True):
            v = ResultValidator.__new__(ResultValidator)
            v.output_dir = Path('output')
            return v

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with climb data."""
        return pd.DataFrame({
            'climb_name': ['Mont Ventoux', 'Alpe d\'Huez', 'Col du Tourmalet', 'Fish Hill'],
            'total_length_km': [21.0, 13.8, 17.1, 3.0],
            'total_elevation_gain_m': [1610, 1120, 1268, 200],
            'avg_gradient_percent': [7.5, 8.1, 7.4, 6.7],
            'fiets_score': [95000, 72000, 68000, 15000],
            'regions': ['Provence', 'Alps', 'Pyrenees', 'Worcestershire']
        })

    def test_exact_match(self, validator, sample_df):
        """Test exact name match."""
        match = validator.find_climb(sample_df, 'Mont Ventoux')
        assert match.found is True
        assert match.match_type == 'exact'
        assert match.matched_name == 'Mont Ventoux'

    def test_exact_match_case_insensitive(self, validator, sample_df):
        """Test exact match is case insensitive."""
        match = validator.find_climb(sample_df, 'MONT VENTOUX')
        assert match.found is True
        assert match.match_type == 'exact'

    def test_partial_match(self, validator, sample_df):
        """Test partial name match (contains)."""
        match = validator.find_climb(sample_df, 'Ventoux')
        assert match.found is True
        assert match.match_type == 'partial'
        assert match.matched_name == 'Mont Ventoux'

    def test_alt_name_match(self, validator, sample_df):
        """Test matching via alternative names."""
        match = validator.find_climb(
            sample_df,
            'The Giant of Provence',  # Won't match
            alt_names=['Mont Ventoux']  # Will match
        )
        assert match.found is True
        assert match.matched_name == 'Mont Ventoux'

    def test_no_match(self, validator, sample_df):
        """Test no match found."""
        match = validator.find_climb(sample_df, 'Non Existent Climb')
        assert match.found is False
        assert match.match_type == 'none'
        assert match.data is None

    def test_match_returns_data(self, validator, sample_df):
        """Test that match includes row data."""
        match = validator.find_climb(sample_df, 'Fish Hill')
        assert match.found is True
        assert match.data['total_length_km'] == 3.0
        assert match.data['total_elevation_gain_m'] == 200

    def test_missing_climb_name_column(self, validator):
        """Test handling of DataFrame without climb_name column."""
        df = pd.DataFrame({'other_column': [1, 2, 3]})
        match = validator.find_climb(df, 'Test')
        assert match.found is False


class TestCrossRegionClimb:
    """Tests for find_cross_region_climb() method."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        with patch('result_validator.HAS_PANDAS', True):
            v = ResultValidator.__new__(ResultValidator)
            v.output_dir = Path('output')
            return v

    @pytest.fixture
    def cross_region_df(self):
        """Create DataFrame with cross-region climb."""
        return pd.DataFrame({
            'climb_name': ['Cold Overton Road', 'US Route 95'],
            'regions': ['Rutland, Leicestershire', 'Oregon, Idaho'],
            'osm_way_ids': ['123,456,789', '111,222,333,444'],
            'total_length_km': [2.0, 20.0],
            'total_elevation_gain_m': [80, 500]
        })

    def test_regions_verified(self, validator, cross_region_df):
        """Test that cross-region climb has regions verified."""
        match, regions_ok = validator.find_cross_region_climb(
            cross_region_df,
            'Cold Overton Road',
            expected_regions=['Rutland', 'Leicestershire']
        )
        assert match.found is True
        assert regions_ok is True

    def test_regions_not_all_present(self, validator, cross_region_df):
        """Test when not all expected regions are present."""
        match, regions_ok = validator.find_cross_region_climb(
            cross_region_df,
            'Cold Overton Road',
            expected_regions=['Rutland', 'Nottinghamshire']  # Wrong region
        )
        assert match.found is True
        assert regions_ok is False

    def test_multiple_way_ids(self, validator, cross_region_df):
        """Test climb with multiple way IDs (merged)."""
        match, regions_ok = validator.find_cross_region_climb(
            cross_region_df,
            'US Route 95',
            expected_regions=['Oregon', 'Idaho']
        )
        assert match.found is True
        assert regions_ok is True
        # Multiple way IDs indicate merged segment
        assert ',' in match.data['osm_way_ids']

    def test_climb_not_found(self, validator, cross_region_df):
        """Test when climb is not found."""
        match, regions_ok = validator.find_cross_region_climb(
            cross_region_df,
            'Non Existent',
            expected_regions=['Somewhere']
        )
        assert match.found is False
        assert regions_ok is False


class TestGetClimbMetrics:
    """Tests for get_climb_metrics() method."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        with patch('result_validator.HAS_PANDAS', True):
            v = ResultValidator.__new__(ResultValidator)
            return v

    def test_extract_metrics(self, validator):
        """Test metric extraction from match."""
        match = ClimbMatch(
            found=True,
            matched_name='Test Climb',
            data={
                'total_length_km': 5.0,
                'total_elevation_gain_m': 300,
                'avg_gradient_percent': 6.0,
                'max_gradient_percent': 12.0,
                'fiets_score': 25000,
                'climb_category': '3'
            }
        )

        metrics = validator.get_climb_metrics(match)
        assert metrics['distance_km'] == 5.0
        assert metrics['elev_gain_m'] == 300
        assert metrics['avg_grade'] == 6.0
        assert metrics['max_grade'] == 12.0
        assert metrics['fiets_score'] == 25000
        assert metrics['category'] == '3'

    def test_not_found_match(self, validator):
        """Test metrics for not-found match."""
        match = ClimbMatch(found=False)
        metrics = validator.get_climb_metrics(match)
        assert metrics == {}

    def test_missing_data(self, validator):
        """Test handling of missing data fields."""
        match = ClimbMatch(
            found=True,
            data={'total_length_km': 5.0}  # Only partial data
        )
        metrics = validator.get_climb_metrics(match)
        assert metrics['distance_km'] == 5.0
        assert metrics['elev_gain_m'] == 0  # Default to 0


class TestCoordinateValidation:
    """Tests for coordinate validation logic."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        with patch('result_validator.HAS_PANDAS', True):
            v = ResultValidator.__new__(ResultValidator)
            return v

    def test_valid_coordinates(self, validator):
        """Test valid coordinate ranges."""
        df = pd.DataFrame({
            'start_lat': [45.5, -33.8, 0.0],
            'start_lon': [-122.6, 151.2, 0.0],
            'end_lat': [45.6, -33.7, 1.0],
            'end_lon': [-122.5, 151.3, 1.0]
        })
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        validator._validate_coordinates(df, result)
        assert result.coordinate_check is True

    def test_invalid_latitude(self, validator):
        """Test invalid latitude detection."""
        df = pd.DataFrame({
            'start_lat': [45.5, 95.0],  # 95 is invalid
            'start_lon': [-122.6, -122.5],
            'end_lat': [45.6, 45.7],
            'end_lon': [-122.5, -122.4]
        })
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        validator._validate_coordinates(df, result)
        assert result.coordinate_check is False
        assert any('invalid latitude' in e.message for e in result.errors)

    def test_invalid_longitude(self, validator):
        """Test invalid longitude detection."""
        df = pd.DataFrame({
            'start_lat': [45.5, 45.6],
            'start_lon': [-122.6, -200.0],  # -200 is invalid
            'end_lat': [45.6, 45.7],
            'end_lon': [-122.5, -122.4]
        })
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        validator._validate_coordinates(df, result)
        assert result.coordinate_check is False
        assert any('invalid longitude' in e.message for e in result.errors)

    def test_null_coordinates_warning(self, validator):
        """Test null coordinates generate warnings."""
        df = pd.DataFrame({
            'start_lat': [45.5, None],
            'start_lon': [-122.6, -122.5],
            'end_lat': [45.6, 45.7],
            'end_lon': [-122.5, -122.4]
        })
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        validator._validate_coordinates(df, result)
        assert any('missing' in w.message for w in result.warnings)


class TestSortOrderValidation:
    """Tests for sort order validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        with patch('result_validator.HAS_PANDAS', True):
            v = ResultValidator.__new__(ResultValidator)
            return v

    def test_correctly_sorted(self, validator):
        """Test properly sorted scores."""
        df = pd.DataFrame({
            'fiets_score': [100000, 80000, 50000, 20000]
        })
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        validator._validate_sort_order(df, result)
        assert result.sort_check is True

    def test_incorrectly_sorted(self, validator):
        """Test incorrectly sorted scores."""
        df = pd.DataFrame({
            'fiets_score': [50000, 80000, 100000, 20000]  # Not descending
        })
        result = ValidationResult(file_path=Path("test.xlsx"), valid=True)
        validator._validate_sort_order(df, result)
        assert result.sort_check is False
        assert any('not sorted' in w.message for w in result.warnings)


class TestFormatValidationSummary:
    """Tests for format_validation_summary() function."""

    def test_format_valid_result(self):
        """Test formatting valid result."""
        result = ValidationResult(
            file_path=Path("test.xlsx"),
            valid=True,
            total_climbs=100
        )
        output = format_validation_summary(result)
        assert "test.xlsx" in output
        assert "100" in output
        assert "YES" in output
        assert "PASS" in output

    def test_format_invalid_result(self):
        """Test formatting invalid result."""
        result = ValidationResult(
            file_path=Path("test.xlsx"),
            valid=False,
            total_climbs=50
        )
        result.errors.append(ValidationError('test', 'Test error'))
        result.column_check = False

        output = format_validation_summary(result)
        assert "NO" in output
        assert "FAIL" in output
        assert "Test error" in output

    def test_format_with_warnings(self):
        """Test formatting with warnings."""
        result = ValidationResult(
            file_path=Path("test.xlsx"),
            valid=True,
            total_climbs=50
        )
        result.warnings.append(ValidationError('field', 'Warning message'))

        output = format_validation_summary(result)
        assert "Warning message" in output


class TestCategoryThresholds:
    """Tests for CATEGORY_THRESHOLDS constant."""

    def test_thresholds_ordered(self):
        """Test that thresholds are in descending order."""
        values = [
            CATEGORY_THRESHOLDS['HC'],
            CATEGORY_THRESHOLDS['1'],
            CATEGORY_THRESHOLDS['2'],
            CATEGORY_THRESHOLDS['3'],
            CATEGORY_THRESHOLDS['4'],
            CATEGORY_THRESHOLDS['5'],
        ]
        assert values == sorted(values, reverse=True)

    def test_threshold_values(self):
        """Test specific threshold values."""
        assert CATEGORY_THRESHOLDS['HC'] == 80000
        assert CATEGORY_THRESHOLDS['1'] == 64000
        assert CATEGORY_THRESHOLDS['2'] == 32000
        assert CATEGORY_THRESHOLDS['3'] == 16000
        assert CATEGORY_THRESHOLDS['4'] == 8000
        assert CATEGORY_THRESHOLDS['5'] == 0


class TestRequiredColumns:
    """Tests for REQUIRED_COLUMNS constant."""

    def test_column_count(self):
        """Test that we have expected number of columns."""
        assert len(REQUIRED_COLUMNS) == 26

    def test_essential_columns_present(self):
        """Test that essential columns are in the list."""
        essential = [
            'climb_name', 'fiets_score', 'climb_category',
            'total_length_km', 'total_elevation_gain_m',
            'avg_gradient_percent', 'start_lat', 'start_lon'
        ]
        for col in essential:
            assert col in REQUIRED_COLUMNS
