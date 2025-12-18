#!/usr/bin/env python3
"""
Result validator for climb analyzer XLSX output files.

Validates output structure, data integrity, and specific climb presence.
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from glob import glob

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


# Expected columns in output XLSX files (26 columns)
REQUIRED_COLUMNS = [
    'rank',
    'climb_name',
    'osm_way_ids',
    'fiets_score',
    'climb_category',
    'total_length_km',
    'total_elevation_gain_m',
    'avg_gradient_percent',
    'max_gradient_percent',
    'start_lat',
    'start_lon',
    'end_lat',
    'end_lon',
    'surface_type',
    'road_classification',
    'elevation_profile',
    'gradient_profile',
    'min_elevation_m',
    'max_elevation_m',
    'steepest_100m_gradient',
    'steepest_500m_gradient',
    'steepest_1km_gradient',
    'osm_link',
    'google_maps_link',
    'strava_link',
    'regions'
]

# Fiets score thresholds for climb categories
CATEGORY_THRESHOLDS = {
    'HC': 80000,    # Hors Catégorie
    '1': 64000,     # Category 1
    '2': 32000,     # Category 2
    '3': 16000,     # Category 3
    '4': 8000,      # Category 4
    '5': 0          # Uncategorized / minor
}


@dataclass
class ValidationError:
    """A single validation error."""
    field: str
    message: str
    severity: str = 'error'  # 'error', 'warning', 'info'
    row: Optional[int] = None


@dataclass
class ClimbMatch:
    """Result of searching for a specific climb."""
    found: bool
    row_index: Optional[int] = None
    matched_name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    match_type: str = 'none'  # 'exact', 'partial', 'alt_name', 'none'


@dataclass
class ValidationResult:
    """Complete validation result for an output file."""
    file_path: Path
    valid: bool
    total_climbs: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    column_check: bool = True
    sort_check: bool = True
    coordinate_check: bool = True
    category_check: bool = True

    def add_error(self, field: str, message: str, row: int = None):
        self.errors.append(ValidationError(field, message, 'error', row))
        self.valid = False

    def add_warning(self, field: str, message: str, row: int = None):
        self.warnings.append(ValidationError(field, message, 'warning', row))


class ResultValidator:
    """Validator for climb analyzer output files."""

    def __init__(self, output_dir: str = 'output'):
        if not HAS_PANDAS:
            raise ImportError(
                "ResultValidator requires 'pandas' and 'openpyxl'. "
                "Install with: pip install pandas openpyxl"
            )
        self.output_dir = Path(output_dir)

    def find_output_file(self, region: str, surface: str = 'all') -> Optional[Path]:
        """
        Find the most recent output file for a region.

        Args:
            region: Region name to search for
            surface: Surface filter type (default 'all')

        Returns:
            Path to the most recent matching file, or None
        """
        # Normalize region name for file matching
        normalized = region.lower().replace(' ', '_').replace(',', '_')
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)

        # Build search patterns
        patterns = [
            f"{normalized}*{surface}*.xlsx",
            f"*{normalized}*{surface}*.xlsx",
            f"{normalized}*.xlsx",
            f"*{normalized}*.xlsx",
        ]

        for pattern in patterns:
            matches = list(self.output_dir.glob(pattern))
            if matches:
                # Return most recently modified
                return max(matches, key=lambda p: p.stat().st_mtime)

        # Try case-insensitive search
        all_files = list(self.output_dir.glob("*.xlsx"))
        for f in sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True):
            if normalized in f.name.lower():
                return f

        return None

    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate an output XLSX file.

        Args:
            file_path: Path to the XLSX file

        Returns:
            ValidationResult with detailed findings
        """
        result = ValidationResult(file_path=file_path, valid=True)

        if not file_path.exists():
            result.add_error('file', f'File not found: {file_path}')
            return result

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            result.add_error('file', f'Failed to read Excel file: {e}')
            return result

        result.total_climbs = len(df)

        # Check columns
        self._validate_columns(df, result)

        # Check data integrity
        if result.column_check:
            self._validate_coordinates(df, result)
            self._validate_scores(df, result)
            self._validate_categories(df, result)
            self._validate_sort_order(df, result)
            self._validate_profiles(df, result)
            self._validate_links(df, result)

        return result

    def _validate_columns(self, df: pd.DataFrame, result: ValidationResult):
        """Check that all required columns are present."""
        missing = []
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                missing.append(col)

        if missing:
            result.column_check = False
            result.add_error('columns', f'Missing columns: {", ".join(missing)}')

        # Check for extra columns (warning only)
        extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        if extra:
            result.add_warning('columns', f'Extra columns: {", ".join(extra)}')

    def _validate_coordinates(self, df: pd.DataFrame, result: ValidationResult):
        """Validate coordinate values."""
        coord_cols = ['start_lat', 'start_lon', 'end_lat', 'end_lon']

        for col in coord_cols:
            if col not in df.columns:
                continue

            # Check for null values
            null_count = df[col].isna().sum()
            if null_count > 0:
                result.add_warning(col, f'{null_count} rows with missing {col}')

            # Check value ranges
            if 'lat' in col:
                invalid = df[(df[col] < -90) | (df[col] > 90)][col]
                if len(invalid) > 0:
                    result.add_error(col, f'{len(invalid)} rows with invalid latitude')
                    result.coordinate_check = False
            else:
                invalid = df[(df[col] < -180) | (df[col] > 180)][col]
                if len(invalid) > 0:
                    result.add_error(col, f'{len(invalid)} rows with invalid longitude')
                    result.coordinate_check = False

    def _validate_scores(self, df: pd.DataFrame, result: ValidationResult):
        """Validate fiets scores."""
        if 'fiets_score' not in df.columns:
            return

        # Check for negative scores
        negative = df[df['fiets_score'] < 0]
        if len(negative) > 0:
            result.add_error('fiets_score', f'{len(negative)} rows with negative scores')

        # Check for unreasonably high scores
        very_high = df[df['fiets_score'] > 500000]
        if len(very_high) > 0:
            result.add_warning('fiets_score', f'{len(very_high)} rows with very high scores (>500k)')

    def _validate_categories(self, df: pd.DataFrame, result: ValidationResult):
        """Validate climb categories match scores."""
        if 'fiets_score' not in df.columns or 'climb_category' not in df.columns:
            return

        mismatch_count = 0
        for idx, row in df.iterrows():
            score = row['fiets_score']
            category = str(row['climb_category']).upper()

            expected_cat = self._score_to_category(score)
            if expected_cat.upper() != category:
                mismatch_count += 1
                if mismatch_count <= 5:  # Only report first 5
                    result.add_warning(
                        'climb_category',
                        f'Row {idx}: score {score:.0f} should be {expected_cat}, got {category}',
                        row=idx
                    )

        if mismatch_count > 5:
            result.add_warning(
                'climb_category',
                f'... and {mismatch_count - 5} more category mismatches'
            )

        if mismatch_count > len(df) * 0.1:  # More than 10% mismatch
            result.category_check = False

    def _score_to_category(self, score: float) -> str:
        """Convert fiets score to climb category."""
        if score >= CATEGORY_THRESHOLDS['HC']:
            return 'HC'
        elif score >= CATEGORY_THRESHOLDS['1']:
            return '1'
        elif score >= CATEGORY_THRESHOLDS['2']:
            return '2'
        elif score >= CATEGORY_THRESHOLDS['3']:
            return '3'
        elif score >= CATEGORY_THRESHOLDS['4']:
            return '4'
        else:
            return '5'

    def _validate_sort_order(self, df: pd.DataFrame, result: ValidationResult):
        """Validate that climbs are sorted by fiets score descending."""
        if 'fiets_score' not in df.columns:
            return

        scores = df['fiets_score'].tolist()
        if scores != sorted(scores, reverse=True):
            result.sort_check = False
            result.add_warning('sort_order', 'Climbs not sorted by fiets_score descending')

    def _validate_profiles(self, df: pd.DataFrame, result: ValidationResult):
        """Validate elevation and gradient profiles."""
        profile_cols = ['elevation_profile', 'gradient_profile']

        for col in profile_cols:
            if col not in df.columns:
                continue

            # Check for empty profiles
            empty = df[df[col].isna() | (df[col] == '') | (df[col] == '[]')]
            if len(empty) > 0:
                result.add_warning(col, f'{len(empty)} rows with empty {col}')

    def _validate_links(self, df: pd.DataFrame, result: ValidationResult):
        """Validate OSM and map links."""
        if 'osm_link' in df.columns:
            # Check OSM links format
            invalid_osm = df[
                df['osm_link'].notna() &
                ~df['osm_link'].str.contains('openstreetmap.org', na=False)
            ]
            if len(invalid_osm) > 0:
                result.add_warning('osm_link', f'{len(invalid_osm)} rows with invalid OSM links')

        if 'google_maps_link' in df.columns:
            invalid_maps = df[
                df['google_maps_link'].notna() &
                ~df['google_maps_link'].str.contains('google.com/maps', na=False)
            ]
            if len(invalid_maps) > 0:
                result.add_warning('google_maps_link', f'{len(invalid_maps)} invalid Google Maps links')

    def find_climb(
        self,
        df: pd.DataFrame,
        name: str,
        alt_names: List[str] = None
    ) -> ClimbMatch:
        """
        Search for a specific climb in the dataframe.

        Args:
            df: DataFrame with climb data
            name: Primary climb name to search
            alt_names: Alternative names to try

        Returns:
            ClimbMatch with search results
        """
        if 'climb_name' not in df.columns:
            return ClimbMatch(found=False)

        all_names = [name] + (alt_names or [])

        for search_name in all_names:
            search_lower = search_name.lower()

            # Exact match
            exact = df[df['climb_name'].str.lower() == search_lower]
            if len(exact) > 0:
                row = exact.iloc[0]
                return ClimbMatch(
                    found=True,
                    row_index=exact.index[0],
                    matched_name=row['climb_name'],
                    data=row.to_dict(),
                    match_type='exact'
                )

            # Partial match (contains)
            partial = df[df['climb_name'].str.lower().str.contains(search_lower, na=False)]
            if len(partial) > 0:
                row = partial.iloc[0]
                return ClimbMatch(
                    found=True,
                    row_index=partial.index[0],
                    matched_name=row['climb_name'],
                    data=row.to_dict(),
                    match_type='partial'
                )

            # Reverse partial (search term contains climb name)
            for idx, row in df.iterrows():
                climb_name = str(row['climb_name']).lower()
                if climb_name in search_lower:
                    return ClimbMatch(
                        found=True,
                        row_index=idx,
                        matched_name=row['climb_name'],
                        data=row.to_dict(),
                        match_type='partial'
                    )

        return ClimbMatch(found=False)

    def find_cross_region_climb(
        self,
        df: pd.DataFrame,
        name: str,
        expected_regions: List[str],
        alt_names: List[str] = None
    ) -> Tuple[ClimbMatch, bool]:
        """
        Find a climb and verify it spans multiple regions.

        Args:
            df: DataFrame with climb data
            name: Climb name to search
            expected_regions: Regions the climb should span
            alt_names: Alternative names

        Returns:
            Tuple of (ClimbMatch, regions_verified)
        """
        match = self.find_climb(df, name, alt_names)

        if not match.found:
            return match, False

        # Check if regions column exists and contains expected regions
        regions_verified = False
        if match.data and 'regions' in match.data:
            regions_str = str(match.data['regions']).lower()
            regions_found = all(
                r.lower() in regions_str for r in expected_regions
            )
            regions_verified = regions_found

        # Also check osm_way_ids for multiple ways (indicates merge)
        if match.data and 'osm_way_ids' in match.data:
            way_ids = str(match.data['osm_way_ids'])
            # Multiple way IDs suggest a merged climb
            way_count = len(way_ids.split(',')) if ',' in way_ids else 1
            if way_count > 1:
                logger.debug(f"Climb {name} has {way_count} way IDs - likely merged")

        return match, regions_verified

    def get_climb_metrics(self, match: ClimbMatch) -> Dict[str, float]:
        """
        Extract numeric metrics from a climb match.

        Args:
            match: ClimbMatch with data

        Returns:
            Dict with distance_km, elev_gain_m, avg_grade
        """
        if not match.found or not match.data:
            return {}

        return {
            'distance_km': float(match.data.get('total_length_km', 0)),
            'elev_gain_m': float(match.data.get('total_elevation_gain_m', 0)),
            'avg_grade': float(match.data.get('avg_gradient_percent', 0)),
            'max_grade': float(match.data.get('max_gradient_percent', 0)),
            'fiets_score': float(match.data.get('fiets_score', 0)),
            'category': str(match.data.get('climb_category', '')),
        }


def format_validation_summary(result: ValidationResult) -> str:
    """Format validation result for display."""
    lines = [
        f"File: {result.file_path.name}",
        f"Total climbs: {result.total_climbs}",
        f"Valid: {'YES' if result.valid else 'NO'}",
        "",
    ]

    if result.errors:
        lines.append("Errors:")
        for err in result.errors:
            lines.append(f"  - [{err.field}] {err.message}")
        lines.append("")

    if result.warnings:
        lines.append("Warnings:")
        for warn in result.warnings[:10]:  # Limit to 10
            lines.append(f"  - [{warn.field}] {warn.message}")
        if len(result.warnings) > 10:
            lines.append(f"  ... and {len(result.warnings) - 10} more warnings")
        lines.append("")

    lines.append("Checks:")
    lines.append(f"  Columns: {'PASS' if result.column_check else 'FAIL'}")
    lines.append(f"  Coordinates: {'PASS' if result.coordinate_check else 'FAIL'}")
    lines.append(f"  Categories: {'PASS' if result.category_check else 'FAIL'}")
    lines.append(f"  Sort order: {'PASS' if result.sort_check else 'FAIL'}")

    return '\n'.join(lines)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python result_validator.py <xlsx_file_or_region>")
        sys.exit(1)

    target = sys.argv[1]
    validator = ResultValidator()

    # Check if it's a file path or region name
    if target.endswith('.xlsx'):
        file_path = Path(target)
    else:
        file_path = validator.find_output_file(target)
        if not file_path:
            print(f"No output file found for region: {target}")
            sys.exit(1)

    print(f"Validating: {file_path}")
    result = validator.validate_file(file_path)
    print(format_validation_summary(result))
