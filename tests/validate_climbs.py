#!/usr/bin/env python3
"""
Climb validation script.

Validates climb-analyzer output against external data sources
(Strava, PJAMM Cycling) to verify accuracy of climb metrics.
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from strava_scraper import StravaScraper, StravaSegmentData, validate_against_strava
from pjamm_scraper import PjammScraper, PjammClimbData, get_pjamm_climb, KNOWN_PJAMM_CLIMBS

logger = logging.getLogger(__name__)

# Unit conversion constants
FEET_TO_METERS = 0.3048
MILES_TO_KM = 1.60934

# Known Strava segment IDs for validation
KNOWN_STRAVA_SEGMENTS = {
    "mauna kea access road": "612474",
    "jvari pass": "8876890",
    "fish hill": "628456",
    "bear mountain": "628459",
    "caesars head": "628123",
}

# Known climb reference data from authoritative sources (ClimbByBike, PJAMM, etc.)
# These are used when scrapers can't fetch live data due to JavaScript rendering
KNOWN_CLIMB_REFERENCE = {
    # Hawaii - Paved Roads
    "mauna kea access road": {
        "source": "ClimbByBike",
        "distance_km": 23.4,
        "elev_gain_m": 2188,
        "avg_grade": 9.4,
    },
    "crater road": {
        "source": "PJAMM",
        "distance_km": 58.0,  # Full 36mi from sea level
        "elev_gain_m": 3021,
        "avg_grade": 5.4,
    },
    "waipoli road": {
        "source": "PJAMM",
        "distance_km": 20.4,  # 12.7 miles
        "elev_gain_m": 1678,  # 5504 ft
        "avg_grade": 7.9,
    },
    "mauna loa road": {
        "source": "PJAMM",
        "distance_km": 18.8,  # 11.7 miles
        "elev_gain_m": 1268,  # ~4160 ft
        "avg_grade": 6.8,
    },
    "stainback highway": {
        "source": "PJAMM",
        "distance_km": 27.0,
        "elev_gain_m": 1350,
        "avg_grade": 5.0,
    },
    "hana highway": {
        "source": "PJAMM",
        "distance_km": 28.0,
        "elev_gain_m": 1320,
        "avg_grade": 4.7,
    },
    # Colorado - Paved Roads
    "independence pass": {
        "source": "PJAMM",
        "distance_km": 25.4,  # 15.8 miles from Aspen
        "elev_gain_m": 1258,  # 4128 ft
        "avg_grade": 4.8,
    },
    "mount evans": {
        "source": "PJAMM",
        "distance_km": 43.5,  # 27 miles
        "elev_gain_m": 2054,  # 6740 ft
        "avg_grade": 4.5,
    },
    "mt evans": {
        "source": "PJAMM",
        "distance_km": 43.5,
        "elev_gain_m": 2054,
        "avg_grade": 4.5,
    },
    "trail ridge road": {
        "source": "PJAMM",
        "distance_km": 77.0,  # ~48 miles
        "elev_gain_m": 2012,  # ~6600 ft
        "avg_grade": 2.6,
    },
    "pikes peak": {
        "source": "PJAMM",
        "distance_km": 31.0,  # 19.3 miles
        "elev_gain_m": 2194,  # 7200 ft
        "avg_grade": 7.1,
    },
    # European Classics
    "mont ventoux": {
        "source": "ClimbByBike",
        "distance_km": 21.5,
        "elev_gain_m": 1617,
        "avg_grade": 7.5,
    },
    "alpe d'huez": {
        "source": "ClimbByBike",
        "distance_km": 13.8,
        "elev_gain_m": 1071,
        "avg_grade": 7.9,
    },
    "col du tourmalet": {
        "source": "ClimbByBike",
        "distance_km": 17.1,
        "elev_gain_m": 1268,
        "avg_grade": 7.4,
    },
    # Trails (note: may differ due to one-way vs round-trip)
    "haleakala": {
        "source": "ClimbByBike",
        "distance_km": 60.0,
        "elev_gain_m": 2970,
        "avg_grade": 5.0,
    },
    "kalalau trail": {
        "source": "AllTrails",
        "distance_km": 17.7,  # Full out-and-back
        "elev_gain_m": 1830,
        "avg_grade": 10.3,
    },
    "ainapo trail": {
        "source": "AllTrails",
        "distance_km": 14.5,
        "elev_gain_m": 2347,
        "avg_grade": 16.2,
    },
}


@dataclass
class ClimbData:
    """Normalized climb data from analyzer output."""
    name: str
    city: str
    state: str
    country: str
    lat: float
    lon: float
    fiets_score: float
    distance_km: float
    elev_gain_m: float
    avg_grade: float
    max_grade: float
    category: str
    osm_link: str = ""


@dataclass
class ValidationResult:
    """Result of validating a single climb."""
    climb: ClimbData
    source: str  # 'strava', 'pjamm', or 'none'
    external_name: Optional[str] = None
    external_distance_km: Optional[float] = None
    external_elev_gain_m: Optional[float] = None
    external_avg_grade: Optional[float] = None
    distance_diff_pct: Optional[float] = None
    elev_diff_pct: Optional[float] = None
    grade_diff_pct: Optional[float] = None
    passed: bool = False
    error: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a file."""
    file_path: str
    total_climbs: int
    sampled: int
    tolerance_pct: float
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.source != 'none')

    @property
    def no_match_count(self) -> int:
        return sum(1 for r in self.results if r.source == 'none')

    @property
    def pass_rate(self) -> float:
        matched = len(self.results) - self.no_match_count
        if matched == 0:
            return 0.0
        return self.passed_count / matched


def detect_units(df: pd.DataFrame) -> str:
    """Detect measurement system from column names."""
    if 'Elev Gain (ft)' in df.columns:
        return 'imperial'
    elif 'Elev Gain (m)' in df.columns:
        return 'metric'
    else:
        # Try to infer from column names
        for col in df.columns:
            if '(ft)' in col or '(mi)' in col:
                return 'imperial'
            if '(m)' in col or '(km)' in col:
                return 'metric'
        raise ValueError("Cannot detect measurement units from columns")


def extract_climb_data(row: pd.Series, units: str) -> ClimbData:
    """Extract and normalize climb data from a DataFrame row."""
    if units == 'imperial':
        distance_km = row.get('Length (mi)', 0) * MILES_TO_KM
        elev_gain_m = row.get('Elev Gain (ft)', 0) * FEET_TO_METERS
    else:
        distance_km = row.get('Length (km)', 0)
        elev_gain_m = row.get('Elev Gain (m)', 0)

    return ClimbData(
        name=str(row.get('Street Name', 'Unknown')),
        city=str(row.get('City', '')),
        state=str(row.get('State', '')),
        country=str(row.get('Country', '')),
        lat=float(row.get('Latitude', 0)),
        lon=float(row.get('Longitude', 0)),
        fiets_score=float(row.get('FIETS Score', 0)),
        distance_km=distance_km,
        elev_gain_m=elev_gain_m,
        avg_grade=float(row.get('Avg Grade (%)', 0)),
        max_grade=float(row.get('Max Grade (%)', 0)),
        category=str(row.get('Category', '')),
        osm_link=str(row.get('OSM Link', ''))
    )


def select_validation_sample(
    df: pd.DataFrame,
    count: int,
    min_score: float,
    named_only: bool = True,
    units: str = 'imperial'
) -> List[ClimbData]:
    """
    Select a random sample of high-scoring climbs for validation.

    Args:
        df: DataFrame with climb data
        count: Number of climbs to select
        min_score: Minimum FIETS score
        named_only: Only include named climbs (not "Unnamed Road")

    Returns:
        List of ClimbData objects
    """
    # Filter by score
    filtered = df[df['FIETS Score'] >= min_score].copy()

    # Filter unnamed roads if requested
    if named_only:
        filtered = filtered[~filtered['Street Name'].str.lower().str.contains('unnamed', na=False)]

    # Limit to top climbs (already sorted by FIETS score)
    if len(filtered) > count * 10:
        filtered = filtered.head(count * 10)

    # Random sample
    if len(filtered) <= count:
        sample = filtered
    else:
        sample = filtered.sample(n=count, random_state=42)

    # Convert to ClimbData
    climbs = []
    for _, row in sample.iterrows():
        climbs.append(extract_climb_data(row, units))

    return climbs


def validate_climb_strava(
    climb: ClimbData,
    scraper: StravaScraper,
    tolerance_pct: float
) -> ValidationResult:
    """Validate a climb against Strava data."""
    result = ValidationResult(climb=climb, source='none')

    # Check known segments first
    normalized_name = climb.name.lower().strip()
    segment_id = KNOWN_STRAVA_SEGMENTS.get(normalized_name)

    if not segment_id:
        # Try partial match on known segments
        for known_name, sid in KNOWN_STRAVA_SEGMENTS.items():
            if known_name in normalized_name or normalized_name in known_name:
                segment_id = sid
                break

    if segment_id:
        # Direct segment lookup
        segment = scraper.get_segment(segment_id)
        if segment.is_valid():
            result.source = 'strava'
            result.external_name = segment.name
            result.external_distance_km = segment.distance_km
            result.external_elev_gain_m = segment.elev_gain_m
            result.external_avg_grade = segment.avg_grade

            # Calculate differences
            if segment.distance_km > 0:
                result.distance_diff_pct = abs(climb.distance_km - segment.distance_km) / segment.distance_km * 100
            if segment.elev_gain_m > 0:
                result.elev_diff_pct = abs(climb.elev_gain_m - segment.elev_gain_m) / segment.elev_gain_m * 100
            if segment.avg_grade > 0:
                result.grade_diff_pct = abs(climb.avg_grade - segment.avg_grade) / segment.avg_grade * 100

            # Check if all metrics within tolerance
            diffs = [d for d in [result.distance_diff_pct, result.elev_diff_pct, result.grade_diff_pct] if d is not None]
            result.passed = all(d <= tolerance_pct for d in diffs)
        else:
            result.error = segment.fetch_error
    else:
        # Try location-based search
        segment = scraper.find_matching_segment(
            name=climb.name,
            lat=climb.lat,
            lon=climb.lon,
            distance_km=climb.distance_km,
            elev_gain_m=climb.elev_gain_m
        )
        if segment and segment.is_valid():
            result.source = 'strava'
            result.external_name = segment.name
            result.external_distance_km = segment.distance_km
            result.external_elev_gain_m = segment.elev_gain_m
            result.external_avg_grade = segment.avg_grade

            # Calculate differences
            if segment.distance_km > 0:
                result.distance_diff_pct = abs(climb.distance_km - segment.distance_km) / segment.distance_km * 100
            if segment.elev_gain_m > 0:
                result.elev_diff_pct = abs(climb.elev_gain_m - segment.elev_gain_m) / segment.elev_gain_m * 100
            if segment.avg_grade > 0:
                result.grade_diff_pct = abs(climb.avg_grade - segment.avg_grade) / segment.avg_grade * 100

            diffs = [d for d in [result.distance_diff_pct, result.elev_diff_pct, result.grade_diff_pct] if d is not None]
            result.passed = all(d <= tolerance_pct for d in diffs)

    return result


def validate_climb_pjamm(
    climb: ClimbData,
    scraper: PjammScraper,
    tolerance_pct: float
) -> ValidationResult:
    """Validate a climb against PJAMM data."""
    result = ValidationResult(climb=climb, source='none')

    # Check known PJAMM climbs
    normalized_name = climb.name.lower().strip()

    pjamm_data = None
    if normalized_name in KNOWN_PJAMM_CLIMBS:
        climb_id = KNOWN_PJAMM_CLIMBS[normalized_name]
        url = f"https://pjammcycling.com/climb/{climb_id}"
        pjamm_data = scraper.get_climb_by_url(url)
    else:
        # Try search
        results = scraper.search_by_name(climb.name)
        if results:
            pjamm_data = results[0]

    if pjamm_data and pjamm_data.is_valid():
        result.source = 'pjamm'
        result.external_name = pjamm_data.name
        result.external_distance_km = pjamm_data.distance_km
        result.external_elev_gain_m = pjamm_data.elev_gain_m
        result.external_avg_grade = pjamm_data.avg_grade

        # Calculate differences
        if pjamm_data.distance_km > 0:
            result.distance_diff_pct = abs(climb.distance_km - pjamm_data.distance_km) / pjamm_data.distance_km * 100
        if pjamm_data.elev_gain_m > 0:
            result.elev_diff_pct = abs(climb.elev_gain_m - pjamm_data.elev_gain_m) / pjamm_data.elev_gain_m * 100
        if pjamm_data.avg_grade > 0:
            result.grade_diff_pct = abs(climb.avg_grade - pjamm_data.avg_grade) / pjamm_data.avg_grade * 100

        diffs = [d for d in [result.distance_diff_pct, result.elev_diff_pct, result.grade_diff_pct] if d is not None]
        result.passed = all(d <= tolerance_pct for d in diffs)

    return result


def validate_climb_reference(climb: ClimbData, tolerance_pct: float) -> ValidationResult:
    """Validate a climb against known reference data."""
    result = ValidationResult(climb=climb, source='none')

    normalized_name = climb.name.lower().strip()

    # Check exact match
    ref_data = KNOWN_CLIMB_REFERENCE.get(normalized_name)

    # Try partial match
    if not ref_data:
        for known_name, data in KNOWN_CLIMB_REFERENCE.items():
            if known_name in normalized_name or normalized_name in known_name:
                ref_data = data
                break

    if ref_data:
        result.source = f"reference ({ref_data['source']})"
        result.external_name = climb.name
        result.external_distance_km = ref_data['distance_km']
        result.external_elev_gain_m = ref_data['elev_gain_m']
        result.external_avg_grade = ref_data['avg_grade']

        # Calculate differences
        if ref_data['distance_km'] > 0:
            result.distance_diff_pct = abs(climb.distance_km - ref_data['distance_km']) / ref_data['distance_km'] * 100
        if ref_data['elev_gain_m'] > 0:
            result.elev_diff_pct = abs(climb.elev_gain_m - ref_data['elev_gain_m']) / ref_data['elev_gain_m'] * 100
        if ref_data['avg_grade'] > 0:
            result.grade_diff_pct = abs(climb.avg_grade - ref_data['avg_grade']) / ref_data['avg_grade'] * 100

        diffs = [d for d in [result.distance_diff_pct, result.elev_diff_pct, result.grade_diff_pct] if d is not None]
        result.passed = all(d <= tolerance_pct for d in diffs)

    return result


def validate_climb(
    climb: ClimbData,
    strava_scraper: StravaScraper,
    pjamm_scraper: PjammScraper,
    tolerance_pct: float,
    sources: List[str]
) -> ValidationResult:
    """
    Validate a climb against available external sources.

    Tries sources in order: known reference data, Strava, then PJAMM.
    """
    result = ValidationResult(climb=climb, source='none')

    # Try known reference data first (always available, no network needed)
    result = validate_climb_reference(climb, tolerance_pct)
    if result.source != 'none':
        return result

    # Try Strava
    if 'strava' in sources:
        result = validate_climb_strava(climb, strava_scraper, tolerance_pct)
        if result.source != 'none':
            return result

    # Fall back to PJAMM
    if 'pjamm' in sources:
        result = validate_climb_pjamm(climb, pjamm_scraper, tolerance_pct)

    return result


def format_report_console(report: ValidationReport) -> str:
    """Format validation report for console output."""
    lines = []
    lines.append("=" * 80)
    lines.append("CLIMB VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"File: {Path(report.file_path).name}")
    lines.append(f"Total climbs: {report.total_climbs:,}")
    lines.append(f"Sampled: {report.sampled}")
    lines.append(f"Tolerance: {report.tolerance_pct}%")
    lines.append("")

    for i, result in enumerate(report.results, 1):
        lines.append("-" * 80)
        lines.append(f"{i}. {result.climb.name}")
        lines.append(f"   Location: {result.climb.city}, {result.climb.state}")
        lines.append(f"   FIETS: {result.climb.fiets_score:,.1f} | Category: {result.climb.category}")
        lines.append(f"   Analyzer: {result.climb.elev_gain_m:.0f}m / {result.climb.distance_km:.2f}km / {result.climb.avg_grade:.1f}%")

        if result.source == 'none':
            lines.append(f"   Source: No match found")
            if result.error:
                lines.append(f"   Error: {result.error}")
        else:
            source_name = result.source.upper()
            lines.append(f"   {source_name}: {result.external_name}")
            lines.append(f"   {source_name} data: {result.external_elev_gain_m:.0f}m / {result.external_distance_km:.2f}km / {result.external_avg_grade:.1f}%")

            # Format differences
            diffs = []
            if result.distance_diff_pct is not None:
                status = "[OK]" if result.distance_diff_pct <= report.tolerance_pct else "[DIFF]"
                diffs.append(f"Distance: {result.distance_diff_pct:.1f}% {status}")
            if result.elev_diff_pct is not None:
                status = "[OK]" if result.elev_diff_pct <= report.tolerance_pct else "[DIFF]"
                diffs.append(f"Elevation: {result.elev_diff_pct:.1f}% {status}")
            if result.grade_diff_pct is not None:
                status = "[OK]" if result.grade_diff_pct <= report.tolerance_pct else "[DIFF]"
                diffs.append(f"Grade: {result.grade_diff_pct:.1f}% {status}")

            lines.append(f"   Comparison: {' | '.join(diffs)}")

            result_str = "PASS" if result.passed else "FAIL"
            lines.append(f"   Result: {result_str}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    matched = report.sampled - report.no_match_count
    lines.append(f"Matched: {matched}/{report.sampled}")
    if matched > 0:
        lines.append(f"  Passed: {report.passed_count} ({report.pass_rate * 100:.0f}%)")
        lines.append(f"  Failed: {report.failed_count}")
    lines.append(f"  No match: {report.no_match_count}")

    # Calculate average differences
    dist_diffs = [r.distance_diff_pct for r in report.results if r.distance_diff_pct is not None]
    elev_diffs = [r.elev_diff_pct for r in report.results if r.elev_diff_pct is not None]
    grade_diffs = [r.grade_diff_pct for r in report.results if r.grade_diff_pct is not None]

    if dist_diffs or elev_diffs or grade_diffs:
        lines.append("")
        lines.append("Average differences:")
        if dist_diffs:
            lines.append(f"  Distance: {sum(dist_diffs)/len(dist_diffs):.1f}%")
        if elev_diffs:
            lines.append(f"  Elevation: {sum(elev_diffs)/len(elev_diffs):.1f}%")
        if grade_diffs:
            lines.append(f"  Grade: {sum(grade_diffs)/len(grade_diffs):.1f}%")

    lines.append("=" * 80)

    return '\n'.join(lines)


def format_report_json(report: ValidationReport) -> str:
    """Format validation report as JSON."""
    data = {
        'file': report.file_path,
        'total_climbs': report.total_climbs,
        'sampled': report.sampled,
        'tolerance_pct': report.tolerance_pct,
        'results': [asdict(r) for r in report.results],
        'summary': {
            'passed': report.passed_count,
            'failed': report.failed_count,
            'no_match': report.no_match_count,
            'pass_rate': report.pass_rate
        }
    }
    return json.dumps(data, indent=2, default=str)


def format_report_csv(report: ValidationReport) -> str:
    """Format validation report as CSV."""
    lines = ['climb_name,fiets_score,analyzer_dist_km,analyzer_elev_m,analyzer_grade,source,external_dist_km,external_elev_m,external_grade,dist_diff_pct,elev_diff_pct,grade_diff_pct,passed']

    for r in report.results:
        lines.append(','.join([
            f'"{r.climb.name}"',
            f'{r.climb.fiets_score:.1f}',
            f'{r.climb.distance_km:.2f}',
            f'{r.climb.elev_gain_m:.0f}',
            f'{r.climb.avg_grade:.1f}',
            r.source,
            f'{r.external_distance_km:.2f}' if r.external_distance_km else '',
            f'{r.external_elev_gain_m:.0f}' if r.external_elev_gain_m else '',
            f'{r.external_avg_grade:.1f}' if r.external_avg_grade else '',
            f'{r.distance_diff_pct:.1f}' if r.distance_diff_pct else '',
            f'{r.elev_diff_pct:.1f}' if r.elev_diff_pct else '',
            f'{r.grade_diff_pct:.1f}' if r.grade_diff_pct else '',
            str(r.passed).lower()
        ]))

    return '\n'.join(lines)


def find_output_file(input_path: str) -> Path:
    """Find output file from input path or region name."""
    path = Path(input_path)

    # Direct file path
    if path.exists() and path.suffix == '.xlsx':
        return path

    # Check output directory for region match
    output_dir = Path('output')
    if output_dir.exists():
        # Normalize region name
        region = input_path.replace(' ', '_').replace('-', '_')

        for xlsx in output_dir.glob('*.xlsx'):
            if region.lower() in xlsx.stem.lower():
                return xlsx

    raise FileNotFoundError(f"Could not find output file for: {input_path}")


def validate_output_file(
    xlsx_path: Path,
    sample_count: int = 5,
    min_fiets_score: float = 5000,
    tolerance_pct: float = 15.0,
    sources: List[str] = None,
    named_only: bool = True
) -> ValidationReport:
    """
    Validate an output file against external sources.

    Args:
        xlsx_path: Path to Excel output file
        sample_count: Number of climbs to validate
        min_fiets_score: Minimum FIETS score for sampling
        tolerance_pct: Acceptable difference percentage
        sources: List of sources to use ('strava', 'pjamm')
        named_only: Only validate named climbs

    Returns:
        ValidationReport with results
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required. Install with: pip install pandas openpyxl")

    sources = sources or ['strava', 'pjamm']

    print(f"Loading {xlsx_path.name}...")
    df = pd.read_excel(xlsx_path)
    total_climbs = len(df)

    print(f"Total climbs: {total_climbs:,}")

    # Detect units
    units = detect_units(df)
    print(f"Units: {units}")

    # Select sample
    print(f"Selecting {sample_count} climbs with FIETS >= {min_fiets_score}...")
    sample = select_validation_sample(df, sample_count, min_fiets_score, named_only, units)

    if not sample:
        print("No climbs match the criteria.")
        return ValidationReport(
            file_path=str(xlsx_path),
            total_climbs=total_climbs,
            sampled=0,
            tolerance_pct=tolerance_pct
        )

    print(f"Selected {len(sample)} climbs for validation\n")

    # Initialize scrapers
    strava_scraper = StravaScraper()
    pjamm_scraper = PjammScraper()

    # Validate each climb
    report = ValidationReport(
        file_path=str(xlsx_path),
        total_climbs=total_climbs,
        sampled=len(sample),
        tolerance_pct=tolerance_pct
    )

    for i, climb in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] Validating: {climb.name}...", end=' ', flush=True)
        result = validate_climb(climb, strava_scraper, pjamm_scraper, tolerance_pct, sources)
        report.results.append(result)

        if result.source == 'none':
            print("No match")
        elif result.passed:
            print(f"PASS ({result.source})")
        else:
            print(f"FAIL ({result.source})")

    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate climb-analyzer output against external sources (Strava, PJAMM)'
    )
    parser.add_argument(
        'input',
        help='Excel file path or region name (e.g., output/Hawaii_*.xlsx or "Hawaii")'
    )
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=5,
        help='Number of climbs to validate (default: 5)'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=5000,
        help='Minimum FIETS score for sample selection (default: 5000)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=15.0,
        help='Acceptable difference percentage (default: 15.0)'
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        default=['strava', 'pjamm'],
        choices=['strava', 'pjamm'],
        help='External sources to use (default: strava pjamm)'
    )
    parser.add_argument(
        '--format',
        choices=['console', 'json', 'csv'],
        default='console',
        help='Output format (default: console)'
    )
    parser.add_argument(
        '--include-unnamed',
        action='store_true',
        help='Include unnamed roads in sample'
    )
    parser.add_argument(
        '--save',
        type=Path,
        help='Save report to file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        # Find the output file
        xlsx_path = find_output_file(args.input)
        print(f"Using file: {xlsx_path}\n")

        # Run validation
        report = validate_output_file(
            xlsx_path=xlsx_path,
            sample_count=args.count,
            min_fiets_score=args.min_score,
            tolerance_pct=args.tolerance,
            sources=args.sources,
            named_only=not args.include_unnamed
        )

        # Format output
        if args.format == 'json':
            output = format_report_json(report)
        elif args.format == 'csv':
            output = format_report_csv(report)
        else:
            output = format_report_console(report)

        print()
        print(output)

        # Save if requested
        if args.save:
            with open(args.save, 'w') as f:
                f.write(output)
            print(f"\nReport saved to: {args.save}")

        # Return exit code based on results
        if report.pass_rate >= 0.8:
            return 0
        elif report.pass_rate >= 0.5:
            return 1
        else:
            return 2

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
