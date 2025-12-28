#!/usr/bin/env python3
"""
Strava segment scraper for third-party climb validation.

Fetches public Strava segment data to validate climb analyzer output
against real-world cycling data.
"""

import re
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin
from datetime import datetime, timedelta
from difflib import SequenceMatcher

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

logger = logging.getLogger(__name__)


@dataclass
class StravaSegmentData:
    """Data extracted from a Strava segment page."""
    segment_id: str
    name: str
    distance_km: float
    elev_gain_m: float
    avg_grade: float
    max_grade: Optional[float] = None
    location: Optional[str] = None
    fetch_error: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if segment data was successfully fetched."""
        return self.fetch_error is None and self.distance_km > 0


class StravaScraper:
    """Scraper for public Strava segment pages."""

    BASE_URL = "https://www.strava.com/segments/"
    EXPLORE_URL = "https://www.strava.com/segments/explore"
    RETRY_DELAYS = [1, 2, 5]  # Exponential backoff delays in seconds
    REQUEST_DELAY = 2.0  # Delay between requests in seconds

    def __init__(self, cache_file: Optional[Path] = None):
        if not HAS_DEPENDENCIES:
            raise ImportError(
                "StravaScraper requires 'requests' and 'beautifulsoup4'. "
                "Install with: pip install requests beautifulsoup4"
            )
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_file = cache_file or Path("tests/.strava_cache.json")
        self._last_request_time = 0
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                    # Filter out expired entries (7 days)
                    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                    self._cache = {
                        k: v for k, v in data.items()
                        if v.get('cached_at', '') > cutoff
                    }
                logger.debug(f"Loaded {len(self._cache)} cached Strava entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load Strava cache: {e}")
                self._cache = {}

    def save_cache(self):
        """Save cache to disk."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved {len(self._cache)} Strava entries to cache")
        except IOError as e:
            logger.warning(f"Failed to save Strava cache: {e}")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_segment(self, segment_id: str) -> StravaSegmentData:
        """
        Fetch and parse a Strava segment page.

        Args:
            segment_id: The Strava segment ID (numeric string)

        Returns:
            StravaSegmentData with segment information or error
        """
        # Check cache first
        cache_key = f"segment:{segment_id}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return StravaSegmentData(**cached.get('data', {}))

        self._rate_limit()
        url = f"{self.BASE_URL}{segment_id}"
        html = self._fetch_with_retry(url)

        if html is None:
            result = StravaSegmentData(
                segment_id=segment_id,
                name="Unknown",
                distance_km=0,
                elev_gain_m=0,
                avg_grade=0,
                fetch_error="Failed to fetch segment page after retries"
            )
        else:
            result = self._parse_segment_page(html, segment_id)

        # Cache result
        self._cache[cache_key] = {
            'data': asdict(result),
            'cached_at': datetime.now().isoformat()
        }
        self.save_cache()
        return result

    def find_matching_segment(
        self,
        name: str,
        lat: float,
        lon: float,
        distance_km: float,
        elev_gain_m: float,
        search_radius_km: float = 5.0,
        name_threshold: float = 0.6
    ) -> Optional[StravaSegmentData]:
        """
        Search for a Strava segment matching the given climb data.

        Uses fuzzy name matching and metric comparison to find the best match.

        Args:
            name: Climb name to search for
            lat: Latitude of climb start
            lon: Longitude of climb start
            distance_km: Expected distance in km
            elev_gain_m: Expected elevation gain in meters
            search_radius_km: Search radius in km (default 5)
            name_threshold: Minimum name similarity ratio (0-1)

        Returns:
            Best matching StravaSegmentData or None
        """
        # Check cache for location search
        cache_key = f"search:{lat:.4f},{lon:.4f}"
        segments = []

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            segments = [StravaSegmentData(**s) for s in cached.get('segments', [])]
        else:
            # Strava explore requires authentication for API, so we can only
            # use known segment IDs. Return None to fall back to other sources.
            logger.debug(f"No cached segments for location ({lat}, {lon})")
            return None

        if not segments:
            return None

        # Score each segment by name similarity and metric match
        best_match = None
        best_score = 0

        for segment in segments:
            if not segment.is_valid():
                continue

            # Name similarity
            name_sim = SequenceMatcher(
                None, name.lower(), segment.name.lower()
            ).ratio()

            if name_sim < name_threshold:
                continue

            # Distance similarity (within 30%)
            if segment.distance_km > 0:
                dist_diff = abs(distance_km - segment.distance_km) / segment.distance_km
                if dist_diff > 0.3:
                    continue
            else:
                continue

            # Elevation similarity (within 30%)
            if segment.elev_gain_m > 0:
                elev_diff = abs(elev_gain_m - segment.elev_gain_m) / segment.elev_gain_m
                if elev_diff > 0.3:
                    continue

            # Combined score: weight name more heavily
            score = (name_sim * 0.6) + ((1 - dist_diff) * 0.2) + ((1 - elev_diff) * 0.2)

            if score > best_score:
                best_score = score
                best_match = segment

        return best_match

    def _fetch_with_retry(self, url: str) -> Optional[str]:
        """
        Fetch URL with exponential backoff retry.

        Args:
            url: URL to fetch

        Returns:
            HTML content or None if all retries failed
        """
        last_error = None

        for attempt, delay in enumerate(self.RETRY_DELAYS):
            try:
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.warning(f"Segment not found: {url}")
                    return None
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    logger.warning(f"Rate limited, waiting {delay * 2}s")
                    time.sleep(delay * 2)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

            # Wait before retry
            if attempt < len(self.RETRY_DELAYS) - 1:
                time.sleep(delay)

        logger.error(f"All retries failed for {url}: {last_error}")
        return None

    def _parse_segment_page(self, html: str, segment_id: str) -> StravaSegmentData:
        """
        Parse Strava segment page HTML.

        Args:
            html: Raw HTML content
            segment_id: Segment ID for reference

        Returns:
            StravaSegmentData with parsed information
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Extract segment name
        name = "Unknown"
        name_elem = soup.find('h1', class_='mb-0') or soup.find('h1')
        if name_elem:
            name = name_elem.get_text(strip=True)

        # Extract stats from the segment stats section
        distance_km = 0.0
        elev_gain_m = 0.0
        avg_grade = 0.0
        max_grade = None
        location = None

        # Try to find stats in various page layouts
        stats = self._extract_stats_from_page(soup)

        if stats:
            distance_km = stats.get('distance_km', 0.0)
            elev_gain_m = stats.get('elev_gain_m', 0.0)
            avg_grade = stats.get('avg_grade', 0.0)
            max_grade = stats.get('max_grade')

        # Extract location
        loc_elem = soup.find('span', class_='location')
        if loc_elem:
            location = loc_elem.get_text(strip=True)

        return StravaSegmentData(
            segment_id=segment_id,
            name=name,
            distance_km=distance_km,
            elev_gain_m=elev_gain_m,
            avg_grade=avg_grade,
            max_grade=max_grade,
            location=location
        )

    def _extract_stats_from_page(self, soup: "BeautifulSoup") -> Dict[str, Any]:
        """Extract stats from various Strava page layouts."""
        stats = {}

        # Look for stat labels and values
        # Strava uses various layouts - try multiple approaches

        # Approach 1: Look for labeled stats
        for stat_elem in soup.find_all(['div', 'span', 'td'], class_=re.compile(r'stat|value')):
            text = stat_elem.get_text(strip=True).lower()

            # Check for distance
            if 'km' in text or 'mi' in text:
                distance = self._parse_distance(text)
                if distance and distance > 0:
                    stats['distance_km'] = distance

            # Check for elevation
            if 'm' in text or 'ft' in text:
                elev = self._parse_elevation(text)
                if elev and elev > 0 and 'elev_gain_m' not in stats:
                    stats['elev_gain_m'] = elev

            # Check for grade
            if '%' in text:
                grade = self._parse_grade(text)
                if grade is not None:
                    if 'avg_grade' not in stats:
                        stats['avg_grade'] = grade
                    elif 'max_grade' not in stats:
                        stats['max_grade'] = grade

        # Approach 2: Look for specific data attributes
        for elem in soup.find_all(attrs={'data-distance': True}):
            try:
                stats['distance_km'] = float(elem['data-distance']) / 1000
            except (ValueError, KeyError):
                pass

        for elem in soup.find_all(attrs={'data-elevation': True}):
            try:
                stats['elev_gain_m'] = float(elem['data-elevation'])
            except (ValueError, KeyError):
                pass

        # Approach 3: Search in script tags for JSON data
        for script in soup.find_all('script'):
            if script.string and 'distance' in script.string.lower():
                # Try to extract numeric values near 'distance', 'elevation', 'grade'
                text = script.string

                # Distance pattern
                dist_match = re.search(r'"distance"\s*:\s*([\d.]+)', text)
                if dist_match and 'distance_km' not in stats:
                    try:
                        # Strava often stores in meters
                        val = float(dist_match.group(1))
                        stats['distance_km'] = val / 1000 if val > 100 else val
                    except ValueError:
                        pass

                # Elevation pattern
                elev_match = re.search(r'"elevation[_]?gain"\s*:\s*([\d.]+)', text, re.I)
                if elev_match and 'elev_gain_m' not in stats:
                    try:
                        stats['elev_gain_m'] = float(elev_match.group(1))
                    except ValueError:
                        pass

                # Grade pattern
                grade_match = re.search(r'"avg[_]?grade"\s*:\s*([\d.]+)', text, re.I)
                if grade_match and 'avg_grade' not in stats:
                    try:
                        stats['avg_grade'] = float(grade_match.group(1))
                    except ValueError:
                        pass

        return stats

    def _parse_distance(self, text: str) -> Optional[float]:
        """Parse distance string to kilometers."""
        text = text.lower().strip()

        # Match patterns like "5.2km", "5.2 km", "3.2mi", "3.2 mi"
        match = re.search(r'([\d.]+)\s*(km|mi|miles?|kilometers?)', text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)

            if 'mi' in unit:
                return value * 1.60934  # Convert miles to km
            return value

        return None

    def _parse_elevation(self, text: str) -> Optional[float]:
        """Parse elevation string to meters."""
        text = text.lower().strip()

        # Match patterns like "500m", "500 m", "1640ft", "1,640 ft"
        match = re.search(r'([\d,]+)\s*(m|meters?|ft|feet)', text)
        if match:
            value = float(match.group(1).replace(',', ''))
            unit = match.group(2)

            if 'ft' in unit or 'feet' in unit:
                return value * 0.3048  # Convert feet to meters
            return value

        return None

    def _parse_grade(self, text: str) -> Optional[float]:
        """Parse grade string to percentage."""
        text = text.lower().strip()

        # Match patterns like "5.2%", "5.2 %"
        match = re.search(r'([\d.]+)\s*%', text)
        if match:
            return float(match.group(1))

        return None

    def clear_cache(self):
        """Clear the segment cache."""
        self._cache.clear()


def validate_against_strava(
    climb_data: Dict[str, Any],
    strava_segment_id: str,
    tolerance_percent: float = 15.0
) -> Dict[str, Any]:
    """
    Validate climb analyzer output against Strava segment data.

    Args:
        climb_data: Dict with keys: name, distance_km, elev_gain_m, avg_grade
        strava_segment_id: Strava segment ID to compare against
        tolerance_percent: Acceptable difference percentage (default 15%)

    Returns:
        Dict with validation results
    """
    scraper = StravaScraper()
    strava_data = scraper.get_segment(strava_segment_id)

    if not strava_data.is_valid():
        return {
            'valid': False,
            'error': strava_data.fetch_error or 'Failed to fetch Strava data',
            'strava_data': None,
            'comparisons': {}
        }

    comparisons = {}

    # Compare distance
    if 'distance_km' in climb_data and strava_data.distance_km > 0:
        diff = abs(climb_data['distance_km'] - strava_data.distance_km)
        diff_pct = (diff / strava_data.distance_km) * 100
        comparisons['distance'] = {
            'climb_analyzer': climb_data['distance_km'],
            'strava': strava_data.distance_km,
            'diff_percent': diff_pct,
            'within_tolerance': diff_pct <= tolerance_percent
        }

    # Compare elevation gain
    if 'elev_gain_m' in climb_data and strava_data.elev_gain_m > 0:
        diff = abs(climb_data['elev_gain_m'] - strava_data.elev_gain_m)
        diff_pct = (diff / strava_data.elev_gain_m) * 100
        comparisons['elevation'] = {
            'climb_analyzer': climb_data['elev_gain_m'],
            'strava': strava_data.elev_gain_m,
            'diff_percent': diff_pct,
            'within_tolerance': diff_pct <= tolerance_percent
        }

    # Compare average grade
    if 'avg_grade' in climb_data and strava_data.avg_grade > 0:
        diff = abs(climb_data['avg_grade'] - strava_data.avg_grade)
        diff_pct = (diff / strava_data.avg_grade) * 100
        comparisons['grade'] = {
            'climb_analyzer': climb_data['avg_grade'],
            'strava': strava_data.avg_grade,
            'diff_percent': diff_pct,
            'within_tolerance': diff_pct <= tolerance_percent
        }

    # Overall validation - all compared metrics must be within tolerance
    all_within_tolerance = all(
        c['within_tolerance'] for c in comparisons.values()
    ) if comparisons else False

    return {
        'valid': all_within_tolerance,
        'strava_data': {
            'name': strava_data.name,
            'distance_km': strava_data.distance_km,
            'elev_gain_m': strava_data.elev_gain_m,
            'avg_grade': strava_data.avg_grade,
            'max_grade': strava_data.max_grade,
            'location': strava_data.location
        },
        'comparisons': comparisons,
        'tolerance_percent': tolerance_percent
    }


def format_validation_result(result: Dict[str, Any]) -> str:
    """Format validation result for display."""
    lines = []

    if result.get('error'):
        return f"Strava validation failed: {result['error']}"

    strava = result.get('strava_data', {})
    if strava:
        lines.append(f"Strava segment: {strava.get('name', 'Unknown')}")
        if strava.get('location'):
            lines.append(f"  Location: {strava['location']}")

    lines.append("")
    lines.append("Comparison (Analyzer vs Strava):")

    for metric, data in result.get('comparisons', {}).items():
        status = "[OK]" if data['within_tolerance'] else "[DIFF]"
        lines.append(
            f"  {metric.capitalize()}: {data['climb_analyzer']:.1f} vs "
            f"{data['strava']:.1f} ({data['diff_percent']:.1f}% diff) {status}"
        )

    overall = "PASS" if result['valid'] else "FAIL"
    lines.append(f"\nOverall: {overall} (tolerance: {result.get('tolerance_percent', 15)}%)")

    return '\n'.join(lines)


if __name__ == '__main__':
    # Test with a known segment
    import sys

    if len(sys.argv) > 1:
        segment_id = sys.argv[1]
    else:
        # Default test segment - Bear Mountain (NY)
        segment_id = "628459"

    print(f"Fetching Strava segment {segment_id}...")

    try:
        scraper = StravaScraper()
        data = scraper.get_segment(segment_id)

        if data.is_valid():
            print(f"\nSegment: {data.name}")
            print(f"Distance: {data.distance_km:.2f} km")
            print(f"Elevation: {data.elev_gain_m:.0f} m")
            print(f"Avg Grade: {data.avg_grade:.1f}%")
            if data.max_grade:
                print(f"Max Grade: {data.max_grade:.1f}%")
            if data.location:
                print(f"Location: {data.location}")
        else:
            print(f"Error: {data.fetch_error}")

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        sys.exit(1)
