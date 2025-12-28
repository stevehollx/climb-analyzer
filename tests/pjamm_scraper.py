#!/usr/bin/env python3
"""
PJAMM Cycling scraper for climb validation.

Fetches public PJAMM climb data to validate climb analyzer output
against curated cycling climb database.
"""

import re
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

logger = logging.getLogger(__name__)


@dataclass
class PjammClimbData:
    """Data extracted from a PJAMM climb page."""
    climb_id: str
    name: str
    location: str
    distance_km: float
    elev_gain_m: float
    avg_grade: float
    max_grade: Optional[float] = None
    pdi_score: Optional[float] = None
    difficulty_rank: Optional[int] = None
    page_url: str = ""
    fetch_error: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if climb data was successfully fetched."""
        return self.fetch_error is None and self.distance_km > 0


class PjammScraper:
    """Scraper for public PJAMM Cycling climb pages."""

    BASE_URL = "https://pjammcycling.com"
    SEARCH_URL = "https://pjammcycling.com/search"
    RETRY_DELAYS = [1, 2, 5]
    REQUEST_DELAY = 1.0  # Delay between requests in seconds

    def __init__(self, cache_file: Optional[Path] = None):
        if not HAS_DEPENDENCIES:
            raise ImportError(
                "PjammScraper requires 'requests' and 'beautifulsoup4'. "
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
        self._cache_file = cache_file or Path("tests/.pjamm_cache.json")
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
                logger.debug(f"Loaded {len(self._cache)} cached PJAMM entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load PJAMM cache: {e}")
                self._cache = {}

    def save_cache(self):
        """Save cache to disk."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved {len(self._cache)} PJAMM entries to cache")
        except IOError as e:
            logger.warning(f"Failed to save PJAMM cache: {e}")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def search_by_name(self, name: str) -> List[PjammClimbData]:
        """
        Search PJAMM for climbs by name.

        Args:
            name: Climb name to search for

        Returns:
            List of matching PjammClimbData objects
        """
        # Check cache first
        cache_key = f"search:{name.lower()}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return [PjammClimbData(**c) for c in cached.get('results', [])]

        # PJAMM uses client-side search, so we need to fetch their climb index
        # For now, try direct URL construction for well-known climbs
        results = []

        # Try common URL patterns
        url_name = self._name_to_url(name)

        # Try fetching the climb page directly
        for pattern in [
            f"{self.BASE_URL}/climb/{url_name}",
            f"{self.BASE_URL}/climb/{url_name}-Cycling",
        ]:
            climb = self._fetch_climb_page(pattern)
            if climb and climb.is_valid():
                results.append(climb)
                break

        # Cache results
        self._cache[cache_key] = {
            'results': [asdict(r) for r in results],
            'cached_at': datetime.now().isoformat()
        }
        self.save_cache()

        return results

    def get_climb_by_url(self, url: str) -> PjammClimbData:
        """
        Fetch climb data from a specific PJAMM URL.

        Args:
            url: Full PJAMM climb URL

        Returns:
            PjammClimbData with climb information or error
        """
        # Check cache
        cache_key = f"url:{url}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return PjammClimbData(**cached.get('data', {}))

        result = self._fetch_climb_page(url)

        # Cache result
        if result:
            self._cache[cache_key] = {
                'data': asdict(result),
                'cached_at': datetime.now().isoformat()
            }
            self.save_cache()

        return result or PjammClimbData(
            climb_id="unknown",
            name="Unknown",
            location="",
            distance_km=0,
            elev_gain_m=0,
            avg_grade=0,
            fetch_error="Failed to fetch climb page"
        )

    def _name_to_url(self, name: str) -> str:
        """Convert climb name to PJAMM URL format."""
        # Remove special characters, replace spaces with hyphens
        url_name = re.sub(r'[^\w\s-]', '', name)
        url_name = re.sub(r'\s+', '-', url_name.strip())
        return url_name

    def _fetch_climb_page(self, url: str) -> Optional[PjammClimbData]:
        """Fetch and parse a PJAMM climb page."""
        self._rate_limit()

        last_error = None
        for attempt, delay in enumerate(self.RETRY_DELAYS):
            try:
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return self._parse_climb_page(response.text, url)
                elif response.status_code == 404:
                    logger.debug(f"PJAMM climb not found: {url}")
                    return None
                elif response.status_code == 429:
                    logger.warning(f"PJAMM rate limited, waiting {delay * 2}s")
                    time.sleep(delay * 2)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"PJAMM request failed (attempt {attempt + 1}): {e}")

            if attempt < len(self.RETRY_DELAYS) - 1:
                time.sleep(delay)

        logger.error(f"All PJAMM retries failed for {url}: {last_error}")
        return None

    def _parse_climb_page(self, html: str, url: str) -> PjammClimbData:
        """Parse PJAMM climb page HTML."""
        soup = BeautifulSoup(html, 'html.parser')

        # Extract climb ID from URL
        climb_id = url.split('/')[-1].split('.')[0] if '/' in url else "unknown"

        # Extract climb name
        name = "Unknown"
        name_elem = soup.find('h1') or soup.find('title')
        if name_elem:
            name = name_elem.get_text(strip=True)
            # Clean up name - remove " | PJAMM Cycling" suffix
            name = re.sub(r'\s*\|\s*PJAMM.*$', '', name)

        # Extract location
        location = ""
        loc_elem = soup.find('span', class_='location') or soup.find(class_='climb-location')
        if loc_elem:
            location = loc_elem.get_text(strip=True)

        # Extract stats
        stats = self._extract_stats(soup)

        return PjammClimbData(
            climb_id=climb_id,
            name=name,
            location=location,
            distance_km=stats.get('distance_km', 0),
            elev_gain_m=stats.get('elev_gain_m', 0),
            avg_grade=stats.get('avg_grade', 0),
            max_grade=stats.get('max_grade'),
            pdi_score=stats.get('pdi_score'),
            difficulty_rank=stats.get('difficulty_rank'),
            page_url=url
        )

    def _extract_stats(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract climb statistics from page."""
        stats = {}

        # Look for stat items (PJAMM uses .stat-item containers)
        for stat in soup.find_all(class_='stat-item'):
            value_elem = stat.find(class_='stat-value')
            title_elem = stat.find(class_='stat-title')

            if value_elem and title_elem:
                value_text = value_elem.get_text(strip=True).lower()
                title_text = title_elem.get_text(strip=True).lower()

                if 'distance' in title_text:
                    stats['distance_km'] = self._parse_distance(value_text)
                elif 'elevation' in title_text or 'gain' in title_text:
                    stats['elev_gain_m'] = self._parse_elevation(value_text)
                elif 'avg' in title_text and 'grade' in title_text:
                    stats['avg_grade'] = self._parse_grade(value_text)
                elif 'max' in title_text and 'grade' in title_text:
                    stats['max_grade'] = self._parse_grade(value_text)
                elif 'pdi' in title_text or 'difficulty' in title_text:
                    try:
                        stats['pdi_score'] = float(re.sub(r'[^\d.]', '', value_text))
                    except ValueError:
                        pass

        # Also search in general text for stats if stat-items not found
        if not stats:
            text = soup.get_text()

            # Distance patterns
            dist_match = re.search(r'(\d+\.?\d*)\s*(km|mi|miles?)', text, re.I)
            if dist_match:
                val = float(dist_match.group(1))
                if 'mi' in dist_match.group(2).lower():
                    val *= 1.60934
                stats['distance_km'] = val

            # Elevation patterns
            elev_match = re.search(r'(\d+[,\d]*)\s*(m|ft|feet|meters?)\s*(gain|elevation)?', text, re.I)
            if elev_match:
                val = float(elev_match.group(1).replace(',', ''))
                if 'ft' in elev_match.group(2).lower() or 'feet' in elev_match.group(2).lower():
                    val *= 0.3048
                stats['elev_gain_m'] = val

            # Grade patterns
            grade_match = re.search(r'avg\.?\s*grade[:\s]*(\d+\.?\d*)\s*%', text, re.I)
            if grade_match:
                stats['avg_grade'] = float(grade_match.group(1))

        return stats

    def _parse_distance(self, text: str) -> float:
        """Parse distance string to kilometers."""
        match = re.search(r'([\d.]+)\s*(km|mi|miles?)?', text)
        if match:
            val = float(match.group(1))
            unit = match.group(2) or ''
            if 'mi' in unit.lower():
                return val * 1.60934
            return val
        return 0

    def _parse_elevation(self, text: str) -> float:
        """Parse elevation string to meters."""
        match = re.search(r'([\d,]+)\s*(m|ft|feet)?', text)
        if match:
            val = float(match.group(1).replace(',', ''))
            unit = match.group(2) or ''
            if 'ft' in unit.lower() or 'feet' in unit.lower():
                return val * 0.3048
            return val
        return 0

    def _parse_grade(self, text: str) -> Optional[float]:
        """Parse grade string to percentage."""
        match = re.search(r'([\d.]+)\s*%?', text)
        if match:
            return float(match.group(1))
        return None

    def clear_cache(self):
        """Clear the climb cache."""
        self._cache.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()


# Well-known PJAMM climb mappings for direct lookup
KNOWN_PJAMM_CLIMBS = {
    "mauna kea": "37.Mauna-Kea",
    "mauna kea access road": "37.Mauna-Kea",
    "haleakala": "38.Haleakala",
    "mont ventoux": "1.Mont-Ventoux",
    "alpe d'huez": "2.Alpe-dHuez",
    "col du galibier": "5.Col-du-Galibier",
    "stelvio": "3.Stelvio",
    "passo dello stelvio": "3.Stelvio",
    "col du tourmalet": "6.Col-du-Tourmalet",
    "mortirolo": "4.Mortirolo",
    "sa calobra": "16.Sa-Calobra",
    "mount washington": "30.Mount-Washington",
    "pikes peak": "31.Pikes-Peak",
}


def get_pjamm_climb(name: str) -> Optional[PjammClimbData]:
    """
    Get PJAMM climb data by name.

    Args:
        name: Climb name to look up

    Returns:
        PjammClimbData or None if not found
    """
    scraper = PjammScraper()

    # Check known climbs first
    normalized = name.lower().strip()
    if normalized in KNOWN_PJAMM_CLIMBS:
        climb_id = KNOWN_PJAMM_CLIMBS[normalized]
        url = f"https://pjammcycling.com/climb/{climb_id}"
        return scraper.get_climb_by_url(url)

    # Try search
    results = scraper.search_by_name(name)
    return results[0] if results else None


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) > 1:
        climb_name = ' '.join(sys.argv[1:])
    else:
        climb_name = "Mauna Kea"

    print(f"Searching PJAMM for: {climb_name}")

    try:
        result = get_pjamm_climb(climb_name)

        if result and result.is_valid():
            print(f"\nClimb: {result.name}")
            print(f"Location: {result.location}")
            print(f"Distance: {result.distance_km:.2f} km")
            print(f"Elevation: {result.elev_gain_m:.0f} m")
            print(f"Avg Grade: {result.avg_grade:.1f}%")
            if result.max_grade:
                print(f"Max Grade: {result.max_grade:.1f}%")
            if result.pdi_score:
                print(f"PDI Score: {result.pdi_score:.0f}")
            print(f"URL: {result.page_url}")
        else:
            error = result.fetch_error if result else "Not found"
            print(f"Error: {error}")

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        sys.exit(1)
