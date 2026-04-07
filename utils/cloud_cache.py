#!/usr/bin/env python3
"""
Cloud Cache Manager for Climb Analyzer.

Manages downloading and uploading climb analyses to/from the GitHub cloud cache repository.
Handles multi-file analyses (split Excel files) and USA state-level caching.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from utils.github_app_client import GitHubAppClient

# Try to import embedded credentials first, fall back to environment variables
try:
    from utils.cloud_cache_config import get_config, is_configured

    _config = get_config()
    _embedded_available = is_configured()
except ImportError:
    _embedded_available = False
    _config = None


class CloudCacheManager:
    """
    Manages cloud cache operations for climb analyses.

    Features:
    - Download pre-analyzed data from GitHub
    - Upload completed analyses to GitHub (via pull request)
    - Handle multi-file analyses (split Excel files)
    - USA state-level caching for large states
    - Country-level caching for all other countries

    Uses GitHub App authentication for public sharing without compromising security.
    """

    # Repository is hardcoded - always use the global climb data repository
    REPO_OWNER = "stevehollx"
    REPO_NAME = "global-road-and-trail-climbs"

    # GitHub App credentials - configurable via embedded config or environment
    if _embedded_available:
        GITHUB_APP_ID = _config["app_id"]
        GITHUB_PRIVATE_KEY = _config["private_key"]
    else:
        # Fall back to environment variables for GitHub App credentials only
        GITHUB_APP_ID = os.environ.get("GITHUB_APP_ID", "")
        GITHUB_PRIVATE_KEY = os.environ.get("GITHUB_PRIVATE_KEY", "")

    # Clean analysis criteria
    CLEAN_SURFACE = "all"
    CLEAN_MIN_SCORE = 0
    CLEAN_CYCLING = False

    def __init__(self):
        """Initialize cloud cache manager."""
        self.github = GitHubAppClient(
            app_id=self.GITHUB_APP_ID,
            private_key=self.GITHUB_PRIVATE_KEY,
            owner=self.REPO_OWNER,
            repo=self.REPO_NAME,
        )

    def is_token_configured(self) -> bool:
        """Check if GitHub App is configured (has valid private key)."""
        return (
            self.GITHUB_PRIVATE_KEY
            and "REPLACE_WITH_YOUR_PRIVATE_KEY_HERE" not in self.GITHUB_PRIVATE_KEY
        )

    def is_clean_analysis(self, surface_filter: str, min_score: float, cycling_only: bool) -> bool:
        """
        Check if analysis meets "clean" criteria for cloud cache.

        Clean analysis = all roads, min_score=0, cycling off.
        Only clean analyses are uploaded to cloud cache for consistency.

        Args:
            surface_filter: Surface filter used
            min_score: Minimum score threshold
            cycling_only: Whether cycling filter was enabled

        Returns:
            True if analysis is clean, False otherwise
        """
        return (
            surface_filter == self.CLEAN_SURFACE
            and min_score == self.CLEAN_MIN_SCORE
            and cycling_only == self.CLEAN_CYCLING
        )

    def is_usa(self, country_name: str) -> bool:
        """
        Check if country is United States of America.

        Args:
            country_name: Country name to check

        Returns:
            True if USA, False otherwise
        """
        if country_name is None:
            return False

        usa_variations = [
            "United States of America",
            "United States",
            "USA",
            "US",
            "united-states-of-america",
            "united-states",
            "usa",
            "us",
        ]
        return country_name.lower() in [v.lower() for v in usa_variations]

    def sanitize_name(self, name: str) -> str:
        """
        Convert name to folder-safe format.

        Args:
            name: Name to sanitize (e.g., "United States of America")

        Returns:
            Sanitized name (e.g., "united-states-of-america")
        """
        if name is None:
            return "unknown"

        # Convert to lowercase
        name = name.lower()
        # Replace spaces and underscores with hyphens for consistency
        name = name.replace(" ", "-").replace("_", "-")
        # Remove special characters
        name = name.replace("/", "-")
        name = name.replace("'", "")
        name = name.replace(".", "")
        return name

    def get_elevation_datasets(self) -> List[str]:
        """
        Get list of elevation datasets from opentopodata config.

        Returns:
            List of dataset names in priority order (e.g., ['ned10m', 'srtm30m'])
        """
        config_path = Path("opentopodata-config.yaml")
        if not config_path.exists():
            return []

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            datasets = config.get("datasets", [])
            # Extract dataset names from the list of dicts
            if isinstance(datasets, list):
                return [d.get("name", "") for d in datasets if isinstance(d, dict) and d.get("name")]
            return []
        except Exception:
            return []

    def load_datasets_used_from_checkpoint(self, region_name: str) -> List[str]:
        """
        Load datasets_used from the most recent analysis checkpoint folder.

        Args:
            region_name: Name of the region to search for

        Returns:
            List of dataset names that were actually used, or empty list if not found
        """
        import json

        checkpoint_base = Path("data/checkpoint_data")
        if not checkpoint_base.exists():
            return []

        # Normalize region name for matching
        normalized = region_name.lower().replace(" ", "_").replace("-", "_")

        # Find matching checkpoint folders (sorted by modification time, newest first)
        matching_dirs = []
        for d in checkpoint_base.iterdir():
            if d.is_dir():
                dir_name_normalized = d.name.lower().replace("-", "_")
                if normalized in dir_name_normalized:
                    matching_dirs.append(d)

        if not matching_dirs:
            return []

        # Sort by modification time (most recent first)
        matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Check each for datasets_used.json
        for analysis_dir in matching_dirs:
            datasets_file = analysis_dir / "datasets_used.json"
            if datasets_file.exists():
                try:
                    with open(datasets_file) as f:
                        return json.load(f)
                except Exception:
                    continue

        return []

    def get_country_continent(self, country_name: str) -> Optional[str]:
        """
        Get continent name for a country.

        Uses geo_lookup to find the region and extract continent from its path.

        Args:
            country_name: Country name

        Returns:
            Continent name (e.g., "North America") or None if country_name is None
        """
        if country_name is None:
            return None

        from climb_analyzer.data.geo_lookup import find_region, is_us_state

        # SAFETY CHECK: If this is actually a US state, return North America
        # This prevents Georgia (state) from being looked up as Georgia (country in Asia)
        if is_us_state(country_name):
            return "North America"

        # Normalize common country name variations for lookup
        country_aliases = {
            "US": "us",
            "USA": "us",
            "United States": "us",
            "United States of America": "us",
            "UK": "united-kingdom",
            "Britain": "united-kingdom",
            "Great Britain": "united-kingdom",
        }

        # Normalize if alias exists
        lookup_name = country_aliases.get(country_name, country_name)

        # Find region using geo_lookup
        region_info = find_region(lookup_name)

        if region_info and "pbf_url" in region_info:
            # Extract continent from pbf_url
            # e.g., https://download.geofabrik.de/europe/france-latest.osm.pbf -> europe
            pbf_url = region_info["pbf_url"]
            url_path = pbf_url.replace("https://download.geofabrik.de/", "")
            # First segment is continent (e.g., "europe", "north-america")
            continent_slug = url_path.split("/")[0]

            # Convert slug to display name
            continent_map = {
                "africa": "Africa",
                "asia": "Asia",
                "australia-oceania": "Oceania",
                "central-america": "Central America",
                "europe": "Europe",
                "north-america": "North America",
                "south-america": "South America",
            }
            return continent_map.get(continent_slug, continent_slug.replace("-", " ").title())

        # Default fallback
        print(f"[WARNING] Country '{country_name}' not found in geo_lookup")
        return "Unknown"

    def get_cache_path(
        self, country: str, region: Optional[str] = None, scope_type: str = "country"
    ) -> Optional[str]:
        """
        Determine cache path based on country/continent and region.

        Args:
            country: Country name OR continent name (for subregion analysis)
            region: Region/state name (for USA states or subregions like Bristol)
            scope_type: Type of analysis ("country", "region")

        Returns:
            Path in repo (e.g., "north-america/united-states-of-america/vermont")
            or None if full USA (not supported)

        Examples:
            get_cache_path("United States of America", "Vermont", "region")
            -> "north-america/united-states-of-america/vermont"

            get_cache_path("Iceland", scope_type="country")
            -> "europe/iceland"

            get_cache_path("Europe", "bristol", "region")
            -> "europe/united-kingdom/england/bristol" (uses geo_lookup)

            get_cache_path("United States of America", scope_type="country")
            -> None (full USA too large)
        """
        from climb_analyzer.data.geo_lookup import find_region, is_us_state

        # Early return if no country provided (e.g., address-based searches)
        if country is None and region is None:
            return None

        # SAFETY CHECK: If country parameter is actually a US state name, fix it
        if is_us_state(country):
            # State name was passed as country - correct it
            region = country
            country = "United States of America"
            scope_type = "region"

        # For region scope, try to get full path from geo_lookup
        if scope_type == "region" and region:
            # Check if this is a US state - use standardized path format
            # US states must use "united-states-of-america" not "us" from Geofabrik URLs
            if is_us_state(region):
                sanitized_region = self.sanitize_name(region)
                return f"north-america/united-states-of-america/{sanitized_region}"

            # Non-US regions: extract path from pbf_url
            region_info = find_region(region)
            if region_info and region_info.get("pbf_url"):
                # Extract path from pbf_url
                # e.g., https://download.geofabrik.de/europe/united-kingdom/england/bristol-latest.osm.pbf
                # -> europe/united-kingdom/england/bristol
                pbf_url = region_info["pbf_url"]
                url_path = pbf_url.replace("https://download.geofabrik.de/", "")
                # Remove "-latest.osm.pbf" suffix
                cache_path = url_path.replace("-latest.osm.pbf", "")
                return cache_path

        # Fallback to original logic for countries
        continent = self.get_country_continent(country)
        sanitized_continent = self.sanitize_name(continent)

        if self.is_usa(country):
            if scope_type == "region" and region:
                # USA state: north-america/united-states-of-america/{state}
                sanitized_region = self.sanitize_name(region)
                return f"{sanitized_continent}/united-states-of-america/{sanitized_region}"
            else:
                # Full USA not supported (too large)
                return None
        else:
            # Other countries: {continent}/{country}
            sanitized_country = self.sanitize_name(country)
            return f"{sanitized_continent}/{sanitized_country}"

    def parse_analysis_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse analysis filename to extract metadata.

        Args:
            filename: Filename to parse

        Returns:
            Dict with metadata or None if doesn't match pattern

        Examples:
            "california_climbs_all-surfaces_all-access_imperial_2025-10-25_v2.1.0_e0000-1.xlsx"
            -> {region: 'california', surface: 'all',
                access: 'all-access', units: 'imperial',
                date: '2025-10-25', part: 1, ext: 'xlsx'}

            "iceland_climbs_all-surfaces_cycling_metric_2025-10-25_v2.1.0_e0000.xlsx"
            -> {region: 'iceland', surface: 'all',
                access: 'cycling', units: 'metric',
                date: '2025-10-25', part: None, ext: 'xlsx'}
        """
        # Pattern: {name}_climbs_{surface}_{date}[_v{version}_e{errors}][-{part}].{ext}
        # Updated to handle both old and new filename formats:
        #   OLD: california_climbs_all_2025-10-25_v2.1.0_e0000.xlsx
        #   NEW: north_carolina_climbs_all-surfaces_cycling_imperial_2025-12-16_v2.2.1_e0000-1.xlsx

        # Try new format first (more specific): {name}_climbs_{surface}_{access}_{units}_{date}...
        new_pattern = r"(.+)_climbs_(all-surfaces|paved|gravel|dirt)_(cycling|all-access)_(imperial|metric)_(\d{4}-\d{2}-\d{2})(?:_v[\d.]+_e\d+)?(?:-(\d+))?\.(\w+)"
        match = re.match(new_pattern, filename)

        if match:
            # Map new surface names to old names for consistency
            surface_map = {"all-surfaces": "all"}
            return {
                "region": match.group(1),
                "surface": surface_map.get(match.group(2), match.group(2)),
                "access": match.group(3),
                "units": match.group(4),
                "date": match.group(5),
                "part": int(match.group(6)) if match.group(6) else None,
                "ext": match.group(7),
            }

        # Fall back to old format: {name}_climbs_{surface}_{date}...
        old_pattern = r"(.+)_climbs_(all|paved|gravel|dirt)_(\d{4}-\d{2}-\d{2})(?:_v[\d.]+_e\d+)?(?:-(\d+))?\.(\w+)"
        match = re.match(old_pattern, filename)

        if match:
            return {
                "region": match.group(1),
                "surface": match.group(2),
                "date": match.group(3),
                "part": int(match.group(4)) if match.group(4) else None,
                "ext": match.group(5),
            }

        # Also check for errors TXT pattern (CSV format inside)
        # Try new format first: {name}_errors_{surface}_{access}_{units}_{date}...
        new_error_pattern = r"(.+)_errors_(all-surfaces|paved|gravel|dirt)_(cycling|all-access)_(imperial|metric)_(\d{4}-\d{2}-\d{2})(?:_v[\d.]+_e\d+)?\.txt"
        error_match = re.match(new_error_pattern, filename)

        if error_match:
            # Map new surface names to old names for consistency
            surface_map = {"all-surfaces": "all"}
            surface = error_match.group(2)
            return {
                "region": error_match.group(1),
                "surface": surface_map.get(surface, surface),
                "access": error_match.group(3),
                "units": error_match.group(4),
                "date": error_match.group(5),
                "part": None,
                "ext": "csv",
                "is_error": True,
            }

        # Fall back to old error format
        old_error_pattern = (
            r"(.+)_errors_(all|paved|gravel|dirt)_(\d{4}-\d{2}-\d{2})(?:_v[\d.]+_e\d+)?\.txt"
        )
        error_match = re.match(old_error_pattern, filename)

        if error_match:
            surface = error_match.group(2)
            return {
                "region": error_match.group(1),
                "surface": surface,
                "date": error_match.group(3),
                "part": None,
                "ext": "csv",
                "is_error": True,
            }

        return None

    def group_analysis_files(self, file_list: List[str]) -> Dict:
        """
        Group files by analysis (base name + date).

        Args:
            file_list: List of filenames

        Returns:
            Dict mapping analysis key to file info

        Example:
            {
                'california_2025-10-25': {
                    'xlsx': ['california-1.xlsx', 'california-2.xlsx'],
                    'csv': 'california_errors.csv',
                    'date': '2025-10-25',
                    'parts': 2
                }
            }
        """
        grouped = {}

        for filename in file_list:
            parsed = self.parse_analysis_filename(filename)
            if not parsed:
                continue

            key = f"{parsed['region']}_{parsed['date']}"

            if key not in grouped:
                grouped[key] = {
                    "xlsx": [],
                    "csv": None,
                    "date": parsed["date"],
                    "parts": 0,
                    "region": parsed["region"],
                }

            if parsed["ext"] in ["xlsx", "xls"]:
                grouped[key]["xlsx"].append(filename)
                grouped[key]["parts"] += 1
            elif parsed["ext"] == "csv" and parsed.get("is_error"):
                grouped[key]["csv"] = filename

        # Sort xlsx files by part number
        for key in grouped:
            grouped[key]["xlsx"].sort()

        return grouped

    def find_latest_analysis(self, grouped_analyses: Dict) -> Optional[Dict]:
        """
        Find most recent analysis from grouped analyses.

        Args:
            grouped_analyses: Result from group_analysis_files()

        Returns:
            Latest analysis dict or None if no analyses
        """
        if not grouped_analyses:
            return None

        # Sort by date (most recent first)
        sorted_keys = sorted(
            grouped_analyses.keys(), key=lambda k: grouped_analyses[k]["date"], reverse=True
        )

        latest_key = sorted_keys[0]
        return grouped_analyses[latest_key]

    def check_cached(
        self, country: str, region: Optional[str] = None, scope_type: str = "country"
    ) -> Dict:
        """
        Check if analysis exists in cloud cache.

        Args:
            country: Country name
            region: Region/state name (for USA states)
            scope_type: Type of analysis

        Returns:
            Dict with cache info:
            {
                'exists': True/False,
                'xlsx_files': ['file-1.xlsx', 'file-2.xlsx'],
                'csv_file': 'errors.csv',
                'date': '2025-10-25',
                'total_parts': 2,
                'path': 'north-america/united-states-of-america/california'
            }
        """
        cache_path = self.get_cache_path(country, region, scope_type)

        if cache_path is None:
            return {"exists": False, "path": None}

        # List files in cache directory
        files = self.github.list_directory_files(cache_path)

        if not files:
            return {"exists": False, "path": cache_path}

        # Group files by analysis
        grouped = self.group_analysis_files(files)

        if not grouped:
            return {"exists": False, "path": cache_path}

        # Get latest analysis
        latest = self.find_latest_analysis(grouped)

        if not latest:
            return {"exists": False, "path": cache_path}

        # Parse version and error count from first xlsx file
        version = None
        error_count = None
        if latest["xlsx"]:
            version, error_count = self.parse_version_and_errors(latest["xlsx"][0])

        return {
            "exists": True,
            "xlsx_files": latest["xlsx"],
            "csv_file": latest["csv"],
            "date": latest["date"],
            "total_parts": latest["parts"],
            "path": cache_path,
            "region": latest["region"],
            "version": version,
            "error_count": error_count,
        }

    def download_cache(self, cache_info: Dict, output_dir: Path) -> Dict:
        """
        Download all files for an analysis.

        Args:
            cache_info: Result from check_cached()
            output_dir: Local directory to save files

        Returns:
            Dict with local file paths:
            {
                'xlsx_files': [Path('/path/file-1.xlsx')],
                'csv_file': Path('/path/errors.csv'),
                'total_parts': 2
            }
        """
        if not cache_info["exists"]:
            return {"xlsx_files": [], "csv_file": None, "total_parts": 0}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_path = cache_info["path"]
        xlsx_files = []
        csv_file = None

        # Download xlsx files
        total_files = len(cache_info["xlsx_files"]) + (1 if cache_info["csv_file"] else 0)
        file_num = 0

        for xlsx_filename in cache_info["xlsx_files"]:
            file_num += 1
            repo_path = f"{cache_path}/{xlsx_filename}"
            local_path = output_dir / xlsx_filename

            print(f"  Downloading {xlsx_filename}... ({file_num}/{total_files})")

            if self.github.download_file(repo_path, local_path):
                xlsx_files.append(local_path)
                file_size_mb = local_path.stat().st_size / 1024**2
                print(f"  ✓ Downloaded {xlsx_filename} ({file_size_mb:.1f} MB)")
            else:
                print(f"  ✗ Failed to download {xlsx_filename}")

        # Download error log file (txt with CSV format inside)
        if cache_info["csv_file"]:
            file_num += 1
            csv_filename = cache_info["csv_file"]
            repo_path = f"{cache_path}/{csv_filename}"
            local_path = output_dir / csv_filename

            print(f"  Downloading {csv_filename}... ({file_num}/{total_files})")

            if self.github.download_file(repo_path, local_path):
                csv_file = local_path
                file_size_kb = local_path.stat().st_size / 1024
                print(f"  ✓ Downloaded {csv_filename} ({file_size_kb:.1f} KB)")
            else:
                print(f"  ✗ Failed to download {csv_filename}")

        return {"xlsx_files": xlsx_files, "csv_file": csv_file, "total_parts": len(xlsx_files)}

    def filter_climbs_by_bbox(
        self,
        xlsx_files: List[Path],
        bbox: Tuple[float, float, float, float],
        region_name: str,
        output_dir: Path,
    ) -> List[Path]:
        """
        Filter multi-file analysis by bounding box.

        Args:
            xlsx_files: List of Excel file paths to filter
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
            region_name: Name of region for output filename
            output_dir: Directory to save filtered results

        Returns:
            List of created output files
        """
        from climb_analyzer import save_large_dataframe_as_split_excel

        print(f"\n  Loading and combining {len(xlsx_files)} Excel file(s)...")

        # Load all xlsx parts
        dfs = []
        for xlsx_file in xlsx_files:
            try:
                df = pd.read_excel(xlsx_file)
                dfs.append(df)
            except Exception as e:
                print(f"  ⚠️  Error loading {xlsx_file.name}: {e}")

        if not dfs:
            print("  ✗ No data loaded from Excel files")
            return []

        # Concatenate all parts
        full_df = pd.concat(dfs, ignore_index=True)
        total_climbs = len(full_df)
        print(f"  ✓ Loaded {total_climbs:,} climbs")

        # Filter by bbox
        min_lat, min_lon, max_lat, max_lon = bbox

        filtered_df = full_df[
            (full_df["Latitude"] >= min_lat)
            & (full_df["Latitude"] <= max_lat)
            & (full_df["Longitude"] >= min_lon)
            & (full_df["Longitude"] <= max_lon)
        ]

        filtered_count = len(filtered_df)
        print(f"  ✓ Filtered to {filtered_count:,} climbs within {region_name} bbox")

        # Save filtered result (may split if still large)
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_filename = output_dir / f"{region_name}_climbs_all_{date_str}"

        created_files = save_large_dataframe_as_split_excel(filtered_df, base_filename)

        return created_files

    @staticmethod
    def parse_version_and_errors(filename: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse version and elevation error count from filename.

        Filename format: {name}_v{version}_e{errors}.xlsx
        Example: Luxembourg_climbs_all_basic_2025-11-19_v2.1.0_e0000.xlsx

        Args:
            filename: Filename to parse

        Returns:
            Tuple of (version_string, error_count)
            Returns (None, None) if parsing fails
        """
        # Match version pattern: _v{major}.{minor}.{patch}
        version_match = re.search(r"_v(\d+\.\d+\.\d+)", filename)
        version = version_match.group(1) if version_match else None

        # Match error count pattern: _e{count}
        error_match = re.search(r"_e(\d+)", filename)
        errors = int(error_match.group(1)) if error_match else None

        return (version, errors)

    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """
        Compare two semantic version strings.

        Args:
            v1: First version (e.g., "2.1.0")
            v2: Second version (e.g., "2.0.1")

        Returns:
            1 if v1 > v2, -1 if v1 < v2, 0 if equal
        """
        parts1 = tuple(int(x) for x in v1.split("."))
        parts2 = tuple(int(x) for x in v2.split("."))

        if parts1 > parts2:
            return 1
        elif parts1 < parts2:
            return -1
        else:
            return 0

    def should_upload_analysis(
        self, local_files: Dict, existing_files: List[str]
    ) -> Tuple[bool, str]:
        """
        Determine if local analysis should be uploaded based on version and error count.

        Rules:
        1. Upload if local version > existing version
        2. Upload if versions equal AND local errors < existing errors
        3. Don't upload if local version < existing version
        4. Don't upload if versions equal AND local errors >= existing errors

        Args:
            local_files: Dict with 'xlsx' (list of Path objects) for local files
            existing_files: List of existing filenames in cloud cache

        Returns:
            Tuple of (should_upload: bool, reason: str)
        """
        if not existing_files:
            return (True, "No existing analysis in cloud cache")

        # Parse local file version/errors (use first file if multiple)
        local_filename = local_files["xlsx"][0].name
        local_version, local_errors = self.parse_version_and_errors(local_filename)

        if local_version is None or local_errors is None:
            return (False, f"Could not parse version/errors from local file: {local_filename}")

        # Parse existing file version/errors (use first file if multiple)
        existing_filename = existing_files[0]
        existing_version, existing_errors = self.parse_version_and_errors(existing_filename)

        if existing_version is None or existing_errors is None:
            # Existing file has no version info - allow upload
            return (
                True,
                f"Existing file has no version info, uploading v{local_version} with {local_errors} errors",
            )

        # Compare versions
        version_cmp = self.compare_versions(local_version, existing_version)

        if version_cmp > 0:
            # Local version is newer
            return (True, f"Local version {local_version} > existing {existing_version}")
        elif version_cmp < 0:
            # Local version is older
            return (
                False,
                f"Local version {local_version} < existing {existing_version} - not uploading",
            )
        else:
            # Versions are equal - compare error counts
            if local_errors < existing_errors:
                # Upload if fewer errors
                return (
                    True,
                    f"Same version ({local_version}) but fewer errors ({local_errors} vs {existing_errors})",
                )
            elif local_errors == existing_errors:
                # Don't upload if same version AND same error count (no improvement)
                return (
                    False,
                    f"Analysis matches existing cloud cache (v{local_version}, {local_errors} errors)",
                )
            else:
                # Don't upload if more errors
                return (
                    False,
                    f"Same version ({local_version}) but more errors ({local_errors} vs {existing_errors}) - not uploading",
                )

    def upload_to_staging(
        self,
        country: str,
        region: Optional[str],
        local_files: Dict,
        scope_type: str,
        datasets_used: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Upload analysis files to staging branch and create pull request.

        Args:
            country: Country name
            region: Region/state name (for USA states)
            local_files: Dict with 'xlsx' (list of paths) and 'csv' (path to error log txt, or None)
            scope_type: Type of analysis
            datasets_used: List of elevation datasets that were actually used during analysis

        Returns:
            PR URL if successful, None otherwise
        """
        cache_path = self.get_cache_path(country, region, scope_type)

        if cache_path is None:
            print("  ✗ Full USA analysis too large for cloud cache")
            return None

        # Check if we have xlsx files to upload (analysis might have failed)
        if not local_files.get("xlsx") or len(local_files["xlsx"]) == 0:
            print("  ✗ No Excel files to upload (analysis may have failed)")
            print("     Check output/ directory for error logs")
            return None

        # Check if existing analysis in cloud cache and validate version/errors
        print("\n  Checking existing cloud cache for version/error validation...")
        cache_info = self.check_cached(country, region, scope_type)

        if cache_info["exists"]:
            existing_files = cache_info["xlsx_files"]
            should_upload, reason = self.should_upload_analysis(local_files, existing_files)

            if not should_upload:
                print(f"  ℹ️  {reason}")
                print(
                    "    Your analysis matches what's already in the cloud cache - no upload needed"
                )
                return None
            else:
                print(f"  ✓ Upload validation passed: {reason}")
        else:
            print("  ✓ No existing analysis found - upload allowed")

        # Create staging branch name
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        location_name = region if scope_type == "region" else country
        branch_name = f"staging/{self.sanitize_name(location_name)}-{timestamp}"

        print(f"\n  Creating staging branch: {branch_name}")

        if not self.github.create_branch(branch_name):
            print("  ✗ Failed to create staging branch")
            return None

        print("  ✓ Created branch")

        # Collect old files to delete if we're replacing an existing analysis
        is_update = cache_info["exists"]
        deleted_files = []  # Track deleted files for PR description

        if is_update and cache_info["xlsx_files"]:
            deleted_files.extend(cache_info["xlsx_files"])
            if cache_info.get("csv_file"):
                deleted_files.append(cache_info["csv_file"])

        # Collect all files to upload
        all_files_to_upload = list(local_files["xlsx"])
        if local_files["csv"]:
            all_files_to_upload.append(local_files["csv"])

        # Calculate total size for display
        total_size_mb = sum(f.stat().st_size for f in all_files_to_upload) / 1024**2
        print(
            f"\n  Uploading {len(all_files_to_upload)} file(s) ({total_size_mb:.1f} MB total) via git push with LFS..."
        )

        # Get app version for commit message
        try:
            from __version__ import __version__ as app_version
        except ImportError:
            app_version = "unknown"

        # Build commit message with version and date
        date_str = datetime.now().strftime("%Y-%m-%d")
        commit_message = f"{'Update' if is_update else 'Add'} {location_name} climb analysis (v{app_version}, {date_str})"

        # Upload using git push with LFS support
        if not self.github.upload_files_via_git(
            local_files=all_files_to_upload,
            repo_path=cache_path,
            branch_name=branch_name,
            commit_message=commit_message,
            delete_old_files=deleted_files if deleted_files else None,
        ):
            print("  ✗ Failed to upload files via git")
            # Cleanup: delete branch
            self.github.delete_branch(branch_name)
            return None

        # Track uploaded files for PR description
        uploaded_files = [f.name for f in all_files_to_upload]

        # Create pull request
        print("\n  Creating pull request...")

        # Parse local version and error count for PR title/body
        local_filename = local_files["xlsx"][0].name
        local_version, local_errors = self.parse_version_and_errors(local_filename)

        # Determine if this is an update or new analysis
        is_update = cache_info["exists"]
        pr_title = f"{'Update' if is_update else 'Add'} {location_name} climb analysis"
        if is_update and local_version:
            pr_title += f" (v{local_version})"

        # Build PR description
        total_size_mb = sum(f.stat().st_size for f in local_files["xlsx"]) / 1024**2
        if local_files["csv"]:
            total_size_mb += local_files["csv"].stat().st_size / 1024**2

        # Build files description - only mention error log if it exists
        files_desc = f"{len(local_files['xlsx'])} Excel file(s)"
        if local_files["csv"]:
            files_desc += " + error log"

        pr_body = f"""## {location_name} Climb Analysis {'Update' if is_update else ''}

**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Version:** {local_version if local_version else 'N/A'}
**Elevation Errors:** {local_errors if local_errors is not None else 'N/A'}
**Files:** {files_desc}
**Total Size:** {total_size_mb:.1f} MB

**Analysis Parameters:**
- Surface Filter: all roads
- Minimum Score: 0
- Score Type: basic
- Cycling Filter: off
"""

        # Add elevation datasets section if available
        # Try loading from checkpoint if not provided
        if not datasets_used:
            datasets_used = self.load_datasets_used_from_checkpoint(location_name)

        if datasets_used:
            pr_body += "\n**Elevation Datasets Used (priority order):**\n"
            for i, ds in enumerate(datasets_used, 1):
                pr_body += f"{i}. {ds}\n"

        # Add comparison info if updating existing analysis
        if is_update and cache_info["xlsx_files"]:
            existing_filename = cache_info["xlsx_files"][0]
            existing_version, existing_errors = self.parse_version_and_errors(existing_filename)
            if existing_version and existing_errors is not None:
                pr_body += f"""
**Replacing Existing Analysis:**
- Previous version: {existing_version}
- Previous errors: {existing_errors}
- Improvement: {"✓ Newer version" if self.compare_versions(local_version, existing_version) > 0 else f"✓ Fewer errors ({local_errors} vs {existing_errors})" if local_errors < existing_errors else "Same version/errors"}
"""

        # List deleted files if any
        if deleted_files:
            pr_body += """
**Files Removed:**
"""
            for filename in deleted_files:
                pr_body += f"- ~~`{filename}`~~ (replaced with newer version)\n"

        pr_body += """
**Files Added:**
"""
        for filename in uploaded_files:
            pr_body += f"- `{filename}`\n"

        pr_body += "\n---\n\n Generated by Climb Analyzer"

        pr_url = self.github.create_pull_request(pr_title, pr_body, branch_name)

        if pr_url:
            return pr_url
        else:
            # Cleanup: delete branch
            self.github.delete_branch(branch_name)
            return None

    def upload_to_release(
        self,
        country: str,
        region: Optional[str],
        local_files: Dict,
        scope_type: str,
        datasets_used: Optional[List[str]] = None,
        climb_count: int = 0,
    ) -> Optional[str]:
        """
        Upload analysis files to a GitHub Release (replaces LFS-based upload).

        Creates a per-region release with XLSX and SQLite files as release assets.
        If a release already exists for this region, updates it by replacing assets.

        Args:
            country: Country name
            region: Region/state name (for USA states)
            local_files: Dict with 'xlsx' (list of paths), 'sqlite' (list of paths), 'csv' (error log path or None)
            scope_type: Type of analysis
            datasets_used: List of elevation datasets used
            climb_count: Number of climbs in the analysis

        Returns:
            Release URL if successful, None otherwise
        """
        # Get app version
        try:
            from __version__ import __version__ as app_version
        except ImportError:
            app_version = "2.2.0"

        # Determine location name for release tag
        location_name = region if scope_type == "region" else country

        # Check if we have files to upload
        xlsx_files = local_files.get("xlsx", [])
        sqlite_files = local_files.get("sqlite", [])
        sqlite_checksums = local_files.get("sqlite_checksums", {})

        # Detect if SQLite files are split (have .001, .002, etc. suffixes)
        import re
        is_split_db = any(re.match(r'.*\.sqlite\.\d{3}$', str(f.name)) for f in sqlite_files)

        if not xlsx_files and not sqlite_files:
            print("  ✗ No files to upload")
            return None

        # Generate release tag (per-region, e.g., "hawaii-v2.2.0")
        sanitized_name = self.sanitize_name(location_name).lower()
        release_tag = f"{sanitized_name}-v{app_version}"

        print(f"\n  Checking for existing release: {release_tag}")

        # Check for existing release
        existing_release = self.github.get_release_by_tag(release_tag)

        if existing_release:
            print(f"  ✓ Found existing release - will update assets")
            release_id = existing_release["id"]
            upload_url = existing_release["upload_url"]

            # Delete old assets to replace with new ones
            existing_assets = self.github.list_release_assets(release_id)
            if existing_assets:
                print(f"    Removing {len(existing_assets)} existing assets...")
                for asset in existing_assets:
                    self.github.delete_release_asset(asset["id"])
        else:
            # Create new release
            print(f"  Creating release: {release_tag}")

            # Parse version and error count from first xlsx file
            if xlsx_files:
                local_version, local_errors = self.parse_version_and_errors(xlsx_files[0].name)
            else:
                local_version, local_errors = app_version, 0

            # Build release body
            date_str = datetime.now().strftime("%Y-%m-%d")

            # Calculate total size
            total_xlsx_size = sum(f.stat().st_size for f in xlsx_files) if xlsx_files else 0
            total_sqlite_size = sum(f.stat().st_size for f in sqlite_files) if sqlite_files else 0
            total_size_mb = (total_xlsx_size + total_sqlite_size) / (1024 ** 2)

            release_body = f"""## {location_name} Climb Analysis

**Date:** {date_str}
**Version:** {local_version}
**Elevation Errors:** {local_errors if local_errors is not None else 0}
**Climb Count:** {climb_count:,}

### Files

| File | Size | Format |
|------|------|--------|
"""
            for f in xlsx_files:
                size_mb = f.stat().st_size / (1024 ** 2)
                release_body += f"| {f.name} | {size_mb:.1f} MB | Excel |\n"

            for f in sqlite_files:
                size_mb = f.stat().st_size / (1024 ** 2)
                release_body += f"| {f.name} | {size_mb:.1f} MB | SQLite |\n"

            if local_files.get("csv"):
                csv_size = local_files["csv"].stat().st_size / (1024 ** 2)
                release_body += f"| {local_files['csv'].name} | {csv_size:.2f} MB | Error Log |\n"

            release_body += f"""
**Total Size:** {total_size_mb:.1f} MB

### Download

- **Excel files**: For viewing in spreadsheet applications
- **SQLite database**: For iOS app import via `ATTACH DATABASE`

### Analysis Parameters

- Surface Filter: all roads
- Minimum Score: 0
- Score Type: basic
- Cycling Filter: off
"""

            # Add elevation datasets if available
            if not datasets_used:
                datasets_used = self.load_datasets_used_from_checkpoint(location_name)

            if datasets_used:
                release_body += "\n### Elevation Datasets Used (priority order)\n\n"
                for i, ds in enumerate(datasets_used, 1):
                    release_body += f"{i}. {ds}\n"

            # Add split database section if files were split
            if is_split_db and sqlite_files:
                release_body += "\n### Split Database Files\n\n"
                release_body += "The SQLite database exceeds GitHub's 2GB limit and has been split:\n\n"
                release_body += "| File | Size | SHA256 |\n"
                release_body += "|------|------|--------|\n"

                for f in sorted(sqlite_files, key=lambda x: x.name):
                    size_gb = f.stat().st_size / (1024 ** 3)
                    checksum = sqlite_checksums.get(f.name, "")
                    # Show first 16 chars of checksum for readability
                    checksum_short = f"`{checksum[:16]}...`" if checksum else ""
                    release_body += f"| {f.name} | {size_gb:.2f} GB | {checksum_short} |\n"

                # Get base filename for reassembly instructions
                base_name = sqlite_files[0].name.rsplit('.', 1)[0]  # Remove .001
                chunk_names = " ".join(f.name for f in sorted(sqlite_files, key=lambda x: x.name))
                release_body += f"\n**To reassemble:**\n```bash\ncat {chunk_names} > {base_name}\n```\n"

            release_body += "\n---\n\nGenerated by [Climb Analyzer](https://github.com/stevehollx/climb-analyzer)"

            # Create DRAFT release (requires PR approval to publish)
            release = self.github.create_release(
                tag_name=release_tag,
                name=f"{location_name} Climb Analysis v{app_version}",
                body=release_body,
                draft=True,  # Draft until PR is approved
                prerelease=False,
            )

            if not release:
                print("  ✗ Failed to create release")
                return None

            release_id = release["id"]
            upload_url = release["upload_url"]
            print(f"  ✓ Created release: {release['html_url']}")

        # Upload all files as release assets
        print(f"\n  Uploading files to release...")

        all_files = []
        all_files.extend(xlsx_files)
        all_files.extend(sqlite_files)
        if local_files.get("csv"):
            all_files.append(local_files["csv"])

        # Add .sha256 checksum file if split database exists
        if is_split_db and sqlite_files:
            # Checksum file is named {original_sqlite_name}.sha256
            # e.g., California.sqlite.sha256 (not California.sqlite.001.sha256)
            first_chunk = sorted(sqlite_files, key=lambda x: x.name)[0]
            base_sqlite_name = first_chunk.name.rsplit('.', 1)[0]  # Remove .001
            checksum_file = first_chunk.parent / f"{base_sqlite_name}.sha256"
            if checksum_file.exists():
                all_files.append(checksum_file)

        uploaded_count = 0
        for local_file in all_files:
            # Determine content type
            if local_file.suffix == ".xlsx":
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif local_file.suffix == ".sqlite" or re.match(r'.*\.sqlite\.\d{3}$', local_file.name):
                # Handle both single .sqlite and split chunks (.sqlite.001, .sqlite.002, etc.)
                content_type = "application/x-sqlite3"
            elif local_file.name.endswith(".sha256"):
                content_type = "text/plain"
            else:
                content_type = "text/plain"

            asset = self.github.upload_release_asset(
                release_id=release_id,
                upload_url=upload_url,
                local_file=local_file,
                content_type=content_type,
            )

            if asset:
                uploaded_count += 1
            else:
                print(f"    ⚠️  Failed to upload {local_file.name}")

        if uploaded_count == len(all_files):
            print(f"  ✓ Successfully uploaded {uploaded_count} files to release (draft)")

            # Get release info by ID (draft releases don't have tags until published)
            release_info = self.github.get_release_by_id(release_id)
            if release_info:
                # Build markdown file content
                md_content = self._build_release_markdown(
                    location_name=location_name,
                    version=app_version,
                    climb_count=climb_count,
                    release_tag=release_tag,
                    release_url=release_info["html_url"],
                    assets=release_info.get("assets", []),
                    datasets_used=datasets_used,
                    elevation_errors=local_errors if local_errors is not None else 0,
                )

                # Create PR with markdown file
                pr_url = self._create_release_pr(
                    location_name=location_name,
                    md_content=md_content,
                    release_tag=release_tag,
                    country=country,
                    scope_type=scope_type,
                )

                if pr_url:
                    print(f"  ✓ Created PR for release approval: {pr_url}")
                    return pr_url
        else:
            print(f"  ⚠️  Uploaded {uploaded_count}/{len(all_files)} files")

        # Fallback: return release URL if PR creation failed
        release_url = f"https://github.com/{self.github.owner}/{self.github.repo}/releases/tag/{release_tag}"
        return release_url

    def _build_release_markdown(
        self,
        location_name: str,
        version: str,
        climb_count: int,
        release_tag: str,
        release_url: str,
        assets: List[Dict],
        datasets_used: Optional[List[str]] = None,
        elevation_errors: int = 0,
    ) -> str:
        """Build markdown content for the release PR in user-preferred format."""
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Calculate file counts and total size
        xlsx_count = sum(1 for a in assets if a.get("name", "").endswith(".xlsx"))
        sqlite_count = sum(1 for a in assets if a.get("name", "").endswith(".sqlite"))
        log_count = sum(1 for a in assets if "error" in a.get("name", "").lower() or a.get("name", "").endswith(".log"))
        total_size_bytes = sum(a.get("size", 0) for a in assets)
        total_size_mb = total_size_bytes / (1024 ** 2)

        # Build files description
        file_parts = []
        if xlsx_count > 0:
            file_parts.append(f"{xlsx_count} Excel file(s)")
        if sqlite_count > 0:
            file_parts.append(f"{sqlite_count} SQLite database(s)")
        if log_count > 0:
            file_parts.append("error log")
        files_desc = " + ".join(file_parts) if file_parts else "No files"

        # Start building markdown
        md = f"""## Info
* Date: {date_str}
* Version: {version}
* Elevation Errors: {elevation_errors}
* Climbs: {climb_count:,}
* Files: {files_desc}
* Total Size: {total_size_mb:.1f} MB
* Release Tag: `{release_tag}`

"""

        # Elevation datasets section
        if datasets_used:
            md += "## Elevation datasets used:\n"
            for i, dataset in enumerate(datasets_used, 1):
                md += f"{i}. {dataset}\n"
            md += "\n"

        # Files table
        md += """## Files

| File | Size | Format |
|------|------|--------|
"""
        for asset in assets:
            name = asset.get("name", "")
            size_bytes = asset.get("size", 0)
            size_mb = size_bytes / (1024 ** 2)
            download_url = asset.get("browser_download_url", "")

            if name.endswith(".xlsx"):
                fmt = "Excel"
            elif name.endswith(".sqlite"):
                fmt = "SQLite"
            elif name.endswith(".log") or "error" in name.lower():
                fmt = "Log"
            else:
                fmt = "Other"

            md += f"| [{name}]({download_url}) | {size_mb:.1f} MB | {fmt} |\n"

        md += f"""
## Release

[View Release]({release_url})

---

*This README was automatically generated by Climb Analyzer.*
*Merging this PR will publish the draft release and update the index.*
"""
        return md

    def _create_release_pr(
        self,
        location_name: str,
        md_content: str,
        release_tag: str,
        country: str = "",
        scope_type: str = "",
    ) -> Optional[str]:
        """Create a PR with the release markdown file."""
        # Sanitize location name for filename
        sanitized_name = self.sanitize_name(location_name).lower()

        # Use geographic path for US states, otherwise use releases folder
        if scope_type == "region" and country.lower() in ["usa", "united states", "united states of america"]:
            md_filename = f"north-america/united-states-of-america/{sanitized_name}/README.md"
        else:
            md_filename = f"releases/{sanitized_name}.md"

        # Create unique branch name
        timestamp = int(datetime.now().timestamp())
        branch_name = f"release/{sanitized_name}-{timestamp}"

        print(f"    Creating branch: {branch_name}")

        # Create branch
        if not self.github.create_branch(branch_name):
            print(f"    ⚠️  Failed to create branch {branch_name}")
            return None

        # Commit markdown file to branch
        commit_message = f"Add release info for {location_name}"
        if not self.github.commit_file(
            branch_name=branch_name,
            file_path=md_filename,
            content=md_content,
            commit_message=commit_message,
        ):
            print(f"    ⚠️  Failed to commit {md_filename}")
            self.github.delete_branch(branch_name)
            return None

        # Create PR - use the same content as the release README
        pr_title = f"Release: {location_name} Climb Analysis"
        pr_url = self.github.create_pull_request(pr_title, md_content, branch_name)

        if not pr_url:
            print(f"    ⚠️  Failed to create PR")
            self.github.delete_branch(branch_name)
            return None

        return pr_url
