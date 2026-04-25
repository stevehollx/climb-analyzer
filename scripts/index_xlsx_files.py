#!/usr/bin/env python3
"""
Index Release Assets Script

This script scans GitHub Releases for XLSX and SQLite files and creates an index.json
containing region information, file metadata, and download URLs.

The script handles per-region releases (e.g., "hawaii-v2.2.0", "california-v2.2.0")
and extracts region name, version, and file information from release assets.

Index format uses path-based region keys (e.g., "north-america/us/hawaii") for iOS
compatibility with geographic hierarchy navigation.
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests


# Mapping from release tag region names to geographic paths
# Used to create path-based keys in index.json
# Format: "tag-name" -> "continent/country/region" or "continent/region"
US_STATES_TO_PATH = {
    "alabama": "north-america/united-states-of-america/alabama",
    "alaska": "north-america/united-states-of-america/alaska",
    "arizona": "north-america/united-states-of-america/arizona",
    "arkansas": "north-america/united-states-of-america/arkansas",
    "california": "north-america/united-states-of-america/california",
    "colorado": "north-america/united-states-of-america/colorado",
    "connecticut": "north-america/united-states-of-america/connecticut",
    "delaware": "north-america/united-states-of-america/delaware",
    "district-of-columbia": "north-america/united-states-of-america/district-of-columbia",
    "florida": "north-america/united-states-of-america/florida",
    "georgia": "north-america/united-states-of-america/georgia",
    "hawaii": "north-america/united-states-of-america/hawaii",
    "idaho": "north-america/united-states-of-america/idaho",
    "illinois": "north-america/united-states-of-america/illinois",
    "indiana": "north-america/united-states-of-america/indiana",
    "iowa": "north-america/united-states-of-america/iowa",
    "kansas": "north-america/united-states-of-america/kansas",
    "kentucky": "north-america/united-states-of-america/kentucky",
    "louisiana": "north-america/united-states-of-america/louisiana",
    "maine": "north-america/united-states-of-america/maine",
    "maryland": "north-america/united-states-of-america/maryland",
    "massachusetts": "north-america/united-states-of-america/massachusetts",
    "michigan": "north-america/united-states-of-america/michigan",
    "minnesota": "north-america/united-states-of-america/minnesota",
    "mississippi": "north-america/united-states-of-america/mississippi",
    "missouri": "north-america/united-states-of-america/missouri",
    "montana": "north-america/united-states-of-america/montana",
    "nebraska": "north-america/united-states-of-america/nebraska",
    "nevada": "north-america/united-states-of-america/nevada",
    "new-hampshire": "north-america/united-states-of-america/new-hampshire",
    "new-jersey": "north-america/united-states-of-america/new-jersey",
    "new-mexico": "north-america/united-states-of-america/new-mexico",
    "new-york": "north-america/united-states-of-america/new-york",
    "north-carolina": "north-america/united-states-of-america/north-carolina",
    "north-dakota": "north-america/united-states-of-america/north-dakota",
    "ohio": "north-america/united-states-of-america/ohio",
    "oklahoma": "north-america/united-states-of-america/oklahoma",
    "oregon": "north-america/united-states-of-america/oregon",
    "pennsylvania": "north-america/united-states-of-america/pennsylvania",
    "rhode-island": "north-america/united-states-of-america/rhode-island",
    "south-carolina": "north-america/united-states-of-america/south-carolina",
    "south-dakota": "north-america/united-states-of-america/south-dakota",
    "tennessee": "north-america/united-states-of-america/tennessee",
    "texas": "north-america/united-states-of-america/texas",
    "utah": "north-america/united-states-of-america/utah",
    "vermont": "north-america/united-states-of-america/vermont",
    "virginia": "north-america/united-states-of-america/virginia",
    "washington": "north-america/united-states-of-america/washington",
    "west-virginia": "north-america/united-states-of-america/west-virginia",
    "wisconsin": "north-america/united-states-of-america/wisconsin",
    "wyoming": "north-america/united-states-of-america/wyoming",
    # US territories
    "puerto-rico": "north-america/united-states-of-america/puerto-rico",
    "us-virgin-islands": "north-america/united-states-of-america/us-virgin-islands",
}

# European countries
EUROPE_COUNTRIES_TO_PATH = {
    "albania": "europe/albania",
    "andorra": "europe/andorra",
    "austria": "europe/austria",
    "azores": "europe/azores",
    "belarus": "europe/belarus",
    "belgium": "europe/belgium",
    "bosnia-herzegovina": "europe/bosnia-herzegovina",
    "bulgaria": "europe/bulgaria",
    "croatia": "europe/croatia",
    "cyprus": "europe/cyprus",
    "czech-republic": "europe/czech-republic",
    "denmark": "europe/denmark",
    "estonia": "europe/estonia",
    "faroe-islands": "europe/faroe-islands",
    "finland": "europe/finland",
    "france": "europe/france",
    "georgia": "europe/georgia",
    "germany": "europe/germany",
    "great-britain": "europe/great-britain",
    "greece": "europe/greece",
    "hungary": "europe/hungary",
    "iceland": "europe/iceland",
    "ireland-and-northern-ireland": "europe/ireland-and-northern-ireland",
    "isle-of-man": "europe/isle-of-man",
    "italy": "europe/italy",
    "kosovo": "europe/kosovo",
    "latvia": "europe/latvia",
    "liechtenstein": "europe/liechtenstein",
    "lithuania": "europe/lithuania",
    "luxembourg": "europe/luxembourg",
    "macedonia": "europe/macedonia",
    "malta": "europe/malta",
    "moldova": "europe/moldova",
    "monaco": "europe/monaco",
    "montenegro": "europe/montenegro",
    "netherlands": "europe/netherlands",
    "norway": "europe/norway",
    "poland": "europe/poland",
    "portugal": "europe/portugal",
    "romania": "europe/romania",
    "russia": "europe/russia",
    "serbia": "europe/serbia",
    "slovakia": "europe/slovakia",
    "slovenia": "europe/slovenia",
    "spain": "europe/spain",
    "sweden": "europe/sweden",
    "switzerland": "europe/switzerland",
    "turkey": "europe/turkey",
    "ukraine": "europe/ukraine",
}

# Other regions
OTHER_REGIONS_TO_PATH = {
    # Canada
    "canada": "north-america/canada",
    "alberta": "north-america/canada/alberta",
    "british-columbia": "north-america/canada/british-columbia",
    "ontario": "north-america/canada/ontario",
    "quebec": "north-america/canada/quebec",
    # Oceania
    "australia": "oceania/australia",
    "new-zealand": "oceania/new-zealand",
    # Asia
    "japan": "asia/japan",
    # South America
    "brazil": "south-america/brazil",
    "argentina": "south-america/argentina",
    "chile": "south-america/chile",
    "colombia": "south-america/colombia",
}


def get_region_path(tag_region: str) -> str:
    """
    Map a release tag region name to its full geographic path.

    Args:
        tag_region: Region name from release tag (e.g., "hawaii", "france")

    Returns:
        Full geographic path (e.g., "north-america/us/hawaii")
    """
    tag_lower = tag_region.lower()

    # Check US states first (most common)
    if tag_lower in US_STATES_TO_PATH:
        return US_STATES_TO_PATH[tag_lower]

    # Check European countries
    if tag_lower in EUROPE_COUNTRIES_TO_PATH:
        return EUROPE_COUNTRIES_TO_PATH[tag_lower]

    # Check other regions
    if tag_lower in OTHER_REGIONS_TO_PATH:
        return OTHER_REGIONS_TO_PATH[tag_lower]

    # Continent-prefix fallback: e.g. "canada-alberta" -> "north-america/canada/alberta",
    # "north-america-mexico" -> "north-america/mexico", "europe-france" -> "europe/france".
    # This catches tags that follow the "<continent-or-country>-<rest>" convention without
    # requiring every permutation to be hardcoded above.
    continent_prefixes = {
        "north-america": "north-america",
        "south-america": "south-america",
        "europe": "europe",
        "asia": "asia",
        "africa": "africa",
        "oceania": "oceania",
        "antarctica": "antarctica",
        # Country-level prefixes that nest under a continent
        "canada": "north-america/canada",
        "usa": "north-america/united-states-of-america",
        "us": "north-america/united-states-of-america",
        "france": "europe/france",
        "germany": "europe/germany",
        "italy": "europe/italy",
        "spain": "europe/spain",
    }
    for prefix, path_root in continent_prefixes.items():
        if tag_lower.startswith(prefix + "-"):
            remainder = tag_lower[len(prefix) + 1:]
            if remainder:
                return f"{path_root}/{remainder}"

    # Default: return as-is if not found (for custom/unknown regions)
    return tag_lower


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment."""
    return os.environ.get("GITHUB_TOKEN")


def extract_region_from_tag(tag_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract region name and version from a release tag.

    Examples:
        hawaii-v2.2.0 -> ("hawaii", "2.2.0")
        california-v2.1.0 -> ("california", "2.1.0")
        new-york-v2.2.0 -> ("new-york", "2.2.0")

    Returns:
        Tuple of (region_name, version) or (None, None) if not a valid tag
    """
    # Pattern: region-name-vX.Y.Z
    match = re.match(r'^(.+?)-v(\d+\.\d+\.\d+)$', tag_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def extract_region_from_filename(filename: str) -> Optional[str]:
    """
    Extract the region name from a filename.
    The region name is the word before "_climbs".

    Examples:
        Hawaii_climbs_all-surfaces_all-access_imperial_2025-01-03_v2.2.0_e0000.xlsx -> Hawaii
        New_York_climbs_all_basic_2025-11-01_v2.1.0_e0000-1.xlsx -> New_York
    """
    match = re.match(r'^(.+?)_climbs', filename)
    if match:
        return match.group(1)
    return None


def extract_climb_count_from_release_body(body: str) -> Optional[int]:
    """
    Extract climb count from release body markdown.

    Handles both old format "**Climb Count:** 1,234" and new format "* Climbs: 1,234"
    """
    # Try old format first: **Climb Count:** 1,234
    match = re.search(r'\*\*Climb Count:\*\*\s*([\d,]+)', body)
    if match:
        return int(match.group(1).replace(',', ''))
    # Try new bulletized format: * Climbs: 1,234
    match = re.search(r'\* Climbs:\s*([\d,]+)', body)
    if match:
        return int(match.group(1).replace(',', ''))
    return None


def extract_elevation_errors_from_release_body(body: str) -> Optional[int]:
    """
    Extract elevation errors from release body markdown.

    Handles both old format "**Elevation Errors:** 0" and new format (not yet defined)
    """
    match = re.search(r'\*\*Elevation Errors:\*\*\s*(\d+)', body)
    if match:
        return int(match.group(1))
    return None


def _partition_id_to_display_name(partition_id: str) -> str:
    """
    Convert a partition ID to a human-readable display name.

    Examples:
        norcal -> Northern California
        socal -> Southern California
        northeast -> Northeast
        ne_sw -> Southern Northeast
        other -> Other Regions
    """
    # Map of known partition IDs to display names
    display_names = {
        # Geofabrik-style
        "norcal": "Northern California",
        "socal": "Southern California",
        "north": "North",
        "south": "South",
        "east": "East",
        "west": "West",
        # Quadtree-style (full names)
        "northeast": "Northeast",
        "northwest": "Northwest",
        "southeast": "Southeast",
        "southwest": "Southwest",
        # Quadtree-style (abbreviated)
        "ne": "Northeast",
        "nw": "Northwest",
        "se": "Southeast",
        "sw": "Southwest",
        # Catch-all
        "other": "Other Regions",
    }

    if partition_id in display_names:
        return display_names[partition_id]

    # Handle nested quadtree patterns (e.g., ne_sw, northeast_se)
    if "_" in partition_id:
        parts = partition_id.split("_")
        if len(parts) == 2:
            outer = display_names.get(parts[0], parts[0].title())
            inner = parts[1].upper() if len(parts[1]) <= 2 else parts[1].title()

            # Generate directional prefix
            inner_map = {"ne": "Eastern", "nw": "Western", "se": "Eastern", "sw": "Western",
                        "northeast": "Eastern", "northwest": "Western",
                        "southeast": "Eastern", "southwest": "Western"}
            prefix = inner_map.get(parts[1], "")
            if not prefix:
                if "north" in parts[1]:
                    prefix = "Northern"
                elif "south" in parts[1]:
                    prefix = "Southern"
                else:
                    prefix = parts[1].title()
            return f"{prefix} {outer}"

    # Fallback: title case the ID
    return partition_id.replace("_", " ").title()


def fetch_checksums(sha256_url: str, token: Optional[str] = None) -> Dict[str, str]:
    """
    Fetch and parse .sha256 checksum file from release assets.

    Args:
        sha256_url: URL to the .sha256 file
        token: Optional GitHub token for authentication

    Returns:
        Dict mapping filename to SHA256 checksum
    """
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(sha256_url, headers=headers, timeout=30)
        if response.status_code == 200:
            checksums = {}
            for line in response.text.strip().split('\n'):
                # Standard format: "sha256hash  filename" (two spaces)
                parts = line.split('  ')
                if len(parts) == 2:
                    checksums[parts[1].strip()] = parts[0].strip()
            return checksums
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to fetch checksums from {sha256_url}: {e}")

    return {}


def fetch_partition_metadata(metadata_url: str, token: Optional[str] = None) -> Dict:
    """
    Fetch partition metadata JSON from release asset.

    Args:
        metadata_url: URL to .sqlite.metadata.json file
        token: Optional GitHub token for authentication

    Returns:
        Parsed metadata dict or empty dict on error
    """
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(metadata_url, headers=headers, timeout=30)
        if response.status_code == 200:
            return json.loads(response.text)
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"  Warning: Could not fetch partition metadata: {e}")

    return {}


def fetch_releases(owner: str, repo: str, token: Optional[str] = None) -> List[Dict]:
    """
    Fetch all releases from a GitHub repository.

    Args:
        owner: Repository owner
        repo: Repository name
        token: Optional GitHub token for authentication

    Returns:
        List of release data dicts
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    releases = []
    page = 1
    per_page = 100

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        params = {"per_page": per_page, "page": page}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)

            if response.status_code == 200:
                page_releases = response.json()
                if not page_releases:
                    break
                releases.extend(page_releases)
                if len(page_releases) < per_page:
                    break
                page += 1
            else:
                print(f"Failed to fetch releases: HTTP {response.status_code}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Network error fetching releases: {e}")
            break

    return releases


def scan_releases(owner: str, repo: str, token: Optional[str] = None) -> Dict[str, Dict]:
    """
    Scan GitHub Releases for climb data files.

    Returns:
        Dictionary keyed by region name with file information
    """
    releases = fetch_releases(owner, repo, token)
    index_data = {}

    for release in releases:
        tag_name = release.get("tag_name", "")
        region_from_tag, version = extract_region_from_tag(tag_name)

        if not region_from_tag:
            # Skip non-region releases (e.g., app version releases)
            continue

        # Get release metadata
        release_url = release.get("html_url", "")
        published_at = release.get("published_at", "")
        release_body = release.get("body", "")

        # Extract climb count and errors from release body
        climb_count = extract_climb_count_from_release_body(release_body)
        elevation_errors = extract_elevation_errors_from_release_body(release_body)

        # Process assets
        assets = release.get("assets", [])
        xlsx_files = []
        xlsx_sizes = []
        xlsx_urls = []
        sqlite_file = None
        sqlite_size = 0
        sqlite_url = None

        # Split SQLite detection (binary chunks: .001, .002, etc.)
        sqlite_split_files = []
        sqlite_split_sizes = []
        sqlite_split_urls = []
        sha256_url = None

        # Gzipped SQLite detection (.sqlite.gz single-file or .sqlite.gz.001/.002 split)
        sqlite_gz_file = None
        sqlite_gz_size = 0
        sqlite_gz_url = None
        sqlite_gz_split_files = []
        sqlite_gz_split_sizes = []
        sqlite_gz_split_urls = []
        sqlite_gz_sha256_url = None

        # Partitioned SQLite detection (geographic partitions: _norcal, _socal, _northeast, etc.)
        sqlite_partition_files = []
        sqlite_partition_sizes = []
        sqlite_partition_urls = []
        sqlite_partition_ids = []
        partition_metadata_urls = {}  # partition_id -> metadata_url

        region_name = None  # Will be extracted from filename

        # Known partition ID patterns (from Geofabrik and Quadtree)
        partition_patterns = [
            # Geofabrik-based
            "norcal", "socal", "north", "south", "east", "west",
            # Quadtree-based
            "northeast", "northwest", "southeast", "southwest",
            "ne", "nw", "se", "sw",
            # Nested quadtree (e.g., ne_sw, southwest_ne)
            r"[ns][ew]_[ns][ew]",
            r"(northeast|northwest|southeast|southwest)_[ns][ew]",
            # "other" catch-all
            "other",
        ]
        partition_regex = re.compile(
            r'_(' + '|'.join(partition_patterns) + r')\.sqlite$', re.IGNORECASE
        )

        for asset in assets:
            name = asset.get("name", "")
            size = asset.get("size", 0)
            download_url = asset.get("browser_download_url", "")

            if name.endswith(".xlsx"):
                xlsx_files.append(name)
                xlsx_sizes.append(size)
                xlsx_urls.append(download_url)

                # Extract region name from xlsx filename
                if not region_name:
                    region_name = extract_region_from_filename(name)

            elif re.match(r'.*\.sqlite\.gz\.\d{3}$', name):
                # Split gzipped SQLite chunk (.sqlite.gz.001, .sqlite.gz.002, ...)
                sqlite_gz_split_files.append(name)
                sqlite_gz_split_sizes.append(size)
                sqlite_gz_split_urls.append(download_url)
                if not region_name:
                    # Remove ".001" to get ".sqlite.gz", then back to base name
                    base_name = name.rsplit('.', 1)[0].replace('.gz', '')
                    region_name = extract_region_from_filename(base_name)

            elif name.endswith(".sqlite.gz"):
                # Single-file gzipped SQLite (not split)
                sqlite_gz_file = name
                sqlite_gz_size = size
                sqlite_gz_url = download_url
                if not region_name:
                    base_name = name[:-len('.gz')]  # strip ".gz" -> ".sqlite"
                    region_name = extract_region_from_filename(base_name)

            elif name.endswith(".sqlite.gz.sha256"):
                # Checksum file for split gzipped SQLite
                sqlite_gz_sha256_url = download_url

            elif re.match(r'.*\.sqlite\.\d{3}$', name):
                # Split SQLite chunk (.sqlite.001, .sqlite.002, etc.)
                sqlite_split_files.append(name)
                sqlite_split_sizes.append(size)
                sqlite_split_urls.append(download_url)

                # Extract region name from sqlite filename if not yet found
                if not region_name:
                    # Remove the .001 suffix to get the base name
                    base_name = name.rsplit('.', 1)[0]  # e.g., California_climbs_....sqlite
                    region_name = extract_region_from_filename(base_name)

            elif name.endswith(".sqlite.sha256"):
                # Checksum file for split SQLite
                sha256_url = download_url

            elif name.endswith(".sqlite.metadata.json"):
                # Partition metadata file
                # e.g., "California_climbs_..._norcal.sqlite.metadata.json" -> "norcal"
                base = name.replace(".sqlite.metadata.json", "")
                partition_match = partition_regex.search(base + ".sqlite")
                if partition_match:
                    part_id = partition_match.group(1).lower()
                    partition_metadata_urls[part_id] = download_url

            elif name.endswith(".sqlite"):
                # Check if this is a partitioned file
                partition_match = partition_regex.search(name)
                if partition_match:
                    partition_id = partition_match.group(1).lower()
                    sqlite_partition_files.append(name)
                    sqlite_partition_sizes.append(size)
                    sqlite_partition_urls.append(download_url)
                    sqlite_partition_ids.append(partition_id)

                    # Extract region name from partition filename if not yet found
                    if not region_name:
                        # Remove the partition suffix to get base region name
                        base_name = partition_regex.sub('.sqlite', name)
                        region_name = extract_region_from_filename(base_name)
                else:
                    # Regular single SQLite file
                    sqlite_file = name
                    sqlite_size = size
                    sqlite_url = download_url

                    # Extract region name from sqlite filename if not yet found
                    if not region_name:
                        region_name = extract_region_from_filename(name)

        # Determine if SQLite is split or partitioned
        is_split_db = len(sqlite_split_files) > 0
        is_partitioned_db = len(sqlite_partition_files) > 0

        # Skip if no files found
        if not xlsx_files and not sqlite_file and not sqlite_split_files and not sqlite_partition_files:
            continue

        # Use tag-based region name if filename extraction failed
        if not region_name:
            region_name = region_from_tag.replace("-", "_").title()

        # Sort xlsx files for consistent ordering (handles split files)
        if xlsx_files:
            sorted_indices = sorted(
                range(len(xlsx_files)),
                key=lambda i: (
                    xlsx_files[i].replace("-1.xlsx", ".xlsx").replace("-2.xlsx", ".xlsx"),
                    0 if not re.search(r'-(\d+)\.xlsx$', xlsx_files[i]) else int(re.search(r'-(\d+)\.xlsx$', xlsx_files[i]).group(1))
                )
            )
            xlsx_files = [xlsx_files[i] for i in sorted_indices]
            xlsx_sizes = [xlsx_sizes[i] for i in sorted_indices]
            xlsx_urls = [xlsx_urls[i] for i in sorted_indices]

        # Sort split SQLite files for consistent ordering (.001, .002, .003, etc.)
        if sqlite_split_files:
            sorted_indices = sorted(
                range(len(sqlite_split_files)),
                key=lambda i: sqlite_split_files[i]
            )
            sqlite_split_files = [sqlite_split_files[i] for i in sorted_indices]
            sqlite_split_sizes = [sqlite_split_sizes[i] for i in sorted_indices]
            sqlite_split_urls = [sqlite_split_urls[i] for i in sorted_indices]

        # Sort split gzipped SQLite chunks (.gz.001, .gz.002, ...)
        if sqlite_gz_split_files:
            sorted_indices = sorted(
                range(len(sqlite_gz_split_files)),
                key=lambda i: sqlite_gz_split_files[i]
            )
            sqlite_gz_split_files = [sqlite_gz_split_files[i] for i in sorted_indices]
            sqlite_gz_split_sizes = [sqlite_gz_split_sizes[i] for i in sorted_indices]
            sqlite_gz_split_urls = [sqlite_gz_split_urls[i] for i in sorted_indices]

        is_split_gz = len(sqlite_gz_split_files) > 0
        has_gz = sqlite_gz_file is not None or is_split_gz

        # Sort partitioned SQLite files for consistent ordering (alphabetically by partition_id)
        partitions_data = []
        if sqlite_partition_files:
            sorted_indices = sorted(
                range(len(sqlite_partition_files)),
                key=lambda i: sqlite_partition_ids[i]
            )
            sqlite_partition_files = [sqlite_partition_files[i] for i in sorted_indices]
            sqlite_partition_sizes = [sqlite_partition_sizes[i] for i in sorted_indices]
            sqlite_partition_urls = [sqlite_partition_urls[i] for i in sorted_indices]
            sqlite_partition_ids = [sqlite_partition_ids[i] for i in sorted_indices]

            # Build partition info for each partition
            for i, partition_id in enumerate(sqlite_partition_ids):
                # Generate display name from partition_id
                display_name = _partition_id_to_display_name(partition_id)

                # Fetch metadata if available
                metadata = {}
                if partition_id in partition_metadata_urls:
                    metadata = fetch_partition_metadata(
                        partition_metadata_urls[partition_id], token
                    )

                # Extract bounds from metadata
                bounds_data = None
                if metadata.get("bounds"):
                    b = metadata["bounds"]
                    bounds_data = {
                        "minLat": b.get("min_lat"),
                        "minLon": b.get("min_lon"),
                        "maxLat": b.get("max_lat"),
                        "maxLon": b.get("max_lon"),
                    }

                partitions_data.append({
                    "partition_id": partition_id,
                    "display_name": display_name,
                    "database_file": sqlite_partition_files[i],
                    "database_size": sqlite_partition_sizes[i],
                    "database_url": sqlite_partition_urls[i],
                    "database_sha256": metadata.get("sha256"),
                    "bounds": bounds_data,
                    "climb_count": metadata.get("climb_count"),
                    "size_mb": metadata.get("file_size_mb") or round(sqlite_partition_sizes[i] / (1024**2), 1),
                    "metadata_available": bool(metadata),
                })

        # Fetch checksums if split database and checksum file exists
        split_checksums = []
        if is_split_db and sha256_url:
            checksums_dict = fetch_checksums(sha256_url, token)
            # Build ordered list of checksums matching split_files order
            for filename in sqlite_split_files:
                split_checksums.append(checksums_dict.get(filename, ""))

        # Build region entry using path-based key (e.g., "north-america/us/hawaii")
        region_key = get_region_path(region_from_tag)

        # Determine if this is a split file set
        has_split_files = len(xlsx_files) > 1

        # Extract last_updated date from published_at
        last_updated = published_at[:10] if published_at else None

        # Determine partition type if partitioned
        partition_type = None
        if is_partitioned_db:
            # Check if using Geofabrik names (norcal, socal) or Quadtree names (northeast, etc.)
            geofabrik_ids = {"norcal", "socal", "north", "south", "east", "west"}
            if any(pid in geofabrik_ids for pid in sqlite_partition_ids):
                partition_type = "geofabrik"
            else:
                partition_type = "quadtree"

        # Use flattened structure for iOS compatibility
        index_data[region_key] = {
            "region_name": region_name.replace("_", " "),
            "version": version,
            "release_tag": tag_name,
            "release_url": release_url,
            "climb_count": climb_count,
            "elevation_errors": elevation_errors,
            # Flattened xlsx fields (no nested object)
            "files": xlsx_files,
            "download_urls": xlsx_urls,
            "file_sizes": xlsx_sizes,
            "total_size": sum(xlsx_sizes),
            "file_count": len(xlsx_files),
            "has_split_files": has_split_files,
            # Database fields - always populated when SQLite exists
            # For split databases: logical name/total size, download via split_urls
            "database_file": (
                sqlite_file if sqlite_file
                else sqlite_split_files[0].rsplit('.', 1)[0] if sqlite_split_files  # e.g., .sqlite.001 -> .sqlite
                else None
            ),
            "database_size": (
                sqlite_size if sqlite_size
                else sum(sqlite_split_sizes) if sqlite_split_sizes
                else None
            ),
            "database_url": (
                sqlite_url if sqlite_url
                else sqlite_split_urls[0] if sqlite_split_urls  # First chunk URL as reference
                else None
            ),
            # Split database fields (binary chunks - iOS-expected schema)
            "is_split": is_split_db,
            "split_files": sqlite_split_files if is_split_db else None,
            "split_urls": sqlite_split_urls if is_split_db else None,
            "split_sizes": sqlite_split_sizes if is_split_db else None,
            "split_checksums": split_checksums if is_split_db and split_checksums else None,
            # Partitioned database fields (geographic partitions)
            "is_partitioned": is_partitioned_db,
            "partition_type": partition_type,
            "partitions": partitions_data if is_partitioned_db else None,
            "total_database_size": (
                sum(sqlite_partition_sizes) if is_partitioned_db
                else sum(sqlite_split_sizes) if is_split_db
                else sqlite_size if sqlite_size
                else None
            ),
            # Gzipped database fields (preferred format - ~3x smaller downloads).
            # When present, the iOS app should prefer these over the raw sqlite path.
            # - database_format: "gzip" if a .sqlite.gz is available, else "sqlite"
            # - database_gz_url: single-file download URL (null if split across chunks)
            # - database_gz_size: total compressed size on disk
            # - database_decompressed_size: size after decompression (matches raw sqlite size)
            # - is_gz_split: true when .sqlite.gz exceeds 2 GB and is split into .gz.00N
            # - gz_split_files/gz_split_urls/gz_split_sizes: chunks to download+concat
            "database_format": "gzip" if has_gz else ("sqlite" if (sqlite_file or is_split_db) else None),
            "database_gz_file": (
                sqlite_gz_file if sqlite_gz_file
                else (sqlite_gz_split_files[0].rsplit('.', 1)[0] if sqlite_gz_split_files else None)
            ),
            "database_gz_size": (
                sqlite_gz_size if sqlite_gz_size
                else (sum(sqlite_gz_split_sizes) if sqlite_gz_split_sizes else None)
            ),
            "database_gz_url": sqlite_gz_url if sqlite_gz_file else None,
            "database_decompressed_size": (
                sqlite_size if sqlite_size
                else (sum(sqlite_split_sizes) if sqlite_split_sizes else None)
            ),
            "is_gz_split": is_split_gz,
            "gz_split_files": sqlite_gz_split_files if is_split_gz else None,
            "gz_split_urls": sqlite_gz_split_urls if is_split_gz else None,
            "gz_split_sizes": sqlite_gz_split_sizes if is_split_gz else None,
            # Dates
            "published_at": published_at,
            "last_updated": last_updated,
        }

    return index_data


def create_summary_stats(index_data: Dict) -> Dict:
    """Create summary statistics for the index."""
    total_regions = len(index_data)
    total_xlsx_files = sum(region.get("file_count", 0) for region in index_data.values())

    # Count SQLite databases (single file, split, or partitioned)
    total_sqlite_files = 0
    total_partitioned_regions = 0
    for region in index_data.values():
        if region.get("is_partitioned") and region.get("partitions"):
            total_sqlite_files += len(region["partitions"])
            total_partitioned_regions += 1
        elif region.get("is_split"):
            total_sqlite_files += 1  # Count as one logical database
        elif region.get("database_file"):
            total_sqlite_files += 1

    total_xlsx_size = sum(region.get("total_size", 0) for region in index_data.values())

    # Calculate SQLite size (single file, split files, or partitioned files)
    total_sqlite_size = 0
    for region in index_data.values():
        if region.get("is_partitioned") and region.get("total_database_size"):
            total_sqlite_size += region["total_database_size"]
        elif region.get("is_split") and region.get("split_sizes"):
            total_sqlite_size += sum(region["split_sizes"])
        elif region.get("database_size"):
            total_sqlite_size += region["database_size"]

    total_climbs = sum(region.get("climb_count") or 0 for region in index_data.values())

    return {
        "total_regions": total_regions,
        "total_xlsx_files": total_xlsx_files,
        "total_sqlite_files": total_sqlite_files,
        "total_partitioned_regions": total_partitioned_regions,
        "total_xlsx_size_bytes": total_xlsx_size,
        "total_sqlite_size_bytes": total_sqlite_size,
        "total_size_mb": round((total_xlsx_size + total_sqlite_size) / (1024 * 1024), 1),
        "total_climbs": total_climbs,
    }


def main():
    """Main function to run the indexer."""
    print("Starting release indexing...")

    # Get repository info from environment or use defaults
    repo_name = os.environ.get("REPO_NAME", "stevehollx/global-road-and-trail-climbs")
    owner, repo = repo_name.split("/")
    token = get_github_token()

    if token:
        print(f"Using authenticated GitHub API access")
    else:
        print(f"Using unauthenticated API access (rate limited)")

    # Scan releases
    print(f"Scanning releases from {repo_name}...")
    index_data = scan_releases(owner, repo, token)

    # Create the final index structure
    final_index = {
        "version": "2.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repository": repo_name,
        "summary": create_summary_stats(index_data),
        "regions": index_data,
    }

    # Write to index.json in the current directory
    output_path = "index.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Index created successfully at {output_path}")
    print(f"  - Found {final_index['summary']['total_regions']} regions")
    print(f"  - Total XLSX files: {final_index['summary']['total_xlsx_files']}")
    print(f"  - Total SQLite files: {final_index['summary']['total_sqlite_files']}")
    if final_index['summary'].get('total_partitioned_regions', 0) > 0:
        print(f"  - Partitioned regions: {final_index['summary']['total_partitioned_regions']}")
    print(f"  - Total climbs: {final_index['summary']['total_climbs']:,}")
    print(f"  - Total size: {final_index['summary']['total_size_mb']:.1f} MB")

    if final_index["summary"]["total_regions"] == 0:
        print("\nNo climb data releases found in the repository.")
        print("The index.json file has been created but is empty.")
        print("It will be populated as releases are created.")
    else:
        print("\nRegions indexed:")
        for region_key, region_data in sorted(index_data.items()):
            climb_str = f"{region_data['climb_count']:,}" if region_data.get("climb_count") else "?"
            partition_info = ""
            if region_data.get("is_partitioned") and region_data.get("partitions"):
                partition_count = len(region_data["partitions"])
                partition_info = f" [{partition_count} partitions]"
            print(f"  - {region_data['region_name']}: {climb_str} climbs, v{region_data['version']}{partition_info}")


if __name__ == "__main__":
    main()
