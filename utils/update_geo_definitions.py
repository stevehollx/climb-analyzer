#!/usr/bin/env python3
"""
Update Geographic Definitions

Crawls Geofabrik to build a single osm_pbf_urls dictionary containing:
- PBF download URLs
- File sizes
- Bounding boxes (from .poly files)
- Hierarchical subregion structure

This replaces the previous approach using Natural Earth shapefiles + Nominatim.
"""

import os
import re
import sys
import time
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urljoin

import requests
from tqdm import tqdm

# Add parent directory to path to import formatting utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from climb_analyzer.utils.formatting import (
    print_banner,
    print_error,
    print_header,
    print_success,
    print_warning,
)

# Debug flag - set to True to see detailed progress messages
DEBUG = False

# Request headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def parse_poly_to_bounds(poly_content: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse an Osmosis .poly file and extract bounding box.

    .poly format:
        RegionName
        1
           lon1 lat1
           lon2 lat2
           ...
        END
        END

    Returns:
        Tuple of (lat_min, lon_min, lat_max, lon_max) or None if parsing fails.
    """
    try:
        lines = poly_content.strip().split('\n')
        coords = []
        in_ring = False

        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0:
                # Skip header (region name)
                continue
            elif line == 'END':
                in_ring = False
            elif line.startswith('!'):
                # Hole section - skip for bounds calculation
                in_ring = True
            elif not in_ring and line and not line.isdigit():
                # Coordinate line (not a section number)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coords.append((lon, lat))
                    except ValueError:
                        pass
            elif line.isdigit():
                # Section number, start a new ring
                in_ring = False

        if not coords:
            return None

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        lat_min = min(lats)
        lat_max = max(lats)
        lon_min = min(lons)
        lon_max = max(lons)

        return (lat_min, lon_min, lat_max, lon_max)

    except Exception as e:
        if DEBUG:
            print(f"    Error parsing .poly: {e}")
        return None


def fetch_poly_bounds(poly_url: str) -> Optional[Tuple[float, float, float, float]]:
    """Download and parse a .poly file to get bounding box."""
    try:
        response = requests.get(poly_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return parse_poly_to_bounds(response.text)
    except Exception as e:
        if DEBUG:
            print(f"    Failed to fetch .poly from {poly_url}: {e}")
        return None


def get_geofabrik_structured_data():
    """
    Crawl Geofabrik to build hierarchical OSM data with bounds.

    Returns dict with structure:
    {
        "continent": {
            "pbf_url": "...",
            "size": 12345,
            "bounds": (lat_min, lon_min, lat_max, lon_max),
            "subregions": {
                "continent/country": {
                    "pbf_url": "...",
                    "size": 12345,
                    "bounds": (...),
                    "subregions": {...}
                }
            }
        }
    }
    """
    HREF_RE = re.compile(r'href="([^"]+)"')
    SIZE_RE_NBSP = re.compile(r"(\d+(?:\.\d+)?)\s*&nbsp;\s*(MB|GB|TB|KB)")

    # Continent URLs
    region_urls = {
        "africa": "https://download.geofabrik.de/africa.html",
        "antarctica": "https://download.geofabrik.de/antarctica.html",
        "asia": "https://download.geofabrik.de/asia.html",
        "australia-oceania": "https://download.geofabrik.de/australia-oceania.html",
        "central-america": "https://download.geofabrik.de/central-america.html",
        "europe": "https://download.geofabrik.de/europe.html",
        "north-america": "https://download.geofabrik.de/north-america.html",
        "south-america": "https://download.geofabrik.de/south-america.html",
    }

    def get_file_size_bytes(size_str, unit_str):
        """Convert file size string to bytes."""
        size_val = float(size_str)
        unit_multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        return int(size_val * unit_multipliers.get(unit_str.upper(), 1))

    def fetch_links_and_sizes(url):
        """Fetch all href links from a URL and extract size information."""
        if DEBUG:
            print(f"  Fetching {url}")

        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        links = HREF_RE.findall(resp.text)

        # Parse sizes from HTML
        size_map = {}
        table_rows = re.findall(r"<tr[^>]*>.*?</tr>", resp.text, re.DOTALL)
        for row in table_rows:
            href_match = re.search(r'href="([^"]*-latest\.osm\.pbf)"', row)
            if href_match:
                size_match = SIZE_RE_NBSP.search(row)
                if size_match:
                    filename = href_match.group(1)
                    size_bytes = get_file_size_bytes(size_match.group(1), size_match.group(2))
                    file_key = filename.replace("-latest.osm.pbf", "")
                    size_map[file_key] = size_bytes

        return links, size_map

    def is_valid_geographic_link(html_href, current_url):
        """Check if a link represents a valid geographic region."""
        link_name = html_href.replace(".html", "")

        if link_name.startswith("http://") or link_name.startswith("https://"):
            return False
        if link_name.startswith("/") or link_name.startswith("../"):
            return False

        invalid_pages = [
            "index", "contact", "imprint", "about", "help", "support", "data",
            "download", "shapefiles", "technical", "technical-details", "geofabrik",
            "openstreetmap", "osm", "vector-data", "routing", "geocoding",
            "overpass-api", "admin-polygons", "energy-networks", "postalcodes",
            "routeable-vector-data", "free", "press", "publications", "students",
        ]

        base_name = link_name.split("/")[-1]
        if base_name.lower() in invalid_pages:
            return False

        if re.match(r"^[a-z0-9_-]+(/[a-z0-9_-]+)*$", link_name):
            return True

        return False

    def recursively_get_subregions(region_url, region_name, parsed_sizes, depth=0, max_depth=5, sub_pbar=None):
        """Recursively crawl subregions from a Geofabrik page."""
        if depth >= max_depth:
            return None

        indent = "  " * depth
        if DEBUG:
            print(f"{indent}Crawling: {region_name} (depth {depth})")

        try:
            links, sizes = fetch_links_and_sizes(region_url)
        except Exception as e:
            if DEBUG:
                print(f"{indent}  Failed to fetch {region_url}: {e}")
            return None

        result = {}

        # Get PBF URL
        region_pbf_url = region_url.replace(".html", "-latest.osm.pbf")
        region_key = region_name
        if region_key in parsed_sizes:
            result["pbf_url"] = region_pbf_url
            result["size"] = parsed_sizes[region_key]
        elif region_key in sizes:
            result["pbf_url"] = region_pbf_url
            result["size"] = sizes[region_key]

        # Get bounds from .poly file
        poly_url = region_url.replace(".html", ".poly")
        bounds = fetch_poly_bounds(poly_url)
        if bounds:
            result["bounds"] = bounds
            if DEBUG:
                print(f"{indent}  Got bounds from .poly: {bounds}")
        else:
            if DEBUG:
                print(f"{indent}  Warning: No bounds for {region_name}")

        # Find subregion links
        html_links = []
        for href in links:
            if href.endswith(".html") and not href.startswith(".."):
                if is_valid_geographic_link(href, region_url):
                    html_links.append(href)

        if html_links:
            result["subregions"] = {}

            for html_href in html_links:
                if any(skip in html_href.lower() for skip in ["special", "diffs", "updates"]):
                    continue

                subregion_url = urljoin(region_url, html_href)
                subregion_name = html_href.replace(".html", "")

                if sub_pbar and depth == 0:
                    sub_pbar.set_description(f"  └─ {subregion_name}")
                    sub_pbar.update(1)

                time.sleep(0.3)  # Rate limiting
                subregion_data = recursively_get_subregions(
                    subregion_url, subregion_name, sizes, depth + 1, max_depth, sub_pbar
                )

                if subregion_data:
                    result["subregions"][subregion_name] = subregion_data

        return result if result else None

    def count_subregions(links, region_url):
        """Count valid subregion links for progress bar."""
        count = 0
        for href in links:
            if href.endswith(".html") and not href.startswith(".."):
                if is_valid_geographic_link(href, region_url):
                    if not any(skip in href.lower() for skip in ["special", "diffs", "updates"]):
                        count += 1
        return count

    # Process all continents
    all_geofabrik_data = {}

    print()
    pbar = tqdm(
        region_urls.items(),
        desc="Crawling continents",
        unit="continent",
        position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        ascii=" ▏▎▍▌▋▊▉█"
    )

    for region_name, region_url in pbar:
        try:
            pbar.set_description(f"Crawling {region_name}")

            # First fetch to count subregions
            links, sizes = fetch_links_and_sizes(region_url)
            subregion_count = count_subregions(links, region_url)

            sub_pbar = None
            if subregion_count > 0:
                sub_pbar = tqdm(
                    total=subregion_count,
                    desc=f"  └─ Scanning",
                    leave=False,
                    position=1,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
                    ascii=" ▏▎▍▌▋▊▉█"
                )

            region_data = recursively_get_subregions(
                region_url, region_name, sizes, depth=0, max_depth=5, sub_pbar=sub_pbar
            )

            if sub_pbar:
                sub_pbar.close()

            if region_data:
                all_geofabrik_data[region_name] = region_data

            time.sleep(1)  # Rate limiting between continents

        except Exception as e:
            print_error(f"Failed to process {region_name}: {e}")
            continue

    print(f"✓ Successfully crawled {len(all_geofabrik_data)} continents\n")
    return all_geofabrik_data


def count_regions_with_bounds(data, stats=None):
    """Count regions with and without bounds."""
    if stats is None:
        stats = {"with_bounds": 0, "without_bounds": 0}

    if "bounds" in data:
        stats["with_bounds"] += 1
    elif "pbf_url" in data:
        stats["without_bounds"] += 1

    if "subregions" in data:
        for subregion_data in data["subregions"].values():
            count_regions_with_bounds(subregion_data, stats)

    return stats


def main():
    print_banner("Updating Geographic Definitions")

    print_header("Crawling Geofabrik (URLs + Bounds from .poly files)")

    geofabrik_data = get_geofabrik_structured_data()

    # Count statistics
    total_stats = {"with_bounds": 0, "without_bounds": 0}
    for continent_data in geofabrik_data.values():
        count_regions_with_bounds(continent_data, total_stats)

    print(f"  Regions with bounds: {total_stats['with_bounds']}")
    print(f"  Regions without bounds: {total_stats['without_bounds']}")

    # Write to geo_definitions.py
    print_header("Writing geo_definitions.py")

    def write_region_dict(f, region_data, indent_level=1):
        """Recursively write region data with proper indentation."""
        indent = "    " * indent_level

        # Write fields in order: pbf_url, size, bounds, subregions
        if "pbf_url" in region_data:
            f.write(f'{indent}"pbf_url": "{region_data["pbf_url"]}",\n')

        if "size" in region_data:
            f.write(f'{indent}"size": {region_data["size"]},\n')

        if "bounds" in region_data:
            bounds = region_data["bounds"]
            f.write(f'{indent}"bounds": ({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]})')
            if "subregions" in region_data and region_data["subregions"]:
                f.write(",\n")
            else:
                f.write("\n")
        elif "subregions" not in region_data or not region_data["subregions"]:
            # Remove trailing comma from size if no bounds and no subregions
            pass

        if "subregions" in region_data and region_data["subregions"]:
            f.write(f'{indent}"subregions": {{\n')
            subregion_items = list(region_data["subregions"].items())
            for i, (subregion_name, subregion_data) in enumerate(subregion_items):
                f.write(f'{indent}    "{subregion_name}": {{\n')
                write_region_dict(f, subregion_data, indent_level + 2)
                f.write(f"{indent}    }}")
                if i < len(subregion_items) - 1:
                    f.write(",")
                f.write("\n")
            f.write(f"{indent}}}\n")

    with open("climb_analyzer/data/geo_definitions.py", "w") as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        f.write("# Geofabrik OSM data - URLs, sizes, and bounding boxes\n")
        f.write(f"# Generated automatically on {current_time} - do not edit manually\n")
        f.write("# Bounds sourced from Geofabrik .poly files (exact extract boundaries)\n\n")

        f.write("osm_pbf_urls = {\n")
        region_items = list(geofabrik_data.items())
        for i, (region, data) in enumerate(region_items):
            f.write(f'    "{region}": {{\n')
            write_region_dict(f, data, indent_level=2)
            f.write("    }")
            if i < len(region_items) - 1:
                f.write(",")
            f.write("\n")
        f.write("}\n")

    print_success("geo_definitions.py written successfully")
    print(f"  Output: climb_analyzer/data/geo_definitions.py")


if __name__ == "__main__":
    main()
