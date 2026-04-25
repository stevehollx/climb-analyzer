#!/usr/bin/env python3
"""
OSM data downloader from Geofabrik.

Downloads .osm.pbf files for countries and regions.
"""

import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Import centralized data paths
try:
    from utils.data_paths import PLANET_OSM_DIR
except ImportError:
    # Fallback for direct script execution
    PLANET_OSM_DIR = Path("data/planet_osm_data")


# Geofabrik URL mappings for common countries/regions
GEOFABRIK_URLS = {
    # North America
    "United States of America": "https://download.geofabrik.de/north-america/us-latest.osm.pbf",
    "Canada": "https://download.geofabrik.de/north-america/canada-latest.osm.pbf",
    "Mexico": "https://download.geofabrik.de/north-america/mexico-latest.osm.pbf",
    "Greenland": "https://download.geofabrik.de/north-america/greenland-latest.osm.pbf",

    # Europe
    "Albania": "https://download.geofabrik.de/europe/albania-latest.osm.pbf",
    "Austria": "https://download.geofabrik.de/europe/austria-latest.osm.pbf",
    "Belarus": "https://download.geofabrik.de/europe/belarus-latest.osm.pbf",
    "Belgium": "https://download.geofabrik.de/europe/belgium-latest.osm.pbf",
    "Bosnia and Herz.": "https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf",
    "Bulgaria": "https://download.geofabrik.de/europe/bulgaria-latest.osm.pbf",
    "Croatia": "https://download.geofabrik.de/europe/croatia-latest.osm.pbf",
    "Czechia": "https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf",
    "Czech Rep.": "https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf",
    "Denmark": "https://download.geofabrik.de/europe/denmark-latest.osm.pbf",
    "Estonia": "https://download.geofabrik.de/europe/estonia-latest.osm.pbf",
    "Finland": "https://download.geofabrik.de/europe/finland-latest.osm.pbf",
    "France": "https://download.geofabrik.de/europe/france-latest.osm.pbf",
    "Germany": "https://download.geofabrik.de/europe/germany-latest.osm.pbf",
    "Greece": "https://download.geofabrik.de/europe/greece-latest.osm.pbf",
    "Hungary": "https://download.geofabrik.de/europe/hungary-latest.osm.pbf",
    "Iceland": "https://download.geofabrik.de/europe/iceland-latest.osm.pbf",
    "Ireland": "https://download.geofabrik.de/europe/ireland-and-northern-ireland-latest.osm.pbf",
    "Italy": "https://download.geofabrik.de/europe/italy-latest.osm.pbf",
    "Latvia": "https://download.geofabrik.de/europe/latvia-latest.osm.pbf",
    "Lithuania": "https://download.geofabrik.de/europe/lithuania-latest.osm.pbf",
    "Luxembourg": "https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf",
    "Netherlands": "https://download.geofabrik.de/europe/netherlands-latest.osm.pbf",
    "Norway": "https://download.geofabrik.de/europe/norway-latest.osm.pbf",
    "Poland": "https://download.geofabrik.de/europe/poland-latest.osm.pbf",
    "Portugal": "https://download.geofabrik.de/europe/portugal-latest.osm.pbf",
    "Romania": "https://download.geofabrik.de/europe/romania-latest.osm.pbf",
    "Russia": "https://download.geofabrik.de/russia-latest.osm.pbf",
    "Serbia": "https://download.geofabrik.de/europe/serbia-latest.osm.pbf",
    "Slovakia": "https://download.geofabrik.de/europe/slovakia-latest.osm.pbf",
    "Slovenia": "https://download.geofabrik.de/europe/slovenia-latest.osm.pbf",
    "Spain": "https://download.geofabrik.de/europe/spain-latest.osm.pbf",
    "Sweden": "https://download.geofabrik.de/europe/sweden-latest.osm.pbf",
    "Switzerland": "https://download.geofabrik.de/europe/switzerland-latest.osm.pbf",
    "Ukraine": "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf",
    "United Kingdom": "https://download.geofabrik.de/europe/great-britain-latest.osm.pbf",

    # South America
    "Argentina": "https://download.geofabrik.de/south-america/argentina-latest.osm.pbf",
    "Bolivia": "https://download.geofabrik.de/south-america/bolivia-latest.osm.pbf",
    "Brazil": "https://download.geofabrik.de/south-america/brazil-latest.osm.pbf",
    "Chile": "https://download.geofabrik.de/south-america/chile-latest.osm.pbf",
    "Colombia": "https://download.geofabrik.de/south-america/colombia-latest.osm.pbf",
    "Ecuador": "https://download.geofabrik.de/south-america/ecuador-latest.osm.pbf",
    "Paraguay": "https://download.geofabrik.de/south-america/paraguay-latest.osm.pbf",
    "Peru": "https://download.geofabrik.de/south-america/peru-latest.osm.pbf",
    "Uruguay": "https://download.geofabrik.de/south-america/uruguay-latest.osm.pbf",
    "Venezuela": "https://download.geofabrik.de/south-america/venezuela-latest.osm.pbf",

    # Asia
    "Afghanistan": "https://download.geofabrik.de/asia/afghanistan-latest.osm.pbf",
    "Bangladesh": "https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf",
    "Cambodia": "https://download.geofabrik.de/asia/cambodia-latest.osm.pbf",
    "China": "https://download.geofabrik.de/asia/china-latest.osm.pbf",
    "India": "https://download.geofabrik.de/asia/india-latest.osm.pbf",
    "Indonesia": "https://download.geofabrik.de/asia/indonesia-latest.osm.pbf",
    "Iran": "https://download.geofabrik.de/asia/iran-latest.osm.pbf",
    "Iraq": "https://download.geofabrik.de/asia/iraq-latest.osm.pbf",
    "Israel": "https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf",
    "Japan": "https://download.geofabrik.de/asia/japan-latest.osm.pbf",
    "Kazakhstan": "https://download.geofabrik.de/asia/kazakhstan-latest.osm.pbf",
    "Malaysia": "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf",
    "Mongolia": "https://download.geofabrik.de/asia/mongolia-latest.osm.pbf",
    "Myanmar": "https://download.geofabrik.de/asia/myanmar-latest.osm.pbf",
    "Nepal": "https://download.geofabrik.de/asia/nepal-latest.osm.pbf",
    "Pakistan": "https://download.geofabrik.de/asia/pakistan-latest.osm.pbf",
    "Philippines": "https://download.geofabrik.de/asia/philippines-latest.osm.pbf",
    "South Korea": "https://download.geofabrik.de/asia/south-korea-latest.osm.pbf",
    "Korea": "https://download.geofabrik.de/asia/south-korea-latest.osm.pbf",
    "Sri Lanka": "https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf",
    "Taiwan": "https://download.geofabrik.de/asia/taiwan-latest.osm.pbf",
    "Thailand": "https://download.geofabrik.de/asia/thailand-latest.osm.pbf",
    "Turkey": "https://download.geofabrik.de/europe/turkey-latest.osm.pbf",
    "Uzbekistan": "https://download.geofabrik.de/asia/uzbekistan-latest.osm.pbf",
    "Vietnam": "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf",

    # Africa
    "Algeria": "https://download.geofabrik.de/africa/algeria-latest.osm.pbf",
    "Egypt": "https://download.geofabrik.de/africa/egypt-latest.osm.pbf",
    "Ethiopia": "https://download.geofabrik.de/africa/ethiopia-latest.osm.pbf",
    "Kenya": "https://download.geofabrik.de/africa/kenya-latest.osm.pbf",
    "Madagascar": "https://download.geofabrik.de/africa/madagascar-latest.osm.pbf",
    "Morocco": "https://download.geofabrik.de/africa/morocco-latest.osm.pbf",
    "Nigeria": "https://download.geofabrik.de/africa/nigeria-latest.osm.pbf",
    "South Africa": "https://download.geofabrik.de/africa/south-africa-latest.osm.pbf",
    "Tanzania": "https://download.geofabrik.de/africa/tanzania-latest.osm.pbf",

    # Oceania
    "Australia": "https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf",
    "New Zealand": "https://download.geofabrik.de/australia-oceania/new-zealand-latest.osm.pbf",
}


# Region name stems that exist in multiple Geofabrik paths and collide when
# using just the URL's last segment (e.g., "georgia-latest.osm.pbf" exists in
# both europe/ and north-america/us/). These require a path prefix in the
# local filename to prevent one being overwritten by the other.
AMBIGUOUS_STEMS = {"georgia"}


def get_local_pbf_filename(url: str) -> str:
    """
    Generate the local filename for a Geofabrik PBF URL.

    For unambiguous regions, returns the URL's last segment unchanged
    (e.g., "california-latest.osm.pbf") to preserve backward compatibility.

    For ambiguous regions (e.g., Georgia the US state vs Georgia the country),
    prefixes the filename with the immediate parent folder from the URL path
    (e.g., "us_georgia-latest.osm.pbf" vs "europe_georgia-latest.osm.pbf").

    Args:
        url: Full Geofabrik download URL

    Returns:
        Disambiguated local filename

    Examples:
        >>> get_local_pbf_filename("https://download.geofabrik.de/europe/france-latest.osm.pbf")
        'france-latest.osm.pbf'
        >>> get_local_pbf_filename("https://download.geofabrik.de/north-america/us/georgia-latest.osm.pbf")
        'us_georgia-latest.osm.pbf'
        >>> get_local_pbf_filename("https://download.geofabrik.de/europe/georgia-latest.osm.pbf")
        'europe_georgia-latest.osm.pbf'
    """
    # Parse out the parts after the geofabrik domain
    # e.g., "north-america/us/georgia-latest.osm.pbf" or "europe/france-latest.osm.pbf"
    base = "https://download.geofabrik.de/"
    if url.startswith(base):
        path = url[len(base):]
    else:
        # Fallback: use everything after the last scheme separator
        path = url.split("://", 1)[-1].split("/", 1)[-1]

    parts = path.split("/")
    filename = parts[-1]  # e.g., "georgia-latest.osm.pbf"

    # Extract stem (name before "-latest.osm.pbf")
    if not filename.endswith("-latest.osm.pbf"):
        return filename
    stem = filename[: -len("-latest.osm.pbf")]

    # If stem is not known to be ambiguous, use the simple filename
    if stem not in AMBIGUOUS_STEMS:
        return filename

    # Prefix with the immediate parent folder for disambiguation.
    # For us/georgia -> "us", for europe/georgia -> "europe"
    if len(parts) >= 2:
        parent = parts[-2]
        return f"{parent}_{filename}"
    return filename


def download_osm_data(
    country_name: str,
    output_dir: Optional[str] = None,
    force: bool = False
) -> Optional[Path]:
    """
    Download OSM .pbf file from Geofabrik.

    Args:
        country_name: Country name (e.g., "Switzerland")
        output_dir: Where to save .pbf file (defaults to data/planet_osm_data)
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded file, or None if failed
    """
    url = GEOFABRIK_URLS.get(country_name)
    if not url:
        print(f"\n⚠️  No Geofabrik extract available for '{country_name}'")
        print("\nAvailable countries:")
        for i, name in enumerate(sorted(GEOFABRIK_URLS.keys())[:20], 1):
            print(f"  {name}")
        if len(GEOFABRIK_URLS) > 20:
            print(f"  ... and {len(GEOFABRIK_URLS) - 20} more")
        print("\nFor other regions, download manually from:")
        print("  https://download.geofabrik.de/")
        return None

    # Create output directory (use new data structure by default)
    output_path = Path(output_dir) if output_dir else PLANET_OSM_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine output filename (with disambiguation for ambiguous regions)
    filename = get_local_pbf_filename(url)
    output_file = output_path / filename

    # Check if already exists
    if output_file.exists() and not force:
        print(f"\n✓ OSM file already exists: {output_file}")
        response = input("  Re-download OSM file? [y/N]: ").strip().lower()
        if response != 'y':
            print("  Using existing OSM file")
            return output_file
        else:
            print("  Re-downloading...")

    print(f"\nDownloading OSM data for {country_name}")
    print(f"  Source: {url}")
    print(f"  Destination: {output_file}")

    try:
        # Get file size for progress bar
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_file, 'wb') as f, tqdm(
            desc="  Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ascii=" ▏▎▍▌▋▊▉█"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"  ✓ Downloaded successfully: {output_file.stat().st_size / (1024**3):.2f} GB")
        return output_file

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        if output_file.exists():
            partial_size = output_file.stat().st_size / (1024**2)
            print(f"  💾 Partial download saved ({partial_size:.1f} MB) - rerun to resume")
        return None
    except KeyboardInterrupt:
        print(f"\n  ⏸️  Download interrupted")
        if output_file.exists():
            partial_size = output_file.stat().st_size / (1024**2)
            print(f"  💾 Partial download saved ({partial_size:.1f} MB) - rerun to resume")
        raise  # Re-raise to allow graceful shutdown
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        if output_file.exists():
            output_file.unlink()
        return None


def get_osm_url_for_location(location_name: str) -> Optional[tuple[str, int]]:
    """
    Get OSM download URL and size for a location (country or US state).

    Args:
        location_name: Name of country, region, or US state
                      Can be:
                      - "Switzerland" (simple name)
                      - "europe/liechtenstein" (hierarchical path)
                      - "north-america/us/california" (nested path)

    Returns:
        Tuple of (pbf_url, size_bytes) or None if not found
    """
    from climb_analyzer.data.geo_definitions import osm_pbf_urls

    def search_hierarchical(data_dict, path_parts):
        """Recursively search hierarchical structure for a region."""
        if not path_parts:
            return None

        # Try to find in subregions at current level
        if 'subregions' in data_dict:
            subregions = data_dict['subregions']

            # Try exact match with full remaining path
            full_key = "/".join(path_parts)
            if full_key in subregions:
                region_data = subregions[full_key]
                if 'pbf_url' in region_data:
                    return (region_data['pbf_url'], region_data.get('size', 0))

            # Try matching just the first part and recurse
            first_part = path_parts[0]
            for key, value in subregions.items():
                if key.endswith('/' + first_part) or key == first_part:
                    if len(path_parts) == 1 and 'pbf_url' in value:
                        return (value['pbf_url'], value.get('size', 0))
                    elif len(path_parts) > 1:
                        # Recurse deeper
                        result = search_hierarchical(value, path_parts[1:])
                        if result:
                            return result

        return None

    # Normalize location name
    location_normalized = location_name.lower().replace(' ', '-')

    # Split path if it contains /
    if '/' in location_normalized:
        path_parts = location_normalized.split('/')
        continent = path_parts[0]

        # Look in the specific continent
        if continent in osm_pbf_urls:
            continent_data = osm_pbf_urls[continent]
            result = search_hierarchical(continent_data, path_parts[1:])
            if result:
                return result

            # Also try searching with full path including continent
            result = search_hierarchical(continent_data, path_parts)
            if result:
                return result

    # Try simple search across all continents
    for continent_key, continent_data in osm_pbf_urls.items():
        if isinstance(continent_data, dict):
            # Try direct match at continent level
            if 'pbf_url' in continent_data:
                # Check if this is the region we want
                if continent_key.lower() == location_normalized:
                    return (continent_data['pbf_url'], continent_data.get('size', 0))

            # Search in subregions
            result = search_hierarchical(continent_data, [location_normalized])
            if result:
                return result

    # Fallback to hardcoded GEOFABRIK_URLS
    url = GEOFABRIK_URLS.get(location_name)
    if url:
        return (url, 0)  # Size unknown from hardcoded list

    return None


def download_osm_for_location(
    location_name: str,
    output_dir: Optional[str] = None,
    force: bool = False
) -> Optional[Path]:
    """
    Download OSM .pbf file for a location (country or US state).

    Uses geo_definitions.py for URL lookup, supporting both countries and US states.

    Args:
        location_name: Location name (e.g., "Switzerland", "Hawaii")
        output_dir: Where to save .pbf file (defaults to data/planet_osm_data)
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded file, or None if failed
    """
    url_data = get_osm_url_for_location(location_name)

    if not url_data:
        print(f"\n⚠️  No Geofabrik extract available for '{location_name}'")
        print("\nFor manual download, visit:")
        print("  https://download.geofabrik.de/")
        return None

    url, expected_size = url_data

    # Create output directory (use new data structure by default)
    output_path = Path(output_dir) if output_dir else PLANET_OSM_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine output filename (with disambiguation for ambiguous regions)
    filename = get_local_pbf_filename(url)
    output_file = output_path / filename

    # Check if already exists
    if output_file.exists() and not force:
        print(f"\n✓ OSM file already exists: {output_file}")
        return output_file

    # Show download info
    size_gb = expected_size / (1024**3) if expected_size > 0 else 0
    size_info = f" (~{size_gb:.1f} GB)" if size_gb > 0 else ""

    print(f"\nDownloading OSM data for {location_name}{size_info}")
    print(f"  Source: {url}")
    print(f"  Destination: {output_file}")

    try:
        # Ensure output directory is writable
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = output_path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            print(f"  ❌ Permission denied: Cannot write to {output_path}")
            print(f"  💡 Try running: chmod 755 {output_path}")
            return None

        # Check for partial download to resume
        resume_pos = 0
        file_mode = 'wb'
        if output_file.exists():
            resume_pos = output_file.stat().st_size
            if resume_pos > 0:
                print(f"  📂 Found partial download ({resume_pos / (1024**2):.1f} MB) - resuming...")
                file_mode = 'ab'  # Append mode to continue download

        # Get file size for progress bar
        response = requests.head(url, allow_redirects=True, timeout=10)
        total_size = int(response.headers.get('content-length', 0))

        # Check if resume is needed and server supports it
        headers = {}
        if resume_pos > 0:
            # HTTP Range header for resume
            headers['Range'] = f'bytes={resume_pos}-'
            # Verify server supports range requests
            if response.headers.get('Accept-Ranges') != 'bytes':
                print(f"  ⚠️  Server doesn't support resume - starting from beginning")
                resume_pos = 0
                file_mode = 'wb'
                headers = {}

        # Download with progress bar
        response = requests.get(url, stream=True, timeout=300, headers=headers)
        response.raise_for_status()

        with open(output_file, file_mode) as f, tqdm(
            desc="  Downloading",
            total=total_size,
            initial=resume_pos,  # Start progress bar from resume position
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ascii=" ▏▎▍▌▋▊▉█"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        # Validate download size against Content-Length
        actual_size = output_file.stat().st_size
        if total_size > 0:
            if actual_size < total_size:
                print(f"  ⚠️  Download incomplete: {actual_size:,} bytes vs expected {total_size:,}")
                print(f"  💾 Partial download saved - rerun to resume")
                return None
            print(f"  ✓ Download verified: {actual_size:,} bytes ({actual_size / (1024**3):.2f} GB)")
        else:
            print(f"  ✓ Downloaded: {actual_size / (1024**3):.2f} GB")
        return output_file

    except PermissionError as e:
        print(f"  ❌ Permission denied: {e}")
        print(f"  💡 Try running: chmod 755 {output_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        if output_file.exists():
            partial_size = output_file.stat().st_size / (1024**2)
            print(f"  💾 Partial download saved ({partial_size:.1f} MB) - rerun to resume")
        return None
    except KeyboardInterrupt:
        print(f"\n  ⏸️  Download interrupted")
        if output_file.exists():
            partial_size = output_file.stat().st_size / (1024**2)
            print(f"  💾 Partial download saved ({partial_size:.1f} MB) - rerun to resume")
        raise  # Re-raise to allow graceful shutdown
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        if output_file.exists():
            output_file.unlink()
        return None


def get_available_countries() -> list:
    """Get list of countries with Geofabrik extracts."""
    return sorted(GEOFABRIK_URLS.keys())


# Test if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        country = sys.argv[1]
        result = download_osm_data(country)
        if result:
            print(f"\nSuccess! File saved to: {result}")
        else:
            print("\nDownload failed.")
            sys.exit(1)
    else:
        print("Usage: python osm_downloader.py <country_name>")
        print("\nExample:")
        print("  python osm_downloader.py Switzerland")
        print("\nAvailable countries:")
        for country in get_available_countries()[:20]:
            print(f"  {country}")
        print(f"  ... and {len(get_available_countries()) - 20} more")
