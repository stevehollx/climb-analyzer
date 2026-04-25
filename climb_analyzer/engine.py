#!/usr/bin/env python3

# Instantiating dependency check before importing other modules
import argparse
import sys
from pathlib import Path


def create_default_config():
    """Creates config.yaml on first run of app"""
    from climb_analyzer.utils.formatting import print_banner

    print_banner("First Run Detected - Creating Default Config")

    # Create a minimal default config.yaml
    default_config = """# Climb Analyzer Configuration
# Generated automatically on first run

# Deployment type: 'local' or 'cloud'
# - local: Use local OSM files (fastest, requires planet files in data/planet_osm_data/)
# - cloud: Use Overpass API (slower, rate-limited, no setup required)
DEPLOYMENT_TYPE: 'cloud'

# Elevation API configuration
TOPO_API_BASE_URL: 'https://api.opentopodata.org/v1'

# Overpass API configuration (for cloud deployment)
OVERPASS_API_URL: 'https://overpass-api.de/api/interpreter'
OVERPASS_API_DELAY_SEC: 0.5
CLOUD_MODE_MAX_RADIUS_KM: 40.0  # ~25 miles, safe for Overpass API
CLOUD_MODE_MAX_RADIUS_MILES: 25.0

# Elevation request settings
ELEVATION_BATCH_SIZE: 100
ELEVATION_REQUEST_TIMEOUT_SEC: 30
ELEVATION_MAX_RETRIES: 3
ELEVATION_BACKOFF_FACTOR: 2
ELEVATION_MAX_CONCURRENT: 10

# Geocoding settings
GEOCODING_MAX_CONCURRENT: 5
GEOCODING_RETRY_ATTEMPTS: 3

# Checkpoint settings
CHECKPOINT_INTERVAL_MIN: 15
CHECKPOINT_MILESTONES_PERC: [10, 25, 50, 75, 90]

# Elevation dataset tiers (for local deployment)
# Options: 'primary', 'primary+secondary', 'primary+secondary+tertiary'
# - primary: SRTM only (~5-50 GB per region, fastest)
# - primary+secondary: SRTM + AW3D30 (~10-100 GB, recommended)
# - primary+secondary+tertiary: SRTM + AW3D30 + ASTER (~15-150 GB, maximum coverage)
ELEVATION_DATASET_TIERS: 'primary+secondary+tertiary'

# Local deployment settings (only used if DEPLOYMENT_TYPE: 'local')
# PLANET_FILE_PATH: (DEPRECATED - use OSM_PLANET_DATA list and -r flag instead)
# ELEVATION_DATASETS: ['aw3d30', 'aster_gdem']
# ELEVATION_COVERAGE: ['Vermont']
# OSM_COVERAGE: ['Vermont']
"""

    try:
        with open("config.yaml", "w") as f:
            f.write(default_config)
        print("✓ Created config.yaml with default settings")
        print("\n" + "=" * 60)
        print("  SETUP COMPLETE")
        print("=" * 60)
        print("\n Default Configuration:")
        print("   • Deployment: Cloud mode (uses Overpass API)")
        print("   • Elevation: opentopodata.org:5000 (start elevation server separately)")
        print("\n Next Steps:")
        print("   1. The analyzer will start automatically")
        print("   2. Choose your analysis scope (address/state/country)")
        print("   3. Data files will be validated and downloaded as needed")
        print("\n🔧 For local deployment (faster):")
        print("   • Edit config.yaml and set DEPLOYMENT_TYPE: 'local'")
        print("   • Download OSM files to data/planet_osm_data/ directory")
        print("   • Run: python data_setup.py (optional, for bulk setup)")
        print("\n" + "=" * 60 + "\n")

        # Pause briefly so user can read the message
        import time

        time.sleep(2)

    except Exception as e:
        print(f"❌ Error creating config.yaml: {e}")
        sys.exit(1)

    return True


if not Path("config.yaml").exists():
    create_default_config()

# Custom libraries
import atexit
import concurrent.futures
import datetime
import gc
import json
import math
import os
import pickle
import shutil
import signal
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# pip install overpy geopy numpy pandas tqdm requests psutil aiohttp osmium rtree reverse_geocoder
# Built-in libraries
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import psutil
import requests

# NOTE: reverse_geocoder is imported locally in functions to avoid loading
# the large spatial index (~8-10GB) at module import time
from geopy.geocoders import Nominatim
from tqdm import tqdm

# Configure tqdm to always show progress bars (disable auto-detection)
# This ensures progress bars work even when stdout is redirected to log files
FORCE_TQDM_OUTPUT = os.environ.get("FORCE_TQDM", "0") == "1"
os.environ.setdefault("TERM", "xterm-256color")  # Ensure TERM is set for tqdm
os.environ["PYTHONIOENCODING"] = "utf-8"  # Force UTF-8 encoding for Unicode progress bars
tqdm.monitor_interval = 0  # Disable monitor thread that can cause issues

# Standard tqdm configuration for solid, resizable progress bars
# Use these defaults for all progress bars to ensure consistent behavior
TQDM_DEFAULTS = {
    "dynamic_ncols": True,  # Auto-adjust to terminal width changes
    "disable": False if FORCE_TQDM_OUTPUT else None,  # Force output when requested
    "file": sys.stderr if FORCE_TQDM_OUTPUT else None,  # Output to stderr for GUI capture
    "mininterval": 0.5 if FORCE_TQDM_OUTPUT else 0.1,  # Update every 0.5s for GUI
    "ascii": " ▏▎▍▌▋▊▉█",  # Gradient block characters for smooth progress bars (9 chars needed for tqdm)
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
}

# Import refactored merger classes from climb_analyzer package
from climb_analyzer.core.merger import BoundaryMerger, _merge_street_batch_worker_top_level
from climb_analyzer.data.elevation import ElevationFetchLog
from climb_analyzer.data.geo_lookup import (
    find_region,
    is_us_state,
)
from climb_analyzer.data.geo_lookup import (
    get_region_bounds as lookup_bounds,
)
from climb_analyzer.processing.checkpoint import ChunkPersistenceManager

# Import cloud cache management
try:
    from utils.cloud_cache import CloudCacheManager
    from utils.config_loader import CLOUD_CACHE_ENABLED, CLOUD_CACHE_REPO

    CLOUD_CACHE_AVAILABLE = True
except ImportError:
    CLOUD_CACHE_AVAILABLE = False
    CLOUD_CACHE_ENABLED = False

# Import data paths
try:
    from utils.data_paths import CHECKPOINT_DIR, OSM_INDEXES_DIR, PLANET_OSM_DIR
except ImportError:
    CHECKPOINT_DIR = Path("data/checkpoint_data")
    OSM_INDEXES_DIR = Path("data/osm_indexes")
    PLANET_OSM_DIR = Path("data/planet_osm_data")

# Memory checker will be imported lazily when needed to avoid startup overhead
MEMORY_CHECK_AVAILABLE = True

# Import version information
try:
    from __version__ import __version__
except ImportError:
    __version__ = "unknown"

# NOTE: Menu functions are also available in geographic_menu.py module
# for use by data_setup.py and other tools. The versions below are kept
# to avoid breaking existing code.

# Try to import configuration
try:
    from utils.config_loader import (
        CHECKPOINT_INTERVAL_MIN,
        CHECKPOINT_MILESTONES_PERC,
        CLOUD_MODE_MAX_RADIUS_KM,
        CLOUD_MODE_MAX_RADIUS_MILES,
        DEPLOYMENT_TYPE,
        ELEVATION_BACKOFF_FACTOR,
        ELEVATION_BATCH_SIZE,
        ELEVATION_MAX_CONCURRENT,
        ELEVATION_MAX_RETRIES,
        ELEVATION_REQUEST_TIMEOUT_SEC,
        ENABLE_CROSS_CHUNK_POSTPROCESS,
        GEOCODING_MAX_CONCURRENT,
        GEOCODING_RETRY_ATTEMPTS,
        OVERPASS_API_DELAY_SEC,
        OVERPASS_API_URL,
        TOPO_API_BASE_URL,
        get_config,
    )
except (ImportError, FileNotFoundError):
    print("\n⚠️  Configuration file not found!")
    create_default_config()
    # Import after creating
    from utils.config_loader import (
        CHECKPOINT_INTERVAL_MIN,
        CHECKPOINT_MILESTONES_PERC,
        CLOUD_MODE_MAX_RADIUS_KM,
        CLOUD_MODE_MAX_RADIUS_MILES,
        DEPLOYMENT_TYPE,
        ELEVATION_BATCH_SIZE,
        ELEVATION_REQUEST_TIMEOUT_SEC,
        ENABLE_CROSS_CHUNK_POSTPROCESS,
        OVERPASS_API_DELAY_SEC,
        OVERPASS_API_URL,
        TOPO_API_BASE_URL,
        get_config,
    )

# OSM processing configuration
# - Local mode: Uses multiprocessing with all CPU cores (rtree/osmium process-safe)
# - Cloud mode: Uses serial processing (Overpass API rate limits require sequential requests)
OSM_MAX_THREADS = cpu_count()  # Use all CPU cores for local mode multiprocessing

# Parallel street merging configuration
MERGE_PARALLEL_ENABLED = True  # Enable parallel processing for street merging
MERGE_MAX_WORKERS = min(16, os.cpu_count() or 1)  # Max workers for parallel merge
MERGE_BATCH_SIZE = 500  # Streets per batch for parallel processing

if DEPLOYMENT_TYPE == "local":
    import osmium
else:
    import overpy


# Custom unpickler to handle module remapping for classes serialized from __main__
# This fixes "Can't get attribute 'ClimbMetrics' on <module '__main__'>" errors
class ClimbAnalyzerUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps classes serialized from __main__ to correct modules."""

    def find_class(self, module, name):
        # Remap classes that were pickled from __main__ to their actual module location
        if module == "__main__":
            if name == "ClimbMetrics":
                module = "climb_analyzer.engine"
            elif name == "ClimbIdentifier":
                module = "climb_analyzer.engine"
            elif name == "SimpleClimbNode":
                module = "climb_analyzer.engine"
            elif name == "ElevationProfile":
                module = "climb_analyzer.engine"
            elif name == "ErrorLogEntry":
                module = "climb_analyzer.data.elevation"
        return super().find_class(module, name)


def safe_pickle_load(file_handle):
    """Load pickle data using custom unpickler that handles __main__ class remapping.

    Use this instead of pickle.load() when loading checkpoint data that may contain
    ClimbMetrics or other classes that were serialized from different module contexts.
    """
    return ClimbAnalyzerUnpickler(file_handle).load()


# Global verbose flag - set by main() after argument parsing
_VERBOSE_MODE = False


def is_verbose() -> bool:
    """Check if verbose mode is enabled via --verbose flag."""
    global _VERBOSE_MODE
    if _VERBOSE_MODE:
        return True
    # Also check sys.modules for args (fallback)
    try:
        if hasattr(sys.modules.get("__main__"), "args"):
            args = sys.modules["__main__"].args
            return getattr(args, "verbose", False)
    except Exception:
        pass
    return False


def verbose_print(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if is_verbose():
        print(*args, **kwargs)


def get_configured_osm_file_path():
    """Get the OSM file path based on PLANET_FILE_PATH config."""
    if PLANET_FILE_PATH and Path(PLANET_FILE_PATH).exists():
        # Return the full OSM file path - SpatialIndexManager will calculate the index paths
        return str(PLANET_FILE_PATH)
    else:
        # Do NOT fall back to arbitrary file - this causes wrong regions to be used
        # Return None to signal that no configured file exists
        return None


def find_osm_file_for_region(region_name: str, canonical_path: Optional[str] = None):
    """
    Find OSM PBF file matching a region name.

    Searches data/planet_osm_data/ directory for files matching the region name.
    Handles various naming conventions (lowercase, with/without hyphens, etc.)

    For ambiguous regions (e.g., Georgia the US state vs Georgia the country),
    the canonical_path parameter is used to disambiguate and prevent using the
    wrong PBF file.

    Args:
        region_name: Name of region (e.g., "Antarctica", "Alaska", "Vermont")
        canonical_path: Optional full path from geo_lookup (e.g., "us/georgia"
                       or "europe/georgia") used to disambiguate collisions

    Returns:
        Path to matching PBF file, or None if not found
    """
    # Use new data structure
    try:
        from utils.data_paths import PLANET_OSM_DIR

        planet_dir = PLANET_OSM_DIR
    except ImportError:
        planet_dir = Path("./data/planet_osm_data")

    if not planet_dir.exists():
        return None

    # Normalize region name to lowercase, replace spaces and underscores with hyphens
    # Handles: "South Carolina" -> "south-carolina", "south_carolina" -> "south-carolina"
    normalized = region_name.lower().replace(" ", "-").replace("_", "-")

    # If a canonical path is provided and the stem is known ambiguous, check
    # for the disambiguated filename first (e.g., "us_georgia-latest.osm.pbf").
    from climb_analyzer.data.osm_downloader import AMBIGUOUS_STEMS
    if canonical_path and normalized in AMBIGUOUS_STEMS:
        # Extract the immediate parent from canonical path
        # e.g., "us/georgia" -> "us", "europe/georgia" -> "europe"
        parts = canonical_path.split("/")
        if len(parts) >= 2:
            parent = parts[-2]
            disambiguated = planet_dir / f"{parent}_{normalized}-latest.osm.pbf"
            if disambiguated.exists():
                return disambiguated
            # Disambiguated file doesn't exist - do NOT fall back to the
            # bare name, as it may be the wrong region's file. Return None
            # so the caller knows to download the correct one.
            return None

    # Try exact match first
    pattern = f"{normalized}-latest.osm.pbf"
    exact_match = planet_dir / pattern
    if exact_match.exists():
        return exact_match

    # Try glob pattern with wildcards
    pattern = f"{normalized}*.osm.pbf"
    matches = list(planet_dir.glob(pattern))
    if matches:
        return matches[0]

    # Try without hyphens (e.g., "newyork" instead of "new-york")
    pattern = f"{normalized.replace('-', '')}*.osm.pbf"
    matches = list(planet_dir.glob(pattern))
    if matches:
        return matches[0]

    # Try with underscores instead of hyphens
    pattern = f"{normalized.replace('-', '_')}*.osm.pbf"
    matches = list(planet_dir.glob(pattern))
    if matches:
        return matches[0]

    return None


def check_opentopodata_ready(max_retries=15, retry_delay=2):
    """
    Check if OpenTopoData server is ready to serve requests.
    Called AFTER region selection and data confirmation.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries

    Returns:
        bool: True if server is ready, False otherwise
    """
    if DEPLOYMENT_TYPE != "local":
        return True  # Not needed in cloud mode

    if not TOPO_API_BASE_URL or "opentopodata" not in TOPO_API_BASE_URL:
        return True  # Using external API

    print("\nChecking OpenTopoData elevation server...")

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(f"{TOPO_API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ OpenTopoData server is ready")
                return True
        except requests.exceptions.RequestException:
            if attempt < max_retries:
                print(f"  Attempt {attempt}/{max_retries} - waiting for server...")
                time.sleep(retry_delay)
            else:
                print("\n⚠️  OpenTopoData server not responding")
                print("   You may need to start it with: docker compose up -d opentopodata")
                print("   Or check the opentopodata-config.yaml has the correct datasets")
                return False

    return False


# Road surface filtering options
ROAD_SURFACE_FILTERS = {
    "paved": [
        'highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|footway|cycleway|bridleway|path"',
        'surface!~"unpaved|gravel|dirt|sand|grass|ground|earth|mud|clay"',
    ],
    "gravel": [
        'highway~"track|path|unclassified|tertiary|residential|service|footway|bridleway"',
        'surface~"gravel|compacted|fine_gravel"',
    ],
    "dirt": ['highway~"track|path|footway|bridleway"', "tracktype"],
    "all": [
        'highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|track|path|footway|cycleway|bridleway"'
    ],
}

TRACKTYPE_DEFINITIONS = {
    "1-solid": "Solid: paved or heavily compacted hardcore surface",
    "2-gravel_rd": "Mostly solid: unpaved track with through traffic, gravel/dirt with some soft areas",
    "3-doubletrack": "Even mixture of hard and soft materials, gravel/dirt with many soft areas",
    "4-mostly_soft": "Mostly soft: soil/sand/grass with some hard material mixed in",
    "5-unimproved": "Soft: soil/sand/grass, no hard material. Almost impossible for cars",
}


# =============================================================================
# CLOUD CACHE HELPER FUNCTIONS
# =============================================================================


def find_analysis_output_files(
    output_dir: Path, region_name: str, surface: str, score: str, date_str: str
) -> Dict:
    """
    Find all output files from a completed analysis.

    Used after analysis completes to locate files for upload to cloud cache.

    Args:
        output_dir: Output directory
        region_name: Name of region/country
        surface: Surface filter used
        score: Score type used
        date_str: Date string (YYYY-MM-DD)

    Returns:
        Dict with 'xlsx' (list of Paths) and 'csv' (Path or None)
    """
    output_dir = Path(output_dir)
    # Replace spaces with underscores to match how filenames are created
    safe_region_name = region_name.replace(" ", "_")
    base_pattern = f"{safe_region_name}_climbs_{surface}_{score}_{date_str}"

    xlsx_files = []
    csv_file = None

    # Find all xlsx files (may be split: file-1.xlsx, file-2.xlsx, etc.)
    for file in output_dir.glob(f"{base_pattern}*.xlsx"):
        xlsx_files.append(file)

    # Find txt file (CSV format inside) - use wildcard to match version suffix
    # Example: Hawaii_errors_all_basic_2025-11-21_v2.0.1_e0000.txt
    csv_pattern = f"{safe_region_name}_errors_{surface}_{score}_{date_str}*.txt"
    csv_matches = list(output_dir.glob(csv_pattern))
    if csv_matches:
        csv_file = csv_matches[0]

    # Sort xlsx files by part number
    xlsx_files.sort()

    return {"xlsx": xlsx_files, "csv": csv_file}


def prompt_download_cache(location_name: str, date: str, parts: int = 1) -> bool:
    """
    Prompt user to download cached analysis.

    Args:
        location_name: Name of location (country or state)
        date: Date of cached analysis
        parts: Number of Excel file parts

    Returns:
        True if user wants to download, False otherwise
    """
    from climb_analyzer.utils.formatting import print_header, print_info, print_success

    parts_str = f" ({parts} files)" if parts > 1 else ""

    print_header("Pre-Analyzed Climbs Available", spacing_before=1)
    print_success(f"{location_name} - {date}{parts_str}", indent=0)
    print_info(
        "Download prior generated climb analysis for this region from cloud cache, instead of processing locally? (saves hours)",
        indent=0,
    )
    print()

    response = (
        input("Use cloud cached analysis instad of locally processing? [Y/n]: ").strip().lower()
    )
    # Default to yes if user just presses Enter
    return response == "y" or response == ""


def prompt_download_parent_cache(country: str, region: str) -> bool:
    """
    Prompt user to download parent country and filter for region.

    Args:
        country: Parent country name
        region: Region/state name

    Returns:
        True if user wants to download and filter, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"✓ {country} analysis available in cloud cache")
    print(f"  Download and filter for {region} locally?")
    print("  (Faster than full processing)")
    print(f"{'='*70}")

    response = input("Download and filter? [y/n]: ").strip().lower()
    return response == "y"


def prompt_run_clean_for_community(location_name: str) -> bool:
    """
    Prompt user to run clean analysis for cloud cache contribution.

    Args:
        location_name: Name of location (country or state)

    Returns:
        True if user wants to run clean analysis, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"⚠️  {location_name} not in cloud cache")
    print(f"{'='*70}")
    print("Run full analysis (all roads, no filters, min_score=0) to share with community?")
    print("Your analysis will be uploaded for review and benefit all users.")
    print(f"{'='*70}")

    response = input("Run clean analysis for community? [y/n]: ").strip().lower()
    return response == "y"


def prompt_contribute_to_cache(file_count: int = 1) -> bool:
    """
    Prompt user to upload completed analysis to cloud cache.

    Args:
        file_count: Number of files to upload

    Returns:
        True if user wants to contribute, False otherwise
    """
    from climb_analyzer.utils.formatting import print_header, print_list_item, print_separator

    files_str = f"{file_count} files" if file_count > 1 else "1 file"
    print_header("Share this analysis with the community?", spacing_before=2)
    print_list_item("Creates pull request for review")
    print_list_item(f"Will upload {files_str}")
    print_list_item("Helps other users save processing time")
    print_separator()

    response = input("Create pull request? [y/n]: ").strip().lower()
    return response == "y"


def get_region_bbox_from_definitions(region_path: str) -> Optional[Tuple]:
    """
    Get bounding box for a region from osm_pbf_urls.

    Uses the consolidated geo_lookup module which contains bounds
    from Geofabrik .poly files for all regions.

    Args:
        region_path: Region path (e.g., 'us/vermont', 'asia/japan', 'bristol')

    Returns:
        (lat_min, lon_min, lat_max, lon_max) or None if not found
    """
    # Extract region name from path (e.g., 'us/vermont' -> 'vermont')
    if "/" in region_path:
        region_name = region_path.split("/")[-1]
    else:
        region_name = region_path

    return lookup_bounds(region_name)


@dataclass
class CheckpointConfig:
    """Global configuration for checkpoint saving"""

    time_interval_minutes: float = CHECKPOINT_INTERVAL_MIN
    progress_milestones: list = None
    save_at_completion: bool = True

    def __post_init__(self):
        if self.progress_milestones is None:
            self.progress_milestones = CHECKPOINT_MILESTONES_PERC


CHECKPOINT_CONFIG = CheckpointConfig()


class Tee:
    """
    A custom file-like object that redirects output to multiple streams.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        # Write the output to every file/stream provided
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate writing

    def flush(self):
        # Flush the buffer for every file/stream
        for f in self.files:
            f.flush()

    def isatty(self):
        # Return True if any of the streams is a TTY (needed for tqdm progress bars)
        return any(hasattr(f, "isatty") and f.isatty() for f in self.files)

    def fileno(self):
        # Return the fileno of the first stream that has one (needed for tqdm)
        for f in self.files:
            if hasattr(f, "fileno"):
                try:
                    return f.fileno()
                except (OSError, ValueError):
                    pass
        raise AttributeError("No underlying file has fileno()")


def estimate_data_size(bounds_data):
    """
    Estimate OSM + DEM data size based on geographic area.
    Returns human-readable string like "~2.5 GB"
    """
    # Calculate area in square degrees
    lat_range = abs(bounds_data["lat_max"] - bounds_data["lat_min"])
    lon_range = abs(bounds_data["lon_max"] - bounds_data["lon_min"])
    area_sq_deg = lat_range * lon_range

    # Rough estimates (very approximate):
    # - OSM: ~50-200 MB per square degree (varies by density)
    # - DEM: ~20-50 MB per square degree for SRTM/ASTER
    # Using conservative mid-range estimates
    osm_mb_per_sq_deg = 100  # MB per square degree
    dem_mb_per_sq_deg = 30  # MB per square degree

    total_mb = (osm_mb_per_sq_deg + dem_mb_per_sq_deg) * area_sq_deg

    # Format as human-readable
    if total_mb < 100:
        return f"~{int(total_mb)} MB"
    elif total_mb < 1024:
        return f"~{int(total_mb)} MB"
    else:
        gb = total_mb / 1024
        if gb < 10:
            return f"~{gb:.1f} GB"
        else:
            return f"~{int(gb)} GB"


def display_selection_menu(data, title):
    """
    Displays a two-column numerical selection menu for geographic regions.
    Shows estimated data size instead of abbreviations.
    """
    if not data:
        print(f"Error: No data found for {title}.")
        return None, None

    data_list = sorted(list(data.items()), key=lambda x: x[0])
    num_entries = len(data_list)
    mid_point = (num_entries + 1) // 2

    max_name_len = 0
    for key, value in data_list:
        max_name_len = max(max_name_len, len(key))

    name_width = max_name_len + 2

    print(f"\n{title} (select by number, separate multiple with commas):")
    print("-" * 80)

    for i in range(mid_point):
        left_key, left_data = data_list[i]

        # Show estimated data size instead of abbreviation
        left_size = estimate_data_size(left_data)
        left_entry = f" {i + 1:2d}. {left_key:<{name_width}} {left_size:>10}"
        right_entry = ""
        if i + mid_point < num_entries:
            right_key, right_data = data_list[i + mid_point]
            right_size = estimate_data_size(right_data)
            right_entry = f" {i + mid_point + 1:2d}. {right_key:<{name_width}} {right_size:>10}"

        print(f"{left_entry} {right_entry}")

    all_keys_list = [key for key, _ in data_list]

    while True:
        try:
            choice_input = input(f"\nEnter {title} number(s) (e.g., 5,15,28): ").strip()

            if not choice_input:
                return [], all_keys_list

            choices = [int(c.strip()) for c in choice_input.split(",") if c.strip().isdigit()]

            selected_keys = []
            for choice in choices:
                index = choice - 1
                if 0 <= index < num_entries:
                    selected_keys.append(all_keys_list[index])
                else:
                    print(
                        f"❌ Invalid selection: {choice}. Please enter a number between 1 and {num_entries}."
                    )
                    raise ValueError

            return selected_keys, all_keys_list

        except ValueError:
            pass
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            pass


def select_us_states(state_data):
    """Wrapper function to select US States."""
    selected_keys, all_keys_list = display_selection_menu(state_data, "US States")
    if selected_keys:
        return "state", selected_keys, None
    else:
        return "state", [], None


def select_countries(country_data):
    """Wrapper function to select Countries with continent grouping."""

    # Define continent groupings based on country names and geography
    continents = {
        "North America": [
            "Canada",
            "United States of America",
            "Mexico",
            "Greenland",
            "Cuba",
            "Haiti",
            "Dominican Rep.",
            "Jamaica",
            "Trinidad and Tobago",
            "Bahamas",
            "Belize",
            "Costa Rica",
            "El Salvador",
            "Guatemala",
            "Honduras",
            "Nicaragua",
            "Panama",
            "Puerto Rico",
        ],
        "South America": [
            "Argentina",
            "Bolivia",
            "Brazil",
            "Chile",
            "Colombia",
            "Ecuador",
            "Guyana",
            "Paraguay",
            "Peru",
            "Suriname",
            "Uruguay",
            "Venezuela",
            "Fr. Guiana",
            "Falkland Is.",
        ],
        "Europe": [
            "Albania",
            "Andorra",
            "Austria",
            "Belarus",
            "Belgium",
            "Bosnia and Herz.",
            "Bulgaria",
            "Croatia",
            "Cyprus",
            "Czechia",
            "Czech Rep.",
            "Denmark",
            "Estonia",
            "Finland",
            "France",
            "Germany",
            "Greece",
            "Hungary",
            "Iceland",
            "Ireland",
            "Italy",
            "Kosovo",
            "Latvia",
            "Lithuania",
            "Luxembourg",
            "Macedonia",
            "Malta",
            "Moldova",
            "Monaco",
            "Montenegro",
            "Netherlands",
            "Norway",
            "Poland",
            "Portugal",
            "Romania",
            "Russia",
            "San Marino",
            "Serbia",
            "Slovakia",
            "Slovenia",
            "Spain",
            "Sweden",
            "Switzerland",
            "Ukraine",
            "United Kingdom",
            "Vatican",
        ],
        "Africa": [
            "Algeria",
            "Angola",
            "Benin",
            "Botswana",
            "Burkina Faso",
            "Burundi",
            "Cameroon",
            "Central African Rep.",
            "Chad",
            "Comoros",
            "Dem. Rep. Congo",
            "Congo",
            "Côte d'Ivoire",
            "Ivory Coast",
            "Djibouti",
            "Egypt",
            "Eq. Guinea",
            "Equatorial Guinea",
            "Eritrea",
            "eSwatini",
            "Swaziland",
            "Ethiopia",
            "Gabon",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Kenya",
            "Lesotho",
            "Liberia",
            "Libya",
            "Madagascar",
            "Malawi",
            "Mali",
            "Mauritania",
            "Mauritius",
            "Morocco",
            "Mozambique",
            "Namibia",
            "Niger",
            "Nigeria",
            "Rwanda",
            "S. Sudan",
            "South Sudan",
            "São Tomé and Principe",
            "Senegal",
            "Seychelles",
            "Sierra Leone",
            "Somalia",
            "Somaliland",
            "South Africa",
            "Sudan",
            "Tanzania",
            "Togo",
            "Tunisia",
            "Uganda",
            "W. Sahara",
            "Zambia",
            "Zimbabwe",
        ],
        "Asia": [
            "Afghanistan",
            "Armenia",
            "Azerbaijan",
            "Bahrain",
            "Bangladesh",
            "Bhutan",
            "Brunei",
            "Cambodia",
            "China",
            "Georgia",
            "India",
            "Indonesia",
            "Iran",
            "Iraq",
            "Israel",
            "Japan",
            "Jordan",
            "Kazakhstan",
            "Kuwait",
            "Kyrgyzstan",
            "Laos",
            "Lebanon",
            "Malaysia",
            "Maldives",
            "Mongolia",
            "Myanmar",
            "Nepal",
            "North Korea",
            "Dem. Rep. Korea",
            "Oman",
            "Pakistan",
            "Palestine",
            "Philippines",
            "Qatar",
            "Saudi Arabia",
            "Singapore",
            "South Korea",
            "Korea",
            "Sri Lanka",
            "Syria",
            "Taiwan",
            "Tajikistan",
            "Thailand",
            "Timor-Leste",
            "Turkey",
            "Turkmenistan",
            "United Arab Emirates",
            "Uzbekistan",
            "Vietnam",
            "Yemen",
        ],
        "Oceania": [
            "Australia",
            "Fiji",
            "Kiribati",
            "Marshall Is.",
            "Micronesia",
            "Nauru",
            "New Zealand",
            "Palau",
            "Papua New Guinea",
            "Samoa",
            "Solomon Is.",
            "Tonga",
            "Tuvalu",
            "Vanuatu",
            "Fr. Polynesia",
            "New Caledonia",
        ],
    }

    # First, let user choose continent
    from climb_analyzer.utils.formatting import print_header

    print_header("Select Continent", spacing_before=1)
    continent_list = sorted(continents.keys())
    for i, continent in enumerate(continent_list, 1):
        country_count = sum(1 for country in continents[continent] if country in country_data)
        print(f"  {i}. {continent:<20} ({country_count} countries)")

    print(f"  {len(continent_list) + 1}. All Countries")
    print()

    while True:
        try:
            continent_choice = input(
                f"\nEnter continent number (1-{len(continent_list) + 1}): "
            ).strip()
            if not continent_choice:
                return "country", [], None

            choice_num = int(continent_choice)
            if choice_num == len(continent_list) + 1:
                # Show all countries
                selected_keys, all_keys_list = display_selection_menu(country_data, "Countries")
                if selected_keys:
                    return "country", selected_keys, None
                else:
                    return "country", [], None
            elif 1 <= choice_num <= len(continent_list):
                # Filter countries by continent
                selected_continent = continent_list[choice_num - 1]
                continent_countries = continents[selected_continent]

                # Create filtered country_data dictionary
                filtered_data = {
                    country: data
                    for country, data in country_data.items()
                    if country in continent_countries
                }

                if not filtered_data:
                    print(f"No countries found for {selected_continent}")
                    continue

                selected_keys, all_keys_list = display_selection_menu(
                    filtered_data, f"{selected_continent} Countries"
                )
                if selected_keys:
                    return "country", selected_keys, None
                else:
                    return "country", [], None
            else:
                print(f"Invalid choice. Please enter 1-{len(continent_list) + 1}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            return "country", [], None


def fix_output_file_permissions(file_path: Path) -> None:
    """
    Fix permissions on output files created by Docker containers.

    Docker containers run as root, creating files with restrictive permissions (600, root:root).
    This function makes files readable by all users (644) and owned by the host user.

    Args:
        file_path: Path to the file to fix permissions on
    """
    try:
        # Make file readable by all (644)
        file_path.chmod(0o644)

        # Try to change ownership if running as root (in Docker)
        if os.geteuid() == 0:
            # Use HOST_UID/HOST_GID environment variables if available (set by docker-compose)
            # Otherwise, infer from parent directory ownership
            host_uid = int(os.environ.get("HOST_UID", 0))
            host_gid = int(os.environ.get("HOST_GID", 0))

            if host_uid == 0 or host_gid == 0:
                # Fallback: get the user who owns the parent directory
                parent_stat = file_path.parent.stat()
                host_uid = parent_stat.st_uid
                host_gid = parent_stat.st_gid

            os.chown(file_path, host_uid, host_gid)
    except (PermissionError, AttributeError, ValueError):
        # AttributeError: os.geteuid() doesn't exist on Windows
        # PermissionError: Can't change ownership (not running as root)
        # ValueError: Invalid UID/GID in environment variables
        # All are fine - permissions are already workable or we can't fix them
        pass


def format_excel_worksheet(worksheet, df):
    """
    Format Excel worksheet with proper column types and auto-sized widths.

    Args:
        worksheet: openpyxl worksheet object
        df: pandas DataFrame used to generate the worksheet
    """

    # Define numeric columns that should be formatted as numbers
    numeric_columns = {
        "From Center (km)": "0.0",
        "Latitude": "0.000000",
        "Longitude": "0.000000",
        "Basic Score": "0",
        "FIETS Score": "0",
        "PDI Score": "0.0",
        "Elev Gain (m)": "0.0",
        "Height (m)": "0.0",
        "Prominence (m)": "0.0",
        "Length (km)": "0.00",
        "Avg Grade (%)": "0.00",
        "Max Grade (%)": "0.00",
    }

    # Get column indices for numeric columns
    header_row = [cell.value for cell in worksheet[1]]

    # Apply number formatting to numeric columns
    for col_name, num_format in numeric_columns.items():
        if col_name in header_row:
            col_idx = header_row.index(col_name) + 1  # openpyxl is 1-indexed
            # Apply formatting to all data rows (skip header)
            for row_idx in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.number_format = num_format

    # Auto-size all columns based on content
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter

        for cell in column:
            try:
                # Calculate length considering cell value
                if cell.value:
                    # For hyperlinks/formulas, use the display text
                    cell_str = str(cell.value)
                    # Account for markdown links like [123](url)
                    if cell_str.startswith("[") and "](" in cell_str:
                        # Extract just the display text from [text](url)
                        cell_str = cell_str.split("]")[0][1:]
                    max_length = max(max_length, len(cell_str))
            except:
                pass

        # Set column width with some padding (add 2 for comfort)
        # Excel column width units are approximately character width
        adjusted_width = min(max_length + 2, 50)  # Cap at 50 to avoid extremely wide columns
        worksheet.column_dimensions[column_letter].width = adjusted_width


def save_large_dataframe_as_split_excel(
    df: pd.DataFrame,
    base_filename: Path,
    sort_column: str = "Basic Score",
    max_file_size_mb: float = 100.0,  # 100 MB max per file for GitHub
    app_version: str = None,
    elevation_errors: int = 0,
    enable_merge: bool = True,  # Enable climb merging before sorting
    region_info: Optional[Dict] = None,  # Region metadata for merging
) -> List[Path]:
    """
    Save a large dataframe as multiple Excel files when it exceeds the file size limit.
    Files are split after sorting so that the hardest climbs are in file-1.

    Args:
        df: DataFrame to save
        base_filename: Base path for output files (without extension)
        sort_column: Column to sort by (descending) to put hardest climbs first
        max_file_size_mb: Maximum file size in MB (default 100MB for GitHub)
        app_version: Application version to append to filename (e.g., "2.0.0")
        elevation_errors: Number of elevation fetch errors to append to filename
        enable_merge: Whether to perform climb merging before sorting (default True)
        region_info: Optional region metadata for merging context

    Returns:
        List of created file paths
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path as TempPath

    import yaml

    # DISK SPACE FIX: Use checkpoint_data directory for temp files
    # Enables recovery if export phase crashes, and uses volume-mounted storage
    temp_dir = TempPath("data/checkpoint_data/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    row_count = len(df)

    # Check for --no-merge flag from command-line
    import sys

    if hasattr(sys.modules.get("__main__"), "args"):
        args = sys.modules["__main__"].args
        if hasattr(args, "no_merge") and args.no_merge:
            enable_merge = False

    # Perform climb merging before sorting if enabled
    if enable_merge:
        try:
            # Load merge rules from config
            with open("config.yaml") as f:
                config = yaml.safe_load(f) or {}
            merge_rules = config.get("merge_rules", {})

            # Apply default merge rules if not in config
            if not merge_rules:
                merge_rules = {
                    "max_merge_distance_km": 0.5,
                    "length_tolerance_percent": 0,
                    "allow_cross_country_merge": False,
                    "allowed_cross_country_merging": [],
                }

            # Check for command-line argument overrides
            # Look for global args object if available
            import sys

            if hasattr(sys.modules.get("__main__"), "args"):
                args = sys.modules["__main__"].args
                if hasattr(args, "allow_cross_country_merge") and args.allow_cross_country_merge:
                    merge_rules["allow_cross_country_merge"] = True
                if hasattr(args, "merge_distance_km") and args.merge_distance_km is not None:
                    merge_rules["max_merge_distance_km"] = args.merge_distance_km

            # Import and run the unified merger
            from climb_analyzer.core.unified_climb_merger import UnifiedClimbMerger

            merger = UnifiedClimbMerger(merge_rules, region_info, verbose=True)
            df = merger.merge_all_climbs(df)
            row_count = len(df)  # Update count after merging
        except Exception as e:
            print(f"⚠️ Warning: Climb merging failed: {e}")
            print("   Continuing without merging...")

    # Sort by difficulty (descending) so hardest climbs are first
    df_sorted = df.sort_values(by=sort_column, ascending=False)

    # Build filename suffix with version and error count
    suffix = ""
    if app_version:
        # Format version for filename (e.g., "2.0.0" -> "v2.0.0")
        suffix += f"_v{app_version}"
    if elevation_errors > 0:
        # Format error count (e.g., 123 -> "e123")
        suffix += f"_e{elevation_errors:04d}"
    elif elevation_errors == 0 and app_version:
        # Include e0000 if there are no errors
        suffix += "_e0000"

    # Initialize file_size_mb (will be set if single file creation succeeds)
    file_size_mb = None

    # Try to save the entire dataset as a single file first
    temp_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, dir=str(temp_dir))
    temp_file.close()

    try:
        with pd.ExcelWriter(temp_file.name, engine="openpyxl") as writer:
            df_sorted.to_excel(writer, sheet_name="Climbs", index=False)
            worksheet = writer.sheets["Climbs"]

            # Skip autofilter for very large datasets (>500K rows) to avoid openpyxl errors
            # For large files, we'll split them anyway, so autofilter on single file isn't needed
            if row_count <= 500000:
                try:
                    worksheet.auto_filter.ref = worksheet.dimensions
                except Exception:
                    pass  # Autofilter failed, not critical

            # Format columns and auto-size widths
            format_excel_worksheet(worksheet, df_sorted)

        # Check file size
        file_size_mb = os.path.getsize(temp_file.name) / (1024 * 1024)

        # If within limits, just move (not rename) to handle cross-device moves
        if file_size_mb <= max_file_size_mb:
            output_file = Path(f"{base_filename}{suffix}.xlsx")

            # Protect against overwriting: rename existing file to .backup
            if output_file.exists():
                backup_file = output_file.with_suffix(".xlsx.backup")
                if backup_file.exists():
                    backup_file.unlink()
                output_file.rename(backup_file)
                print(f" ℹ️  Existing file renamed to {backup_file.name}")

            shutil.move(temp_file.name, output_file)
            fix_output_file_permissions(output_file)
            print(f" Saved {output_file.name} ({row_count:,} rows, {file_size_mb:.1f} MB)")
            return [output_file]
        else:
            # File too large, need to split
            os.unlink(temp_file.name)
            print(
                f" File would be {file_size_mb:.1f} MB, splitting into multiple files (max {max_file_size_mb} MB each)..."
            )
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        print(f"⚠️ Error creating single file: {e}")
        print("   Proceeding with split file creation...")

    # Need to split - use fast estimation instead of slow binary search
    # Estimate rows per file based on file size ratio (much faster!)
    if file_size_mb is not None:
        # Use actual file size if we have it
        num_files_needed = int(file_size_mb / max_file_size_mb) + 1
    else:
        # Estimate: assume ~1MB per 1000 rows as a rough heuristic
        estimated_total_size_mb = row_count / 1000
        num_files_needed = max(int(estimated_total_size_mb / max_file_size_mb) + 1, 2)

    estimated_rows_per_file = row_count // num_files_needed

    # Apply 10% safety margin to ensure files stay under limit
    rows_per_file = int(estimated_rows_per_file * 0.9)

    created_files = []
    start_idx = 0
    file_num = 1

    print(
        f" Splitting {row_count:,} rows into multiple Excel files (max {max_file_size_mb} MB each, hardest climbs first)..."
    )
    print(f"   Estimated {num_files_needed} files needed, ~{rows_per_file:,} rows each")

    while start_idx < row_count:
        end_idx = min(start_idx + rows_per_file, row_count)
        df_chunk = df_sorted.iloc[start_idx:end_idx]

        # Create filename with suffix: basename_v2.0.0_e0000-1.xlsx, -2.xlsx, etc.
        output_file = Path(f"{base_filename}{suffix}-{file_num}.xlsx")

        # Protect against overwriting: rename existing file to .backup
        if output_file.exists():
            backup_file = output_file.with_suffix(".xlsx.backup")
            if backup_file.exists():
                backup_file.unlink()
            output_file.rename(backup_file)
            print(f"   ℹ️  Existing file renamed to {backup_file.name}")

        try:
            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                df_chunk.to_excel(writer, sheet_name="Climbs", index=False)
                worksheet = writer.sheets["Climbs"]

                # Add autofilter with error handling (openpyxl can be finicky)
                try:
                    worksheet.auto_filter.ref = worksheet.dimensions
                except Exception:
                    pass  # Autofilter failed, not critical

                # Format columns and auto-size widths
                format_excel_worksheet(worksheet, df_chunk)

            file_size_mb_chunk = os.path.getsize(output_file) / (1024 * 1024)
            created_files.append(output_file)
            fix_output_file_permissions(output_file)
            print(
                f"   ✓ Saved {output_file.name} ({len(df_chunk):,} rows, {file_size_mb_chunk:.1f} MB)"
            )
        except Exception as e:
            print(f"   ⚠️ Could not save {output_file.name}: {e}")

        start_idx = end_idx
        file_num += 1

    # Check if any files are slightly over limit (expected with estimation)
    over_limit = [
        f for f in created_files if os.path.getsize(f) / (1024 * 1024) > max_file_size_mb * 1.05
    ]
    if over_limit:
        print(
            f"   ⚠️  {len(over_limit)} file(s) slightly over {max_file_size_mb}MB limit (estimation variance)"
        )

    return created_files


def prompt_cross_region_analysis(
    location: List[Tuple[str, List[str]]],
    batch_tracker,
    surface_filter: str,
    score_type: str,
    radius_km: float,
) -> None:
    """
    Prompt user to run cross-region climb analysis after batch completion.

    This function checks if climbs might span across multiple analyzed regions
    and offers to merge them using the cross-region merge script.

    Args:
        location: List of (continent, path) tuples representing analyzed regions
        batch_tracker: BatchProgressTracker instance with completion info
        surface_filter: Surface filter used in analysis
        score_type: Score type used in analysis
        radius_km: Radius used in analysis
    """
    import subprocess
    from pathlib import Path

    # Get list of completed regions
    summary = batch_tracker.get_progress_summary()
    completed_regions = summary["locations"]["completed"]

    if len(completed_regions) < 2:
        # Need at least 2 regions to do cross-region analysis
        return

    # Determine if all regions are in the same country
    # Parse country from region names
    # Region name format: "Continent > Country > ... > Region" or similar
    countries = set()
    for region_name in completed_regions:
        parts = region_name.split(" > ")
        # Extract country: if we have at least 2 parts, second part is likely the country
        # If only 1 part (continent level), it's not same country
        if len(parts) >= 2:
            countries.add(parts[1])  # Second part is typically the country
        elif len(parts) == 1:
            # Only continent level, treat each as different
            countries.add(region_name)  # Treat continent as unique identifier

    same_country = len(countries) == 1

    # Default behavior based on country
    if same_country:
        default_response = "yes"
        default_msg = "all regions within same country - defaulting to YES"
    else:
        default_response = "no"
        default_msg = "regions span multiple countries - defaulting to NO"

    from climb_analyzer.utils.formatting import print_banner, print_list_item

    print_banner("Cross-Region Climb Analysis", spacing_before=1)
    print(f"Analyzed {len(completed_regions)} regions:")
    for region in completed_regions:
        print_list_item(region)

    print("\nSome climbs may extend across regional boundaries.")
    print("Would you like to run cross-region analysis to detect and merge")
    print("climbs that span multiple regions?")
    print(f"\n({default_msg})")

    # Prompt user
    try:
        response = (
            input(f"\nRun cross-region analysis? (y/n) [{default_response}]: ").strip().lower()
        )

        if not response:
            response = default_response

        if response not in ["yes", "y", "no", "n"]:
            print(f"Invalid response '{response}', using default: {default_response}")
            response = default_response

    except KeyboardInterrupt:
        print(f"\nCancelled, using default: {default_response}")
        response = default_response

    if response in ["no", "n"]:
        print("\nSkipping cross-region analysis.")
        return

    # User wants to run cross-region analysis
    print("\nStarting cross-region climb analysis...")
    print(f"This will compare climbs between all {len(completed_regions)} regions.\n")

    # Find output files for each region
    output_dir = Path("output")
    output_files = []

    for region_name in completed_regions:
        # Construct expected filename pattern (without radius since it varies by region)
        # Use same logic as run_single_location_analysis for consistency
        safe_name = "".join(c for c in region_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_name = safe_name.replace(" ", "_")

        # Search for files matching the pattern (glob to handle different radii)
        # Pattern: climbs_{RegionName}_{surface}_region_*km.xlsx
        pattern = f"climbs_{safe_name}_{surface_filter}_region_*.xlsx"
        matching_files = list(output_dir.glob(pattern))

        # Also try CSV if no XLSX found
        if not matching_files:
            pattern = f"climbs_{safe_name}_{surface_filter}_region_*.csv"
            matching_files = list(output_dir.glob(pattern))

        if matching_files:
            # Use most recent file if multiple matches
            most_recent = max(matching_files, key=lambda p: p.stat().st_mtime)
            output_files.append((region_name, most_recent))
        else:
            print(f"⚠️  Could not find output file for {region_name}")
            print(f"    Expected pattern: output/{pattern}")

    if len(output_files) < 2:
        print("\n❌ Error: Need at least 2 output files to run cross-region analysis")
        print(f"   Found only {len(output_files)} output file(s)")
        return

    # Run cross-region analysis for each pair of adjacent regions
    print(f"\nFound {len(output_files)} output files")
    print("Analyzing pairs of regions for cross-region climbs...\n")

    script_path = Path("scripts") / "merge_cross_region_climbs.py"

    if not script_path.exists():
        print(f"❌ Error: Cross-region merge script not found at {script_path}")
        return

    # Compare each adjacent pair (and potentially all pairs if needed)
    total_merged = 0

    for i in range(len(output_files) - 1):
        region1_name, file1 = output_files[i]
        region2_name, file2 = output_files[i + 1]

        print(f"\n{'─' * 80}")
        print("Comparing:")
        print(f"  Region 1: {region1_name}")
        print(f"  Region 2: {region2_name}")
        print(f"{'─' * 80}\n")

        try:
            # Run the merge script
            result = subprocess.run(
                ["python3", str(script_path), str(file1), str(file2)],
                capture_output=False,  # Show output in real-time
                text=True,
            )

            if result.returncode == 0:
                print(f"\n✓ Completed comparison between {region1_name} and {region2_name}")
            else:
                print(
                    f"\n⚠️  Error comparing {region1_name} and {region2_name} (exit code: {result.returncode})"
                )

        except Exception as e:
            print(f"\n❌ Error running cross-region analysis: {e}")
            continue

    print("\n" + "=" * 80)
    print("CROSS-REGION ANALYSIS COMPLETE")
    print("=" * 80)


def offer_automatic_merge(subregion_names: List[str], surface_filter: str, score_type: str) -> bool:
    """
    Automatically merge subregion results after split country analysis.

    This function automatically merges split subregions (e.g., France North + France South)
    without prompting the user.

    Args:
        subregion_names: List of subregion names that were analyzed (e.g., ["France North", "France South"])
        surface_filter: Surface filter used in analysis
        score_type: Score type used in analysis

    Returns:
        True if merge was performed successfully, False otherwise
    """
    import subprocess
    from pathlib import Path

    from climb_analyzer.utils.formatting import print_error, print_header, print_success

    if len(subregion_names) < 2:
        return False

    print_header("Merge Split Regions", spacing_before=1)
    print(f"Analysis complete for {len(subregion_names)} subregions:")
    for region_name in subregion_names:
        print(f"  • {region_name}")

    print("\nAutomatically merging these subregions back together...")
    print("This will combine climbs that span across subregion boundaries.")

    # Find output files for each subregion
    output_dir = Path("output")
    output_files = []

    for region_name in subregion_names:
        # Construct expected filename pattern
        safe_name = "".join(c for c in region_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_name = safe_name.replace(" ", "_")

        # Search for files matching the pattern
        # Pattern: climbs_{RegionName}_{surface}_*.xlsx
        pattern = f"climbs_{safe_name}_{surface_filter}_*.xlsx"
        matching_files = list(output_dir.glob(pattern))

        # Also try CSV if no XLSX found
        if not matching_files:
            pattern = f"climbs_{safe_name}_{surface_filter}_*.csv"
            matching_files = list(output_dir.glob(pattern))

        if matching_files:
            # Use most recent file if multiple matches
            most_recent = max(matching_files, key=lambda p: p.stat().st_mtime)
            output_files.append((region_name, most_recent))
        else:
            print_error(f"Could not find output file for {region_name}")
            print(f"    Expected pattern: output/{pattern}")

    if len(output_files) < 2:
        print_error(f"Need at least 2 output files to merge (found {len(output_files)})")
        return False

    # Run merge script for adjacent pairs
    print(f"\nMerging {len(output_files)} subregions...\n")

    script_path = Path("scripts") / "merge_cross_region_climbs.py"
    if not script_path.exists():
        print_error(f"Merge script not found at {script_path}")
        return False

    merge_successful = True
    for i in range(len(output_files) - 1):
        region1_name, file1 = output_files[i]
        region2_name, file2 = output_files[i + 1]

        print(f"{'─' * 80}")
        print(f"Merging: {region1_name} + {region2_name}")
        print(f"{'─' * 80}\n")

        try:
            result = subprocess.run(
                ["python3", str(script_path), str(file1), str(file2)],
                capture_output=False,
                text=True,
            )

            if result.returncode == 0:
                print_success(f"Merged {region1_name} and {region2_name}")
            else:
                print_error(
                    f"Error merging {region1_name} and {region2_name} (exit code: {result.returncode})"
                )
                merge_successful = False

        except Exception as e:
            print_error(f"Error running merge: {e}")
            merge_successful = False
            continue

    if merge_successful:
        print()
        print_success("All subregions merged successfully!")
    else:
        print_error("Some merges failed. Check output above for details.")

    return merge_successful


def prompt_cross_region_analysis_simple(
    location: List[str],
    batch_tracker,
    surface_filter: str,
    score_type: str,
    radius_km: float,
    scope_type: str,
) -> None:
    """
    Prompt user to run cross-region climb analysis for states/countries.

    Simpler version for region and country batch processing where location is a list of names.

    Args:
        location: List of location names (region or country names)
        batch_tracker: BatchProgressTracker instance with completion info
        surface_filter: Surface filter used in analysis
        score_type: Score type used in analysis
        radius_km: Radius used in analysis
        scope_type: Type of scope ("region" or "country")
    """
    import subprocess
    from pathlib import Path

    # Get list of completed locations
    summary = batch_tracker.get_progress_summary()
    completed_locations = summary["locations"]["completed"]

    if len(completed_locations) < 2:
        # Need at least 2 locations to do cross-region analysis
        return

    # For states, all are in same country (USA), so default to yes
    # For countries, they span multiple countries, so default to no
    if scope_type == "region":
        default_response = "yes"
        default_msg = "all states within USA - defaulting to YES"
    elif scope_type == "country":
        default_response = "no"
        default_msg = "spanning multiple countries - defaulting to NO"
    else:
        default_response = "no"
        default_msg = ""

    from climb_analyzer.utils.formatting import print_banner, print_list_item

    print_banner("Cross-Region Climb Analysis", spacing_before=1)
    print(f"Analyzed {len(completed_locations)} {scope_type}s:")
    for loc in completed_locations:
        print_list_item(loc)

    print(f"\nSome climbs may extend across {scope_type} boundaries.")
    print("Would you like to run cross-region analysis to detect and merge")
    print(f"climbs that span multiple {scope_type}s?")
    if default_msg:
        print(f"\n({default_msg})")

    # Prompt user
    try:
        response = (
            input(f"\nRun cross-region analysis? (y/n) [{default_response}]: ").strip().lower()
        )

        if not response:
            response = default_response

        if response not in ["yes", "y", "no", "n"]:
            print(f"Invalid response '{response}', using default: {default_response}")
            response = default_response

    except KeyboardInterrupt:
        print(f"\nCancelled, using default: {default_response}")
        response = default_response

    if response in ["no", "n"]:
        print("\nSkipping cross-region analysis.")
        return

    # User wants to run cross-region analysis
    print("\nStarting cross-region climb analysis...")
    print(f"This will compare climbs between all {len(completed_locations)} {scope_type}s.\n")

    # Find output files for each location
    output_dir = Path("output")
    output_files = []

    for loc_name in completed_locations:
        # Construct expected filename pattern (without radius since it varies by state)
        safe_name = loc_name.replace(" ", "_")

        # Search for files matching the pattern (glob to handle different radii)
        # Pattern: climbs_{StateName}_{surface}_{scope}_*.xlsx
        pattern = f"climbs_{safe_name}_{surface_filter}_{scope_type}_*.xlsx"
        matching_files = list(output_dir.glob(pattern))

        # Also try CSV if no XLSX found
        if not matching_files:
            pattern = f"climbs_{safe_name}_{surface_filter}_{scope_type}_*.csv"
            matching_files = list(output_dir.glob(pattern))

        if matching_files:
            # Use most recent file if multiple matches
            most_recent = max(matching_files, key=lambda p: p.stat().st_mtime)
            output_files.append((loc_name, most_recent))
        else:
            print(f"⚠️  Could not find output file for {loc_name}")
            print(f"    Expected pattern: output/{pattern}")

    if len(output_files) < 2:
        print("\n❌ Error: Need at least 2 output files to run cross-region analysis")
        print(f"   Found only {len(output_files)} output file(s)")
        return

    # Run cross-region analysis for each pair of adjacent locations
    print(f"\nFound {len(output_files)} output files")
    print(f"Analyzing pairs of {scope_type}s for cross-region climbs...\n")

    script_path = Path("scripts") / "merge_cross_region_climbs.py"

    if not script_path.exists():
        print(f"❌ Error: Cross-region merge script not found at {script_path}")
        return

    # Build list of all unique pairs (not just sequential)
    region_pairs = []
    for i in range(len(output_files)):
        for j in range(i + 1, len(output_files)):
            region_pairs.append((i, j))

    from climb_analyzer.utils.formatting import print_banner, print_info

    print_banner("Cross-Region Analysis")
    print_info(f"Checking {len(region_pairs)} region pair(s) for cross-boundary climbs\n")

    # Compare each pair
    compared_count = 0
    skipped_count = 0

    for i, j in region_pairs:
        loc1_name, file1 = output_files[i]
        loc2_name, file2 = output_files[j]

        print_info(
            f"[{compared_count + skipped_count + 1}/{len(region_pairs)}] {loc1_name} ↔ {loc2_name}"
        )

        try:
            # Run the merge script (it will check adjacency internally)
            result = subprocess.run(
                ["python3", str(script_path), str(file1), str(file2)],
                capture_output=True,  # Capture to check for "not adjacent" message
                text=True,
            )

            # Print the output
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)

            if result.returncode == 0:
                # Check if regions were actually compared or skipped
                if "not adjacent" in result.stdout.lower() or "skipping" in result.stdout.lower():
                    skipped_count += 1
                else:
                    compared_count += 1
            else:
                print(f"⚠️  Error (exit code: {result.returncode})")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print()
    print_banner("Analysis Complete")
    if compared_count > 0:
        print_info(
            f"Compared {compared_count} adjacent pair(s), skipped {skipped_count} non-adjacent"
        )


def run_single_location_analysis(
    location_name: str,
    address: str,
    country: Optional[str],
    radius_km: float,
    center_lat: float,
    center_lon: float,
    formatted_address: str,
    surface_filter: str,
    unit_system: str,
    min_score: float,
    chunk_size_km: float,
    score_type: str,
    enable_geocoding: bool,
    save_to_xlsx: bool = True,
    scope_type: str = "address",
    batch_mode: bool = True,  # Default True since this function is used for batch processing
):
    """Run analysis for a single location with separate checkpoints and exports."""

    # Create location-specific analysis ID based on location name
    safe_location_name = "".join(
        c for c in location_name if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    safe_location_name = safe_location_name.replace(" ", "_")

    print(f"\\nStarting analysis for: {location_name}")
    print(f"Center coordinates: {center_lat:.4f}, {center_lon:.4f}")
    print(f"Analysis radius: {radius_km:.1f}km")
    print(f"Safe filename prefix: {safe_location_name}")

    # Check OpenTopoData server is ready (for local mode)
    # This happens AFTER region selection and data confirmation
    if not check_opentopodata_ready():
        print("\nContinuing with external elevation API...")

    try:
        climbs, df, persistence, should_upload_to_cache, error_logger = analyze_area(
            address=address,
            country=country,
            radius_km=radius_km,
            surface_filter=surface_filter,
            unit_system=unit_system,
            min_score=min_score,
            chunk_size_km=chunk_size_km,
            center_lat=center_lat,
            center_lon=center_lon,
            formatted_address=formatted_address,
            score_type=score_type,
            cycling_only=True,
            enable_geocoding=enable_geocoding,
            max_workers=OSM_MAX_THREADS,
            scope_type=scope_type,
            batch_mode=batch_mode,
        )

        if climbs:
            print(f"\\n✓ Analysis completed for {location_name}")
            print(f"Found {len(climbs)} climbs meeting criteria (min score: {min_score})")

            # Export results to output folder
            if save_to_xlsx:
                # Create output folder if it doesn't exist
                output_dir = Path("output")
                try:
                    output_dir.mkdir(exist_ok=True, mode=0o777)
                    # Ensure the directory is writable
                    import os

                    os.chmod(output_dir, 0o777)
                except Exception as e:
                    print(f"⚠️ Could not create/modify output folder: {e}")
                    # Try /tmp as fallback
                    import tempfile

                    output_dir = Path(tempfile.gettempdir())
                    print(f"   Using temporary directory instead: {output_dir}")

                # Generate location-specific export filename
                # Use consistent naming pattern for cross-region analysis compatibility
                # Pattern: climbs_{safe_name}_{surface}_{score_type}_{scope}_radius.{ext}

                # Get elevation error count from stats collector
                elevation_errors = 0
                try:
                    from utils.elevation_stats_collector import (
                        get_stats_collector,
                        has_elevation_stats,
                    )

                    if has_elevation_stats():
                        stats_collector = get_stats_collector()
                        stats = stats_collector.get_stats()
                        elevation_errors = stats.get("total_coords_failed", 0)
                except Exception:
                    # If stats collector not available, try to get from persistence
                    elevation_errors = 0

                # Save results (will split into multiple files if exceeds Excel limit)
                base_filename = (
                    output_dir
                    / f"climbs_{safe_location_name}_{surface_filter}_{score_type}_{scope_type}_{radius_km:.0f}km"
                )
                created_files = save_large_dataframe_as_split_excel(
                    df, base_filename, app_version=__version__, elevation_errors=elevation_errors
                )

                if created_files:
                    file_count = len(created_files)
                    if file_count == 1:
                        print(f" Results exported to: {created_files[0]}")
                    else:
                        print(f" Results exported to {file_count} files:")
                        for file in created_files:
                            print(f"   {file.name}")

                    # Now close the error logger with file count
                    if error_logger:
                        error_logger.stop_elevation_logging(output_file_count=file_count)
                        if elevation_errors > 0:
                            print(
                                f"📝 Elevation errors logged to: {error_logger.elevation_error_log} ({error_logger.error_count:,} failures)"
                            )
                else:
                    print("⚠️ Export failed")
                    # Still close error logger even if export failed
                    if error_logger:
                        error_logger.stop_elevation_logging(output_file_count=0)

        else:
            print(f"\\n❌ No climbs found for {location_name} with minimum score {min_score}")

    except Exception as e:
        print(f"\\n❌ Error analyzing {location_name}: {e}")
        import traceback

        traceback.print_exc()


# Import SpatialIndexManager from package (supports both pickle and JSONL metadata)
from climb_analyzer.data.spatial_index import SpatialIndexManager

if DEPLOYMENT_TYPE == "local":  # osmium only installed in local mode.

    class RoadWayHandler(osmium.SimpleHandler):
        def __init__(self, bbox, surface_filter, cycling_only):
            super().__init__()
            self.ways = []
            self.bbox = bbox  # (min_lat, min_lon, max_lat, max_lon)
            # Parse surface filter - can be "all" or comma-separated list like "paved,gravel"
            if surface_filter == "all":
                self.surface_filters = ["all"]
            else:
                self.surface_filters = [s.strip() for s in surface_filter.split(",")]
            self.cycling_only = cycling_only
            self.nodes = {}  # Store node locations

        def node(self, n):
            # Store node locations for later use
            self.nodes[n.id] = (n.location.lat, n.location.lon)

        def way(self, w):
            # Check if way is within bounding box
            if not self._is_in_bbox(w):
                return

            # Apply highway filter
            if "highway" not in w.tags:
                return

            highway_type = w.tags.get("highway", "")

            # Apply surface and cycling filters
            if not self._matches_filters(w.tags, highway_type):
                return

            # Convert to your SimpleWay format
            simple_way = self._convert_to_simple_way(w)
            if simple_way:
                self.ways.append(simple_way)


class ReverseGeocoder:
    """Reverse geocoder using offline reverse_geocoder module"""

    def __init__(self, max_concurrent: int = None):
        # max_concurrent is ignored since reverse_geocoder is synchronous and fast
        print(".")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass

    async def reverse_geocode_parallel(
        self,
        coordinates: List[Tuple[float, float]],
        persistence,
        progress_desc: str = "Geocoding",
    ) -> Dict:
        """Main parallel reverse geocoding function using offline reverse_geocoder"""

        if not coordinates:
            return {}

        # Aggressive coordinate deduplication with spatial clustering
        unique_coords, coord_mapping = self._deduplicate_with_clustering(coordinates)

        print(f"Geocoding {len(unique_coords)} unique locations using offline data")

        # Import reverse_geocoder locally to defer loading large spatial index
        import reverse_geocoder as rg

        # Process coordinates with reverse_geocoder (much faster than API calls)
        coord_to_location = {}

        with tqdm(
            total=len(unique_coords),
            desc=progress_desc,
            unit="coord",
            mininterval=0.1,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            # Process in batches for better progress indication
            batch_size = 1700  # Large batches since it's offline
            for i in range(0, len(unique_coords), batch_size):
                batch_coords = unique_coords[i : i + batch_size]

                # Signal handler check
                signal_handler.set_operation(
                    "geocoding",
                    {
                        "completed_count": i,
                        "total_count": len(unique_coords),
                        "coord_to_location": coord_to_location,
                        "coord_mapping": coord_mapping,
                        "timestamp": time.time(),
                    },
                )

                if signal_handler.kill_now:
                    self._save_geocoding_checkpoint(
                        persistence,
                        coord_to_location,
                        coord_mapping,
                        i,
                        len(unique_coords),
                    )
                    print("Geocoding progress saved. Analysis can be resumed.")
                    sys.exit(0)

                try:
                    # Use reverse_geocoder for batch lookup
                    results = rg.search(batch_coords)

                    # Process results
                    for j, (coord, result) in enumerate(zip(batch_coords, results)):
                        if result:
                            # Extract city and state from reverse_geocoder result
                            city = result.get("name", "Unknown")

                            # reverse_geocoder uses 'admin1' for state/province
                            state = result.get("admin1", "Unknown")

                            # Build full address
                            country = result.get("cc", "")  # Country code
                            full_address_parts = [city]
                            if state and state != "Unknown":
                                full_address_parts.append(state)
                            if country:
                                full_address_parts.append(country)
                            full_address = ", ".join(full_address_parts)

                            coord_to_location[coord] = {
                                "city": city,
                                "state": state,
                                "full_address": full_address,
                            }
                        else:
                            coord_to_location[coord] = {
                                "city": "Unknown",
                                "state": "Unknown",
                                "full_address": "Not found",
                            }

                except Exception as e:
                    print(f"Error in reverse geocoding batch: {e}")
                    # Fill with fallback data
                    for coord in batch_coords:
                        coord_to_location[coord] = {
                            "city": "Lookup Failed",
                            "state": "Lookup Failed",
                            "full_address": f"Error: {str(e)[:50]}",
                        }

                # Update progress
                completed = min(i + batch_size, len(unique_coords))
                pbar.n = completed
                pbar.set_postfix(
                    {
                        "success_rate": f"{len([v for v in coord_to_location.values() if 'Error' not in v.get('city', '')])}/{completed}"
                    }
                )
                pbar.refresh()

        # Map results back to all original coordinates
        full_results = {}
        for i, original_coord in enumerate(coordinates):
            clustered_coord = coord_mapping.get(original_coord, original_coord)
            if clustered_coord in coord_to_location:
                full_results[i] = coord_to_location[clustered_coord]
            else:
                full_results[i] = {
                    "city": "Unknown",
                    "state": "Unknown",
                    "full_address": "Not found",
                }

        successful_lookups = len(
            [v for v in coord_to_location.values() if "Error" not in v.get("city", "")]
        )
        print(
            f"Successfully geocoded {successful_lookups}/{len(unique_coords)} unique locations using offline data"
        )

        return full_results

    def _deduplicate_with_clustering(
        self, coordinates: List[Tuple[float, float]], cluster_radius_deg: float = 0.01
    ) -> Tuple[List[Tuple[float, float]], Dict]:
        """Advanced deduplication using spatial clustering to reduce lookups"""

        if not coordinates:
            return [], {}

        # Simple grid-based clustering - coordinates within same grid cell share lookup
        cluster_map = defaultdict(list)
        coord_mapping = {}

        for coord in coordinates:
            # Round to grid cell (approximately 1km at equator)
            cluster_key = (
                round(coord[0] / cluster_radius_deg) * cluster_radius_deg,
                round(coord[1] / cluster_radius_deg) * cluster_radius_deg,
            )
            cluster_map[cluster_key].append(coord)

        # Use cluster center as representative coordinate
        unique_coords = []
        for cluster_key, coords_in_cluster in cluster_map.items():
            if coords_in_cluster:
                # Use the first coordinate in cluster as representative
                representative = coords_in_cluster[0]
                unique_coords.append(representative)

                # Map all coordinates in cluster to the representative
                for coord in coords_in_cluster:
                    coord_mapping[coord] = representative

        return unique_coords, coord_mapping

    def _save_geocoding_checkpoint(
        self,
        persistence,
        coord_to_location: Dict,
        coord_mapping: Dict,
        completed: int,
        total: int,
    ):
        """Save geocoding checkpoint"""
        checkpoint_data = {
            "coord_to_location": coord_to_location,
            "coord_mapping": coord_mapping,
            "completed_count": completed,
            "total_count": total,
            "timestamp": time.time(),
        }

        try:
            geocoding_checkpoint_file = persistence.analysis_dir / "geocoding_parallel_progress.pkl"
            temp_file = persistence.analysis_dir / "geocoding_parallel_progress.tmp"

            with open(temp_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            temp_file.rename(geocoding_checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not save geocoding checkpoint: {e}")


def build_elevation_url(dataset_name: str) -> str:
    """Build full elevation API URL from base URL and dataset name

    Args:
        dataset_name: Dataset name like 'srtm30m', 'ned10m', 'aster30m'

    Returns:
        Full URL like 'http://localhost:5000/v1/srtm30m'
    """
    if not TOPO_API_BASE_URL:
        return None

    # Remove trailing slash if present
    base = TOPO_API_BASE_URL.rstrip("/")

    # Add dataset name
    return f"{base}/{dataset_name}"


def get_available_datasets_from_server() -> List[str]:
    """Query OpenTopoData server for available datasets.

    Returns:
        List of dataset names available on the server, or empty list if query fails
    """
    if not TOPO_API_BASE_URL:
        return []

    # Remove /v1 suffix from base URL since /datasets is at root level
    base = TOPO_API_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    # Try multiple URLs to handle both Docker container names and localhost
    urls_to_try = [
        f"{base}/datasets",  # Original URL (works inside Docker network)
    ]

    # If URL contains Docker container name, also try localhost
    if "opentopodata-server" in base:
        localhost_base = base.replace("opentopodata-server", "localhost")
        urls_to_try.append(f"{localhost_base}/datasets")

    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                # Response format: {"results": [{"name": "ned10m", ...}, ...]}
                if "results" in data and isinstance(data["results"], list):
                    return [ds["name"] for ds in data["results"] if "name" in ds]
        except Exception:
            continue  # Try next URL

    # If all attempts failed, return empty list (will skip filtering)
    return []


def filter_available_datasets(
    dataset_priority: List[str], available_datasets: List[str]
) -> List[str]:
    """Filter dataset priority list to only include available datasets.

    Args:
        dataset_priority: Desired dataset priority order
        available_datasets: Datasets available on the OpenTopoData server

    Returns:
        Filtered dataset list containing only available datasets, in priority order
    """
    if not available_datasets:
        # If we couldn't query server, return full list (will fail at query time)
        return dataset_priority

    return [ds for ds in dataset_priority if ds in available_datasets]


# Region-based dataset priority configuration
# Defines the preferred dataset order for elevation fetching based on geographic region
# Cloud mode dataset priority (limited datasets available in cloud API)
DATASET_PRIORITY_BY_REGION_CLOUD = {
    # United States (excluding Alaska)
    "United States of America": ["ned10m", "srtm30m"],
    # US States (excluding Alaska) - all use NED + SRTM
    "Alabama": ["ned10m", "srtm30m"],
    "Arizona": ["ned10m", "srtm30m"],
    "Arkansas": ["ned10m", "srtm30m"],
    "California": ["ned10m", "srtm30m"],
    "Colorado": ["ned10m", "srtm30m"],
    "Connecticut": ["ned10m", "srtm30m"],
    "Delaware": ["ned10m", "srtm30m"],
    "District Of Columbia": ["ned10m", "srtm30m"],
    "Florida": ["ned10m", "srtm30m"],
    "Georgia": ["ned10m", "srtm30m"],
    "Hawaii": ["ned10m", "srtm30m"],
    "Idaho": ["ned10m", "srtm30m"],
    "Illinois": ["ned10m", "srtm30m"],
    "Indiana": ["ned10m", "srtm30m"],
    "Iowa": ["ned10m", "srtm30m"],
    "Kansas": ["ned10m", "srtm30m"],
    "Kentucky": ["ned10m", "srtm30m"],
    "Louisiana": ["ned10m", "srtm30m"],
    "Maine": ["ned10m", "srtm30m"],
    "Maryland": ["ned10m", "srtm30m"],
    "Massachusetts": ["ned10m", "srtm30m"],
    "Michigan": ["ned10m", "srtm30m"],
    "Minnesota": ["ned10m", "srtm30m"],
    "Mississippi": ["ned10m", "srtm30m"],
    "Missouri": ["ned10m", "srtm30m"],
    "Montana": ["ned10m", "srtm30m"],
    "Nebraska": ["ned10m", "srtm30m"],
    "Nevada": ["ned10m", "srtm30m"],
    "New Hampshire": ["ned10m", "srtm30m"],
    "New Jersey": ["ned10m", "srtm30m"],
    "New Mexico": ["ned10m", "srtm30m"],
    "New York": ["ned10m", "srtm30m"],
    "North Carolina": ["ned10m", "srtm30m"],
    "North Dakota": ["ned10m", "srtm30m"],
    "Ohio": ["ned10m", "srtm30m"],
    "Oklahoma": ["ned10m", "srtm30m"],
    "Oregon": ["ned10m", "srtm30m"],
    "Pennsylvania": ["ned10m", "srtm30m"],
    "Puerto Rico": ["ned10m", "srtm30m"],
    "Rhode Island": ["ned10m", "srtm30m"],
    "South Carolina": ["ned10m", "srtm30m"],
    "South Dakota": ["ned10m", "srtm30m"],
    "Tennessee": ["ned10m", "srtm30m"],
    "Texas": ["ned10m", "srtm30m"],
    "Us Virgin Islands": ["ned10m", "srtm30m"],
    "Utah": ["ned10m", "srtm30m"],
    "Vermont": ["ned10m", "srtm30m"],
    "Virginia": ["ned10m", "srtm30m"],
    "Washington": ["ned10m", "srtm30m"],
    "West Virginia": ["ned10m", "srtm30m"],
    "Wisconsin": ["ned10m", "srtm30m"],
    "Wyoming": ["ned10m", "srtm30m"],
    # Alaska - Cloud mode uses SRTM (limited coverage at high latitudes)
    "Alaska": ["srtm30m"],
    # Canada - split by latitude
    "Canada": ["srtm30m"],  # Southern Canada
    "Canada (>60°N)": ["srtm30m"],  # Northern Canada - cloud mode (limited coverage)
    # Greenland - Not supported in cloud mode
    "Greenland": [],
    # Nordic countries - SRTM for <60N only
    "Iceland": ["srtm30m"],  # Limited coverage - mostly >60N
    "Norway": ["srtm30m"],  # Southern Norway (<60N)
    "Norway (>60°N)": ["srtm30m"],  # Northern Norway - cloud mode (limited coverage)
    "Sweden": ["srtm30m"],  # Southern Sweden (<60N)
    "Sweden (>60°N)": ["srtm30m"],  # Northern Sweden - cloud mode (limited coverage)
    "Finland": ["srtm30m"],  # Southern Finland (<60N)
    "Finland (>60°N)": ["srtm30m"],  # Northern Finland - cloud mode (limited coverage)
    # Russia - split by latitude
    "Russia": ["srtm30m"],  # Southern Russia
    "Russia (>60°N)": ["srtm30m"],  # Northern Russia - cloud mode (limited coverage)
    # Antarctica - Not supported in cloud mode
    "Antarctica": [],
    # Default for most non-US countries
    "_default": ["srtm30m"],
}

# Local mode dataset priority (full dataset availability)
DATASET_PRIORITY_BY_REGION = {
    # United States (excluding Alaska)
    "United States of America": ["ned10m", "srtm30m"],
    # US States (excluding Alaska) - all use NED + SRTM
    "Alabama": ["ned10m", "srtm30m"],
    "Arizona": ["ned10m", "srtm30m"],
    "Arkansas": ["ned10m", "srtm30m"],
    "California": ["ned10m", "srtm30m"],
    "Colorado": ["ned10m", "srtm30m"],
    "Connecticut": ["ned10m", "srtm30m"],
    "Delaware": ["ned10m", "srtm30m"],
    "District Of Columbia": ["ned10m", "srtm30m"],
    "Florida": ["ned10m", "srtm30m"],
    "Georgia": ["ned10m", "srtm30m"],
    "Hawaii": ["ned10m", "srtm30m"],
    "Idaho": ["ned10m", "srtm30m"],
    "Illinois": ["ned10m", "srtm30m"],
    "Indiana": ["ned10m", "srtm30m"],
    "Iowa": ["ned10m", "srtm30m"],
    "Kansas": ["ned10m", "srtm30m"],
    "Kentucky": ["ned10m", "srtm30m"],
    "Louisiana": ["ned10m", "srtm30m"],
    "Maine": ["ned10m", "srtm30m"],
    "Maryland": ["ned10m", "srtm30m"],
    "Massachusetts": ["ned10m", "srtm30m"],
    "Michigan": ["ned10m", "srtm30m"],
    "Minnesota": ["ned10m", "srtm30m"],
    "Mississippi": ["ned10m", "srtm30m"],
    "Missouri": ["ned10m", "srtm30m"],
    "Montana": ["ned10m", "srtm30m"],
    "Nebraska": ["ned10m", "srtm30m"],
    "Nevada": ["ned10m", "srtm30m"],
    "New Hampshire": ["ned10m", "srtm30m"],
    "New Jersey": ["ned10m", "srtm30m"],
    "New Mexico": ["ned10m", "srtm30m"],
    "New York": ["ned10m", "srtm30m"],
    "North Carolina": ["ned10m", "srtm30m"],
    "North Dakota": ["ned10m", "srtm30m"],
    "Ohio": ["ned10m", "srtm30m"],
    "Oklahoma": ["ned10m", "srtm30m"],
    "Oregon": ["ned10m", "srtm30m"],
    "Pennsylvania": ["ned10m", "srtm30m"],
    "Puerto Rico": ["ned10m", "srtm30m"],
    "Rhode Island": ["ned10m", "srtm30m"],
    "South Carolina": ["ned10m", "srtm30m"],
    "South Dakota": ["ned10m", "srtm30m"],
    "Tennessee": ["ned10m", "srtm30m"],
    "Texas": ["ned10m", "srtm30m"],
    "Us Virgin Islands": ["ned10m", "srtm30m"],
    "Utah": ["ned10m", "srtm30m"],
    "Vermont": ["ned10m", "srtm30m"],
    "Virginia": ["ned10m", "srtm30m"],
    "Washington": ["ned10m", "srtm30m"],
    "West Virginia": ["ned10m", "srtm30m"],
    "Wisconsin": ["ned10m", "srtm30m"],
    "Wyoming": ["ned10m", "srtm30m"],
    # Alaska - ArcticDEM primary (best for >60°N), AW3D30 fallback
    "Alaska": ["arctic32m", "aw3d30"],
    # Canada country-level - split by latitude
    "Canada": ["srtm30m", "aw3d30"],  # Southern Canada
    "Canada (>60°N)": ["arctic32m", "aw3d30"],  # Northern Canada
    # Canadian provinces - SRTM primary, AW3D30 fallback. Provinces that span
    # or touch 60°N get arctic32m as tertiary — opentopodata multi-dataset mode
    # serves SRTM where it has coverage (<=60°N) and falls through to arctic32m
    # for coordinates above 60°N where SRTM returns null.
    "British Columbia": ["srtm30m", "arctic32m", "aw3d30"],     # touches 60.0°N
    "Alberta": ["srtm30m", "arctic32m", "aw3d30"],              # touches 60.0°N
    "Saskatchewan": ["srtm30m", "arctic32m", "aw3d30"],         # touches 60.0°N
    "Manitoba": ["srtm30m", "arctic32m", "aw3d30"],             # touches 60.0°N
    "Quebec": ["srtm30m", "arctic32m", "aw3d30"],               # to 62.6°N
    "Newfoundland And Labrador": ["srtm30m", "arctic32m", "aw3d30"],  # to 60.5°N
    # Provinces entirely south of 60°N — no arctic coverage needed
    "Ontario": ["srtm30m", "aw3d30"],                           # max 57.5°N
    "New Brunswick": ["srtm30m", "aw3d30"],                     # max 48.4°N
    "Nova Scotia": ["srtm30m", "aw3d30"],                       # max 47.9°N
    "Prince Edward Island": ["srtm30m", "aw3d30"],              # max 47.7°N
    # Canadian territories - ArcticDEM primary (all or mostly above 60°N)
    "Yukon": ["arctic32m", "aw3d30"],
    "Northwest Territories": ["arctic32m", "aw3d30"],
    "Nunavut": ["arctic32m", "aw3d30"],
    # Greenland - ArcticDEM only (no other datasets have coverage)
    "Greenland": ["arctic32m"],
    # Nordic countries - Arctic with latitude-based switching
    "Iceland": ["arctic32m", "aw3d30"],  # All Iceland is >60N
    "Norway": ["srtm30m", "aw3d30"],  # Southern Norway (<60N)
    "Norway (>60°N)": ["arctic32m", "aw3d30"],  # Northern Norway
    "Sweden": ["srtm30m", "aw3d30"],  # Southern Sweden (<60N)
    "Sweden (>60°N)": ["arctic32m", "aw3d30"],  # Northern Sweden
    "Finland": ["srtm30m", "aw3d30"],  # Southern Finland (<60N)
    "Finland (>60°N)": ["arctic32m", "aw3d30"],  # Northern Finland
    # Russia - split by latitude
    "Russia": ["srtm30m", "aw3d30"],  # Southern Russia
    "Russia (>60°N)": ["arctic32m", "aw3d30"],  # Northern Russia
    # Antarctica - REMA only (no other datasets available)
    "Antarctica": ["rema32m"],
    # Default for most non-US countries
    "_default": ["srtm30m", "aw3d30"],
}


def get_dataset_priority_for_region(
    region_name: str, lat: float = None, cloud_mode: bool = False
) -> List[str]:
    """
    Get dataset priority list for a specific region and latitude.

    Args:
        region_name: Name of the region/country (case-insensitive)
        lat: Latitude (optional, used for latitude-based rules)
        cloud_mode: If True, use cloud API dataset priorities (limited availability)

    Returns:
        List of dataset names in priority order
    """
    # Select appropriate priority dict
    priority_dict = DATASET_PRIORITY_BY_REGION_CLOUD if cloud_mode else DATASET_PRIORITY_BY_REGION

    # Strip continent/country prefix from canonical paths like "us/colorado",
    # "canada/british-columbia", "north-america/greenland". Without this, the
    # title-cased lookup becomes "Us/Colorado" and silently falls to _default,
    # causing arctic32m (and other region-specific priorities) to be skipped.
    lookup_name = region_name.split("/")[-1] if region_name and "/" in region_name else region_name

    # Normalize region name to title case for lookup
    # This ensures "antarctica" matches "Antarctica" in the dict
    normalized_region = lookup_name.replace("-", " ").title() if lookup_name else None

    # Special handling for ambiguous region names that exist in multiple places
    if lat is not None and normalized_region == "Georgia":
        # Georgia (US State): ~30-35°N
        # Georgia (Country): ~41-43°N
        if lat < 38:
            # US State - use NED priority (already in dict as "Georgia")
            pass
        else:
            # Country in Caucasus - use global default
            return priority_dict["_default"]

    # Check for latitude-based rules for Arctic regions
    if lat is not None and normalized_region:
        # Canada, Russia, Norway, Sweden, Finland - use Arctic datasets above 60°N
        arctic_regions = ["Canada", "Russia", "Norway", "Sweden", "Finland"]
        if normalized_region in arctic_regions and lat > 60:
            high_lat_key = f"{normalized_region} (>60°N)"
            return priority_dict.get(
                high_lat_key, priority_dict.get(normalized_region, priority_dict["_default"])
            )

    # Check for region-specific priority (case-insensitive)
    if normalized_region and normalized_region in priority_dict:
        return priority_dict[normalized_region]

    # Fall back to default
    return priority_dict["_default"]


def get_region_name_from_config() -> Optional[str]:
    """
    Get region name from config OSM_COVERAGE.

    Returns:
        Region name if available, None otherwise
    """
    try:
        config = get_config()
        osm_coverage = config.get("OSM_COVERAGE", [])
        if isinstance(osm_coverage, list) and osm_coverage:
            # Return first region from coverage list
            return osm_coverage[0]
    except Exception:
        pass
    return None


class FastElevationFetcher:
    """Simplified parallel elevation fetcher with local GeoTIFF support"""

    def __init__(self, primary_dataset="srtm30m", region_name=None, lat=None):
        """Initialize elevation fetcher

        Args:
            primary_dataset: Primary dataset to use (overridden if region_name is provided)
            region_name: Region/country name for automatic dataset priority selection
            lat: Latitude for latitude-based priority rules (optional)
        """
        # Store region info for dataset priority
        self.region_name = region_name
        self.lat = lat

        # Determine dataset priority order and auto-detect cloud vs local mode
        if region_name:
            # Auto-detect cloud mode from base URL
            cloud_mode = "api.opentopodata.org" in TOPO_API_BASE_URL if TOPO_API_BASE_URL else False

            # Use region-based priority with cloud mode flag
            self.dataset_priority = get_dataset_priority_for_region(region_name, lat, cloud_mode)

            # Preserve the configured cascade before server-availability filtering
            # so downstream consumers (checkpoint, PR body) can document intent
            # even when the local server is missing a dataset this run.
            self.configured_priority = list(self.dataset_priority)

            # Query server for available datasets and filter priority list
            available_datasets = get_available_datasets_from_server()
            original_priority = self.dataset_priority.copy()

            if available_datasets:
                # Filter out datasets that aren't available on the server
                self.dataset_priority = filter_available_datasets(
                    self.dataset_priority, available_datasets
                )

            primary_dataset = self.dataset_priority[0] if self.dataset_priority else primary_dataset
            mode_str = " (cloud mode)" if cloud_mode else " (local mode)"

            if self.dataset_priority:
                # Show verbose dataset info only in verbose mode
                verbose_print(
                    f"\n  [verbose] Dataset priority for {region_name}{mode_str}: {' → '.join(self.dataset_priority)}"
                )

                # Show filtered datasets if any were removed
                if available_datasets and len(self.dataset_priority) < len(original_priority):
                    removed = [ds for ds in original_priority if ds not in self.dataset_priority]
                    verbose_print(f"  [verbose] Filtered out unavailable datasets: {', '.join(removed)}")
        else:
            # Use provided primary_dataset or default
            self.dataset_priority = None

        self.primary_dataset = primary_dataset  # Store for error logging
        self.primary_url = build_elevation_url(primary_dataset)
        self.optimal_batch_size = ELEVATION_BATCH_SIZE
        self.batch_size_adapted = False  # Track if we've found optimal size
        self.min_batch_size = 99  # Minimum acceptable batch size before erroring out

        # Configure dataset cascade for multi-dataset fallback queries
        self.dataset_cascade = self._configure_dataset_cascade()
        self.multi_dataset_url = self._build_multi_dataset_url()

        # Silently remove duplicate datasets if any
        if self.multi_dataset_url and self.dataset_cascade:
            if len(self.dataset_cascade) != len(set(self.dataset_cascade)):
                seen = set()
                self.dataset_cascade = [
                    ds for ds in self.dataset_cascade if not (ds in seen or seen.add(ds))
                ]
                self.multi_dataset_url = self._build_multi_dataset_url()

        # Keep legacy fallback URLs for emergency backup
        self.fallback_urls = self._configure_fallback_endpoints_legacy()

        # Track datasets that returned 404 (not available) - skip these for entire session
        self.unavailable_datasets = set()

        # Track datasets that actually returned valid elevation data
        self.datasets_used = set()

        # Track if we've attempted config sync after "not in config" error (only try once)
        self._config_sync_attempted = False

        # Track elevation fetch statistics for error reporting
        self.total_coords_requested = 0
        self.total_coords_failed = 0
        self.total_unique_coords = 0

        # Track 429 rate limit errors
        self.rate_limit_429_count = 0
        self.rate_limit_429_shown_full_msg = False  # Track if we've shown the long message

        # Track 400 errors (bad request, usually dataset not available)
        self.http_400_shown_datasets = set()  # Datasets we've already warned about

        # Track multi-dataset fallback message
        self.multi_dataset_fallback_shown = False

        # Enable optimized multi-dataset mode (can be disabled for testing)
        self.use_multi_dataset_mode = True

        # Thread lock for printing error messages (prevent concurrent printing)
        import threading

        self._print_lock = threading.Lock()

    def _configure_dataset_cascade(self) -> List[str]:
        """Configure ordered list of datasets for multi-dataset queries"""
        # Use region-based priority if available
        if self.dataset_priority:
            return self.dataset_priority
        else:
            # Standard fallback sequence based on availability
            # Include primary dataset as first in cascade
            return [self.primary_dataset, "srtm30m", "aw3d30"]

    def _build_multi_dataset_url(self) -> str:
        """Build URL with comma-separated datasets for optimized fallback"""
        if not TOPO_API_BASE_URL:
            return None

        # Join datasets with comma (OpenTopoData multi-dataset syntax)
        dataset_list = ",".join(self.dataset_cascade)
        base_url = TOPO_API_BASE_URL.rstrip("/")
        return f"{base_url}/{dataset_list}"

    def _configure_fallback_endpoints_legacy(self) -> List[str]:
        """Legacy fallback endpoints (kept as emergency backup)"""
        fallback_urls = []

        # Use datasets from cascade (excluding primary which is already set)
        fallback_datasets = self.dataset_cascade[1:] if self.dataset_cascade else []

        for dataset in fallback_datasets:
            endpoint = build_elevation_url(dataset)
            if endpoint and endpoint != self.primary_url and endpoint not in fallback_urls:
                fallback_urls.append(endpoint)

        return fallback_urls

    def fetch_elevations_for_coordinates(
        self,
        coordinates: List[Tuple[float, float]],
        persistence,
        progress_desc: str = "Fetching elevation",
        coord_metadata: Optional[Dict[Tuple[float, float], Dict[str, str]]] = None,
        fetch_log: Optional["ElevationFetchLog"] = None,
    ) -> List[Optional[float]]:
        """Main entry point - uses parallel processing for optimal performance"""
        return self._fetch_elevations_parallel(
            coordinates, persistence, progress_desc, coord_metadata, fetch_log
        )

    def _save_optimal_batch_size_to_config(self):
        """Save the optimal batch size back to config.yaml"""
        try:
            import yaml

            config_path = Path("config.yaml")

            config = {}
            if config_path.exists():
                # Load existing config
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}

            # Update the batch size
            old_batch_size = config.get("ELEVATION_BATCH_SIZE", ELEVATION_BATCH_SIZE)
            config["ELEVATION_BATCH_SIZE"] = self.optimal_batch_size

            # Save back to file
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=True)

            print(
                f"Saved optimal batch size to config.yaml: {old_batch_size} → {self.optimal_batch_size}"
            )

        except Exception as e:
            print(f"Warning: Could not save optimal batch size to config.yaml: {e}")

    def _wait_with_backoff(self, wait_seconds, reason=""):
        """Wait for server recovery with progress indication"""
        if wait_seconds > 0:
            print(
                f"Waiting {wait_seconds}s for server recovery{' (' + reason + ')' if reason else ''}..."
            )
            time.sleep(wait_seconds)

    def _reduce_batch_size(self, reason="HTTP error", with_backoff=False):
        """Conservative batch size reduction"""
        # No additional wait - retries already handled waiting

        # Reduce by 10% each time to find optimal batch size gradually
        new_size = max(99, int(self.optimal_batch_size * 0.9))  # Reduce by 10%

        if new_size != self.optimal_batch_size:
            print(
                f"{reason} - reducing batch size conservatively: {self.optimal_batch_size} → {new_size}"
            )
            self.optimal_batch_size = new_size
            self.batch_size_adapted = True
            return True
        return False

    def _test_multi_dataset_health(self) -> bool:
        """Test multi-dataset endpoint with a small query before starting analysis."""
        if not self.use_multi_dataset_mode or not self.multi_dataset_url:
            return True  # Skip health check if multi-dataset mode not enabled

        import logging

        logger = logging.getLogger(__name__)


        # Only show health check message in verbose mode
        verbose_print("\n  [verbose] Testing multi-dataset endpoint health...")

        # Test with a simple coordinate (Paris, France)
        test_coord = (48.8566, 2.3522)

        try:
            import time as time_module

            import requests

            start = time_module.time()
            response = requests.get(
                self.multi_dataset_url,
                params={"locations": f"{test_coord[0]},{test_coord[1]}"},
                timeout=ELEVATION_REQUEST_TIMEOUT_SEC,
                headers={"Connection": "close"},
            )
            elapsed = time_module.time() - start

            if response.ok and response.json().get("status") == "OK":
                logger.info("[MULTI-DS] Health check passed")
                return True
            else:
                logger.warning("[MULTI-DS] Health check failed, disabling multi-dataset mode")
                self.use_multi_dataset_mode = False
                return False

        except Exception as e:
            logger.error(f"[MULTI-DS] Health check exception: {type(e).__name__}: {str(e)}")
            self.use_multi_dataset_mode = False
            return False

    def _fetch_elevations_parallel(
        self,
        coordinates: List[Tuple[float, float]],
        persistence,
        progress_desc: str,
        coord_metadata: Optional[Dict[Tuple[float, float], Dict[str, str]]] = None,
        fetch_log: Optional["ElevationFetchLog"] = None,
        silent_mode: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> List[Optional[float]]:
        """Parallel elevation fetching with smart checkpointing and adaptive batch sizing."""

        # Health check for multi-dataset endpoint (only on first call)
        if not hasattr(self, "_health_check_done"):
            self._test_multi_dataset_health()
            self._health_check_done = True

        # Existing coordinate deduplication logic (unchanged)
        unique_coords = []
        coord_to_index = {}
        original_to_unique = []

        for i, coord in enumerate(coordinates):
            rounded_coord = (round(coord[0], 6), round(coord[1], 6))

            if rounded_coord not in coord_to_index:
                coord_to_index[rounded_coord] = len(unique_coords)
                unique_coords.append(rounded_coord)

            original_to_unique.append(coord_to_index[rounded_coord])

        total_unique = len(unique_coords)
        total_original = len(coordinates)

        # SILENT MODE - no printing, no progress bars
        # All output handled by caller's single progress bar

        # Process unique coordinates in batches with parallel execution
        all_elevations = [None] * total_unique
        coordinate_mapping = {}

        # Track elevation fetch statistics
        total_coords_requested = 0
        total_coords_failed = 0

        # Initialize smart checkpointer with dynamic total calculation
        def calculate_total_batches():
            return (total_unique + self.optimal_batch_size - 1) // self.optimal_batch_size

        total_batches = calculate_total_batches()
        checkpointer = SmartCheckpointer(total_batches, "Elevation Fetching")

        # Prepare all batches upfront
        all_batches = []
        coord_idx = 0
        while coord_idx < total_unique:
            current_batch_size = min(self.optimal_batch_size, total_unique - coord_idx)
            batch_coords = unique_coords[coord_idx : coord_idx + current_batch_size]
            all_batches.append((len(all_batches), coord_idx, batch_coords))
            coord_idx += current_batch_size

        # Dummy progress bar that does nothing
        class DummyProgressBar:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, n=1):
                pass

            def set_postfix(self, *args, **kwargs):
                pass

        pbar = DummyProgressBar()

        with pbar:

            # Direct parallel processing - submit all batch requests immediately
            def make_direct_request(batch_info):
                """Make direct API request without wrapper overhead"""
                batch_num, start_idx, batch_coords = batch_info
                try:
                    # Direct API call with minimal overhead
                    elevations = self._fetch_single_batch(
                        batch_coords, max_retries=5, base_delay=1.0
                    )
                    return (start_idx, batch_coords, elevations, batch_num)
                except Exception:
                    # Silent - errors logged to error_logger.py if configured
                    return (
                        start_idx,
                        batch_coords,
                        [None] * len(batch_coords),
                        batch_num,
                    )

            # Submit ALL batches to ThreadPoolExecutor at once for true parallelism
            with ThreadPoolExecutor(max_workers=ELEVATION_MAX_CONCURRENT) as executor:
                completed_batches = 0

                # Submit ALL requests immediately
                future_to_batch = {
                    executor.submit(make_direct_request, batch_info): batch_info
                    for batch_info in all_batches
                }

                # Process results as they complete with periodic interrupt checking
                remaining_futures = set(future_to_batch.keys())
                while remaining_futures:
                    # Check for shutdown signal
                    if signal_handler.kill_now:
                        print("\nCanceling remaining elevation requests...")
                        for f in remaining_futures:
                            f.cancel()
                        # Checkpoint handled by parent function (process_region_without_chunking)
                        # which uses streaming disk-based format
                        os._exit(0)  # Force immediate exit without waiting for threads

                    # Wait for next future with timeout to allow periodic signal checking
                    try:
                        done, remaining_futures = concurrent.futures.wait(
                            remaining_futures,
                            timeout=0.5,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                    except Exception:
                        continue

                    # Process completed futures
                    for future in done:
                        batch_info = future_to_batch[future]
                        batch_num = batch_info[0]

                        try:
                            (
                                start_idx,
                                batch_coords,
                                batch_elevations,
                                returned_batch_num,
                            ) = future.result()

                            # Store results in the main arrays and track failures
                            batch_failed_count = 0
                            for j, elevation in enumerate(batch_elevations):
                                actual_coord_idx = start_idx + j
                                if actual_coord_idx < total_unique:
                                    all_elevations[actual_coord_idx] = elevation
                                    total_coords_requested += 1
                                    if elevation is not None:
                                        coordinate_mapping[unique_coords[actual_coord_idx]] = (
                                            elevation
                                        )
                                    else:
                                        batch_failed_count += 1
                                        total_coords_failed += 1

                            # Track if entire batch failed (don't print - will show in summary)
                            if (
                                batch_failed_count == len(batch_elevations)
                                and len(batch_elevations) > 0
                            ):
                                # Silently track failed batches
                                pass

                            completed_batches += 1

                            # Update progress bar by number of coordinates processed in this batch
                            pbar.update(len(batch_coords))
                            pbar.set_postfix(
                                {
                                    "completed": completed_batches,
                                    "total": len(all_batches),
                                    "batch_size": self.optimal_batch_size,
                                    "failed": total_coords_failed,
                                }
                            )

                            # Call external progress callback if provided (for parent progress bar)
                            if progress_callback:
                                progress_callback(len(batch_coords))

                            # Checkpointing removed - handled by parent function (process_region_without_chunking)
                            # which uses streaming disk-based format with proper batch tracking

                        except Exception:
                            # Silent - errors logged to error_logger.py if configured
                            completed_batches += 1
                            # Update by batch size estimate
                            pbar.update(self.optimal_batch_size)

                            # Call external progress callback if provided
                            if progress_callback:
                                progress_callback(self.optimal_batch_size)

        # Save optimal batch size to config if it was adapted
        if self.batch_size_adapted and self.optimal_batch_size != ELEVATION_BATCH_SIZE:
            self._save_optimal_batch_size_to_config()

        # Map back to original coordinate order and cleanup (unchanged)
        result_elevations = [all_elevations[unique_idx] for unique_idx in original_to_unique]

        successful_total = sum(1 for e in result_elevations if e is not None)

        # Store statistics for error reporting
        self.total_coords_requested = total_original
        self.total_unique_coords = total_unique
        self.total_coords_failed = total_coords_failed

        # Report to global stats collector
        # SILENT MODE - no summary printing
        # Errors will be logged to error_logger.py if provided
        # Summary will be printed once at the end by caller

        persistence.clear_elevation_progress()

        # MEMORY FIX: Explicitly delete large data structures before return
        # For large regions (France: 1M coords per batch), these can be 100-200MB each
        del all_elevations, coordinate_mapping, unique_coords, original_to_unique
        import gc

        gc.collect()

        check_and_cleanup_memory(force_cleanup=True)

        return result_elevations

    def get_error_statistics(self):
        """Get elevation fetch error statistics for reporting.

        Returns:
            Tuple of (total_coords_requested, failed_coords, total_unique_coords)
        """
        return (self.total_coords_requested, self.total_coords_failed, self.total_unique_coords)

    def get_successful_datasets(self) -> List[str]:
        """Get list of datasets that were successfully used (not unavailable).

        Returns:
            List of dataset names that were not removed due to unavailability
        """
        return [ds for ds in self.dataset_cascade if ds not in self.unavailable_datasets]

    def get_datasets_used(self) -> List[str]:
        """Get list of datasets that actually returned elevation data, in priority order.

        Returns:
            List of dataset names that returned valid elevation data, ordered by
            their position in the dataset_cascade (primary first). Datasets that
            returned data but aren't in the cascade are appended at the end.
        """
        in_cascade = [ds for ds in self.dataset_cascade if ds in self.datasets_used]
        extras = sorted(self.datasets_used - set(self.dataset_cascade))
        return in_cascade + extras

    def _fetch_single_batch(
        self,
        coordinates: List[Tuple[float, float]],
        max_retries: int,
        base_delay: float,
    ) -> List[Optional[float]]:
        """Fetch elevation data using optimized multi-dataset fallback (single request, server-side cascading)."""

        # Try optimized multi-dataset mode first (single request, server-side fallback)
        if self.use_multi_dataset_mode and self.multi_dataset_url:
            # DEBUG: Log multi-dataset attempt
            import logging

            logger = logging.getLogger(__name__)
            datasets = self.multi_dataset_url.split("/v1/")[-1].split("?")[0]
            logger.debug(f"[MULTI-DS] Attempting multi-dataset query: {datasets}")
            logger.debug(f"[MULTI-DS] URL: {self.multi_dataset_url}")
            logger.debug(
                f"[MULTI-DS] Batch size: {len(coordinates)}, Timeout: {ELEVATION_REQUEST_TIMEOUT_SEC}s"
            )

            result = self._fetch_single_batch_from_endpoint(
                coordinates, self.multi_dataset_url, max_retries, base_delay
            )

            # If multi-dataset mode succeeded, return immediately.
            # datasets_used is populated from the per-result "dataset" field
            # in _fetch_single_batch_from_endpoint — don't blanket-add the cascade here.
            if result is not None:
                valid_count = sum(1 for e in result if e is not None)
                if valid_count > 0:
                    logger.info(
                        f"[MULTI-DS] SUCCESS: Got {valid_count}/{len(coordinates)} elevations from multi-dataset query"
                    )
                    return result

            # If multi-dataset failed entirely, fall back to legacy mode
            if not self.multi_dataset_fallback_shown:
                with self._print_lock:
                    if not self.multi_dataset_fallback_shown:  # Double-check after acquiring lock
                        from climb_analyzer.utils.formatting import print_info, print_warning

                        print()
                        print_warning(
                            "Multi-dataset mode failed, falling back to sequential elevation fetching"
                        )
                        print_info(f"   Multi-dataset URL that failed: {self.multi_dataset_url}")
                        print_info(f"   Datasets attempted: {datasets}")
                        print_info(
                            "   Check logs above for specific error (HTTP 400/500/ConnectionError)"
                        )
                        print_info(
                            f"   Falling back to sequential queries: {' → '.join(self.dataset_cascade)}"
                        )
                        self.multi_dataset_fallback_shown = True

        # Legacy fallback mode: Try primary endpoint first, then fallbacks sequentially
        result = self._fetch_single_batch_from_endpoint(
            coordinates, self.primary_url, max_retries, base_delay
        )

        # Check if we got valid data or if we should try fallbacks
        if result is not None:
            valid_count = sum(1 for e in result if e is not None)
            if valid_count > 0:
                # Track the primary dataset that returned data
                self.datasets_used.add(self.primary_dataset)
                return result

        # Try fallback endpoints for coordinates with missing data
        for fallback_url in self.fallback_urls:
            fallback_result = self._fetch_single_batch_from_endpoint(
                coordinates, fallback_url, max_retries=5, base_delay=base_delay
            )

            if fallback_result is not None:
                # Track this fallback dataset if it returned any valid data
                fallback_valid = sum(1 for e in fallback_result if e is not None)
                if fallback_valid > 0:
                    # Extract dataset name from URL
                    fallback_dataset = fallback_url.split("/v1/")[-1].split("?")[0]
                    self.datasets_used.add(fallback_dataset)

                # Combine results - use fallback data where primary failed
                if result is None:
                    result = fallback_result
                else:
                    # Merge results - use fallback where primary returned None
                    for i, (primary_val, fallback_val) in enumerate(zip(result, fallback_result)):
                        if primary_val is None and fallback_val is not None:
                            result[i] = fallback_val

                # Check if we now have enough valid data
                valid_count = sum(1 for e in result if e is not None)
                if valid_count > 0:
                    missing_count = len(coordinates) - valid_count
                    # Continue to next fallback if we still have missing data
                    if missing_count == 0:
                        break

        # Return result or all None if all endpoints failed
        # Note: Don't print here - failure tracking is handled at higher level
        return result if result is not None else [None] * len(coordinates)

    def _fetch_single_batch_from_endpoint(
        self,
        coordinates: List[Tuple[float, float]],
        endpoint_url: str,
        max_retries: int,
        base_delay: float,
    ) -> List[Optional[float]]:
        """Fetch elevation data from a specific endpoint with retry logic."""
        # Extract dataset name from URL for tracking
        dataset_name = endpoint_url.split("/v1/")[-1].split("?")[0]

        # DEBUG: Import logger
        import logging
        import time as time_module

        logger = logging.getLogger(__name__)

        # Skip if this dataset is known to be unavailable (got 404 previously)
        if dataset_name in self.unavailable_datasets:
            return None

        locations_str = "|".join([f"{round(lat, 5)},{round(lon, 5)}" for lat, lon in coordinates])
        params = {"locations": locations_str}
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # DEBUG: Start timing
                request_start = time_module.time()

                response = requests.get(
                    endpoint_url,
                    params=params,
                    timeout=ELEVATION_REQUEST_TIMEOUT_SEC,
                    headers={"Connection": "close"},
                )

                # DEBUG: Calculate elapsed time
                request_elapsed = time_module.time() - request_start
                logger.debug(
                    f"[MULTI-DS] Request to {dataset_name} completed in {request_elapsed:.2f}s, Status: {response.status_code}"
                )

                # Check response before raising for status to get better error info
                if not response.ok:
                    # Extract base URL without parameters
                    base_url = endpoint_url.split("?")[0]

                    # Special handling for 404 - dataset not available
                    if response.status_code == 404:
                        # Mark this dataset as unavailable for entire session
                        self.unavailable_datasets.add(dataset_name)

                        # Print clear error message (only once per dataset)
                        print("\n" + "=" * 70)
                        print(f"⚠️  DATASET NOT AVAILABLE: {dataset_name}")
                        print("=" * 70)
                        print(f"HTTP 404: Dataset '{dataset_name}' not found on server")
                        print(f"URL: {base_url}")
                        print("\nThis dataset will be skipped for all remaining requests.")
                        print("Trying next fallback dataset...")
                        print("=" * 70 + "\n")

                        return None  # Immediately fail, don't retry

                    # Special handling for 400 - bad request (usually dataset not in config)
                    elif response.status_code == 400:
                        # Try to parse JSON error to detect dataset issues
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("error", "")

                            # DEBUG: Log HTTP 400 details for multi-dataset queries
                            is_multi_dataset = "," in dataset_name
                            if is_multi_dataset:
                                logger.error(
                                    f"[MULTI-DS] HTTP 400 error on multi-dataset query: {dataset_name}"
                                )
                                logger.error(f"[MULTI-DS] Error message: {error_msg}")
                                if "duplicate" in error_msg.lower():
                                    logger.error("[MULTI-DS] DUPLICATE DATASETS DETECTED in URL!")
                                    logger.error(
                                        f"[MULTI-DS] Dataset cascade: {self.dataset_cascade}"
                                    )

                            # Check if this is a "Dataset not in config" error
                            if "not in config" in error_msg.lower():
                                # Extract dataset name from error or URL
                                # Error format: "Dataset 'arcticdem32m' not in config."
                                import re

                                match = re.search(r"Dataset '([^']+)' not in config", error_msg)
                                bad_dataset = match.group(1) if match else dataset_name

                                # Try auto-recovery ONCE by syncing server config
                                if not self._config_sync_attempted:
                                    self._config_sync_attempted = True
                                    with self._print_lock:
                                        print(f"\n  ⚠️  Dataset '{bad_dataset}' not in server config")
                                        print("  Attempting automatic server reconfiguration...")

                                    try:
                                        from utils.opentopodata_manager import rebuild_and_restart
                                        if rebuild_and_restart(auto_update_config=True, validate_health=True):
                                            with self._print_lock:
                                                print("  ✓ Server reconfigured - retrying request")
                                            # Clear unavailable datasets - they may be available now
                                            self.unavailable_datasets.clear()
                                            self.http_400_shown_datasets.clear()
                                            # Retry this request
                                            continue
                                        else:
                                            with self._print_lock:
                                                print("  ⚠️  Config sync failed - continuing with fallback datasets")
                                    except Exception as e:
                                        with self._print_lock:
                                            print(f"  ⚠️  Config sync error: {e}")

                                # Mark this dataset as unavailable for entire session
                                self.unavailable_datasets.add(bad_dataset)

                                # Print clear error message (only once per dataset)
                                if bad_dataset not in self.http_400_shown_datasets:
                                    with self._print_lock:
                                        if (
                                            bad_dataset not in self.http_400_shown_datasets
                                        ):  # Double-check
                                            self.http_400_shown_datasets.add(bad_dataset)
                                            from climb_analyzer.utils.formatting import (
                                                print_info,
                                                print_separator,
                                                print_warning,
                                            )

                                            print()
                                            print_warning(f"DATASET NOT CONFIGURED: {bad_dataset}")
                                            print_separator(70, "─")
                                            print_info(
                                                f"HTTP 400: Dataset '{bad_dataset}' not in server config"
                                            )
                                            print_info(f"URL: {base_url}")
                                            print_info(f"Server response: {error_msg}")
                                            print()
                                            print_info(
                                                "This dataset will be skipped for all remaining requests."
                                            )
                                            print_info("Trying next fallback dataset...")
                                            print_separator(70, "─")
                                            print()

                                return None  # Immediately fail, don't retry
                            else:
                                # Other 400 error - print once and fail
                                if dataset_name not in self.http_400_shown_datasets:
                                    self.http_400_shown_datasets.add(dataset_name)
                                    print(f"⚠️  HTTP 400 error from {base_url}: {error_msg}")
                                return None
                        except:
                            # Couldn't parse JSON - print once and fail
                            if dataset_name not in self.http_400_shown_datasets:
                                self.http_400_shown_datasets.add(dataset_name)
                                print(f"⚠️  HTTP 400 error from {base_url} (could not parse error)")
                            return None

                    # Special handling for 429 rate limiting
                    elif response.status_code == 429:
                        if attempt < max_retries:
                            # Silently retry with exponential backoff: 1s, 4s, 16s, 64s, 256s
                            backoff_delay = base_delay * (4**attempt)
                            time.sleep(backoff_delay)
                            continue  # Skip raise_for_status and retry
                        else:
                            # Last attempt failed - increment counter
                            self.rate_limit_429_count += 1

                            # Show full message only once, then compact warnings
                            if not self.rate_limit_429_shown_full_msg:
                                print("\n" + "=" * 70)
                                print("⚠️  ELEVATION API RATE LIMITED (Transient)")
                                print("=" * 70)
                                print(f"HTTP 429 from {base_url} after {max_retries + 1} attempts")
                                print(
                                    "\nThis is likely a temporary rate limit, not the daily quota."
                                )
                                print("The script will continue processing other batches.")
                                print("\nIf you see many of these:")
                                print("  • Daily limit may be approaching (100,000 coords/day)")
                                print(
                                    "  • Consider reducing ELEVATION_MAX_CONCURRENT in config.yaml"
                                )
                                print("\n✓ Progress is automatically saved")
                                print("=" * 70 + "\n")
                                self.rate_limit_429_shown_full_msg = True
                            else:
                                # Compact warning for subsequent 429 errors
                                print(
                                    f"⚠️  HTTP 429 from {base_url} [{self.rate_limit_429_count} times]"
                                )
                            return None

                    # Special handling for 5xx server errors (including 504 gateway timeout)
                    # These are server-wide issues, not dataset-specific, so we retry the same endpoint
                    elif 500 <= response.status_code < 600:
                        # DEBUG: Log HTTP 500 for multi-dataset queries
                        is_multi_dataset = "," in dataset_name
                        if is_multi_dataset and attempt < max_retries:
                            logger.warning(
                                f"[MULTI-DS] HTTP {response.status_code} on multi-dataset query (attempt {attempt + 1}/{max_retries + 1}): {dataset_name}"
                            )
                            logger.warning(
                                f"[MULTI-DS] Request took {request_elapsed:.2f}s before error"
                            )

                        if attempt < max_retries:
                            # Silently retry with exponential backoff: 1s, 2s, 4s, 8s, 16s
                            backoff_delay = base_delay * (2**attempt)
                            time.sleep(backoff_delay)
                            continue
                        else:
                            # Last attempt failed - print error
                            error_name = {
                                500: "Internal Server Error",
                                502: "Bad Gateway",
                                503: "Service Unavailable",
                                504: "Gateway Timeout",
                            }.get(response.status_code, f"Server Error {response.status_code}")

                            # Enhanced error message for multi-dataset queries
                            if is_multi_dataset:
                                print(
                                    f"⚠️  MULTI-DATASET HTTP {response.status_code} ({error_name}) from {base_url} persisted after {max_retries + 1} attempts"
                                )
                                print(f"   Datasets queried: {dataset_name}")
                                print(
                                    f"   Batch size: {len(coordinates)}, Request time: {request_elapsed:.2f}s"
                                )
                                print(
                                    "   This is a server-wide issue. The server may be overloaded."
                                )
                                # Try to get response body for more details
                                try:
                                    if response.text and len(response.text) < 500:
                                        print(f"   Server response: {response.text}")
                                except:
                                    pass
                            else:
                                print(
                                    f"⚠️  HTTP {response.status_code} ({error_name}) from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                                )
                                print("   This is a server-wide issue, not dataset-specific.")
                                print("   The server may be overloaded or experiencing issues.")
                            return None

                    # Other HTTP errors (4xx except 404, 3xx, etc.) - retry with backoff
                    else:
                        if attempt < max_retries:
                            # Silently retry for other HTTP errors too
                            backoff_delay = base_delay * (2**attempt)
                            time.sleep(backoff_delay)
                            continue
                        else:
                            # Last attempt failed - print clean error
                            print(
                                f"⚠️  HTTP {response.status_code} error from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                            )
                            # Don't print HTML response bodies
                            if response.text and not response.text.strip().startswith("<"):
                                print(f"  Response: {response.text[:200]}")
                            return None

                response.raise_for_status()
                data = response.json()

                if data.get("status") != "OK":
                    base_url = endpoint_url.split("?")[0]
                    print(f"API returned status: {data.get('status')} from {base_url}")
                    # This is a successful connection but API-level failure
                    return [None] * len(coordinates)

                elevations = []
                results = data.get("results", [])

                for result in results:
                    if result and "elevation" in result and result["elevation"] is not None:
                        elevations.append(float(result["elevation"]))
                        # Record actual dataset that supplied this elevation so
                        # datasets_used reflects real usage, not just cascade config.
                        ds = result.get("dataset")
                        if ds:
                            self.datasets_used.add(ds)
                    else:
                        elevations.append(None)

                # Ensure the number of elevations matches the number of coordinates
                while len(elevations) < len(coordinates):
                    elevations.append(None)

                # DEBUG: Log if multi-dataset query returned all nulls (indicates data coverage issue)
                is_multi_dataset = "," in dataset_name
                if is_multi_dataset:
                    null_count = sum(1 for e in elevations if e is None)
                    valid_count = len(elevations) - null_count
                    logger.debug(
                        f"[MULTI-DS] Query returned {valid_count}/{len(elevations)} valid elevations ({null_count} nulls)"
                    )
                    if null_count == len(elevations) and len(elevations) > 0:
                        logger.warning(
                            f"[MULTI-DS] All elevations returned null from {dataset_name}"
                        )
                        logger.warning(
                            "[MULTI-DS] This indicates missing elevation data coverage for these coordinates"
                        )
                        # Print warning on first occurrence
                        if not hasattr(self, "_all_null_warning_shown"):
                            from climb_analyzer.utils.formatting import print_info, print_warning

                            print()
                            print_warning("Multi-dataset query returned all null elevations")
                            print_info(f"   Datasets queried: {dataset_name}")
                            print_info(
                                "   This indicates missing elevation data coverage for queried coordinates"
                            )
                            print_info("   Elevation data coverage issue (not a server error)")
                            print()
                            self._all_null_warning_shown = True

                return elevations[: len(coordinates)]

            except requests.exceptions.HTTPError as e:
                # This shouldn't happen anymore since we handle 429/504 before raise_for_status()
                # But keep as fallback for any other HTTP errors
                status_code = e.response.status_code if e.response else "Unknown"
                last_error = e
                base_url = endpoint_url.split("?")[0]

                # Only print on last attempt
                if attempt >= max_retries:
                    if status_code == "Unknown":
                        print(
                            f"⚠️  HTTP error with no status code from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                        )
                        print(f"  Error type: {type(e).__name__}, Details: {str(e)}")
                    else:
                        print(
                            f"⚠️  HTTP {status_code} error from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                        )
                    return None  # Return None to indicate failure

                # Retry silently
                if status_code == 429 or status_code == 504:
                    delay = (
                        base_delay * (4**attempt)
                        if status_code == 429
                        else base_delay * (2**attempt)
                    )
                else:
                    delay = base_delay * (2**attempt)  # Normal exponential backoff

                time.sleep(delay)
                continue

            except requests.exceptions.RequestException as e:
                last_error = e
                base_url = endpoint_url.split("?")[0]

                # DEBUG: Log connection errors for multi-dataset queries
                is_multi_dataset = "," in dataset_name
                if is_multi_dataset and attempt < max_retries:
                    logger.warning(
                        f"[MULTI-DS] ConnectionError on multi-dataset query (attempt {attempt + 1}/{max_retries + 1}): {dataset_name}"
                    )
                    logger.warning(f"[MULTI-DS] Error: {type(e).__name__}")

                # Silently retry - only print on final failure
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                else:
                    # Clean error message - strip query parameters
                    error_str = str(e)
                    import re

                    # Remove everything after ? in URLs (including multiline query params)
                    error_str = re.sub(
                        r"/v1/[^?]+\?[^)]+", lambda m: m.group(0).split("?")[0], error_str
                    )
                    # Also clean up any remaining ? query params
                    error_str = re.sub(r"\?locations=[^\s)]+", "", error_str)

                    # Enhanced error message for multi-dataset queries
                    if is_multi_dataset:
                        print(
                            f"⚠️  MULTI-DATASET Connection error after {max_retries + 1} attempts from {base_url}"
                        )
                        print(f"   Datasets queried: {dataset_name}")
                        print(f"   Batch size: {len(coordinates)}")
                        print(f"   {type(e).__name__}: {error_str}")
                        print(
                            "   This may indicate OpenTopoData server is not responding or overloaded."
                        )
                    else:
                        print(
                            f"⚠️  Connection error after {max_retries + 1} attempts from {base_url} (batch size: {len(coordinates)})"
                        )
                        print(f"  {type(e).__name__}: {error_str}")
                    return None  # Return None to indicate failure

            except (ValueError, KeyError) as e:
                # JSON parsing or key errors - these are usually not retryable
                base_url = endpoint_url.split("?")[0]
                print(f"Failed to parse elevation response from {base_url}: {e}")
                return [None] * len(coordinates)

        # This should never be reached due to the return statements above, but just in case
        return None

    def _fetch_batch_elevations_sync(
        self, coordinates: List[Tuple[float, float]]
    ) -> List[Optional[float]]:
        """Fetch elevation data for a batch of coordinates with aggressive adaptive batch sizing."""

        max_adaptation_attempts = 15  # Try up to 15 times to find working batch size
        base_delay = 1.0
        current_batch_size = len(coordinates)

        # If batch is already at or below minimum, try as-is first
        if current_batch_size <= self.min_batch_size:
            try:
                result = self._fetch_single_batch(coordinates, max_retries=5, base_delay=base_delay)
                # If we get all None values, that could indicate a server overload even at minimum size
                successful_count = sum(1 for e in result if e is not None)
                if successful_count == 0 and current_batch_size > 5:
                    print(
                        f"Got 0 successful elevations with batch size {current_batch_size}. Server may be overloaded."
                    )
                return result
            except Exception as e:
                status_code = (
                    getattr(e.response, "status_code", "Unknown")
                    if hasattr(e, "response")
                    else type(e).__name__
                )
                print(
                    f"Error with minimum batch size ({current_batch_size}): {status_code}. Cannot reduce further."
                )
                return [None] * len(coordinates)

        # For larger batches, implement aggressive adaptive sizing
        attempt = 0
        while attempt <= max_adaptation_attempts:
            try:
                # Try with current batch size
                if current_batch_size < len(coordinates):
                    print(f"Splitting batch into chunks of {current_batch_size}...")
                    # Split into smaller chunks
                    all_results = []
                    for i in range(0, len(coordinates), current_batch_size):
                        chunk = coordinates[i : i + current_batch_size]
                        try:
                            chunk_results = self._fetch_single_batch(
                                chunk, max_retries=5, base_delay=base_delay
                            )
                            all_results.extend(chunk_results)
                        except Exception as e:
                            # Any error should trigger size reduction
                            status_code = (
                                getattr(e.response, "status_code", "Unknown")
                                if hasattr(e, "response")
                                else type(e).__name__
                            )
                            print(f"Error in chunk (status: {status_code}) - reducing batch size")
                            if self._reduce_batch_size(f"Error (status: {status_code}) in chunk"):
                                # Restart the whole batch with new global size
                                return self._fetch_batch_elevations_sync(coordinates)
                            else:
                                # Cannot reduce further
                                print(
                                    "Cannot reduce batch size further. Filling chunk with None values."
                                )
                                all_results.extend([None] * len(chunk))

                    # Check if we got any successful results
                    successful_count = sum(1 for e in all_results if e is not None)
                    if successful_count == 0 and current_batch_size > self.min_batch_size:
                        print(f"Got 0 successful elevations with chunks of {current_batch_size}.")

                        # Try 3 attempts with 10-second waits before reducing batch size
                        retry_count = 0
                        max_retries = 3

                        while retry_count < max_retries:
                            retry_count += 1
                            print(f"Retry {retry_count}/{max_retries} after 10s wait...")
                            self._wait_with_backoff(10, f"retry {retry_count}")

                            # Test with a small chunk
                            if coordinates:
                                test_chunk = coordinates[: min(current_batch_size, 100)]
                                test_result = self._fetch_single_batch(
                                    test_chunk, max_retries=1, base_delay=base_delay
                                )
                                test_successful = sum(1 for e in test_result if e is not None)

                                if test_successful > 0:
                                    print("Server recovered - retrying full batch")
                                    break

                        # If all retries failed, reduce batch size
                        if retry_count >= max_retries:
                            print("All retries failed - reducing batch size by 10%")
                            if self._reduce_batch_size(
                                f"0 successful elevations after {max_retries} retries"
                            ):
                                attempt += 1
                                current_batch_size = self.optimal_batch_size
                                continue

                    # Update global optimal batch size if this worked
                    if current_batch_size != self.optimal_batch_size:
                        print(f"Confirmed optimal batch size: {current_batch_size}")
                        self.optimal_batch_size = current_batch_size
                        self.batch_size_adapted = True

                    return all_results
                else:
                    # Try with full batch
                    result = self._fetch_single_batch(
                        coordinates, max_retries=5, base_delay=base_delay
                    )

                    # Check if we got any successful results
                    successful_count = sum(1 for e in result if e is not None)
                    if successful_count == 0:
                        print(
                            f"Got 0 successful elevations with full batch size {current_batch_size}."
                        )

                        # Try 3 attempts with 10-second waits before reducing batch size
                        retry_count = 0
                        max_retries = 3

                        while retry_count < max_retries:
                            retry_count += 1
                            print(f"Retry {retry_count}/{max_retries} after 10s wait...")
                            self._wait_with_backoff(10, f"retry {retry_count}")

                            # Test with same batch size
                            result_retry = self._fetch_single_batch(
                                coordinates, max_retries=1, base_delay=base_delay
                            )
                            successful_retry_count = sum(1 for e in result_retry if e is not None)

                            if successful_retry_count > 0:
                                print("Server recovered - returning successful result")
                                return result_retry

                        # If all retries failed and batch is larger than minimum, reduce batch size
                        if retry_count >= max_retries and current_batch_size > self.min_batch_size:
                            print("All retries failed - reducing batch size by 10%")
                            if self._reduce_batch_size(
                                f"0 successful elevations after {max_retries} retries"
                            ):
                                attempt += 1
                                current_batch_size = self.optimal_batch_size
                                continue

                    return result

            except Exception as e:
                # Any exception should trigger adaptive sizing
                status_code = (
                    getattr(e.response, "status_code", "Unknown")
                    if hasattr(e, "response")
                    else type(e).__name__
                )
                print(
                    f"HTTP/Connection error (status: {status_code}) - attempt {attempt + 1}/{max_adaptation_attempts + 1}"
                )

                # Reduce batch size globally and try again
                if self._reduce_batch_size(f"Error (status: {status_code})"):
                    current_batch_size = self.optimal_batch_size
                    attempt += 1

                    if current_batch_size >= self.min_batch_size:
                        time.sleep(base_delay * min(attempt, 5))  # Cap delay at 5 seconds
                        continue
                    else:
                        print(
                            f"Error persists even with minimum batch size ({self.min_batch_size}). Failing batch."
                        )
                        return [None] * len(coordinates)
                else:
                    # Cannot reduce batch size further
                    print(f"Cannot reduce batch size below {self.min_batch_size}. Failing batch.")
                    return [None] * len(coordinates)

        # If we exit the loop, we've exhausted our adaptation attempts
        print(
            f"Exhausted {max_adaptation_attempts + 1} attempts for batch size adaptation. Failing batch."
        )
        return [None] * len(coordinates)

    def _resume_elevation_fetching_sync(
        self,
        coordinates: List[Tuple[float, float]],
        existing_mapping: Dict,
        batch_info: Dict,
        persistence,
        progress_desc: str,
    ) -> List[Optional[float]]:
        """Resume elevation fetching from checkpoint using sync requests."""

        print(
            f"Resuming elevation fetching from batch {batch_info.get('completed_batches', 0)}/{batch_info.get('total_batches', 0)}"
        )

        # Rebuild coordinate structure
        unique_coords = []
        coord_to_index = {}
        original_to_unique = []

        for i, coord in enumerate(coordinates):
            rounded_coord = (round(coord[0], 6), round(coord[1], 6))

            if rounded_coord not in coord_to_index:
                coord_to_index[rounded_coord] = len(unique_coords)
                unique_coords.append(rounded_coord)

            original_to_unique.append(coord_to_index[rounded_coord])

        total_unique = len(unique_coords)
        all_elevations = [None] * total_unique

        # Fill in existing elevations
        for i, coord in enumerate(unique_coords):
            if coord in existing_mapping:
                all_elevations[i] = existing_mapping[coord]

        existing_count = sum(1 for e in all_elevations if e is not None)
        remaining_count = total_unique - existing_count

        print(f"Loaded {existing_count} existing elevations, {remaining_count} remaining to fetch")

        if remaining_count == 0:
            print("All elevations already completed!")
            # Map back to original order
            result_elevations = [all_elevations[unique_idx] for unique_idx in original_to_unique]
            return result_elevations

        # Continue fetching missing elevations using current optimal batch size
        # Note: We scan all batches but skip coordinates already in existing_mapping
        # This handles parallel processing where batches complete out of order
        coordinate_mapping = existing_mapping.copy()

        # Prepare all batches that need fetching
        pending_batches = []
        for batch_num in range(0, total_unique, self.optimal_batch_size):
            batch_coords = unique_coords[batch_num : batch_num + self.optimal_batch_size]

            # Skip coordinates we already have
            coords_to_fetch = []
            coord_indices = []
            for j, coord in enumerate(batch_coords):
                coord_idx = batch_num + j
                if coord_idx < total_unique and all_elevations[coord_idx] is None:
                    coords_to_fetch.append(coord)
                    coord_indices.append(coord_idx)

            if coords_to_fetch:
                pending_batches.append((batch_num, coords_to_fetch, coord_indices))

        completed_coords = 0
        total_coords_to_fetch = sum(len(batch[1]) for batch in pending_batches)

        # Set signal handler operation (only coordinate_mapping is used for checkpoint)
        signal_handler.set_operation(
            "elevation_fetching",
            {
                "coordinate_mapping": coordinate_mapping,
                "batch_info": {},  # Not needed - coordinate_mapping tracks everything
            },
        )

        with tqdm(
            total=total_unique,  # Total original coordinates
            initial=existing_count,  # Start at number already completed
            desc=f"{progress_desc} (resumed)",
            unit="coords",
            mininterval=1.0,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            # Process all batches in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=ELEVATION_MAX_CONCURRENT) as executor:

                def fetch_batch(batch_info):
                    batch_num, coords_to_fetch, coord_indices = batch_info
                    try:
                        batch_elevations = self._fetch_single_batch(
                            coords_to_fetch, max_retries=5, base_delay=1.0
                        )
                        return (coord_indices, batch_elevations, coords_to_fetch)
                    except Exception as e:
                        print(f"Error in resumed batch {batch_num}: {e}")
                        return (
                            coord_indices,
                            [None] * len(coords_to_fetch),
                            coords_to_fetch,
                        )

                # Submit all batches immediately
                future_to_batch = {
                    executor.submit(fetch_batch, batch_info): batch_info
                    for batch_info in pending_batches
                }

                # Process results as they complete with periodic interrupt checking
                remaining_futures = set(future_to_batch.keys())
                while remaining_futures:
                    # Check for shutdown signal
                    if signal_handler.kill_now:
                        print("\nCanceling remaining elevation requests...")
                        for f in remaining_futures:
                            f.cancel()
                        print("Elevation progress saved. Analysis can be resumed.")
                        os._exit(0)  # Force immediate exit without waiting for threads

                    # Wait for next future with timeout to allow periodic signal checking
                    try:
                        done, remaining_futures = concurrent.futures.wait(
                            remaining_futures,
                            timeout=0.5,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                    except Exception:
                        continue

                    # Process completed futures
                    completed_batch_count = 0
                    for future in done:
                        try:
                            coord_indices, batch_elevations, coords_to_fetch = future.result()

                            # Store results
                            for idx, elevation in zip(coord_indices, batch_elevations):
                                all_elevations[idx] = elevation
                                if elevation is not None:
                                    coordinate_mapping[unique_coords[idx]] = elevation

                            successful_elevations = sum(
                                1 for e in batch_elevations if e is not None
                            )
                            completed_coords += len(coords_to_fetch)
                            completed_batch_count += 1

                            # Update signal handler with current progress
                            signal_handler.set_operation(
                                "elevation_fetching",
                                {
                                    "coordinate_mapping": coordinate_mapping,
                                    "batch_info": {},  # Not needed - coordinate_mapping tracks everything
                                },
                            )

                            pbar.set_postfix(
                                {
                                    "success": f"{successful_elevations}/{len(coords_to_fetch)}",
                                    "total": len(coordinate_mapping),
                                    "batch_size": self.optimal_batch_size,
                                }
                            )
                            pbar.update(len(coords_to_fetch))

                        except Exception as e:
                            print(f"  Error processing resumed batch result: {e}")
                            pbar.update(1)

        # Map back to original coordinate order
        result_elevations = [all_elevations[unique_idx] for unique_idx in original_to_unique]

        successful_total = sum(1 for e in result_elevations if e is not None)
        print(
            f"Successfully fetched elevation data for {successful_total}/{len(coordinates)} coordinates"
        )

        # Clear checkpoint after successful completion
        persistence.clear_elevation_progress()

        # Clean up memory
        check_and_cleanup_memory(force_cleanup=True)

        return result_elevations


def configure_checkpoints(
    time_minutes: float = 5.0, milestones: list = None, save_at_completion: bool = True
):
    """Configure global checkpoint settings"""
    global CHECKPOINT_CONFIG
    CHECKPOINT_CONFIG.time_interval_minutes = time_minutes
    CHECKPOINT_CONFIG.progress_milestones = milestones or [25, 50, 75, 100]
    CHECKPOINT_CONFIG.save_at_completion = save_at_completion
    # Checkpoint config is internal - don't print


def check_and_cleanup_memory(threshold_percent=80, force_cleanup=False):
    """Check memory usage and cleanup if needed"""
    try:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > threshold_percent or force_cleanup:
            # print(f"Memory usage: {memory_percent:.1f}% - performing cleanup...")
            gc.collect()
            new_memory = psutil.virtual_memory().percent
            # print(f"Memory after cleanup: {new_memory:.1f}%")
            return True
        return False
    except:
        # Fallback if psutil not available
        if force_cleanup:
            gc.collect()
        return False


class ThreadSafeChunkProcessor:
    """Thread-safe wrapper for chunk processing with rate limiting"""

    def __init__(
        self,
        road_analyzer,
        chunk_merger,
        max_workers=OSM_MAX_THREADS,
        api_delay=OVERPASS_API_DELAY_SEC,
    ):
        self.road_analyzer = road_analyzer
        self.chunk_merger = chunk_merger
        self.max_workers = max_workers
        self.api_delay = api_delay
        self.lock = threading.Lock()
        self.api_lock = threading.Lock()  # Separate lock for API rate limiting
        self.processed_chunks = []
        self.chunk_results = {}

    def process_single_chunk(self, chunk_index, chunk_lat, chunk_lon, chunk_radius):
        """Process a single chunk in a thread-safe manner"""

        try:
            # Rate limit API calls
            with self.api_lock:
                time.sleep(self.api_delay)

            # Get roads for this chunk
            chunk_ways = self.road_analyzer.get_roads_in_chunk(chunk_lat, chunk_lon, chunk_radius)

            # Process ways into segments
            chunk_segments = []
            if chunk_ways:
                chunk_segments = self.chunk_merger.process_way_chunk(chunk_ways)

            # Return results (don't save here - do it in main thread)
            return {
                "chunk_index": chunk_index,
                "chunk_segments": chunk_segments,
                "chunk_info": (chunk_lat, chunk_lon, chunk_radius),
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "chunk_index": chunk_index,
                "chunk_segments": [],
                "chunk_info": (chunk_lat, chunk_lon, chunk_radius),
                "success": False,
                "error": str(e),
            }


def process_all_chunks_parallel(
    persistence,
    road_analyzer,
    chunk_merger,
    boundary_merger,
    elevation_fetcher,
    chunks,
    metadata,
    min_score,
    unit_system,
    score_type,
    enable_geocoding,
    max_workers=OSM_MAX_THREADS,
):
    """Parallel version of process_all_chunks with same interface"""

    # Setup signal handling (same as before)
    signal_handler.set_persistence_manager(persistence)
    total_chunks = len(chunks)
    processed_chunks = []
    # CRITICAL MEMORY FIX: Don't accumulate segments in memory during processing
    # With 2.2M segments = 11GB, Docker containers hit memory limits and swap
    # Segments are saved to disk during checkpoints - load later for boundary merge
    # This keeps memory constant at ~1-2GB instead of growing to 15-20GB
    all_merged_segments = None  # Will load from disk after chunk processing completes

    print(
        f"Processing road data with {max_workers} worker thread{'s' if max_workers > 1 else ''} (checkpoint saving enabled)..."
    )

    # Create thread-safe processor
    processor = ThreadSafeChunkProcessor(road_analyzer, chunk_merger, max_workers)

    # Initialize smart checkpointer
    checkpointer = SmartCheckpointer(total_chunks, "Parallel Chunk Processing")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks to thread pool
        future_to_chunk = {}
        for chunk_index, (chunk_lat, chunk_lon, chunk_radius) in enumerate(chunks):
            future = executor.submit(
                processor.process_single_chunk,
                chunk_index,
                chunk_lat,
                chunk_lon,
                chunk_radius,
            )
            future_to_chunk[future] = chunk_index

        # Performance optimization: Track counts instead of recalculating
        active_thread_count = len(future_to_chunk)  # Start with total chunks
        total_segment_count = 0  # Track segments without calling len()
        processing_start_time = time.time()  # Track overall processing time

        # Process completed chunks as they finish
        with tqdm(
            total=total_chunks,
            desc="Processing chunks",
            unit="chunk",
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
        ) as pbar:
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]

                # Signal handler check
                signal_handler.set_operation(
                    "chunk_processing",
                    {
                        "processed_chunks": processed_chunks,
                        "total_chunks": total_chunks,
                        "metadata": metadata,
                    },
                )

                if signal_handler.kill_now:
                    # Cancel remaining futures
                    for remaining_future in future_to_chunk:
                        remaining_future.cancel()

                    persistence.save_progress(processed_chunks, total_chunks, metadata)
                    print("Chunk progress saved. Analysis can be resumed.")
                    sys.exit(0)

                try:
                    # Get result from completed chunk
                    result = future.result()

                    # Decrement active thread count (Performance: O(1) instead of O(n))
                    active_thread_count -= 1

                    if result["success"]:
                        # MEMORY FIX: Don't accumulate in memory - segments saved to disk during checkpoints
                        # Just track the count for progress monitoring
                        processed_chunks.append(result["chunk_index"])

                        # Update segment count (Performance: O(1) instead of calling len())
                        total_segment_count += len(result["chunk_segments"])

                        # Performance: Only update progress bar every 10 chunks to reduce overhead
                        if len(processed_chunks) % 10 == 0:
                            # Calculate segments per second for density-aware performance metric
                            elapsed_time = time.time() - processing_start_time
                            segments_per_sec = (
                                total_segment_count / elapsed_time if elapsed_time > 0 else 0
                            )

                            pbar.set_postfix(
                                {
                                    "total_segments": total_segment_count,
                                    "seg/s": f"{segments_per_sec:.0f}",
                                    "completed": len(processed_chunks),
                                }
                            )
                    else:
                        # Handle failed chunk
                        print(
                            f"\nError processing chunk {result['chunk_index']}: {result['error']}"
                        )
                        processed_chunks.append(result["chunk_index"])

                        # Performance: Update postfix only on errors (rare)
                        pbar.set_postfix(
                            {
                                "status": "error",
                                "total_segments": total_segment_count,
                            }
                        )

                    # Smart checkpoint save (every 15 min or at milestones: 10%, 25%, 50%, 75%, 90%)
                    if checkpointer.should_checkpoint(len(processed_chunks) - 1):
                        # Save chunk data only during checkpoints (not every chunk!)
                        # This reduces disk I/O from 971 saves to ~10-15 saves
                        persistence.save_chunk(
                            result["chunk_index"],
                            result["chunk_segments"],
                            result["chunk_info"],
                        )

                        persistence.save_progress(processed_chunks, total_chunks, metadata)

                        info = checkpointer.get_checkpoint_info(len(processed_chunks) - 1)

                        # Calculate performance metric
                        elapsed_time = time.time() - processing_start_time
                        segments_per_sec = (
                            total_segment_count / elapsed_time if elapsed_time > 0 else 0
                        )

                        pbar.set_postfix(
                            {
                                "total_segments": total_segment_count,
                                "seg/s": f"{segments_per_sec:.0f}",
                                "completed": len(processed_chunks),
                                "next_save": f"{info['time_until_next_min']:.1f}min",
                            }
                        )

                    pbar.update(1)

                    # Memory cleanup
                    if len(processed_chunks) % 100 == 0:
                        check_and_cleanup_memory()

                except Exception as e:
                    print(f"\nError processing chunk {chunk_index}: {e}")
                    # Save empty chunk to mark as processed
                    persistence.save_chunk(chunk_index, [], chunks[chunk_index])
                    processed_chunks.append(chunk_index)
                    pbar.update(1)

    # MEMORY FIX: Load segments from disk now that chunk processing is complete
    print(f"\n✓ Chunk processing complete! Loading {total_segment_count} segments from disk...")
    all_merged_segments = []

    with tqdm(
        total=len(processed_chunks),
        desc="Loading segments from disk",
        ascii=" █",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for chunk_index in processed_chunks:
            chunk_data = persistence.load_chunk(chunk_index)
            if chunk_data:
                all_merged_segments.extend(chunk_data)
            pbar.update(1)

    print(f"Loaded {len(all_merged_segments)} road segments from {len(processed_chunks)} chunks")

    # Continue with the rest of the analysis (same as before)
    return complete_analysis_from_segments(
        all_merged_segments,
        elevation_fetcher,
        boundary_merger,
        metadata,
        min_score,
        unit_system,
        persistence,
        score_type,
        enable_geocoding,
    )


# NOTE: Multiprocessing was attempted but failed due to memory constraints.
# Each worker loads the entire 1.5M way spatial index (~1.5GB per worker).
# With 16 workers = 24GB just for indexes, causing workers to be killed by OS.
# Serial processing with periodic rtree reload is the practical solution.


def continue_chunk_processing_parallel(
    persistence,
    road_analyzer,
    chunk_merger,
    boundary_merger,
    elevation_fetcher,
    chunks,
    metadata,
    existing_segments,
    processed_chunks,
    min_score,
    unit_system,
    score_type,
    enable_geocoding,
    max_workers=OSM_MAX_THREADS,
):
    """Parallel version of continue_chunk_processing"""

    total_chunks = len(chunks)
    processed_set = set(processed_chunks)
    # MEMORY FIX: Don't keep existing segments in memory - they're on disk
    # Will load all segments from disk after chunk processing completes
    existing_segment_count = len(existing_segments) if existing_segments else 0
    all_merged_segments = None  # Don't accumulate in memory

    print(
        f"Continuing chunk processing with {max_workers} worker thread{'s' if max_workers > 1 else ''}..."
    )

    remaining_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in processed_set]

    if not remaining_chunks:
        print("No remaining chunks to process!")
        return complete_analysis_from_segments(
            all_merged_segments,
            elevation_fetcher,
            boundary_merger,
            metadata,
            min_score,
            unit_system,
            persistence,
            score_type,
            enable_geocoding,
        )

    # Create thread-safe processor
    processor = ThreadSafeChunkProcessor(road_analyzer, chunk_merger, max_workers)

    # Initialize smart checkpointer (same as main processing path)
    checkpointer = SmartCheckpointer(total_chunks, "Resume Chunk Processing")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit remaining chunks
        future_to_chunk = {}
        for chunk_index, (chunk_lat, chunk_lon, chunk_radius) in remaining_chunks:
            future = executor.submit(
                processor.process_single_chunk,
                chunk_index,
                chunk_lat,
                chunk_lon,
                chunk_radius,
            )
            future_to_chunk[future] = chunk_index

        # Performance optimization: Track segment count instead of calling len()
        total_segment_count = existing_segment_count  # Start with existing count
        processing_start_time = time.time()  # Track processing time

        # Process results
        with tqdm(
            total=len(remaining_chunks),
            desc="Processing remaining chunks",
            unit="chunk",
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
        ) as pbar:
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]

                try:
                    result = future.result()

                    if result["success"]:
                        # MEMORY FIX: Don't accumulate - segments saved to disk
                        # Performance: Update count instead of calling len()
                        total_segment_count += len(result["chunk_segments"])
                    else:
                        print(f"Error in chunk {result['chunk_index']}: {result['error']}")

                    processed_chunks.append(result["chunk_index"])

                    # Save progress only on checkpoints (every 15min or milestones)
                    if checkpointer.should_checkpoint(len(processed_chunks) - 1):
                        # Save chunk data during checkpoints only
                        persistence.save_chunk(
                            result["chunk_index"],
                            result["chunk_segments"],
                            result["chunk_info"],
                        )

                        persistence.save_progress(processed_chunks, total_chunks, metadata)

                        info = checkpointer.get_checkpoint_info(len(processed_chunks) - 1)

                        # Calculate performance metric
                        elapsed_time = time.time() - processing_start_time
                        segments_per_sec = (
                            total_segment_count / elapsed_time if elapsed_time > 0 else 0
                        )

                        pbar.set_postfix(
                            {
                                "total_segments": total_segment_count,
                                "seg/s": f"{segments_per_sec:.0f}",
                                "completed": len(processed_chunks),
                                "next_save": f"{info['time_until_next_min']:.1f}min",
                            }
                        )

                    # Performance: Only update progress bar every 10 chunks
                    if len(processed_chunks) % 10 == 0:
                        # Calculate performance metric
                        elapsed_time = time.time() - processing_start_time
                        segments_per_sec = (
                            total_segment_count / elapsed_time if elapsed_time > 0 else 0
                        )

                        pbar.set_postfix(
                            {
                                "total_segments": total_segment_count,
                                "seg/s": f"{segments_per_sec:.0f}",
                                "completed": f"{len(processed_chunks)}/{total_chunks}",
                            }
                        )
                    pbar.update(1)

                    # Memory cleanup
                    if len(processed_chunks) % 20 == 0:
                        check_and_cleanup_memory()

                except Exception as e:
                    print(f"\nError processing chunk {chunk_index}: {e}")
                    processed_chunks.append(chunk_index)
                    pbar.update(1)

    # MEMORY FIX: Load segments from disk now that chunk processing is complete
    print(f"\n✓ Chunk processing complete! Loading {total_segment_count} segments from disk...")
    all_merged_segments = []

    with tqdm(
        total=len(processed_chunks),
        desc="Loading segments from disk",
        ascii=" █",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for chunk_index in processed_chunks:
            chunk_data = persistence.load_chunk(chunk_index)
            if chunk_data:
                all_merged_segments.extend(chunk_data)
            pbar.update(1)

    print(f"Loaded {len(all_merged_segments)} road segments from {len(processed_chunks)} chunks")

    return complete_analysis_from_segments(
        all_merged_segments,
        elevation_fetcher,
        boundary_merger,
        metadata,
        min_score,
        unit_system,
        persistence,
        score_type,
        enable_geocoding,
    )


def resume_chunk_processing_parallel(
    analysis_id: str,
    min_score: float,
    unit_system: str,
    score_type: str,
    max_workers: int = OSM_MAX_THREADS,
):
    """Resume function with persistent elevation storage"""

    persistence = ChunkPersistenceManager(analysis_id)
    processed_chunks, total_chunks, metadata = persistence.load_progress()
    elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()

    print(f"  - Processed chunks: {len(processed_chunks)}/{total_chunks}")
    print(f"  - Elevation mapping: {len(elevation_mapping)} coordinates")

    # Check if all chunks are complete
    if len(processed_chunks) >= total_chunks:
        # *** NEW: Check for completed elevation data first ***
        if persistence.has_completed_elevations():

            # Load all chunk data and proceed directly to analysis
            all_merged_segments = []
            print("Loading all processed chunks...")
            with tqdm(
                total=len(processed_chunks),
                desc="Loading saved chunks",
                unit="chunk",
                dynamic_ncols=True,
                ascii=" ▏▎▍▌▋▊▉█",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ) as pbar:
                for chunk_index in processed_chunks:
                    chunk_data = persistence.load_chunk(chunk_index)
                    if chunk_data:
                        all_merged_segments.extend(chunk_data)
                    pbar.update(1)

            print(f"Loaded {len(all_merged_segments)} road segments from all chunks")

            # Initialize components and proceed directly to analysis
            boundary_merger = BoundaryMerger(coordinate_tolerance=0.002)

            return complete_analysis_from_segments(
                all_merged_segments,
                None,  # elevation_fetcher not needed
                boundary_merger,
                metadata,
                min_score,
                unit_system,
                persistence,
                score_type,
            )

        # Check for in-progress elevation checkpoints
        elif elevation_mapping or elevation_batch_info.get("completed_batches", 0) > 0:
            print("Found elevation progress to resume:")
            print(f"  - Coordinate mappings: {len(elevation_mapping)}")
            print(f"  - Completed batches: {elevation_batch_info.get('completed_batches', 0)}")

            # Resume elevation fetching...
            # (existing resume logic)

        else:
            print("INFO: No elevation progress found, starting fresh elevation analysis")
            # (existing fresh start logic)

    # If chunks are not complete, continue chunk processing with parallel processing
    print(f"Continuing chunk processing: {len(processed_chunks)}/{total_chunks} completed")

    # Load existing chunks
    all_merged_segments = []
    print("Loading previously processed chunks...")
    with tqdm(
        total=len(processed_chunks),
        desc="Loading saved chunks",
        unit="chunk",
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for chunk_index in processed_chunks:
            chunk_data = persistence.load_chunk(chunk_index)
            if chunk_data:
                all_merged_segments.extend(chunk_data)
            pbar.update(1)
            if chunk_index % 100 == 0:
                check_and_cleanup_memory()
    print(
        f"Loaded {len(all_merged_segments)} road segments from {len(processed_chunks)} completed chunks"
    )
    check_and_cleanup_memory(force_cleanup=True)

    # Continue processing remaining chunks
    remaining_chunks = total_chunks - len(processed_chunks)

    # Rebuild components and continue
    chunks = metadata.get("chunks", [])
    surface_filter = metadata.get("surface_filter", "all")
    chunk_size_km = metadata.get("chunk_size_km", 30.0)  # Default chunk size
    cycling_only = metadata.get("cycling_only", True)
    enable_geocoding = metadata.get("enable_geocoding", True)

    # Initialize deployment-aware road analyzer
    deployment_type = DEPLOYMENT_TYPE

    if deployment_type == "local":
        # CRITICAL: Local deployment MUST use serial processing (rtree not thread-safe)
        max_workers = 1

        # Determine region name for OSM file selection
        from utils.data_validator import find_osm_file_for_region

        region_name = None
        country = metadata.get("country")
        formatted_address = metadata.get("formatted_address")

        # For US states, extract state name
        if country == "United States" and formatted_address:
            state_name = formatted_address.split(",")[0].strip()
            if is_us_state(state_name):
                region_name = state_name
            else:
                region_name = country
        elif country:
            region_name = country

        # Find the correct OSM file for this region
        osm_file = None
        if region_name:
            osm_file = find_osm_file_for_region(region_name)
            # DO NOT fall back to cached file if region name doesn't match
        else:
            # Only use configured/cached file if no region name specified
            try:
                osm_file_path = get_configured_osm_file_path()
                osm_file = Path(osm_file_path) if osm_file_path else None
            except FileNotFoundError:
                osm_file = None

        if osm_file and osm_file.exists():
            from climb_analyzer.utils.formatting import print_dim

            print_dim(f"Using OSM file: {osm_file.name}")
            road_analyzer = ChunkedRoadNetworkAnalyzer(
                surface_filter,
                chunk_size_km,
                cycling_only,
                osm_file_path=str(osm_file),
            )
        else:
            from climb_analyzer.utils.formatting import print_dim

            # No region-specific OSM file found, will auto-detect
            print_dim("   Will auto-detect OSM file from data/planet_osm_data/...")
            road_analyzer = ChunkedRoadNetworkAnalyzer(
                surface_filter,
                chunk_size_km,
                cycling_only,
            )
    else:
        # Cloud deployment - also use serial for API rate limits
        max_workers = 1
        road_analyzer = ChunkedRoadNetworkAnalyzer(surface_filter, chunk_size_km, cycling_only)

    chunk_merger = MemoryEfficientMerger()
    boundary_merger = BoundaryMerger(coordinate_tolerance=0.002)
    # Get region name for dataset priority selection
    region_name = get_region_name_from_config()
    elevation_fetcher = FastElevationFetcher(region_name=region_name)

    # Use serial processing - parallel causes thread contention and memory issues
    # Note: Don't pass all_merged_segments to avoid memory accumulation
    # Segments will be loaded from disk after chunk processing completes
    return continue_chunk_processing(
        persistence,
        road_analyzer,
        chunk_merger,
        boundary_merger,
        elevation_fetcher,
        chunks,
        metadata,
        [],  # Pass empty list instead of all_merged_segments to avoid memory issues
        processed_chunks,
        min_score,
        unit_system,
        score_type,
        enable_geocoding,
    )


class SmartCheckpointer:
    """Intelligent checkpoint manager with time-based and milestone-based saving"""

    def __init__(self, total_items: int, operation_name: str = "Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_progress = 0
        self.milestone_index = 0

        # Pre-calculate next milestone
        self.next_milestone = (
            CHECKPOINT_CONFIG.progress_milestones[0]
            if CHECKPOINT_CONFIG.progress_milestones
            else 100
        )

    def should_checkpoint(self, current_item: int, force_check: bool = False) -> bool:
        """
        Determine if we should save a checkpoint now.

        Args:
            current_item: Current item being processed (0-based)
            force_check: Force a time check even if not at a milestone

        Returns:
            True if checkpoint should be saved
        """
        current_time = time.time()
        current_progress = ((current_item + 1) / self.total_items) * 100

        # Time-based checkpoint check
        time_elapsed = (current_time - self.last_checkpoint_time) / 60.0  # minutes
        time_trigger = time_elapsed >= CHECKPOINT_CONFIG.time_interval_minutes

        # Progress milestone check
        milestone_trigger = current_progress >= self.next_milestone

        # Force completion checkpoint
        completion_trigger = (
            current_item + 1 >= self.total_items and CHECKPOINT_CONFIG.save_at_completion
        )

        # Decide whether to checkpoint
        should_save = time_trigger or milestone_trigger or completion_trigger

        if should_save:
            self._update_after_checkpoint(current_time, current_progress)

        return should_save

    def _update_after_checkpoint(self, current_time: float, current_progress: float):
        """Update internal state after checkpoint"""
        self.last_checkpoint_time = current_time
        self.last_checkpoint_progress = current_progress

        # Move to next milestone
        if (
            current_progress >= self.next_milestone
            and self.milestone_index < len(CHECKPOINT_CONFIG.progress_milestones) - 1
        ):
            self.milestone_index += 1
            self.next_milestone = CHECKPOINT_CONFIG.progress_milestones[self.milestone_index]

    def get_checkpoint_info(self, current_item: int) -> Dict[str, Any]:
        """Get info about current checkpoint status"""
        current_progress = ((current_item + 1) / self.total_items) * 100
        time_since_last = (time.time() - self.last_checkpoint_time) / 60.0

        return {
            "progress_pct": current_progress,
            "next_milestone": self.next_milestone,
            "time_since_last_min": time_since_last,
            "time_until_next_min": max(
                0, CHECKPOINT_CONFIG.time_interval_minutes - time_since_last
            ),
        }


@dataclass
class ClimbSegment:
    """Represents a climbing segment with elevation profile"""

    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    distance_km: float
    elevation_gain_m: float
    avg_gradient: float
    max_gradient: float
    category: str
    points: List[Tuple[float, float, float]]  # (lat, lon, elevation)


@dataclass
class ClimbMetrics:
    street_name: str
    climb_category: str
    climb_score: float
    elevation_gain: float
    height: float
    prominence: float
    length_km: float
    distance_km: float  # Distance between first and last elevation points
    avg_grade: float
    max_grade: float
    min_elevation: float
    max_elevation: float
    surface: str
    tracktype: str  # For tracks, the tracktype classification
    tracktype_definition: str  # Human readable definition
    way_ids: List[int]  # OSM way IDs that make up this road
    osm_links: List[str]  # Links to OSM for each way
    city_state: str = "Unknown"  # City and state information
    country: str = "Unknown"  # Country information for boundary detection
    distance_from_center_km: float = 0.0  # Distance from search center
    mid_lat: float = 0.0  # Midpoint latitude for location lookup
    mid_lon: float = 0.0  # Midpoint longitude for location lookup
    start_lat: float = 0.0  # Start latitude of climb
    start_lon: float = 0.0  # Start longitude of climb
    nodes: List["SimpleNode"] = None  # Nodes for location/distance lookup
    fiets_score: float = 0.0
    pdi_score: float = 0.0
    cycling_access: str = "Unknown"
    connected_climbs: List[str] = None  # List of connected climb street names
    elevation_profile: str = ""  # Compact elevation profile string
    highway_type: str = "unknown"


@dataclass(frozen=True)
class ClimbEndpoints:
    """Minimal data structure for connected climb detection using spatial indexing."""

    climb_idx: int
    street_name: str
    way_id: int  # OSM way ID for the starting segment
    start_coord: Tuple[float, float]  # (lat, lon) rounded to 6 decimals
    end_coord: Tuple[float, float]
    grid_cells: frozenset  # Grid cells this climb touches (immutable for hashing)
    max_elevation: float = 0.0  # Max elevation for connection direction checking
    surface: str = "unknown"  # Surface type for same-name different-surface connections


def get_grid_cell(lat: float, lon: float, cell_size: float = 0.01) -> Tuple[int, int]:
    """Convert coordinate to grid cell. cell_size=0.01 ≈ 1km at equator."""
    return (int(lat / cell_size), int(lon / cell_size))


def get_adjacent_cells(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return cell and 8 adjacent cells (3x3 grid around center)."""
    x, y = cell
    return [
        (x - 1, y - 1),
        (x, y - 1),
        (x + 1, y - 1),
        (x - 1, y),
        (x, y),
        (x + 1, y),
        (x - 1, y + 1),
        (x, y + 1),
        (x + 1, y + 1),
    ]


class GracefulKiller:
    """Handles graceful shutdown with checkpoint saving."""

    def __init__(self):
        self.kill_now = False
        self.persistence_manager = None
        self.current_operation = "unknown"
        self.checkpoint_data = {}
        self._lock = threading.Lock()

        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
        atexit.register(self._cleanup)

    def set_persistence_manager(self, persistence_manager):
        with self._lock:
            self.persistence_manager = persistence_manager

    def set_operation(self, operation: str, checkpoint_data: Dict = None):
        with self._lock:
            self.current_operation = operation
            if checkpoint_data:
                self.checkpoint_data = checkpoint_data.copy()

    def _exit_gracefully(self, signum, frame):
        with self._lock:
            if self.kill_now:
                print("\nForced shutdown. Exiting immediately...")
                os._exit(1)  # Force exit without cleanup

            self.kill_now = True
            print(f"\nGraceful shutdown during: {self.current_operation}")

            if self.persistence_manager and self.checkpoint_data:
                try:
                    print("Saving emergency checkpoint...")
                    self._save_emergency_checkpoint()
                    print("Emergency checkpoint saved. Analysis can be resumed.")
                except Exception as e:
                    print(f"Failed to save emergency checkpoint: {e}")

            os._exit(0)

    def _save_emergency_checkpoint(self):
        # ACTIVE CHECKPOINT OPERATIONS (for process_region_without_chunking flow)
        if self.current_operation == "region_elevation_fetching":
            # Handle emergency checkpoint for region elevation fetching
            node_elevations = self.checkpoint_data.get("node_elevations", {})
            checkpoint_info = {
                "batch_idx": self.checkpoint_data.get("batch_idx", 0),
                "coords_seen": self.checkpoint_data.get("coords_seen", []),
                "elevations_fetched": self.checkpoint_data.get("elevations_fetched", 0),
                "elevations_failed": self.checkpoint_data.get("elevations_failed", 0),
            }
            self.persistence_manager.save_elevation_progress(node_elevations, checkpoint_info)
        elif self.current_operation == "filtering_roads":
            # Handle emergency checkpoint for road filtering
            processed = self.checkpoint_data.get("processed_count", 0)
            matched = self.checkpoint_data.get("matched_count", 0)
            print(f"  Road filtering interrupted: {processed:,} processed, {matched:,} matched")
            print("  ✓ Partial progress saved - will resume from checkpoint on next run")
        elif self.current_operation == "way_to_segment_conversion":
            # Handle emergency checkpoint for way to segment conversion
            # Note: Partial progress is saved to segments.checkpoint.jsonl file
            # On next run, the file will be overwritten (restart from beginning)
            print(
                f"  Partial segment conversion completed: {self.checkpoint_data.get('segment_count', 0)} segments"
            )
            print("  ⚠️  Note: Conversion will restart from beginning on next run")
        elif self.current_operation == "external_sort_phase1":
            # Handle emergency checkpoint for external sort phase 1
            chunks_created = self.checkpoint_data.get("chunks_created", 0)
            segments_processed = self.checkpoint_data.get("segments_processed", 0)
            print(
                f"  Sort Phase 1 interrupted: {chunks_created} chunks created ({segments_processed:,} segments)"
            )
            print("  ⚠️  Partial chunks will be cleaned up - sort will restart on next run")
        elif self.current_operation == "external_sort_phase2":
            # Handle emergency checkpoint for external sort phase 2
            merged_count = self.checkpoint_data.get("merged_count", 0)
            print(f"  Sort Phase 2 interrupted: {merged_count:,} segments merged")
            print("  ⚠️  Partial sorted file created - sort will restart on next run")
        elif self.current_operation == "segment_merging":
            # Handle emergency checkpoint for segment merging
            processed = self.checkpoint_data.get("processed", 0)
            merged = self.checkpoint_data.get("merged", 0)
            print(f"  Segment merging interrupted: {processed:,} processed → {merged:,} merged")
            print("  ⚠️  Partial merged file created - merge will restart on next run")
        # LEGACY OPERATIONS REMOVED:
        # - "chunk_processing" (legacy chunked processing)
        # - "elevation_fetching" (old parallel chunk elevation format)
        # - "deduplication" (not used in regional extraction flow)
        # - "boundary_merge" (not needed without chunking)

    def _cleanup(self):
        pass


signal_handler = GracefulKiller()


class LocationIndex:
    """Persistent location index for OSM nodes"""

    def __init__(self, osm_file_path: str, index_dir: Optional[str] = None):
        self.osm_file_path = osm_file_path
        self.index_dir = Path(index_dir) if index_dir else OSM_INDEXES_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / f"{Path(osm_file_path).stem}_locations.idx"

    def create_index(self):
        """Create node location index from OSM file"""
        print("Creating location index (this may take several minutes for large files)...")

        # Use osmium's node location storage
        # This creates a memory-mapped file for fast access
        handler = osmium.SimpleHandler()

        # Apply with location storage
        handler.apply_file(
            self.osm_file_path,
            locations=True,
            idx="sparse_file_array," + str(self.index_file),
        )

        print(f"Location index created: {self.index_file}")

    def exists(self):
        """Check if index exists"""
        return self.index_file.exists()


class SimpleWay:
    """Lightweight way object."""

    def __init__(self):
        self.id = None
        self.tags = {}
        self.nodes = []


class SimpleNode:
    """Lightweight node object."""

    def __init__(self, node_data):
        self.id = node_data.get("id", 0)
        self.lat = float(node_data.get("lat", 0.0))
        self.lon = float(node_data.get("lon", 0.0))


class ChunkedRoadNetworkAnalyzer:
    """
    Road network analyzer that processes data in geographic chunks.

    Chunk Size Optimization:
    - Default 30km chunks provide optimal performance (benchmarked on Georgia dataset)
    - Memory usage is constant (~4.6GB for spatial index) regardless of chunk size
    - Larger chunks reduce overhead: 30km=37 chunks vs 5km=1,253 chunks (33x faster)
    - Performance scales with number of chunks, not chunk size
    - Use benchmark_comprehensive.py to optimize for your specific dataset

    Memory is dominated by spatial index loading, not chunk processing.
    """

    def __init__(
        self,
        surface_filter: str = "all",
        chunk_size_km: float = 30.0,  # 30km minimizes overhead while keeping memory constant (~4.6GB)
        cycling_only: bool = True,
        osm_file_path: str = None,
        silent_loading: bool = False,  # Suppress progress messages (for multiprocessing workers)
        progress_counter=None,  # Shared counter for tracking loading progress across workers
    ):
        self.geolocator = Nominatim(user_agent="climb_analyzer")
        self.surface_filter = surface_filter
        # Parse surface filter into list - can be "all" or comma-separated like "paved,gravel"
        if surface_filter == "all":
            self.surface_filters = ["all"]
        else:
            self.surface_filters = [s.strip() for s in surface_filter.split(",")]
        self.chunk_size_km = chunk_size_km
        self.cycling_only = cycling_only
        self.osm_file_path = osm_file_path  # Store the osm_file_path
        self.deployment_type = DEPLOYMENT_TYPE
        self.silent_loading = silent_loading  # Store for use in _initialize_deployment_type
        self.progress_counter = progress_counter  # Store for passing to SpatialIndexManager

        # Initialize deployment-specific components
        self._initialize_deployment_type()

    def _convert_spatial_way_to_simple_way(self, way_data):
        """Convert spatial index way data to SimpleWay object"""
        try:
            way = SimpleWay()
            way.id = way_data["id"]
            way.tags = way_data["tags"]
            way.nodes = []

            # Convert coordinate tuples to SimpleNode objects
            for i, node_tuple in enumerate(way_data["nodes"]):
                # Handle both old format (lat, lon) and new format (lat, lon, node_id)
                if len(node_tuple) == 3:
                    lat, lon, node_id = node_tuple
                else:
                    # Fallback for old format indexes (backward compatibility)
                    lat, lon = node_tuple
                    node_id = f"{way.id}_{i}"

                node_data = {
                    "id": node_id,
                    "lat": lat,
                    "lon": lon,
                }
                way.nodes.append(SimpleNode(node_data))

            return way

        except Exception as e:
            print(f"Error converting spatial way {way_data.get('id', '?')}: {e}")
            return None

    def _way_matches_filters(self, way):
        """Check if way matches surface and cycling filters"""
        try:
            if not hasattr(way, "tags") or not way.tags:
                return False

            highway_type = way.tags.get("highway", "").lower()
            surface = way.tags.get("surface", "")

            # ALWAYS exclude motorways unless explicitly allowed for cycling/pedestrians
            # This runs regardless of surface filter or cycling filter settings
            if highway_type in ["motorway", "motorway_link"]:
                bicycle = way.tags.get("bicycle", "").lower()
                foot = way.tags.get("foot", "").lower()
                # Only allow if explicitly marked for cycling or pedestrian access
                allowed_values = ["yes", "designated", "permissive", "use_sidepath"]
                if bicycle not in allowed_values and foot not in allowed_values:
                    return False

            # Surface filter logic - check if way matches ANY of the selected surface types
            if "all" not in self.surface_filters:
                matches_any_surface = False

                for surface_type in self.surface_filters:
                    if surface_type == "paved":
                        # Paved filter logic
                        if highway_type in [
                            "trunk",
                            "primary",
                            "secondary",
                            "tertiary",
                            "unclassified",
                            "residential",
                            "service",
                        ]:
                            matches_any_surface = True
                            break
                        elif surface not in ["unpaved", "gravel", "dirt", "sand", "grass"]:
                            matches_any_surface = True
                            break

                    elif surface_type == "gravel":
                        # Gravel filter logic
                        if highway_type in ["track", "path", "unclassified", "tertiary"]:
                            if surface in ["gravel", "compacted", "fine_gravel", ""]:
                                matches_any_surface = True
                                break
                        elif surface in ["gravel", "compacted", "fine_gravel"]:
                            matches_any_surface = True
                            break

                    elif surface_type == "dirt":
                        # Dirt filter logic
                        if highway_type in ["track", "path", "footway", "bridleway"]:
                            matches_any_surface = True
                            break
                        elif "tracktype" in way.tags:
                            matches_any_surface = True
                            break

                if not matches_any_surface:
                    return False

            # Cycling filter
            if self.cycling_only:
                bicycle = way.tags.get("bicycle", "").lower()
                access = way.tags.get("access", "").lower()

                if bicycle == "no" or access in ["no", "private"]:
                    return False

                # For footways, require explicit cycling permission
                if highway_type == "footway":
                    if bicycle not in ["yes", "designated", "permissive"]:
                        return False

            return True

        except Exception as e:
            print(f"Error checking filters for way {getattr(way, 'id', '?')}: {e}")
            return False

    def get_roads_in_chunk_osmium(
        self, chunk_lat: float, chunk_lon: float, chunk_radius: float, debug: bool = False
    ) -> List:
        """Get road network data using spatial index for maximum speed."""

        # Calculate bounding box
        lat_offset = chunk_radius / 111.0
        lon_offset = chunk_radius / (111.0 * math.cos(math.radians(chunk_lat)))

        min_lat = chunk_lat - lat_offset
        max_lat = chunk_lat + lat_offset
        min_lon = chunk_lon - lon_offset
        max_lon = chunk_lon + lon_offset

        # Debug output only when requested
        if debug:
            print(f"Chunk center: ({chunk_lat:.6f}, {chunk_lon:.6f}), radius: {chunk_radius:.2f}km")
            print(f"Bbox: lat {min_lat:.6f} to {max_lat:.6f}, lon {min_lon:.6f} to {max_lon:.6f}")

        # Use spatial index to query ways in bounding box
        try:
            ways_data = self.spatial_index_manager.query_bbox(min_lat, min_lon, max_lat, max_lon)

            if debug:
                print(f"Spatial index returned {len(ways_data)} ways for this chunk")
                if len(ways_data) > 0:
                    print(f"Sample way IDs: {[w['id'] for w in ways_data[:3]]}")

            # Convert to SimpleWay objects and filter
            ways = []

            for way_data in ways_data:
                way = self._convert_spatial_way_to_simple_way(way_data)

                if way:
                    if self._way_matches_filters(way):
                        ways.append(way)
                    elif debug:
                        print(
                            f"  Way {way.id} failed filters: highway={way.tags.get('highway', 'none')}"
                        )
                elif debug:
                    print(f"  Failed to convert way {way_data.get('id', '?')}")

            if debug:
                print(f"Final result: {len(ways)} ways for this chunk")

            return ways

        except Exception as e:
            print(f"Error using spatial index: {e}")
            traceback.print_exc()
            return []

    def get_all_roads_in_region(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        persistence_manager=None,
        use_streaming: bool = False,
    ) -> List:
        """
        Extract ALL roads from region at once (deployment-aware).

        This method routes to the appropriate implementation based on deployment type:
        - Local mode: Uses spatial index to query OSM file
        - Cloud mode: Uses Overpass API bbox query

        Both approaches are more efficient than chunking for typical address searches.

        Args:
            min_lat: Minimum latitude of bounding box
            min_lon: Minimum longitude of bounding box
            max_lat: Maximum latitude of bounding box
            max_lon: Maximum longitude of bounding box
            persistence_manager: Optional ChunkPersistenceManager for streaming mode
            use_streaming: If True, write ways to checkpoint file instead of returning list

        Returns:
            List of SimpleWay objects matching filters (or empty list if streaming mode)
        """
        # Route to appropriate implementation based on deployment type
        if self.deployment_type == "cloud":
            return self.get_all_roads_in_region_api(
                min_lat, min_lon, max_lat, max_lon, persistence_manager, use_streaming
            )

        # Local mode implementation
        from climb_analyzer.utils.formatting import print_info

        print()  # Spacing
        print_info(
            f"Bounding box: lat [{min_lat:.4f}, {max_lat:.4f}], lon [{min_lon:.4f}, {max_lon:.4f}]"
        )

        # Use spatial index to query ways in batches for memory efficiency
        try:
            batch_size = 50000

            # Check if spatial index manager is available
            if not self.spatial_index_manager:
                print("✗ Spatial index manager not initialized")
                print("  This usually means the OSM index was not built or found")
                return []

            # Force index loading before progress bar to avoid output overlap
            # This ensures "Loading ways" completes before "Processing" starts
            if self.spatial_index_manager.spatial_idx is None:
                self.spatial_index_manager.load_index(silent=False)

            # Use the batched query generator to process in chunks
            # This avoids loading all 4M+ ways into memory at once
            way_batches = self.spatial_index_manager.query_bbox_batched(
                min_lat, min_lon, max_lat, max_lon, batch_size=batch_size
            )

            # STREAMING MODE: Write filtered ways to checkpoint file
            if use_streaming and persistence_manager:
                checkpoint_file = persistence_manager.init_filtered_ways_checkpoint()

                processed_count = 0
                matched_count = 0

                print("\nFiltering and streaming roads to checkpoint...")
                # Wrap the batches iterator with tqdm to show progress
                # We don't know the total count upfront, so use dynamic progress
                batch_num = 0
                with tqdm(
                    desc="Processing",
                    unit=" ways",
                    bar_format="{desc}: {n_fmt} ways ({rate_fmt}) | {elapsed}",
                    ascii=" ▏▎▍▌▋▊▉█",
                    dynamic_ncols=True,
                    mininterval=0.5,
                ) as pbar:

                    for batch in way_batches:
                        filtered_batch = []

                        # Filter the batch
                        for way_data in batch:
                            way = self._convert_spatial_way_to_simple_way(way_data)
                            if way and self._way_matches_filters(way):
                                filtered_batch.append(way)

                        # Write filtered batch to checkpoint
                        if filtered_batch:
                            persistence_manager.append_filtered_ways_batch(
                                filtered_batch, checkpoint_file
                            )
                            matched_count += len(filtered_batch)

                        processed_count += len(batch)
                        pbar.update(len(batch))
                        pbar.set_postfix({"matched": f"{matched_count:,}"})
                        batch_num += 1

                        # Explicitly free batch memory to prevent accumulation
                        del filtered_batch
                        del batch

                        # Force garbage collection every 5 batches to prevent OOM
                        if batch_num % 5 == 0:
                            import gc

                            gc.collect()

                        # Check for graceful shutdown every 10 batches
                        if batch_num % 10 == 0:
                            signal_handler.set_operation(
                                "filtering_roads",
                                {
                                    "processed_count": processed_count,
                                    "matched_count": matched_count,
                                },
                            )

                            if signal_handler.kill_now:
                                print("\n\n🛑 Graceful shutdown during road filtering")
                                print(
                                    f"  Processed {processed_count:,} ways, matched {matched_count:,}"
                                )
                                print("  ✓ Partial progress saved to checkpoint")
                                print(
                                    "\n⏸️  Run again to continue from checkpoint (existing data will be kept)."
                                )
                                import sys

                                sys.exit(0)

                # Progress bar will auto-close when exiting the with block
                print(
                    f"\n✓ Streamed {matched_count:,} filtered roads to checkpoint from {processed_count:,} ways"
                )
                return []  # Return empty list in streaming mode

            # NORMAL MODE: Accumulate ways in memory (for small regions)
            else:
                ways = []
                processed_count = 0

                for batch in way_batches:
                    # Process each way in the batch
                    for way_data in batch:
                        way = self._convert_spatial_way_to_simple_way(way_data)
                        if way and self._way_matches_filters(way):
                            ways.append(way)

                    processed_count += len(batch)

                print(
                    f"✓ Extracted {len(ways):,} roads matching filters from {processed_count:,} ways"
                )
                return ways

        except Exception as e:
            print(f"Error extracting all ways from region: {e}")
            import traceback

            traceback.print_exc()
            return []

    def get_all_roads_in_region_api(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        persistence_manager=None,
        use_streaming: bool = False,
    ) -> List:
        """
        Extract ALL roads from region using Overpass API bbox query (cloud mode).

        Uses a single bbox query instead of chunk-by-chunk processing.
        Similar to local mode's get_all_roads_in_region() but uses Overpass API.

        Args:
            min_lat: Minimum latitude of bounding box
            min_lon: Minimum longitude of bounding box
            max_lat: Maximum latitude of bounding box
            max_lon: Maximum longitude of bounding box
            persistence_manager: Optional ChunkPersistenceManager for streaming mode
            use_streaming: If True, write filtered ways to checkpoint file

        Returns:
            List of SimpleWay objects matching filters (or empty list if streaming mode)
        """
        from climb_analyzer.utils.formatting import print_info

        print()  # Spacing
        print_info(
            f"Bounding box: lat [{min_lat:.4f}, {max_lat:.4f}], lon [{min_lon:.4f}, {max_lon:.4f}]"
        )
        print_info("Using single bbox query to Overpass API...")

        max_retries = 5
        base_delay = 15.0

        for attempt in range(max_retries):
            try:
                # Build Overpass query with bbox filter instead of around filter
                query = self._build_bbox_overpass_query(
                    min_lat, min_lon, max_lat, max_lon, self.cycling_only
                )

                api_url = (
                    OVERPASS_API_URL
                    if OVERPASS_API_URL
                    else "https://overpass-api.de/api/interpreter"
                )

                print("Sending bbox query to Overpass API (timeout: 240s)...")
                response = requests.post(
                    api_url,
                    data=query,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=240,  # Cloud mode: longer timeout for regional extraction (Overpass can be slow)
                )

                # Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        backoff_delay = base_delay * (3**attempt)
                        print(f"⚠️  Rate limited (HTTP 429), retrying in {backoff_delay:.0f}s...")
                        time.sleep(backoff_delay)
                        continue
                    else:
                        print(f"⚠️  HTTP 429 (rate limited) persisted after {max_retries} attempts")
                        return []

                # Handle 504 Gateway Timeout with retry
                if response.status_code == 504:
                    if attempt < max_retries - 1:
                        backoff_delay = base_delay * (2**attempt)
                        print(f"⚠️  Gateway timeout (HTTP 504), retrying in {backoff_delay:.0f}s...")
                        time.sleep(backoff_delay)
                        continue
                    else:
                        print(
                            f"⚠️  HTTP 504 (gateway timeout) persisted after {max_retries} attempts"
                        )
                        return []

                # Handle other HTTP errors
                if response.status_code != 200:
                    print(f"HTTP Error {response.status_code} from Overpass API")
                    return []

                # Success - process response
                response_text = response.text
                json_start = response_text.find("{")
                if json_start == -1:
                    print("⚠️  Invalid response from Overpass API (no JSON found)")
                    return []

                json_text = response_text[json_start:]

                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    return []

                # Process ways with memory efficiency
                ways = []

                # Build node lookup only for nodes we need
                all_nodes = {}
                for element in data.get("elements", []):
                    if element.get("type") == "node":
                        all_nodes[element["id"]] = element

                print(f"Received {len(data.get('elements', []))} elements from API")

                # Create and filter ways
                from tqdm import tqdm

                way_elements = [e for e in data.get("elements", []) if e.get("type") == "way"]
                print(f"Processing {len(way_elements):,} ways...")

                # STREAMING MODE: Write filtered ways to checkpoint file
                if use_streaming and persistence_manager:
                    checkpoint_file = persistence_manager.init_filtered_ways_checkpoint()
                    filtered_ways = []

                    for element in tqdm(
                        way_elements,
                        desc="Converting ways",
                        unit="ways",
                        miniters=len(way_elements) // 100 if way_elements else 1,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        ascii=" ▏▎▍▌▋▊▉█",
                        dynamic_ncols=True,
                    ):
                        way = self._create_simple_way(element, all_nodes)
                        # Filter by surface type and other criteria
                        if way and len(way.nodes) > 1 and self._way_matches_filters(way):
                            filtered_ways.append(way)

                    # Write filtered ways to checkpoint
                    if filtered_ways:
                        persistence_manager.append_filtered_ways_batch(
                            filtered_ways, checkpoint_file
                        )

                    # Clean up memory
                    del data, all_nodes, filtered_ways
                    gc.collect()

                    print(f"✓ Extracted {len(way_elements):,} roads from region via Overpass API")
                    return []  # Return empty list in streaming mode

                # NON-STREAMING MODE: Return list of filtered ways
                else:
                    for element in tqdm(
                        way_elements,
                        desc="Converting ways",
                        unit="ways",
                        miniters=len(way_elements) // 100 if way_elements else 1,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        ascii=" ▏▎▍▌▋▊▉█",
                        dynamic_ncols=True,
                    ):
                        way = self._create_simple_way(element, all_nodes)
                        # Filter by surface type and other criteria
                        if way and len(way.nodes) > 1 and self._way_matches_filters(way):
                            ways.append(way)

                    # Clean up memory
                    del data, all_nodes
                    gc.collect()

                    print(f"✓ Extracted {len(ways):,} roads from region via Overpass API")
                    return ways

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    backoff_delay = base_delay * (2**attempt)
                    print(f"⚠️  Request timeout, retrying in {backoff_delay:.0f}s...")
                    time.sleep(backoff_delay)
                    continue
                else:
                    print(f"⚠️  Request timeout after {max_retries} attempts")
                    return []

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    backoff_delay = base_delay * (2**attempt)
                    print(f"⚠️  Connection error, retrying in {backoff_delay:.0f}s...")
                    time.sleep(backoff_delay)
                    continue
                else:
                    print(f"⚠️  Connection error after {max_retries} attempts: {type(e).__name__}")
                    return []

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Unexpected error, retrying: {e}")
                    time.sleep(base_delay)
                    continue
                else:
                    print(f"Error extracting ways from Overpass API: {e}")
                    import traceback

                    traceback.print_exc()
                    return []

        # Should not reach here, but return empty list as fallback
        return []

    def _build_bbox_overpass_query(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        cycling_only: bool = False,
    ) -> str:
        """Build Overpass query with bbox filter instead of around filter."""

        # Bbox in query settings applies to all way selectors
        # Format: [bbox:south,west,north,east]
        bbox_filter = f"[bbox:{min_lat},{min_lon},{max_lat},{max_lon}]"

        # Get base surface filter
        surface_filters = ROAD_SURFACE_FILTERS.get(self.surface_filter, ROAD_SURFACE_FILTERS["all"])

        if cycling_only:
            # Modify highway filter for cycling restrictions
            if self.surface_filter == "paved":
                highway_query = """
                way[highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|cycleway"]
                    [access!~"no|private"][bicycle!~"no|private"];
                way[highway~"path|bridleway"]
                    [access!~"no|private"][bicycle!~"no|private"];
                way[highway~"footway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"];
                """
            elif self.surface_filter == "gravel":
                highway_query = """
                way[highway~"track|unclassified|tertiary|residential|service|cycleway"]
                    [access!~"no|private"][bicycle!~"no|private"];
                way[highway~"path|bridleway"]
                    [access!~"no|private"][bicycle!~"no|private"];
                way[highway~"footway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"];
                """
            elif self.surface_filter == "dirt":
                highway_query = """
                way[highway~"track"]
                    [access!~"no|private"][bicycle!~"no|private"];
                way[highway~"path|footway|cycleway|bridleway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"];
                """
            else:  # 'all'
                highway_query = """
                way[highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|track|cycleway"]
                    [access!~"no|private"][bicycle!~"no|private"];
                way[highway~"path|bridleway"]
                    [access!~"no|private"][bicycle!~"no|private"];
                way[highway~"footway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"];
                """

            # Add surface filter if it exists
            if len(surface_filters) > 1:
                surface_condition = f"[{surface_filters[1]}]"
                # Apply surface filter to each way query line (before the semicolon)
                lines = highway_query.split("\n")
                filtered_lines = []
                for line in lines:
                    if "way[highway" in line:
                        # Insert surface condition before the semicolon
                        line = line.replace(";", f"{surface_condition};")
                    filtered_lines.append(line)
                highway_query = "\n".join(filtered_lines)

            query = f"""
            [out:json][timeout:180][maxsize:1073741824]{bbox_filter};
            (
              {highway_query}
            );
            (._;>;);
            out geom;
            """
        else:
            # Use original surface filters without cycling restrictions
            if len(surface_filters) == 1:
                # No surface restrictions, just highway type
                query = f"""
                [out:json][timeout:180][maxsize:1073741824]{bbox_filter};
                (
                  way[{surface_filters[0]}];
                );
                (._;>;);
                out geom;
                """
            else:
                # Highway type + surface restrictions
                query = f"""
                [out:json][timeout:180][maxsize:1073741824]{bbox_filter};
                (
                  way[{surface_filters[0]}]
                      [{surface_filters[1]}];
                );
                (._;>;);
                out geom;
                """

        return query

    def _convert_spatial_way_to_simple_way(self, way_data: dict) -> SimpleWay:
        """Convert spatial index way data to SimpleWay object."""
        way = SimpleWay()
        way.id = way_data["id"]
        way.tags = way_data["tags"]

        # Convert node coordinates to SimpleNode objects
        way.nodes = []
        for i, node_tuple in enumerate(way_data["nodes"]):
            # Handle both old format (lat, lon) and new format (lat, lon, node_id)
            if len(node_tuple) == 3:
                lat, lon, node_id = node_tuple
            else:
                # Fallback for old format indexes (backward compatibility)
                lat, lon = node_tuple
                node_id = f"{way.id}_{i}"

            node_data = {"id": node_id, "lat": lat, "lon": lon}
            way.nodes.append(SimpleNode(node_data))

        return way

    def _way_matches_filters(self, way):
        """Check if way matches surface and cycling filters"""
        try:
            if not hasattr(way, "tags") or not way.tags:
                # print(f"  Way {getattr(way, 'id', '?')} has no tags")
                return False

            highway_type = way.tags.get("highway", "").lower()
            if not highway_type:
                # print(f"  Way {way.id} has no highway tag")
                return False

            # ALWAYS exclude motorways unless explicitly allowed for cycling/pedestrians
            # This runs regardless of surface filter or cycling filter settings
            if highway_type in ["motorway", "motorway_link"]:
                bicycle = way.tags.get("bicycle", "").lower()
                foot = way.tags.get("foot", "").lower()
                # Only allow if explicitly marked for cycling or pedestrian access
                allowed_values = ["yes", "designated", "permissive", "use_sidepath"]
                if bicycle not in allowed_values and foot not in allowed_values:
                    return False

            # Surface filter logic
            if self.surface_filter == "paved":
                if highway_type not in [
                    "trunk",
                    "primary",
                    "secondary",
                    "tertiary",
                    "unclassified",
                    "residential",
                    "service",
                ]:
                    surface = way.tags.get("surface", "")
                    if surface in ["unpaved", "gravel", "dirt", "sand", "grass"]:
                        # print(f"  Way {way.id} failed paved filter: highway={highway_type}, surface={surface}")
                        return False

            # Cycling filter
            if self.cycling_only:
                bicycle = way.tags.get("bicycle", "").lower()
                access = way.tags.get("access", "").lower()

                if bicycle == "no" or access in ["no", "private"]:
                    # print(f"  Way {way.id} failed cycling filter: bicycle={bicycle}, access={access}")
                    return False

                # For footways, require explicit cycling permission
                if highway_type == "footway":
                    if bicycle not in ["yes", "designated", "permissive"]:
                        # print(f"  Way {way.id} footway failed cycling filter: bicycle={bicycle}")
                        return False

            return True

        except Exception as e:
            print(f"Error checking filters for way {getattr(way, 'id', '?')}: {e}")
            return False

    def _fallback_to_osmium(self, chunk_lat: float, chunk_lon: float, chunk_radius: float) -> List:
        """Fallback to osmium processing if spatial index fails."""
        bbox = self._calculate_bbox(chunk_lat, chunk_lon, chunk_radius)
        handler = RoadWayHandler(bbox, self.surface_filter, self.cycling_only)
        handler.apply_file(self.osm_file_path, locations=True, idx="flex_mem")
        return handler.ways

    def _initialize_deployment_type(self):
        """Initialize deployment type and related components."""
        if self.deployment_type == "local":
            # Local raw file deployment - USE SPATIAL INDEX AND API ELEVATION
            planet_dir = Path("./data/planet_osm_data")

            # Only auto-detect OSM file if not explicitly provided
            if not self.osm_file_path:
                # Try config.yaml first
                try:
                    import yaml

                    with open("config.yaml") as f:
                        config = yaml.safe_load(f)
                    configured_path = config.get("PLANET_FILE_PATH", "")
                    if configured_path and Path(configured_path).exists():
                        self.osm_file_path = configured_path
                except:
                    pass

                # If still no OSM file path found after config check, error out
                # NEVER auto-detect a random file - this causes wrong region selection
                if not self.osm_file_path:
                    raise FileNotFoundError(
                        "No OSM file found for this region. "
                        "Please download the correct OSM file for your region using:\n"
                        "  python climb_analyzer_main.py download-osm --region <region_name>\n"
                        "Or check that the region name matches an existing .osm.pbf file in data/planet_osm_data/"
                    )

            # Initialize spatial index manager
            self.spatial_index_manager = SpatialIndexManager(
                self.osm_file_path,
                cache_dir=str(OSM_INDEXES_DIR),
                silent=self.silent_loading,
                progress_counter=self.progress_counter,
            )

            # Build index if it doesn't exist
            if not self.spatial_index_manager.exists():
                print(f"Building spatial index for {Path(self.osm_file_path).name}...")
                print("This is a one-time process and may take several minutes.")
                from climb_analyzer.data.index_builder import build_spatial_index

                success = build_spatial_index(
                    self.osm_file_path,
                    output_dir=str(OSM_INDEXES_DIR),
                    surface_filter="all",
                    cycling_only=False,
                )
                if not success:
                    raise FileNotFoundError(
                        f"Failed to build spatial index for {Path(self.osm_file_path).name}"
                    )

        else:  # cloud deployment
            # Cloud deployment uses API calls, no local files needed
            if OVERPASS_API_URL:
                self.overpass_api = overpy.Overpass(url=OVERPASS_API_URL)
            else:
                raise ValueError("No Overpass API URL configured for cloud deployment")

    def get_roads_in_chunk(self, chunk_lat: float, chunk_lon: float, chunk_radius: float) -> List:
        """Get road network data for a single chunk - deployment type aware."""

        if self.deployment_type == "local":
            # Use local OSM file parsing
            return self.get_roads_in_chunk_osmium(chunk_lat, chunk_lon, chunk_radius)
        else:
            # Use API-based approach (cloud or local API)
            return self.get_roads_in_chunk_w_API(chunk_lat, chunk_lon, chunk_radius)

    def get_roads_in_chunk_osmium(
        self, chunk_lat: float, chunk_lon: float, chunk_radius: float
    ) -> List:
        """Get road network data using spatial index for maximum speed."""

        # Calculate bounding box
        lat_offset = chunk_radius / 111.0
        lon_offset = chunk_radius / (111.0 * math.cos(math.radians(chunk_lat)))

        min_lat = chunk_lat - lat_offset
        max_lat = chunk_lat + lat_offset
        min_lon = chunk_lon - lon_offset
        max_lon = chunk_lon + lon_offset

        # Use spatial index to query ways in bounding box
        try:
            ways_data = self.spatial_index_manager.query_bbox(min_lat, min_lon, max_lat, max_lon)

            # Convert to SimpleWay objects
            ways = []
            for way_data in ways_data:
                way = self._convert_spatial_way_to_simple_way(way_data)
                if way and self._way_matches_filters(way):
                    ways.append(way)

            return ways

        except Exception as e:
            print(f"Error using spatial index: {e}")
            # Fallback to osmium if spatial index fails
            return self._fallback_to_osmium(chunk_lat, chunk_lon, chunk_radius)

    def _calculate_bbox(self, lat: float, lon: float, radius_km: float) -> tuple:
        """Calculate bounding box from center point and radius."""
        # Approximate conversion (1 degree ≈ 111km at equator)
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * math.cos(math.radians(lat)))

        return (
            lat - lat_offset,  # min_lat
            lon - lon_offset,  # min_lon
            lat + lat_offset,  # max_lat
            lon + lon_offset,  # max_lon
        )

    def get_roads_in_chunk_w_API(
        self, chunk_lat: float, chunk_lon: float, chunk_radius: float
    ) -> List:
        """Get road network data for a single chunk from Overpass API with retry logic."""
        max_retries = 5
        base_delay = 15.0  # Start with 15 seconds for first retry

        for attempt in range(max_retries):
            try:
                query = self.build_overpass_query(
                    chunk_lat, chunk_lon, chunk_radius, self.cycling_only
                )

                api_url = (
                    OVERPASS_API_URL
                    if OVERPASS_API_URL
                    else "https://overpass-api.de/api/interpreter"
                )

                response = requests.post(
                    api_url,  # Changed from local_api_url
                    data=query,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=120,  # Cloud mode: query has 60s timeout, add 60s buffer for network/processing
                )

                # Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        backoff_delay = base_delay * (
                            3**attempt
                        )  # Conservative backoff: 15s, 45s, 135s, 405s (~7min)
                        time.sleep(backoff_delay)
                        continue
                    else:
                        print(
                            f"⚠️  HTTP 429 (rate limited) persisted after {max_retries} attempts for chunk ({chunk_lat:.4f}, {chunk_lon:.4f})"
                        )
                        return []

                # Handle 504 Gateway Timeout with retry
                if response.status_code == 504:
                    if attempt < max_retries - 1:
                        backoff_delay = base_delay * (
                            2**attempt
                        )  # Moderate backoff for 504: 15s, 30s, 60s, 120s
                        # Silently retry - only print if all retries fail
                        time.sleep(backoff_delay)
                        continue
                    else:
                        print(
                            f"⚠️  HTTP 504 (gateway timeout) persisted after {max_retries} attempts for chunk ({chunk_lat:.4f}, {chunk_lon:.4f})"
                        )
                        return []

                # Handle other HTTP errors (print immediately as these are unexpected)
                if response.status_code != 200:
                    print(
                        f"HTTP Error {response.status_code} for chunk ({chunk_lat:.4f}, {chunk_lon:.4f})"
                    )
                    # Don't retry other error codes
                    return []

                # Success - process response
                response_text = response.text
                json_start = response_text.find("{")
                if json_start == -1:
                    return []

                json_text = response_text[json_start:]

                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed for chunk: {e}")
                    return []

                # Process ways with memory efficiency
                ways = []

                # Build node lookup only for nodes we need
                all_nodes = {}
                for element in data.get("elements", []):
                    if element.get("type") == "node":
                        all_nodes[element["id"]] = element

                # Create ways
                for element in data.get("elements", []):
                    if element.get("type") == "way":
                        way = self._create_simple_way(element, all_nodes)
                        if way and len(way.nodes) > 1:
                            ways.append(way)

                # Clean up memory
                del data, all_nodes
                gc.collect()

                return ways

            except requests.exceptions.Timeout:
                # Network timeout (different from HTTP 504)
                if attempt < max_retries - 1:
                    backoff_delay = base_delay * (2**attempt)
                    # Silently retry - only print on final failure
                    time.sleep(backoff_delay)
                    continue
                else:
                    print(
                        f"⚠️  Request timeout after {max_retries} attempts for chunk ({chunk_lat:.4f}, {chunk_lon:.4f})"
                    )
                    return []

            except requests.exceptions.RequestException as e:
                # Other network errors (connection reset, etc)
                if attempt < max_retries - 1:
                    backoff_delay = base_delay * (2**attempt)
                    # Silently retry - only print on final failure
                    time.sleep(backoff_delay)
                    continue
                else:
                    print(
                        f"⚠️  Connection error after {max_retries} attempts for chunk ({chunk_lat:.4f}, {chunk_lon:.4f}): {type(e).__name__}"
                    )
                    return []

            except Exception as e:
                # Unexpected errors (JSON parsing, etc)
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                    continue
                else:
                    print(f"Error processing chunk ({chunk_lat:.4f}, {chunk_lon:.4f}): {e}")
                    return []

        # Should not reach here, but return empty list as fallback
        return []

    def get_coordinates_from_address(
        self, address: str, country: str = None
    ) -> Tuple[float, float, str]:
        """Convert street address to coordinates with fallback strategies"""
        try:
            if country:
                search_query = f"{address}, {country}"
            else:
                search_query = address

            print(f"Geocoding address: {search_query}")
            location = self.geolocator.geocode(search_query, timeout=10)

            if location:
                formatted_address = location.address
                print(f"Found location: {formatted_address}")
                print(f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}")
                return location.latitude, location.longitude, formatted_address

            # Fallback strategies if full address fails
            print("Full address not found. Trying fallback strategies...")

            # Strategy 1: Try without house number
            address_parts = address.split(",")
            if len(address_parts) > 1:
                # Remove the first part (likely house number + street)
                street_part = address_parts[0].strip()
                # Try to extract just the street name (remove house number)
                street_words = street_part.split()
                if len(street_words) > 1 and street_words[0].isdigit():
                    street_name = " ".join(street_words[1:])
                    fallback_address = ", ".join([street_name] + address_parts[1:])
                    if country:
                        fallback_query = f"{fallback_address}, {country}"
                    else:
                        fallback_query = fallback_address

                    print(f"Trying street name only: {fallback_query}")
                    location = self.geolocator.geocode(fallback_query, timeout=10)
                    if location:
                        print(f"Found using street name: {location.address}")
                        print(f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}")
                        return location.latitude, location.longitude, location.address

            # Strategy 2: Try just city, state, zip
            if len(address_parts) >= 2:
                city_state_zip = ", ".join(address_parts[1:]).strip()
                if country:
                    fallback_query = f"{city_state_zip}, {country}"
                else:
                    fallback_query = city_state_zip

                print(f"Trying city/state/zip: {fallback_query}")
                location = self.geolocator.geocode(fallback_query, timeout=10)
                if location:
                    print(f"Found using city/state: {location.address}")
                    print(f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}")
                    print("Note: Using city center coordinates since exact address wasn't found")
                    return location.latitude, location.longitude, location.address

            # Strategy 3: Manual coordinate entry
            print(f"\nGeocoding failed for address: {search_query}")
            print("You can:")
            print("1. Enter coordinates manually")
            print("2. Try a simpler address (just city, state)")
            print("3. Exit")

            choice = input("Enter your choice (1-3): ").strip()

            if choice == "1":
                try:
                    lat = float(input("Enter latitude (decimal degrees): ").strip())
                    lon = float(input("Enter longitude (decimal degrees): ").strip())
                    manual_address = (
                        input("Enter description for this location: ").strip()
                        or "Manual coordinates"
                    )
                    print(f"Using manual coordinates: {lat:.6f}, {lon:.6f}")
                    return lat, lon, manual_address
                except ValueError:
                    raise ValueError("Invalid coordinates entered")
            elif choice == "2":
                new_address = input("Enter a simpler address: ").strip()
                if new_address:
                    return self.get_coordinates_from_address(new_address, country)
                else:
                    raise ValueError("No address provided")
            else:
                raise ValueError("Geocoding cancelled by user")

        except Exception as e:
            if "cancelled by user" in str(e) or "Invalid coordinates" in str(e):
                raise ValueError(str(e))
            else:
                raise ValueError(f"Error geocoding address: {e}")

    def calculate_chunks(
        self, center_lat: float, center_lon: float, radius_km: float
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate geographic chunks to process data in smaller pieces.
        Returns list of (lat, lon, chunk_radius) tuples.
        """
        if radius_km <= self.chunk_size_km:
            return [(center_lat, center_lon, radius_km)]

        chunks = []
        # Simple grid-based chunking
        chunk_radius = self.chunk_size_km / 2

        # Calculate how many chunks we need in each direction
        chunks_per_side = math.ceil(radius_km / self.chunk_size_km)

        for i in range(-chunks_per_side, chunks_per_side + 1):
            for j in range(-chunks_per_side, chunks_per_side + 1):
                # Calculate chunk center
                lat_offset = i * self.chunk_size_km / 111.0  # Rough km to degree conversion
                lon_offset = j * self.chunk_size_km / (111.0 * math.cos(math.radians(center_lat)))

                chunk_lat = center_lat + lat_offset
                chunk_lon = center_lon + lon_offset

                # Check if chunk is within the original radius
                distance = self.calculate_distance(center_lat, center_lon, chunk_lat, chunk_lon)
                if distance <= radius_km:
                    chunks.append((chunk_lat, chunk_lon, chunk_radius))

        print(f"Processing area in {len(chunks)} chunks of ~{self.chunk_size_km}km radius each")
        return chunks

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        # Safety check for None values
        if any(coord is None for coord in [lat1, lon1, lat2, lon2]):
            return 0.0

        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def build_overpass_query(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        cycling_only: bool = False,
    ) -> str:
        """Build Overpass query with surface filtering and optional cycling restrictions."""

        distance_filter = f"(around:{radius_km * 1000},{center_lat},{center_lon})"

        # Get base surface filter
        surface_filters = ROAD_SURFACE_FILTERS.get(self.surface_filter, ROAD_SURFACE_FILTERS["all"])

        if cycling_only:
            # Modify highway filter for cycling restrictions
            highway_filter = surface_filters[0]

            if self.surface_filter == "paved":
                # Remove footway, require cycling access for paths/bridleways
                highway_query = f"""
                way[highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|cycleway"]
                    [access!~"no|private"][bicycle!~"no|private"]
                    {distance_filter};
                way[highway~"path|bridleway"]
                    [access!~"no|private"][bicycle!~"no|private"]
                    {distance_filter};
                way[highway~"footway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"]
                    {distance_filter};
                """

            elif self.surface_filter == "gravel":
                highway_query = f"""
                way[highway~"track|unclassified|tertiary|residential|service|cycleway"]
                    [access!~"no|private"][bicycle!~"no|private"]
                    {distance_filter};
                way[highway~"path|bridleway"]
                    [access!~"no|private"][bicycle!~"no|private"]
                    {distance_filter};
                way[highway~"footway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"]
                    {distance_filter};
                """

            elif self.surface_filter == "dirt":
                # Remove steps, add cycling requirements
                highway_query = f"""
                way[highway~"track"]
                    [access!~"no|private"][bicycle!~"no|private"]
                    {distance_filter};
                way[highway~"path|footway|cycleway|bridleway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"]
                    {distance_filter};
                """

            else:  # 'all'
                highway_query = f"""
                way[highway~"trunk|primary|secondary|tertiary|unclassified|residential|service|track|cycleway"]
                    [access!~"no|private"][bicycle!~"no|private"]
                    {distance_filter};
                way[highway~"path|bridleway"]
                    [access!~"no|private"][bicycle!~"no|private"]
                    {distance_filter};
                way[highway~"footway"]
                    [bicycle~"yes|designated|permissive"]
                    [access!~"no|private"]
                    {distance_filter};
                """

            # Add surface filter if it exists
            if len(surface_filters) > 1:
                surface_condition = f"[{surface_filters[1]}]"
                # Apply surface filter to both query parts
                highway_query = highway_query.replace(
                    distance_filter, f"{surface_condition}{distance_filter}"
                )

            query = f"""
            [out:json][timeout:60][maxsize:1073741824];
            (
              {highway_query}
            );
            (._;>;);
            out geom;
            """

        else:
            # Use original surface filters without cycling restrictions
            if len(surface_filters) == 1:
                # No surface restrictions, just highway type
                query = f"""
                [out:json][timeout:60][maxsize:1073741824];
                (
                  way[{surface_filters[0]}]
                      {distance_filter};
                );
                (._;>;);
                out geom;
                """
            else:
                # Highway type + surface restrictions
                query = f"""
                [out:json][timeout:60][maxsize:1073741824];
                (
                  way[{surface_filters[0]}]
                      [{surface_filters[1]}]
                      {distance_filter};
                );
                (._;>;);
                out geom;
                """

        return query

    def _create_simple_way(self, element, all_nodes):
        """Create a simple way object."""
        way_id = element["id"]
        tags = element.get("tags", {})
        nodes = []

        # Handle geometry data (preferred for memory efficiency)
        if "geometry" in element:
            for idx, geom_point in enumerate(element["geometry"]):
                if "lat" in geom_point and "lon" in geom_point:
                    node_data = {
                        "id": f"{way_id}_{idx}",
                        "lat": geom_point["lat"],
                        "lon": geom_point["lon"],
                    }
                    node = SimpleNode(node_data)
                    nodes.append(node)

        # Handle regular node references (fallback)
        elif "nodes" in element:
            for node_id in element["nodes"]:
                if node_id in all_nodes:
                    node = SimpleNode(all_nodes[node_id])
                    nodes.append(node)

        if nodes:
            way = SimpleWay()
            way.id = way_id
            way.tags = tags
            way.nodes = nodes
            return way

        return None


def _merge_street_batch_worker_top_level(args):
    """
    Top-level worker function for parallel street merging.
    Must be at module level for pickle serialization in multiprocessing.

    Args:
        args: Tuple of (street_items_batch, coordinate_tolerance, distance_tolerance_m)

    Returns:
        List of merged segments for all streets in batch
    """
    street_items_batch, coordinate_tolerance, distance_tolerance_m = args

    # Create a temporary merger instance for this worker
    merger = BoundaryMerger(coordinate_tolerance=coordinate_tolerance)
    merger.distance_tolerance_m = distance_tolerance_m

    batch_results = []
    for street_name, street_segments in street_items_batch:
        try:
            # Determine merge strategy based on street characteristics
            common_names = {
                "service",
                "track",
                "path",
                "footway",
                "cycleway",
                "bridleway",
                "steps",
                "residential",
                "unclassified",
            }
            is_common_name = street_name.lower() in common_names

            if is_common_name and len(street_segments) > 50:
                merged = merger._merge_large_street_with_spatial_grouping(
                    street_segments, street_name
                )
            else:
                merged = merger._merge_street(street_segments, street_name)

            batch_results.extend(merged)
        except Exception as e:
            # Log error but continue processing other streets
            import traceback

            print(f"\nError in worker processing street '{street_name}': {e}")
            traceback.print_exc()
            # Include unmerged segments as fallback
            batch_results.extend(street_segments)

    return batch_results


class BoundaryMerger:
    """Handles merging of road segments that cross region boundaries."""

    def __init__(self, coordinate_tolerance: float = 0.002):
        self.coordinate_tolerance = coordinate_tolerance
        self.distance_tolerance_m = 200
        self.parallel_enabled = MERGE_PARALLEL_ENABLED
        self.max_workers = MERGE_MAX_WORKERS
        self.batch_size = MERGE_BATCH_SIZE

    def _calculate_safe_worker_count(self, total_segments: int) -> int:
        """
        Calculate safe number of workers based on available memory.

        Each worker needs memory for:
        - Python interpreter overhead: ~50-100 MB
        - Batch of streets to process: ~500 streets × avg segments
        - Temporary merge results: 2-3x input size during processing

        Conservative estimate: ~500 MB per worker for large datasets
        """
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)

            # Conservative estimates
            MB_PER_WORKER = 500  # Assume 500MB per worker
            SAFETY_MARGIN = 0.7  # Only use 70% of available memory

            # Calculate how many workers we can safely spawn
            available_mb = (available_gb * 1024) * SAFETY_MARGIN
            safe_workers = int(available_mb / MB_PER_WORKER)

            # Clamp to reasonable bounds
            safe_workers = max(1, min(safe_workers, self.max_workers, os.cpu_count() or 1))

            print("\nMemory-based worker calculation:")
            print(f"  Total RAM: {mem.total / (1024**3):.1f} GB")
            print(f"  Available RAM: {available_gb:.1f} GB")
            print(f"  Safe workers (with {int(SAFETY_MARGIN*100)}% safety margin): {safe_workers}")
            print(f"  Configured max workers: {self.max_workers}")

            if safe_workers < self.max_workers:
                print(
                    f"  ⚠️  Reducing workers from {self.max_workers} to {safe_workers} due to memory constraints"
                )

            return safe_workers

        except Exception as e:
            print(f"  ⚠️  Could not calculate memory-safe worker count: {e}")
            print("  Defaulting to conservative 4 workers")
            return min(4, self.max_workers)

    def _coordinates_match(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> bool:
        """Coordinate matching using actual distance."""
        if (
            abs(coord1[0] - coord2[0]) < self.coordinate_tolerance
            and abs(coord1[1] - coord2[1]) < self.coordinate_tolerance
        ):
            return True

        distance_km = self.calculate_distance(coord1[0], coord1[1], coord2[0], coord2[1])
        distance_m = distance_km * 1000

        return distance_m < self.distance_tolerance_m

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        # Safety check for None values
        if any(coord is None for coord in [lat1, lon1, lat2, lon2]):
            return 0.0

        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _merge_large_street_with_spatial_grouping(
        self, segments: List[Dict], street_name: str
    ) -> List[Dict]:
        """Handle very large streets (>100 segments) by grouping spatially first."""

        # Group segments by approximate location (larger grid for very large streets)
        location_groups = self._group_segments_by_location(segments, grid_size=0.005)  # ~500m grid

        merged_segments = []

        # Show nested progress bar for large streets (>200 segments) to show processing progress
        # This helps avoid the appearance of hanging during long street merges
        show_nested_progress = len(segments) > 200 and len(location_groups) > 5

        # Debug: Print when we encounter large streets to help diagnose progress bar visibility
        if len(segments) > 200:
            status = (
                "✓ nested progress"
                if show_nested_progress
                else f"⚠ only {len(location_groups)} groups"
            )
            print(
                f"   Large street: {street_name[:30]} ({len(segments)} segs, {len(location_groups)} groups) - {status}"
            )

        iterator = enumerate(location_groups)
        if show_nested_progress:
            display_name = street_name[:20] + "..." if len(street_name) > 20 else street_name
            iterator = tqdm(
                iterator,
                total=len(location_groups),
                desc=f"  └─ {display_name}",
                unit="cluster",
                leave=False,
                position=1,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                ascii=" ▏▎▍▌▋▊▉█",
                dynamic_ncols=True,
            )

        for i, group_segments in iterator:
            if len(group_segments) <= 20:
                # Use regular merging for reasonable-sized groups
                group_merged = self._merge_street(group_segments, f"{street_name}_cluster_{i}")
                merged_segments.extend(group_merged)
            else:
                # For very large groups within a location, sub-divide further
                sub_groups = self._group_segments_by_location(
                    group_segments, grid_size=0.001
                )  # ~100m grid

                for j, sub_group in enumerate(sub_groups):
                    if len(sub_group) <= 10:
                        sub_merged = self._merge_street(sub_group, f"{street_name}_cluster_{i}_{j}")
                        merged_segments.extend(sub_merged)
                    else:
                        # If still too large, just keep as separate segments
                        merged_segments.extend(sub_group)

        return merged_segments

    def merge_boundary_segments(
        self, all_segments: List[Dict], persistence: ChunkPersistenceManager
    ) -> List[Dict]:
        """
        Merge segments that represent the same road across different chunks.
        With progress tracking and checkpointing for large datasets.

        Automatically uses parallel processing for large datasets.
        """
        print(f"\nPerforming boundary merge on {len(all_segments):,} segments...")

        # Check for existing progress
        progress_data = persistence.load_boundary_merge_progress()

        if progress_data:
            print("Resuming boundary merge from checkpoint...")
            print(f"  - Streets processed: {progress_data.get('streets_processed', 0)}")
            print(f"  - Total streets: {progress_data.get('total_streets', 0)}")

            # Check if it was running in parallel mode
            if progress_data.get("parallel_mode", False):
                print("  - Resuming in parallel mode")
                # TODO: Implement parallel resume (for now, continue with serial)

            return self._resume_boundary_merge(all_segments, progress_data, persistence)

        # Decide whether to use parallel or serial processing
        # Use parallel for datasets with >1000 segments (threshold can be tuned)
        if self.parallel_enabled and len(all_segments) > 1000 and self.max_workers > 1:
            print(f"✓ Using PARALLEL processing with {self.max_workers} workers")
            return self._perform_boundary_merge_parallel(all_segments, persistence)
        else:
            print("✓ Using SERIAL processing (dataset too small or parallel disabled)")
            return self._perform_boundary_merge(all_segments, persistence)

    def _perform_boundary_merge(
        self, all_segments: List[Dict], persistence: ChunkPersistenceManager
    ) -> List[Dict]:
        """Perform boundary merge with smart checkpointing."""

        # Group segments by street name (unchanged logic)
        print("Grouping segments by street name...")
        segments_by_name = defaultdict(list)

        with tqdm(
            total=len(all_segments),
            desc="Grouping by street name",
            unit="segments",
            miniters=len(all_segments) // 100,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for segment in all_segments:
                street_name = segment.get("street_name", "").strip()
                if street_name:
                    segments_by_name[street_name].append(segment)

                if pbar.n % max(1, len(all_segments) // 100) == 0:
                    pbar.set_postfix({"unique_streets": len(segments_by_name)})
                pbar.update(1)

        # Prepare for processing
        streets_to_merge = {name: segs for name, segs in segments_by_name.items() if len(segs) > 1}
        single_segments = []

        for name, segs in segments_by_name.items():
            if len(segs) == 1:
                single_segments.extend(segs)

        print(
            "Grouping complete. \n\n",
        )

        # Process street groups with smart checkpointing
        merged_segments = single_segments.copy()
        processed_streets = []
        processed_street_segments = {}

        street_items = list(streets_to_merge.items())

        # RANDOMIZE street order to distribute heavy processing throughout
        import random

        random.shuffle(street_items)
        print("Randomized street processing order to distribute workload evenly")
        print("\n   Note: Common street names (like 'service', 'track', 'path') have thousands")
        print("   of segments and will temporarily slow progress when encountered.")
        print("   This is normal - progress will speed up as these complete.\n")

        # Initialize smart checkpointer
        checkpointer = SmartCheckpointer(len(street_items), "Boundary Merge")

        with tqdm(
            total=len(street_items),
            desc="Merging street segments",
            unit="streets",
            miniters=1,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            postfix_update_counter = 0  # Counter to reduce flickering
            for i, (street_name, street_segments) in enumerate(street_items):
                # Signal handler check every 10 streets
                if i % 10 == 0:
                    signal_handler.set_operation(
                        "boundary_merge",
                        {
                            "streets_processed": i,
                            "total_streets": len(street_items),
                            "processed_street_names": processed_streets,
                            "processed_street_segments": processed_street_segments,
                            "merged_segments_count": len(merged_segments),
                            "single_segments": single_segments,
                            "segments_by_name_remaining": {
                                k: v
                                for k, v in streets_to_merge.items()
                                if k not in processed_streets
                            },
                            "timestamp": time.time(),
                        },
                    )

                if signal_handler.kill_now:
                    progress_data = {
                        "streets_processed": i,
                        "total_streets": len(street_items),
                        "processed_street_names": processed_streets,
                        "processed_street_segments": processed_street_segments,
                        "merged_segments_count": len(merged_segments),
                        "single_segments": single_segments,
                        "segments_by_name_remaining": {
                            k: v for k, v in streets_to_merge.items() if k not in processed_streets
                        },
                        "timestamp": time.time(),
                    }
                    persistence.save_boundary_merge_progress(progress_data)
                    print("Boundary merge progress saved. Analysis can be resumed.")
                    sys.exit(0)

                try:
                    # Process this street (existing logic)
                    display_name = (
                        street_name[:12] + "..." if len(street_name) > 15 else street_name
                    )

                    # ALWAYS update immediately for large streets (reduces perceived hang)
                    # Only throttle updates for small streets to reduce flickering
                    should_update_postfix = (len(street_segments) > 100) or (
                        postfix_update_counter % 10 == 0
                    )

                    if should_update_postfix:
                        if len(street_segments) > 500:
                            pbar.set_postfix_str(
                                f"LARGE STREET: {display_name} ({len(street_segments):,} segments) - processing..."
                            )
                        elif len(street_segments) > 100:
                            pbar.set_postfix_str(
                                f"Processing: {display_name} ({len(street_segments)} segments)"
                            )
                        elif len(street_segments) > 50:
                            pbar.set_postfix_str(
                                f"Processing {display_name} ({len(street_segments)} segs)"
                            )

                    postfix_update_counter += 1

                    merged_street_segments = self._merge_segments_for_street(
                        street_segments, street_name, persistence
                    )
                    merged_segments.extend(merged_street_segments)
                    processed_streets.append(street_name)
                    processed_street_segments[street_name] = merged_street_segments

                    # SMART CHECKPOINT CHECK
                    if checkpointer.should_checkpoint(i):
                        progress_data = {
                            "streets_processed": i + 1,
                            "total_streets": len(street_items),
                            "processed_street_names": processed_streets,
                            "processed_street_segments": processed_street_segments,
                            "merged_segments_count": len(merged_segments),
                            "single_segments": single_segments,
                            "segments_by_name_remaining": {
                                k: v
                                for k, v in streets_to_merge.items()
                                if k not in processed_streets
                            },
                            "timestamp": time.time(),
                        }
                        persistence.save_boundary_merge_progress(progress_data)

                        # Update progress bar with checkpoint info
                        info = checkpointer.get_checkpoint_info(i)
                        pbar.set_postfix(
                            {
                                "current": display_name,
                                "segments": len(street_segments),
                                "total_merged": len(merged_segments),
                                "next_save": f"{info['time_until_next_min']:.1f}min",
                            }
                        )
                    else:
                        pbar.set_postfix(
                            {
                                "current": display_name,
                                "segments": len(street_segments),
                                "total_merged": len(merged_segments),
                            }
                        )

                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing street '{street_name}': {e}")
                    # Save checkpoint before failing
                    progress_data = {
                        "streets_processed": i,
                        "total_streets": len(street_items),
                        "processed_street_names": processed_streets,
                        "processed_street_segments": processed_street_segments,
                        "merged_segments_count": len(merged_segments),
                        "single_segments": single_segments,
                        "segments_by_name_remaining": {
                            k: v for k, v in streets_to_merge.items() if k not in processed_streets
                        },
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    persistence.save_boundary_merge_progress(progress_data)
                    raise

        print(
            f"Boundary merge completed: {len(all_segments)} -> {len(merged_segments)} segments"
        )
        persistence.clear_boundary_merge_progress()
        check_and_cleanup_memory(force_cleanup=True)

        return merged_segments

    def _perform_boundary_merge_parallel(
        self, all_segments: List[Dict], persistence: ChunkPersistenceManager
    ) -> List[Dict]:
        """
        Parallel boundary merge using ProcessPoolExecutor.

        Processes streets in batches across multiple CPU cores for dramatic speedup.
        Includes chunked output to prevent memory accumulation.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Group segments by street name
        print(f"Grouping {len(all_segments):,} segments by street name...")
        segments_by_name = defaultdict(list)

        with tqdm(
            total=len(all_segments),
            desc="Grouping by street name",
            unit="segments",
            miniters=len(all_segments) // 100,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for segment in all_segments:
                street_name = segment.get("street_name", "").strip()
                if street_name:
                    segments_by_name[street_name].append(segment)

                if pbar.n % max(1, len(all_segments) // 100) == 0:
                    pbar.set_postfix({"unique_streets": len(segments_by_name)})
                pbar.update(1)

        # Separate single-segment streets (no merging needed)
        streets_to_merge = {name: segs for name, segs in segments_by_name.items() if len(segs) > 1}
        single_segments = []

        for name, segs in segments_by_name.items():
            if len(segs) == 1:
                single_segments.extend(segs)

        from climb_analyzer.utils.formatting import print_dim

        print(f"\nProcessing {len(streets_to_merge):,} streets with multiple segments")
        print_dim(f"Skipping {len(single_segments):,} single-segment streets (no merge needed)")

        # Calculate safe worker count based on available memory
        safe_workers = self._calculate_safe_worker_count(len(all_segments))
        actual_workers = safe_workers

        print_dim(f"Using {actual_workers} parallel workers with batch size {self.batch_size}\n")

        street_items = list(streets_to_merge.items())

        # RANDOMIZE street order to distribute heavy processing throughout
        # This prevents all the slow "service", "track", "path" streets from being processed first
        import random

        random.shuffle(street_items)
        print_dim("Randomized street processing order to distribute workload evenly")
        print_dim("\n   Note: Common street names (like 'service', 'track', 'path') have thousands")
        print_dim("   of segments and will temporarily slow progress when encountered.")
        print_dim("   This is normal - progress will speed up as these complete.\n")

        # Create batches for parallel processing
        batches = []
        for i in range(0, len(street_items), self.batch_size):
            batch = street_items[i : i + self.batch_size]
            batches.append((batch, self.coordinate_tolerance, self.distance_tolerance_m))

        print(f"Split into {len(batches)} batches for parallel processing\n")

        # Initialize smart checkpointer
        checkpointer = SmartCheckpointer(len(street_items), "Boundary Merge")

        # Process batches in parallel with streaming output
        merged_segments = single_segments.copy()
        processed_streets_count = 0
        batch_checkpoint_dir = persistence.analysis_dir / "batch_checkpoints"
        batch_checkpoint_dir.mkdir(exist_ok=True)

        try:
            executor = ProcessPoolExecutor(max_workers=actual_workers)
        except Exception as e:
            print(f"\n⚠️  Warning: Could not initialize parallel processing: {e}")
            print("Falling back to SERIAL processing (slower but more reliable)\n")
            # Fall back to serial processing
            return self._perform_boundary_merge(all_segments, persistence)

        with executor:
            # Test worker pool with a small test batch first
            print("Testing worker pool with first batch...")
            try:
                test_future = executor.submit(_merge_street_batch_worker_top_level, batches[0])
                # Wait a moment to see if it crashes immediately
                import time

                time.sleep(0.5)
                if test_future.done() and test_future.exception():
                    raise test_future.exception()
                print("✓ Worker pool test successful\n")
            except Exception as e:
                print(f"\n❌ Worker pool test failed: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback

                traceback.print_exc()
                print("\nFalling back to SERIAL processing (slower but more reliable)\n")
                executor.shutdown(wait=False)
                return self._perform_boundary_merge(all_segments, persistence)

            # Submit all batches
            print(f"Submitting {len(batches)} batches to worker pool...")
            try:
                future_to_batch = {
                    executor.submit(_merge_street_batch_worker_top_level, batch_args): idx
                    for idx, batch_args in enumerate(batches)
                }
                print(f"✓ All {len(batches)} batches submitted successfully\n")
            except Exception as e:
                print(f"\n❌ Error submitting batches: {e}")
                print("Falling back to SERIAL processing\n")
                executor.shutdown(wait=False)
                return self._perform_boundary_merge(all_segments, persistence)

            # Process completed batches with progress bar
            with tqdm(
                total=len(street_items),
                desc="Merging street segments (parallel)",
                unit="streets",
                miniters=1,
                dynamic_ncols=True,
                ascii=" ▏▎▍▌▋▊▉█",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ) as pbar:

                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]

                    try:
                        # Get results from completed batch
                        batch_results = future.result()

                        # Stream to checkpoint file immediately (memory efficient)
                        batch_checkpoint_file = batch_checkpoint_dir / f"batch_{batch_idx:04d}.pkl"
                        with open(batch_checkpoint_file, "wb") as f:
                            pickle.dump(batch_results, f)

                        # Track progress
                        batch_street_count = len(batches[batch_idx][0])
                        processed_streets_count += batch_street_count

                        # Update progress bar
                        pbar.update(batch_street_count)
                        pbar.set_postfix(
                            {
                                "batch": f"{len([f for f in future_to_batch.values() if f <= batch_idx])}/{len(batches)}",
                                "workers": actual_workers,
                                "mem_saved": f"{len(batch_results):,} segs on disk",
                            }
                        )

                        # Checkpoint if needed
                        if checkpointer.should_checkpoint(processed_streets_count - 1):
                            progress_data = {
                                "streets_processed": processed_streets_count,
                                "total_streets": len(street_items),
                                "batches_completed": [
                                    idx for future, idx in future_to_batch.items() if future.done()
                                ],
                                "batch_checkpoint_dir": str(batch_checkpoint_dir),
                                "single_segments": single_segments,
                                "timestamp": time.time(),
                                "parallel_mode": True,
                            }
                            persistence.save_boundary_merge_progress(progress_data)

                            info = checkpointer.get_checkpoint_info(processed_streets_count - 1)
                            pbar.write(
                                f"  Checkpoint saved: {processed_streets_count}/{len(street_items)} streets, "
                                f"next save in {info['time_until_next_min']:.1f}min"
                            )

                        # Signal handler check
                        if signal_handler.kill_now:
                            pbar.write("\n\nInterrupted! Saving progress...")
                            progress_data = {
                                "streets_processed": processed_streets_count,
                                "total_streets": len(street_items),
                                "batches_completed": [
                                    idx for future, idx in future_to_batch.items() if future.done()
                                ],
                                "batch_checkpoint_dir": str(batch_checkpoint_dir),
                                "single_segments": single_segments,
                                "timestamp": time.time(),
                                "parallel_mode": True,
                            }
                            persistence.save_boundary_merge_progress(progress_data)
                            pbar.write("Progress saved. Analysis can be resumed.")
                            sys.exit(0)

                    except Exception as e:
                        pbar.write(f"❌ Error processing batch {batch_idx}: {e}")
                        # Continue with other batches
                        processed_streets_count += len(batches[batch_idx][0])
                        pbar.update(len(batches[batch_idx][0]))

        # Merge all batch checkpoint files
        print("\nMerging batch results from disk...")

        for batch_idx in tqdm(
            range(len(batches)),
            desc="Loading batch checkpoints",
            unit=" batch",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            batch_checkpoint_file = batch_checkpoint_dir / f"batch_{batch_idx:04d}.pkl"
            if batch_checkpoint_file.exists():
                try:
                    with open(batch_checkpoint_file, "rb") as f:
                        batch_results = safe_pickle_load(f)
                    merged_segments.extend(batch_results)

                    # Delete checkpoint file to free disk space
                    batch_checkpoint_file.unlink()
                except Exception as e:
                    tqdm.write(f"⚠️  Warning: Could not load batch {batch_idx}: {e}")

        # Cleanup batch checkpoint directory
        try:
            if batch_checkpoint_dir.exists() and not any(batch_checkpoint_dir.iterdir()):
                batch_checkpoint_dir.rmdir()
        except Exception:
            pass

        print(
            f"\nParallel boundary merge completed: {len(all_segments):,} -> {len(merged_segments):,} segments"
        )
        print(f"Used {self.max_workers} workers with {self.batch_size} streets/batch")
        persistence.clear_boundary_merge_progress()
        check_and_cleanup_memory(force_cleanup=True)

        return merged_segments

    def _resume_boundary_merge(
        self,
        all_segments: List[Dict],
        progress_data: Dict,
        persistence: ChunkPersistenceManager,
    ) -> List[Dict]:
        """Resume boundary merge from checkpoint."""

        streets_processed = progress_data.get("streets_processed", 0)
        processed_street_names = set(progress_data.get("processed_street_names", []))
        processed_street_segments = progress_data.get("processed_street_segments", {})  # NEW
        single_segments = progress_data.get("single_segments", [])
        segments_by_name_remaining = progress_data.get("segments_by_name_remaining", {})  # NEW

        print(f"Resuming boundary merge from street {streets_processed}")

        # Rebuild merged segments from checkpoint data
        merged_segments = single_segments.copy()

        # Add already processed street segments
        for street_name, street_segments in processed_street_segments.items():
            merged_segments.extend(street_segments)

        print(f"Restored {len(merged_segments)} segments from checkpoint")

        # Check if we have remaining work
        if not segments_by_name_remaining:
            print("All streets already processed!")
            return merged_segments

        print(f"Streets remaining to process: {len(segments_by_name_remaining)}")

        # Process remaining streets
        checkpoint_interval = max(1, len(segments_by_name_remaining) // 20)
        street_items = list(segments_by_name_remaining.items())

        with tqdm(
            total=len(street_items),
            desc="Processing remaining streets",
            unit="streets",
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for i, (street_name, street_segments) in enumerate(street_items):
                try:
                    merged_street_segments = self._merge_segments_for_street(
                        street_segments, street_name, persistence
                    )

                    merged_segments.extend(merged_street_segments)
                    processed_street_names.add(street_name)
                    processed_street_segments[street_name] = merged_street_segments  # NEW

                    # Save checkpoint periodically
                    if i % checkpoint_interval == 0 or i == len(street_items) - 1:
                        # Update remaining work
                        remaining_items = {k: v for j, (k, v) in enumerate(street_items) if j > i}

                        progress_data_updated = {
                            "streets_processed": streets_processed + i + 1,
                            "total_streets": progress_data.get("total_streets", 0),
                            "processed_street_names": list(processed_street_names),
                            "processed_street_segments": processed_street_segments,  # NEW
                            "merged_segments_count": len(merged_segments),
                            "single_segments": single_segments,
                            "segments_by_name_remaining": remaining_items,  # NEW
                            "timestamp": time.time(),
                        }
                        persistence.save_boundary_merge_progress(progress_data_updated)

                    pbar.set_postfix(
                        {
                            "street": (
                                street_name[:20] + "..." if len(street_name) > 20 else street_name
                            ),
                            "total_merged": len(merged_segments),
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing street '{street_name}': {e}")
                    # Update remaining work before saving error checkpoint
                    remaining_items = {k: v for j, (k, v) in enumerate(street_items) if j >= i}

                    progress_data["error"] = str(e)
                    progress_data["segments_by_name_remaining"] = remaining_items
                    progress_data["processed_street_segments"] = processed_street_segments
                    persistence.save_boundary_merge_progress(progress_data)
                    raise

        print(
            f"Boundary merge completed: {len(all_segments)} -> {len(merged_segments)} segments"
        )

        # Clear checkpoint after successful completion
        persistence.clear_boundary_merge_progress()

        return merged_segments

    def _group_segments_by_location(
        self, segments: List[Dict], grid_size: float = 0.01
    ) -> List[List[Dict]]:
        """Group segments by approximate geographic location to reduce merge complexity."""
        location_groups = defaultdict(list)

        for segment in segments:
            nodes = segment.get("nodes", [])
            if nodes:
                # Use first node location for grouping
                first_node = nodes[0]
                if hasattr(first_node, "lat") and hasattr(first_node, "lon"):
                    # Round to grid for grouping (0.01 degrees â‰ˆ 1km)
                    grid_lat = round(float(first_node.lat) / grid_size) * grid_size
                    grid_lon = round(float(first_node.lon) / grid_size) * grid_size
                    location_groups[(grid_lat, grid_lon)].append(segment)

        return list(location_groups.values())

    def _connect_segments_iteratively(self, merged_segments: List[Dict], max_nodes: int = None):
        """
        [DEPRECATED] Old O(n²) iterative merging - replaced by endpoint index.

        This method is no longer used but kept for reference.
        See _merge_street() for the new O(n) implementation.

        Args:
            merged_segments: List of segments to process (modified in place)
            max_nodes: If provided, only process segments with <= this many nodes
        """
        changes_made = True
        iteration = 0
        max_iterations = 10  # Prevent infinite loops

        while changes_made and iteration < max_iterations:
            changes_made = False
            iteration += 1

            # Process segments in current list
            i = 0
            while i < len(merged_segments):
                current_segment = merged_segments[i]

                # Skip if max_nodes filter doesn't match
                if max_nodes is not None and len(current_segment.get("nodes", [])) > max_nodes:
                    i += 1
                    continue

                # Look for segments to connect to this one
                j = i + 1
                while j < len(merged_segments):
                    other_segment = merged_segments[j]

                    # Check if these segments can be connected
                    connection_type = self._check_segment_connectivity(
                        current_segment, other_segment
                    )

                    if connection_type:
                        # Connect the segments
                        connected_segment = self._connect_segments(
                            current_segment, other_segment, connection_type
                        )

                        # Replace current segment with connected one
                        merged_segments[i] = connected_segment
                        current_segment = connected_segment

                        # Remove the other segment (it's now part of the connected segment)
                        merged_segments.pop(j)

                        changes_made = True
                        # Don't increment j since we removed an element
                    else:
                        j += 1

                i += 1

            # Memory cleanup for large datasets
            if iteration % 3 == 0:
                import gc

                gc.collect()

    def _merge_street(self, segments: List[Dict], street_name: str) -> List[Dict]:
        """
        Segment merging using endpoint index.

        O(n) complexity instead of O(n²) by using a spatial index of segment endpoints.

        Techniques applied:
        1. Cache endpoint coordinates during growth loop
        2. Remove used segments from endpoint_index
        3. Early exit when no neighbors exist
        4. Eliminate redundant list copies in _connect_segments
        """

        if len(segments) <= 1:
            return segments

        # Handle 2-segment streets (very common case)
        if len(segments) == 2:
            connection_type = self._check_segment_connectivity(segments[0], segments[1])
            if connection_type:
                return [self._connect_segments(segments[0], segments[1], connection_type)]
            return segments

        # Build endpoint index: coordinate → list of (segment_idx, endpoint_type)
        # endpoint_type: 'start' or 'end'
        from collections import defaultdict

        endpoint_index = defaultdict(list)

        for idx, seg in enumerate(segments):
            nodes = seg.get("nodes", [])
            if not nodes:
                continue

            start_coord = self._get_node_coordinates(nodes[0])
            end_coord = self._get_node_coordinates(nodes[-1])

            # Round coordinates for consistent matching
            start_coord = (round(start_coord[0], 6), round(start_coord[1], 6))
            end_coord = (round(end_coord[0], 6), round(end_coord[1], 6))

            endpoint_index[start_coord].append((idx, "start"))
            endpoint_index[end_coord].append((idx, "end"))

        # Merge segments using endpoint index
        merged = []
        used = set()

        for start_idx in range(len(segments)):
            if start_idx in used:
                continue

            # Start a new merged segment
            current_seg = segments[start_idx].copy()
            used.add(start_idx)

            # Grow this segment by connecting neighbors
            changes_made = True
            max_growth_iterations = 100  # Prevent infinite loops

            # OPTIMIZATION: Cache initial endpoints to avoid repeated calculations
            nodes = current_seg.get("nodes", [])
            if nodes:
                start_coord = self._get_node_coordinates(nodes[0])
                end_coord = self._get_node_coordinates(nodes[-1])
                current_endpoints = (
                    (round(start_coord[0], 6), round(start_coord[1], 6)),
                    (round(end_coord[0], 6), round(end_coord[1], 6)),
                )
            else:
                current_endpoints = None

            for _ in range(max_growth_iterations):
                if not changes_made:
                    break

                changes_made = False
                nodes = current_seg.get("nodes", [])
                if not nodes:
                    break

                # Early exit when no neighbors exist
                start_coord, end_coord = current_endpoints
                if not endpoint_index[start_coord] and not endpoint_index[end_coord]:
                    break  # No possible neighbors, done growing

                # Try to extend from start endpoint
                for neighbor_idx, neighbor_endpoint in endpoint_index[start_coord]:
                    if neighbor_idx in used:
                        continue

                    neighbor_seg = segments[neighbor_idx]
                    neighbor_nodes = neighbor_seg.get("nodes", [])
                    if not neighbor_nodes:
                        continue

                    # Connect based on which endpoint matches
                    if neighbor_endpoint == "start":
                        # Neighbor's start connects to our start -> reverse neighbor and prepend
                        current_seg = self._connect_segments(
                            neighbor_seg, current_seg, reverse_first=True
                        )
                    else:
                        # Neighbor's end connects to our start -> prepend neighbor
                        current_seg = self._connect_segments(
                            neighbor_seg, current_seg, reverse_first=False
                        )

                    used.add(neighbor_idx)
                    changes_made = True

                    # Remove used segment from endpoint_index
                    neighbor_nodes = neighbor_seg.get("nodes", [])
                    if neighbor_nodes:
                        neighbor_start = self._get_node_coordinates(neighbor_nodes[0])
                        neighbor_end = self._get_node_coordinates(neighbor_nodes[-1])
                        neighbor_start = (round(neighbor_start[0], 6), round(neighbor_start[1], 6))
                        neighbor_end = (round(neighbor_end[0], 6), round(neighbor_end[1], 6))

                        # Remove this neighbor from all its endpoint entries
                        endpoint_index[neighbor_start] = [
                            (idx, ep)
                            for idx, ep in endpoint_index[neighbor_start]
                            if idx != neighbor_idx
                        ]
                        endpoint_index[neighbor_end] = [
                            (idx, ep)
                            for idx, ep in endpoint_index[neighbor_end]
                            if idx != neighbor_idx
                        ]

                    # OPTIMIZATION: Update cached endpoints after merge
                    nodes = current_seg.get("nodes", [])
                    if nodes:
                        start_coord = self._get_node_coordinates(nodes[0])
                        end_coord = self._get_node_coordinates(nodes[-1])
                        current_endpoints = (
                            (round(start_coord[0], 6), round(start_coord[1], 6)),
                            (round(end_coord[0], 6), round(end_coord[1], 6)),
                        )
                    break

                # Try to extend from end endpoint (if we didn't extend from start)
                if not changes_made:
                    for neighbor_idx, neighbor_endpoint in endpoint_index[end_coord]:
                        if neighbor_idx in used:
                            continue

                        neighbor_seg = segments[neighbor_idx]
                        neighbor_nodes = neighbor_seg.get("nodes", [])
                        if not neighbor_nodes:
                            continue

                        # Connect based on which endpoint matches
                        if neighbor_endpoint == "start":
                            # Neighbor's start connects to our end -> append neighbor
                            current_seg = self._connect_segments(
                                current_seg, neighbor_seg, reverse_first=False
                            )
                        else:
                            # Neighbor's end connects to our end -> reverse neighbor and append
                            current_seg = self._connect_segments(
                                current_seg, neighbor_seg, reverse_first=True, reverse_second=True
                            )

                        used.add(neighbor_idx)
                        changes_made = True

                        # Remove used segment from endpoint_index
                        neighbor_nodes = neighbor_seg.get("nodes", [])
                        if neighbor_nodes:
                            neighbor_start = self._get_node_coordinates(neighbor_nodes[0])
                            neighbor_end = self._get_node_coordinates(neighbor_nodes[-1])
                            neighbor_start = (
                                round(neighbor_start[0], 6),
                                round(neighbor_start[1], 6),
                            )
                            neighbor_end = (round(neighbor_end[0], 6), round(neighbor_end[1], 6))

                            # Remove this neighbor from all its endpoint entries
                            endpoint_index[neighbor_start] = [
                                (idx, ep)
                                for idx, ep in endpoint_index[neighbor_start]
                                if idx != neighbor_idx
                            ]
                            endpoint_index[neighbor_end] = [
                                (idx, ep)
                                for idx, ep in endpoint_index[neighbor_end]
                                if idx != neighbor_idx
                            ]

                        # OPTIMIZATION: Update cached endpoints after merge
                        nodes = current_seg.get("nodes", [])
                        if nodes:
                            start_coord = self._get_node_coordinates(nodes[0])
                            end_coord = self._get_node_coordinates(nodes[-1])
                            current_endpoints = (
                                (round(start_coord[0], 6), round(start_coord[1], 6)),
                                (round(end_coord[0], 6), round(end_coord[1], 6)),
                            )
                        break

            merged.append(current_seg)

        return merged

    def _connect_segments(
        self, seg1: Dict, seg2: Dict, reverse_first: bool = False, reverse_second: bool = False
    ) -> Dict:
        """
        Fast segment connection with optional reversal.
        OPTIMIZED: Eliminates redundant list copies by building the final list once.

        Args:
            seg1: First segment
            seg2: Second segment
            reverse_first: Reverse seg1's nodes before connecting
            reverse_second: Reverse seg2's nodes before connecting (used when appending to end)

        Returns:
            Merged segment
        """
        # OPTIMIZATION: Build connected_nodes directly without intermediate copies
        nodes1 = seg1["nodes"]
        nodes2 = seg2["nodes"]

        if reverse_first and reverse_second:
            # Both reversed: reverse(nodes1) + reverse(nodes2[1:])
            connected_nodes = nodes1[::-1] + nodes2[-2::-1]
        elif reverse_first:
            # Only first reversed: reverse(nodes1) + nodes2[1:]
            connected_nodes = nodes1[::-1] + nodes2[1:]
        elif reverse_second:
            # Only second reversed: nodes1 + reverse(nodes2[1:])
            connected_nodes = nodes1 + nodes2[-2::-1]
        else:
            # Neither reversed: nodes1 + nodes2[1:]
            connected_nodes = nodes1 + nodes2[1:]

        # Create merged segment with all fields
        merged_segment = {
            "way_ids": seg1["way_ids"] + seg2["way_ids"],
            "nodes": connected_nodes,
            "street_name": seg1["street_name"],
            "surface": seg1["surface"],
            "tracktype": seg1["tracktype"],
            "tracktype_definition": seg1["tracktype_definition"],
            "cycling_access": seg1.get("cycling_access", "Unknown"),
            "highway_type": seg1.get("highway_type", "unknown"),
        }

        # Update surface/tracktype if seg1 has unknown but seg2 has info
        if seg1["surface"] == "unknown" and seg2["surface"] != "unknown":
            merged_segment["surface"] = seg2["surface"]

        if seg1["tracktype"] in ["unknown", "-"] and seg2["tracktype"] not in ["unknown", "-"]:
            merged_segment["tracktype"] = seg2["tracktype"]
            merged_segment["tracktype_definition"] = seg2["tracktype_definition"]

        if (
            seg1.get("cycling_access", "Unknown") == "Unknown"
            and seg2.get("cycling_access", "Unknown") != "Unknown"
        ):
            merged_segment["cycling_access"] = seg2.get("cycling_access", "Unknown")

        if (
            seg1.get("highway_type", "unknown") == "unknown"
            and seg2.get("highway_type", "unknown") != "unknown"
        ):
            merged_segment["highway_type"] = seg2.get("highway_type", "unknown")

        return merged_segment

    def _merge_segments_for_street(
        self,
        segments: List[Dict],
        street_name: str,
        persistence: ChunkPersistenceManager,
    ) -> List[Dict]:
        """Enhanced merging with spatial optimization for large streets with common names."""

        if len(segments) <= 1:
            return segments

        # Special handling for very common generic road names
        common_names = {
            "service",
            "track",
            "path",
            "footway",
            "cycleway",
            "bridleway",
            "steps",
            "residential",
            "unclassified",
        }
        is_common_name = street_name.lower() in common_names

        # Apply spatial grouping only for common names when segments > 50
        if is_common_name and len(segments) > 50:
            return self._merge_large_street_with_spatial_grouping(segments, street_name)
        else:
            # Use basic merging for all other streets regardless of length
            return self._merge_street(segments, street_name)

    def _check_segment_connectivity(self, segment1: Dict, segment2: Dict) -> Optional[str]:
        """Check if two segments can be connected."""
        nodes1 = segment1.get("nodes", [])
        nodes2 = segment2.get("nodes", [])

        if not nodes1 or not nodes2:
            return None

        # Get endpoint coordinates
        start1 = self._get_node_coordinates(nodes1[0])
        end1 = self._get_node_coordinates(nodes1[-1])
        start2 = self._get_node_coordinates(nodes2[0])
        end2 = self._get_node_coordinates(nodes2[-1])

        # Check all possible connections
        connections = []
        if self._coordinates_match(end1, start2):
            connections.append("end1_to_start2")
        if self._coordinates_match(end1, end2):
            connections.append("end1_to_end2")
        if self._coordinates_match(start1, start2):
            connections.append("start1_to_start2")
        if self._coordinates_match(start1, end2):
            connections.append("start1_to_end2")

        return connections[0] if connections else None

    def _get_node_coordinates(self, node) -> Tuple[float, float]:
        """Extract coordinates from a node."""
        if hasattr(node, "lat") and hasattr(node, "lon"):
            return (float(node.lat), float(node.lon))
        return (0.0, 0.0)

    def _connect_segments(self, segment1: Dict, segment2: Dict, connection_type: str) -> Dict:
        """Connect two segments based on connection type."""
        nodes1 = segment1["nodes"].copy()
        nodes2 = segment2["nodes"].copy()

        if connection_type == "end1_to_start2":
            connected_nodes = nodes1 + nodes2[1:]
        elif connection_type == "end1_to_end2":
            connected_nodes = nodes1 + nodes2[-2::-1]
        elif connection_type == "start1_to_start2":
            connected_nodes = nodes1[::-1] + nodes2[1:]
        elif connection_type == "start1_to_end2":
            connected_nodes = nodes2 + nodes1[1:]
        else:
            connected_nodes = nodes1 + nodes2

        # Create merged segment - FIXED: Include all fields
        merged_segment = {
            "way_ids": segment1["way_ids"] + segment2["way_ids"],
            "nodes": connected_nodes,
            "street_name": segment1["street_name"],
            "surface": segment1["surface"],
            "tracktype": segment1["tracktype"],
            "tracktype_definition": segment1["tracktype_definition"],
            "cycling_access": segment1.get("cycling_access", "Unknown"),  # ADDED
            "highway_type": segment1.get("highway_type", "unknown"),  # ADDED
        }

        # Update surface/tracktype if current is unknown but other has info
        if segment1["surface"] == "unknown" and segment2["surface"] != "unknown":
            merged_segment["surface"] = segment2["surface"]

        if segment1["tracktype"] in ["unknown", "-"] and segment2["tracktype"] not in [
            "unknown",
            "-",
        ]:
            merged_segment["tracktype"] = segment2["tracktype"]
            merged_segment["tracktype_definition"] = segment2["tracktype_definition"]

        # ADDED: Update cycling_access and highway_type if segment1 has unknown values
        if (
            segment1.get("cycling_access", "Unknown") == "Unknown"
            and segment2.get("cycling_access", "Unknown") != "Unknown"
        ):
            merged_segment["cycling_access"] = segment2.get("cycling_access", "Unknown")

        if (
            segment1.get("highway_type", "unknown") == "unknown"
            and segment2.get("highway_type", "unknown") != "unknown"
        ):
            merged_segment["highway_type"] = segment2.get("highway_type", "unknown")

        return merged_segment


class MemoryEfficientMerger:
    """Memory-efficient road segment merger."""

    def __init__(self):
        self.way_lookup = {}  # For deduplication

    def process_way_chunk(self, ways_chunk: List, show_progress: bool = False) -> List[Dict]:
        """Process a chunk of ways and return merged segments.

        Args:
            ways_chunk: List of OSM ways to process
            show_progress: If True, show a blue progress bar for processing
        """
        # Deduplicate ways by ID to avoid processing the same way multiple times
        unique_ways = {}

        if show_progress:
            pbar = tqdm(
                total=len(ways_chunk),
                desc="  Deduplicating ways",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ascii=" ▏▎▍▌▋▊▉█",
                dynamic_ncols=True,
                colour="blue",
            )
            for way in ways_chunk:
                if hasattr(way, "id") and way.id not in unique_ways:
                    unique_ways[way.id] = way
                pbar.update(1)
            pbar.close()
        else:
            for way in ways_chunk:
                if hasattr(way, "id") and way.id not in unique_ways:
                    unique_ways[way.id] = way

        # Group by street name
        street_groups = defaultdict(list)

        if show_progress:
            pbar = tqdm(
                total=len(unique_ways),
                desc="  Grouping by street",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ascii=" ▏▎▍▌▋▊▉█",
                dynamic_ncols=True,
                colour="blue",
            )
            for way in unique_ways.values():
                street_name = self._get_street_name(way)
                if street_name:
                    street_groups[street_name].append(way)
                pbar.update(1)
            pbar.close()
        else:
            for way in unique_ways.values():
                street_name = self._get_street_name(way)
                if street_name:
                    street_groups[street_name].append(way)

        # Merge within groups
        merged_segments = []

        if show_progress:
            pbar = tqdm(
                total=len(street_groups),
                desc="  Merging segments",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ascii=" ▏▎▍▌▋▊▉█",
                dynamic_ncols=True,
                colour="blue",
            )
            for street_name, ways_group in street_groups.items():
                segments = self._merge_ways_in_group(ways_group)
                merged_segments.extend(segments)
                pbar.update(1)
            pbar.close()
        else:
            for street_name, ways_group in street_groups.items():
                segments = self._merge_ways_in_group(ways_group)
                merged_segments.extend(segments)

        # Clean up local variables (Python's reference counting handles this automatically)
        del unique_ways, street_groups
        # NOTE: Removed gc.collect() - it was scanning ALL objects in memory (140M+)
        # causing exponential slowdown (4.82s -> 21s/chunk). Python's reference
        # counting already frees these local variables without needing full GC scan.

        for segment in merged_segments:
            # Add start coordinates for efficient lookup later
            if segment["nodes"] and len(segment["nodes"]) > 0:
                first_node = segment["nodes"][0]
                if hasattr(first_node, "lat") and hasattr(first_node, "lon"):
                    segment["start_lat"] = float(first_node.lat)
                    segment["start_lon"] = float(first_node.lon)

        return merged_segments

    def _get_highway_type(self, way) -> str:
        """Extract highway type from OSM way."""
        if not hasattr(way, "tags") or not way.tags:
            return "unknown"

        highway_type = way.tags.get("highway", "").strip().lower()
        return highway_type if highway_type else "unknown"

    def _get_street_name(self, way) -> Optional[str]:
        """Extract street name."""
        if not hasattr(way, "tags") or not way.tags:
            return None

        name_tags = [
            "name",
            "ref",
            "addr:street",
            "tiger:name_base",
            "unsigned_ref",
            "highway",
        ]

        for tag in name_tags:
            if tag in way.tags and way.tags[tag]:
                return str(way.tags[tag]).strip()

        return None

    def _merge_ways_in_group(self, ways_group: List) -> List[Dict]:
        """Merge ways in a group."""
        if not ways_group:
            return []

        merged_segments = []
        used_ways = set()

        for start_way in ways_group:
            if start_way.id in used_ways:
                continue

            # Create merged segment
            segment = {
                "way_ids": [start_way.id],
                "nodes": start_way.nodes.copy() if start_way.nodes else [],
                "street_name": self._get_street_name(start_way),
                "surface": self._get_surface(start_way),
                "tracktype": self._get_tracktype(start_way)[0],
                "tracktype_definition": self._get_tracktype(start_way)[1],
                "cycling_access": self._get_cycling_access(start_way),
                "highway_type": self._get_highway_type(start_way),
            }
            used_ways.add(start_way.id)

            # Try to extend (simplified merging for memory efficiency)
            extended = True
            while extended and len(used_ways) < len(ways_group):
                extended = False

                # Collect and score potential connections (avoid dead-end spurs)
                candidates = []
                for way2 in ways_group:
                    if way2.id in used_ways:
                        continue

                    # Simple endpoint connection check
                    if self._can_connect_simple(segment["nodes"], way2.nodes):
                        # Count how many other ways way2 connects to (simple topology score)
                        way2_first = way2.nodes[0]
                        way2_last = way2.nodes[-1]
                        connection_count = 0

                        for other_way in ways_group:
                            if other_way.id == way2.id or other_way.id in used_ways:
                                continue
                            other_first = other_way.nodes[0]
                            other_last = other_way.nodes[-1]

                            # Check if way2's endpoints match other way's endpoints
                            if (
                                self._nodes_match(way2_first, other_first)
                                or self._nodes_match(way2_first, other_last)
                                or self._nodes_match(way2_last, other_first)
                                or self._nodes_match(way2_last, other_last)
                            ):
                                connection_count += 1

                        candidates.append((way2, connection_count))

                # Sort by connection count (prefer well-connected ways over spurs)
                candidates.sort(key=lambda x: x[1], reverse=True)

                # Try best candidate first
                for way2, conn_count in candidates:
                    segment["nodes"].extend(way2.nodes[1:])  # Simple append, skip duplicate
                    segment["way_ids"].append(way2.id)
                    used_ways.add(way2.id)
                    extended = True
                    break

            merged_segments.append(segment)

        cycling_access = self._get_cycling_access(start_way)

        return merged_segments

    def _can_connect_simple(self, segment_nodes: List, way_nodes: List) -> bool:
        """Simple connection check for memory efficiency."""
        if not segment_nodes or not way_nodes:
            return False

        # Check if last node of segment connects to first node of way
        if hasattr(segment_nodes[-1], "id") and hasattr(way_nodes[0], "id"):
            return segment_nodes[-1].id == way_nodes[0].id

        return False

    def _get_surface(self, way) -> str:
        """Extract surface information."""
        if not hasattr(way, "tags") or not way.tags:
            return "unknown"

        if "surface" in way.tags:
            return str(way.tags["surface"]).strip().lower()

        highway_type = way.tags.get("highway", "").strip().lower()
        surface_mapping = {
            "motorway": "asphalt",
            "trunk": "asphalt",
            "primary": "asphalt",
            "secondary": "asphalt",
            "tertiary": "asphalt",
            "unclassified": "paved",
            "residential": "asphalt",
            "service": "paved",
            "track": "unpaved",
            "path": "unpaved",
            "footway": "unpaved",
        }
        return surface_mapping.get(highway_type, "unknown")

    def _get_tracktype(self, way) -> Tuple[str, str]:
        """Extract tracktype information."""

        if not hasattr(way, "tags") or not way.tags:
            return "unknown", "No tracktype information"

        if way.tags.get("highway") != "track":
            return "not_applicable", "Not a track"

        if "tracktype" in way.tags:
            tracktype_raw = str(way.tags["tracktype"]).strip().lower()

            # Map OSM tracktype values to our definition keys
            osm_to_definition_key = {
                "grade1": "1-solid",
                "grade2": "2-gravel_rd",
                "grade3": "3-doubletrack",
                "grade4": "4-mostly_soft",
                "grade5": "5-unimproved",
                # Also handle numeric values without "grade" prefix
                "1": "1-solid",
                "2": "2-gravel_rd",
                "3": "3-doubletrack",
                "4": "4-mostly_soft",
                "5": "5-unimproved",
            }

            definition_key = osm_to_definition_key.get(tracktype_raw)

            if definition_key and definition_key in TRACKTYPE_DEFINITIONS:
                return (
                    definition_key,
                    TRACKTYPE_DEFINITIONS[definition_key],
                )  # Changed this line
            else:
                return tracktype_raw, f"Unknown tracktype: {tracktype_raw}"

        return "unspecified", "Track with unspecified surface quality"

    def _get_cycling_access(self, way) -> str:
        """Extract cycling access information from OSM way."""
        if not hasattr(way, "tags") or not way.tags:
            return "Unknown"

        highway = way.tags.get("highway", "").strip().lower()
        return determine_cycling_access(way.tags, highway)


def find_all_existing_analyses() -> Optional[str]:
    """Find all existing analyses at script startup and show recent 5 options."""
    import threading

    base_dir = CHECKPOINT_DIR

    # Create the directory if it doesn't exist
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create checkpoint directory {base_dir}: {e}")
        return None

    # Check if directory exists after creation attempt
    if not base_dir.exists():
        return None

    # Quick check: if directory is empty, return immediately
    try:
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return None
    except:
        return None

    # Show progress indicator for long-running scans
    print("Scanning checkpoint folder for information on past saved sessions", end="", flush=True)

    # Timer to print dots every 10 seconds
    stop_timer = threading.Event()

    def print_dots():
        while not stop_timer.is_set():
            if stop_timer.wait(10):  # Wait 10 seconds or until stopped
                break
            print(".", end="", flush=True)

    timer_thread = threading.Thread(target=print_dots, daemon=True)
    timer_thread.start()

    try:
        all_analyses = []
        for analysis_dir in subdirs:
            if not analysis_dir.is_dir():
                continue

            progress_file = analysis_dir / "progress.pkl"
            elevation_progress_file = analysis_dir / "elevation_progress.pkl"
            deduplication_progress_file = analysis_dir / "deduplication_progress.pkl"
            boundary_merge_progress_file = analysis_dir / "boundary_merge_progress.pkl"

            # Check for streaming mode checkpoint files
            filtered_ways_file = analysis_dir / "filtered_ways.jsonl"
            merged_segments_file = analysis_dir / "merged_segments.jsonl"
            segments_sorted_file = analysis_dir / "segments_sorted.jsonl"
            segments_checkpoint_file = analysis_dir / "segments.checkpoint.jsonl"

            # Check for either chunked mode or streaming mode checkpoints
            is_chunked = progress_file.exists()
            is_streaming = (
                filtered_ways_file.exists()
                or merged_segments_file.exists()
                or segments_sorted_file.exists()
                or segments_checkpoint_file.exists()
            )

            if is_chunked or is_streaming:
                try:
                    # Load chunk progress (for chunked mode)
                    if is_chunked:
                        with open(progress_file, "rb") as f:
                            progress_data = safe_pickle_load(f)

                        completed = len(progress_data.get("processed_chunks", []))
                        total = progress_data.get("total_chunks", 0)
                    else:
                        # Streaming mode - treat as single "chunk"
                        completed = 0
                        total = 1  # Treat streaming as 1 "chunk" that needs completion

                        # Check which stage we're in
                        if merged_segments_file.exists():
                            completed = 1  # Merging complete, ready for elevation
                        elif segments_sorted_file.exists():
                            completed = 0  # Sorting done, merging pending
                        elif filtered_ways_file.exists() or segments_checkpoint_file.exists():
                            completed = 0  # Filtering done, sorting/merging pending

                    # Load elevation progress
                    elevation_coords = 0
                    if elevation_progress_file.exists():
                        try:
                            with open(elevation_progress_file, "rb") as f:
                                elevation_data = safe_pickle_load(f)
                            # Check both in-memory and disk-based elevation storage
                            elevation_coords = len(elevation_data.get("coordinate_mapping", {}))
                            # If using disk-based storage, check elevations_fetched count
                            if elevation_coords == 0:
                                batch_info = elevation_data.get("batch_info", {})
                                elevation_coords = batch_info.get("elevations_fetched", 0)
                        except:
                            elevation_coords = 0

                    # Load deduplication progress
                    deduplication_in_progress = False
                    deduplication_info = ""
                    if deduplication_progress_file.exists():
                        try:
                            with open(deduplication_progress_file, "rb") as f:
                                dedupe_data = safe_pickle_load(f)

                            step1_complete = dedupe_data.get("step1_complete", False)
                            step1_in_progress = dedupe_data.get(
                                "step1_in_progress", False
                            )  # ADD THIS LINE
                            step2_progress = dedupe_data.get("step2_progress", 0)

                            # MODIFY THIS CONDITION TO INCLUDE step1_in_progress
                            if step1_complete or step1_in_progress or step2_progress > 0:
                                deduplication_in_progress = True

                                if step1_in_progress:
                                    # Handle interrupted Step 1
                                    processed = dedupe_data.get("step1_segments_processed", 0)
                                    total = dedupe_data.get("step1_total_segments", 0)
                                    if total > 0:
                                        progress_pct = (processed / total) * 100
                                        deduplication_info = f"Deduplication interrupted at {progress_pct:.1f}% ({processed}/{total})"
                                    else:
                                        deduplication_info = "Deduplicatoin interrupted"
                                elif step1_complete and step2_progress > 0:
                                    deduplication_info = f"Step 2: {step2_progress} segments"
                                elif step1_complete:
                                    deduplication_info = "Step 1 complete, Step 2 ready"
                                else:
                                    deduplication_info = f"Step 1: {step2_progress} segments"
                        except:
                            deduplication_in_progress = True
                            deduplication_info = "Status unknown"

                    # Check boundary merge progress
                    boundary_merge_in_progress = boundary_merge_progress_file.exists()

                    # Get file modification time for sorting
                    # For chunked mode: use progress.pkl
                    # For streaming mode: use the most recent checkpoint file
                    if is_chunked and progress_file.exists():
                        mod_time = progress_file.stat().st_mtime
                    elif elevation_progress_file.exists():
                        mod_time = elevation_progress_file.stat().st_mtime
                    elif merged_segments_file.exists():
                        mod_time = merged_segments_file.stat().st_mtime
                    elif segments_sorted_file.exists():
                        mod_time = segments_sorted_file.stat().st_mtime
                    elif filtered_ways_file.exists():
                        mod_time = filtered_ways_file.stat().st_mtime
                    elif segments_checkpoint_file.exists():
                        mod_time = segments_checkpoint_file.stat().st_mtime
                    else:
                        # Fallback to directory modification time
                        mod_time = analysis_dir.stat().st_mtime

                    # Consider analysis resumable
                    chunks_incomplete = total > 0 and completed < total
                    elevation_in_progress = elevation_coords > 0

                    if (
                        chunks_incomplete
                        or elevation_in_progress
                        or deduplication_in_progress
                        or boundary_merge_in_progress
                    ):
                        # Determine phase
                        if chunks_incomplete:
                            phase = "chunk_processing"
                            percentage = (completed / total) * 100 if total > 0 else 0.0
                        elif boundary_merge_in_progress:
                            phase = "boundary_merge"
                            percentage = 100.0
                        elif deduplication_in_progress:
                            phase = "deduplication"
                            percentage = 100.0
                        elif elevation_in_progress:
                            phase = "elevation_fetching"
                            percentage = 100.0
                        else:
                            phase = "unknown"
                            percentage = 0.0

                        all_analyses.append(
                            {
                                "id": analysis_dir.name,
                                "completed": completed,
                                "total": total,
                                "percentage": percentage,
                                "elevation_coords": elevation_coords,
                                "deduplication_in_progress": deduplication_in_progress,
                                "deduplication_info": deduplication_info,
                                "boundary_merge_in_progress": boundary_merge_in_progress,
                                "phase": phase,
                                "mod_time": mod_time,
                            }
                        )

                except Exception as e:
                    print(f"Error reading progress for {analysis_dir.name}: {e}")
                    continue

    finally:
        # Stop the timer thread
        stop_timer.set()
        timer_thread.join(timeout=1)
        # Print newline to move to next line after dots
        print()

    if all_analyses:
        # Sort by modification time (most recent first) and show top 5
        all_analyses.sort(key=lambda x: x["mod_time"], reverse=True)
        top_analyses = all_analyses[:5]

        print(f"\n{'=' * 80}")
        print(f"CHECKPOINT DETECTED: Found {len(all_analyses)} resumable analyses")
        print(f"{'=' * 80}")
        print(f"Showing most recent {len(top_analyses)}:")

        for i, analysis in enumerate(top_analyses, 1):
            print(f"\n{i}. Analysis ID: {analysis['id']}")

            if analysis["phase"] == "chunk_processing":
                print(f"   Status: Processing ({analysis['percentage']:.1f}% complete)")
            elif analysis["phase"] == "boundary_merge":
                print("   Status: Merging Road Segments")
            elif analysis["phase"] == "deduplication":
                print(f"   Status: Deduplication ({analysis['deduplication_info']})")
            elif analysis["phase"] == "elevation_fetching":
                print("   Status: Elevation Fetching")
                print(f"   Elevation: {analysis['elevation_coords']} coordinates")

            # Show last modified time
            mod_date = datetime.datetime.fromtimestamp(analysis["mod_time"])
            print(f"   Last Modified: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n0. Start new analysis")
        print(f"{'=' * 80}")

        while True:
            choice = input(f"\nSelect analysis to resume (0-{len(top_analyses)}): ").strip()

            if choice == "0" or not choice:
                return None

            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(top_analyses):
                    selected = top_analyses[choice_num - 1]
                    return selected["id"]
                else:
                    print(f"Please enter a number between 0 and {len(top_analyses)}")
            except ValueError:
                print("Please enter a valid number")

    return None


def find_matching_analysis_for_cli(
    region_name: str, surface_filter: str, scope_type: str, score_type: str
) -> Optional[str]:
    """
    Find existing analysis matching the specified parameters for CLI mode.
    Returns the most recent matching analysis ID, or None if not found.

    Args:
        region_name: Name of the region (e.g., "Luxembourg")
        surface_filter: Surface filter (e.g., "all", "paved", "unpaved")
        scope_type: Scope type (e.g., "country", "region")
        score_type: Scoring algorithm (e.g., "basic", "fiets", "pdi")

    Returns:
        Analysis ID of most recent matching analysis, or None
    """

    base_dir = CHECKPOINT_DIR
    if not base_dir.exists():
        return None

    # Build the base pattern (without timestamp)
    # Clean the region name same way as in main code
    clean_region = "".join(c for c in region_name if c.isalnum() or c in ("_", "-"))
    base_pattern = f"{clean_region}_{surface_filter}_{scope_type}"

    # Find all matching directories
    matching_analyses = []
    try:
        for analysis_dir in base_dir.iterdir():
            if not analysis_dir.is_dir():
                continue

            # Check if directory name starts with our pattern
            if analysis_dir.name.startswith(base_pattern + "_"):
                # Verify there's actual checkpoint data
                progress_file = analysis_dir / "progress.pkl"
                elevation_file = analysis_dir / "elevation_progress.pkl"

                if progress_file.exists() or elevation_file.exists():
                    # Extract timestamp from directory name
                    try:
                        timestamp_str = analysis_dir.name.split("_")[-1]
                        timestamp = int(timestamp_str)
                        matching_analyses.append((timestamp, analysis_dir.name))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Warning: Error scanning checkpoints: {e}")
        return None

    if not matching_analyses:
        return None

    # Sort by timestamp (most recent first) and return the most recent
    matching_analyses.sort(reverse=True)
    return matching_analyses[0][1]


def resume_analysis_from_startup(analysis_id: str):
    """Resume analysis directly from startup with default parameters."""

    print(f"\n{'=' * 80}")
    print(f"RESUMING ANALYSIS: {analysis_id}")
    print(f"{'=' * 80}")

    # Detect checkpoint type (streaming vs chunked)
    from pathlib import Path

    checkpoint_dir = CHECKPOINT_DIR / analysis_id
    is_streaming = (
        (checkpoint_dir / "filtered_ways.jsonl").exists()
        or (checkpoint_dir / "merged_segments.jsonl").exists()
        or (checkpoint_dir / "segments.checkpoint.jsonl").exists()
    )
    is_chunked = (checkpoint_dir / "progress.pkl").exists()

    if is_streaming and not is_chunked:
        # Streaming mode checkpoint - can't resume from interactive menu
        print("\n⚠️  This is a streaming mode checkpoint.")
        print(
            "Streaming mode checkpoints resume automatically when you re-run the original command."
        )
        print("\nTo continue this analysis:")
        print("  1. Exit this menu (enter 0)")
        print("  2. Re-run with the same region/parameters")
        print("  3. It will automatically resume from checkpoint\n")
        return

    # Load metadata to get original parameters (chunked mode)
    persistence = ChunkPersistenceManager(analysis_id)
    _, _, metadata = persistence.load_progress()

    # Extract original parameters from metadata
    min_score = 0  # Default, can be overridden later
    unit_system = metadata.get("unit_system", "imperial")
    score_type = metadata.get("score_type", "basic")

    # Show original analysis parameters
    print("Original parameters:")
    print(f"  - Address: {metadata.get('address', 'Unknown')}")
    print(f"  - Surface filter: {metadata.get('surface_filter', 'unknown')}")
    print(f"  - Radius: {metadata.get('radius_km', 'unknown')} km")
    print(f"  - Unit system: {unit_system}")
    print(f"  - Score type: {score_type}")

    # Allow user to adjust minimum score if desired
    try:
        default_score = "0"
        if score_type == "basic":
            score_desc = "basic score (elevation*distance)"
        elif score_type == "fiets":
            score_desc = "FIETS score"
        else:  # pdi
            score_desc = "PDI score"

        min_score_input = input(
            f"\nMinimum {score_desc} to display (default {default_score}, press Enter to use default): "
        ).strip()
        if min_score_input:
            min_score = float(min_score_input)
        else:
            min_score = float(default_score)
    except ValueError:
        min_score = 0

    print(f"\nResuming analysis with minimum score: {min_score}")

    # Call resume function directly
    try:
        total_start_time = time.time()
        # climbs, df, persistence = resume_chunk_processing(analysis_id, min_score, unit_system, score_type)
        climbs, df, persistence, should_upload_to_cache = resume_chunk_processing_parallel(
            analysis_id, min_score, unit_system, score_type, max_workers=OSM_MAX_THREADS
        )

        total_duration = time.time() - total_start_time
        print(f"\nTOTAL PROCESSING TIME: {total_duration:.2f} seconds")

        # Automatically save results to XLSX in output folder
        if df is not None and len(df) > 0:
            # Create output folder if it doesn't exist and ensure write permissions
            output_dir = Path("output")
            try:
                output_dir.mkdir(exist_ok=True, mode=0o777)
                # Ensure the directory is writable
                import os

                os.chmod(output_dir, 0o777)
            except Exception as e:
                print(f"⚠️ Could not create/modify output folder: {e}")
                # Try /tmp as fallback
                import tempfile

                output_dir = Path(tempfile.gettempdir())
                print(f"   Using temporary directory instead: {output_dir}")

            address = metadata.get("address", "unknown")
            surface_filter = metadata.get("surface_filter", "all")
            radius_km = metadata.get("radius_km", 0)

            safe_name = "".join(c for c in address if c.isalnum() or c in (" ", "-", "_")).rstrip()
            safe_name = safe_name.replace(" ", "_")[:50]

            # Get elevation error count from stats collector
            elevation_errors = 0
            try:
                from utils.elevation_stats_collector import get_stats_collector, has_elevation_stats

                if has_elevation_stats():
                    stats_collector = get_stats_collector()
                    stats = stats_collector.get_stats()
                    elevation_errors = stats.get("total_coords_failed", 0)
            except Exception:
                elevation_errors = 0

            # Save results (will split into multiple files if exceeds Excel limit)
            base_filename = (
                output_dir
                / f"climbs_{safe_name}_{surface_filter}_{score_type}_resumed_{radius_km:.0f}km"
            )
            created_files = save_large_dataframe_as_split_excel(
                df, base_filename, app_version=__version__, elevation_errors=elevation_errors
            )

            if created_files:
                if len(created_files) == 1:
                    print(f"Results saved to {created_files[0]}")
                else:
                    print(f"Results saved to {len(created_files)} files:")
                    for file in created_files:
                        print(f"   {file.name}")
            else:
                print("⚠️ Could not save results")
                print("   Results are still in memory and can be accessed programmatically")
        else:
            print("No results to save (no climbs found above minimum score)")

        # Cleanup - default to keeping checkpoints unless --delete-checkpoints is specified
        # (No interactive prompt - use --delete-checkpoints flag to enable cleanup)
        print("\n✓ Checkpoint files preserved for potential resume")
        print("  Use --delete-checkpoints flag to auto-delete checkpoints after completion")

        cleanup_choice = "3"  # Default: keep checkpoints
        if cleanup_choice == "1":
            persistence.cleanup()
        elif cleanup_choice == "2":
            try:
                base_dir = CHECKPOINT_DIR
                if base_dir.exists():
                    for item in base_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    print("Cleaned up all files in checkpoint directories")
                else:
                    print("No checkpoint directory found")
            except Exception as e:
                print(f"Error cleaning up checkpoint contents: {e}")

    except Exception as e:
        print(f"Error during resumed analysis: {e}")
        traceback.print_exc()


def find_existing_analysis(
    base_id: str, base_dir: Path, auto_resume: bool = False
) -> Optional[str]:
    """
    Find existing analyses and show recent 5 options.

    Args:
        base_id: Base analysis ID pattern to match
        base_dir: Checkpoints directory
        auto_resume: If True, automatically resume most recent match without prompting

    Returns:
        Analysis ID to resume, or None to start fresh
    """

    # Create the directory if it doesn't exist
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create checkpoint directory {base_dir}: {e}")
        return None

    # Check if directory exists after creation attempt
    if not base_dir.exists():
        return None

    matching_analyses = []
    for analysis_dir in base_dir.iterdir():

        # Case-insensitive matching for checkpoint directories
        if analysis_dir.is_dir() and analysis_dir.name.lower().startswith(base_id.lower() + "_"):
            progress_file = analysis_dir / "progress.pkl"
            elevation_progress_file = analysis_dir / "elevation_progress.pkl"
            deduplication_progress_file = analysis_dir / "deduplication_progress.pkl"

            # Also check for streaming mode checkpoint files
            filtered_ways_file = analysis_dir / "filtered_ways.jsonl"
            merged_segments_file = analysis_dir / "merged_segments.jsonl"
            segments_sorted_file = analysis_dir / "segments_sorted.jsonl"
            elevation_db_file = analysis_dir / "elevation_data.db"

            # Check for either chunked mode (progress.pkl) or streaming mode (.jsonl files or elevation db)
            is_chunked_mode = progress_file.exists()
            is_streaming_mode = (
                filtered_ways_file.exists()
                or merged_segments_file.exists()
                or segments_sorted_file.exists()
                or elevation_db_file.exists()
            )

            # Also check for segments.checkpoint.jsonl (streaming mode intermediate checkpoint)
            segments_checkpoint_file = analysis_dir / "segments.checkpoint.jsonl"
            if segments_checkpoint_file.exists():
                is_streaming_mode = True

            if is_chunked_mode or is_streaming_mode:
                try:
                    # Load chunk progress (for chunked mode)
                    if is_chunked_mode:
                        with open(progress_file, "rb") as f:
                            progress_data = safe_pickle_load(f)

                        completed = len(progress_data.get("processed_chunks", []))
                        total = progress_data.get("total_chunks", 0)
                    else:
                        # Streaming mode - treat as single "chunk"
                        completed = 0
                        total = 1  # Treat streaming as 1 "chunk" that needs completion

                        # Check which stage we're in
                        if merged_segments_file.exists():
                            completed = 1  # Merging complete, ready for elevation
                        elif segments_sorted_file.exists():
                            completed = 0  # Sorting done, merging pending
                        elif filtered_ways_file.exists():
                            completed = 0  # Filtering done, sorting/merging pending

                    # Load elevation progress
                    elevation_coords = 0
                    if elevation_progress_file.exists():
                        try:
                            with open(elevation_progress_file, "rb") as f:
                                elevation_data = safe_pickle_load(f)
                            # Check both in-memory and disk-based elevation storage
                            elevation_coords = len(elevation_data.get("coordinate_mapping", {}))
                            # If using disk-based storage, check elevations_fetched count
                            if elevation_coords == 0:
                                batch_info = elevation_data.get("batch_info", {})
                                elevation_coords = batch_info.get("elevations_fetched", 0)
                        except:
                            elevation_coords = 0

                    # If .pkl didn't have elevation count, check the database file
                    if elevation_coords == 0 and elevation_db_file.exists():
                        # Count entries in the database (streaming mode)
                        try:
                            # BUGFIX: Use SQLite directly instead of shelve
                            import sqlite3

                            conn = sqlite3.connect(str(elevation_db_file))
                            cursor = conn.execute("SELECT COUNT(*) FROM elevations")
                            elevation_coords = cursor.fetchone()[0]
                            conn.close()
                        except Exception as e:
                            # Log the error so we can debug why resume isn't working
                            print(
                                f"⚠️  Could not read elevation database for {analysis_dir.name}: {e}"
                            )
                            elevation_coords = 0

                    # Load deduplication progress
                    deduplication_in_progress = False
                    deduplication_info = ""
                    if deduplication_progress_file.exists():
                        try:
                            with open(deduplication_progress_file, "rb") as f:
                                dedupe_data = safe_pickle_load(f)

                            step1_complete = dedupe_data.get("step1_complete", False)
                            step1_in_progress = dedupe_data.get(
                                "step1_in_progress", False
                            )  # ADD THIS LINE
                            step2_progress = dedupe_data.get("step2_progress", 0)

                            # MODIFY THIS CONDITION TO INCLUDE step1_in_progress
                            if step1_complete or step1_in_progress or step2_progress > 0:
                                deduplication_in_progress = True

                                if step1_in_progress:
                                    # Handle interrupted Step 1
                                    processed = dedupe_data.get("step1_segments_processed", 0)
                                    total = dedupe_data.get("step1_total_segments", 0)
                                    if total > 0:
                                        progress_pct = (processed / total) * 100
                                        deduplication_info = f"Step 1 interrupted at {progress_pct:.1f}% ({processed}/{total})"
                                    else:
                                        deduplication_info = "Step 1 interrupted"
                                elif step1_complete and step2_progress > 0:
                                    deduplication_info = f"Step 2: {step2_progress} segments"
                                elif step1_complete:
                                    deduplication_info = "Step 1 complete, Step 2 ready"
                                else:
                                    deduplication_info = f"Step 1: {step2_progress} segments"
                        except:
                            deduplication_in_progress = True
                            deduplication_info = "Status unknown"

                    # Get file modification time for sorting
                    # For chunked mode: use progress.pkl
                    # For streaming mode: use the most recent checkpoint file
                    if is_chunked_mode and progress_file.exists():
                        mod_time = progress_file.stat().st_mtime
                    elif elevation_progress_file.exists():
                        mod_time = elevation_progress_file.stat().st_mtime
                    elif merged_segments_file.exists():
                        mod_time = merged_segments_file.stat().st_mtime
                    elif segments_sorted_file.exists():
                        mod_time = segments_sorted_file.stat().st_mtime
                    elif filtered_ways_file.exists():
                        mod_time = filtered_ways_file.stat().st_mtime
                    else:
                        # Fallback to directory modification time
                        mod_time = analysis_dir.stat().st_mtime

                    # Consider analysis resumable
                    chunks_incomplete = total > 0 and completed < total
                    elevation_in_progress = elevation_coords > 0

                    if chunks_incomplete or elevation_in_progress or deduplication_in_progress:
                        # Determine phase
                        if chunks_incomplete:
                            phase = "chunk_processing"
                            percentage = (completed / total) * 100 if total > 0 else 0.0
                        elif deduplication_in_progress:
                            phase = "deduplication"
                            percentage = 100.0
                        elif elevation_in_progress:
                            phase = "elevation_fetching"
                            percentage = 100.0
                        else:
                            phase = "unknown"
                            percentage = 0.0

                        matching_analyses.append(
                            {
                                "id": analysis_dir.name,
                                "completed": completed,
                                "total": total,
                                "percentage": percentage,
                                "elevation_coords": elevation_coords,
                                "deduplication_in_progress": deduplication_in_progress,
                                "deduplication_info": deduplication_info,
                                "phase": phase,
                                "mod_time": mod_time,
                            }
                        )

                except Exception as e:
                    # Silent for streaming mode checkpoints without progress.pkl (expected)
                    if not (is_streaming_mode and "progress.pkl" in str(e)):
                        print(f"Error reading progress for {analysis_dir.name}: {e}")
                    continue

    if matching_analyses:
        # Sort by modification time (most recent first) and show top 5
        matching_analyses.sort(key=lambda x: x["mod_time"], reverse=True)
        top_analyses = matching_analyses[:5]

        # Auto-resume mode: automatically resume most recent without prompting
        if auto_resume:
            most_recent = top_analyses[0]
            # Show compact checkpoint status
            if most_recent["phase"] == "chunk_processing":
                status = f"chunk processing ({most_recent['percentage']:.0f}%)"
            elif most_recent["phase"] == "deduplication":
                status = "deduplication"
            elif most_recent["phase"] == "elevation_fetching":
                status = f"elevation fetching ({most_recent['elevation_coords']:,} coords)"
            else:
                status = most_recent["phase"]
            print(f"\n✓ Resuming from checkpoint: {status}")
            return most_recent["id"]

        # Interactive mode: show options and prompt user
        print(
            f"\nFound {len(matching_analyses)} resumable analyses (showing most recent {len(top_analyses)}):"
        )

        for i, analysis in enumerate(top_analyses, 1):
            print(f"\n{i}. Analysis ID: {analysis['id']}")

            if analysis["phase"] == "chunk_processing":
                print(f"   Status: Processing ({analysis['percentage']:.1f}% complete)")
            elif analysis["phase"] == "deduplication":
                print(f"   Status: Deduplication ({analysis['deduplication_info']})")
            elif analysis["phase"] == "elevation_fetching":
                print("   Status: Elevation Fetching")
                print(f"   Elevation: {analysis['elevation_coords']} coordinates")

            # Show last modified time
            mod_date = datetime.datetime.fromtimestamp(analysis["mod_time"])
            print(f"   Last Modified: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n0. Start new analysis")

        while True:
            choice = input(
                f"\nSelect analysis to resume (0-{len(top_analyses)}, default 0): "
            ).strip()

            if not choice or choice == "0":
                return None

            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(top_analyses):
                    selected = top_analyses[choice_num - 1]
                    return selected["id"]
                else:
                    print(f"Please enter a number between 0 and {len(top_analyses)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        if not auto_resume:
            print("\nNo resumable analyses found")

    return None


def resume_chunk_processing(analysis_id: str, min_score: float, unit_system: str, score_type: str):
    """Resume chunk processing from where it left off."""

    persistence = ChunkPersistenceManager(analysis_id)
    processed_chunks, total_chunks, metadata = persistence.load_progress()
    elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()

    print(f"  - Elevation mapping: {len(elevation_mapping)} coordinates")

    # Check if all chunks are complete
    if len(processed_chunks) >= total_chunks:
        print("Processing complete, checking elevation status...")

        if elevation_mapping:
            print("Found elevation progress.")
            print(f"  - Elevation coordinates: {len(elevation_mapping)}")
            print(f"  - Batch info: {elevation_batch_info}")

            # Load all chunk data
            all_merged_segments = []
            print("Loading all processed chunks...")
            with tqdm(
                total=len(processed_chunks),
                desc="Loading saved chunks",
                unit="chunk",
                dynamic_ncols=True,
                ascii=" ▏▎▍▌▋▊▉█",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ) as pbar:
                for chunk_index in processed_chunks:
                    chunk_data = persistence.load_chunk(chunk_index)
                    if chunk_data:
                        all_merged_segments.extend(chunk_data)
                    pbar.update(1)

            print(f"Loaded {len(all_merged_segments)} road segments from all chunks")

            # Get deployment type from metadata
            deployment_type = metadata.get("deployment_type", "cloud")
            use_local_elevation = metadata.get("use_local_elevation", False)

            # Get region name for dataset priority selection
            region_name = get_region_name_from_config()
            elevation_fetcher = FastElevationFetcher(region_name=region_name)
            boundary_merger = BoundaryMerger(coordinate_tolerance=0.002)

            return complete_analysis_from_segments(
                all_merged_segments,
                elevation_fetcher,
                boundary_merger,
                metadata,
                min_score,
                unit_system,
                persistence,
                score_type,
                is_resumed=True,
            )
        else:
            print("INFO: No elevation progress, starting fresh elevation analysis")

            # Load all chunk data and start elevation
            all_merged_segments = []
            print("Loading all processed chunks...")
            with tqdm(
                total=len(processed_chunks),
                desc="Loading saved chunks",
                unit="chunk",
                dynamic_ncols=True,
                ascii=" ▏▎▍▌▋▊▉█",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ) as pbar:
                for chunk_index in processed_chunks:
                    chunk_data = persistence.load_chunk(chunk_index)
                    if chunk_data:
                        all_merged_segments.extend(chunk_data)
                    pbar.update(1)

            print(f"Loaded {len(all_merged_segments)} road segments from all chunks")

            # Initialize components
            # Get region name for dataset priority selection
            region_name = get_region_name_from_config()
            elevation_fetcher = FastElevationFetcher(region_name=region_name)
            boundary_merger = BoundaryMerger(coordinate_tolerance=0.002)

            return complete_analysis_from_segments(
                all_merged_segments,
                elevation_fetcher,
                boundary_merger,
                metadata,
                min_score,
                unit_system,
                persistence,
                score_type,
                is_resumed=True,
            )

    # If chunks are not complete, continue processing
    print("Resuming processing...")

    # Load existing data
    all_merged_segments = []
    print("Loading previously processed data...")
    with tqdm(
        total=len(processed_chunks),
        desc="Loading saved chunks",
        unit="chunk",
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for chunk_index in processed_chunks:
            chunk_data = persistence.load_chunk(chunk_index)
            if chunk_data:
                all_merged_segments.extend(chunk_data)
            pbar.update(1)
            if chunk_index % 100 == 0:
                check_and_cleanup_memory()
    print(
        f"Loaded {len(all_merged_segments)} road segments from {len(processed_chunks)} completed chunks"
    )
    check_and_cleanup_memory(force_cleanup=True)

    # Continue processing remaining chunks
    remaining_chunks = total_chunks - len(processed_chunks)

    # Rebuild components and continue
    chunks = metadata.get("chunks", [])
    surface_filter = metadata.get("surface_filter", "all")
    chunk_size_km = metadata.get("chunk_size_km", 30.0)  # Default chunk size

    road_analyzer = ChunkedRoadNetworkAnalyzer(surface_filter, chunk_size_km)
    chunk_merger = MemoryEfficientMerger()
    boundary_merger = BoundaryMerger(coordinate_tolerance=0.002)
    # Get region name for dataset priority selection
    region_name = get_region_name_from_config()
    elevation_fetcher = FastElevationFetcher(region_name=region_name)

    return continue_chunk_processing(
        persistence,
        road_analyzer,
        chunk_merger,
        boundary_merger,
        elevation_fetcher,
        chunks,
        metadata,
        all_merged_segments,
        processed_chunks,
        min_score,
        unit_system,
        score_type,
        True,
    )


def continue_chunk_processing(
    persistence,
    road_analyzer,
    chunk_merger,
    boundary_merger,
    elevation_fetcher,
    chunks,
    metadata,
    existing_segments,
    processed_chunks,
    min_score,
    unit_system,
    score_type,
    enable_geocoding,
):
    """Continue processing chunks from where we left off (serial version)."""

    total_chunks = len(chunks)
    processed_set = set(processed_chunks)
    # MEMORY FIX: Don't accumulate segments in memory - track count only
    total_segments_count = 0

    print("Continuing chunk processing with serial mode (checkpoint saving enabled)...")

    # Disable automatic GC for performance
    gc.disable()

    # Initialize smart checkpointer
    checkpointer = SmartCheckpointer(total_chunks, "Resume Chunk Processing")

    # Track processing start time for seg/s calculation
    processing_start_time = time.time()

    remaining_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in processed_set]

    with tqdm(
        total=len(remaining_chunks),
        desc="Processing remaining chunks",
        unit="chunk",
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
    ) as pbar:
        for chunk_index, (chunk_lat, chunk_lon, chunk_radius) in remaining_chunks:
            try:
                chunk_ways = road_analyzer.get_roads_in_chunk(chunk_lat, chunk_lon, chunk_radius)

                if chunk_ways:
                    chunk_segments = chunk_merger.process_way_chunk(chunk_ways)
                    persistence.save_chunk(
                        chunk_index,
                        chunk_segments,
                        (chunk_lat, chunk_lon, chunk_radius),
                    )
                    total_segments_count += len(chunk_segments)
                else:
                    # Save empty chunk to mark as processed
                    persistence.save_chunk(chunk_index, [], (chunk_lat, chunk_lon, chunk_radius))

                processed_chunks.append(chunk_index)

                # Calculate segments per second
                elapsed_time = time.time() - processing_start_time
                seg_per_sec = total_segments_count / elapsed_time if elapsed_time > 0 else 0

                # Smart checkpoint save (not every chunk)
                if checkpointer.should_checkpoint(chunk_index):
                    persistence.save_progress(processed_chunks, total_chunks, metadata)

                    info = checkpointer.get_checkpoint_info(chunk_index)
                    pbar.set_postfix(
                        {
                            "total_segments": total_segments_count,
                            "seg/s": f"{seg_per_sec:.0f}",
                            "completed": len(processed_chunks),
                            "next_save": f"{info['time_until_next_min']:.1f}min",
                        }
                    )
                else:
                    pbar.set_postfix(
                        {
                            "total_segments": total_segments_count,
                            "seg/s": f"{seg_per_sec:.0f}",
                            "completed": len(processed_chunks),
                        }
                    )

                # Manual GC every 10 chunks for memory management
                if len(processed_chunks) % 10 == 0:
                    gc.collect()

                pbar.update(1)

            except Exception as e:
                print(f"\nError processing chunk {chunk_index}: {e}")
                print("Progress has been saved. You can resume this analysis later.")
                raise

    # Re-enable automatic GC
    gc.enable()

    print(f"Completed all chunks! Total segments: {total_segments_count}")

    # Load all segments from disk (they were saved chunk by chunk)
    all_merged_segments = persistence.load_all_segments()

    # Now proceed to elevation analysis
    return complete_analysis_from_segments(
        all_merged_segments,
        elevation_fetcher,
        boundary_merger,
        metadata,
        min_score,
        unit_system,
        persistence,
        score_type,
        enable_geocoding,
    )


def process_all_chunks(
    persistence,
    road_analyzer,
    chunk_merger,
    boundary_merger,
    elevation_fetcher,
    chunks,
    metadata,
    min_score,
    unit_system,
    score_type,
    enable_geocoding,
):
    # Setup signal handling
    signal_handler.set_persistence_manager(persistence)
    total_chunks = len(chunks)
    processed_chunks = []
    total_segments_count = 0  # Track count instead of storing all segments

    print("Processing road data in chunks with checkpoint saving...")

    # ... existing code ...

    with tqdm(
        total=total_chunks,
        desc="Processing chunks",
        unit="chunk",
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for chunk_index, (chunk_lat, chunk_lon, chunk_radius) in enumerate(chunks):
            # Signal handler check
            signal_handler.set_operation(
                "chunk_processing",
                {
                    "processed_chunks": processed_chunks,
                    "total_chunks": total_chunks,
                    "metadata": metadata,
                },
            )

            if signal_handler.kill_now:
                persistence.save_progress(processed_chunks, total_chunks, metadata)
                print("Chunk progress saved. Analysis can be resumed.")
                sys.exit(0)

            try:
                chunk_ways = road_analyzer.get_roads_in_chunk(chunk_lat, chunk_lon, chunk_radius)

                if chunk_ways:
                    chunk_segments = chunk_merger.process_way_chunk(chunk_ways)
                    persistence.save_chunk(
                        chunk_index,
                        chunk_segments,
                        (chunk_lat, chunk_lon, chunk_radius),
                    )
                    total_segments_count += len(chunk_segments)
                else:
                    # Save empty chunk to mark as processed
                    persistence.save_chunk(chunk_index, [], (chunk_lat, chunk_lon, chunk_radius))

                processed_chunks.append(chunk_index)

                # Save progress
                persistence.save_progress(processed_chunks, total_chunks, metadata)

                pbar.set_postfix({"status": "processed", "total_segments": total_segments_count})
                pbar.update(1)

                # Clean up memory
                del chunk_ways
                check_and_cleanup_memory()

                # Small delay to be respectful to the API
                time.sleep(0.5)

            except Exception as e:
                print(f"\nError processing chunk {chunk_index}: {e}")
                print("Progress has been saved. You can resume this analysis later.")
                raise

    print(f"\nCollected {total_segments_count} road segments from all chunks")

    # Load all segments from disk (they were saved chunk by chunk)
    all_merged_segments = persistence.load_all_segments()

    return complete_analysis_from_segments(
        all_merged_segments,
        elevation_fetcher,
        boundary_merger,
        metadata,
        min_score,
        unit_system,
        persistence,
        score_type,
        enable_geocoding,
    )


def _get_or_init_way_boundaries(segment: Dict) -> List[Tuple[int, int, int]]:
    """
    Get way_boundaries from a segment, initializing if not present.

    way_boundaries tracks which node indices belong to which way_id.
    Format: [(start_idx, end_idx, way_id), ...]

    Args:
        segment: Segment dict with 'nodes' and 'way_ids' keys

    Returns:
        List of (start_idx, end_idx, way_id) tuples
    """
    bounds = segment.get("way_boundaries")
    if bounds is not None:
        return list(bounds)  # Return copy to avoid mutation

    # Initialize from way_ids - ALL ways cover full range (when bounds unknown)
    # Fix: was only returning first way_id, now returns all (matches merger.py Fix #12)
    way_ids = segment.get("way_ids", [])
    nodes = segment.get("nodes", [])
    if way_ids:
        return [(0, len(nodes), wid) for wid in way_ids]
    return []


def merge_adjacent_ways_simple(ways: List[Dict]) -> List[Dict]:
    """
    Simple merging of adjacent ways on the same street.

    Merges ways where endpoints are within 200m of each other.
    Much simpler than boundary_merger - designed for whole-region extraction.

    Args:
        ways: List of way segments on the same street

    Returns:
        List of merged segments
    """
    if len(ways) <= 1:
        return ways

    # Build endpoint index for fast lookups
    endpoint_index = {}
    for idx, way in enumerate(ways):
        nodes = way.get("nodes", [])
        if len(nodes) < 2:
            continue

        start = nodes[0]
        end = nodes[-1]

        # Round to ~10m precision for matching
        start_coord = (round(start["lat"], 4), round(start["lon"], 4))
        end_coord = (round(end["lat"], 4), round(end["lon"], 4))

        if start_coord not in endpoint_index:
            endpoint_index[start_coord] = []
        if end_coord not in endpoint_index:
            endpoint_index[end_coord] = []

        endpoint_index[start_coord].append((idx, "start"))
        endpoint_index[end_coord].append((idx, "end"))

    # Merge connected ways
    merged = []
    used = set()

    for start_idx, way in enumerate(ways):
        if start_idx in used:
            continue

        nodes = way.get("nodes", [])
        if len(nodes) < 2:
            merged.append(way)
            used.add(start_idx)
            continue

        # Start building merged segment
        current = way.copy()
        used.add(start_idx)

        # Try to extend at both ends
        changed = True
        while changed:
            changed = False
            current_nodes = current["nodes"]

            if len(current_nodes) < 2:
                break

            start_coord = (round(current_nodes[0]["lat"], 4), round(current_nodes[0]["lon"], 4))
            end_coord = (round(current_nodes[-1]["lat"], 4), round(current_nodes[-1]["lon"], 4))

            # Try to connect at end
            for neighbor_idx, neighbor_end in endpoint_index.get(end_coord, []):
                if neighbor_idx in used:
                    continue

                neighbor = ways[neighbor_idx]
                neighbor_nodes = neighbor.get("nodes", [])

                if neighbor_end == "start":
                    # end → neighbor start: append neighbor (skip first node)
                    # Track way_boundaries
                    current_bounds = _get_or_init_way_boundaries(current)
                    neighbor_bounds = _get_or_init_way_boundaries(neighbor)
                    current_len = len(current_nodes)

                    # Neighbor indices shift: original i → current_len + (i - 1)
                    # Skip node 0, so start from max(s, 1)
                    adjusted_neighbor = []
                    for s, e, wid in neighbor_bounds:
                        new_s = max(s, 1) + current_len - 1
                        new_e = e - 1 + current_len
                        if new_s < new_e:
                            adjusted_neighbor.append((new_s, new_e, wid))

                    current["nodes"] = current_nodes + neighbor_nodes[1:]
                    current["way_ids"] = current.get("way_ids", []) + neighbor.get("way_ids", [])
                    current["way_boundaries"] = current_bounds + adjusted_neighbor
                    used.add(neighbor_idx)
                    changed = True
                    break
                elif neighbor_end == "end":
                    # end → neighbor end: append reversed neighbor (skip last node)
                    # Track way_boundaries with reversal
                    current_bounds = _get_or_init_way_boundaries(current)
                    neighbor_bounds = _get_or_init_way_boundaries(neighbor)
                    current_len = len(current_nodes)
                    neighbor_len = len(neighbor_nodes)

                    # Reversed: original index i → current_len + (neighbor_len - 2 - i)
                    # Skip last node (neighbor_len - 1)
                    adjusted_neighbor = []
                    for s, e, wid in neighbor_bounds:
                        e_clamped = min(e, neighbor_len - 1)
                        if s >= e_clamped:
                            continue
                        new_start = current_len + (neighbor_len - 2 - (e_clamped - 1))
                        new_end = current_len + (neighbor_len - 2 - s) + 1
                        adjusted_neighbor.append((new_start, new_end, wid))

                    adjusted_neighbor.sort(key=lambda x: x[0])

                    current["nodes"] = current_nodes + neighbor_nodes[-2::-1]
                    current["way_ids"] = current.get("way_ids", []) + neighbor.get("way_ids", [])
                    current["way_boundaries"] = current_bounds + adjusted_neighbor
                    used.add(neighbor_idx)
                    changed = True
                    break

            if changed:
                continue

            # Try to connect at start
            for neighbor_idx, neighbor_end in endpoint_index.get(start_coord, []):
                if neighbor_idx in used:
                    continue

                neighbor = ways[neighbor_idx]
                neighbor_nodes = neighbor.get("nodes", [])

                if neighbor_end == "end":
                    # neighbor end → start: prepend neighbor (skip last node)
                    # Track way_boundaries
                    neighbor_bounds = _get_or_init_way_boundaries(neighbor)
                    current_bounds = _get_or_init_way_boundaries(current)
                    neighbor_len = len(neighbor_nodes)
                    prepend_len = neighbor_len - 1  # Skip last node

                    # Trim neighbor bounds to prepend_len
                    trimmed_neighbor = []
                    for s, e, wid in neighbor_bounds:
                        new_e = min(e, prepend_len)
                        if s < new_e:
                            trimmed_neighbor.append((s, new_e, wid))

                    # Shift current bounds by prepend_len
                    shifted_current = [(s + prepend_len, e + prepend_len, wid)
                                       for s, e, wid in current_bounds]

                    current["nodes"] = neighbor_nodes[:-1] + current_nodes
                    current["way_ids"] = neighbor.get("way_ids", []) + current.get("way_ids", [])
                    current["way_boundaries"] = trimmed_neighbor + shifted_current
                    used.add(neighbor_idx)
                    changed = True
                    break
                elif neighbor_end == "start":
                    # neighbor start → start: prepend reversed neighbor (skip first node)
                    # Track way_boundaries with reversal
                    neighbor_bounds = _get_or_init_way_boundaries(neighbor)
                    current_bounds = _get_or_init_way_boundaries(current)
                    neighbor_len = len(neighbor_nodes)
                    prepend_len = neighbor_len - 1  # Skip first node

                    # Reverse and adjust neighbor bounds (skip node 0)
                    reversed_neighbor = []
                    for s, e, wid in neighbor_bounds:
                        s_clamped = max(s, 1)  # Skip node 0
                        if s_clamped >= e:
                            continue
                        # Reversed: index i → prepend_len - i
                        new_start = prepend_len - e + 1
                        new_end = prepend_len - s_clamped + 1
                        reversed_neighbor.append((new_start, new_end, wid))

                    reversed_neighbor.sort(key=lambda x: x[0])

                    # Shift current bounds by prepend_len
                    shifted_current = [(s + prepend_len, e + prepend_len, wid)
                                       for s, e, wid in current_bounds]

                    current["nodes"] = neighbor_nodes[:0:-1] + current_nodes
                    current["way_ids"] = neighbor.get("way_ids", []) + current.get("way_ids", [])
                    current["way_boundaries"] = reversed_neighbor + shifted_current
                    used.add(neighbor_idx)
                    changed = True
                    break

        merged.append(current)

    return merged


def merge_large_street_chunked(segments: List[Dict], chunk_size: int = 1000) -> List[Dict]:
    """
    Merge large streets in chunks to avoid memory exhaustion, then iteratively
    merge across boundaries until no more merges are possible.

    Strategy:
    1. Sort segments geographically (by start coordinate)
    2. Process in batches of chunk_size, merging within each batch
    3. Iteratively try to merge across boundaries using only endpoint data
    4. Stop when no more boundary merges are possible

    This limits memory to ~chunk_size segments during initial merge, then only
    uses lightweight endpoint data for boundary merging.

    Args:
        segments: List of segments for a large street
        chunk_size: Number of segments to process per chunk (default 1000)

    Returns:
        List of merged segments
    """
    if len(segments) <= chunk_size:
        return merge_adjacent_ways_simple(segments)

    # Sort by geographic location (lat, lon of first node) so nearby segments are together
    segments.sort(
        key=lambda s: (
            (round(s["nodes"][0]["lat"], 2), round(s["nodes"][0]["lon"], 2))
            if s.get("nodes") and len(s["nodes"]) > 0
            else (0, 0)
        )
    )

    # Phase 1: Process in chunks, merging within each chunk
    merged_segments = []
    for i in range(0, len(segments), chunk_size):
        chunk = segments[i : i + chunk_size]
        merged_chunk = merge_adjacent_ways_simple(chunk)
        merged_segments.extend(merged_chunk)

    # Phase 2: Iteratively merge across boundaries using lightweight endpoint matching
    merged_segments = merge_boundary_iterative(merged_segments)

    return merged_segments


def merge_boundary_iterative(segments: List[Dict]) -> List[Dict]:
    """
    Iteratively merge segments across boundaries using only endpoint data.

    Uses a lightweight approach that only examines start/end coordinates to find
    mergeable pairs. Continues until no more merges are possible.

    Memory efficient: Only stores endpoint index + segment references, not full node data.

    Args:
        segments: List of segments (potentially from multiple batches)

    Returns:
        List of maximally merged segments
    """
    if len(segments) <= 1:
        return segments

    iteration = 0
    while True:
        iteration += 1

        # Build lightweight endpoint index
        # Maps rounded coordinates -> list of (segment_idx, endpoint_type)
        endpoint_index = {}

        for idx, seg in enumerate(segments):
            nodes = seg.get("nodes", [])
            if len(nodes) < 2:
                continue

            start = nodes[0]
            end = nodes[-1]

            # Round to ~10m precision for matching (same as merge_adjacent_ways_simple)
            start_coord = (round(start["lat"], 4), round(start["lon"], 4))
            end_coord = (round(end["lat"], 4), round(end["lon"], 4))

            if start_coord not in endpoint_index:
                endpoint_index[start_coord] = []
            if end_coord not in endpoint_index:
                endpoint_index[end_coord] = []

            endpoint_index[start_coord].append((idx, "start"))
            endpoint_index[end_coord].append((idx, "end"))

        # Try to find mergeable pairs
        merged_segments = []
        used = set()
        merges_made = 0

        for idx, seg in enumerate(segments):
            if idx in used:
                continue

            nodes = seg.get("nodes", [])
            if len(nodes) < 2:
                merged_segments.append(seg)
                used.add(idx)
                continue

            # Try to find one segment to merge with (greedy approach)
            current = seg.copy()
            used.add(idx)

            current_nodes = current["nodes"]
            start_coord = (round(current_nodes[0]["lat"], 4), round(current_nodes[0]["lon"], 4))
            end_coord = (round(current_nodes[-1]["lat"], 4), round(current_nodes[-1]["lon"], 4))

            merged_this_segment = False

            # Try to connect at end first
            for neighbor_idx, neighbor_end in endpoint_index.get(end_coord, []):
                if neighbor_idx in used or neighbor_idx == idx:
                    continue

                neighbor = segments[neighbor_idx]
                neighbor_nodes = neighbor.get("nodes", [])

                if len(neighbor_nodes) < 2:
                    continue

                # Merge the segments
                if neighbor_end == "start":
                    # end → neighbor start: append neighbor (skip first node)
                    current["nodes"] = current_nodes + neighbor_nodes[1:]
                    current["way_ids"] = current.get("way_ids", []) + neighbor.get("way_ids", [])
                    used.add(neighbor_idx)
                    merges_made += 1
                    merged_this_segment = True
                    break
                elif neighbor_end == "end":
                    # end → neighbor end: append reversed neighbor (skip last node)
                    current["nodes"] = current_nodes + neighbor_nodes[-2::-1]
                    current["way_ids"] = current.get("way_ids", []) + neighbor.get("way_ids", [])
                    used.add(neighbor_idx)
                    merges_made += 1
                    merged_this_segment = True
                    break

            # If we didn't merge at end, try to connect at start
            if not merged_this_segment:
                for neighbor_idx, neighbor_end in endpoint_index.get(start_coord, []):
                    if neighbor_idx in used or neighbor_idx == idx:
                        continue

                    neighbor = segments[neighbor_idx]
                    neighbor_nodes = neighbor.get("nodes", [])

                    if len(neighbor_nodes) < 2:
                        continue

                    # Merge the segments
                    if neighbor_end == "end":
                        # neighbor end → start: prepend neighbor (skip last node)
                        current["nodes"] = neighbor_nodes[:-1] + current_nodes
                        current["way_ids"] = neighbor.get("way_ids", []) + current.get(
                            "way_ids", []
                        )
                        used.add(neighbor_idx)
                        merges_made += 1
                        merged_this_segment = True
                        break
                    elif neighbor_end == "start":
                        # neighbor start → start: prepend reversed neighbor (skip first node)
                        current["nodes"] = neighbor_nodes[:0:-1] + current_nodes
                        current["way_ids"] = neighbor.get("way_ids", []) + current.get(
                            "way_ids", []
                        )
                        used.add(neighbor_idx)
                        merges_made += 1
                        merged_this_segment = True
                        break

            merged_segments.append(current)

        # If no merges were made, we're done
        if merges_made == 0:
            break

        # Update segments for next iteration
        segments = merged_segments

        # Safety check: don't iterate forever
        if iteration > 50:
            break

    return merged_segments


def should_use_streaming_mode(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> bool:
    """
    Intelligently determine if streaming mode should be used based on region size
    and available system memory.

    Args:
        min_lat: Minimum latitude
        min_lon: Minimum longitude
        max_lat: Maximum latitude
        max_lon: Maximum longitude

    Returns:
        True if streaming mode should be used
    """
    # Calculate region area
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    area_deg2 = lat_span * lon_span

    # Estimate number of ways based on area
    # California (500 deg²) has ~4M ways, so ~8000 ways/deg²
    # This is a rough estimate - actual density varies
    estimated_ways = int(area_deg2 * 8000)

    # Estimate memory needed:
    # Each SimpleWay object is roughly 500-1000 bytes (nodes, tags, etc.)
    # Use conservative estimate of 1KB per way
    estimated_memory_gb = (estimated_ways * 1024) / (1024**3)

    # Try to get available memory using psutil (if available)
    try:
        import psutil

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        has_psutil = True
    except ImportError:
        # Fallback: assume 8GB available if psutil not installed
        available_gb = 8.0
        has_psutil = False

    # Decision logic:
    # 1. If estimated memory > 50% of available memory, use streaming
    # 2. If region > 100 deg², use streaming (safety threshold)
    # 3. If estimated ways > 1M, use streaming

    use_streaming = (
        estimated_memory_gb > (available_gb * 0.5) or area_deg2 > 100 or estimated_ways > 1_000_000
    )

    if use_streaming:
        print("Memory-aware mode: Streaming enabled")
        print(f"  Region: {area_deg2:.1f} deg² (~{estimated_ways:,} estimated ways)")
        print(f"  Estimated memory needed: {estimated_memory_gb:.1f} GB")
        if has_psutil:
            print(f"  Available memory: {available_gb:.1f} GB")
    else:
        print("Memory-aware mode: In-memory processing")
        print(f"  Region: {area_deg2:.1f} deg² (~{estimated_ways:,} estimated ways)")

    return use_streaming


def get_tracktype_helper(way) -> Tuple[str, str]:
    """Extract tracktype information from way."""
    if not hasattr(way, "tags") or not way.tags:
        return "unknown", "No tracktype information"

    if way.tags.get("highway") != "track":
        return "not_applicable", "Not a track"

    if "tracktype" in way.tags:
        tracktype_raw = str(way.tags["tracktype"]).strip().lower()

        osm_to_definition_key = {
            "grade1": "1-solid",
            "grade2": "2-gravel_rd",
            "grade3": "3-doubletrack",
            "grade4": "4-mostly_soft",
            "grade5": "5-unimproved",
            "1": "1-solid",
            "2": "2-gravel_rd",
            "3": "3-doubletrack",
            "4": "4-mostly_soft",
            "5": "5-unimproved",
        }

        definition_key = osm_to_definition_key.get(tracktype_raw)
        if definition_key and definition_key in TRACKTYPE_DEFINITIONS:
            return definition_key, TRACKTYPE_DEFINITIONS[definition_key]
        else:
            return tracktype_raw, f"Unknown tracktype: {tracktype_raw}"

    return "unspecified", "Track with unspecified surface quality"


def get_surface_helper(way) -> str:
    """Extract surface information from way."""
    if not hasattr(way, "tags") or not way.tags:
        return "unknown"

    if "surface" in way.tags:
        return str(way.tags["surface"]).strip().lower()

    highway_type = way.tags.get("highway", "").strip().lower()
    surface_mapping = {
        "motorway": "asphalt",
        "trunk": "asphalt",
        "primary": "asphalt",
        "secondary": "asphalt",
        "tertiary": "asphalt",
        "unclassified": "paved",
        "residential": "asphalt",
        "service": "paved",
        "track": "unpaved",
        "path": "unpaved",
        "footway": "unpaved",
    }
    return surface_mapping.get(highway_type, "unknown")


def convert_way_to_segment(way) -> Dict:
    """Convert a single way to a segment dictionary."""
    tracktype, tracktype_def = get_tracktype_helper(way)
    nodes = [{"lat": node.lat, "lon": node.lon, "id": node.id} for node in way.nodes]

    return {
        "way_ids": [way.id],
        "way_boundaries": [(0, len(nodes), way.id)],  # Track which nodes belong to this way
        "nodes": nodes,
        "street_name": way.tags.get("name", "Unnamed Road"),
        "surface": get_surface_helper(way),
        "tracktype": tracktype,
        "tracktype_definition": tracktype_def,
        "cycling_access": way.tags.get("bicycle", "Unknown"),
        "highway_type": way.tags.get("highway", "unknown"),
    }


def convert_ways_to_segments_batched(
    persistence, checkpoint_file: Path, signal_handler=None
) -> Path:
    """
    Convert ways to segments by reading from checkpoint in batches.
    Writes segments to a checkpoint file to avoid memory exhaustion.

    Args:
        persistence: ChunkPersistenceManager instance
        checkpoint_file: Path to filtered ways checkpoint file
        signal_handler: Optional GracefulKiller instance for checkpoint support

    Returns:
        Path to segments checkpoint file
    """
    # Initialize segments checkpoint file
    segments_checkpoint = persistence.analysis_dir / "segments.checkpoint.jsonl"
    if segments_checkpoint.exists():
        segments_checkpoint.unlink()

    # Count total ways for progress bar
    total_ways = persistence.count_filtered_ways(checkpoint_file)

    segment_count = 0
    batch_num = 0

    # Read ways in batches from checkpoint and write segments
    with open(segments_checkpoint, "w") as f:
        with tqdm(
            total=total_ways,
            desc="  Converting",
            unit=" ways",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ascii=" ▏▎▍▌▋▊▉█",
            dynamic_ncols=True,
        ) as pbar:

            for way_batch in persistence.read_filtered_ways_batched(
                checkpoint_file, batch_size=10000
            ):
                # Convert each way in the batch
                for way in way_batch:
                    segment = convert_way_to_segment(way)
                    f.write(json.dumps(segment) + "\n")
                    segment_count += 1
                    pbar.update(1)

                batch_num += 1

                # Check for graceful shutdown every batch
                if signal_handler and batch_num % 10 == 0:
                    signal_handler.set_operation(
                        "way_to_segment_conversion",
                        {
                            "segment_count": segment_count,
                            "total_ways": total_ways,
                        },
                    )

                    if signal_handler.kill_now:
                        f.flush()  # Ensure all data is written
                        print("\n\n🛑 Graceful shutdown - partial conversion saved")
                        print(f"  Converted {segment_count:,} / {total_ways:,} ways")
                        print(
                            "  ⚠️  Partial segments file created - will be overwritten on next run"
                        )
                        print("\n⏸️  Run again to restart conversion from beginning.")
                        import sys

                        sys.exit(0)

    print(f"\n  ✓ Converted {segment_count:,} ways to segment format (checkpoint)")
    return segments_checkpoint


def convert_ways_to_segments_inmemory(all_ways: List) -> List[Dict]:
    """
    Convert ways to segments (in-memory mode for small regions).

    Args:
        all_ways: List of way objects

    Returns:
        List of segment dictionaries
    """
    from tqdm import tqdm

    print("  Converting ways to segment format...")
    all_segments = []

    for way in tqdm(
        all_ways,
        desc="  Converting",
        unit="ways",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ascii=" █",
        dynamic_ncols=True,
    ):
        segment = convert_way_to_segment(way)
        all_segments.append(segment)

    print(f"  ✓ Converted {len(all_segments):,} ways to segment format")
    return all_segments


def get_sort_key_for_segment(seg: dict) -> str:
    """
    Get sorting key for a segment, with geographic clustering to prevent memory issues.

    Adds a geographic suffix to group nearby segments together for streets that
    commonly have thousands of segments scattered across large regions. This prevents
    memory exhaustion during the merge phase.

    Strategy:
    - Unnamed roads: Very fine grid (0.01° ≈ 1km) - handles very dense urban areas
    - Long highways/routes: Medium grid (0.2° ≈ 20km) - follow long corridors
    - Regular named streets: No clustering (usually localized)

    Common large streets in California:
    - "Unnamed Road": 50,000+ segments → needs clustering
    - "US Route 101", "Interstate 5": 10,000+ segments → needs clustering
    - "Main Street": Usually <100 segments per city → no clustering needed

    Args:
        seg: Segment dictionary with street_name and nodes

    Returns:
        Sorting key string with optional geographic suffix
    """
    street_name = seg.get("street_name", "Unnamed Road")

    # Determine if street needs geographic clustering
    needs_clustering = False
    grid_size = 0.01  # Default: 1km grid for unnamed roads

    # Check for patterns that indicate a long/distributed street
    street_lower = street_name.lower()

    # Always cluster unnamed roads (very fine grid for dense areas)
    if street_name == "Unnamed Road":
        needs_clustering = True
        grid_size = 0.01  # 1km grid (~0.6 miles) - handles Boise, LA, SF density

    # Cluster highways and routes (coarser grid since they're continuous)
    elif any(
        keyword in street_lower
        for keyword in [
            "highway",
            "interstate",
            "route",
            "freeway",
            "us-",
            "state route",
            "sr-",
            "i-",
            "us ",
            "ca-",
            "california",
        ]
    ):
        needs_clustering = True
        grid_size = 0.2  # 20km grid for highways (they're more continuous)

    # Apply geographic clustering if needed
    if needs_clustering and seg.get("nodes"):
        first_node = seg["nodes"][0]
        lat_bucket = round(first_node["lat"] / grid_size) * grid_size
        lon_bucket = round(first_node["lon"] / grid_size) * grid_size
        # Use abs() for longitude and explicit direction
        lon_dir = "E" if lon_bucket >= 0 else "W"
        return f"{street_name}_{lat_bucket:.2f}N_{abs(lon_bucket):.2f}{lon_dir}"

    return street_name


def sort_segments_by_street(segments_checkpoint: Path, signal_handler=None) -> Path:
    """
    Sort segments checkpoint file by street name using external merge sort.

    Memory-efficient approach that never loads all segments at once:
    1. Split into sorted chunks that fit in memory
    2. Merge sorted chunks using k-way merge

    Uses geographic clustering for unnamed roads to prevent memory exhaustion
    from large groups of unnamed segments.

    Args:
        segments_checkpoint: Path to unsorted segments checkpoint
        signal_handler: Optional GracefulKiller instance for checkpoint support

    Returns:
        Path to sorted segments checkpoint
    """
    import heapq
    import sys

    sorted_checkpoint = segments_checkpoint.parent / "segments_sorted.jsonl"
    temp_dir = segments_checkpoint.parent / "sort_temp"
    temp_dir.mkdir(exist_ok=True)

    print("  Sorting segments by street name (external sort)...")

    try:
        # Phase 1: Split into sorted chunks
        CHUNK_SIZE = 100000  # Sort 100k segments at a time in memory

        # Count total segments first
        print("    Counting segments...")
        total_segments = 0
        with open(segments_checkpoint) as f:
            for line in f:
                if line.strip():
                    total_segments += 1

        print(f"    Splitting {total_segments:,} segments into sorted chunks...")
        chunk_files = []
        chunk = []
        chunk_num = 0
        segments_processed = 0

        with open(segments_checkpoint) as f:
            with tqdm(
                total=total_segments,
                desc="    Phase 1",
                unit=" segs",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ascii=" ▏▎▍▌▋▊▉█",
                dynamic_ncols=True,
            ) as pbar:

                for line in f:
                    if not line.strip():
                        continue

                    seg = json.loads(line)
                    sort_key = get_sort_key_for_segment(seg)
                    chunk.append((sort_key, line))
                    segments_processed += 1
                    pbar.update(1)

                    # When chunk is full, sort and write it
                    if len(chunk) >= CHUNK_SIZE:
                        chunk.sort(key=lambda x: x[0])
                        chunk_file = temp_dir / f"chunk_{chunk_num:04d}.jsonl"
                        with open(chunk_file, "w") as cf:
                            for _, original_line in chunk:
                                cf.write(original_line)
                        chunk_files.append(chunk_file)
                        chunk = []
                        chunk_num += 1

                        # Check for graceful shutdown after each chunk
                        if signal_handler and chunk_num % 5 == 0:
                            signal_handler.set_operation(
                                "external_sort_phase1",
                                {
                                    "chunks_created": len(chunk_files),
                                    "segments_processed": segments_processed,
                                    "total_segments": total_segments,
                                },
                            )

                            if signal_handler.kill_now:
                                print("\n\n🛑 Graceful shutdown during sort Phase 1")
                                print(
                                    f"  Created {len(chunk_files)} sorted chunks ({segments_processed:,} / {total_segments:,} segments)"
                                )
                                print(
                                    "  ⚠️  Partial sort progress - will restart from beginning on next run"
                                )
                                print("\n⏸️  Run again to restart sorting.")
                                sys.exit(0)

                # Write final chunk
                if chunk:
                    chunk.sort(key=lambda x: x[0])
                    chunk_file = temp_dir / f"chunk_{chunk_num:04d}.jsonl"
                    with open(chunk_file, "w") as cf:
                        for _, original_line in chunk:
                            cf.write(original_line)
                    chunk_files.append(chunk_file)

        print(f"    Created {len(chunk_files)} sorted chunks")

        # Phase 2: K-way merge of sorted chunks
        print("    Phase 2: Merging sorted chunks...")

        # Open all chunk files
        chunk_handles = []
        for chunk_file in chunk_files:
            chunk_handles.append(open(chunk_file))

        # Initialize heap with first line from each chunk
        heap = []
        for i, fh in enumerate(chunk_handles):
            line = fh.readline()
            if line.strip():
                seg = json.loads(line)
                sort_key = get_sort_key_for_segment(seg)
                # Heap entry: (sort_key, chunk_index, line)
                heapq.heappush(heap, (sort_key, i, line))

        # Merge all chunks
        merged_count = 0
        with open(sorted_checkpoint, "w") as out_f:
            with tqdm(
                total=total_segments,
                desc="    Merging",
                unit=" segs",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ascii=" ▏▎▍▌▋▊▉█",
                dynamic_ncols=True,
            ) as pbar:

                while heap:
                    # Pop smallest element
                    sort_key, chunk_idx, line = heapq.heappop(heap)
                    out_f.write(line)
                    merged_count += 1
                    pbar.update(1)

                    # Check for graceful shutdown periodically
                    if signal_handler and merged_count % 50000 == 0:
                        signal_handler.set_operation(
                            "external_sort_phase2",
                            {
                                "merged_count": merged_count,
                                "total_segments": total_segments,
                            },
                        )

                        if signal_handler.kill_now:
                            out_f.flush()
                            print("\n\n🛑 Graceful shutdown during sort Phase 2")
                            print(f"  Merged {merged_count:,} / {total_segments:,} segments")
                            print(
                                "  ⚠️  Partial sorted file created - will restart from beginning on next run"
                            )
                            print("\n⏸️  Run again to restart sorting.")
                            # Close chunk handles before exiting
                            for fh in chunk_handles:
                                fh.close()
                            sys.exit(0)

                    # Read next line from same chunk
                    next_line = chunk_handles[chunk_idx].readline()
                    if next_line.strip():
                        seg = json.loads(next_line)
                        sort_key = get_sort_key_for_segment(seg)
                        heapq.heappush(heap, (sort_key, chunk_idx, next_line))

        # Close all chunk files
        for fh in chunk_handles:
            fh.close()

        # Clean up temporary chunks
        for chunk_file in chunk_files:
            chunk_file.unlink()
        temp_dir.rmdir()

        print(f"    ✓ Sorted {merged_count:,} segments by street name")
        return sorted_checkpoint

    except Exception as e:
        print(f"    ⚠️  Sort failed: {e}, processing unsorted")
        import traceback

        traceback.print_exc()

        # Clean up temp dir if it exists
        try:
            if temp_dir.exists():
                for f in temp_dir.iterdir():
                    f.unlink()
                temp_dir.rmdir()
        except:
            pass

        return segments_checkpoint


def read_segments_from_checkpoint_batched(checkpoint_file: Path, batch_size: int = 10000):
    """
    Read segments from checkpoint file in batches.
    Generator that yields batches of segments.
    """
    if not checkpoint_file.exists():
        return

    batch = []
    with open(checkpoint_file) as f:
        for line in f:
            if line.strip():
                segment = json.loads(line)
                batch.append(segment)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

    if batch:
        yield batch


def merge_segments_streaming(persistence, segments_checkpoint: Path, signal_handler=None) -> Path:
    """
    Merge segments by street name using streaming to avoid memory exhaustion.

    Strategy: Sort by street name, then process one street at a time sequentially.
    This ensures we never hold more than one street's segments in memory.

    Args:
        persistence: ChunkPersistenceManager instance
        segments_checkpoint: Path to segments checkpoint file
        signal_handler: Optional GracefulKiller instance for checkpoint support

    Returns:
        Path to merged segments checkpoint file
    """
    print("\n  Merging segments by street (streaming mode)...")

    # Phase 1: Sort segments by street name
    sorted_checkpoint = sort_segments_by_street(segments_checkpoint, signal_handler)

    # Phase 2: Process sorted file, merging segments for each street
    merged_checkpoint = persistence.analysis_dir / "merged_segments.jsonl"
    if merged_checkpoint.exists():
        merged_checkpoint.unlink()

    # Count total segments for progress bar
    total_segments = 0
    with open(sorted_checkpoint) as f:
        for line in f:
            if line.strip():
                total_segments += 1

    merged_count = 0
    streets_with_multiple = 0
    streets_skipped_too_large = 0
    current_street = None
    current_street_segments = []
    total_processed = 0

    # Memory safety: Skip merging streets with too many segments
    MAX_SEGMENTS_PER_STREET = 5000

    print(f"  Processing {total_segments:,} segments...")

    # Track if we're processing a large street (using chunked merge)
    processing_large_street = False

    with open(sorted_checkpoint) as f_in, open(merged_checkpoint, "w") as f_out:
        with tqdm(
            total=total_segments,
            desc="  Merging",
            unit=" segs",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ascii=" ▏▎▍▌▋▊▉█",
            dynamic_ncols=True,
        ) as pbar:

            for line in f_in:
                if not line.strip():
                    continue

                segment = json.loads(line)
                # Use same sort key as sorting phase for consistency
                street_name = get_sort_key_for_segment(segment)
                total_processed += 1
                pbar.update(1)

                # Check for graceful shutdown periodically
                if signal_handler and total_processed % 10000 == 0:
                    signal_handler.set_operation(
                        "segment_merging",
                        {
                            "processed": total_processed,
                            "merged": merged_count,
                            "total": total_segments,
                        },
                    )

                    if signal_handler.kill_now:
                        # Flush current street's segments
                        if current_street_segments:
                            if len(current_street_segments) == 1:
                                f_out.write(json.dumps(current_street_segments[0]) + "\n")
                            elif processing_large_street:
                                # Use chunked merge for large streets
                                merged = merge_large_street_chunked(current_street_segments)
                                for seg in merged:
                                    f_out.write(json.dumps(seg) + "\n")
                            else:
                                merged = merge_adjacent_ways_simple(current_street_segments)
                                for seg in merged:
                                    f_out.write(json.dumps(seg) + "\n")
                        f_out.flush()
                        print("\n\n🛑 Graceful shutdown during segment merging")
                        print(f"  Processed {total_processed:,} / {total_segments:,} segments")
                        print(f"  Merged into {merged_count:,} segments")
                        print(
                            "  ⚠️  Partial merged file created - will restart from beginning on next run"
                        )
                        print("\n⏸️  Run again to restart merging.")
                        import sys

                        sys.exit(0)

                # If we're on a new street, process the previous street's segments
                if current_street is not None and street_name != current_street:
                    # End of previous street - process accumulated segments
                    if len(current_street_segments) == 1:
                        f_out.write(json.dumps(current_street_segments[0]) + "\n")
                        merged_count += 1
                    elif len(current_street_segments) > 1:
                        streets_with_multiple += 1
                        if processing_large_street:
                            # Use chunked merge for large streets
                            # Suppressed: internal implementation detail (chunked merge is always used)
                            # pbar.write(
                            #     f"    → Merging {len(current_street_segments):,} segments for '{current_street}' using chunked algorithm..."
                            # )
                            merged = merge_large_street_chunked(current_street_segments)
                            for seg in merged:
                                f_out.write(json.dumps(seg) + "\n")
                                merged_count += 1
                        else:
                            # Normal merge for regular streets
                            merged = merge_adjacent_ways_simple(current_street_segments)
                            for seg in merged:
                                f_out.write(json.dumps(seg) + "\n")
                                merged_count += 1

                    # Reset for new street
                    current_street_segments = []
                    processing_large_street = False

                # Check if we're accumulating too many segments for this street
                if len(current_street_segments) >= MAX_SEGMENTS_PER_STREET:
                    if not processing_large_street:
                        # First time hitting limit - mark for chunked merge processing
                        streets_skipped_too_large += 1
                        # Suppressed: internal implementation detail (chunked merge is always used)
                        # pbar.write(
                        #     f"    ⚠️  Large street detected: '{current_street}' (>{MAX_SEGMENTS_PER_STREET} segments) - will use chunked merge"
                        # )
                        processing_large_street = True

                # Continue accumulating segments (even for large streets)
                current_street = street_name
                current_street_segments.append(segment)

            # Process final street (if any segments remain)
            if current_street_segments:
                if len(current_street_segments) == 1:
                    f_out.write(json.dumps(current_street_segments[0]) + "\n")
                    merged_count += 1
                elif processing_large_street:
                    streets_with_multiple += 1
                    pbar.write(
                        f"    → Merging {len(current_street_segments):,} segments for '{current_street}' using chunked algorithm..."
                    )
                    merged = merge_large_street_chunked(current_street_segments)
                    for seg in merged:
                        f_out.write(json.dumps(seg) + "\n")
                        merged_count += 1
                else:
                    streets_with_multiple += 1
                    merged = merge_adjacent_ways_simple(current_street_segments)
                    for seg in merged:
                        f_out.write(json.dumps(seg) + "\n")
                        merged_count += 1

    print(f"\n    ✓ Merged {total_processed:,} segments → {merged_count:,} road segments")
    print(f"    ✓ Merged {streets_with_multiple:,} streets with multiple ways")
    if streets_skipped_too_large > 0:
        print(
            f"    ✓ Merged {streets_skipped_too_large} large streets (>5,000 segments each) using chunked algorithm"
        )

    return merged_checkpoint


def process_region_without_chunking(
    persistence,
    road_analyzer,
    chunk_merger,
    boundary_merger,
    elevation_fetcher,
    min_lat,
    min_lon,
    max_lat,
    max_lon,
    metadata,
    min_score,
    unit_system,
    score_type,
    enable_geocoding,
    surface_filter=None,
    cycling_only=False,
):
    """
    Process an entire region without chunking.

    This is more efficient for local mode analysis because:
    1. We have pre-downloaded OSM files (no API rate limit concerns)
    2. Extract all ways at once instead of chunk-by-chunk
    3. Run optimized in-memory merge on contiguous ways
    4. No boundary merging complexity
    5. Simpler code path with fewer intermediate steps

    Used for:
    - State/country analysis (always)
    - Address searches in local mode (new)
    """

    # Setup signal handling
    signal_handler.set_persistence_manager(persistence)

    from pathlib import Path

    from climb_analyzer.utils.formatting import print_dim, print_section_simple

    print_section_simple("Optimized Full-Region Extraction Mode", spacing_before=1)
    print_dim("Using full-region extraction\n")

    # ALWAYS use streaming mode for full-region extraction
    # Benefits: constant memory usage, resumability, progress tracking, no OOM crashes
    # Drawback: ~1-2 second overhead for tiny regions (negligible)
    use_streaming = True
    #print("Using streaming mode (memory-efficient, resumable processing)")

    # MEMORY FIX: Delete legacy checkpoint files that trigger the old non-streaming code path
    # These files cause the system to load GBs into memory instead of streaming from disk
    legacy_files = [
        persistence.analysis_dir / "final_processed_segments.pkl",  # Triggers old pickle load path
        persistence.analysis_dir / "completed_elevations.pkl",  # Old format elevation storage
    ]
    for legacy_file in legacy_files:
        if legacy_file.exists():
            try:
                legacy_file.unlink()
                print(f"  ✓ Removed legacy checkpoint file: {legacy_file.name}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not remove {legacy_file.name}: {e}")

    # Cloud cache upload flag: Check if this is a clean analysis
    should_upload_to_cache = False
    if CLOUD_CACHE_AVAILABLE and CLOUD_CACHE_ENABLED:
        from utils.config_loader import DEPLOYMENT_TYPE

        deployment_type = DEPLOYMENT_TYPE
        scope_type = metadata.get("scope_type", "unknown")
        if deployment_type == "local":
            # Check if analysis settings qualify for cloud cache upload
            is_clean = (
                surface_filter == "all"
                and min_score == 0
                and not cycling_only
                and score_type == "basic"
                and scope_type in ["country", "region"]
            )
            if is_clean:
                should_upload_to_cache = True
                print(
                    "Analysis will be posted to cloud cache to share with others after completion."
                )

    # Check for existing checkpoints and show resumability status
    if use_streaming:
        checkpoints_found = []
        # Use hasattr to check if method exists (for backwards compatibility with older container builds)
        if (
            hasattr(persistence, "has_filtered_ways_checkpoint")
            and persistence.has_filtered_ways_checkpoint()
        ):
            checkpoints_found.append("filtered ways")
        elif (persistence.analysis_dir / "filtered_ways.jsonl").exists():
            checkpoints_found.append("filtered ways")

        if (persistence.analysis_dir / "segments.checkpoint.jsonl").exists():
            checkpoints_found.append("segments")
        if (persistence.analysis_dir / "merged_segments.jsonl").exists():
            checkpoints_found.append("merged segments")

        elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()
        if elevation_mapping or elevation_batch_info:
            checkpoints_found.append("elevation progress")

        # Only show checkpoint status if resuming (don't show "starting fresh")
        if checkpoints_found:
            # Show the furthest checkpoint stage
            furthest = checkpoints_found[-1]  # Last in list is furthest along
            print(f"Resuming from: {furthest}")

    # **EARLY CHECKPOINT CHECK: Skip all preprocessing if elevations are complete**
    # This is critical for recovery from OOM crashes during later stages (geocoding, profiling, export)
    completed_elevations = persistence.load_completed_elevations()
    merged_segments_checkpoint = persistence.analysis_dir / "merged_segments.jsonl"

    if completed_elevations and merged_segments_checkpoint.exists():
        merged_count = persistence.count_segments("merged_segments")

        print(f"\n✓ Fast recovery: {merged_count:,} segments, {len(completed_elevations):,} elevations")
        print("  Skipping preprocessing, starting climb analysis...")

        # Set variables needed for climb analysis
        node_elevations = completed_elevations
        skip_elevation_extraction = True
        total_segments = merged_count
        total_coords_requested = len(completed_elevations)
        total_elevations_fetched = len(completed_elevations)
        total_elevations_failed = 0

        # Skip to climb analysis (jump to line ~9418)
        # Set flag to bypass preprocessing steps
        skip_preprocessing = True
    else:
        skip_preprocessing = False

    # Extract region name from metadata for display
    region_name = metadata.get("region_name") or metadata.get("country") or metadata.get("address") or "Unknown Region"

    if not skip_preprocessing:
        from climb_analyzer.utils.formatting import print_header
        print_header(f"Analyzing Region: {region_name}", spacing_before=2)

        # Extract ALL ways from the region
        print("\nStep 1: Extracting all road ways from bounding box...")

        # Check if we have a checkpoint from previous run
        checkpoint_file = persistence.analysis_dir / "filtered_ways.jsonl"
        has_checkpoint = (
            hasattr(persistence, "has_filtered_ways_checkpoint")
            and persistence.has_filtered_ways_checkpoint()
        ) or checkpoint_file.exists()

        if use_streaming and has_checkpoint:
            way_count = persistence.count_filtered_ways(checkpoint_file)
            print(f"Resuming from checkpoint with {way_count:,} filtered ways")
            all_ways = []  # We'll read from checkpoint in batches
        else:
            # Extract ways (streaming mode will write to checkpoint)
            all_ways = road_analyzer.get_all_roads_in_region(
                min_lat,
                min_lon,
                max_lat,
                max_lon,
                persistence_manager=persistence if use_streaming else None,
                use_streaming=use_streaming,
            )

        # Handle streaming mode differently - skip merging for very large regions
        if use_streaming:
            checkpoint_file = persistence.analysis_dir / "filtered_ways.jsonl"
            if not checkpoint_file.exists() or checkpoint_file.stat().st_size == 0:
                print("⚠️  No roads found in region. Analysis cannot continue.")
                import pandas as pd

                return [], pd.DataFrame(), persistence, False, None

            way_count = persistence.count_filtered_ways(checkpoint_file)
            print(f"✓ Checkpoint contains {way_count:,} filtered roads\n")

            # Check if segments checkpoint already exists
            segments_checkpoint = persistence.analysis_dir / "segments.checkpoint.jsonl"
            if segments_checkpoint.exists():
                segment_count = persistence.count_segments("segments.checkpoint")
                print(f"Found existing segments checkpoint with {segment_count:,} segments")
                print("   Skipping Step 2 (conversion already complete)\n")
            else:
                # Process ways in batches from checkpoint
                print("Step 2: Converting ways to segments...")
                segments_checkpoint = convert_ways_to_segments_batched(
                    persistence, checkpoint_file, signal_handler
                )

                segment_count = persistence.count_segments("segments.checkpoint")
                print(f"✓ Checkpoint contains {segment_count:,} segments\n")

            # Check if merged segments checkpoint already exists
            merged_segments_checkpoint = persistence.analysis_dir / "merged_segments.jsonl"
            if merged_segments_checkpoint.exists():
                merged_count = persistence.count_segments("merged_segments")
                print(f"Found existing merged segments checkpoint with {merged_count:,} segments")
                print("   Skipping Step 3 (merging already complete)\n")
            else:
                # STREAMING MODE: Merge segments by street using sort-based streaming
                print("Step 3: Merging segments by street (streaming mode)...")
                merged_segments_checkpoint = merge_segments_streaming(
                    persistence, segments_checkpoint, signal_handler
                )

                merged_count = persistence.count_segments("merged_segments")
                print(f"✓ Merged checkpoint contains {merged_count:,} road segments\n")

            merged_segments = []  # Empty - will process from checkpoint

    # Determine the segment source and count
    # (Always using streaming mode - segments are in checkpoint file)
    total_segments = merged_count

    # Skip boundary merge - different reasons for address vs regional analysis
    scope_type = metadata.get("scope_type", "unknown")

    # Step 4 removed - boundary merging now happens during post-processing
    # after climb analysis (more memory efficient - operates on climbs not segments)
    print(f"\n  Ready to process {total_segments:,} merged segments")

    # **CHECKPOINT CHECK: Skip elevation extraction if already completed**
    completed_elevations = persistence.load_completed_elevations()
    skip_elevation_extraction = False

    if completed_elevations:
        print(f"✓ Elevation data cached: {len(completed_elevations):,} nodes")
        node_elevations = completed_elevations
        skip_elevation_extraction = True
        # Set dummy values for stats (will be loaded from error logger history)
        total_coords_requested = len(completed_elevations)
        total_elevations_fetched = len(completed_elevations)
        total_elevations_failed = 0
    # Only extract elevations if not already completed
    if not skip_elevation_extraction:
        # Process in batches to avoid memory exhaustion on large states
        from collections import defaultdict
        from datetime import datetime
        from pathlib import Path

        from utils.error_logger import ErrorLogger

        # Initialize error logger for tracking elevation failures
        region_name = metadata.get("address", "Unknown Region")

        # Apply same scope_info formatting as xlsx output (see lines 11230-11242)
        # Extract just the region name from path formats like "us > hawaii" or "europe/france"
        if " > " in region_name:
            formatted_name = region_name.split(" > ")[-1]
        elif "/" in region_name:
            formatted_name = region_name.split("/")[-1]
        else:
            formatted_name = region_name

        # Convert to title case for cleaner filenames (e.g., "california" -> "California")
        formatted_name = formatted_name.replace("-", " ").title()

        # Make safe for filenames
        safe_name = "".join(c for c in formatted_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_name = safe_name.replace(" ", "_")[:50]

        # Create error log filename to match output file naming
        # Format: <region>_errors_<date>.txt (no surface filter, matches xlsx date at end)
        # Example: California_errors_2025-12-31.txt
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_filename = f"{safe_name}_errors_{date_str}"

        error_logger = ErrorLogger(
            region_name=region_name,
            output_dir=Path("output"),
            base_filename=base_filename,
            surface_filter=surface_filter,
            min_score=min_score,
            score_type=score_type,
            cycling_allowed=cycling_only,
            app_version=__version__,
        )

        # Collect all coordinates and metadata in batches
        coord_set = set()
        all_coordinates = []
        coord_to_node_ids = defaultdict(list)
        coord_metadata = {}  # Maps coordinate -> {way_id, street_name}

        total_nodes_processed = 0
        skipped_segments = 0

        # Calculate optimal batch size based on available memory
        def calculate_batch_size(total_segments: int, available_memory_gb: float) -> int:
            """
            Calculate optimal batch size for coordinate extraction.

            Conservative estimates:
            - Each segment: ~100 nodes average
            - Each coordinate: ~200 bytes (tuple + metadata + dict overhead)
            - Safety factor: Use only 40% of available memory for this operation

            Args:
                total_segments: Total number of segments to process
                available_memory_gb: Available system memory in GB

            Returns:
                Optimal batch size (number of segments)
            """
            # Conservative memory estimates
            BYTES_PER_COORD = 200  # Coordinate tuple + metadata + dict overhead
            AVG_NODES_PER_SEGMENT = 100  # Conservative average
            SAFETY_FACTOR = 0.40  # Use only 40% of available memory

            # Calculate memory budget
            available_bytes = available_memory_gb * 1024**3 * SAFETY_FACTOR

            # Estimate memory per segment
            bytes_per_segment = AVG_NODES_PER_SEGMENT * BYTES_PER_COORD

            # Calculate batch size
            batch_size = int(available_bytes / bytes_per_segment)

            # Enforce reasonable bounds
            MIN_BATCH = 5000  # Minimum for performance
            MAX_BATCH = 100000  # Maximum for progress visibility

            batch_size = max(MIN_BATCH, min(batch_size, MAX_BATCH))

            # Don't exceed total segments (but ensure minimum of 1)
            if total_segments > 0:
                batch_size = min(batch_size, total_segments)

            # Final safety check - never return 0
            batch_size = max(1, batch_size)

            return batch_size

        # Get available memory and calculate batch size
        try:
            import psutil

            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            total_memory_gb = memory.total / (1024**3)

            BATCH_SIZE = calculate_batch_size(total_segments, available_memory_gb)

            print(f"Memory: {available_memory_gb:.1f}GB available / {total_memory_gb:.1f}GB total")
            print(f"Using batch size: {BATCH_SIZE:,} segments (optimized for available memory)")
        except Exception:
            # Fallback to conservative default if psutil fails
            BATCH_SIZE = 25000
            print(
                f"Could not detect memory, using conservative batch size: {BATCH_SIZE:,} segments"
            )

        num_batches = (total_segments + BATCH_SIZE - 1) // BATCH_SIZE

        print("\nStep 4: Extracting elevations (streaming batches)...")

        # Start streaming elevation errors to file
        error_logger.start_elevation_logging()

        # Initialize smart checkpointer for batch processing
        from climb_analyzer.processing.checkpoint import SmartCheckpointer

        checkpointer = SmartCheckpointer(num_batches, "Region Elevation Fetching")

        # Check for existing checkpoint to resume from
        elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()
        start_batch_idx = 0

        # PERFORMANCE FIX 3: Use SQLite instead of shelve for 100x faster operations
        # SQLite provides fast bulk writes, efficient indexing, and incremental sync
        import sqlite3

        class ElevationDatabase:
            """Fast SQLite-based elevation storage with incremental sync support."""

            def __init__(self, db_path):
                self.db_path = str(db_path)
                self.conn = sqlite3.connect(self.db_path)
                self.conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging for speed
                self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL
                self.conn.execute("PRAGMA cache_size=10000")  # 10MB cache

                # Create table if it doesn't exist
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS elevations (
                        node_id TEXT PRIMARY KEY,
                        elevation REAL NOT NULL
                    )
                """
                )
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_node ON elevations(node_id)")
                self.conn.commit()

                # Track pending writes for incremental sync
                self.pending_writes = {}
                self.write_transaction = None

            def __setitem__(self, node_id, elevation):
                """Buffer write for bulk insert."""
                self.pending_writes[node_id] = elevation

                # MEMORY FIX: Auto-flush if buffer gets too large (prevents OOM)
                # Safety limit: 500k entries ≈ 15-30MB in memory
                if len(self.pending_writes) >= 500000:
                    self.flush()
                    self.conn.commit()  # Also commit to ensure data persisted

            def __getitem__(self, node_id):
                """Get elevation for a node."""
                cursor = self.conn.execute(
                    "SELECT elevation FROM elevations WHERE node_id=?", (node_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(node_id)
                return row[0]

            def get(self, node_id, default=None):
                """Get elevation for a node with default value if not found."""
                cursor = self.conn.execute(
                    "SELECT elevation FROM elevations WHERE node_id=?", (node_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    return default
                return row[0]

            def __contains__(self, node_id):
                """Check if node exists."""
                cursor = self.conn.execute(
                    "SELECT 1 FROM elevations WHERE node_id=? LIMIT 1", (node_id,)
                )
                return cursor.fetchone() is not None

            def __len__(self):
                """Get total count of elevations."""
                cursor = self.conn.execute("SELECT COUNT(*) FROM elevations")
                return cursor.fetchone()[0]

            def keys(self):
                """Iterate over all node IDs."""
                cursor = self.conn.execute("SELECT node_id FROM elevations")
                return (row[0] for row in cursor)

            def flush(self):
                """Write all pending changes to database (incremental sync)."""
                if not self.pending_writes:
                    return 0

                # Use INSERT OR REPLACE for upsert behavior
                self.conn.executemany(
                    "INSERT OR REPLACE INTO elevations (node_id, elevation) VALUES (?, ?)",
                    self.pending_writes.items(),
                )
                count = len(self.pending_writes)
                self.pending_writes.clear()
                return count

            def sync(self):
                """Flush pending writes and sync to disk (for checkpoints)."""
                count = self.flush()
                self.conn.commit()
                return count

            def close(self):
                """Flush, commit, and close database."""
                self.flush()
                self.conn.commit()
                self.conn.close()

        # Use disk-based storage for elevations to avoid memory exhaustion
        # Store elevation database in checkpoints folder (persistent across runs)
        elevation_db_path = persistence.analysis_dir / "elevation_data.db"

        # Use SQLite instead of shelve (100x faster for large datasets)
        elevation_db = ElevationDatabase(elevation_db_path)
        print(f"Using disk-based elevation storage: {elevation_db_path}")

        # Use bloom filter for coordinate deduplication (memory efficient)
        try:
            from pybloom_live import BloomFilter

            # CRITICAL FIX: Increase capacity for very large regions like France (112M+ coords)
            # France extrapolation: 45M entries @ batch 32 → ~112M total for 80 batches
            # Memory cost: 150M capacity @ 0.001 error rate = ~225MB (vs 3GB+ for set)
            # This is acceptable for large region analysis - much better than OOM
            global_coords_bloom = BloomFilter(capacity=150000000, error_rate=0.001)
            # MEMORY FIX: Also use bloom filter for node IDs to prevent unbounded memory growth
            # For large regions like France (potentially 200M+ nodes), this saves several GB
            fetched_node_ids_bloom = BloomFilter(capacity=200000000, error_rate=0.001)
            print(
                "   Using bloom filters for coordinate & node ID deduplication (memory optimized)"
            )
            print("   Bloom filter capacity: 150M coordinates, 200M node IDs")
            use_bloom = True
        except ImportError:
            # Fallback to set if bloom filter not available
            global_coords_seen = set()
            fetched_node_ids = set()
            use_bloom = False

        # MEMORY FIX: Flush interval for bulk database writes
        # CRITICAL: For large regions (France, USA), pending_writes can grow to GBs if we wait too long
        # Reduced from 10 to 2 to prevent OOM while still getting bulk insert benefits
        FLUSH_INTERVAL = 2  # Flush buffer every N batches (was 10, caused OOM on large regions)

        total_elevations_fetched = 0
        total_elevations_failed = 0

        # Check if we can resume from disk database (streaming mode without .pkl file)
        existing_elevation_count = len(elevation_db)
        can_resume_from_db = existing_elevation_count > 0

        if elevation_mapping or elevation_batch_info or can_resume_from_db:
            # STREAMING FORMAT CHECKPOINT RESUME
            if elevation_batch_info and "elevations_fetched" in elevation_batch_info:
                # Streaming format - has elevations_fetched directly (not nested in checkpoint_data)
                is_complete = elevation_batch_info.get("complete", False)

                if is_complete:
                    print("✓ Elevation processing already complete - using saved data")
                else:
                    print("Found existing elevation checkpoint (streaming mode) - resuming...")

                # Extract batch progress to determine where to resume
                last_batch_idx = elevation_batch_info.get("batch_idx", -1)
                start_batch_idx = last_batch_idx + 1  # Resume from next batch

                total_elevations_fetched = elevation_batch_info.get("elevations_fetched", 0)
                total_elevations_failed = elevation_batch_info.get("elevations_failed", 0)

                # Check if elevation database file exists (persistent storage)
                # The elevation_db is already opened above, so data should be there
                existing_elevation_count = len(elevation_db)

                if existing_elevation_count > 0:
                    print(
                        f"  ✓ Loaded {existing_elevation_count:,} elevations from persistent database"
                    )
                    if not is_complete:
                        print(f"  Total progress: {total_elevations_fetched:,} coordinates fetched")

                    # Check if already complete by verifying all batches were processed
                    if is_complete or start_batch_idx >= num_batches:
                        if not is_complete:
                            print(
                                f"  ✓ All batches processed ({start_batch_idx}/{num_batches}) - skipping elevation stage"
                            )
                        start_batch_idx = num_batches  # Skip all batches
                    else:
                        print(f"  Resuming from batch {start_batch_idx + 1}/{num_batches}")
                elif elevation_mapping:
                    # Fallback: load from coordinate_mapping if database is empty
                    for node_id, elevation in elevation_mapping.items():
                        elevation_db[node_id] = elevation
                        # MEMORY FIX: Track in bloom filter instead of unbounded set
                        if use_bloom:
                            fetched_node_ids_bloom.add(node_id)
                        else:
                            fetched_node_ids.add(node_id)
                    print(f"  ✓ Restored {len(elevation_mapping):,} elevations to database")
                    if not is_complete:
                        print(f"  Total progress: {total_elevations_fetched:,} coordinates fetched")
            elif can_resume_from_db:
                # Resume from disk database: either no metadata file or old/incompatible format
                # All batches will run, but duplicate coordinates will be skipped
                print("Found existing elevation database (streaming mode) - resuming...")
                print(f"  ✓ Loaded {existing_elevation_count:,} elevations from disk database")
                print("  Rebuilding coordinate tracking from database...")

                # Populate tracking from existing database
                # MEMORY FIX: Use bloom filter to avoid loading all node IDs into memory
                print("  Populating in-memory tracking from database...")
                if use_bloom:
                    node_count = 0
                    for node_id in elevation_db.keys():
                        fetched_node_ids_bloom.add(node_id)
                        node_count += 1
                    print(
                        f"  ✓ Ready to resume - tracking {node_count:,} node IDs in bloom filter (memory-efficient)"
                    )
                else:
                    for node_id in elevation_db.keys():
                        fetched_node_ids.add(node_id)
                    print(
                        f"  ✓ Ready to resume - tracking {len(fetched_node_ids):,} node IDs in memory"
                    )
                print("  ✓ Will skip coordinates already in database")
                print("  Note: All batches will run, but duplicate coordinates will be skipped")
                # Don't set start_batch_idx - let all batches run but skip coordinates in db

        # Track progress for smooth updates during elevation fetching
        # We'll update the progress bar fractionally based on coordinate progress within batches
        segments_accounted_for = 0  # Segments we've already counted in progress bar
        current_batch_segment_count = 0  # How many segments in current batch
        current_batch_coord_count = 0  # How many coords to fetch in current batch
        current_batch_coords_done = 0  # How many coords fetched so far in current batch
        fractional_segments_shown = 0.0  # Track fractional progress (for internal calculation only)

        with tqdm(
            total=total_segments,
            desc="Processing",
            unit="segs",
            **TQDM_DEFAULTS,  # Apply global defaults (includes file=stderr when FORCE_TQDM=1)
            smoothing=0.1,  # Smooth out the rate calculation
            initial=(
                min(start_batch_idx * BATCH_SIZE, total_segments) if start_batch_idx > 0 else 0
            ),  # Start from checkpoint position (capped at total to show 100% when complete)
        ) as pbar:
            # Force immediate render when resuming from checkpoint
            if start_batch_idx > 0:
                pbar.refresh()

            # Initialize postfix to show elevation error rate from the start
            total_elevation_attempts = total_elevations_fetched + total_elevations_failed
            error_rate = (
                (total_elevations_failed / total_elevation_attempts * 100)
                if total_elevation_attempts > 0
                else 0
            )
            pbar.set_postfix({"elev_err": f"{total_elevations_failed:,} ({error_rate:.1f}%)"})

            # Set signal handler operation for elevation fetching phase
            # This ensures emergency checkpoints save elevation progress
            signal_handler.set_operation(
                "region_elevation_fetching",
                {
                    "batch_idx": start_batch_idx - 1 if start_batch_idx > 0 else 0,
                    "node_elevations": {},  # Empty - elevations stored in disk database
                    "coords_seen": [],  # Empty - using bloom filter
                    "elevations_fetched": total_elevations_fetched,
                    "elevations_failed": total_elevations_failed,
                    "elevation_db": elevation_db,  # Store reference so signal handler can sync it
                },
            )

            # Define progress callback to update main progress bar incrementally
            def update_progress_callback(n_coords):
                """Called by elevation fetcher when coordinates are processed within a batch."""
                nonlocal current_batch_coords_done, fractional_segments_shown

                current_batch_coords_done += n_coords

                # Calculate error rate for display (do this first so it's always available)
                total_elevation_attempts = total_elevations_fetched + total_elevations_failed
                error_rate = (
                    (total_elevations_failed / total_elevation_attempts * 100)
                    if total_elevation_attempts > 0
                    else 0
                )

                # Calculate what fraction of the current batch is complete
                if current_batch_coord_count > 0:
                    batch_progress_fraction = current_batch_coords_done / current_batch_coord_count
                    # Calculate how many segments worth of progress that represents (fractional)
                    segments_to_show_fractional = (
                        batch_progress_fraction * current_batch_segment_count
                    )
                    # Round to whole number for display
                    segments_to_show_whole = int(segments_to_show_fractional)

                    # Calculate increment since last update (only whole segments)
                    segments_already_shown_whole = int(fractional_segments_shown)
                    increment_whole = segments_to_show_whole - segments_already_shown_whole

                    if increment_whole > 0:
                        pbar.update(increment_whole)
                        fractional_segments_shown = segments_to_show_fractional
                        # Update postfix with elevation error count and rate after each update
                        pbar.set_postfix(
                            {"elev_err": f"{total_elevations_failed:,} ({error_rate:.1f}%)"}
                        )
                    else:
                        # Even if no progress increment, still update postfix (error rate may have changed)
                        pbar.set_postfix(
                            {"elev_err": f"{total_elevations_failed:,} ({error_rate:.1f}%)"},
                            refresh=False,
                        )
                else:
                    # No coords in batch, just update postfix
                    pbar.set_postfix(
                        {"elev_err": f"{total_elevations_failed:,} ({error_rate:.1f}%)"},
                        refresh=False,
                    )

            # Adjust segments_accounted_for if resuming from checkpoint
            if start_batch_idx > 0:
                segments_accounted_for = start_batch_idx * BATCH_SIZE

            # Open checkpoint file once and read sequentially (avoid O(n²) re-reading)
            checkpoint_file_handle = None
            if use_streaming:
                checkpoint_file_handle = open(merged_segments_checkpoint)
                # If resuming, skip to the start batch with progress indicator
                if start_batch_idx > 0:
                    lines_to_skip = start_batch_idx * BATCH_SIZE
                    for _ in tqdm(
                        range(lines_to_skip),
                        desc="Seeking to checkpoint",
                        unit="lines",
                        leave=False,
                        **TQDM_DEFAULTS,
                    ):
                        line = checkpoint_file_handle.readline()
                        if not line:
                            break

            def get_next_batch_from_checkpoint(batch_size):
                """Read next batch sequentially from open file handle."""
                batch = []
                while len(batch) < batch_size:
                    line = checkpoint_file_handle.readline()
                    if not line:
                        break  # EOF
                    if line.strip():
                        segment = json.loads(line)
                        batch.append(segment)
                return batch

            for batch_idx in range(start_batch_idx, num_batches):
                # Calculate batch boundaries for indexing
                batch_start = batch_idx * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, total_segments)

                if use_streaming:
                    # Read next batch sequentially from checkpoint file
                    batch_segments = get_next_batch_from_checkpoint(BATCH_SIZE)
                    if not batch_segments:
                        break  # No more batches
                else:
                    # Slice in-memory segments
                    batch_segments = merged_segments[batch_start:batch_end]

                # Extract coordinates for this batch only
                batch_coords_set = set()
                batch_coord_to_node_ids = defaultdict(list)
                batch_coord_metadata = {}

                for seg_idx_in_batch, segment in enumerate(batch_segments):
                    seg_idx = batch_start + seg_idx_in_batch

                    try:
                        if "nodes" not in segment:
                            skipped_segments += 1
                            continue

                        nodes = segment.get("nodes", [])
                        way_ids = segment.get("way_ids", [])
                        base_id = way_ids[0] if way_ids else f"seg{seg_idx}"
                        street_name = segment.get("street_name", "Unnamed Road")
                        segment["_segment_index"] = seg_idx

                        for i, node in enumerate(nodes):
                            if isinstance(node, dict) and "lat" in node and "lon" in node:
                                try:
                                    coord = (
                                        round(float(node["lat"]), 6),
                                        round(float(node["lon"]), 6),
                                    )
                                    node_id = f"{base_id}_{i}"

                                    # Check if this node already has elevation (for resume from db)
                                    # MEMORY FIX: Use bloom filter when available to avoid unbounded memory growth
                                    if use_bloom:
                                        if node_id not in fetched_node_ids_bloom:
                                            batch_coords_set.add(coord)
                                    else:
                                        if node_id not in fetched_node_ids:
                                            batch_coords_set.add(coord)

                                    if coord not in batch_coord_metadata:
                                        batch_coord_metadata[coord] = {
                                            "way_id": str(base_id),
                                            "street_name": street_name,
                                        }

                                    # node_id already created above for db check
                                    batch_coord_to_node_ids[coord].append(node_id)
                                    total_nodes_processed += 1
                                except:
                                    continue

                    except Exception:
                        skipped_segments += 1
                        continue

                # Fetch elevations for this batch (only for new coords not seen before)
                if use_bloom:
                    batch_coords_list = [
                        c for c in batch_coords_set if c not in global_coords_bloom
                    ]
                else:
                    batch_coords_list = [c for c in batch_coords_set if c not in global_coords_seen]

                # Set up tracking variables for this batch to enable smooth progress updates
                current_batch_segment_count = len(batch_segments)
                current_batch_coord_count = len(batch_coords_list)
                current_batch_coords_done = 0
                fractional_segments_shown = 0.0  # Reset for each batch

                if batch_coords_list:
                    try:
                        # Fetch elevations with progress callback to update main progress bar incrementally
                        elevations = elevation_fetcher._fetch_elevations_parallel(
                            batch_coords_list,
                            persistence,
                            None,  # No progress description - silent mode
                            batch_coord_metadata,
                            None,  # No fetch log for individual batches
                            silent_mode=True,
                            progress_callback=update_progress_callback,
                        )

                        # Map elevations to node IDs and log failures
                        batch_success = 0
                        for coord, elevation in zip(batch_coords_list, elevations):
                            if elevation is not None:
                                # Store by node_id for original segment lookups
                                for node_id in batch_coord_to_node_ids[coord]:
                                    # PERFORMANCE FIX: Write to SQLite database (internally buffered)
                                    # SQLite buffers writes and commits in bulk - much faster than shelve
                                    elevation_db[node_id] = elevation
                                    # MEMORY FIX: Track in bloom filter instead of unbounded set
                                    if use_bloom:
                                        fetched_node_ids_bloom.add(node_id)
                                    else:
                                        fetched_node_ids.add(node_id)

                                # ALSO store by coordinate for merged climb lookups
                                # This enables elevation profile generation for multi-way merged climbs
                                # Round to 6 decimals to match lookup precision
                                coord_key = f"coord_{round(coord[0], 6)}_{round(coord[1], 6)}"
                                elevation_db[coord_key] = elevation
                                if use_bloom:
                                    global_coords_bloom.add(coord)
                                else:
                                    global_coords_seen.add(coord)
                                batch_success += 1
                            else:
                                # Log elevation failure with metadata
                                metadata = batch_coord_metadata.get(coord, {})
                                error_logger.log_coordinate_failure(
                                    coordinate=coord,
                                    street_name=metadata.get("street_name", "Unknown"),
                                    osm_way_id=metadata.get("way_id", "Unknown"),
                                    datasets_tried=["ned10m", "srtm30m"],  # Based on config
                                    primary_dataset=elevation_fetcher.primary_dataset,
                                )

                        total_elevations_fetched += batch_success
                        total_elevations_failed += len(batch_coords_list) - batch_success

                    except Exception as e:
                        pbar.write(f"⚠️  Batch {batch_idx + 1} elevation fetch failed: {e}")
                        total_elevations_failed += len(batch_coords_list)
                else:
                    # No coordinates to fetch - just update progress for the segments directly
                    pbar.update(len(batch_segments))

                # After batch is complete, ensure progress bar shows full batch completion
                # (account for any rounding/fractional differences)
                expected_position = segments_accounted_for + len(batch_segments)
                if pbar.n < expected_position:
                    pbar.update(expected_position - pbar.n)

                # Update accounting for next batch
                segments_accounted_for += len(batch_segments)

                # Note: Database sync now follows checkpoint settings (config.yaml)
                # Syncs occur every 15 min or at progress milestones (10%, 25%, 50%, 75%, 90%)
                # This eliminates sync-after-every-batch overhead while maintaining data safety

                # Ensure postfix is visible after each batch (in case it got cleared)
                total_elevation_attempts = total_elevations_fetched + total_elevations_failed
                error_rate = (
                    (total_elevations_failed / total_elevation_attempts * 100)
                    if total_elevation_attempts > 0
                    else 0
                )
                pbar.set_postfix({"elev_err": f"{total_elevations_failed:,} ({error_rate:.1f}%)"})

                # PERFORMANCE FIX: Flush SQLite buffer every N batches (incremental sync)
                # Only writes NEW elevations since last flush - much faster than full database sync
                # Flush every batch in final 3 batches to minimize data loss on crash
                is_final_batches = (num_batches - batch_idx) <= 3
                if is_final_batches or (batch_idx + 1) % FLUSH_INTERVAL == 0:
                    elevation_db.flush()

                # Signal handler check - allow graceful cancellation
                # NOTE: Elevations are already persisted in disk database (elevation_db)
                # We only checkpoint batch progress and statistics, not the elevations themselves
                signal_handler.set_operation(
                    "region_elevation_fetching",
                    {
                        "batch_idx": batch_idx,
                        "node_elevations": {},  # Empty - elevations already on disk
                        "coords_seen": [],  # Empty - using bloom filter
                        "elevations_fetched": total_elevations_fetched,
                        "elevations_failed": total_elevations_failed,
                        "elevation_db": elevation_db,  # Store reference so signal handler can sync it
                    },
                )

                if signal_handler.kill_now:
                    print("\n\n🛑 Graceful shutdown requested - saving checkpoint...")
                    # CRITICAL: Sync SQLite database to disk (incremental - only new writes)
                    try:
                        count = elevation_db.sync()
                        print(
                            f"  ✓ Synced {count:,} new elevations to disk ({len(elevation_db):,} total)"
                        )
                    except Exception as e:
                        print(f"  ⚠️  Warning: Failed to sync elevation database: {e}")

                    checkpoint_data = {
                        "batch_idx": batch_idx,
                        "node_elevations": {},  # Empty - elevations in disk database
                        "coords_seen": [],  # Empty - using bloom filter
                        "elevations_fetched": total_elevations_fetched,
                        "elevations_failed": total_elevations_failed,
                        "elevation_db_path": str(elevation_db_path),  # Save path for resume
                    }
                    persistence.save_elevation_progress({}, checkpoint_data)
                    from datetime import datetime

                    current_time = datetime.now().strftime("%H:%M")
                    print(
                        f"✓ Checkpoint saved at batch {batch_idx + 1}/{num_batches} ({current_time})"
                    )
                    print(f"  Elevations saved: {len(elevation_db):,}")
                    print("\n⏸️  Analysis paused. Run again to resume from this checkpoint.")
                    elevation_db.close()  # Close shelve database properly
                    sys.exit(0)

                # Check if we should save a checkpoint (based on config.yaml settings)
                if checkpointer.should_checkpoint(batch_idx):
                    from datetime import datetime

                    # CRITICAL: Sync SQLite database to disk (incremental - only new writes)
                    # This ensures elevations are persisted before checkpoint metadata
                    try:
                        elevation_db.sync()
                    except Exception as e:
                        pbar.write(f"⚠️  Database sync failed during checkpoint: {e}")

                    checkpoint_data = {
                        "batch_idx": batch_idx,
                        "node_elevations": {},  # Empty - elevations in disk database
                        "coords_seen": [],  # Empty - using bloom filter
                        "elevations_fetched": total_elevations_fetched,
                        "elevations_failed": total_elevations_failed,
                        "elevation_db_path": str(elevation_db_path),  # Save path for resume
                    }
                    persistence.save_elevation_progress({}, checkpoint_data)

                    current_time = datetime.now().strftime("%H:%M")
                    pbar.write(
                        f"Checkpoint saved at batch {batch_idx + 1}/{num_batches} ({current_time})"
                    )

                # MEMORY FIX: Clear batch memory explicitly (critical for large regions)
                # batch_segments can be 50-100MB of JSON objects, must be deleted to prevent accumulation
                # Note: Some variables may not exist depending on execution path
                try:
                    del batch_coords_set
                except (NameError, UnboundLocalError):
                    pass
                try:
                    del batch_coord_to_node_ids
                except (NameError, UnboundLocalError):
                    pass
                try:
                    del batch_coord_metadata
                except (NameError, UnboundLocalError):
                    pass
                try:
                    del batch_coords_list
                except (NameError, UnboundLocalError):
                    pass
                try:
                    del batch_segments
                except (NameError, UnboundLocalError):
                    pass
                try:
                    del elevations
                except (NameError, UnboundLocalError):
                    pass

                import gc

                gc.collect()

        # Close checkpoint file handle if it was opened
        if checkpoint_file_handle is not None:
            checkpoint_file_handle.close()

        # CRITICAL: Flush any remaining buffered elevations before completion
        try:
            elevation_db.flush()
        except Exception as e:
            print(f"⚠️  Warning: Final flush failed: {e}")

        # Calculate success rate
        total_coords_requested = total_elevations_fetched + total_elevations_failed
        success_rate = (
            (total_elevations_fetched / total_coords_requested * 100)
            if total_coords_requested > 0
            else 0
        )

        print(f"\n✓ Processed {total_segments:,} segments")
        print(f"✓ Total nodes processed: {total_nodes_processed:,}")
        print(f"✓ Unique coordinates: {total_coords_requested:,}")
        print(f"✓ Elevations fetched: {total_elevations_fetched:,} ({success_rate:.1f}%)")
        if total_elevations_failed > 0:
            fail_rate = total_elevations_failed / total_coords_requested * 100
            if fail_rate > 50:
                print(
                    f"⚠️  WARNING: {total_elevations_failed:,} elevations failed ({fail_rate:.1f}%)"
                )
            else:
                print(f"  Failed: {total_elevations_failed:,} ({fail_rate:.1f}%)")
        print(f"✓ Mapped to {len(elevation_db):,} nodes")
        if skipped_segments > 0:
            print(f"  (Skipped {skipped_segments:,} segments with no/invalid nodes)")

        # Update successful datasets in error logger (will be written when stopped)
        successful_datasets = elevation_fetcher.get_successful_datasets()
        error_logger.set_successful_datasets(successful_datasets)

        # Save datasets info: configured priority + actually-used.
        # Use configured_priority (pre-server-filter) so intent is documented
        # even if the local opentopodata server is missing a tier this run.
        datasets_used = elevation_fetcher.get_datasets_used()
        datasets_priority = list(
            getattr(elevation_fetcher, "configured_priority", None)
            or getattr(elevation_fetcher, "dataset_priority", [])
            or []
        )
        if datasets_used or datasets_priority:
            persistence.save_datasets_used(datasets_used, priority=datasets_priority)

        # DON'T clear elevation progress - keep it for resume functionality
        # The elevation_data.db file persists in checkpoints folder
        # Update checkpoint to mark as complete BEFORE sync/close (in case close fails)
        if elevation_mapping or elevation_batch_info or start_batch_idx > 0:
            # Update checkpoint to show all batches complete
            completion_checkpoint = {
                "elevations_fetched": total_elevations_fetched,
                "elevations_failed": total_elevations_failed,
                "complete": True,  # Mark as complete
            }
            persistence.save_elevation_progress({}, completion_checkpoint)
            print("✓ Elevation processing complete - checkpoint updated")

        # Sync and close SQLite database (with defensive exception handling)
        try:
            elevation_db.sync()  # Ensure all data is written to disk
        except Exception as e:
            print(f"⚠️  Warning: Final sync failed: {e}")
            # Data may still be in WAL file - continue to close

        try:
            elevation_db.close()
        except Exception as e:
            print(f"⚠️  Warning: Database close failed: {e}")

        # Elevations are now stored on disk in SQLite database
        node_elevations = elevation_db_path  # Pass path instead of dict

        # Save completed elevations path (NOT the dict - that would load GBs into memory!)
        # The save_completed_elevations method accepts a Path and stores just the path reference
        persistence.save_completed_elevations(elevation_db_path)

    # Initialize error logger if not already initialized (happens when skipping elevation extraction)
    if skip_elevation_extraction:
        from utils.error_logger import ErrorLogger

        region_name = metadata.get("address", "Unknown Region")
        safe_name = "".join(c for c in region_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
        surface_filter = metadata.get("surface_filter", "all")
        min_score = metadata.get("min_score", 0)
        score_type = metadata.get("score_type", "basic")
        cycling_only = metadata.get("cycling_only", False)
        error_logger = ErrorLogger(
            region_name=safe_name or region_name,
            surface_filter=surface_filter,
            min_score=min_score,
            score_type=score_type,
            cycling_allowed=cycling_only,
            output_dir=persistence.analysis_dir,
            base_filename="error",
            app_version=__version__,
        )

    # Continue with climb analysis
    analyzer = ClimbAnalyzer(metadata.get("surface_filter", "all"), unit_system, score_type)

    # Handle elevation database path for lazy loading
    if isinstance(node_elevations, Path):
        # Open SQLite elevation database for reading
        analyzer.elevation_db = ElevationDatabase(node_elevations)
        analyzer.node_elevations = analyzer.elevation_db  # Database acts like a dict
    else:
        # Fallback for when elevations are in memory
        analyzer.node_elevations = node_elevations
        analyzer.elevation_db = None

    # Load merged segments for analysis
    print("\nStep 5: Analyzing climbs...")
    # MEMORY FIX: ALWAYS use streaming analysis for regions (never load all segments into memory)
    # The old non-streaming path (analyze_merged_roads) loads all segments + climbs into memory
    climbs = analyzer.analyze_merged_roads_streaming(
        merged_segments_checkpoint, persistence, batch_size=50000
    )

    # Close elevation database if it was opened (but keep the file for resume)
    if hasattr(analyzer, "elevation_db") and analyzer.elevation_db:
        analyzer.elevation_db.close()
        # NOTE: Do NOT delete the elevation database - it's stored in checkpoints
        # folder and is needed for resume functionality

    # MEMORY FIX: Handle streaming mode for very large datasets
    # If analyzer has climbs_temp_file, we're using streaming mode
    using_streaming_export = hasattr(analyzer, "climbs_temp_file") and analyzer.climbs_temp_file
    climb_count = analyzer.climbs_count if using_streaming_export else len(climbs)

    # Update error logger with climb count (but don't stop logging yet - will be done after xlsx save)
    if not skip_elevation_extraction:
        error_logger.set_analysis_stats(
            climb_count=climb_count,
            total_coords=total_coords_requested,
            failed_coords=total_elevations_failed,
        )
        # Note: error_logger.stop_elevation_logging() will be called after xlsx files are created
        # so we can include the file count in the summary

    # Elevation profiles are now generated during streaming analysis (see analyze_merged_roads_streaming)
    # This avoids iterating through all climbs again and works for any dataset size

    # Get analysis center from metadata
    analysis_center = (metadata.get("center_lat"), metadata.get("center_lon"))

    if climb_count == 0:
        print("No climbs found above threshold")
        import pandas as pd

        # Stop elevation logging if it was started
        if not skip_elevation_extraction and error_logger:
            error_logger.stop_elevation_logging(output_file_count=0)

        return [], pd.DataFrame(), persistence, False, None

    # Set elevation statistics in analyzer for reporting
    analyzer.total_coords_requested = total_coords_requested
    analyzer.total_coords_failed = total_elevations_failed
    if total_coords_requested > 0:
        analyzer.elevation_success_rate = (total_elevations_fetched / total_coords_requested) * 100
    else:
        analyzer.elevation_success_rate = 100.0

    # Build scope info string for results header
    scope_info = None
    scope_type = metadata.get("scope_type")
    if scope_type == "address":
        scope_info = metadata.get("address", "Unknown Address")
    elif scope_type in ("country", "region"):
        # Try to get region name from metadata, then from the persistence
        # analysis_id as a fallback (e.g. "Kansas_all_region_1776687510" -> "Kansas")
        # since scope_info drives the output filename prefix.
        region_name = (
            metadata.get("region_name") or metadata.get("country") or metadata.get("address")
        )
        if not region_name and persistence is not None:
            aid = getattr(persistence, "analysis_id", "") or ""
            # analysis_id format: "<RegionName>_<surface>_<scope>_<timestamp>"
            for suffix in ("_all_region_", "_all_country_", "_all_address_"):
                if suffix in aid:
                    region_name = aid.split(suffix, 1)[0]
                    break
        if region_name:
            # Extract just the region name from path formats like "us > hawaii" or "europe/france"
            if " > " in region_name:
                scope_info = region_name.split(" > ")[-1]
            elif "/" in region_name:
                scope_info = region_name.split("/")[-1]
            else:
                scope_info = region_name
            # Convert to title case for cleaner filenames (e.g., "hawaii" -> "Hawaii")
            scope_info = scope_info.replace("-", " ").replace("_", " ").title()

    # Generate results
    print("\nGenerating results...")
    if enable_geocoding:
        print("Performing location lookup and distance calculations...")
    else:
        print("Skipping location lookup (faster results)...")

    df = analyzer.print_climb_results(
        min_score, unit_system, analysis_center, persistence, enable_geocoding, scope_info
    )

    # Step 6: Post-process boundary merges (memory-efficient)
    # Only run for regional analysis (not address searches) as they may have chunk boundaries
    scope_type = metadata.get("scope_type", "unknown")
    enable_boundary_merge = metadata.get("enable_cross_chunk_postprocess", True)

    # Boundary merge memory threshold: skip for very large datasets to prevent OOM
    # Estimated memory usage: ~10KB per climb + overhead for merge data structures
    # Safe limit: 200K climbs (~2GB base + ~2GB overhead = ~4GB total)
    # Step 6: Climb merging now happens in save_large_dataframe_as_split_excel()
    # This allows merging before file splitting and uses the unified merger
    # which handles within-region, cross-region, and country boundary rules
    if scope_type == "address":
        print("\nStep 6: Boundary merge not needed for address search")
    elif scope_type in ("country", "region") and enable_boundary_merge:
        print("\nStep 6: Climb merging will be performed during file save using unified merger")

    # Return with cloud cache upload flag and error_logger (to be closed after xlsx save)
    # error_logger will be None if skip_elevation_extraction is True
    return_error_logger = error_logger if not skip_elevation_extraction else None
    return climbs, df, persistence, should_upload_to_cache, return_error_logger


def process_all_chunks_serial(
    persistence,
    road_analyzer,
    chunk_merger,
    boundary_merger,
    elevation_fetcher,
    chunks,
    metadata,
    min_score,
    unit_system,
    score_type,
    enable_geocoding,
):
    """Serial version of process_all_chunks for local deployment"""

    # Setup signal handling
    signal_handler.set_persistence_manager(persistence)

    # Cloud cache upload not supported in chunked mode
    should_upload_to_cache = False
    total_chunks = len(chunks)
    processed_chunks = []
    total_segments_count = 0  # Track count instead of storing all segments

    print("Processing road data serially with checkpoint saving...")

    # Disable automatic GC for performance - we'll run it manually
    gc.disable()

    # Initialize smart checkpointer
    checkpointer = SmartCheckpointer(total_chunks, "Serial Chunk Processing")

    # Track processing start time for seg/s calculation
    processing_start_time = time.time()

    print("\nNote: Progress is based on map chunks completed. Dense urban areas")
    print("      may process at a slower rate but still show steady progress.\n")
    with tqdm(
        total=total_chunks,
        desc="Processing road segment chunks ",
        unit="chunk",
        mininterval=0.1,
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
    ) as pbar:
        for chunk_index, (chunk_lat, chunk_lon, chunk_radius) in enumerate(chunks):
            # Signal handler check
            signal_handler.set_operation(
                "chunk_processing",
                {
                    "processed_chunks": processed_chunks,
                    "total_chunks": total_chunks,
                    "metadata": metadata,
                },
            )

            if signal_handler.kill_now:
                persistence.save_progress(processed_chunks, total_chunks, metadata)
                print("Chunk progress saved. Analysis can be resumed.")
                sys.exit(0)

            try:
                # Periodic rtree index reload to prevent performance degradation
                # Reload every 5 chunks - testing showed 34% slowdown can occur within just 1 chunk
                # This suggests rtree internal state degrades based on query complexity, not just count
                if chunk_index > 0 and chunk_index % 5 == 0:
                    if (
                        hasattr(road_analyzer, "spatial_index_manager")
                        and road_analyzer.spatial_index_manager
                    ):
                        road_analyzer.spatial_index_manager.reload_index()

                # Get roads for this chunk
                chunk_ways = road_analyzer.get_roads_in_chunk(chunk_lat, chunk_lon, chunk_radius)

                if chunk_ways:
                    chunk_ways_count = len(chunk_ways)

                    # Show sub-progress bar for chunks with 500+ ways
                    if chunk_ways_count > 500:
                        print(
                            f"\n  Chunk {chunk_index} has {chunk_ways_count} ways - showing blue progress..."
                        )
                        chunk_segments = chunk_merger.process_way_chunk(
                            chunk_ways, show_progress=True
                        )
                    else:
                        chunk_segments = chunk_merger.process_way_chunk(
                            chunk_ways, show_progress=False
                        )

                    chunk_segments_count = len(chunk_segments)

                    # Save chunk data to disk
                    persistence.save_chunk(
                        chunk_index,
                        chunk_segments,
                        (chunk_lat, chunk_lon, chunk_radius),
                    )

                    # Track segment count (no need to keep in memory - already saved to disk)
                    total_segments_count += chunk_segments_count
                else:
                    # Save empty chunk to mark as processed
                    persistence.save_chunk(chunk_index, [], (chunk_lat, chunk_lon, chunk_radius))
                    chunk_ways_count = 0
                    chunk_segments_count = 0

                processed_chunks.append(chunk_index)

                # Calculate segments per second
                elapsed_time = time.time() - processing_start_time
                seg_per_sec = total_segments_count / elapsed_time if elapsed_time > 0 else 0

                # Smart checkpoint save
                if checkpointer.should_checkpoint(chunk_index):
                    persistence.save_progress(processed_chunks, total_chunks, metadata)

                    info = checkpointer.get_checkpoint_info(chunk_index)
                    pbar.set_postfix(
                        {
                            "ways": chunk_ways_count,
                            "segs": chunk_segments_count,
                            "total_segs": total_segments_count,
                            "seg/s": f"{seg_per_sec:.0f}",
                            "next_save": f"{info['time_until_next_min']:.1f}min",
                        }
                    )
                else:
                    pbar.set_postfix(
                        {
                            "ways": chunk_ways_count,
                            "segs": chunk_segments_count,
                            "total_segs": total_segments_count,
                            "seg/s": f"{seg_per_sec:.0f}",
                        }
                    )

                # Manual GC every 10 chunks for memory management
                if len(processed_chunks) % 10 == 0:
                    gc.collect()

                # Update progress bar immediately after each chunk
                pbar.update(1)
                pbar.refresh()  # Force refresh

            except Exception as e:
                print(f"\nError processing chunk {chunk_index}: {e}")
                # Save empty chunk to mark as processed
                persistence.save_chunk(chunk_index, [], (chunk_lat, chunk_lon, chunk_radius))
                processed_chunks.append(chunk_index)
                pbar.update(1)

    # Re-enable automatic GC
    gc.enable()
    gc.collect()

    print(f"\nCollected {total_segments_count} road segments from all chunks")

    # Load all segments from disk (they were saved chunk by chunk)
    all_merged_segments = persistence.load_all_segments()

    # Continue with the rest of the analysis
    return complete_analysis_from_segments(
        all_merged_segments,
        elevation_fetcher,
        boundary_merger,
        metadata,
        min_score,
        unit_system,
        persistence,
        score_type,
        enable_geocoding,
        should_upload_to_cache=should_upload_to_cache,
    )


def analyze_area(
    address: str,
    country: str = None,
    radius_km: float = 15,
    surface_filter: str = "all",
    unit_system: str = "imperial",
    min_score: float = 0,
    chunk_size_km: float = 30.0,  # Optimized: larger chunks reduce overhead while maintaining constant memory usage
    center_lat: float = None,
    center_lon: float = None,
    formatted_address: str = None,
    score_type: str = "basic",
    cycling_only: bool = True,
    enable_geocoding: bool = True,
    max_workers: int = OSM_MAX_THREADS,
    skip_data_validation: bool = False,  # Skip validation if already done in main()
    scope_type: str = "address",  # "address", "region", or "country"
    batch_mode: bool = False,  # Enable auto-resume in batch/CLI mode
    ignore_checkpoints: bool = False,  # Ignore existing checkpoints and start fresh (--ignore-checkpoints flag)
    skip_cloud_cache_check: bool = False,  # Skip cloud cache check if already done in main()
):
    """Analysis for areas with resumable chunk processing."""

    deployment_type = DEPLOYMENT_TYPE

    # Determine region name early (needed for OSM file lookup)
    # NOTE: For address searches, we don't have formatted_address yet (geocoding happens later)
    # So we'll need to defer OSM file selection until after geocoding
    region_name = None  # Will be determined later for address searches
    region_canonical_path = None  # Full path like "us/georgia" for disambiguation

    if scope_type != "address":
        # For state/country searches, we have formatted_address already
        if country == "United States" and formatted_address:
            # Extract state name from formatted address
            # Format can be:
            #   "Hawaii, United States"
            #   "7300, Dunhill Terrace Northeast, Atlanta, Fulton County, Georgia, 30328, United States"
            # Need to find the state name within the comma-separated parts
            address_parts = [part.strip() for part in formatted_address.split(",")]

            # Look for a state name in the address parts (case-insensitive)
            region_name = country  # Default to country
            for part in address_parts:
                # Check if this part is a US state using geo_lookup
                if is_us_state(part):
                    region_name = part
                    region_canonical_path = f"us/{part.lower().replace(' ', '-')}"
                    break
                # Try title case match
                part_title = part.title()
                if is_us_state(part_title):
                    region_name = part_title
                    region_canonical_path = f"us/{part_title.lower().replace(' ', '-')}"
                    break
        elif scope_type == "region" and address:
            # For region-based analysis, extract just the region name (not full path)
            # address format is like "antarctica", "europe > andorra", or "us/north-carolina"
            if "/" in address:
                region_canonical_path = address  # Preserve "us/north-carolina"
                region_name = address.split("/")[-1]  # "us/north-carolina" -> "north-carolina"
            elif " > " in address:
                # "us > georgia" format - convert to canonical path
                parts = address.split(" > ")
                region_canonical_path = "/".join(p.strip().lower().replace(" ", "-") for p in parts)
                region_name = parts[-1]  # "europe > andorra" -> "andorra"
            else:
                region_name = address
            # Convert to title case for cleaner display/filenames
            region_name = region_name.replace("-", " ").title()
        else:
            # Country-scope: 'address' can be a canonical path like "canada/alberta".
            # Strip any continent/country prefix and title-case so downstream
            # consumers (analysis_id, metadata, filenames) get a clean region name.
            raw = country if country else address
            if raw and "/" in raw:
                region_canonical_path = raw
                raw = raw.split("/")[-1]
            if raw:
                region_name = raw.replace("-", " ").replace("_", " ").title()
            else:
                region_name = raw
    # For address searches, region_name stays None and will be determined after geocoding

    # For address searches, geocode FIRST to get the state name
    if scope_type == "address" and not formatted_address:
        # Need to geocode to get state for OSM file selection
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="climb_analyzer", timeout=10)
        print(f"Geocoding address: {address}")
        try:
            location_result = geolocator.geocode(address, country_codes=country, timeout=10)
            if location_result:
                formatted_address = location_result.address
                center_lat = location_result.latitude
                center_lon = location_result.longitude
                print(f"Found location: {formatted_address}")
                print(f"Coordinates: {center_lat:.6f}, {center_lon:.6f}")

                # Now extract region for region_name
                if "United States" in formatted_address:
                    # US address - extract state name
                    address_parts = [part.strip() for part in formatted_address.split(",")]
                    for part in address_parts:
                        if is_us_state(part):
                            region_name = part
                            break
                    if not region_name:
                        region_name = "United States"
                else:
                    # Non-US address - extract country name
                    address_parts = [part.strip() for part in formatted_address.split(",")]
                    # Country is typically the last part in the address
                    if address_parts:
                        region_name = address_parts[-1]
                        print(f"Detected international region: {region_name}")
                    else:
                        region_name = "International"
            else:
                print(f"Could not geocode address: {address}")
                region_name = "International"
        except Exception as e:
            print(f"Geocoding error: {e}")
            region_name = "International"

    # Cloud mode: Validate radius limits and block regional scope
    if deployment_type == "cloud":
        # Block region/country scope in cloud mode
        if scope_type in ["region", "country"]:
            print("\n" + "=" * 70)
            print("❌ ERROR: Regional analysis not supported in cloud mode")
            print("=" * 70)
            print(f"\nYou requested {scope_type} analysis, which requires downloading")
            print("entire state/country OSM data via Overpass API.")
            print("\nThis is not supported because:")
            print("  • Very large data transfers (hundreds of MB to GB)")
            print("  • API timeout limits (180 seconds max)")
            print("  • API memory limits (1-2 GB response size)")
            print("  • Risk of server-side errors and rate limiting")
            print("\n" + "─" * 70)
            print("RECOMMENDED SOLUTION: Switch to local mode")
            print("─" * 70)
            print("\nRun the setup command to download OSM files locally:")
            print("  $ ./climb-analyzer setup")
            print("\nLocal mode benefits:")
            print("  ✓ Analyze entire states/countries")
            print("  ✓ 3-5x faster processing")
            print("  ✓ No API rate limits")
            print("  ✓ Works offline")
            print("  ✓ Uses pre-built spatial indexes")
            print("=" * 70 + "\n")
            sys.exit(1)

        # Validate address search radius (allow small rounding tolerance)
        if scope_type == "address" and radius_km > (CLOUD_MODE_MAX_RADIUS_KM + 0.5):
            miles_requested = radius_km / 1.60934
            print("\n" + "=" * 70)
            print("❌ ERROR: Search radius exceeds cloud mode limit")
            print("=" * 70)
            print(f"\nRequested radius: {radius_km:.1f} km ({miles_requested:.1f} miles)")
            print(
                f"Cloud mode limit:  {CLOUD_MODE_MAX_RADIUS_KM:.1f} km ({CLOUD_MODE_MAX_RADIUS_MILES:.1f} miles)"
            )
            print("\nCloud mode uses Overpass API with the following limits:")
            print("  • Query timeout: 180 seconds")
            print("  • Memory limit: 1-2 GB response size")
            print("  • Rate limiting: ~2 requests/second")
            print(
                f"\nSearches over {CLOUD_MODE_MAX_RADIUS_MILES:.0f} miles risk timeouts and failures."
            )
            print("\n" + "─" * 70)
            print("OPTIONS:")
            print("─" * 70)
            print(f"\n1. Reduce search radius to {CLOUD_MODE_MAX_RADIUS_MILES:.0f} miles or less")
            print(f"   Re-run with: --distance {CLOUD_MODE_MAX_RADIUS_MILES:.0f}")
            print("\n2. Switch to local mode (RECOMMENDED for large areas)")
            print("   Run: ./climb-analyzer setup")
            print("\n   Local mode benefits:")
            print("     ✓ No radius limits")
            print("     ✓ 3-5x faster processing")
            print("     ✓ No API rate limits")
            print("     ✓ Works offline")
            print("=" * 70 + "\n")
            sys.exit(1)

    # Validate data availability for local deployment (skip if already validated)
    if deployment_type == "local" and not skip_data_validation:
        from utils.data_validator import ensure_data_available

        # Get bounding box
        if scope_type == "address":
            # For address searches, use center + radius
            if center_lat is not None and center_lon is not None:
                lat_center, lon_center = center_lat, center_lon
            else:
                # Will be geocoded later, skip validation for now
                lat_center, lon_center = None, None

            # Only validate if we have coordinates
            if lat_center is not None and lon_center is not None and radius_km is not None:
                # Calculate approximate bbox from center + radius
                import math

                lat_offset = radius_km / 111.0
                lon_offset = radius_km / (111.0 * math.cos(math.radians(lat_center)))

                lat_min = lat_center - lat_offset
                lat_max = lat_center + lat_offset
                lon_min = lon_center - lon_offset
                lon_max = lon_center + lon_offset

                # Validate data availability
                data_available = ensure_data_available(
                    region_name, lat_min, lat_max, lon_min, lon_max
                )

                if not data_available:
                    # User chose to switch to cloud mode
                    print("\n→ Switching to cloud deployment mode")
                    deployment_type = "cloud"
        else:
            # For state/country, get bbox from region data
            from climb_analyzer.data.manager import DataManager

            manager = DataManager()

            is_state = scope_type == "region"
            bounds = manager.get_region_bounds(region_name or address, is_state=is_state)

            if bounds:
                # get_region_bounds returns (lat_min, lon_min, lat_max, lon_max)
                lat_min, lon_min, lat_max, lon_max = bounds
                data_available = ensure_data_available(
                    region_name or address, lat_min, lat_max, lon_min, lon_max
                )

                if not data_available:
                    print("\n→ Switching to cloud deployment mode")
                    deployment_type = "cloud"

    # Cloud Cache: Offer to run clean analysis if filters applied and not cached
    should_upload_to_cache = False

    if (
        CLOUD_CACHE_AVAILABLE
        and CLOUD_CACHE_ENABLED
        and deployment_type == "local"
        and not skip_cloud_cache_check
    ):
        try:
            cloud_cache = CloudCacheManager()

            # FIX: For country-level analysis, country param is None, use address instead
            country_for_cache = (
                country if country else (address if scope_type == "country" else None)
            )
            cache_path = cloud_cache.get_cache_path(country_for_cache, region_name, scope_type)

            if cache_path:
                is_clean = cloud_cache.is_clean_analysis(
                    surface_filter, min_score, cycling_only
                )
                location_display = region_name if scope_type == "region" else country_for_cache
                print(f"\nChecking cloud cache for {location_display}...")
                cache_info = cloud_cache.check_cached(country_for_cache, region_name, scope_type)

                if cache_info["exists"]:
                    # Build version info string
                    version_info = ""
                    if cache_info.get("version"):
                        version_info = f" v{cache_info['version']}"
                    if cache_info.get("error_count") is not None:
                        version_info += f" - {cache_info['error_count']} elevation errors"

                    print(
                        f"✓ {location_display} found in cloud cache (analyzed: {cache_info['date']}{version_info})"
                    )
                else:
                    print(f"✗ {location_display} not found in cloud cache")

                if not cache_info["exists"]:
                    # Region not in cloud cache - check if we should upload after analysis
                    if is_clean:
                        # User is already running clean analysis - enable upload after completion
                        should_upload_to_cache = True
                        print("  Analysis will be uploaded to cloud cache after completion.")
                    else:
                        # User has filters, not cached - auto-run clean analysis per CLOUD_CACHE_ENABLED config
                        print(f"\n{'='*70}")
                        print(f"ℹ️  {location_display} not in cloud cache")
                        print(f"{'='*70}")
                        print("CLOUD_CACHE_ENABLED is True - automatically running clean analysis")
                        print("to share with community (all roads, no filters, min_score=0)")
                        print(f"{'='*70}\n")

                        # Override to clean settings
                        print("Overriding to clean analysis settings for cloud cache contribution:")
                        print("  • Surface: all roads")
                        print("  • Min score: 0")
                        print("  • Cycling filter: off")
                        print("  • Score type: basic\n")

                        surface_filter = "all"
                        min_score = 0
                        cycling_only = False
                        score_type = "basic"
                        should_upload_to_cache = True
        except Exception as e:
            print(f"\n⚠️  Cloud cache check failed: {e}")

    # Initialize components based on deployment type
    if deployment_type == "local":
        # SERIAL PROCESSING FOR LOCAL MODE (memory constraints)
        # Note: Multiprocessing attempted but each worker loads entire 1.5GB spatial index = 24GB for 16 workers
        max_workers = 1

        # Find the correct OSM file for this region
        from utils.data_validator import find_osm_file_for_region

        osm_file = None

        # Try to find OSM file by region name
        if region_name:
            # For region scope, extract the simple region name for OSM file lookup
            # e.g., "europe > andorra" -> "andorra"
            osm_region_name = region_name
            if scope_type == "region" and " > " in region_name:
                parts = region_name.split(" > ")
                osm_region_name = parts[-1]  # Extract last part for OSM file lookup

            print(f"Looking for OSM file for region: {osm_region_name}")
            osm_file = find_osm_file_for_region(osm_region_name, canonical_path=region_canonical_path)
            if osm_file:
                print(f"✓ Found region-specific OSM file: {osm_file}")
            else:
                # Will auto-detect most recent OSM file below
                osm_file = None
        else:
            # Only use configured/cached file if no region name specified
            try:
                osm_file_path = get_configured_osm_file_path()
                osm_file = Path(osm_file_path) if osm_file_path else None
            except FileNotFoundError:
                osm_file = None

        if osm_file and osm_file.exists():
            from climb_analyzer.utils.formatting import print_dim

            print_dim(f"Using OSM file: {osm_file.name}")
            road_analyzer = ChunkedRoadNetworkAnalyzer(
                surface_filter,
                chunk_size_km,
                cycling_only,
                osm_file_path=str(osm_file),
            )
        else:
            from climb_analyzer.utils.formatting import print_dim

            # No region-specific OSM file found, will auto-detect
            print_dim("   Will auto-detect OSM file from data/planet_osm_data/...")
            road_analyzer = ChunkedRoadNetworkAnalyzer(
                surface_filter,
                chunk_size_km,
                cycling_only,
            )
    else:
        # Cloud deployment - use serial processing to respect Overpass API rate limits
        max_workers = 1
        road_analyzer = ChunkedRoadNetworkAnalyzer(surface_filter, chunk_size_km, cycling_only)

    # Initialize all required components
    # Get region name for dataset priority selection
    # Use region_name parameter if provided, otherwise get from config
    region_name_for_elevation = region_name if region_name else get_region_name_from_config()
    elevation_fetcher = FastElevationFetcher(region_name=region_name_for_elevation)
    chunk_merger = MemoryEfficientMerger()
    boundary_merger = BoundaryMerger(coordinate_tolerance=0.002)

    # Rest of the function continues as before...
    from climb_analyzer.utils.formatting import print_section_simple

    print_section_simple("Starting Analysis", spacing_before=1)
    if scope_type == "address":
        print(f"Region: {radius_km:.1f}km radius around {address}")
    else:
        print(f"Region: {region_name or address}")
    # Show analysis settings in compact format
    cycling_str = "cycling-only" if cycling_only else "all-access"
    print(f"Settings: {surface_filter} surface | {score_type} scoring | {cycling_str}")

    # Verbose: show detailed configuration
    verbose_print(f"\n  [verbose] Deployment: {DEPLOYMENT_TYPE}")
    verbose_print(f"  [verbose] Max workers: {max_workers}")
    verbose_print(f"  [verbose] Chunk size: {chunk_size_km} km")
    verbose_print(f"  [verbose] Unit system: {unit_system}")
    verbose_print(f"  [verbose] Min score: {min_score}")

    total_start_time = time.time()

    # Create analysis ID
    if scope_type == "address":
        radius_str = str(radius_km).replace(".", "")
        base_analysis_id = f"{address}_{surface_filter}_{radius_str}km"
    else:
        # For state/country, use region name instead of radius
        base_analysis_id = f"{region_name or address}_{surface_filter}_{scope_type}"
    base_analysis_id = "".join(c for c in base_analysis_id if c.isalnum() or c in ("_", "-"))[:40]

    # Check for existing incomplete analyses
    # Auto-resume by default for all runs (batch, single-region, CLI)
    # This allows seamless resumption after interruptions
    base_dir = CHECKPOINT_DIR

    # Skip checkpoint detection entirely if --ignore-checkpoints is set
    if ignore_checkpoints:
        print("--ignore-checkpoints: Starting fresh analysis (existing checkpoints ignored)")
        existing_analysis_id = None
    else:
        # Default: auto-resume from checkpoints if they exist
        existing_analysis_id = find_existing_analysis(
            base_analysis_id, base_dir, auto_resume=True
        )

    if existing_analysis_id:
        print(f"Found existing analysis: {existing_analysis_id}")

        # Determine if this is streaming mode or chunked mode checkpoint
        checkpoint_dir = base_dir / existing_analysis_id
        is_streaming_checkpoint = (
            (checkpoint_dir / "filtered_ways.jsonl").exists()
            or (checkpoint_dir / "merged_segments.jsonl").exists()
            or (checkpoint_dir / "segments_sorted.jsonl").exists()
        )
        is_chunked_checkpoint = (checkpoint_dir / "progress.pkl").exists()

        if is_streaming_checkpoint:
            # For streaming mode: just reuse the existing analysis_id
            # The process_region function will detect and resume from checkpoints
            # Auto-resume is the default behavior (use --ignore-checkpoints to start fresh)
            analysis_id = existing_analysis_id
            print(f"✓ Auto-resuming analysis: {analysis_id}")

        elif is_chunked_checkpoint:
            # Check what phase the analysis is in
            persistence = ChunkPersistenceManager(existing_analysis_id)
            processed_chunks, total_chunks, metadata = persistence.load_progress()
            elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()

            # Determine phase
            chunks_complete = len(processed_chunks) >= total_chunks
            elevation_in_progress = len(elevation_mapping) > 0

            if batch_mode:
                # Batch mode: automatically resume (no prints needed)
                resume_choice = "y"
            else:
                # Interactive mode: prompt user
                if chunks_complete and elevation_in_progress:
                    print("PHASE: ELEVATION FETCHING")
                    resume_choice = input("Resume elevation fetching? (y/n): ").strip().lower()
                elif chunks_complete and not elevation_in_progress:
                    print("PHASE: READY FOR ELEVATION")
                    resume_choice = input("Start elevation analysis? (y/n): ").strip().lower()
                else:
                    print("PHASE: CHUNK PROCESSING")
                    resume_choice = input("Resume chunk processing? (y/n): ").strip().lower()

            if resume_choice == "y":
                return resume_chunk_processing(
                    existing_analysis_id, min_score, unit_system, score_type
                )
            else:
                print("Session resume declined - starting new analysis")
                analysis_id = f"{base_analysis_id}_{int(time.time())}"
        else:
            print("⚠️  Unknown checkpoint type - starting new analysis")
            analysis_id = f"{base_analysis_id}_{int(time.time())}"
    else:
        # No existing analysis found - create new
        analysis_id = f"{base_analysis_id}_{int(time.time())}"
    # print(f"\nStarting new analysis: {analysis_id}")

    # Get center coordinates
    if center_lat is not None and center_lon is not None:
        from climb_analyzer.utils.formatting import print_dim

        print_dim(f"Using pre-calculated coordinates for: {formatted_address or address}")
        print(f"Center: {formatted_address or address}")
        print(f"Coordinates: {center_lat:.6f}, {center_lon:.6f}\n")
        final_formatted_address = formatted_address or address
    else:
        center_lat, center_lon, final_formatted_address = (
            road_analyzer.get_coordinates_from_address(address, country)
        )
        print(f"Center: {final_formatted_address}")
        print(f"Coordinates: {center_lat:.6f}, {center_lon:.6f}\n")

    # Calculate chunks (only for address searches with radius)
    if scope_type == "address" and radius_km is not None:
        chunks = road_analyzer.calculate_chunks(center_lat, center_lon, radius_km)
        total_chunks = len(chunks)
    else:
        chunks = []
        total_chunks = 0

    # Save analysis metadata
    analysis_metadata = {
        "address": address,
        "country": country,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": radius_km,
        "surface_filter": surface_filter,
        "unit_system": unit_system,
        "chunk_size_km": chunk_size_km,
        "total_chunks": total_chunks,
        "chunks": chunks,
        "formatted_address": final_formatted_address,
        "score_type": score_type,
        "cycling_only": cycling_only,
        "enable_geocoding": enable_geocoding,
        "max_workers": max_workers,
        "deployment_type": deployment_type,
        "scope_type": scope_type,
        "region_name": region_name,
        "enable_cross_chunk_postprocess": ENABLE_CROSS_CHUNK_POSTPROCESS,
    }

    # Initialize persistence manager
    persistence = ChunkPersistenceManager(analysis_id)

    # Decide on processing strategy based on scope and deployment type
    # Use optimized full-region extraction when:
    #   1. Local mode: All scope types (addresses, states, countries)
    #   2. Cloud mode: Address searches only (within radius limits validated above)
    # Use chunking when:
    #   1. Cloud mode: State/country (already blocked above, so this won't happen)
    #   2. Legacy code path (shouldn't reach this anymore)

    use_regional_extraction = deployment_type == "local" or (
        deployment_type == "cloud" and scope_type == "address"
    )

    if use_regional_extraction:
        if deployment_type == "local":
            print("\n✓ Local mode detected - using optimized full-region extraction")
        else:  # cloud mode
            print(
                "\n✓ Cloud mode with address search - using optimized full-region extraction"
            )
            print(
                f"   Search radius: {radius_km:.1f} km (~{radius_km/1.60934:.1f} miles) - within cloud mode limit"
            )

        if scope_type == "address":
            print(f"   Address search within {radius_km}km radius will extract all ways at once")
        else:
            print(f"   {scope_type.capitalize()} analysis will extract all ways at once")

        # Calculate bounding box for the entire region
        import math

        if scope_type == "address":
            # For address searches, use radius to calculate bbox
            lat_offset = radius_km / 111.0
            lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))

            min_lat = center_lat - lat_offset
            max_lat = center_lat + lat_offset
            min_lon = center_lon - lon_offset
            max_lon = center_lon + lon_offset
        else:
            # For state/country, get bbox from region data.
            # Use canonical path (e.g., "us/georgia") when available to avoid
            # ambiguity with same-named countries (e.g., Georgia the country).
            from climb_analyzer.data.manager import DataManager

            manager = DataManager()

            is_state = scope_type == "region"
            bounds_lookup_name = region_canonical_path or region_name or address
            bounds = manager.get_region_bounds(bounds_lookup_name, is_state=is_state)

            if not bounds:
                raise ValueError(f"Could not determine bounds for region: {bounds_lookup_name}")

            # get_region_bounds returns (lat_min, lon_min, lat_max, lon_max)
            min_lat, min_lon, max_lat, max_lon = bounds

        return process_region_without_chunking(
            persistence,
            road_analyzer,
            chunk_merger,
            boundary_merger,
            elevation_fetcher,
            min_lat,
            min_lon,
            max_lat,
            max_lon,
            analysis_metadata,
            min_score,
            unit_system,
            score_type,
            enable_geocoding,
            surface_filter=surface_filter,
            cycling_only=cycling_only,
        )

    # Fallback: Process chunks (legacy code path, should not reach here anymore)
    # Serial processing includes periodic rtree reload to prevent performance degradation
    print("\n⚠️  Using legacy chunked processing (this should not happen)")
    if deployment_type == "local":
        return process_all_chunks_serial(
            persistence,
            road_analyzer,
            chunk_merger,
            boundary_merger,
            elevation_fetcher,
            chunks,
            analysis_metadata,
            min_score,
            unit_system,
            score_type,
            enable_geocoding,
        )
    else:
        # Cloud mode: Use serial processing
        return process_all_chunks_serial(
            persistence,
            road_analyzer,
            chunk_merger,
            boundary_merger,
            elevation_fetcher,
            chunks,
            analysis_metadata,
            min_score,
            unit_system,
            score_type,
            enable_geocoding,
        )

    # Get center coordinates
    if center_lat is not None and center_lon is not None:
        from climb_analyzer.utils.formatting import print_dim

        print_dim(f"Using pre-calculated coordinates for: {formatted_address or address}")
        print(f"Center: {formatted_address or address}")
        print(f"Coordinates: {center_lat:.6f}, {center_lon:.6f}\n")
        final_formatted_address = formatted_address or address
    else:
        center_lat, center_lon, final_formatted_address = (
            road_analyzer.get_coordinates_from_address(address, country)
        )
        print(f"Center: {final_formatted_address}")
        print(f"Coordinates: {center_lat:.6f}, {center_lon:.6f}\n")

    # Calculate chunks (only for address searches with radius)
    if scope_type == "address" and radius_km is not None:
        chunks = road_analyzer.calculate_chunks(center_lat, center_lon, radius_km)
        total_chunks = len(chunks)
    else:
        chunks = []
        total_chunks = 0

    # Save analysis metadata
    analysis_metadata = {
        "address": address,
        "country": country,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": radius_km,
        "surface_filter": surface_filter,
        "unit_system": unit_system,
        "chunk_size_km": chunk_size_km,
        "total_chunks": total_chunks,
        "chunks": chunks,
        "formatted_address": final_formatted_address,
        "score_type": score_type,
        "cycling_only": cycling_only,
        "enable_geocoding": enable_geocoding,
        "max_workers": max_workers,
        "deployment_type": deployment_type,
        "scope_type": scope_type,
        "region_name": region_name,
        "enable_cross_chunk_postprocess": ENABLE_CROSS_CHUNK_POSTPROCESS,
    }

    # Initialize persistence manager
    persistence = ChunkPersistenceManager(analysis_id)

    # Process chunks
    return process_all_chunks_parallel(
        persistence,
        road_analyzer,
        chunk_merger,
        boundary_merger,
        elevation_fetcher,
        chunks,
        analysis_metadata,
        min_score,
        unit_system,
        score_type,
        enable_geocoding,
        max_workers,
    )


def complete_analysis_from_segments(
    all_merged_segments: List[Dict],
    elevation_fetcher: FastElevationFetcher,
    boundary_merger: BoundaryMerger,
    metadata: Dict,
    min_score: float,
    unit_system: str,
    persistence: ChunkPersistenceManager,
    score_type: str = "basic",
    enable_geocoding: bool = True,
    is_resumed: bool = False,
    should_upload_to_cache: bool = False,
):
    # Check for existing completed elevation data FIRST
    completed_elevations = persistence.load_completed_elevations()

    if completed_elevations:
        print(f"Loading completed elevation data for {len(completed_elevations)} nodes...")
        node_elevations = completed_elevations

    else:
        # MEMORY FIX: DO NOT load pickled segments - this loads GBs into memory!
        # The legacy pickle save/load logic has been removed to prevent OOM on large regions.
        # For region analysis, use process_region_without_chunking() (streaming mode) instead.
        # This function should only be used for small address-based searches where
        # all_merged_segments is already in memory from the search query.

        # Check if final processed segments exist (legacy file from old runs)
        final_segments_file = persistence.analysis_dir / "final_processed_segments.pkl"
        if final_segments_file.exists():
            # Delete legacy file to prevent future runs from trying to load it
            try:
                final_segments_file.unlink()
                print("⚠️  Removed legacy final_processed_segments.pkl file")
            except Exception as e:
                print(f"⚠️  Warning: Could not remove legacy file: {e}")

        # For large regions, this code path should NOT be reached.
        # If all_merged_segments is empty, we cannot proceed.
        if not all_merged_segments:
            raise RuntimeError(
                "complete_analysis_from_segments called with no segments and no completed elevations. "
                "For region analysis, use process_region_without_chunking() (streaming mode) instead."
            )

        # Process segments (only for address searches with small datasets)
        print("Processing segments...")
        all_merged_segments = process_segments(
            all_merged_segments, boundary_merger, persistence, auto_resume=is_resumed
        )
        check_and_cleanup_memory(force_cleanup=True)

        print("\nExtracting coordinates for elevation analysis...")
        all_coordinates, coordinate_to_node_ids = extract_coordinates_from_segments(
            all_merged_segments
        )

        # Initialize elevation fetcher
        print("\nInitializing elevation data source...")
        print("Initializing API-based elevation fetcher...")
        # Get region name for dataset priority selection
        region_name = get_region_name_from_config()
        elevation_fetcher = FastElevationFetcher(region_name=region_name)

        # Get elevation data with checkpoint support
        print("\nFetching elevation data with checkpoint support...")
        elevation_mapping, elevation_batch_info = persistence.load_elevation_progress()

        # Ensure signal handler has persistence manager for elevation fetching
        signal_handler.set_persistence_manager(persistence)

        if elevation_mapping:
            print("Resuming elevation fetching from existing checkpoint...")
            elevations = elevation_fetcher._resume_elevation_fetching_sync(
                all_coordinates,
                elevation_mapping,
                elevation_batch_info,
                persistence,
                "Fetching elevations",
            )
        else:
            print("Starting fresh elevation analysis...")
            elevations = elevation_fetcher._fetch_elevations_parallel(
                all_coordinates, persistence, "Fetching elevations", None, None
            )

        # Create elevation mapping
        node_elevations = {}
        for i, (coord, elevation) in enumerate(zip(all_coordinates, elevations)):
            if elevation is not None:
                for node_id in coordinate_to_node_ids[coord]:
                    node_elevations[node_id] = elevation

        print(f"Successfully mapped elevations for {len(node_elevations)} nodes")

        # Track per-way elevation failures for error reporting
        try:
            from utils.way_elevation_tracker import track_way_elevation_failures

            print("Tracking per-way elevation failures...")

            # Create coordinate-to-elevation mapping for fast lookup
            coord_to_elevation = {coord: elev for coord, elev in zip(all_coordinates, elevations)}

            for segment in all_merged_segments:
                if "nodes" not in segment:
                    continue

                # Extract coordinates and elevations for this segment
                segment_coords = []
                segment_elevs = []

                for node in segment["nodes"]:
                    if hasattr(node, "lat") and hasattr(node, "lon"):
                        coord = (round(float(node.lat), 6), round(float(node.lon), 6))
                        segment_coords.append(coord)

                        # Fast O(1) lookup in dictionary
                        segment_elevs.append(coord_to_elevation.get(coord, None))

                # Track failures for this way
                if segment_coords:
                    way_id = str(segment.get("id", "unknown"))
                    way_name = segment.get("name", f"Unnamed segment {way_id}")

                    track_way_elevation_failures(
                        way_id=way_id,
                        way_name=way_name,
                        coordinates=segment_coords,
                        elevations=segment_elevs,
                    )

        except ImportError:
            print("Warning: Could not import way_elevation_tracker for per-way failure tracking")
        except Exception as e:
            print(f"Warning: Error tracking per-way failures: {e}")

        # *** NEW: Save completed elevation data permanently ***
        persistence.save_completed_elevations(node_elevations)

    # Continue with analysis using node_elevations...
    print("\nAnalyzing climbs...")
    analyzer = ClimbAnalyzer(metadata.get("surface_filter", "all"), unit_system, score_type)
    analyzer.node_elevations = node_elevations

    climbs = analyzer.analyze_merged_roads(all_merged_segments, persistence)

    # ============================================================================
    # DEPRECATED: Non-streaming elevation profile generation (Legacy code path)
    # ============================================================================
    # This code path is NO LONGER USED for large region analyses.
    # All large regions now use streaming mode (see lines 12520-12560 in
    # analyze_merged_roads_streaming) which provides the same functionality
    # with significantly lower memory consumption.
    #
    # This code is maintained only for backwards compatibility with small
    # address-based searches that don't use checkpointing.
    #
    # Features implemented here that MUST also exist in streaming mode:
    # - Coordinate-based elevation lookup (coord_{lat}_{lon} keys)
    # - Coordinate rounding to 6 decimals for precision matching
    # - Node validation (isinstance dict check)
    # - Elevation profile orientation reversal (low→high)
    # - Error handling with graceful degradation
    # ============================================================================

    # Generate elevation profiles for all climbs
    if climbs:
        from climb_analyzer.data.elevation_profile import (
            generate_elevation_profile,
            downsample_profile,
        )

        print("\n")  # Spacing before progress bar
        for climb in tqdm(
            climbs,
            desc="Generating elevation profiles",
            unit="climbs",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ascii=" ▏▎▍▌▋▊▉█",
            dynamic_ncols=True,
            mininterval=0.5,
        ):
            if climb.nodes and len(climb.nodes) > 2:
                try:
                    # Extract elevations for this climb's nodes
                    elevations = []
                    for idx, node in enumerate(climb.nodes):
                        # Use coordinate-based lookup (works for merged segments with multiple way_ids)
                        # Elevations are stored as "coord_{lat}_{lon}" for merged climb compatibility
                        if isinstance(node, dict) and "lat" in node and "lon" in node:
                            # Round to 6 decimals to match storage precision
                            lat = round(node["lat"], 6)
                            lon = round(node["lon"], 6)
                            coord_key = f"coord_{lat}_{lon}"
                            elevation = node_elevations.get(coord_key, 0.0)
                            elevations.append(elevation)
                        else:
                            elevations.append(0.0)

                    # CRITICAL: Ensure profile always goes from lowest to highest elevation
                    # If first elevation is higher than last, reverse the nodes and elevations
                    if elevations and elevations[0] > elevations[-1]:
                        climb.nodes = list(reversed(climb.nodes))
                        elevations = list(reversed(elevations))

                        # Also update start coordinate to be at the lowest point
                        if climb.nodes:
                            climb.start_lat = climb.nodes[0]["lat"]
                            climb.start_lon = climb.nodes[0]["lon"]

                    # Convert elevations to feet if using imperial units
                    # Elevations are stored in meters, but profile should match metrics units
                    profile_elevations = elevations
                    if analyzer.unit_system == "imperial":
                        profile_elevations = [e * 3.28084 for e in elevations]

                    # Generate compact profile
                    climb.elevation_profile = generate_elevation_profile(
                        climb.nodes, profile_elevations
                    )

                    # Downsample long profiles to fit Excel cell limit (32k chars)
                    if len(climb.elevation_profile) > 30000:
                        climb.elevation_profile = downsample_profile(
                            climb.elevation_profile,
                            max_chars=30000,
                        )

                except Exception as e:
                    climb.elevation_profile = ""
                    print(
                        f"\n[ERROR] Failed to generate elevation profile for {climb.street_name}: {e}"
                    )
            else:
                climb.elevation_profile = ""

    # Rest of the function remains the same...

    # GET ANALYSIS CENTER FROM METADATA
    analysis_center = (metadata.get("center_lat"), metadata.get("center_lon"))

    # Add safety check before geocoding
    if not climbs:
        print("No climbs found above threshold")
        return [], pd.DataFrame(), persistence, False

    # Build scope info string for results header
    scope_info = None
    scope_type = metadata.get("scope_type")
    if scope_type == "address":
        scope_info = metadata.get("address", "Unknown Address")
    elif scope_type in ("country", "region"):
        # Fallback chain: region_name → country → address → persistence.analysis_id.
        # Missing region_name in metadata produces "region_climbs_*" output files
        # which is confusing and wastes upload bandwidth. analysis_id always embeds
        # the region name (e.g. "Kansas_all_region_1776687510" → "Kansas").
        region_name = (
            metadata.get("region_name") or metadata.get("country") or metadata.get("address")
        )
        if not region_name and persistence is not None:
            aid = getattr(persistence, "analysis_id", "") or ""
            for suffix in ("_all_region_", "_all_country_", "_all_address_"):
                if suffix in aid:
                    region_name = aid.split(suffix, 1)[0]
                    break
        if region_name:
            # Extract just the region name from path formats like "us > hawaii" or "europe/france"
            if " > " in region_name:
                scope_info = region_name.split(" > ")[-1]
            elif "/" in region_name:
                scope_info = region_name.split("/")[-1]
            else:
                scope_info = region_name
            # Convert to title case for cleaner filenames (e.g., "hawaii" -> "Hawaii")
            scope_info = scope_info.replace("-", " ").replace("_", " ").title()

    # Print results WITH analysis center for location lookup
    print("\nGenerating results...")
    if enable_geocoding:
        print("Performing location lookup and distance calculations...")
    else:
        print("Skipping location lookup (faster results)...")

    df = analyzer.print_climb_results(
        min_score, unit_system, analysis_center, persistence, enable_geocoding, scope_info
    )

    # Return with cloud cache upload flag (error_logger=None for this code path)
    return climbs, df, persistence, should_upload_to_cache, None


def process_segments(all_merged_segments, boundary_merger, persistence, auto_resume=False):
    """Helper function to do deduplication and boundary merging

    Args:
        auto_resume: If True, automatically resume from checkpoints without prompting
    """
    print("\nDeduplicating segments...")
    all_merged_segments = deduplicate_segments(
        all_merged_segments, persistence, auto_resume=auto_resume
    )

    all_merged_segments = boundary_merger.merge_boundary_segments(
        all_merged_segments, persistence
    )

    return all_merged_segments


def extract_coordinates_from_segments(
    segments: List[Dict],
) -> Tuple[List[Tuple[float, float]], Dict]:
    """Extract unique coordinates from segments."""
    coordinates = []
    coord_to_node_ids = defaultdict(list)

    for segment in segments:
        if "nodes" in segment:
            for node in segment["nodes"]:
                if hasattr(node, "lat") and hasattr(node, "lon") and hasattr(node, "id"):
                    coord = (round(float(node.lat), 6), round(float(node.lon), 6))

                    # Only add if not already seen
                    if coord not in coord_to_node_ids:
                        coordinates.append(coord)

                    coord_to_node_ids[coord].append(node.id)

    return coordinates, coord_to_node_ids


class ClimbAnalyzer:
    """Climb analyzer with detailed climb metrics and batch location lookup."""

    def __init__(
        self,
        surface_filter: str = "all",
        unit_system: str = "imperial",
        score_type: str = "basic",
    ):
        self.surface_filter = surface_filter
        self.unit_system = unit_system
        self.score_type = score_type
        self.node_elevations = {}
        self.climbs = []
        self.geolocator = Nominatim(user_agent="climb_analyzer")
        # Elevation fetch statistics
        self.total_coords_requested = 0
        self.total_coords_failed = 0
        self.elevation_success_rate = 100.0

    def batch_reverse_geocode_climbs(
        self, climbs: List, analysis_center: Tuple[float, float], persistence
    ) -> List[Dict]:
        """Batch reverse geocoding using offline reverse_geocoder module"""

        if not climbs:
            return []

        print("Looking up locations with offline reverse geocoding...")

        # Extract coordinates from climbs
        coordinates = []
        for climb in climbs:
            start_coord = self._get_climb_start_coordinates(climb)
            coordinates.append(start_coord if start_coord else (0.0, 0.0))

        # Use offline reverse geocoding (much faster than API calls)
        try:
            # Filter out invalid coordinates
            valid_coords = []
            coord_indices = []
            for i, coord in enumerate(coordinates):
                if coord and coord != (0.0, 0.0):
                    valid_coords.append(coord)
                    coord_indices.append(i)

            if valid_coords:
                print(f"Looking up {len(valid_coords)} coordinates using offline data...")

                # Import reverse_geocoder locally to defer loading large spatial index
                import reverse_geocoder as rg

                # Batch lookup with reverse_geocoder
                results = rg.search(valid_coords)

                # Build results mapping
                location_results = {}
                for i, result in enumerate(results):
                    original_index = coord_indices[i]

                    if result:
                        city = result.get("name", "Unknown")
                        state = result.get("admin1", "Unknown")
                        country = result.get("cc", "")

                        full_address_parts = [city]
                        if state and state != "Unknown":
                            full_address_parts.append(state)
                        if country:
                            full_address_parts.append(country)
                        full_address = ", ".join(full_address_parts)

                        location_results[original_index] = {
                            "city": city,
                            "state": state,
                            "country": country,
                            "full_address": full_address,
                        }
                    else:
                        location_results[original_index] = {
                            "city": "Unknown",
                            "state": "Unknown",
                            "country": "Unknown",
                            "full_address": "Not found",
                        }
            else:
                location_results = {}

        except ImportError:
            print(
                "Warning: reverse_geocoder module not found. Install with: pip install reverse_geocoder"
            )
            print("Falling back to basic distance calculation...")
            location_results = {}
        except Exception as e:
            print(f"Error with reverse_geocoder: {e}")
            print("Falling back to basic distance calculation...")
            location_results = {}

        # Calculate distances and compile final results
        results = []
        for i, climb in enumerate(climbs):
            location_info = location_results.get(
                i,
                {
                    "city": "Unknown",
                    "state": "Unknown",
                    "country": "Unknown",
                    "full_address": "Not found",
                },
            )

            # Calculate distance from analysis center
            start_coord = coordinates[i]
            if start_coord and start_coord != (0.0, 0.0):
                distance_km = self.calculate_distance(
                    analysis_center[0],
                    analysis_center[1],
                    start_coord[0],
                    start_coord[1],
                )
            else:
                distance_km = 0.0

            results.append(
                {
                    "city": location_info["city"],
                    "state": location_info["state"],
                    "country": location_info.get("country", "Unknown"),
                    "distance_km": distance_km,
                    "start_lat": start_coord[0] if start_coord else None,
                    "start_lon": start_coord[1] if start_coord else None,
                }
            )

        return results

    def _batch_geocode_climbs_chunked(
        self,
        climbs: List,
        analysis_center: Tuple[float, float],
        persistence,
        chunk_size: int = 50000,
    ) -> List[Dict]:
        """
        Batch reverse geocoding with chunked processing and memory cleanup.
        Loads reverse_geocoder for each chunk, then explicitly frees memory.
        """
        import gc
        import pickle

        if not climbs:
            return []

        # Cache file for geocoding results
        cache_file = persistence.analysis_dir / "geocoding_cache.pkl"

        # Check if we have cached results
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_results = safe_pickle_load(f)
                if len(cached_results) == len(climbs):
                    print(f"✓ Loaded {len(cached_results):,} geocoded locations from cache")
                    return cached_results
            except Exception:
                pass  # Cache invalid, continue with fresh geocoding

        all_results = []
        num_chunks = (len(climbs) + chunk_size - 1) // chunk_size

        # Use tqdm progress bar for chunks
        for chunk_idx in tqdm(range(num_chunks), desc="Geocoding climbs", unit="chunk"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(climbs))
            chunk_climbs = climbs[start_idx:end_idx]

            # Extract coordinates for this chunk
            coordinates = []
            for climb in chunk_climbs:
                start_coord = self._get_climb_start_coordinates(climb)
                coordinates.append(start_coord if start_coord else (0.0, 0.0))

            # Import and use reverse_geocoder for this chunk only
            try:
                import reverse_geocoder as rg

                # Filter valid coordinates
                valid_coords = []
                coord_indices = []
                for i, coord in enumerate(coordinates):
                    if coord and coord != (0.0, 0.0):
                        valid_coords.append(coord)
                        coord_indices.append(i)

                if valid_coords:
                    # Geocode this chunk
                    results = rg.search(valid_coords)

                    # Build results mapping
                    location_results = {}
                    for i, result in enumerate(results):
                        original_index = coord_indices[i]
                        if result:
                            location_results[original_index] = {
                                "city": result.get("name", "Unknown"),
                                "state": result.get("admin1", "Unknown"),
                                "country": result.get("cc", ""),
                            }
                else:
                    location_results = {}

                # Calculate distances and compile chunk results
                chunk_results = []
                for i, _ in enumerate(chunk_climbs):
                    location_info = location_results.get(
                        i, {"city": "Unknown", "state": "Unknown", "country": ""}
                    )

                    coord = coordinates[i]
                    if coord and coord != (0.0, 0.0):
                        distance_km = self.calculate_distance(
                            analysis_center[0], analysis_center[1], coord[0], coord[1]
                        )
                    else:
                        distance_km = 0.0

                    chunk_results.append(
                        {
                            "city": location_info["city"],
                            "state": location_info["state"],
                            "country": location_info.get("country", "Unknown"),
                            "distance_km": distance_km,
                        }
                    )

                all_results.extend(chunk_results)

                # CRITICAL: Explicitly free memory after each chunk
                del rg
                del results
                del location_results
                del valid_coords
                gc.collect()

            except ImportError:
                print("reverse_geocoder not available, using basic distance calculation")
                chunk_results = self._calculate_distances_only(chunk_climbs, analysis_center)
                all_results.extend(chunk_results)
            except Exception as e:
                print(f"Error geocoding chunk: {e}")
                chunk_results = self._calculate_distances_only(chunk_climbs, analysis_center)
                all_results.extend(chunk_results)

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(all_results, f)
            print(f"✓ Cached {len(all_results):,} geocoding results")
        except Exception:
            pass  # Cache save failed, not critical

        return all_results

    def analyze_merged_roads(
        self, merged_roads: List[Dict], persistence: ChunkPersistenceManager = None
    ) -> List:
        """Analyze roads with optional checkpointing for large datasets."""

        self.climbs = []

        print(f"Analyzing {len(merged_roads)} road segments...")

        with tqdm(
            total=len(merged_roads),
            desc="Analyzing climbs",
            unit="roads",
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for road_segment in merged_roads:
                try:
                    climb_results = self.calculate_climb_metrics(road_segment)
                    if climb_results:
                        self.climbs.extend(
                            climb_results
                        )  # extend since calculate_climb_metrics returns a list

                    if len(self.climbs) % 100 == 0:
                        pbar.set_postfix({"climbs_found": len(self.climbs)})

                    pbar.update(1)

                except Exception:
                    pbar.update(1)
                    continue

        print(f"Found {len(self.climbs)} climbs")

        # Detect connected climbs
        print("Detecting connected climbs...")
        self._detect_connected_climbs()

        return self.climbs

    def analyze_merged_roads_streaming(
        self,
        checkpoint_path: Path,
        persistence: ChunkPersistenceManager = None,
        batch_size: int = 50000,
    ) -> List:
        """
        Analyze roads by streaming from checkpoint in batches (memory-efficient).

        This avoids loading all segments into memory at once, which can cause OOM
        for large countries like Finland, Sweden, etc.

        Supports checkpointing: saves progress every 500K segments and at completion.
        Can resume from checkpoint if interrupted.

        Args:
            checkpoint_path: Path to merged segments checkpoint file
            persistence: Optional persistence manager for checkpointing
            batch_size: Number of segments to process per batch

        Returns:
            List of climbs found
        """
        # Note: read_segments_from_checkpoint_batched is defined in this module
        # No import needed

        # Use temporary file to store climbs on disk instead of in memory
        import pickle

        from tqdm import tqdm

        # Setup checkpoint directory
        if persistence:
            checkpoint_dir = persistence.analysis_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Fallback to output directory if no persistence (shouldn't happen for large analyses)
            checkpoint_dir = Path("output")
            checkpoint_dir.mkdir(exist_ok=True)

        # Check for existing climb analysis checkpoint
        climb_checkpoint_file = checkpoint_dir / "climb_analysis_progress.pkl"
        resume_from_index = 0
        climbs_count = 0
        temp_climbs_file = None
        skip_analysis = False  # Flag to skip analysis if already completed

        if climb_checkpoint_file.exists():
            try:
                with open(climb_checkpoint_file, "rb") as f:
                    checkpoint_data = safe_pickle_load(f)
                    resume_from_index = checkpoint_data.get("segments_processed", 0)
                    climbs_count = checkpoint_data.get("climbs_found", 0)
                    temp_climbs_file_str = checkpoint_data.get("temp_climbs_file")
                    # Convert to absolute path - checkpoint stores relative path
                    if temp_climbs_file_str:
                        temp_climbs_file = Path(temp_climbs_file_str)
                        # Make absolute if relative (resolve relative to cwd)
                        if not temp_climbs_file.is_absolute():
                            temp_climbs_file = Path.cwd() / temp_climbs_file
                        # DEBUG: Show what path we resolved to
                        # print(f"[DEBUG] Checkpoint temp file path: {temp_climbs_file}")
                        # print(f"[DEBUG] File exists: {temp_climbs_file.exists()}")
                    else:
                        temp_climbs_file = None
                    is_completed = checkpoint_data.get("completed", False)

                    # Check if analysis was already completed
                    if is_completed and temp_climbs_file and temp_climbs_file.exists():
                        print("  Found existing climb analysis checkpoint")
                        print(f"  ✓ Analysis already completed: {climbs_count:,} climbs found")
                        print()

                        # Set instance variables to use existing climbs
                        self.climbs_temp_file = temp_climbs_file
                        self.climbs_count = climbs_count

                        # Set flag to skip analysis loop but continue to spatial index
                        skip_analysis = True

                    elif is_completed and temp_climbs_file and not temp_climbs_file.exists():
                        # Analysis completed AND exported - temp file was cleaned up after successful export
                        print("✅ Analysis already fully completed and exported")
                        print(f"   Previous run analyzed {climbs_count:,} climbs")
                        print("   Checkpoint temp files cleaned up after export")
                        print()

                        # Set analyzer's climb count so main function knows how many climbs existed
                        # This prevents "No climbs found above threshold" message
                        self.climbs_count = climbs_count
                        self.climbs_temp_file = temp_climbs_file  # Set even though file doesn't exist (for streaming mode detection)

                        # Look for existing Excel files in output directory
                        # Pattern: <region>_climbs_<filter>_<date>_v<version>_e<errors>-<filenum>.xlsx
                        output_dir = Path("output")
                        existing_files = []
                        if output_dir.exists():
                            region = None
                            surface = "all"
                            score_type = "basic"

                            # Try to find files matching the checkpoint metadata
                            # Metadata is stored as pickle (.pkl), not JSON
                            checkpoint_metadata_file = checkpoint_dir / "metadata.pkl"
                            if checkpoint_metadata_file.exists():
                                try:
                                    with open(checkpoint_metadata_file, "rb") as f:
                                        checkpoint_meta = safe_pickle_load(f)

                                    region = (
                                        checkpoint_meta.get("region_name")
                                        or checkpoint_meta.get("country")
                                        or checkpoint_meta.get("address", "")
                                    )
                                    surface = checkpoint_meta.get("surface_filter", "all")
                                    score_type = checkpoint_meta.get("score_type", "basic")
                                except Exception:
                                    pass

                            # Fallback: Extract region info from analysis_id
                            # Format: Region_surface_scope_scoretype_timestamp
                            if not region and hasattr(self, 'analysis_id'):
                                try:
                                    parts = self.analysis_id.split("_")
                                    if len(parts) >= 4:
                                        region = parts[0]  # First part is region name
                                        surface = parts[1] if len(parts) > 1 else "all"
                                        score_type = parts[3] if len(parts) > 3 else "basic"
                                except Exception:
                                    pass

                            if region:
                                # Build pattern: region_climbs_surface_date_*.xlsx
                                safe_region = region.replace(" ", "_").replace("-", "_")
                                pattern = f"{safe_region}_climbs_{surface}_*.xlsx"

                                existing_files = sorted(output_dir.glob(pattern))

                                if existing_files:
                                    print(
                                        f"  Found {len(existing_files)} existing output file(s):"
                                    )
                                    for f in existing_files:
                                        file_size_mb = f.stat().st_size / (1024 * 1024)
                                        print(f"   ✓ {f.name} ({file_size_mb:.1f} MB)")
                                    print()

                        if not existing_files:
                            print("💡 To re-run analysis:")
                            print(f"   Delete checkpoint: rm -rf {checkpoint_dir}")
                            print("   Then restart analysis")
                            print()

                        # Return empty list but analyzer.climbs_count is set
                        # Main function will skip "no climbs found" message because count > 0
                        return []

                    elif resume_from_index > 0 and temp_climbs_file and temp_climbs_file.exists():
                        # Resume from partial checkpoint
                        print("  Found existing climb analysis checkpoint")
                        print(
                            f"   ✓ Resuming from segment {resume_from_index:,} ({climbs_count:,} climbs found so far)"
                        )
                        print()
            except Exception as e:
                print(f"⚠️  Could not load climb analysis checkpoint: {e}")
                print("   Starting from beginning...")
                resume_from_index = 0
                climbs_count = 0
                temp_climbs_file = None

        # Create temp file if not resuming
        if not temp_climbs_file:
            temp_climbs_file = checkpoint_dir / f"climbs_temp_{int(time.time())}.pkl"

        # Only run analysis if not already completed
        if not skip_analysis:
            # First, count total segments for progress bar
            print("Counting segments...")
            total_segments = 0
            for batch in read_segments_from_checkpoint_batched(
                checkpoint_path, batch_size=batch_size
            ):
                total_segments += len(batch)
            print(f"✓ Found {total_segments:,} segments to analyze")

            # Now process in batches, writing climbs to disk
            print("Analyzing climbs in batches (streaming to disk)...")
            print("    Checkpointing enabled: Progress saved every 500K segments")

            # Open temp file in append mode if resuming, write mode if starting fresh
            file_mode = "ab" if resume_from_index > 0 else "wb"

            with open(temp_climbs_file, file_mode) as f:
                with tqdm(
                    total=total_segments,
                    initial=resume_from_index,  # Start progress bar at resumed position
                    desc="Analyzing climbs",
                    unit="roads",
                    dynamic_ncols=True,
                    ascii=" ▏▎▍▌▋▊▉█",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                ) as pbar:
                    # Force immediate render when resuming from checkpoint
                    if resume_from_index > 0:
                        pbar.refresh()

                    batch_climbs = []
                    segments_processed = 0
                    last_checkpoint_index = resume_from_index

                    for batch in read_segments_from_checkpoint_batched(
                        checkpoint_path, batch_size=batch_size
                    ):
                        for road_segment in batch:
                            # Skip already processed segments if resuming
                            if segments_processed < resume_from_index:
                                segments_processed += 1
                                pbar.update(1)
                                continue

                            try:
                                climb_results = self.calculate_climb_metrics(road_segment)
                                for climb_metrics in climb_results:
                                    # Generate elevation profile immediately (while nodes are in memory)
                                    if climb_metrics.nodes and len(climb_metrics.nodes) > 2:
                                        try:
                                            elevations = []

                                            # For merged climbs with multiple way_ids, use coordinate-based lookup
                                            # Elevations are stored as "coord_{lat}_{lon}" for this purpose
                                            for idx, node in enumerate(climb_metrics.nodes):
                                                if (
                                                    isinstance(node, dict)
                                                    and "lat" in node
                                                    and "lon" in node
                                                ):
                                                    # Look up by coordinate (works for all segments, merged or not)
                                                    lat = round(node["lat"], 6)
                                                    lon = round(node["lon"], 6)
                                                    coord_key = f"coord_{lat}_{lon}"
                                                    elevation = self.node_elevations.get(
                                                        coord_key, 0.0
                                                    )
                                                    elevations.append(elevation)
                                                elif hasattr(node, "lat") and hasattr(node, "lon"):
                                                    # Handle object nodes with .lat/.lon attributes
                                                    lat = round(node.lat, 6)
                                                    lon = round(node.lon, 6)
                                                    coord_key = f"coord_{lat}_{lon}"
                                                    elevation = self.node_elevations.get(
                                                        coord_key, 0.0
                                                    )
                                                    elevations.append(elevation)
                                                else:
                                                    elevations.append(0.0)

                                            # CRITICAL: Ensure profile always goes from lowest to highest elevation
                                            # If first elevation is higher than last, reverse the nodes and elevations
                                            if elevations and elevations[0] > elevations[-1]:
                                                climb_metrics.nodes = list(
                                                    reversed(climb_metrics.nodes)
                                                )
                                                elevations = list(reversed(elevations))

                                                # Also update start coordinate to be at the lowest point
                                                if climb_metrics.nodes:
                                                    first_node = climb_metrics.nodes[0]
                                                    if isinstance(first_node, dict):
                                                        climb_metrics.start_lat = first_node["lat"]
                                                        climb_metrics.start_lon = first_node["lon"]
                                                    elif hasattr(first_node, "lat"):
                                                        climb_metrics.start_lat = first_node.lat
                                                        climb_metrics.start_lon = first_node.lon

                                            from climb_analyzer.data.elevation_profile import (
                                                generate_elevation_profile,
                                                downsample_profile,
                                            )

                                            # Convert elevations to feet if using imperial units
                                            # Elevations are stored in meters, but profile should match metrics units
                                            profile_elevations = elevations
                                            if self.unit_system == "imperial":
                                                profile_elevations = [
                                                    e * 3.28084 for e in elevations
                                                ]

                                            climb_metrics.elevation_profile = (
                                                generate_elevation_profile(
                                                    climb_metrics.nodes, profile_elevations
                                                )
                                            )

                                            # Downsample long profiles to fit Excel cell limit (32k chars)
                                            if len(climb_metrics.elevation_profile) > 30000:
                                                climb_metrics.elevation_profile = (
                                                    downsample_profile(
                                                        climb_metrics.elevation_profile,
                                                        max_chars=30000,
                                                    )
                                                )

                                        except Exception as e:
                                            climb_metrics.elevation_profile = ""
                                            print(
                                                f"\n[ERROR] Failed to generate elevation profile for {climb_metrics.street_name}: {e}"
                                            )
                                    else:
                                        climb_metrics.elevation_profile = ""

                                    batch_climbs.append(climb_metrics)
                                    climbs_count += 1

                                if climbs_count % 100 == 0:
                                    pbar.set_postfix({"climbs_found": climbs_count})

                                # Write climbs to disk every 10K to free memory
                                if len(batch_climbs) >= 10000:
                                    pickle.dump(batch_climbs, f)
                                    batch_climbs = []

                                segments_processed += 1
                                pbar.update(1)

                                # Save checkpoint every 500K segments
                                if segments_processed - last_checkpoint_index >= 500000:
                                    checkpoint_data = {
                                        "segments_processed": segments_processed,
                                        "climbs_found": climbs_count,
                                        "temp_climbs_file": str(temp_climbs_file),
                                        "timestamp": time.time(),
                                    }
                                    with open(climb_checkpoint_file, "wb") as cp_f:
                                        pickle.dump(checkpoint_data, cp_f)
                                    last_checkpoint_index = segments_processed
                                    pbar.write(
                                        f"    Checkpoint saved: {segments_processed:,} segments ({climbs_count:,} climbs)"
                                    )

                            except Exception:
                                segments_processed += 1
                                pbar.update(1)
                                continue

                    # Write any remaining climbs
                    if batch_climbs:
                        pickle.dump(batch_climbs, f)

                # Save final checkpoint
                checkpoint_data = {
                    "segments_processed": total_segments,
                    "climbs_found": climbs_count,
                    "temp_climbs_file": str(temp_climbs_file),
                    "timestamp": time.time(),
                    "completed": True,
                }
                with open(climb_checkpoint_file, "wb") as cp_f:
                    pickle.dump(checkpoint_data, cp_f)
                print("    Final checkpoint saved")

                print(f"Found {climbs_count:,} climbs (saved to disk)")

        # Always use spatial-indexed connected climb detection (memory-efficient streaming)
        # This works for all dataset sizes and enables connected climbs + elevation profiles
        if True:  # Always use streaming approach
            print()
            print(f"Detecting connected climbs for {climbs_count:,} climbs...")
            #print("   Using spatial-indexed detection (memory-efficient streaming)")
            print()

            # Build spatial grid index of endpoints (first pass - only endpoints, not all nodes)
            grid_index, all_endpoints = self._build_endpoint_index_streaming(
                temp_climbs_file, climbs_count
            )

            # Detect connections using spatial index (fast - only compares nearby climbs)
            self._detect_connected_climbs_spatial(
                temp_climbs_file, grid_index, all_endpoints, climbs_count, persistence
            )

            # Free memory before loading climbs
            del grid_index, all_endpoints
            import gc

            gc.collect()

            # MEMORY FIX: For very large datasets (7M+ climbs), don't load all into memory!
            # Instead, we'll return the temp file path and stream during export
            from climb_analyzer.utils.formatting import print_dim

            print()
            print(f"✓ Climbs ready for export ({climbs_count:,} climbs)")
            print_dim("   Using streaming mode for export (memory optimized)")
            print()

            # Store the temp file path and count for streaming export
            self.climbs_temp_file = temp_climbs_file
            self.climbs_count = climbs_count
            self.climbs = []  # Empty list - will stream from temp file during export

        # OLD CODE PATH - Commented out (now always using streaming approach above)
        # else:
        #     # Small dataset - load climbs for connected climb detection
        #     print("Loading climbs for connected climb analysis...")
        #     self.climbs = []
        #     with open(temp_climbs_file, "rb") as f:
        #         try:
        #             while True:
        #                 batch_climbs = pickle.load(f)
        #                 self.climbs.extend(batch_climbs)
        #         except EOFError:
        #             pass
        #
        #     print(f"Loaded {len(self.climbs):,} climbs into memory")
        #
        #     # Detect connected climbs (requires nodes)
        #     print("Detecting connected climbs...")
        #     self._detect_connected_climbs()
        #
        #     # Build connection map for updating temp file
        #     connection_names = {}
        #     for i, climb in enumerate(self.climbs):
        #         if hasattr(climb, "connected_climbs") and climb.connected_climbs:
        #             connection_names[i] = climb.connected_climbs
        #
        #     print(f"Found connections for {len(connection_names):,} climbs")
        #
        #     # Free memory: Remove nodes from all climbs (no longer needed after connected climb detection)
        #     print("Freeing memory (removing node data)...")
        #     nodes_removed = 0
        #     for climb in self.climbs:
        #         if climb.nodes:
        #             nodes_removed += len(climb.nodes)
        #             climb.nodes = None  # Clear nodes to free memory
        #     print(f"  Freed memory from {nodes_removed:,} nodes")
        #
        #     # CONSOLIDATION: Update temp file with connection info, then use streaming export
        #     # This enables consistent streaming export for all dataset sizes
        #     if connection_names:
        #         self._update_climbs_with_connections(
        #             temp_climbs_file, connection_names, climbs_count
        #         )
        #
        #     self.climbs_temp_file = temp_climbs_file
        #     self.climbs_count = climbs_count
        #     self.climbs = []  # Clear in-memory climbs - will stream from temp file during export

        # Clean up temp file ONLY if not using streaming export
        # For large datasets, temp file is kept and cleaned up after export
        if not hasattr(self, "climbs_temp_file"):
            if temp_climbs_file.exists():
                temp_climbs_file.unlink()

        return self.climbs

    def _stream_to_excel(
        self,
        min_score,
        units,
        enable_geocoding,
        analysis_center,
        persistence,
        scope_info,
        surface_filter,
        score_type,
        cycling_only,
    ):
        """
        Stream climbs directly from temp file to Excel using external sorting.

        For 7M+ climbs, uses external merge-sort to avoid loading all climbs into memory:
        1. Pass 1: Load climbs in batches, filter, sort each batch, write to temp files
        2. Pass 1.5: Merge-sort all batches to single sorted file (streaming)
        3. Pass 1.75: Reverse geocode climbs in batches (streaming)
        4. Pass 2: Stream from sorted file to Excel with automatic file splitting

        Memory usage: ~500MB-1GB (vs 11GB+ for loading all at once)

        Returns:
            List of created Excel file paths (for compatibility with save infrastructure)
        """
        import gc
        import heapq
        import pickle
        import tempfile
        from pathlib import Path

        import pandas as pd
        from tqdm import tqdm

        # DISK SPACE FIX: Use checkpoint_data directory for temp files
        # Enables recovery if export phase crashes, and uses volume-mounted storage
        if persistence and hasattr(persistence, "analysis_dir"):
            temp_dir = persistence.analysis_dir / "tmp"
        else:
            # Fallback if no persistence context
            temp_dir = Path("data/checkpoint_data/tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Check if temp file exists (might be deleted after previous export)
        if not self.climbs_temp_file.exists():
            print("   Analysis was already exported in a previous run")
            print("   Looking for existing output files...")

            # Look for existing Excel files in output directory
            output_dir = Path("output")
            existing_files = []

            # Try to find checkpoint metadata to build filename pattern
            if persistence:
                checkpoint_dir = persistence.analysis_dir
                region = None
                surface = "all"
                score_type = "basic"

                # Metadata is stored as pickle (.pkl), not JSON
                checkpoint_metadata_file = checkpoint_dir / "metadata.pkl"
                if checkpoint_metadata_file.exists():
                    try:
                        with open(checkpoint_metadata_file, "rb") as f:
                            checkpoint_meta = safe_pickle_load(f)

                        region = (
                            checkpoint_meta.get("region_name")
                            or checkpoint_meta.get("country")
                            or checkpoint_meta.get("address", "")
                        )
                        surface = checkpoint_meta.get("surface_filter", "all")
                        score_type = checkpoint_meta.get("score_type", "basic")
                    except Exception:
                        pass

                # Fallback: Extract region info from analysis_id
                # Format: Region_surface_scope_scoretype_timestamp
                if not region:
                    try:
                        analysis_id = persistence.analysis_id
                        parts = analysis_id.split("_")
                        if len(parts) >= 4:
                            region = parts[0]  # First part is region name
                            surface = parts[1] if len(parts) > 1 else "all"
                            score_type = parts[3] if len(parts) > 3 else "basic"
                    except Exception:
                        pass

                if region:
                    # Build pattern: region_climbs_surface_date_*.xlsx
                    safe_region = region.replace(" ", "_").replace("-", "_")
                    pattern = f"{safe_region}_climbs_{surface}_*.xlsx"

                    existing_files = sorted(output_dir.glob(pattern))

            if existing_files:
                print(f"   ✓ Found {len(existing_files)} existing file(s):")
                for f in existing_files:
                    file_size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"      {f.name} ({file_size_mb:.1f} MB)")
                return existing_files
            else:
                print("   ⚠️ Could not locate existing output files")
                print("   Files may have been moved or deleted")
                return []

        # Determine unit labels
        if units == "imperial":
            elev_unit = "ft"
            dist_unit = "mi"
        else:
            elev_unit = "m"
            dist_unit = "km"

        # Determine score extraction function
        if self.score_type == "fiets":
            get_score = lambda c: c.fiets_score
        elif self.score_type == "pdi":
            get_score = lambda c: c.pdi_score
        else:  # 'basic'
            get_score = lambda c: c.climb_score

        # RESUME NOTE: Export temp files preserved in checkpoint_data/tmp/ for recovery
        # If you need to restart export after a crash, these files will be reused in a future update
        # For now, if export fails, you can restart and it will regenerate from climb analysis

        # Pass 1: Load climbs in batches, filter, sort each batch, write to temp files
        from climb_analyzer.utils.formatting import print_dim

        print_dim("   Pass 1: Loading, filtering, and sorting climbs in batches...")
        print_dim("   Using external sorting to keep memory under 1GB (vs 11GB+ for loading all)")

        temp_sorted_files = []
        filtered_climbs = []
        batch_sort_size = 500000  # Sort and flush every 500K climbs
        total_filtered = 0

        with open(self.climbs_temp_file, "rb") as f:
            pbar = tqdm(
                total=self.climbs_count, desc="  Loading climbs", unit="climbs", **TQDM_DEFAULTS
            )
            try:
                while True:
                    # Load batches of climbs (as written by analysis phase)
                    batch_climbs = safe_pickle_load(f)

                    for climb in batch_climbs:
                        # Filter by score threshold
                        if get_score(climb) >= min_score:
                            filtered_climbs.append(climb)
                            total_filtered += 1

                            # Flush to disk when batch is full
                            if len(filtered_climbs) >= batch_sort_size:
                                # Sort this batch by score (descending)
                                filtered_climbs.sort(key=get_score, reverse=True)

                                # Write sorted batch to temp file (one climb per pickle dump for streaming)
                                temp_file = tempfile.NamedTemporaryFile(
                                    mode="wb", delete=False, suffix=".pkl", dir=str(temp_dir)
                                )
                                for climb_item in filtered_climbs:
                                    pickle.dump(climb_item, temp_file)
                                temp_file.close()
                                temp_sorted_files.append(temp_file.name)

                                from climb_analyzer.utils.formatting import Colors

                                pbar.write(
                                    f"{Colors.DIM}     Sorted batch {len(temp_sorted_files)}: {len(filtered_climbs):,} climbs → {Path(temp_file.name).name}{Colors.RESET}"
                                )

                                # Clear memory
                                del filtered_climbs
                                filtered_climbs = []
                                gc.collect()

                    pbar.update(len(batch_climbs))

            except EOFError:
                pass
            finally:
                pbar.close()

        # CHECKPOINT PRESERVATION: Keep climbs_temp_file for recovery
        # DO NOT delete - allows resume if export phase crashes

        # Handle remaining climbs in last batch
        if filtered_climbs:
            filtered_climbs.sort(key=get_score, reverse=True)
            temp_file = tempfile.NamedTemporaryFile(
                mode="wb", delete=False, suffix=".pkl", dir=str(temp_dir)
            )
            for climb in filtered_climbs:
                pickle.dump(climb, temp_file)
            temp_file.close()
            temp_sorted_files.append(temp_file.name)
            print_dim(
                f"     Sorted batch {len(temp_sorted_files)}: {len(filtered_climbs):,} climbs → {Path(temp_file.name).name}"
            )
            del filtered_climbs
            gc.collect()

        if not temp_sorted_files:
            print("No climbs found after filtering")
            return pd.DataFrame()

        print(
            f"   ✓ Created {len(temp_sorted_files)} sorted batches with {total_filtered:,} total climbs"
        )
        print_dim("   Pass 1.5: Merge-sorting batches into single sorted stream...")

        # Merge-sort all batches using heapq
        # Load iterators for each sorted batch file
        def batch_iterator(filepath):
            """Generator that yields climbs from a sorted batch file (one at a time for memory efficiency)"""
            with open(filepath, "rb") as f:
                while True:
                    try:
                        yield safe_pickle_load(f)
                    except EOFError:
                        break

        # Create heap with (negative_score, batch_idx, climb, iterator) tuples
        # Negative score because heapq is min-heap, we want max-heap (highest scores first)
        batch_iterators = [batch_iterator(f) for f in temp_sorted_files]
        heap = []

        for idx, it in enumerate(batch_iterators):
            try:
                climb = next(it)
                heapq.heappush(heap, (-get_score(climb), idx, climb, it))
            except StopIteration:
                pass

        # Merge-sort to a single sorted temp file (stream, don't accumulate in memory!)
        final_sorted_file = tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix="_final_sorted.pkl", dir=str(temp_dir)
        )
        merge_pbar = tqdm(
            total=total_filtered, desc="  Merge-sorting", unit="climbs", **TQDM_DEFAULTS
        )

        batch_for_disk = []
        write_batch_size = 50000  # Write to disk every 50K climbs

        while heap:
            neg_score, batch_idx, climb, iterator = heapq.heappop(heap)
            batch_for_disk.append(climb)
            merge_pbar.update(1)

            # Write to disk periodically to keep memory low
            if len(batch_for_disk) >= write_batch_size:
                pickle.dump(batch_for_disk, final_sorted_file)
                del batch_for_disk
                batch_for_disk = []
                gc.collect()

            # Get next climb from same batch
            try:
                next_climb = next(iterator)
                heapq.heappush(heap, (-get_score(next_climb), batch_idx, next_climb, iterator))
            except StopIteration:
                pass

        # Write final batch
        if batch_for_disk:
            pickle.dump(batch_for_disk, final_sorted_file)
            del batch_for_disk
            gc.collect()

        final_sorted_file.close()

        # Ensure progress bar shows 100%
        merge_pbar.n = merge_pbar.total
        merge_pbar.refresh()
        merge_pbar.close()

        # CHECKPOINT PRESERVATION: Keep sorted batch files for recovery
        # DO NOT delete - allows resume if export phase crashes
        print(f"   ✓ Merge-sorted {total_filtered:,} climbs to disk")
        print(
            f"   ✓ Preserved {len(temp_sorted_files)} sorted batch files in checkpoint_data for recovery"
        )

        # Pass 1.75: Reverse geocode climbs in streaming batches
        # For large datasets, geocode in batches and save to temp file
        print_dim(f"   Pass 1.75: Reverse geocoding {total_filtered:,} climbs in batches...")

        geocode_batch_size = 10000  # Geocode 10K climbs at a time
        geocode_cache = {}  # Cache results keyed by (lat, lon) tuple
        geocode_temp_file = tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix="_geocode.pkl", dir=str(temp_dir)
        )

        with open(final_sorted_file.name, "rb") as f:
            geocode_pbar = tqdm(
                total=total_filtered, desc="  Geocoding", unit="climbs", **TQDM_DEFAULTS
            )
            try:
                while True:
                    # Load batch of climbs
                    batch_climbs = safe_pickle_load(f)

                    # Extract unique coordinates from batch
                    unique_coords = set()
                    for climb in batch_climbs:
                        coord = (round(climb.start_lat, 5), round(climb.start_lon, 5))
                        if coord not in geocode_cache:
                            unique_coords.add(coord)

                    # Geocode unique coordinates in this batch
                    if unique_coords and enable_geocoding:
                        import reverse_geocoder as rg

                        coords_list = [(lat, lon) for lat, lon in unique_coords]
                        try:
                            results = rg.search(coords_list)
                            for coord, result in zip(coords_list, results):
                                geocode_cache[coord] = {
                                    "city": result.get("name", ""),
                                    "state": result.get("admin1", ""),
                                    "country": result.get("cc", ""),
                                }
                        except Exception:
                            # If geocoding fails, use empty values
                            for coord in coords_list:
                                geocode_cache[coord] = {"city": "", "state": "", "country": ""}

                    geocode_pbar.update(len(batch_climbs))

                    # Flush cache to disk every 50K entries to keep memory low
                    if len(geocode_cache) >= 50000:
                        pickle.dump(geocode_cache, geocode_temp_file)
                        geocode_cache = {}
                        gc.collect()

            except EOFError:
                pass

            # Write remaining cached geocode results
            if geocode_cache:
                pickle.dump(geocode_cache, geocode_temp_file)

            # Ensure progress bar shows 100%
            geocode_pbar.n = geocode_pbar.total
            geocode_pbar.refresh()
            geocode_pbar.close()

        geocode_temp_file.close()
        print(f"   ✓ Reverse geocoded {total_filtered:,} climbs")

        print_dim(f"   Pass 2: Streaming {total_filtered:,} climbs to Excel...")

        # Load geocode cache from temp file
        geocode_lookup = {}
        with open(geocode_temp_file.name, "rb") as gcf:
            try:
                while True:
                    batch_geocode = safe_pickle_load(gcf)
                    geocode_lookup.update(batch_geocode)
            except EOFError:
                pass
        print(f"   ✓ Loaded geocode data for {len(geocode_lookup):,} unique locations")

        # Stream from sorted file to Excel (don't load all into memory)
        from datetime import datetime
        from pathlib import Path

        # Import version info
        try:
            from __version__ import __version__
        except ImportError:
            __version__ = "unknown"

        # Get elevation error count from stats collector
        elevation_errors = 0
        try:
            from utils.elevation_stats_collector import get_stats_collector, has_elevation_stats

            if has_elevation_stats():
                stats_collector = get_stats_collector()
                stats = stats_collector.get_stats()
                elevation_errors = stats.get("total_coords_failed", 0)
        except Exception:
            elevation_errors = 0

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Build base filename using pattern:
        # Format: {region}_climbs_{surface}_{access}_{units}_{date}
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Surface: all-surfaces, paved, gravel, dirt (convert internal "all" to "all-surfaces")
        surface_str = "all-surfaces" if surface_filter == "all" else surface_filter
        # Access: cycling or all-access
        access_str = "cycling" if cycling_only else "all-access"
        # Units: imperial or metric
        units_str = units if units in ("imperial", "metric") else "imperial"
        filter_str = f"{surface_str}_{access_str}_{units_str}"

        # Recover scope_info from persistence.analysis_id if caller didn't pass it.
        # analysis_id format: "<RegionName>_<surface>_<scope>_<timestamp>"
        # e.g. "Ohio_all_region_1776700362" → "Ohio". For canonical paths that
        # had slashes stripped during sanitization (e.g. "canadaalberta_all_country_..."),
        # peel off known continent/country prefixes to recover the trailing region.
        if not scope_info and persistence is not None:
            aid = getattr(persistence, "analysis_id", "") or ""
            for suffix in ("_all_region_", "_all_country_", "_all_address_"):
                if suffix in aid:
                    recovered = aid.split(suffix, 1)[0]
                    if recovered:
                        recovered_lower = recovered.lower()
                        # Strip known parent prefixes left over from "canada/alberta"
                        # -> "canadaalberta" sanitization.
                        for prefix in (
                            "northamerica", "southamerica", "europe", "africa",
                            "asia", "oceania", "australia", "antarctica",
                            "canada", "us", "mexico",
                        ):
                            if recovered_lower.startswith(prefix) and len(recovered_lower) > len(prefix):
                                recovered = recovered[len(prefix):]
                                break
                        scope_info = recovered.replace("_", " ").replace("-", " ").title()
                        print(
                            f"  ℹ️  Recovered scope_info='{scope_info}' from persistence.analysis_id"
                        )
                        break

        # Extract safe name from scope_info (e.g., "France", "Colorado", etc.)
        if scope_info:
            safe_name = "".join(
                c for c in scope_info if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            safe_name = safe_name.replace(" ", "_")
            # Ensure we don't end up with an empty string after filtering
            if not safe_name:
                safe_name = "region"
        else:
            safe_name = "region"
            print(
                f"⚠️  WARN: scope_info was None/empty AND persistence.analysis_id lookup failed — "
                f"output files will use 'region' prefix."
            )

        base_filename = f"{safe_name}_climbs_{filter_str}_{date_str}"

        # Build filename suffix with version and error count
        # Format: _v{version}_e{errors} (e.g., "_v2.0.1_e0000")
        suffix = ""
        if __version__:
            suffix += f"_v{__version__}"
        if elevation_errors > 0:
            suffix += f"_e{elevation_errors:04d}"
        elif elevation_errors == 0 and __version__:
            suffix += "_e0000"

        # Excel limits: 1,048,576 rows max.
        # ROWS_PER_FILE is set well below that, primarily to bound the amount
        # of in-flight data that could be lost if the process is killed mid-write.
        # Smaller files also mean each file finalizes to disk sooner, which
        # matters because write-only openpyxl still keeps the current sheet's
        # xml worksheet stream open until wb.save().
        MAX_EXCEL_ROWS = 1048576
        ROWS_PER_FILE = 500000  # Lowered from 950K for better crash resilience

        # Import openpyxl directly so we can use write_only mode (memory-flat
        # streaming writer). pandas.ExcelWriter does not expose write_only.
        from openpyxl import Workbook

        row_num = 0
        created_files = []
        current_file_num = 1
        current_wb = None
        current_ws = None
        current_excel_file = None
        current_file_rows = 0
        header_keys = None  # Column headers, captured from the first row built

        # Track column widths during streaming write (memory-efficient)
        column_widths = {}  # {column_letter: max_width}

        def _track_column_widths(row_data, is_header=False):
            """Track maximum width for each column as rows are written"""
            for idx, (col_name, value) in enumerate(row_data.items()):
                # Convert column index to letter (A, B, C, ...)
                col_letter = (
                    chr(65 + idx) if idx < 26 else chr(65 + idx // 26 - 1) + chr(65 + idx % 26)
                )

                # Calculate value width
                value_str = str(value) if value is not None else ""
                value_width = len(value_str)

                # Headers get extra weight
                if is_header:
                    value_width = max(value_width, len(col_name))

                # Track maximum
                if col_letter not in column_widths:
                    column_widths[col_letter] = value_width
                else:
                    column_widths[col_letter] = max(column_widths[col_letter], value_width)

        def _apply_excel_formatting(worksheet):
            """Apply tracked column widths to a write-only worksheet.

            Note: write-only worksheets do not support auto_filter on
            worksheet.dimensions (dimensions are not computed until save),
            and column widths must be set BEFORE rows are appended. This
            function is called when the sheet is fresh, right after header.
            """
            try:
                # Apply tracked column widths
                for col_letter, max_width in column_widths.items():
                    adjusted_width = min(max_width + 2, 50)
                    worksheet.column_dimensions[col_letter].width = adjusted_width
            except Exception as e:
                print(f"   ⚠️  Warning: Could not apply Excel formatting: {e}")

        def _finalize_current_file():
            """Save and close the current workbook atomically via temp file.

            Writing to a .tmp file and renaming on success ensures the
            visible .xlsx file is either a complete valid file or absent -
            never a truncated 2KB zip header. If the process dies during
            save, the .tmp file is left behind and the original (if any)
            is untouched.
            """
            nonlocal current_wb, current_ws, current_excel_file, current_file_rows

            if current_wb is None:
                return

            tmp_path = current_excel_file.with_suffix(".xlsx.tmp")
            try:
                current_wb.save(str(tmp_path))
                # Atomic rename - the final file only appears when save succeeded
                tmp_path.replace(current_excel_file)
                created_files.append(current_excel_file)
                file_size_mb = current_excel_file.stat().st_size / (1024**2)
                print(
                    f"   ✓ Saved {current_excel_file.name} ({current_file_rows:,} rows, {file_size_mb:.1f} MB)"
                )
            finally:
                # Drop references so the workbook (and any internal buffers)
                # can be garbage collected before we start the next one
                current_wb = None
                current_ws = None
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                gc.collect()

        def _create_new_file():
            """Helper to create a new Excel file when splitting."""
            nonlocal current_file_num, current_wb, current_ws, current_excel_file, current_file_rows, column_widths

            # Finalize the previous file (atomic rename via .tmp)
            _finalize_current_file()

            # Pick filename: {base}{suffix}.xlsx for single-file outputs,
            # {base}{suffix}-N.xlsx for split outputs
            if current_file_num == 1 and total_filtered <= ROWS_PER_FILE:
                current_excel_file = output_dir / f"{base_filename}{suffix}.xlsx"
            elif current_file_num == 1:
                current_excel_file = output_dir / f"{base_filename}{suffix}-1.xlsx"
            else:
                current_excel_file = output_dir / f"{base_filename}{suffix}-{current_file_num}.xlsx"

            # Protect against overwriting existing files: rename to .backup
            if current_excel_file.exists():
                backup_file = current_excel_file.with_suffix(".xlsx.backup")
                if backup_file.exists():
                    backup_file.unlink()
                current_excel_file.rename(backup_file)
                print(f"   ℹ️  Existing file renamed to {backup_file.name}")

            # Use write_only mode: rows are serialized to XML and streamed
            # to a temp file on disk as they are appended, so memory stays
            # flat regardless of total row count.
            current_wb = Workbook(write_only=True)
            current_ws = current_wb.create_sheet("Climbs")
            current_file_rows = 0
            current_file_num += 1

            # Reset column width tracking for the new file
            column_widths.clear()

        # Create first file
        _create_new_file()

        def _write_header_if_needed(row_dict):
            """Write header row when the sheet is empty and apply formatting.

            In write_only mode, column widths must be set before any rows are
            appended, so we apply formatting here based on the header row's
            own width tracking. Column widths are then refined per-row as data
            flows through.
            """
            nonlocal header_keys
            if current_file_rows > 0:
                return
            header_keys = list(row_dict.keys())
            # Track header widths
            _track_column_widths(row_dict, is_header=True)
            # Apply initial column widths based on header (rows will refine these,
            # but only the values set *before* first append take effect)
            _apply_excel_formatting(current_ws)
            current_ws.append(header_keys)

        with open(final_sorted_file.name, "rb") as f:
            with tqdm(
                total=total_filtered, desc="  Writing Excel", unit="climbs", **TQDM_DEFAULTS
            ) as pbar:
                try:
                    while True:
                        batch_climbs = safe_pickle_load(f)

                        for climb in batch_climbs:
                            # Check if we need to start a new file
                            if current_file_rows >= ROWS_PER_FILE:
                                # Finalize current file (atomic save) before
                                # starting the next split. This ensures each
                                # completed split is fully on disk.
                                _create_new_file()

                            # Convert units
                            if units == "imperial":
                                elev_gain = climb.elevation_gain * 3.28084
                                height = climb.height * 3.28084
                                prominence = climb.prominence * 3.28084
                                length = climb.length_km * 0.621371
                            else:
                                elev_gain = climb.elevation_gain
                                height = climb.height
                                prominence = climb.prominence
                                length = climb.length_km

                            way_ids_str = str(climb.way_ids[0]) if climb.way_ids else "N/A"
                            osm_links_str = climb.osm_links[0] if climb.osm_links else "N/A"
                            # All way IDs for merged climbs (comma-separated)
                            all_way_ids_str = (
                                ", ".join(str(wid) for wid in climb.way_ids)
                                if climb.way_ids
                                else "N/A"
                            )

                            # Get geocode data for this climb's coordinates
                            coord = (round(climb.start_lat, 5), round(climb.start_lon, 5))
                            geocode_data = geocode_lookup.get(coord, {})

                            # Calculate distance from analysis center (if provided)
                            center_distance = 0.0
                            if (
                                analysis_center
                                and analysis_center[0] is not None
                                and analysis_center[1] is not None
                            ):
                                from math import atan2, cos, radians, sin, sqrt

                                R = 6371  # Earth radius in km
                                lat1, lon1 = radians(analysis_center[0]), radians(
                                    analysis_center[1]
                                )
                                lat2, lon2 = radians(climb.start_lat), radians(climb.start_lon)
                                dlat, dlon = lat2 - lat1, lon2 - lon1
                                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                                center_distance = R * c

                            # Convert distance to imperial if needed
                            if units == "imperial":
                                center_distance_display = center_distance * 0.621371  # km to mi
                            else:
                                center_distance_display = center_distance

                            row = {
                                "Street Name": climb.street_name,
                                "City": geocode_data.get("city", ""),
                                "State": geocode_data.get("state", ""),
                                "Country": geocode_data.get("country", ""),
                                f"From Center ({dist_unit})": round(center_distance_display, 1),
                                "Latitude": round(climb.start_lat, 5),
                                "Longitude": round(climb.start_lon, 5),
                                "Cycling": climb.cycling_access,
                                "Category": climb.climb_category,
                                "Basic Score": int(climb.climb_score),
                                "FIETS Score": round(climb.fiets_score, 1),
                                "PDI Score": round(climb.pdi_score, 1),
                                f"Elev Gain ({elev_unit})": (
                                    round(elev_gain, 2) if elev_gain < 1 else int(elev_gain)
                                ),
                                f"Height ({elev_unit})": (
                                    round(height, 2) if height < 1 else int(height)
                                ),
                                f"Prominence ({elev_unit})": (
                                    round(prominence, 2) if prominence < 1 else int(prominence)
                                ),
                                f"Length ({dist_unit})": round(length, 2),
                                "Avg Grade (%)": round(min(climb.avg_grade, 50.0), 2),
                                "Max Grade (%)": round(climb.max_grade, 2),
                                "Highway Type": climb.highway_type,
                                "Surface": climb.surface,
                                "Tracktype": climb.tracktype,
                                "Start Way ID": way_ids_str,
                                "OSM Link": osm_links_str,
                                "All Way IDs": all_way_ids_str,
                                "Connected Climbs": (
                                    ", ".join(
                                        f"{name} ({way_id})"
                                        for name, way_id in climb.connected_climbs
                                    )
                                    if climb.connected_climbs
                                    else "None"
                                ),
                                f"Elevation Profile ({elev_unit})": getattr(
                                    climb, "elevation_profile", ""
                                )
                                or "",
                            }

                            # Write header on first row of a new file, then
                            # append this row. Both happen via the write-only
                            # worksheet which streams to a temp file on disk,
                            # so memory stays flat regardless of row count.
                            _write_header_if_needed(row)
                            _track_column_widths(row, is_header=False)

                            # Use consistent column order from the header
                            current_ws.append([row[k] for k in header_keys])

                            row_num += 1
                            current_file_rows += 1
                            pbar.update(1)

                except EOFError:
                    pass

                # Ensure progress bar shows 100%
                pbar.n = pbar.total
                pbar.refresh()

        # Finalize the last file (atomic save via .tmp rename)
        _finalize_current_file()

        # CHECKPOINT PRESERVATION: Keep temp files for recovery
        # DO NOT delete - allows resume if export phase crashes or analysis is re-run
        print_dim("   ✓ Preserved temp files in checkpoint_data:")
        print_dim(f"      - {Path(final_sorted_file.name).name}")
        print_dim(f"      - {Path(geocode_temp_file.name).name}")

        # Print summary
        if len(created_files) == 1:
            print(f"\n✓ Saved {total_filtered:,} climbs to {created_files[0].name}")
        else:
            print(f"\n✓ Saved {total_filtered:,} climbs to {len(created_files)} files:")
            for f in created_files:
                print(f"   {f.name}")

        # ============================================================================
        # SQLite Export - Generate iOS-compatible database alongside Excel
        # ============================================================================
        sqlite_files = []
        sqlite_checksums = {}
        sqlite_gz_files = []
        sqlite_gz_checksums = {}
        gz_decompressed_size = 0

        try:
            from climb_analyzer.data.sqlite_export import SQLiteExporter

            # Determine units for SQLite (ft or m)
            db_units = "ft" if units == "imperial" else "m"

            # Single SQLite file per region.
            # Files >2GB are split at upload time for GitHub releases.
            # iOS app handles spatial subsetting via R-tree index at query time.
            print(f"\nGenerating SQLite database for iOS app...")
            sqlite_filename = f"{base_filename}{suffix}.sqlite"
            sqlite_file = output_dir / sqlite_filename
            single_exporter = SQLiteExporter(sqlite_file, file_id=sqlite_filename, units=db_units)
            single_exporter.open()

            # Re-read the sorted temp file and geocode data
            with open(final_sorted_file.name, "rb") as f:
                geocode_lookup = {}
                if geocode_temp_file and Path(geocode_temp_file.name).exists():
                    with open(geocode_temp_file.name, "rb") as gf:
                        try:
                            while True:
                                geocode_lookup.update(safe_pickle_load(gf))
                        except EOFError:
                            pass

                with tqdm(
                    total=total_filtered, desc="  Writing SQLite", unit="climbs", **TQDM_DEFAULTS
                ) as pbar:
                    batch_rows = []

                    try:
                        while True:
                            batch_climbs = safe_pickle_load(f)

                            for climb in batch_climbs:
                                # Build row dict matching Excel format
                                geocode_key = f"{climb.start_lat:.5f},{climb.start_lon:.5f}"
                                geocode_data = geocode_lookup.get(geocode_key, {})

                                # Distance from center
                                center_distance = 0.0
                                if (
                                    analysis_center
                                    and analysis_center[0] is not None
                                    and analysis_center[1] is not None
                                ):
                                    from math import radians, sin, cos, sqrt, atan2
                                    lat1, lon1 = radians(analysis_center[0]), radians(analysis_center[1])
                                    lat2, lon2 = radians(climb.start_lat), radians(climb.start_lon)
                                    dlat, dlon = lat2 - lat1, lon2 - lon1
                                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                                    center_distance = 6371 * 2 * atan2(sqrt(a), sqrt(1-a))

                                # Unit conversions
                                if units == "imperial":
                                    elev_gain = climb.elevation_gain * 3.28084
                                    height = climb.height * 3.28084
                                    prominence = climb.prominence * 3.28084
                                    length = climb.length_km * 0.621371
                                    center_distance_display = center_distance * 0.621371
                                    elev_unit = "ft"
                                    dist_unit = "mi"
                                else:
                                    elev_gain = climb.elevation_gain
                                    height = climb.height
                                    prominence = climb.prominence
                                    length = climb.length_km
                                    center_distance_display = center_distance
                                    elev_unit = "m"
                                    dist_unit = "km"

                                # Way IDs
                                way_ids_str = str(climb.way_ids[0]) if climb.way_ids else ""
                                osm_links_str = climb.osm_links[0] if climb.osm_links else ""
                                all_way_ids_str = ", ".join(str(w) for w in (climb.way_ids or []))

                                row = {
                                    "Street Name": climb.street_name,
                                    "City": geocode_data.get("city", ""),
                                    "State": geocode_data.get("state", ""),
                                    "Country": geocode_data.get("country", ""),
                                    f"From Center ({dist_unit})": round(center_distance_display, 1),
                                    "Latitude": round(climb.start_lat, 5),
                                    "Longitude": round(climb.start_lon, 5),
                                    "Cycling": climb.cycling_access,
                                    "Category": climb.climb_category,
                                    "Basic Score": int(climb.climb_score),
                                    "FIETS Score": round(climb.fiets_score, 1),
                                    "PDI Score": round(climb.pdi_score, 1),
                                    f"Elev Gain ({elev_unit})": round(elev_gain, 2) if elev_gain < 1 else int(elev_gain),
                                    f"Height ({elev_unit})": round(height, 2) if height < 1 else int(height),
                                    f"Prominence ({elev_unit})": round(prominence, 2) if prominence < 1 else int(prominence),
                                    f"Length ({dist_unit})": round(length, 2),
                                    "Avg Grade (%)": round(min(climb.avg_grade, 50.0), 2),
                                    "Max Grade (%)": round(climb.max_grade, 2),
                                    "Highway Type": climb.highway_type,
                                    "Surface": climb.surface,
                                    "Tracktype": climb.tracktype,
                                    "Start Way ID": way_ids_str,
                                    "OSM Link": osm_links_str,
                                    "All Way IDs": all_way_ids_str,
                                    "Connected Climbs": (
                                        ", ".join(f"{name} ({way_id})" for name, way_id in climb.connected_climbs)
                                        if climb.connected_climbs else "None"
                                    ),
                                    f"Elevation Profile ({elev_unit})": getattr(climb, "elevation_profile", "") or "",
                                }

                                batch_rows.append(row)
                                if len(batch_rows) >= 5000:
                                    single_exporter.write_rows(batch_rows)
                                    batch_rows = []

                                pbar.update(1)

                    except EOFError:
                        pass

                    # Write final batch
                    if batch_rows:
                        single_exporter.write_rows(batch_rows)

            single_exporter.close()
            sqlite_size_mb = sqlite_file.stat().st_size / (1024**2)
            decompressed_size = sqlite_file.stat().st_size
            print(f"✓ Saved SQLite database: {sqlite_file.name} ({single_exporter.row_count:,} rows, {sqlite_size_mb:.1f} MB)")

            # SQLite distribution strategy for GitHub Releases (2GB per-asset limit):
            #
            # 1. Raw .sqlite path (backward compat for older iOS app versions):
            #    - If .sqlite < 1.95 GB: upload as single file
            #    - If .sqlite >= 1.95 GB: split into .sqlite.001/.002/...
            #
            # 2. Gzipped .sqlite.gz path (preferred, ~3x smaller downloads):
            #    - If .sqlite.gz < 1.95 GB: upload as single file (typical)
            #    - If .sqlite.gz >= 1.95 GB: split into .sqlite.gz.001/.002/...
            #
            # Both paths are uploaded so old and new iOS app versions both work.
            # The gz file is produced via streaming compression, so memory stays flat.
            from utils.file_splitter import should_split_file, split_file, gzip_file

            # --- Path 1: gzip the sqlite (preferred path) ---
            gz_decompressed_size = decompressed_size
            try:
                gz_path = gzip_file(sqlite_file, compresslevel=6, delete_original=False)
                gz_size = gz_path.stat().st_size
                if should_split_file(gz_path):
                    print(
                        f"\n  Compressed file still exceeds 1.95GB - splitting .sqlite.gz for GitHub releases..."
                    )
                    chunks, sqlite_gz_checksums = split_file(gz_path, delete_original=True)
                    sqlite_gz_files = chunks
                    print(f"  ✓ Created {len(chunks)} gz chunks with SHA256 checksums")
                else:
                    sqlite_gz_files = [gz_path]
                    # Single-file checksum for verification on-device
                    from utils.file_splitter import calculate_sha256
                    sqlite_gz_checksums[gz_path.name] = calculate_sha256(gz_path)
            except Exception as e:
                print(f"⚠️  gzip step failed, continuing with raw sqlite only: {e}")
                import traceback
                traceback.print_exc()

            # --- Path 2: raw .sqlite (backward compat) ---
            # Still performed so existing app versions keep working.
            sqlite_files.append(sqlite_file)
            if should_split_file(sqlite_file):
                print(f"\n  Raw SQLite exceeds 1.95GB - also producing .sqlite.001/.002/... for backward compat...")
                chunks, sqlite_checksums = split_file(sqlite_file, delete_original=True)
                sqlite_files = chunks
                print(f"  ✓ Created {len(chunks)} raw chunks with SHA256 checksums")

        except Exception as e:
            print(f"⚠️  SQLite export failed: {e}")
            import traceback
            traceback.print_exc()

        # Return dict with both Excel, raw SQLite, and gzipped SQLite paths.
        # sqlite/sqlite_checksums: raw sqlite or its .001/.002 split chunks.
        # sqlite_gz/sqlite_gz_checksums: single .sqlite.gz or its .gz.001/.002 split chunks.
        return {
            "xlsx": created_files,
            "sqlite": sqlite_files,
            "sqlite_checksums": sqlite_checksums,
            "sqlite_gz": sqlite_gz_files,
            "sqlite_gz_checksums": sqlite_gz_checksums,
            "sqlite_decompressed_size": gz_decompressed_size,
            "climb_count": total_filtered,
        }

    # ============================================================================
    # DEAD CODE - Chunked export function (never executes)
    # Streaming export threshold (100K) is lower than chunked threshold (500K)
    # Consolidated to streaming export only for all dataset sizes
    # ============================================================================

    # def _export_large_dataset_chunked(
    #     self, sorted_climbs, location_data, units, min_score, chunk_size=50000
    # ):
    #     """
    #     Export very large datasets (>500K climbs) directly to Excel in chunks.
    #
    #     This avoids loading all climb data into memory at once by writing
    #     the DataFrame in batches.
    #     """
    #     import pandas as pd
    #     from tqdm import tqdm
    #
    #     print(f"   Processing {len(sorted_climbs):,} climbs in chunks of {chunk_size:,}...")
    #
    #     # Determine unit labels
    #     if units == "imperial":
    #         elev_unit = "ft"
    #         dist_unit = "mi"
    #     else:
    #         elev_unit = "m"
    #         dist_unit = "km"
    #
    #     # Build DataFrame in chunks and concatenate
    #     all_chunks = []
    #     num_chunks = (len(sorted_climbs) + chunk_size - 1) // chunk_size
    #
    #     for chunk_idx in tqdm(
    #         range(num_chunks), desc="Building DataFrame", unit="chunk", **TQDM_DEFAULTS
    #     ):
    #         start_idx = chunk_idx * chunk_size
    #         end_idx = min(start_idx + chunk_size, len(sorted_climbs))
    #         chunk_climbs = sorted_climbs[start_idx:end_idx]
    #
    #         chunk_data = []
    #         for local_i, climb in enumerate(chunk_climbs):
    #             # Calculate absolute index for location_data lookup
    #             i = start_idx + local_i
    #             # Convert units
    #             if units == "imperial":
    #                 elev_gain = climb.elevation_gain * 3.28084
    #                 height = climb.height * 3.28084
    #                 prominence = climb.prominence * 3.28084
    #                 length = climb.length_km * 0.621371
    #                 center_distance = location_data[i]["distance_km"] * 0.621371
    #             else:
    #                 elev_gain = climb.elevation_gain
    #                 height = climb.height
    #                 prominence = climb.prominence
    #                 length = climb.length_km
    #                 center_distance = location_data[i]["distance_km"]
    #
    #             way_ids_str = str(climb.way_ids[0]) if climb.way_ids else "N/A"
    #             osm_links_str = climb.osm_links[0] if climb.osm_links else "N/A"
    #             # All way IDs for merged climbs (comma-separated)
    #             all_way_ids_str = ", ".join(str(wid) for wid in climb.way_ids) if climb.way_ids else "N/A"
    #
    #             row = {
    #                 "Street Name": climb.street_name,
    #                 "City": location_data[i]["city"],
    #                 "State": location_data[i]["state"],
    #                 "Country": location_data[i].get("country", "Unknown"),
    #                 f"From Center ({dist_unit})": round(center_distance, 1),
    #                 "Latitude": round(climb.start_lat, 5),
    #                 "Longitude": round(climb.start_lon, 5),
    #                 "Cycling": climb.cycling_access,
    #                 "Category": climb.climb_category,
    #                 "Basic Score": int(climb.climb_score),
    #                 "FIETS Score": round(climb.fiets_score, 1),
    #                 "PDI Score": round(climb.pdi_score, 1),
    #                 f"Elev Gain ({elev_unit})": (
    #                     round(elev_gain, 2) if elev_gain < 1 else int(elev_gain)
    #                 ),
    #                 f"Height ({elev_unit})": round(height, 2) if height < 1 else int(height),
    #                 f"Prominence ({elev_unit})": (
    #                     round(prominence, 2) if prominence < 1 else int(prominence)
    #                 ),
    #                 f"Length ({dist_unit})": round(length, 2),
    #                 "Avg Grade (%)": round(min(climb.avg_grade, 50.0), 2),
    #                 "Max Grade (%)": round(climb.max_grade, 2),
    #                 "Highway Type": climb.highway_type,
    #                 "Surface": climb.surface,
    #                 "Tracktype": climb.tracktype,
    #                 "Way ID": way_ids_str,
    #                 "OSM Link": osm_links_str,
    #                 "All Way IDs": all_way_ids_str,
    #                 "Connected Climbs": (
    #                     ", ".join(climb.connected_climbs) if climb.connected_climbs else "None"
    #                 ),
    #                 "Elevation Profile": getattr(climb, "elevation_profile", "") or "",
    #             }
    #             chunk_data.append(row)
    #
    #         # Create DataFrame for this chunk
    #         chunk_df = pd.DataFrame(chunk_data)
    #         all_chunks.append(chunk_df)
    #
    #         # Clear memory
    #         del chunk_data
    #
    #     # Concatenate all chunks
    #     print("   Concatenating chunks...")
    #     final_df = pd.concat(all_chunks, ignore_index=True)
    #     del all_chunks
    #
    #     print(f"✓ DataFrame created with {len(final_df):,} rows")
    #     return final_df

    def _detect_connected_climbs(self):
        """
        Detect which climbs are connected to each other by analyzing endpoints.

        A climb B is considered "connected" to climb A only if:
        1. Climb A's END point matches Climb B's START point (continuation)
        2. Climb B continues upward (its max elevation >= Climb A's max elevation)

        This ensures we only show connections that represent continuing climbs,
        not climbs that share a node but go in opposite directions.

        Updates each climb's connected_climbs field with a list of connected climb names.
        """
        if len(self.climbs) < 2:
            return

        def get_coord(node):
            """Extract rounded coordinate from node (handles both dict and object)."""
            if isinstance(node, dict):
                return (round(node["lat"], 6), round(node["lon"], 6))
            elif hasattr(node, "lat") and hasattr(node, "lon"):
                return (round(node.lat, 6), round(node.lon, 6))
            return None

        # Build index of climb START points: coord -> list of (climb_idx, climb)
        start_point_index = {}
        climb_info = {}  # climb_idx -> (street_name, max_elevation)

        for i, climb in enumerate(self.climbs):
            climb_info[i] = (climb.street_name, climb.max_elevation)
            if not climb.nodes or len(climb.nodes) < 2:
                continue

            start_coord = get_coord(climb.nodes[0])
            if start_coord:
                if start_coord not in start_point_index:
                    start_point_index[start_coord] = []
                start_point_index[start_coord].append((i, climb))

        # Find connections: climb A's END connects to climb B's START (and B goes up)
        climb_connections = {i: set() for i in range(len(self.climbs))}

        for i, climb in enumerate(self.climbs):
            if not climb.nodes or len(climb.nodes) < 2:
                continue

            # Get this climb's END coordinate
            end_coord = get_coord(climb.nodes[-1])
            if not end_coord:
                continue

            # Check if any other climb STARTS at this climb's END
            if end_coord in start_point_index:
                for other_idx, other_climb in start_point_index[end_coord]:
                    # Skip same climb
                    if other_idx == i:
                        continue
                    # Skip same street name AND same surface (they should be merged, not connected)
                    # Allow same-name connections if surfaces differ
                    if (other_climb.street_name == climb.street_name and
                        getattr(other_climb, 'surface', 'unknown') == getattr(climb, 'surface', 'unknown')):
                        continue

                    # Bidirectional connection - add both directions
                    climb_connections[i].add(other_idx)
                    climb_connections[other_idx].add(i)

        # Update each climb's connected_climbs field
        connections_found = 0
        for i, climb in enumerate(self.climbs):
            connected_climb_names = []
            for connected_idx in climb_connections[i]:
                connected_name = climb_info[connected_idx][0]
                if connected_name not in connected_climb_names:  # Avoid duplicates
                    connected_climb_names.append(connected_name)

            if connected_climb_names:
                climb.connected_climbs = sorted(connected_climb_names)
                connections_found += 1
            else:
                climb.connected_climbs = []

        print(f"Found connections for {connections_found} climbs")

    def _build_endpoint_index_streaming(
        self, temp_climbs_file: Path, climbs_count: int
    ) -> Tuple[Dict, List]:
        """Build spatial grid index of climb endpoints by streaming through pickle file."""
        import pickle
        from collections import defaultdict

        # Grid index: grid_cell -> [ClimbEndpoints]
        grid_index = defaultdict(list)
        all_endpoints = []

        print("Building spatial index of climb endpoints...")
        with open(temp_climbs_file, "rb") as f:
            climb_idx = 0
            try:
                while True:
                    # Load batches of climbs (as written by analysis phase)
                    batch_climbs = safe_pickle_load(f)

                    for climb in batch_climbs:
                        if not climb.nodes or len(climb.nodes) < 2:
                            climb_idx += 1
                            continue

                        # Extract only start/end coordinates
                        start_node = climb.nodes[0]
                        end_node = climb.nodes[-1]

                        # Handle both dict and object node types
                        if isinstance(start_node, dict):
                            start_coord = (round(start_node["lat"], 6), round(start_node["lon"], 6))
                            end_coord = (round(end_node["lat"], 6), round(end_node["lon"], 6))
                        else:
                            start_coord = (round(start_node.lat, 6), round(start_node.lon, 6))
                            end_coord = (round(end_node.lat, 6), round(end_node.lon, 6))

                        # Determine which grid cells this climb touches
                        start_cell = get_grid_cell(*start_coord)
                        end_cell = get_grid_cell(*end_coord)
                        cells = frozenset([start_cell, end_cell])

                        # Get first way ID for this climb
                        first_way_id = climb.way_ids[0] if climb.way_ids else 0

                        endpoint = ClimbEndpoints(
                            climb_idx=climb_idx,
                            street_name=climb.street_name,
                            way_id=first_way_id,
                            start_coord=start_coord,
                            end_coord=end_coord,
                            grid_cells=cells,
                            max_elevation=climb.max_elevation,
                            surface=getattr(climb, 'surface', 'unknown'),
                        )

                        all_endpoints.append(endpoint)

                        # Add to grid index
                        for cell in cells:
                            grid_index[cell].append(endpoint)

                        climb_idx += 1
            except EOFError:
                pass  # Normal end of file

        print(f"✓ Indexed {climb_idx:,} climbs across {len(grid_index):,} grid cells")
        return dict(grid_index), all_endpoints

    def _detect_connected_climbs_spatial(
        self,
        temp_climbs_file: Path,
        grid_index: Dict,
        all_endpoints: List,
        climbs_count: int,
        persistence=None,
    ):
        """Detect connected climbs using spatial grid index with checkpoint support."""
        import gc
        import pickle
        from collections import defaultdict

        from tqdm import tqdm

        # Check for existing checkpoint
        connections_checkpoint_file = None
        resume_from_index = 0
        connections = defaultdict(set)

        if persistence:
            connections_checkpoint_file = persistence.analysis_dir / "connected_climbs_progress.pkl"

            if connections_checkpoint_file.exists():
                try:
                    with open(connections_checkpoint_file, "rb") as f:
                        checkpoint_data = safe_pickle_load(f)
                        resume_from_index = checkpoint_data.get("endpoints_processed", 0)
                        # Restore connections dict (convert lists back to sets)
                        saved_connections = checkpoint_data.get("connections", {})
                        connections = defaultdict(set)
                        for k, v in saved_connections.items():
                            connections[k] = set(v)

                        if resume_from_index > 0:
                            print("✓ Found existing connected climbs checkpoint")
                            print(
                                f"Resuming from endpoint {resume_from_index:,}/{len(all_endpoints):,}"
                            )
                            print()
                except Exception as e:
                    print(f"⚠️  Could not load checkpoint: {e}")
                    print("   Starting from beginning...")
                    resume_from_index = 0
                    connections = defaultdict(set)

        print("Detecting connected climbs using spatial index...")
        if persistence:
            print("    Checkpointing enabled: Progress saved every 500K endpoints")
        print()

        last_checkpoint_index = resume_from_index

        with tqdm(
            total=len(all_endpoints),
            initial=resume_from_index,
            desc="Checking connections",
            unit="climbs",
            **TQDM_DEFAULTS,
        ) as pbar:
            # Force immediate render when resuming from checkpoint
            if resume_from_index > 0:
                pbar.refresh()

            for idx, endpoint in enumerate(all_endpoints):
                # Skip already processed endpoints if resuming
                if idx < resume_from_index:
                    continue

                # Track nearby climbs we've already checked (per endpoint, not global)
                nearby_seen = set()

                # Get all nearby climbs from adjacent grid cells
                for cell in endpoint.grid_cells:
                    for adj_cell in get_adjacent_cells(cell):
                        if adj_cell in grid_index:
                            for other in grid_index[adj_cell]:
                                # Only check each pair once: process only if other climb has higher index
                                if (
                                    other.climb_idx > endpoint.climb_idx
                                    and other.climb_idx not in nearby_seen
                                ):
                                    nearby_seen.add(other.climb_idx)

                                    # Skip same street name AND same surface (they should be merged)
                                    # Allow same-name connections if surfaces differ
                                    if (other.street_name == endpoint.street_name and
                                        other.surface == endpoint.surface):
                                        continue

                                    # Bidirectional connections - no upward filter
                                    # Connection 1: endpoint's END matches other's START
                                    if endpoint.end_coord == other.start_coord:
                                        connections[endpoint.climb_idx].add(other.climb_idx)
                                        connections[other.climb_idx].add(endpoint.climb_idx)

                                    # Connection 2: other's END matches endpoint's START
                                    if other.end_coord == endpoint.start_coord:
                                        connections[other.climb_idx].add(endpoint.climb_idx)
                                        connections[endpoint.climb_idx].add(other.climb_idx)

                # Update progress bar after processing endpoint
                pbar.update(1)

                # Periodic garbage collection and checkpointing
                if idx > 0 and idx % 50000 == 0:
                    gc.collect()

                # Save checkpoint every 500K endpoints
                if (
                    persistence
                    and idx - last_checkpoint_index >= 500000
                    and idx > resume_from_index
                ):
                    # Convert sets to lists for pickling
                    checkpoint_data = {
                        "endpoints_processed": idx,
                        "connections": {k: list(v) for k, v in connections.items()},
                        "timestamp": time.time(),
                    }
                    with open(connections_checkpoint_file, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    last_checkpoint_index = idx
                    pbar.write(f"    Checkpoint saved: {idx:,} endpoints checked")

        # Save final checkpoint
        if persistence:
            checkpoint_data = {
                "endpoints_processed": len(all_endpoints),
                "connections": {k: list(v) for k, v in connections.items()},
                "timestamp": time.time(),
                "completed": True,
            }
            with open(connections_checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            print("    Final checkpoint saved")

        # Build map: climb_idx -> [(street_name, way_id), ...]
        # Store tuples of (street_name, way_id) for each connected climb
        endpoint_map = {ep.climb_idx: (ep.street_name, ep.way_id) for ep in all_endpoints}
        connection_names = {}
        for climb_idx, connected_indices in connections.items():
            # Filter out any indices that don't exist in endpoint_map (safety check)
            valid_indices = {idx for idx in connected_indices if idx in endpoint_map}
            if valid_indices:
                # Store tuples of (street_name, way_id) for each connection
                connection_tuples = {endpoint_map[idx] for idx in valid_indices}
                # Sort by street name for consistent ordering
                connection_names[climb_idx] = sorted(connection_tuples, key=lambda x: x[0])

        print(f"✓ Found connections for {len(connection_names):,} climbs")

        # Update climbs with connection info
        self._update_climbs_with_connections(temp_climbs_file, connection_names, climbs_count)

    def _update_climbs_with_connections(
        self, temp_climbs_file: Path, connection_names: Dict, climbs_count: int
    ):
        """
        Update climbs in pickle file with connection information.
        Uses streaming approach to avoid loading all climbs into memory.
        """
        import gc
        import pickle

        # Create temporary output file
        temp_output = temp_climbs_file.parent / f"{temp_climbs_file.stem}_updated.pkl"

        print()
        print("Updating climbs with connection information (streaming mode)...")
        print()

        climb_idx = 0
        batch_count = 0

        # Stream: read batch -> update -> write immediately
        with open(temp_climbs_file, "rb") as f_in, open(temp_output, "wb") as f_out:
            try:
                while True:
                    batch_climbs = safe_pickle_load(f_in)

                    # Update this batch with connection information
                    for climb in batch_climbs:
                        climb.connected_climbs = connection_names.get(climb_idx, [])
                        climb_idx += 1

                    # Write updated batch immediately (don't accumulate)
                    pickle.dump(batch_climbs, f_out)
                    batch_count += 1

                    # Periodic progress and GC
                    if batch_count % 50 == 0:
                        # Update in-place using \r (carriage return)
                        print(
                            f"\r  Processed {climb_idx:,}/{climbs_count:,} climbs ({100*climb_idx/climbs_count:.1f}%)",
                            end="",
                            flush=True,
                        )
                        gc.collect()

            except EOFError:
                pass

        # Replace original file with updated version
        import shutil

        shutil.move(str(temp_output), str(temp_climbs_file))

        # Print newline to complete the in-place progress line
        print()
        print(f"✓ Updated {climb_idx:,} climbs with connection data")

    def _compile_geocoding_results(
        self,
        climbs: List,
        coordinates: List,
        coord_to_location: Dict,
        analysis_center: Tuple[float, float],
    ) -> List[Dict]:
        """Compile final results from offline geocoding data"""

        results = []
        for i, climb in enumerate(climbs):
            coord = coordinates[i]

            if coord and coord != (0.0, 0.0) and coord in coord_to_location:
                location_info = coord_to_location[coord]

                # Calculate distance from analysis center to climb start
                distance_km = self.calculate_distance(
                    analysis_center[0], analysis_center[1], coord[0], coord[1]
                )

                results.append(
                    {
                        "city": location_info["city"],
                        "state": location_info["state"],
                        "country": location_info.get("country", "Unknown"),
                        "distance_km": distance_km,
                        "start_lat": coord[0],
                        "start_lon": coord[1],
                    }
                )
            else:
                results.append(
                    {
                        "city": "Unknown",
                        "state": "Unknown",
                        "country": "Unknown",
                        "distance_km": 0.0,
                        "start_lat": None,
                        "start_lon": None,
                    }
                )

        return results

    def _perform_geocoding_with_checkpoints(
        self,
        climbs: List,
        analysis_center: Tuple[float, float],
        persistence: ChunkPersistenceManager,
    ) -> List[Dict]:
        """Perform geocoding using offline reverse_geocoder
        (much faster, minimal checkpointing needed)"""

        print("Using offline reverse geocoding for fast location lookup...")

        try:
            # Extract coordinates from climbs
            coordinates = []
            for climb in climbs:
                start_coord = self._get_climb_start_coordinates(climb)
                coordinates.append(start_coord if start_coord else (0.0, 0.0))

            # Filter valid coordinates
            valid_coords = []
            coord_to_climb_map = {}
            for i, coord in enumerate(coordinates):
                if coord and coord != (0.0, 0.0):
                    valid_coords.append(coord)
                    coord_to_climb_map[coord] = i

            # Import reverse_geocoder locally to defer loading large spatial index
            import reverse_geocoder as rg

            # Batch reverse geocoding (very fast with offline data)
            coord_to_location = {}
            if valid_coords:
                print(f"Processing {len(valid_coords)} coordinates...")

                # Since reverse_geocoder is fast, we can process larger batches
                batch_size = 5000
                with tqdm(
                    total=len(valid_coords),
                    desc="Offline geocoding",
                    unit="coords",
                    dynamic_ncols=True,
                    ascii=" ▏▎▍▌▋▊▉█",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                ) as pbar:
                    for i in range(0, len(valid_coords), batch_size):
                        batch = valid_coords[i : i + batch_size]

                        # Signal handler check (less frequent since it's fast)
                        if i % 10000 == 0:
                            signal_handler.set_operation(
                                "geocoding",
                                {
                                    "processed_coords": i,
                                    "total_coords": len(valid_coords),
                                    "coord_to_location": coord_to_location,
                                    "timestamp": time.time(),
                                },
                            )

                        if signal_handler.kill_now:
                            print(
                                "Geocoding interrupted - most data should be available since it's offline"
                            )
                            break

                        try:
                            results = rg.search(batch)

                            for coord, result in zip(batch, results):
                                if result:
                                    city = result.get("name", "Unknown")
                                    state = result.get("admin1", "Unknown")
                                    country = result.get("cc", "")

                                    full_address_parts = [city]
                                    if state and state != "Unknown":
                                        full_address_parts.append(state)
                                    if country:
                                        full_address_parts.append(country)
                                    full_address = ", ".join(full_address_parts)

                                    coord_to_location[coord] = {
                                        "city": city,
                                        "state": state,
                                        "country": country,
                                        "full_address": full_address,
                                    }
                                else:
                                    coord_to_location[coord] = {
                                        "city": "Unknown",
                                        "state": "Unknown",
                                        "country": "Unknown",
                                        "full_address": "Not found",
                                    }

                        except Exception as e:
                            print(f"Error in batch geocoding: {e}")
                            # Fill with fallback
                            for coord in batch:
                                coord_to_location[coord] = {
                                    "city": "Lookup Failed",
                                    "state": "Lookup Failed",
                                    "country": "Lookup Failed",
                                    "full_address": f"Error: {str(e)[:50]}",
                                }

                        pbar.update(len(batch))

            # Compile results for all climbs
            results = self._compile_geocoding_results(
                climbs, coordinates, coord_to_location, analysis_center
            )

            print(f"Offline geocoding completed for {len(valid_coords)} coordinates")
            return results

        except ImportError:
            print("reverse_geocoder module not found. Falling back to distance-only calculation.")
            return self._calculate_distances_only(climbs, analysis_center)
        except Exception as e:
            print(f"Error with offline geocoding: {e}. Falling back to distance-only calculation.")
            return self._calculate_distances_only(climbs, analysis_center)

    def _resume_geocoding(
        self,
        climbs: List,
        checkpoint_data: Dict,
        analysis_center: Tuple[float, float],
        persistence: ChunkPersistenceManager,
    ) -> List[Dict]:
        """Resume geocoding from checkpoint."""

        processed_coords = checkpoint_data.get("processed_coords", 0)
        total_coords = checkpoint_data.get("total_coords", 0)
        coord_to_location = checkpoint_data.get("coord_to_location", {})
        climb_to_coord = checkpoint_data.get("climb_to_coord", [])
        unique_coords = checkpoint_data.get("unique_coords", {})
        current_coord_idx = checkpoint_data.get("current_coord_idx", 0)

        print(f"Resuming geocoding from coordinate {current_coord_idx}/{total_coords}")
        print(f"Already processed: {processed_coords} coordinates")

        if current_coord_idx >= total_coords:
            print("All coordinates already processed!")
            return self._compile_geocoding_results(
                climbs,
                climb_to_coord,
                unique_coords,
                coord_to_location,
                analysis_center,
            )

        # Continue processing remaining coordinates
        remaining_coords = list(unique_coords.items())[current_coord_idx:]
        checkpoint_interval = max(1, len(remaining_coords) // 20)

        with tqdm(
            total=len(remaining_coords),
            desc="Resuming geocoding",
            unit="coords",
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for i, (rounded_coord, actual_coord) in enumerate(remaining_coords):
                # Skip if already processed
                if rounded_coord in coord_to_location:
                    pbar.update(1)
                    continue

                try:
                    location = self.geolocator.reverse(
                        f"{actual_coord[0]:.6f},{actual_coord[1]:.6f}",
                        timeout=10,
                        exactly_one=True,
                    )

                    if location and location.address:
                        address_parts = location.raw.get("address", {})

                        city = (
                            address_parts.get("city")
                            or address_parts.get("town")
                            or address_parts.get("village")
                            or address_parts.get("hamlet")
                            or address_parts.get("county")
                            or "Unknown"
                        )

                        state = (
                            address_parts.get("state")
                            or address_parts.get("province")
                            or address_parts.get("region")
                            or "Unknown"
                        )

                        coord_to_location[rounded_coord] = {
                            "city": city,
                            "state": state,
                            "full_address": location.address,
                        }
                    else:
                        coord_to_location[rounded_coord] = {
                            "city": "Unknown",
                            "state": "Unknown",
                            "full_address": "Location not found",
                        }

                except Exception as e:
                    coord_to_location[rounded_coord] = {
                        "city": "Lookup Failed",
                        "state": "Lookup Failed",
                        "full_address": f"Error: {str(e)[:50]}",
                    }

                processed_coords += 1

                # Save checkpoint periodically
                if i % checkpoint_interval == 0 or i == len(remaining_coords) - 1:
                    updated_checkpoint = {
                        "total_climbs": len(climbs),
                        "processed_coords": processed_coords,
                        "total_coords": total_coords,
                        "coord_to_location": coord_to_location,
                        "climb_to_coord": climb_to_coord,
                        "unique_coords": unique_coords,
                        "current_coord_idx": current_coord_idx + i + 1,
                        "analysis_center": analysis_center,
                        "timestamp": time.time(),
                    }
                    self._save_geocoding_checkpoint(updated_checkpoint, persistence)

                pbar.set_postfix({"completed": processed_coords, "total": total_coords})
                pbar.update(1)
                time.sleep(0.2)

        # Compile final results
        results = self._compile_geocoding_results(
            climbs, climb_to_coord, unique_coords, coord_to_location, analysis_center
        )

        # Clear checkpoint after successful completion
        self._clear_geocoding_checkpoint(persistence)

        return results

    def _compile_geocoding_results(
        self,
        climbs: List,
        climb_to_coord: List,
        unique_coords: Dict,
        coord_to_location: Dict,
        analysis_center: Tuple[float, float],
    ) -> List[Dict]:
        """Compile final results from geocoding data."""

        results = []
        for i, climb in enumerate(climbs):
            rounded_coord = climb_to_coord[i]

            if rounded_coord and rounded_coord in coord_to_location:
                location_info = coord_to_location[rounded_coord]

                # Calculate distance from analysis center to climb start
                actual_coord = unique_coords[rounded_coord]
                distance_km = self.calculate_distance(
                    analysis_center[0],
                    analysis_center[1],
                    actual_coord[0],
                    actual_coord[1],
                )

                results.append(
                    {
                        "city": location_info["city"],
                        "state": location_info["state"],
                        "country": location_info.get("country", "Unknown"),
                        "distance_km": distance_km,
                        "start_lat": actual_coord[0],
                        "start_lon": actual_coord[1],
                    }
                )
            else:
                results.append(
                    {
                        "city": "Unknown",
                        "state": "Unknown",
                        "country": "Unknown",
                        "distance_km": 0.0,
                        "start_lat": None,
                        "start_lon": None,
                    }
                )

        return results

    def _save_geocoding_checkpoint(
        self, checkpoint_data: Dict, persistence: ChunkPersistenceManager
    ):
        """Save geocoding checkpoint to disk."""
        geocoding_checkpoint_file = persistence.analysis_dir / "geocoding_progress.pkl"
        temp_file = persistence.analysis_dir / "geocoding_progress.tmp"

        try:
            with open(temp_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            temp_file.rename(geocoding_checkpoint_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            print(f"Error saving geocoding checkpoint: {e}")

    def _clear_geocoding_checkpoint(self, persistence: ChunkPersistenceManager):
        """Clear geocoding checkpoint after successful completion."""
        geocoding_checkpoint_file = persistence.analysis_dir / "geocoding_progress.pkl"
        try:
            if geocoding_checkpoint_file.exists():
                geocoding_checkpoint_file.unlink()
        except Exception as e:
            print(f"Error clearing geocoding checkpoint: {e}")

    def _calculate_distances_only(
        self, climbs: List, analysis_center: Tuple[float, float]
    ) -> List[Dict]:
        """Calculate distances from analysis center without reverse geocoding."""
        results = []

        for climb in climbs:
            # Get starting coordinates from first node
            start_coord = self._get_climb_start_coordinates(climb)

            if start_coord:
                # Calculate distance from analysis center to climb start
                distance_km = self.calculate_distance(
                    analysis_center[0],
                    analysis_center[1],
                    start_coord[0],
                    start_coord[1],
                )

                results.append(
                    {
                        "city": "N/A",
                        "state": "N/A",
                        "country": "N/A",
                        "distance_km": distance_km,
                        "start_lat": start_coord[0],
                        "start_lon": start_coord[1],
                    }
                )
            else:
                results.append(
                    {
                        "city": "N/A",
                        "state": "N/A",
                        "country": "N/A",
                        "distance_km": 0.0,
                        "start_lat": None,
                        "start_lon": None,
                    }
                )

        return results

    def _get_climb_start_coordinates(self, climb) -> Optional[Tuple[float, float]]:
        """Extract starting coordinates for a climb."""
        # First check if climb has start_lat/start_lon fields (preferred)
        if hasattr(climb, "start_lat") and hasattr(climb, "start_lon"):
            if climb.start_lat != 0.0 or climb.start_lon != 0.0:
                return (climb.start_lat, climb.start_lon)

        # Fallback: Check if climb has direct node access
        if hasattr(climb, "nodes") and climb.nodes:
            first_node = climb.nodes[0]
            if hasattr(first_node, "lat") and hasattr(first_node, "lon"):
                return (float(first_node.lat), float(first_node.lon))

        # Alternative: Use OSM API to get way coordinates (less efficient)
        if hasattr(climb, "way_ids") and climb.way_ids:
            return self._get_way_start_from_osm(climb.way_ids[0])

        return None

    def _get_way_start_from_osm(self, way_id: int) -> Optional[Tuple[float, float]]:
        """Get starting coordinates of an OSM way using Overpass API."""
        try:
            # Query for the specific way
            query = f"""
            [out:json][timeout:10];
            way({way_id});
            (._;>;);
            out geom;
            """

            if OVERPASS_API_URL:
                response = requests.post(
                    OVERPASS_API_URL,
                    data=query,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15,
                )
            else:
                # Use public Overpass API as fallback
                response = requests.post(
                    "https://overpass-api.de/api/interpreter",
                    data=query,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15,
                )

            if response.status_code == 200:
                data = response.json()

                # Find the way and get its first node coordinates
                for element in data.get("elements", []):
                    if element.get("type") == "way" and element.get("id") == way_id:
                        # Check for geometry data first (preferred)
                        if "geometry" in element and element["geometry"]:
                            first_point = element["geometry"][0]
                            if "lat" in first_point and "lon" in first_point:
                                return (first_point["lat"], first_point["lon"])

                        # Fallback: get first node ID and find its coordinates
                        if "nodes" in element and element["nodes"]:
                            first_node_id = element["nodes"][0]

                            # Find the node with this ID
                            for node_element in data.get("elements", []):
                                if (
                                    node_element.get("type") == "node"
                                    and node_element.get("id") == first_node_id
                                ):
                                    return (node_element["lat"], node_element["lon"])

        except Exception as e:
            print(f"Warning: Could not fetch coordinates for way {way_id}: {e}")

        return None

    def smooth_elevations(
        self, elevations: List[float], distances_km: List[float], window_size_m: float = 25.0
    ) -> List[float]:
        """
        Apply moving average smoothing to elevation data to reduce noise from elevation errors.

        This smoothing helps eliminate unrealistic grade spikes caused by ±1-5m elevation
        precision errors in SRTM/NED data, while preserving the overall climb profile.

        Args:
            elevations: List of elevation values (meters)
            distances_km: List of cumulative distances between points (km)
            window_size_m: Window size for moving average in meters (default 25m)

        Returns:
            List of smoothed elevation values
        """
        if len(elevations) < 3:
            return elevations[:]  # Return copy, no smoothing possible

        # Convert distances to meters and calculate cumulative distances
        cumulative_dist = [0.0]
        for d_km in distances_km:
            cumulative_dist.append(cumulative_dist[-1] + d_km * 1000)

        smoothed = []

        for i in range(len(elevations)):
            # Find points within window_size_m of current point
            current_pos = cumulative_dist[i]
            window_elevations = []

            for j in range(len(elevations)):
                dist_from_current = abs(cumulative_dist[j] - current_pos)
                if dist_from_current <= window_size_m / 2:
                    window_elevations.append(elevations[j])

            # Calculate moving average
            if window_elevations:
                smoothed.append(sum(window_elevations) / len(window_elevations))
            else:
                # Fallback: use original elevation
                smoothed.append(elevations[i])

        return smoothed

    def calculate_max_grade_smoothed(
        self,
        elevations: List[float],
        distances_km: List[float],
        smoothing_window_m: float = 25.0,
        grade_window_m: float = 30.0,
        outlier_cap_pct: float = 35.0,
    ) -> float:
        """
        Calculate maximum grade using smoothed elevation data and larger grade windows.

        This approach provides accurate max_grade values that are:
        - Smoothed to eliminate elevation precision errors
        - Calculated over realistic distances (not point-to-point)
        - Capped at physically realistic values for outlier rejection
        - Truthful to actual terrain steepness

        Args:
            elevations: List of elevation values (meters)
            distances_km: List of distances between consecutive points (km)
            smoothing_window_m: Window size for elevation smoothing (default 25m)
            grade_window_m: Minimum distance for grade calculation (default 30m)
            outlier_cap_pct: Maximum grade for outlier rejection (default 35%)

        Returns:
            Maximum grade percentage, or 0 if insufficient data
        """
        if len(elevations) < 2 or len(distances_km) < 1:
            return 0.0

        # Step 1: Apply smoothing to reduce elevation noise
        smoothed_elevations = self.smooth_elevations(elevations, distances_km, smoothing_window_m)

        # Step 2: Calculate cumulative distances
        cumulative_dist = [0.0]
        for d_km in distances_km:
            cumulative_dist.append(cumulative_dist[-1] + d_km * 1000)

        total_distance_m = cumulative_dist[-1]

        # Step 3: Calculate grades over grade_window_m distances
        max_grade = 0.0

        for i in range(len(smoothed_elevations)):
            current_pos = cumulative_dist[i]

            # Look ahead to find a point at least grade_window_m away
            for j in range(i + 1, len(smoothed_elevations)):
                distance_between = cumulative_dist[j] - current_pos

                # Only calculate grade if we have sufficient distance
                if distance_between >= grade_window_m:
                    elevation_change = smoothed_elevations[j] - smoothed_elevations[i]

                    # Only consider positive grades for max_grade
                    if elevation_change > 0:
                        grade = (elevation_change / distance_between) * 100

                        # Apply outlier cap (but allow if consistent)
                        if grade <= outlier_cap_pct:
                            max_grade = max(max_grade, grade)
                        elif grade > outlier_cap_pct and grade < outlier_cap_pct * 1.5:
                            # Allow grades up to 1.5x cap if they appear consistent
                            # Check if nearby grades are also high
                            nearby_high_grades = 0
                            for k in range(max(0, i - 2), min(len(smoothed_elevations), i + 3)):
                                if k != i and k < len(smoothed_elevations) - 1:
                                    test_dist = cumulative_dist[k + 1] - cumulative_dist[k]
                                    if test_dist >= 5.0:
                                        test_grade = (
                                            (smoothed_elevations[k + 1] - smoothed_elevations[k])
                                            / test_dist
                                        ) * 100
                                        if test_grade > outlier_cap_pct * 0.7:  # 70% of cap
                                            nearby_high_grades += 1

                            # If 2+ nearby segments also show high grades, it's likely real
                            if nearby_high_grades >= 2:
                                max_grade = max(max_grade, min(grade, outlier_cap_pct * 1.5))

                    # Only use first valid window for each starting point
                    break

        return max_grade

    def calculate_fiets_score(self, elevation_gain_m: float, distance_km: float) -> float:
        """Calculate FIETS difficulty index"""
        if distance_km <= 0:
            return 0
        return (elevation_gain_m**2) / (distance_km * 10)

    def calculate_pdi_score(
        self,
        road_segment: Dict,
        elevation_gain_m: float,
        distance_km: float,
        min_elev_m: float,
        max_elev_m: float,
    ) -> float:
        """Calculate PDI difficulty index with proper parameters"""

        # Extract data
        total_distance_m = distance_km * 1000

        # Calculate elevation loss from nodes
        elevation_loss_m = self.calculate_elevation_loss(road_segment["nodes"])

        # Work calculation (PDI formula)
        work = (8.6 * total_distance_m + 735 * (elevation_gain_m - 0.25 * elevation_loss_m)) / 1600

        # Elevation factor
        elevation_factor = 1 + (min_elev_m**2 + max_elev_m**2) / (7.2e6)

        # Surface factor (map your surface types to 0-5 scale)
        surface_index = self.map_surface_to_pdi_scale(road_segment["surface"])
        surface_factor = 1 + 0.2 * surface_index

        # Final PDI
        if total_distance_m > 0:
            pdi_score = elevation_factor * surface_factor * (work**2 / total_distance_m)
        else:
            pdi_score = 0

        return pdi_score

    def calculate_elevation_loss(self, nodes: List) -> float:
        """Calculate total elevation loss along the route."""
        if not nodes or len(nodes) < 2:
            return 0.0

        elevation_loss = 0.0

        for i in range(len(nodes) - 1):
            if hasattr(nodes[i], "id") and hasattr(nodes[i + 1], "id"):
                if nodes[i].id in self.node_elevations and nodes[i + 1].id in self.node_elevations:
                    elev_change = (
                        self.node_elevations[nodes[i + 1].id] - self.node_elevations[nodes[i].id]
                    )
                    if elev_change < 0:  # Negative change = elevation loss
                        elevation_loss += abs(elev_change)

        return elevation_loss

    def map_surface_to_pdi_scale(self, surface: str) -> float:
        """Map your surface types to PDI's 0-5 scale"""
        surface_mapping = {
            "asphalt": 0.0,
            "paved": 0.0,
            "concrete": 0.0,
            "gravel": 1.5,
            "compacted": 1.0,
            "dirt": 2.5,
            "unpaved": 2.0,
            "track": 2.0,
            "unknown": 1.0,  # Conservative middle ground
        }
        return surface_mapping.get(surface.lower(), 1.0)

    def _find_highest_point_index(self, elevations: List[float]) -> int:
        """
        Find index of the highest elevation point.

        Args:
            elevations: List of elevation values

        Returns:
            Index of the highest elevation point (first occurrence if tied)
        """
        if not elevations:
            return 0
        max_elev = max(elevations)
        return elevations.index(max_elev)

    def _split_segment_at_peak(
        self, road_segment: Dict, peak_idx: int, nodes: List, elevations: List[float]
    ) -> Tuple[Dict, Dict]:
        """
        Split a road segment at its highest elevation point into two climbs.

        Both resulting segments will include the peak node as their endpoint.
        Segment A goes from original start to peak, Segment B goes from original
        end to peak (reversed to be ascending).

        Args:
            road_segment: Original road segment dict
            peak_idx: Index of the highest elevation point in the valid nodes/elevations
            nodes: List of valid nodes (with elevations)
            elevations: List of corresponding elevations

        Returns:
            Tuple of (segment_a, segment_b) dicts, or (segment_a, None) if segment_b
            would be too short
        """
        way_ids = road_segment.get("way_ids", [])
        way_boundaries = road_segment.get("way_boundaries")
        street_name = road_segment.get("street_name", "Unknown")
        highway_type = road_segment.get("highway_type", "unknown")
        surface = road_segment.get("surface", "unknown")
        tracktype = road_segment.get("tracktype", "-")
        tracktype_definition = road_segment.get("tracktype_definition", "-")
        cycling_access = road_segment.get("cycling_access", "Unknown")

        # Determine way_ids for each split segment using way_boundaries
        # way_boundaries format: [(start_idx, end_idx, way_id), ...]
        if way_boundaries:
            # Segment A: way_ids that have nodes BEFORE the peak
            # A way starting exactly at peak only contributes the peak node, belongs to B
            way_ids_a = []
            for start_idx, end_idx, wid in way_boundaries:
                if start_idx < peak_idx:  # Way starts BEFORE peak (has nodes before peak)
                    way_ids_a.append(wid)
            # Segment B: way_ids that have nodes AFTER the peak
            # A way ending exactly at peak only contributes the peak node, belongs to A
            way_ids_b = []
            for start_idx, end_idx, wid in way_boundaries:
                if end_idx > peak_idx:  # Way ends AFTER peak (has nodes after peak)
                    way_ids_b.append(wid)
        else:
            # Fallback: use all way_ids for both segments (backward compatibility)
            way_ids_a = way_ids if way_ids else []
            way_ids_b = way_ids if way_ids else []

        # Segment A: from start to peak (inclusive)
        nodes_a = nodes[0 : peak_idx + 1]
        segment_a = {
            "street_name": street_name,
            "nodes": nodes_a,
            "way_ids": way_ids_a,
            "highway_type": highway_type,
            "surface": surface,
            "tracktype": tracktype,
            "tracktype_definition": tracktype_definition,
            "cycling_access": cycling_access,
            "_from_split": True,  # Flag to prevent re-splitting
        }

        # Segment B: from peak to end, then reversed (so it ascends to peak)
        nodes_b_original = nodes[peak_idx:]  # Includes peak node
        if len(nodes_b_original) < 2:
            # Segment B too short, only return segment A
            return segment_a, None

        # Reverse segment B so it goes from lowest point (original end) to peak
        nodes_b = list(reversed(nodes_b_original))
        segment_b = {
            "street_name": street_name,
            "nodes": nodes_b,
            "way_ids": way_ids_b,
            "highway_type": highway_type,
            "surface": surface,
            "tracktype": tracktype,
            "tracktype_definition": tracktype_definition,
            "cycling_access": cycling_access,
            "_from_split": True,  # Flag to prevent re-splitting
        }

        return segment_a, segment_b

    def calculate_climb_metrics(self, road_segment: Dict) -> List:
        """
        Calculate detailed climb metrics with node storage and selected score type.

        If the segment has its highest elevation point in the middle (not at start or end),
        it will be split into two climbs - one from each end ascending to the peak.

        Returns:
            List of ClimbMetrics objects (usually 1, but 2 if split at peak)
        """
        # DEBUG: Confirm new code is running
        if not hasattr(self, "_code_version_printed"):
            self._code_version_printed = True
            # print("\n[DEBUG] calculate_climb_metrics v2 (peak-splitting) is active\n")

        street_name = road_segment.get("street_name", "Unknown")
        nodes = road_segment.get("nodes", [])
        way_ids = road_segment.get("way_ids", [])
        highway_type = road_segment.get("highway_type", "unknown")
        surface = road_segment.get("surface", "unknown")
        tracktype = road_segment.get("tracktype", "-")
        from_split = road_segment.get("_from_split", False)

        if len(nodes) < 2:
            return []

        # Extract coordinates and elevations
        coordinates = []
        elevations = []
        valid_nodes = []
        found_count = 0
        missing_count = 0

        for i, node in enumerate(nodes):
            # Use coordinate-based lookup (works for merged segments with multiple way_ids)
            # This matches the elevation profile generation logic
            if isinstance(node, dict) and "lat" in node and "lon" in node:
                lat = round(node["lat"], 6)
                lon = round(node["lon"], 6)
                coord_key = f"coord_{lat}_{lon}"

                if coord_key in self.node_elevations:
                    coordinates.append((float(node["lat"]), float(node["lon"])))
                    elevations.append(self.node_elevations[coord_key])
                    valid_nodes.append(node)
                    found_count += 1
                else:
                    missing_count += 1
            elif hasattr(node, "lat") and hasattr(node, "lon"):
                lat = round(node.lat, 6)
                lon = round(node.lon, 6)
                coord_key = f"coord_{lat}_{lon}"

                if coord_key in self.node_elevations:
                    coordinates.append((float(node.lat), float(node.lon)))
                    elevations.append(self.node_elevations[coord_key])
                    valid_nodes.append(node)
                    found_count += 1
                else:
                    missing_count += 1

        if len(elevations) < 2:
            return []

        # Check if we should split at the highest point
        # Only split if: (1) not already from a split, and (2) highest point is in the middle
        if not from_split:
            peak_idx = self._find_highest_point_index(elevations)

            # Check if peak is in the middle (not at start or end)
            if peak_idx > 0 and peak_idx < len(elevations) - 1:
                # Split the segment at the peak
                segment_a, segment_b = self._split_segment_at_peak(
                    road_segment, peak_idx, valid_nodes, elevations
                )

                results = []
                # Recursively process both segments
                climb_a = self.calculate_climb_metrics(segment_a)
                if climb_a:
                    results.extend(climb_a)

                if segment_b:
                    climb_b = self.calculate_climb_metrics(segment_b)
                    if climb_b:
                        results.extend(climb_b)

                return results

        # CRITICAL: Ensure elevations go from LOW to HIGH (ascending)
        # If first elevation > last elevation, reverse arrays
        if elevations[0] > elevations[-1]:
            elevations = list(reversed(elevations))
            coordinates = list(reversed(coordinates))
            valid_nodes = list(reversed(valid_nodes))

        # Calculate distances between consecutive points
        distances = []
        total_distance = 0.0

        for i in range(len(coordinates) - 1):
            dist = self.calculate_distance(
                coordinates[i][0],
                coordinates[i][1],
                coordinates[i + 1][0],
                coordinates[i + 1][1],
            )
            distances.append(dist)
            total_distance += dist

        if total_distance == 0:
            return []

        # Calculate elevation metrics
        min_elevation = min(elevations)
        max_elevation = max(elevations)
        height = max_elevation  # Absolute maximum elevation (highest point)

        # Calculate elevation gain (sum of positive elevation changes)
        elevation_gain = 0.0

        for i in range(len(elevations) - 1):
            elev_change = elevations[i + 1] - elevations[i]
            if elev_change > 0:
                elevation_gain += elev_change

        # Calculate max grade using smoothed elevation data to eliminate precision errors
        # This uses a 25m smoothing window and 30m grade calculation window
        # with a 35% outlier cap (allowing up to 52.5% if consistent)
        max_grade = self.calculate_max_grade_smoothed(
            elevations=elevations,
            distances_km=distances,
            smoothing_window_m=25.0,
            grade_window_m=30.0,
            outlier_cap_pct=35.0,
        )

        # Calculate average grade using elevation_gain (not height) to avoid noise from max-min
        # This gives a more accurate representation of the actual climbing gradient
        if total_distance > 0:
            avg_grade = (elevation_gain / (total_distance * 1000)) * 100
            # Cap at 50% to match max_grade logic and avoid unrealistic values from very short segments
            avg_grade = min(avg_grade, 50.0)
        else:
            avg_grade = 0

        # Calculate climb scores based on selected score type
        basic_score = total_distance * 1000 * avg_grade if avg_grade > 0 else 0
        fiets_score = self.calculate_fiets_score(elevation_gain, total_distance)
        pdi_score = self.calculate_pdi_score(
            road_segment, elevation_gain, total_distance, min_elevation, max_elevation
        )

        # Select the appropriate score based on score_type
        if self.score_type == "fiets":
            climb_score = fiets_score
        elif self.score_type == "pdi":
            climb_score = pdi_score
        else:  # 'basic'
            climb_score = basic_score

        # Calculate prominence: elevation gain from start to highest point
        # This represents the "true" climb height from where you begin
        start_elevation = elevations[0]
        prominence = max_elevation - start_elevation

        # Distance between start and end points
        if len(coordinates) >= 2:
            distance_km = self.calculate_distance(
                coordinates[0][0],
                coordinates[0][1],
                coordinates[-1][0],
                coordinates[-1][1],
            )
        else:
            distance_km = total_distance

        # Determine climb category
        category = self.categorize_climb(avg_grade, total_distance * 1000, elevation_gain)

        cycling_access = road_segment.get("cycling_access", "Unknown")

        # Create OSM links
        osm_links = []
        for way_id in way_ids:
            osm_links.append(f"[{way_id}](https://www.openstreetmap.org/way/{way_id})")

        return [
            ClimbMetrics(
                street_name=street_name,
                climb_category=category,
                climb_score=climb_score,
                elevation_gain=elevation_gain,
                height=height,
                prominence=prominence,
                length_km=total_distance,
                distance_km=distance_km,
                avg_grade=avg_grade,
                max_grade=max_grade,
                min_elevation=min_elevation,
                max_elevation=max_elevation,
                highway_type=highway_type,
                surface=surface,
                tracktype=tracktype,
                tracktype_definition=road_segment.get("tracktype_definition", "-"),
                way_ids=way_ids,
                osm_links=osm_links,
                nodes=valid_nodes,  # Store the valid nodes (with elevations) for location lookup
                fiets_score=fiets_score,
                pdi_score=pdi_score,
                cycling_access=cycling_access,
                connected_climbs=[],  # Will be populated by _detect_connected_climbs()
                elevation_profile="",  # Will be populated during analysis if nodes exist
                start_lat=coordinates[0][0] if coordinates else 0.0,
                start_lon=coordinates[0][1] if coordinates else 0.0,
            )
        ]

    def categorize_climb(self, avg_grade: float, length_m: float, elevation_gain_m: float) -> str:
        """Categorize climb based on climb score (cycling categorization system)."""
        # Calculate climb score for categorization
        climb_score = length_m * avg_grade

        if climb_score > 80000:
            return "HC"  # Hors Catégorie (beyond categorization)
        elif climb_score > 64000:
            return "1"
        elif climb_score > 32000:
            return "2"
        elif climb_score > 16000:
            return "3"
        elif climb_score > 8000:
            return "4"
        else:
            return "N/A"  # Below category 4 threshold

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        # Safety check for None values
        if any(coord is None for coord in [lat1, lon1, lat2, lon2]):
            return 0.0

        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def print_climb_results(
        self,
        min_score: float = 0.0,
        unit_system: str = None,
        analysis_center: Tuple[float, float] = None,
        persistence=None,
        enable_geocoding: bool = True,
        scope_info: str = None,
    ):
        """Filter and sort by selected score type, show selected on screen, export all to XLS

        Args:
            scope_info: Optional string describing the analysis scope (e.g., "Luxembourg", "Boulder, CO", etc.)
        """

        # CONSOLIDATION: Always use streaming export for consistent memory usage (1-2GB)
        # Streaming works efficiently for all dataset sizes through optimized batch processing
        if hasattr(self, "climbs_temp_file") and self.climbs_temp_file:
            from climb_analyzer.utils.formatting import print_dim

            print(f"Streaming {self.climbs_count:,} climbs directly to Excel...")
            print_dim("   Memory-efficient streaming export (1-2GB peak usage)")

            # Use streaming export - writes directly to Excel without loading all climbs
            # Get additional parameters from analyzer instance
            surface_filter = getattr(self, "surface_filter", "all")
            score_type = getattr(self, "score_type", "basic")
            cycling_only = getattr(self, "cycling_only", False)

            return self._stream_to_excel(
                min_score=min_score,
                units=unit_system or self.unit_system,
                enable_geocoding=enable_geocoding,
                analysis_center=analysis_center,
                persistence=persistence,
                scope_info=scope_info,
                surface_filter=surface_filter,
                score_type=score_type,
                cycling_only=cycling_only,
            )

        # ============================================================================
        # OLD IN-MEMORY EXPORT PATH - COMMENTED OUT
        # Consolidated to streaming export only for consistent memory usage
        # Streaming handles all dataset sizes efficiently (1-2GB peak memory)
        # ============================================================================

        # If streaming didn't execute (no temp file), this is likely an error state
        # All region/country analyses should now use streaming mode
        print("⚠️ Warning: Streaming export not available (no temp file)")
        print("   This may indicate an issue with the analysis phase")
        return pd.DataFrame()

        # # OLD CODE - In-memory export (uses 1-20GB+ memory)
        # if not self.climbs:
        #     print("No climbs found")
        #     return pd.DataFrame()
        #
        # # Use provided unit system or fall back to instance unit system
        # units = unit_system or self.unit_system
        #
        # # Filter by the SELECTED score type (skip if already filtered during streaming load)
        # if hasattr(self, "climbs_temp_file") and self.climbs_temp_file:
        #     # Already filtered during load in streaming mode
        #     filtered_climbs = self.climbs
        # else:
        #     # Normal filtering for non-streaming mode
        #     if self.score_type == "fiets":
        #         filtered_climbs = [c for c in self.climbs if c.fiets_score >= min_score]
        #     elif self.score_type == "pdi":
        #         filtered_climbs = [c for c in self.climbs if c.pdi_score >= min_score]
        #     else:  # 'basic'
        #         filtered_climbs = [c for c in self.climbs if c.climb_score >= min_score]
        #
        # if not filtered_climbs:
        #     score_type_name = self.score_type.upper() if self.score_type != "basic" else "basic"
        #     print(f"No climbs found with {score_type_name} score >= {min_score}")
        #     return None
        #
        # # Sort by the SELECTED score type
        # if self.score_type == "fiets":
        #     sorted_climbs = sorted(filtered_climbs, key=lambda x: x.fiets_score, reverse=True)
        # elif self.score_type == "pdi":
        #     sorted_climbs = sorted(filtered_climbs, key=lambda x: x.pdi_score, reverse=True)
        # else:  # 'basic'
        #     sorted_climbs = sorted(filtered_climbs, key=lambda x: x.climb_score, reverse=True)
        #
        # # Use optimized geocoding
        # # For large datasets, use chunked geocoding with memory cleanup
        # location_data = []
        # if enable_geocoding and len(sorted_climbs) > 500000:
        #     print(f"   Using chunked geocoding for {len(sorted_climbs):,} climbs...")
        #     location_data = self._batch_geocode_climbs_chunked(
        #         sorted_climbs, analysis_center, persistence
        #     )
        # elif enable_geocoding and analysis_center and persistence:
        #     location_data = self.batch_reverse_geocode_climbs(
        #         sorted_climbs, analysis_center, persistence
        #     )
        # elif enable_geocoding and analysis_center:
        #     location_data = self._calculate_distances_only(sorted_climbs, analysis_center)
        # elif analysis_center:
        #     location_data = self._calculate_distances_only(sorted_climbs, analysis_center)
        # else:
        #     location_data = [
        #         {"city": "N/A", "state": "N/A", "country": "N/A", "distance_km": 0.0}
        #         for _ in sorted_climbs
        #     ]
        #
        # # Prepare terminal width for potential use
        # import shutil
        #
        # terminal_width = shutil.get_terminal_size((160, 20)).columns
        #
        # score_type_display = (
        #     self.score_type.upper() if self.score_type != "basic" else "BASIC (elevation*distance)"
        # )
        #
        # # Print simple summary instead of full results table
        # print(f"\n✓ Found {len(sorted_climbs):,} climbs (saving to file...)")
        #
        # # For very large datasets (>500K climbs), use chunked export
        # # if len(sorted_climbs) > 500000:
        #     print(
        #         f"   Large dataset ({len(sorted_climbs):,} climbs) - using memory-efficient export..."
        #     )
        #     # Return empty DataFrame - caller will handle chunked Excel export
        #     return self._export_large_dataset_chunked(
        #         sorted_climbs, location_data, units, min_score
        #     )
        #
        # # Prepare data for SCREEN display (only selected score) and XLS export (all scores)
        # screen_data = []
        # xls_data = []
        #
        # for i, climb in enumerate(sorted_climbs):
        #     # Convert units based on system
        #     if units == "imperial":
        #         elev_gain = climb.elevation_gain * 3.28084  # m to ft
        #         height = climb.height * 3.28084  # m to ft
        #         prominence = climb.prominence * 3.28084  # m to ft
        #         length = climb.length_km * 0.621371  # km to mi
        #         distance = climb.distance_km * 0.621371  # km to mi
        #         center_distance = location_data[i]["distance_km"] * 0.621371  # km to mi
        #         elev_unit = "ft"
        #         dist_unit = "mi"
        #     else:
        #         elev_gain = climb.elevation_gain
        #         height = climb.height
        #         prominence = climb.prominence
        #         length = climb.length_km
        #         distance = climb.distance_km
        #         center_distance = location_data[i]["distance_km"]
        #         elev_unit = "m"
        #         dist_unit = "km"
        #
        #     # Format way IDs and OSM links
        #     way_ids_str = str(climb.way_ids[0]) if climb.way_ids else "N/A"
        #     osm_links_str = climb.osm_links[0] if climb.osm_links else "N/A"
        #     # All way IDs for merged climbs (comma-separated)
        #     all_way_ids_str = ", ".join(str(wid) for wid in climb.way_ids) if climb.way_ids else "N/A"
        #
        #     # Choose score display based on score type
        #     if self.score_type == "fiets":
        #         score_display = f"{climb.fiets_score:.1f}"
        #         score_label = "FIETS Score"
        #     elif self.score_type == "pdi":
        #         score_display = f"{climb.pdi_score:.1f}"
        #         score_label = "PDI Score"
        #     else:  # basic
        #         score_display = str(int(climb.climb_score))
        #         score_label = "Basic Score"
        #
        #     # SCREEN data - only selected score (Connected Climbs suppressed from terminal)
        #     screen_row = {
        #         "Street Name": climb.street_name,
        #         "City": location_data[i]["city"],
        #         "State": location_data[i]["state"],
        #         f"From Center ({dist_unit})": round(center_distance, 1),
        #         "Cycling": climb.cycling_access,
        #         "Category": climb.climb_category,
        #         score_label: score_display,
        #         f"Elev Gain ({elev_unit})": int(elev_gain),
        #         f"Height ({elev_unit})": int(height),
        #         f"Prominence ({elev_unit})": int(prominence),
        #         f"Length ({dist_unit})": round(length, 2),
        #         "Avg Grade (%)": round(min(climb.avg_grade, 50.0), 2),
        #         "Max Grade (%)": round(climb.max_grade, 2),
        #         "Highway Type": climb.highway_type,
        #         "Surface": climb.surface,
        #         "Tracktype": climb.tracktype,
        #         "Way ID": way_ids_str,
        #         "OSM Link": osm_links_str,
        #     }
        #     screen_data.append(screen_row)
        #
        #     # XLS data - ALL three scores
        #     xls_row = {
        #         "Street Name": climb.street_name,
        #         "City": location_data[i]["city"],
        #         "State": location_data[i]["state"],
        #         "Country": location_data[i].get("country", "Unknown"),
        #         f"From Center ({dist_unit})": round(center_distance, 1),
        #         "Latitude": round(climb.start_lat, 5),
        #         "Longitude": round(climb.start_lon, 5),
        #         "Cycling": climb.cycling_access,
        #         "Category": climb.climb_category,
        #         "Basic Score": int(climb.climb_score),
        #         "FIETS Score": round(climb.fiets_score, 1),
        #         "PDI Score": round(climb.pdi_score, 1),
        #         f"Elev Gain ({elev_unit})": int(elev_gain),
        #         f"Height ({elev_unit})": int(height),
        #         f"Prominence ({elev_unit})": int(prominence),
        #         f"Length ({dist_unit})": round(length, 2),
        #         "Avg Grade (%)": round(min(climb.avg_grade, 50.0), 2),
        #         "Max Grade (%)": round(climb.max_grade, 2),
        #         "Highway Type": climb.highway_type,
        #         "Surface": climb.surface,
        #         "Tracktype": climb.tracktype,
        #         "Way ID": way_ids_str,
        #         "OSM Link": osm_links_str,
        #         "All Way IDs": all_way_ids_str,
        #         "Connected Climbs": (
        #             ", ".join(climb.connected_climbs) if climb.connected_climbs else "None"
        #         ),
        #         "Elevation Profile": getattr(climb, "elevation_profile", "") or "",
        #     }
        #     xls_data.append(xls_row)
        #
        # # Build screen DataFrame but don't print it (results saved to file only)
        # screen_df = pd.DataFrame(screen_data)
        #
        # # Print brief elevation statistics summary if there are failures
        # if self.total_coords_requested > 0 and self.total_coords_failed > 0:
        #     from climb_analyzer.utils.formatting import print_warning
        #
        #     failure_rate = (self.total_coords_failed / self.total_coords_requested) * 100
        #     if failure_rate > 10:
        #         print_warning(
        #             f"⚠️  High elevation fetch failure rate ({failure_rate:.1f}%) - review results carefully"
        #         )
        #
        # # Return XLS dataframe with all scores
        # return pd.DataFrame(xls_data)


def get_score_type_choice() -> str:
    """Get climb score type choice from user input."""
    from climb_analyzer.utils.formatting import print_header

    print_header("Climb Score Type", spacing_before=1)
    print("1. Basic (elevation gain × distance) - Traditional simple scoring")
    print("2. FIETS (elevation²/distance) - Dutch FIETS difficulty index")
    print("3. PDI (Performance & Difficulty Index) - Advanced multi-factor scoring")
    print()

    while True:
        choice = input("Enter your choice (1-3, default 3): ").strip() or "3"

        if choice == "1":
            return "basic"
        elif choice == "2":
            return "fiets"
        elif choice == "3":
            return "pdi"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def get_surface_filter_choice() -> str:
    """Get surface filter choice from user input."""
    from climb_analyzer.utils.formatting import print_header

    print_header("Surface Filter Options", spacing_before=1)
    print("1. Paved roads only (highways, primary, secondary roads)")
    print("2. Gravel roads (tracks with gravel surface)")
    print("3. Dirt trails/tracks (OSM track classification)")
    print("4. All road types")
    print()

    while True:
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            return "paved"
        elif choice == "2":
            return "gravel"
        elif choice == "3":
            return "dirt"
        elif choice == "4":
            return "all"
        elif not choice:
            print("Defaulting to paved roads.")
            return "paved"
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def get_unit_system_choice() -> str:
    """Get unit system choice from user input."""
    from climb_analyzer.utils.formatting import print_header

    print_header("Unit System", spacing_before=1)
    print("1. Metric (meters, kilometers)")
    print("2. Imperial (feet, miles)")
    print("3. Auto (region-native: imperial for US states, metric elsewhere)")
    print()

    while True:
        choice = input("Enter your choice (1-3, default auto): ").strip() or "3"
        if choice == "1":
            return "metric"
        elif choice == "2":
            return "imperial"
        elif choice == "3":
            return "auto"
        else:
            print("Invalid choice. Defaulting to auto (region-native).")
            return "auto"


def get_cycling_filter_choice() -> bool:
    """Get cycling filter choice from user input."""
    from climb_analyzer.utils.formatting import print_header

    print_header("Cycling Accessibility Filter", spacing_before=1)
    print("Y - Only include roads/trails accessible to cyclists")
    print("N - Include all roads regardless of cycling access")
    print()

    while True:
        choice = input("Filter to cycling-accessible routes only? (y/N): ").strip() or "N"

        if choice.lower() == "y" or choice.lower() == "yes":
            return True
        elif choice.lower() == "n" or choice.lower() == "no":
            return False
        else:
            print("Invalid choice. Please enter Y or N.")


# REMOVED: get_reverse_geocoding_choice() function
# Geocoding is now always enabled since it's fast with local offline libraries (reverse_geocoder)


def get_analysis_scope_choice(
    deployment_type: str = "cloud",
) -> Tuple[str, Union[str, List[str]], Optional[float]]:
    """Get analysis scope choice from user input - simplified two-option menu."""
    from climb_analyzer.utils.formatting import print_header
    from utils.geographic_menu import select_regions_hierarchical

    print_header("Analysis Scope", spacing_before=1)
    print("1. Address with radius (search near a specific location)")
    print("2. Select country or subregion (hierarchical region selection)")
    print()

    while True:
        choice = input("Enter your choice (1-2, or 0 to cancel): ").strip()

        if choice == "0":
            print("Analysis cancelled.")
            return "cancelled", [], None
        elif choice == "1":
            return "address", "", None
        elif choice == "2":
            # Check if cloud mode - regional analysis not supported
            if DEPLOYMENT_TYPE == "cloud":
                print("\n" + "=" * 70)
                print("❌ ERROR: Regional analysis not supported in cloud mode")
                print("=" * 70)
                print("\nRegional analysis (countries, states, provinces) requires")
                print("downloading entire region OSM data via Overpass API.")
                print("\nThis is not supported because:")
                print("  • Very large data transfers (hundreds of MB to GB)")
                print("  • API timeout limits (180 seconds max)")
                print("  • API memory limits (1-2 GB response size)")
                print("  • Risk of server-side errors and rate limiting")
                print("\n" + "─" * 70)
                print("RECOMMENDED SOLUTION: Switch to local mode")
                print("─" * 70)
                print("\nRun the setup command to download OSM files locally:")
                print("  $ ./climb-analyzer setup")
                print("\nLocal mode benefits:")
                print("  ✓ Analyze entire states/countries")
                print("  ✓ 3-5x faster processing")
                print("  ✓ No API rate limits")
                print("  ✓ Works offline")
                print("  ✓ Uses pre-built spatial indexes")
                print("=" * 70 + "\n")
                print("Returning to main menu...\n")
                continue

            # Use the hierarchical geographic menu system - go directly to continent selection
            selection_type, selection_data, _ = select_regions_hierarchical()

            if selection_type == "region" and selection_data:
                # selection_data is a list of (continent, path) tuples
                return "region", selection_data, None
            else:
                # No selection or cancelled - go back to main menu
                continue
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")


def calculate_state_bounds(
    state_name: str, country: str = "United States"
) -> Tuple[float, float, float]:
    """
    Calculate approximate bounds for a state.
    Returns (center_lat, center_lon, radius_km)

    Uses the consolidated osm_pbf_urls for lookups.
    """
    state_key = state_name.strip()

    # Use geo_lookup to find bounds (handles exact and partial matches)
    bounds = lookup_bounds(state_key)
    if bounds:
        lat_min, lon_min, lat_max, lon_max = bounds
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        # Calculate rough radius from bounding box
        lat_diff = lat_max - lat_min
        lon_diff = lon_max - lon_min
        radius = max(lat_diff, lon_diff) * 111  # Convert degrees to km (rough)
        print(f"\nFound match for '{state_name}'")
        return center_lat, center_lon, radius

    # State not found
    print(f"Warning: State/Province '{state_name}' not found in database.")

    # Default to center of continental US with large radius
    print("Using default coordinates (center of continental US)")
    return (39.0, -98.0, 400)


def calculate_country_bounds(country_name: str) -> Tuple[float, float, float]:
    """
    Calculate approximate bounds for a country.
    Returns (center_lat, center_lon, radius_km)

    Uses the consolidated osm_pbf_urls for lookups.
    """
    country_key = country_name.strip()

    # Use geo_lookup to find bounds (handles exact and partial matches)
    bounds = lookup_bounds(country_key)
    if bounds:
        lat_min, lon_min, lat_max, lon_max = bounds
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        # Calculate approximate radius in km
        lat_diff = abs(lat_max - lat_min)
        lon_diff = abs(lon_max - lon_min)
        radius = max(lat_diff, lon_diff) * 111.32  # Rough km per degree

        print(f"\nFound match for '{country_name}'")
        return center_lat, center_lon, radius

    # Country not found
    print(f"Warning: Country '{country_name}' not found in database.")

    # Default global center
    print("Using default coordinates (global center)")
    return (0.0, 0.0, 1000)


def deduplicate_segments(
    segments: List[Dict], persistence: ChunkPersistenceManager, auto_resume: bool = False
) -> List[Dict]:
    """Deduplication with checkpoint support for resumability.

    Args:
        segments: List of segments to deduplicate
        persistence: ChunkPersistenceManager for checkpointing
        auto_resume: If True, automatically resume from checkpoint without prompting
                     (used when already in a resumed analysis session)
    """

    print(f"Starting deduplication of {len(segments)} segments...")

    # Check if we should skip ALL deduplication
    try:
        from climb_analyzer.config import SKIP_ALL_DEDUPLICATION

        skip_all = SKIP_ALL_DEDUPLICATION
    except ImportError:
        skip_all = False

    if skip_all:
        print("\n⚠️  SKIPPING ALL DEDUPLICATION (performance optimization)")
        print("   Reason: SKIP_ALL_DEDUPLICATION = True in config")
        print("   Impact: Duplicates will be handled during street merging")
        print("   Benefit: Avoids potential memory issues with large datasets\n")

        # Clear any existing checkpoints
        try:
            persistence.clear_deduplication_progress()
        except:
            pass

        return segments  # Return all segments unchanged

    # Check for existing deduplication progress
    try:
        dedupe_checkpoint = persistence.load_deduplication_progress()
    except Exception as e:
        print(f"⚠️  Warning: Could not load deduplication checkpoint: {e}")
        print("  → Starting fresh deduplication...")
        dedupe_checkpoint = None

    if dedupe_checkpoint:
        print("Found existing deduplication progress:")
        print(
            f"  - Exact duplicate removal complete: {dedupe_checkpoint.get('step1_complete', False)}"
        )
        print(
            f"  - Step 2 progress: {dedupe_checkpoint.get('step2_progress', 0)} segments processed"
        )

        # If auto_resume is True, skip the prompt and resume automatically
        if auto_resume:
            print("  → Automatically resuming from checkpoint...")
            return _resume_deduplication(segments, dedupe_checkpoint, persistence)

        # Try to prompt, but handle errors gracefully
        try:
            resume_choice = input("Resume deduplication from checkpoint? (y/n): ").strip().lower()
            if resume_choice == "y":
                return _resume_deduplication(segments, dedupe_checkpoint, persistence)
            else:
                print("  → Starting fresh deduplication...")
        except (EOFError, OSError):
            # stdin not available or closed - auto-resume to be safe
            print(
                "  → Cannot prompt (stdin unavailable), automatically resuming from checkpoint..."
            )
            return _resume_deduplication(segments, dedupe_checkpoint, persistence)

    # Start fresh deduplication
    return _deduplicate(segments, persistence)


def _deduplicate(segments: List[Dict], persistence: ChunkPersistenceManager) -> List[Dict]:
    """Deduplication with smart checkpointing."""

    print("Removing exact way ID duplicates with smart checkpointing...")
    seen_way_sets = set()
    deduplicated = []

    # Initialize smart checkpointer
    checkpointer = SmartCheckpointer(len(segments), "Way ID Deduplication")

    with tqdm(
        total=len(segments),
        desc="Removing way ID duplicates",
        unit="segments",
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for i, segment in enumerate(segments):
            # Signal handler check every 1000 segments (reduced frequency for performance)
            if i % 1000 == 0:
                signal_handler.set_operation(
                    "deduplication",
                    {
                        "step1_in_progress": True,
                        "step1_segments_processed": i,
                        "step1_total_segments": len(segments),
                        "step1_kept": len(deduplicated),
                        # Don't copy the huge arrays - too slow!
                        # "original_segments": segments,
                        # "seen_way_sets": seen_way_sets,
                        # "deduplicated_so_far": deduplicated,
                        "timestamp": time.time(),
                    },
                )

            if signal_handler.kill_now:
                checkpoint_data = {
                    "step1_in_progress": True,
                    "step1_segments_processed": i,
                    "step1_total_segments": len(segments),
                    "step1_kept": len(deduplicated),
                    # DON'T save 1.5M segments - causes crash!
                    # "original_segments": segments,
                    # "seen_way_sets": seen_way_sets,
                    # "deduplicated_so_far": deduplicated,
                    "timestamp": time.time(),
                }
                persistence.save_deduplication_progress(checkpoint_data)
                print("Deduplication progress saved. Analysis can be resumed.")
                sys.exit(0)

            # Process the segment
            way_ids = tuple(sorted(segment.get("way_ids", [])))
            if way_ids and way_ids not in seen_way_sets:
                seen_way_sets.add(way_ids)
                deduplicated.append(segment)
            elif not way_ids:  # Keep segments without way_ids
                deduplicated.append(segment)

            # SMART CHECKPOINT CHECK - DISABLED FOR PERFORMANCE
            # Deduplication is fast enough that we don't need checkpointing
            # Saving 1.5M segments repeatedly is the bottleneck
            if False and checkpointer.should_checkpoint(i):
                checkpoint_data = {
                    "step1_in_progress": True,
                    "step1_segments_processed": i + 1,
                    "step1_total_segments": len(segments),
                    "step1_kept": len(deduplicated),
                    # Don't save the huge arrays - too slow!
                    # "original_segments": segments,
                    # "seen_way_sets": seen_way_sets,
                    # "deduplicated_so_far": deduplicated,
                    "timestamp": time.time(),
                }
                persistence.save_deduplication_progress(checkpoint_data)

                # Update progress bar with checkpoint info
                info = checkpointer.get_checkpoint_info(i)
                pbar.set_postfix(
                    {
                        "kept": len(deduplicated),
                        "progress": f"{info['progress_pct']:.1f}%",
                        "next_save": f"{info['time_until_next_min']:.1f}min",
                    }
                )
            else:
                pbar.set_postfix({"kept": len(deduplicated)})

            pbar.update(1)

    print(f"Exact duplicate removal complete: {len(segments)} -> {len(deduplicated)} segments")
    check_and_cleanup_memory(force_cleanup=True)

    # Check if we should skip spatial deduplication (performance optimization)
    try:
        from climb_analyzer.config import SKIP_SPATIAL_DEDUPLICATION, SPATIAL_DEDUP_THRESHOLD

        skip_spatial = SKIP_SPATIAL_DEDUPLICATION
        threshold = SPATIAL_DEDUP_THRESHOLD
    except ImportError:
        # Default: skip spatial dedup for large datasets
        skip_spatial = False
        threshold = 10000

    if skip_spatial or len(deduplicated) > threshold:
        print("\n⚠️  SKIPPING spatial deduplication (performance optimization)")
        print(
            f"   Reason: {'User configured' if skip_spatial else f'Dataset too large ({len(deduplicated):,} > {threshold:,} segments)'}"
        )
        print("   Impact: Minor - street merging will handle most overlaps")
        print("   Benefit: Saves hours/days of processing time\n")

        # Clear checkpoint and return
        persistence.clear_deduplication_progress()
        return deduplicated

    # Final checkpoint save - DON'T save the actual segment data (too large)
    checkpoint_data = {
        "step1_complete": True,
        # "step1_result": deduplicated,  # DON'T save 1.5M segments - causes crash!
        "original_count": len(segments),
        "step1_count": len(deduplicated),
        "timestamp": time.time(),
    }
    try:
        persistence.save_deduplication_progress(checkpoint_data)
        print("Deduplication checkpoint saved")
    except Exception as e:
        print(f"⚠️  Warning: Could not save deduplication checkpoint: {e}")
        print("  → Continuing without checkpoint (deduplication complete anyway)")

    # Proceed to spatial deduplication for small datasets
    if len(deduplicated) <= 1000:
        return _deduplicate_small_dataset(deduplicated, persistence)
    else:
        return _deduplicate_with_spatial_index(deduplicated, persistence)


def _resume_deduplication(
    segments: List[Dict], checkpoint: Dict, persistence: ChunkPersistenceManager
) -> List[Dict]:
    """Resume deduplication from checkpoint."""

    # NEW: Handle Step 1 interruption
    if checkpoint.get("step1_in_progress", False):
        print("Resuming from interrupted Step 1...")

        processed_count = checkpoint.get("step1_segments_processed", 0)
        total_count = checkpoint.get("step1_total_segments", len(segments))
        deduplicated_so_far = checkpoint.get("deduplicated_so_far", [])
        seen_way_sets = checkpoint.get("seen_way_sets", set())

        print(f"Exact deduplication progress: {processed_count}/{total_count} segments processed")
        print(f"Kept so far: {len(deduplicated_so_far)} segments")

        # Continue from where we left off
        checkpoint_interval = max(100, min(1000, len(segments) // 20))

        with tqdm(
            total=len(segments),
            desc="Resuming way ID deduplication",
            unit="segments",
            initial=processed_count,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for i in range(processed_count, len(segments)):
                # Signal handler check
                if i % 100 == 0:
                    signal_handler.set_operation(
                        "deduplication",
                        {
                            "step1_in_progress": True,
                            "step1_segments_processed": i,
                            "step1_total_segments": len(segments),
                            "step1_kept": len(deduplicated_so_far),
                            "original_segments": segments,
                            "seen_way_sets": seen_way_sets,
                            "deduplicated_so_far": deduplicated_so_far,
                            "timestamp": time.time(),
                        },
                    )

                if signal_handler.kill_now:
                    checkpoint_data = {
                        "step1_in_progress": True,
                        "step1_segments_processed": i,
                        "step1_total_segments": len(segments),
                        "step1_kept": len(deduplicated_so_far),
                        "original_segments": segments,
                        "seen_way_sets": seen_way_sets,
                        "deduplicated_so_far": deduplicated_so_far,
                        "timestamp": time.time(),
                    }
                    persistence.save_deduplication_progress(checkpoint_data)
                    print("Deduplication progress saved. Analysis can be resumed.")
                    sys.exit(0)

                segment = segments[i]
                way_ids = tuple(sorted(segment.get("way_ids", [])))
                if way_ids and way_ids not in seen_way_sets:
                    seen_way_sets.add(way_ids)
                    deduplicated_so_far.append(segment)
                elif not way_ids:  # Keep segments without way_ids
                    deduplicated_so_far.append(segment)

                # Periodic checkpoint save
                if i % checkpoint_interval == 0 and i > processed_count:
                    checkpoint_data = {
                        "step1_in_progress": True,
                        "step1_segments_processed": i,
                        "step1_total_segments": len(segments),
                        "step1_kept": len(deduplicated_so_far),
                        "original_segments": segments,
                        "seen_way_sets": seen_way_sets,
                        "deduplicated_so_far": deduplicated_so_far,
                        "timestamp": time.time(),
                    }
                    persistence.save_deduplication_progress(checkpoint_data)

                pbar.set_postfix({"kept": len(deduplicated_so_far)})
                pbar.update(1)

        print(
            f"Exact deduplication completed: {len(segments)} -> {len(deduplicated_so_far)} segments"
        )

        # Update checkpoint to mark Step 1 complete
        checkpoint_data = {
            "step1_complete": True,
            "step1_result": deduplicated_so_far,
            "original_count": len(segments),
            "step1_count": len(deduplicated_so_far),
            "timestamp": time.time(),
        }
        persistence.save_deduplication_progress(checkpoint_data)

        # Proceed to Step 2 with completed Step 1 data
        if len(deduplicated_so_far) <= 1000:
            return _deduplicate_small_dataset(deduplicated_so_far, persistence)
        else:
            return _deduplicate_with_spatial_index(deduplicated_so_far, persistence)

    # EXISTING: Handle Step 1 complete, Step 2 in progress
    elif checkpoint.get("step1_complete", False):
        print("Exact deduplication already complete, resuming Step 2...")
        deduplicated = checkpoint.get("step1_result", None)

        # If step1_result not in checkpoint (removed to save space), use original segments
        if deduplicated is None or len(deduplicated) == 0:
            print("  Step 1 result not in checkpoint (removed for space), using original segments")
            print("  Note: Will skip exact duplicate removal (already done in previous run)")
            deduplicated = segments

        if len(deduplicated) <= 1000:
            final_segments = _deduplicate_small_dataset(deduplicated, persistence, checkpoint)
        else:
            final_segments = _deduplicate_with_spatial_index(deduplicated, persistence, checkpoint)

        return final_segments

    # FALLBACK: Restart from beginning
    else:
        print("Resuming from exact deduplication (checkpoint unclear)...")
        return _deduplicate(segments, persistence)


def _deduplicate_small_dataset(
    segments: List[Dict], persistence: ChunkPersistenceManager, checkpoint: Dict = None
) -> List[Dict]:
    """Small dataset deduplication with checkpointing."""
    print("\nUsing standard deduplication (small dataset)")

    start_index = 0
    final_segments = []
    processed_indices = set()  # FIX: Initialize processed_indices
    duplicate_count = 0  # FIX: Initialize duplicate_count

    if checkpoint and "step2_progress" in checkpoint:
        start_index = checkpoint["step2_progress"]
        final_segments = checkpoint.get("step2_result", [])
        processed_indices = set(
            checkpoint.get("processed_indices", [])
        )  # FIX: Load from checkpoint
        duplicate_count = checkpoint.get("duplicate_count", 0)  # FIX: Load from checkpoint
        print(f"Resuming from segment {start_index}/{len(segments)}")

    checkpoint_interval = max(1, len(segments) // 20)  # Save every 5%

    with tqdm(
        total=len(segments),
        desc="Standard deduplication",
        unit="segments",
        initial=start_index,
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for i in range(start_index, len(segments)):
            if i in processed_indices:
                pbar.update(1)
                continue

            # Signal handler check every 100 segments
            if i % 100 == 0:
                signal_handler.set_operation(
                    "deduplication",
                    {
                        "step1_complete": True,
                        "step1_result": segments,
                        "step2_progress": i,
                        "step2_result": final_segments,
                        "processed_indices": list(processed_indices),
                        "duplicate_count": duplicate_count,
                        "timestamp": time.time(),
                    },
                )

            if signal_handler.kill_now:
                checkpoint_data = {
                    "step1_complete": True,
                    "step1_result": segments,
                    "step2_progress": i,
                    "step2_result": final_segments,
                    "processed_indices": list(processed_indices),
                    "duplicate_count": duplicate_count,
                    "timestamp": time.time(),
                }
                persistence.save_deduplication_progress(checkpoint_data)
                print("Deduplication progress saved. Analysis can be resumed.")
                sys.exit(0)

            segment = segments[i]
            is_duplicate = False

            # For small datasets, check against all previously processed segments
            # This is O(n²) but acceptable for small datasets
            for j in range(len(final_segments)):
                if _segments_are_similar(segment, final_segments[j]):
                    is_duplicate = True
                    duplicate_count += 1
                    break

            # Keep this segment if it's not a duplicate
            if not is_duplicate:
                final_segments.append(segment)

            processed_indices.add(i)

            # Save checkpoint periodically
            if i % checkpoint_interval == 0 or i == len(segments) - 1:
                checkpoint_data = {
                    "step1_complete": True,
                    "step1_result": segments,
                    "step2_progress": i + 1,
                    "step2_result": final_segments,
                    "processed_indices": list(processed_indices),
                    "duplicate_count": duplicate_count,
                    "timestamp": time.time(),
                }
                persistence.save_deduplication_progress(checkpoint_data)

            pbar.set_postfix({"kept": len(final_segments), "duplicates": duplicate_count})
            pbar.update(1)

    print(f"Standard deduplication complete: {len(segments)} -> {len(final_segments)} segments")
    return final_segments


def _deduplicate_with_spatial_index(
    segments: List[Dict], persistence: ChunkPersistenceManager, checkpoint: Dict = None
) -> List[Dict]:
    """Spatial index deduplication with smart checkpointing."""
    print("Using spatial indexing with smart checkpointing (large dataset)")

    # Calculate grid parameters and build spatial index (unchanged)
    grid_params = _calculate_grid_parameters(segments)
    spatial_index = _build_spatial_index(segments, grid_params)
    print(f"Built spatial index with {len(spatial_index)} occupied grid cells")

    start_index = 0
    final_segments = []
    processed_indices = set()
    duplicate_count = 0

    if checkpoint and "step2_progress" in checkpoint:
        start_index = checkpoint["step2_progress"]
        final_segments = checkpoint.get("step2_result", [])
        processed_indices = set(checkpoint.get("processed_indices", []))
        duplicate_count = checkpoint.get("duplicate_count", 0)
        print(f"Resuming spatial deduplication from segment {start_index}/{len(segments)}")

    # Initialize smart checkpointer
    checkpointer = SmartCheckpointer(len(segments) - start_index, "Spatial Deduplication")

    with tqdm(
        total=len(segments),
        desc="Spatial deduplication",
        unit="segments",
        initial=start_index,
        dynamic_ncols=True,
        ascii=" █",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for i in range(start_index, len(segments)):
            # Signal handler check every 100 segments
            if i % 100 == 0:
                signal_handler.set_operation(
                    "deduplication",
                    {
                        "step1_complete": True,
                        "step1_result": segments,
                        "step2_progress": i,
                        "step2_result": final_segments,
                        "processed_indices": list(processed_indices),
                        "duplicate_count": duplicate_count,
                        "timestamp": time.time(),
                    },
                )

            if signal_handler.kill_now:
                checkpoint_data = {
                    "step1_complete": True,
                    "step1_result": segments,
                    "step2_progress": i,
                    "step2_result": final_segments,
                    "processed_indices": list(processed_indices),
                    "duplicate_count": duplicate_count,
                    "timestamp": time.time(),
                }
                persistence.save_deduplication_progress(checkpoint_data)
                print("Deduplication progress saved. Analysis can be resumed.")
                sys.exit(0)

            # Process the segment (existing logic)
            segment = segments[i]
            candidates = _get_nearby_candidates(segment, spatial_index, grid_params)

            is_duplicate = False
            for candidate_idx in candidates:
                if candidate_idx in processed_indices or candidate_idx <= i:
                    continue

                candidate_segment = segments[candidate_idx]
                if _segments_are_similar(segment, candidate_segment):
                    processed_indices.add(candidate_idx)
                    duplicate_count += 1

            if not is_duplicate:
                final_segments.append(segment)

            processed_indices.add(i)

            # SMART CHECKPOINT CHECK
            progress_idx = i - start_index
            if checkpointer.should_checkpoint(progress_idx):
                checkpoint_data = {
                    "step1_complete": True,
                    "step1_result": segments,
                    "step2_progress": i + 1,
                    "step2_result": final_segments,
                    "processed_indices": list(processed_indices),
                    "duplicate_count": duplicate_count,
                    "grid_params": grid_params,
                    "timestamp": time.time(),
                }
                persistence.save_deduplication_progress(checkpoint_data)

                # Update progress bar with checkpoint info
                info = checkpointer.get_checkpoint_info(progress_idx)
                pbar.set_postfix(
                    {
                        "kept": len(final_segments),
                        "duplicates": duplicate_count,
                        "candidates": len(candidates),
                        "next_save": f"{info['time_until_next_min']:.1f}min",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "kept": len(final_segments),
                        "duplicates": duplicate_count,
                        "candidates_avg": len(candidates),
                    }
                )

            pbar.update(1)

    print(f"Spatial deduplication complete: {len(segments)} -> {len(final_segments)} segments")
    return final_segments


def _calculate_grid_parameters(segments: List[Dict]) -> Dict:
    """Calculate grid parameters for large datasets."""

    # For massive datasets like yours, sample much fewer segments
    sample_size = min(500, len(segments))
    step = max(1, len(segments) // sample_size)
    sample_segments = segments[::step]

    all_coords = []
    for segment in sample_segments[:sample_size]:
        start_coord, end_coord = _get_segment_endpoints(segment)
        if start_coord:
            all_coords.append(start_coord)
        if end_coord and end_coord != start_coord:
            all_coords.append(end_coord)

    if not all_coords:
        return {
            "min_lat": 0,
            "max_lat": 1,
            "min_lon": 0,
            "max_lon": 1,
            "cells_lat": 100,
            "cells_lon": 100,
            "cell_size_deg": 0.01,
        }

    lats = [coord[0] for coord in all_coords]
    lons = [coord[1] for coord in all_coords]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    # For 870k segments, use MANY small cells (target 100-150 segments per cell)
    target_segments_per_cell = 125
    estimated_cells_needed = len(segments) // target_segments_per_cell

    # Calculate grid dimensions
    cells_per_side = int(math.sqrt(estimated_cells_needed))
    cells_lat = cells_lon = max(100, min(300, cells_per_side))  # Much finer grid

    cell_size_lat = lat_span / cells_lat if lat_span > 0 else 0.01
    cell_size_lon = lon_span / cells_lon if lon_span > 0 else 0.01
    cell_size_deg = max(cell_size_lat, cell_size_lon)

    return {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "cells_lat": cells_lat,
        "cells_lon": cells_lon,
        "cell_size_deg": cell_size_deg,
    }


def _segments_are_similar(seg1: Dict, seg2: Dict, tolerance_deg: float = 0.0001) -> bool:
    """Similarity check using coordinate differences."""

    start1, end1 = _get_segment_endpoints(seg1)
    start2, end2 = _get_segment_endpoints(seg2)

    if not all([start1, end1, start2, end2]):
        return False

    # Fast coordinate difference check (no expensive distance calculations)
    return (
        abs(start1[0] - start2[0]) < tolerance_deg
        and abs(start1[1] - start2[1]) < tolerance_deg
        and abs(end1[0] - end2[0]) < tolerance_deg
        and abs(end1[1] - end2[1]) < tolerance_deg
    )


def _build_spatial_index(segments: List[Dict], grid_params: Dict) -> Dict:
    """Build spatial index mapping grid cells to segment indices."""

    spatial_index = defaultdict(list)

    for i, segment in enumerate(segments):
        start_coord, end_coord = _get_segment_endpoints(segment)

        if not start_coord or not end_coord:
            # Put segments without coordinates in cell (0,0)
            spatial_index[(0, 0)].append(i)
            continue

        # Get all grid cells this segment spans
        cells = _get_segment_cells(start_coord, end_coord, grid_params)

        for cell in cells:
            spatial_index[cell].append(i)

    return dict(spatial_index)


def _get_segment_cells(
    start_coord: Tuple[float, float], end_coord: Tuple[float, float], grid_params: Dict
) -> Set[Tuple[int, int]]:
    """Get all grid cells that a segment spans."""

    # Get cells for both endpoints
    start_cell = _coord_to_cell(start_coord, grid_params)
    end_cell = _coord_to_cell(end_coord, grid_params)

    cells = {start_cell, end_cell}

    # For segments spanning multiple cells, add intermediate cells
    if start_cell != end_cell:
        min_row = min(start_cell[0], end_cell[0])
        max_row = max(start_cell[0], end_cell[0])
        min_col = min(start_cell[1], end_cell[1])
        max_col = max(start_cell[1], end_cell[1])

        # Add all cells in the bounding box (conservative approach)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cells.add((row, col))

    return cells


def _coord_to_cell(coord: Tuple[float, float], grid_params: Dict) -> Tuple[int, int]:
    """Convert coordinate to grid cell."""

    lat, lon = coord

    # Calculate cell indices
    lat_cell = int((lat - grid_params["min_lat"]) / grid_params["cell_size_deg"])
    lon_cell = int((lon - grid_params["min_lon"]) / grid_params["cell_size_deg"])

    # Clamp to valid range
    lat_cell = max(0, min(grid_params["cells_lat"] - 1, lat_cell))
    lon_cell = max(0, min(grid_params["cells_lon"] - 1, lon_cell))

    return (lat_cell, lon_cell)


def _get_nearby_candidates(segment: Dict, spatial_index: Dict, grid_params: Dict) -> List[int]:
    """Get candidate segments from nearby grid cells with strict limits."""

    start_coord, end_coord = _get_segment_endpoints(segment)

    if not start_coord or not end_coord:
        return []

    # Get cells this segment spans (direct cells only for large datasets)
    segment_cells = _get_segment_cells(start_coord, end_coord, grid_params)

    # Collect candidates from direct cells only - NO adjacent cells for massive datasets
    candidates = []
    max_candidates = 200  # Hard limit

    for cell in segment_cells:
        cell_candidates = spatial_index.get(cell, [])
        candidates.extend(cell_candidates)

        # Stop early if we have enough candidates
        if len(candidates) >= max_candidates:
            candidates = candidates[:max_candidates]
            break

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    return unique_candidates


def _get_segment_endpoints(
    segment: Dict,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Extract start and end coordinates from a segment."""

    nodes = segment.get("nodes", [])
    if not nodes:
        return None, None

    start_coord = extract_node_coordinates(nodes[0])
    end_coord = extract_node_coordinates(nodes[-1])

    return start_coord, end_coord


def extract_node_coordinates(node):
    """Extract coordinates from a node object."""
    try:
        if hasattr(node, "lat") and hasattr(node, "lon"):
            lat = float(node.lat)
            lon = float(node.lon)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance using Haversine formula (in km)."""
    # Safety check for None values
    if any(coord is None for coord in [lat1, lon1, lat2, lon2]):
        return 0.0

    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1_rad) * math.cos(
        lat2_rad
    ) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def determine_cycling_access(tags: dict, highway_type: str = None) -> str:
    """
    Determine cycling access based on OSM tags and highway type.
    Uses reasonable assumptions about cycling legality on public roads.
    Returns: 'Yes', 'No', 'Unknown', or 'Limited'
    """
    if not tags:
        return "Unknown"

    highway = highway_type or tags.get("highway", "").strip().lower()
    bicycle = tags.get("bicycle", "").strip().lower()
    access = tags.get("access", "").strip().lower()

    # Check explicit bicycle restrictions first
    if bicycle in ["no", "private"]:
        return "No"
    elif bicycle in ["yes", "designated", "permissive"]:
        return "Yes"
    elif bicycle in ["dismount"]:
        return "Limited"

    # Check general access restrictions
    if access in ["no", "private"]:
        return "No"
    elif access in ["customers", "delivery", "permit"]:
        return "Limited"

    # Roads that are clearly cycling-friendly
    cycling_friendly_highways = {
        "cycleway": "Yes",
        "path": "Yes",  # Multi-use paths
        "bridleway": "Yes",  # Horse paths typically allow cycling
        "track": "Yes",  # Farm tracks, forest roads
        "residential": "Yes",  # Local streets
        "unclassified": "Yes",  # Minor public roads
        "tertiary": "Yes",  # Local connecting roads
        "secondary": "Yes",  # Regional roads
        "primary": "Yes",  # Major roads (legal but not always pleasant)
        "service": "Yes",  # Driveways, parking areas
        "living_street": "Yes",  # Shared space streets
        "road": "Yes",  # Generic road type
        "minor": "Yes",  # Minor roads
        "trunk": "Limited",  # Major roads - legal but often not ideal
    }

    # Roads that typically prohibit cycling
    prohibited_highways = {
        "motorway": "No",
        "motorway_link": "No",
        "steps": "No",
        "corridor": "No",  # Indoor corridors
        "elevator": "No",  # Elevators
    }

    # Special handling for footways
    if highway == "footway":
        # Footways allow cycling if explicitly marked, otherwise limited/unknown
        if bicycle in ["yes", "designated", "permissive"]:
            return "Yes"
        else:
            return "Limited"  # Pedestrian priority, cycling may be restricted

    # Apply highway-based defaults
    if highway in cycling_friendly_highways:
        return cycling_friendly_highways[highway]
    elif highway in prohibited_highways:
        return prohibited_highways[highway]

    # For unknown highway types, make reasonable assumptions
    # Most public roads allow cycling unless explicitly restricted
    if highway and highway not in ["", "unknown", None]:
        # If it's some kind of road/way we don't recognize, assume cycling is allowed
        # This covers regional highway types and new OSM classifications
        return "Yes"

    return "Unknown"


def batch_mode_menu():
    """Interactive menu for batch mode configuration."""
    print("\n" + "=" * 70)
    print("  BATCH MODE CONFIGURATION")
    print("=" * 70)

    # Step 1: Ask if batching US states or countries
    print("\n1. What would you like to batch analyze?")
    print("   a. US States")
    print("   b. Countries")

    while True:
        batch_type = input("\nEnter your choice (a/b): ").strip().lower()
        if batch_type in ["a", "b"]:
            break
        print("Invalid choice. Please enter 'a' or 'b'.")

    # Step 2: Get the list of states or countries
    if batch_type == "a":
        print("\n2. Enter US states to analyze (comma-separated)")
        print("   Examples: Colorado, California, Utah")
        print("             Vermont, New Hampshire, Maine")
        locations_input = input("\nStates: ").strip()
        scope_type = "region"
    else:
        print("\n2. Enter countries to analyze (comma-separated)")
        print("   Examples: Switzerland, Austria, Italy")
        print("             Japan, South Korea")
        locations_input = input("\nCountries: ").strip()
        scope_type = "country"

    if not locations_input:
        print("No locations entered. Exiting batch mode.")
        sys.exit(0)

    # Parse comma-separated locations
    locations = [loc.strip() for loc in locations_input.split(",") if loc.strip()]

    if not locations:
        print("No valid locations entered. Exiting batch mode.")
        sys.exit(0)

    print(f"\nLocations to analyze: {', '.join(locations)}")

    # Step 3: Ask about cleanup after each run
    print("\n3. Delete OSM map/index and elevation data after each location?")
    print("   This saves disk space but requires re-downloading for future runs.")
    cleanup_choice = input("   Delete after each run? (y/n, default y): ").strip().lower() or "y"
    cleanup_after = cleanup_choice in ["y", "yes"]

    print(f"\n   Cleanup after each run: {'Yes' if cleanup_after else 'No'}")

    return scope_type, locations, cleanup_after


def run_batch_subprocess(location, scope_type, args, cleanup_after):
    """
    Run a single state/country analysis as a subprocess.
    This ensures tqdm progress bars work properly and each run is independent.
    """
    import subprocess

    # Build command line arguments
    cmd = [sys.executable, __file__]

    # Add batch mode flag to prevent interactive prompts in subprocess
    cmd.append("--batch")

    # Skip storage prompt in subprocess (parent already asked)
    cmd.append("--skip-storage-prompt")

    # Add all the configuration arguments
    cmd.extend(["--scope", scope_type])
    cmd.extend(["--surface-filter", args.surface_filter])
    cmd.extend(["--units", args.units])
    cmd.extend(["--score-type", args.score_type])
    # Cycling filter - use the modern --cycling-filter flag
    if args.cycling_filter:
        cmd.append("--cycling-filter")

    if args.min_score:
        cmd.extend(["--min-score", str(args.min_score)])

    # Add the specific location
    if scope_type == "region":
        cmd.extend(["--states", location])
    else:
        cmd.extend(["--countries", location])

    # Run the subprocess
    from climb_analyzer.utils.formatting import print_header

    print_header(f"Starting Analysis: {location}", width=80, spacing_before=1)

    try:
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            # Check if output files were created despite non-zero exit code
            from pathlib import Path

            output_files = list(Path("output").glob(f"*{location}*"))

            if output_files:
                # Files were created - analysis succeeded despite exit code
                print(f"\n✓ Completed analysis for {location}")
                print(
                    f"   (Note: Subprocess exited with code {result.returncode} but output files were created)"
                )
                # Cleanup if requested
                if cleanup_after:
                    print(f"\nCleaning up data for {location}...")
                    from utils.batch_cleanup import get_cleanup_targets, perform_cleanup

                    cleanup_targets, _ = get_cleanup_targets()
                    if cleanup_targets:
                        perform_cleanup(cleanup_targets, verbose=True)
                        print(f"✓ Cleanup completed for {location}")
                return 0  # SUCCESS - files exist
            else:
                # No files - real failure
                print(f"\n⚠️  Warning: Analysis for {location} exited with code {result.returncode}")
                return result.returncode  # FAILURE - no files
        else:
            print(f"\n✓ Completed analysis for {location}")

        # Cleanup if requested
        if cleanup_after:
            print(f"\nCleaning up data for {location}...")
            from utils.batch_cleanup import get_cleanup_targets, perform_cleanup

            cleanup_targets, _ = get_cleanup_targets()
            if cleanup_targets:
                perform_cleanup(cleanup_targets, verbose=True)
                print(f"✓ Cleanup completed for {location}")

        return 0  # SUCCESS

    except Exception as e:
        print(f"\n❌ Error running analysis for {location}: {e}")
        return 1


def show_extended_help():
    """Show extended help with comprehensive examples."""
    print(
        """
CLIMB ANALYZER - EXTENDED HELP
===============================

0. WEB GUI MODE (RECOMMENDED FOR NEW USERS)
   Launch web-based graphical interface:
   $ python climb_analyzer.py -g

   The GUI will start on http://localhost:3000 with:
   - Dashboard overview
   - Visual analysis configuration
   - Interactive map visualization
   - Data download manager
   - System configuration

1. ADDRESS ANALYSIS
   Analyze climbs within 25 miles of Boulder, CO:
   $ python climb_analyzer.py -a "Boulder, CO" --distance 25

   With custom settings:
   $ python climb_analyzer.py -a "Boulder, CO" --distance 25 -u metric -t fiets -s paved -m 250

2. SINGLE REGION ANALYSIS
   Analyze entire state of Colorado:
   $ python climb_analyzer.py -r Colorado

   With abbreviations:
   $ python climb_analyzer.py -r CO -s paved -u imperial

3. BATCH MULTI-REGION ANALYSIS
   Analyze multiple New England states:
   $ python climb_analyzer.py -r "Vermont,New Hampshire,Maine"

   All New England states with abbreviations:
   $ python climb_analyzer.py -r "VT,NH,ME,MA,CT,RI"

   With cleanup after each region:
   $ python climb_analyzer.py -r "VT,NH,ME,MA,CT,RI" -X

   European countries:
   $ python climb_analyzer.py -r "Switzerland,Austria,Italy" -u metric -t fiets

4. DATA MANAGEMENT
   a. Download data without analysis:
      $ python climb_analyzer.py -D -r Vermont

   b. Update geographic boundaries:
      $ python climb_analyzer.py -U

   c. Delete checkpoints:
      $ python climb_analyzer.py -C

   d. Delete OSM data:
      $ python climb_analyzer.py -P

   e. Delete elevation data:
      $ python climb_analyzer.py -E

   f. Delete all data:
      $ python climb_analyzer.py -A

5. ADVANCED COMBINATIONS
   a. Gravel climbs only, FIETS scoring:
      $ python climb_analyzer.py -r Vermont -s gravel -t fiets -m 200 -u metric

   b. Cycling-accessible roads only:
      $ python climb_analyzer.py -r Colorado --cycling-filter

   c. All roads (default, cycling filter disabled):
      $ python climb_analyzer.py -r Colorado

6. INTERACTIVE MODE
   Launch interactive menus:
   $ python climb_analyzer.py -i

   Or just (default):
   $ python climb_analyzer.py

For more documentation, see:
  • CLI_ARGUMENTS_SPEC.md
  • DOCKER_SETUP.md
  • INSTALLATION.md
"""
    )


def launch_gui():
    """Launch the web-based GUI interface."""
    import subprocess
    from pathlib import Path

    from climb_analyzer.utils.formatting import (
        print_banner,
        print_error,
        print_info,
        print_success,
        print_warning,
    )

    print_banner("Starting Web GUI", spacing_before=1)

    # Check if GUI directory exists
    gui_dir = Path(__file__).parent / "gui"
    if not gui_dir.exists():
        print_error("GUI directory not found at: gui/")
        print_info("The web interface may not be installed.", indent=2)
        sys.exit(1)

    # Check for Node.js
    try:
        node_check = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=False
        )
        if node_check.returncode != 0:
            print_error("Node.js not found")
            print_info("Please install Node.js 18+ to use the web GUI", indent=2)
            print_info("Visit: https://nodejs.org/", indent=2)
            sys.exit(1)

        node_version = node_check.stdout.strip()
        print_success(f"Node.js detected: {node_version}")
    except FileNotFoundError:
        print_error("Node.js not found")
        print_info("Please install Node.js 18+ to use the web GUI", indent=2)
        print_info("Visit: https://nodejs.org/", indent=2)
        sys.exit(1)

    # Check if node_modules exists, if not run npm install
    node_modules = gui_dir / "node_modules"
    if not node_modules.exists():
        print_info("First time setup - installing dependencies...")
        print_info("This may take a few minutes...", indent=2)
        result = subprocess.run(["npm", "install"], cwd=gui_dir, check=False)
        if result.returncode != 0:
            print_error("Failed to install dependencies")
            sys.exit(1)
        print_success("Dependencies installed successfully")
        print()

    # Start the dev server in background
    print_info("Starting Next.js development server in background...")
    print()

    try:
        # Run npm run dev in background with nohup
        log_file = gui_dir / "gui-server.log"
        pid_file = gui_dir / "gui-server.pid"

        with open(log_file, "w") as log:
            process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=gui_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        # Save PID for later shutdown
        with open(pid_file, "w") as f:
            f.write(str(process.pid))

        # Wait for server to be ready by checking logs
        import time

        print_info("Waiting for server to be ready...")

        max_wait = 30  # Maximum 30 seconds
        start_time = time.time()
        server_ready = False

        while time.time() - start_time < max_wait:
            # Check if process is still running
            if process.poll() is not None:
                print_error("GUI server process died")
                print_info(f"Check logs: {log_file}")
                sys.exit(1)

            # Check logs for "Ready" message
            try:
                with open(log_file) as f:
                    log_content = f.read()
                    if "Ready in" in log_content or "ready -" in log_content.lower():
                        server_ready = True
                        break
            except:
                pass

            time.sleep(0.5)

        if server_ready:
            print_success("✓ GUI server is ready!")
            print()
            print_info("GUI available at:")
            print_info("  http://localhost:3000              # From this machine", indent=2)
            print_info("  http://<your-server-ip>:3000      # From network", indent=2)
            print()
            print_info(f"Server logs: {log_file}")
            print()
        else:
            print_warning("Server started but not ready yet (still initializing)")
            print()
            print_info(f"Check status: tail -f {log_file}")
            print()

        # Keep the container alive by waiting for the subprocess
        # This ensures Docker port mappings stay active
        print_info("GUI server running... (Press Ctrl+C to stop)")
        print()
        try:
            process.wait()
        except KeyboardInterrupt:
            print()
            print_info("Stopping GUI server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            # Clean up PID file
            if pid_file.exists():
                pid_file.unlink()

            print_success("GUI server stopped")

    except KeyboardInterrupt:
        print()
        print_info("Interrupted - GUI server may still be running")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error starting GUI: {e}")
        sys.exit(1)


def run_update_geo_boundaries():
    """Update geographic boundary data."""
    import subprocess

    print("\n🔄 Updating geographic boundaries...")
    result = subprocess.run(["python3", "utils/update_geo_definitions.py"])
    if result.returncode == 0:
        print("✓ Geographic boundaries updated successfully")
    else:
        print("❌ Update failed")
        sys.exit(1)


def delete_checkpoints(skip_confirm=False):
    """Delete all checkpoint files."""
    import shutil

    if not skip_confirm:
        confirm = input("\n⚠️  Delete all checkpoint files? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    # Delete checkpoint directory
    checkpoint_dir = CHECKPOINT_DIR
    try:
        if checkpoint_dir.exists():
            try:
                shutil.rmtree(checkpoint_dir)
                print("✓ Deleted all checkpoints")
            except OSError as e:
                # Handle "Device or resource busy" - delete contents instead
                deleted = 0
                failed = 0
                items = list(checkpoint_dir.iterdir())
                for item in items:
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        deleted += 1
                    except OSError:
                        failed += 1
                if deleted > 0:
                    print(f"✓ Deleted {deleted} checkpoint items")
                if failed > 0:
                    print(f"⚠️  {failed} items could not be deleted (in use)")
                if deleted == 0 and failed == 0:
                    print("✓ Checkpoint directory is empty (nothing to delete)")
        else:
            print("   No checkpoint directory found")
    except Exception as e:
        print(f"⚠️  Error accessing checkpoint directory: {e}")


def delete_planet_data(skip_confirm=False):
    """Delete all OSM planet data."""
    import shutil

    if not skip_confirm:
        confirm = input("\n⚠️  Delete all OSM .pbf files and indices? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    planet_dir = PLANET_OSM_DIR
    if not planet_dir.exists():
        print("   No OSM planet data directory found")
        return

    deleted_count = 0
    failed_count = 0
    for pbf_file in list(planet_dir.glob("*.pbf")):
        try:
            pbf_file.unlink()
            deleted_count += 1

            # Delete associated index files
            idx_file = pbf_file.with_suffix(".pbf.idx")
            if idx_file.exists():
                idx_file.unlink()

            # Delete rtree indices
            idx_dir = pbf_file.with_suffix("")
            if idx_dir.is_dir():
                shutil.rmtree(idx_dir)
        except OSError:
            failed_count += 1

    if deleted_count > 0:
        print(f"✓ Deleted {deleted_count} OSM files and indices")
    if failed_count > 0:
        print(f"⚠️  {failed_count} files could not be deleted (in use)")
    if deleted_count == 0 and failed_count == 0:
        print("✓ OSM planet data directory is empty (nothing to delete)")


def _cleanup_elevation_configs():
    """Clean opentopodata-config.yaml and config.yaml after elevation deletion."""
    from pathlib import Path

    # 1. Reset opentopodata-config.yaml to minimal test config
    opentopodata_config = Path("opentopodata-config.yaml")
    if opentopodata_config.exists():
        opentopodata_config.write_text("""datasets:
- name: test-dataset
  path: /app/tests/data/datasets/test-etopo1-resampled-1deg/
  filename_epsg: 4326
  filename_tile_size: 1
max_locations_per_request: 500
access_control_allow_origin: '*'
""")
        print("   ✓ Reset opentopodata-config.yaml to minimal config")

    # 2. Clear elevation entries from project config.yaml
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            if "ELEVATION_DATASETS" in config or "ELEVATION_DATA" in config:
                config["ELEVATION_DATASETS"] = {}
                config["ELEVATION_DATA"] = []
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                print("   ✓ Cleared ELEVATION_DATASETS and ELEVATION_DATA from config.yaml")
        except Exception as e:
            print(f"   ⚠️  Could not update config.yaml: {e}")


def delete_elevation_data(skip_confirm=False):
    """Delete all elevation data."""
    import shutil
    from pathlib import Path

    if not skip_confirm:
        confirm = input("\n⚠️  Delete all elevation dataset files? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    # Import centralized path
    try:
        from utils.data_paths import ELEVATION_DATA_DIR

        elevation_dir = ELEVATION_DATA_DIR
    except ImportError:
        elevation_dir = Path("data/elevation_data")

    if elevation_dir.exists():
        try:
            shutil.rmtree(elevation_dir)
            elevation_dir.mkdir(parents=True)
            print("✓ Deleted all elevation data")
        except OSError:
            # Handle "Device or resource busy" - delete contents instead
            deleted = 0
            failed = 0
            items = list(elevation_dir.iterdir())
            for item in items:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted += 1
                except OSError:
                    failed += 1
            if deleted > 0:
                print(f"✓ Deleted {deleted} elevation data items")
            if failed > 0:
                print(f"⚠️  {failed} items could not be deleted (in use)")
            if deleted == 0 and failed == 0:
                print("✓ Elevation data directory is empty (nothing to delete)")
    else:
        print("   No elevation data directory found")

    # Clean up config files after elevation data deletion
    _cleanup_elevation_configs()


def delete_all_data(skip_confirm=False):
    """Delete all data: checkpoints + OSM + elevation."""
    if not skip_confirm:
        print("\n⚠️  WARNING: This will delete ALL data:")
        print("  • Checkpoint files")
        print("  • OSM .pbf files and indices")
        print("  • Elevation dataset files")
        confirm = input("\nContinue? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    delete_checkpoints(skip_confirm=True)
    delete_planet_data(skip_confirm=True)
    delete_elevation_data(skip_confirm=True)
    print("\n✓ All data deleted successfully")


def delete_unavailable_cache(dataset=None, skip_confirm=False):
    """Delete .unavailable files that cache tiles marked as unavailable.

    This is useful when tiles were incorrectly marked as unavailable due to
    authentication issues or server problems.

    Args:
        dataset: If specified, only delete for this dataset (e.g., 'srtm30m', 'ned10m')
        skip_confirm: If True, skip confirmation prompt
    """
    from pathlib import Path

    elevation_data_dir = Path("data/elevation_data")
    if not elevation_data_dir.exists():
        print("No elevation data directory found")
        return

    # Map dataset names to directories
    dataset_dirs = {
        'srtm30m': 'srtm30m',
        'ned10m': 'ned10m',
        'aster30m': 'aster',  # Directory is 'aster', dataset is 'aster30m'
        'aw3d30': 'aw3d30',
        'arctic32m': 'arcticdem',
        'rema32m': 'rema',
    }

    # Find .unavailable files
    unavailable_files = []
    if dataset:
        # Specific dataset
        dir_name = dataset_dirs.get(dataset.lower(), dataset.lower())
        unavailable_file = elevation_data_dir / dir_name / ".unavailable"
        if unavailable_file.exists():
            unavailable_files.append(unavailable_file)
        else:
            print(f"No .unavailable file found for dataset: {dataset}")
            return
    else:
        # All datasets
        for subdir in elevation_data_dir.iterdir():
            if subdir.is_dir():
                unavailable_file = subdir / ".unavailable"
                if unavailable_file.exists():
                    unavailable_files.append(unavailable_file)

    if not unavailable_files:
        print("No .unavailable files found")
        return

    # Show what will be deleted
    print("\n  Unavailable cache files found:")
    total_tiles = 0
    for f in unavailable_files:
        try:
            with open(f) as fp:
                tiles = [line.strip() for line in fp if line.strip()]
                total_tiles += len(tiles)
                print(f"  • {f.parent.name}: {len(tiles)} tiles")
        except Exception:
            print(f"  • {f.parent.name}: (could not read)")

    if not skip_confirm:
        print(f"\n⚠️  This will allow {total_tiles} tiles to be re-attempted on next download")
        confirm = input("Continue? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    # Delete the files
    deleted = 0
    for f in unavailable_files:
        try:
            f.unlink()
            deleted += 1
            print(f"  ✓ Deleted {f.parent.name}/.unavailable")
        except Exception as e:
            print(f"  ⚠️  Could not delete {f}: {e}")

    print(f"\n✓ Cleared unavailable cache for {deleted} dataset(s)")
    print("  Tiles will be re-attempted on next elevation download")


def delete_osm_indexes(skip_confirm=False, region_name=None):
    """Delete OSM spatial index files.

    Args:
        skip_confirm: If True, skip confirmation prompt
        region_name: If specified, only delete indexes for this region (normalized name)
    """
    if not skip_confirm and region_name is None:
        confirm = input("\n⚠️  Delete all OSM spatial index files? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    if not OSM_INDEXES_DIR.exists():
        print("   No OSM indexes directory found")
        return

    deleted_count = 0
    failed_count = 0

    if region_name:
        # Delete only indexes for specific region
        normalized = region_name.lower().replace(" ", "-").replace("_", "-")
        patterns = [f"{normalized}-latest.osm_*", f"{normalized}.osm_*", f"{normalized}_*"]
        for pattern in patterns:
            for idx_file in OSM_INDEXES_DIR.glob(pattern):
                try:
                    if idx_file.is_dir():
                        shutil.rmtree(idx_file)
                    else:
                        idx_file.unlink()
                    deleted_count += 1
                except OSError:
                    failed_count += 1
    else:
        # Delete all index files
        for idx_file in list(OSM_INDEXES_DIR.iterdir()):
            try:
                if idx_file.is_dir():
                    shutil.rmtree(idx_file)
                else:
                    idx_file.unlink()
                deleted_count += 1
            except OSError:
                failed_count += 1

    if deleted_count > 0:
        scope = f"for {region_name}" if region_name else ""
        print(f"✓ Deleted {deleted_count} OSM index files {scope}".strip())
    if failed_count > 0:
        print(f"⚠️  {failed_count} index files could not be deleted (in use)")
    if deleted_count == 0 and failed_count == 0:
        print("   No OSM index files found to delete")


def parse_tile_coordinates(filename: str) -> Optional[Tuple[int, int]]:
    """Parse latitude and longitude from elevation tile filename.

    Handles multiple naming conventions:
    - SRTM/NED: N44W117.tif, n44w117.tif
    - ASTER: ASTGTMV003_N44W117_dem.tif
    - AW3D30: N044W117.tif

    Returns:
        (lat, lon) tuple or None if parsing fails
    """
    import re

    # Try different patterns
    # Pattern 1: Standard SRTM/NED (N44W117 or n44w117)
    match = re.search(r'([NSns])(\d{1,3})([EWew])(\d{1,3})', filename)
    if match:
        lat_dir, lat_val, lon_dir, lon_val = match.groups()
        lat = int(lat_val)
        lon = int(lon_val)

        if lat_dir.upper() == 'S':
            lat = -lat
        if lon_dir.upper() == 'W':
            lon = -lon

        return (lat, lon)

    return None


def delete_elevation_tiles_in_bbox(bbox: Tuple[float, float, float, float], verbose=True):
    """Delete elevation tiles that fall within the bounding box.

    Args:
        bbox: (lat_min, lon_min, lat_max, lon_max) bounding box
        verbose: Print progress messages
    """
    import math

    try:
        from utils.data_paths import ELEVATION_DATA_DIR
    except ImportError:
        ELEVATION_DATA_DIR = Path("data/elevation_data")

    if not ELEVATION_DATA_DIR.exists():
        if verbose:
            print("   No elevation data directory found")
        return

    lat_min, lon_min, lat_max, lon_max = bbox

    # Calculate which tile coordinates fall within bbox
    # Tiles are named by their lower-left corner (floor)
    tile_lat_min = int(math.floor(lat_min))
    tile_lat_max = int(math.floor(lat_max))
    tile_lon_min = int(math.floor(lon_min))
    tile_lon_max = int(math.floor(lon_max))

    deleted_count = 0
    datasets_affected = set()

    for dataset_dir in ELEVATION_DATA_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Skip VRT directories (arctic32m-vrt, rema32m-vrt)
        if dataset_dir.name.endswith('-vrt'):
            continue

        # Check all tile files in dataset
        for tile_file in list(dataset_dir.glob("*")):
            if tile_file.is_dir():
                continue

            # Skip non-elevation files
            if not tile_file.suffix.lower() in ['.tif', '.hgt']:
                continue

            coords = parse_tile_coordinates(tile_file.name)
            if coords:
                tile_lat, tile_lon = coords
                # Check if tile falls within bbox
                if (tile_lat_min <= tile_lat <= tile_lat_max and
                    tile_lon_min <= tile_lon <= tile_lon_max):
                    try:
                        tile_file.unlink()
                        deleted_count += 1
                        datasets_affected.add(dataset_dir.name)
                    except OSError as e:
                        if verbose:
                            print(f"   ⚠️  Could not delete {tile_file.name}: {e}")

    if verbose:
        if deleted_count > 0:
            print(f"✓ Deleted {deleted_count} elevation tiles from: {', '.join(sorted(datasets_affected))}")
        else:
            print("   No elevation tiles found in region bbox")

    return deleted_count > 0


def update_config_after_region_deletion(region_name: str, osm_deleted: bool, elevation_deleted: bool):
    """Update config.yaml after deleting region data.

    Args:
        region_name: Name of the region whose data was deleted
        osm_deleted: Whether OSM data was deleted
        elevation_deleted: Whether elevation data was deleted
    """
    try:
        import yaml
        config_path = Path("config.yaml")

        if not config_path.exists():
            return

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        modified = False

        if osm_deleted:
            # Remove from OSM_COVERAGE
            osm_coverage = config.get('OSM_COVERAGE', [])
            if region_name in osm_coverage:
                osm_coverage.remove(region_name)
                config['OSM_COVERAGE'] = osm_coverage
                modified = True

            # Remove from OSM_PLANET_DATA
            normalized = region_name.lower().replace(" ", "-").replace("_", "-")
            pbf_name = f"{normalized}-latest.osm.pbf"
            osm_planet_data = config.get('OSM_PLANET_DATA', [])
            if pbf_name in osm_planet_data:
                osm_planet_data.remove(pbf_name)
                config['OSM_PLANET_DATA'] = osm_planet_data
                modified = True

        if elevation_deleted:
            # Remove from ELEVATION_COVERAGE (if it exists)
            elev_coverage = config.get('ELEVATION_COVERAGE', [])
            if region_name in elev_coverage:
                elev_coverage.remove(region_name)
                config['ELEVATION_COVERAGE'] = elev_coverage
                modified = True

        if modified:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"   ✓ Updated config.yaml")

    except Exception as e:
        print(f"   ⚠️  Could not update config.yaml: {e}")


def rebuild_opentopodata_after_deletion():
    """Update opentopodata config and restart service after elevation deletion."""
    try:
        from utils.opentopodata_manager import rebuild_and_restart, update_config

        print("\n🔄 Updating OpenTopoData configuration...")

        # Update config based on remaining elevation data
        elevation_data_dir = Path("data/elevation_data")
        config_path = Path("opentopodata/config.yaml")
        update_config(elevation_data_dir, config_path)

        # Restart the service
        print("🔄 Restarting OpenTopoData service...")
        rebuild_and_restart()

        print("✓ OpenTopoData service updated")
    except ImportError:
        print("   ⚠️  opentopodata_manager not available - manual restart required")
    except Exception as e:
        print(f"   ⚠️  Could not restart OpenTopoData: {e}")


def delete_region_data(region_name: str, delete_checkpoints=True, delete_osm=True,
                       delete_elevation=True, rebuild_opentopodata=True):
    """Delete all data for a specific region only.

    This is used for per-region cleanup after analysis, ensuring only data
    for the specified region is removed while leaving other regions intact.

    Args:
        region_name: Name of the region (e.g., "Washington", "Hawaii")
        delete_checkpoints: Delete checkpoint directories for this region
        delete_osm: Delete OSM PBF file and indexes for this region
        delete_elevation: Delete elevation tiles within region's bounding box
        rebuild_opentopodata: Rebuild opentopodata service after elevation deletion
    """
    print(f"\n🧹 Cleaning up data for region: {region_name}")

    normalized = region_name.lower().replace(" ", "-").replace("_", "-")
    elevation_deleted = False

    if delete_checkpoints:
        # Sanitize region name the same way as analysis_id generation (line 11839)
        # "North Dakota" -> "NorthDakota" (removes spaces, keeps alphanumeric)
        sanitized = "".join(c for c in region_name if c.isalnum() or c in ("_", "-"))

        deleted_count = 0
        seen_dirs = set()

        # Try multiple patterns to catch all variations
        for pattern in [f"{sanitized}_*", f"{sanitized.lower()}_*", f"{region_name}_*", f"{normalized}_*"]:
            for checkpoint_dir in CHECKPOINT_DIR.glob(pattern):
                if checkpoint_dir.exists() and checkpoint_dir not in seen_dirs:
                    seen_dirs.add(checkpoint_dir)
                    try:
                        shutil.rmtree(checkpoint_dir)
                        deleted_count += 1
                    except OSError as e:
                        print(f"   ⚠️  Could not delete {checkpoint_dir.name}: {e}")

        if deleted_count > 0:
            print(f"✓ Deleted {deleted_count} checkpoint directory(s)")
        else:
            print("   No checkpoint directories found for this region")

    if delete_osm:
        # Delete OSM PBF file
        pbf_deleted = False
        for pbf_pattern in [f"{normalized}-latest.osm.pbf", f"{normalized}.osm.pbf"]:
            pbf_file = PLANET_OSM_DIR / pbf_pattern
            if pbf_file.exists():
                try:
                    pbf_file.unlink()
                    print(f"✓ Deleted OSM file: {pbf_pattern}")
                    pbf_deleted = True
                except OSError as e:
                    print(f"   ⚠️  Could not delete {pbf_pattern}: {e}")

        if not pbf_deleted:
            print("   No OSM PBF file found for this region")

        # Delete OSM spatial indexes
        delete_osm_indexes(skip_confirm=True, region_name=region_name)

    if delete_elevation:
        # Get region bounding box and delete tiles within it
        bbox = get_region_bbox_from_definitions(region_name)
        if bbox:
            elevation_deleted = delete_elevation_tiles_in_bbox(bbox, verbose=True)
        else:
            print(f"   ⚠️  Could not determine bounding box for {region_name}")
            print("      Elevation tiles not deleted")

    # Update config.yaml
    update_config_after_region_deletion(region_name, delete_osm, delete_elevation)

    # Rebuild opentopodata if elevation was deleted
    if delete_elevation and elevation_deleted and rebuild_opentopodata:
        rebuild_opentopodata_after_deletion()

    print(f"✓ Region cleanup complete: {region_name}")


def download_data_for_regions(regions):
    """Download data for regions without analysis."""
    from climb_analyzer.data.manager import DataManager

    manager = DataManager()

    print(f"\n🔄 Downloading data for {len(regions)} region(s)")

    # Show currently available elevation data
    try:
        from utils.config_loader import get_config

        config = get_config()
        elevation_datasets = config.get("ELEVATION_DATASETS", {})

        if elevation_datasets:
            print("\n Available elevation data:")
            for dataset_name, regions_list in elevation_datasets.items():
                if regions_list:
                    regions_str = ", ".join(regions_list)
                    print(f"  • {dataset_name}: {regions_str}")
                else:
                    print(f"  • {dataset_name}: (no regions)")
        else:
            print("\n No elevation data currently available")
    except Exception:
        # Don't fail if we can't read elevation data
        pass

    for i, region in enumerate(regions, 1):
        print(f"\n[{i}/{len(regions)}] {region['canonical_name']}")

        # Get bounds
        is_state = region["type"] == "state"
        bounds = manager.get_region_bounds(region["canonical_name"], is_state=is_state)

        if not bounds:
            print(f"  ❌ Could not determine bounds for {region['canonical_name']}")
            continue

        # Download OSM data
        print("  Downloading OSM data...")
        pbf_path = manager.download_osm_data(region["canonical_name"], is_state=is_state)
        if pbf_path:
            print(f"  ✓ OSM data: {pbf_path.name}")
            # Build index
            if not manager.build_osm_index(pbf_path):
                print("  ⚠️  Spatial index build failed")
        else:
            print("  ❌ OSM download failed")

        # Download elevation data
        print("  Downloading elevation data...")

        # Determine which datasets to download based on config
        datasets_to_download = []
        credentials = None

        try:
            from utils.config_loader import get_config

            config = get_config()

            # Get tier setting (defaults to tertiary for maximum coverage)
            tier_setting = config.get("ELEVATION_DATASET_TIERS", "primary+secondary+tertiary")

            # NOTE: As of December 2025, no datasets require credentials.
            # All datasets now use public sources:
            # - SRTM: OpenTopography S3 (public)
            # - AW3D30: JAXA FTP (public)
            # - NED, ArcticDEM, REMA: AWS S3 (public)
            credentials = None

            # Get region-aware dataset list (same priority logic used during elevation fetching)
            # This ensures download phase matches analysis phase requirements
            region_name = region["canonical_name"]
            center_lat = (bounds[0] + bounds[2]) / 2  # Center latitude for latitude-based rules
            datasets_to_download = get_dataset_priority_for_region(region_name, center_lat, cloud_mode=False)

            # Apply tier filtering if user wants to limit datasets
            # tier_setting: "primary" = 1 dataset, "primary+secondary" = 2 (max now)
            if tier_setting != "primary+secondary":
                max_datasets = 1 if tier_setting == "primary" else 2
                datasets_to_download = datasets_to_download[:max_datasets]

            # If no datasets (shouldn't happen), default to srtm30m
            if not datasets_to_download:
                datasets_to_download = ["srtm30m"]

            print(f"  Region datasets: {', '.join(datasets_to_download)}")

        except Exception as e:
            print(f"  ⚠️  Could not read tier config, using default (srtm30m): {e}")
            datasets_to_download = ["srtm30m"]

        success, new_files = manager.download_elevation_data(
            region["canonical_name"], bounds, datasets_to_download, credentials=None
        )

        if success:
            print(f"  ✓ Elevation data: {new_files} new file(s) downloaded")
        else:
            print("  ⚠️  Elevation data download had issues (may be partially complete)")

    # Update config.yaml with downloaded data
    print("\n📝 Updating config.yaml...")
    try:
        from utils.sync_config_data import update_config_yaml

        result = update_config_yaml()
        if result.get("changes"):
            print(f"  ✓ Config updated: {len(result['changes'])} change(s)")
        else:
            print("  ✓ Config is up to date")
    except Exception as e:
        print(f"  ⚠️  Could not update config: {e}")

    # Rebuild OpenTopoData server if elevation data was downloaded
    print("\n🔄 Rebuilding OpenTopoData server...")
    try:
        from utils.opentopodata_manager import rebuild_and_restart

        if rebuild_and_restart(validate_health=True):
            print("  ✓ OpenTopoData server rebuilt and ready")
        else:
            print("  ⚠️  OpenTopoData rebuild had issues (server may still work)")
    except Exception as e:
        print(f"  ⚠️  Could not rebuild OpenTopoData: {e}")
        print("     You can manually rebuild with: python utils/opentopodata_manager.py rebuild")

    print("\n✓ Data download complete")


def run_cleanup_subcommand(args):
    """Handle the cleanup subcommand for global data deletion."""
    print("\n Cleanup Mode")

    # Validate at least one option specified
    has_option = any([
        getattr(args, 'cleanup_checkpoints', False),
        getattr(args, 'cleanup_elevation', False),
        getattr(args, 'cleanup_osm', False),
        getattr(args, 'cleanup_all', False),
        getattr(args, 'cleanup_unavailable', False),
    ])

    if not has_option:
        print("Error: Specify at least one of: --checkpoints, --elevation, --osm, --all, --unavailable-cache")
        print("\nUsage: ./climb-analyzer cleanup [--checkpoints] [--elevation] [--osm] [--all] [--unavailable-cache] [--force]")
        sys.exit(1)

    skip_confirm = getattr(args, 'force', False)

    if args.cleanup_all or args.cleanup_checkpoints:
        delete_checkpoints(skip_confirm=skip_confirm)

    if args.cleanup_all or args.cleanup_osm:
        delete_planet_data(skip_confirm=skip_confirm)
        delete_osm_indexes(skip_confirm=True)  # Always skip confirm for indexes if OSM confirmed

    if args.cleanup_all or args.cleanup_elevation:
        delete_elevation_data(skip_confirm=skip_confirm)
        # Rebuild opentopodata after elevation deletion
        rebuild_opentopodata_after_deletion()

    if getattr(args, 'cleanup_unavailable', False):
        delete_unavailable_cache(
            dataset=getattr(args, 'dataset', None),
            skip_confirm=skip_confirm
        )

    print("\n✓ Cleanup complete")


def main():
    """Main function to run the climb analyzer."""
    from pathlib import Path

    from utils.batch_cleanup import get_cleanup_targets, perform_cleanup
    from utils.error_logger import ErrorLogger, LogRotator
    from utils.version_checker import check_for_updates

    # Check for updates (silent mode - only shows message if update available)
    check_for_updates(silent=True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Climb Analyzer - Analyze road and trail climbs from OpenStreetMap",
        epilog="Use --help-extended for detailed examples and usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === SUBCOMMANDS ===
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Cleanup subcommand
    cleanup_parser = subparsers.add_parser(
        'cleanup',
        help='Delete data without running analysis',
        description='Delete checkpoints, elevation data, or OSM data globally.'
    )
    cleanup_parser.add_argument(
        '--checkpoints',
        action='store_true',
        dest='cleanup_checkpoints',
        help='Delete all checkpoint files'
    )
    cleanup_parser.add_argument(
        '--elevation',
        action='store_true',
        dest='cleanup_elevation',
        help='Delete all elevation data'
    )
    cleanup_parser.add_argument(
        '--osm',
        action='store_true',
        dest='cleanup_osm',
        help='Delete all OSM data and indexes'
    )
    cleanup_parser.add_argument(
        '--all',
        action='store_true',
        dest='cleanup_all',
        help='Delete all data (checkpoints + elevation + OSM)'
    )
    cleanup_parser.add_argument(
        '--unavailable-cache',
        action='store_true',
        dest='cleanup_unavailable',
        help='Delete .unavailable files (tiles marked as unavailable will be re-attempted)'
    )
    cleanup_parser.add_argument(
        '--dataset',
        type=str,
        metavar='NAME',
        help='Only clear unavailable cache for specific dataset (e.g., srtm30m, ned10m, aster30m)'
    )
    cleanup_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )

    # === ANALYSIS MODES (Mutually Exclusive) ===
    analysis_group = parser.add_mutually_exclusive_group()

    analysis_group.add_argument(
        "-a",
        "--address",
        type=str,
        metavar="ADDRESS",
        help="Address for radius-based analysis (e.g., 'Boulder, CO')",
    )

    analysis_group.add_argument(
        "-r",
        "--run-region",
        type=str,
        metavar="REGION",
        help="Analyze region(s) - use full state names, not acronyms. "
        "Examples: 'Vermont', 'Switzerland', or 'New Zealand,South Korea,United Kingdom' for batch",
    )

    analysis_group.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive mode with menus (default if no arguments)",
    )

    analysis_group.add_argument(
        "-g",
        "--gui",
        action="store_true",
        help="Launch web-based GUI interface (starts Next.js dev server on http://localhost:3000)",
    )

    # === ANALYSIS PARAMETERS ===
    parser.add_argument(
        "--distance",
        type=float,
        metavar="MILES",
        help="Search radius/distance in miles (required with --address)",
    )

    parser.add_argument(
        "-s",
        "--surface-filter",
        type=str,
        default="all",
        help="Surface filter: comma-separated list of paved,gravel,dirt or 'all' [default: all]. Examples: 'paved', 'paved,gravel', 'all'",
    )

    parser.add_argument(
        "--cycling-filter",
        action="store_true",
        help="Enable cycling accessibility filter [default: disabled]",
    )

    parser.add_argument(
        "-u",
        "--units",
        type=str,
        default="auto",
        choices=["metric", "imperial", "auto"],
        help="Unit system (auto=region-native, imperial=US, metric=international) [default: auto]",
    )

    parser.add_argument(
        "-t",
        "--score-type",
        type=str,
        default=None,
        choices=["basic", "fiets", "pdi"],
        help="Scoring: basic (grade*dist), fiets (gradient), pdi (difficulty) [default: basic]",
    )

    parser.add_argument(
        "-m",
        "--min-score",
        type=float,
        help="Minimum climb score threshold (requires -t/--score-type to be specified)",
    )

    # === DATA MANAGEMENT ===
    data_group = parser.add_argument_group("Data Management")

    data_group.add_argument(
        "-U",
        "--update-geo-boundaries",
        action="store_true",
        help="Update country/state boundary data from sources",
    )

    data_group.add_argument(
        "-D",
        "--data-download",
        action="store_true",
        help="Download OSM and elevation data without running analysis",
    )

    # Per-region cleanup flags (used with -r for post-analysis cleanup)
    data_group.add_argument(
        "-c",
        "--cleanup-checkpoints",
        action="store_true",
        help="Delete checkpoints for THIS region after successful analysis (use with -r)",
    )

    data_group.add_argument(
        "-Z",
        "--cleanup-all-data",
        action="store_true",
        help="Delete checkpoints + OSM + elevation for THIS region after successful analysis (use with -r)",
    )

    # Note: Checkpoints are KEPT by default after analysis. No -K flag needed.

    data_group.add_argument(
        "--ignore-checkpoints",
        action="store_true",
        help="Ignore existing checkpoints and start fresh analysis (default: auto-resume if checkpoints exist)",
    )

    data_group.add_argument(
        "--no-cloud-upload",
        action="store_true",
        help="Skip uploading to cloud cache (default: auto-upload clean analyses)",
    )

    # NOTE: Old global deletion flags (-C, -P, -E, -X) have been replaced by:
    #   ./climb-analyzer cleanup --checkpoints/--osm/--elevation/--all

    # === INFORMATIONAL ===
    info_group = parser.add_argument_group("Informational")

    info_group.add_argument(
        "--list-regions",
        action="store_true",
        help="List all available regions for analysis and exit",
    )

    # === CLIMB MERGING ===
    merge_group = parser.add_argument_group("Climb Merging")

    merge_group.add_argument(
        "--allow-cross-country-merge",
        action="store_true",
        help="Allow merging climbs across country boundaries (overrides config)",
    )

    merge_group.add_argument(
        "--merge-distance-km",
        type=float,
        help="Maximum distance in km between climb endpoints for merging (default: 0.5)",
    )

    merge_group.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable all climb merging (for debugging)",
    )

    merge_group.add_argument(
        "--merge-regions",
        nargs="+",
        metavar="FILES",
        help="Post-process: Merge cross-region climbs from existing .xlsx files. "
        "Examples: --merge-regions VT.xlsx NH.xlsx  OR  --merge-regions output/state_*.xlsx",
    )

    # === HELP & OUTPUT ===
    parser.add_argument(
        "--help-extended",
        action="store_true",
        help="Show extended help with examples",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging and technical details",
    )

    # === INTERNAL FLAGS ===
    # Internal flags for batch subprocesses only - not for user use
    parser.add_argument(
        "--skip-storage-prompt",
        action="store_true",
        help=argparse.SUPPRESS,  # Internal flag for batch subprocesses
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help=argparse.SUPPRESS,  # Internal flag - enables batch subprocess mode
    )

    parser.add_argument(
        "--scope",
        type=str,
        help=argparse.SUPPRESS,  # Internal flag - specifies state/country scope
    )

    parser.add_argument(
        "--states",
        type=str,
        help=argparse.SUPPRESS,  # Internal flag - state names for batch processing
    )

    parser.add_argument(
        "--countries",
        type=str,
        help=argparse.SUPPRESS,  # Internal flag - country names for batch processing
    )

    args = parser.parse_args()

    # Handle cleanup subcommand first (no other processing needed)
    if args.command == 'cleanup':
        run_cleanup_subcommand(args)
        return

    # Validate: -m/--min-score requires -t/--score-type to be explicitly specified
    if args.min_score is not None and args.score_type is None:
        parser.error(
            "-m/--min-score requires -t/--score-type to be specified.\n"
            "       Score types have different scales:\n"
            "         -t basic  : grade × distance (typical range: 0 - 100,000+)\n"
            "         -t fiets  : gradient-based (typical range: 0 - 20+)\n"
            "         -t pdi    : difficulty index (typical range: 0 - 100+)\n"
            "       Example: -t basic -m 5000"
        )

    # Default score_type to "basic" if not specified
    if args.score_type is None:
        args.score_type = "basic"

    # Store args globally so it can be accessed by save_large_dataframe_as_split_excel
    import sys

    sys.modules["__main__"].args = args

    # Set global verbose flag
    global _VERBOSE_MODE
    _VERBOSE_MODE = getattr(args, "verbose", False)

    # Handle extended help
    if args.help_extended:
        show_extended_help()
        return

    # Handle post-processing merge operations
    if args.merge_regions:
        import glob

        from climb_analyzer.utils.formatting import print_banner, print_success

        print_banner("Cross-Region Climb Merger")
        print("Post-processing: Merging climbs across region boundaries\n")

        # Expand wildcards
        all_files = []
        for pattern in args.merge_regions:
            expanded = glob.glob(pattern)
            if expanded:
                all_files.extend(expanded)
            else:
                all_files.append(pattern)  # Keep as-is if not a wildcard

        # Convert to Path objects and filter for xlsx files
        xlsx_files = [Path(f) for f in all_files if f.endswith(".xlsx")]

        if len(xlsx_files) < 2:
            print(f"Error: Need at least 2 .xlsx files to merge. Found: {len(xlsx_files)}")
            return

        print(f"Found {len(xlsx_files)} files to process\n")

        # Call batch merge function
        from scripts.batch_merge_regions import batch_merge_regions

        batch_merge_regions(xlsx_files)

        print_success("\n✓ Cross-region merge complete")
        return

    # Handle --list-regions (show available regions and exit)
    if args.list_regions:
        from climb_analyzer.data.geo_lookup import print_available_regions

        print_available_regions()
        return

    # Initialize variables to replace deprecated arguments
    # These act as local variables to handle batch mode without deprecated CLI args
    batch_mode = False
    batch_states = None
    batch_countries = None

    # Handle internal arguments from subprocess (these may already be set by argparse)
    # If not set, initialize them for compatibility
    if not hasattr(args, "batch") or args.batch is None:
        args.batch = False
    if not hasattr(args, "states") or args.states is None:
        args.states = None
    if not hasattr(args, "countries") or args.countries is None:
        args.countries = None
    if not hasattr(args, "scope") or args.scope is None:
        args.scope = None

    # Legacy compatibility attributes
    args.no_cycling_filter = False
    args.cleanup_after_batch = False

    # Handle --run-region (auto-detect batch vs single region)
    used_run_region = False  # Track if user used new -r flag
    cached_parsed_regions = None  # Cache parsed regions to avoid double prompting
    if args.run_region:
        used_run_region = True
        from utils.region_detector import parse_regions, print_unknown_region_error

        parsed_regions = parse_regions(args.run_region)
        cached_parsed_regions = parsed_regions  # Cache for reuse later

        # Check for unknown regions and exit with helpful error
        unknown_regions = [r for r in parsed_regions if r["type"] == "unknown"]
        if unknown_regions:
            print_unknown_region_error(unknown_regions)
            return

        if len(parsed_regions) > 1:
            # Multiple regions = enable batch subprocess mode (run each as separate subprocess)
            print("   Multiple regions detected - enabling batch mode")
            batch_mode = True
            args.batch = True  # Set for compatibility
            # Convert to legacy format for batch processing
            if all(r["type"] == "state" for r in parsed_regions):
                batch_states = args.run_region
                batch_countries = None
                args.states = batch_states  # Set for compatibility
                args.countries = batch_countries
            elif all(r["type"] in ["country", "subregion"] for r in parsed_regions):
                batch_countries = args.run_region
                batch_states = None
                args.countries = batch_countries  # Set for compatibility
                args.states = batch_states
            else:
                # Mixed types - split into separate state and country batches
                state_regions = [r["input_name"] for r in parsed_regions if r["type"] == "state"]
                country_regions = [
                    r["input_name"] for r in parsed_regions if r["type"] in ["country", "subregion"]
                ]

                # For mixed types, we need to run them as separate batches
                # Process states first, then countries
                print(
                    f"   Mixed region types detected: {len(state_regions)} state(s), {len(country_regions)} country/region(s)"
                )
                print(f"   States: {', '.join(state_regions)}")
                print(f"   Countries/Regions: {', '.join(country_regions)}")

                # Store both for batch processing
                batch_states = ",".join(state_regions) if state_regions else None
                batch_countries = ",".join(country_regions) if country_regions else None
                args.states = batch_states  # Set for compatibility
                args.countries = batch_countries
        else:
            # Single region with --run-region
            # Set batch flag to avoid interactive prompts (even for single region)
            args.batch = True
            batch_mode = False  # Not true batch mode (multiple regions), but non-interactive

            # Check memory BEFORE proceeding with direct processing
            region = parsed_regions[0]

            # Memory check for single regions (only in local mode)
            try:
                from utils.config_loader import DEPLOYMENT_TYPE

                if DEPLOYMENT_TYPE == "local" and MEMORY_CHECK_AVAILABLE:
                    print("\n   Checking available memory for region analysis...")
                    from utils.memory_checker import (
                        check_memory_for_region,
                        print_memory_report,
                        print_split_guidance,
                    )

                    # Estimate memory needs based on region
                    memory_result = check_memory_for_region(region["canonical_name"], None)

                    if not memory_result["is_sufficient"]:
                        print_memory_report(
                            memory_result["segments"],
                            memory_result["required_gb"],
                            memory_result["available_gb"],
                            memory_result["is_sufficient"],
                            memory_result["message"],
                        )

                        # Show split guidance
                        print_split_guidance(memory_result)

                        # Automatically proceed with analysis
                        print("\nProceeding with analysis...")
            except Exception as e:
                print(f"   Warning: Could not check memory: {e}")
                # Continue anyway if memory check fails

    # Handle GUI mode (early return)
    if args.gui:
        launch_gui()
        return

    # Handle data management operations (early return)
    if args.update_geo_boundaries:
        run_update_geo_boundaries()
        return

    # NOTE: Old global deletion flags (-C, -P, -E, -X) have been removed.
    # Use: ./climb-analyzer cleanup --checkpoints/--osm/--elevation/--all
    # Per-region cleanup: -c/--cleanup-checkpoints and -Z/--cleanup-all-data (with -r)

    # Handle data download
    if args.data_download:
        if not args.run_region:
            print("❌ Error: --data-download requires --run-region to specify which regions")
            sys.exit(1)
        from utils.region_detector import parse_regions

        regions = parse_regions(args.run_region)
        download_data_for_regions(regions)
        return

    # Handle batch mode with interactive menu (when batch mode is set but no region specified)
    if batch_mode and not batch_states and not batch_countries and not args.run_region:
        scope_type, locations, cleanup_after = batch_mode_menu()

        # Check if any countries need to be split into subregions
        from utils.large_country_handler import get_country_regions, should_split_country

        expanded_locations = []
        split_country_groups = {}  # Maps parent country name to list of subregion names

        for location in locations:
            if scope_type == "country" and should_split_country(location):
                # This country needs to be split
                regions = get_country_regions(location)
                print(
                    f"\n⚠️  {location} is a large country that will be split into {len(regions)} subregions:"
                )

                subregion_names = []
                for region_name, bbox in regions:
                    print(f"   • {region_name}")
                    expanded_locations.append(region_name)
                    subregion_names.append(region_name)

                # Track this group for later merging
                split_country_groups[location] = subregion_names

                print(
                    "\nAfter analyzing all subregions, you'll be prompted to merge them back together."
                )
            else:
                # Regular location, not a split country
                expanded_locations.append(location)

        # Replace locations with expanded version
        locations = expanded_locations

        # Now run each location as a separate subprocess
        from climb_analyzer.utils.formatting import print_banner, print_header

        print_banner(
            f"Starting Batch Analysis - {len(locations)} location(s)", width=100, spacing_before=1
        )

        failed_locations = []
        completed_subregions = {}  # Maps parent country to list of completed subregion names

        for i, location in enumerate(locations, 1):
            # Prominent progress header - wider and bold
            print_banner(
                f"[{i}/{len(locations)}] Processing: {location}", width=100, spacing_before=1
            )
            return_code = run_batch_subprocess(location, scope_type, args, cleanup_after)
            if return_code != 0:
                failed_locations.append(location)
            else:
                # Check if this location is a subregion of a split country
                for parent_country, subregion_list in split_country_groups.items():
                    if location in subregion_list:
                        if parent_country not in completed_subregions:
                            completed_subregions[parent_country] = []
                        completed_subregions[parent_country].append(location)
                        break

        # Print modern batch completion summary
        from climb_analyzer.utils.formatting import print_banner, print_error, print_success

        print_banner("Batch Analysis Complete", spacing_before=1)
        print(f"Total locations: {len(locations)}")

        successful_count = len(locations) - len(failed_locations)
        if successful_count > 0:
            print_success(f"Completed: {successful_count}/{len(locations)}")

        if failed_locations:
            print_error(f"Failed: {len(failed_locations)}/{len(locations)}")
            print(f"  Failed locations: {', '.join(failed_locations)}")
        else:
            print_success("All locations completed successfully!")
        print()

        # Offer to merge split subregions for each parent country
        if completed_subregions:
            surface_filter = getattr(args, "surface_filter", "all")
            score_type = getattr(args, "score_type", "basic")

            for parent_country, subregion_list in completed_subregions.items():
                # Only offer merge if we have at least 2 subregions completed
                if len(subregion_list) >= 2:
                    # Check if all subregions for this country completed
                    expected_subregions = split_country_groups.get(parent_country, [])
                    if len(subregion_list) == len(expected_subregions):
                        print(f"\n✓ All subregions for {parent_country} completed successfully")
                    else:
                        print(
                            f"\n⚠️  Only {len(subregion_list)}/{len(expected_subregions)} subregions for {parent_country} completed"
                        )

                    # Offer to merge
                    offer_automatic_merge(subregion_list, surface_filter, score_type)

        # Check if we should run cross-region climb analysis
        # For states/countries, convert location list to format expected by prompt function
        if len(locations) >= 2 and successful_count >= 2:
            # Create a dummy batch_tracker to get completed locations
            from utils.batch_progress import BatchProgressTracker

            batch_tracker = BatchProgressTracker()
            # Mark successful locations as completed
            for loc in locations:
                if loc not in failed_locations:
                    batch_tracker.mark_completed(loc)

            # Get filter parameters (use defaults from args if not set)
            surface_filter = getattr(args, "surface_filter", "all")
            score_type = getattr(args, "score_type", "basic")

            # For state/country batch, we don't have radius_km, so pass None
            # The prompt function will handle it
            prompt_cross_region_analysis(
                locations, batch_tracker, surface_filter, score_type, radius_km=None
            )

        return

    # Handle batch mode with command-line specified locations
    if args.batch and (args.states or args.countries):
        # Handle mixed types - both states and countries can be set
        all_locations = []

        if args.states and args.countries:
            # Mixed batch - process states first, then countries
            state_locations = [(s.strip(), "state") for s in args.states.split(",") if s.strip()]
            country_locations = [
                (c.strip(), "country") for c in args.countries.split(",") if c.strip()
            ]
            all_locations = state_locations + country_locations
        elif args.states:
            # Only states
            all_locations = [(s.strip(), "state") for s in args.states.split(",") if s.strip()]
        else:
            # Only countries
            all_locations = [(c.strip(), "country") for c in args.countries.split(",") if c.strip()]

        # Check if any countries need to be split into subregions
        # Track which subregions belong to which parent country for merging later
        from utils.large_country_handler import get_country_regions, should_split_country

        expanded_locations = []
        split_country_groups = {}  # Maps parent country name to list of subregion names

        for location, scope_type in all_locations:
            if scope_type == "country" and should_split_country(location):
                # This country needs to be split
                regions = get_country_regions(location)
                print(
                    f"\n⚠️  {location} is a large country that will be split into {len(regions)} subregions:"
                )

                subregion_names = []
                for region_name, bbox in regions:
                    print(f"   • {region_name}")
                    # Add each subregion as a separate location
                    # We'll use a special marker to indicate this is a subregion
                    expanded_locations.append(
                        (region_name, "subregion", location)
                    )  # (name, type, parent)
                    subregion_names.append(region_name)

                # Track this group for later merging
                split_country_groups[location] = subregion_names

                print(
                    "\nAfter analyzing all subregions, you'll be prompted to merge them back together."
                )
            else:
                # Regular location, not a split country
                expanded_locations.append((location, scope_type, None))  # No parent

        # Replace all_locations with expanded version
        all_locations = expanded_locations

        # If only one location, don't spawn subprocess - process directly
        # But keep args.batch = True to skip interactive prompts
        # Import functions needed for batch storage calculations
        from climb_analyzer.data.data_coverage_checker import calculate_storage_requirements_for_batch

        # Initialize cleanup_after (may be updated later by storage prompt)
        # Use new per-region cleanup flags
        cleanup_after = getattr(args, 'cleanup_all_data', False)

        if len(all_locations) == 1:
            location, scope_type, parent = all_locations[0]
            # Single location - process directly without spawning subprocess
            # Set the location for direct processing below
            if scope_type == "region":
                args.states = location
                args.countries = None
            elif scope_type == "subregion":
                # Subregions are processed as countries with bounding boxes
                args.countries = location
                args.states = None
            else:
                args.countries = location
                args.states = None
            # Keep args.batch = True to avoid interactive prompts
            # Skip batch subprocess loop and fall through to direct processing below
        else:
            # Multiple locations - run each as a separate subprocess
            # Calculate storage for all locations (need to separate by type for storage calc)
            # Handle new 3-tuple format: (location, scope_type, parent)
            state_locs = [loc for loc, typ, parent in all_locations if typ == "state"]
            country_locs = [
                loc for loc, typ, parent in all_locations if typ in ("country", "subregion")
            ]

            storage_info_parts = []
            if state_locs:
                storage_info = calculate_storage_requirements_for_batch(state_locs, "state")
                if storage_info:
                    storage_info_parts.append(storage_info)
            if country_locs:
                storage_info = calculate_storage_requirements_for_batch(country_locs, "country")
                if storage_info:
                    storage_info_parts.append(storage_info)

            if storage_info_parts:
                from climb_analyzer.utils.formatting import print_header

                print()
                print_header("Storage Requirements")
                print("\n".join(storage_info_parts))

                # Use CLI argument for cleanup decision (no prompt in batch mode)
                # Default to False (keep files) unless --delete-data-on-complete is set
                if cleanup_after:
                    print("\n✓ Files will be deleted after analysis (--delete-data-on-complete)")
                else:
                    print("\n✓ Files will be kept after analysis (use -X to delete)")
                print()

            from climb_analyzer.utils.formatting import print_banner, print_header

            print_banner(
                f"Starting Batch Analysis - {len(all_locations)} location(s)",
                width=100,
                spacing_before=1,
            )

            failed_locations = []
            completed_subregions = {}  # Maps parent country to list of completed subregion names

            for i, (location, scope_type, parent) in enumerate(all_locations, 1):
                # Prominent progress header - wider and bold
                display_type = "subregion" if scope_type == "subregion" else scope_type
                print_banner(
                    f"[{i}/{len(all_locations)}] Processing: {location} ({display_type})",
                    width=100,
                    spacing_before=1,
                )

                # For subregions, we process them as countries but with special handling
                actual_scope = "country" if scope_type == "subregion" else scope_type
                return_code = run_batch_subprocess(location, actual_scope, args, cleanup_after)

                if return_code != 0:
                    failed_locations.append(location)
                else:
                    # Track completed subregions for merging
                    if scope_type == "subregion" and parent:
                        if parent not in completed_subregions:
                            completed_subregions[parent] = []
                        completed_subregions[parent].append(location)

            # Print modern batch completion summary
            from climb_analyzer.utils.formatting import print_banner, print_error, print_success

            print_banner("Batch Analysis Complete", spacing_before=1)
            print(f"Total locations: {len(all_locations)}")

            successful_count = len(all_locations) - len(failed_locations)
            if successful_count > 0:
                print_success(f"Completed: {successful_count}/{len(all_locations)}")

            if failed_locations:
                print_error(f"Failed: {len(failed_locations)}/{len(all_locations)}")
                print(f"  Failed locations: {', '.join(failed_locations)}")
            else:
                print_success("All locations completed successfully!")
            print()

            # Offer to merge split subregions for each parent country
            if completed_subregions:
                surface_filter = getattr(args, "surface_filter", "all")
                score_type = getattr(args, "score_type", "basic")

                for parent_country, subregion_list in completed_subregions.items():
                    # Only offer merge if we have at least 2 subregions completed
                    if len(subregion_list) >= 2:
                        # Check if all subregions for this country completed
                        expected_subregions = split_country_groups.get(parent_country, [])
                        if len(subregion_list) == len(expected_subregions):
                            print(f"\n✓ All subregions for {parent_country} completed successfully")
                        else:
                            print(
                                f"\n⚠️  Only {len(subregion_list)}/{len(expected_subregions)} subregions for {parent_country} completed"
                            )

                        # Offer to merge
                        offer_automatic_merge(subregion_list, surface_filter, score_type)

            return

    LOG_FILE = "climb_analyzer.log"

    # Initialize logging with rotation (keep last 3 runs)
    log_rotator = LogRotator(Path(LOG_FILE), max_runs=3)
    log_rotator.rotate_if_needed()

    try:
        log_file = open(LOG_FILE, "a")  # Append mode for rotation
        original_stdout = sys.__stdout__
        original_stderr = sys.__stderr__
        sys.stdout = Tee(original_stdout, log_file)
        # Use rate-limited Tee for stderr to prevent massive log files from tqdm progress bars
        # Progress bars will update terminal every 0.1s but only log every 10 minutes
        from climb_analyzer.utils.tee import RateLimitedTee

        sys.stderr = RateLimitedTee(original_stderr, log_file, log_interval_seconds=600)

        # Add run separator
        mode_label = "Batch Mode" if args.batch else "Interactive Mode"
        log_rotator.add_run_separator(mode_label)

    except Exception as e:
        # Handle potential file opening errors
        print(f"Error opening log file: {e}")

    # Check deployment type
    try:
        from climb_analyzer.utils.formatting import print_banner, print_dim, print_info
        from utils.config_loader import DEPLOYMENT_TYPE

        deployment_type = DEPLOYMENT_TYPE

        # Print deployment banner
        print("\n")  # Ensure clean line before banner
        sys.stdout.flush()
        print_banner(f"CLIMB ANALYZER - {deployment_type.upper()} DEPLOYMENT", spacing_before=0)
        print()  # Ensure clean line after banner
        sys.stdout.flush()

        # Show cloud cache status on one line
        if CLOUD_CACHE_AVAILABLE and CLOUD_CACHE_ENABLED:
            print_dim("Cloud cache: Enabled")
        elif not CLOUD_CACHE_ENABLED:
            print_dim("Cloud cache: Disabled")
    except ImportError:
        deployment_type = "cloud"
        print("Warning: Could not determine deployment type, using cloud")

    # Only check for existing analyses in interactive mode
    # Skip this check when user explicitly specifies a region/address to analyze
    # Default to interactive mode if no CLI arguments provided
    # Also skip for batch subprocesses (args.batch or internal args like args.countries/args.states)
    is_interactive_mode = args.interactive or (
        not args.run_region
        and not args.address
        and not args.batch
        and not args.countries
        and not args.states
    )
    if (
        is_interactive_mode
        and not args.run_region
        and not args.address
        and not args.batch
        and not args.countries
        and not args.states
    ):
        existing_analysis_id = find_all_existing_analyses()

        if existing_analysis_id:
            print(f"Resuming analysis: {existing_analysis_id}\n")
            resume_analysis_from_startup(existing_analysis_id)
            return  # Exit after resume completes

        print_info("No existing analyses found - starting new analysis configuration")
    print()

    # Batch mode cleanup prompt (before starting analysis)
    # Skip if this is a subprocess (parent already prompted)
    cleanup_after_analysis = False
    if args.batch and not args.skip_storage_prompt:
        cleanup_targets, _ = get_cleanup_targets()
        if cleanup_targets:
            # Use new per-region cleanup flags
            cleanup_after_analysis = getattr(args, 'cleanup_all_data', False)

    # init
    configure_checkpoints()
    total_start_time = time.time()

    # Reset elevation statistics for this analysis run
    from utils.elevation_stats_collector import reset_stats

    reset_stats()

    # Check if we have CLI arguments mode (--run-region or --address)
    # This needs to be determined early to control prompting behavior
    has_cli_mode = args.run_region or args.address

    # === STEP 1: Determine analysis scope FIRST (needed for smart score defaults) ===
    # For interactive mode, get scope before score type
    # For CLI mode, scope is already determined from args
    scope_type_for_defaults = None  # Will be used to set smart score defaults

    # Determine scope type early for non-interactive modes
    interactive_scope_result = None  # Will store result if we get scope interactively
    if args.address:
        scope_type_for_defaults = "address"
    elif args.run_region or args.states or args.countries:
        scope_type_for_defaults = "region"  # region/state/country all use min_score=0 default
    elif args.batch:
        scope_type_for_defaults = "region"
    else:
        # Interactive mode - need to get scope choice first
        interactive_scope_result = get_analysis_scope_choice(deployment_type)
        scope_type_for_defaults, _, _ = interactive_scope_result

    # Get surface filter preference (CLI mode, batch mode, or interactive)
    if has_cli_mode or args.batch:
        surface_filter = args.surface_filter
    else:
        surface_filter = get_surface_filter_choice()

    # Get cycling filter preference (CLI args or interactive)
    if has_cli_mode or args.batch:
        cycling_only = args.cycling_filter
    else:
        cycling_only = get_cycling_filter_choice()

    # Get unit system preference (CLI mode, batch mode, or interactive)
    if has_cli_mode or args.batch:
        unit_system = args.units
    else:
        unit_system = get_unit_system_choice()

    # Note: If unit_system is "auto", it will be resolved to actual units later
    # once we know the region (US state = imperial, other = metric)
    # Initialize is_us_address for address searches (will be set later if needed)
    is_us_address = None

    # === STEP 2: Get score type and minimum score (with smart defaults based on scope) ===
    # Get climb score type preference (CLI mode, batch mode, or interactive)
    if has_cli_mode or args.batch:
        score_type = args.score_type
    else:
        score_type = get_score_type_choice()

    # Get minimum score threshold with SMART DEFAULTS based on scope type
    if args.min_score is not None:
        min_score = args.min_score
    elif has_cli_mode or args.batch:
        # Use 0 default in CLI/batch mode to show all climbs
        min_score = 0
    else:
        # Interactive mode - prompt with 0 default
        try:
            default_score = "0"
            if score_type == "basic":
                score_desc = "basic score (elevation*distance)"
            elif score_type == "fiets":
                score_desc = "FIETS score"
            else:  # pdi
                score_desc = "PDI score"

            min_score = float(
                input(f"Enter minimum {score_desc} to display (default {default_score}): ").strip()
                or default_score
            )
        except ValueError:
            min_score = 0

    # Geocoding is always enabled (fast with offline reverse_geocoder)
    enable_geocoding = True

    # === STEP 3: Get full analysis scope details (address, radius, etc.) ===
    # Get analysis scope (command-line args, batch mode, or interactive)
    if args.run_region and not args.states and not args.countries:
        # Single region from --run-region (args.states/countries not set means single region)
        # Reuse cached parsed_regions to avoid double prompting user
        if cached_parsed_regions is not None:
            parsed_regions = cached_parsed_regions
        else:
            from utils.region_detector import parse_regions

            parsed_regions = parse_regions(args.run_region)

        if len(parsed_regions) == 1:
            region = parsed_regions[0]
            if region["type"] == "state":
                # States are treated as regions
                scope_type = "region"
                canonical_path = region["canonical_name"]
                if "/" in canonical_path:
                    continent, subregion = canonical_path.split("/", 1)
                    # Format expected by data_coverage_checker: [(continent, (path,))]
                    location = [(continent, (canonical_path,))]
                else:
                    # Legacy format - treat as simple region
                    location = [canonical_path]
                radius_km = None
                print(f"Analysis scope: region ({canonical_path})")
            elif region["type"] == "country":
                scope_type = "country"
                location = [region["canonical_name"]]
                radius_km = None
                print(f"Analysis scope: {scope_type} ({region['canonical_name']})")
            elif region["type"] == "subregion":
                # Geofabrik subregion - convert to region scope format
                # canonical_name format: "continent/subregion" (e.g., "europe/andorra")
                scope_type = "region"
                canonical_path = region["canonical_name"]
                if "/" in canonical_path:
                    continent, subregion = canonical_path.split("/", 1)
                    # Format expected by data_coverage_checker: [(continent, (path,))]
                    location = [(continent, (canonical_path,))]
                    radius_km = None
                    print(f"Analysis scope: {scope_type} ({canonical_path})")
                else:
                    # Fallback - shouldn't happen but handle gracefully
                    scope_type = "country"
                    location = [region["canonical_name"]]
                    radius_km = None
                    print(f"Analysis scope: {scope_type} ({region['canonical_name']})")
            else:
                scope_type = "country"  # Default to country for unknown types
                location = [region["canonical_name"]]
                radius_km = None
                print(f"Analysis scope: {scope_type} ({region['canonical_name']})")
        else:
            # This shouldn't happen due to earlier logic, but handle it
            print("❌ Error: Multiple regions detected but not converted to batch mode")
            return
    elif args.states:
        # Command-line specified states (treated as regions)
        scope_type = "region"
        state_names = [s.strip() for s in args.states.split(",") if s.strip()]
        location = state_names
        radius_km = None
        print(f"Analysis scope: region ({', '.join(location)})")
    elif args.countries:
        # Command-line specified countries
        # But check if they're actually US states first
        location = [c.strip() for c in args.countries.split(",") if c.strip()]

        # Auto-detect if all entries are US states
        # Uses geo_lookup's is_us_state function
        us_states = []
        non_us_countries = []

        for loc in location:
            loc_is_state = False
            # Check if this is a US state using geo_lookup
            if is_us_state(loc):
                us_states.append(loc)
                loc_is_state = True
            if not loc_is_state:
                non_us_countries.append(loc)

        # Determine scope based on what we found
        if us_states and not non_us_countries:
            # All are US states - convert to region scope
            scope_type = "region"
            location = us_states
            radius_km = None
            print(f"Analysis scope: region (US States: {', '.join(location)})")
        elif non_us_countries and not us_states:
            # All are countries - keep country scope
            scope_type = "country"
            location = non_us_countries
            radius_km = None
            print(f"Analysis scope: country ({', '.join(location)})")
        else:
            # Mixed US states and countries - error
            print("❌ Error: Cannot mix US states and countries in same analysis")
            print(f"   US States detected: {', '.join(us_states)}")
            print(f"   Countries detected: {', '.join(non_us_countries)}")
            print(
                "   Please run separate analyses for states (--states) and countries (--countries)"
            )
            return
    elif args.batch:
        # Map "state" to "region" for batch mode
        scope_type = "region" if args.scope == "state" else args.scope
        # In batch mode, get location from config file
        if args.scope in ("state", "region"):
            try:
                config = get_config()
                osm_coverage = config.get("OSM_COVERAGE", [])
                if isinstance(osm_coverage, list) and osm_coverage:
                    location = osm_coverage  # Use the regions from config
                else:
                    location = [osm_coverage] if osm_coverage else []
                print(
                    f"Batch mode - Analysis scope: {scope_type} ({', '.join(location) if isinstance(location, list) else location})"
                )
            except Exception as e:
                print(f"⚠️  Could not read regions from config: {e}")
                location = []
        else:
            location = None
        radius_km = None
    else:
        # Use the cached result from earlier if we already got scope interactively
        if interactive_scope_result:
            scope_type, location, radius_km = interactive_scope_result
        else:
            # Shouldn't reach here, but handle it
            scope_type, location, radius_km = get_analysis_scope_choice(deployment_type)

    if scope_type == "address":
        # Get address from command-line or user input
        if args.address:
            address = args.address
            print(f"Address: {address}")
        else:
            from climb_analyzer.utils.formatting import print_header

            print_header("Street Address", spacing_before=1)
            address = input(
                "Enter the street address to analyze (include country if outside your region): "
            ).strip()
            if not address:
                print("Please enter a valid address.")
                return

        # No separate country prompt - let geocoding handle it
        country = None  # Will be auto-detected from address

        # For auto mode, detect country from address to determine appropriate radius unit
        is_us_address = None
        if unit_system == "auto":
            # Helper function for pattern-based US detection (no API call)
            def _detect_us_from_address_pattern(addr: str) -> Optional[bool]:
                """Quick US detection from common address patterns."""
                import re
                addr_lower = addr.lower()

                # US state abbreviations
                us_states = [
                    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
                    'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
                    'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
                    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
                    'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy', 'dc'
                ]

                # Check for ", XX" pattern (city, state abbreviation)
                state_match = re.search(r',\s*([a-z]{2})\b', addr_lower)
                if state_match and state_match.group(1) in us_states:
                    return True

                # Explicit country mentions
                if 'united states' in addr_lower or ', usa' in addr_lower:
                    return True

                return None  # Unknown - need API call

            # Try pattern detection first (instant, no network)
            pattern_result = _detect_us_from_address_pattern(address)
            if pattern_result is not None:
                is_us_address = pattern_result
                if is_us_address:
                    print("✓ US address detected - using imperial units")
                else:
                    print("✓ Non-US address detected - using metric units")
            else:
                # Fall back to Nominatim with retry
                from geopy.geocoders import Nominatim

                print("Detecting region from address...")
                for attempt in range(3):
                    try:
                        geolocator = Nominatim(user_agent="climb_analyzer", timeout=15)
                        location_result = geolocator.geocode(address, timeout=15)
                        if location_result:
                            is_us_address = "United States" in location_result.address
                            if is_us_address:
                                print("✓ US address detected - using imperial units")
                            else:
                                print("✓ Non-US address detected - using metric units")
                        break  # Success
                    except Exception as e:
                        if attempt < 2:
                            print(f"   Geocoding attempt {attempt + 1} failed, retrying...")
                            time.sleep(2 * (attempt + 1))  # 2s, 4s backoff
                        else:
                            print(f"Note: Could not auto-detect region ({e}), defaulting to metric")
                            is_us_address = False

        # Get search radius from command-line or user input
        if args.distance:
            radius = args.distance
            # Convert to km for internal processing based on unit system
            if unit_system == "metric":
                radius_km = radius
                print(f"Radius: {radius} kilometers")
            elif unit_system == "auto":
                # Use detected region to determine unit
                if is_us_address:
                    radius_km = radius * 1.60934
                    print(f"Radius: {radius} miles (US address)")
                else:
                    radius_km = radius
                    print(f"Radius: {radius} kilometers (non-US address)")
            else:  # imperial
                radius_km = radius * 1.60934
                print(f"Radius: {radius} miles")
        else:
            # Determine default unit for radius prompt based on unit system
            if unit_system == "metric":
                radius_unit = "kilometers"
            elif unit_system == "auto":
                # Use detected region: miles for US, kilometers for non-US
                radius_unit = "miles" if is_us_address else "kilometers"
            else:  # imperial
                radius_unit = "miles"

            # Loop until valid radius is entered (for cloud mode validation)
            # Set appropriate default value based on unit
            default_radius = "15" if radius_unit == "miles" else "24"
            while True:
                try:
                    radius_input = (
                        input(
                            f"Enter the search radius (default {default_radius} {radius_unit}): "
                        ).strip()
                        or default_radius
                    )
                    radius = float(radius_input)

                    # Convert to km for internal processing
                    if radius_unit == "miles":
                        radius_km = radius * 1.60934
                    else:
                        radius_km = radius

                    # Cloud mode: validate radius limit (allow small rounding tolerance)
                    if (
                        deployment_type == "cloud"
                        and scope_type == "address"
                        and radius_km > (CLOUD_MODE_MAX_RADIUS_KM + 0.5)
                    ):
                        miles_requested = radius_km / 1.60934
                        print("\n" + "=" * 70)
                        print("❌ ERROR: Search radius exceeds cloud mode limit")
                        print("=" * 70)
                        print(
                            f"\nRequested radius: {radius_km:.1f} km ({miles_requested:.1f} miles)"
                        )
                        print(
                            f"Cloud mode limit:  {CLOUD_MODE_MAX_RADIUS_KM:.1f} km ({CLOUD_MODE_MAX_RADIUS_MILES:.1f} miles)"
                        )
                        print("\nCloud mode uses Overpass API with the following limits:")
                        print("  • Query timeout: 180 seconds")
                        print("  • Memory limit: 1-2 GB response size")
                        print("  • Rate limiting: ~2 requests/second")
                        print(
                            f"\nSearches over {CLOUD_MODE_MAX_RADIUS_MILES:.0f} miles risk timeouts and failures."
                        )
                        print("\n" + "─" * 70)
                        print("OPTIONS:")
                        print("─" * 70)
                        print(
                            f"\n1. Enter a smaller radius ({CLOUD_MODE_MAX_RADIUS_MILES:.0f} {radius_unit} or less)"
                        )
                        print("\n2. Switch to local mode (RECOMMENDED for large areas)")
                        print("   Run: ./climb-analyzer setup")
                        print("\n   Local mode benefits:")
                        print("     ✓ No radius limits")
                        print("     ✓ 3-5x faster processing")
                        print("     ✓ No API rate limits")
                        print("     ✓ Works offline")
                        print("=" * 70 + "\n")
                        # Loop back to re-prompt
                        continue

                    # Valid radius - break out of loop
                    break

                except ValueError:
                    radius = 15
                    radius_km = 15 * 1.60934 if radius_unit == "miles" else 15
                    break

    # Default chunk size - optimized via comprehensive benchmarking
    # 30km chunks provide best performance: fewer chunks = less overhead, memory constant at ~4.6GB
    chunk_size_km = 30.0

    # Validate data coverage and auto-download if needed (only for local deployment)
    if deployment_type == "local":
        from climb_analyzer.data.data_coverage_checker import batch_mode_validate, validate_and_prepare_data

        # Determine address and radius for address-based searches
        if scope_type == "address":
            validation_address = address if "address" in locals() else None
            validation_radius = radius_km if "radius_km" in locals() else None
        else:
            validation_address = None
            validation_radius = None

        # Run validation (batch mode auto-downloads, interactive mode prompts)
        if args.batch:
            data_ready = batch_mode_validate(scope_type, location)
        else:
            data_ready = validate_and_prepare_data(
                scope_type, location, validation_address, validation_radius
            )

        if not data_ready:
            print("\n❌ Data validation failed. Cannot proceed with analysis.")
            print("Please ensure you have the required OSM and elevation data files.")
            sys.exit(1)

    # Check if this is tuple format (Geofabrik subregions) vs string format (US states)
    # Tuple format: [(continent, (path,))] - from -r "bristol" or hierarchical menu
    # String format: ["Vermont"] or ["VT", "NH"] - from legacy --states flag
    is_geofabrik_region = (
        scope_type == "region"
        and isinstance(location, list)
        and location
        and isinstance(location[0], tuple)
    )

    if scope_type == "region" and not is_geofabrik_region:
        # Handle multiple state analysis (US states only - string format)
        if isinstance(location, list) and len(location) > 1:
            print(f"\nMultiple state analysis: {', '.join(location)}")
            print("Processing each state separately...\n")

            # Initialize batch progress tracker
            from utils.batch_progress import BatchProgressTracker

            batch_tracker = BatchProgressTracker()
            batch_tracker.start_batch(
                locations=location,
                metadata={
                    "scope_type": scope_type,
                    "surface_filter": surface_filter,
                    "unit_system": unit_system,
                    "score_type": score_type,
                    "min_score": min_score,
                },
            )

            # Show progress summary
            batch_tracker.print_progress()

            # Process each state
            for i, state_name in enumerate(location, 1):
                # Skip if already completed
                if batch_tracker.is_completed(state_name):
                    print(f"\n=== [{i}/{len(location)}] {state_name} - ALREADY COMPLETED ===")
                    print(f"✓ Skipping {state_name} (completed in previous run)")
                    continue

                # Show failed status if applicable
                if batch_tracker.is_failed(state_name):
                    print(
                        f"\n=== [{i}/{len(location)}] {state_name} - RETRYING (previously failed) ==="
                    )

                try:
                    print(f"\n=== [{i}/{len(location)}] Analyzing {state_name} ===")
                    batch_tracker.mark_started(state_name)

                    center_lat, center_lon, radius_km = calculate_state_bounds(state_name)
                    address = f"{state_name} State Analysis"
                    country = "United States"
                    formatted_address = f"{state_name}, United States"
                    print(f"Calculated center: {center_lat:.4f}, {center_lon:.4f}")
                    print(f"Estimated coverage: {radius_km * 2:.0f}km diameter")

                    # Configure chunk size for this state if needed
                    state_chunk_size = 30.0  # Optimized default

                    # Run separate analysis for this state (always save to XLSX)
                    run_single_location_analysis(
                        state_name,
                        address,
                        country,
                        radius_km,
                        center_lat,
                        center_lon,
                        formatted_address,
                        surface_filter,
                        unit_system,
                        min_score,
                        state_chunk_size,
                        score_type,
                        enable_geocoding,
                        True,  # Always save to XLSX
                        "region",  # scope_type
                    )

                    # Mark as completed
                    batch_tracker.mark_completed(state_name)
                    print(f"\n✓ {state_name} completed successfully")

                    if i < len(location):
                        print(f"\n{'-'*60}\n")

                except ValueError as e:
                    error_msg = f"Error processing {state_name}: {e}"
                    print(error_msg)
                    batch_tracker.mark_failed(state_name, str(e))
                    continue
                except Exception as e:
                    error_msg = f"Unexpected error processing {state_name}: {e}"
                    print(error_msg)
                    batch_tracker.mark_failed(state_name, str(e))
                    import traceback

                    traceback.print_exc()
                    continue

            # Print final summary
            from climb_analyzer.utils.formatting import print_banner

            print_banner("Batch Analysis Complete", spacing_before=1)
            batch_tracker.print_progress()

            # Check if we should run cross-region climb analysis
            if len(location) >= 2:
                prompt_cross_region_analysis_simple(
                    location, batch_tracker, surface_filter, score_type, radius_km, "state"
                )

            return
        else:
            # Single state analysis
            if isinstance(location, list):
                location = location[0] if location else ""
            if not location:
                print("No state selected.")
                return

            try:
                center_lat, center_lon, radius_km = calculate_state_bounds(location)
                address = f"{location} State Analysis"
                country = "United States"
                formatted_address = f"{location}, United States"
                print(f"State analysis: {location}")
                print(f"Calculated center: {center_lat:.4f}, {center_lon:.4f}")
                print(f"Estimated coverage: {radius_km * 2:.0f}km diameter")
            except ValueError as e:
                print(f"Error: {e}")
                return

    elif scope_type == "country":
        # Handle multiple country analysis
        if isinstance(location, list) and len(location) > 1:
            print(f"\nMultiple country analysis: {', '.join(location)}")
            print("Processing each country separately...\n")

            # Initialize batch progress tracker
            from utils.batch_progress import BatchProgressTracker

            batch_tracker = BatchProgressTracker()
            batch_tracker.start_batch(
                locations=location,
                metadata={
                    "scope_type": scope_type,
                    "surface_filter": surface_filter,
                    "unit_system": unit_system,
                    "score_type": score_type,
                    "min_score": min_score,
                },
            )

            # Show progress summary
            batch_tracker.print_progress()

            # Process each country
            for i, country_name in enumerate(location, 1):
                # Skip if already completed
                if batch_tracker.is_completed(country_name):
                    print(f"\n=== [{i}/{len(location)}] {country_name} - ALREADY COMPLETED ===")
                    print(f"✓ Skipping {country_name} (completed in previous run)")
                    continue

                # Show failed status if applicable
                if batch_tracker.is_failed(country_name):
                    print(
                        f"\n=== [{i}/{len(location)}] {country_name} - RETRYING (previously failed) ==="
                    )

                try:
                    print(f"\n=== [{i}/{len(location)}] Analyzing {country_name} ===")
                    batch_tracker.mark_started(country_name)

                    center_lat, center_lon, radius_km = calculate_country_bounds(country_name)
                    address = country_name
                    country = None
                    formatted_address = country_name
                    print(f"Estimated coverage: {radius_km * 2:.0f}km diameter")

                    # Memory check for multi-country batch processing
                    if MEMORY_CHECK_AVAILABLE and deployment_type == "local":
                        from utils.memory_checker import (
                            check_memory_for_region,
                            print_memory_report,
                            print_split_guidance,
                        )

                        # Find the OSM file for this country
                        planet_dir = PLANET_OSM_DIR
                        osm_file_path = None
                        if planet_dir.exists():
                            # Try exact match first
                            pattern = f"{country_name.lower().replace(' ', '-')}-latest.osm.pbf"
                            osm_files = list(planet_dir.glob(pattern))
                            if not osm_files:
                                # Try with underscores
                                pattern = f"{country_name.lower().replace(' ', '_')}-latest.osm.pbf"
                                osm_files = list(planet_dir.glob(pattern))
                            if osm_files:
                                osm_file_path = osm_files[0]

                        memory_result = check_memory_for_region(country_name, osm_file_path)

                        print_memory_report(
                            memory_result["segments"],
                            memory_result["required_gb"],
                            memory_result["available_gb"],
                            memory_result["is_sufficient"],
                            memory_result["message"],
                        )

                        if not memory_result["is_sufficient"]:
                            print_split_guidance(memory_result)
                            confirm = (
                                input(f"\nContinue analyzing {country_name} anyway? (y/n): ")
                                .strip()
                                .lower()
                            )
                            if confirm != "y":
                                print(f"Skipping {country_name} due to insufficient memory.")
                                batch_tracker.mark_failed(country_name, "Insufficient memory")
                                continue

                    # Configure chunk size for this country if needed
                    country_chunk_size = 30.0  # Optimized default

                    # Run separate analysis for this country (always save to XLSX)
                    run_single_location_analysis(
                        country_name,
                        address,
                        country,
                        radius_km,
                        center_lat,
                        center_lon,
                        formatted_address,
                        surface_filter,
                        unit_system,
                        min_score,
                        country_chunk_size,
                        score_type,
                        enable_geocoding,
                        True,  # Always save to XLSX
                        "country",  # scope_type
                    )

                    # Mark as completed
                    batch_tracker.mark_completed(country_name)
                    print(f"\n✓ {country_name} completed successfully")

                    if i < len(location):
                        print(f"\n{'-'*60}\n")

                except ValueError as e:
                    error_msg = f"Error processing {country_name}: {e}"
                    print(error_msg)
                    batch_tracker.mark_failed(country_name, str(e))
                    continue
                except Exception as e:
                    error_msg = f"Unexpected error processing {country_name}: {e}"
                    print(error_msg)
                    batch_tracker.mark_failed(country_name, str(e))
                    import traceback

                    traceback.print_exc()
                    continue

            # Print final summary
            from climb_analyzer.utils.formatting import print_banner

            print_banner("Batch Analysis Complete", spacing_before=1)
            batch_tracker.print_progress()

            # Check if we should run cross-region climb analysis
            if len(location) >= 2:
                prompt_cross_region_analysis_simple(
                    location, batch_tracker, surface_filter, score_type, radius_km, "country"
                )

            return
        else:
            # Single country analysis
            if isinstance(location, list):
                location = location[0] if location else ""
            if not location:
                print("No country selected.")
                return

            try:
                center_lat, center_lon, radius_km = calculate_country_bounds(location)
                address = location
                country = None
                formatted_address = location
                print(f"Country analysis: {location}")
                print(f"Estimated coverage: {radius_km * 2:.0f}km diameter")

                # Memory check for country analysis (single country path)
                if MEMORY_CHECK_AVAILABLE and deployment_type == "local":
                    from utils.memory_checker import (
                        check_memory_for_region,
                        print_memory_report,
                        print_split_guidance,
                    )

                    # Find the OSM file for this country
                    planet_dir = PLANET_OSM_DIR
                    osm_file_path = None
                    if planet_dir.exists():
                        # Try exact match first
                        pattern = f"{location.lower().replace(' ', '-')}-latest.osm.pbf"
                        osm_files = list(planet_dir.glob(pattern))
                        if not osm_files:
                            # Try with underscores
                            pattern = f"{location.lower().replace(' ', '_')}-latest.osm.pbf"
                            osm_files = list(planet_dir.glob(pattern))
                        if osm_files:
                            osm_file_path = osm_files[0]

                    memory_result = check_memory_for_region(location, osm_file_path)

                    print_memory_report(
                        memory_result["segments"],
                        memory_result["required_gb"],
                        memory_result["available_gb"],
                        memory_result["is_sufficient"],
                        memory_result["message"],
                    )

                    if not memory_result["is_sufficient"]:
                        print_split_guidance(memory_result)
                        print("\nProceeding with analysis...")

            except ValueError as e:
                print(f"Error: {e}")
                return

    elif is_geofabrik_region:
        # Handle Geofabrik region-based analysis (from -r "bristol" or hierarchical menu)
        # location is a list of (continent, path) tuples
        if not isinstance(location, list) or not location:
            print("No regions selected.")
            return

        print(f"\nRegion-based analysis: {len(location)} region(s) selected")

        # Process single region
        if len(location) == 1:
            # Handle both tuple format and string format
            if isinstance(location[0], tuple):
                # Tuple format: (continent, path)
                continent, path = location[0]
                region_name = " > ".join([continent] + [p.split("/")[-1] for p in path])
                print(f"Analyzing region: {region_name}")
                region_display_name = path[-1].split("/")[-1] if path else continent
                # Keep full canonical path for unambiguous lookups (e.g., "us/georgia" not "georgia")
                region_canonical_path = path[-1] if path else region_display_name
            else:
                # String format: "Hawaii" or "north-america/us/hawaii"
                region_str = location[0]
                region_name = region_str  # Set region_name for later use
                print(f"Analyzing region: {region_str}")
                # Extract region name (last part of path)
                region_display_name = region_str.split("/")[-1]
                region_canonical_path = region_str
                # For lookups, we'll use geo_lookup to search osm_pbf_urls
                continent = None
                path = None

            # Normalize for dataset priority lookup (title case)
            # This will be used for FastElevationFetcher
            region_name_for_dataset = region_display_name.replace("-", " ").title()

            # Get bounds from osm_pbf_urls via geo_lookup
            try:
                # Normalize region name for lookup
                normalized_name = region_display_name.replace("-", " ").title()

                # Use full canonical path for bounds lookup to avoid ambiguity
                # (e.g., "us/georgia" resolves to US state, not "georgia" the country)
                found_bounds = lookup_bounds(region_canonical_path)

                if found_bounds:
                    # Bounds format: (lat_min, lon_min, lat_max, lon_max)
                    lat_min, lon_min, lat_max, lon_max = found_bounds
                    center_lat = (lat_min + lat_max) / 2
                    center_lon = (lon_min + lon_max) / 2
                    lat_diff = abs(lat_max - lat_min)
                    lon_diff = abs(lon_max - lon_min)

                    # Special handling for polar regions (Antarctica, Greenland)
                    # These wrap around longitude or span very wide longitude ranges
                    if normalized_name in ["Antarctica", "Greenland"] or lon_diff > 180:
                        # Use only latitude range for radius calculation
                        # Don't use longitude as it wraps around poles
                        radius_km = lat_diff * 111.32
                    else:
                        # Normal calculation for non-polar regions
                        radius_km = max(lat_diff, lon_diff) * 111.32
                else:
                    # Default fallback - assume moderate size region
                    print(
                        f"Warning: Could not find bounds for {region_display_name}, using default"
                    )
                    center_lat, center_lon, radius_km = 0.0, 0.0, 200.0

                address = region_name
                country = None
                formatted_address = region_name
                # Override region_name with normalized version for dataset priority
                region_name = region_name_for_dataset
                print(f"Estimated coverage: {radius_km * 2:.0f}km diameter")

                # Memory check for region analysis
                if MEMORY_CHECK_AVAILABLE and deployment_type == "local":
                    from utils.memory_checker import (
                        check_memory_for_region,
                        print_memory_report,
                        print_split_guidance,
                    )

                    # Find the OSM file for this region
                    planet_dir = PLANET_OSM_DIR
                    osm_file_path = None
                    if planet_dir.exists():
                        # Try various naming patterns (handle underscores too)
                        search_name = region_display_name.lower().replace(" ", "-").replace("_", "-")
                        patterns = [
                            f"{search_name}-latest.osm.pbf",
                            f"{search_name}.osm.pbf",
                            f"{search_name}_latest.osm.pbf",
                        ]
                        for pattern in patterns:
                            osm_files = list(planet_dir.glob(pattern))
                            if osm_files:
                                osm_file_path = osm_files[0]
                                break

                    memory_result = check_memory_for_region(region_display_name, osm_file_path)

                    print_memory_report(
                        memory_result["segments"],
                        memory_result["required_gb"],
                        memory_result["available_gb"],
                        memory_result["is_sufficient"],
                        memory_result["message"],
                    )

                    if not memory_result["is_sufficient"]:
                        print_split_guidance(memory_result)
                        print("\nProceeding with analysis...")

            except Exception as e:
                print(f"Error calculating region bounds: {e}")
                return

        else:
            # Multiple regions - process each separately
            print("Processing each region separately...")
            print("Note: Cross-region climb detection will be enabled")

            # Initialize batch progress tracker
            from utils.batch_progress import BatchProgressTracker

            batch_tracker = BatchProgressTracker()
            region_names = [
                " > ".join([cont] + [p.split("/")[-1] for p in path]) for cont, path in location
            ]
            batch_tracker.start_batch(
                locations=region_names,
                metadata={
                    "scope_type": scope_type,
                    "surface_filter": surface_filter,
                    "unit_system": unit_system,
                    "score_type": score_type,
                    "min_score": min_score,
                },
            )

            # Show progress summary
            batch_tracker.print_progress()

            # Cloud Cache: Check all regions before starting batch
            if CLOUD_CACHE_AVAILABLE and CLOUD_CACHE_ENABLED and deployment_type == "local":
                print("\n" + "=" * 80)
                print("Checking cloud cache for batch regions...")
                print("=" * 80)
                try:
                    cloud_cache = CloudCacheManager()
                    cached_regions = []

                    for continent, path in location:
                        region_display_name = path[-1].split("/")[-1] if path else continent

                        # Determine country/region for cache check
                        if scope_type == "region":
                            country_name = "United States"
                            region_name = region_display_name
                        elif scope_type == "country":
                            country_name = region_display_name
                            region_name = None
                        else:
                            continue  # Skip non-country/state scopes

                        cache_path = cloud_cache.get_cache_path(
                            country_name, region_name, scope_type
                        )
                        if cache_path:
                            cache_info = cloud_cache.check_cached(
                                country_name, region_name, scope_type
                            )
                            if cache_info["exists"]:
                                location_name = (
                                    region_name if scope_type == "region" else country_name
                                )
                                # Build version info string
                                version_info = ""
                                if cache_info.get("version"):
                                    version_info = f" v{cache_info['version']}"
                                if cache_info.get("error_count") is not None:
                                    version_info += f" - {cache_info['error_count']}e"

                                print(
                                    f"  ✓ {location_name:30s} available in cloud cache ({cache_info['date']}{version_info})"
                                )
                                cached_regions.append((continent, path, cache_info))
                            else:
                                location_name = (
                                    region_name if scope_type == "region" else country_name
                                )
                                print(f"  ✗ {location_name:30s} not in cloud cache")

                    # Offer to download cached regions
                    if cached_regions:
                        print(f"\nFound {len(cached_regions)} region(s) in cloud cache.")
                        response = (
                            input("Download cached analyses before batch? (y/n): ").strip().lower()
                        )
                        if response == "y":
                            for continent, path, cache_info in cached_regions:
                                region_display_name = path[-1].split("/")[-1] if path else continent
                                location_name = region_display_name
                                print(f"\nDownloading {location_name}...")
                                downloaded = cloud_cache.download_cache(cache_info, Path("output"))
                                if downloaded["xlsx_files"]:
                                    total_mb = (
                                        sum(f.stat().st_size for f in downloaded["xlsx_files"])
                                        / 1024**2
                                    )
                                    print(
                                        f"  ✓ Downloaded {len(downloaded['xlsx_files'])} file(s), {total_mb:.1f} MB"
                                    )
                            print("\n✓ Cached analyses downloaded to output/ directory")

                except Exception as e:
                    print(f"\n⚠️  Cloud cache check failed: {e}")
                    print("   Proceeding with batch analysis...")

                print("=" * 80 + "\n")

            # Process each region
            for i, (continent, path) in enumerate(location, 1):
                region_name = " > ".join([continent] + [p.split("/")[-1] for p in path])
                region_display_name = path[-1].split("/")[-1] if path else continent
                region_canonical_path = path[-1] if path else region_display_name

                # Skip if already completed
                if batch_tracker.is_completed(region_name):
                    print(f"\n=== [{i}/{len(location)}] {region_name} - ALREADY COMPLETED ===")
                    print(f"✓ Skipping {region_name} (completed in previous run)")
                    continue

                # Show failed status if applicable
                if batch_tracker.is_failed(region_name):
                    print(
                        f"\n=== [{i}/{len(location)}] {region_name} - RETRYING (previously failed) ==="
                    )

                try:
                    print(f"\n=== [{i}/{len(location)}] Analyzing {region_name} ===")
                    batch_tracker.mark_started(region_name)

                    # Get bounds for this region using full canonical path to avoid ambiguity
                    try:
                        bounds = lookup_bounds(region_canonical_path)
                        if bounds:
                            lat_min, lon_min, lat_max, lon_max = bounds
                            center_lat = (lat_min + lat_max) / 2
                            center_lon = (lon_min + lon_max) / 2
                            lat_diff = abs(lat_max - lat_min)
                            lon_diff = abs(lon_max - lon_min)
                            radius_km = max(lat_diff, lon_diff) * 111.32
                        else:
                            raise ValueError(
                                f"Could not find bounds for region: {region_display_name}"
                            )
                    except Exception as e:
                        print(f"Error: {e}")
                        batch_tracker.mark_failed(region_name, str(e))
                        continue

                    address = region_name
                    country = None
                    formatted_address = region_name
                    print(f"Estimated coverage: {radius_km * 2:.0f}km diameter")

                    # Configure chunk size for this region
                    region_chunk_size = 30.0  # Optimized default

                    # Run separate analysis for this region (always save to XLSX)
                    run_single_location_analysis(
                        region_name,
                        address,
                        country,
                        radius_km,
                        center_lat,
                        center_lon,
                        formatted_address,
                        surface_filter,
                        unit_system,
                        min_score,
                        region_chunk_size,
                        score_type,
                        enable_geocoding,
                        True,  # Always save to XLSX
                        "region",  # scope_type
                    )

                    # Mark as completed
                    batch_tracker.mark_completed(region_name)
                    print(f"\n✓ {region_name} completed successfully")

                    if i < len(location):
                        print(f"\n{'-'*60}\n")

                except ValueError as e:
                    error_msg = f"Error processing {region_name}: {e}"
                    print(error_msg)
                    batch_tracker.mark_failed(region_name, str(e))
                    continue
                except Exception as e:
                    error_msg = f"Unexpected error processing {region_name}: {e}"
                    print(error_msg)
                    batch_tracker.mark_failed(region_name, str(e))
                    import traceback

                    traceback.print_exc()
                    continue

            # Print final summary
            from climb_analyzer.utils.formatting import print_banner

            print_banner("Batch Analysis Complete", spacing_before=1)
            batch_tracker.print_progress()

            # Check if we should run cross-region climb analysis
            if len(location) >= 2:
                prompt_cross_region_analysis(
                    location, batch_tracker, surface_filter, score_type, radius_km
                )

            return

    # Configure chunk size for single location analysis

    max_workers = OSM_MAX_THREADS

    # Display analysis summary
    from climb_analyzer.utils.formatting import print_dim

    print("\nAnalysis criteria:")
    print(f"Location: {address}")
    if country:
        print(f"Country: {country}")
    print_dim(f"Surface filter: {surface_filter}")
    print_dim(f"Score type: {score_type}")
    print_dim(f"Search radius: {radius_km:.1f} km")
    print_dim(f"Minimum climb score: {min_score}")

    # Memory check for radius-based analysis (address searches only)
    if MEMORY_CHECK_AVAILABLE and deployment_type == "local" and scope_type == "address":
        # Ensure required variables are set for address searches
        if "center_lat" in locals() and "center_lon" in locals() and "radius_km" in locals():
            from utils.memory_checker import check_memory_for_radius, print_memory_report

            memory_result = check_memory_for_radius(
                center_lat, center_lon, radius_km, country if "country" in locals() else None
            )

            print_memory_report(
                memory_result["segments"],
                memory_result["required_gb"],
                memory_result["available_gb"],
                memory_result["is_sufficient"],
                memory_result["message"],
            )

            if not memory_result["is_sufficient"]:
                print("\n⚠️  INSUFFICIENT MEMORY FOR ANALYSIS")
                if memory_result["recommendations"]:
                    print("\nRecommendations:")
                    for rec in memory_result["recommendations"]:
                        print(f"  • {rec}")
                print("\nYou can:")
                print("  1. Reduce the search radius")
                print("  2. Free up system memory and try again")
                print("  3. Run the analysis on a system with more RAM")

                print("\nProceeding with analysis...")

    # Check OpenTopoData server is ready (for local mode)
    # This happens AFTER region selection and data confirmation
    if not check_opentopodata_ready():
        print("\nContinuing with external elevation API...")

    # Resolve "auto" unit system to actual units based on region
    if unit_system == "auto":
        is_us_region = False

        if scope_type == "address":
            # For address searches, use the geocoded result from earlier
            is_us_region = is_us_address if is_us_address is not None else False
            if is_us_region:
                unit_system = "imperial"
                print_dim("   Auto-detected unit system: imperial (US address)")
            else:
                unit_system = "metric"
                print_dim("   Auto-detected unit system: metric (non-US address)")
        elif scope_type == "region":
            # For region searches, check if it's a US state using geo_lookup
            # Get the region name to check
            region_to_check = None
            if isinstance(location, list) and location:
                if isinstance(location[0], tuple):
                    # Tuple format: (continent, path)
                    _, path = location[0]
                    region_to_check = path[-1].split("/")[-1] if path else None
                else:
                    # String format
                    region_to_check = location[0].split("/")[-1] if "/" in location[0] else location[0]
            elif isinstance(location, str):
                region_to_check = location.split("/")[-1] if "/" in location else location

            if region_to_check and is_us_state(region_to_check):
                is_us_region = True
                unit_system = "imperial"
                print_dim("   Auto-detected unit system: imperial (US state)")
            else:
                unit_system = "metric"
                print_dim("   Auto-detected unit system: metric (non-US region)")
        elif scope_type == "country":
            # For country searches, check if it's United States
            country_name = location[0] if isinstance(location, list) else location
            if country_name and country_name.lower() in ["united states", "united states of america", "usa", "us"]:
                is_us_region = True
                unit_system = "imperial"
                print_dim("   Auto-detected unit system: imperial (United States)")
            else:
                unit_system = "metric"
                print_dim("   Auto-detected unit system: metric (non-US country)")
        else:
            # Default fallback
            unit_system = "metric"
            print_dim("   Auto-detected unit system: metric (default)")

    # Cloud Cache: Check if analysis is already available
    if CLOUD_CACHE_AVAILABLE and CLOUD_CACHE_ENABLED and deployment_type == "local":
        try:
            cloud_cache = CloudCacheManager()

            # Determine variables for cloud cache check
            if is_geofabrik_region:
                # Geofabrik subregions (e.g., england/bristol) - skip cloud cache for now
                # Cloud cache is only set up for US states and countries
                country_name = None
                region_name = None
            elif scope_type == "region":
                # US states
                country_name = "United States"
                region_name = location if isinstance(location, str) else location[0]
            elif scope_type == "country":
                country_name = location if isinstance(location, str) else location[0]
                region_name = None
            else:
                # Address scope - skip cloud cache
                country_name = None
                region_name = None

            if country_name:
                cache_path = cloud_cache.get_cache_path(country_name, region_name, scope_type)

                if (
                    cache_path is None
                    and cloud_cache.is_usa(country_name)
                    and scope_type == "country"
                ):
                    # Full USA - too large
                    print("\n   Full USA analysis too large for cloud cache.")
                    print("   Analyze individual states instead to download/contribute.")
                elif cache_path:
                    # Check if cached
                    location_name = region_name if scope_type == "region" else country_name
                    print(f"\nChecking cloud cache for {location_name}...")
                    cache_info = cloud_cache.check_cached(country_name, region_name, scope_type)

                    if cache_info["exists"]:
                        # Build version info string
                        version_info = ""
                        if cache_info.get("version"):
                            version_info = f" v{cache_info['version']}"
                        if cache_info.get("error_count") is not None:
                            version_info += f" - {cache_info['error_count']} elevation errors"

                        print(
                            f"✓ {location_name} found in cloud cache (analyzed: {cache_info['date']}{version_info})"
                        )
                    else:
                        print(f"✗ {location_name} not found in cloud cache - will run analysis\n")

                    if cache_info["exists"]:
                        location_name = region_name if scope_type == "region" else country_name
                        if prompt_download_cache(
                            location_name, cache_info["date"], cache_info["total_parts"]
                        ):
                            print("\nDownloading cached analysis from GitHub...")
                            downloaded = cloud_cache.download_cache(cache_info, Path("output"))

                            if downloaded["xlsx_files"]:
                                total_mb = (
                                    sum(f.stat().st_size for f in downloaded["xlsx_files"])
                                    / 1024**2
                                )
                                print(
                                    f"\n✓ Download complete! {len(downloaded['xlsx_files'])} file(s), {total_mb:.1f} MB"
                                )
                                print("  Files saved to output/ directory")
                                sys.exit(0)  # Exit - analysis complete
        except Exception as e:
            print(f"\n⚠️  Cloud cache check failed: {e}")
            print("   Proceeding with local analysis...")

    # Run analysis
    try:
        # Set flag if cloud cache was already checked in main() to avoid duplicate checks
        cloud_cache_already_checked = (
            CLOUD_CACHE_AVAILABLE
            and CLOUD_CACHE_ENABLED
            and deployment_type == "local"
            and (scope_type in ["region", "country"])
        )

        climbs, df, persistence, should_upload_to_cache, error_logger = analyze_area(
            address=address,
            country=country,
            radius_km=radius_km,
            surface_filter=surface_filter,
            unit_system=unit_system,
            min_score=min_score,
            chunk_size_km=chunk_size_km,
            center_lat=center_lat if scope_type != "address" else None,
            center_lon=center_lon if scope_type != "address" else None,
            formatted_address=formatted_address if scope_type != "address" else None,
            score_type=score_type,
            cycling_only=cycling_only,
            enable_geocoding=enable_geocoding,
            max_workers=OSM_MAX_THREADS,
            skip_data_validation=True,  # Already validated in main() above
            scope_type=scope_type,  # Pass scope to enable optimized processing
            batch_mode=args.batch,  # Enable auto-resume in batch/CLI mode
            ignore_checkpoints=args.ignore_checkpoints,  # Start fresh, ignore existing checkpoints
            skip_cloud_cache_check=cloud_cache_already_checked,  # Don't check again if already checked in main()
        )

    except Exception as e:
        # Handle user cancellation gracefully
        if "cancelled by user" in str(e).lower():
            print(f"\n{e}")
            print("Exiting...")
            return

        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return

    total_duration = (time.time() - total_start_time) / 3600
    if total_duration >= 1:
        print(f"\nTOTAL PROCESSING TIME: {total_duration:.2f} hours")
    else:
        total_duration = total_duration * 60
        print(f"\nTOTAL PROCESSING TIME: {total_duration:.2f} minutes")

    # Create output folder if it doesn't exist (needed for both saving and error logging)
    output_dir = Path("output")
    try:
        output_dir.mkdir(exist_ok=True, mode=0o777)
        # Ensure the directory is writable
        import os

        os.chmod(output_dir, 0o777)
    except Exception as e:
        print(f"⚠️ Could not create/modify output folder: {e}")
        # Try /tmp as fallback
        import tempfile

        output_dir = Path(tempfile.gettempdir())
        print(f"   Using temporary directory instead: {output_dir}")

    # Create a safe filename (used in multiple places below)
    # Apply same formatting as streaming mode (lines 10325-10337) for consistent naming
    if scope_type == "address":
        safe_name = "".join(c for c in address if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_name = safe_name.replace(" ", "_")[:50]
    else:
        # Handle both string and list location formats
        # Extract just the region name (e.g., "california" from "us/california" or "us > california")
        if isinstance(location, list):
            # For region scope, location is a list of tuples: [("continent", ("path",))]
            # Extract readable names from the structure
            location_parts = []
            for item in location:
                if isinstance(item, tuple):
                    # Region scope format: ("europe", ("europe/andorra",))
                    continent, paths = item
                    if paths:
                        # Extract last part of path: "europe/andorra" -> "andorra"
                        last_path = paths[-1] if isinstance(paths, tuple) else paths
                        region_part = last_path.split("/")[-1] if "/" in last_path else last_path
                        location_parts.append(str(region_part))
                    else:
                        location_parts.append(str(continent))
                elif isinstance(item, list):
                    # Nested list - flatten it
                    location_parts.extend([str(x) for x in item])
                else:
                    # Simple string (state/country names)
                    location_parts.append(str(item))
            # Use only the last part (most specific region name)
            region_name = location_parts[-1] if location_parts else "unknown"
        else:
            region_name = str(location)

        # Extract just the region name from path formats like "us > california" or "us/california"
        if " > " in region_name:
            region_name = region_name.split(" > ")[-1]
        elif "/" in region_name:
            region_name = region_name.split("/")[-1]

        # Convert to title case for cleaner filenames (e.g., "california" -> "California")
        formatted_name = region_name.replace("-", " ").replace("_", " ").title()

        # Make safe for filenames
        safe_name = "".join(c for c in formatted_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_name = safe_name.replace(" ", "_")[:50]

    # Create consistent base filename with date (used in error logging and cloud cache)
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Build filter string (surface only - score type removed from filenames)
    filter_str = f"{surface_filter}"

    # Automatically save results to XLSX in output folder (MOVED BEFORE CLEANUP)
    # Check if df is a DataFrame (needs saving) or a list of files (already saved by streaming mode)
    import pandas as pd

    if df is not None and isinstance(df, list) and len(df) > 0:
        # Streaming mode already saved files - df is a list of created file paths
        created_files = df

        if len(created_files) == 1:
            print(f"Results saved to {created_files[0]}")
            filename = created_files[0]
        else:
            print(f"Results saved to {len(created_files)} files:")
            for file in created_files:
                print(f"   {file.name if hasattr(file, 'name') else Path(file).name}")
            filename = created_files[0]

        # Close the error logger from analyze_area (if it exists)
        if error_logger:
            error_logger.stop_elevation_logging(output_file_count=len(created_files))

    elif df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
        # Normal mode - df is a DataFrame that needs to be saved

        # Base filename format: <region>_climbs_<filters>_<date>
        base_filename = f"{safe_name}_climbs_{filter_str}_{date_str}"

        # Get elevation error count from stats collector
        elevation_errors = 0
        try:
            from utils.elevation_stats_collector import get_stats_collector, has_elevation_stats

            if has_elevation_stats():
                stats_collector = get_stats_collector()
                stats = stats_collector.get_stats()
                elevation_errors = stats.get("total_coords_failed", 0)
        except Exception:
            elevation_errors = 0

        # Save results (will split into multiple files if exceeds Excel limit)
        base_file = output_dir / base_filename
        created_files = save_large_dataframe_as_split_excel(
            df, base_file, app_version=__version__, elevation_errors=elevation_errors
        )

        if created_files:
            if len(created_files) == 1:
                print(f"Results saved to {created_files[0]}")
                filename = created_files[0]
            else:
                print(f"Results saved to {len(created_files)} files:")
                for file in created_files:
                    print(f"   {file.name}")
                filename = created_files[0]  # Set to first file for compatibility

            # Auto-merge split climbs if boundary merge was skipped
            # This happens when dataset is too large (>200K climbs) for in-memory merge
            if len(created_files) >= 2 and len(df) > 200000:
                try:
                    from utils.auto_merge_splits import merge_split_files

                    merge_split_files(created_files, auto_replace=True)
                except Exception as e:
                    print(f"\n⚠️  Auto-merge skipped: {e}")
                    print("   Files were still saved successfully")

            # Close the error logger from analyze_area (if it exists)
            if error_logger:
                error_logger.stop_elevation_logging(output_file_count=len(created_files))
        else:
            print("⚠️ Could not save results")
            print("   Results are still in memory and can be accessed programmatically")
            filename = None
            # Still close error logger even if save failed
            if error_logger:
                error_logger.stop_elevation_logging(output_file_count=0)
    else:
        # Check if this is a streaming export result
        # New format returns dict with xlsx, sqlite, climb_count
        # Legacy format returns list of Path objects
        sqlite_files = []
        sqlite_gz_files = []
        sqlite_gz_checksums = {}
        sqlite_decompressed_size = 0
        export_climb_count = 0

        if isinstance(df, dict) and "xlsx" in df:
            # New return format with xlsx, sqlite, sqlite_gz, and climb_count
            created_files = df["xlsx"]
            sqlite_files = df.get("sqlite", [])
            sqlite_gz_files = df.get("sqlite_gz", []) or []
            sqlite_gz_checksums = df.get("sqlite_gz_checksums", {}) or {}
            sqlite_decompressed_size = df.get("sqlite_decompressed_size", 0)
            export_climb_count = df.get("climb_count", 0)
        elif isinstance(df, list) and df and all(isinstance(f, Path) for f in df):
            # Legacy format (list of files)
            created_files = df
        else:
            created_files = []

        if created_files:
            # Streaming export already completed - files were saved in streaming mode
            if len(created_files) == 1:
                print(f"Results saved to {created_files[0].name} (streaming mode)")
                filename = created_files[0]
            else:
                print(f"Results saved to {len(created_files)} files (streaming mode):")
                for f in created_files:
                    print(f"   {f.name}")
                filename = created_files[0]  # Set to first file for compatibility

            # Also report SQLite files if generated
            if sqlite_files:
                print(f"SQLite database: {sqlite_files[0].name}")

            # Close error logger with correct file count
            if error_logger:
                error_logger.stop_elevation_logging(output_file_count=len(created_files))
        else:
            # Check if this was a clean analysis where output files were lost
            if should_upload_to_cache:
                from climb_analyzer.utils.formatting import print_separator, print_warning

                print()
                print_separator()
                print_warning("Analysis completed but output files not found")
                print()
                print("This was a clean analysis eligible for cloud cache upload.")
                print()
                print("💡 To generate output files and upload to cloud cache:")
                print("   1. Delete the checkpoint directory for this analysis:")
                print("      rm -rf data/checkpoint_data/<region>_*")
                print("   2. Re-run the analysis")
                print("   3. Output files will be generated and cloud cache upload will be offered")
                print_separator()
                print()
            else:
                print("No results to save (no climbs found above minimum score)")
            filename = None
            # Close error logger if no results
            if error_logger:
                error_logger.stop_elevation_logging(output_file_count=0)

    # Error reporting (elevation fetch statistics)
    # Use global stats collector for clean architecture
    from utils.elevation_stats_collector import (
        get_stats_collector,
        get_way_failures,
        has_elevation_stats,
    )

    if has_elevation_stats():
        # Use same base_filename format as streaming mode: <region>_errors_<date>
        # (no filter_str - matches xlsx naming pattern)
        error_base_filename = f"{safe_name}_errors_{date_str}"

        error_logger = ErrorLogger(
            region_name="Analysis Results",
            surface_filter=surface_filter,
            min_score=min_score,
            score_type=score_type,
            cycling_allowed=cycling_only,
            output_dir=output_dir,
            base_filename=error_base_filename,
            app_version=__version__,
        )
        stats_collector = get_stats_collector()
        stats = stats_collector.get_stats()
        way_failures = get_way_failures()

        # Print error report to console with per-way failures
        way_pct, country_pct = error_logger.print_error_report(
            total_coords_in_ways=stats["total_coords_requested"],
            failed_coords=stats["total_coords_failed"],
            total_coords_in_country=stats["total_unique_coords"],
            way_failures=way_failures,
        )

        # Log to error.log with report filename and per-way failures
        if filename:
            report_name = filename.name if hasattr(filename, "name") else str(filename)
            error_logger.log_elevation_errors(
                report_filename=report_name,
                total_coords_in_ways=stats["total_coords_requested"],
                failed_coords=stats["total_coords_failed"],
                total_coords_in_country=stats["total_unique_coords"],
                way_percentage=way_pct,
                country_percentage=country_pct,
                way_failures=way_failures,
            )

    # Cloud Cache: Upload completed clean analysis
    if CLOUD_CACHE_AVAILABLE and CLOUD_CACHE_ENABLED and should_upload_to_cache:
        try:
            cloud_cache = CloudCacheManager()

            # Determine variables for cloud cache upload
            # Use geo_lookup to find the region's parent continent/country
            if scope_type == "region":
                # Handle different location formats:
                # - String: "bristol" or "england > bristol"
                # - List with string: ["england > bristol"]
                # - List with tuple: [(continent, [path, parts])]
                if isinstance(location, str):
                    region_name = location
                elif isinstance(location, (list, tuple)) and len(location) > 0:
                    if isinstance(location[0], tuple):
                        # Tuple format: (continent, path_list)
                        _, path_parts = location[0]
                        region_name = path_parts[-1] if path_parts else ""
                    else:
                        # String in list/tuple
                        region_name = str(location[0])
                else:
                    region_name = ""

                # Extract the base region name (handle hierarchical paths)
                if isinstance(region_name, str):
                    # Handle display format "england > bristol"
                    if " > " in region_name:
                        region_name = region_name.split(" > ")[-1]
                    # Handle path format "europe/united-kingdom/england/bristol"
                    elif "/" in region_name:
                        region_name = region_name.split("/")[-1]

                # Determine country_name based on region type
                # For US states: set country to USA explicitly
                # For non-US regions: country_name can be None, get_cache_path uses find_region()
                if is_us_state(region_name):
                    country_name = "United States of America"
                else:
                    # Non-US regions (like Bristol) - get_cache_path will extract
                    # the full hierarchical path from find_region(region).pbf_url
                    country_name = None
            elif scope_type == "country":
                country_name = location if isinstance(location, str) else location[0]
                region_name = None
            else:
                country_name = None
                region_name = None

            if country_name or region_name:
                cache_path = cloud_cache.get_cache_path(country_name, region_name, scope_type)

                if cache_path:
                    is_clean = cloud_cache.is_clean_analysis(
                        surface_filter, min_score, cycling_only
                    )

                    if is_clean and scope_type in ["country", "region"]:
                        # Use created_files directly if available (already have the actual files)
                        # This avoids pattern-matching issues with hierarchical region names
                        if 'created_files' in dir() and created_files:
                            output_files = {
                                "xlsx": [f for f in created_files if str(f).endswith('.xlsx')],
                                "sqlite": sqlite_files if 'sqlite_files' in dir() and sqlite_files else [],
                                "sqlite_gz": sqlite_gz_files if 'sqlite_gz_files' in dir() and sqlite_gz_files else [],
                                "sqlite_gz_checksums": sqlite_gz_checksums if 'sqlite_gz_checksums' in dir() else {},
                                "sqlite_decompressed_size": sqlite_decompressed_size if 'sqlite_decompressed_size' in dir() else 0,
                                "csv": None,
                                "climb_count": export_climb_count if 'export_climb_count' in dir() else 0,
                            }

                            # Search for error log file in output directory
                            # Error files use different naming patterns, so search broadly
                            error_patterns = [
                                f"*{safe_name}*_errors_*.txt",
                                f"*{safe_name.lower().replace(' ', '_')}*_errors_*.txt",
                                f"*{safe_name.lower().replace(' ', '-')}*_errors_*.txt",
                                f"us__*_errors_{filter_str}_{date_str}.txt",
                            ]
                            for pattern in error_patterns:
                                matches = list(output_dir.glob(pattern))
                                if matches:
                                    # Use most recent error file
                                    output_files["csv"] = max(matches, key=lambda p: p.stat().st_mtime)
                                    break
                        else:
                            # Fallback to pattern search
                            output_files = find_analysis_output_files(
                                Path("output"), safe_name, surface_filter, score_type, date_str
                            )

                        if output_files["xlsx"] or output_files["csv"]:
                            file_count = len(output_files["xlsx"]) + (
                                1 if output_files["csv"] else 0
                            )

                            # Auto-upload by default unless --no-cloud-upload flag is set
                            should_upload = not args.no_cloud_upload

                            if should_upload:
                                from climb_analyzer.utils.formatting import (
                                    print_header,
                                    print_info,
                                    print_success,
                                    print_warning,
                                )

                                print_header("Uploading to Cloud Cache", spacing_before=2)
                                print_info("Creating GitHub release with your analysis...", indent=0)

                                # Load datasets_used from persistence if available
                                datasets_used = None
                                if persistence:
                                    datasets_used = persistence.load_datasets_used()

                                release_url = cloud_cache.upload_to_release(
                                    country_name, region_name, output_files, scope_type, datasets_used,
                                    climb_count=output_files.get("climb_count", 0)
                                )

                                if release_url:
                                    total_mb = (
                                        sum(f.stat().st_size for f in output_files["xlsx"])
                                        / 1024**2
                                    )
                                    # Add SQLite files to total
                                    if output_files.get("sqlite"):
                                        total_mb += sum(f.stat().st_size for f in output_files["sqlite"]) / 1024**2
                                    if output_files.get("csv"):
                                        total_mb += output_files["csv"].stat().st_size / 1024**2

                                    print_header(
                                        "PR Created for Review", spacing_before=2
                                    )
                                    print_success(f"PR/Release URL: {release_url}")
                                    print_success(
                                        f"Total uploaded: {total_mb:.1f} MB across {file_count} files"
                                    )
                                    print_success("Thank you for contributing your time and data to the community!")
                                else:
                                    print()  # Empty line for spacing
                                    print(
                                        "  ℹ️  No upload performed. Your local analysis is saved in output/"
                                    )
        except Exception as e:
            from climb_analyzer.utils.formatting import print_warning

            print_warning(f"Cloud cache upload failed: {e}")
            print_warning("Your local analysis is still saved in output/")

    # Per-region cleanup (new -c and -Z flags)
    # -c: Delete only checkpoints for this region
    # -Z: Delete checkpoints + OSM + elevation for this region
    cleanup_checkpoints_flag = getattr(args, 'cleanup_checkpoints', False)
    cleanup_all_data_flag = getattr(args, 'cleanup_all_data', False)

    if cleanup_all_data_flag and region_name:
        # Full cleanup: checkpoints + OSM + elevation for this region only
        print(f"\nPerforming per-region cleanup for: {region_name}")
        delete_region_data(
            region_name,
            delete_checkpoints=True,
            delete_osm=True,
            delete_elevation=True,
            rebuild_opentopodata=True
        )
        cleanup_choice = "3"  # Already cleaned up, don't do additional cleanup
    elif cleanup_checkpoints_flag and region_name:
        # Checkpoint-only cleanup for this region
        print(f"\nDeleting checkpoints for: {region_name}")
        delete_region_data(
            region_name,
            delete_checkpoints=True,
            delete_osm=False,
            delete_elevation=False,
            rebuild_opentopodata=False
        )
        cleanup_choice = "3"  # Already cleaned up
    elif cleanup_after_analysis:
        # Legacy batch mode cleanup (for backward compatibility with batch_mode_menu)
        print("\nPerforming batch mode cleanup...")
        perform_cleanup(cleanup_targets, verbose=True)
        cleanup_choice = "3"
    elif args.batch:
        # Batch mode - default to keeping checkpoints (use -c or -Z for cleanup)
        cleanup_choice = "3"  # Keep checkpoints (default for batch mode)
    else:
        # Interactive mode - default to keeping checkpoints
        print("\n✓ Checkpoint files preserved for potential resume")
        print("  Use -c/--cleanup-checkpoints or -Z/--cleanup-all-data for per-region cleanup")
        cleanup_choice = "3"  # Default: keep checkpoints

    if cleanup_choice == "1":
        persistence.cleanup()
    elif cleanup_choice == "2":
        # Clean up all files inside checkpoint directories, but keep the base directory
        try:
            base_dir = CHECKPOINT_DIR
            if base_dir.exists():
                for item in base_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                print("Cleaned up all files in checkpoint directories")
            else:
                print("No checkpoint directory found")
        except Exception as e:
            print(f"Error cleaning up checkpoint contents: {e}")

    # Restore original stdout and stderr
    if isinstance(sys.stdout, Tee):
        sys.stdout = sys.__stdout__
    if isinstance(sys.stderr, Tee):
        sys.stderr = sys.__stderr__
    if "log_file" in locals() and log_file:
        log_file.close()


if __name__ == "__main__":
    main()
