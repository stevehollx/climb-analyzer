#!/usr/bin/env python3
"""
Data validation for local deployment.

Checks if required OSM + DEM data exists before analysis.
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
from difflib import SequenceMatcher

from utils.dataset_selector import get_required_datasets
from climb_analyzer.data.geo_lookup import is_us_state


def _validate_region_match(region_name: str, filename: str, min_similarity: float = 0.6) -> bool:
    """
    Validate that a filename actually matches the region being searched.

    Only exact normalized matches are accepted. Substring containment ("kansas"
    inside "arkansas") is EXPLICITLY rejected — it caused wrong-region selection
    that wasted hours of analysis time on the wrong data.

    Args:
        region_name: Region being searched (e.g., "France", "New York")
        filename: OSM filename (e.g., "france-latest.osm.pbf", "hawaii-latest.osm.pbf")
        min_similarity: Retained for signature compatibility; no longer used.

    Returns:
        True if filename exactly matches region_name (after whitespace/delim
        normalization), False otherwise.
    """
    # Normalize both strings for comparison
    region_normalized = region_name.lower().replace(" ", "").replace("-", "").replace("_", "")

    # Extract the region part from filename (before "-latest" or ".osm.pbf")
    filename_base = filename.lower().replace(".osm.pbf", "").replace("-latest", "")
    filename_normalized = filename_base.replace(" ", "").replace("-", "").replace("_", "")

    # Strip known disambiguation prefixes (e.g., "us_georgia" -> "georgia",
    # "europe_georgia" -> "georgia") so both forms exact-match correctly.
    for prefix in ("us", "europe", "asia", "africa", "oceania", "southamerica", "northamerica"):
        if filename_normalized.startswith(prefix) and len(filename_normalized) > len(prefix):
            candidate = filename_normalized[len(prefix):]
            if candidate == region_normalized:
                return True

    return region_normalized == filename_normalized


def find_osm_file_for_region(region_name: str, canonical_path: Optional[str] = None) -> Path:
    """
    Find OSM .pbf file for a region.

    Prioritizes merged files (created for cross-border searches) over individual region files.

    For ambiguous regions (e.g., Georgia the US state vs Georgia the country),
    pass canonical_path to disambiguate (e.g., "us/georgia" vs "europe/georgia").

    Args:
        region_name: Name of region/country (or US state name)
                    Can be:
                    - Simple name: "liechtenstein"
                    - Hierarchical path: "europe/liechtenstein"
                    - Nested path: "north-america/us/california"
        canonical_path: Optional full canonical path for disambiguation

    Returns:
        Path to OSM file, or None if not found
    """
    planet_dir = Path("data/planet_osm_data")
    if not planet_dir.exists():
        return None

    # If a canonical_path is provided separately, use it for disambiguation first
    if canonical_path and '/' in canonical_path:
        try:
            from climb_analyzer.data.osm_downloader import AMBIGUOUS_STEMS
            path_parts = canonical_path.split('/')
            stem_name = path_parts[-1].lower().replace(" ", "-").replace("_", "-")
            if stem_name in AMBIGUOUS_STEMS and len(path_parts) >= 2:
                parent = path_parts[-2]
                disambiguated = planet_dir / f"{parent}_{stem_name}-latest.osm.pbf"
                if disambiguated.exists():
                    return disambiguated
                # Specific file missing - don't fall through to ambiguous match
                return None
        except ImportError:
            pass

    # Extract the actual region name from hierarchical path FIRST
    # e.g., "europe/liechtenstein" -> "liechtenstein"
    # e.g., "north-america/us/california" -> "california"
    if '/' in region_name:
        region_parts = region_name.split('/')
        # Use the last part as the actual region name
        actual_region_name = region_parts[-1]
        # For ambiguous regions (e.g., "us/georgia" vs "europe/georgia"),
        # check for the disambiguated filename first
        try:
            from climb_analyzer.data.osm_downloader import AMBIGUOUS_STEMS
            normalized = actual_region_name.lower().replace(" ", "-").replace("_", "-")
            if normalized in AMBIGUOUS_STEMS and len(region_parts) >= 2:
                parent = region_parts[-2]
                disambiguated = planet_dir / f"{parent}_{normalized}-latest.osm.pbf"
                if disambiguated.exists():
                    return disambiguated
                # Don't fall through to ambiguous lookup if the specific file
                # isn't present - we'd risk returning the wrong one
                return None
        except ImportError:
            pass
    else:
        actual_region_name = region_name

    # Normalize region name for matching (handle spaces, underscores -> hyphens)
    normalized_region = actual_region_name.lower().replace(" ", "-").replace("_", "-")

    # PRIORITY 1: Check for merged files that contain this specific region
    # Merged files are created when search area spans multiple regions
    # Format: merged-<region1>-<region2>-<region3>-latest.osm.pbf
    # Example: merged-north-carolina-south-carolina-tennessee-latest.osm.pbf
    # IMPORTANT: Only use merged file if it actually contains the region we're searching
    merged_files = list(planet_dir.glob("merged-*.osm.pbf"))
    if merged_files:
        # Check if any merged file contains this region in its name
        for merged_file in merged_files:
            # Extract region names from merged filename
            # merged-north-carolina-south-carolina-tennessee-latest.osm.pbf
            # -> check if "georgia" is in the name
            filename_lower = merged_file.stem.lower()
            if filename_lower.startswith("merged-") and filename_lower.endswith("-latest"):
                # Remove "merged-" prefix and "-latest" suffix
                regions_part = filename_lower[7:-7]  # Remove "merged-" and "-latest"

                # Check if our normalized region appears in the merged filename
                # For "georgia" searching in "north-carolina-south-carolina-tennessee":
                #   - Check if "georgia" is in the full string (handles multi-word regions)
                #   - Use word boundary check: ensure it's a complete region name
                # Split on hyphens to get individual words, then reconstruct possible region names
                # E.g., "north-carolina-south-carolina" -> ["north", "carolina", "south", "carolina"]
                # Then check combinations: "north", "carolina", "north-carolina", "south", "south-carolina"

                # Simple approach: check if normalized_region appears as a complete segment
                # Split the regions_part by known delimiters and check
                region_segments = regions_part.split("-")

                # Build all possible multi-word combinations (up to 3 words)
                possible_regions = []
                for i in range(len(region_segments)):
                    # Single word
                    possible_regions.append(region_segments[i])
                    # Two words
                    if i + 1 < len(region_segments):
                        possible_regions.append(f"{region_segments[i]}-{region_segments[i+1]}")
                    # Three words
                    if i + 2 < len(region_segments):
                        possible_regions.append(f"{region_segments[i]}-{region_segments[i+1]}-{region_segments[i+2]}")

                # Check if our region matches any of these combinations
                if normalized_region in possible_regions:
                    return merged_file

        # No merged file contains this region - continue to individual files

    # PRIORITY 2: Check if this is a US state
    if is_us_state(actual_region_name):
        # Look for state-specific file first (e.g., hawaii-latest.osm.pbf).
        # Require exact stem match: "kansas-latest" or "us_kansas-latest".
        # Substring matching caused "kansas" to pick up "arkansas-latest.osm.pbf".
        # NOTE: Path.stem only strips one extension; for "kansas-latest.osm.pbf"
        # it returns "kansas-latest.osm", so we strip ".osm.pbf" from the name.
        state_normalized = actual_region_name.lower().replace(" ", "-").replace("_", "-")
        expected_stems = {
            f"{state_normalized}-latest",
            f"us_{state_normalized}-latest",
        }
        for pbf_file in planet_dir.glob("*.osm.pbf"):
            stem_lower = pbf_file.name.lower().replace(".osm.pbf", "")
            if stem_lower.startswith("merged-"):
                continue
            if stem_lower in expected_stems:
                return pbf_file

        # If no state file, look for whole United States OSM file
        us_patterns = ["united-states", "us-latest", "usa"]
        for pbf_file in planet_dir.glob("*.osm.pbf"):
            stem_lower = pbf_file.stem.lower()
            if stem_lower.startswith("merged-"):
                continue
            if any(pattern in stem_lower for pattern in us_patterns):
                return pbf_file

        # No US or state file found
        return None

    # PRIORITY 3: Exact match for country/region name
    # Stem must exactly equal "<region>-latest" (with optional parent prefix
    # like "canada_british-columbia-latest"). Substring match is forbidden —
    # e.g., "mexico" must not match "new-mexico-latest.osm.pbf".
    normalized_name = actual_region_name.lower().replace(" ", "-").replace("_", "-")
    exact_stems = {f"{normalized_name}-latest"}
    for pbf_file in planet_dir.glob("*.osm.pbf"):
        stem_lower = pbf_file.name.lower().replace(".osm.pbf", "")
        if stem_lower.startswith("merged-"):
            continue
        if stem_lower in exact_stems:
            return pbf_file
        # Allow parent-prefixed form like "canada_yukon-latest" or
        # "europe_switzerland-latest" when validation confirms exact region match.
        if "_" in stem_lower and stem_lower.endswith(f"_{normalized_name}-latest"):
            return pbf_file

    # No match found
    return None


def validate_local_data(
    region_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    deployment_type: str = 'local'
) -> Tuple[bool, List[str]]:
    """
    Validate that all required data exists for local deployment.

    Args:
        region_name: Name of region (e.g., "Switzerland")
        lat_min, lat_max, lon_min, lon_max: Bounding box
        deployment_type: 'local' or 'cloud' (default: 'local')

    Returns:
        (is_valid, list_of_missing_items)
    """
    missing = []

    # Check OSM .pbf file
    osm_file = find_osm_file_for_region(region_name)
    if not osm_file or not osm_file.exists():
        # Provide helpful message for US states
        if is_us_state(region_name):
            missing.append(f"OSM data file (.pbf) for United States (needed for {region_name})")
        else:
            missing.append(f"OSM data file (.pbf) for {region_name}")

    # Check spatial index
    if osm_file:
        from climb_analyzer.data.spatial_index import SpatialIndexManager
        index_mgr = SpatialIndexManager(str(osm_file), cache_dir="data/osm_indexes")
        if not index_mgr.exists():
            missing.append(f"Spatial index for {osm_file.name}")

    # Check DEM data - check for actual file patterns, not subdirectories
    required_datasets = get_required_datasets(lat_min, lat_max, lon_min, lon_max, deployment_type)
    elevation_dir = Path("data/elevation_data")

    for dataset in required_datasets:
        dataset_found = False

        if dataset == 'srtm30m':
            # SRTM files are stored as N##E###.hgt directly in elevation_data/
            srtm_files = list(elevation_dir.glob("*.hgt"))
            dataset_found = len(srtm_files) > 0

        elif dataset == 'aw3d30':
            # AW3D30 files are stored in subdirectories like N045E005/, N050E005/
            aw3d_dirs = [d for d in elevation_dir.iterdir() if d.is_dir() and d.name.startswith('N')]
            dataset_found = len(aw3d_dirs) > 0

        elif dataset == 'aster':
            # ASTER files are stored as ASTGTMV003_*.tif directly in elevation_data/
            aster_files = list(elevation_dir.glob("ASTGTMV003_*_dem.tif"))
            dataset_found = len(aster_files) > 0

        elif dataset == 'ned10m':
            # NED files - check for common patterns
            ned_files = list(elevation_dir.glob("*ned*.tif")) + list(elevation_dir.glob("*NED*.tif"))
            dataset_found = len(ned_files) > 0

        elif dataset == 'arcticdem':
            # ArcticDEM files
            arctic_files = list(elevation_dir.glob("*arctic*.tif"))
            dataset_found = len(arctic_files) > 0

        elif dataset == 'rema':
            # REMA files
            rema_files = list(elevation_dir.glob("*rema*.tif"))
            dataset_found = len(rema_files) > 0

        if not dataset_found:
            missing.append(f"DEM dataset: {dataset}")

    return (len(missing) == 0, missing)


def ensure_data_available(
    region_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> bool:
    """
    Ensure all required data is available. Automatically downloads if missing.

    This is a wrapper around validate_and_prepare_data() for backward compatibility.

    Args:
        region_name: Name of region
        lat_min, lat_max, lon_min, lon_max: Bounding box

    Returns:
        True if data is available (or successfully downloaded)
    """
    from climb_analyzer.data.data_coverage_checker import validate_and_prepare_data

    # Determine scope type based on region_name
    # Check if it's a US state (treated as region)
    if is_us_state(region_name):
        scope_type = "region"
    else:
        scope_type = "country"

    # Use the new auto-download logic
    return validate_and_prepare_data(scope_type, region_name)


# Test if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        region = sys.argv[1]
        # Test bounds (approximate)
        is_valid, missing = validate_local_data(region, 40, 50, 5, 15)

        if is_valid:
            print(f"✓ All data available for {region}")
        else:
            print(f"❌ Missing data for {region}:")
            for item in missing:
                print(f"  • {item}")
    else:
        print("Usage: python data_validator.py <region_name>")
        print("\nExample:")
        print("  python data_validator.py Switzerland")
