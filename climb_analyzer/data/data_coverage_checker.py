#!/usr/bin/env python3
"""
Data Coverage Checker and Auto-Downloader

Validates that required OSM planet files and elevation tiles are available
for the user's selected analysis region. Automatically downloads missing data.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from climb_analyzer.data.geo_lookup import find_region, get_region_bounds, is_us_state
from climb_analyzer.utils.formatting import print_header


def get_bbox_for_scope(
    scope_type: str, location: any
) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box for the analysis scope.

    Args:
        scope_type: "address", "region", or "country"
        location: Location identifier (region name, country name, or list of (continent, path) tuples)

    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon) or None if not found
    """
    if scope_type == "address":
        # For address, we'll need to geocode first - return None to signal this
        return None

    elif scope_type == "region":
        # Handle regions (including US states)
        # Regions can be passed as simple names or as tuples
        if isinstance(location, list) and location and isinstance(location[0], str):
            # Legacy state format or simple region names
            locations = location if isinstance(location, list) else [location]

            # Get bboxes for all states/regions and merge them
            min_lat, min_lon = float("inf"), float("inf")
            max_lat, max_lon = float("-inf"), float("-inf")

            for loc in locations:
                # Use geo_lookup to find the region
                bounds = get_region_bounds(loc)
                if bounds:
                    min_lat = min(min_lat, bounds[0])
                    min_lon = min(min_lon, bounds[1])
                    max_lat = max(max_lat, bounds[2])
                    max_lon = max(max_lon, bounds[3])

            if min_lat != float("inf"):
                return (min_lat, min_lon, max_lat, max_lon)
            return None

        else:
            # Region selection - location is a list of (continent, path) tuples
            if not isinstance(location, list) or not location:
                return None

            # Get bboxes for all selected regions and merge them
            min_lat, min_lon = float("inf"), float("inf")
            max_lat, max_lon = float("-inf"), float("-inf")

            for continent, path in location:
                # Use the full path for region lookup to avoid ambiguous names
                # e.g., "us/georgia" should find US state, not "georgia" the country
                if isinstance(path, (list, tuple)) and path:
                    # path is a tuple like ("us/georgia",) - use the full path
                    region_name = path[-1] if path else continent
                else:
                    region_name = continent

                # Use geo_lookup to find bounds - use full path for disambiguation
                bounds = get_region_bounds(region_name)

                if bounds:
                    # Validate bbox is not a point (some regions have bad data)
                    lat_span = abs(bounds[2] - bounds[0])
                    lon_span = abs(bounds[3] - bounds[1])
                    if lat_span > 0.01 and lon_span > 0.01:  # At least ~1km span
                        min_lat = min(min_lat, bounds[0])
                        min_lon = min(min_lon, bounds[1])
                        max_lat = max(max_lat, bounds[2])
                        max_lon = max(max_lon, bounds[3])
                    else:
                        print(f"⚠️  Warning: Region bounds for {region_name} are too small (point)")
                else:
                    print(f"⚠️  Warning: No bounds found for region {region_name}")

            if min_lat != float("inf"):
                return (min_lat, min_lon, max_lat, max_lon)
            return None

    elif scope_type == "country":
        # Handle both single country and multiple countries
        locations = location if isinstance(location, list) else [location]

        # Get bboxes for all countries and merge them
        min_lat, min_lon = float("inf"), float("inf")
        max_lat, max_lon = float("-inf"), float("-inf")

        for loc in locations:
            # Use geo_lookup to find the region bounds
            bounds = get_region_bounds(loc)
            if bounds:
                min_lat = min(min_lat, bounds[0])
                min_lon = min(min_lon, bounds[1])
                max_lat = max(max_lat, bounds[2])
                max_lon = max(max_lon, bounds[3])

        if min_lat != float("inf"):
            return (min_lat, min_lon, max_lat, max_lon)
        return None

    return None


def check_osm_coverage(
    bbox: Tuple[float, float, float, float],
    scope_type: str,
    location: any,
    center_point: Optional[Tuple[float, float]] = None,
) -> Tuple[bool, str]:
    """
    Check if we have OSM planet file coverage for the bounding box.

    Args:
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        scope_type: "address", "region", or "country"
        location: Location name(s)
        center_point: Optional (lat, lon) center for address searches

    Returns:
        (has_coverage: bool, message: str)
    """
    planet_dir = Path("data/planet_osm_data")

    # Check if planet directory exists
    if not planet_dir.exists():
        return False, "No data/planet_osm_data directory found. Please create one and add OSM files."

    # Look for .osm.pbf files
    osm_files = list(planet_dir.glob("*.osm.pbf"))

    if not osm_files:
        return False, "No OSM planet files (.osm.pbf) found in data/planet_osm_data directory."

    # For address-based searches, check if existing files cover the bbox
    if scope_type == "address":
        from utils.region_mapper import find_regions_for_bbox

        # Find which regions are needed for this bbox, passing center point
        # to ensure we include region containing the address
        needed_regions = find_regions_for_bbox(
            bbox, prefer_smallest=True, center_point=center_point
        )

        if not needed_regions:
            return False, "Could not determine which regions are needed for this area"

        # Check if we have files for all needed regions
        missing_regions = []
        found_regions = []

        for region_path, region_url, region_size in needed_regions:
            # Extract the expected filename from the URL
            expected_filename = region_url.split("/")[-1]

            # Check if this file exists
            if (planet_dir / expected_filename).exists():
                found_regions.append(region_path)
            else:
                missing_regions.append(region_path)

        if missing_regions:
            if found_regions:
                return (
                    False,
                    f"Partial coverage: have {len(found_regions)}, missing {len(missing_regions)} regions",
                )
            else:
                return False, f"Missing OSM files for {len(missing_regions)} region(s)"
        else:
            # All individual files exist
            if len(found_regions) == 1:
                # Single region - use individual file
                return (
                    True,
                    f"Found OSM file: {(planet_dir / (needed_regions[0][1].split('/')[-1])).name}",
                )
            else:
                # Multiple regions - check if merged file exists for these specific regions
                # Build expected merged filename from region names
                region_names = []
                for region_path, _, _ in needed_regions:
                    region_name = region_path.split("/")[-1]
                    region_names.append(region_name)
                region_names.sort()
                expected_merged_filename = f"merged-{'-'.join(region_names)}-latest.osm.pbf"
                expected_merged_path = planet_dir / expected_merged_filename

                if expected_merged_path.exists():
                    # Exact merged file for these regions exists
                    return True, f"Found merged OSM file: {expected_merged_filename}"
                else:
                    # Multiple regions but no merged file - need to merge
                    return (
                        False,
                        f"Have {len(found_regions)} region files but need merge (no merged file found)",
                    )

    # For non-address searches, use the old location-based logic
    from utils.data_validator import find_osm_file_for_region

    # Handle multiple regions
    if isinstance(location, list) and location and isinstance(location[0], tuple):
        # List of (continent, path) tuples - check each region
        missing_regions = []
        found_regions = []

        for continent, path in location:
            # Handle empty path (e.g., Antarctica where path is empty tuple)
            if isinstance(path, (list, tuple)) and len(path) > 0:
                location_name = "/".join(path)
            elif isinstance(path, (list, tuple)) and len(path) == 0:
                # Empty path means the continent itself is the region
                location_name = continent
            else:
                location_name = str(path)

            osm_file = find_osm_file_for_region(location_name)

            if osm_file and osm_file.exists():
                found_regions.append(location_name)
            else:
                missing_regions.append(location_name)

        # Report results
        if missing_regions and found_regions:
            return False, f"Found {len(found_regions)} region(s), missing {len(missing_regions)} region(s): {', '.join(missing_regions)}"
        elif missing_regions:
            return False, f"Missing OSM file(s) for: {', '.join(missing_regions)}"
        else:
            return True, f"Found OSM files for {len(found_regions)} region(s)"

    # Handle single region
    # Extract location name from various formats
    if isinstance(location, str):
        location_name = location
    elif isinstance(location, list) and location:
        # List of strings
        location_name = location[0]
    else:
        location_name = "Unknown"

    osm_file = find_osm_file_for_region(location_name)

    if osm_file and osm_file.exists():
        return True, f"Found OSM file: {osm_file.name}"

    # File not found - provide helpful message
    if is_us_state(location_name):
        return False, f"Missing OSM file for {location_name} (will download state-specific file)"
    else:
        return False, f"Missing OSM file for {location_name}"


def check_elevation_coverage(
    bbox: Tuple[float, float, float, float], scope_type: str, location: any
) -> Tuple[bool, List[str], str]:
    """
    Check if we have elevation tile coverage for the bounding box.

    Now performs tile-level validation to ensure specific tiles exist
    for the region, not just that dataset directories exist.

    Args:
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        scope_type: "address", "region", or "country"
        location: Location name(s)

    Returns:
        (has_coverage: bool, missing_datasets: List[str], message: str)
    """
    from utils.config_loader import DEPLOYMENT_TYPE
    from utils.dataset_selector import get_required_datasets
    from utils.tile_validator import check_region_elevation_coverage

    elevation_dir = Path("data/elevation_data")
    min_lat, min_lon, max_lat, max_lon = bbox

    # Use the proper dataset selection logic (respects local vs cloud deployment)
    required_datasets = get_required_datasets(
        min_lat, max_lat, min_lon, max_lon, deployment_type=DEPLOYMENT_TYPE
    )

    if not elevation_dir.exists():
        # No elevation data at all
        return False, required_datasets, "Missing all elevation data"

    # Check tile-level coverage for each dataset
    has_any_complete, coverage_report = check_region_elevation_coverage(
        elevation_dir, required_datasets, min_lat, max_lat, min_lon, max_lon
    )

    # Analyze coverage report
    datasets_with_complete_coverage = []
    datasets_with_incomplete_coverage = []
    datasets_with_no_coverage = []

    # Check each REQUIRED dataset (not just those in coverage_report)
    for dataset in required_datasets:
        report = coverage_report.get(
            dataset, {"has_all": False, "existing": [], "missing": [], "unavailable": [], "total_required": 0}
        )

        # Count existing + unavailable as "covered" (unavailable tiles are known to not exist)
        existing_count = len(report["existing"])
        unavailable_count = len(report.get("unavailable", []))
        missing_count = len(report["missing"])
        total_required = report["total_required"]

        # Dataset is complete if no tiles are actually missing (existing + unavailable = all tiles)
        # NOTE: Must also check total_required > 0, otherwise empty/missing directories
        # would incorrectly be marked as "complete" (missing_count=0 when no tiles calculated)
        if report["has_all"] or (missing_count == 0 and total_required > 0):
            datasets_with_complete_coverage.append(dataset)
        elif total_required == 0 or (existing_count == 0 and unavailable_count == 0):
            # No tiles at all for this required dataset
            datasets_with_no_coverage.append(dataset)
        else:
            # Has some tiles but not all
            datasets_with_incomplete_coverage.append(dataset)

    # Determine which datasets need to be downloaded
    missing_datasets = datasets_with_no_coverage + datasets_with_incomplete_coverage

    # Build informative message
    msg_parts = []

    if datasets_with_complete_coverage:
        complete_info = []
        for dataset in datasets_with_complete_coverage:
            report = coverage_report.get(dataset, {})
            existing_count = len(report.get("existing", []))
            unavailable_count = len(report.get("unavailable", []))
            if unavailable_count > 0:
                complete_info.append(f"{dataset.upper()} ({existing_count} tiles, {unavailable_count} unavailable)")
            else:
                complete_info.append(dataset.upper())
        msg_parts.append(f"Complete coverage: {', '.join(complete_info)}")

    if datasets_with_incomplete_coverage:
        incomplete_info = []
        for dataset in datasets_with_incomplete_coverage:
            report = coverage_report[dataset]
            existing_count = len(report['existing'])
            unavailable_count = len(report.get('unavailable', []))
            missing_count = len(report['missing'])
            # Show: existing/needed (unavailable excluded from needed count)
            needed_count = existing_count + missing_count  # Excludes unavailable
            info = f"{dataset.upper()} ({existing_count}/{needed_count} tiles"
            if unavailable_count > 0:
                info += f", {unavailable_count} unavailable"
            info += ")"
            incomplete_info.append(info)
        msg_parts.append(f"Incomplete coverage: {', '.join(incomplete_info)}")

    if datasets_with_no_coverage:
        msg_parts.append(f"No coverage: {', '.join([d.upper() for d in datasets_with_no_coverage])}")

    msg = "\n   ".join(msg_parts) if msg_parts else "No elevation data found"

    # Return success ONLY if ALL required datasets have complete coverage
    # Otherwise return the list of datasets that need to be downloaded
    if missing_datasets:
        return False, missing_datasets, msg
    else:
        return True, [], msg


def calculate_storage_requirements_for_batch(
    locations: List[str],
    scope_type: str
) -> str:
    """
    Calculate storage requirements for a batch of regions.
    Shows only what will be downloaded for the specified regions.

    Args:
        locations: List of location names (countries or regions)
        scope_type: "country" or "region"

    Returns:
        Formatted string showing storage requirements, or empty string if none
    """
    from climb_analyzer.data.geo_definitions import osm_pbf_urls

    planet_dir = Path("data/planet_osm_data")
    osm_to_download = []
    osm_total_size = 0

    # Check which OSM files are missing
    for location in locations:
        loc_normalized = location.lower()

        # Find OSM file info
        found = False
        if scope_type == "country":
            for continent, continent_data in osm_pbf_urls.items():
                for region_path, region_info in continent_data.items():
                    region_name = region_path.split('/')[-1]
                    if region_name.lower() == loc_normalized:
                        pbf_url = region_info.get('pbf_url')
                        if pbf_url:
                            filename = pbf_url.split('/')[-1]
                            file_path = planet_dir / filename

                            # Only include if file doesn't exist
                            if not file_path.exists():
                                size_bytes = region_info.get('size', 0)
                                osm_to_download.append((filename, size_bytes))
                                osm_total_size += size_bytes
                                found = True
                                break
                if found:
                    break

    # Format output
    if not osm_to_download:
        return ""

    lines = []

    if osm_to_download:
        lines.append(f"OSM Planet Files: {osm_total_size / 1e9:.2f} GB (will download)")
        for filename, size in osm_to_download[:3]:
            lines.append(f"  • {filename} ({size / 1e9:.2f} GB)")
        if len(osm_to_download) > 3:
            lines.append(f"  • ... and {len(osm_to_download) - 3} more files")

    # Note: Elevation data is auto-downloaded during analysis, so we don't need to calculate it here
    lines.append("")
    lines.append("Note: Elevation data will be downloaded automatically during analysis")

    return "\n".join(lines)


def auto_download_osm_data(
    bbox: Tuple[float, float, float, float],
    scope_type: str,
    location: any,
) -> bool:
    """
    Automatically download missing OSM data for the region.

    Args:
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        scope_type: "address", "region", or "country"
        location: Location name(s)

    Returns:
        True if successful, False otherwise
    """
    from climb_analyzer.data.geo_definitions import osm_pbf_urls

    print_header("Downloading OSM Data", spacing_before=1)

    planet_dir = Path("data/planet_osm_data")
    planet_dir.mkdir(parents=True, exist_ok=True)

    # Determine which regions to download
    regions_to_download = []

    if scope_type == "country":
        # Handle both single country and multiple countries
        locations = location if isinstance(location, list) else [location]

        for loc in locations:
            loc_normalized = loc.lower()

            # Search for country in osm_pbf_urls subregions
            found = False
            for continent, continent_data in osm_pbf_urls.items():
                # Look in the subregions (where individual countries are)
                if 'subregions' in continent_data:
                    for region_path, region_info in continent_data['subregions'].items():
                        region_name = region_path.split('/')[-1]
                        if region_name.lower() == loc_normalized:
                            pbf_url = region_info.get('pbf_url')
                            if pbf_url:
                                filename = pbf_url.split('/')[-1]
                                regions_to_download.append((loc, pbf_url, filename))
                                found = True
                                break
                if found:
                    break

            if not found:
                print(f"⚠️  Could not find OSM download URL for: {loc}")
                return False

    elif scope_type == "region":
        # Handle region selection - can be list of strings or list of (continent, path) tuples
        locations = location if isinstance(location, list) else [location]

        # Check if location is list of strings (like ["Hawaii"]) or tuples
        if locations and isinstance(locations[0], str):
            # Simple region names - search for them recursively in nested subregions
            def normalize_region_name(name):
                """Normalize region name for comparison (handle spaces vs hyphens)"""
                return name.lower().replace(' ', '-').replace('_', '-')

            def search_region_recursive(data_dict, target_name_normalized):
                """Recursively search for region in nested subregions"""
                for region_path, region_info in data_dict.items():
                    # Check if this region matches
                    region_name = region_path.split('/')[-1]
                    region_name_normalized = normalize_region_name(region_name)

                    if region_name_normalized == target_name_normalized:
                        pbf_url = region_info.get('pbf_url')
                        if pbf_url:
                            return pbf_url

                    # Recursively search subregions
                    if isinstance(region_info, dict) and 'subregions' in region_info:
                        result = search_region_recursive(region_info['subregions'], target_name_normalized)
                        if result:
                            return result
                return None

            for loc in locations:
                loc_normalized = normalize_region_name(loc)

                # IMPORTANT: Check if this is a US state first (handles Georgia disambiguation)
                # US states should ONLY search in north-america/us/, not all continents
                loc_is_us_state = is_us_state(loc)
                pbf_url = None

                if loc_is_us_state:
                    # US State - search in north-america/us/subregions
                    # Structure: osm_pbf_urls['north-america']['subregions']['north-america/us']['subregions']['us/utah']
                    if 'north-america' in osm_pbf_urls:
                        na_data = osm_pbf_urls['north-america']
                        if 'subregions' in na_data and 'north-america/us' in na_data['subregions']:
                            us_data = na_data['subregions']['north-america/us']
                            if 'subregions' in us_data:
                                # Search for keys starting with "us/" that match the state name
                                for region_path, region_info in us_data['subregions'].items():
                                    if region_path.startswith('us/'):
                                        region_name = region_path.split('/')[-1]
                                        region_name_normalized = normalize_region_name(region_name)
                                        if region_name_normalized == loc_normalized:
                                            pbf_url = region_info.get('pbf_url')
                                            break
                else:
                    # Not a US state - search all continents
                    for continent, continent_data in osm_pbf_urls.items():
                        if 'subregions' in continent_data:
                            pbf_url = search_region_recursive(continent_data['subregions'], loc_normalized)
                            if pbf_url:
                                break

                if pbf_url:
                    filename = pbf_url.split('/')[-1]
                    regions_to_download.append((loc, pbf_url, filename))
                else:
                    print(f"⚠️  Could not find OSM download URL for: {loc}")
                    return False
        else:
            # Tuple format: (continent, path)
            # path can be a string like 'england/bristol' or a tuple like ('england/bristol',)

            def search_recursive_for_path(data_dict, target_path):
                """Recursively search nested subregions for a path ending with target"""
                for region_key, region_info in data_dict.items():
                    # Check if this is our target
                    if region_key == target_path or region_key.endswith('/' + target_path.split('/')[-1]):
                        pbf_url = region_info.get('pbf_url') if isinstance(region_info, dict) else None
                        if pbf_url:
                            return pbf_url
                    # Recursively search subregions
                    if isinstance(region_info, dict) and 'subregions' in region_info:
                        result = search_recursive_for_path(region_info['subregions'], target_path)
                        if result:
                            return result
                return None

            for continent, path in locations:
                # Convert path to string - get the last path segment for searching
                if isinstance(path, (list, tuple)) and len(path) > 0:
                    # path is like ('england/bristol',) - take the last element
                    target_path = path[-1] if isinstance(path[-1], str) else str(path[-1])
                elif isinstance(path, (list, tuple)) and len(path) == 0:
                    target_path = continent
                else:
                    target_path = str(path)

                # Extract just the final region name for matching (e.g., 'england/bristol' -> 'bristol')
                target_name = target_path.split('/')[-1] if '/' in target_path else target_path

                # Search recursively through all continents for the region
                pbf_url = None
                for search_continent in osm_pbf_urls.keys():
                    if 'subregions' in osm_pbf_urls[search_continent]:
                        pbf_url = search_recursive_for_path(
                            osm_pbf_urls[search_continent]['subregions'],
                            target_path
                        )
                        if pbf_url:
                            break

                if pbf_url:
                    filename = pbf_url.split('/')[-1]
                    regions_to_download.append((target_name, pbf_url, filename))
                else:
                    print(f"⚠️  No OSM download URL for: {continent}/{target_path}")
                    return False

    else:
        print(f"⚠️  OSM auto-download not supported for scope type: {scope_type}")
        return False

    # Download each region
    for region_name, url, filename in regions_to_download:
        output_path = planet_dir / filename

        # Check if already exists
        if output_path.exists():
            print(f"✓ {filename} already exists, skipping")
            continue

        print(f"\n📥 Downloading {region_name}...")
        print(f"   URL: {url}")
        print(f"   Destination: {output_path}")

        try:
            # Use requests with tqdm for clean progress bar
            import requests

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"   {filename}",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    ncols=100,
                    ascii=" █"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            if output_path.exists():
                # Get file size
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
            else:
                print(f"❌ Download failed for {filename}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            return False
        except Exception as e:
            print(f"❌ Unexpected error during download: {e}")
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            return False

    print("\n✓ All OSM data downloaded successfully")
    return True


def auto_download_elevation_data(
    bbox: Tuple[float, float, float, float], missing_datasets: List[str], location: any
) -> bool:
    """
    Automatically download missing elevation data for the region.
    Also updates OpenTopoData config and restarts the server.

    Args:
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        missing_datasets: List of dataset names to download
        location: Location name for display

    Returns:
        True if successful, False otherwise
    """
    from climb_analyzer.data.setup import download_dem_for_region

    print_header(f"Downloading Elevation Data for {location}", spacing_before=1)

    min_lat, min_lon, max_lat, max_lon = bbox

    # Use absolute path to elevation_data directory for proper Docker volume mounting
    output_dir = Path.cwd() / "data" / "elevation_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    success, new_files_count = download_dem_for_region(
        region_name=location,
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        datasets=missing_datasets,
    )

    if not success:
        print("\n❌ Elevation data download failed")
        return False

    print(f"\n✓ Elevation data download completed ({new_files_count} new file(s))")

    # Auto-update OpenTopoData config and restart server ONLY if new files were downloaded
    if new_files_count == 0:
        print("\n✓ All required elevation data already exists - no server rebuild needed")
        return True

    try:
        from utils.opentopodata_manager import rebuild_and_restart

        print(f"\n🔄 Updating OpenTopoData server with {new_files_count} new file(s)...")

        # Rebuild and restart with new data
        if rebuild_and_restart(
            elevation_data_dir=output_dir, auto_update_config=True, validate_health=True
        ):
            print("\n✓ OpenTopoData server updated and ready")
            return True
        else:
            print("\n⚠️  Elevation data downloaded but server restart failed")
            print("    You may need to manually restart: docker compose up -d --build opentopodata")
            # Still return True since data was downloaded
            return True

    except ImportError as e:
        print(f"\n⚠️  Could not import opentopodata_manager: {e}")
        print("    Elevation data downloaded but server not restarted")
        print("    Run manually: docker compose up -d --build opentopodata")
        return True
    except Exception as e:
        print(f"\n⚠️  Error restarting OpenTopoData: {e}")
        print("    Elevation data downloaded but server not restarted")
        print("    Run manually: docker compose up -d --build opentopodata")
        return True


def validate_and_prepare_data(
    scope_type: str, location: any, address: Optional[str] = None, radius_km: Optional[float] = None
) -> bool:
    """
    Main entry point: Validate data coverage and auto-download if needed.

    Args:
        scope_type: "address", "region", or "country"
        location: Location identifier (region/country name or list of names)
        address: Street address (if scope_type == "address")
        radius_km: Search radius in km (if scope_type == "address")

    Returns:
        True if all required data is available (after downloads if needed), False otherwise
    """
    print_header("Validating Data Coverage", spacing_before=1)

    # Get bounding box for the scope
    center_coords = None  # Will hold (lat, lon) for address searches

    if scope_type == "address":
        # For address-based search, we need to geocode first
        print(f"  Address-based search: {address}")
        print(f"   Radius: {radius_km:.1f} km")

        # Geocode the address to get coordinates
        from geopy.geocoders import Nominatim

        try:
            geolocator = Nominatim(user_agent="climb_analyzer", timeout=10)
            location_result = geolocator.geocode(address, timeout=10)

            if not location_result:
                print(f"❌ Could not geocode address: {address}")
                return False

            lat, lon = location_result.latitude, location_result.longitude
            center_coords = (lat, lon)  # Store for later use
            print(f"   Coordinates: {lat:.4f}, {lon:.4f}")

            # Calculate bounding box from center point and radius
            # Rough approximation: 1 degree latitude ≈ 111 km
            # 1 degree longitude ≈ 111 km * cos(latitude)
            import math

            lat_delta = radius_km / 111.0
            lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

            bbox = (
                lat - lat_delta,  # min_lat
                lon - lon_delta,  # min_lon
                lat + lat_delta,  # max_lat
                lon + lon_delta,  # max_lon
            )
            location_name = address

        except Exception as e:
            print(f"❌ Error geocoding address: {e}")
            return False

    else:
        # Get bbox for state or country
        bbox = get_bbox_for_scope(scope_type, location)

        if bbox is None:
            print(f"❌ Could not determine bounding box for {scope_type}: {location}")
            return False

        # Handle different location formats
        if isinstance(location, str):
            location_name = location
        elif isinstance(location, list):
            if location and isinstance(location[0], tuple):
                # List of (continent, path) tuples from region selection
                # path is a list like ['liechtenstein'] or ['us', 'california']
                # or empty tuple () for continents like Antarctica
                location_names = []
                for continent, path in location:
                    if isinstance(path, (list, tuple)) and len(path) > 0:
                        # Join path components with /
                        location_names.append("/".join(path))
                    elif isinstance(path, (list, tuple)) and len(path) == 0:
                        # Empty path means the continent itself is the region
                        location_names.append(continent)
                    else:
                        location_names.append(str(path))
                location_name = ", ".join(location_names)
            else:
                # List of strings
                location_name = ", ".join(str(x) for x in location)
        else:
            location_name = str(location)

        print(f"  {scope_type.title()}: {location_name}")

    print(f"   Bounding box: {bbox}")

    # Check OSM coverage
    print("\n  Checking OSM planet file coverage...")
    osm_ok, osm_msg = check_osm_coverage(bbox, scope_type, location, center_point=center_coords)

    if osm_ok:
        print(f"   ✓ {osm_msg}")
    else:
        print(f"      {osm_msg}")

        # Offer to download OSM data automatically
        print(f"\n Downloading OSM data for {location_name}...")

        # For address-based searches, find the best region using coordinates
        if scope_type == "address":
            from utils.region_mapper import find_regions_for_bbox

            # Use the center coordinates we geocoded earlier
            print(
                f"   Finding best Geofabrik region for coordinates ({center_coords[0]:.4f}, {center_coords[1]:.4f})..."
            )

            # Find all regions that cover this area, passing center point
            # to ensure we include region containing the address even if search extends beyond
            regions = find_regions_for_bbox(bbox, prefer_smallest=True, center_point=center_coords)

            if not regions:
                print("   ⚠️  Could not find a matching Geofabrik region")
                print("\n   Please download manually from:")
                print("   https://download.geofabrik.de/")
                return False

            # Check if we need just one region or multiple
            if len(regions) == 1:
                # Single region covers the area completely
                region_path, pbf_url, size_bytes = regions[0]
                size_gb = size_bytes / (1024**3)
                print(f"     Best match: {region_path} ({size_gb:.2f} GB)")

                from climb_analyzer.data.osm_downloader import download_osm_for_location

                osm_file = download_osm_for_location(region_path, output_dir="data/planet_osm_data")

                if not osm_file:
                    print(f"\n⚠️  Auto-download failed for {region_path}")
                    print("   Please download manually from:")
                    print(f"   {pbf_url}")
                    return False

                print(f"\n✓ OSM data downloaded successfully: {osm_file.name}")

            elif len(regions) > 1:
                # Multiple NON-overlapping regions needed (e.g., search spans state borders)
                # This happens when the search area crosses boundaries between separate OSM regions
                print(f"     Search area spans {len(regions)} regions:")
                total_size = sum(size for _, _, size in regions)
                for i, (path, url, size) in enumerate(regions[:5], 1):
                    size_gb = size / (1024**3)
                    print(f"      {i}. {path} ({size_gb:.2f} GB)")
                if len(regions) > 5:
                    print(f"      ... and {len(regions) - 5} more")

                print(f"   📦 Total download size: {total_size / (1024**3):.2f} GB")

                # Download all needed regions (or verify they exist)
                from utils.osm_merger import check_osmium_available, merge_osm_files

                from climb_analyzer.data.osm_downloader import download_osm_for_location

                print(f"\n Checking/downloading {len(regions)} OSM regions...")
                downloaded_files = []

                for i, (region_path, pbf_url, size_bytes) in enumerate(regions, 1):
                    expected_filename = pbf_url.split("/")[-1]
                    expected_path = Path("./data/planet_osm_data") / expected_filename

                    if expected_path.exists():
                        print(f"   [{i}/{len(regions)}] ✓ Already have: {expected_path.name}")
                        downloaded_files.append(expected_path)
                    else:
                        print(f"\n   [{i}/{len(regions)}] Downloading {region_path}...")
                        osm_file = download_osm_for_location(region_path, output_dir="data/planet_osm_data")

                        if not osm_file:
                            print(f"\n⚠️  Auto-download failed for {region_path}")
                            print("   Please download manually from:")
                            print(f"   {pbf_url}")
                            return False

                        downloaded_files.append(osm_file)
                        print(f"   ✓ Downloaded: {osm_file.name}")

                print(f"\n✓ All {len(regions)} regions available")

                # Check if we need to merge (only if more than 1 file)
                if len(downloaded_files) > 1:
                    # Check if osmium is available
                    if not check_osmium_available():
                        print("\n" + "=" * 70)
                        print("  ⚠️  OSMIUM-TOOL REQUIRED FOR MULTI-REGION ANALYSIS")
                        print("=" * 70)
                        print("\n❌ Cannot proceed: osmium-tool is not installed")
                        print(f"\nYour search area spans {len(downloaded_files)} regions:")
                        for i, f in enumerate(downloaded_files, 1):
                            print(f"   {i}. {f.name}")
                        print("\nThese files MUST be merged before indexing, otherwise:")
                        print("  • Roads crossing state borders will be missing")
                        print("  • Analysis results will be incomplete")
                        print("  • Climbs spanning multiple states won't be detected")
                        print("\n Please install osmium-tool:")
                        print("  • Ubuntu/Debian: sudo apt-get install osmium-tool")
                        print("  • macOS: brew install osmium-tool")
                        print("  • Arch Linux: sudo pacman -S osmium-tool")
                        print("\nAfter installation, re-run the analysis.")
                        print("The downloaded files will be automatically merged.")
                        return False
                    else:
                        # Merge the files
                        # Generate filename from region names for reusability across searches
                        region_names = []
                        for region_path, _, _ in regions:
                            # Extract last part of path (e.g., "north-carolina" from "north-america/us/north-carolina")
                            region_name = region_path.split("/")[-1]
                            region_names.append(region_name)

                        # Sort alphabetically for consistent naming
                        region_names.sort()

                        # Create merged filename: merged-north-carolina-south-carolina-tennessee-latest.osm.pbf
                        merged_filename = f"merged-{'-'.join(region_names)}-latest.osm.pbf"
                        merged_path = Path("./data/planet_osm_data") / merged_filename

                        print(f"\n🔀 Merging {len(downloaded_files)} OSM files...")
                        print(f"   Output: {merged_filename}")
                        success = merge_osm_files(
                            downloaded_files,
                            merged_path,
                            remove_inputs=False,  # Keep originals in case merge fails
                        )

                        if success:
                            print(f"✓ Regions merged successfully: {merged_path.name}")
                            print(" This merged file covers your entire search area")
                            osm_file = merged_path
                        else:
                            print("\n" + "=" * 70)
                            print("  ❌ MERGE FAILED")
                            print("=" * 70)
                            print("\nCannot proceed with multi-region analysis.")
                            print("The OSM files could not be merged.")
                            print("\nPlease check:")
                            print("  • Disk space is sufficient")
                            print("  • osmium-tool is working: osmium --version")
                            print("  • File permissions are correct")
                            return False
                else:
                    # Only one file downloaded
                    osm_file = downloaded_files[0]

            else:
                # This shouldn't happen, but handle it gracefully
                print("   ❌ No suitable regions found")
                return False

        else:
            # For state/country/region, use existing logic
            from climb_analyzer.data.osm_downloader import download_osm_for_location

            # Handle multiple regions
            if isinstance(location, list) and location and isinstance(location[0], tuple):
                # Multiple regions - download each separately
                print(f"\n Downloading {len(location)} OSM region(s)...")
                print()  # Blank line before progress bar
                downloaded_files = []

                pbar = tqdm(location, desc="Downloading OSM regions", unit="region",
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

                for continent, path in pbar:
                    # Handle empty path (e.g., Antarctica where path is empty tuple)
                    if isinstance(path, (list, tuple)) and len(path) > 0:
                        region_name = "/".join(path)
                    elif isinstance(path, (list, tuple)) and len(path) == 0:
                        # Empty path means the continent itself is the region
                        region_name = continent
                    else:
                        region_name = str(path)

                    pbar.set_description(f"Downloading {region_name}")
                    osm_file = download_osm_for_location(region_name, output_dir="data/planet_osm_data")

                    if not osm_file:
                        pbar.close()
                        print(f"\n⚠️  Auto-download failed for {region_name}. Please download manually:")
                        print("   Download from: https://download.geofabrik.de/")
                        return False

                    downloaded_files.append(osm_file)

                pbar.close()
                print(f"\n✓ All {len(location)} OSM region(s) downloaded successfully")

                # Build spatial indexes for all downloaded files
                print(f"\n🔨 Building spatial indexes for {len(downloaded_files)} file(s)...")
                print()  # Blank line before progress bar
                try:
                    from climb_analyzer.data.index_builder import build_spatial_index

                    index_pbar = tqdm(downloaded_files, desc="Building indexes", unit="file",
                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

                    for osm_file_path in index_pbar:
                        index_pbar.set_description(f"Indexing {osm_file_path.name}")
                        index_success = build_spatial_index(
                            osm_file_path=str(osm_file_path),
                            output_dir="data/osm_indexes",
                            surface_filter="all",
                            cycling_only=False,
                        )

                        if not index_success:
                            index_pbar.write(f"   ⚠️  Index build failed for {osm_file_path.name} (will be built on first use)")

                    index_pbar.close()

                except Exception as e:
                    print(f"\n   ⚠️  Error building spatial indexes: {e}")
                    print("   Indexes will be built automatically on first analysis")

            else:
                # Single region
                osm_file = download_osm_for_location(location_name, output_dir="data/planet_osm_data")

                if not osm_file:
                    print("\n⚠️  Auto-download failed. Please download manually:")
                    print("   Download from: https://download.geofabrik.de/")
                    return False

                print(f"\n✓ OSM data downloaded successfully: {osm_file.name}")

                # Build spatial index for the new OSM file
                print(f"\n🔨 Building spatial index for {osm_file.name}...")
                try:
                    from climb_analyzer.data.index_builder import build_spatial_index

                    index_success = build_spatial_index(
                        osm_file_path=str(osm_file),
                        output_dir="data/osm_indexes",
                        surface_filter="all",
                        cycling_only=False,
                    )

                    if index_success:
                        print("   ✓ Spatial index built successfully")
                    else:
                        print("   ⚠️  Spatial index build failed (will be built on first use)")

                except Exception as e:
                    print(f"   ⚠️  Error building spatial index: {e}")
                    print("   Index will be built automatically on first analysis")

        # Re-check coverage (pass center_point for address searches to ensure consistency)
        osm_ok, osm_msg = check_osm_coverage(bbox, scope_type, location, center_point=center_coords)
        if not osm_ok:
            print("\n❌ OSM file downloaded but still not detected. Please check manually.")
            return False

    # Memory check happens in climb_analyzer.py (disabled here to avoid startup issues)

    # Check elevation coverage
    print("\n  Checking elevation data coverage...")
    elev_ok, missing_datasets, elev_msg = check_elevation_coverage(bbox, scope_type, location)

    if elev_ok:
        print(f"   ✓ {elev_msg}")
    else:
        # Download missing elevation data (checkpoint status doesn't affect this)
        print(f"   ⚠️  {elev_msg}")
        print(f"\n Downloading missing elevation data: {', '.join(missing_datasets).upper()}")

        success = auto_download_elevation_data(bbox, missing_datasets, location_name)
        if not success:
            return False

    # Check if OpenTopoData server is running (for local deployment)
    from utils.config_loader import DEPLOYMENT_TYPE

    if DEPLOYMENT_TYPE == "local":
        print("\n🔌 Checking OpenTopoData server status...")

        try:
            from utils.opentopodata_manager import check_server_health, get_configured_datasets, rebuild_and_restart
            from utils.dataset_selector import get_required_datasets

            # Initial check with generous timeout (60 seconds)
            # Server might be starting up or initializing (especially with many tiles)
            if check_server_health(max_retries=30, retry_delay=2.0):
                print("  ✓ OpenTopoData server is running")

                # Verify required datasets are configured on the server
                min_lat, min_lon, max_lat, max_lon = bbox
                required = set(get_required_datasets(min_lat, max_lat, min_lon, max_lon, deployment_type="local"))
                configured = get_configured_datasets()

                if configured:
                    missing_from_config = required - configured
                    if missing_from_config:
                        print(f"  ⚠️  Server missing datasets: {', '.join(sorted(missing_from_config))}")
                        print("  Updating server configuration...")
                        if rebuild_and_restart(auto_update_config=True, validate_health=True):
                            print("  ✓ Server configuration updated")
                        else:
                            print("  ⚠️  Config update failed - elevation fetching may fail")
                    else:
                        # Check if any datasets on disk aren't configured (stale config)
                        elevation_dir = Path("data/elevation_data")
                        if elevation_dir.exists():
                            on_disk = {d.name for d in elevation_dir.iterdir() if d.is_dir() and any(d.glob("*.tif")) or any(d.glob("*.hgt"))}
                            # Filter to known dataset names
                            known_datasets = {"ned10m", "srtm30m", "aw3d30", "aster30m", "rema32m", "arctic32m"}
                            on_disk = on_disk & known_datasets
                            missing_from_config = on_disk - configured
                            if missing_from_config:
                                print(f"  ⚠️  Datasets on disk not configured: {', '.join(sorted(missing_from_config))}")
                                print("  Updating server configuration...")
                                if rebuild_and_restart(auto_update_config=True, validate_health=True):
                                    print("  ✓ Server configuration updated")
                else:
                    # Couldn't query datasets - try to rebuild anyway if data exists
                    elevation_dir = Path("data/elevation_data")
                    if elevation_dir.exists() and any(elevation_dir.iterdir()):
                        print("  ⚠️  Could not query server datasets - updating configuration...")
                        if rebuild_and_restart(auto_update_config=True, validate_health=True):
                            print("  ✓ Server configuration updated")

                return True
            else:
                print("   ❌ OpenTopoData server not responding")
                print("\n Starting OpenTopoData server...")

                # rebuild_and_restart includes its own health check with longer timeout
                if rebuild_and_restart(auto_update_config=True, validate_health=True):
                    print("   ✓ Server started successfully")
                    return True
                else:
                    print("\n⚠️  Could not start OpenTopoData server automatically")
                    print("    Please start it manually:")
                    print("      docker compose up -d --build opentopodata")
                    return False

        except Exception as e:
            print(f"   ⚠️  Error checking server: {e}")
            print("\n Please ensure OpenTopoData is running:")
            print("      docker compose up -d --build opentopodata")
            return False

    return True


def batch_mode_validate(scope_type: str, location: any) -> bool:
    """
    Batch mode validation - auto-download without prompting.

    Args:
        scope_type: "address", "region", or "country"
        location: Location identifier

    Returns:
        True if all data is ready, False otherwise
    """
    print_header("Batch Mode - Validating Data Coverage", spacing_before=1)

    # Get bounding box
    bbox = get_bbox_for_scope(scope_type, location)

    if bbox is None:
        print(f"❌ Could not determine bounding box for {scope_type}: {location}")
        return False

    # Format location name for display
    if isinstance(location, str):
        location_name = location
    elif isinstance(location, list):
        # Handle list of tuples like [(continent, (path,))]
        if location and isinstance(location[0], tuple):
            # Extract paths from tuples
            paths = []
            for item in location:
                if isinstance(item, tuple) and len(item) == 2:
                    # Format: (continent, (path,))
                    if isinstance(item[1], tuple):
                        paths.extend(item[1])
                    else:
                        paths.append(str(item[1]))
                else:
                    paths.append(str(item))
            location_name = ", ".join(paths)
        else:
            # Regular list of strings
            location_name = ", ".join(str(x) for x in location)
    else:
        location_name = str(location)

    print(f"  {scope_type.title()}: {location_name}")
    print(f"   Bounding box: {bbox}")

    # Check OSM coverage
    print("\n  Checking OSM planet file coverage...")
    osm_ok, osm_msg = check_osm_coverage(bbox, scope_type, location)

    if osm_ok:
        print(f"   ✓ {osm_msg}")
    else:
        print(f"   ❌ {osm_msg}")
        print("    Auto-downloading OSM data...")

        success = auto_download_osm_data(bbox, scope_type, location)
        if not success:
            return False

    # Check elevation coverage
    print("\n  Checking elevation data coverage...")
    elev_ok, missing_datasets, elev_msg = check_elevation_coverage(bbox, scope_type, location)

    if elev_ok:
        print(f"   ✓ {elev_msg}")
    else:
        # Download missing elevation data (checkpoint status doesn't affect this)
        print(f"   ⚠️  {elev_msg}")
        print(f"    Auto-downloading: {', '.join(missing_datasets).upper()}\n")
        sys.stdout.flush()

        success = auto_download_elevation_data(bbox, missing_datasets, location_name)
        if not success:
            return False

    # Check if OpenTopoData server is running (for local deployment)
    from utils.config_loader import DEPLOYMENT_TYPE

    if DEPLOYMENT_TYPE == "local":
        print("\n🔌 Checking OpenTopoData server status...")

        try:
            from utils.opentopodata_manager import check_server_health, get_configured_datasets, rebuild_and_restart
            from utils.dataset_selector import get_required_datasets

            # Initial check with generous timeout (60 seconds)
            if check_server_health(max_retries=30, retry_delay=2.0):
                print("  ✓ OpenTopoData server is running")

                # Verify required datasets are configured on the server
                min_lat, min_lon, max_lat, max_lon = bbox
                required = set(get_required_datasets(min_lat, max_lat, min_lon, max_lon, deployment_type="local"))
                configured = get_configured_datasets()

                if configured:
                    missing_from_config = required - configured
                    if missing_from_config:
                        print(f"  ⚠️  Server missing datasets: {', '.join(sorted(missing_from_config))}")
                        print("  Updating server configuration...")
                        if rebuild_and_restart(auto_update_config=True, validate_health=True):
                            print("  ✓ Server configuration updated")
                        else:
                            print("  ⚠️  Config update failed - elevation fetching may fail")
                    else:
                        # Check if any datasets on disk aren't configured (stale config)
                        elevation_dir = Path("data/elevation_data")
                        if elevation_dir.exists():
                            on_disk = {d.name for d in elevation_dir.iterdir() if d.is_dir() and any(d.glob("*.tif")) or any(d.glob("*.hgt"))}
                            known_datasets = {"ned10m", "srtm30m", "aw3d30", "aster30m", "rema32m", "arctic32m"}
                            on_disk = on_disk & known_datasets
                            missing_from_config = on_disk - configured
                            if missing_from_config:
                                print(f"  ⚠️  Datasets on disk not configured: {', '.join(sorted(missing_from_config))}")
                                print("  Updating server configuration...")
                                if rebuild_and_restart(auto_update_config=True, validate_health=True):
                                    print("  ✓ Server configuration updated")
                else:
                    # Couldn't query datasets - try to rebuild anyway if data exists
                    elevation_dir = Path("data/elevation_data")
                    if elevation_dir.exists() and any(elevation_dir.iterdir()):
                        print("  ⚠️  Could not query server datasets - updating configuration...")
                        if rebuild_and_restart(auto_update_config=True, validate_health=True):
                            print("  ✓ Server configuration updated")

                return True
            else:
                print("   ❌ Server not responding - auto-starting...")

                # Batch mode: automatically start server without prompting
                if rebuild_and_restart(auto_update_config=True, validate_health=True):
                    print("   ✓ Server started successfully")
                    return True
                else:
                    print("   ❌ Failed to start OpenTopoData server")
                    return False

        except Exception as e:
            print(f"   ❌ Error checking/starting server: {e}")
            return False

    return True


if __name__ == "__main__":
    # Test the coverage checker
    import sys

    if len(sys.argv) < 3:
        print("Usage: python data_coverage_checker.py <scope_type> <location>")
        print("Example: python data_coverage_checker.py state Vermont")
        print("Example: python data_coverage_checker.py country Switzerland")
        sys.exit(1)

    scope_type = sys.argv[1]
    location = sys.argv[2]

    success = validate_and_prepare_data(scope_type, location)

    if success:
        print("\n✓ All data ready for analysis!")
        sys.exit(0)
    else:
        print("\n❌ Data validation failed")
        sys.exit(1)
