#!/usr/bin/env python3
"""
Data Setup for Climb Analyzer (Local Deployment)

Downloads and prepares all required data:
1. OSM .pbf file from Geofabrik
2. Spatial R-tree index
3. DEM (elevation) data for the region

Usage:
    python data_setup.py --country "Switzerland"
    python data_setup.py --state "Vermont"
    python data_setup.py --interactive
"""

import argparse
import os
import sys
from pathlib import Path

from climb_analyzer.utils.formatting import print_section_simple

# Import our modules
from utils.geographic_menu import select_countries, select_us_states
from climb_analyzer.data.manager import DataManager
from climb_analyzer.data.geo_lookup import is_us_state, get_region_bounds


def get_earthdata_credentials() -> tuple[str, str] | None:
    """
    DEPRECATED: NASA Earthdata credentials no longer needed (December 2025).

    All elevation datasets now use public sources:
    - SRTM: OpenTopography S3 (public)
    - NED: AWS S3 (public)
    - AW3D30: JAXA FTP (public)
    - ASTER: Data source retired December 2025

    Returns:
        None - credentials are no longer required
    """
    return None


# def get_jaxa_credentials() -> tuple[str, str] | None:
#     """
#     Get JAXA credentials from environment or prompt user.
#
#     NOTE: As of 2024, AW3D30 uses public FTP and doesn't require credentials.
#     This function is no longer used.
#
#     Returns:
#         Tuple of (username, password) or None if user skips
#     """
#     # Check if credentials already exist in environment
#     username = os.environ.get('JAXA_USERNAME')
#     password = os.environ.get('JAXA_PASSWORD')
#
#     if username and password:
#         print(f"\n✓ Using existing JAXA credentials (username: {username})")
#         return (username, password)
#
#     # Check for credentials in .credentials/jaxa file
#     # The .credentials directory is mounted from host at /app/.credentials (see docker-compose.yml line 26)
#     try:
#         from pathlib import Path
#
#         jaxa_path = Path("/app/.credentials/jaxa")
#
#         if jaxa_path.exists():
#             try:
#                 with open(jaxa_path) as f:
#                     lines = f.readlines()
#                     if len(lines) >= 2:
#                         username = lines[0].strip()
#                         password = lines[1].strip()
#                         if username and password:
#                             print(f"\n✓ Found JAXA credentials (username: {username})")
#                             # Store in environment for this session
#                             os.environ['JAXA_USERNAME'] = username
#                             os.environ['JAXA_PASSWORD'] = password
#                             return (username, password)
#             except Exception:
#                 pass
#     except Exception:
#         pass
#
#     # Inform user that JAXA credentials are optional
#     print("\n" + "=" * 70)
#     print("JAXA Credentials (Optional)")
#     print("=" * 70)
#     print("\nGood news! The AW3D30 dataset is currently available via public FTP.")
#     print("No credentials are required for AW3D30.")
#     print("\nIf you have JAXA credentials, you can enter them for future-proofing.")
#     print("Otherwise, you can skip this step.")
#     print("=" * 70)
#
#     response = input("\nDo you have JAXA credentials to set up? [y/N]: ").strip().lower()
#     if response != 'y':
#         print("\nSkipping JAXA credentials. AW3D30 will use public FTP access.")
#         return None
#
#     username = input("JAXA username: ").strip()
#     password = getpass.getpass("JAXA password: ")
#
#     if username and password:
#         # Store in environment for this session
#         os.environ['JAXA_USERNAME'] = username
#         os.environ['JAXA_PASSWORD'] = password
#         print("✓ Credentials stored in environment variables")
#
#         # Ask if user wants to save credentials
#         save_creds = input("\nSave JAXA credentials for future use? [Y/n]: ").strip().lower()
#         if save_creds in ['', 'y', 'yes']:
#             try:
#                 from setup_wizard import setup_jaxa_credentials
#                 setup_jaxa_credentials(username, password)
#                 print("✓ Credentials will be available for future use")
#             except Exception as e:
#                 print(f"⚠️  Could not save credentials: {e}")
#                 print("   You'll need to enter them again next time")
#
#         return (username, password)
#
#     return None


def load_geographic_bounds():
    """
    Legacy function - no longer needed.
    Bounds are now accessed via geo_lookup.get_region_bounds().
    """
    # Return empty dicts for backward compatibility
    # Callers should migrate to using geo_lookup functions directly
    return {}, {}


def get_required_datasets(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> list[str]:
    """
    Determine which DEM datasets to download based on region bounds.

    Dataset coverage:
    - NED 10m: US only (18°N to 72°N, -180° to -60°W) - highest quality
    - SRTM 30m: 60°N to 56°S
    - AW3D30: Global 84°N to 84°S (fallback for non-US)
    - ArcticDEM 32m: Arctic regions (>60°N)
    - REMA 32m: Antarctica (≤-60°S)

    Returns:
        List of dataset names to download
    """
    datasets = []

    # Antarctica (lat_max <= -60)
    if lat_max <= -60:
        datasets.append("rema")
        return datasets

    # Check region characteristics
    is_arctic = lat_max > 60  # Region extends into Arctic
    is_us_region = (lat_min >= 18 and lat_max <= 72 and
                    lon_min >= -180 and lon_max <= -60)

    # US regions get NED (highest quality) - no need for AW3D30 fallback
    if is_us_region:
        datasets.append("ned10m")
        if is_arctic:
            datasets.append("arcticdem")  # For Alaska
        # SRTM as backup for areas below 60°N
        if lat_min < 60:
            datasets.append("srtm")
        return datasets  # No AW3D30 needed - NED covers all US

    # Non-US regions
    if is_arctic:
        datasets.append("arcticdem")

    # SRTM for mid-latitudes (60°N to 56°S)
    if lat_min >= -56 and lat_max <= 60:
        datasets.append("srtm")
    elif lat_min < 60:  # Region spans both Arctic and mid-latitudes
        datasets.append("srtm")

    # AW3D30 as fallback (only for non-US regions)
    datasets.append("aw3d30")

    return datasets


def explain_dataset_selection(
    datasets: list[str], lat_min: float, lat_max: float, lon_min: float, lon_max: float
):
    """Explain which datasets are being downloaded and why."""
    print("\nDatasets to download:")
    for ds in datasets:
        ds_upper = ds.upper()
        if ds_upper == "NED10M":
            print(f"  • {ds_upper} - National Elevation Dataset 10m (US high-res)")
        elif ds_upper == "AW3D30":
            print(f"  • {ds_upper} - ALOS World 3D 30m (global coverage)")
        elif ds_upper == "REMA":
            print(f"  • {ds_upper} - Antarctica elevation model 32m")
        elif ds_upper == "ARCTICDEM":
            print(f"  • {ds_upper} - Arctic elevation model 32m")
        elif ds_upper == "SRTM":
            print(f"  • {ds_upper} - Shuttle Radar Topography Mission 30m")
        else:
            print(f"  • {ds_upper}")


def download_dem_for_region(
    region_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    datasets: list[str],
) -> tuple:
    """
    Download DEM data for a region.

    Args:
        region_name: Name of region
        lat_min, lat_max, lon_min, lon_max: Bounding box
        datasets: List of dataset names to download

    Returns:
        Tuple of (success: bool, new_files_count: int)
    """
    print_section_simple(f"Downloading DEM data for {region_name}", spacing_before=1)
    print(f"Bounding box: ({lat_min:.2f}, {lon_min:.2f}) to ({lat_max:.2f}, {lon_max:.2f})")

    # Show which datasets will be downloaded
    explain_dataset_selection(datasets, lat_min, lat_max, lon_min, lon_max)

    print("\n   DEM download can take 30 minutes to several hours")
    print("  depending on region size and number of datasets.\n")

    # Note: All elevation datasets now use public sources (December 2025)
    # No credentials required for SRTM, NED, or AW3D30

    if not datasets:
        print("\n⚠️  No datasets to download")
        return (False, 0)

    try:
        from .dem_downloaders import (
            ArcticDEMDownloader,
            AW3D30Downloader,
            NEDDownloader,
            REMADownloader,
            SRTMDownloader,
        )

        base_dir = Path("data/elevation_data")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Add buffer to bbox to account for roads that extend slightly beyond boundaries
        # OSM data often includes roads just outside official boundaries
        # Buffer: 0.1 degrees ≈ 11 km, ensures we get all necessary tiles
        BBOX_BUFFER = 0.1
        bbox = (
            lat_min - BBOX_BUFFER,
            lon_min - BBOX_BUFFER,
            lat_max + BBOX_BUFFER,
            lon_max + BBOX_BUFFER
        )

        # Create subdirectories for each dataset to avoid coordinate conflicts
        # Each dataset gets its own folder to prevent OpenTopoData from seeing
        # duplicate tile coordinates with different extensions
        srtm_dir = base_dir / "srtm30m"
        aw3d30_dir = base_dir / "aw3d30"
        rema_dir = base_dir / "rema32m"
        arcticdem_dir = base_dir / "arctic32m"
        ned10m_dir = base_dir / "ned10m"

        for dir_path in [srtm_dir, aw3d30_dir, rema_dir, arcticdem_dir, ned10m_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize downloaders with their specific subdirectories
        # Aliases must match ALL keys used in DATASET_PRIORITY_BY_REGION
        # (engine.py) — a missing alias silently drops that dataset here too.
        downloaders = {
            "aw3d30": AW3D30Downloader(aw3d30_dir),
            "rema": REMADownloader(rema_dir),
            "rema32m": REMADownloader(rema_dir),
            "arcticdem": ArcticDEMDownloader(arcticdem_dir),
            "arctic32m": ArcticDEMDownloader(arcticdem_dir),
            "srtm": SRTMDownloader(srtm_dir),
            "srtm30m": SRTMDownloader(srtm_dir),
            "ned10m": NEDDownloader(ned10m_dir),
        }

        # Sort datasets by download priority (primary/high-quality first)
        # Priority order: ned, arctic, rema, srtm, aw3d30
        priority_order = {
            'ned10m': 1,
            'arcticdem': 2,
            'rema': 3,
            'srtm': 4,
            'srtm30m': 4,  # Same priority as srtm
            'aw3d30': 5,
        }
        datasets_sorted = sorted(datasets, key=lambda d: priority_order.get(d.lower(), 99))

        # Track download results by priority and new files count
        download_results = {}
        total_new_files = 0
        for dataset in datasets_sorted:
            dataset_lower = dataset.lower()
            if dataset_lower in downloaders:
                print(f"\n  Downloading {dataset.upper()}...")
                downloader = downloaders[dataset_lower]
                try:
                    success, new_files_count = downloader.download_bbox(bbox)
                    download_results[dataset_lower] = success
                    total_new_files += new_files_count
                    if not success and new_files_count == 0:
                        # Download returned False - could be because tiles were skipped or unavailable
                        print(f"     {dataset.upper()}: No tiles downloaded (may be covered by higher-priority datasets or unavailable)")
                    elif not success:
                        print(f"  ⚠️  {dataset.upper()} download had issues")
                except Exception as e:
                    print(f"  ❌ {dataset.upper()} download failed: {e}")
                    download_results[dataset_lower] = False
            else:
                print(f"  ⚠️  Unknown dataset: {dataset}")

        # Consider it success if we have at least one dataset (including existing ones)
        # Check both download results AND existing files on disk
        # Priority: NED (US high-res), SRTM (primary), AW3D30 (secondary)

        def has_dataset_files(dataset_dir: Path) -> bool:
            """Check if dataset directory has any elevation files."""
            if not dataset_dir.exists():
                return False
            patterns = ['*.tif', '*.hgt', '*.img', '*.vrt']
            for pattern in patterns:
                if list(dataset_dir.glob(pattern)):
                    return True
            return False

        has_ned = download_results.get('ned10m', False) or has_dataset_files(ned10m_dir)
        has_primary = (download_results.get('srtm', False) or download_results.get('srtm30m', False)
                      or has_dataset_files(srtm_dir))
        has_secondary = download_results.get('aw3d30', False) or has_dataset_files(aw3d30_dir)
        has_polar = (download_results.get('rema', False) or download_results.get('arcticdem', False)
                    or has_dataset_files(rema_dir) or has_dataset_files(arcticdem_dir))

        if has_ned or has_primary or has_secondary or has_polar:
            print(f"\n  ✓ DEM data downloaded successfully ({total_new_files} new file(s))")

            # Prepare elevation data (deduplicate overlapping tiles)
            from .dem_downloaders import prepare_elevation_data, update_opentopodata_config

            print("\n  Preparing elevation data for OpenTopoData...")
            prepare_success = prepare_elevation_data(base_dir)
            if not prepare_success:
                print("  ⚠️  Elevation data preparation had issues, continuing anyway...")

            # Update OpenTopoData configuration
            config_updated = update_opentopodata_config(
                elevation_data_dir=base_dir,
                config_path=Path("opentopodata-config.yaml")
            )

            # Determine if server rebuild is needed
            # Rebuild if: (1) config was updated, OR (2) server is running but missing required datasets
            needs_rebuild = False

            if config_updated and total_new_files > 0:
                print("\n  🔄 New files downloaded - rebuilding OpenTopoData server...")
                needs_rebuild = True
            elif config_updated and total_new_files == 0:
                # Config was updated but no new files - means datasets exist but weren't in config
                print("\n  🔄 Config updated with existing datasets - rebuilding OpenTopoData server...")
                needs_rebuild = True
            elif total_new_files == 0:
                print("\n  ✓ All required files already exist - no server rebuild needed")

            if needs_rebuild:
                # Always rebuild when new files are downloaded (server must reload data files)
                # Only skip rebuild if config changed but no new files (can wait for next restart)
                from utils.opentopodata_manager import check_server_health

                if total_new_files > 0:
                    # New files downloaded - MUST rebuild to load them
                    print("  [DEBUG] New files downloaded - forcing OpenTopoData rebuild...")
                    rebuild_opentopodata_server()
                else:
                    # Only config changed, no new files - can defer rebuild
                    print("  [DEBUG] Checking if OpenTopoData server is already running...")
                    if check_server_health(max_retries=3, retry_delay=1.0, quiet=True):
                        print("  ✓ OpenTopoData server already running and healthy - skipping rebuild")
                        print("     (Config was updated - server will load new config on next restart)")
                    else:
                        print("  [DEBUG] Server not responding - proceeding with rebuild...")
                        rebuild_opentopodata_server()

            return (True, total_new_files)
        else:
            print("\n  ❌ DEM download failed - no primary or secondary datasets available")
            return (False, 0)

    except Exception as e:
        print(f"\n  ❌ DEM download error: {e}")
        import traceback

        traceback.print_exc()
        return (False, 0)


def is_running_in_container() -> bool:
    """Check if we're running inside a Docker container."""
    import os
    return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')


def rebuild_opentopodata_server() -> bool:
    """
    Rebuild and restart the OpenTopoData server to load updated configuration.

    Uses Docker socket if available (container with mounted socket),
    otherwise provides manual instructions.

    Returns:
        True if successful or instructions provided
    """
    import subprocess

    print("\n[Rebuilding OpenTopoData Server]")

    # Check if we have Docker socket access (works from container or host)
    has_docker = False
    docker_check_error = None
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            check=False,
            timeout=5
        )
        has_docker = (result.returncode == 0)
        if not has_docker:
            docker_check_error = f"docker ps returned code {result.returncode}: {result.stderr.decode()[:200]}"
    except FileNotFoundError as e:
        docker_check_error = f"docker command not found: {e}"
    except subprocess.TimeoutExpired:
        docker_check_error = "docker ps timed out after 5s"
    except Exception as e:
        docker_check_error = f"unexpected error: {e}"

    if not has_docker:
        # No Docker access - provide manual instructions
        print(f"     Docker socket not accessible ({docker_check_error})")
        print("\n  To rebuild OpenTopoData with the updated config:")
        print("\n  From the HOST machine (open a NEW terminal window), run:")
        print("     cd /path/to/climb-analyzer")
        print("     ./utils/rebuild-opentopodata.sh")
        print("\n  Or manually:")
        print("     cd /path/to/climb-analyzer/opentopodata")
        print("     docker stop opentopodata-server")
        print("     make build && docker run --rm -itd --name opentopodata-server --network climb-network \\")
        print("       -v $(pwd)/../elevation_data:/app/data:ro -v $(pwd)/../opentopodata-config.yaml:/app/config.yaml:ro \\")
        print("       -p 5000:5000 opentopodata:$(cat VERSION)")
        print("\n  ⏳ After running the rebuild, the analyzer will automatically detect the server.")
        print("  ✓ Configuration updated - ready for rebuild")
        return True

    # We have Docker access - use the rebuild script
    print("  Rebuilding and restarting OpenTopoData container...")

    try:
        # Find the rebuild script - check current directory and parent directory
        from pathlib import Path

        # Try utils directory first (preferred location)
        script_path = Path("./utils/rebuild-opentopodata.sh")

        # If not found, try finding it relative to this script's location
        if not script_path.exists():
            script_dir = Path(__file__).parent
            script_path = script_dir / "utils" / "rebuild-opentopodata.sh"

        # If still not found, try parent directory with utils (in case we're in a subdirectory)
        if not script_path.exists():
            script_path = Path("../utils/rebuild-opentopodata.sh")

        if not script_path.exists():
            print("     utils/rebuild-opentopodata.sh not found - skipping automatic rebuild")
            print("  💡 The OpenTopoData server may need manual restart if config changed")
            return False

        # Run the rebuild script
        print(f"  → Running {script_path.resolve()}...")
        result = subprocess.run(
            ["bash", str(script_path.resolve())],
            check=False,
            text=True,
            cwd=script_path.parent.resolve()
        )

        if result.returncode == 0:
            print("  ✓ OpenTopoData server rebuilt and started")
            print("  ℹ️  Server available at http://localhost:5000")
            return True
        else:
            print("  ❌ Rebuild script failed")
            print("  💡 Try running manually: ./utils/rebuild-opentopodata.sh")
            return False

    except Exception as e:
        print(f"  ❌ Failed to rebuild server: {e}")
        print("  💡 Try running manually: ./utils/rebuild-opentopodata.sh")
        return False


def setup_data_for_region(
    region_name: str,
    bounds_data: dict,
    skip_osm: bool = False,
    skip_index: bool = False,
    skip_dem: bool = False,
) -> bool:
    """
    Download all data for a region.

    Args:
        region_name: Name of region
        bounds_data: Dict with lat_min, lat_max, lon_min, lon_max
        skip_osm: Skip OSM download
        skip_index: Skip index building
        skip_dem: Skip DEM download

    Returns:
        True if all steps successful
    """
    print(f"\n{'=' * 70}")
    print(f"  DATA SETUP FOR: {region_name}")
    print(f"{'=' * 70}\n")

    # Extract bounds
    lat_min = bounds_data["lat_min"]
    lat_max = bounds_data["lat_max"]
    lon_min = bounds_data["lon_min"]
    lon_max = bounds_data["lon_max"]

    # Determine required DEM datasets
    datasets = get_required_datasets(lat_min, lat_max, lon_min, lon_max)

    # Initialize DataManager (handles config updates automatically)
    data_mgr = DataManager()

    # Step 1: Download OSM data
    osm_file = None
    if not skip_osm:
        # Determine if this is a US state
        region_is_state = is_us_state(region_name)
        osm_file = data_mgr.download_osm_data(region_name, is_state=region_is_state)
        if not osm_file:
            print("\n❌ OSM download failed")
            return False
    else:
        # Find existing OSM file
        from utils.data_validator import find_osm_file_for_region

        osm_file = find_osm_file_for_region(region_name)
        if not osm_file:
            print(f"\n❌ No OSM file found for {region_name}")
            return False
        print(f"\n✓ Using existing OSM file: {osm_file}")

    # Step 2: Build spatial index
    if not skip_index:
        success = data_mgr.build_osm_index(osm_file)
        if not success:
            print("\n❌ Index build failed")
            return False
    else:
        print_section_simple("Skipping spatial index build", spacing_before=1)

    # Step 3: Download DEM data
    if not skip_dem:
        # Note: All elevation datasets now use public sources (December 2025)
        # No credentials required for SRTM, NED, or AW3D30
        success, new_files = data_mgr.download_elevation_data(
            region_name,
            (lat_min, lon_min, lat_max, lon_max),
            datasets,
            credentials=None
        )
        if not success:
            print("\n❌ DEM download failed")
            return False
    else:
        print_section_simple("Skipping DEM download", spacing_before=1)

    # Step 5: Final verification
    print_section_simple("Verifying all data", spacing_before=1)
    from utils.data_validator import validate_local_data

    is_valid, missing = validate_local_data(region_name, lat_min, lat_max, lon_min, lon_max, deployment_type='local')

    if is_valid:
        print("\n" + "=" * 70)
        print("  ✓ ALL DATA SETUP COMPLETE")
        print("=" * 70)
        print("\nYou can now run:")
        print("  python climb_analyzer.py")
        print(f"\nAnd select '{region_name}' for fast local analysis!\n")
        return True
    else:
        print("\n⚠️  Setup completed but some data is still missing:")
        for item in missing:
            print(f"  • {item}")
        return False


def interactive_setup():
    """Interactive setup wizard."""
    print("\n" + "=" * 70)
    print("  CLIMB ANALYZER - DATA DOWNLOAD WIZARD")
    print("=" * 70)
    print("\nThis wizard will download all required data for local deployment:")
    print("  1. OSM road network data (.pbf file)")
    print("  2. Generate a spatial index of OSM data")
    print("  3. Download appropriate elevation data")
    print("  4. Generate OpenTopoData configuration file")
    print("\n⚠️  Note: Downloads can be several GB and may take several hours\n")

    # Load bounds
    COUNTRY_BOUNDS, US_STATE_BOUNDS = load_geographic_bounds()

    # Select region type
    print("Select region type:")
    print("  1. Country")
    print("  2. US State")

    while True:
        choice = input("\nEnter choice (1-2): ").strip()
        if choice == "1":
            # Select countries
            region_type, selected_regions, _ = select_countries(COUNTRY_BOUNDS)
            scope_type = "country"
            break
        elif choice == "2":
            # Select US states
            region_type, selected_regions, _ = select_us_states(US_STATE_BOUNDS)
            scope_type = "state"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    if not selected_regions:
        print("\nNo regions selected. Exiting.")
        return

    # Process each selected region using the working data download flow
    all_success = True
    for region_name in selected_regions:
        # Use the working data coverage checker flow that's proven to work
        from climb_analyzer.data.data_coverage_checker import validate_and_prepare_data

        print(f"\n{'='*70}")
        print(f"  DOWNLOADING DATA FOR: {region_name}")
        print(f"{'='*70}\n")

        success = validate_and_prepare_data(scope_type, region_name)

        if success:
            # Build spatial index if needed
            from utils.data_validator import find_osm_file_for_region
            from climb_analyzer.data.index_builder import build_spatial_index, verify_index

            osm_file = find_osm_file_for_region(region_name)
            if osm_file and osm_file.exists():
                # Check if index already exists
                if not verify_index(str(osm_file), index_dir="data/osm_indexes"):
                    print(f"\n[Building Spatial Index for {region_name}]")
                    build_success = build_spatial_index(str(osm_file), output_dir="data/osm_indexes")
                    if not build_success:
                        print(f"⚠️  Index build failed for {region_name}")
                        all_success = False
                else:
                    print(f"✓ Spatial index already exists for {region_name}")

            print(f"\n✓ Data setup complete for {region_name}")
        else:
            all_success = False
            print(f"\n⚠️  Setup failed for {region_name}")

    if all_success:
        print("\n" + "=" * 70)
        print("  ✓ ALL REGIONS SETUP SUCCESSFULLY")
        print("=" * 70)
        print("\nYou can now run:")
        print("  ./climb-analyzer")
        print("\nOr:")
        print("  python climb_analyzer.py\n")
    else:
        print("\n⚠️  Some regions failed to setup completely")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare data for Climb Analyzer local deployment"
    )
    parser.add_argument("--country", type=str, help="Country name (e.g., 'Switzerland')")
    parser.add_argument("--state", type=str, help="US State name (e.g., 'Vermont')")
    parser.add_argument("--interactive", action="store_true", help="Run interactive setup wizard")
    parser.add_argument(
        "--skip-osm", action="store_true", help="Skip OSM download (use existing file)"
    )
    parser.add_argument("--skip-index", action="store_true", help="Skip spatial index building")
    parser.add_argument("--skip-dem", action="store_true", help="Skip DEM download")

    args = parser.parse_args()

    # Load bounds
    COUNTRY_BOUNDS, US_STATE_BOUNDS = load_geographic_bounds()

    # Interactive mode
    if args.interactive or (not args.country and not args.state):
        interactive_setup()
        return

    # Country mode
    if args.country:
        if args.country not in COUNTRY_BOUNDS:
            print(f"❌ Country '{args.country}' not found in database")
            print("\nAvailable countries:")
            for country in sorted(COUNTRY_BOUNDS.keys())[:20]:
                print(f"  {country}")
            if len(COUNTRY_BOUNDS) > 20:
                print(f"  ... and {len(COUNTRY_BOUNDS) - 20} more")
            sys.exit(1)

        bounds_data = COUNTRY_BOUNDS[args.country]
        success = setup_data_for_region(
            args.country,
            bounds_data,
            skip_osm=args.skip_osm,
            skip_index=args.skip_index,
            skip_dem=args.skip_dem,
        )
        sys.exit(0 if success else 1)

    # State mode
    if args.state:
        if args.state not in US_STATE_BOUNDS:
            print(f"❌ State '{args.state}' not found in database")
            print("\nAvailable states:")
            for state in sorted(US_STATE_BOUNDS.keys())[:20]:
                print(f"  {state}")
            sys.exit(1)

        bounds_data = US_STATE_BOUNDS[args.state]
        success = setup_data_for_region(
            args.state,
            bounds_data,
            skip_osm=args.skip_osm,
            skip_index=args.skip_index,
            skip_dem=args.skip_dem,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
