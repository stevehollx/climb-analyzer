#!/usr/bin/env python3
"""
Data Manager for Climb Analyzer

Centralized management of OSM planet files and elevation data.
Handles checking, downloading, and validating data sources.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from climb_analyzer.data.geo_lookup import find_region, get_region_bounds as lookup_bounds
from climb_analyzer.data.index_builder import build_spatial_index, verify_index
from climb_analyzer.data.osm_downloader import download_osm_for_location

# Import centralized data paths
try:
    from utils.data_paths import PLANET_OSM_DIR, ELEVATION_DATA_DIR, OSM_INDEXES_DIR
except ImportError:
    PLANET_OSM_DIR = Path("data/planet_osm_data")
    ELEVATION_DATA_DIR = Path("data/elevation_data")
    OSM_INDEXES_DIR = Path("data/osm_indexes")


class DataManager:
    """
    Centralized manager for OSM and elevation data.

    Responsibilities:
    - Check if data exists for a region
    - Download missing OSM planet files
    - Download missing elevation data
    - Build/verify spatial indexes
    - Update config.yaml with available data
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.planet_dir = PLANET_OSM_DIR
        self.elevation_dir = ELEVATION_DATA_DIR
        self.config = self._load_config()

        # Create directories if they don't exist
        self.planet_dir.mkdir(parents=True, exist_ok=True)
        self.elevation_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load config.yaml"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_config(self):
        """Save config.yaml"""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def get_available_osm_regions(self) -> List[str]:
        """
        Get list of regions with OSM data available.

        Returns:
            List of region names (countries or states)
        """
        available = self.config.get("OSM_COVERAGE", [])
        return available if isinstance(available, list) else []

    def get_available_elevation_datasets(self) -> Dict[str, List[str]]:
        """
        Get dict of available elevation datasets by region.

        Returns:
            Dict mapping dataset name to list of regions
        """
        return self.config.get("ELEVATION_DATASETS", {})

    def check_osm_data_exists(self, region_name: str) -> Optional[Path]:
        """
        Check if OSM data exists for a region.

        Args:
            region_name: Country or state name

        Returns:
            Path to .pbf file if exists, None otherwise
        """
        # Check if listed in config
        available = self.get_available_osm_regions()
        if region_name not in available:
            return None

        # Look for matching .pbf file
        # Try different naming patterns
        patterns = [
            f"*{region_name.lower().replace(' ', '-')}*.pbf",
            f"*{region_name.lower().replace(' ', '_')}*.pbf",
        ]

        for pattern in patterns:
            matches = list(self.planet_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def check_elevation_data_exists(self, region_name: str, dataset: str) -> bool:
        """
        Check if elevation data exists for a region and dataset.

        Args:
            region_name: Country or state name
            dataset: Dataset name (e.g., 'srtm', 'ned10m', 'aster')

        Returns:
            True if data exists
        """
        elevation_datasets = self.get_available_elevation_datasets()
        regions = elevation_datasets.get(dataset, [])
        return region_name in regions

    def download_osm_data(self, region_name: str, is_state: bool = False) -> Optional[Path]:
        """
        Download OSM data for a region.

        Args:
            region_name: Country or state name
            is_state: True if region_name is a US state

        Returns:
            Path to downloaded .pbf file, or None if failed
        """
        print(f"\n{'='*80}")
        print(f"  Downloading OSM data for {region_name}")
        print(f"{'='*80}\n")

        # Use download_osm_for_location which properly handles both countries and US states
        pbf_path = download_osm_for_location(region_name, output_dir=str(self.planet_dir))

        if pbf_path:
            # Update config with new region
            available = self.get_available_osm_regions()
            if region_name not in available:
                available.append(region_name)
                self.config["OSM_COVERAGE"] = sorted(available)
                self._save_config()

            return pbf_path

        return None

    def build_osm_index(self, pbf_path: Path) -> bool:
        """
        Build spatial index for OSM .pbf file.

        Args:
            pbf_path: Path to .pbf file

        Returns:
            True if successful
        """
        print(f"\n{'='*80}")
        print(f"  Building spatial index for {pbf_path.name}")
        print(f"{'='*80}\n")

        success = build_spatial_index(str(pbf_path))

        if success:
            # Verify the index was created
            if verify_index(str(pbf_path)):
                print("  ✓ Spatial index built and verified")
                return True
            else:
                print("  ⚠️  Index built but verification failed")
                return False

        return False

    def download_elevation_data(
        self,
        region_name: str,
        bounds: Tuple[float, float, float, float],
        datasets: List[str],
        credentials: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> tuple:
        """
        Download elevation data for a region.

        Args:
            region_name: Country or state name
            bounds: Bounding box (lat_min, lon_min, lat_max, lon_max)
            datasets: List of dataset names to download
            credentials: Dict mapping dataset type to (username, password)
                       e.g., {'earthdata': (user, pass)}

        Returns:
            Tuple of (success: bool, new_files_count: int)
            - success: True if at least one dataset downloaded successfully
            - new_files_count: Total number of new files downloaded across all datasets
        """
        print(f"\n{'='*80}")
        print(f"  Downloading elevation data for {region_name}")
        print(f"{'='*80}\n")

        lat_min, lon_min, lat_max, lon_max = bounds
        print(f"  Bounding box: ({lat_min:.2f}, {lon_min:.2f}) to ({lat_max:.2f}, {lon_max:.2f})")
        print(f"  Datasets: {', '.join(datasets)}")

        # Import downloaders
        try:
            from .dem_downloaders import (
                ArcticDEMDownloader,
                ASTERDownloader,
                AW3D30Downloader,
                NEDDownloader,
                REMADownloader,
                SRTMDownloader,
            )
        except ImportError as e:
            print(f"  ❌ Error importing dem_downloaders: {e}")
            return False

        # Extract credentials
        creds = credentials or {}
        earthdata_creds = creds.get("earthdata")

        # Initialize downloaders
        downloaders = {
            "ned10m": NEDDownloader(self.elevation_dir / "ned10m"),
            "srtm": SRTMDownloader(self.elevation_dir / "srtm30m", credentials=earthdata_creds),
            "srtm30m": SRTMDownloader(self.elevation_dir / "srtm30m", credentials=earthdata_creds),  # Alias
            "aster": ASTERDownloader(self.elevation_dir / "aster30m", credentials=earthdata_creds),
            "aw3d30": AW3D30Downloader(self.elevation_dir / "aw3d30"),  # Uses public FTP
            "arcticdem": ArcticDEMDownloader(self.elevation_dir / "arctic32m"),
            "rema": REMADownloader(self.elevation_dir / "rema32m"),
        }

        bbox = (lat_min, lon_min, lat_max, lon_max)
        all_success = True
        total_new_files = 0  # Track total new files downloaded

        for dataset_name in datasets:
            dataset_lower = dataset_name.lower()

            if dataset_lower not in downloaders:
                print(f"\n  ⚠️  Unknown dataset: {dataset_name}")
                continue

            print(f"\n  Downloading {dataset_name.upper()}...")

            downloader = downloaders[dataset_lower]
            try:
                success, new_files_count = downloader.download_bbox(bbox)
                total_new_files += new_files_count

                if success:
                    # Update config with new dataset availability
                    elevation_datasets = self.get_available_elevation_datasets()
                    if dataset_lower not in elevation_datasets:
                        elevation_datasets[dataset_lower] = []

                    regions = elevation_datasets[dataset_lower]
                    if region_name not in regions:
                        regions.append(region_name)
                        elevation_datasets[dataset_lower] = sorted(regions)

                    self.config["ELEVATION_DATASETS"] = elevation_datasets
                    self._save_config()

                    if new_files_count > 0:
                        print(f"  ✓ {dataset_name.upper()} downloaded {new_files_count} new file(s)")
                    else:
                        print(f"  ✓ {dataset_name.upper()} - all files already exist")
                else:
                    print(f"  ⚠️  {dataset_name.upper()} download had issues")
                    all_success = False

            except Exception as e:
                print(f"  ❌ {dataset_name.upper()} download failed: {e}")
                all_success = False

        # Return success status and new files count
        return (all_success, total_new_files)

    def restart_opentopodata(self) -> bool:
        """
        Restart OpenTopoData container to pick up new elevation data.

        Returns:
            True if successful
        """
        print(f"\n{'='*80}")
        print("  Restarting OpenTopoData server")
        print(f"{'='*80}\n")

        try:
            # First check if opentopodata container exists
            check_result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=opentopodata-server", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=False
            )

            container_exists = "opentopodata-server" in check_result.stdout

            if not container_exists:
                print("     OpenTopoData container not found - skipping restart")
                print("  Note: Start OpenTopoData manually with: cd opentopodata && make daemon")
                return True  # Return True to not block the workflow

            # Stop existing container
            subprocess.run(
                ["docker", "stop", "opentopodata-server"],
                capture_output=True,
                check=False,
                timeout=30
            )

            # Start container
            result = subprocess.run(
                ["docker", "start", "opentopodata-server"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )

            if result.returncode == 0:
                print("  ✓ OpenTopoData restarted successfully")
                return True
            else:
                print(f"  ⚠️  OpenTopoData restart had issues: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("  ⚠️  OpenTopoData restart timed out")
            return False
        except Exception as e:
            print(f"  ⚠️  Error restarting OpenTopoData: {e}")
            print("  Note: You may need to restart it manually with: cd opentopodata && make daemon")
            return True  # Return True to not block the workflow

    def get_region_bounds(
        self, region_name: str, is_state: bool = False
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Get bounding box for a region.

        Uses the consolidated osm_pbf_urls dictionary which contains bounds
        from Geofabrik .poly files for all regions.

        Args:
            region_name: Country, state name, or hierarchical region path
                        Examples: "france", "bristol", "california", "europe > isle-of-man"
            is_state: Deprecated - no longer used (kept for backward compatibility)

        Returns:
            Tuple of (lat_min, lon_min, lat_max, lon_max) or None
        """
        # Handle hierarchical paths like "europe > isle-of-man" or "england > bristol"
        if " > " in region_name:
            # Extract the last part (actual region name)
            parts = [p.strip() for p in region_name.split(" > ")]
            search_name = parts[-1]
        else:
            search_name = region_name

        # Use the new geo_lookup module which searches osm_pbf_urls recursively
        # This handles all regions: countries, states, subregions (like Bristol)
        return lookup_bounds(search_name)

    def ensure_data_ready(
        self,
        region_name: str,
        is_state: bool = False,
        elevation_datasets: Optional[List[str]] = None,
        credentials: Optional[Dict[str, Tuple[str, str]]] = None,
        batch_mode: bool = False,
        delete_after: bool = False,
    ) -> Tuple[bool, Optional[Path]]:
        """
        Ensure all required data is ready for a region.
        Downloads missing data automatically.

        Args:
            region_name: Country or state name
            is_state: True if US state
            elevation_datasets: List of elevation datasets needed
            credentials: Credentials dict for downloads
            batch_mode: If True, skip confirmations
            delete_after: If True (batch mode), delete data after use

        Returns:
            Tuple of (success: bool, pbf_path: Optional[Path])
        """
        print(f"\n{'='*80}")
        print(f"  Checking data availability for {region_name}")
        print(f"{'='*80}\n")

        # Check OSM data
        pbf_path = self.check_osm_data_exists(region_name)

        if not pbf_path:
            print(f"  OSM data not found for {region_name}")

            if not batch_mode:
                response = input(f"  Download OSM data for {region_name}? [Y/n]: ").strip().lower()
                if response and response != "y":
                    print("  ❌ OSM data required. Aborting.")
                    return False, None

            pbf_path = self.download_osm_data(region_name, is_state=is_state)

            if not pbf_path:
                print("  ❌ Failed to download OSM data")
                return False, None

            # Build spatial index
            if not self.build_osm_index(pbf_path):
                print("  ⚠️  Spatial index build failed")
        else:
            print(f"  ✓ OSM data found: {pbf_path.name}")

            # Verify index exists
            index_path = pbf_path.with_suffix(".idx")
            if not index_path.exists():
                print("  Spatial index not found, building...")
                self.build_osm_index(pbf_path)

        # Check elevation data
        if elevation_datasets:
            bounds = self.get_region_bounds(region_name, is_state=is_state)
            if not bounds:
                print(f"  ⚠️  Could not determine bounds for {region_name}")
                return True, pbf_path

            missing_datasets = []
            for dataset in elevation_datasets:
                if not self.check_elevation_data_exists(region_name, dataset):
                    missing_datasets.append(dataset)

            if missing_datasets:
                print(f"\n  Missing elevation datasets: {', '.join(missing_datasets)}")

                if not batch_mode:
                    response = input("  Download elevation data? [Y/n]: ").strip().lower()
                    if response and response != "y":
                        print("  ⚠️  Continuing without elevation data")
                        return True, pbf_path

                # Download missing elevation data
                success = self.download_elevation_data(
                    region_name, bounds, missing_datasets, credentials=credentials
                )

                if success and self.config.get("DEPLOYMENT_TYPE") == "local":
                    # Restart OpenTopoData to pick up new data
                    self.restart_opentopodata()

        return True, pbf_path

    def cleanup_region_data(self, region_name: str, is_state: bool = False):
        """
        Delete OSM and elevation data for a region (for batch mode cleanup).

        Args:
            region_name: Country or state name
            is_state: True if US state
        """
        print(f"\n  Cleaning up data for {region_name}...")

        # Remove OSM file
        pbf_path = self.check_osm_data_exists(region_name)
        if pbf_path and pbf_path.exists():
            pbf_path.unlink()
            # Also remove index
            index_path = pbf_path.with_suffix(".idx")
            if index_path.exists():
                index_path.unlink()
            print("    ✓ Removed OSM data")

        # Remove from config
        available = self.get_available_osm_regions()
        if region_name in available:
            available.remove(region_name)
            self.config["OSM_COVERAGE"] = available

        # Remove elevation data references
        elevation_datasets = self.get_available_elevation_datasets()
        for dataset, regions in elevation_datasets.items():
            if region_name in regions:
                regions.remove(region_name)
                elevation_datasets[dataset] = regions

        self.config["ELEVATION_DATASETS"] = elevation_datasets
        self._save_config()

        print(f"    ✓ Cleaned up {region_name} data")


# DEPRECATED: NASA Earthdata credentials no longer needed (December 2025)
# All elevation datasets now use public sources:
# - SRTM: OpenTopography S3 (public)
# - AW3D30: JAXA FTP (public)
# - NED: AWS S3 (public)
# - ASTER: Data source retired December 2025

def get_credentials_from_netrc() -> Dict[str, Tuple[str, str]]:
    """DEPRECATED: Returns empty dict. NASA Earthdata credentials no longer needed."""
    return {}


def get_credentials_from_env() -> Dict[str, Tuple[str, str]]:
    """DEPRECATED: Returns empty dict. NASA Earthdata credentials no longer needed."""
    return {}


def prompt_for_credentials(datasets: List[str]) -> Dict[str, Tuple[str, str]]:
    """DEPRECATED: Returns empty dict. NASA Earthdata credentials no longer needed."""
    return {}
