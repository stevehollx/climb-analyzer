#!/usr/bin/env python3
"""
DEM (Digital Elevation Model) Downloaders

This module provides downloaders for various global DEM datasets:
- NED 10m: National Elevation Dataset 10m (USGS) - US only
- SRTM 30m: Shuttle Radar Topography Mission 30m (NASA) - Global 60°N to 56°S
- AW3D30: ALOS World 3D 30m (JAXA)
- ASTER GDEM v3: 30m (NASA/METI)
- REMA: Reference Elevation Model of Antarctica 32m (PGC)
- ArcticDEM: Arctic Digital Elevation Model 32m (PGC)

All downloaders support bounding box-based downloads and handle tile management.
"""

import math
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
import yaml
from tqdm import tqdm

# Type alias for bounding box
BoundingBox = Tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)


def _print_download_header(title: str, subtitle: str = "", bbox_str: str = "", output_dir: str = "", source: str = ""):
    """Print a modern formatted header for DEM downloads using formatter.py."""
    from climb_analyzer.utils.formatting import print_header, print_info

    # Print main header using formatter
    print_header(title, spacing_before=1)

    # Print additional details
    if subtitle:
        print_info(subtitle, indent=2)
        print()
    if bbox_str:
        print(f"  Bounding box: {bbox_str}")
    if output_dir:
        print(f"  Output directory: {output_dir}")
    if source:
        print(f"  Source: {source}")
    if bbox_str or output_dir or source:
        print()


# =============================================================================
# DEPRECATED: NASA Earthdata authentication code (December 2025)
# All elevation datasets now use public sources:
# - SRTM: OpenTopography S3 (public)
# - AW3D30: JAXA FTP (public)
# - NED: AWS S3 (public)
# - ASTER: NASA LP DAAC Data Pool retired December 2025
# =============================================================================

def setup_earthdata_netrc(username: str, password: str) -> bool:
    """DEPRECATED: No-op. NASA Earthdata credentials no longer needed."""
    return True


class SessionWithHeaderRedirection(requests.Session):
    """DEPRECATED: NASA Earthdata auth class, no longer needed."""

    def __init__(self, username: str, password: str):
        super().__init__()
        # No auth needed anymore


def create_earthdata_session(username: str, password: str) -> requests.Session:
    """DEPRECATED: Returns plain session. NASA Earthdata credentials no longer needed."""
    return requests.Session()


class BaseDEMDownloader:
    """Base class for DEM downloaders with common functionality."""

    # Dataset priority (lower number = higher priority, don't download if covered)
    DATASET_PRIORITY = {
        'rema32m': 1,
        'arcticdem32m': 2,
        'rema': 3,
        'ned10m': 4,
        'srtm30m': 5,
        'aw3d30': 6,
        'aster30m': 7,
    }

    def __init__(self, output_dir: Path, dataset_name: str):
        """
        Initialize DEM downloader.

        Args:
            output_dir: Directory to save downloaded tiles
            dataset_name: Name of the dataset for logging
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.unavailable_file = self.output_dir / ".unavailable"

        # Determine elevation_data root (parent of this dataset's dir)
        # e.g., elevation_data/srtm30m -> elevation_data
        self.elevation_data_dir = self.output_dir.parent

    def is_tile_unavailable(self, tile_name: str) -> bool:
        """
        Check if a tile is known to be unavailable (404 from previous attempts).

        Args:
            tile_name: Name of the tile to check

        Returns:
            True if tile is known to be unavailable
        """
        if not self.unavailable_file.exists():
            return False

        try:
            with open(self.unavailable_file) as f:
                unavailable_tiles = set(line.strip() for line in f if line.strip())
            return tile_name in unavailable_tiles
        except Exception:
            return False

    def mark_tile_unavailable(self, tile_name: str):
        """
        Mark a tile as unavailable (404) to skip on future runs.

        Args:
            tile_name: Name of the tile that returned 404
        """
        try:
            # Read existing unavailable tiles
            existing_tiles = set()
            if self.unavailable_file.exists():
                with open(self.unavailable_file) as f:
                    existing_tiles = set(line.strip() for line in f if line.strip())

            # Add new tile and write back
            existing_tiles.add(tile_name)
            with open(self.unavailable_file, "w") as f:
                for tile in sorted(existing_tiles):
                    f.write(f"{tile}\n")
        except Exception:
            # Silently fail - not critical
            pass

    def _clear_unavailable_tiles(self, tiles_to_remove: list):
        """
        Remove specific tiles from the unavailable cache.

        Used to roll back false positive unavailable marks when
        a systemic issue (auth failure, server down) is detected.

        Args:
            tiles_to_remove: List of tile names to remove from unavailable cache
        """
        if not tiles_to_remove:
            return

        try:
            if not self.unavailable_file.exists():
                return

            # Read existing unavailable tiles
            with open(self.unavailable_file) as f:
                existing_tiles = set(line.strip() for line in f if line.strip())

            # Remove the specified tiles
            tiles_to_remove_set = set(tiles_to_remove)
            remaining_tiles = existing_tiles - tiles_to_remove_set

            # Write back remaining tiles
            if remaining_tiles:
                with open(self.unavailable_file, "w") as f:
                    for tile in sorted(remaining_tiles):
                        f.write(f"{tile}\n")
            else:
                # No tiles left, remove the file
                self.unavailable_file.unlink()
        except Exception:
            # Silently fail - not critical
            pass

    def _tile_has_valid_data(self, tile_path: Path, min_valid_percent: float = 1.0) -> bool:
        """
        Check if a tile has sufficient valid (non-NODATA) data.

        This prevents treating tiles that are mostly NODATA (like border regions)
        as valid coverage that would block lower-priority datasets from downloading.

        Args:
            tile_path: Path to the tile file
            min_valid_percent: Minimum percentage of valid pixels required (default 1.0%)

        Returns:
            True if tile has >= min_valid_percent valid data, False otherwise
        """
        try:
            import subprocess

            # Use gdalinfo to get statistics
            result = subprocess.run(
                ['gdalinfo', '-stats', str(tile_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                # If gdalinfo fails, assume tile is valid (conservative approach)
                return True

            # Look for STATISTICS_VALID_PERCENT in output
            for line in result.stdout.split('\n'):
                if 'STATISTICS_VALID_PERCENT' in line:
                    # Extract percentage value
                    # Format: "    STATISTICS_VALID_PERCENT=0.1558"
                    parts = line.split('=')
                    if len(parts) == 2:
                        try:
                            valid_percent = float(parts[1].strip())
                            return valid_percent >= min_valid_percent
                        except ValueError:
                            pass

            # If no STATISTICS_VALID_PERCENT found, assume tile is valid
            return True

        except Exception:
            # On any error, conservatively assume tile is valid
            return True

    def is_tile_covered_by_higher_priority(self, tile_coords: str) -> tuple[bool, str | None]:
        """
        Check if a tile coordinate is already covered by a higher-priority dataset with valid data.

        This prevents redundant downloads but allows overlapping tiles to coexist. If a higher-priority
        dataset has the tile but it's mostly NODATA, this will still return False to allow downloading
        from the current dataset (preserving fallback coverage).

        Args:
            tile_coords: Normalized tile coordinates (e.g., 'N30W082')

        Returns:
            Tuple of (is_covered, dataset_name)
            - is_covered: True if a higher-priority dataset has this tile WITH VALID DATA
            - dataset_name: Name of the covering dataset, or None
        """
        # Get priority of current dataset
        my_priority = self.DATASET_PRIORITY.get(self.dataset_name, 999)

        # Check all higher-priority datasets
        for dataset_name, priority in self.DATASET_PRIORITY.items():
            if priority >= my_priority:
                # Same or lower priority, skip
                continue

            # Check multiple possible dataset directory locations
            # 1. elevation_data/{dataset} - download directory
            # 2. opentopodata/data/{dataset} - OpenTopoData directory (production)
            possible_dirs = [
                self.elevation_data_dir / dataset_name,
                self.elevation_data_dir.parent / "opentopodata" / "data" / dataset_name,
                Path("opentopodata/data") / dataset_name,  # Relative path
            ]

            for dataset_dir in possible_dirs:
                if not dataset_dir.exists():
                    continue

                # Look for files matching this coordinate
                # Common patterns: N30W082.hgt, USGS_13_n30w082.tif, N030W082.tif, etc.
                # IMPORTANT: Use exact matching to avoid false positives
                patterns = [
                    f"{tile_coords}.*",  # Direct match - N37W120.hgt
                    f"{tile_coords.lower()}.*",  # Lowercase - n37w120.hgt
                    f"USGS_*{tile_coords.lower()}.*",  # NED prefix - USGS_13_n37w120.tif
                    f"ASTGTMV003_{tile_coords}_*",  # ASTER format
                ]

                for pattern in patterns:
                    matching_files = list(dataset_dir.glob(pattern))
                    if matching_files:
                        # Verify this is an exact match, not a partial match
                        # E.g., n37w120.tif should match, but n37w081.tif should NOT match pattern "n37w120.*"
                        for f in matching_files:
                            # Extract coordinate from filename (handles various formats)
                            fname = f.stem.lower()  # n37w120 or usgs_13_n37w120
                            target = tile_coords.lower()  # n37w120

                            # Check if filename contains exact coordinate
                            if target in fname:
                                # Make sure it's not a substring match
                                # n37w120 should match "n37w120.hgt" but not "n37w1200.hgt"
                                # Find where the target appears
                                idx = fname.find(target)
                                if idx != -1:
                                    # Check that the match is complete (not followed by digit)
                                    after_idx = idx + len(target)
                                    if after_idx >= len(fname) or not fname[after_idx].isdigit():
                                        # Found matching file - verify it has valid data before considering it as coverage
                                        if self._tile_has_valid_data(f):
                                            return True, dataset_name
                                        # Tile exists but has mostly NODATA - don't consider it as coverage

        return False, None

    def download_file(
        self,
        url: str,
        output_path: Path,
        desc: Optional[str] = None,
        timeout: int = 300,
        auth: tuple[str, str] | None = None,
        session: requests.Session | None = None,
        silent_404: bool = True,
        position: int = 0,
        leave: bool = True,
        colour: str | None = None,
    ) -> tuple[bool, int | None]:
        """
        Download a file with progress bar.

        Args:
            url: URL to download (supports HTTP/HTTPS/FTP)
            output_path: Path to save file
            desc: Description for progress bar
            timeout: Request timeout in seconds
            auth: Optional tuple of (username, password) for HTTP basic auth
            session: Optional requests.Session for authenticated downloads
            silent_404: If True, don't print errors for 404s (tiles over water/uncovered areas)
            position: tqdm position (0=main bar, 1=sub bar for nested displays)
            leave: If False, remove progress bar when done (for sub-bars)
            colour: Progress bar color (e.g., 'white' for grey sub-bars)

        Returns:
            Tuple of (success: bool, status_code: int | None)
            status_code is None for FTP or non-HTTP errors
        """
        try:
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle FTP URLs separately
            if url.startswith("ftp://"):
                from ftplib import FTP
                from urllib.parse import urlparse

                parsed = urlparse(url)
                ftp_host = parsed.hostname
                ftp_path = parsed.path

                # Connect and download
                ftp = FTP(ftp_host, timeout=timeout)
                ftp.login()  # Anonymous login

                try:
                    # Get file size for progress bar
                    ftp.voidcmd("TYPE I")  # Binary mode
                    total_size = ftp.size(ftp_path)

                    with open(output_path, "wb") as f:
                        if total_size and total_size > 0:
                            with tqdm(
                                total=total_size,
                                unit="B",
                                unit_scale=True,
                                unit_divisor=1024,
                                desc=f"{desc or output_path.name}",
                                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                                ncols=100,
                                ascii=" █",
                                position=position,
                                leave=leave,
                                colour=colour,
                            ) as pbar:

                                def callback(data):
                                    f.write(data)
                                    pbar.update(len(data))

                                ftp.retrbinary(f"RETR {ftp_path}", callback)
                        else:
                            # No size available, download without progress
                            ftp.retrbinary(f"RETR {ftp_path}", f.write)

                    ftp.quit()
                    return (True, None)  # FTP doesn't have HTTP status codes

                except Exception as e:
                    ftp.quit()
                    # Check if it's a file not found error (ocean/water tile)
                    error_str = str(e)
                    if "550" in error_str or "No such file" in error_str.lower() or "not found" in error_str.lower():
                        # File doesn't exist on FTP (expected for ocean tiles) - fail silently
                        if output_path.exists():
                            output_path.unlink()
                        return (False, None)
                    else:
                        # Unexpected FTP error - re-raise
                        raise e

            # Handle HTTP/HTTPS URLs with requests
            else:
                if session:
                    response = session.get(url, stream=True, timeout=timeout)
                else:
                    response = requests.get(url, stream=True, timeout=timeout, auth=auth)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(output_path, "wb") as f:
                    if total_size > 0:
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=f"{desc or output_path.name}",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                            ncols=100,
                            ascii=" █",
                            position=position,
                            leave=leave,
                            colour=colour,
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # No content length, just download
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                return (True, response.status_code)

        except requests.exceptions.HTTPError as e:
            # Extract status code if available
            status_code = e.response.status_code if hasattr(e, "response") else None

            # Handle 404s silently (tiles over water or uncovered areas are expected)
            if status_code == 404 and silent_404:
                # Don't print error, this is expected for water/uncovered areas
                pass
            else:
                print(f"Error downloading {url}: {e}")

            if output_path.exists():
                output_path.unlink()
            return (False, status_code)

        except requests.exceptions.RequestException as e:
            # Other network errors (timeouts, connection errors, etc.)
            print(f"Error downloading {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return (False, None)

        except Exception as e:
            # Only print error if it's not an expected FTP 550 (file not found) error
            error_str = str(e)
            if not ("550" in error_str or "No such file" in error_str.lower()):
                print(f"Error downloading {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return (False, None)

    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """
        Extract a zip archive.

        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to

        Returns:
            True if successful, False otherwise
        """
        try:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            return True
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False

    def download_bbox(self, bbox: BoundingBox) -> tuple:
        """
        Download DEM tiles for a bounding box.
        Must be implemented by subclasses.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            Tuple of (success: bool, new_files_count: int)
            - success: True if at least one tile available (existing or downloaded)
            - new_files_count: Number of new files actually downloaded
        """
        raise NotImplementedError("Subclass must implement download_bbox()")

    def export_tile_urls(self, bbox: BoundingBox, output_file: Path) -> bool:
        """
        Export tile URLs for a bounding box to a file for manual download.
        Must be implemented by subclasses.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
            output_file: Path to output file

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclass must implement export_tile_urls()")


class AW3D30Downloader(BaseDEMDownloader):
    """
    Downloader for ALOS World 3D 30m (AW3D30) dataset.

    JAXA's AW3D30 is a high-quality global DEM with 30m resolution.
    Data is organized in 5° x 5° tiles.

    Source: https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm
    """

    # AW3D30 public FTP server - no authentication required
    BASE_URL = "ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v2404/"

    def __init__(self, output_dir: Path, credentials: tuple[str, str] | None = None):
        """Initialize AW3D30 downloader.

        Args:
            output_dir: Directory to save downloads
            credentials: Not used - kept for backward compatibility
        """
        super().__init__(output_dir, "AW3D30")
        # Credentials not needed for public FTP access
        self.credentials = None

    def get_tile_name(self, lat: float, lon: float) -> str:
        """
        Get tile name for a given coordinate.

        AW3D30 tiles are named based on the southwest corner of each 5° x 5° tile.
        Format: N000E000 where N/S is latitude and E/W is longitude.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Tile name (e.g., "N035E135")
        """
        # Round down to nearest 5 degrees
        tile_lat = math.floor(lat / 5) * 5
        tile_lon = math.floor(lon / 5) * 5

        # Format: N/S + 3-digit lat + E/W + 3-digit lon
        lat_str = f"{'N' if tile_lat >= 0 else 'S'}{abs(tile_lat):03d}"
        lon_str = f"{'E' if tile_lon >= 0 else 'W'}{abs(tile_lon):03d}"

        return f"{lat_str}{lon_str}"

    def get_tiles_for_bbox(self, bbox: BoundingBox) -> List[str]:
        """
        Get list of tile names covering a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            List of tile names
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        tiles = set()

        # Iterate over 5° grid
        lat = math.floor(min_lat / 5) * 5
        while lat <= max_lat:
            lon = math.floor(min_lon / 5) * 5
            while lon <= max_lon:
                tiles.add(self.get_tile_name(lat, lon))
                lon += 5
            lat += 5

        return sorted(tiles)

    def _subtile_intersects_bbox(self, subtile_name: str, bbox: BoundingBox) -> bool:
        """
        Check if a 1° x 1° subtile intersects with a bounding box.

        Args:
            subtile_name: Subtile name (e.g., "N041E023")
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            True if subtile intersects with bbox
        """
        # Parse subtile name to get lat/lon
        # Format: N041E023 or S041W023
        import re
        match = re.match(r'([NS])(\d{2,3})([EW])(\d{2,3})', subtile_name)
        if not match:
            return False

        lat_dir, lat_val, lon_dir, lon_val = match.groups()
        tile_lat = int(lat_val) if lat_dir == 'N' else -int(lat_val)
        tile_lon = int(lon_val) if lon_dir == 'E' else -int(lon_val)

        # 1° x 1° subtile bounds
        subtile_min_lat = tile_lat
        subtile_max_lat = tile_lat + 1
        subtile_min_lon = tile_lon
        subtile_max_lon = tile_lon + 1

        # Check intersection
        min_lat, min_lon, max_lat, max_lon = bbox
        return not (subtile_max_lat <= min_lat or subtile_min_lat >= max_lat or
                    subtile_max_lon <= min_lon or subtile_min_lon >= max_lon)

    def download_tile(self, tile_name: str, bbox: BoundingBox = None, main_pbar=None) -> tuple:
        """
        Download a single AW3D30 tile (5x5 degree tile containing 1x1 degree subtiles).

        Args:
            tile_name: Tile name (e.g., "N045E005" for the 5x5 degree tile)
            bbox: Optional bounding box (min_lat, min_lon, max_lat, max_lon) to filter subtiles
            main_pbar: Optional main progress bar to update

        Returns:
            Tuple of (success: bool, is_water_tile: bool, files_downloaded: int)
            - success: True if tile is complete (downloaded or already existed)
            - is_water_tile: True if tile directory doesn't exist (water/ocean area)
            - files_downloaded: Number of new files actually downloaded
        """

        # AW3D30 v2404 structure: release_v2404/{5x5_tile}/{1x1_tile}.zip
        # The 5x5 tile (e.g., N045E005) is a directory containing 1x1 degree tiles
        # We need to download all the zip files inside that directory

        # OPTIMIZATION: Check if all subtiles already exist locally before connecting to FTP
        # This avoids unnecessary FTP connections when tiles are already downloaded
        import re
        match = re.match(r'([NS])(\d{3})([EW])(\d{3})', tile_name)
        if match:
            lat_dir, lat_str, lon_dir, lon_str = match.groups()
            base_lat = int(lat_str) * (1 if lat_dir == 'N' else -1)
            base_lon = int(lon_str) * (1 if lon_dir == 'E' else -1)

            # Check which subtiles we would download given the bbox filter
            subtiles_needed = []
            for lat in range(base_lat, base_lat + 5):
                for lon in range(base_lon, base_lon + 5):
                    subtile_lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):03d}"
                    subtile_lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                    subtile_name = f"{subtile_lat_str}{subtile_lon_str}"

                    # If bbox provided, check if subtile intersects with bbox
                    if bbox and not self._subtile_intersects_bbox(subtile_name, bbox):
                        continue

                    # Check if this subtile already exists
                    original_file = self.output_dir / f"ALPSMLC30_{subtile_name}_DSM.tif"
                    renamed_file = self.output_dir / f"{subtile_name}.tif"
                    if not (original_file.exists() or renamed_file.exists()):
                        subtiles_needed.append(subtile_name)

            # If all subtiles already exist, return immediately without FTP connection
            if not subtiles_needed:
                return (True, False, 0)  # Already complete, no new files

        try:
            import zipfile
            from ftplib import FTP, error_perm

            # Parse FTP URL to get host and path
            # BASE_URL format: ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v2404/
            ftp_host = "ftp.eorc.jaxa.jp"
            ftp_path = f"/pub/ALOS/ext1/AW3D30/release_v2404/{tile_name}/"

            # Connect to FTP server and list directory
            ftp = FTP(ftp_host, timeout=30)
            ftp.login()  # Anonymous login

            try:
                # Get list of files in the directory
                files = []
                try:
                    ftp.dir(ftp_path, files.append)
                except error_perm as e:
                    # FTP error 450 or 550 means directory doesn't exist (water/ocean tile)
                    error_msg = str(e)
                    if "450" in error_msg or "550" in error_msg or "No such file" in error_msg:
                        ftp.quit()
                        # Suppress output for water tiles - this is expected
                        return (False, True, 0)  # Water tile
                    else:
                        ftp.quit()
                        raise e

                # Extract zip filenames from directory listing
                zip_files = []
                for line in files:
                    # FTP dir format: "-rw-r--r--   1 ftp      ftp      123456 Jan 01 12:00 filename.zip"
                    parts = line.split()
                    if parts and parts[-1].endswith(".zip"):
                        zip_files.append(parts[-1])

                # Get set of subtile names that exist on FTP (for later unavailable check)
                subtiles_on_ftp = set(z.replace(".zip", "") for z in zip_files)

                if not zip_files:
                    ftp.quit()
                    return (False, False, 0)  # No files to download

                # Check which subtiles already exist and intersect with bbox (if provided)
                # Each zip file like "N051W131.zip" extracts to a folder with ALPSMLC30_N051W131_DSM.tif
                subtiles_to_download = []
                skipped_existing = 0
                for zip_file in zip_files:
                    # Extract subtile name (e.g., "N051W131" from "N051W131.zip")
                    subtile_name = zip_file.replace(".zip", "")

                    # If bbox provided, check if subtile intersects with bbox
                    if bbox and not self._subtile_intersects_bbox(subtile_name, bbox):
                        continue

                    # Check if the main DSM tif file already exists in output directory
                    # Check both original format and renamed format (from preparation)
                    original_file = self.output_dir / f"ALPSMLC30_{subtile_name}_DSM.tif"
                    renamed_file = self.output_dir / f"{subtile_name}.tif"
                    if not (original_file.exists() or renamed_file.exists()):
                        subtiles_to_download.append(zip_file)
                    else:
                        skipped_existing += 1

                # Check for subtiles that are needed but NOT on FTP server (water/ocean tiles)
                # These should be marked as unavailable so we don't keep trying to download them
                subtiles_not_on_ftp = []
                for lat in range(base_lat, base_lat + 5):
                    for lon in range(base_lon, base_lon + 5):
                        subtile_lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):03d}"
                        subtile_lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                        subtile_name = f"{subtile_lat_str}{subtile_lon_str}"

                        # Skip if not needed for bbox
                        if bbox and not self._subtile_intersects_bbox(subtile_name, bbox):
                            continue

                        # Skip if already exists locally
                        original_file = self.output_dir / f"ALPSMLC30_{subtile_name}_DSM.tif"
                        renamed_file = self.output_dir / f"{subtile_name}.tif"
                        if original_file.exists() or renamed_file.exists():
                            continue

                        # If needed but not on FTP, it's a water tile
                        if subtile_name not in subtiles_on_ftp:
                            subtiles_not_on_ftp.append(subtile_name)

                if not subtiles_to_download:
                    # Before returning, cache any subtiles not on FTP as unavailable
                    if subtiles_not_on_ftp:
                        subtile_unavailable_file = self.output_dir / ".unavailable_subtiles"
                        unavailable_subtiles = set()
                        if subtile_unavailable_file.exists():
                            try:
                                with open(subtile_unavailable_file, 'r') as f:
                                    unavailable_subtiles = set(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
                            except Exception:
                                pass

                        new_unavailable = [s for s in subtiles_not_on_ftp if s not in unavailable_subtiles]
                        if new_unavailable:
                            unavailable_subtiles.update(new_unavailable)
                            try:
                                with open(subtile_unavailable_file, 'w') as f:
                                    f.write("# Subtiles that are permanently unavailable (ocean/water areas)\n")
                                    f.write("# These 1x1 degree tiles don't exist on JAXA's FTP server\n\n")
                                    for subtile in sorted(unavailable_subtiles):
                                        f.write(f"{subtile}\n")
                            except Exception:
                                pass  # Silently handle unavailable cache save errors

                    ftp.quit()
                    return (True, False, 0)  # Already complete, no new files

                # Show what's being downloaded for smaller batches
                if len(subtiles_to_download) <= 10 and len(subtiles_to_download) > 0:
                    print(f"       Downloading: {', '.join([z.replace('.zip', '') for z in subtiles_to_download])}")

                # Load subtile-level unavailable cache
                subtile_unavailable_file = self.output_dir / ".unavailable_subtiles"
                unavailable_subtiles = set()
                if subtile_unavailable_file.exists():
                    try:
                        with open(subtile_unavailable_file, 'r') as f:
                            unavailable_subtiles = set(line.strip() for line in f if line.strip())
                    except Exception:
                        pass

                # Filter out unavailable subtiles BEFORE attempting download
                original_count = len(subtiles_to_download)
                subtiles_to_download = [z for z in subtiles_to_download if z.replace(".zip", "") not in unavailable_subtiles]
                skipped_unavailable = original_count - len(subtiles_to_download)

                # Download and extract each missing subtile with sub-progress bars
                success_count = 0
                downloaded_files = []
                failed_subtiles = []
                for zip_file in subtiles_to_download:
                    subtile_name = zip_file.replace(".zip", "")

                    subtile_url = f"ftp://{ftp_host}{ftp_path}{zip_file}"
                    output_path = self.output_dir / zip_file

                    # Download with sub-progress bar (grey, disappearing)
                    success, _ = self.download_file(
                        subtile_url,
                        output_path,
                        desc=f"     ├─ {zip_file}",
                        position=1,
                        leave=False,
                        colour='white'
                    )

                    if success:
                        # Extract directly to output directory (flat structure)
                        try:
                            with zipfile.ZipFile(output_path, "r") as zip_ref:
                                extracted = zip_ref.namelist()
                                zip_ref.extractall(self.output_dir)
                            output_path.unlink()  # Clean up zip
                            success_count += 1
                            downloaded_files.append((zip_file, extracted))
                        except Exception:
                            if output_path.exists():
                                output_path.unlink()
                            # Mark as unavailable (extraction failed)
                            failed_subtiles.append(subtile_name)
                    else:
                        # Download failed - likely ocean/water tile
                        failed_subtiles.append(subtile_name)

                # Also mark subtiles that weren't on FTP as unavailable (water/ocean tiles)
                if subtiles_not_on_ftp:
                    failed_subtiles.extend(subtiles_not_on_ftp)

                # Save newly discovered unavailable subtiles to cache
                if failed_subtiles:
                    unavailable_subtiles.update(failed_subtiles)
                    try:
                        with open(subtile_unavailable_file, 'w') as f:
                            f.write("# Subtiles that are permanently unavailable (ocean/water areas)\n")
                            f.write("# These 1x1 degree tiles don't exist on JAXA's FTP server\n\n")
                            for subtile in sorted(unavailable_subtiles):
                                f.write(f"{subtile}\n")
                    except Exception:
                        pass  # Silently handle cache save errors

                ftp.quit()

                if success_count > 0:
                    return (True, False, success_count)  # Downloaded files
                else:
                    return (False, False, 0)  # Failed to download

            except Exception as e:
                ftp.quit()
                raise e

        except Exception as e:
            # Check if it's a water tile error (FTP directory not found)
            error_msg = str(e)
            if "450" in error_msg or "550" in error_msg or "No such file" in error_msg:
                # Water/ocean tile - this is expected, suppress the error
                return (False, True, 0)  # Water tile
            else:
                # Actual error - suppress to keep progress bars clean
                return (False, False, 0)  # Error

    def _tile_has_any_subtiles(self, tile_name: str) -> bool:
        """
        Check if any subtiles exist for a 5x5 degree AW3D30 tile.

        A 5x5 degree tile (e.g., N050W135) contains 1x1 degree subtiles.
        After preparation, these are renamed to simple format (e.g., N051W131.tif).
        This checks both the original format (ALPSMLC30_*_DSM.tif) and the
        renamed format (N*.tif or S*.tif).

        Args:
            tile_name: 5x5 degree tile name (e.g., "N050W135")

        Returns:
            True if any subtiles exist for this 5x5 tile
        """
        import re

        # Extract base coordinates from tile name (e.g., "N050W135" -> lat=50, lon=-135)
        match = re.match(r'([NS])(\d{3})([EW])(\d{3})', tile_name)
        if not match:
            return False

        lat_dir, lat_str, lon_dir, lon_str = match.groups()
        base_lat = int(lat_str) * (1 if lat_dir == 'N' else -1)
        base_lon = int(lon_str) * (1 if lon_dir == 'E' else -1)

        # Check for any subtiles within the 5x5 degree tile
        # Look for both original format (ALPSMLC30_*_DSM.tif) and renamed format (N*.tif/S*.tif)
        for lat in range(base_lat, base_lat + 5):
            for lon in range(base_lon, base_lon + 5):
                # Generate subtile name
                subtile_lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):03d}"
                subtile_lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                subtile_name = f"{subtile_lat_str}{subtile_lon_str}"

                # Check for either format (with or without subdirectory)
                renamed_file = self.output_dir / f"{subtile_name}.tif"
                renamed_file_subdir = self.output_dir / subtile_name / f"{subtile_name}.tif"
                original_file = self.output_dir / f"ALPSMLC30_{subtile_name}_DSM.tif"
                original_file_subdir = self.output_dir / subtile_name / f"ALPSMLC30_{subtile_name}_DSM.tif"

                if (renamed_file.exists() or original_file.exists() or
                    renamed_file_subdir.exists() or original_file_subdir.exists()):
                    return True

        return False

    def _tile_has_all_subtiles(self, tile_name: str, bbox: BoundingBox = None, debug: bool = False) -> bool:
        """
        Check if ALL needed subtiles exist for a 5x5 degree AW3D30 tile.

        This is used to skip tiles where all subtiles are already downloaded.
        Unlike _tile_has_any_subtiles(), this checks that ALL subtiles (or all
        subtiles within the bbox if provided) exist.

        Args:
            tile_name: 5x5 degree tile name (e.g., "N050W135")
            bbox: Optional bounding box to filter which subtiles are needed
            debug: If True, print detailed file checking info

        Returns:
            True if all needed subtiles exist for this tile
        """
        import re

        match = re.match(r'([NS])(\d{3})([EW])(\d{3})', tile_name)
        if not match:
            return False

        lat_dir, lat_str, lon_dir, lon_str = match.groups()
        base_lat = int(lat_str) * (1 if lat_dir == 'N' else -1)
        base_lon = int(lon_str) * (1 if lon_dir == 'E' else -1)

        # Load subtile-level unavailable cache
        subtile_unavailable_file = self.output_dir / ".unavailable_subtiles"
        unavailable_subtiles = set()
        if subtile_unavailable_file.exists():
            try:
                with open(subtile_unavailable_file, 'r') as f:
                    unavailable_subtiles = set(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
            except Exception:
                pass

        needed_subtiles = []
        existing_subtiles = []
        missing_subtiles = []

        # Check all subtiles within the 5x5 degree tile
        for lat in range(base_lat, base_lat + 5):
            for lon in range(base_lon, base_lon + 5):
                subtile_lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):03d}"
                subtile_lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                subtile_name = f"{subtile_lat_str}{subtile_lon_str}"

                # If bbox provided, check if subtile intersects with bbox
                if bbox and not self._subtile_intersects_bbox(subtile_name, bbox):
                    continue

                needed_subtiles.append(subtile_name)

                # Check if this subtile exists (check both flat and subdirectory structure)
                renamed_file = self.output_dir / f"{subtile_name}.tif"
                renamed_file_subdir = self.output_dir / subtile_name / f"{subtile_name}.tif"
                original_file = self.output_dir / f"ALPSMLC30_{subtile_name}_DSM.tif"
                original_file_subdir = self.output_dir / subtile_name / f"ALPSMLC30_{subtile_name}_DSM.tif"

                # DETAILED DEBUG: Show exactly what paths we're checking for this subtile
                if debug:
                    print(f"      Checking subtile: {subtile_name}")
                    paths_to_check = [
                        (renamed_file, f"{subtile_name}.tif (flat)"),
                        (original_file, f"ALPSMLC30_{subtile_name}_DSM.tif (flat)"),
                        (renamed_file_subdir, f"{subtile_name}/{subtile_name}.tif (subdir)"),
                        (original_file_subdir, f"{subtile_name}/ALPSMLC30_{subtile_name}_DSM.tif (subdir)")
                    ]
                    for path_obj, path_desc in paths_to_check:
                        exists_status = "EXISTS ✓" if path_obj.exists() else "NOT FOUND"
                        print(f"        Path: {path_obj.name if path_obj.parent == self.output_dir else str(path_obj.relative_to(self.output_dir))} - {exists_status}")

                found_file = None
                if renamed_file.exists():
                    found_file = f"{subtile_name}.tif (flat)"
                elif original_file.exists():
                    found_file = f"ALPSMLC30_{subtile_name}_DSM.tif (flat)"
                elif renamed_file_subdir.exists():
                    found_file = f"{subtile_name}/{subtile_name}.tif (subdir)"
                elif original_file_subdir.exists():
                    found_file = f"{subtile_name}/ALPSMLC30_{subtile_name}_DSM.tif (subdir)"

                if found_file:
                    if debug:
                        print(f"        Result: FOUND - {found_file}")
                    existing_subtiles.append((subtile_name, found_file))
                elif subtile_name in unavailable_subtiles:
                    # Subtile is known to be unavailable (ocean/water) - treat as complete
                    if debug:
                        print(f"        Result: UNAVAILABLE (ocean tile, will never exist)")
                    existing_subtiles.append((subtile_name, "unavailable (ocean)"))
                else:
                    if debug:
                        print(f"        Result: MISSING")
                    missing_subtiles.append(subtile_name)

        if debug:
            print(f"    {tile_name}: {len(needed_subtiles)} subtiles needed")
            print(f"      Existing: {len(existing_subtiles)}")
            for subtile, path in existing_subtiles[:3]:  # Show first 3
                print(f"        ✓ {path}")
            if len(existing_subtiles) > 3:
                print(f"        ... and {len(existing_subtiles) - 3} more")
            print(f"      Missing: {len(missing_subtiles)}")
            for subtile in missing_subtiles[:3]:  # Show first 3
                print(f"        ✗ {subtile}")
            if len(missing_subtiles) > 3:
                print(f"        ... and {len(missing_subtiles) - 3} more")

        return len(missing_subtiles) == 0  # All needed subtiles exist

    def download_bbox(self, bbox: BoundingBox) -> tuple:
        """
        Download AW3D30 tiles for a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            Tuple of (success: bool, new_files_count: int)
        """
        tiles = self.get_tiles_for_bbox(bbox)

        # Check which tiles already exist or are known to be unavailable
        existing_tiles = []
        tiles_to_download = []
        unavailable_tiles = []

        for tile in tiles:
            # Check if tile is marked as completely unavailable (water/ocean)
            if self.is_tile_unavailable(tile):
                unavailable_tiles.append(tile)
            # Check if all needed subtiles already exist (skip if complete)
            elif self._tile_has_all_subtiles(tile, bbox, debug=False):
                existing_tiles.append(tile)
            else:
                # Some or all subtiles missing - download_tile() will check which ones
                tiles_to_download.append(tile)

        # Print summary about skipped tiles
        if existing_tiles:
            print(f"  {len(existing_tiles)} tiles already complete")
        if unavailable_tiles:
            print(f"  {len(unavailable_tiles)} tiles unavailable (ocean/uncovered)")

        if not tiles_to_download:
            if existing_tiles:
                print(f"\n✓ All {len(existing_tiles)} AW3D30 tiles already downloaded")
            else:
                print(f"\n✓ All AW3D30 tiles skipped (marked as unavailable)")
            return (True, 0)  # Success, but no new files

        # Print header
        _print_download_header(
            "AW3D30 Download",
            subtitle="ALOS World 3D 30m\nDownloading from JAXA public FTP server (no authentication required)",
            bbox_str=str(bbox),
            output_dir=str(self.output_dir.absolute()),
            source=self.BASE_URL
        )

        print(f"\nDownloading {len(tiles_to_download)} AW3D30 5° tiles with missing subtiles")
        print(f"  (each tile contains up to 25 1° subtiles - only missing ones will be downloaded)\n")

        success_count = len(existing_tiles)  # Count existing as successful
        new_files_count = 0  # Track newly downloaded files
        water_tiles = []  # Track tiles that don't exist (water/ocean areas)

        # Main progress bar for overall tile download
        with tqdm(
            total=len(tiles_to_download),
            desc="   AW3D30",
            unit=" tiles",
            position=0,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100,
            ascii=" █"
        ) as main_pbar:

            for tile in tiles_to_download:
                success, is_water, files_downloaded = self.download_tile(tile, bbox, main_pbar)
                if success:
                    success_count += 1
                    new_files_count += files_downloaded  # Only count actually downloaded files
                elif is_water:
                    water_tiles.append(tile)
                    # Mark as unavailable so we don't retry on future runs
                    self.mark_tile_unavailable(tile)
                main_pbar.update(1)
                time.sleep(1)  # Rate limiting

        # Adjust the tile count to exclude water tiles from the total
        actual_tile_count = len(tiles) - len(water_tiles)

        if water_tiles:
            print(f"\n     {len(water_tiles)} tiles skipped (water/ocean areas with no elevation data)")

        print(f"\nAW3D30 download complete: {success_count}/{actual_tile_count} tiles successful")

        return (success_count > 0, new_files_count)

    def export_tile_urls(self, bbox: BoundingBox, output_file: Path) -> bool:
        """
        Export AW3D30 tile URLs for manual download.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
            output_file: Path to output file

        Returns:
            True if successful
        """
        tiles = self.get_tiles_for_bbox(bbox)

        try:
            with open(output_file, 'w') as f:
                f.write("# AW3D30 (ALOS World 3D 30m) Tile URLs\n")
                f.write(f"# Bounding box: {bbox}\n")
                f.write("# Dataset: JAXA AW3D30\n")
                f.write("# Resolution: 30m\n")
                f.write(f"# Base URL: {self.BASE_URL}\n")
                f.write("# Note: Each 5x5 degree tile contains multiple 1x1 degree subtiles as zip files\n")
                f.write("# You'll need to list and download all .zip files from each directory\n\n")

                for tile in tiles:
                    tile_url = f"{self.BASE_URL}{tile}/"
                    f.write(f"# Tile: {tile}\n")
                    f.write(f"{tile_url}\n\n")

            print(f"✓ Exported {len(tiles)} AW3D30 tile directory URLs to {output_file}")
            print("  Note: Each URL is a directory. Use FTP client to list and download all .zip files.")
            return True

        except Exception as e:
            print(f"✗ Error exporting URLs: {e}")
            return False


class ASTERDownloader(BaseDEMDownloader):
    """
    DEPRECATED: ASTER GDEM v3 dataset downloader.

    WARNING: As of December 2025, NASA LP DAAC Data Pool was retired.
    This downloader no longer works. Use SRTM (OpenTopography) or AW3D30 instead.

    - SRTM: 30m resolution, 60°N to 56°S coverage (now via OpenTopography S3)
    - AW3D30: 30m resolution, 84°N to 84°S coverage (still works via JAXA FTP)

    Both SRTM and AW3D30 have better accuracy than ASTER according to research.
    See: https://www.mdpi.com/2072-4292/12/21/3482
    """

    # BROKEN: NASA LP DAAC Data Pool was retired December 2025
    # This URL now returns 404
    BASE_URL = "https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/"

    def __init__(self, output_dir: Path, credentials: tuple[str, str] | None = None):
        """Initialize ASTER GDEM downloader.

        NOTE: NASA LP DAAC Data Pool was retired December 2025.
        ASTER downloads no longer work. Use SRTM or AW3D30 instead.

        Args:
            output_dir: Directory to save downloads
            credentials: DEPRECATED - no longer used
        """
        super().__init__(output_dir, "ASTER GDEM")
        # DEPRECATED: Credentials no longer used - data source retired
        self.credentials = None
        self.session = None
        self._credentials_validated = False

    def _validate_credentials(self) -> bool:
        """DEPRECATED: Returns False. NASA LP DAAC Data Pool retired December 2025."""
        return False

    def get_tile_name(self, lat: float, lon: float) -> str:
        """
        Get tile name for a given coordinate.

        ASTER tiles are 1° x 1° and named based on the southwest corner.
        Format: ASTGTMV003_N00E000

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Tile name (e.g., "ASTGTMV003_N35E139")
        """
        tile_lat = math.floor(lat)
        tile_lon = math.floor(lon)

        lat_str = f"{'N' if tile_lat >= 0 else 'S'}{abs(tile_lat):02d}"
        lon_str = f"{'E' if tile_lon >= 0 else 'W'}{abs(tile_lon):03d}"

        return f"ASTGTMV003_{lat_str}{lon_str}"

    def get_tiles_for_bbox(self, bbox: BoundingBox) -> List[str]:
        """
        Get list of tile names covering a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            List of tile names
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        tiles = set()

        # Iterate over 1° grid
        lat = math.floor(min_lat)
        while lat <= max_lat:
            lon = math.floor(min_lon)
            while lon <= max_lon:
                tiles.add(self.get_tile_name(lat, lon))
                lon += 1
            lat += 1

        return sorted(tiles)

    def download_tile(self, tile_name: str, main_pbar=None) -> tuple[bool, bool]:
        """
        Download a single ASTER GDEM tile.

        Args:
            tile_name: Tile name (e.g., "ASTGTMV003_N35E139")
            main_pbar: Optional main progress bar to update

        Returns:
            Tuple of (success: bool, was_404: bool)
        """
        # ASTER tiles are distributed as ZIP files (no _dem suffix in filename)
        filename = f"{tile_name}.zip"
        url = urljoin(self.BASE_URL, filename)

        output_path = self.output_dir / filename

        # Check if the main DEM file already exists (ASTER extracts files directly, not into a subdirectory)
        dem_file = self.output_dir / f"{tile_name}_dem.tif"
        if dem_file.exists():
            return (True, False)

        # Download the tile with sub-progress bar (grey, disappearing)
        success, status_code = self.download_file(
            url,
            output_path,
            desc=f"     ├─ {filename}",
            session=self.session,
            silent_404=True,
            position=1,
            leave=False,
            colour='white'
        )

        if not success:
            # Distinguish auth errors from true 404s
            if status_code in (401, 403):
                # Authentication error - don't cache as unavailable
                return (False, False)
            was_404 = status_code == 404
            return (False, was_404)

        # Extract the archive
        try:
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)

            # Clean up archive
            output_path.unlink()

            # Remove _num.tif file (we only need _dem.tif for elevation data)
            # OpenTopoData treats both files as duplicates which causes config errors
            num_file = self.output_dir / f"{tile_name}_num.tif"
            if num_file.exists():
                num_file.unlink()

            return (True, False)

        except Exception:
            # Suppress error to keep progress bars clean
            if output_path.exists():
                output_path.unlink()
            return (False, False)

    def download_bbox(self, bbox: BoundingBox) -> tuple:
        """
        Download ASTER GDEM tiles for a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            Tuple of (success: bool, new_files_count: int)
        """
        tiles = self.get_tiles_for_bbox(bbox)

        # Check which tiles already exist or are known to be unavailable
        existing_tiles = []
        tiles_to_download = []
        unavailable_tiles = []

        for tile in tiles:
            dem_file = self.output_dir / f"{tile}_dem.tif"
            if dem_file.exists():
                existing_tiles.append(tile)
            elif self.is_tile_unavailable(tile):
                unavailable_tiles.append(tile)
            else:
                # Download all missing tiles (no overlap check - preserving all datasets for fallback)
                tiles_to_download.append(tile)

        # Print summary about unavailable tiles
        if unavailable_tiles:
            from climb_analyzer.utils.formatting import print_dim

            print_dim(
                f"\n  Skipping {len(unavailable_tiles)} tiles known to be unavailable (ocean/uncovered)"
            )

        # If all exist, print summary and return
        if existing_tiles and not tiles_to_download:
            total_available = len(existing_tiles)
            print(f"\n✓ All {total_available} available ASTER GDEM tiles already exist")
            if unavailable_tiles:
                print(f"  ({len(unavailable_tiles)} tiles are unavailable for this region)")
            return (True, 0)  # Success, but no new files

        # Print header only if we need to download
        _print_download_header("ASTER GDEM Download", subtitle="Tertiary Dataset - 30m Resolution")

        if existing_tiles and tiles_to_download:
            print(
                f"\nFound {len(tiles)} ASTER GDEM tiles: {len(existing_tiles)} already exist, {len(tiles_to_download)} to download"
            )
        else:
            print(f"\nFound {len(tiles)} ASTER GDEM tiles: {', '.join(tiles)}")

        print("\n⚠️  ASTER GDEM is a tertiary (backup) dataset.")
        print("   If you have SRTM and AW3D30, ASTER is optional.")
        print("\nManual download:")
        print("  1. Visit: https://search.earthdata.nasa.gov/")
        print("  2. Search for: ASTGTM (ASTER GDEM)")
        print(f"  3. Download tiles: {', '.join(tiles_to_download)}")
        print("  4. Extract to: elevation_data/aster/")
        print("=" * 70)

        # DEPRECATED: NASA LP DAAC Data Pool retired December 2025
        print("\n⚠️  ASTER automated download unavailable - data source retired December 2025")
        print("   Use SRTM (OpenTopography S3) or AW3D30 (JAXA FTP) instead")
        return (False, 0)


class REMADownloader(BaseDEMDownloader):
    """
    Downloader for REMA (Reference Elevation Model of Antarctica) 32m mosaic.

    REMA provides high-resolution elevation data for Antarctica.
    Data is distributed as regional mosaics via AWS S3.

    REMA uses a tile grid system (100km x 100km tiles) in Antarctic Polar Stereographic
    projection (EPSG:3031). Tiles are identified by grid coordinates (e.g., "18_33").

    Source: https://www.pgc.umn.edu/data/rema/
    """

    # REMA data available via AWS S3 public bucket
    # Using v2.0 32m mosaic tiles
    S3_BASE_URL = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/rema/mosaics/v2.0/32m/"

    # Tile index GeoPackage URL
    TILE_INDEX_URL = "https://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/REMA_Mosaic_Index_v2_gpkg.zip"

    def __init__(self, output_dir: Path):
        """Initialize REMA downloader."""
        super().__init__(output_dir, "rema32m")  # Must match DATASET_PRIORITY key
        self.index_dir = self.output_dir / ".index"
        self.index_file = self.index_dir / "REMA_Mosaic_Index_v2_gpkg.gpkg"
        self._tile_index = None

    def _ensure_tile_index(self) -> bool:
        """
        Download and cache REMA tile index if not already present.

        Returns:
            True if index is available, False otherwise
        """
        if self._tile_index is not None:
            return True

        # Check if index file exists
        if self.index_file.exists():
            try:
                import geopandas as gpd
                import warnings
                # Suppress pyogrio date format warnings (non-critical)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyogrio')
                    self._tile_index = gpd.read_file(
                        self.index_file,
                        layer='REMA_Mosaic_Index_v2_32m'
                    )
                return True
            except Exception as e:
                print(f"⚠️  Error loading tile index: {e}")
                # Try to download fresh copy
                self.index_file.unlink()

        # Download tile index
        print("Downloading REMA tile index (one-time setup)...")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        try:
            import zipfile

            # Download to temporary file
            zip_path = self.index_dir / "tile_index.zip"
            response = requests.get(self.TILE_INDEX_URL, stream=True, timeout=60)
            response.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract GeoPackage
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.index_dir)

            # Clean up zip file
            zip_path.unlink()

            # Load index
            import geopandas as gpd
            import warnings
            # Suppress pyogrio date format warnings (non-critical)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyogrio')
                self._tile_index = gpd.read_file(
                    self.index_file,
                    layer='REMA_Mosaic_Index_v2_32m'
                )

            print(f"✓ Loaded REMA tile index ({len(self._tile_index)} tiles)")
            return True

        except Exception as e:
            print(f"❌ Failed to download tile index: {e}")
            print("   REMA downloads require tile index for spatial queries.")
            return False

    def _get_tiles_for_bbox(self, bbox: BoundingBox) -> List[dict]:
        """
        Get list of REMA tiles intersecting with a bounding box.

        Uses spatial index to find tiles that intersect with the bbox.
        Transforms bbox from WGS84 (EPSG:4326) to Antarctic Polar Stereographic (EPSG:3031).

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon) in WGS84

        Returns:
            List of dicts with tile info: {'tile_id': str, 'coverage': float, 'geometry': ...}
        """
        if not self._ensure_tile_index():
            return []

        min_lat, min_lon, max_lat, max_lon = bbox

        try:
            from pyproj import Transformer
            from shapely.geometry import box

            # Transform bbox from WGS84 to EPSG:3031 (Antarctic Polar Stereographic)
            transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3031', always_xy=True)

            # Transform corners
            min_x, min_y = transformer.transform(min_lon, min_lat)
            max_x, max_y = transformer.transform(max_lon, max_lat)

            # Create bounding box in EPSG:3031
            bbox_3031 = box(min_x, min_y, max_x, max_y)

            # Find intersecting tiles
            intersecting = self._tile_index[self._tile_index.intersects(bbox_3031)]

            # Filter to tiles with data (data_percent > 0)
            tiles_with_data = intersecting[intersecting['data_percent'] > 0]

            # Convert to list of dicts
            tiles = []
            for _, row in tiles_with_data.iterrows():
                tiles.append({
                    'tile_id': row['tile'],
                    'coverage': row['data_percent'],
                    'geometry': row.geometry
                })

            return tiles

        except Exception as e:
            print(f"❌ Error querying tile index: {e}")
            return []

    def download_bbox(self, bbox: BoundingBox) -> tuple:
        """
        Download REMA tiles for a bounding box (Antarctica only).

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            Tuple of (success: bool, new_files_count: int)
        """
        min_lat, min_lon, max_lat, max_lon = bbox

        # Validate that bbox is in Antarctica (< -60° latitude)
        if max_lat > -60:
            print("   REMA is only available for Antarctica (latitude < -60°)")
            print(f"   Your bounding box extends to {max_lat:.1f}° latitude.")
            print("   Using fallback datasets (AW3D30, ASTER) for this region.")
            return (True, 0)

        _print_download_header(
            "REMA Download",
            subtitle="Reference Elevation Model of Antarctica 32m",
            bbox_str=f"({min_lat:.4f}, {min_lon:.4f}, {max_lat:.4f}, {max_lon:.4f})",
            output_dir=str(self.output_dir.absolute()),
            source=self.S3_BASE_URL
        )

        # Get required tiles using spatial index
        tiles = self._get_tiles_for_bbox(bbox)

        if not tiles:
            print("⚠️  No REMA tiles found for this area")
            return (True, 0)

        print(f"   Found {len(tiles)} REMA tiles with data coverage")

        # Check which tiles already exist
        existing_tiles = []
        missing_tiles = []

        for tile_info in tiles:
            tile_id = tile_info['tile_id']
            tile_filename = f"{tile_id}_32m_v2.0_dem.tif"
            tile_path = self.output_dir / tile_filename

            if tile_path.exists():
                existing_tiles.append(tile_filename)
            else:
                missing_tiles.append(tile_info)

        if existing_tiles:
            print(f"✓ {len(existing_tiles)} tiles already exist")

        if not missing_tiles:
            print("✓ All required REMA tiles are already downloaded")
            return (True, 0)

        print(f"Downloading {len(missing_tiles)} missing tiles...\n")

        # Download missing tiles with nested progress bars
        new_files = 0
        failed_tiles = []

        # Main progress bar for overall tile download
        with tqdm(
            total=len(missing_tiles),
            desc="   REMA",
            unit=" tiles",
            position=0,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100,
            ascii=" █"
        ) as main_pbar:

            for tile_info in missing_tiles:
                tile_id = tile_info['tile_id']
                tile_filename = f"{tile_id}_32m_v2.0_dem.tif"

                # Construct URL: {base_url}{tile_id}/{tile_id}_32m_v2.0_dem.tif
                tile_url = f"{self.S3_BASE_URL}{tile_id}/{tile_filename}"
                tile_path = self.output_dir / tile_filename

                try:
                    response = requests.get(tile_url, stream=True, timeout=60)

                    if response.status_code == 200:
                        # Download successful - show per-file progress
                        tile_path.parent.mkdir(parents=True, exist_ok=True)
                        total_size = int(response.headers.get('content-length', 0))

                        # Sub progress bar for individual file (grey, disappears when done)
                        with open(tile_path, 'wb') as f:
                            with tqdm(
                                total=total_size,
                                desc=f"     ├─ {tile_filename}",
                                unit='B',
                                unit_scale=True,
                                unit_divisor=1024,
                                position=1,
                                leave=False,  # Disappears when done
                                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]',
                                ncols=100,
                                ascii=" █",
                                colour='white'  # Grey-ish color for sub-bar
                            ) as file_pbar:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    file_pbar.update(len(chunk))

                        new_files += 1
                        main_pbar.update(1)

                    elif response.status_code == 404:
                        # Tile doesn't exist (shouldn't happen with index, but handle gracefully)
                        failed_tiles.append(tile_id)
                        main_pbar.update(1)
                    else:
                        failed_tiles.append(tile_id)
                        main_pbar.update(1)

                except requests.exceptions.Timeout:
                    failed_tiles.append(tile_id)
                    main_pbar.update(1)
                except Exception:
                    failed_tiles.append(tile_id)
                    main_pbar.update(1)

        # Print summary
        print(f"\n✓ Download complete: {new_files} new file(s)")
        if failed_tiles:
            print(f"⚠️  Failed to download {len(failed_tiles)} tile(s)")

        # Auto-rebuild VRT if new tiles were downloaded
        if new_files > 0:
            self._rebuild_vrt()

        return (True, new_files)

    def _rebuild_vrt(self):
        """Rebuild VRT file for REMA32m after downloading tiles."""
        try:
            print("\n  Building VRT file for OpenTopoData...")
            from scripts.manage_arctic_vrt import ArcticVRTManager

            vrt_manager = ArcticVRTManager('rema32m', base_dir=self.output_dir.parent.parent)
            success = vrt_manager.rebuild_vrt(verbose=False)

            if success:
                print(f"  ✓ VRT file created: {vrt_manager.vrt_path.relative_to(vrt_manager.base_dir)}")
            else:
                print("  ⚠️  VRT creation failed - OpenTopoData may not be able to use these tiles")
        except Exception as e:
            print(f"  ⚠️  VRT rebuild failed: {e}")
            print("     You can manually rebuild with: python scripts/manage_arctic_vrt.py --dataset rema32m --rebuild")


class ArcticDEMDownloader(BaseDEMDownloader):
    """
    Downloader for ArcticDEM 32m mosaic.

    ArcticDEM provides high-resolution elevation data for the Arctic (>60°N).
    Coverage: Alaska, Northern Canada, Greenland, Iceland, Scandinavia, Northern Russia
    Data is distributed as regional mosaic tiles via AWS S3.

    ArcticDEM uses a tile grid system (100km x 100km tiles) in NSIDC Sea Ice Polar
    Stereographic North projection (EPSG:3413). Tiles are identified by grid coordinates.

    Source: https://www.pgc.umn.edu/data/arcticdem/
    """

    # ArcticDEM tiles available via AWS S3 public bucket
    # Using v4.1 32m mosaic tiles
    S3_BASE_URL = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/32m/"

    # Tile index GeoPackage URL
    TILE_INDEX_URL = "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ArcticDEM_Mosaic_Index_v4_1_gpkg.zip"

    def __init__(self, output_dir: Path):
        """Initialize ArcticDEM downloader."""
        super().__init__(output_dir, "arcticdem32m")  # Must match DATASET_PRIORITY key
        self.index_dir = self.output_dir / ".index"
        self.index_file = self.index_dir / "ArcticDEM_Mosaic_Index_v4_1_gpkg.gpkg"
        self._tile_index = None

    def _ensure_tile_index(self) -> bool:
        """
        Download and cache ArcticDEM tile index if not already present.

        Returns:
            True if index is available, False otherwise
        """
        if self._tile_index is not None:
            return True

        # Check if index file exists
        if self.index_file.exists():
            try:
                import geopandas as gpd
                import warnings
                # Suppress pyogrio date format warnings (non-critical)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyogrio')
                    self._tile_index = gpd.read_file(
                        self.index_file,
                        layer='ArcticDEM_Mosaic_Index_v4_1_32m'
                    )
                return True
            except Exception as e:
                print(f"⚠️  Error loading tile index: {e}")
                # Try to download fresh copy
                self.index_file.unlink()

        # Download tile index
        print("Downloading ArcticDEM tile index (one-time setup)...")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        try:
            import zipfile

            # Download to temporary file
            zip_path = self.index_dir / "tile_index.zip"
            response = requests.get(self.TILE_INDEX_URL, stream=True, timeout=60)
            response.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract GeoPackage
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.index_dir)

            # Clean up zip file
            zip_path.unlink()

            # Load index
            import geopandas as gpd
            import warnings
            # Suppress pyogrio date format warnings (non-critical)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyogrio')
                self._tile_index = gpd.read_file(
                    self.index_file,
                    layer='ArcticDEM_Mosaic_Index_v4_1_32m'
                )

            print(f"✓ Loaded ArcticDEM tile index ({len(self._tile_index)} tiles)")
            return True

        except Exception as e:
            print(f"❌ Failed to download tile index: {e}")
            print("   ArcticDEM downloads require tile index for spatial queries.")
            return False

    def _get_tiles_for_bbox(self, bbox: BoundingBox) -> List[dict]:
        """
        Get list of ArcticDEM tiles intersecting with a bounding box.

        Uses spatial index to find tiles that intersect with the bbox.
        Transforms bbox from WGS84 (EPSG:4326) to NSIDC Polar Stereographic North (EPSG:3413).

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon) in WGS84

        Returns:
            List of dicts with tile info: {'tile_id': str, 'coverage': float, 'geometry': ...}
        """
        if not self._ensure_tile_index():
            return []

        min_lat, min_lon, max_lat, max_lon = bbox

        try:
            from pyproj import Transformer
            from shapely.geometry import box

            # Transform bbox from WGS84 to EPSG:3413 (NSIDC Sea Ice Polar Stereographic North)
            transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3413', always_xy=True)

            # Transform corners
            min_x, min_y = transformer.transform(min_lon, min_lat)
            max_x, max_y = transformer.transform(max_lon, max_lat)

            # Create bounding box in EPSG:3413
            bbox_3413 = box(min_x, min_y, max_x, max_y)

            # Find intersecting tiles
            intersecting = self._tile_index[self._tile_index.intersects(bbox_3413)]

            # Filter to tiles with data (data_percent > 0)
            tiles_with_data = intersecting[intersecting['data_percent'] > 0]

            # Convert to list of dicts
            tiles = []
            for _, row in tiles_with_data.iterrows():
                tiles.append({
                    'tile_id': row['tile'],
                    'coverage': row['data_percent'],
                    'geometry': row.geometry
                })

            return tiles

        except Exception as e:
            print(f"❌ Error querying tile index: {e}")
            return []

    def download_bbox(self, bbox: BoundingBox) -> tuple:
        """
        Download ArcticDEM tiles for a bounding box (Arctic only, >60°N).

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            Tuple of (success: bool, new_files_count: int)
        """
        min_lat, min_lon, max_lat, max_lon = bbox

        # Validate that bbox is in Arctic region (> 60°N latitude)
        if max_lat < 60:
            print("   ArcticDEM is only available for Arctic regions (latitude > 60°N)")
            print(f"   Your bounding box is entirely below 60°N (max: {max_lat:.1f}°).")
            print("   Using fallback datasets (SRTM, AW3D30, ASTER) for this region.")
            return (True, 0)

        # Clip bbox to Arctic region
        if min_lat < 60:
            print(f"   Arctic region detected (coverage extends to {max_lat:.1f}°N)")
            min_lat = 60  # Clip to Arctic boundary

        _print_download_header(
            "ArcticDEM Download",
            subtitle="Arctic Digital Elevation Model 32m",
            bbox_str=f"({min_lat:.4f}, {min_lon:.4f}, {max_lat:.4f}, {max_lon:.4f})",
            output_dir=str(self.output_dir.absolute()),
            source=self.S3_BASE_URL
        )

        # Get required tiles using spatial index
        tiles = self._get_tiles_for_bbox(bbox)

        if not tiles:
            print("⚠️  No ArcticDEM tiles found for this area")
            return (True, 0)

        print(f"   Found {len(tiles)} ArcticDEM tiles with data coverage")

        # Check which tiles already exist
        existing_tiles = []
        missing_tiles = []

        for tile_info in tiles:
            tile_id = tile_info['tile_id']
            tile_filename = f"{tile_id}_32m_v4.1_dem.tif"
            tile_path = self.output_dir / tile_filename

            if tile_path.exists():
                existing_tiles.append(tile_filename)
            else:
                missing_tiles.append(tile_info)

        if existing_tiles:
            print(f"✓ {len(existing_tiles)} tiles already exist")

        if not missing_tiles:
            print("✓ All required ArcticDEM tiles are already downloaded")
            return (True, 0)

        print(f"Downloading {len(missing_tiles)} missing tiles...\n")

        # Download missing tiles with nested progress bars
        new_files = 0
        failed_tiles = []

        # Main progress bar for overall tile download
        with tqdm(
            total=len(missing_tiles),
            desc="   ArcticDEM",
            unit=" tiles",
            position=0,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100,
            ascii=" █"
        ) as main_pbar:

            for tile_info in missing_tiles:
                tile_id = tile_info['tile_id']
                tile_filename = f"{tile_id}_32m_v4.1_dem.tif"

                # Construct URL: {base_url}{tile_id}/{tile_id}_32m_v4.1_dem.tif
                tile_url = f"{self.S3_BASE_URL}{tile_id}/{tile_filename}"
                tile_path = self.output_dir / tile_filename

                try:
                    response = requests.get(tile_url, stream=True, timeout=60)

                    if response.status_code == 200:
                        # Download successful - show per-file progress
                        tile_path.parent.mkdir(parents=True, exist_ok=True)
                        total_size = int(response.headers.get('content-length', 0))

                        # Sub progress bar for individual file (grey, disappears when done)
                        with open(tile_path, 'wb') as f:
                            with tqdm(
                                total=total_size,
                                desc=f"     ├─ {tile_filename}",
                                unit='B',
                                unit_scale=True,
                                unit_divisor=1024,
                                position=1,
                                leave=False,  # Disappears when done
                                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]',
                                ncols=100,
                                ascii=" █",
                                colour='white'  # Grey-ish color for sub-bar
                            ) as file_pbar:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    file_pbar.update(len(chunk))

                        new_files += 1
                        main_pbar.update(1)

                    elif response.status_code == 404:
                        # Tile doesn't exist (shouldn't happen with index, but handle gracefully)
                        failed_tiles.append(tile_id)
                        main_pbar.update(1)
                    else:
                        failed_tiles.append(tile_id)
                        main_pbar.update(1)

                except requests.exceptions.Timeout:
                    failed_tiles.append(tile_id)
                    main_pbar.update(1)
                except Exception:
                    failed_tiles.append(tile_id)
                    main_pbar.update(1)

        # Print summary
        print(f"\n✓ Download complete: {new_files} new file(s)")
        if failed_tiles:
            print(f"⚠️  Failed to download {len(failed_tiles)} tile(s)")

        # Auto-rebuild VRT if new tiles were downloaded
        if new_files > 0:
            self._rebuild_vrt()

        return (True, new_files)

    def _rebuild_vrt(self):
        """Rebuild VRT file for Arctic32m after downloading tiles."""
        try:
            print("\n  Building VRT file for OpenTopoData...")
            from scripts.manage_arctic_vrt import ArcticVRTManager

            vrt_manager = ArcticVRTManager('arctic32m', base_dir=self.output_dir.parent.parent)
            success = vrt_manager.rebuild_vrt(verbose=False)

            if success:
                print(f"  ✓ VRT file created: {vrt_manager.vrt_path.relative_to(vrt_manager.base_dir)}")
            else:
                print("  ⚠️  VRT creation failed - OpenTopoData may not be able to use these tiles")
        except Exception as e:
            print(f"  ⚠️  VRT rebuild failed: {e}")
            print("     You can manually rebuild with: python scripts/manage_arctic_vrt.py --dataset arctic32m --rebuild")


class NEDDownloader(BaseDEMDownloader):
    """
    Downloader for NED (National Elevation Dataset) 10m.

    NED provides high-resolution elevation data for the United States.
    10m (1/3 arc-second) resolution available for most of the US.

    Source: https://www.usgs.gov/the-national-map-data-delivery
    """

    # NED data is available through The National Map
    BASE_URL = "https://prd-tnm.s3.amazonaws.com/index.html"

    def __init__(self, output_dir: Path):
        """Initialize NED downloader."""
        super().__init__(output_dir, "NED 10m")

    def get_tile_name(self, lat: float, lon: float) -> str:
        """
        Get NED tile name for a given coordinate.

        NED tiles are 1° x 1° and named based on the UPPER-LEFT (northwest) corner.
        Format: nXXwYYY or nXXeYYY or sXXwYYY or sXXeYYY

        Example: Coordinate (34.5, -84.5) is in tile n35w085
        (covers 34-35°N, 84-85°W, named by upper-left corner at 35°N, 85°W)

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Tile name using upper-left corner (e.g., "n35w085")
        """
        # NED uses upper-left corner naming, so we need ceiling for lat
        tile_lat = math.ceil(lat)  # Upper edge
        tile_lon = math.floor(lon)  # Left edge

        lat_str = f"{'n' if tile_lat >= 0 else 's'}{abs(tile_lat):02d}"
        lon_str = f"{'e' if tile_lon >= 0 else 'w'}{abs(tile_lon):03d}"

        return f"{lat_str}{lon_str}"

    def get_srtm_format_tile_name(self, ned_tile_name: str) -> str:
        """
        Convert NED tile name (upper-left) to SRTM format (lower-left).

        OpenTopoData expects tile filenames to match the lower-left corner (SRTM format).
        NED tiles use upper-left corner naming, so we subtract 1 from latitude.

        Args:
            ned_tile_name: NED tile name (e.g., "n35w085")

        Returns:
            SRTM-format tile name (e.g., "n34w085")
        """
        import re
        match = re.search(r'([ns]\d\d)', ned_tile_name)
        if not match:
            return ned_tile_name

        old_northing = match.group(1)
        n_or_s = old_northing[0]
        ns_value = int(old_northing[1:3])

        # Calculate new northing (subtract 1 for north, add 1 for south)
        if old_northing == 'n00':
            new_northing = 's01'
        elif n_or_s == 'n':
            new_northing = 'n' + str(ns_value - 1).zfill(2)
        elif n_or_s == 's':
            new_northing = 's' + str(ns_value + 1).zfill(2)
        else:
            return ned_tile_name

        return ned_tile_name.replace(old_northing, new_northing)

    def get_tiles_for_bbox(self, bbox: BoundingBox) -> List[str]:
        """
        Get list of tile names covering a bounding box.

        NED tiles use upper-left corner naming, so we need to expand the bbox
        upward to ensure we get tiles that cover the northern edge.

        For example:
        - Coordinate (34.5, -84.5) needs tile n35w085 (covers 34-35°N)
        - Coordinate (35.1, -84.5) needs tile n36w085 (covers 35-36°N)

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            List of tile names using NED upper-left corner naming
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        tiles = set()

        # For NED upper-left corner naming with ceil(), we need to ensure
        # we cover all coordinates in the bbox. Since get_tile_name uses
        # ceil(lat), we need to iterate through all integer latitudes that
        # could produce tiles covering our bbox.
        #
        # Example: if max_lat = 35.1, we need tile n36w084 (covers 35-36°N)
        # So we iterate up to and including floor(max_lat) + 1
        lat = math.floor(min_lat)
        max_lat_iter = math.floor(max_lat) + 1

        while lat <= max_lat_iter:
            lon = math.floor(min_lon)
            while lon <= max_lon:
                tiles.add(self.get_tile_name(lat, lon))
                lon += 1
            lat += 1

        return sorted(tiles)

    def download_tile(self, tile_name: str, main_pbar=None) -> tuple[bool, bool]:
        """
        Download a single NED tile.

        Args:
            tile_name: Tile name in NED format/upper-left corner (e.g., "n35w084")
            main_pbar: Optional main progress bar to update

        Returns:
            Tuple of (success: bool, was_404: bool)
            was_404 indicates tile doesn't exist (water/uncovered area)
        """
        # NED tiles are distributed as TIFF files from AWS
        # URL format: https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/{tile_name}/USGS_13_{tile_name}.tif
        base_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
        filename = f"USGS_13_{tile_name}.tif"
        url = f"{base_url}/{tile_name}/{filename}"

        # We'll download to a temp path with USGS_13_ prefix, then rename to final format
        temp_output_path = self.output_dir / filename

        # Calculate SRTM-format name (lower-left corner) without USGS_13_ prefix
        # This is the final format OpenTopoData expects
        srtm_tile_name = self.get_srtm_format_tile_name(tile_name)
        final_output_path = self.output_dir / f"{srtm_tile_name}.tif"

        # Skip if already downloaded in final format
        if final_output_path.exists() and final_output_path.stat().st_size > 1_000_000:
            return (True, False)

        # Also check temp path in case download was interrupted
        if temp_output_path.exists() and temp_output_path.stat().st_size > 1_000_000:
            # Rename to final format
            temp_output_path.rename(final_output_path)
            return (True, False)

        # Download the tile with sub-progress bar (grey, disappearing)
        success, status_code = self.download_file(
            url,
            temp_output_path,
            desc=f"     ├─ {filename}",
            silent_404=True,
            position=1,
            leave=False,
            colour='white'
        )

        if not success:
            # Check if it was a 404 (tile doesn't exist - water/uncovered)
            was_404 = status_code == 404
            return (False, was_404)

        # Rename from temp name (USGS_13_n35w084.tif) to final format (n34w084.tif)
        if temp_output_path.exists():
            temp_output_path.rename(final_output_path)

        return (True, False)

    def _rename_ned_tiles_to_srtm_format(self) -> None:
        """
        Rename NED tiles to SRTM-compatible format for OpenTopoData.

        OpenTopoData expects tile filenames to match the lower-left corner,
        but NED files use upper-left corner naming. This subtracts 1 from
        the latitude to convert NED naming to SRTM naming.

        For example:
            USGS_13_n35w085.tif (NED, upper-left) -> USGS_13_n34w085.tif (SRTM, lower-left)
            Tile covers 34-35°N, 84-85°W
            NED names it by upper-left corner (35N, 85W)
            SRTM format uses lower-left corner (34N, 85W)
        """
        import re

        print("\n📝 Renaming NED tiles to SRTM-compatible format...")

        # Find all NED tiles (only original USGS format, not already renamed)
        ned_tiles = list(self.output_dir.glob("USGS_13_[ns][0-9][0-9][ew][0-9][0-9][0-9].tif"))

        if not ned_tiles:
            print("     No tiles to rename (may already be in SRTM format)")
            return

        renamed_count = 0
        skipped_count = 0

        for tile_path in ned_tiles:
            filename = tile_path.name

            # Extract tile name (e.g., 'n35w085' from 'USGS_13_n35w085.tif')
            match = re.search(r'USGS_13_([ns]\d\d[ew]\d\d\d)\.tif', filename)
            if not match:
                print(f"  ⚠️  Could not parse tile name from: {filename}")
                skipped_count += 1
                continue

            ned_tile_name = match.group(1)

            # Convert to SRTM format (lower-left corner)
            srtm_tile_name = self.get_srtm_format_tile_name(ned_tile_name)

            # Create new filename
            new_filename = f"USGS_13_{srtm_tile_name}.tif"
            new_path = tile_path.parent / new_filename

            # Skip if source and destination are the same (shouldn't happen)
            if ned_tile_name == srtm_tile_name:
                skipped_count += 1
                continue

            # Skip if target already exists
            if new_path.exists():
                # Remove the old NED-format file since we have the SRTM-format version
                print(f"     {srtm_tile_name} already exists, removing duplicate {ned_tile_name}")
                tile_path.unlink()
                skipped_count += 1
                continue

            # Rename the file (this moves it, doesn't copy)
            tile_path.rename(new_path)
            renamed_count += 1

        if renamed_count > 0:
            print(f"  ✓ Renamed {renamed_count} NED tiles to SRTM format")
            print("    Example: USGS_13_n35w085.tif → USGS_13_n34w085.tif")
            print("    (upper-left corner → lower-left corner naming)")
        if skipped_count > 0:
            print(f"  ⏭️  Skipped {skipped_count} tiles (already converted or duplicates)")

    def download_bbox(self, bbox: BoundingBox) -> tuple:
        """
        Download NED tiles for a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            Tuple of (success: bool, new_files_count: int)
        """
        min_lat, min_lon, max_lat, max_lon = bbox

        # Validate that bbox is in US coverage area
        # NED covers: Continental US (25-50°N, -125 to -65°W) and Hawaii (18-23°N, -161 to -154°W)
        # Alaska has separate coverage but is not included in this check
        is_continental = (24 < max_lat < 50 and -126 < min_lon < -65)
        is_hawaii = (17 < max_lat < 23 and -161 < min_lon < -154)

        if not (is_continental or is_hawaii):
            print("WARNING: NED 10m is primarily available for the continental United States and Hawaii")
            print("Your bounding box may be outside the primary coverage area.")

        tiles = self.get_tiles_for_bbox(bbox)

        # Check which tiles already exist or are known to be unavailable
        existing_tiles = []
        tiles_to_download = []
        unavailable_tiles = []

        for tile in tiles:
            # For each NED tile name (upper-left format), check if we have it in final format
            # Final format: {lower-left}.tif (without USGS_13_ prefix)

            # Calculate SRTM-format name (lower-left corner) - this is the final format
            srtm_tile_name = self.get_srtm_format_tile_name(tile)
            final_filename = f"{srtm_tile_name}.tif"
            final_file = self.output_dir / final_filename

            # Also check temp format in case previous download didn't complete rename
            temp_filename = f"USGS_13_{tile}.tif"
            temp_file = self.output_dir / temp_filename

            # Check if tile exists in final or temp format with valid size
            file_exists = False
            if final_file.exists() and final_file.stat().st_size > 1_000_000:
                file_exists = True
            elif temp_file.exists() and temp_file.stat().st_size > 1_000_000:
                # Rename to final format
                temp_file.rename(final_file)
                file_exists = True

            if file_exists:
                existing_tiles.append(tile)
            else:
                # Check unavailable using SRTM format (lower-left corner) since that's
                # how files are stored and how .unavailable entries are named
                if self.is_tile_unavailable(srtm_tile_name):
                    unavailable_tiles.append(tile)
                else:
                    # Download all missing tiles (no overlap check - preserving all datasets for fallback)
                    tiles_to_download.append(tile)

        # Print summary
        if unavailable_tiles:
            print(
                f"\n  Skipping {len(unavailable_tiles)} tiles known to be unavailable (ocean/uncovered)"
            )

        if existing_tiles and tiles_to_download:
            print(
                f"\nFound {len(tiles)} NED tiles: {len(existing_tiles)} already exist, {len(tiles_to_download)} to download"
            )
            print(f"  Existing: {', '.join(sorted(existing_tiles)[:10])}{'...' if len(existing_tiles) > 10 else ''}")
            print(f"  To download: {', '.join(sorted(tiles_to_download)[:10])}{'...' if len(tiles_to_download) > 10 else ''}")
        elif existing_tiles and not tiles_to_download:
            total_available = len(existing_tiles)
            print(f"\n✓ All {total_available} available NED tiles already exist")
            if unavailable_tiles:
                print(f"  ({len(unavailable_tiles)} tiles are unavailable for this region)")
            return (True, 0)  # Success, but no new files downloaded
        elif tiles_to_download:
            print(
                f"\nFound {len(tiles_to_download)} NED tiles to download: {', '.join(tiles_to_download)}"
            )
        else:
            print("\n⚠️  No tiles to download (all tiles are unavailable)")
            return (False, 0)

        success_count = len(existing_tiles)  # Count existing as successful
        new_files_count = 0  # Track newly downloaded files
        not_found_count = 0  # 404s (water/uncovered areas)
        error_count = 0  # Real errors
        tiles_marked_unavailable = []  # Track tiles we mark unavailable this run

        # Main progress bar for overall tile download
        with tqdm(
            total=len(tiles_to_download),
            desc="   NED",
            unit=" tiles",
            position=0,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100,
            ascii=" █"
        ) as main_pbar:

            for tile in tiles_to_download:
                success, was_404 = self.download_tile(tile, main_pbar)
                if success:
                    success_count += 1
                    new_files_count += 1  # Track new downloads
                elif was_404:
                    # Mark tile as unavailable using SRTM format (lower-left corner)
                    # This matches the format used by tile_validator and file storage
                    srtm_tile_name = self.get_srtm_format_tile_name(tile)
                    self.mark_tile_unavailable(srtm_tile_name)
                    tiles_marked_unavailable.append(srtm_tile_name)
                    not_found_count += 1
                else:
                    error_count += 1
                main_pbar.update(1)

        # Safety check: only rollback unavailable marks if ALL tiles failed
        # (If some succeeded, server is working and 404s are genuine ocean tiles)
        # This handles island regions like Hawaii where >95% of tiles are ocean
        if tiles_to_download and success_count == 0 and not_found_count > 0:
            if tiles_marked_unavailable:
                print("\n⚠️  All tiles returned 404 - possible server or network issue")
                print("  Clearing unavailable marks to prevent false positives")
                self._clear_unavailable_tiles(tiles_marked_unavailable)

        # Print summary
        print(f"\n✓ Successfully downloaded {success_count} NED tiles")
        if not_found_count > 0:
            print(f"  ({not_found_count} tiles unavailable - ocean/uncovered areas)")
        if error_count > 0:
            print(f"  ({error_count} tiles failed to download)")

        # Note: Tiles are automatically renamed to SRTM format during download
        # (see download_tile method for details)

        return (success_count > 0, new_files_count)


class SRTMDownloader(BaseDEMDownloader):
    """
    Downloader for SRTM (Shuttle Radar Topography Mission) 30m.

    SRTM provides global elevation data at 30m (1 arc-second) resolution.
    Coverage: 60°N to 56°S (most populated areas of Earth).

    Data source: OpenTopography S3 (public, no authentication required)
    - https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1

    Note: As of December 2025, NASA LP DAAC Data Pool was retired.
    SRTM data is now served from OpenTopography's public S3 bucket.
    """

    # SRTM data via OpenTopography S3 - public access, no auth required
    BASE_URL = "https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/"

    def __init__(self, output_dir: Path, credentials: tuple[str, str] | None = None):
        """Initialize SRTM downloader.

        Args:
            output_dir: Directory to save downloads
            credentials: Deprecated - no longer required (kept for backward compatibility)
        """
        super().__init__(output_dir, "SRTM 30m")
        # Credentials no longer needed - OpenTopography S3 is public
        self.credentials = None
        self.session = None
        self._credentials_validated = True  # Always "validated" since no auth needed

    def _validate_credentials(self) -> bool:
        """
        Validate access to SRTM data source.

        Returns:
            True - OpenTopography S3 is public, no credentials needed.
        """
        # OpenTopography S3 is public - always accessible
        return True

    def get_tile_name(self, lat: float, lon: float) -> str:
        """
        Get SRTM tile name for a given coordinate.

        SRTM tiles are 1° x 1° and named based on the southwest corner.
        Format: N00E000 or S00W000

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Tile name (e.g., "N35E139")
        """
        tile_lat = math.floor(lat)
        tile_lon = math.floor(lon)

        lat_str = f"{'N' if tile_lat >= 0 else 'S'}{abs(tile_lat):02d}"
        lon_str = f"{'E' if tile_lon >= 0 else 'W'}{abs(tile_lon):03d}"

        return f"{lat_str}{lon_str}"

    def get_tiles_for_bbox(self, bbox: BoundingBox) -> List[str]:
        """
        Get list of tile names covering a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            List of tile names
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        tiles = set()

        # Iterate over 1° grid
        lat = math.floor(min_lat)
        while lat <= max_lat:
            lon = math.floor(min_lon)
            while lon <= max_lon:
                tiles.add(self.get_tile_name(lat, lon))
                lon += 1
            lat += 1

        return sorted(tiles)

    def download_tile(self, tile_name: str, main_pbar=None) -> tuple[bool, bool]:
        """
        Download a single SRTM tile.

        Args:
            tile_name: Tile name (e.g., "N35E139")
            main_pbar: Optional main progress bar to update

        Returns:
            Tuple of (success: bool, was_404: bool)
            was_404 indicates tile doesn't exist (water/uncovered area)
        """
        # OpenTopography provides GeoTIFF files directly (no zip extraction needed)
        filename = f"{tile_name}.tif"
        url = urljoin(self.BASE_URL, filename)

        tif_file = self.output_dir / filename

        # Skip if already downloaded
        if tif_file.exists():
            return (True, False)

        # Download the tile directly (no session needed - public S3)
        success, status_code = self.download_file(
            url,
            tif_file,
            desc=f"     ├─ {filename}",
            session=None,  # No auth needed
            silent_404=True,
            position=1,
            leave=False,
            colour='white'
        )

        if not success:
            # 404 = tile doesn't exist (water/uncovered area)
            was_404 = status_code == 404
            return (False, was_404)

        return (True, False)

    def download_bbox(self, bbox: BoundingBox) -> tuple:
        """
        Download SRTM tiles for a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)

        Returns:
            Tuple of (success: bool, new_files_count: int)
        """
        min_lat, min_lon, max_lat, max_lon = bbox

        # Validate coverage
        if max_lat > 60 or min_lat < -56:
            print("WARNING: SRTM coverage is limited to 60°N to 56°S")
            if max_lat > 60:
                print(f"  Your bounding box extends to {max_lat}°N (above 60°N)")
            if min_lat < -56:
                print(f"  Your bounding box extends to {min_lat}°S (below 56°S)")
            print("  Consider using ArcticDEM (>60°N) or REMA (<60°S) for polar regions.")

        tiles = self.get_tiles_for_bbox(bbox)

        # Check which tiles already exist or are known to be unavailable
        existing_tiles = []
        tiles_to_download = []
        unavailable_tiles = []

        for tile in tiles:
            tif_file = self.output_dir / f"{tile}.tif"
            hgt_file = self.output_dir / f"{tile}.hgt"
            if tif_file.exists() or hgt_file.exists():
                existing_tiles.append(tile)
            elif self.is_tile_unavailable(tile):
                unavailable_tiles.append(tile)
            else:
                # Download all missing tiles (no overlap check - preserving all datasets for fallback)
                tiles_to_download.append(tile)

        # Print summary
        if unavailable_tiles:
            print(
                f"\n  Skipping {len(unavailable_tiles)} tiles known to be unavailable (ocean/uncovered)"
            )

        if existing_tiles and tiles_to_download:
            print(
                f"\nFound {len(tiles)} SRTM tiles: {len(existing_tiles)} already exist, {len(tiles_to_download)} to download"
            )
        elif existing_tiles and not tiles_to_download:
            total_available = len(existing_tiles)
            print(f"\n✓ All {total_available} available SRTM tiles already exist")
            if unavailable_tiles:
                print(f"  ({len(unavailable_tiles)} tiles are unavailable for this region)")
            return (True, 0)  # Success, but no new files
        elif tiles_to_download:
            print(
                f"\nFound {len(tiles_to_download)} SRTM tiles to download: {', '.join(tiles_to_download)}"
            )
        else:
            print("\n⚠️  No tiles to download (all tiles are unavailable)")
            return (False, 0)

        success_count = len(existing_tiles)  # Count existing as successful
        new_files_count = 0  # Track newly downloaded files
        not_found_count = 0  # 404s (water/uncovered areas)
        error_count = 0  # Real errors
        tiles_marked_unavailable = []  # Track tiles we mark unavailable this run

        # Main progress bar for overall tile download
        with tqdm(
            total=len(tiles_to_download),
            desc="   SRTM",
            unit=" tiles",
            position=0,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100,
            ascii=" █"
        ) as main_pbar:

            for tile in tiles_to_download:
                success, was_404 = self.download_tile(tile, main_pbar)
                if success:
                    success_count += 1
                    new_files_count += 1  # Track new downloads
                elif was_404:
                    # Cache as unavailable (tile is ocean/uncovered area)
                    self.mark_tile_unavailable(tile)
                    tiles_marked_unavailable.append(tile)
                    not_found_count += 1
                else:
                    error_count += 1
                main_pbar.update(1)
                time.sleep(0.3)  # Rate limiting (reduced - public S3 is faster)

        # Safety check: only rollback unavailable marks if ALL tiles failed
        # (If some succeeded, server is working and 404s are genuine ocean tiles)
        # This handles island regions like Hawaii where >95% of tiles are ocean
        if tiles_to_download and success_count == 0 and not_found_count > 0:
            if tiles_marked_unavailable:
                print("\n⚠️  All tiles returned 404 - possible server issue")
                print("  Clearing unavailable marks to prevent false positives")
                # Remove the tiles we just marked from the unavailable file
                self._clear_unavailable_tiles(tiles_marked_unavailable)
                tiles_marked_unavailable = []

        # Report results
        if success_count > 0:
            print(f"\n✓ SRTM download complete: {success_count}/{len(tiles)} tiles downloaded")
            if not_found_count > 0:
                print(f"     {not_found_count} tiles not available (likely water/uncovered areas)")
            if error_count > 0:
                print(f"  ⚠️  {error_count} tiles failed with errors")
            return (True, new_files_count)
        else:
            # All tiles failed
            if not_found_count == len(tiles_to_download):
                print("\n⚠️  No SRTM tiles found for this region")
                print("  This area may be entirely over water or not covered by SRTM")
                print("  (SRTM coverage: 60°N to 56°S)")
            else:
                print("\n❌ SRTM download failed: no tiles downloaded successfully")
                print("  Possible causes:")
                print("  - Network issues")
                print("  - OpenTopography S3 server unavailable")
            return (False, 0)


# Convenience functions


def download_dem_for_country(
    country_name: str, bbox: BoundingBox, output_dir: Path, datasets: Optional[List[str]] = None
) -> bool:
    """
    Download DEM data for a specific country.

    Args:
        country_name: Name of the country (for logging)
        bbox: Bounding box for the country
        output_dir: Output directory for DEM data
        datasets: List of datasets to download (auto-selected if None)

    Returns:
        True if successful
    """
    print(f"\nDownloading DEM data for: {country_name}")
    print(f"Bounding box: {bbox}")

    min_lat, min_lon, max_lat, max_lon = bbox

    # Auto-select datasets based on location
    if datasets is None:
        datasets = []

        if max_lat < -60:
            datasets.append("rema")
        elif min_lat > 60:
            datasets.append("arcticdem")
        else:
            # SRTM primary (60°N to 56°S), AW3D30 secondary (global coverage)
            datasets.extend(["srtm30m", "aw3d30"])

    success = False
    for dataset in datasets:
        if dataset == "ned10m" or dataset == "ned":
            downloader = NEDDownloader(output_dir / "ned10m")
            success |= downloader.download_bbox(bbox)
        elif dataset == "srtm30m" or dataset == "srtm":
            downloader = SRTMDownloader(output_dir / "srtm30m")
            success |= downloader.download_bbox(bbox)
        elif dataset == "aw3d30":
            downloader = AW3D30Downloader(output_dir / "aw3d30")
            success |= downloader.download_bbox(bbox)
        elif dataset == "aster":
            downloader = ASTERDownloader(output_dir / "aster30m")
            success |= downloader.download_bbox(bbox)
        elif dataset == "rema":
            downloader = REMADownloader(output_dir / "rema32m")
            success |= downloader.download_bbox(bbox)
        elif dataset == "arcticdem":
            downloader = ArcticDEMDownloader(output_dir / "arctic32m")
            success |= downloader.download_bbox(bbox)

    return success


def create_test_dataset_config(config_path: Path) -> bool:
    """
    Create OpenTopoData config with test dataset for initial setup/testing.

    This creates a minimal config with the test-etopo1 dataset that ships
    with OpenTopoData for testing purposes.

    Args:
        config_path: Path to opentopodata-config.yaml

    Returns:
        True if successful
    """
    print("\n[Creating Test Dataset Configuration]")

    test_config = {
        "datasets": [
            {
                "name": "test-dataset",
                "path": "tests/data/datasets/test-etopo1-resampled-1deg/",
                "filename_epsg": 4326,
                "filename_tile_size": 1,
            }
        ],
        "max_locations_per_request": 500,
        "access_control_allow_origin": "*",
    }

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(test_config, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ Created test config at {config_path}")
        print("  ✓ Configured with 'test-dataset' for testing")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create config: {e}")
        return False


def prepare_elevation_data(elevation_data_dir: Path) -> bool:
    """
    Prepare elevation data for OpenTopoData by cleaning up files.

    With subdirectory-based organization, each dataset has its own namespace
    in OpenTopoData, so duplicate coordinates across datasets are no longer an issue.

    This function:
    1. Removes duplicate file types from AW3D30 (keeps only DSM files)
    2. Flattens AW3D30 directory structure
    3. Renames AW3D30 files to simple format

    NOTE: Cross-dataset deduplication is NO LONGER PERFORMED since each dataset
    now lives in its own subdirectory and is configured separately in OpenTopoData.
    This allows keeping all downloaded data for maximum coverage.

    Args:
        elevation_data_dir: Path to elevation_data directory

    Returns:
        True if successful
    """
    elevation_data_dir = Path(elevation_data_dir)

    if not elevation_data_dir.exists():
        print(f"  ⚠️  Elevation data directory not found: {elevation_data_dir}")
        return False

    # Track if any actual work was done
    work_done = False

    # Step 1: Clean up AW3D30 files (silently unless there's work to do)
    aw3d30_dir = elevation_data_dir / "aw3d30"
    if aw3d30_dir.exists():
        # Remove MSK and STK files (keep only DSM)
        msk_files = list(aw3d30_dir.rglob("*_MSK.tif"))
        stk_files = list(aw3d30_dir.rglob("*_STK.tif"))
        removed_count = 0

        for f in msk_files + stk_files:
            f.unlink()
            removed_count += 1

        if removed_count > 0:
            print(f"  Cleaned up {removed_count} duplicate AW3D30 files")
            work_done = True

        # Flatten directory structure and rename files
        dsm_files = list(aw3d30_dir.rglob("*_DSM.tif"))
        moved_count = 0

        for f in dsm_files:
            if f.parent != aw3d30_dir:
                import re
                match = re.search(r"(N\d{3}[EW]\d{3})", f.name)
                if match:
                    coord = match.group(1)
                    new_name = aw3d30_dir / f"{coord}.tif"
                    f.rename(new_name)
                    moved_count += 1

        # Remove nested coordinate-named directories
        import re
        import shutil
        srtm_pattern = re.compile(r'^[NS]\d{3}[EW]\d{3}$')
        removed_dirs = 0
        for item in aw3d30_dir.iterdir():
            if item.is_dir() and srtm_pattern.match(item.name):
                try:
                    shutil.rmtree(item)
                    removed_dirs += 1
                except Exception:
                    pass

        # Remove any remaining empty directories
        for d in sorted(aw3d30_dir.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except:
                    pass

        if moved_count > 0 or removed_dirs > 0:
            work_done = True

    # Step 2: Count tiles (silent - just for internal tracking)
    dataset_subdirs = ["ned10m", "srtm30m", "aw3d30", "aster30m", "arctic32m", "rema32m"]
    tile_counts = {}

    for dataset_name in dataset_subdirs:
        dataset_dir = elevation_data_dir / dataset_name
        if not dataset_dir.exists():
            continue

        patterns = ["*.hgt", "*.tif"]
        files = []
        for pattern in patterns:
            files.extend(dataset_dir.glob(pattern))

        elevation_files = [f for f in files if "_num.tif" not in f.name]

        if elevation_files:
            tile_counts[dataset_name] = len(elevation_files)

    # Only show summary if work was done
    if work_done:
        print(f"  ✓ Prepared {sum(tile_counts.values())} elevation tiles")

    return True


def update_opentopodata_config(elevation_data_dir: Path, config_path: Path) -> bool:
    """
    Update opentopodata-config.yaml based on downloaded elevation datasets.

    Scans the elevation data directory for downloaded DEM files and updates
    the OpenTopoData configuration to serve those datasets.

    Args:
        elevation_data_dir: Path to elevation_data directory
        config_path: Path to opentopodata-config.yaml

    Returns:
        True if successful
    """
    datasets = []

    # Check for NED (.tif files in ned10m subfolder)
    ned_dir = elevation_data_dir / "ned10m"
    if ned_dir.exists():
        ned_files = list(ned_dir.glob("USGS_13_*.tif")) + list(ned_dir.glob("[ns][0-9][0-9][ew][0-9][0-9][0-9].tif"))
        if ned_files:
            datasets.append(
                {
                    "name": "ned10m",
                    "path": "data/ned10m/",
                    "filename_epsg": 4326,
                    "filename_tile_size": 1,
                    "filename_pattern": "{lat}{lon}.tif",
                }
            )

    # Check for SRTM (.hgt files in srtm30m subfolder)
    srtm_dir = elevation_data_dir / "srtm30m"
    if srtm_dir.exists():
        srtm_files = list(srtm_dir.glob("*.hgt"))
        if srtm_files:
            datasets.append(
                {
                    "name": "srtm30m",
                    "path": "data/srtm30m/",
                    "filename_epsg": 4326,
                    "filename_tile_size": 1,
                    "filename_pattern": "{lat}{lon}.hgt",
                }
            )

    # Check for ASTER (ASTGTMV003_*_dem.tif files in aster30m subfolder)
    aster_dir = elevation_data_dir / "aster30m"
    if aster_dir.exists():
        aster_files = list(aster_dir.glob("ASTGTMV003_*_dem.tif"))
        if aster_files:
            datasets.append(
                {
                    "name": "aster30m",
                    "path": "data/aster30m/",
                    "filename_epsg": 4326,
                    "filename_tile_size": 1,
                    "filename_pattern": "ASTGTMV003_{lat}{lon}_dem.tif",
                }
            )

    # Check for AW3D30 (looks for both old nested structure and new flattened structure)
    aw3d30_dir = elevation_data_dir / "aw3d30"
    if aw3d30_dir.exists():
        aw3d30_flat_files = list(aw3d30_dir.glob("[NS][0-9][0-9][0-9][EW][0-9][0-9][0-9].tif"))
        aw3d30_nested_files = list(aw3d30_dir.rglob("ALPSMLC30_*_DSM.tif"))
        aw3d30_tiles = len(aw3d30_flat_files) + len(aw3d30_nested_files)

        if aw3d30_tiles > 0:
            datasets.append(
                {
                    "name": "aw3d30",
                    "path": "data/aw3d30/",
                    "filename_epsg": 4326,
                    "filename_tile_size": 1,
                    "filename_pattern": "{lat}{lon}.tif",
                }
            )

    # Check for ArcticDEM VRT file (raw tiles can't be used directly by OpenTopoData)
    arctic_vrt = elevation_data_dir / "arctic32m-vrt" / "arctic32m.vrt"
    if arctic_vrt.exists():
        datasets.append(
            {
                "name": "arctic32m",
                "path": "data/arctic32m-vrt/arctic32m.vrt",
            }
        )

    # Check for REMA VRT file (raw tiles can't be used directly by OpenTopoData)
    rema_vrt = elevation_data_dir / "rema32m-vrt" / "rema32m.vrt"
    if rema_vrt.exists():
        datasets.append(
            {
                "name": "rema32m",
                "path": "data/rema32m-vrt/rema32m.vrt",
            }
        )

    if not datasets:
        # No datasets found - write minimal config with test dataset only
        # This ensures the config is cleaned up when all elevation data is deleted
        minimal_config = {
            "datasets": [
                {
                    "name": "test-dataset",
                    "path": "/app/tests/data/datasets/test-etopo1-resampled-1deg/",
                    "filename_epsg": 4326,
                    "filename_tile_size": 1,
                }
            ],
            "max_locations_per_request": 500,
            "access_control_allow_origin": "*",
        }
        try:
            with open(config_path, "w") as f:
                yaml.dump(minimal_config, f, default_flow_style=False)
            print("  ✓ Cleared opentopodata config (no elevation data found)")
            return True
        except Exception as e:
            print(f"  ⚠️  Could not update config: {e}")
            return False

    # Read existing config to check for changes
    existing_datasets = []
    existing_dataset_names = set()
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing_config = yaml.safe_load(f)
                if existing_config and "datasets" in existing_config:
                    existing_dataset_names = {d.get("name") for d in existing_config["datasets"]}
                    # Keep datasets that aren't in our new list
                    new_dataset_names = {d["name"] for d in datasets}
                    existing_datasets = [
                        d
                        for d in existing_config["datasets"]
                        if d.get("name") not in new_dataset_names
                    ]
        except Exception as e:
            print(f"  ⚠️  Could not read existing config: {e}")

    # Combine existing and new datasets
    all_datasets = datasets + existing_datasets

    # Check if config would change
    new_dataset_names = {d["name"] for d in all_datasets}
    config_changed = new_dataset_names != existing_dataset_names

    # Create new config
    config = {
        "datasets": all_datasets,
        "max_locations_per_request": 500,
        "access_control_allow_origin": "*",
    }

    # Only write if config changed
    if not config_changed:
        return True

    # Write config
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ Updated OpenTopoData config ({len(all_datasets)} dataset(s))")

        # Also copy to opentopodata/config.yaml so it's available inside the repo
        opentopodata_config_path = Path("opentopodata") / "config.yaml"
        if opentopodata_config_path.parent.exists():
            try:
                with open(opentopodata_config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception:
                pass  # Silent - not critical

        return True
    except Exception as e:
        print(f"  ❌ Failed to write config: {e}")
        return False


def export_elevation_tile_urls(
    bbox: BoundingBox, datasets: List[str], output_dir: Path = Path(".")
) -> bool:
    """
    Export tile download URLs for specified datasets to files.

    Creates one file per dataset with all tile URLs for manual downloading.

    Args:
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        datasets: List of dataset names ('ned10m', 'srtm30m', 'aw3d30', 'aster', etc.)
        output_dir: Directory to save URL files

    Returns:
        True if at least one export succeeded
    """
    print("\n" + "=" * 80)
    print("EXPORTING ELEVATION TILE URLS")
    print("=" * 80)
    print(f"\nBounding box: {bbox}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Output directory: {output_dir.absolute()}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    success = False

    for dataset in datasets:
        dataset_lower = dataset.lower()
        output_file = output_dir / f"{dataset_lower}_urls.txt"

        try:
            if dataset_lower == "aw3d30":
                downloader = AW3D30Downloader(Path("elevation_data/aw3d30"))
                if downloader.export_tile_urls(bbox, output_file):
                    success = True

            elif dataset_lower in ["srtm30m", "srtm"]:
                downloader = SRTMDownloader(Path("elevation_data/srtm30m"))
                # For now, note that SRTM needs implementation
                with open(output_file, 'w') as f:
                    f.write("# SRTM 30m Tile URLs\n")
                    f.write(f"# Bounding box: {bbox}\n")
                    f.write("# Dataset: NASA SRTM\n")
                    f.write("# Resolution: 30m\n")
                    f.write("# Note: SRTM tiles require authentication via NASA Earthdata\n")
                    f.write("# Use the automated downloader with your credentials instead\n")
                print(f"✓ Created placeholder for {dataset_lower} at {output_file}")
                success = True

            elif dataset_lower in ["ned10m", "ned"]:
                with open(output_file, 'w') as f:
                    f.write("# NED 10m Tile URLs\n")
                    f.write(f"# Bounding box: {bbox}\n")
                    f.write("# Dataset: USGS National Elevation Dataset\n")
                    f.write("# Resolution: 10m\n")
                    f.write("# Coverage: USA only\n")
                    f.write("# Note: NED tiles require authentication via USGS TNM API\n")
                    f.write("# Use the automated downloader instead\n")
                print(f"✓ Created placeholder for {dataset_lower} at {output_file}")
                success = True

            elif dataset_lower == "aster":
                with open(output_file, 'w') as f:
                    f.write("# ASTER GDEM v3 Tile URLs\n")
                    f.write(f"# Bounding box: {bbox}\n")
                    f.write("# Dataset: NASA ASTER GDEM v3\n")
                    f.write("# Resolution: 30m\n")
                    f.write("# Note: ASTER tiles require authentication via NASA Earthdata\n")
                    f.write("# Use the automated downloader with your credentials instead\n")
                print(f"✓ Created placeholder for {dataset_lower} at {output_file}")
                success = True

            else:
                print(f"⚠️  Unknown dataset: {dataset}")

        except Exception as e:
            print(f"✗ Error exporting {dataset}: {e}")

    if success:
        print(f"\n✓ URL files saved to: {output_dir.absolute()}")
        print("\nTo download manually:")
        print("  1. Review the URL files")
        print("  2. Use wget, curl, or FTP client to download tiles")
        print("  3. Extract files to elevation_data/<dataset>/ directory")

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export elevation tile URLs for manual download")
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box as 'min_lat,min_lon,max_lat,max_lon'",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of datasets (e.g., 'aw3d30,srtm30m')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for URL files (default: current directory)",
    )

    args = parser.parse_args()

    # Parse bounding box
    try:
        bbox_parts = [float(x.strip()) for x in args.bbox.split(",")]
        if len(bbox_parts) != 4:
            raise ValueError("Bounding box must have 4 values")
        bbox = tuple(bbox_parts)
    except Exception as e:
        print(f"✗ Invalid bounding box format: {e}")
        print("  Expected format: 'min_lat,min_lon,max_lat,max_lon'")
        exit(1)

    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",")]

    # Export URLs
    output_dir = Path(args.output_dir)
    success = export_elevation_tile_urls(bbox, datasets, output_dir)

    exit(0 if success else 1)
