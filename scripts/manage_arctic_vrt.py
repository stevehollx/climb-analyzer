#!/usr/bin/env python3
"""
ArcticDEM/REMA VRT Management Script

This script manages the VRT (Virtual Raster) files for ArcticDEM and REMA datasets
used by OpenTopoData. It can:
1. List tiles currently in the VRT
2. Download new tiles for a region
3. Rebuild the VRT with all available tiles

DUAL USAGE:
    1. **Automatic** (during analysis):
       - Imported by dem_downloaders.py
       - VRT automatically rebuilt after downloading Arctic/Antarctic tiles
       - No user action needed - happens during Alaska/Greenland/Antarctica analysis

    2. **Manual** (maintenance/debugging):
       - Run directly for VRT management
       - Useful for troubleshooting, adding tiles, verifying VRT files

AUTOMATIC USAGE:
    When you analyze Arctic or Antarctic regions (Alaska, Greenland, Antarctica),
    the system automatically:
    - Downloads elevation tiles via dem_downloaders.py
    - Imports ArcticVRTManager from this script
    - Rebuilds VRT for OpenTopoData
    - No manual intervention needed!

MANUAL USAGE:
    # List current tiles in VRT
    docker exec climb-analyzer python /app/scripts/manage_arctic_vrt.py --dataset arctic32m --list

    # Download tiles for a new region
    docker exec climb-analyzer python /app/scripts/manage_arctic_vrt.py \\
        --dataset arctic32m --download --bbox 63,-25,67,-13

    # Rebuild VRT after adding new tiles
    docker exec climb-analyzer python /app/scripts/manage_arctic_vrt.py \\
        --dataset arctic32m --rebuild

    # Do everything: download + rebuild
    docker exec climb-analyzer python /app/scripts/manage_arctic_vrt.py \\
        --dataset arctic32m --download --bbox 63,-25,67,-13 --rebuild

DOCKER NOTE:
    This script requires GDAL tools and should run inside the climb-analyzer container.
    All dependencies are pre-installed in the container.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Set
import xml.etree.ElementTree as ET

# Add parent directory to path to import DEM downloaders
sys.path.insert(0, str(Path(__file__).parent.parent))
from climb_analyzer.data.dem_downloaders import ArcticDEMDownloader, REMADownloader


class ArcticVRTManager:
    """Manages ArcticDEM/REMA VRT files for OpenTopoData."""

    DATASET_CONFIG = {
        'arctic32m': {
            'tiles_dir': 'elevation_data/arctic32m',
            'vrt_dir': 'elevation_data/arctic32m-vrt',
            'vrt_name': 'arctic32m.vrt',
            'downloader': ArcticDEMDownloader,
            'description': 'ArcticDEM 32m resolution (Arctic regions)',
        },
        'rema32m': {
            'tiles_dir': 'elevation_data/rema32m',
            'vrt_dir': 'elevation_data/rema32m-vrt',
            'vrt_name': 'rema32m.vrt',
            'downloader': REMADownloader,
            'description': 'REMA 32m resolution (Antarctica)',
        }
    }

    def __init__(self, dataset: str, base_dir: Path = None):
        """Initialize VRT manager.

        Args:
            dataset: Dataset name ('arctic32m' or 'rema32m')
            base_dir: Base directory (defaults to script parent dir)
        """
        if dataset not in self.DATASET_CONFIG:
            raise ValueError(f"Unknown dataset '{dataset}'. Valid: {list(self.DATASET_CONFIG.keys())}")

        self.dataset = dataset
        self.config = self.DATASET_CONFIG[dataset]

        # Set paths
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.tiles_dir = self.base_dir / self.config['tiles_dir']
        self.vrt_dir = self.base_dir / self.config['vrt_dir']
        self.vrt_path = self.vrt_dir / self.config['vrt_name']

        # Ensure directories exist
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.vrt_dir.mkdir(parents=True, exist_ok=True)

    def list_tiles_in_vrt(self) -> List[str]:
        """List all tiles referenced in the VRT file.

        Returns:
            List of tile filenames (not full paths)
        """
        if not self.vrt_path.exists():
            print(f"⚠️  VRT file not found: {self.vrt_path}")
            return []

        try:
            tree = ET.parse(self.vrt_path)
            root = tree.getroot()

            tiles = []
            for source_filename in root.findall('.//SourceFilename'):
                filename = source_filename.text
                # Extract just the filename from relative paths
                if filename:
                    tile_name = Path(filename).name
                    tiles.append(tile_name)

            return sorted(set(tiles))  # Remove duplicates and sort

        except Exception as e:
            print(f"❌ Error parsing VRT file: {e}")
            return []

    def list_tiles_on_disk(self) -> List[str]:
        """List all tile files currently on disk.

        Returns:
            List of tile filenames
        """
        if not self.tiles_dir.exists():
            return []

        tiles = [f.name for f in self.tiles_dir.glob('*.tif')]
        return sorted(tiles)

    def print_tile_status(self):
        """Print status of tiles in VRT vs on disk."""
        vrt_tiles = set(self.list_tiles_in_vrt())
        disk_tiles = set(self.list_tiles_on_disk())

        print(f"\n{'=' * 70}")
        print(f"TILE STATUS: {self.dataset}")
        print(f"{'=' * 70}")

        if self.vrt_path.exists():
            print(f"VRT file: {self.vrt_path.relative_to(self.base_dir)}")
        else:
            print(f"VRT file: NOT FOUND")

        print(f"Tiles directory: {self.tiles_dir.relative_to(self.base_dir)}")
        print()

        print(f"Tiles in VRT:     {len(vrt_tiles)}")
        print(f"Tiles on disk:    {len(disk_tiles)}")
        print()

        # Check for discrepancies
        missing_from_disk = vrt_tiles - disk_tiles
        missing_from_vrt = disk_tiles - vrt_tiles

        if missing_from_disk:
            print(f"⚠️  {len(missing_from_disk)} tiles in VRT but MISSING from disk:")
            for tile in sorted(missing_from_disk)[:10]:
                print(f"   - {tile}")
            if len(missing_from_disk) > 10:
                print(f"   ... and {len(missing_from_disk) - 10} more")
            print()

        if missing_from_vrt:
            print(f"⚠️  {len(missing_from_vrt)} tiles on disk but NOT in VRT:")
            for tile in sorted(missing_from_vrt)[:10]:
                print(f"   - {tile}")
            if len(missing_from_vrt) > 10:
                print(f"   ... and {len(missing_from_vrt) - 10} more")
            print(f"\n💡 Run with --rebuild to add these tiles to the VRT")
            print()

        if not missing_from_disk and not missing_from_vrt and vrt_tiles:
            print("✓ VRT and disk tiles are in sync")
            print()

        # Show sample tiles
        if disk_tiles:
            print("Sample tiles on disk (first 10):")
            for tile in sorted(disk_tiles)[:10]:
                print(f"   {tile}")
            if len(disk_tiles) > 10:
                print(f"   ... and {len(disk_tiles) - 10} more")

        print(f"{'=' * 70}\n")

    def download_tiles(self, bbox: Tuple[float, float, float, float],
                      force: bool = False) -> Tuple[bool, int]:
        """Download tiles for a bounding box.

        Args:
            bbox: Tuple of (min_lat, min_lon, max_lat, max_lon)
            force: If True, re-download existing tiles

        Returns:
            Tuple of (success, num_new_files)
        """
        print(f"\n{'=' * 70}")
        print(f"DOWNLOADING TILES: {self.dataset}")
        print(f"{'=' * 70}")
        print(f"Bounding box: {bbox}")
        print(f"Description: {self.config['description']}")
        print(f"Output directory: {self.tiles_dir.relative_to(self.base_dir)}")
        print()

        # Initialize downloader
        downloader_class = self.config['downloader']
        downloader = downloader_class(self.tiles_dir)

        # Download
        success, new_files = downloader.download_bbox(bbox, force=force)

        if success:
            print(f"\n✓ Download complete: {new_files} new files")
        else:
            print(f"\n❌ Download failed")

        print(f"{'=' * 70}\n")

        return success, new_files

    def rebuild_vrt(self, verbose: bool = True) -> bool:
        """Rebuild the VRT file from all tiles on disk.

        Args:
            verbose: Print progress messages

        Returns:
            True if successful
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"REBUILDING VRT: {self.dataset}")
            print(f"{'=' * 70}")

        # Get all tiles
        tiles = list(self.tiles_dir.glob('*.tif'))

        if not tiles:
            print(f"❌ No tiles found in {self.tiles_dir}")
            return False

        if verbose:
            print(f"Found {len(tiles)} tiles")
            print(f"VRT output: {self.vrt_path.relative_to(self.base_dir)}")
            print()

        # Build relative paths from VRT directory to tiles
        # VRT is in arctic32m-vrt/, tiles are in arctic32m/.tiles/
        # So relative path is: ../arctic32m/.tiles/filename.tif
        relative_paths = []
        for tile in tiles:
            # Get path relative to VRT directory
            try:
                rel_path = os.path.relpath(tile, self.vrt_dir)
                relative_paths.append(rel_path)
            except ValueError:
                # On Windows, relpath fails if on different drives
                relative_paths.append(str(tile.absolute()))

        # Create temporary file list
        temp_list = self.vrt_dir / '.tile_list.txt'
        with open(temp_list, 'w') as f:
            for path in sorted(relative_paths):
                f.write(f"{path}\n")

        try:
            # Build VRT using gdalbuildvrt
            cmd = [
                'gdalbuildvrt',
                '-input_file_list', str(temp_list),
                str(self.vrt_path)
            ]

            if verbose:
                print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.vrt_dir),  # Run from VRT directory
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                print(f"❌ gdalbuildvrt failed:")
                print(result.stderr)
                return False

            # Fix VRT file to use relativeToVRT="1" and convert absolute paths to relative
            vrt_content = self.vrt_path.read_text()
            vrt_content = vrt_content.replace('relativeToVRT="0"', 'relativeToVRT="1"')

            # Convert absolute paths to relative paths.
            # NOTE: The opentopodata container mounts `data/elevation_data`
            # at `/app/data` (not `/app`), so inside that container the tiles
            # live at `/app/data/arctic32m/` — not `/app/elevation_data/...`.
            # We strip whatever absolute prefix gdalbuildvrt produced and
            # leave `../<dataset>/<file>.tif` (relative to VRT dir).
            import re as _re
            if self.dataset == 'arctic32m':
                vrt_content = _re.sub(
                    r'<SourceFilename relativeToVRT="1">[^<]*?/arctic32m/([^<]+)</SourceFilename>',
                    r'<SourceFilename relativeToVRT="1">../arctic32m/\1</SourceFilename>',
                    vrt_content,
                )
            elif self.dataset == 'rema32m':
                vrt_content = _re.sub(
                    r'<SourceFilename relativeToVRT="1">[^<]*?/rema32m/([^<]+)</SourceFilename>',
                    r'<SourceFilename relativeToVRT="1">../rema32m/\1</SourceFilename>',
                    vrt_content,
                )

            self.vrt_path.write_text(vrt_content)

            if verbose:
                print(result.stdout)
                print(f"\n✓ VRT rebuilt successfully: {self.vrt_path}")
                print(f"   Contains {len(tiles)} tiles")
                print(f"   Fixed relative paths in VRT file")
                print(f"{'=' * 70}\n")

            return True

        except FileNotFoundError:
            print("❌ gdalbuildvrt not found. Please install GDAL.")
            return False

        except Exception as e:
            print(f"❌ Error building VRT: {e}")
            return False

        finally:
            # Clean up temp file
            if temp_list.exists():
                temp_list.unlink()

    def verify_vrt(self) -> bool:
        """Verify the VRT file is valid and can be opened by GDAL.

        Returns:
            True if valid
        """
        if not self.vrt_path.exists():
            print(f"❌ VRT file not found: {self.vrt_path}")
            return False

        try:
            cmd = ['gdalinfo', str(self.vrt_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"❌ VRT validation failed:")
                print(result.stderr)
                return False

            print(f"✓ VRT file is valid")

            # Extract key info
            lines = result.stdout.split('\n')
            for line in lines[:20]:  # First 20 lines usually have the key info
                if 'Size is' in line or 'Coordinate System' in line or 'Origin' in line:
                    print(f"   {line.strip()}")

            return True

        except FileNotFoundError:
            print("❌ gdalinfo not found. Please install GDAL.")
            return False

        except Exception as e:
            print(f"❌ Error verifying VRT: {e}")
            return False


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bounding box string.

    Args:
        bbox_str: Comma-separated "min_lat,min_lon,max_lat,max_lon"

    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    parts = [float(x.strip()) for x in bbox_str.split(',')]
    if len(parts) != 4:
        raise ValueError("Bounding box must be: min_lat,min_lon,max_lat,max_lon")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description='Manage ArcticDEM/REMA VRT files for OpenTopoData',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset',
        choices=['arctic32m', 'rema32m'],
        required=True,
        help='Dataset to manage'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List tiles currently in VRT and on disk'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download new tiles for a region'
    )

    parser.add_argument(
        '--bbox',
        type=str,
        help='Bounding box: min_lat,min_lon,max_lat,max_lon (e.g., "63,-25,67,-13")'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild VRT from all tiles on disk'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify VRT file is valid'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download of existing tiles'
    )

    parser.add_argument(
        '--base-dir',
        type=Path,
        help='Base directory (defaults to script parent directory)'
    )

    args = parser.parse_args()

    # Create manager
    manager = ArcticVRTManager(args.dataset, args.base_dir)

    # Default action: list status
    if not any([args.list, args.download, args.rebuild, args.verify]):
        args.list = True

    success = True

    # List tiles
    if args.list:
        manager.print_tile_status()

    # Download tiles
    if args.download:
        if not args.bbox:
            print("❌ --bbox required for download")
            print("   Example: --bbox 63,-25,67,-13")
            return 1

        try:
            bbox = parse_bbox(args.bbox)
            dl_success, new_files = manager.download_tiles(bbox, force=args.force)
            success = success and dl_success

            # Auto-rebuild if we downloaded new files
            if dl_success and new_files > 0 and not args.rebuild:
                print("💡 New files downloaded. Rebuilding VRT...")
                args.rebuild = True

        except ValueError as e:
            print(f"❌ Invalid bounding box: {e}")
            return 1

    # Rebuild VRT
    if args.rebuild:
        rebuild_success = manager.rebuild_vrt()
        success = success and rebuild_success

        # Auto-verify after rebuild
        if rebuild_success:
            args.verify = True

    # Verify VRT
    if args.verify:
        verify_success = manager.verify_vrt()
        success = success and verify_success

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
