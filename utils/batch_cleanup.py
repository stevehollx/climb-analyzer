#!/usr/bin/env python3
"""
Batch mode cleanup utilities for Climb Analyzer.

Handles cleanup of:
- Planet PBF files
- Planet spatial index files
- Elevation data/cache
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import sys

# Add parent directory to path for formatting imports
sys.path.insert(0, str(Path(__file__).parent.parent / "climb_analyzer" / "utils"))
try:
    from formatting import print_header, print_list_item, format_size
except ImportError:
    # Fallback if formatting module not available
    def print_header(text, **kwargs):
        print(f"\n{text}\n")
    def print_list_item(text, **kwargs):
        print(f"  - {text}")
    def format_size(size):
        return f"{size / 1e9:.2f} GB"


def get_cleanup_targets() -> List[Tuple[str, Path, str]]:
    """
    Get list of cleanup targets with descriptions.

    Returns:
        List of (name, path, description) tuples
    """
    targets = []

    # OSM Planet files (show but don't include in cleanup)
    osm_files = list(Path("data/planet_osm_data").glob("*.osm.pbf")) if Path("data/planet_osm_data").exists() else []

    # Planet PBF files (downloaded region files)
    planet_pbf_pattern = Path(".").glob("planet-*.osm.pbf")
    planet_pbf_files = list(planet_pbf_pattern)
    if planet_pbf_files:
        total_size = sum(f.stat().st_size for f in planet_pbf_files if f.exists())
        targets.append(("Downloaded Region Files", planet_pbf_files, f"{total_size / 1e9:.2f} GB"))

    # Planet spatial index files
    index_files = []
    index_patterns = [
        "planet_*_index.pickle",
        "planet_*_metadata.json",
        "*_spatial_index.pickle",
        "*_spatial_metadata.json"
    ]

    for pattern in index_patterns:
        index_files.extend(Path(".").glob(pattern))

    if index_files:
        total_size = sum(f.stat().st_size for f in index_files if f.exists())
        targets.append(("Spatial Index Files", index_files, f"{total_size / 1e9:.2f} GB"))

    # Elevation data/cache
    elevation_dirs = []
    elevation_patterns = [
        "data/elevation_data",
        "elevation_cache",
        ".elevation_cache"
    ]

    for pattern in elevation_patterns:
        path = Path(pattern)
        if path.exists() and path.is_dir():
            # Calculate directory size
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            elevation_dirs.append(path)
            targets.append(("Elevation Data", [path], f"{total_size / 1e9:.2f} GB"))

    return targets, osm_files


def print_cleanup_summary(targets: List[Tuple[str, List, str]], osm_files: List = None):
    """
    Print summary of storage requirements and cleanup targets.

    Args:
        targets: List of cleanup targets
        osm_files: List of OSM planet files (for display only, not cleaned up)
    """
    if not targets and not osm_files:
        print("No storage information available.")
        return

    print_header("Storage Requirements")

    # Show OSM planet files if available (not cleaned up)
    if osm_files:
        osm_total = sum(f.stat().st_size for f in osm_files if f.exists())
        print(f"OSM Planet Files: {format_size(osm_total)}")
        for f in osm_files[:3]:
            print_list_item(str(f.name))
        if len(osm_files) > 3:
            print_list_item(f"... and {len(osm_files) - 3} more files")
        print()

    # Show cleanup targets
    for name, files, description in targets:
        print(f"{name}: {description}")
        if isinstance(files, list) and len(files) <= 3:
            for f in files:
                print_list_item(str(f))
        elif isinstance(files, list) and len(files) > 3:
            for f in files[:2]:
                print_list_item(str(f))
            print_list_item(f"... and {len(files) - 2} more")
        print()


def prompt_cleanup(batch_mode: bool = False, default_yes: bool = True) -> bool:
    """
    Prompt user for cleanup confirmation.

    Args:
        batch_mode: If True, show batch mode message (deprecated - no longer used)
        default_yes: If True, default to yes (Y/n), else default to no (y/N)

    Returns:
        True if user confirms cleanup
    """
    targets, osm_files = get_cleanup_targets()

    if not targets:
        return False

    print_cleanup_summary(targets, osm_files)

    prompt = "\nDelete these files after analysis? (Y/n): " if default_yes else "\nDelete these files after analysis? (y/N): "

    try:
        response = input(prompt).strip().lower()

        if default_yes:
            return response != 'n'
        else:
            return response == 'y'

    except (KeyboardInterrupt, EOFError):
        print("\nCleanup cancelled.")
        return False


def perform_cleanup(targets: List[Tuple[str, List, str]] = None, verbose: bool = True):
    """
    Perform cleanup of specified targets.

    Args:
        targets: List of cleanup targets (auto-detected if None)
        verbose: Print progress messages
    """
    if targets is None:
        targets, _ = get_cleanup_targets()  # Ignore OSM files (not cleaned)

    if not targets:
        if verbose:
            print("No files to clean up.")
        return

    if verbose:
        print("\n" + "=" * 70)
        print("PERFORMING CLEANUP")
        print("=" * 70)

    total_deleted = 0
    total_size_freed = 0

    for name, files, description in targets:
        if verbose:
            print(f"\nCleaning up {name}...")

        for file_path in files:
            file_path = Path(file_path)

            try:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_path.unlink()
                    total_deleted += 1
                    total_size_freed += size
                    if verbose:
                        print(f"  ✓ Deleted: {file_path}")

                elif file_path.is_dir():
                    # Calculate size before deletion
                    size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
                    shutil.rmtree(file_path)
                    total_deleted += 1
                    total_size_freed += size
                    if verbose:
                        print(f"  ✓ Deleted directory: {file_path}")

            except Exception as e:
                if verbose:
                    print(f"  ✗ Error deleting {file_path}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"Cleanup complete: {total_deleted} items deleted, {total_size_freed / 1e9:.2f} GB freed")
        print("=" * 70)


def cleanup_planet_files():
    """Quick cleanup of planet PBF files only."""
    planet_files = list(Path(".").glob("planet-*.osm.pbf"))

    if not planet_files:
        return

    print(f"\nCleaning up {len(planet_files)} planet PBF file(s)...")

    for f in planet_files:
        try:
            size = f.stat().st_size
            f.unlink()
            print(f"  ✓ Deleted {f} ({size / 1e9:.2f} GB)")
        except Exception as e:
            print(f"  ✗ Error deleting {f}: {e}")


def cleanup_index_files():
    """Quick cleanup of spatial index files only."""
    index_patterns = [
        "planet_*_index.pickle",
        "planet_*_metadata.json",
        "*_spatial_index.pickle",
        "*_spatial_metadata.json"
    ]

    index_files = []
    for pattern in index_patterns:
        index_files.extend(Path(".").glob(pattern))

    if not index_files:
        return

    print(f"\nCleaning up {len(index_files)} spatial index file(s)...")

    for f in index_files:
        try:
            f.unlink()
            print(f"  ✓ Deleted {f}")
        except Exception as e:
            print(f"  ✗ Error deleting {f}: {e}")


def cleanup_elevation_data():
    """Quick cleanup of elevation data only."""
    elevation_patterns = ["data/elevation_data", "elevation_cache", ".elevation_cache"]

    for pattern in elevation_patterns:
        path = Path(pattern)
        if path.exists() and path.is_dir():
            try:
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                shutil.rmtree(path)
                print(f"  ✓ Deleted {path} ({size / 1e9:.2f} GB)")
            except Exception as e:
                print(f"  ✗ Error deleting {path}: {e}")


if __name__ == "__main__":
    """Interactive cleanup when run directly."""
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup utility for Climb Analyzer")
    parser.add_argument("--all", action="store_true", help="Clean up all targets")
    parser.add_argument("--planet", action="store_true", help="Clean up planet PBF files only")
    parser.add_argument("--index", action="store_true", help="Clean up spatial index files only")
    parser.add_argument("--elevation", action="store_true", help="Clean up elevation data only")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.planet:
        cleanup_planet_files()
    elif args.index:
        cleanup_index_files()
    elif args.elevation:
        cleanup_elevation_data()
    elif args.all:
        if args.yes:
            perform_cleanup()
        else:
            if prompt_cleanup(batch_mode=False, default_yes=True):
                perform_cleanup()
    else:
        # Interactive mode
        if prompt_cleanup(batch_mode=False, default_yes=False):
            perform_cleanup()
