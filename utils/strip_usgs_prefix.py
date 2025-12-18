#!/usr/bin/env python3
"""
Strip USGS_13_ prefix from NED tiles for OpenTopoData compatibility

OpenTopoData expects tiles in format: {lat}{lon}.tif
But NED tiles are: USGS_13_{lat}{lon}.tif

This script creates symlinks without the prefix.
"""

import sys
from pathlib import Path


def create_symlinks(ned_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Create symlinks from {lat}{lon}.tif to USGS_13_{lat}{lon}.tif

    Args:
        ned_dir: Directory containing NED tiles
        dry_run: If True, only print what would be done

    Returns:
        Tuple of (created_count, skipped_count)
    """
    print(f"\n{'DRY RUN: ' if dry_run else ''}Creating symlinks for NED tiles...")
    print(f"Directory: {ned_dir}")

    # Find all NED tiles with USGS_13_ prefix
    ned_tiles = list(ned_dir.glob("USGS_13_*.tif"))

    if not ned_tiles:
        print("     No tiles found")
        return (0, 0)

    print(f"  Found {len(ned_tiles)} tiles")

    created_count = 0
    skipped_count = 0

    for tile_path in ned_tiles:
        # Remove USGS_13_ prefix
        original_name = tile_path.name
        new_name = original_name.replace("USGS_13_", "")
        symlink_path = tile_path.parent / new_name

        # Skip if symlink already exists
        if symlink_path.exists():
            skipped_count += 1
            continue

        if dry_run:
            print(f"  Would create: {new_name} -> {original_name}")
            if created_count < 5:
                created_count += 1
        else:
            symlink_path.symlink_to(original_name)
            if created_count < 5:
                print(f"  ✓ {new_name} -> {original_name}")
            created_count += 1

    if dry_run:
        created_count = len([t for t in ned_tiles if not (t.parent / t.name.replace("USGS_13_", "")).exists()])

    print(f"\n  {'Would create' if dry_run else 'Created'} {created_count} symlinks")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} (already exist)")

    return (created_count, skipped_count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create symlinks without USGS_13_ prefix for OpenTopoData"
    )
    parser.add_argument(
        "ned_dir",
        type=str,
        nargs="?",
        default="elevation_data/ned10m",
        help="Path to NED tiles directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done",
    )

    args = parser.parse_args()
    ned_dir = Path(args.ned_dir)

    if not ned_dir.exists():
        print(f"❌ Directory not found: {ned_dir}")
        sys.exit(1)

    created, skipped = create_symlinks(ned_dir, dry_run=args.dry_run)

    if created > 0:
        print(f"\n✓ Successfully processed {created + skipped} tiles")
        if not args.dry_run:
            print("\nNext steps:")
            print("  1. Restart OpenTopoData: python opentopodata_manager.py restart")
    else:
        print("\n✓ All symlinks already exist")

    sys.exit(0)
