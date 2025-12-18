#!/usr/bin/env python3
"""
Migration script to reorganize data directories into the new structure.

Moves:
  planet-osm/          → data/planet_osm_data/
  checkpoints/         → data/checkpoint_data/
  osm_indexes/         → data/osm_indexes/
  elevation_data/      → data/elevation_data/

Run with --dry-run to see what would be moved without making changes.
"""

import shutil
import argparse
from pathlib import Path
import sys

from data_paths import (
    DATA_ROOT,
    PLANET_OSM_DIR,
    CHECKPOINT_DIR,
    OSM_INDEXES_DIR,
    ELEVATION_DATA_DIR,
    LEGACY_PLANET_OSM_DIR,
    LEGACY_CHECKPOINT_DIR,
    LEGACY_OSM_INDEXES_DIR,
    LEGACY_ELEVATION_DATA_DIR,
    ensure_data_directories
)


def migrate_directory(source: Path, dest: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Migrate data from source to destination directory.

    Returns:
        Tuple of (files_moved, bytes_moved)
    """
    if not source.exists():
        print(f"  ⚠ Source directory does not exist: {source}")
        return 0, 0

    if not list(source.iterdir()):
        print(f"  ⚠ Source directory is empty: {source}")
        return 0, 0

    files_moved = 0
    bytes_moved = 0

    print(f"\n  Moving: {source} → {dest}")

    # Get list of items to move
    items = list(source.iterdir())

    for item in items:
        dest_item = dest / item.name

        # Check if destination already exists
        if dest_item.exists():
            print(f"    ⚠ Skipping {item.name} (already exists at destination)")
            continue

        # Get size
        if item.is_file():
            size = item.stat().st_size
            bytes_moved += size
            files_moved += 1
            size_mb = size / (1024 * 1024)

            if dry_run:
                print(f"    [DRY RUN] Would move: {item.name} ({size_mb:.1f} MB)")
            else:
                print(f"    Moving: {item.name} ({size_mb:.1f} MB)")
                shutil.move(str(item), str(dest_item))

        elif item.is_dir():
            # Count files in directory
            file_count = sum(1 for _ in item.rglob('*') if _.is_file())
            dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            bytes_moved += dir_size
            files_moved += file_count
            size_mb = dir_size / (1024 * 1024)

            if dry_run:
                print(f"    [DRY RUN] Would move: {item.name}/ ({file_count} files, {size_mb:.1f} MB)")
            else:
                print(f"    Moving: {item.name}/ ({file_count} files, {size_mb:.1f} MB)")
                shutil.move(str(item), str(dest_item))

    return files_moved, bytes_moved


def main():
    parser = argparse.ArgumentParser(description='Migrate data to new directory structure')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    args = parser.parse_args()

    print("=" * 80)
    print("Data Directory Migration")
    print("=" * 80)

    if args.dry_run:
        print("\n⚠ DRY RUN MODE - No changes will be made\n")

    # Ensure new directories exist
    print("\n1. Creating new directory structure...")
    ensure_data_directories()

    # Perform migrations
    print("\n2. Migrating data...")

    migrations = [
        (LEGACY_PLANET_OSM_DIR, PLANET_OSM_DIR, "Planet OSM files"),
        (LEGACY_CHECKPOINT_DIR, CHECKPOINT_DIR, "Checkpoint data"),
        (LEGACY_OSM_INDEXES_DIR, OSM_INDEXES_DIR, "OSM indexes"),
        (LEGACY_ELEVATION_DATA_DIR, ELEVATION_DATA_DIR, "Elevation data"),
    ]

    total_files = 0
    total_bytes = 0

    for source, dest, description in migrations:
        print(f"\n  {description}:")
        files, bytes_moved = migrate_directory(source, dest, args.dry_run)
        total_files += files
        total_bytes += bytes_moved

    # Summary
    print("\n" + "=" * 80)
    print("Migration Summary")
    print("=" * 80)
    print(f"  Total files moved: {total_files}")
    print(f"  Total data moved: {total_bytes / (1024 * 1024):.1f} MB")

    if args.dry_run:
        print("\n⚠ This was a DRY RUN - no changes were made")
        print("  Run without --dry-run to perform the migration")
    else:
        print("\n✓ Migration completed successfully!")
        print("\nNext steps:")
        print("  1. Review the migrated data in the 'data/' directory")
        print("  2. Update config.yaml if needed")
        print("  3. Delete old empty directories when ready:")
        for source, _, _ in migrations:
            if source.exists() and not list(source.iterdir()):
                print(f"     rm -rf {source}")

    return 0 if not args.dry_run or total_files > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
