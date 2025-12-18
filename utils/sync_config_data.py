#!/usr/bin/env python3
"""
Sync config.yaml with actual data on disk.

Scans the data directories to detect what OSM files, elevation datasets,
and checkpoints exist, then updates config.yaml to reflect reality.

This is useful after:
- Manual deletion of data files
- Failed downloads that left partial data
- Moving/restoring data from backups
"""

import yaml
from pathlib import Path
from typing import Dict, List, Set
import sys

# Try to import from utils, fall back to local
try:
    from utils.data_paths import (
        PLANET_OSM_DIR,
        CHECKPOINT_DIR,
        OSM_INDEXES_DIR,
        ELEVATION_DATA_DIR
    )
except ImportError:
    from data_paths import (
        PLANET_OSM_DIR,
        CHECKPOINT_DIR,
        OSM_INDEXES_DIR,
        ELEVATION_DATA_DIR
    )


def scan_osm_files() -> List[str]:
    """Scan for .osm.pbf files and return list of filenames."""
    if not PLANET_OSM_DIR.exists():
        return []

    files = []
    for pbf_file in PLANET_OSM_DIR.glob('*.osm.pbf'):
        # Store just the filename
        files.append(pbf_file.name)

    return sorted(files)


def scan_elevation_datasets() -> List[str]:
    """Scan elevation_data directory and return list of dataset names."""
    if not ELEVATION_DATA_DIR.exists():
        return []

    datasets = []
    for dataset_dir in ELEVATION_DATA_DIR.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
            # Check if dataset has any .tif files
            has_tifs = any(dataset_dir.rglob('*.tif'))
            if has_tifs:
                datasets.append(dataset_dir.name)

    return sorted(datasets)


def scan_checkpoints() -> List[str]:
    """Scan checkpoint_data directory and return list of regions with checkpoints."""
    if not CHECKPOINT_DIR.exists():
        return []

    regions = []
    for checkpoint_dir in CHECKPOINT_DIR.iterdir():
        if checkpoint_dir.is_dir():
            # Check if it contains actual checkpoint files
            if any(checkpoint_dir.glob('*.pkl')) or any(checkpoint_dir.glob('*.json')):
                region_name = checkpoint_dir.name.replace('_', ' ').title()
                regions.append(region_name)

    return sorted(regions)


def scan_osm_indexes() -> List[str]:
    """Scan osm_indexes directory and return list of index files."""
    if not OSM_INDEXES_DIR.exists():
        return []

    # Group by base name (e.g., california-latest)
    index_bases = set()
    for index_file in OSM_INDEXES_DIR.iterdir():
        if index_file.is_file() and ('spatial' in index_file.name or 'metadata' in index_file.name):
            # Extract base name (e.g., "california-latest" from "california-latest.osm_spatial.dat")
            base = index_file.name.split('.osm_')[0] if '.osm_' in index_file.name else index_file.stem
            index_bases.add(base)

    return sorted(list(index_bases))


def update_config_yaml(dry_run: bool = False) -> Dict:
    """
    Update config.yaml with scanned data.

    Returns:
        Dictionary with scan results
    """
    config_file = Path('config.yaml')

    # Load existing config
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Scan data directories
    print("Scanning data directories...")
    osm_files = scan_osm_files()
    elevation_datasets = scan_elevation_datasets()
    checkpoint_regions = scan_checkpoints()
    osm_indexes = scan_osm_indexes()

    print(f"  Found {len(osm_files)} OSM planet files")
    print(f"  Found {len(elevation_datasets)} elevation datasets")
    print(f"  Found {len(checkpoint_regions)} regions with checkpoints")
    print(f"  Found {len(osm_indexes)} OSM indexes")

    # Update config
    changes = []

    # Update OSM_PLANET_DATA
    old_osm = set(config.get('OSM_PLANET_DATA', []))
    new_osm = set(osm_files)

    if old_osm != new_osm:
        added = new_osm - old_osm
        removed = old_osm - new_osm

        if added:
            changes.append(f"  + Added OSM planet files: {', '.join(sorted(added))}")
        if removed:
            changes.append(f"  - Removed OSM planet files: {', '.join(sorted(removed))}")

        config['OSM_PLANET_DATA'] = sorted(osm_files)

    # Update OSM_INDEXES
    old_idx = set(config.get('OSM_INDEXES', []))
    new_idx = set(osm_indexes)

    if old_idx != new_idx:
        added = new_idx - old_idx
        removed = old_idx - new_idx

        if added:
            changes.append(f"  + Added OSM indexes: {', '.join(sorted(added))}")
        if removed:
            changes.append(f"  - Removed OSM indexes: {', '.join(sorted(removed))}")

        config['OSM_INDEXES'] = sorted(osm_indexes)

    # Update ELEVATION_DATA
    old_elev = set(config.get('ELEVATION_DATA', []))
    new_elev = set(elevation_datasets)

    if old_elev != new_elev:
        added = new_elev - old_elev
        removed = old_elev - new_elev

        if added:
            changes.append(f"  + Added elevation datasets: {', '.join(sorted(added))}")
        if removed:
            changes.append(f"  - Removed elevation datasets: {', '.join(sorted(removed))}")

        config['ELEVATION_DATA'] = sorted(elevation_datasets)

    # Write config back
    if changes:
        print("\nChanges detected:")
        for change in changes:
            print(change)

        if not dry_run:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"\n✓ Updated {config_file}")
        else:
            print("\n[DRY RUN] Would update config.yaml")
    else:
        print("\n✓ Config already in sync with data on disk")

    return {
        'osm_files': osm_files,
        'osm_indexes': osm_indexes,
        'elevation_datasets': elevation_datasets,
        'checkpoint_regions': checkpoint_regions,
        'changes': changes
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Sync config.yaml with actual data on disk')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would change without modifying config.yaml')
    args = parser.parse_args()

    print("=" * 80)
    print("Config.yaml Data Sync")
    print("=" * 80)
    print()

    if args.dry_run:
        print("⚠ DRY RUN MODE - config.yaml will not be modified\n")

    result = update_config_yaml(dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  OSM Planet Files: {len(result['osm_files'])}")
    print(f"  OSM Indexes: {len(result['osm_indexes'])}")
    print(f"  Elevation Datasets: {len(result['elevation_datasets'])}")
    print(f"  Checkpoint Regions: {len(result['checkpoint_regions'])}")

    if not result['changes']:
        print("\n✓ Config is already in sync")
    elif args.dry_run:
        print("\n⚠ Run without --dry-run to apply changes")

    return 0


if __name__ == '__main__':
    sys.exit(main())
