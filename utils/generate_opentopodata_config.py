#!/usr/bin/env python3
"""
Auto-generate OpenTopoData config.yaml based on existing elevation datasets.

Scans the elevation_data directory and creates a config entry for each dataset
that has tiles.
"""

from pathlib import Path
import yaml
import sys

# Add path to climb_analyzer package
sys.path.insert(0, str(Path(__file__).parent.parent))
from climb_analyzer.utils.formatting import print_banner, print_success


# Dataset configurations with their specific patterns
DATASET_CONFIGS = {
    'ned10m': {
        'name': 'ned10m',
        'path': 'data/ned10m/',
        'filename_epsg': 4326,
        'filename_tile_size': 1,
        'filename_pattern': '{lat}{lon}.tif',
        'priority': 1,  # Highest priority (US only, best resolution)
    },
    'srtm30m': {
        'name': 'srtm30m',
        'path': 'data/srtm30m/',
        'filename_epsg': 4326,
        'filename_tile_size': 1,
        'priority': 2,  # Global coverage
    },
    'aw3d30': {
        'name': 'aw3d30',
        'path': 'data/aw3d30/',
        'filename_epsg': 4326,
        'filename_tile_size': 1,
        'filename_pattern': '{lat}{lon}.tif',
        'priority': 3,  # Global fallback
    },
    'aster30m': {
        'name': 'aster30m',
        'path': 'data/aster30m/',
        'filename_epsg': 4326,
        'filename_tile_size': 1,
        'filename_pattern': 'ASTGTMV003_{lat}{lon}_dem.tif',
        'priority': 4,  # Last resort
    },
    'rema32m': {
        'name': 'rema32m',
        'path': 'data/rema32m-vrt/',
        'filename_epsg': 4326,
        'filename_tile_size': 1,
        'priority': 0,  # Antarctica only, highest priority there
    },
    'arctic32m': {
        'name': 'arctic32m',
        'path': 'data/arctic32m-vrt/',
        'filename_epsg': 4326,
        'filename_tile_size': 1,
        'priority': 0,  # Arctic only, highest priority there (requires VRT file)
    },
}


def count_tiles_in_dataset(dataset_dir: Path) -> int:
    """Count number of tiles in a dataset directory."""
    if not dataset_dir.exists():
        return 0

    # Count common elevation file types
    tile_count = 0
    for ext in ['*.tif', '*.hgt', '*.tiff', '*.vrt']:
        tile_count += len(list(dataset_dir.glob(ext)))

    return tile_count


def detect_datasets(elevation_data_dir: Path) -> list:
    """
    Scan elevation_data directory and return list of datasets with tiles.

    Returns:
        List of dataset config dicts, sorted by priority
    """
    datasets = []

    print(f"Scanning elevation data directory: {elevation_data_dir}")
    print()

    for dataset_name, config in DATASET_CONFIGS.items():
        # Check if dataset uses VRT file (path ends with .vrt or -vrt/)
        if config['path'].endswith('.vrt'):
            # Specific VRT file path
            vrt_file = elevation_data_dir / config['path'].replace('data/', '')
            if vrt_file.exists():
                print(f"  ✓ {dataset_name:12s} VRT file")
                dataset_config = {k: v for k, v in config.items() if k != 'priority'}
                datasets.append(dataset_config)
            else:
                print(f"  ✗ {dataset_name:12s} (VRT file not found)")
        elif config['path'].endswith('-vrt/'):
            # VRT directory - find .vrt files inside
            vrt_dir = elevation_data_dir / config['path'].replace('data/', '')
            vrt_files = list(vrt_dir.glob('*.vrt')) if vrt_dir.exists() else []
            if vrt_files:
                print(f"  ✓ {dataset_name:12s} VRT file ({len(vrt_files)} found)")
                dataset_config = {k: v for k, v in config.items() if k != 'priority'}
                datasets.append(dataset_config)
            else:
                print(f"  ✗ {dataset_name:12s} (no VRT file in directory)")
        else:
            # For tile-based datasets, count tiles in directory
            dataset_dir = elevation_data_dir / dataset_name
            tile_count = count_tiles_in_dataset(dataset_dir)

            if tile_count > 0:
                print(f"  ✓ {dataset_name:12s} {tile_count:6,d} tiles")

                # Create config entry (exclude priority from final config)
                dataset_config = {k: v for k, v in config.items() if k != 'priority'}
                datasets.append(dataset_config)
            else:
                print(f"  ✗ {dataset_name:12s} (no tiles)")

    print()

    # Sort by priority (lower number = higher priority)
    datasets.sort(key=lambda d: DATASET_CONFIGS[d['name']]['priority'])

    return datasets


def generate_config(elevation_data_dir: Path, output_path: Path) -> bool:
    """
    Generate OpenTopoData config.yaml file.

    Configures individual datasets that can be queried via comma-separated URLs
    for multi-dataset fallback (e.g., ned10m,srtm30m for US regions).

    Args:
        elevation_data_dir: Path to elevation_data directory
        output_path: Path where config.yaml should be written

    Returns:
        True if successful
    """
    print_banner("OpenTopoData Config Generator", spacing_before=0)

    # Detect datasets
    datasets = detect_datasets(elevation_data_dir)

    if not datasets:
        print("ERROR: No elevation datasets found!")
        print(f"       Checked directory: {elevation_data_dir}")
        return False

    print(f"Found {len(datasets)} dataset(s) with tiles")
    print()

    # Build config with individual datasets (no multi-dataset parents)
    # OpenTopoData will handle fallback via comma-separated queries
    config = {
        'datasets': datasets,
        'max_locations_per_request': 500,
        'access_control_allow_origin': '*',
    }

    # Write config
    print(f"Writing config to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print()
    print_banner("Config generated successfully!", spacing_before=0)
    print("Datasets configured (in priority order):")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset['name']}")
    print()
    print("Note: Multi-dataset queries use comma-separated names (e.g., ned10m,srtm30m)")

    return True


def main():
    """Main entry point."""
    # Determine paths
    # Use current working directory (script is called with 'cd $BASE_DIR' before execution)
    # This ensures it works both on host and inside Docker containers
    script_dir = Path.cwd()
    elevation_data_dir = script_dir / "data" / "elevation_data"

    # Output path (rebuild script will copy to opentopodata/ before build)
    output_path = script_dir / "opentopodata-config.yaml"

    # Generate config
    success = generate_config(elevation_data_dir, output_path)

    if success:
        print()
        print_success(f"Config written to: {output_path}")
        print()
        print("Note: The rebuild script will copy this to opentopodata/config.yaml before building")
        return 0
    else:
        print("✗ Config generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
