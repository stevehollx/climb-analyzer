"""
Centralized data directory paths for the climb analyzer.

All data storage paths are defined here to ensure consistency across the application.
"""

from pathlib import Path
import os

# Root data directory - all application data stored here
DATA_ROOT = Path(os.getenv('CA_DATA_ROOT', './data'))

# Data subdirectories
PLANET_OSM_DIR = DATA_ROOT / 'planet_osm_data'
CHECKPOINT_DIR = DATA_ROOT / 'checkpoint_data'
OSM_INDEXES_DIR = DATA_ROOT / 'osm_indexes'
ELEVATION_DATA_DIR = DATA_ROOT / 'elevation_data'

# Legacy paths (for backward compatibility during migration)
LEGACY_PLANET_OSM_DIR = Path('./planet-osm')
LEGACY_CHECKPOINT_DIR = Path('./checkpoints')
LEGACY_OSM_INDEXES_DIR = Path('./osm_indexes')
LEGACY_ELEVATION_DATA_DIR = Path('./elevation_data')


def ensure_data_directories():
    """
    Ensure all data directories exist, creating them if necessary.

    Creates the following directory structure:
    data/
    ├── planet_osm_data/     # OSM PBF files
    ├── checkpoint_data/      # Analysis checkpoints
    ├── osm_indexes/          # Spatial indexes
    └── elevation_data/       # DEM tiles
    """
    for directory in [DATA_ROOT, PLANET_OSM_DIR, CHECKPOINT_DIR, OSM_INDEXES_DIR, ELEVATION_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")


def get_planet_osm_path(filename: str) -> Path:
    """Get full path to a planet OSM file."""
    return PLANET_OSM_DIR / filename


def get_checkpoint_path(region_name: str) -> Path:
    """Get checkpoint directory path for a region."""
    return CHECKPOINT_DIR / region_name


def get_osm_index_path(region_name: str) -> Path:
    """Get OSM index path for a region."""
    return OSM_INDEXES_DIR / f"{region_name}.idx"


def get_elevation_dataset_path(dataset_name: str) -> Path:
    """Get elevation dataset directory path."""
    return ELEVATION_DATA_DIR / dataset_name


if __name__ == '__main__':
    # When run directly, ensure all directories exist
    ensure_data_directories()
    print("\nData directory structure:")
    print(f"  Root: {DATA_ROOT.absolute()}")
    print(f"  - Planet OSM: {PLANET_OSM_DIR.relative_to(DATA_ROOT)}")
    print(f"  - Checkpoints: {CHECKPOINT_DIR.relative_to(DATA_ROOT)}")
    print(f"  - OSM Indexes: {OSM_INDEXES_DIR.relative_to(DATA_ROOT)}")
    print(f"  - Elevation Data: {ELEVATION_DATA_DIR.relative_to(DATA_ROOT)}")
