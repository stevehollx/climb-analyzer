# Climb Analyzer Package

A modular Python package for analyzing road and trail climbs from OpenStreetMap data.

## Package Structure

```
climb_analyzer/
├── __init__.py                 # Main package initialization
├── core/                       # Core data structures
│   ├── __init__.py
│   └── segment.py             # ClimbSegment and ClimbMetrics classes
├── data/                       # Data fetching and management
│   ├── __init__.py
│   ├── elevation.py           # Elevation data fetching
│   ├── geocoding.py           # Reverse geocoding
│   └── spatial_index.py       # Spatial indexing for OSM data
├── processing/                 # Processing and checkpointing
│   ├── __init__.py
│   └── checkpoint.py          # Checkpoint and persistence management
└── utils/                      # Utility functions and helpers
    ├── __init__.py
    ├── graceful_killer.py     # Signal handling for graceful shutdown
    ├── helpers.py             # Standalone utility functions
    └── tee.py                 # Output stream splitting
```

## Modules

### Core (`climb_analyzer.core`)

**segment.py**
- `ClimbSegment`: Dataclass representing a climbing segment with elevation profile
- `ClimbMetrics`: Comprehensive metrics for a climb including elevation stats, scores, and metadata

### Data (`climb_analyzer.data`)

**elevation.py**
- `FastElevationFetcher`: Parallel elevation fetcher with local GeoTIFF support
  - Automatic fallback between datasets (NED10m → SRTM30m → ASTER30m)
  - Adaptive batch sizing for optimal performance
  - Smart checkpointing for resumable operations

**geocoding.py**
- `ReverseGeocoder`: Offline reverse geocoding using reverse_geocoder library
  - Spatial clustering to minimize redundant lookups
  - Fast offline processing
  - Checkpoint support for long operations

**spatial_index.py**
- `SpatialIndexManager`: Manages spatial indexing for OSM ways
  - Efficient bounding box queries
  - Pre-built index loading
- `LocationIndex`: Persistent location index for OSM nodes
  - Memory-mapped file for fast coordinate lookups

### Processing (`climb_analyzer.processing`)

**checkpoint.py**
- `CheckpointConfig`: Global configuration for checkpoint saving
- `SmartCheckpointer`: Intelligent checkpoint manager with time-based and milestone-based saving
- `ChunkPersistenceManager`: Manages persistent storage of chunk processing progress and data
- `configure_checkpoints()`: Configure global checkpoint settings

### Utils (`climb_analyzer.utils`)

**graceful_killer.py**
- `GracefulKiller`: Signal handler for graceful shutdown with checkpoint saving

**tee.py**
- `Tee`: File-like object that redirects output to multiple streams

**helpers.py**
- `calculate_distance_km()`: Calculate distance using Haversine formula
- `determine_cycling_access()`: Determine cycling access based on OSM tags
- `extract_node_coordinates()`: Extract coordinates from a node object
- `extract_coordinates_from_segments()`: Extract unique coordinates from segments
- `check_and_cleanup_memory()`: Check memory usage and cleanup if needed
- `build_elevation_url()`: Build elevation API URL from dataset name
- `get_configured_osm_file_path()`: Get OSM file path from configuration

## Usage Examples

### Basic Import

```python
from climb_analyzer import (
    ClimbSegment,
    ClimbMetrics,
    FastElevationFetcher,
    ReverseGeocoder,
    SpatialIndexManager,
    calculate_distance_km,
)
```

### Fetching Elevation Data

```python
from climb_analyzer.data.elevation import FastElevationFetcher
from climb_analyzer.processing.checkpoint import ChunkPersistenceManager

# Initialize fetcher
fetcher = FastElevationFetcher(primary_dataset="srtm30m")

# Create persistence manager for checkpointing
persistence = ChunkPersistenceManager(analysis_id="my_analysis")

# Fetch elevations
coordinates = [(40.7128, -74.0060), (34.0522, -118.2437)]
elevations = fetcher.fetch_elevations_for_coordinates(
    coordinates,
    persistence,
    progress_desc="Fetching elevations"
)
```

### Reverse Geocoding

```python
from climb_analyzer.data.geocoding import ReverseGeocoder

# Initialize geocoder
geocoder = ReverseGeocoder()

# Geocode coordinates
coordinates = [(40.7128, -74.0060), (34.0522, -118.2437)]
locations = await geocoder.reverse_geocode_parallel(
    coordinates,
    persistence,
    progress_desc="Geocoding locations"
)
```

### Spatial Indexing

```python
from climb_analyzer.data.spatial_index import SpatialIndexManager

# Initialize spatial index manager
index_mgr = SpatialIndexManager(
    osm_file_path="path/to/file.osm.pbf",
    cache_dir="osm_indexes"
)

# Load existing index
index_mgr.load_index()

# Query bounding box
ways = index_mgr.query_bbox(
    min_lat=40.0,
    min_lon=-75.0,
    max_lat=41.0,
    max_lon=-74.0
)
```

### Using Utility Functions

```python
from climb_analyzer.utils.helpers import (
    calculate_distance_km,
    determine_cycling_access,
    check_and_cleanup_memory,
)

# Calculate distance between two points
distance = calculate_distance_km(40.7128, -74.0060, 34.0522, -118.2437)

# Determine if a road allows cycling
tags = {"highway": "residential", "bicycle": "yes"}
access = determine_cycling_access(tags)

# Check and cleanup memory if needed
check_and_cleanup_memory(threshold_percent=80, force_cleanup=False)
```

### Checkpoint Management

```python
from climb_analyzer.processing.checkpoint import (
    CheckpointConfig,
    SmartCheckpointer,
    ChunkPersistenceManager,
)

# Configure checkpoints
config = CheckpointConfig(
    time_interval_minutes=5.0,
    progress_milestones=[25, 50, 75, 100],
    save_at_completion=True
)

# Create checkpointer
checkpointer = SmartCheckpointer(total_items=1000, operation_name="Processing")

# Create persistence manager
persistence = ChunkPersistenceManager(analysis_id="my_analysis")

# Check if checkpoint should be saved
for i in range(1000):
    # ... process item ...

    if checkpointer.should_checkpoint(i):
        persistence.save_progress(processed_chunks, total_chunks, metadata)
```

### Graceful Shutdown

```python
from climb_analyzer.utils.graceful_killer import GracefulKiller

# Create signal handler
signal_handler = GracefulKiller()

# Set persistence manager for checkpoint saving
signal_handler.set_persistence_manager(persistence)

# Set current operation
signal_handler.set_operation(
    "chunk_processing",
    {
        "processed_chunks": [0, 1, 2],
        "total_chunks": 10,
        "metadata": {}
    }
)

# During processing, check for shutdown signal
if signal_handler.kill_now:
    print("Shutdown requested, saving checkpoint...")
    # Checkpoint will be saved automatically
```

## Dependencies

- `pandas`: Data manipulation
- `psutil`: Memory monitoring
- `requests`: HTTP requests for elevation API
- `reverse_geocoder`: Offline reverse geocoding
- `tqdm`: Progress bars
- `rtree`: Spatial indexing (local deployment only)
- `osmium`: OSM file processing (local deployment only)
- `geopy`: Geocoding utilities
- `pyyaml`: Configuration file handling

## Configuration

The package expects a `config.yaml` file with the following settings:

```yaml
ELEVATION_BATCH_SIZE: 100
ELEVATION_MAX_CONCURRENT: 4
ELEVATION_REQUEST_TIMEOUT_SEC: 30
ELEVATION_MAX_RETRIES: 3
ELEVATION_BACKOFF_FACTOR: 2
CHECKPOINT_INTERVAL_MIN: 5.0
CHECKPOINT_MILESTONES_PERC: [25, 50, 75, 100]
TOPO_API_BASE_URL: "http://localhost:5000/v1"
PLANET_FILE_PATH: "path/to/planet.osm.pbf"
```

## Design Principles

1. **Modularity**: Each module has a single, well-defined responsibility
2. **Minimal Dependencies**: Modules only import what they need
3. **Self-Contained**: Each module can be understood and tested independently
4. **PEP 8 Compliance**: Follows Python style guidelines
5. **Type Hints**: Uses type hints for better code clarity
6. **Docstrings**: Comprehensive docstrings for all public interfaces
7. **Error Handling**: Robust error handling with informative messages
8. **Checkpointing**: Built-in checkpoint support for long-running operations
9. **Resource Management**: Automatic memory cleanup and resource management

## Future Improvements

- Add async/await support throughout for better concurrency
- Implement dependency injection for better testability
- Add comprehensive unit tests
- Create a CLI interface
- Add more elevation dataset providers
- Implement caching for frequently accessed data
- Add logging configuration
- Create a plugin system for extensibility

## License

This package is part of the Climb Analyzer project.
