# Technical Architecture

Deep dive into Climb Analyzer's internal systems for developers and technical users.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIMB ANALYZER PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   OSM Data  │───▶│   Spatial   │───▶│  Elevation  │───▶│    Climb    │  │
│  │  Extraction │    │   Index     │    │   Fetch     │    │  Detection  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                 │                  │                   │          │
│         ▼                 ▼                  ▼                   ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │    Way      │    │   R-tree    │    │   Multi-    │    │   Scoring   │  │
│  │   Merging   │    │   Query     │    │   Dataset   │    │   Engine    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                 │                  │                   │          │
│         └────────────────┴──────────────────┴───────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌─────────────────┐                                │
│                          │  Boundary Merge │                                │
│                          │  & Export       │                                │
│                          └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Package Structure

The climb analyzer is organized as a modular Python package:

```
climb_analyzer/
├── cli.py                 # CLI entry point (re-exports main from engine)
├── config.py              # Configuration constants (parallel merge settings)
├── engine.py              # Main analysis engine (~20K lines, argparse)
├── examples.py            # Example usage code
├── __init__.py            # Package exports
├── __main__.py            # Enables: python -m climb_analyzer
│
├── core/                  # Core analysis logic
│   ├── segment.py         # ClimbSegment, ClimbMetrics data structures
│   ├── merger.py          # BoundaryMerger class (72KB)
│   ├── boundary_merge.py  # DataFrame-based cross-region merging
│   └── unified_climb_merger.py  # Unified climb merging
│
├── data/                  # Data acquisition and management
│   ├── elevation.py       # FastElevationFetcher with fallbacks
│   ├── elevation_downloader.py  # Elevation data downloading
│   ├── elevation_profile.py     # Elevation profile calculations
│   ├── geocoding.py       # ReverseGeocoder class
│   ├── geo_definitions.py # Geographic boundaries (140KB)
│   ├── geo_lookup.py      # Region lookup utilities
│   ├── spatial_index.py   # SpatialIndexManager, LocationIndex, LazyWayMetadataDict
│   ├── index_builder.py   # Spatial index construction
│   ├── manager.py         # DataManager for coordinating operations
│   ├── osm_downloader.py  # OSM data downloading
│   ├── dem_downloaders.py # DEM downloading utilities
│   ├── data_coverage_checker.py  # Coverage checking
│   └── setup.py           # Data setup utilities
│
├── processing/            # Processing utilities
│   └── checkpoint.py      # SmartCheckpointer, ChunkPersistenceManager
│
└── utils/                 # Internal package utilities
    ├── formatting.py      # Output formatting (banners, colors)
    ├── helpers.py         # General helper functions
    ├── ascii_logo.py      # ASCII art logo display
    ├── graceful_killer.py # Signal handling for clean shutdown
    ├── batch_cleanup.py   # Batch cleanup utilities
    └── tee.py             # Tee output to file and console

utils/                     # Top-level utility scripts
├── cloud_cache.py         # Cloud cache operations
├── cloud_cache_config.py  # Cloud cache configuration
├── config_loader.py       # Configuration loading
├── region_detector.py     # Region auto-detection
├── region_mapper.py       # Region name mapping
├── large_country_handler.py  # Large country auto-splitting
├── geographic_menu.py     # Interactive geographic menu
├── github_app_client.py   # GitHub App authentication
├── github_client.py       # GitHub API client
├── error_logger.py        # Error logging utilities
├── elevation_stats_collector.py  # Elevation statistics
└── ...                    # Additional utilities
```

**Entry Points:**
- `./climb-analyzer` - Bash wrapper script (Docker orchestration)
- `python climb_analyzer_main.py` - Direct Python entry
- `python -m climb_analyzer` - Module invocation

---

## 1. OSM Data Extraction

### 1.1 Data Sources

Two deployment modes control data source:

| Mode | Source | Speed | Setup |
|------|--------|-------|-------|
| **Local** | `.osm.pbf` files | Fast | Requires download |
| **Cloud** | Overpass API | Slower | Zero setup |

### 1.2 Way Extraction Process

```
┌─────────────────────────────────────────────────────────────────┐
│                     OSM PBF PROCESSING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  .osm.pbf file                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ Osmium Handler  │  ◄── Streaming parser (low memory)        │
│  │ (WayHandler)    │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ├──▶ Filter by highway type                           │
│           │    (trunk, primary, secondary, tertiary,            │
│           │     unclassified, residential, track, path...)      │
│           │                                                     │
│           ├──▶ Filter by surface (paved/gravel/dirt/all)        │
│           │                                                     │
│           ├──▶ Extract node coordinates                         │
│           │                                                     │
│           └──▶ Extract tags (name, surface, highway type)       │
│                                                                 │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ JSONL Metadata  │  ◄── Flat file format for streaming        │
│  │ (way_id, tags,  │                                            │
│  │  nodes, bbox)   │                                            │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Classes:**
- `SpatialIndexManager` (`climb_analyzer/data/spatial_index.py`)
- `LazyWayMetadataDict` - Memory-efficient lazy-loading with LRU cache

### 1.3 Highway Type Filtering

```python
ROAD_SURFACE_FILTERS = {
    "all": [
        'highway~"trunk|primary|secondary|tertiary|unclassified|'
        'residential|service|track|path|footway|cycleway|bridleway"'
    ],
    "paved": [
        'highway~"..."',
        'surface~"asphalt|concrete|paved|..."'
    ],
    "gravel": [...],
    "dirt": [...]
}
```

---

## 2. Spatial Indexing

### 2.1 R-tree Index Structure

The spatial index enables O(log n) geographic queries instead of O(n) linear scans.

```
┌─────────────────────────────────────────────────────────────────┐
│                      R-TREE INDEX                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌───────────────┐                            │
│                    │   Root Node   │                            │
│                    │  (Global BBox)│                            │
│                    └───────┬───────┘                            │
│              ┌─────────────┼─────────────┐                      │
│              ▼             ▼             ▼                      │
│        ┌─────────┐   ┌─────────┐   ┌─────────┐                  │
│        │ Region1 │   │ Region2 │   │ Region3 │                  │
│        │  BBox   │   │  BBox   │   │  BBox   │                  │
│        └────┬────┘   └────┬────┘   └────┬────┘                  │
│             │             │             │                       │
│        ┌────┴────┐   ┌────┴────┐   ┌────┴────┐                  │
│        ▼         ▼   ▼         ▼   ▼         ▼                  │
│     ┌─────┐  ┌─────┐ ...      ┌─────┐  ┌─────┐                  │
│     │Way1 │  │Way2 │          │WayN │  │WayM │                  │
│     │BBox │  │BBox │          │BBox │  │BBox │                  │
│     └─────┘  └─────┘          └─────┘  └─────┘                  │
│                                                                 │
│  Query: "Find all ways in bbox (lat1,lon1,lat2,lon2)"           │
│  Result: Way IDs that intersect query bbox                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Files Generated:**
- `{region}_spatial.idx` - R-tree index data
- `{region}_spatial.dat` - R-tree node data
- `{region}_metadata.jsonl` - Way metadata (tags, nodes, bbox)
- `{region}_metadata.jsonl.idx_cache` - Pickled position index

### 2.2 Lazy Loading Architecture

```python
class LazyWayMetadataDict:
    """
    Memory-efficient lazy-loading dictionary for way metadata.

    Instead of loading all 4M ways into memory (~16GB), this builds
    an index of file positions and loads way data on-demand.

    Memory usage: ~16 bytes per way (just the position index)
    vs. ~4KB per way if fully loaded
    """

    def __init__(self, jsonl_file_path, cache_size=50000):
        self.way_positions = {}  # Maps way_id -> file_position
        self.cache = {}          # LRU cache for recently accessed
        self.cache_size = cache_size

    def __getitem__(self, way_id):
        # Check LRU cache first
        if way_id in self.cache:
            return self.cache[way_id]

        # Seek to position and load
        pos = self.way_positions[way_id]
        with open(self.jsonl_file, 'r') as f:
            f.seek(pos)
            entry = json.loads(f.readline())

        # Add to LRU cache
        self._add_to_cache(way_id, entry)
        return entry
```

---

## 3. Geographic Chunking

### 3.1 Chunk Calculation

Large regions are divided into manageable chunks to control memory usage.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHUNK GRID SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original Region (e.g., France)                                 │
│  ┌─────────────────────────────────────┐                        │
│  │  ┌───┬───┬───┬───┬───┐              │                        │
│  │  │ 1 │ 2 │ 3 │ 4 │ 5 │              │                        │
│  │  ├───┼───┼───┼───┼───┤              │                        │
│  │  │ 6 │ 7 │ 8 │ 9 │10 │  ◄── Each chunk ~15-25km radius      │
│  │  ├───┼───┼───┼───┼───┤                                       │
│  │  │11 │12 │13 │14 │15 │                                       │
│  │  └───┴───┴───┴───┴───┘                                       │
│  └─────────────────────────────────────┘                        │
│                                                                 │
│  chunk_size_km = 25 (configurable)                              │
│  chunks_per_side = ceil(radius_km / chunk_size_km)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Algorithm:**
```python
def calculate_chunks(center_lat, center_lon, radius_km):
    if radius_km <= chunk_size_km:
        return [(center_lat, center_lon, radius_km)]

    chunks = []
    chunk_radius = chunk_size_km / 2
    chunks_per_side = ceil(radius_km / chunk_size_km)

    for i in range(-chunks_per_side, chunks_per_side + 1):
        for j in range(-chunks_per_side, chunks_per_side + 1):
            # Convert grid position to lat/lon offset
            lat_offset = i * chunk_size_km / 111.0  # km to degrees
            lon_offset = j * chunk_size_km / (111.0 * cos(radians(center_lat)))

            chunk_lat = center_lat + lat_offset
            chunk_lon = center_lon + lon_offset

            # Only include if within original radius
            if haversine(center_lat, center_lon, chunk_lat, chunk_lon) <= radius_km:
                chunks.append((chunk_lat, chunk_lon, chunk_radius))

    return chunks
```

### 3.2 Large Country Auto-Splitting

Countries exceeding memory limits are automatically split into predefined regions.

```python
LARGE_COUNTRIES = {
    "France": {
        "split_method": "latitude",
        "split_at": 46.0,  # Near Lyon
        "regions": [
            ("France North", (46.0, -5.0, 51.2, 9.6)),
            ("France South", (41.3, -5.0, 46.0, 9.6))
        ]
    },
    "United States": {
        "split_method": "custom",
        "regions": [
            ("USA Northeast", (38.0, -83.0, 48.0, -67.0)),
            ("USA Southeast", (24.0, -90.0, 38.0, -75.0)),
            ("USA Midwest", (36.0, -105.0, 49.0, -83.0)),
            ("USA Southwest", (28.0, -125.0, 42.0, -102.0)),
            ("USA Northwest", (42.0, -125.0, 49.0, -102.0)),
            ("USA Alaska", (51.0, -180.0, 72.0, -130.0)),
            ("USA Hawaii", (18.0, -161.0, 23.0, -154.0))
        ]
    },
    # ... Germany, Canada, Russia, China, Brazil, Australia, Argentina, India
}
```

---

## 4. Elevation Fetching

### 4.1 Dataset Priority System

```
┌─────────────────────────────────────────────────────────────────┐
│                  ELEVATION DATASET PRIORITY                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ Region/Latitude │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  lat < -60° (Antarctica)                                   │ │
│  │    ├──▶ 1. REMA 32m                                        │ │
│  │    ├──▶ 2. AW3D30                                          │ │
│  │    └──▶ 3. ASTER 30m                                       │ │
│  │                                                            │ │
│  │  lat > 60° (Arctic)                                        │ │
│  │    ├──▶ 1. ArcticDEM 32m                                   │ │
│  │    ├──▶ 2. AW3D30                                          │ │
│  │    └──▶ 3. ASTER 30m                                       │ │
│  │                                                            │ │
│  │  USA regions                                               │ │
│  │    ├──▶ 1. NED 10m (highest resolution)                    │ │
│  │    ├──▶ 2. SRTM 30m                                        │ │
│  │    ├──▶ 3. AW3D30                                          │ │
│  │    └──▶ 4. ASTER 30m                                       │ │
│  │                                                            │ │
│  │  Global (-60° to 60°)                                      │ │
│  │    ├──▶ 1. SRTM 30m                                        │ │
│  │    ├──▶ 2. AW3D30                                          │ │
│  │    └──▶ 3. ASTER 30m                                       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Parallel Batch Fetching

```
┌─────────────────────────────────────────────────────────────────┐
│                ELEVATION BATCH PROCESSING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Coordinates to fetch: [(lat1,lon1), (lat2,lon2), ...]          │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────┐                        │
│  │    Split into batches               │                        │
│  │    (ELEVATION_BATCH_SIZE = 100)     │                        │
│  └─────────────────────────────────────┘                        │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ThreadPoolExecutor                         │    │
│  │         (ELEVATION_MAX_CONCURRENT = 10)                 │    │
│  │                                                         │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐   │    │
│  │  │ Worker1 │ │ Worker2 │ │ Worker3 │ ... │ WorkerN │   │    │
│  │  │ Batch1  │ │ Batch2  │ │ Batch3  │     │ BatchN  │   │    │
│  │  └────┬────┘ └────┬────┘ └────┬────┘     └────┬────┘   │    │
│  │       │           │           │               │        │    │
│  │       ▼           ▼           ▼               ▼        │    │
│  │  ┌─────────────────────────────────────────────────┐   │    │
│  │  │          OpenTopoData API                       │   │    │
│  │  │  GET /v1/{dataset}?locations=lat1,lon1|lat2,... │   │    │
│  │  └─────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────┐                        │
│  │    Aggregate results                │                        │
│  │    Retry failed with fallback       │                        │
│  └─────────────────────────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**API Request Format:**
```
GET /v1/srtm30m?locations=44.5234,-72.8123|44.5240,-72.8130|...
```

**Configuration:**
```yaml
ELEVATION_BATCH_SIZE: 100          # Coordinates per request
ELEVATION_MAX_CONCURRENT: 10       # Parallel requests
ELEVATION_REQUEST_TIMEOUT_SEC: 30  # Request timeout
ELEVATION_MAX_RETRIES: 3           # Retry attempts
ELEVATION_BACKOFF_FACTOR: 2        # Exponential backoff
```

---

## 5. Way Merging

### 5.1 Street-Based Merging

Roads from OSM are individual "ways" that often represent fragments of a continuous road.
The merger reconnects these fragments.

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAY MERGING PROCESS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Individual OSM ways                                     │
│                                                                 │
│  Way A (id:123)    Way B (id:456)    Way C (id:789)            │
│  ●────────●        ●────────●        ●────────●                 │
│       "Main St"         "Main St"         "Main St"             │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│                                                                 │
│  Step 1: Group by street name                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │ "Main St" → [Way A, Way B, Way C]        │                   │
│  │ "Oak Ave" → [Way D, Way E]               │                   │
│  │ "service" → [Way F, Way G, Way H, ...]   │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│                                                                 │
│  Step 2: Find connected endpoints                               │
│  ┌──────────────────────────────────────────┐                   │
│  │ Endpoint hash: (lat, lon) → way_ids      │                   │
│  │                                          │                   │
│  │ Way A end ≈ Way B start? Connect!        │                   │
│  │ tolerance: 0.002° ≈ 220m                 │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│                                                                 │
│  Step 3: Merge connected ways                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │ Merged: Way A + Way B + Way C            │                   │
│  │ ●────────────────────────────────●       │                   │
│  │ way_ids: [123, 456, 789]                 │                   │
│  │ nodes: concatenated                      │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Parallel vs Serial Processing

```python
class BoundaryMerger:
    """
    Processing modes based on dataset size:

    Serial Mode (< 1000 segments):
    - Progressive merging with checkpointing
    - Memory-efficient streaming
    - ~1000 segments/second

    Parallel Mode (>= 1000 segments):
    - ProcessPoolExecutor with batch workers
    - Default 16 workers (memory-aware scaling)
    - ~5000-10000 segments/second
    """

    def merge_boundary_segments(self, segments, persistence):
        if len(segments) < 1000 or not self.parallel_enabled:
            return self._perform_boundary_merge(segments, persistence)
        else:
            return self._perform_boundary_merge_parallel(segments, persistence)
```

### 5.3 Spatial Grouping for Large Streets

Common street names (e.g., "service", "track") can have thousands of segments.
Spatial grouping prevents O(n²) comparisons.

```python
def _merge_large_street_with_spatial_grouping(self, segments, street_name):
    """
    For streets with >50 segments, group spatially first.

    Grid size: 0.01° ≈ 1.1km
    Only compare segments in same or adjacent grid cells.
    """
    GRID_SIZE = 0.01  # ~1.1 km at equator

    # Build spatial grid
    grid = defaultdict(list)
    for seg in segments:
        lat, lon = seg['start_lat'], seg['start_lon']
        cell = (int(lat / GRID_SIZE), int(lon / GRID_SIZE))
        grid[cell].append(seg)

    # Merge within and across adjacent cells
    merged = []
    for cell, cell_segments in grid.items():
        # Get segments from 9 neighboring cells
        neighbors = get_neighbor_cells(cell)
        nearby_segments = flatten([grid[n] for n in neighbors])
        merged.extend(self._merge_street(nearby_segments, street_name))

    return merged
```

---

## 6. Climb Detection

### 6.1 Climb Identification Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                  CLIMB DETECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Merged road segment with elevation profile              │
│                                                                 │
│  Elevation Profile:                                             │
│        ▲                                                        │
│    500m│           ╱╲                                           │
│    400m│         ╱    ╲                                         │
│    300m│       ╱        ╲                                       │
│    200m│     ╱            ╲                                     │
│    100m│___╱                ╲___                                │
│        └──────────────────────────▶                             │
│                Distance                                         │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│                                                                 │
│  Step 1: Find highest point (peak)                              │
│  ┌──────────────────────────────────────────┐                   │
│  │ peak_idx = index of max elevation        │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│                                                                 │
│  Step 2: Split at peak if bidirectional climb                   │
│  ┌──────────────────────────────────────────┐                   │
│  │ If both sides have significant grade:    │                   │
│  │   Segment A: start → peak (ascending)    │                   │
│  │   Segment B: end → peak (reversed)       │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│                                                                 │
│  Step 3: Calculate metrics for each climb                       │
│  ┌──────────────────────────────────────────┐                   │
│  │ - Elevation gain (m/ft)                  │                   │
│  │ - Distance (km/mi)                       │                   │
│  │ - Average grade (%)                      │                   │
│  │ - Maximum grade (smoothed)               │                   │
│  │ - Basic/FIETS/PDI scores                 │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Grade Calculation

```python
def calculate_max_grade_smoothed(
    self,
    elevations: List[float],
    distances_km: List[float],
    smoothing_window_m: float = 25.0,   # Smooth elevation noise
    grade_window_m: float = 30.0,       # Min distance for grade calc
    outlier_cap_pct: float = 35.0       # Max realistic grade
) -> float:
    """
    Algorithm:
    1. Apply rolling average smoothing to reduce GPS/DEM noise
    2. Calculate grades over grade_window_m distances (not point-to-point)
    3. Cap outliers at physically realistic values
    4. Allow higher grades if consistently steep in area
    """

    # Step 1: Smooth elevations
    smoothed = self.smooth_elevations(elevations, distances_km, smoothing_window_m)

    # Step 2: Calculate cumulative distances
    cumulative_dist = [0.0]
    for d_km in distances_km:
        cumulative_dist.append(cumulative_dist[-1] + d_km * 1000)

    # Step 3: Calculate grades over windows
    max_grade = 0.0
    for i in range(len(smoothed)):
        for j in range(i + 1, len(smoothed)):
            distance_between = cumulative_dist[j] - cumulative_dist[i]

            if distance_between >= grade_window_m:
                elevation_change = smoothed[j] - smoothed[i]
                if elevation_change > 0:
                    grade = (elevation_change / distance_between) * 100

                    # Apply outlier cap with consistency check
                    if grade <= outlier_cap_pct:
                        max_grade = max(max_grade, grade)
                break

    return max_grade
```

---

## 7. Scoring Algorithms

### 7.1 Basic Score

Simple and intuitive metric for comparing climbs.

```
Basic Score = Elevation Gain (m) × Distance (km)

Example:
  - 500m gain over 8km = 4,000 points
  - 1000m gain over 15km = 15,000 points
```

### 7.2 FIETS Index

European-style scoring emphasizing steepness.

```
FIETS = H² / (D × 10) + Altitude Bonus

Where:
  H = Total elevation gain (meters)
  D = Distance (km)

Altitude Bonus (for high mountains):
  If summit > 1000m: bonus = (summit - 1000) / 1000

Example:
  - 500m gain over 8km = 500² / (8 × 10) = 3,125
  - Summit at 1500m adds: (1500-1000)/1000 = 0.5 bonus points
```

### 7.3 PDI (PJAMM Difficulty Index)

Physics-based scoring accounting for surface and altitude effects.

```
PDI = Elevation Factor × Surface Factor × (Work² / Distance)

Where:

Work = (8.6 × D_m + 735 × (Gain - 0.25 × Loss)) / 1600

Elevation Factor = 1 + (min_elev² + max_elev²) / 7,200,000
  - Accounts for altitude effect on performance

Surface Factor = 1 + 0.2 × surface_index
  - asphalt/paved: 0.0
  - compacted: 1.0
  - gravel: 1.5
  - dirt/unpaved: 2.0-2.5

Example:
  - 500m gain, 100m loss, 8000m distance, paved, 200-700m elevation:
  - Work = (8.6×8000 + 735×(500-25)) / 1600 = 260.6
  - Elev Factor = 1 + (200² + 700²) / 7.2M = 1.074
  - Surface Factor = 1.0
  - PDI = 1.074 × 1.0 × (260.6² / 8000) = 9.12
```

### 7.4 Climb Categories

```python
CATEGORY_THRESHOLDS = {
    'basic': {
        'HC':    80000,   # Hors Catégorie
        'Cat 1': 50000,
        'Cat 2': 30000,
        'Cat 3': 15000,
        'Cat 4': 8000
    },
    'fiets': {
        'HC':    8.0,
        'Cat 1': 5.0,
        'Cat 2': 3.0,
        'Cat 3': 1.5,
        'Cat 4': 0.5
    },
    'pdi': {
        'HC':    50,
        'Cat 1': 30,
        'Cat 2': 15,
        'Cat 3': 8,
        'Cat 4': 4
    }
}
```

---

## 8. Boundary Merge System

### 8.1 Cross-Region Climb Merging

When regions are processed separately, climbs at boundaries get split.
The boundary merge system reconnects them.

```
┌─────────────────────────────────────────────────────────────────┐
│                  BOUNDARY MERGE ALGORITHM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Before Merge:                                                  │
│                                                                 │
│  ┌─────────────────┬─────────────────┐                          │
│  │    Region A     │    Region B     │                          │
│  │                 │                 │                          │
│  │   Climb 1       │                 │                          │
│  │   Route D117    │                 │                          │
│  │   8.2 km        │      Climb 2    │                          │
│  │   ───────●      │      ●───────   │                          │
│  │         endpoint│ endpoint        │                          │
│  │                 │      Route D117 │                          │
│  │                 │      3.1 km     │                          │
│  └─────────────────┴─────────────────┘                          │
│                                                                 │
│       │                                                         │
│       ▼                                                         │
│                                                                 │
│  Step 1: Group by street name                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │ "Route D117" → [Climb 1, Climb 2]        │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  Step 2: Check endpoint proximity (< 500m)                      │
│  ┌──────────────────────────────────────────┐                   │
│  │ haversine(Climb1.end, Climb2.start)      │                   │
│  │ = 0.12 km < 0.5 km threshold             │                   │
│  │ → CONNECTED!                             │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  Step 3: Merge metrics                                          │
│  ┌──────────────────────────────────────────┐                   │
│  │ Combined Distance: 8.2 + 3.1 = 11.3 km   │                   │
│  │ Combined Gain: sum of individual gains   │                   │
│  │ Recalculate: avg grade, scores           │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  After Merge:                                                   │
│                                                                 │
│  ┌─────────────────────────────────────────┐                    │
│  │           Complete Climb                │                    │
│  │           Route D117                    │                    │
│  │           11.3 km                       │                    │
│  │           ─────────────────────         │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Merge Types

| Type | Trigger | Action |
|------|---------|--------|
| **Boundary** | Climbs at region boundary with matching endpoints | Merge into single climb |
| **Duplicate** | Same climb in both regions (similar length) | Keep one, discard duplicate |
| **Cross-State** | Climbs spanning US state boundaries | Merge (enabled by default) |
| **Cross-Country** | Climbs spanning country borders | Merge only in Schengen or if `--allow-cross-country-merge` |

### 8.3 Memory Efficiency

```
Traditional Approach:
  Load all 7.9M segments into memory → 16GB+ RAM required

Boundary Merge Approach:
  Operate on ~50K climbs (post-analysis) → ~500MB RAM
  Same algorithm, 32x less memory
```

---

## 9. Checkpointing System

### 9.1 Smart Checkpointing

Analyses can be interrupted and resumed without losing progress.

```
┌─────────────────────────────────────────────────────────────────┐
│                  CHECKPOINT TRIGGERS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Time-based:                                                    │
│    Save every CHECKPOINT_INTERVAL_MIN (default: 15 minutes)     │
│                                                                 │
│  Progress-based:                                                │
│    Save at milestones: [10%, 25%, 50%, 75%, 90%]               │
│                                                                 │
│  Completion:                                                    │
│    Save at 100% completion                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Checkpoint Data Structure

```
data/checkpoint_data/{analysis_id}/
├── progress.pkl           # Current progress state
├── metadata.pkl           # Analysis configuration
├── elevation_progress.pkl # Elevation fetch progress
├── elevation_complete.pkl # Completed elevation data
├── chunks/
│   ├── chunk_0.pkl       # Processed chunk data
│   ├── chunk_1.pkl
│   └── ...
└── elevations/
    └── node_elevations.shelve  # Persistent elevation cache
```

### 9.3 ChunkPersistenceManager

```python
class ChunkPersistenceManager:
    """
    Manages persistent storage of chunk processing progress.

    Key methods:
    - save_completed_elevations(): Persist elevation data
    - save_chunk_progress(): Save processing state
    - load_checkpoint(): Resume from saved state
    - cleanup(): Remove checkpoint files after completion
    """

    def __init__(self, analysis_id: str):
        self.analysis_dir = CHECKPOINT_DIR / analysis_id
        self.chunk_dir = self.analysis_dir / "chunks"
        self.elevation_dir = self.analysis_dir / "elevations"

    def save_chunk_progress(self, chunk_index, segments, elevations):
        """Atomic save with temp file + rename pattern."""
        temp_file = self.chunk_dir / f"chunk_{chunk_index}.tmp"
        final_file = self.chunk_dir / f"chunk_{chunk_index}.pkl"

        with open(temp_file, 'wb') as f:
            pickle.dump({'segments': segments, 'elevations': elevations}, f)

        temp_file.rename(final_file)  # Atomic on POSIX
```

---

## 10. Performance Characteristics

### 10.1 Memory Usage by Component

| Component | Memory | Notes |
|-----------|--------|-------|
| Spatial Index (position only) | ~16 bytes/way | 4M ways = 64MB |
| Spatial Index (full load) | ~4KB/way | 4M ways = 16GB |
| LRU Cache | 50K ways × 4KB | ~200MB |
| Segment Processing | ~2KB/segment | 1M segments = 2GB |
| Elevation Cache | ~16 bytes/coord | 10M coords = 160MB |

### 10.2 Processing Speed

| Operation | Speed | Notes |
|-----------|-------|-------|
| Index Building | ~200K ways/min | 4M ways = 20 min |
| Elevation Fetch | ~10K coords/min | Depends on API |
| Serial Merge | ~1K segments/sec | Single-threaded |
| Parallel Merge | ~5-10K segments/sec | 16 workers |
| Climb Detection | ~5K roads/min | With scoring |

### 10.3 Disk Usage

| Data Type | Size | Example |
|-----------|------|---------|
| OSM PBF | ~1GB per large state | California = 1.1GB |
| Spatial Index | ~30% of PBF | California = 350MB |
| Elevation Tiles | ~100MB-5GB per dataset | NED 10m = 2GB for California |
| Checkpoints | ~100MB-1GB | Depends on progress |
| Output Excel | ~1-10MB | Depends on climb count |

---

## 11. Error Handling

### 11.1 Elevation Fetch Failures

```python
class ElevationFetchLog:
    """Tracks failed and fallback elevation fetches."""

    def add_failed_fetch(self, coord, way_id, street_name, datasets_tried):
        """Record failed fetch with metadata for debugging."""
        self.failed_fetches.append({
            'coordinate': coord,
            'way_id': way_id,
            'street_name': street_name,
            'datasets_tried': datasets_tried
        })
```

### 11.2 Graceful Degradation

- **Elevation failure**: Road excluded from climb analysis (logged)
- **API timeout**: Exponential backoff with fallback datasets
- **Memory pressure**: Automatic chunk size reduction
- **Interrupted analysis**: Resume from last checkpoint

---

## File Reference

### Core Package (`climb_analyzer/`)

| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point (re-exports main from engine) |
| `config.py` | Configuration constants (parallel merge, deduplication) |
| `engine.py` | Main analysis engine (~20K lines, CLI argparse) |
| `examples.py` | Example usage code |
| `__init__.py` | Package initialization and exports |
| `__main__.py` | Module entry point (`python -m climb_analyzer`) |

### Core Module (`climb_analyzer/core/`)

| File | Purpose |
|------|---------|
| `segment.py` | ClimbSegment and ClimbMetrics data structures |
| `merger.py` | BoundaryMerger class (72KB, serial/parallel) |
| `boundary_merge.py` | DataFrame-based cross-region climb merging |
| `unified_climb_merger.py` | Unified climb merging logic |

### Data Module (`climb_analyzer/data/`)

| File | Purpose |
|------|---------|
| `elevation.py` | FastElevationFetcher with dataset fallbacks |
| `elevation_downloader.py` | Elevation data downloading |
| `elevation_profile.py` | Elevation profile calculations |
| `geocoding.py` | ReverseGeocoder class |
| `geo_definitions.py` | Geographic boundaries (140KB) |
| `geo_lookup.py` | Region lookup utilities |
| `spatial_index.py` | SpatialIndexManager, LocationIndex, LazyWayMetadataDict |
| `index_builder.py` | Spatial index construction |
| `manager.py` | DataManager for coordinating data operations |
| `osm_downloader.py` | OSM data downloading |
| `dem_downloaders.py` | DEM (Digital Elevation Model) downloading |
| `data_coverage_checker.py` | Elevation data coverage checking |
| `setup.py` | Data setup utilities |

### Processing Module (`climb_analyzer/processing/`)

| File | Purpose |
|------|---------|
| `checkpoint.py` | SmartCheckpointer, ChunkPersistenceManager |

### Internal Utils (`climb_analyzer/utils/`)

| File | Purpose |
|------|---------|
| `formatting.py` | Output formatting (banners, colored text) |
| `helpers.py` | General helper functions |
| `ascii_logo.py` | ASCII art logo display |
| `graceful_killer.py` | Signal handling for clean shutdown |
| `batch_cleanup.py` | Batch cleanup utilities |
| `tee.py` | Tee output to file and console |

### Top-Level Utils (`utils/`)

| File | Purpose |
|------|---------|
| `cloud_cache.py` | Cloud cache operations |
| `cloud_cache_config.py` | Cloud cache configuration |
| `config_loader.py` | Configuration loading |
| `region_detector.py` | Region auto-detection from input |
| `region_mapper.py` | Region name mapping and normalization |
| `large_country_handler.py` | Auto-splitting for large countries |
| `geographic_menu.py` | Interactive geographic menu |
| `github_app_client.py` | GitHub App authentication |
| `github_client.py` | GitHub API client |
| `error_logger.py` | Error logging utilities |
| `elevation_stats_collector.py` | Elevation fetch statistics |
| `data_validator.py` | Data validation utilities |
| `opentopodata_manager.py` | OpenTopoData server management |

---

Back to [API Reference](api.md) | [Documentation Home](../index.md)
