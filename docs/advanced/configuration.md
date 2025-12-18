# Configuration

Complete reference for `config.yaml` settings.

## File Location

```
climb-analyzer/config.yaml
```

## Core Settings

### Deployment Type

```yaml
# 'local' - Run your own elevation server
# 'cloud' - Use api.opentopodata.org
DEPLOYMENT_TYPE: 'local'
```

**Local**: Unlimited queries, requires elevation data download
**Cloud**: No setup, limited to 100K coords/day

## Elevation API

### Server URL

```yaml
# Local mode
TOPO_API_BASE_URL: 'http://opentopodata-server:5000/v1'

# Cloud mode
TOPO_API_BASE_URL: 'https://api.opentopodata.org/v1'
```

### Performance

```yaml
# Coordinates per API request (1-200)
ELEVATION_BATCH_SIZE: 100

# Parallel requests (1-64)
ELEVATION_MAX_CONCURRENT: 16  # Local
# ELEVATION_MAX_CONCURRENT: 2   # Cloud (rate limited)

# Dataset fallback depth
# 'primary' | 'primary+secondary' | 'primary+secondary+tertiary'
ELEVATION_DATASET_TIERS: 'primary+secondary+tertiary'
```

## Checkpointing

```yaml
# Save checkpoint every N minutes
CHECKPOINT_INTERVAL_MIN: 15.0

# Save at these progress percentages
CHECKPOINT_MILESTONES_PERC:
  - 10
  - 25
  - 50
  - 75
  - 90
```

## OSM Processing

```yaml
# Geographic chunk size for large region processing
OSM_CHUNK_SIZE_KM: 30

# Minimum climb length filter (feet)
MIN_CLIMB_LENGTH_FT: 0
```

## Cloud Cache

```yaml
# Enable/disable cloud cache integration
CLOUD_CACHE_ENABLED: true

# Repository for cached analyses
CLOUD_CACHE_REPO: 'stevehollx/global-road-and-trail-climbs'
```

## Cross-Chunk Processing

```yaml
# Merge climbs split at chunk boundaries (v2.1.0+)
ENABLE_CROSS_CHUNK_POSTPROCESS: true
```

## Data Paths

```yaml
# OSM data directory
OSM_DATA_DIR: 'data/planet_osm_data'

# Spatial index directory
OSM_INDEX_DIR: 'data/osm_indexes'

# Elevation data directory
ELEVATION_DATA_DIR: 'data/elevation_data'

# Checkpoint data directory
CHECKPOINT_DIR: 'data/checkpoint_data'

# Analysis output directory
OUTPUT_DIR: 'output'
```

## Complete Example

```yaml
# Deployment
DEPLOYMENT_TYPE: 'local'
TOPO_API_BASE_URL: 'http://opentopodata-server:5000/v1'

# Elevation
ELEVATION_BATCH_SIZE: 100
ELEVATION_MAX_CONCURRENT: 16
ELEVATION_DATASET_TIERS: 'primary+secondary+tertiary'

# Checkpointing
CHECKPOINT_INTERVAL_MIN: 15.0
CHECKPOINT_MILESTONES_PERC:
  - 10
  - 25
  - 50
  - 75
  - 90

# OSM Processing
OSM_CHUNK_SIZE_KM: 30
MIN_CLIMB_LENGTH_FT: 0

# Cloud Cache
CLOUD_CACHE_ENABLED: true
CLOUD_CACHE_REPO: 'stevehollx/global-road-and-trail-climbs'

# Cross-Chunk
ENABLE_CROSS_CHUNK_POSTPROCESS: true

# Paths
OSM_DATA_DIR: 'data/planet_osm_data'
OSM_INDEX_DIR: 'data/osm_indexes'
ELEVATION_DATA_DIR: 'data/elevation_data'
CHECKPOINT_DIR: 'data/checkpoint_data'
OUTPUT_DIR: 'output'
```

## Environment Variables

Settings can also be set via environment variables:

```bash
export DEPLOYMENT_TYPE=cloud
export ELEVATION_MAX_CONCURRENT=2
export CLOUD_CACHE_ENABLED=false
```

Environment variables override `config.yaml` settings.

## Configuration Profiles

### High-Performance Local

For powerful machines with lots of storage:

```yaml
DEPLOYMENT_TYPE: 'local'
ELEVATION_BATCH_SIZE: 200
ELEVATION_MAX_CONCURRENT: 32
ELEVATION_DATASET_TIERS: 'primary+secondary+tertiary'
```

### Cloud Quick-Start

For testing or small analyses:

```yaml
DEPLOYMENT_TYPE: 'cloud'
ELEVATION_BATCH_SIZE: 100
ELEVATION_MAX_CONCURRENT: 2
```

### Low-Memory

For limited RAM systems:

```yaml
ELEVATION_BATCH_SIZE: 50
ELEVATION_MAX_CONCURRENT: 4
OSM_CHUNK_SIZE_KM: 20
```

### Frequent Checkpointing

For unstable systems:

```yaml
CHECKPOINT_INTERVAL_MIN: 5.0
CHECKPOINT_MILESTONES_PERC:
  - 5
  - 10
  - 15
  - 20
  - 25
  - 30
  - 40
  - 50
  - 60
  - 70
  - 80
  - 90
  - 95
```

## Validating Configuration

```bash
# Check current config via GUI
./climb-analyzer -g
# Navigate to /config

# Or view directly
cat config.yaml
```

---

Next: [How to Contribute](../contributing/how-to-contribute.md)
