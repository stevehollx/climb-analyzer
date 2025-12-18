# Setup Wizard

The setup wizard guides you through initial configuration of Climb Analyzer.

## Running the Wizard

```bash
./climb-analyzer setup
```

## What the Wizard Does

### 1. Docker Check

Verifies Docker is installed and running:

```
Checking Docker installation...
✓ Docker version 24.0.7
✓ Docker Compose version 2.23.0
```

### 2. Container Build

Builds the Climb Analyzer Docker container:

```
Building Docker container...
✓ Container built successfully
```

### 3. Data Directories

Creates required directories:

```
data/
├── planet_osm_data/     # OSM .pbf files
├── osm_indexes/         # Spatial indexes
├── elevation_data/      # DEM tiles
│   ├── ned10m/
│   ├── srtm30m/
│   ├── aster30m/
│   └── aw3d30/
├── checkpoint_data/     # Analysis checkpoints
└── output/              # Results
```

### 4. Configuration

Creates or updates `config.yaml`:

```yaml
DEPLOYMENT_TYPE: 'local'  # or 'cloud'
TOPO_API_BASE_URL: 'http://opentopodata-server:5000/v1'
ELEVATION_BATCH_SIZE: 100
ELEVATION_MAX_CONCURRENT: 16
```

### 5. Web GUI (Optional)

```
Install web GUI? (y/n): y

Installing GUI dependencies...
✓ npm install complete
✓ GUI build complete

Start GUI now? (y/n): y
✓ GUI running at http://localhost:3000
```

### 6. Elevation Credentials (Optional)

For local mode, NASA Earthdata credentials are needed:

```
Configure NASA Earthdata credentials? (y/n): y

NASA Earthdata is required for downloading elevation data.
Create a free account at: https://urs.earthdata.nasa.gov/users/new

Username: your_username
Password: ********

✓ Credentials saved to .credentials/netrc
```

## Cloud vs Local Mode

The wizard asks you to choose a deployment mode:

### Cloud Mode

- Uses `api.opentopodata.org` for elevation lookups
- No elevation data download required
- Limited to 100,000 coordinate lookups per day
- Good for small regions or testing

### Local Mode (Recommended)

- Runs your own OpenTopoData server
- Requires downloading elevation data (10-350 GB)
- Unlimited queries, faster performance
- Best for large regions or frequent use

## Post-Setup

After the wizard completes:

```bash
# Run interactive mode
./climb-analyzer

# Or run a specific region
./climb-analyzer -r "Vermont"

# Or start the GUI
./climb-analyzer -g
```

## Re-Running Setup

You can run the wizard again at any time:

```bash
./climb-analyzer setup
```

It will:
- Preserve existing data
- Update configuration
- Rebuild containers if needed

## Manual Configuration

If you prefer manual setup, edit `config.yaml` directly:

```yaml
# Deployment type
DEPLOYMENT_TYPE: 'local'

# Elevation API
TOPO_API_BASE_URL: 'http://opentopodata-server:5000/v1'
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

# Cloud cache
CLOUD_CACHE_ENABLED: true
CLOUD_CACHE_REPO: 'stevehollx/global-road-and-trail-climbs'
```

---

Next: [CLI Reference](../user-guide/cli-reference.md)
