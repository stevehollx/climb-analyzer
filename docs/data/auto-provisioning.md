# Auto Provisioning

Climb Analyzer automatically downloads and configures required data when you run an analysis.

## Overview

When you select a region for analysis, the system:

1. Checks for required OSM data
2. Downloads missing OSM files
3. Builds spatial index
4. Downloads required elevation data
5. Updates OpenTopoData configuration
6. Restarts the elevation server
7. Validates everything is ready

All automatically, with minimal user interaction.

## How It Works

### Interactive Mode

You'll be prompted before downloads:

```
Select region: Vermont

Checking data availability...

OSM Data:
  ✗ Vermont OSM file not found
  Download vermont-latest.osm.pbf (142 MB)? (y/n): y

Downloading...
✓ Downloaded in 2m 15s

Building spatial index...
✓ Index built in 8m 30s

Elevation Data:
  ✗ NED tiles not found for region
  Download elevation data? (y/n): y

Downloading NED tiles...
✓ Downloaded 25 tiles in 4m 20s

Updating OpenTopoData server...
✓ Config updated
✓ Server restarted
✓ Health check passed

✅ All data ready! Starting analysis...
```

### Batch Mode

Downloads happen automatically without prompts:

```bash
./climb-analyzer -r "Vermont,New Hampshire,Maine"
```

System silently:
- Downloads each region's OSM file
- Builds spatial indexes
- Downloads elevation data
- Configures server

## What Gets Downloaded

### OSM Data

From [Geofabrik](https://download.geofabrik.de/):

| Region | Example File | Size |
|--------|--------------|------|
| Vermont | vermont-latest.osm.pbf | 142 MB |
| California | california-latest.osm.pbf | 1.2 GB |
| Switzerland | switzerland-latest.osm.pbf | 380 MB |

### Elevation Data

Based on region:

| Region | Dataset | Typical Size |
|--------|---------|--------------|
| US state | NED 10m | 2-10 GB |
| European country | SRTM 30m | 500 MB - 2 GB |
| Arctic region | ArcticDEM | 5-20 GB |

## Manual Triggering

### Download Only (No Analysis)

```bash
./climb-analyzer -D -r "Vermont"
```

Downloads OSM and elevation data but doesn't run analysis.

### Update Geographic Boundaries

```bash
./climb-analyzer -U
```

Updates state/country boundary data.

## Storage Locations

```
data/
├── planet_osm_data/           # OSM .pbf files
│   ├── vermont-latest.osm.pbf
│   └── switzerland-latest.osm.pbf
├── osm_indexes/               # Spatial indexes
│   ├── vermont.idx
│   └── vermont.dat
└── elevation_data/            # DEM tiles
    ├── ned10m/
    ├── srtm30m/
    └── aw3d30/
```

## Time Estimates

First-time setup for a region:

| Region | OSM Download | Index Build | Elevation | Total |
|--------|--------------|-------------|-----------|-------|
| Vermont | 2-3 min | 8-10 min | 4-5 min | ~15 min |
| California | 15-20 min | 45-60 min | 30-40 min | ~2 hours |
| Switzerland | 5-8 min | 20-25 min | 10-15 min | ~45 min |

### Subsequent Runs

After initial setup:
- No re-downloads
- Indexes reused
- Analysis starts immediately

## Troubleshooting

### "Download failed"

```bash
# Check internet connection
ping download.geofabrik.de

# Manual download
wget https://download.geofabrik.de/north-america/us/vermont-latest.osm.pbf \
     -P data/planet_osm_data/
```

### "Index build failed"

```bash
# Clear and rebuild
rm data/osm_indexes/vermont.*
./climb-analyzer -r "Vermont"
```

### "Server restart failed"

```bash
# Check Docker
docker ps

# Manual restart
docker restart opentopodata-server

# Check logs
docker logs opentopodata-server
```

### "NASA credentials required"

For NED elevation data:

1. Create account at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. Run setup wizard: `./climb-analyzer setup`
3. Enter credentials when prompted

## Disabling Auto-Provisioning

For manual control, set in `config.yaml`:

```yaml
AUTO_PROVISION_OSM: false
AUTO_PROVISION_ELEVATION: false
```

Then manually download data:

```bash
# OSM
wget https://download.geofabrik.de/.../region.osm.pbf -P data/planet_osm_data/

# Build index
./climb-analyzer build-index data/planet_osm_data/region.osm.pbf

# Elevation (via data_setup.py)
python data_setup.py --region "Region Name"
```

---

Next: [Coverage Strategy](coverage-strategy.md)
