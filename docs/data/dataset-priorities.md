# Dataset Priorities

Elevation datasets are selected automatically based on region and deployment mode.

**Note:** As of December 2025, all datasets use public sources with no authentication required.

## Priority Tables

### Cloud Mode (api.opentopodata.org)

Limited dataset availability (only datasets hosted by opentopodata.org):

| Region | Primary | Secondary |
|--------|---------|-----------|
| USA (not Alaska) | ned10m | srtm30m |
| Alaska | srtm30m | - |
| Canada South | srtm30m | - |
| Canada North (>60°N) | srtm30m | - |
| Greenland | *(not supported)* | - |
| Iceland/Nordic | srtm30m | - |
| Russia North (>60°N) | srtm30m | - |
| Russia South | srtm30m | - |
| Antarctica | *(not supported)* | - |
| All Other | srtm30m | - |

### Local Mode (OpenTopoData Server)

Full dataset availability:

| Region | Primary | Secondary |
|--------|---------|-----------|
| USA (not Alaska) | ned10m | srtm30m |
| Alaska | arcticdem32m | aw3d30 |
| Canada South | srtm30m | aw3d30 |
| Canada North (>60°N) | arcticdem32m | aw3d30 |
| Greenland | arcticdem32m | - |
| Iceland/Nordic | arcticdem32m | aw3d30 |
| Russia North (>60°N) | arcticdem32m | aw3d30 |
| Russia South | srtm30m | aw3d30 |
| Antarctica | rema32m | - |
| All Other | srtm30m | aw3d30 |

## How Priorities Work

### Automatic Selection

When you run an analysis, the system:

1. Detects deployment mode (cloud vs local)
2. Determines region from coordinates
3. Selects appropriate priority chain
4. Queries datasets in order until valid elevation found

### Example: Colorado Analysis

```python
Region: Colorado (USA, not Alaska)
Mode: Local

Priority chain: ned10m → srtm30m

Query flow:
  coord (39.7, -104.9):
    1. ned10m → 1609.3m ✓ (use this)

  coord (39.7, -104.9) with NED void:
    1. ned10m → NODATA
    2. srtm30m → 1610.0m ✓ (fallback)
```

### Example: Iceland Analysis

```python
Region: Iceland (Nordic, >60°N)
Mode: Local

Priority chain: arcticdem32m → aw3d30

Query flow:
  coord (64.1, -21.9):
    1. arcticdem32m → 125.4m ✓ (use this)
```

## Configuring Tiers

Control how many fallback datasets to use:

```yaml
# config.yaml

# Primary only (fastest, may have gaps)
ELEVATION_DATASET_TIERS: 'primary'

# Primary + secondary (recommended, good coverage)
ELEVATION_DATASET_TIERS: 'primary+secondary'
```

## Dataset Details

### NED 10m (National Elevation Dataset)

- **Best resolution** in the system (~10m)
- **USA only** coverage
- **Highly accurate** for most US terrain
- May have voids in some areas (restricted zones, water)
- **Source:** AWS S3 (public, no auth)

### SRTM 30m (Shuttle Radar Topography Mission)

- **Global baseline** coverage
- Covers 60°N to 56°S latitude
- **Reliable** and widely used
- No coverage above 60°N
- **Source:** OpenTopography S3 (public, no auth)

### AW3D30 (ALOS World 3D 30m)

- **Global coverage** 84°N to 84°S
- Includes void-filling from ArcticDEM and REMA
- **Better accuracy** than SRTM in many areas (RMSE 5.68m vs 8.28m)
- **Source:** JAXA FTP (public, no auth)

### ArcticDEM

- **Arctic-specific** (60°N and above)
- **High quality** for northern regions
- Covers Alaska, Canada North, Greenland, Nordic countries
- **Local mode only**
- **Source:** AWS S3 (public, no auth)

### REMA (Reference Elevation Model of Antarctica)

- **Antarctica only**
- Only option for Antarctic terrain
- **Local mode only**
- **Source:** AWS S3 (public, no auth)

### ASTER (DEPRECATED)

**ASTER GDEM is no longer available.** As of December 2025, NASA LP DAAC Data Pool was retired.

Use AW3D30 instead - it provides:
- Better coverage (84°N to 84°S vs 83°N to 83°S)
- Better accuracy (RMSE 5.68m vs 11.98m)
- Public access via JAXA FTP

## Storage Trade-offs

### Minimal Setup

Download only primary dataset:

```
ELEVATION_DATASET_TIERS: 'primary'
Storage: ~5-50 GB per region
Coverage: ~99%
```

### Recommended Setup

Download primary + secondary:

```
ELEVATION_DATASET_TIERS: 'primary+secondary'
Storage: ~10-100 GB per region
Coverage: ~99.9%
```

## Troubleshooting

### "Dataset not available" Errors

The dataset isn't downloaded or configured:

```bash
# Download missing data
./climb-analyzer -D -r "Region"

# Or check OpenTopoData config
cat opentopodata-config.yaml
```

### High Elevation Error Rate

May be using cloud mode with limited datasets:

```yaml
# Switch to local mode
DEPLOYMENT_TYPE: 'local'
```

Or download more datasets locally.

### Wrong Dataset Being Used

Check the priority chain output:

```
Using dataset priority for Alaska (local mode): arcticdem32m → aw3d30
```

If unexpected, verify:
1. Deployment mode in config
2. Region detection from coordinates

---

Next: [Auto Provisioning](auto-provisioning.md)
