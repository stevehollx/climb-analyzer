# Elevation System

Climb Analyzer uses multiple elevation datasets with intelligent fallback for complete global coverage.

## Overview

The elevation system provides:

- **Complete coverage**: 99.99%+ of coordinates return valid elevation
- **Intelligent fallback**: Multiple datasets queried if primary fails
- **Dual mode**: Cloud (limited) vs Local (full) operation
- **Automatic configuration**: Datasets detected and configured automatically

## Datasets

### Primary Datasets

| Dataset | Resolution | Coverage | Best For |
|---------|------------|----------|----------|
| **NED 10m** | ~10m | USA only | Highest accuracy for US |
| **SRTM 30m** | ~30m | 60°N - 56°S | Global baseline |

### Secondary Datasets

| Dataset | Resolution | Coverage | Best For |
|---------|------------|----------|----------|
| **AW3D30** | ~30m | Global | SRTM gaps |
| **ASTER 30m** | ~30m | Global | Cloud mode fallback |

### Polar Datasets

| Dataset | Resolution | Coverage | Best For |
|---------|------------|----------|----------|
| **ArcticDEM** | ~32m | 60°N+ | Arctic regions |
| **REMA** | ~32m | Antarctica | Antarctic regions |

## Dataset Priorities by Region

### USA (excluding Alaska)

```
Primary:   NED 10m
Secondary: SRTM 30m
```

### Alaska

```
Local:  ArcticDEM → AW3D30 → ASTER
Cloud:  ASTER only
```

### Canada

```
South (<60°N): SRTM → AW3D30 → ASTER
North (>60°N): ArcticDEM → AW3D30 → ASTER
```

### Arctic Regions (Iceland, Norway, Sweden, Finland, Northern Russia)

```
Local:  ArcticDEM → AW3D30 → ASTER
Cloud:  ASTER only
```

### Rest of World

```
Primary:   SRTM 30m
Secondary: AW3D30
Tertiary:  ASTER 30m
```

## Cloud vs Local Mode

### Cloud Mode

Uses `api.opentopodata.org`:

| Pros | Cons |
|------|------|
| No data download | 100K coords/day limit |
| Quick start | Limited datasets |
| No disk space | Slower queries |

**Available datasets**: SRTM, ASTER

### Local Mode (Recommended)

Runs your own OpenTopoData server:

| Pros | Cons |
|------|------|
| Unlimited queries | Requires data download |
| All datasets | 10-350 GB disk space |
| Faster processing | Initial setup time |

**Available datasets**: All (NED, SRTM, ASTER, AW3D30, ArcticDEM, REMA)

## NODATA Fallback

When primary dataset returns no data (gaps, void fills):

1. Query primary dataset
2. If NODATA, query secondary
3. If still NODATA, query tertiary
4. Return best available elevation

**Example flow:**
```
Request: Elevation for restricted area in NED coverage
Step 1: Query NED → returns NODATA
Step 2: Query SRTM → returns 250.0m ✓
Result: 250.0m (from fallback)
```

## Error Handling

### HTTP 404 - Dataset Not Available

```
⚠️  DATASET NOT AVAILABLE: arcticdem32m
HTTP 404: Dataset 'arcticdem32m' not found on server
This dataset will be skipped for all remaining requests.
```

### HTTP 5xx - Server Error

Retries with exponential backoff:

```
⚠️  HTTP 503 (Service Unavailable)
Retrying with backoff: 1s, 4s, 16s...
```

### HTTP 429 - Rate Limit

```
⚠️  ELEVATION API RATE LIMITED
HTTP 429 after 6 attempts
Daily limit may be approaching (100,000 coords/day)
```

## Configuration

### Dataset Tiers

In `config.yaml`:

```yaml
# Use only primary dataset
ELEVATION_DATASET_TIERS: 'primary'

# Use primary + secondary
ELEVATION_DATASET_TIERS: 'primary+secondary'

# Use all available (default)
ELEVATION_DATASET_TIERS: 'primary+secondary+tertiary'
```

### Performance Settings

```yaml
# Coordinates per API request
ELEVATION_BATCH_SIZE: 100

# Parallel requests
ELEVATION_MAX_CONCURRENT: 16  # Local mode
# ELEVATION_MAX_CONCURRENT: 2   # Cloud mode (rate limited)
```

## Data Sources

### NASA SRTM

- **Coverage**: 60°N to 56°S
- **Resolution**: 1 arc-second (~30m)
- **Source**: [NASA Earthdata](https://earthdata.nasa.gov/)

### USGS NED

- **Coverage**: United States
- **Resolution**: 1/3 arc-second (~10m)
- **Source**: [USGS](https://www.usgs.gov/core-science-systems/ngp/3dep)

### ASTER GDEM

- **Coverage**: Global
- **Resolution**: 1 arc-second (~30m)
- **Source**: [NASA/METI](https://asterweb.jpl.nasa.gov/gdem.asp)

### JAXA AW3D30

- **Coverage**: Global
- **Resolution**: 1 arc-second (~30m)
- **Source**: [JAXA](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/)

### ArcticDEM

- **Coverage**: Arctic (60°N+)
- **Resolution**: ~32m mosaic
- **Source**: [PGC](https://www.pgc.umn.edu/data/arcticdem/)

### REMA

- **Coverage**: Antarctica
- **Resolution**: ~32m mosaic
- **Source**: [PGC](https://www.pgc.umn.edu/data/rema/)

## Storage Requirements

| Dataset | Per Region | Full Coverage |
|---------|------------|---------------|
| NED 10m | 2-10 GB | ~350 GB (USA) |
| SRTM 30m | 500 MB - 2 GB | ~100 GB (global) |
| ASTER 30m | 500 MB - 2 GB | ~100 GB (global) |
| AW3D30 | 500 MB - 2 GB | ~100 GB (global) |
| ArcticDEM | 5-20 GB | ~200 GB (Arctic) |

## Downloading Data

### Via Setup Wizard

```bash
./climb-analyzer setup
# Select elevation data download
```

### Via CLI

```bash
./climb-analyzer -D -r "Vermont"
```

### NASA Earthdata Credentials

Required for NED and some SRTM downloads:

1. Create free account at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. Enter credentials during setup
3. Stored in `.credentials/netrc`

---

Next: [Dataset Priorities](dataset-priorities.md)
