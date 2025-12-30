# Data Coverage Strategy

How Climb Analyzer achieves complete elevation coverage worldwide.

## Goals

1. **99.99%+ coverage**: Valid elevation for nearly all coordinates
2. **Best accuracy**: Use highest-resolution data when available
3. **Global reach**: Support every region on Earth
4. **Efficient storage**: Don't download unnecessary data

## Coverage Approach

### Multi-Dataset Fallback

Instead of relying on a single dataset, we query multiple in sequence:

```
Primary → Secondary → Tertiary → Return best available
```

**Why?** No single dataset has perfect coverage:
- NED has voids in restricted areas
- SRTM doesn't cover above 60°N
- ASTER has accuracy issues in some terrain

### Region-Specific Optimization

Different regions use different primary datasets:

| Region | Why This Primary |
|--------|------------------|
| USA | NED has 10m resolution (3x better than SRTM) |
| Europe | SRTM well-tested, high quality |
| Arctic | ArcticDEM is only option above 60°N |
| Antarctica | REMA is only option |

### No Tile Reconciliation

Overlapping tiles are preserved (not deduplicated) so fallback works:

```
Vermont coverage:
  - NED tiles: 25 tiles (primary)
  - SRTM tiles: 4 tiles (fallback)

If NED has a gap, SRTM tile is available to query.
```

**Trade-off**: More storage, but complete coverage.

## Storage vs Coverage

### Conservative (Primary Only)

```yaml
ELEVATION_DATASET_TIERS: 'primary'
```

- Storage: 10-50 GB per region
- Coverage: ~99%
- Risk: Some NODATA gaps

### Balanced (Primary + Secondary)

```yaml
ELEVATION_DATASET_TIERS: 'primary+secondary'
```

- Storage: 20-100 GB per region
- Coverage: ~99.9%
- Risk: Rare gaps in extreme terrain

### Maximum (All Datasets)

```yaml
ELEVATION_DATASET_TIERS: 'primary+secondary+tertiary'
```

- Storage: 30-150 GB per region
- Coverage: ~99.99%
- Risk: Minimal (only unmapped areas)

## Global Coverage Map

### Well Covered

| Region | Primary Dataset | Quality |
|--------|-----------------|---------|
| USA (lower 48) | NED 10m | Excellent |
| Europe | SRTM 30m | Very Good |
| Japan | SRTM 30m | Very Good |
| Australia | SRTM 30m | Very Good |

### Moderate Coverage

| Region | Primary Dataset | Notes |
|--------|-----------------|-------|
| Canada South | SRTM 30m | Some gaps in remote areas |
| South America | SRTM 30m | Amazon basin has gaps |
| Africa | SRTM 30m | Desert areas may have voids |

### Challenging Regions

| Region | Solution | Notes |
|--------|----------|-------|
| Alaska | ArcticDEM | Large tiles, good coverage |
| Northern Canada | ArcticDEM | Some remote gaps |
| Greenland | ArcticDEM | Ice sheet edges challenging |
| Antarctica | REMA | Research stations well covered |

## Handling NODATA

### What Causes NODATA?

- **Water bodies**: Lakes, reservoirs (intentional)
- **Radar shadows**: Steep terrain blocked radar
- **Snow/ice**: Reflected radar incorrectly
- **Restricted areas**: Military, sensitive zones

### How We Handle It

1. **Fallback datasets**: Query next dataset in chain
2. **Interpolation**: Some datasets interpolate internally
3. **Skip**: If all datasets fail, mark coordinate as error

### Error Reporting

Analysis output includes error rate:

```
Processing: 100%|████████████████████| (elev_err: 0.3%)
```

Error file lists specific failures:

```
Vermont_errors_all_basic_2025-12-17.txt
```

## Optimizing for Your Use Case

### Local/Regional Analysis

Download only needed region:

```bash
./climb-analyzer -D -r "Vermont"
# Downloads: ~150 MB OSM + ~2 GB elevation
```

### Continental Analysis

Pre-download entire continent:

```bash
# Download USA elevation data
./climb-analyzer -D -r "USA"
# Downloads: ~350 GB NED + ~20 GB SRTM backup
```

### Global Analysis

For global coverage, expect:

- OSM: ~100 GB (all countries)
- Elevation: ~500 GB (all datasets)

## Recommendations

### For Most Users

Use default settings:

```yaml
ELEVATION_DATASET_TIERS: 'primary+secondary+tertiary'
```

Download data per-region as needed.

### For Limited Storage

Use primary only:

```yaml
ELEVATION_DATASET_TIERS: 'primary'
```

Accept ~1% elevation gaps.

### For Cloud Mode

Limited to available datasets:

```yaml
DEPLOYMENT_TYPE: 'cloud'
```

Uses api.opentopodata.org (SRTM, ASTER only).

---

Next: [GitHub App Setup](../advanced/github-app-setup.md)
