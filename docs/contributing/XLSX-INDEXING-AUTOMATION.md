# XLSX Indexing Automation

This document describes the automated XLSX file indexing system that runs when Pull Requests are approved.

## Overview

The automation scans the repository for XLSX climb data files and generates an `index.json` file containing region paths and download URLs for all files. This index enables programmatic discovery and download of climb data.

## How It Works

### 1. Trigger Events

The indexing workflow triggers on:
- **PR Approval**: When a reviewer approves a Pull Request
- **PR Merge**: When a Pull Request is merged into the main branch
- **Manual Trigger**: Can be run manually from the Actions tab for testing

### 2. File Detection

The indexer scans all directories for XLSX files with the naming pattern:
```
{RegionName}_climbs_{type}_{date}_v{version}_e{elevation}.xlsx
```

Examples:
- `Luxembourg_climbs_all_basic_2025-11-01_v2.0.0_e0000.xlsx`
- `United-States_climbs_all_basic_2025-11-01_v2.0.0_e0000.xlsx`

### 3. Split File Handling

Large datasets may be split into multiple files with numeric suffixes:
- `Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-1.xlsx`
- `Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-2.xlsx`
- `Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-3.xlsx`

The indexer automatically:
- Detects split files by the `-N` suffix pattern
- Groups all parts of a split dataset together
- Orders them correctly in the index

### 4. Region Name Extraction

The region name is extracted from the filename as the text before `_climbs`:
- `Luxembourg_climbs_...` → Region: `Luxembourg`
- `United-States_climbs_...` → Region: `United-States`
- `New-Zealand_climbs_...` → Region: `New-Zealand`

## Generated Index Structure

The `index.json` file contains metadata for all published climb data files:

```json
{
  "version": "2.0.0",
  "generated_at": "2026-01-03T18:07:06Z",
  "repository": "stevehollx/global-road-and-trail-climbs",
  "summary": {
    "total_regions": 5,
    "total_xlsx_files": 9,
    "total_sqlite_files": 5,
    "total_xlsx_size_bytes": 123456789,
    "total_sqlite_size_bytes": 234567890,
    "total_size_mb": 342.5,
    "total_climbs": 1234567
  },
  "regions": {
    "north-america/united-states-of-america/hawaii": {
      "region_name": "Hawaii",
      "version": "2.3.0",
      "release_tag": "hawaii-v2.3.0",
      "release_url": "https://github.com/stevehollx/global-road-and-trail-climbs/releases/tag/hawaii-v2.3.0",
      "climb_count": 83237,
      "elevation_errors": 0,
      "files": ["Hawaii_climbs_all-surfaces_all-access_imperial_2026-01-03_v2.3.0_e0000.xlsx"],
      "download_urls": [
        "https://github.com/stevehollx/global-road-and-trail-climbs/releases/download/hawaii-v2.3.0/Hawaii_climbs_...xlsx"
      ],
      "file_sizes": [15166008],
      "file_count": 1,
      "has_split_files": false,
      "database_file": "Hawaii_climbs_all-surfaces_all-access_imperial_2026-01-03_v2.3.0_e0000.sqlite",
      "database_size": 33030144,
      "database_url": "https://github.com/stevehollx/global-road-and-trail-climbs/releases/download/hawaii-v2.3.0/Hawaii_climbs_...sqlite",
      "published_at": "2026-01-03T18:04:19Z"
    },
    "europe/belgium": {
      "region_name": "Belgium",
      "version": "2.0.0",
      "files": [
        "Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-1.xlsx",
        "Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-2.xlsx"
      ],
      "file_sizes": [50000000, 45000000],
      "file_count": 2,
      "has_split_files": true,
      "database_file": "Belgium_climbs_...sqlite",
      "database_size": 120000000
    }
  }
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `region_name` | Human-readable region name |
| `files` | List of XLSX filenames |
| `download_urls` | Direct download URLs for XLSX files |
| `file_sizes` | Size in bytes for each XLSX file |
| `database_file` | SQLite database filename |
| `database_size` | SQLite file size in bytes |
| `database_url` | Direct download URL for SQLite file |
| `climb_count` | Total number of climbs in this region |

## Database Files

Each region includes a SQLite database alongside the XLSX files. The SQLite format is optimized for the iOS companion app with:

- **UUID primary keys** - Stable identifiers across data updates
- **Geohash columns** - 6 precision levels (p1-p6) for efficient spatial queries
- **R-tree spatial index** - Fast bounding-box location lookups
- **19 optimized indexes** - For common filter and sort operations
- **Aggregated stats** - Per-file statistics in `file_stats` table

## File Locations

- **Workflow**: `.github/workflows/index-xlsx-files.yml`
- **Indexer Script**: `scripts/index_xlsx_files.py`
- **Test Script**: `scripts/test_indexer.py`
- **Output**: `index.json` (repository root)

## Testing

### Run Tests Locally

```bash
# Run the test suite
python scripts/test_indexer.py

# Run the indexer directly (for current files)
python scripts/index_xlsx_files.py
```

### Manual Workflow Trigger

1. Go to the [Actions tab](https://github.com/stevehollx/global-road-and-trail-climbs/actions)
2. Select "Index XLSX Files on PR Approval"
3. Click "Run workflow"
4. Select the branch and click "Run workflow"

## Using the Index

### Programmatic Access

```python
import json
import requests

# Fetch the index
index_url = "https://raw.githubusercontent.com/stevehollx/global-road-and-trail-climbs/main/index.json"
response = requests.get(index_url)
index_data = response.json()

# Get all files for a specific region
hawaii_data = index_data["regions"]["north-america/united-states-of-america/hawaii"]
for url in hawaii_data["download_urls"]:
    print(f"Downloading: {url}")
    # Download XLSX file...

# Download the SQLite database for iOS app
sqlite_url = hawaii_data["database_url"]
print(f"Database: {sqlite_url}")
```

### Finding Split Files

```python
# Check if a region has split files
for region_path, region_info in index_data["regions"].items():
    if region_info["has_split_files"]:
        print(f"{region_path} has {region_info['file_count']} split files")
```

## Troubleshooting

### Common Issues

1. **Index not updating after PR merge**
   - Check the Actions tab for workflow run status
   - Verify XLSX files follow the naming convention
   - Ensure files are in continent/country directories

2. **Files not appearing in index**
   - File must end with `.xlsx`
   - Filename must contain `_climbs_`
   - Must have a region name before `_climbs`

3. **Workflow permissions error**
   - Ensure GitHub Actions has write permissions
   - Check repository settings → Actions → Workflow permissions

### Debugging

View workflow logs:
1. Go to Actions tab
2. Click on the workflow run
3. Click on "index-files" job
4. Expand "Run indexer" step

## Contributing

When adding new XLSX files:
1. Place files in the appropriate `continent/country` directory
2. Follow the naming convention: `{Region}_climbs_*.xlsx`
3. For large datasets (>100MB), split into numbered parts
4. The index will update automatically when your PR is approved

## Support

For issues or questions:
- Check the [workflow runs](https://github.com/stevehollx/global-road-and-trail-climbs/actions)
- Review this documentation
- Open an issue if problems persistsholl@minipc:/mnt/usb1/global-road-and-trail-climbs/docs
