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

The `index.json` file contains:

```json
{
  "version": "1.0.0",
  "generated_at": "2025-11-01T12:00:00Z",
  "repository": "stevehollx/global-road-and-trail-climbs",
  "summary": {
    "total_regions": 5,
    "total_files": 9,
    "regions_with_split_files": 3,
    "by_continent": {
      "europe": 2,
      "asia": 1,
      "north-america": 1,
      "oceania": 1
    }
  },
  "regions": {
    "europe/luxembourg": {
      "region_name": "Luxembourg",
      "files": ["Luxembourg_climbs_all_basic_2025-11-01_v2.0.0_e0000.xlsx"],
      "download_urls": [
        "https://raw.githubusercontent.com/stevehollx/global-road-and-trail-climbs/main/europe/luxembourg/Luxembourg_climbs_all_basic_2025-11-01_v2.0.0_e0000.xlsx"
      ],
      "file_count": 1,
      "has_split_files": false
    },
    "europe/belgium": {
      "region_name": "Belgium",
      "files": [
        "Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-1.xlsx",
        "Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-2.xlsx",
        "Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-3.xlsx"
      ],
      "download_urls": [
        "https://raw.githubusercontent.com/stevehollx/global-road-and-trail-climbs/main/europe/belgium/Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-1.xlsx",
        "https://raw.githubusercontent.com/stevehollx/global-road-and-trail-climbs/main/europe/belgium/Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-2.xlsx",
        "https://raw.githubusercontent.com/stevehollx/global-road-and-trail-climbs/main/europe/belgium/Belgium_climbs_all_basic_2025-11-01_v2.0.0_e0000-3.xlsx"
      ],
      "file_count": 3,
      "has_split_files": true
    }
  }
}
```

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
luxembourg_data = index_data["regions"]["europe/luxembourg"]
for url in luxembourg_data["download_urls"]:
    print(f"Downloading: {url}")
    # Download the file...
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
