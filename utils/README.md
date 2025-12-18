# Utilities

This directory contains reusable utility scripts for maintaining and managing the Climb Analyzer application.

## Contents

### Data Management
- **update_geo_definitions.py** - Update geographic boundaries from Natural Earth data
- **opentopodata_manager.py** - Manage OpenTopoData server operations
- **download_country_dem.py** - Download country-specific DEM data
- **country_dem_config.py** - Country-specific DEM configuration

### OSM Data Utilities
- **cli_download_osm.py** - CLI tool for downloading OSM data
- **osm_merger.py** - Merge multiple OSM files
- **cli_build_index.py** - CLI tool for building spatial indexes
- **build_hawaii_index.py** - Build spatial index specifically for Hawaii

### Elevation Data Utilities
- **cli_download_elevation.py** - CLI tool for downloading elevation data
- **prepare_elevation_data.py** - Prepare elevation data for use
- **elevation_stats_collector.py** - Collect elevation statistics
- **tile_validator.py** - Validate elevation tile integrity
- **way_elevation_tracker.py** - Track elevation for OSM ways

**Note**: Tile reconciliation has been removed to preserve all datasets for complete coverage and fallback support.

### Data Validation & Coverage
- **check_import_location.py** - Check import locations
- **check_missing_countries.py** - Identify missing countries
- **get_missing_country_bounds.py** - Retrieve missing country boundaries
- **data_coverage_checker.py** - Validate data coverage

### Debugging & Inspection
- **inspect_checkpoint.py** - Inspect checkpoint files
- **error_logger.py** - Error logging utilities

### Data Cleanup & Fixes
- **restore_ned_originals.py** - Restore original NED tiles
- **strip_usgs_prefix.py** - Strip USGS prefix from files

## Usage

These utilities are designed to be run independently for maintenance tasks:

### Updating Geographic Definitions
```bash
python utils/update_geo_definitions.py
```

### Managing OpenTopoData Server
```bash
python utils/opentopodata_manager.py
```

### Downloading OSM Data
```bash
python utils/cli_download_osm.py --region Vermont
```

### Downloading Elevation Data
```bash
python utils/cli_download_elevation.py --region Vermont
```

### Building Spatial Index
```bash
python utils/cli_build_index.py --input vermont.osm.pbf
```

### Inspecting Checkpoints
```bash
python utils/inspect_checkpoint.py --file checkpoints/analysis_123.json
```

### Validating Tiles
```bash
python utils/tile_validator.py --directory elevation_data/
```

## Note

Most of these utilities are standalone and can be run independently. Some may require configuration files or environment setup. Check each script's docstring or help text for specific usage instructions.

## Integration with Main Application

While these are utility scripts, many are also imported and used by the main application. Moving them to `/utils` helps organize the codebase while keeping them accessible for both standalone use and programmatic import.
