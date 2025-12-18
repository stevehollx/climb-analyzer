# Scripts Directory

This directory contains standalone scripts that can be run independently of the main climb analyzer application.

## Production Scripts

These scripts are production-ready and can be used by end users.

### merge_cross_region_climbs.py

Merge climbs that span across regional boundaries. This script identifies climbs that were split at regional borders and combines their metrics.

**Container-Aware Wrapper:**

This script automatically detects whether it's running inside or outside the Docker container:

- ✅ **Outside container**: Automatically re-executes via `docker exec` (no Python dependencies needed on host!)
- ✅ **Inside container**: Runs directly with full access to dependencies

**Usage:**

```bash
# From host machine (automatic container wrapper)
python scripts/merge_cross_region_climbs.py region1.xlsx region2.xlsx

# The script will automatically:
# 1. Detect it's running outside the container
# 2. Check if climb-analyzer container is running
# 3. Re-execute itself inside the container via docker exec
# 4. Return results to your terminal

# You can also run it manually inside container:
docker exec climb-analyzer python /app/scripts/merge_cross_region_climbs.py region1.xlsx region2.xlsx
```

**Requirements:**
- Docker container must be running: `docker-compose up -d`
- Files must be in mounted volume (e.g., `output/` directory)
- NO Python dependencies needed on host machine!

**Dependencies (containerized):**
- pandas >= 2.0.0
- openpyxl >= 3.1.0
- climb_analyzer modules (formatting utilities)

All dependencies are pre-installed in the Docker container.

**What it does:**
1. Loads climb data from two region files
2. Identifies climbs with the same name near regional boundaries
3. Merges segments with combined metrics (total length, elevation gain)
4. Filters out duplicates (same length = complete climb captured in both regions)
5. Creates backup files before making changes
6. Updates both input files with merged data

**Use cases:**
- After analyzing France North + France South subregions
- When analyzing adjacent US states (e.g., Vermont + New Hampshire)
- Any multi-region analysis where climbs may cross boundaries

**Output:**
- Updated XLSX files with merged climbs
- Backup files (.backup.xlsx) of original data
- Console summary of merges performed

---

### manage_arctic_vrt.py

Manage Arctic DEM virtual raster files for elevation data.

**Usage:**

```bash
# Create VRT for Arctic region
python scripts/manage_arctic_vrt.py create --region arctic --output elevation_data/arctic.vrt

# Validate existing VRT
python scripts/manage_arctic_vrt.py validate --vrt elevation_data/arctic.vrt

# List available tiles
python scripts/manage_arctic_vrt.py list --region arctic
```

**Dependencies:**
- GDAL/OGR tools
- Python gdal bindings

**What it does:**
- Creates virtual raster (VRT) files that combine multiple Arctic DEM tiles
- Validates VRT structure and tile availability
- Lists available tiles for Arctic regions
- Optimizes elevation data access for polar regions

**Use cases:**
- Setting up elevation data for Alaska, Greenland, Northern Canada
- Managing large Arctic DEM datasets efficiently
- Optimizing elevation queries for high-latitude regions

---

## Automated Usage

Both scripts can be invoked automatically by the climb analyzer:

- **merge_cross_region_climbs.py**: Called automatically after batch analysis with split regions
  - See `offer_automatic_merge()` in climb_analyzer.py
  - User prompted to merge after subregion analysis completes

- **manage_arctic_vrt.py**: Called during elevation data setup
  - Automatically configured for polar regions
  - Part of the data preparation workflow

---

## Troubleshooting Scripts

If you have test/troubleshooting scripts, place them in `scripts/troubleshooting/` directory.

These scripts are for development and debugging only, not production use:
- One-time verification scripts
- Issue-specific debugging scripts
- Performance testing scripts

---

## Requirements File

For standalone usage of these scripts outside the Docker container, install dependencies:

```bash
# Install all script dependencies
pip install -r scripts/requirements.txt
```

---

## Contributing

When adding new scripts:

1. **Production scripts** go in `scripts/` root
   - Must be well-documented
   - Should have clear usage examples
   - Must handle errors gracefully
   - Should work both standalone and automated

2. **Development scripts** go in `scripts/troubleshooting/`
   - Can be less polished
   - Document what issue they were created for
   - Mark when they can be deleted

3. **All scripts should**:
   - Include usage documentation
   - List dependencies clearly
   - Have `#!/usr/bin/env python3` shebang
   - Be executable: `chmod +x script.py`
   - Include error handling
   - Validate inputs

---

## Common Issues

### "ModuleNotFoundError: No module named 'pandas'"

**Solution**: Install required dependencies
```bash
pip install pandas openpyxl
```

### "Permission denied" when running scripts

**Solution**: Make script executable
```bash
chmod +x scripts/merge_cross_region_climbs.py
```

### Script works in Docker but not standalone

**Possible causes**:
- Missing Python dependencies (install via pip)
- Missing system dependencies (install via apt/brew)
- Environment differences (check Python version, PATH)

**Solution**: Check script's dependency list and install all requirements

---

## Script Maintenance

### When to remove a script:
- One-time migration completed
- Issue it was created for is resolved
- Functionality moved into main application
- No longer compatible with current architecture

### When to keep a script:
- Used regularly by users or automation
- Provides standalone utility value
- Part of documented workflows
- Required for data management

---

## Contact

For issues with scripts, please open a GitHub issue or check the main project documentation.
