# ArcticDEM/REMA VRT Management Guide

This guide explains how to manage the VRT (Virtual Raster) files for ArcticDEM and REMA elevation datasets used by OpenTopoData.

## Overview

### What is a VRT?

A VRT (Virtual Raster) is a GDAL format that creates a virtual mosaic from multiple raster tiles. OpenTopoData treats VRT files as a single-file dataset, which is ideal for ArcticDEM/REMA tiles that don't align to a regular grid.

### Directory Structure

```
elevation_data/
├── arctic32m/                  # ArcticDEM tiles
│   └── .tiles/                 # Hidden folder with actual tiles
│       ├── S2304416_E1102112.tif
│       ├── S2304416_E1202304.tif
│       └── ...
└── arctic32m-vrt/              # VRT folder
    └── arctic32m.vrt           # VRT file (references ../arctic32m/.tiles/*.tif)

elevation_data/
├── rema/                       # REMA tiles (Antarctica)
│   └── .tiles/                 # Hidden folder with actual tiles
│       └── ...
└── rema-vrt/                   # VRT folder
    └── rema32m.vrt             # VRT file
```

**Key Points:**
- Tiles are stored in a **hidden subdirectory** (`.tiles/`)
- VRT file is in a **separate folder** containing ONLY the VRT
- VRT uses **relative paths** to reference tiles
- This structure allows OpenTopoData to auto-detect the VRT as a single-file dataset

## Using the Management Script

The `manage_arctic_vrt.py` script provides all functionality needed to maintain VRT files.

### 1. Check Current Status

```bash
python3 scripts/manage_arctic_vrt.py --dataset arctic32m --list
```

This shows:
- Number of tiles in VRT vs on disk
- Any discrepancies (tiles missing from VRT or disk)
- Sample of tile names

**Example output:**
```
======================================================================
TILE STATUS: arctic32m
======================================================================
VRT file: elevation_data/arctic32m-vrt/arctic32m.vrt
Tiles directory: elevation_data/arctic32m/.tiles

Tiles in VRT:     17
Tiles on disk:    17

✅ VRT and disk tiles are in sync

Sample tiles on disk (first 10):
   S2304416_E1102112.tif
   S2304416_E1202304.tif
   ...
======================================================================
```

### 2. Download Tiles for a New Region

When you need to analyze a new Arctic or Antarctic region, download tiles for that area:

```bash
# ArcticDEM - Alaska example
python3 scripts/manage_arctic_vrt.py \
    --dataset arctic32m \
    --download \
    --bbox 60,-152,65,-141

# REMA - Antarctic Peninsula example
python3 scripts/manage_arctic_vrt.py \
    --dataset rema32m \
    --download \
    --bbox -70,-70,-63,-55
```

**Bounding box format:** `min_lat,min_lon,max_lat,max_lon`

**Tips:**
- Use generous bounding boxes to ensure coverage
- For countries, use approximate bounds:
  - Iceland: `63,-25,67,-13`
  - Alaska: `51,-180,72,-130`
  - Greenland: `59,-75,84,-10`
- The script will:
  1. Download only new tiles (skips existing)
  2. Automatically rebuild the VRT after download
  3. Verify the VRT is valid

### 3. Rebuild VRT

If you manually add tiles or the VRT gets out of sync:

```bash
python3 scripts/manage_arctic_vrt.py --dataset arctic32m --rebuild
```

This will:
- Find all `.tif` files in the tiles directory
- Regenerate the VRT to include all tiles
- Use relative paths for portability
- Verify the VRT is valid

### 4. Verify VRT

Check if the VRT file is valid and readable by GDAL:

```bash
python3 scripts/manage_arctic_vrt.py --dataset arctic32m --verify
```

### 5. Combined Operations

Download tiles for a new region and rebuild:

```bash
python3 scripts/manage_arctic_vrt.py \
    --dataset arctic32m \
    --download \
    --bbox 60,-152,65,-141 \
    --rebuild
```

The `--rebuild` is optional after download because the script automatically rebuilds if new tiles were downloaded.

## Common Workflows

### Workflow 1: Adding Support for a New Country

**Example: Adding support for Alaska**

1. **Check current coverage:**
   ```bash
   python3 scripts/manage_arctic_vrt.py --dataset arctic32m --list
   ```

2. **Download tiles for Alaska:**
   ```bash
   python3 scripts/manage_arctic_vrt.py \
       --dataset arctic32m \
       --download \
       --bbox 51,-180,72,-130
   ```

3. **Verify the download:**
   ```bash
   python3 scripts/manage_arctic_vrt.py --dataset arctic32m --list
   ```

   You should see the tile count increase.

4. **The VRT is automatically rebuilt and verified!**

5. **Restart OpenTopoData server** (on remote server):
   ```bash
   ssh sholl@10.0.0.101
   docker restart opentopodata-server
   ```

6. **Test elevation fetching:**
   ```bash
   curl 'http://localhost:5000/v1/arctic32m?locations=61.2176,-149.8997'
   # Should return elevation for Anchorage, Alaska
   ```

### Workflow 2: Manual VRT Rebuild

If you manually copied tiles or suspect the VRT is corrupted:

```bash
# Check status
python3 scripts/manage_arctic_vrt.py --dataset arctic32m --list

# Rebuild VRT from all tiles
python3 scripts/manage_arctic_vrt.py --dataset arctic32m --rebuild

# Verify it worked
python3 scripts/manage_arctic_vrt.py --dataset arctic32m --verify
```

### Workflow 3: Re-downloading Tiles

If tiles are corrupted or you want fresh data:

```bash
python3 scripts/manage_arctic_vrt.py \
    --dataset arctic32m \
    --download \
    --bbox 63,-25,67,-13 \
    --force
```

The `--force` flag will re-download even if tiles already exist.

## Tile Naming Convention

ArcticDEM and REMA tiles are named using their lower-left corner coordinates in the projection coordinate system:

**Format:** `S######_E######.tif` or `N######_W######.tif`

**Examples:**
- `S2504800_E1102112.tif` - Lower-left at (1102112, -2504800) in EPSG:3413
- `N2705184_W1001920.tif` - Lower-left at (-1001920, 2705184) in EPSG:3413

**Coordinate breakdown:**
- First letter: `N` (north) or `S` (south) for Y-axis
- First number: Absolute Y coordinate in meters
- Second letter: `E` (east) or `W` (west) for X-axis
- Second number: Absolute X coordinate in meters

## OpenTopoData Configuration

The VRT datasets are configured in `opentopodata/config.yaml`:

```yaml
datasets:
  - name: arctic32m
    path: data/arctic32m-vrt/

  # When REMA is set up:
  - name: rema32m
    path: data/rema-vrt/
```

**Important:**
- `path` points to the VRT **folder**, not the VRT file
- OpenTopoData auto-detects the single VRT file in the folder
- No `filename_epsg` or `filename_tile_size` needed for VRT datasets

## Climb Analyzer Configuration

Dataset priorities are configured in `climb_analyzer.py`:

```python
DATASET_PRIORITY_BY_REGION = {
    "Iceland": ["arctic32m", "aw3d30", "aster30m"],
    "Alaska": ["arctic32m", "aw3d30", "aster30m"],
    "Canada (>60°N)": ["arctic32m", "aw3d30", "aster30m"],
    "Greenland": ["arctic32m", "aw3d30"],
    "Russia (>60°N)": ["arctic32m", "aw3d30", "aster30m"],
    "Antarctica": ["rema32m", "aw3d30"],
}
```

The first dataset in the list is tried first, falling back to subsequent datasets if unavailable.

## Troubleshooting

### "No tiles found"

**Problem:** VRT rebuild fails with "No tiles found in elevation_data/arctic32m/.tiles"

**Solution:**
1. Check tiles directory exists:
   ```bash
   ls -la elevation_data/arctic32m/.tiles/
   ```

2. If empty, download tiles:
   ```bash
   python3 scripts/manage_arctic_vrt.py --dataset arctic32m --download --bbox ...
   ```

### "VRT and disk tiles out of sync"

**Problem:** Status shows tiles on disk but not in VRT (or vice versa)

**Solution:**
```bash
python3 scripts/manage_arctic_vrt.py --dataset arctic32m --rebuild
```

### OpenTopoData returns "Dataset not configured"

**Problem:** API returns HTTP 400 with "Dataset 'arctic32m' not in server config"

**Solution:**
1. Check `opentopodata/config.yaml` includes the dataset
2. Restart OpenTopoData server:
   ```bash
   docker restart opentopodata-server
   ```

### OpenTopoData returns "elevation: null"

**Problem:** API returns OK status but elevation is null

**Possible causes:**
1. **Coordinate outside tile coverage:**
   - Check if coordinates fall within downloaded area
   - Download additional tiles if needed

2. **VRT file corruption:**
   ```bash
   python3 scripts/manage_arctic_vrt.py --dataset arctic32m --rebuild --verify
   ```

3. **Permission issues:**
   ```bash
   # On remote server
   ssh sholl@10.0.0.101
   ls -la /mnt/usb1/ca9/elevation_data/arctic32m-vrt/
   ls -la /mnt/usb1/ca9/elevation_data/arctic32m/.tiles/
   ```

### "gdalbuildvrt not found"

**Problem:** VRT rebuild fails because GDAL is not installed

**Solution:**
```bash
# macOS
brew install gdal

# Ubuntu/Debian
sudo apt-get install gdal-bin

# Check installation
gdalbuildvrt --version
```

## Advanced Usage

### Custom Base Directory

If running from a different location:

```bash
python3 scripts/manage_arctic_vrt.py \
    --dataset arctic32m \
    --base-dir /path/to/ca9 \
    --list
```

### Programmatic Usage

You can import and use the manager in Python scripts:

```python
from pathlib import Path
from scripts.manage_arctic_vrt import ArcticVRTManager

# Create manager
manager = ArcticVRTManager('arctic32m', base_dir=Path('/Volumes/usb1-drive/ca9'))

# List tiles
vrt_tiles = manager.list_tiles_in_vrt()
disk_tiles = manager.list_tiles_on_disk()
print(f"VRT has {len(vrt_tiles)} tiles, disk has {len(disk_tiles)} tiles")

# Download tiles
bbox = (63, -25, 67, -13)  # Iceland
success, new_files = manager.download_tiles(bbox)

# Rebuild VRT
if new_files > 0:
    manager.rebuild_vrt()
    manager.verify_vrt()
```

## Performance Considerations

### VRT Performance

According to OpenTopoData documentation and testing:
- **Small datasets** (<20 tiles): VRT performance is excellent
- **Medium datasets** (20-100 tiles): VRT performance is acceptable
- **Large datasets** (100+ tiles): VRT can become slow for batch queries

For our use case (Iceland = 17 tiles), VRT performance is optimal.

### Optimization Tips

1. **Only download needed tiles:** Don't download the entire Arctic unnecessarily
2. **Use appropriate bounding boxes:** Add ~0.5° buffer around regions of interest
3. **Monitor disk usage:** Each tile is 2-30 MB depending on terrain complexity
4. **Regular verification:** Run `--verify` periodically to ensure VRT integrity

## File Sizes

**Typical tile sizes:**
- Flat terrain (ocean, ice sheet): 2-5 MB
- Moderate terrain (hills): 10-15 MB
- Complex terrain (mountains): 20-30 MB

**Total space for Iceland (17 tiles):** ~250 MB

**Total space for Alaska (estimated ~150 tiles):** ~2.5 GB

## Related Documentation

- [OpenTopoData VRT Documentation](https://www.opentopodata.org/notes/buffering-tiles/)
- [GDAL VRT Format](https://gdal.org/drivers/raster/vrt.html)
- [ArcticDEM Download Documentation](../dem_downloaders.py)
- [OpenTopoData Configuration](../opentopodata/config.yaml)

## Version History

- **v1.0** (2025-01-23): Initial implementation with Arctic32m support for Iceland
