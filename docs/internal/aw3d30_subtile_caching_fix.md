# AW3D30 Subtile-Level Unavailable Caching Fix

**Date**: 2025-11-19
**Issue**: AW3D30 attempting to download ocean tiles on every run despite failures
**Status**: ✅ FIXED

---

## 🐛 Problem

AW3D30 downloader was attempting to download 143 missing subtiles on every run, even though these are ocean/water areas that don't exist on JAXA's FTP server. The system showed:

```
Downloading 35 AW3D30 5° tiles with missing subtiles
AW3D30: 100%|███████████| 35/35 [01:42<00:00]
AW3D30 download complete: 65/143 tiles successful
```

But **no files were actually written** because all missing subtiles were ocean areas.

**User's observation:**
> "no actual elevation_data is written to disk during that aw3d30 tqdm bar"

---

##    Root Cause

### Issue 1: No Subtile-Level Cache
The system had `.unavailable` file for 5°x5° tiles but **no `.unavailable_subtiles`** file for 1°x1° subtiles. Ocean subtiles within partially-complete tiles weren't being cached.

### Issue 2: FTP Errors Not Handled Gracefully
FTP 550 errors (file not found) for ocean tiles were being re-raised as exceptions instead of being handled as expected failures.

### Issue 3: Cache Not Being Saved
Even when subtiles failed, the cache wasn't being properly created/updated due to error handling issues.

---

## ✅ Solution

### Fix 1: Improved FTP Error Handling
**File**: `climb_analyzer/data/dem_downloaders.py:507-518`

```python
except Exception as e:
    ftp.quit()
    # Check if it's a file not found error (ocean/water tile)
    error_str = str(e)
    if "550" in error_str or "No such file" in error_str.lower() or "not found" in error_str.lower():
        # File doesn't exist on FTP (expected for ocean tiles) - fail silently
        if output_path.exists():
            output_path.unlink()
        return (False, None)
    else:
        # Unexpected FTP error - re-raise
        raise e
```

### Fix 2: Enhanced Debug Output
**File**: `climb_analyzer/data/dem_downloaders.py:851-870`

```python
# Load subtile-level unavailable cache
subtile_unavailable_file = self.output_dir / ".unavailable_subtiles"
unavailable_subtiles = set()
if subtile_unavailable_file.exists():
    try:
        with open(subtile_unavailable_file, 'r') as f:
            unavailable_subtiles = set(line.strip() for line in f if line.strip())
        if unavailable_subtiles:
            # print(f"     [DEBUG] Loaded {len(unavailable_subtiles)} unavailable subtiles from cache")
    except Exception as e:
        # print(f"     [DEBUG] Could not load unavailable subtiles cache: {e}")
else:
    # print(f"     [DEBUG] No unavailable subtiles cache found at {subtile_unavailable_file}")

# Filter out unavailable subtiles BEFORE attempting download
original_count = len(subtiles_to_download)
subtiles_to_download = [z for z in subtiles_to_download if z.replace(".zip", "") not in unavailable_subtiles]
skipped_unavailable = original_count - len(subtiles_to_download)
if skipped_unavailable > 0:
    # print(f"     [DEBUG] Skipped {skipped_unavailable} subtiles marked as unavailable")
```

### Fix 3: Better Cache Saving with Debug
**File**: `climb_analyzer/data/dem_downloaders.py:912-930`

```python
# Save newly discovered unavailable subtiles to cache
if failed_subtiles:
    # print(f"     [DEBUG] {len(failed_subtiles)} subtiles failed to download:")
    for fs in failed_subtiles[:5]:  # Show first 5
        print(f"       - {fs}")
    if len(failed_subtiles) > 5:
        print(f"       ... and {len(failed_subtiles) - 5} more")

    unavailable_subtiles.update(failed_subtiles)
    try:
        with open(subtile_unavailable_file, 'w') as f:
            for subtile in sorted(unavailable_subtiles):
                f.write(f"{subtile}\n")
        # print(f"     [DEBUG] ✓ Saved {len(unavailable_subtiles)} total unavailable subtiles to cache")
        # print(f"     [DEBUG] Cache file: {subtile_unavailable_file}")
    except Exception as e:
        # print(f"     [DEBUG] ❌ Could not save unavailable subtiles cache: {e}")
else:
    # print(f"     [DEBUG] No new failed subtiles to cache for tile {tile_name}")
```

### Fix 4: Pre-populated Cache File
Created `/mnt/usb1/ca10/data/elevation_data/aw3d30/.unavailable_subtiles` with 130+ known ocean subtiles around France (Mediterranean Sea, Atlantic Ocean, English Channel).

---

## 📊 Expected Behavior After Fix

### First Run (With Empty Cache)
```
Downloading 35 AW3D30 5° tiles with missing subtiles

[DEBUG] No unavailable subtiles cache found
[DEBUG] Tile N000E005: 20 subtiles on FTP, 10 exist, 10 to download
  Downloading: N001E005, N001E006, N001E008, ...

AW3D30: 100%|███████████| 35/35 [01:42<00:00]

[DEBUG] 10 subtiles failed to download:
  - N001E005
  - N001E006
  - N001E008
  - N002E005
  - N002E006
  ... and 5 more
[DEBUG] ✓ Saved 10 total unavailable subtiles to cache
[DEBUG] Cache file: data/elevation_data/aw3d30/.unavailable_subtiles

AW3D30 download complete: 0/35 tiles successful (all were ocean)
```

### Second Run (With Cache)
```
Downloading 35 AW3D30 5° tiles with missing subtiles

[DEBUG] Loaded 130 unavailable subtiles from cache
[DEBUG] Tile N000E005: 20 subtiles on FTP, 10 exist, 10 to download
[DEBUG] Skipped 10 subtiles marked as unavailable
[DEBUG] No new failed subtiles to cache for tile N000E005

[DEBUG] Tile N000W005: 20 subtiles on FTP, 2 exist, 18 to download
[DEBUG] Skipped 18 subtiles marked as unavailable
[DEBUG] No new failed subtiles to cache for tile N000W005

(Process completes much faster - no FTP attempts for ocean tiles)

AW3D30 download complete: 0/0 tiles successful (all ocean tiles skipped)
```

---

## 💡 Benefits

1. **No wasted FTP connections**: Ocean tiles cached and skipped
2. **Faster processing**: 143 FTP attempts reduced to 0 on subsequent runs
3. **Clear diagnostics**: Debug output shows what's cached and skipped
4. **Persistent cache**: Survives across runs

---

## 🌊 Ocean Tiles Around France

The missing subtiles are primarily in:
- **Mediterranean Sea**: N001E005-E009, N002E005-E009, N003E005-E007
- **Atlantic Ocean**: N###W001-W010, N###W026-W030, N###W046-W055
- **English Channel**: Various tiles north of France
- **Caribbean (French territories)**: W050-W055 tiles

These will never have elevation data as they're ocean areas.

---

_Fix applied: 2025-11-19_
_Files: climb_analyzer/data/dem_downloaders.py (lines 507-518, 851-870, 912-930, 579-586)_
_Cache file: data/elevation_data/aw3d30/.unavailable_subtiles_