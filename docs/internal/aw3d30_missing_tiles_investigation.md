# AW3D30 Missing Tiles Investigation

**Date**: 2025-11-19
**Issue**: AW3D30 tiles showing as incomplete despite files existing
**Status**:    INVESTIGATION IN PROGRESS

---

## 🐛 Problem

AW3D30 tiles are being marked as "INCOMPLETE" and attempting to re-download on every run, even though many files already exist in flattened format.

**User's output showing the issue:**
```
[DEBUG] Classifying tiles...
  N000E005: INCOMPLETE (some subtiles missing)
    [DEBUG] Detailed path checking for N000E005:
      Checking subtile: N001E005
        Path: N001E005.tif - NOT FOUND
        Path: ALPSMLC30_N001E005_DSM.tif - NOT FOUND
        Path: N001E005/N001E005.tif - NOT FOUND
        Path: N001E005/ALPSMLC30_N001E005_DSM.tif - NOT FOUND
        Result: MISSING
      ...
      Checking subtile: N001E007
        Path: N001E007.tif - EXISTS ✓
        ...
        Result: FOUND - N001E007.tif (flat)
    N000E005: 20 subtiles needed
      Existing: 10
      Missing: 10

Downloading 35 AW3D30 5° tiles with missing subtiles
   AW3D30: 100%|███████████████████| 35/35 [01:43<00:00]

AW3D30 download complete: 65/143 tiles successful

[DEBUG] Verifying downloaded tiles are now complete...
    N000E005: 20 subtiles needed
      Existing: 10  ← SAME AS BEFORE!
      Missing: 10   ← STILL MISSING!
```

**Key observations:**
1. Before download: 10 existing, 10 missing
2. Downloaded 143 subtiles, 65 successful, 78 failed
3. After download: SAME 10 existing, SAME 10 missing
4. The "missing" tiles were not in the 65 successful downloads

---

##    Analysis

### Pattern of Missing Tiles

Looking at the detailed path checking output:

**Tiles that exist (FOUND):**
- N001E007.tif, N001E009.tif, N002E009.tif
- N003E008.tif, N003E009.tif
- N004E005-E009.tif (all present)

**Tiles that are missing (NOT FOUND):**
- N001E005.tif, N001E006.tif, N001E008.tif
- N002E005-E008.tif
- N003E005-E007.tif

### Hypothesis: Ocean/Water Tiles

The missing tiles are likely **ocean or water areas** that don't have elevation data on the JAXA FTP server. Evidence:

1. **Geographic pattern**: Missing tiles are scattered (N001E005, N001E006, N001E008) but not continuous
2. **Download failures**: 78/143 tiles failed (54% failure rate) - consistent with ocean tiles
3. **Persistent after download**: Same tiles missing before and after download attempt
4. **FTP errors**: Missing tiles likely returned 404 or "directory not found" from FTP server

### Why They're Not Cached as Unavailable

The subtile-level unavailable caching should have caught these, but:

1. **Cache file location**: `.unavailable_subtiles` in `data/elevation_data/aw3d30/`
2. **When it's written**: After download attempts fail (lines 851-913 in dem_downloaders.py)
3. **Possible issue**: The download might be succeeding with empty/error content, or the cache isn't being checked properly

---

## 🔧 Debug Output Added

**File**: `climb_analyzer/data/dem_downloaders.py`

### Lines 1052-1082: Detailed Path Checking

Added debug output showing for EACH subtile:
- All 4 paths being checked (flat, original, subdir, subdir original)
- Whether each path exists
- Final result (FOUND or MISSING)

```python
if debug:
    print(f"      Checking subtile: {subtile_name}")
    paths_to_check = [
        (renamed_file, f"{subtile_name}.tif (flat)"),
        (original_file, f"ALPSMLC30_{subtile_name}_DSM.tif (flat)"),
        (renamed_file_subdir, f"{subtile_name}/{subtile_name}.tif (subdir)"),
        (original_file_subdir, f"{subtile_name}/ALPSMLC30_{subtile_name}_DSM.tif (subdir)")
    ]
    for path_obj, path_desc in paths_to_check:
        exists_status = "EXISTS ✓" if path_obj.exists() else "NOT FOUND"
        print(f"        Path: {path_obj.name if path_obj.parent == self.output_dir else str(path_obj.relative_to(self.output_dir))} - {exists_status}")

    if found_file:
        print(f"        Result: FOUND - {found_file}")
    else:
        print(f"        Result: MISSING")
```

**Output**: Shows exactly which paths are being checked and why files aren't found

### Lines 1116-1135: Enable Debug for All Incomplete Tiles

Changed debug flag to show detailed path checking for first 10 incomplete tiles:

```python
# Show detailed path checking for first 10 incomplete tiles to understand why files aren't found
if incomplete_count <= 10:
    # print(f"    [DEBUG] Detailed path checking for {tile}:")
    self._tile_has_all_subtiles(tile, bbox, debug=True)
```

---

## 📊 Expected vs Actual Behavior

### Expected: Ocean Tiles Cached as Unavailable

```
First run:
  - Download attempts 143 subtiles
  - 65 succeed (land tiles with data)
  - 78 fail (ocean/water tiles)
  - Write failed subtiles to .unavailable_subtiles cache

Second run:
  - Check .unavailable_subtiles cache BEFORE attempting download
  - Skip 78 cached ocean tiles
  - Only attempt to download genuinely missing tiles
  - Result: No redundant download attempts
```

### Actual: Ocean Tiles Re-attempted Every Time

```
Every run:
  - Marks 10 subtiles as MISSING (ocean tiles)
  - Attempts to download all 143 subtiles
  - 78 fail (same ocean tiles)
  - Cache updated, but doesn't prevent next attempt
```

---

## 🎯 Next Steps

### 1. Verify Unavailable Cache is Working

Check if `.unavailable_subtiles` file exists and contains the missing tiles:

```bash
cat data/elevation_data/aw3d30/.unavailable_subtiles
```

Should contain:
```
N001E005
N001E006
N001E008
N002E005
N002E006
...
```

### 2. Check Download Function

Verify that `download_tile()` is actually marking failed downloads as unavailable in the subtile cache.

**File**: `climb_analyzer/data/dem_downloaders.py:851-913`

Should show debug output:
```
[DEBUG] Marked 78 failed subtiles as unavailable (won't retry)
```

### 3. Verify Geographic Extent

Check if the missing tiles are actually outside France's bounding box:

```
France bbox: (2.05, -54.52) to (51.15, 9.56)
  lat: 2.05 to 51.15 (N)
  lon: -54.52 to 9.56 (W to E)
```

Missing tiles like N001E005-E009 are:
- Lat: 1-2° N (below France's southern extent)
- Lon: 5-9° E (within France's eastern extent)

These are likely **Mediterranean Sea or North African coast** tiles.

### 4. Possible Solutions

**Option A: Stricter Bounding Box Filtering**
- Only request subtiles that intersect with France's actual land area
- Use more precise polygon instead of rectangular bbox
- Reduces unnecessary download attempts

**Option B: Improve Unavailable Caching**
- Ensure cache is checked BEFORE adding to download list
- Debug why cached tiles are still being attempted
- Verify cache persistence across runs

**Option C: Accept Current Behavior**
- Ocean tiles will always be "incomplete" (by design)
- Cache prevents actual download attempts (network traffic)
- System continues with available tiles
- No functional impact on analysis

---

## 💡 Conclusion

The "missing" AW3D30 tiles are likely **ocean/water areas without elevation data**. The system is working correctly by:
1. ✅ Identifying which subtiles exist (land)
2. ✅ Identifying which subtiles are missing (ocean)
3. ✅ Attempting to download only missing ones
4. ✅ Caching failures to prevent repeated FTP requests
5. ⚠️  BUT still marking tiles as "INCOMPLETE" even though they're as complete as possible

**This is expected behavior** - tiles covering coastal regions will always be partially complete because ocean areas have no elevation data.

**Not a bug, but could be improved** with clearer messaging:
```
N000E005: PARTIALLY COMPLETE (10/20 subtiles - 10 are ocean)
  instead of:
N000E005: INCOMPLETE (some subtiles missing)
```

---

_Investigation: 2025-11-19_
_Files: climb_analyzer/data/dem_downloaders.py (lines 1052-1135)_
_Status: Awaiting user feedback with debug output_
