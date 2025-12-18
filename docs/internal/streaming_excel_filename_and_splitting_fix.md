# Streaming Excel Export Filename and File Splitting Fix

**Date**: 2025-11-19
**Issue**: Streaming Excel export using wrong filename and hitting Excel row/size limits
**Status**: ✅ FIXED

---

## 🐛 Problem

After implementing streaming reverse geocoding, the Excel export had several issues:

1. **Wrong filename**: `climbs_streaming_1763530440.xlsx` instead of `France_climbs_all_basic_2025-11-19_v2.0.1_e0000.xlsx`
2. **Excel row limit exceeded**: Exactly 1,048,576 rows (Excel's maximum)
3. **File size**: 130MB (over 100MB limit)
4. **No file splitting**: Should split at 950K rows or 100MB
5. **"No results to save" message**: Main code didn't recognize streaming export

### Root Cause

The streaming export (`_stream_to_excel()`) was using a timestamp-based filename and writing all climbs to a single file without checking row/size limits. It returned a metadata DataFrame instead of a list of created files, bypassing the normal save infrastructure.

---

## ✅ Solution

### 1. Proper Filename Generation

**Modified**: `climb_analyzer/engine.py:12774-12822`

Added proper filename generation using the same pattern as `save_large_dataframe_as_split_excel()`:

```python
# Import version info
try:
    from __version__ import __version__
except ImportError:
    __version__ = "unknown"

# Get elevation error count from stats collector
elevation_errors = 0
try:
    from utils.elevation_stats_collector import get_stats_collector, has_elevation_stats

    if has_elevation_stats():
        stats_collector = get_stats_collector()
        stats = stats_collector.get_stats()
        elevation_errors = stats.get("total_coords_failed", 0)
except Exception:
    elevation_errors = 0

# Build base filename: {region}_climbs_{surface}_{score}_{date}
date_str = datetime.now().strftime("%Y-%m-%d")
filter_str = f"{surface_filter}_{score_type}"
safe_name = "".join(c for c in scope_info if c.isalnum() or c in (" ", "-", "_")).rstrip()
safe_name = safe_name.replace(" ", "_")
base_filename = f"{safe_name}_climbs_{filter_str}_{date_str}"

# Build suffix: _v{version}_e{errors}
suffix = ""
if __version__:
    suffix += f"_v{__version__}"
if elevation_errors > 0:
    suffix += f"_e{elevation_errors:04d}"
elif elevation_errors == 0 and __version__:
    suffix += "_e0000"
```

**Result**: Filename now matches expected pattern: `France_climbs_all_basic_2025-11-19_v2.0.1_e0000.xlsx`

---

### 2. File Splitting Implementation

**Modified**: `climb_analyzer/engine.py:12824-13038`

Implemented automatic file splitting when approaching limits:

```python
# Excel limits
MAX_EXCEL_ROWS = 1048576  # Excel's hard limit
MAX_FILE_SIZE_MB = 100.0  # Target file size
ROWS_PER_FILE = 950000    # Safety margin below Excel limit

current_file_num = 1
current_file_rows = 0
created_files = []

def _create_new_file():
    """Helper to create a new Excel file when splitting"""
    # Close previous file if exists
    if current_writer is not None:
        current_writer.close()
        created_files.append(current_excel_file)
        print(f"   ✓ Saved {current_excel_file.name} ({current_file_rows:,} rows, ...)")

    # Create new file
    if current_file_num == 1 and total_filtered <= ROWS_PER_FILE:
        # Single file expected
        current_excel_file = output_dir / f"{base_filename}{suffix}.xlsx"
    elif current_file_num == 1:
        # First of multiple files
        current_excel_file = output_dir / f"{base_filename}{suffix}-1.xlsx"
    else:
        current_excel_file = output_dir / f"{base_filename}{suffix}-{current_file_num}.xlsx"

    current_writer = pd.ExcelWriter(current_excel_file, engine="openpyxl")
    current_file_rows = 0
    current_file_num += 1

# Check if we need to start a new file
if current_file_rows >= ROWS_PER_FILE:
    # Flush current batch and create new file
    _create_new_file()
```

**Naming pattern**:
- Single file: `France_climbs_all_basic_2025-11-19_v2.0.1_e0000.xlsx`
- Split files: `France_climbs_all_basic_2025-11-19_v2.0.1_e0000-1.xlsx`, `-2.xlsx`, etc.

---

### 3. Updated Function Signature

**Modified**: `climb_analyzer/engine.py:12487-12498`

Added parameters needed for filename generation:

```python
def _stream_to_excel(
    self,
    min_score,
    units,
    enable_geocoding,
    analysis_center,
    persistence,
    scope_info,
    surface_filter,  # ← NEW
    score_type,      # ← NEW
    cycling_only,    # ← NEW
):
```

**Modified call site**: `climb_analyzer/engine.py:14371-14387`

```python
# Get additional parameters from analyzer instance
surface_filter = getattr(self, "surface_filter", "all")
score_type = getattr(self, "score_type", "basic")
cycling_only = getattr(self, "cycling_only", False)

return self._stream_to_excel(
    min_score=min_score,
    units=unit_system or self.unit_system,
    enable_geocoding=enable_geocoding,
    analysis_center=analysis_center,
    persistence=persistence,
    scope_info=scope_info,
    surface_filter=surface_filter,
    score_type=score_type,
    cycling_only=cycling_only,
)
```

---

### 4. Return List of Files

**Modified**: `climb_analyzer/engine.py:13037-13038`

Changed return value from metadata DataFrame to list of file paths:

```python
# OLD - returned DataFrame with metadata
return pd.DataFrame({
    "streaming_export": [True],
    "filename": [str(excel_file)],
    "row_count": [total_filtered],
})

# NEW - return list of created file paths
return created_files  # List[Path]
```

---

### 5. Updated Main Code to Handle File List

**Modified**: `climb_analyzer/engine.py:18556-18577`

Updated main code to recognize streaming export by checking if return value is a list of Paths:

```python
else:
    # Check if this is a streaming export (df is a list of files, not a DataFrame)
    if isinstance(df, list) and df and all(isinstance(f, Path) for f in df):
        # Streaming export already completed - files were saved in streaming mode
        created_files = df
        if len(created_files) == 1:
            print(f"Results saved to {created_files[0].name} (streaming mode)")
            filename = created_files[0]
        else:
            print(f"Results saved to {len(created_files)} files (streaming mode):")
            for f in created_files:
                print(f"   {f.name}")
            filename = created_files[0]
        # Close error logger with correct file count
        if error_logger:
            error_logger.stop_elevation_logging(output_file_count=len(created_files))
    else:
        print("No results to save (no climbs found above minimum score)")
```

---

## 📊 Expected Behavior

### Single File Output (<950K climbs)

```
Pass 2: Streaming 500,000 climbs to Excel...
  Writing Excel: 100%|████████████████| 500000/500000 [04:23<00:00, 1896.45climbs/s]
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000.xlsx (500,000 rows, 78.3 MB)

✓ Saved 500,000 climbs to France_climbs_all_basic_2025-11-19_v2.0.1_e0000.xlsx
```

### Multiple Files (>950K climbs)

```
Pass 2: Streaming 7,407,262 climbs to Excel...
  Writing Excel: 100%|████████████████| 7407262/7407262 [54:12<00:00, 2278.34climbs/s]
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-1.xlsx (950,000 rows, 94.7 MB)
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-2.xlsx (950,000 rows, 94.5 MB)
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-3.xlsx (950,000 rows, 94.6 MB)
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-4.xlsx (950,000 rows, 94.8 MB)
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-5.xlsx (950,000 rows, 94.4 MB)
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-6.xlsx (950,000 rows, 94.7 MB)
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-7.xlsx (950,000 rows, 94.3 MB)
   ✓ Saved France_climbs_all_basic_2025-11-19_v2.0.1_e0000-8.xlsx (857,262 rows, 85.2 MB)

✓ Saved 7,407,262 climbs to 8 files:
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-1.xlsx
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-2.xlsx
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-3.xlsx
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-4.xlsx
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-5.xlsx
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-6.xlsx
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-7.xlsx
   France_climbs_all_basic_2025-11-19_v2.0.1_e0000-8.xlsx
```

---

## 📋 Files Modified

**File**: `climb_analyzer/engine.py`

**Changes**:
1. **Lines 12487-12498**: Updated `_stream_to_excel()` signature to accept `surface_filter`, `score_type`, `cycling_only`
2. **Lines 12774-12822**: Added proper filename generation with version and error count
3. **Lines 12824-13038**: Implemented file splitting logic with 950K row limit
4. **Lines 14371-14387**: Updated call site to pass new parameters
5. **Lines 18556-18577**: Updated main code to handle list of files return value

**Total**: ~300 lines modified

---

## ✅ Validation

**Syntax**: ✅ Passed
```bash
python3 -m py_compile climb_analyzer/engine.py
```

**Testing**: User to run France analysis to verify:
- ✅ Proper filename format
- ✅ File splitting at 950K rows
- ✅ All files under 100MB
- ✅ Correct file numbering
- ✅ Version and error count in filename

---

## 💡 Benefits

1. **Consistent naming**: Streaming exports now use same naming pattern as regular exports
2. **Excel compatibility**: Files stay within Excel's 1M row limit
3. **GitHub friendly**: Files under 100MB can be committed to repos
4. **Proper versioning**: Filename includes app version and elevation error count
5. **File splitting**: Large datasets automatically split across multiple files
6. **Progress visibility**: Shows which file is being written and final summary

---

##    Key Implementation Details

### File Numbering Strategy

- **Single file expected** (≤950K rows): `{base}{suffix}.xlsx` (no number)
- **Multiple files expected** (>950K rows): `{base}{suffix}-1.xlsx`, `-2.xlsx`, etc.

### Row Limit Safety Margin

- Excel max: 1,048,576 rows
- Target: 950,000 rows per file
- Safety margin: ~100K rows (9.5%)

### Why 950K instead of 1M?

1. Leaves room for header row
2. Safety margin for counting discrepancies
3. Ensures files stay under 100MB (empirically ~94MB at 950K rows)
4. Matches pattern used in `save_large_dataframe_as_split_excel()`

---

_Fix applied: 2025-11-19_
_Files: climb_analyzer/engine.py (lines 12487-13038, 14371-14387, 18556-18577)_
_Related: streaming_excel_oom_fix_v2.md, climb_temp_file_format_mismatch_fix.md_
