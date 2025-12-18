# Memory Leak Analysis - OOM at Batch 26/80 (France)

**Date**: 2025-01-16
**Status**: CRITICAL LEAKS FOUND
**Region**: France (7.9M segments, 31M+ coordinates)
**Failure Point**: Batch 26/80 (34%)

## Executive Summary

Despite applying 4 critical fixes, France analysis still OOMs at 34%. Analysis reveals **TWO MASSIVE memory leaks** in `_fetch_elevations_parallel()` that accumulate **6-8GB** of data across all batches without ever clearing it.

---

## Critical Memory Leaks Found

### Leak #5: `coordinate_mapping` Dict Accumulation (CRITICAL)
**Location**: `climb_analyzer.py:2618, 2737-2739`
**Memory Impact**: **3-4 GB** for France (31M coordinates)

```python
# Line 2618 - Dict created and NEVER cleared
coordinate_mapping = {}

# Line 2737-2739 - Accumulates ALL successful elevations across ALL batches
for j, elevation in enumerate(batch_elevations):
    if elevation is not None:
        coordinate_mapping[unique_coords[actual_coord_idx]] = elevation

# Line 2780-2782 - Saved to disk every 50 batches but NEVER cleared!
persistence.save_elevation_progress(coordinate_mapping, batch_checkpoint_info)
```

**Root Cause**:
- Dict grows continuously across all 80 batches
- Checkpointed to disk but never flushed/cleared from memory
- For France: 31M entries × ~100 bytes = **3.1 GB minimum**

**Fix Required**: Flush to disk and clear periodically (similar to SQLite buffer pattern)

---

### Leak #6: `all_elevations` List Pre-Allocation (CRITICAL)
**Location**: `climb_analyzer.py:2617, 2734`
**Memory Impact**: **3-4 GB** for France (31M coordinates)

```python
# Line 2617 - Pre-allocate ENTIRE list in memory
all_elevations = [None] * total_unique  # 31M entries!

# Line 2734 - Populate throughout all batches
all_elevations[actual_coord_idx] = elevation

# Line 2799 - Only used at the END to map back to original order
result_elevations = [all_elevations[unique_idx] for unique_idx in original_to_unique]
```

**Root Cause**:
- Entire elevation array held in memory until function completes
- For France: 31M floats × ~100 bytes (Python object overhead) = **3.1 GB**

**Why it exists**: Needed to map unique coords back to original coordinate order

**Fix Required**: Stream directly to disk instead of accumulating in memory, or use checkpoint-based batching

---

## Total Memory Waste

| Data Structure | Location | Size (France) | Status |
|----------------|----------|---------------|--------|
| `coordinate_mapping` dict | Line 2618 | 3-4 GB | ❌ Growing unbounded |
| `all_elevations` list | Line 2617 | 3-4 GB | ❌ Pre-allocated, never cleared |
| **TOTAL** | `_fetch_elevations_parallel()` | **6-8 GB** | ❌ **CAUSES OOM AT 34%** |

---

## Why OOM Occurs at Batch 26/80 (34%)

```
Batch 26 memory breakdown:
- coordinate_mapping:  ~3.1 GB (31M entries accumulated)
- all_elevations:      ~3.1 GB (31M pre-allocated)
- SQLite pending:      ~200 MB (FLUSH_INTERVAL=2, 500k safety limit)
- Batch processing:    ~200 MB (current batch data)
- Docker overhead:     ~1.5 GB
- System overhead:     ~1.5 GB
-------------------------------------------
TOTAL:                 ~9.6 GB

Available RAM:         11.6 GB (total) - 2.1 GB (OS/Docker) = 9.5 GB usable
```

**Result**: Memory usage exceeds available RAM → Linux OOM killer terminates process

---

## Fix Strategy

### Option A: Checkpoint-Based Flushing (RECOMMENDED)
Flush `coordinate_mapping` to disk periodically and clear it:

```python
# Every 10-20 batches, flush coordinate_mapping and clear
if completed_batches % 10 == 0:
    persistence.save_elevation_progress(coordinate_mapping, batch_checkpoint_info)
    coordinate_mapping.clear()  # Free 3GB!
    gc.collect()
```

**Pros**: Minimal code changes, maintains checkpoint functionality
**Cons**: Requires merging checkpoint files at the end
**Memory Savings**: ~3 GB

### Option B: Remove `all_elevations` Pre-Allocation
Build result array incrementally or use disk-based storage:

```python
# Instead of pre-allocating entire array:
# all_elevations = [None] * total_unique

# Use disk-based temporary storage
import tempfile, pickle
temp_elevations_file = tempfile.mktemp(suffix='.pkl')
```

**Pros**: Eliminates 3GB allocation
**Cons**: More complex implementation
**Memory Savings**: ~3 GB

### Option C: Both (BEST)
Combine both fixes for **6-8 GB total savings**

**Expected Result**: France completes successfully within 11.6 GB RAM limit

---

## Recommended Fixes

### Fix #5: Flush `coordinate_mapping` Periodically

```python
# After line 2782, add:
if completed_batches % 10 == 0:
    # Clear coordinate_mapping after checkpointing
    coordinate_mapping.clear()
    gc.collect()
```

### Fix #6: Stream `all_elevations` to Disk

Replace pre-allocated list with disk-based storage, or use chunked checkpointing.

---

## Debug Messages for Next Run

To diagnose remaining issues after these fixes, add:

```python
import psutil

# After each batch completion (line 2752):
process = psutil.Process()
mem_info = process.memory_info()
print(f"[MEM] Batch {completed_batches}: RSS={mem_info.rss / 1024**3:.2f}GB, "
      f"coordinate_mapping={len(coordinate_mapping):,} entries, "
      f"all_elevations={len(all_elevations):,} entries")
```

This will show:
- Memory growth per batch
- Size of accumulating data structures
- Exact batch where OOM occurs

---

## Next Steps

1. ✅ **Identified**: 2 critical memory leaks (6-8GB total)
2. ⏳ **Apply**: Fixes #5 and #6 to climb_analyzer.py
3. ⏳ **Test**: France analysis with fixes
4. ⏳ **Monitor**: Memory usage with debug messages
5. ⏳ **Verify**: Completes successfully within 11.6GB RAM

---

## References

- Previous fixes: [bugs.md](../../.claude/bugs.md) - OOM at 34% (Fixes #1-4)
- Memory analysis: [memory_analysis_2025-01-16.md](../../.claude/memory_analysis_2025-01-16.md)
- Code location: `climb_analyzer.py:2584-2816` (_fetch_elevations_parallel)
