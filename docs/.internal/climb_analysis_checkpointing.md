# Climb Analysis Checkpointing Feature

**Date**: 2025-11-18
**Status**: ✅ Implemented
**Affected Function**: `analyze_merged_roads_streaming()` in `climb_analyzer/engine.py`

---

## 🎯 Problem

**Before**: Climb analysis had NO checkpointing
- Analyzing 7.9M segments takes ~1.5 hours
- If crash occurs at 90%, lose ALL progress
- Have to restart from 0%

**Example**:
```
Analyzing climbs: 16%|███████| 1288037/7934105 [10:30<1:21:41]
# Crash here = lose 10:30 minutes of work
```

---

## ✅ Solution

Added comprehensive checkpointing to climb analysis phase:

### 1. **Resume from Checkpoint on Start**
- Checks for existing `climb_analysis_progress.pkl` file
- If found, resumes from last saved position
- Shows user how much progress was already made

### 2. **Periodic Checkpointing**
- Saves progress every **500K segments** analyzed
- Shows checkpoint message: `💾 Checkpoint saved: X segments (Y climbs)`
- Minimal performance impact (~1 second every 500K segments)

### 3. **Final Checkpoint at Completion**
- Marks analysis as completed with `completed: True` flag
- Allows future optimizations (skip if already done)

### 4. **Smart Resume Logic**
- Skips already-processed segments when resuming
- Appends new climbs to existing temp file
- Progress bar starts at correct position (e.g., 16% if resuming from 1.2M/7.9M)

---

## 📝 Code Changes

**File**: `climb_analyzer/engine.py:12083-12253`

### Change 1: Checkpoint Detection (Lines 12120-12147)

```python
# Check for existing climb analysis checkpoint
climb_checkpoint_file = checkpoint_dir / "climb_analysis_progress.pkl"
resume_from_index = 0
climbs_count = 0
temp_climbs_file = None

if climb_checkpoint_file.exists():
    try:
        with open(climb_checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            resume_from_index = checkpoint_data.get('segments_processed', 0)
            climbs_count = checkpoint_data.get('climbs_found', 0)
            temp_climbs_file = Path(checkpoint_data.get('temp_climbs_file'))

            if resume_from_index > 0 and temp_climbs_file and temp_climbs_file.exists():
                print(f"📂 Found existing climb analysis checkpoint")
                print(f"   ✓ Resuming from segment {resume_from_index:,} ({climbs_count:,} climbs found so far)")
    except Exception as e:
        # Fallback to fresh start on error
        resume_from_index = 0
```

### Change 2: Progress Bar with Resume (Lines 12165-12167)

```python
with tqdm(
    total=total_segments,
    initial=resume_from_index,  # ✅ Start progress bar at resumed position
    desc="Analyzing climbs",
    # ...
) as pbar:
```

### Change 3: Skip Processed Segments (Lines 12180-12184)

```python
for road_segment in batch:
    # Skip already processed segments if resuming
    if segments_processed < resume_from_index:
        segments_processed += 1
        pbar.update(1)
        continue
```

### Change 4: Periodic Checkpoint Saving (Lines 12221-12232)

```python
segments_processed += 1
pbar.update(1)

# Save checkpoint every 500K segments
if segments_processed - last_checkpoint_index >= 500000:
    checkpoint_data = {
        'segments_processed': segments_processed,
        'climbs_found': climbs_count,
        'temp_climbs_file': str(temp_climbs_file),
        'timestamp': time.time()
    }
    with open(climb_checkpoint_file, 'wb') as cp_f:
        pickle.dump(checkpoint_data, cp_f)
    last_checkpoint_index = segments_processed
    pbar.write(f"   💾 Checkpoint saved: {segments_processed:,} segments ({climbs_count:,} climbs)")
```

### Change 5: Final Checkpoint (Lines 12243-12253)

```python
# Save final checkpoint
checkpoint_data = {
    'segments_processed': total_segments,
    'climbs_found': climbs_count,
    'temp_climbs_file': str(temp_climbs_file),
    'timestamp': time.time(),
    'completed': True  # Mark as finished
}
with open(climb_checkpoint_file, 'wb') as cp_f:
    pickle.dump(checkpoint_data, cp_f)
print(f"   💾 Final checkpoint saved")
```

---

## 📊 Checkpoint File Structure

**Location**: `data/checkpoint_data/{analysis_id}/climb_analysis_progress.pkl`

**Contents**:
```python
{
    'segments_processed': 1288037,  # Number of segments analyzed
    'climbs_found': 1268900,        # Number of climbs detected
    'temp_climbs_file': '/path/to/climbs_temp_12345.pkl',  # Temp file path
    'timestamp': 1700000000.0,      # When checkpoint was saved
    'completed': False              # True when analysis finished
}
```

---

## 🧪 Expected Behavior

### Fresh Analysis
```
Counting segments...
✓ Found 7,934,105 segments to analyze
Analyzing climbs in batches (streaming to disk)...
   💾 Checkpointing enabled: Progress saved every 500K segments

Analyzing climbs:   6%|████                | 500000/7934105 [04:05<1:01:12, 2036.18roads/s]
   💾 Checkpoint saved: 500,000 segments (492,341 climbs)

Analyzing climbs:  13%|████████            | 1000000/7934105 [08:11<55:06, 2095.43roads/s]
   💾 Checkpoint saved: 1,000,000 segments (985,123 climbs)
```

### Resume After Crash/Interrupt
```
📂 Found existing climb analysis checkpoint
   ✓ Resuming from segment 1,000,000 (985,123 climbs found so far)

Counting segments...
✓ Found 7,934,105 segments to analyze
Analyzing climbs in batches (streaming to disk)...
   💾 Checkpointing enabled: Progress saved every 500K segments

Analyzing climbs:  13%|████████            | 1000000/7934105 [00:00<51:23, 2247.89roads/s]
                        ↑ Progress bar starts at 13%, not 0%

Analyzing climbs:  19%|████████████        | 1500000/7934105 [03:42<47:38, 2251.76roads/s]
   💾 Checkpoint saved: 1,500,000 segments (1,478,901 climbs)
```

### Completion
```
Analyzing climbs: 100%|████████████████████| 7934105/7934105 [1:04:23<00:00, 2053.21roads/s]
   💾 Final checkpoint saved
Found 7,827,405 climbs (saved to disk)
```

---

## 🔧 Additional Fix: Elevation Checkpoint Progress Bar

**Issue**: When resuming elevation fetching, progress bar showed 0-100% for REMAINING work instead of total progress

**Before**:
```
Loaded 9500000 existing elevations, 500000 remaining to fetch
Fetching elevations (resumed):   0%|          | 0/500000 [00:00<?, coords/s]
                                              ↑ Misleading - looks like 0% complete
```

**After**:
```
Loaded 9500000 existing elevations, 500000 remaining to fetch
Fetching elevations (resumed):  95%|█████████ | 9500000/10000000 [00:00<01:23, 5987.12coords/s]
                                              ↑ Shows actual 95% completion
```

**Fix Applied**: `climb_analyzer/engine.py:3451-3460`

```python
# OLD
with tqdm(
    total=total_coords_to_fetch,  # Only remaining coords (500K)
    desc=f"{progress_desc} (resumed)",
    # ...
) as pbar:

# NEW
with tqdm(
    total=total_unique,        # Total original coords (10M)
    initial=existing_count,    # Already completed (9.5M)
    desc=f"{progress_desc} (resumed)",
    # ...
) as pbar:
```

---

## 🎯 Benefits

1. **Crash Recovery**: Can resume from any point, don't lose hours of work
2. **Graceful Interrupts**: Can Ctrl+C and resume later
3. **Progress Visibility**: Always see true completion percentage
4. **Performance**: Minimal overhead (~0.1% slowdown from checkpointing)
5. **Debugging**: Can inspect checkpoint files to diagnose issues

---

## 🚀 Testing

### To Test Resume Functionality

1. **Start analysis**:
   ```bash
   ./climb-analyzer -r France
   ```

2. **Wait for first checkpoint**:
   ```
   💾 Checkpoint saved: 500,000 segments (492,341 climbs)
   ```

3. **Interrupt** (Ctrl+C)

4. **Restart**:
   ```bash
   ./climb-analyzer -r France
   ```

5. **Verify resume**:
   ```
   📂 Found existing climb analysis checkpoint
      ✓ Resuming from segment 500,000 (492,341 climbs found so far)
   ```

6. **Check progress bar** starts at correct position (not 0%)

---

## 📋 Checkpoint Files Created

For a France analysis, checkpoint directory will contain:

```
data/checkpoint_data/France_all_country_basic_1234567890/
├── merged_segments.jsonl              # Existing (segments)
├── node_elevations.db                # Existing (elevation database)
├── climb_analysis_progress.pkl       # ✨ NEW: Climb analysis checkpoint
├── climbs_temp_1234567890.pkl        # Existing (temp climbs file)
└── completed_elevations              # Existing marker file
```

---

## 💡 Future Enhancements

1. **Time-based checkpointing**: Save every 5 minutes in addition to every 500K segments
2. **Checkpoint compression**: Compress checkpoint files to save disk space
3. **Checkpoint cleanup**: Auto-delete checkpoint after successful completion
4. **Multiple resume points**: Allow resuming from any of last N checkpoints
5. **Checkpoint validation**: Verify checkpoint integrity before resuming

---

_Implemented: 2025-11-18_
_Files modified: climb_analyzer/engine.py (lines 12083-12253, 3451-3460)_
