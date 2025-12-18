# Elevation Profile Blank/Incomplete Investigation

**Date**: 2025-11-19
**Issue**: Elevation profiles appear blank in Excel export despite fixes for coordinate-based lookups
**Status**: 🔍 INVESTIGATING

---

## 📋 Problem Statement

After implementing:
1. Coordinate-based elevation storage (`coord_{lat}_{lon}`)
2. Coordinate-based elevation lookups
3. Debug logging

The user reports elevation profiles are still blank/incomplete in Excel exports. However, debug messages show >50% of elevations are being found (no warnings triggered).

---

## 🔍 Investigation Steps Completed

### Step 1: Verify Elevation Profile Generation Code ✅

**Location**: `engine.py:12458-12533`

**Flow**:
1. Line 12455: Create `climb_metrics` object via `calculate_climb_metrics()`
2. Line 12458: Check if nodes exist and len > 2
3. Lines 12468-12483: Fetch elevations using coordinate-based lookup
4. Line 12514-12518: Generate and assign elevation profile
5. Line 12534: Add to batch

**Edge Cases Handled**:
- Line 12532: If nodes None or ≤2 nodes → `elevation_profile = ""`
- Line 12529: If exception during generation → `elevation_profile = ""`

**Conclusion**: Code correctly assigns elevation profiles

---

### Step 2: Verify Connection Update Preserves Profiles ✅

**Location**: `engine.py:13787-13789`

```python
for climb in batch_climbs:
    climb.connected_climbs = connection_names.get(climb_idx, [])
    climb_idx += 1
```

**Conclusion**: Only modifies `connected_climbs` field, all other fields including `elevation_profile` are preserved

---

### Step 3: Verify Excel Export Reads Profile Field ✅

**Location**: `engine.py:13280`

```python
"Elevation Profile": getattr(climb, "elevation_profile", "") or "",
```

**Conclusion**: Export correctly reads elevation_profile attribute with fallback to empty string

---

### Step 4: Verify ClimbMetrics Dataclass Definition ✅

**Location**: `engine.py:4382-4414`

```python
@dataclass
class ClimbMetrics:
    ...
    elevation_profile: str = ""  # Line 4413
    ...
```

**Conclusion**: ClimbMetrics is properly decorated as dataclass with elevation_profile field

---

## 🐛 Potential Root Cause: Dataclass Field Initialization

### Issue Found: ClimbMetrics Constructor Doesn't Pass elevation_profile

**Location**: `engine.py:14630-14656`

When ClimbMetrics is created, elevation_profile is **not passed as a parameter**:

```python
return ClimbMetrics(
    street_name=street_name,
    climb_category=category,
    ...
    connected_climbs=[],  # Line 14653 - included
    # elevation_profile NOT included!
)
```

### Dataclass Behavior with Missing Fields

When a dataclass field has a default value and is NOT passed to `__init__`:
- Python SHOULD initialize it to the default value
- But this depends on field order and whether all required fields come first

**Potential Issue**: If there's a field ordering problem, dataclass might not properly initialize `elevation_profile` to `""`.

---

## 🧪 Hypothesis Testing Needed

### Hypothesis 1: elevation_profile Not Initialized
If elevation_profile field doesn't exist on climb object when created:
- Later assignment at line 12514 creates a NEW attribute (not a dataclass field)
- Pickle might not save it correctly
- Or getattr() at export time returns default ""

**Test**: Add debug logging to check if elevation_profile exists right after creation:
```python
climb_metrics = self.calculate_climb_metrics(road_segment)
# print(f"[DEBUG] Has elevation_profile: {hasattr(climb_metrics, 'elevation_profile')}")
# print(f"[DEBUG] Initial value: {getattr(climb_metrics, 'elevation_profile', 'MISSING')}")
```

### Hypothesis 2: Nodes Missing or ≤2
If most climbs have None nodes or ≤2 nodes:
- elevation_profile set to "" explicitly (line 12532)
- Excel would show blank profiles

**Test**: Check how many climbs have >2 nodes vs ≤2:
```python
if climb_metrics.nodes and len(climb_metrics.nodes) > 2:
    # print(f"[DEBUG] Processing climb with {len(climb_metrics.nodes)} nodes")
else:
    # print(f"[DEBUG] Skipping profile generation - nodes: {len(climb_metrics.nodes) if climb_metrics.nodes else 0}")
```

### Hypothesis 3: generate_elevation_profile Returns Empty String
If elevation profile generation returns "" even with valid data:
- Could be issue in elevation_profile.py

**Test**: Check return value:
```python
profile = generate_elevation_profile(climb_metrics.nodes, elevations)
# print(f"[DEBUG] Generated profile length: {len(profile)}")
# print(f"[DEBUG] Profile preview: {profile[:50]}")
climb_metrics.elevation_profile = profile
```

---

## 🔧 Proposed Fix

### Fix 1: Explicitly Pass elevation_profile in Constructor

**Change**: `engine.py:14656`

```python
return ClimbMetrics(
    ...
    connected_climbs=[],
    elevation_profile="",  # EXPLICITLY initialize to empty string
)
```

**Rationale**: Ensures field is always initialized, even if dataclass default doesn't work

---

### Fix 2: Add Debug Logging to Verify Field Existence

**Add after line 12455**:

```python
climb_metrics = self.calculate_climb_metrics(road_segment)

# DEBUG: Verify elevation_profile field exists
if not hasattr(climb_metrics, 'elevation_profile'):
    print(f"\n⚠️  WARNING: ClimbMetrics missing elevation_profile field!")
    climb_metrics.elevation_profile = ""  # Create it
```

---

### Fix 3: Add Debug Logging for Profile Generation Success

**Add after line 12518**:

```python
climb_metrics.elevation_profile = generate_elevation_profile(
    climb_metrics.nodes, elevations
)

# DEBUG: Check if profile was generated
import random
if random.random() < 0.01:  # 1% sample
    print(f"\n[DEBUG] Profile generated for {climb_metrics.street_name}")
    print(f"   Nodes: {len(climb_metrics.nodes)}")
    print(f"   Elevations found: {sum(1 for e in elevations if e != 0.0)}/{len(elevations)}")
    print(f"   Profile length: {len(climb_metrics.elevation_profile)} chars")
    if climb_metrics.elevation_profile:
        print(f"   Profile preview: {climb_metrics.elevation_profile[:80]}...")
```

---

## 📊 Debug Output to Collect

When running Luxembourg analysis with these debug additions, look for:

1. **Field existence warnings**: If you see "WARNING: ClimbMetrics missing elevation_profile field" → Fix 1 needed
2. **Node count**: If most climbs show "Skipping profile generation - nodes: 0" → Issue with nodes not being stored
3. **Profile generation**: If profiles are generated but length is 0 → Issue in generate_elevation_profile function
4. **Profile preservation**: Compare profile length at generation vs export time → Check if pickle/unpickle loses data

---

## 🔗 Related Files

- **engine.py:4382-4414** - ClimbMetrics dataclass definition
- **engine.py:12458-12533** - Elevation profile generation during analysis
- **engine.py:13280** - Elevation profile export to Excel
- **engine.py:14630-14656** - ClimbMetrics object creation
- **climb_analyzer/data/elevation_profile.py** - Profile generation function

---

## ✅ Fixes Applied

### Fix 1: Explicitly Initialize elevation_profile in Constructor ✅

**Location**: `engine.py:14654`

**Change**:
```python
return ClimbMetrics(
    ...
    connected_climbs=[],
    elevation_profile="",  # EXPLICITLY initialize to empty string (NEW)
    start_lat=coordinates[0][0] if coordinates else 0.0,
    start_lon=coordinates[0][1] if coordinates else 0.0,
)
```

**Rationale**: Ensures elevation_profile field is always present on ClimbMetrics objects, even if dataclass default initialization has issues.

---

### Fix 2: Debug Logging - Field Existence Check ✅

**Location**: `engine.py:12457-12460`

**Added**:
```python
# DEBUG: Verify elevation_profile field exists (should be initialized to "")
if climb_metrics and not hasattr(climb_metrics, 'elevation_profile'):
    print(f"\n⚠️  WARNING: ClimbMetrics missing elevation_profile field for {climb_metrics.street_name}!")
    climb_metrics.elevation_profile = ""  # Create it if missing
```

**Purpose**: Catches cases where elevation_profile field doesn't exist and creates it

---

### Fix 3: Enhanced Debug Logging - Profile Generation ✅

**Location**: `engine.py:12526-12535`

**Changed sampling rate from 0.1% to 1%** and added more details:
```python
if random.random() < 0.01:  # Log 1% of climbs (was 0.001)
    print(f"\n[DEBUG] ✓ Elevation profile generated for {climb_metrics.street_name}")
    print(f"   Nodes: {len(climb_metrics.nodes)}, Elevations with data: {non_zero_elevations}/{len(elevations)} ({non_zero_elevations*100//len(elevations)}%)")
    print(f"   Profile length: {len(climb_metrics.elevation_profile)} chars")
    if climb_metrics.elevation_profile:
        print(f"   Profile preview: {climb_metrics.elevation_profile[:100]}...")
    else:
        print(f"   ⚠️  Profile is EMPTY despite {non_zero_elevations} elevations!")
```

**Purpose**: Shows when profiles are successfully generated with actual length and preview

---

### Fix 4: Debug Logging - Skipped Profiles ✅

**Location**: `engine.py:12541-12545`

**Added**:
```python
else:
    # DEBUG: Log why profile generation was skipped
    import random
    if random.random() < 0.001:  # Log 0.1% of skipped cases
        node_count = len(climb_metrics.nodes) if climb_metrics.nodes else 0
        print(f"\n[DEBUG] Profile generation skipped for {climb_metrics.street_name}: {node_count} nodes (need >2)")
    climb_metrics.elevation_profile = ""
```

**Purpose**: Shows why profiles are being skipped (missing nodes or too few nodes)

---

### Fix 5: Debug Logging - Export Phase ✅

**Location**: `engine.py:13297-13306`

**Added**:
```python
# DEBUG: Sample check elevation profiles at export time
import random
if random.random() < 0.01:  # 1% sample
    profile = row["Elevation Profile"]
    if profile:
        print(f"\n[DEBUG] ✓ Exporting profile for {climb.street_name}: {len(profile)} chars")
    else:
        has_attr = hasattr(climb, 'elevation_profile')
        attr_val = getattr(climb, 'elevation_profile', 'MISSING')
        print(f"\n[DEBUG] ⚠️  NO profile for {climb.street_name}: hasattr={has_attr}, value='{attr_val}'")
```

**Purpose**: Verifies that elevation_profile exists and has data when exporting to Excel

---

## 📊 Expected Debug Output

When running Luxembourg analysis, you should now see:

### During Analysis Phase:
```
[DEBUG] ✓ Elevation profile generated for Rue de la Montagne
   Nodes: 45, Elevations with data: 44/45 (97%)
   Profile length: 234 chars
   Profile preview: ▁▂▂▃▃▄▄▅▅▆▆▇▇██...

[DEBUG] Profile generation skipped for Short Street: 2 nodes (need >2)
```

### During Export Phase:
```
[DEBUG] ✓ Exporting profile for Rue de la Montagne: 234 chars

[DEBUG] ⚠️  NO profile for Short Street: hasattr=True, value=''
```

### Warning Messages (If field initialization failed):
```
⚠️  WARNING: ClimbMetrics missing elevation_profile field for Street Name!
```

---

## 🎯 What to Look For

1. **No field warnings** - If you see warnings about missing elevation_profile field, there's a deeper dataclass issue
2. **Profiles being generated** - Should see multiple "[DEBUG] ✓ Elevation profile generated" messages
3. **Profile lengths >0** - Generated profiles should have >0 characters
4. **Profiles preserved** - Profiles generated during analysis should still exist at export time
5. **Empty profiles identified** - Should see which climbs have empty profiles and why

---

_Investigation started: 2025-11-19_
_Fixes applied: 2025-11-19_
_Status: Ready for test run - awaiting Luxembourg analysis output_
