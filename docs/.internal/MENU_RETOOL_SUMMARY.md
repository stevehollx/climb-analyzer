# Climb Analyzer Menu System Retool - Summary

## Overview
The main menu system for climb-analyzer has been retooled to simplify the user experience and provide better navigation through geographic regions.

## Changes Made

### 1. Simplified Main Menu (2 Options)

**Before:** 3 options
- Address with radius
- Entire US state
- Entire country

**After:** 2 options
- Address with radius (search near a specific location)
- Select country or subregion (hierarchical region selection)

### 2. Hierarchical Region Selection

The new menu system provides a hierarchical navigation experience:

```
Main Menu
├─ 1. Address (geocode a specific location)
└─ 2. Select Region(s) (hierarchical navigation)
    ├─ Select Continent
    │   └─ Select Country/Region
    │       └─ Select Subregion (if available)
    │           └─ Multiple selections supported (comma-separated)
    └─ Option 0: Go Back (at each level)
```

### 3. Key Features

#### Multiple Selection Support
- Users can select multiple countries by comma-separating numbers (e.g., "1,2,5")
- Users can select multiple subregions within a country (e.g., "3,7,12")
- Users can select "all" to download all regions at the current level
- Each region is processed separately with its own output file

#### Navigation
- **Option 0** is available at every level to go back up the menu hierarchy
- From country selection → back to continent selection
- From subregion selection → back to country selection
- From address input → back to main menu
- From continent selection → cancel entire operation

#### Cross-Region Support
- When multiple subregions are selected, they are processed separately
- Each region gets its own analysis and output file
- Batch progress tracking shows status of each region
- Failed regions can be retried individually

### 4. File Changes

#### [geographic_menu.py](geographic_menu.py)
- **Completely rewritten** to work with `geo_definitions.py` structure
- Simplified to use only "subregions" key (consistent with geo_definitions format)
- Added `select_region_or_address()` - main entry point for menu system
- Added `select_regions_hierarchical()` - handles continent → country → subregion navigation
- Added `select_from_subregions()` - recursive navigation through region hierarchy
- Supports multiple selections via comma-separated input
- Displays file sizes and subregion counts for each option
- Uses ▶ marker to indicate regions with child subregions

#### [climb_analyzer.py](climb_analyzer.py:9318)
- **Updated `get_analysis_scope_choice()`** to use new 2-option menu
- Removed old 3-option menu (address/state/country)
- Now calls `select_region_or_address()` from geographic_menu
- **Added "region" scope type handler** ([climb_analyzer.py:11168](climb_analyzer.py:11168))
  - Handles single region analysis
  - Handles multiple region analysis with batch processing
  - Extracts region bounds from `country_data` or `state_data`
  - Uses `BatchProgressTracker` for multi-region analysis
  - Generates separate output files for each region

### 5. Data Structure

The system now correctly uses the hierarchical structure from [geo_definitions.py](geo_definitions.py):

```python
osm_pbf_urls = {
    "continent-name": {
        "pbf_url": "...",  # Continent-level download
        "size": ...,
        "subregions": {
            "continent/country": {
                "pbf_url": "...",  # Country-level download
                "size": ...,
                "subregions": {
                    "country/subregion": {
                        "pbf_url": "...",  # Subregion-level download
                        "size": ...
                    }
                }
            }
        }
    }
}
```

### 6. Cross-Subregion vs Cross-Country Logic

**Design Decision (per user requirements):**
- ✅ **Multiple subregions within same country:** Supported - processed separately with individual outputs
- ❌ **Multiple countries spanning climbs:** NOT supported - this is an intentional design decision
- Each region is processed independently, producing separate result files
- No attempt to find climbs that cross country boundaries

### 7. Backward Compatibility

Legacy functions are maintained for backward compatibility:
- `select_countries()` - redirects to new hierarchical system
- `select_us_states()` - redirects to new hierarchical system

### 8. Testing

A test script [test_menu.py](test_menu.py) has been created to verify:
- Subregion counting works correctly
- Menu structure is valid
- Region navigation functions properly
- All 8 continents are accessible
- US states can be navigated: North America → US → States

**Test Results:**
```
✓ North America has 78 regions
✓ Found 8 continents with valid data
✓ Successfully navigated to US subregions (53 states/territories)
```

## Usage Examples

### Example 1: Single Country Selection
```
1. Address or 2. Region? → 2
Select Continent: → 7 (North America)
Select Region: → 3 (Mexico)
→ Analyzes entire country of Mexico
```

### Example 2: Multiple US States
```
1. Address or 2. Region? → 2
Select Continent: → 7 (North America)
Select Region: → 4 (US)
Select State(s): → 5,33,48 (California, New York, Washington)
→ Analyzes each state separately
→ Creates 3 separate output files
```

### Example 3: All European Countries
```
1. Address or 2. Region? → 2
Select Continent: → 6 (Europe)
Select Region: → all
→ Analyzes all European countries
→ Creates separate output file for each country
```

### Example 4: Going Back
```
1. Address or 2. Region? → 2
Select Continent: → 7 (North America)
Select Region: → 4 (US)
Select State(s): → 0 (Go back)
Select Region: → 0 (Go back)
Select Continent: → 0 (Cancel)
→ Returns to main program
```

## Benefits

1. **Simpler Interface:** 2 options instead of 3
2. **Better Navigation:** Clear hierarchy with back buttons
3. **More Flexible:** Can select any region at any level
4. **Multiple Selection:** Comma-separated or "all" options
5. **Better UX:** Shows sizes and subregion counts
6. **Consistent Structure:** Uses geo_definitions.py directly
7. **Resumable:** Batch processing tracks progress
8. **Error Handling:** Failed regions can be retried

## Migration Notes

The old menu system with 3 options (address/state/country) has been replaced with the new 2-option system. However:

- **No breaking changes** for batch mode or command-line arguments
- Existing `--scope state` and `--scope country` still work
- Legacy functions redirect to new system with compatibility layer
- User experience is improved while maintaining functionality

## Files Modified

1. **[geographic_menu.py](geographic_menu.py)** - Complete rewrite
2. **[climb_analyzer.py](climb_analyzer.py:9318)** - Updated main menu and added region handler
3. **[test_menu.py](test_menu.py)** - New test file created

## Files Unchanged

- [geo_definitions.py](geo_definitions.py) - Data structure already correct
- Batch processing system - Works with new menu
- Analysis engine - No changes needed
- Output generation - No changes needed
