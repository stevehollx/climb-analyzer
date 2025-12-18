# Quick Start

Get your first climb analysis in 5 minutes.

## Your First Analysis

### Option 1: Download from Cloud Cache

The fastest way to get climb data is to download pre-computed analyses:

```bash
./climb-analyzer -r "Vermont"
```

If Vermont is in the cloud cache, you'll see:

```
✓ Vermont analysis found in cloud cache (2025-10-25)
Download cached analysis? (yes/no): yes
Downloading...
✓ Downloaded 1 file, 8.2 MB
```

### Option 2: Run Your Own Analysis

For regions not in the cache, or to get fresh data:

```bash
# Analyze Rhode Island (small, fast)
./climb-analyzer -r "Rhode Island"

# The system will:
# 1. Download OSM data (if needed)
# 2. Build spatial index (if needed)
# 3. Download elevation data (if needed)
# 4. Run the analysis
```

## Common Commands

### Analyze a US State

```bash
# Full state name
./climb-analyzer -r "Colorado"

# Abbreviation
./climb-analyzer -r "CO"

# Multiple states
./climb-analyzer -r "VT,NH,ME"
```

### Analyze a Country

```bash
# Country name
./climb-analyzer -r "Switzerland"

# With metric units
./climb-analyzer -r "Switzerland" -u metric
```

### Address-Based Analysis

Analyze climbs within a radius of a city or street address:

```bash
# 25 mile radius around Boulder, CO
./climb-analyzer -a "Boulder, CO" --radius 25

# 25 mile radius around Boulder, CO
./climb-analyzer -a "109 South Lee Street, Stockbridge, GA 30281" --radius 10
```

### Interactive Mode

For guided menu-based usage:

```bash
./climb-analyzer
```

## Filtering Options

### Surface Type

```bash
# Paved roads only
./climb-analyzer -r "Vermont" -s paved

# Gravel roads only
./climb-analyzer -r "Vermont" -s gravel

# All surfaces (default)
./climb-analyzer -r "Vermont" -s all
```

### Cycling Accessibility

```bash
# All roads (default)
./climb-analyzer -r "Vermont"

# Only cycling-accessible
./climb-analyzer -r "Vermont" --cycling-filter
```

### Scoring Method

```bash
# Basic score (default)
./climb-analyzer -r "Vermont" -t basic

# FIETS index
./climb-analyzer -r "Vermont" -t fiets

# PDI (PJAMM Difficulty Index)
./climb-analyzer -r "Vermont" -t pdi
```

## Output Files

Results are saved to the `output/` directory:

```
output/
└── Vermont_climbs_all_basic_2025-12-17_v2.1.0_e6000.xlsx
```

Filename format: `{Region}_climbs_{surface}_{date}_v{version}_e{elevation_error_count}.xlsx`

### Opening Results

- **Excel/LibreOffice**: Open `.xlsx` file directly
- **Web GUI**: Visualize on interactive map
- **iOS App**: Import for mobile viewing

## Using the Web GUI

Start the GUI for visual analysis:

```bash
# Start GUI server
./climb-analyzer -g

# Open in browser
# http://localhost:3000
```

The GUI provides:

- Interactive map visualization
- Filtering by category, score, surface
- Elevation profiles
- Download management

## Data Management

### Check What Data You Have

```bash
./climb-analyzer
# Select: Data Management → View Data Status
```

### Download Data Without Analysis

```bash
# Download OSM + elevation for Vermont
./climb-analyzer -D -r "Vermont"
```

### Clean Up Data

```bash
# Delete all checkpoints
./climb-analyzer -C

# Delete all OSM data
./climb-analyzer -P

# Delete all elevation data
./climb-analyzer -E

# Delete everything
./climb-analyzer -A
```

## Tips for First-Time Users

1. **Start small** - Try Rhode Island or a small European country first
2. **Use cloud cache** - Download existing analyses when available
3. **Watch memory** - Large states like California need 8GB+ RAM
4. **Be patient** - First run downloads data; subsequent runs are faster

## Example Workflow

```bash
# 1. Check if data exists in cloud cache
./climb-analyzer -r "Hawaii"
# → Downloads from cache if available

# 2. If not in cache, run analysis
# System auto-downloads OSM and elevation data

# 3. View results in GUI
./climb-analyzer -g
# Open http://localhost:3000

# 4. Optionally contribute to cloud cache
# (prompted after completing a "clean" analysis)
```

---

Next: [CLI Reference](../user-guide/cli-reference.md) for complete command documentation.
