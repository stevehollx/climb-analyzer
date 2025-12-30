# CLI Reference

Complete command-line interface documentation for Climb Analyzer.

## Basic Usage

```bash
./climb-analyzer [OPTIONS]
```

Without options, launches interactive mode.

## Analysis Modes

Three mutually exclusive modes:

### Address Mode (`-a`)

Analyze climbs within a radius of an address:

```bash
./climb-analyzer -a "Boulder, CO" --distance 25
./climb-analyzer -a "Seattle, WA" --distance 50 -u metric
```

| Option | Description |
|--------|-------------|
| `-a`, `--address` | Address for center point |
| `--distance` | Search radius in miles (required with `-a`) |

### Region Mode (`-r`)

Analyze a state, country, or comma-separated list:

```bash
# Single region
./climb-analyzer -r "Colorado"
./climb-analyzer -r "CO"
./climb-analyzer -r "Switzerland"

# Multiple regions (batch mode)
./climb-analyzer -r "VT,NH,ME"
./climb-analyzer -r "Vermont,New Hampshire,Maine"
```

| Option | Description |
|--------|-------------|
| `-r`, `--run-region` | Region name(s), comma-separated for batch |

!!! note "Auto-Detection"
    The system automatically detects whether input is a US state, country, or province.

### Interactive Mode (`-i`)

Launch menu-based interface:

```bash
./climb-analyzer
./climb-analyzer -i
./climb-analyzer --interactive
```

## Analysis Parameters

### Surface Filter (`-s`)

Filter by road surface type (single or comma-separated):

```bash
./climb-analyzer -r "Vermont" -s all           # All surfaces (default)
./climb-analyzer -r "Vermont" -s paved         # Paved only
./climb-analyzer -r "Vermont" -s gravel        # Gravel only
./climb-analyzer -r "Vermont" -s dirt          # Dirt only
./climb-analyzer -r "Vermont" -s paved,gravel  # Paved and gravel
```

### Cycling Filter

Filter to cycling-accessible roads only:

```bash
./climb-analyzer -r "Vermont" --cycling-filter
```

!!! warning "Default Changed"
    Cycling filter is **disabled** by default (shows all roads). Use `--cycling-filter` to enable.

### Units (`-u`)

Set measurement units:

```bash
./climb-analyzer -r "Vermont" -u auto      # Region-native (default)
./climb-analyzer -r "Vermont" -u imperial  # Miles, feet
./climb-analyzer -r "Vermont" -u metric    # Kilometers, meters
```

!!! note "Auto Units"
    Default `auto` uses imperial for US regions, metric for international.

### Score Type (`-t`)

Select scoring algorithm:

```bash
./climb-analyzer -r "Vermont" -t basic  # distance × grade (default)
./climb-analyzer -r "Vermont" -t fiets  # FIETS index
./climb-analyzer -r "Vermont" -t pdi    # PJAMM Difficulty Index
```

### Minimum Score (`-m`)

Filter climbs by minimum score:

```bash
./climb-analyzer -r "Vermont" -m 10000  # Only climbs scoring 10000+
```

## Data Management

### Download Data (`-D`)

Download OSM and elevation data without running analysis:

```bash
./climb-analyzer -D -r "Vermont"
./climb-analyzer -D -r "Colorado,Utah"
```

### Update Boundaries (`-U`)

Update geographic boundary data:

```bash
./climb-analyzer -U
```

### Checkpoint Management

Checkpoints are **kept by default** after analysis for potential resume. Control behavior:

```bash
# Ignore existing checkpoints - start fresh
./climb-analyzer -r "Vermont" --ignore-checkpoints
```

| Option | Description |
|--------|-------------|
| (default) | Keep checkpoint files after analysis |
| `--ignore-checkpoints` | Ignore existing checkpoints, start fresh |

### Per-Region Cleanup (Post-Analysis)

Delete data for a specific region after successful analysis:

```bash
# Delete only checkpoints for this region after analysis
./climb-analyzer -r "Vermont" -c
./climb-analyzer -r "Vermont" --cleanup-checkpoints

# Delete checkpoints + OSM + elevation for this region after analysis
./climb-analyzer -r "Vermont" -Z
./climb-analyzer -r "Vermont" --cleanup-all-data
```

| Option | Description |
|--------|-------------|
| `-c`, `--cleanup-checkpoints` | Delete checkpoints for THIS region after analysis |
| `-Z`, `--cleanup-all-data` | Delete checkpoints + OSM + elevation for THIS region after analysis |

!!! note "Per-Region Only"
    These flags only delete data for the region being analyzed. Other regions' data is preserved.

### Cloud Upload Control

```bash
# Skip auto-upload to cloud cache
./climb-analyzer -r "Vermont" --no-cloud-upload
```

!!! note
    Clean analyses (default settings) are auto-uploaded. Use `--no-cloud-upload` to disable.

### Global Cleanup Subcommand

Delete all data without running analysis:

```bash
# Delete all checkpoints
./climb-analyzer cleanup --checkpoints

# Delete all OSM data and indexes
./climb-analyzer cleanup --osm

# Delete all elevation data
./climb-analyzer cleanup --elevation

# Delete everything (checkpoints + OSM + elevation)
./climb-analyzer cleanup --all

# Skip confirmation prompts
./climb-analyzer cleanup --all --force

# Clear unavailable tile cache (re-attempt tiles marked as unavailable)
./climb-analyzer cleanup --unavailable-cache

# Clear unavailable cache for specific dataset only
./climb-analyzer cleanup --unavailable-cache --dataset srtm30m
```

| Option | Description |
|--------|-------------|
| `--checkpoints` | Delete ALL checkpoint files |
| `--osm` | Delete ALL OSM .pbf files and indexes |
| `--elevation` | Delete ALL elevation data |
| `--all` | Delete ALL data |
| `--unavailable-cache` | Delete .unavailable files (allows re-attempt of marked tiles) |
| `--dataset NAME` | Only clear unavailable cache for specific dataset (srtm30m, ned10m, aster30m, etc.) |
| `--force` | Skip confirmation prompts |

!!! warning "Global Deletion"
    The cleanup subcommand deletes data for ALL regions. Use `-c` or `-Z` with `-r` for per-region cleanup.

!!! tip "Unavailable Cache"
    If tiles were incorrectly marked as unavailable due to authentication or server issues, use `--unavailable-cache` to clear the cache and re-attempt downloading those tiles.

## GUI Commands

### Start GUI (`-g`)

Launch the web GUI server:

```bash
./climb-analyzer -g
# GUI available at http://localhost:3000
```

### Stop GUI

```bash
./climb-analyzer --stop-gui
```

### Restart GUI

```bash
./climb-analyzer --restart-gui
```

## Climb Merging

Control how climbs at boundaries are merged:

### Cross-Country Merging

```bash
# Allow merging climbs across country boundaries
./climb-analyzer -r "France,Germany" --allow-cross-country-merge
```

!!! note "Schengen Exceptions"
    Cross-country merging is auto-enabled for Schengen Zone countries.

### Merge Distance

```bash
# Set maximum distance for endpoint matching (default: 0.5 km)
./climb-analyzer -r "Vermont" --merge-distance-km 1.0
```

### Disable Merging

```bash
# Disable all climb merging (for debugging)
./climb-analyzer -r "Vermont" --no-merge
```

### Post-Process Merge

Merge climbs from existing Excel files:

```bash
# Merge specific files
./climb-analyzer --merge-regions VT.xlsx NH.xlsx

# Merge using glob pattern
./climb-analyzer --merge-regions output/state_*.xlsx
```

## Informational

### List Regions

```bash
# List all available regions for analysis
./climb-analyzer --list-regions
```

## Help

### Basic Help (`-h`)

```bash
./climb-analyzer -h
./climb-analyzer --help
```

### Extended Help

```bash
./climb-analyzer --help-extended
```

### Verbose Mode

```bash
# Enable verbose output for debugging
./climb-analyzer -r "Vermont" -v
./climb-analyzer -r "Vermont" --verbose
```

## Complete Options Table

### Analysis Modes

| Flag | Long Form | Description | Default |
|------|-----------|-------------|---------|
| `-a` | `--address` | Address for radius analysis | - |
| `-r` | `--run-region` | Region(s) to analyze | - |
| `-i` | `--interactive` | Interactive mode | yes |
| `-g` | `--gui` | Launch web GUI | - |

### Analysis Parameters

| Flag | Long Form | Description | Default |
|------|-----------|-------------|---------|
| - | `--distance` | Search radius in miles (with `-a`) | - |
| `-s` | `--surface-filter` | Surface type filter | all |
| - | `--cycling-filter` | Cycling accessible only | off |
| `-u` | `--units` | Unit system | auto |
| `-t` | `--score-type` | Scoring algorithm | basic |
| `-m` | `--min-score` | Minimum score threshold | 0 |

### Data Management

| Flag | Long Form | Description | Default |
|------|-----------|-------------|---------|
| `-U` | `--update-geo-boundaries` | Update boundaries | - |
| `-D` | `--data-download` | Download data only | - |
| - | `--ignore-checkpoints` | Ignore existing checkpoints | - |
| - | `--no-cloud-upload` | Skip cloud cache upload | - |
| `-c` | `--cleanup-checkpoints` | Delete checkpoints for THIS region after analysis | - |
| `-Z` | `--cleanup-all-data` | Delete all data for THIS region after analysis | - |

> **Note:** Checkpoints are kept by default after analysis. No flag needed to preserve them.

### Cleanup Subcommand

| Flag | Long Form | Description |
|------|-----------|-------------|
| - | `cleanup --checkpoints` | Delete ALL checkpoints |
| - | `cleanup --osm` | Delete ALL OSM data |
| - | `cleanup --elevation` | Delete ALL elevation data |
| - | `cleanup --all` | Delete ALL data |
| - | `cleanup --unavailable-cache` | Delete .unavailable files (re-attempt marked tiles) |
| - | `cleanup --dataset NAME` | Only clear unavailable cache for specific dataset |
| - | `cleanup --force` | Skip confirmation prompts |

### Climb Merging

| Flag | Long Form | Description | Default |
|------|-----------|-------------|---------|
| - | `--allow-cross-country-merge` | Merge across country borders | off |
| - | `--merge-distance-km` | Max merge endpoint distance | 0.5 |
| - | `--no-merge` | Disable all merging | - |
| - | `--merge-regions` | Post-process merge files | - |

### Informational

| Flag | Long Form | Description | Default |
|------|-----------|-------------|---------|
| - | `--list-regions` | List available regions | - |
| `-v` | `--verbose` | Verbose output | off |
| `-h` | `--help` | Show help | - |
| - | `--help-extended` | Extended help | - |

## Examples

### Basic Analysis

```bash
# Analyze a US state
./climb-analyzer -r "Vermont"

# Analyze a country with metric units
./climb-analyzer -r "Switzerland" -u metric

# Analyze paved roads only with FIETS scoring
./climb-analyzer -r "Colorado" -s paved -t fiets
```

### Batch Processing

```bash
# Multiple states
./climb-analyzer -r "VT,NH,ME,MA,CT,RI"

# European countries
./climb-analyzer -r "Switzerland,Austria,Italy" -u metric -t fiets

# With per-region cleanup after each (deletes checkpoints + OSM + elevation)
./climb-analyzer -r "VT,NH,ME" -Z
```

### Data Management

```bash
# Pre-download data for offline use
./climb-analyzer -D -r "Colorado,Utah"

# Clean up all data globally
./climb-analyzer cleanup --all

# Delete only checkpoints for a specific region after analysis
./climb-analyzer -r "Vermont" -c
```

### Advanced Filtering

```bash
# High-difficulty gravel climbs only
./climb-analyzer -r "Vermont" -s gravel -t pdi -m 200

# Cycling-accessible paved roads
./climb-analyzer -r "Colorado" -s paved --cycling-filter
```

---

Next: [Interactive Mode](interactive-mode.md)
