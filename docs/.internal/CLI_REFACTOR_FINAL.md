# CLI Argument Refactor - Final Design

## Overview
Comprehensive CLI refactor based on user requirements to enable fully programmatic usage with intelligent region auto-detection.

## Key Design Decisions

1. **`--address` is separate from `--subregion`** - Address-based analysis uses `--address` + `--radius`, not `--subregion address`
2. **`--subregion` auto-detects type** - No need to specify if it's state/country/province, uses `geo_definitions.py` to match
3. **`--run-region` supports comma-separated** - Single argument for both single and batch analysis
4. **Single-letter aliases** - Quick typing with `-X` style flags
5. **`--delete-data-on-complete`** - Cleanup after analysis finishes

## Complete CLI Argument Structure

### Analysis Mode (Mutually Exclusive)
```python
analysis_group = parser.add_mutually_exclusive_group()

analysis_group.add_argument(
    "-a", "--address",
    type=str,
    metavar="ADDRESS",
    help="Address for radius-based analysis (e.g., 'Boulder, CO')"
)

analysis_group.add_argument(
    "-r", "--run-region",
    type=str,
    metavar="REGION",
    help="Analyze region(s). Single: 'Colorado' or Batch: 'Vermont,Maine,NH'. Auto-detects region type."
)

analysis_group.add_argument(
    "-i", "--interactive",
    action="store_true",
    help="Launch interactive mode with menus (default if no arguments)"
)
```

### Analysis Parameters
```python
parser.add_argument(
    "--radius",
    type=float,
    metavar="RADIUS",
    help="Search radius in miles/km (required with --address)"
)

parser.add_argument(
    "-s", "--surface-filter",
    type=str,
    default="all",
    choices=["paved", "gravel", "dirt", "all"],
    help="Surface filter: paved, gravel, dirt, or all [default: all]"
)

parser.add_argument(
    "--no-cycling-filter",
    action="store_true",
    help="Disable cycling accessibility filter"
)

parser.add_argument(
    "-u", "--units",
    type=str,
    default="imperial",
    choices=["metric", "imperial"],
    help="Unit system [default: imperial]"
)

parser.add_argument(
    "-t", "--score-type",
    type=str,
    default="basic",
    choices=["basic", "fiets", "pdi"],
    help="Scoring: basic (grade*dist), fiets (gradient), pdi (difficulty) [default: basic]"
)

parser.add_argument(
    "-m", "--min-score",
    type=float,
    help="Minimum climb score threshold"
)

parser.add_argument(
    "-g", "--geocoding",
    type=str,
    default="yes",
    choices=["yes", "no"],
    help="Enable reverse geocoding [default: yes]"
)
```

### Data Management Operations
```python
data_group = parser.add_argument_group("Data Management")

data_group.add_argument(
    "-U", "--update-geo-boundaries",
    action="store_true",
    help="Update country/state boundary data from sources"
)

data_group.add_argument(
    "-D", "--data-download",
    action="store_true",
    help="Download OSM and elevation data without running analysis"
)

data_group.add_argument(
    "-C", "--delete-checkpoints",
    action="store_true",
    help="Delete checkpoint files"
)

data_group.add_argument(
    "-P", "--delete-planet-data",
    action="store_true",
    help="Delete all OSM .pbf files and indices"
)

data_group.add_argument(
    "-E", "--delete-elevation-data",
    action="store_true",
    help="Delete all elevation dataset files"
)

data_group.add_argument(
    "-X", "--delete-data-on-complete",
    action="store_true",
    help="Delete OSM and elevation data after analysis completes"
)
```

### Help
```python
parser.add_argument(
    "-h", "--help",
    action="help",
    help="Show this help message and exit"
)

parser.add_argument(
    "--help-extended",
    action="store_true",
    help="Show extended help with examples"
)
```

## Region Auto-Detection Logic

### Implementation
```python
def detect_region_type(region_name: str) -> tuple[str, str]:
    """
    Detect if region is a US state, country, or other based on geo_definitions.

    Args:
        region_name: Name of region to detect

    Returns:
        Tuple of (region_type, canonical_name)
        region_type: "state" | "country" | "subregion" | "unknown"
        canonical_name: Standardized name from geo_definitions
    """
    from geo_definitions import state_data, country_data

    # Check US states first (case-insensitive, handle abbreviations)
    region_lower = region_name.lower().strip()

    # Try exact match
    for state_name, state_info in state_data.items():
        if state_name.lower() == region_lower:
            return ("state", state_name)
        # Check abbreviations
        if 'abbr' in state_info and state_info['abbr'].lower() == region_lower:
            return ("state", state_name)

    # Check countries
    for country_name, country_info in country_data.items():
        if country_name.lower() == region_lower:
            return ("country", country_name)
        # Check country codes
        if 'code' in country_info and country_info['code'].lower() == region_lower:
            return ("country", country_name)

    # Check for subregions within countries (provinces, states in other countries)
    for country_name, country_info in country_data.items():
        if 'subregions' in country_info:
            for subregion in country_info['subregions']:
                if subregion.lower() == region_lower:
                    return ("subregion", f"{country_name}:{subregion}")

    # Unknown - return as-is and let downstream handle it
    return ("unknown", region_name)


def parse_regions(region_string: str) -> list[dict]:
    """
    Parse comma-separated region string and detect types.

    Args:
        region_string: Comma-separated regions like "Colorado,Vermont" or "Switzerland,Austria"

    Returns:
        List of dicts with keys: name, type, canonical_name
    """
    regions = [r.strip() for r in region_string.split(',') if r.strip()]

    parsed_regions = []
    for region in regions:
        region_type, canonical_name = detect_region_type(region)
        parsed_regions.append({
            'input_name': region,
            'type': region_type,
            'canonical_name': canonical_name
        })

    return parsed_regions
```

### Usage in Main
```python
if args.run_region:
    regions = parse_regions(args.run_region)

    # Check if all regions have same type
    region_types = set(r['type'] for r in regions)

    if len(region_types) > 1:
        print("⚠️  Warning: Mixed region types detected:")
        for r in regions:
            print(f"  - {r['input_name']}: {r['type']}")
        confirm = input("\nContinue? [y/N]: ")
        if confirm.lower() != 'y':
            sys.exit(1)

    # Determine if single or batch
    is_batch = len(regions) > 1

    if is_batch:
        print(f"\n🔄 Running BATCH analysis for {len(regions)} regions")
        # Use batch runner
        run_batch_analysis(regions, args)
    else:
        print(f"\n🔄 Running SINGLE region analysis: {regions[0]['canonical_name']}")
        # Use single region analysis
        run_single_region_analysis(regions[0], args)
```

## Complete Help Text

```
usage: climb_analyzer.py [-h] [--help-extended]
                         [-a ADDRESS | -r REGION | -i]
                         [--radius RADIUS] [-s {paved,gravel,dirt,all}]
                         [--no-cycling-filter] [-u {metric,imperial}]
                         [-t {basic,fiets,pdi}] [-m MIN_SCORE]
                         [-g {yes,no}] [-U] [-D] [-C] [-P] [-E] [-X]

Climb Analyzer - Analyze road and trail climbs from OpenStreetMap

ANALYSIS MODES (choose one):
  -a ADDRESS, --address ADDRESS
                        Address for radius-based analysis (e.g., 'Boulder, CO')
  -r REGION, --run-region REGION
                        Analyze region(s). Single: 'Colorado' or Batch: 'VT,NH,ME'
                        Auto-detects: US states, countries, provinces
  -i, --interactive     Launch interactive mode with menus (default if no args)

ANALYSIS PARAMETERS:
  --radius RADIUS       Search radius in miles/km (required with --address)
  -s {paved,gravel,dirt,all}, --surface-filter
                        Surface filter [default: all]
  --no-cycling-filter   Disable cycling accessibility filter
  -u {metric,imperial}, --units
                        Unit system [default: imperial]
  -t {basic,fiets,pdi}, --score-type
                        Scoring algorithm [default: basic]
  -m MIN_SCORE, --min-score MIN_SCORE
                        Minimum climb score threshold
  -g {yes,no}, --geocoding
                        Enable reverse geocoding [default: yes]

DATA MANAGEMENT:
  -U, --update-geo-boundaries
                        Update country/state boundary data
  -D, --data-download   Download OSM and elevation data without analysis
  -C, --delete-checkpoints
                        Delete checkpoint files
  -P, --delete-planet-data
                        Delete all OSM .pbf files and indices
  -E, --delete-elevation-data
                        Delete all elevation dataset files
  -X, --delete-data-on-complete
                        Delete OSM and elevation data after analysis completes

HELP:
  -h, --help            Show this help message and exit
  --help-extended       Show extended help with examples

EXAMPLES:

  Address analysis:
    python climb_analyzer.py -a "Boulder, CO" --radius 25 -u imperial -t fiets

  Single region:
    python climb_analyzer.py -r Colorado -s paved -u metric

  Batch regions:
    python climb_analyzer.py -r "Vermont,New Hampshire,Maine" -X

  With abbreviations:
    python climb_analyzer.py -r "CO,UT,WY" -s paved -t pdi -m 1000

  Country analysis:
    python climb_analyzer.py -r Switzerland -u metric -t fiets

  Download data only:
    python climb_analyzer.py -D -r Vermont

  Update boundaries:
    python climb_analyzer.py -U

  Clean up data:
    python climb_analyzer.py -C        # Delete checkpoints
    python climb_analyzer.py -P        # Delete OSM data
    python climb_analyzer.py -E        # Delete elevation data

For more examples: python climb_analyzer.py --help-extended
```

## Extended Help Examples

```python
def show_extended_help():
    """Show extended help with comprehensive examples."""
    print("""
CLIMB ANALYZER - EXTENDED HELP
===============================

1. QUICK ADDRESS ANALYSIS
   Analyze climbs within 25 miles of Boulder, CO:

   $ python climb_analyzer.py -a "Boulder, CO" --radius 25

   With custom settings:
   $ python climb_analyzer.py -a "Boulder, CO" --radius 25 \\
       -u metric -t fiets -s paved -m 250

2. SINGLE REGION ANALYSIS
   Analyze entire state of Colorado:

   $ python climb_analyzer.py -r Colorado

   With abbreviations:
   $ python climb_analyzer.py -r CO -s paved -u imperial

3. BATCH MULTI-REGION ANALYSIS
   Analyze multiple New England states:

   $ python climb_analyzer.py -r "Vermont,New Hampshire,Maine"

   With cleanup after each region:
   $ python climb_analyzer.py -r "VT,NH,ME,MA,CT,RI" -X

   European countries:
   $ python climb_analyzer.py -r "Switzerland,Austria,Italy" -u metric -t fiets

4. DATA MANAGEMENT

   a. Download data without analysis:
      $ python climb_analyzer.py -D -r Vermont

   b. Update geographic boundaries:
      $ python climb_analyzer.py -U

   c. Delete checkpoints (resume from fresh):
      $ python climb_analyzer.py -C

   d. Delete OSM data (free up space):
      $ python climb_analyzer.py -P

   e. Delete elevation data:
      $ python climb_analyzer.py -E

   f. Run analysis and cleanup after:
      $ python climb_analyzer.py -r Colorado -X

5. ADVANCED COMBINATIONS

   a. Download and analyze with cleanup:
      $ python climb_analyzer.py -D -r "CO,UT" && \\
        python climb_analyzer.py -r "CO,UT" -X

   b. Update boundaries then analyze:
      $ python climb_analyzer.py -U && \\
        python climb_analyzer.py -r "Vermont,NH"

   c. Gravel climbs only, FIETS scoring:
      $ python climb_analyzer.py -r "Vermont" \\
        -s gravel -t fiets -m 200 -u metric

6. REGION AUTO-DETECTION

   The CLI automatically detects region types:
   - US States: Colorado, CO, Vermont, VT
   - Countries: Switzerland, Austria, Japan
   - Provinces: Ontario (Canada), Bavaria (Germany)

   You can mix in batch mode:
   $ python climb_analyzer.py -r "Colorado,California,Oregon"

   Or use abbreviations:
   $ python climb_analyzer.py -r "CO,CA,OR"

7. INTERACTIVE MODE

   Launch interactive menus:
   $ python climb_analyzer.py -i

   Or just:
   $ python climb_analyzer.py

8. DOCKER WRAPPER

   Use the convenient wrapper script:
   $ ./climb-analyzer -r Colorado
   $ ./climb-analyzer -a "Seattle, WA" --radius 50
   $ ./climb-analyzer -D -r "VT,NH,ME"

9. SCORING TYPES

   - basic:  grade × distance (default, simple)
   - fiets:  gradient-based (Fietsklim formula)
   - pdi:    difficulty index (comprehensive)

   Examples:
   $ python climb_analyzer.py -r CO -t basic -m 6000
   $ python climb_analyzer.py -r Switzerland -t fiets -m 250
   $ python climb_analyzer.py -r "VT,NH" -t pdi -m 1.0

10. TIPS & TRICKS

    a. Quick analysis with cleanup:
       $ python climb_analyzer.py -r CO -s paved -X

    b. Disable geocoding for speed:
       $ python climb_analyzer.py -r Vermont -g no

    c. Metric units with FIETS scoring:
       $ python climb_analyzer.py -r Switzerland -u metric -t fiets

    d. Download data in advance:
       $ python climb_analyzer.py -D -r "CO,UT,WY,MT,ID"
       $ python climb_analyzer.py -r CO  # Uses cached data

For more documentation, see:
  • CLI_REFACTOR_FINAL.md
  • DOCKER_SETUP.md
  • INSTALLATION.md
""")
```

## Implementation Files and Changes

### 1. climb_analyzer.py

**Lines 11125-11216: Replace argument parser**

```python
def main():
    """Main function to run the climb analyzer."""
    from batch_cleanup import get_cleanup_targets, perform_cleanup, prompt_cleanup
    from error_logger import ErrorLogger, LogRotator

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Climb Analyzer - Analyze road and trail climbs from OpenStreetMap",
        epilog="Use --help-extended for detailed examples and usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === ANALYSIS MODES (Mutually Exclusive) ===
    analysis_group = parser.add_mutually_exclusive_group()

    analysis_group.add_argument(
        "-a", "--address",
        type=str,
        metavar="ADDRESS",
        help="Address for radius-based analysis (e.g., 'Boulder, CO')"
    )

    analysis_group.add_argument(
        "-r", "--run-region",
        type=str,
        metavar="REGION",
        help="Analyze region(s). Single: 'Colorado' or Batch: 'VT,NH,ME'. Auto-detects type."
    )

    analysis_group.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive mode with menus (default if no arguments)"
    )

    # === ANALYSIS PARAMETERS ===
    parser.add_argument(
        "--radius",
        type=float,
        metavar="RADIUS",
        help="Search radius in miles/km (required with --address)"
    )

    parser.add_argument(
        "-s", "--surface-filter",
        type=str,
        default="all",
        choices=["paved", "gravel", "dirt", "all"],
        help="Surface filter: paved, gravel, dirt, or all [default: all]"
    )

    parser.add_argument(
        "--no-cycling-filter",
        action="store_true",
        help="Disable cycling accessibility filter"
    )

    parser.add_argument(
        "-u", "--units",
        type=str,
        default="imperial",
        choices=["metric", "imperial"],
        help="Unit system [default: imperial]"
    )

    parser.add_argument(
        "-t", "--score-type",
        type=str,
        default="basic",
        choices=["basic", "fiets", "pdi"],
        help="Scoring: basic (grade*dist), fiets (gradient), pdi (difficulty) [default: basic]"
    )

    parser.add_argument(
        "-m", "--min-score",
        type=float,
        help="Minimum climb score threshold"
    )

    parser.add_argument(
        "-g", "--geocoding",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="Enable reverse geocoding [default: yes]"
    )

    # === DATA MANAGEMENT ===
    data_group = parser.add_argument_group("Data Management")

    data_group.add_argument(
        "-U", "--update-geo-boundaries",
        action="store_true",
        help="Update country/state boundary data from sources"
    )

    data_group.add_argument(
        "-D", "--data-download",
        action="store_true",
        help="Download OSM and elevation data without running analysis"
    )

    data_group.add_argument(
        "-C", "--delete-checkpoints",
        action="store_true",
        help="Delete checkpoint files"
    )

    data_group.add_argument(
        "-P", "--delete-planet-data",
        action="store_true",
        help="Delete all OSM .pbf files and indices"
    )

    data_group.add_argument(
        "-E", "--delete-elevation-data",
        action="store_true",
        help="Delete all elevation dataset files"
    )

    data_group.add_argument(
        "-X", "--delete-data-on-complete",
        action="store_true",
        help="Delete OSM and elevation data after analysis completes"
    )

    # === HELP ===
    parser.add_argument(
        "--help-extended",
        action="store_true",
        help="Show extended help with examples"
    )

    args = parser.parse_args()

    # Handle extended help
    if args.help_extended:
        show_extended_help()
        return

    # Handle data management operations
    if args.update_geo_boundaries:
        run_update_geo_boundaries()
        return

    if args.delete_checkpoints:
        delete_checkpoints()
        return

    if args.delete_planet_data:
        delete_planet_data()
        return

    if args.delete_elevation_data:
        delete_elevation_data()
        return

    # Handle data download
    if args.data_download:
        if not args.run_region:
            print("❌ Error: --data-download requires --run-region to specify which regions")
            sys.exit(1)
        regions = parse_regions(args.run_region)
        download_data_for_regions(regions)
        return

    # Handle address-based analysis
    if args.address:
        if not args.radius:
            print("❌ Error: --address requires --radius to specify search radius")
            sys.exit(1)
        run_address_analysis(args)
        return

    # Handle region-based analysis
    if args.run_region:
        regions = parse_regions(args.run_region)

        if len(regions) > 1:
            # Batch mode
            run_batch_region_analysis(regions, args)
        else:
            # Single region
            run_single_region_analysis(regions[0], args)
        return

    # Default: Interactive mode
    run_interactive_mode(args)
```

### 2. New Helper Functions (add to climb_analyzer.py)

```python
def detect_region_type(region_name: str) -> tuple[str, str]:
    """[See implementation above]"""
    # ... [Full implementation from above]
    pass


def parse_regions(region_string: str) -> list[dict]:
    """[See implementation above]"""
    # ... [Full implementation from above]
    pass


def run_address_analysis(args):
    """Run address-based radius analysis."""
    print(f"\n🔄 Address Analysis: {args.address} (radius: {args.radius} {args.units})")

    # Call existing analyze_area() with address mode
    # ... existing address analysis code ...


def run_single_region_analysis(region: dict, args):
    """Run analysis for single region."""
    print(f"\n🔄 Single Region Analysis: {region['canonical_name']} ({region['type']})")

    # Call existing analyze_area() with region bounds
    # ... existing region analysis code ...

    # Handle cleanup if requested
    if args.delete_data_on_complete:
        cleanup_region_data(region)


def run_batch_region_analysis(regions: list[dict], args):
    """Run batch analysis for multiple regions."""
    print(f"\n🔄 Batch Analysis: {len(regions)} regions")

    for i, region in enumerate(regions, 1):
        print(f"\n[{i}/{len(regions)}] Processing: {region['canonical_name']}")
        run_single_region_analysis(region, args)


def download_data_for_regions(regions: list[dict]):
    """Download data for regions without analysis."""
    from data_manager import DataManager
    manager = DataManager()

    for region in regions:
        print(f"\nDownloading data for: {region['canonical_name']}")
        # Use DataManager to download
        # ... implementation ...


def run_update_geo_boundaries():
    """Update geographic boundary data."""
    import subprocess
    print("\n🔄 Updating geographic boundaries...")
    result = subprocess.run(["python3", "update_geo_definitions.py"])
    if result.returncode == 0:
        print("✅ Geographic boundaries updated successfully")
    else:
        print("❌ Update failed")
        sys.exit(1)


def delete_checkpoints():
    """Delete all checkpoint files."""
    from pathlib import Path
    import shutil

    confirm = input("\n⚠️  Delete all checkpoint files? [y/N]: ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return

    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir()
        print("✅ Deleted all checkpoints")
    else:
        print("   No checkpoint directory found")


def delete_planet_data():
    """Delete all OSM planet data."""
    from pathlib import Path

    confirm = input("\n⚠️  Delete all OSM .pbf files and indices? [y/N]: ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return

    planet_dir = Path("planet-osm")
    if not planet_dir.exists():
        print("   No planet-osm directory found")
        return

    deleted_count = 0
    for pbf_file in planet_dir.glob("*.pbf"):
        pbf_file.unlink()
        deleted_count += 1

        # Delete associated index files
        idx_file = pbf_file.with_suffix('.pbf.idx')
        if idx_file.exists():
            idx_file.unlink()

        # Delete rtree indices
        idx_dir = pbf_file.with_suffix('')
        if idx_dir.is_dir():
            import shutil
            shutil.rmtree(idx_dir)

    print(f"✅ Deleted {deleted_count} OSM files and indices")


def delete_elevation_data():
    """Delete all elevation data."""
    from pathlib import Path
    import shutil

    confirm = input("\n⚠️  Delete all elevation dataset files? [y/N]: ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return

    elevation_dir = Path("elevation_data")
    if elevation_dir.exists():
        shutil.rmtree(elevation_dir)
        elevation_dir.mkdir()
        print("✅ Deleted all elevation data")
    else:
        print("   No elevation_data directory found")


def cleanup_region_data(region: dict):
    """Clean up OSM and elevation data for a specific region."""
    from data_manager import DataManager
    manager = DataManager()

    is_state = region['type'] == 'state'
    manager.cleanup_region_data(region['canonical_name'], is_state=is_state)
    print(f"✅ Cleaned up data for {region['canonical_name']}")
```

### 3. setup_wizard.py

**Update print_next_steps() function (lines 1330-1363):**

```python
def print_next_steps():
    """Print next steps after setup."""
    print_header("Setup Complete!")

    print("Your climb analyzer is ready to use!\n")

    print("Quick Start:\n")

    print("1. Interactive mode:")
    print("   ./climb-analyzer\n")

    print("2. Address analysis:")
    print("   ./climb-analyzer -a 'Boulder, CO' --radius 25 -u imperial\n")

    print("3. Single region:")
    print("   ./climb-analyzer -r Colorado -s paved -u metric\n")

    print("4. Batch regions:")
    print("   ./climb-analyzer -r 'Vermont,NH,Maine' -s paved -X\n")

    print("5. Data management:")
    print("   ./climb-analyzer -D -r Vermont        # Download data")
    print("   ./climb-analyzer -U                   # Update boundaries")
    print("   ./climb-analyzer -C                   # Delete checkpoints")
    print("   ./climb-analyzer -P                   # Delete OSM data")
    print("   ./climb-analyzer -E                   # Delete elevation\n")

    print("6. Help:")
    print("   ./climb-analyzer --help               # Show help")
    print("   ./climb-analyzer --help-extended      # Extended help\n")

    print("📚 Documentation:")
    print("   • CLI_REFACTOR_FINAL.md - CLI reference")
    print("   • DOCKER_SETUP.md - Docker guide")
    print("   • INSTALLATION.md - Full setup guide\n")
```

**Remove geocoding prompts if present** (search for geocoding references in wizard)

### 4. Update bash wrapper: climb-analyzer

**Add new command mappings:**

```bash
case "${1:-run}" in
    # ... existing cases ...

    -U|--update-geo-boundaries)
        print_header "Updating Geographic Boundaries"
        check_docker
        exec $COMPOSE_CMD run --rm climb-analyzer python climb_analyzer.py -U
        ;;

    -D|--data-download)
        print_header "Downloading Data"
        check_docker
        shift
        exec $COMPOSE_CMD run --rm climb-analyzer python climb_analyzer.py -D "$@"
        ;;

    -C|--delete-checkpoints)
        print_header "Deleting Checkpoints"
        check_docker
        exec $COMPOSE_CMD run --rm climb-analyzer python climb_analyzer.py -C
        ;;

    -P|--delete-planet-data)
        print_header "Deleting OSM Data"
        check_docker
        exec $COMPOSE_CMD run --rm climb-analyzer python climb_analyzer.py -P
        ;;

    -E|--delete-elevation-data)
        print_header "Deleting Elevation Data"
        check_docker
        exec $COMPOSE_CMD run --rm climb-analyzer python climb_analyzer.py -E
        ;;

    # ... rest of cases ...
esac
```

## Interactive Menu Updates

### Do we need to update interactive menus?

**Answer: NO major changes needed.** The interactive menus in `geographic_menu.py` and `climb_analyzer.py` can remain largely unchanged because:

1. They operate independently from CLI arguments
2. CLI arguments bypass the interactive menus
3. The only potential enhancement would be to expose the same cleanup options in menus

**Optional Enhancement:** Add cleanup prompt to interactive menu after analysis:

```python
# In interactive mode, after analysis completes
if analysis_successful:
    cleanup = input("\n🗑️  Delete OSM and elevation data for this region? [y/N]: ")
    if cleanup.lower() == 'y':
        cleanup_region_data(region)
```

## Testing Plan

### Test Cases

```bash
# 1. Address analysis
python climb_analyzer.py -a "Boulder, CO" --radius 25
python climb_analyzer.py -a "Seattle, WA" --radius 50 -u metric -t fiets

# 2. Single region
python climb_analyzer.py -r Colorado
python climb_analyzer.py -r VT -s paved -u metric
python climb_analyzer.py -r Switzerland -t fiets -m 250

# 3. Batch regions
python climb_analyzer.py -r "Vermont,New Hampshire,Maine"
python climb_analyzer.py -r "CO,UT,WY" -s paved -X
python climb_analyzer.py -r "Switzerland,Austria,Italy" -u metric

# 4. Data download
python climb_analyzer.py -D -r Vermont
python climb_analyzer.py -D -r "CO,CA,OR"

# 5. Data management
python climb_analyzer.py -U
python climb_analyzer.py -C
python climb_analyzer.py -P
python climb_analyzer.py -E

# 6. Interactive
python climb_analyzer.py
python climb_analyzer.py -i

# 7. Edge cases
python climb_analyzer.py -r "Colorado,Switzerland"  # Mixed types
python climb_analyzer.py -a "Boulder"  # Missing radius
python climb_analyzer.py -D  # Missing region
```

## Summary of Single-Letter Flags

| Flag | Long Form | Purpose |
|------|-----------|---------|
| `-a` | `--address` | Address for radius analysis |
| `-r` | `--run-region` | Run region analysis (single or batch) |
| `-i` | `--interactive` | Launch interactive mode |
| `-s` | `--surface-filter` | Surface type filter |
| `-u` | `--units` | Unit system |
| `-t` | `--score-type` | Scoring algorithm |
| `-m` | `--min-score` | Minimum score threshold |
| `-g` | `--geocoding` | Enable/disable geocoding |
| `-U` | `--update-geo-boundaries` | Update boundaries |
| `-D` | `--data-download` | Download data only |
| `-C` | `--delete-checkpoints` | Delete checkpoints |
| `-P` | `--delete-planet-data` | Delete OSM data |
| `-E` | `--delete-elevation-data` | Delete elevation data |
| `-X` | `--delete-data-on-complete` | Cleanup after analysis |

## Implementation Priority

### Phase 1: Core CLI (High Priority)
1. ✅ Implement new argument parser
2. ✅ Add region auto-detection logic
3. ✅ Implement `-r/--run-region` with batch support
4. ✅ Implement `-a/--address` separate from subregion
5. ✅ Add `-X/--delete-data-on-complete`

### Phase 2: Data Management (Medium Priority)
6. ✅ Implement data cleanup functions (`-C`, `-P`, `-E`)
7. ✅ Implement `-D/--data-download`
8. ✅ Implement `-U/--update-geo-boundaries`

### Phase 3: Documentation (Medium Priority)
9. ✅ Update help text and extended help
10. ✅ Update setup wizard quickstart output
11. ✅ Update bash wrapper script

### Phase 4: Testing (High Priority)
12. 🧪 Test all argument combinations
13. 🧪 Test region auto-detection with various inputs
14. 🧪 Test batch mode with mixed region types
15. 🧪 Test data management operations

## Questions Answered

1. ✅ **`--scope` handling:** Now `--address` (separate) and `--run-region` (auto-detects type)
2. ✅ **Region auto-detection:** Yes, using `geo_definitions.py`
3. ✅ **Batch support:** Built into `-r/--run-region` with comma-separated values
4. ✅ **Single-letter flags:** Added for quick typing
5. ✅ **Delete on complete:** `-X/--delete-data-on-complete` flag
6. ✅ **Interactive menus:** No major updates needed, they work independently
