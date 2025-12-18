# Climb Analyzer Test Suite

Comprehensive test suite for validating the climb analyzer across multiple regions, input modes, and edge cases.

## Overview

This test suite contains **19 test scenarios** covering:
- CLI and interactive input modes
- Single region and batch (multi-region) analysis
- Address-based radius searches
- Custom bounding box analysis
- Cross-region/cross-country climb merging
- Region disambiguation (Georgia US vs Georgia Europe)

## Quick Start

```bash
# List all available tests
python tests/run_tests.py --list

# Run all tests
python tests/run_tests.py

# Run quick validation tests only (early exit)
python tests/run_tests.py --early-exit-only

# Start from a specific test
python tests/run_tests.py --start 5

# Run specific tests
python tests/run_tests.py --test 3,7,12

# Skip long-running tests
python tests/run_tests.py --skip 8,9,10
```

## Dependencies

```bash
pip install pexpect>=4.8.0        # Interactive test automation
pip install beautifulsoup4>=4.9.0 # Strava page scraping
pip install requests>=2.25.0      # HTTP requests
pip install pandas>=1.3.0         # XLSX validation
pip install openpyxl>=3.0.0       # Excel file reading
```

## Test Cases

### Full Analysis Tests (1-13)

| ID | Region | Mode | Description |
|----|--------|------|-------------|
| 1 | Georgia (Europe) | CLI | European country with `--allow-cross-country-merge` |
| 2 | Luxembourg | Interactive | Small European country via menu navigation |
| 3 | Hawaii | CLI | US state with notable climbs (Mauna Kea, Crater Rd) |
| 4 | New York | Interactive | US state with paved surface filter |
| 5 | Bristol (UK) | Interactive | UK city via nested menu (Europe > UK > England > Bristol) |
| 6 | Worcestershire | CLI | UK county with paved filter |
| 7 | Isle of Wight | CLI | UK island with cycling filter |
| 8 | Rutland, Leicestershire | CLI Batch | Cross-region merge test (Cold Overton Road) |
| 9 | Luxembourg, Belgium | CLI Batch | Cross-country merge test (Wemperhardt) |
| 10 | Oregon, Idaho | CLI Batch | Cross-state merge test (US Route 95 / ION Highway) |
| 11 | China (Lijiang) | CLI Bbox | Custom bounding box (Tiger Leaping Gorge area) |
| 12 | Cleveland SC 30mi | CLI Address | Address-based 30-mile radius search |
| 13 | Cleveland SC 30mi | Interactive | Same as 12, via interactive mode |

### Early Exit Tests (14-19)

These tests verify region selection without running full analysis:

| ID | Region | Mode | Purpose |
|----|--------|------|---------|
| 14 | Georgia (Europe) | Interactive | Verify European Georgia selection |
| 15 | Georgia (Europe) | CLI | Verify CLI picks European Georgia by default |
| 16 | Georgia (US) | CLI | Test disambiguation handling |
| 17 | Georgia (US) | Interactive | Verify US state Georgia selection via menu |
| 18 | Georgia, Luxembourg | CLI Batch | Batch with two European regions |
| 19 | us/Georgia, Luxembourg | CLI Batch | Mixed batch with qualifier prefix |

## File Structure

```
tests/
├── run_tests.py              # Main test orchestrator
├── test_colors.py            # ANSI color output utilities
├── validation_climbs.json    # Test case definitions & expected climbs
├── interactive_sequences.py  # pexpect menu navigation sequences
├── result_validator.py       # XLSX output validation
├── strava_scraper.py         # Strava segment data fetcher
└── README_CLIMB_TESTS.md     # This file
```

## Validation Methods

### 1. Output Structure Validation

Validates the 26-column XLSX output format:

| Column | Description |
|--------|-------------|
| rank | Climb ranking by fiets score |
| climb_name | Name of the climb |
| osm_way_ids | OpenStreetMap way IDs |
| fiets_score | Difficulty score |
| climb_category | HC, 1, 2, 3, 4, or 5 |
| total_length_km | Total climb distance |
| total_elevation_gain_m | Elevation gain in meters |
| avg_gradient_percent | Average gradient |
| max_gradient_percent | Maximum gradient |
| start_lat, start_lon | Start coordinates |
| end_lat, end_lon | End coordinates |
| surface_type | Road surface |
| road_classification | Road type |
| elevation_profile | JSON elevation data |
| gradient_profile | JSON gradient data |
| min_elevation_m | Minimum elevation |
| max_elevation_m | Maximum elevation |
| steepest_100m_gradient | Steepest 100m section |
| steepest_500m_gradient | Steepest 500m section |
| steepest_1km_gradient | Steepest 1km section |
| osm_link | OpenStreetMap link |
| google_maps_link | Google Maps link |
| strava_link | Strava segment link |
| regions | Source regions |

### 2. Specific Climb Validation

Each test defines expected climbs that must appear in output:

```json
{
  "name": "Mauna Kea Access Road",
  "alt_names": ["Mauna Kea Summit Road", "Saddle Road"],
  "strava_segment_id": "612474",
  "expected_elev_gain_m": 4200,
  "expected_length_km": 42,
  "expected_avg_grade": 10.0
}
```

### 3. Strava Validation

For climbs with Strava segment IDs, the test suite fetches public Strava data and compares:
- Distance (km)
- Elevation gain (m)
- Average gradient (%)

**Tolerance:** 15% difference allowed (configurable in `validation_climbs.json`)

### 4. Cross-Region Merge Validation

Tests 8-10 verify that climbs spanning multiple regions are properly merged:
- Checks `regions` column contains all expected regions
- Verifies multiple `osm_way_ids` (indicates merged segments)

## CLI Options

```
usage: run_tests.py [-h] [--start N] [--test IDS] [--skip IDS]
                    [--early-exit-only] [--verbose] [--no-color]
                    [--continue-on-error] [--list]

Options:
  --start N           Start from test number N
  --test IDS          Run specific tests (comma-separated, e.g., "3,7,12")
  --skip IDS          Skip specific tests (comma-separated, e.g., "8,9,10")
  --early-exit-only   Only run early exit tests (14-19)
  --verbose, -v       Verbose output
  --no-color          Disable colored output
  --continue-on-error Continue running tests after critical errors
  --list              List all tests without running them
```

## Output Format

The test runner uses colored output:
- `[✓ PASS]` Green - Test passed
- `[✗ FAIL]` Red - Test failed
- `[○ SKIP]` Yellow - Test skipped
- `[▶ RUN ]` Blue - Test running
- `[⚠ WARN]` Yellow - Warning
- `[✗ ERROR]` Red - Error

### Example Output

```
╔══════════════════════════════════════════════════════════════╗
║              CLIMB ANALYZER TEST SUITE                       ║
╚══════════════════════════════════════════════════════════════╝

[ℹ INFO] Running 19 tests: [1, 2, 3, ...]
[ℹ INFO] Started at: 2025-12-17 10:30:00
────────────────────────────────────────────────────────────────

[▶ RUN ] Test 1: Georgia (Europe) - CLI
[✓ PASS] Test 1: Georgia (Europe) - CLI (2h 15m)
       847 climbs found

[▶ RUN ] Test 2: Luxembourg - Interactive
[✓ PASS] Test 2: Luxembourg - Interactive (45m 30s)
       156 climbs found

...

════════════════════════════════════════════════════════════════
╔══════════════════════════════════════════════════════════════╗
║                    TEST SUMMARY                              ║
╚══════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────┐
│  Total tests: 19                                             │
│  Duration: 8h 45m                                            │
├──────────────────────────────────────────────────────────────┤
│  Passed:  17                                                 │
│  Failed:  1                                                  │
│  Skipped: 1                                                  │
│  Errors:  0                                                  │
└──────────────────────────────────────────────────────────────┘

ALL TESTS PASSED
```

## Runtime Expectations

| Test Type | Typical Duration |
|-----------|------------------|
| Early exit (14-19) | 1-2 minutes each |
| Small regions (Luxembourg, Isle of Wight) | 30-60 minutes |
| Medium regions (Hawaii, Bristol) | 1-3 hours |
| Large regions (New York, Oregon+Idaho) | 4-10 hours |
| Batch tests (8-10) | 2-6 hours |

**Note:** Tests have no timeout. Large regions can take 8-10+ hours to complete.

## Adding New Tests

1. Add test definition to `validation_climbs.json`:
```json
{
  "test_20": {
    "test_name": "New Test - CLI",
    "region": "RegionName",
    "description": "Description of test",
    "climbs": [
      {
        "name": "Expected Climb Name",
        "alt_names": ["Alternative", "Names"],
        "strava_segment_id": "12345",
        "expected_elev_gain_m": 500,
        "expected_length_km": 10,
        "expected_avg_grade": 5.0
      }
    ]
  }
}
```

2. Add CLI command to `interactive_sequences.py`:
```python
CLI_COMMANDS = {
    ...
    20: ["./climb-analyzer", "-r", "RegionName"],
}
```

3. Or add interactive sequence for menu-based tests:
```python
INTERACTIVE_SEQUENCES = {
    ...
    20: InteractiveSequence(
        test_id=20,
        description="New test via interactive",
        steps=[
            MenuStep(expect_pattern=PATTERNS['select_mode'], send_response="1"),
            # ... more steps
        ]
    ),
}
```

## Troubleshooting

### "pexpect not installed"
Interactive tests (2, 4, 5, 13, 14, 17) require pexpect:
```bash
pip install pexpect
```

### "No output file found"
- Check that the analysis completed successfully
- Verify the output directory (`output/`) contains XLSX files
- Check region name normalization (spaces vs underscores)

### Strava validation fails
- Strava may rate-limit requests - wait and retry
- Segment may have been removed or made private
- Tolerance can be adjusted in `validation_climbs.json` metadata

### Test hangs indefinitely
- Large regions can take many hours - check progress messages
- Use `--skip` to skip known long-running tests
- Use `Ctrl+C` to abort and resume with `--start N`
