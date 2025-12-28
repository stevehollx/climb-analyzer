#!/usr/bin/env python3
"""Test validation of known climbs with reference data."""

import pandas as pd
from validate_climbs import (
    extract_climb_data, detect_units, validate_climb_reference,
    KNOWN_CLIMB_REFERENCE, FEET_TO_METERS, MILES_TO_KM
)

# Load Hawaii output
print("Loading Hawaii output file...")
df = pd.read_excel('../output/Hawaii_climbs_all-surfaces_all-access_imperial_2025-12-18_v2.2.2_e0000.xlsx')
units = detect_units(df)
print(f"Loaded {len(df):,} climbs, units: {units}\n")

# Test known Hawaii climbs
known_hawaii_climbs = [
    "Mauna Kea Access Road",
    "Kalalau Trail",
    "Ainapo Trail",
]

print("=" * 80)
print("KNOWN CLIMB VALIDATION")
print("=" * 80)

for climb_name in known_hawaii_climbs:
    print(f"\nSearching for: {climb_name}")

    # Find climb in output
    matches = df[df['Street Name'].str.contains(climb_name, case=False, na=False)]

    if len(matches) == 0:
        print(f"  NOT FOUND in analyzer output")
        continue

    row = matches.iloc[0]
    climb = extract_climb_data(row, units)

    print(f"  Found: {climb.name}")
    print(f"  Analyzer: {climb.elev_gain_m:.0f}m / {climb.distance_km:.2f}km / {climb.avg_grade:.1f}%")

    # Validate against reference
    result = validate_climb_reference(climb, tolerance_pct=15.0)

    if result.source == 'none':
        print(f"  No reference data available")
    else:
        print(f"  Source: {result.source}")
        print(f"  Reference: {result.external_elev_gain_m:.0f}m / {result.external_distance_km:.2f}km / {result.external_avg_grade:.1f}%")
        print(f"  Distance diff: {result.distance_diff_pct:.1f}%")
        print(f"  Elevation diff: {result.elev_diff_pct:.1f}%")
        print(f"  Grade diff: {result.grade_diff_pct:.1f}%")
        print(f"  Result: {'PASS' if result.passed else 'FAIL'}")

print("\n" + "=" * 80)
print("REFERENCE DATA AVAILABLE:")
print("=" * 80)
for name, data in KNOWN_CLIMB_REFERENCE.items():
    print(f"  {name}: {data['distance_km']}km, {data['elev_gain_m']}m, {data['avg_grade']}% ({data['source']})")
