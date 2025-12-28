#!/usr/bin/env python3
"""Validate paved road climbs from output files."""

import pandas as pd
from validate_climbs import (
    extract_climb_data, detect_units, validate_climb_reference,
    KNOWN_CLIMB_REFERENCE, FEET_TO_METERS, MILES_TO_KM
)

def validate_paved_climbs(xlsx_path, max_climbs=10):
    """Validate top paved road climbs from an output file."""
    print(f"Loading {xlsx_path}...")
    df = pd.read_excel(xlsx_path)
    units = detect_units(df)
    print(f"Loaded {len(df):,} climbs, units: {units}")

    # Filter for paved roads only
    paved = df[
        (df['Surface'].isin(['asphalt', 'paved'])) &
        (df['Highway Type'].isin(['primary', 'secondary', 'tertiary', 'unclassified', 'residential']))
    ]
    print(f"Paved road climbs: {len(paved):,}")

    # Get top climbs by FIETS score
    top = paved.nlargest(max_climbs, 'FIETS Score')

    print()
    print("=" * 80)
    print("PAVED ROAD CLIMB VALIDATION")
    print("=" * 80)

    results = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        climb = extract_climb_data(row, units)

        print(f"\n{i}. {climb.name}")
        print(f"   Location: {climb.city}, {climb.state}")
        print(f"   Analyzer: {climb.elev_gain_m:.0f}m / {climb.distance_km:.2f}km / {climb.avg_grade:.1f}%")

        result = validate_climb_reference(climb, tolerance_pct=15.0)

        if result.source == 'none':
            print(f"   Reference: No match found")
            results.append((climb.name, None, None, None, "NO MATCH"))
        else:
            print(f"   Source: {result.source}")
            print(f"   Reference: {result.external_elev_gain_m:.0f}m / {result.external_distance_km:.2f}km / {result.external_avg_grade:.1f}%")
            print(f"   Distance diff: {result.distance_diff_pct:.1f}%")
            print(f"   Elevation diff: {result.elev_diff_pct:.1f}%")
            print(f"   Grade diff: {result.grade_diff_pct:.1f}%")
            status = "PASS" if result.passed else "FAIL"
            print(f"   Result: {status}")
            results.append((climb.name, result.distance_diff_pct, result.elev_diff_pct, result.grade_diff_pct, status))

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    matched = [r for r in results if r[4] != "NO MATCH"]
    passed = [r for r in results if r[4] == "PASS"]
    failed = [r for r in results if r[4] == "FAIL"]
    no_match = [r for r in results if r[4] == "NO MATCH"]

    print(f"Validated: {len(results)} climbs")
    print(f"  Matched: {len(matched)}")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  No reference: {len(no_match)}")

    if matched:
        avg_dist = sum(r[1] for r in matched) / len(matched)
        avg_elev = sum(r[2] for r in matched) / len(matched)
        avg_grade = sum(r[3] for r in matched) / len(matched)
        print(f"\nAverage differences (matched climbs):")
        print(f"  Distance: {avg_dist:.1f}%")
        print(f"  Elevation: {avg_elev:.1f}%")
        print(f"  Grade: {avg_grade:.1f}%")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        xlsx_path = sys.argv[1]
    else:
        xlsx_path = "../output/Hawaii_climbs_all-surfaces_all-access_imperial_2025-12-18_v2.2.2_e0000.xlsx"

    validate_paved_climbs(xlsx_path)
