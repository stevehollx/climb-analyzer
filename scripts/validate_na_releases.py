#!/usr/bin/env python3
"""
Print a validation table for all N America regions on GitHub.
For each release shows: xlsx present, sqlite.gz present, elevation datasets used.

Usage:
    python3 scripts/validate_na_releases.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "stevehollx/global-road-and-trail-climbs"
CHECKPOINT_DIR = Path("data/checkpoint_data")

# All expected N America regions (US states + Canadian provinces/territories + others)
NA_REGIONS = {
    # US states
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "district-of-columbia", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "new-hampshire",
    "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "puerto-rico", "rhode-island",
    "south-carolina", "south-dakota", "tennessee", "texas", "us-virgin-islands",
    "utah", "vermont", "virginia", "washington", "west-virginia", "wisconsin",
    "wyoming",
    # Canadian provinces
    "british-columbia", "alberta", "saskatchewan", "manitoba", "ontario",
    "quebec", "new-brunswick", "nova-scotia", "newfoundland-and-labrador",
    "prince-edward-island",
    # Canadian territories
    "yukon", "northwest-territories", "nunavut",
    # Other N America
    "mexico", "greenland",
}


def gh(args):
    result = subprocess.run(["gh"] + args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def tag_to_state_key(tag):
    return re.sub(r"-v\d+\.\d+\.\d+$", "", tag)


def find_datasets_used(state_key):
    """Read datasets_used.json from the most recent matching checkpoint."""
    prefix = "_".join(w.capitalize() for w in state_key.split("-"))
    matches = sorted(
        CHECKPOINT_DIR.glob(f"{prefix}_all_region_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return None
    ds_file = matches[0] / "datasets_used.json"
    if not ds_file.exists():
        return None
    try:
        return json.loads(ds_file.read_text())
    except Exception:
        return None


US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "district-of-columbia", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "new-hampshire",
    "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "puerto-rico", "rhode-island",
    "south-carolina", "south-dakota", "tennessee", "texas", "us-virgin-islands",
    "utah", "vermont", "virginia", "washington", "west-virginia", "wisconsin",
    "wyoming",
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--us-only", action="store_true",
                    help="Only validate US states (50 + DC + territories)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any region is missing xlsx or sqlite.gz")
    args = ap.parse_args()

    print("Fetching releases from GitHub...")
    releases_raw = json.loads(gh(["api", f"repos/{REPO}/releases?per_page=100"]))

    target_set = US_STATES if args.us_only else NA_REGIONS
    rows = []
    for r in releases_raw:
        tag = r.get("tag_name", "")
        state_key = tag_to_state_key(tag)
        if state_key not in target_set:
            continue

        assets = [a["name"] for a in r.get("assets", [])]
        has_xlsx = any(a.endswith(".xlsx") for a in assets)
        has_sqlite_gz = any(
            a.endswith(".sqlite.gz") or re.match(r".*\.sqlite\.gz\.\d{3}$", a)
            for a in assets
        )
        datasets = find_datasets_used(state_key)
        ds_str = ", ".join(datasets) if datasets else "unknown"

        if has_xlsx and has_sqlite_gz:
            status = "OK"
        elif has_xlsx and not has_sqlite_gz:
            status = "MISSING sqlite.gz"
        elif not has_xlsx:
            status = "MISSING xlsx"
        else:
            status = "MISSING both"

        rows.append({
            "region": state_key,
            "draft": "draft" if r.get("draft") else "pub",
            "xlsx": "YES" if has_xlsx else "NO",
            "sqlite_gz": "YES" if has_sqlite_gz else "NO",
            "datasets": ds_str,
            "status": status,
        })

    # Expected primary dataset per user reference table
    expected_primary = {
        "alaska": "arctic32m",
        "greenland": "arctic32m",
        "yukon": "arctic32m",
        "northwest-territories": "arctic32m",
        "nunavut": "arctic32m",
    }
    for r in rows:
        exp = expected_primary.get(r["region"])
        if exp and r["datasets"] != "unknown":
            first_ds = r["datasets"].split(",")[0].strip()
            if first_ds != exp:
                r["status"] = f"WRONG PRIMARY (expected {exp}, got {first_ds})"

    # Sort: failures first, then alphabetically
    rows.sort(key=lambda r: (r["status"] == "OK", r["region"]))

    # Print table
    col_w = {
        "region": max(len(r["region"]) for r in rows) + 2,
        "draft": 6,
        "xlsx": 5,
        "sqlite_gz": 10,
        "datasets": max(len(r["datasets"]) for r in rows) + 2,
        "status": max(len(r["status"]) for r in rows) + 2,
    }

    header = (
        f"{'Region':<{col_w['region']}}"
        f"{'Pub?':<{col_w['draft']}}"
        f"{'xlsx':<{col_w['xlsx']}}"
        f"{'sqlite.gz':<{col_w['sqlite_gz']}}"
        f"{'Datasets Used':<{col_w['datasets']}}"
        f"Status"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)

    ok_count = 0
    fail_count = 0
    for r in rows:
        line = (
            f"{r['region']:<{col_w['region']}}"
            f"{r['draft']:<{col_w['draft']}}"
            f"{r['xlsx']:<{col_w['xlsx']}}"
            f"{r['sqlite_gz']:<{col_w['sqlite_gz']}}"
            f"{r['datasets']:<{col_w['datasets']}}"
            f"{r['status']}"
        )
        print(line)
        if r["status"] == "OK":
            ok_count += 1
        else:
            fail_count += 1

    print(sep)
    print(f"\nSummary: {ok_count} OK, {fail_count} with issues")

    # Missing from GitHub entirely
    released_keys = {r["region"] for r in rows}
    missing = sorted(target_set - released_keys)
    if missing:
        print(f"\nNo release found for: {', '.join(missing)}")

    if args.strict and (fail_count > 0 or missing):
        sys.exit(1)
    if not args.strict and fail_count > 0:
        # Non-strict: warn but still exit 0 so the queue continues.
        sys.exit(0)


if __name__ == "__main__":
    main()
