#!/usr/bin/env python3
"""
Batch Cross-Region Climb Merger

This script merges climbs across multiple region files (XLSX/CSV) by automatically
detecting adjacent regions and calling the existing cross-region merge logic.

REUSES EXISTING CODE:
    This is a wrapper around scripts/merge_cross_region_climbs.py
    All merge logic, boundary checks, and adjacency detection use the existing
    implementations. No new merge analysis code is created.

Key Features:
    - Accepts wildcard patterns (e.g., "output/state_*.xlsx")
    - Automatically detects which region pairs are adjacent (10km buffer)
    - Only merges adjacent regions (prevents Idaho + Florida merges)
    - Reuses all existing merge logic and boundary checks

Usage:
    # Merge all state files (only adjacent pairs will merge)
    ./climb-analyzer --merge-regions "output/state_*.xlsx"

    # Merge specific regions that you know are adjacent
    ./climb-analyzer --merge-regions "output/Idaho*.xlsx" "output/Oregon*.xlsx"

    # With custom threshold and auto-replace
    python scripts/batch_merge_regions.py "output/*.xlsx" --threshold 0.5 --auto-replace

Examples:
    # Adjacent US states (will merge where they border)
    ./climb-analyzer --merge-regions "output/Vermont*.xlsx" "output/NewHampshire*.xlsx"

    # Split country files (will merge if adjacent)
    ./climb-analyzer --merge-regions "output/France_North*.xlsx" "output/France_South*.xlsx"

    # Batch mode: checks all pairs, only merges adjacent
    ./climb-analyzer --merge-regions "output/US_*.xlsx"

Requirements:
    - Files must be in XLSX or CSV format
    - Files must have Latitude, Longitude, and Street Name columns
    - Docker container must be running (auto-wrapper available)
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple, Set
from itertools import combinations

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing merge functions (REUSE, don't reimplement)
from scripts.merge_cross_region_climbs import (
    load_climb_file,
    calculate_region_bbox,
    regions_are_adjacent,
    bbox_distance_km,
    find_and_merge_cross_region_climbs,
    replace_climbs_in_files,
)

# Import formatting utilities
from climb_analyzer.utils.formatting import (
    print_banner, print_header, print_success, print_warning,
    print_info, print_error, print_list_item, print_separator, STANDARD_WIDTH
)


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """
    Expand wildcard patterns to actual file paths.

    Args:
        patterns: List of file patterns (may include wildcards)

    Returns:
        List of expanded file paths (absolute paths)

    Raises:
        FileNotFoundError: If no files match the patterns
    """
    expanded_files = []

    for pattern in patterns:
        # Expand wildcards
        matches = glob.glob(pattern, recursive=False)

        if not matches:
            # Try treating as literal path (no wildcards)
            if Path(pattern).exists():
                matches = [pattern]
            else:
                print_warning(f"No files match pattern: {pattern}")
                continue

        # Convert to absolute paths
        for match in matches:
            abs_path = str(Path(match).absolute())
            if abs_path not in expanded_files:  # Avoid duplicates
                expanded_files.append(abs_path)

    if not expanded_files:
        raise FileNotFoundError(f"No files found matching patterns: {patterns}")

    return sorted(expanded_files)


def validate_files(file_paths: List[str]) -> List[str]:
    """
    Validate that files exist and are readable.

    Args:
        file_paths: List of file paths to validate

    Returns:
        List of valid file paths

    Raises:
        ValueError: If any files are invalid
    """
    valid_files = []
    invalid_files = []

    for file_path in file_paths:
        path = Path(file_path)

        # Check existence
        if not path.exists():
            invalid_files.append(f"{file_path} (does not exist)")
            continue

        # Check file type
        if path.suffix.lower() not in ['.xlsx', '.csv']:
            invalid_files.append(f"{file_path} (not XLSX/CSV)")
            continue

        # Check readable
        if not os.access(path, os.R_OK):
            invalid_files.append(f"{file_path} (not readable)")
            continue

        valid_files.append(file_path)

    if invalid_files:
        print_error("Invalid files detected:")
        for invalid in invalid_files:
            print_list_item(f"✗ {invalid}")
        raise ValueError(f"Found {len(invalid_files)} invalid file(s)")

    return valid_files


def find_adjacent_pairs(file_paths: List[str], buffer_km: float = 10.0) -> List[Tuple[str, str, float]]:
    """
    Find all pairs of files with adjacent regions.

    Uses existing regions_are_adjacent() logic from merge_cross_region_climbs.py

    Args:
        file_paths: List of file paths
        buffer_km: Maximum distance in km to consider regions adjacent

    Returns:
        List of tuples: (file1, file2, distance_km) for adjacent pairs only
    """
    print_header("Analyzing Region Adjacencies")
    print_info(f"Checking {len(file_paths)} file(s) for adjacent region pairs...")
    print_info(f"Adjacency threshold: {buffer_km} km")
    print()

    adjacent_pairs = []
    non_adjacent_count = 0

    # Check all possible pairs
    total_pairs = len(list(combinations(file_paths, 2)))
    checked = 0

    for file1, file2 in combinations(file_paths, 2):
        checked += 1

        try:
            # Load files and calculate bounding boxes (REUSE existing code)
            df1 = load_climb_file(file1)
            df2 = load_climb_file(file2)

            bbox1 = calculate_region_bbox(df1)
            bbox2 = calculate_region_bbox(df2)

            # Check adjacency (REUSE existing code)
            distance = bbox_distance_km(bbox1, bbox2)
            are_adjacent = regions_are_adjacent(bbox1, bbox2, buffer_km=buffer_km)

            file1_name = Path(file1).stem
            file2_name = Path(file2).stem

            if are_adjacent:
                adjacent_pairs.append((file1, file2, distance))
                print_success(f"✓ Adjacent: {file1_name} ↔ {file2_name} ({distance:.1f} km)")
            else:
                non_adjacent_count += 1
                if distance < 100:  # Only show close non-adjacent pairs
                    print_info(f"  Not adjacent: {file1_name} ↔ {file2_name} ({distance:.1f} km)")

        except Exception as e:
            print_warning(f"Error checking {Path(file1).name} ↔ {Path(file2).name}: {e}")
            continue

    print()
    print_separator()
    print_info(f"Checked {total_pairs} pair(s): {len(adjacent_pairs)} adjacent, {non_adjacent_count} non-adjacent")
    print_separator()
    print()

    return adjacent_pairs


def merge_adjacent_regions(
    adjacent_pairs: List[Tuple[str, str, float]],
    distance_threshold_km: float = 0.5,
    auto_replace: bool = False
) -> int:
    """
    Merge climbs for all adjacent region pairs.

    Uses existing find_and_merge_cross_region_climbs() logic.

    Args:
        adjacent_pairs: List of (file1, file2, distance) tuples
        distance_threshold_km: Distance threshold for climb matching
        auto_replace: Whether to auto-replace without prompting

    Returns:
        Number of region pairs successfully merged
    """
    if not adjacent_pairs:
        print_warning("No adjacent region pairs found - nothing to merge")
        return 0

    print_header(f"Merging {len(adjacent_pairs)} Adjacent Region Pair(s)")
    print()

    successful_merges = 0
    total_climbs_merged = 0

    for idx, (file1, file2, distance) in enumerate(adjacent_pairs, 1):
        file1_name = Path(file1).stem
        file2_name = Path(file2).stem

        print_separator()
        print_info(f"[{idx}/{len(adjacent_pairs)}] Merging: {file1_name} ↔ {file2_name}")
        print_info(f"           Region distance: {distance:.1f} km")
        print()

        try:
            # Call existing merge function (REUSE existing code)
            merged_climbs = find_and_merge_cross_region_climbs(
                file1, file2, distance_threshold_km
            )

            if merged_climbs:
                total_climbs_merged += len(merged_climbs)

                # Ask user about replacement (unless auto-replace enabled)
                if auto_replace:
                    should_replace = True
                    print_info("(Auto-replace mode enabled)")
                else:
                    response = input(f"\n  Replace climbs in original files? [y/N]: ").strip().lower()
                    should_replace = response in ['y', 'yes']

                if should_replace:
                    replace_climbs_in_files(merged_climbs, file1, file2)
                    successful_merges += 1
                else:
                    print_info("Skipped - no changes made to original files")
            else:
                print_info("No cross-boundary climbs found for this pair")

        except Exception as e:
            print_error(f"Failed to merge: {e}")
            continue

        print()

    print_separator()
    print()
    print_header("Merge Summary")
    print_success(f"Successfully merged {successful_merges}/{len(adjacent_pairs)} region pair(s)")
    print_info(f"Total cross-boundary climbs merged: {total_climbs_merged}")
    print()

    return successful_merges


def batch_merge_regions(file_paths: List[Path], distance_threshold_km: float = 0.5,
                       buffer_km: float = 10.0, auto_replace: bool = True) -> int:
    """
    Batch merge climbs across multiple adjacent region files.

    This is a programmatic API for use by climb-analyzer CLI.
    For command-line usage, use main() instead.

    Args:
        file_paths: List of Path objects to .xlsx files
        distance_threshold_km: Max distance between climb endpoints for merging
        buffer_km: Distance buffer for adjacency detection
        auto_replace: Auto-replace climbs without prompting

    Returns:
        Number of successful merges
    """
    # Convert Path objects to strings
    file_strs = [str(f) for f in file_paths]

    # Validate files
    print_info("Validating files...")
    valid_files = validate_files(file_strs)
    print_success(f"All {len(valid_files)} file(s) are valid\n")

    if len(valid_files) < 2:
        print_warning("Need at least 2 files to merge")
        return 0

    # Find adjacent pairs
    adjacent_pairs = find_adjacent_pairs(valid_files, buffer_km=buffer_km)

    if not adjacent_pairs:
        print_warning("No adjacent region pairs found - nothing to merge")
        return 0

    # Display what will be merged
    print_header(f"Found {len(adjacent_pairs)} Adjacent Region Pair(s)")
    for file1, file2, distance in adjacent_pairs:
        print_list_item(f"• {Path(file1).stem} ↔ {Path(file2).stem} ({distance:.1f} km)")
    print()

    # Merge adjacent regions
    successful = merge_adjacent_regions(
        adjacent_pairs,
        distance_threshold_km=distance_threshold_km,
        auto_replace=auto_replace
    )

    return successful


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch merge climbs across multiple adjacent region files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all matching files (only adjacent pairs)
  python batch_merge_regions.py "output/state_*.xlsx"

  # Merge specific adjacent regions
  python batch_merge_regions.py "output/Idaho*.xlsx" "output/Oregon*.xlsx"

  # With custom settings
  python batch_merge_regions.py "output/*.xlsx" --threshold 1.0 --buffer 15 --auto-replace

Notes:
  - Only adjacent regions (within buffer distance) will be merged
  - Non-adjacent regions are automatically skipped
  - Uses existing merge logic from merge_cross_region_climbs.py
  - Original files are backed up before modification
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='File patterns to merge (supports wildcards, e.g., "output/*.xlsx")'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Distance threshold in km for matching climbs (default: 0.5)'
    )
    parser.add_argument(
        '--buffer', '-b',
        type=float,
        default=10.0,
        help='Maximum distance in km to consider regions adjacent (default: 10.0)'
    )
    parser.add_argument(
        '--auto-replace',
        action='store_true',
        help='Automatically replace climbs without prompting'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be merged without making changes'
    )

    args = parser.parse_args()

    try:
        print_banner("Batch Cross-Region Climb Merger")
        print()

        # Expand wildcards to file paths
        print_info("Expanding file patterns...")
        file_paths = expand_file_patterns(args.files)
        print_success(f"Found {len(file_paths)} file(s)")
        for file_path in file_paths:
            print_list_item(f"• {Path(file_path).name}")
        print()

        # Validate files
        print_info("Validating files...")
        valid_files = validate_files(file_paths)
        print_success(f"All {len(valid_files)} file(s) are valid")
        print()

        if len(valid_files) < 2:
            print_warning("Need at least 2 files to merge")
            return 0

        # Find adjacent pairs
        adjacent_pairs = find_adjacent_pairs(valid_files, buffer_km=args.buffer)

        if not adjacent_pairs:
            print_warning("No adjacent region pairs found - nothing to merge")
            return 0

        # Dry run mode - show what would be merged
        if args.dry_run:
            print_header("Dry Run - No Changes Will Be Made")
            print_info(f"Would merge {len(adjacent_pairs)} adjacent region pair(s):")
            for file1, file2, distance in adjacent_pairs:
                print_list_item(f"• {Path(file1).stem} ↔ {Path(file2).stem} ({distance:.1f} km)")
            print()
            print_info("Run without --dry-run to perform actual merges")
            return 0

        # Merge adjacent regions
        successful = merge_adjacent_regions(
            adjacent_pairs,
            distance_threshold_km=args.threshold,
            auto_replace=args.auto_replace
        )

        if successful > 0:
            print_success(f"✓ Batch merge complete - {successful} region pair(s) merged")
        else:
            print_warning("No regions were merged")

        return 0

    except Exception as e:
        print_error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
