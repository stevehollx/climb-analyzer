#!/usr/bin/env python3
"""
Automatic Cross-Region Climb Merger

This module provides functions to automatically merge climbs that were split
at chunk boundaries when in-memory cross-chunk merge was skipped for large datasets.
"""

from pathlib import Path
from typing import List
import sys

# Import the merge functionality
sys.path.insert(0, str(Path(__file__).parent.parent))


def merge_split_files(file_paths: List[Path], auto_replace: bool = True) -> int:
    """
    Automatically merge split climbs across multiple files.

    This is used when in-memory cross-chunk merge was skipped due to dataset size,
    to merge climbs that may have been split at chunk boundaries.

    Args:
        file_paths: List of output file paths to merge
        auto_replace: If True, automatically update files without prompting

    Returns:
        Number of climbs merged (0 if no merges found or error)
    """
    if not file_paths or len(file_paths) < 2:
        # Need at least 2 files to merge
        return 0

    try:
        # Import merge functionality
        from scripts.merge_cross_region_climbs import (
            find_and_merge_cross_region_climbs,
            replace_climbs_in_files
        )

        total_merges = 0

        # Sort files by their suffix number for consistent ordering
        sorted_files = sorted(file_paths, key=lambda p: p.stem)

        print(f"\n{'='*80}")
        print("AUTO-MERGE: Checking for climbs split at chunk boundaries")
        print(f"{'='*80}\n")

        # Compare ALL pairs of files (not just adjacent)
        # This is necessary because files are sorted by Basic Score, not geographic order
        # A climb split across chunks might end up in non-adjacent files!
        num_files = len(sorted_files)
        total_comparisons = (num_files * (num_files - 1)) // 2

        print(f"Checking {num_files} file(s) for split climbs...")
        print(f"Will compare all {total_comparisons} file pair(s)\n")

        comparison_count = 0
        # Compare all pairs: (0,1), (0,2), ..., (0,n), (1,2), (1,3), ..., (n-1,n)
        for i in range(len(sorted_files)):
            for j in range(i + 1, len(sorted_files)):
                file1 = sorted_files[i]
                file2 = sorted_files[j]
                comparison_count += 1

                print(f"[{comparison_count}/{total_comparisons}] Comparing: {file1.name} <-> {file2.name}")

                try:
                    merged_climbs = find_and_merge_cross_region_climbs(
                        str(file1),
                        str(file2),
                        distance_threshold_km=0.5  # Default threshold
                    )

                    if merged_climbs and auto_replace:
                        replace_climbs_in_files(merged_climbs, str(file1), str(file2))
                        total_merges += len(merged_climbs)

                except Exception as e:
                    print(f"  ⚠️  Error merging {file1.name} and {file2.name}: {e}")
                    continue

        if total_merges > 0:
            print(f"\n{'='*80}")
            print(f"✓ AUTO-MERGE COMPLETE: {total_merges} climb(s) merged")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print("✓ AUTO-MERGE COMPLETE: No split climbs found")
            print(f"{'='*80}\n")

        return total_merges

    except Exception as e:
        print(f"\n⚠️  Auto-merge failed: {e}")
        print("   Files were still saved successfully")
        return 0


def merge_all_output_files(output_dir: Path = Path("output"), pattern: str = "*_climbs_*.xlsx") -> int:
    """
    Merge all output files in a directory (for batch mode).

    Args:
        output_dir: Directory containing output files
        pattern: Glob pattern to match output files

    Returns:
        Number of climbs merged
    """
    if not output_dir.exists():
        return 0

    # Find all matching files
    files = sorted(output_dir.glob(pattern))

    if len(files) < 2:
        return 0

    print(f"\n{'='*80}")
    print("BATCH AUTO-MERGE: Checking all output files for split climbs")
    print(f"{'='*80}\n")
    print(f"Found {len(files)} output file(s)")

    return merge_split_files(files, auto_replace=True)


if __name__ == "__main__":
    # For testing
    import argparse
    parser = argparse.ArgumentParser(description="Automatically merge split climbs")
    parser.add_argument("files", nargs="+", help="Files to merge")
    args = parser.parse_args()

    file_paths = [Path(f) for f in args.files]
    merge_split_files(file_paths)
