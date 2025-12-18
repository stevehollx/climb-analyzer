#!/usr/bin/env python3
"""
OSM PBF File Merger

Merges multiple OSM .pbf files into a single file using osmium-tool.
Useful when an analysis area spans multiple Geofabrik regions.
"""

import subprocess
from pathlib import Path
from typing import List, Optional


def check_osmium_available() -> bool:
    """Check if osmium-tool is installed and available."""
    try:
        result = subprocess.run(
            ["osmium", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def merge_osm_files(
    input_files: List[Path],
    output_file: Path,
    remove_inputs: bool = False
) -> bool:
    """
    Merge multiple OSM .pbf files into one using osmium-tool.

    Args:
        input_files: List of .pbf files to merge
        output_file: Output merged .pbf file
        remove_inputs: If True, delete input files after successful merge

    Returns:
        True if successful, False otherwise
    """
    if len(input_files) < 2:
        print("⚠️  Need at least 2 files to merge")
        return False

    if not check_osmium_available():
        print("❌ osmium-tool not found!")
        print("   Install with: brew install osmium-tool  (macOS)")
        print("   or: apt-get install osmium-tool  (Ubuntu/Debian)")
        return False

    print(f"\n🔀 Merging {len(input_files)} OSM files...")
    for f in input_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"   • {f.name} ({size_mb:.0f} MB)")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build osmium merge command
    cmd = ["osmium", "merge"]
    cmd.extend([str(f) for f in input_files])
    cmd.extend(["-o", str(output_file)])

    try:
        print(f"\n   Running: osmium merge...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for large files
        )

        if result.returncode != 0:
            print(f"❌ Merge failed: {result.stderr}")
            return False

        if not output_file.exists():
            print("❌ Merge completed but output file not found")
            return False

        size_mb = output_file.stat().st_size / (1024**2)
        print(f"✓ Merged successfully: {output_file.name} ({size_mb:.0f} MB)")

        # Remove input files if requested
        if remove_inputs:
            print("\n🗑️  Removing input files...")
            for f in input_files:
                try:
                    f.unlink()
                    print(f"   • Deleted {f.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not delete {f.name}: {e}")

        return True

    except subprocess.TimeoutExpired:
        print("❌ Merge timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"❌ Merge error: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python osm_merger.py <output.osm.pbf> <input1.osm.pbf> <input2.osm.pbf> [...]")
        print("\nExample:")
        print("  python osm_merger.py merged.osm.pbf georgia.osm.pbf tennessee.osm.pbf")
        sys.exit(1)

    output = Path(sys.argv[1])
    inputs = [Path(f) for f in sys.argv[2:]]

    # Verify input files exist
    for f in inputs:
        if not f.exists():
            print(f"❌ Input file not found: {f}")
            sys.exit(1)

    success = merge_osm_files(inputs, output)
    sys.exit(0 if success else 1)
