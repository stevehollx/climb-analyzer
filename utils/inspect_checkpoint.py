#!/usr/bin/env python3
"""
Inspect elevation checkpoint files to verify data structure.
"""

import pickle
import sys
from pathlib import Path
from datetime import datetime

def inspect_checkpoint(checkpoint_path):
    """Inspect and display checkpoint contents."""
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        print(f"\n{'='*60}")
        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"{'='*60}")

        # Get file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
        print(f"Size: {file_size_mb:.2f} MB")
        print(f"Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Display top-level structure
        print("Top-level keys:")
        for key in data.keys():
            print(f"  - {key}")
        print()

        # Display coordinate_mapping info
        if 'coordinate_mapping' in data:
            coord_map = data['coordinate_mapping']
            print(f"Coordinate mapping: {len(coord_map):,} node IDs")
            if coord_map:
                sample_keys = list(coord_map.keys())[:3]
                print("  Sample entries:")
                for key in sample_keys:
                    print(f"    {key}: {coord_map[key]}")
        print()

        # Display checkpoint_data info
        if 'checkpoint_data' in data:
            checkpoint_data = data['checkpoint_data']
            print("Checkpoint data:")
            for key, value in checkpoint_data.items():
                if key == 'node_elevations':
                    print(f"  - {key}: {len(value):,} entries")
                    if value:
                        sample = list(value.items())[:3]
                        print("    Sample entries:")
                        for k, v in sample:
                            print(f"      {k}: {v}")
                elif key == 'coords_seen':
                    print(f"  - {key}: {len(value):,} coordinates")
                    if value:
                        print(f"    Sample: {value[:3]}")
                else:
                    print(f"  - {key}: {value}")
        print()

        # Display timestamp if present
        if 'timestamp' in data:
            ts = datetime.fromtimestamp(data['timestamp'])
            print(f"Checkpoint timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"Error reading checkpoint: {e}")
        return False

def main():
    """Find and inspect all elevation checkpoints."""
    checkpoint_base = Path("checkpoints")

    if not checkpoint_base.exists():
        print("No checkpoint directory found.")
        return

    # Find all elevation_progress.pkl files
    checkpoints = list(checkpoint_base.glob("*/elevation_progress.pkl"))

    if not checkpoints:
        print("No elevation checkpoints found.")
        return

    print(f"Found {len(checkpoints)} elevation checkpoint(s)\n")

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for checkpoint in checkpoints:
        inspect_checkpoint(checkpoint)

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Newest: {checkpoints[0].parent.name}")
    print(f"        Modified: {datetime.fromtimestamp(checkpoints[0].stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    if len(checkpoints) > 1:
        print(f"Oldest: {checkpoints[-1].parent.name}")
        print(f"        Modified: {datetime.fromtimestamp(checkpoints[-1].stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Suggest cleanup if there are old checkpoints
    if len(checkpoints) > 3:
        print(f"⚠️  You have {len(checkpoints)} checkpoint files.")
        print("   Consider cleaning up old checkpoints from failed/cancelled runs:")
        print("   rm -rf checkpoints/*/elevation_progress.pkl")
        print()

if __name__ == "__main__":
    main()
