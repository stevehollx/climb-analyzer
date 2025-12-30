#!/usr/bin/env python3
"""
Spatial index builder for OSM data.

Builds R-tree spatial indexes for fast bbox queries.
"""

import shutil
import time
from pathlib import Path

# Import centralized data paths
try:
    from utils.data_paths import OSM_INDEXES_DIR
    DEFAULT_INDEX_DIR = str(OSM_INDEXES_DIR)
except ImportError:
    DEFAULT_INDEX_DIR = "data/osm_indexes"


def build_spatial_index(
    osm_file_path: str,
    output_dir: str = None,
    surface_filter: str = "all",
    cycling_only: bool = False,
) -> bool:
    """
    Build spatial index from OSM .pbf file.

    Args:
        osm_file_path: Path to .osm.pbf file
        output_dir: Where to save index files
        surface_filter: Filter for road surfaces ("all", "paved", "gravel", "dirt")
        cycling_only: If True, only index cycling-friendly roads

    Returns:
        True if successful, False otherwise
    """
    from climb_analyzer.data.spatial_index import SpatialIndexManager

    # Use default if not specified
    if output_dir is None:
        output_dir = DEFAULT_INDEX_DIR

    osm_path = Path(osm_file_path)
    if not osm_path.exists():
        print(f"❌ OSM file not found: {osm_file_path}")
        return False

    print("\nBuilding spatial index")
    print(f"  Input: {osm_file_path}")
    print(f"  Output: {output_dir}")
    print(f"  Surface filter: {surface_filter}")
    print(f"  Cycling only: {cycling_only}")

    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create index manager
        index_mgr = SpatialIndexManager(osm_file_path, cache_dir=output_dir)

        # Check if index already exists
        if index_mgr.exists():
            # Verify the index is valid by checking:
            # 1. Can open the rtree index files
            # 2. Metadata file exists and is non-empty
            idx_base = str(output_path / osm_path.stem) + "_spatial"
            metadata_path = output_path / f"{osm_path.stem}_metadata.jsonl"

            is_valid = True

            # Check rtree index files
            try:
                from rtree import index as rtree_index
                test_idx = rtree_index.Rtree(idx_base)
                test_idx.close()
            except Exception as e:
                print(f"  ⚠️  Existing index is corrupted ({e}), rebuilding...")
                is_valid = False

            # Check metadata file
            if is_valid and not metadata_path.exists():
                print(f"  ⚠️  Metadata file missing, rebuilding...")
                is_valid = False
            elif is_valid and metadata_path.stat().st_size < 100:
                print(f"  ⚠️  Metadata file incomplete (only {metadata_path.stat().st_size} bytes), rebuilding...")
                is_valid = False

            # Check for empty .idx file (sign of interrupted build)
            idx_file = Path(idx_base + ".idx")
            if is_valid and idx_file.exists() and idx_file.stat().st_size == 0:
                print(f"  ⚠️  Index file is empty (interrupted build), rebuilding...")
                is_valid = False

            if is_valid:
                print("  ✓ Spatial index already exists and is valid")
                return True
            else:
                # Delete corrupted/partial index files
                print("  Cleaning up partial/corrupted index files...")
                for ext in [".idx", ".dat"]:
                    idx_file = Path(idx_base + ext)
                    if idx_file.exists():
                        try:
                            idx_file.unlink()
                            print(f"    Deleted: {idx_file.name}")
                        except PermissionError:
                            print(f"    ⚠️  Cannot delete {idx_file.name} (permission denied)")
                            print(f"    Please manually delete: {idx_file}")
                            return False
                if metadata_path.exists():
                    try:
                        metadata_path.unlink()
                        print(f"    Deleted: {metadata_path.name}")
                    except PermissionError:
                        print(f"    ⚠️  Cannot delete {metadata_path.name} (permission denied)")
                        print(f"    Please manually delete: {metadata_path}")
                        return False
                # Continue to rebuild

        # Estimate time based on file size (only show if building)
        file_size_gb = osm_path.stat().st_size / (1024**3)
        estimated_min = int(file_size_gb * 5)  # Rough estimate: 5 min per GB
        estimated_max = int(estimated_min * 1.5)

        if estimated_min == estimated_max:
            print(f"\n   Estimated time: ~{estimated_min} minutes")
        else:
            print(f"\n  Estimated time: {estimated_min}-{estimated_max} minutes")

        # Build index
        start_time = time.time()
        print("  Building spatial index...")
        print("  This will process all highways and create an R-tree for fast queries\n")

        try:
            # Import required libraries
            import json

            import osmium
            from rtree import index as rtree_index

            # Create spatial index using temp files (atomic write pattern)
            idx_path_str = str(output_path / osm_path.stem) + "_spatial"
            idx_path_tmp = idx_path_str + ".tmp"  # Temp path for atomic writes
            jsonl_path = output_path / f"{osm_path.stem}_metadata.jsonl"
            jsonl_path_tmp = jsonl_path.with_suffix(".jsonl.tmp")

            # Clean up any leftover temp files from interrupted builds
            for ext in [".idx", ".dat"]:
                tmp_file = Path(idx_path_tmp + ext)
                if tmp_file.exists():
                    tmp_file.unlink()
                    print(f"  Cleaned up leftover temp file: {tmp_file.name}")
            if jsonl_path_tmp.exists():
                jsonl_path_tmp.unlink()
                print(f"  Cleaned up leftover temp file: {jsonl_path_tmp.name}")

            # Ensure no leftover final index files exist (could be corrupted)
            for ext in [".idx", ".dat"]:
                idx_file = Path(idx_path_str + ext)
                if idx_file.exists():
                    idx_file.unlink()

            # Create rtree with temp path
            spatial_idx = rtree_index.Rtree(idx_path_tmp)

            # Open metadata temp file for streaming writes (memory-efficient)
            metadata_file = open(jsonl_path_tmp, "w")

            class WayIndexer(osmium.SimpleHandler):
                def __init__(self, spatial_idx, metadata_file):
                    super().__init__()
                    self.spatial_idx = spatial_idx
                    self.metadata_file = metadata_file
                    self.node_count = 0
                    self.way_count = 0
                    self.indexed_ways = 0
                    self.write_buffer = []
                    self.buffer_size = 1000  # Flush every 1000 ways

                def node(self, n):
                    # Count nodes for progress tracking
                    # Note: We don't store nodes manually - osmium handles location caching
                    self.node_count += 1

                    if self.node_count % 1000000 == 0:
                        print(f"\r  Processed {self.node_count:,} nodes", end="", flush=True)

                def flush_buffer(self):
                    """Flush buffered metadata to disk"""
                    if self.write_buffer:
                        for entry in self.write_buffer:
                            self.metadata_file.write(json.dumps(entry) + "\n")
                        self.write_buffer.clear()
                        self.metadata_file.flush()

                def way(self, w):
                    self.way_count += 1

                    if self.way_count % 50000 == 0:
                        print(
                            f"\r  Processed {self.way_count:,} ways, indexed {self.indexed_ways:,} highways",
                            end="",
                            flush=True,
                        )

                    # Only index ways with highway tag
                    if "highway" not in w.tags:
                        return

                    # Get coordinates from node references (osmium provides locations)
                    lats = []
                    lons = []
                    node_coords = []

                    for node_ref in w.nodes:
                        # Use osmium's location cache instead of manual dictionary
                        if node_ref.location.valid():
                            lat, lon = node_ref.location.lat, node_ref.location.lon
                            node_id = node_ref.ref  # OSM node ID for connectivity checking
                            lats.append(lat)
                            lons.append(lon)
                            node_coords.append((lat, lon, node_id))

                    if len(node_coords) < 2:
                        return

                    # Calculate bounding box
                    min_lat, max_lat = min(lats), max(lats)
                    min_lon, max_lon = min(lons), max(lons)
                    bbox = (min_lon, min_lat, max_lon, max_lat)

                    # Validate bbox
                    if bbox == (0.0, 0.0, 0.0, 0.0) or min_lat == 0.0:
                        return

                    # Buffer metadata for streaming write (memory-efficient)
                    metadata_entry = {
                        "way_id": int(w.id),
                        "tags": dict(w.tags),
                        "nodes": node_coords,
                        "bbox": bbox,
                    }
                    self.write_buffer.append(metadata_entry)

                    # Flush buffer periodically to disk
                    if len(self.write_buffer) >= self.buffer_size:
                        self.flush_buffer()

                    # Insert into spatial index
                    self.spatial_idx.insert(w.id, bbox)
                    self.indexed_ways += 1

            print("  Processing OSM file...")
            indexer = WayIndexer(spatial_idx, metadata_file)

            # Use sparse_mem_array for better memory efficiency with large files
            # flex_mem keeps all nodes in RAM (~1.5GB for 67M nodes)
            # sparse_mem_array uses less memory at the cost of slightly slower processing
            try:
                indexer.apply_file(str(osm_file_path), locations=True, idx="sparse_mem_array")
            except Exception as e:
                # Fallback to flex_mem if sparse_mem_array fails
                print(f"\n  Note: sparse_mem_array failed ({e}), trying flex_mem...")
                indexer.apply_file(str(osm_file_path), locations=True, idx="flex_mem")

            # Flush any remaining buffered metadata
            indexer.flush_buffer()

            # Close spatial index
            spatial_idx.close()

            elapsed = time.time() - start_time
            print(f"\n\n  Indexing complete in {elapsed/60:.1f} minutes:")
            print(f"    - Nodes processed: {indexer.node_count:,}")
            print(f"    - Ways processed: {indexer.way_count:,}")
            print(f"    - Highways indexed: {indexer.indexed_ways:,}")

            # Prepend metadata header to the JSONL temp file
            print(f"\n  Finalizing metadata file...")
            metadata_file.close()

            # Read existing temp file and prepend metadata
            with open(jsonl_path_tmp, "r") as f:
                existing_lines = f.readlines()

            with open(jsonl_path_tmp, "w") as f:
                # Write metadata header first
                index_metadata = {
                    "_index_metadata": {
                        "total_ways": indexer.indexed_ways,
                        "created_at": time.time(),
                        "source_file": osm_path.name,
                    }
                }
                f.write(json.dumps(index_metadata) + "\n")

                # Write all way data
                f.writelines(existing_lines)

            # Atomic rename: move temp files to final names
            # This ensures all-or-nothing completion
            print(f"  Committing index files (atomic rename)...")
            shutil.move(str(Path(idx_path_tmp + ".idx")), idx_path_str + ".idx")
            shutil.move(str(Path(idx_path_tmp + ".dat")), idx_path_str + ".dat")
            shutil.move(str(jsonl_path_tmp), str(jsonl_path))

            print(f"  ✓ Metadata saved: {indexer.indexed_ways:,} highways")
            print(f"  ✓ Index files: {idx_path_str}.idx, {idx_path_str}.dat")
            print(f"  ✓ Metadata: {jsonl_path.name}")

            # Clean up objects to free memory
            print(f"\n  Cleaning up memory...")

            # Note: Metadata was streamed to disk, so no large dictionary to clear
            # Force garbage collection to free osmium caches
            import gc
            gc.collect()
            print(f"    ✓ Memory cleanup completed")

            return True

        except ImportError as e:
            print(f"\n  ❌ Missing dependency: {e}")
            print("  Install required packages: pip install osmium rtree")
            return False
        except Exception as e:
            print(f"\n  ❌ Index build failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")
        print("  Install rtree: pip install rtree")
        return False
    except Exception as e:
        print(f"  ❌ Index build failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_index(osm_file_path: str, index_dir: str = None) -> bool:
    """
    Verify that spatial index exists and is valid.

    Args:
        osm_file_path: Path to OSM file
        index_dir: Directory containing index files (defaults to centralized path)

    Returns:
        True if index exists and appears valid
    """
    from climb_analyzer.data.spatial_index import SpatialIndexManager

    # Use default if not specified
    if index_dir is None:
        index_dir = DEFAULT_INDEX_DIR

    try:
        index_mgr = SpatialIndexManager(osm_file_path, cache_dir=index_dir)
        exists = index_mgr.exists()

        if exists:
            print("  ✓ Spatial index found")
            return True
        else:
            print("  ❌ Spatial index not found")
            return False

    except Exception as e:
        print(f"  ❌ Error checking index: {e}")
        return False


# Test if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        osm_file = sys.argv[1]
        result = build_spatial_index(osm_file)
        if result:
            print("\nSuccess!")
        else:
            print("\nFailed.")
            sys.exit(1)
    else:
        print("Usage: python index_builder.py <osm_file.pbf>")
        print("\nExample:")
        print("  python index_builder.py planet-osm/switzerland-latest.osm.pbf")
