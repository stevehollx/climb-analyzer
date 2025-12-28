"""
Road segment merging functionality.

OVERVIEW:
This module handles the critical task of merging road segments that were split at
region boundaries during analysis. When regions are processed, individual roads
may be split at arbitrary boundaries. This merger reconnects these segments into
complete roads for accurate climb analysis.

ARCHITECTURE:
The system uses a way-based processing approach:
1. Fetch complete OSM ways (roads) from the database
2. Convert ways to segment arrays with elevation data
3. Merge segments that belong to the same physical road

PROCESSING MODES:
1. **Serial Mode**: For datasets < 1000 segments or when parallel disabled
   - Progressive merging with checkpointing
   - Memory-efficient streaming operations
   - Spatial grouping for very large streets (>100 segments per street)

2. **Parallel Mode**: For large datasets (>1000 segments) with multiple CPU cores
   - Batch processing with ProcessPoolExecutor
   - Default: 16 workers (configurable)
   - Memory-aware worker count calculation
   - Automatic fallback to serial on errors

MEMORY MANAGEMENT:
- Disk-based streaming for large datasets (replaced memory-intensive loading)
- Spatial grouping prevents O(n²) comparisons for large street networks
- Progressive checkpointing allows resuming interrupted merges
- Worker count scales based on available RAM

MERGE STRATEGY:
- Groups segments by street name
- Detects duplicates using way_ids and endpoint hashing
- Connects segments with matching endpoints (±coordinate tolerance)
- Preserves elevation profiles and geometric accuracy
- Special handling for common street names (service, residential, etc.)

CLASSES:
    BoundaryMerger: Main class for segment merging with serial/parallel support

FUNCTIONS:
    _merge_street_batch_worker_top_level: Module-level worker for multiprocessing

USAGE:
    merger = BoundaryMerger(coordinate_tolerance=0.002)
    merged_segments = merger.merge_boundary_segments(all_segments, persistence)

CONFIGURATION:
    - coordinate_tolerance: Degrees for endpoint matching (default: 0.002 ≈ 220m)
    - distance_tolerance_m: Meters for duplicate detection (default: 5m)
    - parallel_enabled: Enable/disable parallel processing
    - max_workers: Maximum worker processes for parallel mode

PERFORMANCE:
    - Serial: ~1000 segments/second
    - Parallel: ~5000-10000 segments/second (16 workers)
    - Checkpointing overhead: ~5% for large merges
    - Memory usage: ~200MB + (segments * 2KB)
"""

import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from tqdm import tqdm


# Custom unpickler to handle module remapping for classes serialized from __main__
class _MergerUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps classes serialized from __main__ to correct modules."""

    def find_class(self, module, name):
        if module == "__main__":
            if name in ("ClimbMetrics", "ClimbIdentifier", "SimpleClimbNode", "ElevationProfile"):
                module = "climb_analyzer.engine"
            elif name == "ErrorLogEntry":
                module = "climb_analyzer.data.elevation"
        return super().find_class(module, name)


def _safe_pickle_load(file_handle):
    """Load pickle data using custom unpickler that handles __main__ class remapping."""
    return _MergerUnpickler(file_handle).load()


def _merge_street_batch_worker_top_level(args):
    """
    Top-level worker function for parallel street merging.
    Must be at module level for pickle serialization in multiprocessing.

    Args:
        args: Tuple of (street_items_batch, coordinate_tolerance, distance_tolerance_m)

    Returns:
        List of merged segments for all streets in batch
    """
    street_items_batch, coordinate_tolerance, distance_tolerance_m = args

    # Import here to avoid circular dependencies
    from climb_analyzer.core.merger import BoundaryMerger

    # Create a temporary merger instance for this worker
    merger = BoundaryMerger(coordinate_tolerance=coordinate_tolerance)
    merger.distance_tolerance_m = distance_tolerance_m

    batch_results = []
    for street_name, street_segments in street_items_batch:
        try:
            # Determine merge strategy based on street characteristics
            common_names = {"service", "track", "path", "footway", "cycleway",
                           "bridleway", "steps", "residential", "unclassified"}
            is_common_name = street_name.lower() in common_names

            if is_common_name and len(street_segments) > 50:
                merged = merger._merge_large_street_with_spatial_grouping(
                    street_segments, street_name
                )
            else:
                merged = merger._merge_street(street_segments, street_name)

            batch_results.extend(merged)
        except Exception as e:
            # Log error but continue processing other streets
            import traceback
            print(f"\nError in worker processing street '{street_name}': {e}")
            traceback.print_exc()
            # Include unmerged segments as fallback
            batch_results.extend(street_segments)

    return batch_results


class BoundaryMerger:
    """
    Handles merging of road segments that cross region boundaries.

    This class provides both serial and parallel merge implementations with automatic
    mode selection based on dataset size and available resources.

    Architecture:
        - Way-based processing: Fetches complete ways, then merges segment representations
        - Streaming operations: Processes data from disk to handle large regions
        - Spatial grouping: Prevents O(n²) complexity for streets with many segments
        - Checkpointing: Allows resuming interrupted merge operations

    Attributes:
        coordinate_tolerance (float): Degrees for endpoint matching (default: 0.002 ≈ 220m at mid-latitudes)
        distance_tolerance_m (int): Meters for duplicate detection (default: 200m)
        parallel_enabled (bool): Whether to use parallel processing for large datasets
        max_workers (int): Maximum worker processes for parallel mode
        batch_size (int): Number of streets per worker batch in parallel mode

    Methods:
        merge_boundary_segments: Main entry point for merging
        _perform_boundary_merge: Serial merge implementation
        _perform_boundary_merge_parallel: Parallel merge implementation
        _merge_street: Merge segments for a single street
        _merge_large_street_with_spatial_grouping: Optimized merge for large streets
    """

    def __init__(self, coordinate_tolerance: float = 0.002):
        """
        Initialize the BoundaryMerger.

        Args:
            coordinate_tolerance: Maximum distance in degrees for endpoint matching.
                                 Default 0.002° ≈ 220m at equator, ~155m at 45° latitude.
                                 Used to determine if two segment endpoints are the same point.

        Configuration Loading:
            Attempts to load from climb_analyzer.config:
            - MERGE_PARALLEL_ENABLED: Enable/disable parallel processing
            - MERGE_MAX_WORKERS: Maximum worker processes
            - MERGE_BATCH_SIZE: Streets per worker batch

            Falls back to safe defaults if config unavailable.
        """
        # Import config values here to avoid issues during initialization
        # Config may not be available in all contexts (e.g., testing, standalone scripts)
        try:
            from climb_analyzer.config import (
                MERGE_BATCH_SIZE,
                MERGE_MAX_WORKERS,
                MERGE_PARALLEL_ENABLED,
            )
            self.parallel_enabled = MERGE_PARALLEL_ENABLED
            self.max_workers = MERGE_MAX_WORKERS
            self.batch_size = MERGE_BATCH_SIZE
        except ImportError:
            # Fallback defaults if config not available
            # Conservative settings that work on most systems
            self.parallel_enabled = True
            self.max_workers = min(16, os.cpu_count() or 1)  # Cap at 16 to avoid excessive overhead
            self.batch_size = 500  # Balance between parallelism and memory usage

        # Coordinate tolerance: Used for endpoint matching
        # 0.002° ≈ 220m at equator, scales with latitude
        self.coordinate_tolerance = coordinate_tolerance

        # Distance tolerance: Used for duplicate detection
        # Segments closer than this are considered duplicates
        # Set conservatively to avoid false positives
        self.distance_tolerance_m = 200

    def _calculate_safe_worker_count(self, total_segments: int) -> int:
        """
        Calculate safe number of workers based on available memory.

        Each worker needs memory for:
        - Python interpreter overhead: ~50-100 MB
        - Batch of streets to process: ~500 streets × avg segments
        - Temporary merge results: 2-3x input size during processing

        Conservative estimate: ~500 MB per worker for large datasets
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)

            # Conservative estimates
            MB_PER_WORKER = 500  # Assume 500MB per worker
            SAFETY_MARGIN = 0.7  # Only use 70% of available memory

            # Calculate how many workers we can safely spawn
            available_mb = (available_gb * 1024) * SAFETY_MARGIN
            safe_workers = int(available_mb / MB_PER_WORKER)

            # Clamp to reasonable bounds
            safe_workers = max(1, min(safe_workers, self.max_workers, os.cpu_count() or 1))

            print("\nMemory-based worker calculation:")
            print(f"  Total RAM: {mem.total / (1024**3):.1f} GB")
            print(f"  Available RAM: {available_gb:.1f} GB")
            print(f"  Safe workers (with {int(SAFETY_MARGIN*100)}% safety margin): {safe_workers}")
            print(f"  Configured max workers: {self.max_workers}")

            if safe_workers < self.max_workers:
                print(f"  ⚠️  Reducing workers from {self.max_workers} to {safe_workers} due to memory constraints")

            return safe_workers

        except Exception as e:
            print(f"  ⚠️  Could not calculate memory-safe worker count: {e}")
            print("  Defaulting to conservative 4 workers")
            return min(4, self.max_workers)

    def _coordinates_match(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> bool:
        """Coordinate matching using actual distance."""
        if (
            abs(coord1[0] - coord2[0]) < self.coordinate_tolerance
            and abs(coord1[1] - coord2[1]) < self.coordinate_tolerance
        ):
            return True

        distance_km = self.calculate_distance(coord1[0], coord1[1], coord2[0], coord2[1])
        distance_m = distance_km * 1000

        return distance_m < self.distance_tolerance_m

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula (in km)."""
        R = 6371
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _is_dead_end(self, segment: Dict, endpoint_type: str, all_segments: List[Dict], used: List[bool]) -> bool:
        """
        Check if a segment's endpoint is a dead-end (not connected to other unused segments).

        Args:
            segment: The segment to check
            endpoint_type: 'start' or 'end'
            all_segments: List of all segments in the merge group
            used: Boolean array indicating which segments are already merged

        Returns:
            True if the endpoint is isolated (dead-end), False if it connects to other segments
        """
        if not segment.get("nodes") or len(segment["nodes"]) < 2:
            return True

        # Get the endpoint to check
        if endpoint_type == "start":
            endpoint = segment["nodes"][0]
        else:  # end
            endpoint = segment["nodes"][-1]

        endpoint_coord = (endpoint.get("lat"), endpoint.get("lon"))
        endpoint_id = endpoint.get("id")

        # Check if this endpoint connects to any other unused segment
        for idx, other_segment in enumerate(all_segments):
            if used[idx]:
                continue
            if other_segment is segment:
                continue
            if not other_segment.get("nodes") or len(other_segment["nodes"]) < 2:
                continue

            other_start = other_segment["nodes"][0]
            other_end = other_segment["nodes"][-1]
            other_start_coord = (other_start.get("lat"), other_start.get("lon"))
            other_end_coord = (other_end.get("lat"), other_end.get("lon"))
            other_start_id = other_start.get("id")
            other_end_id = other_end.get("id")

            # Check for node ID match first (exact connection)
            if endpoint_id and (endpoint_id == other_start_id or endpoint_id == other_end_id):
                return False

            # Check for coordinate proximity
            if self._coordinates_match(endpoint_coord, other_start_coord) or \
               self._coordinates_match(endpoint_coord, other_end_coord):
                return False

        return True

    def _score_merge_potential(self, segment: Dict, neighbor: Dict, neighbor_endpoint: str,
                                all_segments: List[Dict], used: List[bool]) -> int:
        """
        Score a potential merge by counting how many additional segments can connect
        to the neighbor's far endpoint (topology-based chain prioritization).

        Higher scores indicate the neighbor is part of a longer potential chain.

        Args:
            segment: Current segment
            neighbor: Neighbor to potentially merge
            neighbor_endpoint: Which end of neighbor connects ('start' or 'end')
            all_segments: List of all segments in merge group
            used: Boolean array of used segments

        Returns:
            Score (number of potential connections at far endpoint)
        """
        if not neighbor.get("nodes") or len(neighbor["nodes"]) < 2:
            return 0

        # Get the neighbor's FAR endpoint (the one NOT connecting to current segment)
        if neighbor_endpoint == "start":
            far_endpoint = neighbor["nodes"][-1]  # Far end is the last node
        else:  # neighbor_endpoint == "end"
            far_endpoint = neighbor["nodes"][0]  # Far end is the first node

        far_coord = (far_endpoint.get("lat"), far_endpoint.get("lon"))
        far_id = far_endpoint.get("id")

        # Count how many unused segments can connect at this far endpoint
        connection_count = 0
        for idx, other_segment in enumerate(all_segments):
            if used[idx]:
                continue
            if other_segment is neighbor or other_segment is segment:
                continue
            if not other_segment.get("nodes") or len(other_segment["nodes"]) < 2:
                continue

            other_start = other_segment["nodes"][0]
            other_end = other_segment["nodes"][-1]
            other_start_coord = (other_start.get("lat"), other_start.get("lon"))
            other_end_coord = (other_end.get("lat"), other_end.get("lon"))
            other_start_id = other_start.get("id")
            other_end_id = other_end.get("id")

            # Check for connection via node ID or proximity
            if far_id and (far_id == other_start_id or far_id == other_end_id):
                connection_count += 1
            elif self._coordinates_match(far_coord, other_start_coord) or \
                 self._coordinates_match(far_coord, other_end_coord):
                connection_count += 1

        return connection_count

    def _merge_large_street_with_spatial_grouping(
        self, segments: List[Dict], street_name: str
    ) -> List[Dict]:
        """Handle very large streets (>100 segments) by grouping spatially first."""

        # Group segments by approximate location (larger grid for very large streets)
        location_groups = self._group_segments_by_location(segments, grid_size=0.005)  # ~500m grid

        merged_segments = []

        # Process without nested progress bars (they interfere with parallel processing)
        for i, group_segments in enumerate(location_groups):
            if len(group_segments) <= 20:
                # Use regular merging for reasonable-sized groups
                group_merged = self._merge_street(
                    group_segments, f"{street_name}_cluster_{i}"
                )
                merged_segments.extend(group_merged)
            else:
                # For very large groups within a location, sub-divide further
                sub_groups = self._group_segments_by_location(
                    group_segments, grid_size=0.001
                )  # ~100m grid

                for j, sub_group in enumerate(sub_groups):
                    if len(sub_group) <= 10:
                        sub_merged = self._merge_street(
                            sub_group, f"{street_name}_cluster_{i}_{j}"
                        )
                        merged_segments.extend(sub_merged)
                    else:
                        # If still too large, just keep as separate segments
                        merged_segments.extend(sub_group)

        return merged_segments

    def _group_segments_by_location(
        self, segments: List[Dict], grid_size: float = 0.005
    ) -> List[List[Dict]]:
        """Group segments by spatial proximity using grid."""
        location_map = defaultdict(list)

        for segment in segments:
            if not segment.get("nodes"):
                continue

            # Use first node as representative location
            first_node = segment["nodes"][0]
            lat = first_node.get("lat", 0)
            lon = first_node.get("lon", 0)

            # Round to grid
            grid_lat = round(lat / grid_size) * grid_size
            grid_lon = round(lon / grid_size) * grid_size
            grid_key = (grid_lat, grid_lon)

            location_map[grid_key].append(segment)

        return list(location_map.values())

    def _are_segments_duplicates(self, seg1: Dict, seg2: Dict, tolerance_deg: float = 0.0001) -> bool:
        """
        Check if two segments are duplicates (same coordinates for all nodes).

        This handles the case where spatial deduplication was skipped and we have
        segments that represent the exact same road section.
        """
        nodes1 = seg1.get("nodes", [])
        nodes2 = seg2.get("nodes", [])

        if len(nodes1) != len(nodes2):
            return False

        # Check if all nodes match (same direction)
        forward_match = all(
            abs(n1.get("lat", 0) - n2.get("lat", 0)) < tolerance_deg and
            abs(n1.get("lon", 0) - n2.get("lon", 0)) < tolerance_deg
            for n1, n2 in zip(nodes1, nodes2)
        )

        if forward_match:
            return True

        # Check if all nodes match (reversed direction)
        reverse_match = all(
            abs(n1.get("lat", 0) - n2.get("lat", 0)) < tolerance_deg and
            abs(n1.get("lon", 0) - n2.get("lon", 0)) < tolerance_deg
            for n1, n2 in zip(nodes1, reversed(nodes2))
        )

        return reverse_match

    def _merge_street(self, segments: List[Dict], street_name: str) -> List[Dict]:
        """
        Merge street segments using O(n) endpoint index algorithm.

        Includes duplicate detection to handle cases where spatial dedup was skipped.
        """
        if len(segments) <= 1:
            return segments

        # STEP 1: Remove duplicates within this street's segments (O(n) with hashing)
        # This handles cases where spatial deduplication was skipped
        deduplicated = []
        seen_way_sets = set()
        seen_endpoint_pairs = {}  # Maps (start_coord, end_coord) -> segment index for fast lookup
        duplicate_count = 0

        for segment in segments:
            # Check way_ids first (fastest - O(1))
            way_ids = tuple(sorted(segment.get("way_ids", [])))
            if way_ids and way_ids in seen_way_sets:
                duplicate_count += 1
                continue

            # Check endpoint pair hash (fast - O(1) average case)
            nodes = segment.get("nodes", [])
            if len(nodes) >= 2:
                start_node = nodes[0]
                end_node = nodes[-1]

                # Round to tolerance for hashing
                start_coord = (round(start_node.get("lat", 0), 4), round(start_node.get("lon", 0), 4))
                end_coord = (round(end_node.get("lat", 0), 4), round(end_node.get("lon", 0), 4))

                # Check both forward and reverse endpoint pairs
                endpoint_pair_forward = (start_coord, end_coord)
                endpoint_pair_reverse = (end_coord, start_coord)

                existing_idx = None
                if endpoint_pair_forward in seen_endpoint_pairs:
                    existing_idx = seen_endpoint_pairs[endpoint_pair_forward]
                elif endpoint_pair_reverse in seen_endpoint_pairs:
                    existing_idx = seen_endpoint_pairs[endpoint_pair_reverse]

                if existing_idx is not None:
                    # Potential duplicate - verify with full coordinate check
                    existing_segment = deduplicated[existing_idx]
                    if self._are_segments_duplicates(segment, existing_segment):
                        # Merge the way_ids from the duplicate
                        existing_way_ids = set(existing_segment.get("way_ids", []))
                        segment_way_ids = set(segment.get("way_ids", []))
                        existing_segment["way_ids"] = list(existing_way_ids | segment_way_ids)
                        duplicate_count += 1
                        continue

            # Not a duplicate - add to results
            deduplicated.append(segment)
            if way_ids:
                seen_way_sets.add(way_ids)

            # Add to endpoint pair index
            if len(nodes) >= 2:
                seen_endpoint_pairs[endpoint_pair_forward] = len(deduplicated) - 1

        if duplicate_count > 0:
            # Use global print if available, otherwise skip
            try:
                print(f"  Removed {duplicate_count} duplicates from '{street_name}' ({len(segments)} -> {len(deduplicated)} segments)")
            except:
                pass

        # Use deduplicated segments for merging
        segments = deduplicated

        if len(segments) <= 1:
            return segments

        # STEP 2: Build endpoint index for O(1) lookups
        # Index by both node ID (if available) and coordinates
        # Use COARSE rounding (4 decimals ≈ 11m) for spatial proximity, then filter with actual tolerance
        endpoint_index = defaultdict(list)
        node_id_index = defaultdict(list)

        for idx, segment in enumerate(segments):
            if not segment.get("nodes") or len(segment["nodes"]) < 2:
                continue

            start_node = segment["nodes"][0]
            end_node = segment["nodes"][-1]

            # Index by coordinate with COARSE rounding (4 decimals ≈ 11m at equator)
            # This groups nearby endpoints together, then we filter with actual tolerance
            start_coord = (round(start_node["lat"], 4), round(start_node["lon"], 4))
            end_coord = (round(end_node["lat"], 4), round(end_node["lon"], 4))
            endpoint_index[start_coord].append((idx, "start"))
            endpoint_index[end_coord].append((idx, "end"))

            # Index by node ID (preferred for exact matching)
            start_id = start_node.get("id") if isinstance(start_node, dict) else getattr(start_node, "id", None)
            end_id = end_node.get("id") if isinstance(end_node, dict) else getattr(end_node, "id", None)

            if start_id:
                node_id_index[start_id].append((idx, "start"))
            if end_id:
                node_id_index[end_id].append((idx, "end"))

        merged_segments = []
        used = [False] * len(segments)

        # Sort segments by connectivity (process well-connected segments first, spurs last)
        # This prevents starting with dead-end spurs and building chains outward from them
        def count_connections(idx):
            """Count how many other segments this segment can connect to."""
            seg = segments[idx]
            if not seg.get("nodes") or len(seg["nodes"]) < 2:
                return 0

            count = 0
            start_node = seg["nodes"][0]
            end_node = seg["nodes"][-1]
            start_id = start_node.get("id")
            end_id = end_node.get("id")
            # Use same coarse rounding as index (4 decimals)
            start_coord = (round(start_node["lat"], 4), round(start_node["lon"], 4))
            end_coord = (round(end_node["lat"], 4), round(end_node["lon"], 4))

            # Check connections via node_id_index and endpoint_index
            if start_id and node_id_index.get(start_id):
                count += len([i for i, e in node_id_index[start_id] if i != idx])
            elif endpoint_index.get(start_coord):
                count += len([i for i, e in endpoint_index[start_coord] if i != idx])

            if end_id and node_id_index.get(end_id):
                count += len([i for i, e in node_id_index[end_id] if i != idx])
            elif endpoint_index.get(end_coord):
                count += len([i for i, e in endpoint_index[end_coord] if i != idx])

            return count

        # Create list of (index, segment, connection_count) and sort by connection count (descending)
        indexed_segments = [(idx, seg, count_connections(idx)) for idx, seg in enumerate(segments)]
        indexed_segments.sort(key=lambda x: x[2], reverse=True)

        for start_idx, segment, conn_count in indexed_segments:
            if used[start_idx]:
                continue

            if not segment.get("nodes") or len(segment["nodes"]) < 2:
                merged_segments.append(segment)
                used[start_idx] = True
                continue

            # Start a new merged segment
            current_segment = segment.copy()
            used[start_idx] = True

            # Cache current endpoints (both node IDs and coordinates)
            # Use same coarse rounding as index (4 decimals)
            start_node = current_segment["nodes"][0]
            end_node = current_segment["nodes"][-1]

            start_coord = (round(start_node["lat"], 4), round(start_node["lon"], 4))
            end_coord = (round(end_node["lat"], 4), round(end_node["lon"], 4))
            start_id = start_node.get("id") if isinstance(start_node, dict) else getattr(start_node, "id", None)
            end_id = end_node.get("id") if isinstance(end_node, dict) else getattr(end_node, "id", None)

            current_endpoints = (start_coord, end_coord, start_id, end_id)

            # Try to grow in both directions
            max_growth_iterations = 100
            for _ in range(max_growth_iterations):
                start_coord, end_coord, start_id, end_id = current_endpoints

                # Check if neighbors exist (prefer node ID index, fall back to coordinate index)
                start_neighbors = []
                end_neighbors = []

                # Try node ID matching first (exact)
                if start_id and node_id_index[start_id]:
                    start_neighbors = node_id_index[start_id]
                elif endpoint_index[start_coord]:
                    start_neighbors = endpoint_index[start_coord]

                if end_id and node_id_index[end_id]:
                    end_neighbors = node_id_index[end_id]
                elif endpoint_index[end_coord]:
                    end_neighbors = endpoint_index[end_coord]

                # Early exit when no neighbors exist
                if not start_neighbors and not end_neighbors:
                    break

                merge_happened = False

                # Try to connect at start
                # Collect valid neighbors with scores (prioritize longer chains, avoid dead-ends)
                valid_start_neighbors = []
                for neighbor_idx, neighbor_endpoint in start_neighbors:
                    if used[neighbor_idx]:
                        continue

                    neighbor = segments[neighbor_idx]
                    if not neighbor.get("nodes") or len(neighbor["nodes"]) < 2:
                        continue

                    # Dead-end detection: skip neighbors that are dead-end spurs
                    # Unless current segment is also a dead-end (allow two dead-ends to connect)
                    neighbor_far_endpoint = "end" if neighbor_endpoint == "start" else "start"
                    if self._is_dead_end(neighbor, neighbor_far_endpoint, segments, used):
                        # Check if current segment's start is also a dead-end
                        if not self._is_dead_end(current_segment, "start", segments, used):
                            continue  # Skip this dead-end spur

                    # Score this merge by chain potential
                    score = self._score_merge_potential(current_segment, neighbor, neighbor_endpoint, segments, used)
                    valid_start_neighbors.append((neighbor_idx, neighbor_endpoint, neighbor, score))

                # Sort by score (highest first) and try best merge
                valid_start_neighbors.sort(key=lambda x: x[3], reverse=True)

                for neighbor_idx, neighbor_endpoint, neighbor, score in valid_start_neighbors:
                    # Connect the neighbor
                    connected = self._connect_segments(current_segment, neighbor, neighbor_endpoint, "start")
                    if connected:
                        current_segment = connected
                        used[neighbor_idx] = True
                        merge_happened = True

                        # Update endpoint cache (use coarse rounding)
                        start_node = current_segment["nodes"][0]
                        end_node = current_segment["nodes"][-1]
                        new_start = (round(start_node["lat"], 4), round(start_node["lon"], 4))
                        new_end = (round(end_node["lat"], 4), round(end_node["lon"], 4))
                        current_endpoints = (new_start, new_end)

                        # Remove used segment from index (use coarse rounding)
                        neighbor_start = neighbor["nodes"][0]
                        neighbor_end = neighbor["nodes"][-1]
                        neighbor_start_coord = (round(neighbor_start["lat"], 4), round(neighbor_start["lon"], 4))
                        neighbor_end_coord = (round(neighbor_end["lat"], 4), round(neighbor_end["lon"], 4))
                        endpoint_index[neighbor_start_coord] = [
                            (idx, ep) for idx, ep in endpoint_index[neighbor_start_coord]
                            if idx != neighbor_idx
                        ]
                        endpoint_index[neighbor_end_coord] = [
                            (idx, ep) for idx, ep in endpoint_index[neighbor_end_coord]
                            if idx != neighbor_idx
                        ]
                        if neighbor_start_id:
                            node_id_index[neighbor_start_id] = [
                                (idx, ep) for idx, ep in node_id_index[neighbor_start_id]
                                if idx != neighbor_idx
                            ]
                        if neighbor_end_id:
                            node_id_index[neighbor_end_id] = [
                                (idx, ep) for idx, ep in node_id_index[neighbor_end_id]
                                if idx != neighbor_idx
                            ]
                        break

                if merge_happened:
                    continue

                # Try to connect at end
                # Collect valid neighbors with scores (prioritize longer chains, avoid dead-ends)
                valid_end_neighbors = []
                for neighbor_idx, neighbor_endpoint in end_neighbors:
                    if used[neighbor_idx]:
                        continue

                    neighbor = segments[neighbor_idx]
                    if not neighbor.get("nodes") or len(neighbor["nodes"]) < 2:
                        continue

                    # Dead-end detection: skip neighbors that are dead-end spurs
                    # Unless current segment is also a dead-end (allow two dead-ends to connect)
                    neighbor_far_endpoint = "end" if neighbor_endpoint == "start" else "start"
                    if self._is_dead_end(neighbor, neighbor_far_endpoint, segments, used):
                        # Check if current segment's end is also a dead-end
                        if not self._is_dead_end(current_segment, "end", segments, used):
                            continue  # Skip this dead-end spur

                    # Score this merge by chain potential
                    score = self._score_merge_potential(current_segment, neighbor, neighbor_endpoint, segments, used)
                    valid_end_neighbors.append((neighbor_idx, neighbor_endpoint, neighbor, score))

                # Sort by score (highest first) and try best merge
                valid_end_neighbors.sort(key=lambda x: x[3], reverse=True)

                for neighbor_idx, neighbor_endpoint, neighbor, score in valid_end_neighbors:
                    # Connect the neighbor
                    connected = self._connect_segments(current_segment, neighbor, neighbor_endpoint, "end")
                    if connected:
                        current_segment = connected
                        used[neighbor_idx] = True
                        merge_happened = True

                        # Update endpoint cache (coordinates + node IDs, use coarse rounding)
                        start_node = current_segment["nodes"][0]
                        end_node = current_segment["nodes"][-1]
                        new_start_coord = (round(start_node["lat"], 4), round(start_node["lon"], 4))
                        new_end_coord = (round(end_node["lat"], 4), round(end_node["lon"], 4))
                        new_start_id = start_node.get("id") if isinstance(start_node, dict) else getattr(start_node, "id", None)
                        new_end_id = end_node.get("id") if isinstance(end_node, dict) else getattr(end_node, "id", None)
                        current_endpoints = (new_start_coord, new_end_coord, new_start_id, new_end_id)

                        # Remove used segment from indexes (use coarse rounding)
                        neighbor_start = neighbor["nodes"][0]
                        neighbor_end = neighbor["nodes"][-1]
                        neighbor_start_id = neighbor_start.get("id") if isinstance(neighbor_start, dict) else getattr(neighbor_start, "id", None)
                        neighbor_end_id = neighbor_end.get("id") if isinstance(neighbor_end, dict) else getattr(neighbor_end, "id", None)
                        neighbor_start_coord = (round(neighbor_start["lat"], 4), round(neighbor_start["lon"], 4))
                        neighbor_end_coord = (round(neighbor_end["lat"], 4), round(neighbor_end["lon"], 4))
                        endpoint_index[neighbor_start_coord] = [
                            (idx, ep) for idx, ep in endpoint_index[neighbor_start_coord]
                            if idx != neighbor_idx
                        ]
                        endpoint_index[neighbor_end_coord] = [
                            (idx, ep) for idx, ep in endpoint_index[neighbor_end_coord]
                            if idx != neighbor_idx
                        ]
                        if neighbor_start_id:
                            node_id_index[neighbor_start_id] = [
                                (idx, ep) for idx, ep in node_id_index[neighbor_start_id]
                                if idx != neighbor_idx
                            ]
                        if neighbor_end_id:
                            node_id_index[neighbor_end_id] = [
                                (idx, ep) for idx, ep in node_id_index[neighbor_end_id]
                                if idx != neighbor_idx
                            ]
                        break

                if not merge_happened:
                    break

            merged_segments.append(current_segment)

        return merged_segments

    def _connect_segments(self, seg1: Dict, seg2: Dict, seg2_endpoint: str, connection_point: str) -> Dict:
        """Connect two segments without unnecessary list copies."""
        nodes1 = seg1["nodes"]
        nodes2 = seg2["nodes"]

        # Determine if we need to reverse either segment
        reverse_first = (connection_point == "start")
        reverse_second = (seg2_endpoint == "start")

        # Get or create way_boundaries for each segment
        # Format: [(start_idx, end_idx, way_id), ...]
        bounds1 = seg1.get("way_boundaries")
        if bounds1 is None:
            # Initialize from way_ids if not present
            # Create boundary entry for EACH way_id (all covering full node range)
            way_ids_1 = seg1.get("way_ids", [])
            if way_ids_1:
                bounds1 = [(0, len(nodes1), wid) for wid in way_ids_1]
            else:
                bounds1 = []

        bounds2 = seg2.get("way_boundaries")
        if bounds2 is None:
            # Create boundary entry for EACH way_id (all covering full node range)
            way_ids_2 = seg2.get("way_ids", [])
            if way_ids_2:
                bounds2 = [(0, len(nodes2), wid) for wid in way_ids_2]
            else:
                bounds2 = []

        # Build connected nodes directly based on reversal flags
        if reverse_first and reverse_second:
            connected_nodes = nodes1[::-1] + nodes2[-2::-1]
            # Reverse bounds1 indices, then append reversed bounds2 with offset
            len1 = len(nodes1)
            new_bounds1 = [(len1 - end, len1 - start, wid) for (start, end, wid) in reversed(bounds1)]
            # For bounds2: reverse, then offset by (len1 - 1) since we skip 1 node
            len2 = len(nodes2)
            offset = len1 - 1
            new_bounds2 = [(len2 - end + offset, len2 - start + offset, wid) for (start, end, wid) in reversed(bounds2)]
        elif reverse_first:
            connected_nodes = nodes1[::-1] + nodes2[1:]
            len1 = len(nodes1)
            new_bounds1 = [(len1 - end, len1 - start, wid) for (start, end, wid) in reversed(bounds1)]
            # bounds2 offset by (len1 - 1), skip first node
            offset = len1 - 1
            new_bounds2 = [(start + offset, end + offset, wid) for (start, end, wid) in bounds2]
        elif reverse_second:
            connected_nodes = nodes1 + nodes2[-2::-1]
            new_bounds1 = list(bounds1)
            # Reverse bounds2 and offset by len1, skip last node of bounds2
            len2 = len(nodes2)
            offset = len(nodes1) - 1
            new_bounds2 = [(len2 - end + offset, len2 - start + offset, wid) for (start, end, wid) in reversed(bounds2)]
        else:
            connected_nodes = nodes1 + nodes2[1:]
            new_bounds1 = list(bounds1)
            # bounds2 offset by (len1 - 1), skip first node
            offset = len(nodes1) - 1
            new_bounds2 = [(start + offset, end + offset, wid) for (start, end, wid) in bounds2]

        # Merge way_boundaries
        merged_bounds = new_bounds1 + new_bounds2

        # Create merged segment
        merged = seg1.copy()
        merged["nodes"] = connected_nodes
        merged["way_boundaries"] = merged_bounds

        # Merge way_ids (keep for backward compatibility)
        way_ids_1 = set(seg1.get("way_ids", []))
        way_ids_2 = set(seg2.get("way_ids", []))
        merged["way_ids"] = list(way_ids_1 | way_ids_2)

        return merged

    def _merge_segments_for_street(
        self,
        segments: List[Dict],
        street_name: str,
        persistence,
    ) -> List[Dict]:
        """Merge segments with spatial optimization for large streets with common names."""

        if len(segments) <= 1:
            return segments

        # Special handling for very common generic road names
        common_names = {
            "service",
            "track",
            "path",
            "footway",
            "cycleway",
            "bridleway",
            "steps",
            "residential",
            "unclassified",
        }
        is_common_name = street_name.lower() in common_names

        # Apply spatial grouping only for common names when segments > 50
        if is_common_name and len(segments) > 50:
            return self._merge_large_street_with_spatial_grouping(segments, street_name)
        else:
            # Use basic merging for all other streets regardless of length
            return self._merge_street(segments, street_name)

    def merge_boundary_segments(self, all_segments: List[Dict], persistence) -> List[Dict]:
        """
        Merge segments that represent the same road across region boundaries.

        This is the main entry point for boundary merging. It automatically selects
        the optimal processing strategy based on dataset size and available resources.

        STRATEGY SELECTION:
            - Serial mode: datasets < 1000 segments OR parallel disabled OR single core
            - Parallel mode: datasets >= 1000 segments AND parallel enabled AND multiple cores

        FEATURES:
            - Automatic checkpoint creation for long-running merges
            - Resume capability for interrupted merges
            - Progress tracking with ETA
            - Memory monitoring and cleanup
            - Duplicate detection and filtering

        CHECKPOINTING:
            Creates checkpoints every N streets (adaptive based on dataset size):
            - Small datasets (<10K): checkpoint every 1000 streets
            - Medium datasets (10K-100K): checkpoint every 500 streets
            - Large datasets (>100K): checkpoint every 100 streets

            Checkpoints allow resuming after:
            - Process interruption (Ctrl+C)
            - System crashes
            - Memory exhaustion
            - Timeout errors

        PERFORMANCE:
            Serial mode: ~1,000 segments/second
            Parallel mode (16 workers): ~5,000-10,000 segments/second
            Checkpoint overhead: ~5% for large merges

        Args:
            all_segments: List of segment dictionaries to merge
                         Each segment should have: nodes, way_ids, street_name
            persistence: ChunkPersistenceManager instance for checkpointing

        Returns:
            List of merged segments with duplicate segments removed

        Example:
            merger = BoundaryMerger(coordinate_tolerance=0.002)
            merged = merger.merge_boundary_segments(segments, persistence)
            print(f"Merged {len(segments)} segments into {len(merged)} segments")
        """
        print(f"\nPerforming boundary merge on {len(all_segments):,} segments...")

        # Check for existing progress
        progress_data = persistence.load_boundary_merge_progress()

        if progress_data:
            print("Resuming boundary merge from checkpoint...")
            print(f"  - Streets processed: {progress_data.get('streets_processed', 0)}")
            print(f"  - Total streets: {progress_data.get('total_streets', 0)}")

            # Check if it was running in parallel mode
            if progress_data.get('parallel_mode', False):
                print("  - Resuming in parallel mode")
                # TODO: Implement parallel resume (for now, continue with serial)

            return self._resume_boundary_merge(all_segments, progress_data, persistence)

        # Decide whether to use parallel or serial processing
        # Use parallel for datasets with >1000 segments (threshold can be tuned)
        if self.parallel_enabled and len(all_segments) > 1000 and self.max_workers > 1:
            print(f"✓ Using PARALLEL processing with {self.max_workers} workers")
            return self._perform_boundary_merge_parallel(all_segments, persistence)
        else:
            print("✓ Using SERIAL processing (dataset too small or parallel disabled)")
            return self._perform_boundary_merge(all_segments, persistence)

    def _perform_boundary_merge(
        self, all_segments: List[Dict], persistence
    ) -> List[Dict]:
        """Perform boundary merge with smart checkpointing (serial mode)."""

        # Import these from main module if they exist, otherwise stub them
        try:
            from climb_analyzer import SmartCheckpointer, check_and_cleanup_memory, signal_handler
        except ImportError:
            # Create stub implementations for standalone usage
            class StubSignalHandler:
                kill_now = False
                def set_operation(self, *args, **kwargs):
                    pass
            signal_handler = StubSignalHandler()

            def check_and_cleanup_memory(force_cleanup=False):
                pass

            class SmartCheckpointer:
                def __init__(self, total, name):
                    self.total = total
                    self.checkpoint_interval = max(1, total // 20)
                def should_checkpoint(self, i):
                    return i % self.checkpoint_interval == 0
                def get_checkpoint_info(self, i):
                    return {'time_until_next_min': 5.0}

        # Group segments by street name
        print("Grouping segments by street name...")
        segments_by_name = defaultdict(list)

        with tqdm(
            total=len(all_segments),
            desc="Grouping by street name",
            unit="segments",
            miniters=len(all_segments) // 100,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for segment in all_segments:
                street_name = segment.get("street_name", "").strip()
                if street_name:
                    segments_by_name[street_name].append(segment)

                if pbar.n % max(1, len(all_segments) // 100) == 0:
                    pbar.set_postfix({"unique_streets": len(segments_by_name)})
                pbar.update(1)

        # Prepare for processing
        streets_to_merge = {name: segs for name, segs in segments_by_name.items() if len(segs) > 1}
        single_segments = []

        for name, segs in segments_by_name.items():
            if len(segs) == 1:
                single_segments.extend(segs)

        print("Grouping complete. \n\n")

        # FREE MEMORY: Delete large data structures we no longer need
        del all_segments  # Original segment list no longer needed
        del segments_by_name  # Temporary grouping dict no longer needed

        # Force garbage collection
        import gc
        gc.collect()

        print("   Freed memory from segment grouping\n")

        # Process street groups with smart checkpointing
        merged_segments = single_segments.copy()
        processed_streets = []
        processed_street_segments = {}

        street_items = list(streets_to_merge.items())

        # RANDOMIZE street order to distribute heavy processing throughout
        random.shuffle(street_items)
        print("Randomized street processing order to distribute workload evenly")
        print("\n   Note: Common street names (like 'service', 'track', 'path') have thousands")
        print("   of segments and will temporarily slow progress when encountered.")
        print("   This is normal - progress will speed up as these complete.\n")

        # Initialize smart checkpointer
        checkpointer = SmartCheckpointer(len(street_items), "Boundary Merge")

        with tqdm(
            total=len(street_items),
            desc="Merging street segments",
            unit="streets",
            miniters=1,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            postfix_update_counter = 0
            for i, (street_name, street_segments) in enumerate(street_items):
                # Signal handler check every 10 streets
                if i % 10 == 0:
                    signal_handler.set_operation(
                        "boundary_merge",
                        {
                            "streets_processed": i,
                            "total_streets": len(street_items),
                            "processed_street_names": processed_streets,
                            "processed_street_segments": processed_street_segments,
                            "merged_segments_count": len(merged_segments),
                            "single_segments": single_segments,
                            "segments_by_name_remaining": {
                                k: v
                                for k, v in streets_to_merge.items()
                                if k not in processed_streets
                            },
                            "timestamp": time.time(),
                        },
                    )

                if signal_handler.kill_now:
                    progress_data = {
                        "streets_processed": i,
                        "total_streets": len(street_items),
                        "processed_street_names": processed_streets,
                        "processed_street_segments": processed_street_segments,
                        "merged_segments_count": len(merged_segments),
                        "single_segments": single_segments,
                        "segments_by_name_remaining": {
                            k: v for k, v in streets_to_merge.items() if k not in processed_streets
                        },
                        "timestamp": time.time(),
                    }
                    persistence.save_boundary_merge_progress(progress_data)
                    print("Boundary merge progress saved. Analysis can be resumed.")
                    sys.exit(0)

                try:
                    # Process this street
                    display_name = (
                        street_name[:12] + "..." if len(street_name) > 15 else street_name
                    )

                    # Show detailed progress for different street sizes
                    if postfix_update_counter % 10 == 0:
                        if len(street_segments) > 500:
                            pbar.set_postfix_str(
                                f"SPATIAL: {display_name} ({len(street_segments):,} segs)"
                            )
                        elif len(street_segments) > 100:
                            pbar.set_postfix_str(
                                f"LARGE: {display_name} ({len(street_segments)} segs)"
                            )
                        elif len(street_segments) > 50:
                            pbar.set_postfix_str(
                                f"Processing {display_name} ({len(street_segments)} segs)"
                            )

                    postfix_update_counter += 1

                    merged_street_segments = self._merge_segments_for_street(
                        street_segments, street_name, persistence
                    )
                    merged_segments.extend(merged_street_segments)
                    processed_streets.append(street_name)
                    processed_street_segments[street_name] = merged_street_segments

                    # SMART CHECKPOINT CHECK
                    if checkpointer.should_checkpoint(i):
                        progress_data = {
                            "streets_processed": i + 1,
                            "total_streets": len(street_items),
                            "processed_street_names": processed_streets,
                            "processed_street_segments": processed_street_segments,
                            "merged_segments_count": len(merged_segments),
                            "single_segments": single_segments,
                            "segments_by_name_remaining": {
                                k: v
                                for k, v in streets_to_merge.items()
                                if k not in processed_streets
                            },
                            "timestamp": time.time(),
                        }
                        persistence.save_boundary_merge_progress(progress_data)

                        # Update progress bar with checkpoint info
                        info = checkpointer.get_checkpoint_info(i)
                        pbar.set_postfix(
                            {
                                "current": display_name,
                                "segments": len(street_segments),
                                "total_merged": len(merged_segments),
                                "next_save": f"{info['time_until_next_min']:.1f}min",
                            }
                        )
                    else:
                        pbar.set_postfix(
                            {
                                "current": display_name,
                                "segments": len(street_segments),
                                "total_merged": len(merged_segments),
                            }
                        )

                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing street '{street_name}': {e}")
                    # Save checkpoint before failing
                    progress_data = {
                        "streets_processed": i,
                        "total_streets": len(street_items),
                        "processed_street_names": processed_streets,
                        "processed_street_segments": processed_street_segments,
                        "merged_segments_count": len(merged_segments),
                        "single_segments": single_segments,
                        "segments_by_name_remaining": {
                            k: v for k, v in streets_to_merge.items() if k not in processed_streets
                        },
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    persistence.save_boundary_merge_progress(progress_data)
                    raise

        print(
            f"Boundary merge completed: {len(all_segments)} -> {len(merged_segments)} segments"
        )
        persistence.clear_boundary_merge_progress()
        check_and_cleanup_memory(force_cleanup=True)

        return merged_segments

    def _perform_boundary_merge_parallel(
        self, all_segments: List[Dict], persistence
    ) -> List[Dict]:
        """
        Parallel boundary merge using ProcessPoolExecutor.

        Processes streets in batches across multiple CPU cores for dramatic speedup.
        Includes output streaming to prevent memory accumulation.
        """
        from tqdm import tqdm

        # Import these from main module if they exist, otherwise stub them
        try:
            from climb_analyzer import SmartCheckpointer, check_and_cleanup_memory, signal_handler
        except ImportError:
            # Create stub implementations
            class StubSignalHandler:
                kill_now = False
                def set_operation(self, *args, **kwargs):
                    pass
            signal_handler = StubSignalHandler()

            def check_and_cleanup_memory(force_cleanup=False):
                pass

            class SmartCheckpointer:
                def __init__(self, total, name):
                    self.total = total
                    self.checkpoint_interval = max(1, total // 20)
                def should_checkpoint(self, i):
                    return i % self.checkpoint_interval == 0
                def get_checkpoint_info(self, i):
                    return {'time_until_next_min': 5.0}

        # Group segments by street name
        print(f"Grouping {len(all_segments):,} segments by street name...")
        segments_by_name = defaultdict(list)

        with tqdm(
            total=len(all_segments),
            desc="Grouping by street name",
            unit="segments",
            miniters=len(all_segments) // 100,
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for segment in all_segments:
                street_name = segment.get("street_name", "").strip()
                if street_name:
                    segments_by_name[street_name].append(segment)

                if pbar.n % max(1, len(all_segments) // 100) == 0:
                    pbar.set_postfix({"unique_streets": len(segments_by_name)})
                pbar.update(1)

        # Separate single-segment streets (no merging needed)
        streets_to_merge = {name: segs for name, segs in segments_by_name.items() if len(segs) > 1}
        single_segments = []

        for name, segs in segments_by_name.items():
            if len(segs) == 1:
                single_segments.extend(segs)

        print(f"\nProcessing {len(streets_to_merge):,} streets with multiple segments")
        print(f"Skipping {len(single_segments):,} single-segment streets (no merge needed)")

        # FREE MEMORY: Delete large data structures we no longer need
        # This is critical - all_segments is ~1.5M items consuming gigabytes
        del all_segments  # Original segment list no longer needed
        del segments_by_name  # Temporary grouping dict no longer needed

        # Force garbage collection to free memory immediately
        import gc
        gc.collect()

        print("   Freed memory from segment grouping\n")

        # Calculate safe worker count based on available memory
        safe_workers = self._calculate_safe_worker_count(len(single_segments) + sum(len(segs) for segs in streets_to_merge.values()))
        actual_workers = safe_workers

        from climb_analyzer.utils.formatting import print_dim

        print_dim(f"Using {actual_workers} parallel workers with batch size {self.batch_size}\n")

        street_items = list(streets_to_merge.items())

        # RANDOMIZE street order to distribute heavy processing throughout
        random.shuffle(street_items)
        print_dim("Randomized street processing order to distribute workload evenly")
        print_dim("\n   Note: Common street names (like 'service', 'track', 'path') have thousands")
        print_dim("   of segments and will temporarily slow progress when encountered.")
        print_dim("   This is normal - progress will speed up as these complete.\n")

        # Create batches for parallel processing
        batches = []
        for i in range(0, len(street_items), self.batch_size):
            batch = street_items[i:i + self.batch_size]
            batches.append((
                batch,
                self.coordinate_tolerance,
                self.distance_tolerance_m
            ))

        print(f"Split into {len(batches)} batches for parallel processing\n")

        # Initialize smart checkpointer
        checkpointer = SmartCheckpointer(len(street_items), "Boundary Merge")

        # Process batches in parallel with streaming output
        merged_segments = single_segments.copy()
        processed_streets_count = 0
        batch_checkpoint_dir = persistence.analysis_dir / "batch_checkpoints"
        batch_checkpoint_dir.mkdir(exist_ok=True)

        try:
            executor = ProcessPoolExecutor(max_workers=actual_workers)
        except Exception as e:
            print(f"\n⚠️  Warning: Could not initialize parallel processing: {e}")
            print("Falling back to SERIAL processing (slower but more reliable)\n")
            return self._perform_boundary_merge(all_segments, persistence)

        with executor:
            # Test worker pool with a small test batch first
            print("Testing worker pool with first batch...")
            try:
                test_future = executor.submit(_merge_street_batch_worker_top_level, batches[0])
                # Wait a moment to see if it crashes immediately
                time.sleep(0.5)
                if test_future.done() and test_future.exception():
                    raise test_future.exception()
                print("✓ Worker pool test successful\n")
            except Exception as e:
                print(f"\n❌ Worker pool test failed: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("\nFalling back to SERIAL processing (slower but more reliable)\n")
                executor.shutdown(wait=False)
                return self._perform_boundary_merge(all_segments, persistence)

            # Submit all batches
            print(f"Submitting {len(batches)} batches to worker pool...")
            try:
                future_to_batch = {
                    executor.submit(_merge_street_batch_worker_top_level, batch_args): idx
                    for idx, batch_args in enumerate(batches)
                }
                print(f"✓ All {len(batches)} batches submitted successfully\n")
            except Exception as e:
                print(f"\n❌ Error submitting batches: {e}")
                print("Falling back to SERIAL processing\n")
                executor.shutdown(wait=False)
                return self._perform_boundary_merge(all_segments, persistence)

            # Process completed batches with progress bar
            with tqdm(
                total=len(street_items),
                desc="Merging street segments (parallel)",
                unit="streets",
                miniters=1,
                dynamic_ncols=True,
                ascii=" ▏▎▍▌▋▊▉█",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ) as pbar:

                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]

                    try:
                        # Get results from completed batch
                        batch_results = future.result()

                        # Stream to checkpoint file immediately (memory efficient)
                        batch_checkpoint_file = batch_checkpoint_dir / f"batch_{batch_idx:04d}.pkl"
                        with open(batch_checkpoint_file, 'wb') as f:
                            pickle.dump(batch_results, f)

                        # Track progress
                        batch_street_count = len(batches[batch_idx][0])
                        processed_streets_count += batch_street_count

                        # Update progress bar
                        pbar.update(batch_street_count)
                        # Count completed batches
                        completed_batches = sum(1 for f in future_to_batch.keys() if f.done())
                        pbar.set_postfix({
                            "batch": f"{completed_batches}/{len(batches)}",
                            "workers": actual_workers,
                            "mem_saved": f"{len(batch_results):,} segs on disk"
                        })

                        # Checkpoint if needed
                        if checkpointer.should_checkpoint(processed_streets_count - 1):
                            # Get list of completed batch indices
                            completed_batch_indices = [
                                future_to_batch[f] for f in future_to_batch.keys()
                                if f.done()
                            ]
                            progress_data = {
                                "streets_processed": processed_streets_count,
                                "total_streets": len(street_items),
                                "batches_completed": completed_batch_indices,
                                "batch_checkpoint_dir": str(batch_checkpoint_dir),
                                "single_segments": single_segments,
                                "timestamp": time.time(),
                                "parallel_mode": True
                            }
                            persistence.save_boundary_merge_progress(progress_data)

                            info = checkpointer.get_checkpoint_info(processed_streets_count - 1)
                            print(f"\n  Checkpoint saved: {processed_streets_count}/{len(street_items)} streets, "
                                  f"next save in {info['time_until_next_min']:.1f}min")

                        # Signal handler check
                        if signal_handler.kill_now:
                            print("\n\nInterrupted! Saving progress...")
                            # Get list of completed batch indices
                            completed_batch_indices = [
                                future_to_batch[f] for f in future_to_batch.keys()
                                if f.done()
                            ]
                            progress_data = {
                                "streets_processed": processed_streets_count,
                                "total_streets": len(street_items),
                                "batches_completed": completed_batch_indices,
                                "batch_checkpoint_dir": str(batch_checkpoint_dir),
                                "single_segments": single_segments,
                                "timestamp": time.time(),
                                "parallel_mode": True
                            }
                            persistence.save_boundary_merge_progress(progress_data)
                            print("Progress saved. Analysis can be resumed.")
                            sys.exit(0)

                    except Exception as e:
                        print(f"\n❌ Error processing batch {batch_idx}: {e}")
                        # Continue with other batches
                        processed_streets_count += len(batches[batch_idx][0])
                        pbar.update(len(batches[batch_idx][0]))

        # Merge all batch checkpoint files
        print("\nMerging batch results from disk...")

        for batch_idx in tqdm(
            range(len(batches)),
            desc="Loading batch checkpoints",
            unit=" batch",
            ncols=100,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ):
            batch_checkpoint_file = batch_checkpoint_dir / f"batch_{batch_idx:04d}.pkl"
            if batch_checkpoint_file.exists():
                try:
                    with open(batch_checkpoint_file, 'rb') as f:
                        batch_results = _safe_pickle_load(f)
                    merged_segments.extend(batch_results)

                    # Delete checkpoint file to free disk space
                    batch_checkpoint_file.unlink()
                except Exception as e:
                    tqdm.write(f"⚠️  Warning: Could not load batch {batch_idx}: {e}")

        # Cleanup batch checkpoint directory
        try:
            if batch_checkpoint_dir.exists() and not any(batch_checkpoint_dir.iterdir()):
                batch_checkpoint_dir.rmdir()
        except Exception:
            pass

        print(f"\nParallel boundary merge completed: {len(all_segments):,} -> {len(merged_segments):,} segments")
        print(f"Used {self.max_workers} workers with {self.batch_size} streets/batch")
        persistence.clear_boundary_merge_progress()
        check_and_cleanup_memory(force_cleanup=True)

        return merged_segments

    def _resume_boundary_merge(
        self,
        all_segments: List[Dict],
        progress_data: Dict,
        persistence,
    ) -> List[Dict]:
        """Resume boundary merge from checkpoint."""

        streets_processed = progress_data.get("streets_processed", 0)
        processed_street_names = set(progress_data.get("processed_street_names", []))
        processed_street_segments = progress_data.get("processed_street_segments", {})
        single_segments = progress_data.get("single_segments", [])
        segments_by_name_remaining = progress_data.get("segments_by_name_remaining", {})

        print(f"Resuming boundary merge from street {streets_processed}")

        # Rebuild merged segments from checkpoint data
        merged_segments = single_segments.copy()

        # Add already processed street segments
        for street_name, street_segments in processed_street_segments.items():
            merged_segments.extend(street_segments)

        print(f"Restored {len(merged_segments)} segments from checkpoint")

        # Check if we have remaining work
        if not segments_by_name_remaining:
            print("All streets already processed!")
            return merged_segments

        print(f"Streets remaining to process: {len(segments_by_name_remaining)}")

        # Process remaining streets
        checkpoint_interval = max(1, len(segments_by_name_remaining) // 20)
        street_items = list(segments_by_name_remaining.items())

        # Calculate total for proper progress display (processed + remaining)
        total_streets = streets_processed + len(street_items)

        with tqdm(
            total=total_streets,
            initial=streets_processed,  # Show already-processed streets
            desc="Processing streets",
            unit="streets",
            dynamic_ncols=True,
            ascii=" ▏▎▍▌▋▊▉█",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            # Force immediate render when resuming
            if streets_processed > 0:
                pbar.refresh()
            for i, (street_name, street_segments) in enumerate(street_items):
                try:
                    merged_street_segments = self._merge_segments_for_street(
                        street_segments, street_name, persistence
                    )

                    merged_segments.extend(merged_street_segments)
                    processed_street_names.add(street_name)
                    processed_street_segments[street_name] = merged_street_segments

                    # Save checkpoint periodically
                    if i % checkpoint_interval == 0 or i == len(street_items) - 1:
                        # Update remaining work
                        remaining_items = {k: v for j, (k, v) in enumerate(street_items) if j > i}

                        progress_data_updated = {
                            "streets_processed": streets_processed + i + 1,
                            "total_streets": progress_data.get("total_streets", 0),
                            "processed_street_names": list(processed_street_names),
                            "processed_street_segments": processed_street_segments,
                            "merged_segments_count": len(merged_segments),
                            "single_segments": single_segments,
                            "segments_by_name_remaining": remaining_items,
                            "timestamp": time.time(),
                        }
                        persistence.save_boundary_merge_progress(progress_data_updated)

                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing street '{street_name}': {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)

        print(f"Resume completed: {len(merged_segments)} total segments")
        persistence.clear_boundary_merge_progress()

        return merged_segments


__all__ = [
    'BoundaryMerger',
    '_merge_street_batch_worker_top_level',
]

# Backward compatibility alias (deprecated)
CrossChunkRoadMerger = BoundaryMerger
