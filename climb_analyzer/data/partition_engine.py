#!/usr/bin/env python3
"""
Geographic Partition Engine

Partitions large climb datasets into multiple geographic regions to:
1. Keep files under GitHub's 2GB release asset limit
2. Enable sql.js WASM to load databases on iOS Safari (300MB-2GB memory limit)

Two-tier approach:
- Tier 1: Use predefined Geofabrik subregions (e.g., California -> NorCal/SoCal)
- Tier 2: Automatic Quadtree subdivision when predefined partitions don't exist or are too large

Size estimation: ~1,286 bytes per climb row (with indexes)
- 1.5GB target = ~1.1M climbs max per partition
- 1.8GB safe limit = ~1.3M climbs max per partition
- 900K climbs target provides safety margin
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Size thresholds
MAX_CLIMBS_PER_PARTITION = 900_000  # ~1.3GB, safe under 2GB limit
SIZE_THRESHOLD_BYTES = 1_500_000_000  # 1.5GB - trigger partitioning above this
BYTES_PER_CLIMB = 1_286  # Average bytes per climb row including indexes
MAX_QUADTREE_DEPTH = 4  # Cap at 16 partitions max per region

# Bounds type: (min_lat, min_lon, max_lat, max_lon)
Bounds = Tuple[float, float, float, float]


@dataclass
class PartitionInfo:
    """Information about a single geographic partition."""
    partition_id: str  # e.g., "norcal", "northeast", "ne_sw"
    display_name: str  # e.g., "Northern California", "Northeast"
    bounds: Bounds  # (min_lat, min_lon, max_lat, max_lon)
    climb_count: int
    file_path: Optional[Path] = None  # Set after export
    file_size: Optional[int] = None  # Set after export


def estimate_database_size(climb_count: int) -> int:
    """Estimate the SQLite database size for a given number of climbs."""
    return climb_count * BYTES_PER_CLIMB


def needs_partitioning(climb_count: int) -> bool:
    """Check if a dataset needs to be partitioned based on climb count."""
    estimated_size = estimate_database_size(climb_count)
    return estimated_size > SIZE_THRESHOLD_BYTES


def get_climb_coords(climb: Any) -> Tuple[float, float]:
    """
    Extract latitude and longitude from a climb object.

    Handles both ClimbMetrics objects (with start_lat/start_lon) and
    dictionaries (with lat/lon keys).
    """
    if hasattr(climb, 'start_lat'):
        return (climb.start_lat, climb.start_lon)
    elif isinstance(climb, dict):
        return (climb.get('lat') or climb.get('Latitude'),
                climb.get('lon') or climb.get('Longitude'))
    else:
        raise ValueError(f"Unknown climb type: {type(climb)}")


def _in_bounds(lat: float, lon: float, bounds: Bounds) -> bool:
    """Check if coordinates are within bounds (min_lat, min_lon, max_lat, max_lon)."""
    min_lat, min_lon, max_lat, max_lon = bounds
    return min_lat <= lat < max_lat and min_lon <= lon < max_lon


def _generate_partition_name(quadrant_path: str) -> str:
    """
    Convert quadrant path to user-friendly display name.

    Args:
        quadrant_path: e.g., "ne", "sw_nw", "ne_se_nw"

    Returns:
        User-friendly name like "Northeast", "Northern Southwest", etc.
    """
    if not quadrant_path:
        return "Full Region"

    # Map quadrants to cardinal directions
    mapping = {
        "ne": "Northeast",
        "nw": "Northwest",
        "se": "Southeast",
        "sw": "Southwest"
    }

    parts = quadrant_path.split("_")

    if len(parts) == 1:
        return mapping.get(parts[0], parts[0].upper())
    elif len(parts) == 2:
        # e.g., "ne_sw" -> "Southern Northeast"
        outer = mapping.get(parts[0], parts[0].upper())
        inner = mapping.get(parts[1], parts[1].upper())
        # Extract directional prefix from inner quadrant
        if inner.endswith("east"):
            prefix = "Eastern"
        elif inner.endswith("west"):
            prefix = "Western"
        elif inner.startswith("North"):
            prefix = "Northern"
        elif inner.startswith("South"):
            prefix = "Southern"
        else:
            prefix = inner
        return f"{prefix} {outer}"
    else:
        # Deep nesting - use abbreviated form
        return f"Region {quadrant_path.replace('_', '-').upper()}"


def _calculate_bounds(climbs: List[Any]) -> Bounds:
    """Calculate bounding box for a list of climbs."""
    if not climbs:
        return (0, 0, 0, 0)

    lats = []
    lons = []
    for climb in climbs:
        lat, lon = get_climb_coords(climb)
        if lat is not None and lon is not None:
            lats.append(lat)
            lons.append(lon)

    if not lats:
        return (0, 0, 0, 0)

    return (min(lats), min(lons), max(lats), max(lons))


def _quadtree_partition(
    climbs: List[Any],
    bounds: Bounds,
    max_climbs: int = MAX_CLIMBS_PER_PARTITION,
    depth: int = 0,
    quadrant_path: str = ""
) -> List[Tuple[str, str, List[Any], Bounds]]:
    """
    Recursively partition climbs using quadtree subdivision.

    Args:
        climbs: List of climb objects
        bounds: Bounding box (min_lat, min_lon, max_lat, max_lon)
        max_climbs: Maximum climbs per partition
        depth: Current recursion depth
        quadrant_path: Path of quadrants taken (e.g., "ne_sw")

    Returns:
        List of (partition_id, display_name, climbs_list, bounds) tuples
    """
    # Base case: partition is small enough or max depth reached
    if len(climbs) <= max_climbs or depth >= MAX_QUADTREE_DEPTH:
        partition_id = quadrant_path if quadrant_path else "full"
        display_name = _generate_partition_name(quadrant_path)
        return [(partition_id, display_name, climbs, bounds)]

    min_lat, min_lon, max_lat, max_lon = bounds
    mid_lat = (min_lat + max_lat) / 2
    mid_lon = (min_lon + max_lon) / 2

    # Define quadrants: (name, bounds)
    quadrants = {
        "ne": (mid_lat, mid_lon, max_lat, max_lon),  # Northeast
        "nw": (mid_lat, min_lon, max_lat, mid_lon),  # Northwest
        "se": (min_lat, mid_lon, mid_lat, max_lon),  # Southeast
        "sw": (min_lat, min_lon, mid_lat, mid_lon),  # Southwest
    }

    results = []

    for quad_name, quad_bounds in quadrants.items():
        # Filter climbs to this quadrant
        quad_climbs = []
        for climb in climbs:
            lat, lon = get_climb_coords(climb)
            if lat is not None and lon is not None and _in_bounds(lat, lon, quad_bounds):
                quad_climbs.append(climb)

        if quad_climbs:
            new_path = f"{quadrant_path}_{quad_name}" if quadrant_path else quad_name
            results.extend(_quadtree_partition(
                quad_climbs, quad_bounds, max_climbs, depth + 1, new_path
            ))

    return results


def partition_climbs(
    climbs: List[Any],
    predefined_partitions: Optional[Dict[str, Dict]] = None,
    max_climbs: int = MAX_CLIMBS_PER_PARTITION
) -> List[PartitionInfo]:
    """
    Partition climbs into geographic regions, each under the size limit.

    Strategy:
    1. If predefined_partitions provided (from Geofabrik), try those first
    2. Verify each partition stays under max_climbs
    3. If any partition too large, apply Quadtree subdivision to it
    4. If no predefined_partitions, use Quadtree from start

    Args:
        climbs: List of climb objects (ClimbMetrics or dicts)
        predefined_partitions: Optional dict of partition definitions with bounds
            Format: {"partition_id": {"display_name": str, "bounds": (min_lat, min_lon, max_lat, max_lon)}}
        max_climbs: Maximum climbs per partition (default: 900K)

    Returns:
        List of PartitionInfo objects describing each partition
    """
    if not climbs:
        return []

    # Calculate overall bounds
    overall_bounds = _calculate_bounds(climbs)

    # If no predefined partitions, use quadtree
    if not predefined_partitions:
        logger.info(f"No predefined partitions - using Quadtree subdivision for {len(climbs):,} climbs")
        raw_partitions = _quadtree_partition(climbs, overall_bounds, max_climbs)

        return [
            PartitionInfo(
                partition_id=pid,
                display_name=name,
                bounds=bounds,
                climb_count=len(partition_climbs)
            )
            for pid, name, partition_climbs, bounds in raw_partitions
        ]

    # Try predefined partitions first
    logger.info(f"Trying predefined partitions for {len(climbs):,} climbs")
    partitions_result = []

    for partition_id, partition_def in predefined_partitions.items():
        bounds = partition_def["bounds"]
        display_name = partition_def["display_name"]

        # Filter climbs to this partition
        partition_climbs = []
        for climb in climbs:
            lat, lon = get_climb_coords(climb)
            if lat is not None and lon is not None and _in_bounds(lat, lon, bounds):
                partition_climbs.append(climb)

        if not partition_climbs:
            logger.debug(f"Partition {partition_id} has no climbs, skipping")
            continue

        logger.info(f"Partition {partition_id}: {len(partition_climbs):,} climbs")

        # Check if this partition is too large
        if len(partition_climbs) > max_climbs:
            logger.warning(f"Partition {partition_id} exceeds limit ({len(partition_climbs):,} > {max_climbs:,})")
            logger.info(f"Applying Quadtree subdivision to {partition_id}")

            # Subdivide this partition using quadtree
            sub_partitions = _quadtree_partition(partition_climbs, bounds, max_climbs)

            for sub_id, sub_name, sub_climbs, sub_bounds in sub_partitions:
                # Prefix with parent partition ID
                full_id = f"{partition_id}_{sub_id}" if sub_id != "full" else partition_id
                full_name = f"{display_name} - {sub_name}" if sub_name != "Full Region" else display_name

                partitions_result.append(PartitionInfo(
                    partition_id=full_id,
                    display_name=full_name,
                    bounds=sub_bounds,
                    climb_count=len(sub_climbs)
                ))
        else:
            partitions_result.append(PartitionInfo(
                partition_id=partition_id,
                display_name=display_name,
                bounds=bounds,
                climb_count=len(partition_climbs)
            ))

    # Handle climbs not in any predefined partition (boundary cases)
    assigned_climbs = set()
    for partition in partitions_result:
        for climb in climbs:
            lat, lon = get_climb_coords(climb)
            if lat is not None and lon is not None and _in_bounds(lat, lon, partition.bounds):
                assigned_climbs.add(id(climb))

    unassigned = [c for c in climbs if id(c) not in assigned_climbs]
    if unassigned:
        logger.info(f"{len(unassigned)} climbs not in any predefined partition - adding to 'other' partition")
        other_bounds = _calculate_bounds(unassigned)

        # Check if "other" needs subdivision
        if len(unassigned) > max_climbs:
            other_partitions = _quadtree_partition(unassigned, other_bounds, max_climbs)
            for sub_id, sub_name, sub_climbs, sub_bounds in other_partitions:
                partitions_result.append(PartitionInfo(
                    partition_id=f"other_{sub_id}",
                    display_name=f"Other - {sub_name}",
                    bounds=sub_bounds,
                    climb_count=len(sub_climbs)
                ))
        else:
            partitions_result.append(PartitionInfo(
                partition_id="other",
                display_name="Other Regions",
                bounds=other_bounds,
                climb_count=len(unassigned)
            ))

    return partitions_result


def get_climbs_for_partition(
    climbs: List[Any],
    partition: PartitionInfo
) -> List[Any]:
    """
    Filter climbs to only those within a partition's bounds.

    Args:
        climbs: Full list of climbs
        partition: PartitionInfo describing the target partition

    Returns:
        List of climbs within the partition bounds
    """
    result = []
    for climb in climbs:
        lat, lon = get_climb_coords(climb)
        if lat is not None and lon is not None and _in_bounds(lat, lon, partition.bounds):
            result.append(climb)
    return result
