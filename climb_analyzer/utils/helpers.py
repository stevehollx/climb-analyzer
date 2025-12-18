"""
Utility helper functions for climb analysis.

This module contains standalone utility functions used throughout the climb analyzer.
"""

import math
from pathlib import Path
from typing import Tuple, Optional


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(
        lat1_rad
    ) * math.cos(lat2_rad) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def determine_cycling_access(tags: dict, highway_type: str = None) -> str:
    """
    Determine cycling access based on OSM tags and highway type.

    Uses reasonable assumptions about cycling legality on public roads.

    Args:
        tags: Dictionary of OSM tags
        highway_type: Optional highway type override

    Returns:
        One of: 'Yes', 'No', 'Unknown', or 'Limited'
    """
    if not tags:
        return "Unknown"

    highway = highway_type or tags.get("highway", "").strip().lower()
    bicycle = tags.get("bicycle", "").strip().lower()
    access = tags.get("access", "").strip().lower()

    # Check explicit bicycle restrictions first
    if bicycle in ["no", "private"]:
        return "No"
    elif bicycle in ["yes", "designated", "permissive"]:
        return "Yes"
    elif bicycle in ["dismount"]:
        return "Limited"

    # Check general access restrictions
    if access in ["no", "private"]:
        return "No"
    elif access in ["customers", "delivery", "permit"]:
        return "Limited"

    # Roads that are clearly cycling-friendly
    cycling_friendly_highways = {
        "cycleway": "Yes",
        "path": "Yes",  # Multi-use paths
        "bridleway": "Yes",  # Horse paths typically allow cycling
        "track": "Yes",  # Farm tracks, forest roads
        "residential": "Yes",  # Local streets
        "unclassified": "Yes",  # Minor public roads
        "tertiary": "Yes",  # Local connecting roads
        "secondary": "Yes",  # Regional roads
        "primary": "Yes",  # Major roads (legal but not always pleasant)
        "service": "Yes",  # Driveways, parking areas
        "living_street": "Yes",  # Shared space streets
        "road": "Yes",  # Generic road type
        "minor": "Yes",  # Minor roads
        "trunk": "Limited",  # Major roads - legal but often not ideal
    }

    # Roads that typically prohibit cycling
    prohibited_highways = {
        "motorway": "No",
        "motorway_link": "No",
        "steps": "No",
        "corridor": "No",  # Indoor corridors
        "elevator": "No",  # Elevators
    }

    # Special handling for footways
    if highway == "footway":
        # Footways allow cycling if explicitly marked, otherwise limited/unknown
        if bicycle in ["yes", "designated", "permissive"]:
            return "Yes"
        else:
            return "Limited"  # Pedestrian priority, cycling may be restricted

    # Apply highway-based defaults
    if highway in cycling_friendly_highways:
        return cycling_friendly_highways[highway]
    elif highway in prohibited_highways:
        return prohibited_highways[highway]

    # For unknown highway types, make reasonable assumptions
    # Most public roads allow cycling unless explicitly restricted
    if highway and highway not in ["", "unknown", None]:
        # If it's some kind of road/way we don't recognize, assume cycling is allowed
        # This covers regional highway types and new OSM classifications
        return "Yes"

    return "Unknown"


def extract_node_coordinates(node) -> Optional[Tuple[float, float]]:
    """
    Extract coordinates from a node object.

    Args:
        node: Node object with lat/lon attributes

    Returns:
        Tuple of (lat, lon) or None if invalid
    """
    try:
        if hasattr(node, "lat") and hasattr(node, "lon"):
            lat = float(node.lat)
            lon = float(node.lon)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def extract_coordinates_from_segments(segments: list) -> Tuple[list, dict]:
    """
    Extract unique coordinates from segments.

    Args:
        segments: List of segment dictionaries containing nodes

    Returns:
        Tuple of (coordinates list, coord_to_node_ids mapping)
    """
    from collections import defaultdict

    coordinates = []
    coord_to_node_ids = defaultdict(list)

    for segment in segments:
        if "nodes" in segment:
            for node in segment["nodes"]:
                if (
                    hasattr(node, "lat")
                    and hasattr(node, "lon")
                    and hasattr(node, "id")
                ):
                    coord = (round(float(node.lat), 6), round(float(node.lon), 6))

                    # Only add if not already seen
                    if coord not in coord_to_node_ids:
                        coordinates.append(coord)

                    coord_to_node_ids[coord].append(node.id)

    return coordinates, coord_to_node_ids


def get_memory_usage_percent() -> float:
    """
    Get current memory usage as a percentage.

    Returns:
        Memory usage percentage (0-100), or 0 if unable to determine
    """
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0


def get_available_memory_mb() -> float:
    """
    Get available memory in megabytes.

    Returns:
        Available memory in MB, or 0 if unable to determine
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return 0.0


def calculate_optimal_batch_size(
    base_batch_size: int = 50000,
    min_batch_size: int = 5000,
    max_batch_size: int = 100000,
    target_memory_percent: float = 70.0
) -> int:
    """
    Calculate optimal batch size based on current memory availability.

    Dynamically adjusts batch size to prevent memory exhaustion while
    maximizing processing efficiency.

    Args:
        base_batch_size: Default batch size when memory is plentiful
        min_batch_size: Minimum batch size (prevents excessive I/O)
        max_batch_size: Maximum batch size (prevents single-batch OOM)
        target_memory_percent: Target maximum memory usage percentage

    Returns:
        Optimal batch size for current memory conditions
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        current_percent = mem.percent

        # If we're above target, reduce batch size
        if current_percent > target_memory_percent:
            # Reduce exponentially based on how far over we are
            reduction_factor = (100 - current_percent) / (100 - target_memory_percent)
            reduction_factor = max(0.1, min(1.0, reduction_factor))
            adjusted = int(base_batch_size * reduction_factor)
            return max(min_batch_size, min(adjusted, max_batch_size))

        # If we have plenty of memory, use base size
        return base_batch_size

    except ImportError:
        # Fallback if psutil not available - use conservative default
        return min_batch_size


def check_and_cleanup_memory(threshold_percent: int = 80, force_cleanup: bool = False) -> bool:
    """
    Check memory usage and cleanup if needed.

    Args:
        threshold_percent: Memory usage percentage threshold to trigger cleanup
        force_cleanup: Force garbage collection regardless of memory usage

    Returns:
        True if cleanup was performed, False otherwise
    """
    import gc

    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > threshold_percent or force_cleanup:
            gc.collect()
            return True
        return False
    except ImportError:
        # Fallback if psutil not available
        if force_cleanup:
            gc.collect()
        return False


def build_elevation_url(dataset_name: str) -> Optional[str]:
    """
    Build full elevation API URL from base URL and dataset name.

    Args:
        dataset_name: Dataset name like 'srtm30m', 'ned10m', 'aster30m'

    Returns:
        Full URL like 'http://localhost:5000/v1/srtm30m' or None if base URL not configured
    """
    try:
        from utils.config_loader import TOPO_API_BASE_URL
    except ImportError:
        return None

    if not TOPO_API_BASE_URL:
        return None

    # Remove trailing slash if present
    base = TOPO_API_BASE_URL.rstrip("/")

    # Add dataset name
    return f"{base}/{dataset_name}"


def get_configured_osm_file_path() -> str:
    """
    Get the OSM file path based on PLANET_FILE_PATH config, fallback to first available.

    Returns:
        Path to OSM spatial index base (without extension)

    Raises:
        FileNotFoundError: If no spatial index files found
    """
    try:
        from utils.config_loader import PLANET_FILE_PATH
    except ImportError:
        PLANET_FILE_PATH = None

    if PLANET_FILE_PATH and Path(PLANET_FILE_PATH).exists():
        # Use configured planet file path, but convert to spatial index base path
        planet_path = Path(PLANET_FILE_PATH)
        return str(planet_path.with_suffix(""))
    else:
        # Fallback to first available index file (original behavior)
        planet_dir = Path("data/planet_osm_data")
        idx_files = list(planet_dir.glob("*.idx"))
        if idx_files:
            return str(idx_files[0].with_suffix(""))
        else:
            raise FileNotFoundError(
                "No spatial index files found for local deployment."
            )
