"""
Climb Analyzer - Modular package for analyzing road and trail climbs from OpenStreetMap.

This package provides tools for analyzing elevation profiles, calculating climb metrics,
and identifying challenging climbs from OpenStreetMap data.
"""

__version__ = "1.0.0"

# Core data structures
from climb_analyzer.core.segment import ClimbMetrics, ClimbSegment
from climb_analyzer.core.merger import BoundaryMerger, _merge_street_batch_worker_top_level

# Configuration
from climb_analyzer.config import (
    MERGE_PARALLEL_ENABLED,
    MERGE_MAX_WORKERS,
    MERGE_BATCH_SIZE
)

# Data processing
from climb_analyzer.data.elevation import FastElevationFetcher
from climb_analyzer.data.geocoding import ReverseGeocoder
from climb_analyzer.data.spatial_index import LocationIndex, SpatialIndexManager

# Processing utilities
from climb_analyzer.processing.checkpoint import (
    CheckpointConfig,
    ChunkPersistenceManager,
    SmartCheckpointer,
    configure_checkpoints,
)

# Utilities
from climb_analyzer.utils.graceful_killer import GracefulKiller
from climb_analyzer.utils.helpers import (
    build_elevation_url,
    calculate_distance_km,
    check_and_cleanup_memory,
    determine_cycling_access,
    extract_coordinates_from_segments,
    extract_node_coordinates,
    get_configured_osm_file_path,
)
from climb_analyzer.utils.tee import Tee

__all__ = [
    # Core
    "ClimbSegment",
    "ClimbMetrics",
    "BoundaryMerger",
    "_merge_street_batch_worker_top_level",
    # Data
    "SpatialIndexManager",
    "LocationIndex",
    "ReverseGeocoder",
    "FastElevationFetcher",
    # Processing
    "CheckpointConfig",
    "SmartCheckpointer",
    "ChunkPersistenceManager",
    "configure_checkpoints",
    # Utils
    "GracefulKiller",
    "Tee",
    "calculate_distance_km",
    "determine_cycling_access",
    "extract_node_coordinates",
    "extract_coordinates_from_segments",
    "check_and_cleanup_memory",
    "build_elevation_url",
    "get_configured_osm_file_path",
]
