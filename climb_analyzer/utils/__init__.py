"""
Utility modules for the climb_analyzer package.

This utilities package contains modules specific to the refactored climb_analyzer
package for:
- Graceful shutdown handling
- Terminal output formatting and logging (Tee)
- ASCII logo display
- Helper functions for coordinate calculations and OSM data processing

Note: This is separate from the top-level /utils/ directory, which contains
utilities used by main application scripts (data management, elevation processing, etc.).
"""

from climb_analyzer.utils.graceful_killer import GracefulKiller
from climb_analyzer.utils.helpers import (
    build_elevation_url,
    calculate_distance_km,
    calculate_optimal_batch_size,
    check_and_cleanup_memory,
    determine_cycling_access,
    extract_coordinates_from_segments,
    extract_node_coordinates,
    get_available_memory_mb,
    get_configured_osm_file_path,
    get_memory_usage_percent,
)
from climb_analyzer.utils.tee import Tee

__all__ = [
    "GracefulKiller",
    "Tee",
    "calculate_distance_km",
    "calculate_optimal_batch_size",
    "determine_cycling_access",
    "extract_node_coordinates",
    "extract_coordinates_from_segments",
    "check_and_cleanup_memory",
    "get_memory_usage_percent",
    "get_available_memory_mb",
    "build_elevation_url",
    "get_configured_osm_file_path",
]
