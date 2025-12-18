"""
Core data structures for climb segments and metrics.

This module defines the fundamental data classes used to represent climb segments
and their associated metrics.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ClimbSegment:
    """
    Represents a climbing segment with elevation profile.

    Attributes:
        name: Name of the climb/road
        start_lat: Starting latitude
        start_lon: Starting longitude
        end_lat: Ending latitude
        end_lon: Ending longitude
        distance_km: Total distance in kilometers
        elevation_gain_m: Total elevation gain in meters
        avg_gradient: Average gradient as percentage
        max_gradient: Maximum gradient as percentage
        category: Climb category (HC, 1, 2, 3, 4, etc.)
        points: List of (lat, lon, elevation) tuples
    """

    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    distance_km: float
    elevation_gain_m: float
    avg_gradient: float
    max_gradient: float
    category: str
    points: List[Tuple[float, float, float]]  # (lat, lon, elevation)


@dataclass
class ClimbMetrics:
    """
    Comprehensive metrics for a climb.

    This class contains all the calculated metrics and metadata for a climb,
    including elevation statistics, scores, location information, and OSM references.

    Attributes:
        street_name: Name of the street/road
        climb_category: Category of the climb (HC, 1, 2, 3, 4, etc.)
        climb_score: Overall climb difficulty score
        elevation_gain: Total elevation gain in meters
        height: Total height climbed
        prominence: Prominence of the climb
        length_km: Total length in kilometers
        distance_km: Distance between first and last elevation points
        avg_grade: Average grade as percentage
        max_grade: Maximum grade as percentage
        min_elevation: Minimum elevation in meters
        max_elevation: Maximum elevation in meters
        surface: Road surface type
        tracktype: For tracks, the tracktype classification
        tracktype_definition: Human readable definition of tracktype
        way_ids: List of OSM way IDs that make up this road
        osm_links: Links to OSM for each way
        city_state: City and state information
        distance_from_center_km: Distance from search center in kilometers
        mid_lat: Midpoint latitude for location lookup
        mid_lon: Midpoint longitude for location lookup
        start_lat: Start latitude of climb
        start_lon: Start longitude of climb
        nodes: List of nodes for location/distance lookup
        fiets_score: Fiets score (alternative scoring system)
        pdi_score: PDI score (alternative scoring system)
        cycling_access: Cycling access classification (Yes/No/Unknown/Limited)
        connected_climbs: List of connected climb street names
        highway_type: OSM highway type
    """

    street_name: str
    climb_category: str
    climb_score: float
    elevation_gain: float
    height: float
    prominence: float
    length_km: float
    distance_km: float
    avg_grade: float
    max_grade: float
    min_elevation: float
    max_elevation: float
    surface: str
    tracktype: str
    tracktype_definition: str
    way_ids: List[int]
    osm_links: List[str]
    city_state: str = "Unknown"
    distance_from_center_km: float = 0.0
    mid_lat: float = 0.0
    mid_lon: float = 0.0
    start_lat: float = 0.0
    start_lon: float = 0.0
    nodes: List = None  # List of SimpleNode objects
    fiets_score: float = 0.0
    pdi_score: float = 0.0
    cycling_access: str = "Unknown"
    connected_climbs: List[str] = None
    highway_type: str = "unknown"
    elevation_profile: str = None  # Pipe-delimited elevation profile: "dist,ele,grade|..."
