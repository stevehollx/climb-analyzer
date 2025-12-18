#!/usr/bin/env python3
"""
Utility to track way-level elevation fetch failures.

This module provides utilities to associate elevation fetch results
with specific ways/segments for detailed error reporting.
"""

from typing import List, Tuple, Optional, Dict, Any
from utils.elevation_stats_collector import record_way_failure


def track_way_elevation_failures(
    way_id: str,
    way_name: str,
    coordinates: List[Tuple[float, float]],
    elevations: List[Optional[float]]
):
    """
    Track elevation fetch failures for a specific way.

    Args:
        way_id: Unique identifier for the way
        way_name: Display name of the way
        coordinates: List of (lat, lon) tuples for this way
        elevations: List of elevation values (None where fetch failed)
    """
    if len(coordinates) != len(elevations):
        return  # Mismatch, can't track properly

    total_coords = len(coordinates)
    failed_count = 0

    # Track each failure
    for idx, (coord, elev) in enumerate(zip(coordinates, elevations)):
        if elev is None:
            record_way_failure(
                way_id=way_id,
                way_name=way_name,
                coord_index=idx,
                total_coords=total_coords
            )
            failed_count += 1

    return failed_count


def track_segment_elevation_failures(
    segment: Dict[str, Any],
    elevations: List[Optional[float]]
):
    """
    Track elevation fetch failures for a segment.

    Args:
        segment: Segment dictionary with 'id', 'name', and 'coordinates'
        elevations: List of elevation values (None where fetch failed)

    Returns:
        Number of failed coordinates
    """
    way_id = str(segment.get('id', 'unknown'))
    way_name = segment.get('name', f"Unnamed way {way_id}")
    coordinates = segment.get('coordinates', [])

    return track_way_elevation_failures(way_id, way_name, coordinates, elevations)


def batch_track_elevation_failures(
    segments: List[Dict[str, Any]],
    all_elevations: List[Optional[float]]
):
    """
    Track elevation failures for multiple segments.

    This assumes elevations are ordered same as flattened coordinates from all segments.

    Args:
        segments: List of segment dictionaries
        all_elevations: Flat list of all elevations for all segments

    Returns:
        Total number of failed coordinates across all segments
    """
    total_failed = 0
    elev_idx = 0

    for segment in segments:
        coords = segment.get('coordinates', [])
        num_coords = len(coords)

        # Extract elevations for this segment
        segment_elevs = all_elevations[elev_idx:elev_idx + num_coords]

        # Track failures for this segment
        failed = track_segment_elevation_failures(segment, segment_elevs)
        total_failed += failed

        elev_idx += num_coords

    return total_failed
