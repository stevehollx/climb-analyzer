#!/usr/bin/env python3
"""
Region Mapper - Maps coordinates to OSM Geofabrik regions

Given a bounding box (from an address search or other source), finds the
smallest Geofabrik subregion(s) that cover the area.
"""

from typing import List, Tuple, Optional
from pathlib import Path

from climb_analyzer.data.geo_definitions import osm_pbf_urls
from climb_analyzer.data.geo_lookup import find_region, get_region_bounds, is_us_state


def point_in_bbox(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    """Check if a point is within a bounding box."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def bbox_intersects(bbox1: Tuple[float, float, float, float],
                    bbox2: Tuple[float, float, float, float]) -> bool:
    """Check if two bounding boxes intersect."""
    min_lat1, min_lon1, max_lat1, max_lon1 = bbox1
    min_lat2, min_lon2, max_lat2, max_lon2 = bbox2

    # Check if boxes don't overlap (then return False)
    if max_lat1 < min_lat2 or min_lat1 > max_lat2:
        return False
    if max_lon1 < min_lon2 or min_lon1 > max_lon2:
        return False

    return True


def bbox_contains(outer: Tuple[float, float, float, float],
                  inner: Tuple[float, float, float, float]) -> bool:
    """Check if outer bbox completely contains inner bbox."""
    min_lat_outer, min_lon_outer, max_lat_outer, max_lon_outer = outer
    min_lat_inner, min_lon_inner, max_lat_inner, max_lon_inner = inner

    return (min_lat_outer <= min_lat_inner and
            max_lat_outer >= max_lat_inner and
            min_lon_outer <= min_lon_inner and
            max_lon_outer >= max_lon_inner)


def get_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate approximate area of bbox in square degrees."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return (max_lat - min_lat) * (max_lon - min_lon)


def is_dateline_crossing_region(bbox: Tuple[float, float, float, float]) -> bool:
    """
    Check if a region's bounds cross the International Date Line.

    These regions have longitude spans > 350 degrees (e.g., us-pacific with -180 to +180).
    Standard bbox intersection/containment checks don't work correctly for these.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    lon_span = max_lon - min_lon
    return lon_span > 350  # Nearly full globe longitude span


def is_state_level_region(region_path: str) -> bool:
    """Check if this is a US state-level region (under us/)."""
    # US states are under paths like "north-america/us/georgia"
    return "/us/" in region_path and region_path.count("/") >= 2


def is_macro_region(region_path: str) -> bool:
    """Check if this is a US macro region like us-south, us-west, etc."""
    # Macro regions are direct children of north-america, like "north-america/us-south"
    macro_prefixes = ["us-south", "us-west", "us-midwest", "us-northeast", "us-pacific"]
    region_name = region_path.split("/")[-1]
    return region_name in macro_prefixes


def compute_bbox_union(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Compute the bounding box that contains all given bboxes."""
    if not bboxes:
        return (0, 0, 0, 0)

    min_lats = [b[0] for b in bboxes]
    min_lons = [b[1] for b in bboxes]
    max_lats = [b[2] for b in bboxes]
    max_lons = [b[3] for b in bboxes]

    return (min(min_lats), min(min_lons), max(max_lats), max(max_lons))


def regions_cover_bbox(
    region_bboxes: List[Tuple[float, float, float, float]],
    search_bbox: Tuple[float, float, float, float]
) -> bool:
    """
    Check if a set of region bboxes collectively cover the search bbox.

    This is a simplified check - it verifies that the union of regions
    fully contains the search bbox. A more precise check would verify
    actual coverage without gaps, but for adjacent state regions this works.
    """
    if not region_bboxes:
        return False

    union_bbox = compute_bbox_union(region_bboxes)
    return bbox_contains(union_bbox, search_bbox)


def find_regions_for_bbox(
    search_bbox: Tuple[float, float, float, float],
    prefer_smallest: bool = True,
    center_point: Optional[Tuple[float, float]] = None
) -> List[Tuple[str, str, int]]:
    """
    Find Geofabrik regions that cover the search bounding box.

    Args:
        search_bbox: (min_lat, min_lon, max_lat, max_lon) to search within
        prefer_smallest: If True, prefer smallest regions that cover the area
        center_point: Optional (lat, lon) center point - ensures region containing
                     center is included even if search extends beyond its borders

    Returns:
        List of (region_path, pbf_url, size_bytes) tuples, sorted by size

    Example:
        >>> bbox = (33.94, -84.40, 33.97, -84.36)  # Sandy Springs, GA
        >>> regions = find_regions_for_bbox(bbox)
        >>> # Returns [('north-america/us/georgia', 'https://...', size)]
    """
    matching_regions = []

    # Helper function to recursively search through the hierarchical structure
    def search_subregions(continent: str, data: dict, parent_path: str = ""):
        """Recursively search for regions that cover the bbox."""

        # Check if this level has a pbf_url (it's a downloadable region)
        if 'pbf_url' in data:
            region_path = f"{continent}" if not parent_path else f"{continent}/{parent_path}"

            # Try to get bounds for this region from the data itself (preferred) or geo_lookup
            region_bbox = data.get('bounds')

            if not region_bbox:
                # Fall back to geo_lookup
                region_name = parent_path.split('/')[-1] if parent_path else continent
                region_bbox = get_region_bounds(region_name)

            # If we have bounds, check if this region covers our search area
            if region_bbox:
                # Skip date-line crossing regions (like us-pacific with -180 to +180 longitude)
                # These have broken bbox checks that match everything
                if is_dateline_crossing_region(region_bbox):
                    # Skip this region - it crosses the date line and would falsely match
                    pass
                elif bbox_contains(region_bbox, search_bbox):
                    # This region fully contains our search area
                    # Priority: smaller regions are better (more specific)
                    matching_regions.append((
                        region_path,
                        data['pbf_url'],
                        data.get('size', 0),
                        get_bbox_area(region_bbox),  # Store area for sorting
                        True,  # Fully contains
                        region_bbox  # Store bbox for later union check
                    ))
                elif bbox_intersects(region_bbox, search_bbox):
                    # This region partially overlaps - might need it
                    # Lower priority than fully-containing regions
                    matching_regions.append((
                        region_path,
                        data['pbf_url'],
                        data.get('size', 0),
                        get_bbox_area(region_bbox),
                        False,  # Partial overlap
                        region_bbox  # Store bbox for later union check
                    ))

        # Recurse into subregions
        if 'subregions' in data:
            for subregion_key, subregion_data in data['subregions'].items():
                # Extract the subregion name from the key (it might be "continent/region")
                if '/' in subregion_key:
                    subregion_name = subregion_key.split('/')[-1]
                else:
                    subregion_name = subregion_key

                new_path = f"{parent_path}/{subregion_name}" if parent_path else subregion_name
                search_subregions(continent, subregion_data, new_path)

    # Search through all continents
    for continent, continent_data in osm_pbf_urls.items():
        if isinstance(continent_data, dict):
            search_subregions(continent, continent_data)

    # Sort results: prioritize fully-containing regions, then by area (smallest first)
    if prefer_smallest and matching_regions:
        # Sort by: (1) fully contains (True first), (2) area (smallest first)
        matching_regions.sort(key=lambda x: (not x[4], x[3]))

    # Filter out redundant parent regions when child regions exist
    # Example: If we have both "north-america" and "north-america/us/georgia",
    # we should only keep the more specific "north-america/us/georgia"
    #
    # Strategy:
    # 1. For each region, check if a more specific child exists
    # 2. Only keep the region if no child exists OR if it's a sibling (not parent-child)

    filtered_regions = []
    for path, url, size, area, contains, region_bbox in matching_regions:
        # Check if this region is a parent of any other region in the list
        is_parent_of_another = False
        for other_path, _, _, _, _, _ in matching_regions:
            if path != other_path and other_path.startswith(path + '/'):
                # This region is a parent of another region in the results
                # Example: "north-america" is parent of "north-america/us/georgia"
                is_parent_of_another = True
                break

        # Only include if it's NOT a redundant parent
        if not is_parent_of_another:
            filtered_regions.append((path, url, size, area, contains, region_bbox))

    # Additional filtering: if we have regions that fully contain the search area,
    # filter out regions that only partially overlap UNLESS they contain the center point
    has_full_coverage = any(contains for _, _, _, _, contains, _ in filtered_regions)

    if has_full_coverage and center_point:
        # For address searches with a center point:
        # Keep regions that either (1) fully contain OR (2) contain the center point
        center_lat, center_lon = center_point

        def region_contains_center(path, url, size, area, contains, region_bbox):
            # If it fully contains the search bbox, keep it
            if contains:
                return True

            # Check if it contains the center point by looking up bounds
            region_name = path.split('/')[-1]
            bounds = get_region_bounds(region_name)

            if bounds:
                lat_min, lon_min, lat_max, lon_max = bounds
                return (lat_min <= center_lat <= lat_max and
                        lon_min <= center_lon <= lon_max)

            return False

        filtered_regions = [r for r in filtered_regions if region_contains_center(*r)]

    elif has_full_coverage:
        # No center point provided - use original logic
        # Keep only regions that fully contain the search area
        filtered_regions = [r for r in filtered_regions if r[4]]

    # NEW: Prefer state-level regions over macro regions when states can cover the area
    # This avoids downloading huge files like us-south when NC+TN+GA would suffice
    state_regions = [(p, u, s, a, c, b) for p, u, s, a, c, b in filtered_regions
                     if is_state_level_region(p)]
    macro_regions = [(p, u, s, a, c, b) for p, u, s, a, c, b in filtered_regions
                     if is_macro_region(p)]
    other_regions = [(p, u, s, a, c, b) for p, u, s, a, c, b in filtered_regions
                     if not is_state_level_region(p) and not is_macro_region(p)]

    # If we have both state-level and macro regions, check if states can cover the bbox
    if state_regions and macro_regions:
        state_bboxes = [b for _, _, _, _, _, b in state_regions]
        if regions_cover_bbox(state_bboxes, search_bbox):
            # States can cover the area - exclude macro regions
            filtered_regions = state_regions + other_regions
        # else: keep macro regions as they may be needed to fill gaps

    # Return without the area, contains, and bbox fields
    return [(path, url, size) for path, url, size, area, contains, region_bbox in filtered_regions]


def find_best_region_for_address(
    lat: float,
    lon: float,
    radius_km: float = 15.0
) -> Optional[Tuple[str, str, int]]:
    """
    Find the best (smallest) Geofabrik region for an address search.

    Args:
        lat: Latitude of the address
        lon: Longitude of the address
        radius_km: Search radius in kilometers

    Returns:
        Tuple of (region_path, pbf_url, size_bytes) or None if not found

    Example:
        >>> region = find_best_region_for_address(33.9551, -84.3819, 1.6)
        >>> # Returns ('north-america/us/georgia', 'https://...', size)
    """
    import math

    # Calculate bounding box from center point and radius
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(latitude)
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

    search_bbox = (
        lat - lat_delta,  # min_lat
        lon - lon_delta,  # min_lon
        lat + lat_delta,  # max_lat
        lon + lon_delta,  # max_lon
    )

    # Find regions that cover this bbox, passing center point to ensure
    # we include the region containing the address even if search extends beyond
    regions = find_regions_for_bbox(search_bbox, prefer_smallest=True, center_point=(lat, lon))

    if not regions:
        return None

    # Return the smallest region (first in sorted list)
    return regions[0]


def find_regions_for_multi_point(
    locations: List[Tuple[float, float]]
) -> List[Tuple[str, str, int]]:
    """
    Find regions needed to cover multiple locations.

    May return multiple regions if locations span multiple states/countries.

    Args:
        locations: List of (lat, lon) tuples

    Returns:
        List of (region_path, pbf_url, size_bytes) tuples
    """
    # Calculate overall bounding box
    if not locations:
        return []

    lats = [loc[0] for loc in locations]
    lons = [loc[1] for loc in locations]

    search_bbox = (
        min(lats),  # min_lat
        min(lons),  # min_lon
        max(lats),  # max_lat
        max(lons),  # max_lon
    )

    # Find regions that cover this bbox
    regions = find_regions_for_bbox(search_bbox, prefer_smallest=True)

    # If we get multiple partial overlaps, we might need to merge them
    # For now, return all matching regions sorted by size
    return regions


if __name__ == "__main__":
    # Test with the example from the issue
    print("Testing region mapper with Sandy Springs, GA example:")
    lat, lon = 33.9551, -84.3819
    radius_km = 1.6

    result = find_best_region_for_address(lat, lon, radius_km)

    if result:
        region_path, pbf_url, size_bytes = result
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_bytes / (1024 * 1024 * 1024)

        print(f"\nBest region: {region_path}")
        print(f"URL: {pbf_url}")
        print(f"Size: {size_gb:.2f} GB ({size_mb:.0f} MB)")
    else:
        print("\nNo region found!")

    # Test with a bbox
    print("\n\nTesting with explicit bbox:")
    bbox = (33.940603841441444, -84.39940108826382, 33.96960095855856, -84.36444271173617)
    regions = find_regions_for_bbox(bbox)

    print(f"Found {len(regions)} matching regions:")
    for i, (path, url, size) in enumerate(regions[:5], 1):
        print(f"{i}. {path} ({size / (1024**3):.2f} GB)")
