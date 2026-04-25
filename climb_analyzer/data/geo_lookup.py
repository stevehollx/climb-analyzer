"""
Geographic Lookup Utilities

Provides functions to search the hierarchical osm_pbf_urls structure
for regions by name, get bounds, and list all regions.
"""

from typing import Dict, List, Optional, Tuple, Union

# Import will be available after regeneration
try:
    from climb_analyzer.data.geo_definitions import osm_pbf_urls
except ImportError:
    osm_pbf_urls = {}


def find_region(
    region_name: str,
    data: Optional[Dict] = None,
    parent_path: str = ""
) -> Optional[Dict]:
    """
    Recursively search osm_pbf_urls for a region by name.

    Searches by:
    1. Exact key match (e.g., "europe", "europe/france")
    2. Last segment match (e.g., "france" matches "europe/france")
    3. Normalized name match (case-insensitive, hyphens/underscores/spaces)

    Args:
        region_name: Region name to find (e.g., "france", "bristol", "california")
        data: Optional dict to search (defaults to osm_pbf_urls root)
        parent_path: Internal use for building full path

    Returns:
        Dict with 'bounds', 'pbf_url', 'size', 'path' keys, or None if not found
    """
    # Early return if no region name provided
    if region_name is None:
        return None

    if data is None:
        data = osm_pbf_urls

    # Normalize search term
    search_normalized = _normalize_name(region_name)

    for key, value in data.items():
        if not isinstance(value, dict):
            continue

        # The key itself is the path (e.g., "asia/china", "china/xinjiang")
        # For top-level, key is continent name (e.g., "europe", "asia")
        # For nested levels, key contains full relative path (e.g., "europe/france")
        # We use the key directly as the path since it's already the hierarchical path
        current_path = key

        # Check for match at this level
        key_normalized = _normalize_name(key)
        last_segment = key.split("/")[-1] if "/" in key else key
        last_segment_normalized = _normalize_name(last_segment)

        # Match conditions:
        # 1. Exact key match
        # 2. Last segment match (e.g., "bristol" matches "england/bristol")
        # 3. Normalized match
        if (key == region_name or
            last_segment == region_name or
            key_normalized == search_normalized or
            last_segment_normalized == search_normalized):

            result = {
                "path": current_path,
                "key": key,
            }
            if "bounds" in value:
                result["bounds"] = value["bounds"]
            if "pbf_url" in value:
                result["pbf_url"] = value["pbf_url"]
            if "size" in value:
                result["size"] = value["size"]
            if "subregions" in value:
                result["has_subregions"] = True
            return result

        # Recursively search subregions
        if "subregions" in value:
            found = find_region(region_name, value["subregions"], current_path)
            if found:
                return found

    return None


def get_region_bounds(region_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box for a region.

    Args:
        region_name: Region name (e.g., "france", "bristol", "california")

    Returns:
        Tuple of (lat_min, lon_min, lat_max, lon_max) or None if not found
    """
    region = find_region(region_name)
    if region and "bounds" in region:
        return region["bounds"]
    return None


def get_pbf_url(region_name: str) -> Optional[str]:
    """
    Get OSM PBF download URL for a region.

    Args:
        region_name: Region name (e.g., "france", "bristol")

    Returns:
        URL string or None if not found
    """
    region = find_region(region_name)
    if region and "pbf_url" in region:
        return region["pbf_url"]
    return None


def get_all_regions(
    data: Optional[Dict] = None,
    parent_path: str = "",
    include_bounds: bool = False
) -> List[Union[str, Tuple[str, Tuple]]]:
    """
    Flatten the hierarchical structure into a list of region names/paths.

    Args:
        data: Optional dict to search (defaults to osm_pbf_urls root)
        parent_path: Internal use for building full path
        include_bounds: If True, returns list of (path, bounds) tuples

    Returns:
        List of region paths (or (path, bounds) tuples if include_bounds=True)
    """
    if data is None:
        data = osm_pbf_urls

    regions = []

    for key, value in data.items():
        if not isinstance(value, dict):
            continue

        # Keys may contain path prefixes (e.g., "north-america/us" or "us/louisiana")
        # Extract just the final region name for building the path
        key_name = key.split('/')[-1] if '/' in key else key
        current_path = f"{parent_path}/{key_name}" if parent_path else key_name

        if include_bounds and "bounds" in value:
            regions.append((current_path, value["bounds"]))
        elif not include_bounds:
            regions.append(current_path)

        # Recursively get subregions
        if "subregions" in value:
            regions.extend(
                get_all_regions(value["subregions"], current_path, include_bounds)
            )

    return regions


def get_continents() -> List[str]:
    """Get list of continent names (top-level keys)."""
    return list(osm_pbf_urls.keys())


def get_countries(continent: str) -> List[str]:
    """
    Get list of countries/regions within a continent.

    Args:
        continent: Continent name (e.g., "europe", "asia")

    Returns:
        List of country/region names
    """
    continent_data = osm_pbf_urls.get(continent, {})
    subregions = continent_data.get("subregions", {})
    return list(subregions.keys())


def search_by_bounds(
    lat: float,
    lon: float,
    data: Optional[Dict] = None
) -> List[Dict]:
    """
    Find all regions that contain a given lat/lon point.

    Args:
        lat: Latitude
        lon: Longitude
        data: Optional dict to search

    Returns:
        List of matching region dicts, most specific (smallest) first
    """
    if data is None:
        data = osm_pbf_urls

    matches = []

    def search_recursive(d: Dict, parent_path: str = ""):
        for key, value in d.items():
            if not isinstance(value, dict):
                continue

            current_path = f"{parent_path}/{key}" if parent_path else key

            if "bounds" in value:
                lat_min, lon_min, lat_max, lon_max = value["bounds"]
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    # Calculate area for sorting (smaller = more specific)
                    area = (lat_max - lat_min) * (lon_max - lon_min)
                    matches.append({
                        "path": current_path,
                        "key": key,
                        "bounds": value["bounds"],
                        "area": area,
                        "pbf_url": value.get("pbf_url"),
                        "size": value.get("size"),
                    })

            if "subregions" in value:
                search_recursive(value["subregions"], current_path)

    search_recursive(data)

    # Sort by area (smallest first = most specific)
    matches.sort(key=lambda x: x["area"])

    return matches


def _normalize_name(name: str) -> str:
    """Normalize a region name for comparison."""
    if name is None:
        return ""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def print_available_regions() -> None:
    """
    Print all available regions in a hierarchical tree format.

    Output format:
    NORTH AMERICA
      United States (50 states)
        ├── Alabama
        ├── Alaska
        ...
    """
    def format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes >= 1024 * 1024 * 1024:
            return f"{size_bytes / (1024**3):.1f} GB"
        elif size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024**2):.0f} MB"
        else:
            return f"{size_bytes / 1024:.0f} KB"

    def print_tree(data: Dict, indent: int = 0, prefix: str = "") -> None:
        """Recursively print regions as tree."""
        items = list(data.items())

        for i, (key, value) in enumerate(items):
            if not isinstance(value, dict):
                continue

            is_last = (i == len(items) - 1)

            # Extract display name from key
            if '/' in key:
                display_name = key.split('/')[-1].replace('-', ' ').title()
            else:
                display_name = key.replace('-', ' ').title()

            # Build the connector characters
            if indent == 0:
                # Top level (continents) - no tree chars
                connector = ""
                child_prefix = "  "
            else:
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")

            # Get size if available
            size_str = ""
            if "size" in value:
                size_str = f" ({format_size(value['size'])})"

            # Check for subregions
            subregions = value.get("subregions", {})
            subregion_count = len(subregions) if subregions else 0

            # Print the region
            if indent == 0:
                # Continents in uppercase
                print(f"\n{display_name.upper()}")
            elif subregion_count > 0:
                # Show subregion count for countries with subregions
                print(f"{prefix}{connector}{display_name} ({subregion_count} subregions){size_str}")
            else:
                print(f"{prefix}{connector}{display_name}{size_str}")

            # Recursively print subregions
            if subregions:
                print_tree(subregions, indent + 1, child_prefix)

    print("=" * 70)
    print("AVAILABLE REGIONS FOR ANALYSIS")
    print("=" * 70)
    print("\nUse these region names with: ./climb-analyzer -r \"<region>\"")
    print("Examples:")
    print("  ./climb-analyzer -r Vermont")
    print("  ./climb-analyzer -r Switzerland")
    print("  ./climb-analyzer -r \"North Carolina\"")
    print("  ./climb-analyzer -r \"Vermont,New Hampshire\"  (batch mode)")

    # Print the tree
    print_tree(osm_pbf_urls)

    print("\n" + "-" * 70)
    print("Full region list: https://download.geofabrik.de/")
    print("=" * 70)


# Convenience function for backward compatibility with state_data
def is_us_state(region_name: str) -> bool:
    """
    Check if a region name is a US state.

    Uses explicit US path lookup to avoid ambiguity with countries
    that share state names (e.g., Georgia the country vs US state).

    Args:
        region_name: Region name to check

    Returns:
        True if the region is found under north-america/us
    """
    if region_name is None:
        return False

    # Look up with explicit US path first to avoid ambiguity
    us_region = find_region(f"us/{region_name}")
    if us_region and "path" in us_region:
        path = us_region["path"]
        pbf_url = us_region.get("pbf_url", "")
        if path.startswith("us/") or "north-america/us" in pbf_url:
            return True

    # Fallback: check generic lookup but verify it's under us/
    region = find_region(region_name)
    if region and "path" in region:
        path = region["path"]
        pbf_url = region.get("pbf_url", "")
        return path.startswith("us/") or "north-america/us" in pbf_url
    return False


if __name__ == "__main__":
    # Test the module
    print("Testing geo_lookup module...")

    # Test find_region
    france = find_region("france")
    print(f"France: {france}")

    bristol = find_region("bristol")
    print(f"Bristol: {bristol}")

    california = find_region("california")
    print(f"California: {california}")

    # Test get_region_bounds
    bounds = get_region_bounds("switzerland")
    print(f"Switzerland bounds: {bounds}")

    # Test get_all_regions
    regions = get_all_regions()
    print(f"Total regions: {len(regions)}")

    # Test search_by_bounds
    paris_matches = search_by_bounds(48.8566, 2.3522)
    print(f"Regions containing Paris: {[m['path'] for m in paris_matches]}")
