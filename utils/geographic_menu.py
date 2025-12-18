#!/usr/bin/env python3
"""
Shared geographic selection menus for Climb Analyzer tools.

Provides hierarchical region selection: continent -> country -> subregion(s)
and address-based selection.
"""

from typing import Dict, List, Tuple, Optional, Any
from climb_analyzer.data.geo_definitions import osm_pbf_urls
from climb_analyzer.data.geo_lookup import find_region, get_region_bounds, is_us_state, get_continents, get_countries


def format_region_name(name: str) -> str:
    """
    Format region name for display, preserving acronyms like US.

    Args:
        name: Region name (e.g., "us", "new-york", "north-america", "us-midwest")

    Returns:
        Title-cased name with preserved acronyms (e.g., "US", "New York", "North America", "US Midwest")
    """
    # Handle US acronym specifically
    if name.lower() == "us":
        return "US"

    # Handle regions starting with "us-" (e.g., "us-midwest" -> "US Midwest")
    if name.lower().startswith("us-"):
        rest_of_name = name[3:]  # Remove "us-" prefix
        return "US " + rest_of_name.replace('-', ' ').title()

    # Title case with hyphens replaced by spaces
    return name.replace('-', ' ').title()


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size (e.g., "2.5 GB")
    """
    if size_bytes is None:
        return "Unknown"

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def print_two_column_menu(items: List[Tuple[int, str, str, str]], use_columns: bool = True):
    """
    Print menu items in one or two columns.

    Args:
        items: List of (number, name, size, marker) tuples
        use_columns: If True and items > 20, use two columns
    """
    if not use_columns or len(items) <= 20:
        # Single column
        for num, name, size, marker in items:
            print(f"  {num:2d}. {name:<35} {size:>12}{marker}")
    else:
        # Two columns - sort down left column first
        mid = (len(items) + 1) // 2
        left_items = items[:mid]
        right_items = items[mid:]

        for i in range(mid):
            left = left_items[i]
            left_str = f"  {left[0]:2d}. {left[1]:<30} {left[2]:>10}{left[3]}"

            if i < len(right_items):
                right = right_items[i]
                right_str = f"  {right[0]:2d}. {right[1]:<30} {right[2]:>10}{right[3]}"
                print(f"{left_str:<55} {right_str}")
            else:
                print(left_str)


def count_subregions(region_data: Dict) -> int:
    """
    Recursively count total number of downloadable subregions.

    Args:
        region_data: Region data dictionary

    Returns:
        Total count of regions/subregions
    """
    count = 0

    # Check for direct URL (leaf node)
    if "pbf_url" in region_data or "url" in region_data:
        count = 1

    # Check for subregions
    if "subregions" in region_data:
        for subregion in region_data["subregions"].values():
            if isinstance(subregion, dict):
                count += count_subregions(subregion)

    return count


def select_from_subregions(
    continent: str,
    region_path: List[str],
    region_data: Dict,
    region_display_name: str,
    allow_multiple: bool = True
) -> Tuple[List[Tuple[str, List[str]]], bool]:
    """
    Display subregions and allow user to select one or more.

    Args:
        continent: Continent name
        region_path: Current path in the hierarchy
        region_data: Current region data
        region_display_name: Human-readable name for display
        allow_multiple: Allow multiple selections (comma-separated)

    Returns:
        Tuple of (selected_regions, go_back)
        - selected_regions: List of (continent, path) tuples
        - go_back: True if user wants to go back
    """
    # Get subregions if they exist
    subregions = region_data.get("subregions", {})

    if not subregions:
        # Leaf node - no subregions, just download this region
        if "url" in region_data or "pbf_url" in region_data:
            size_str = format_size(region_data.get("size", 0))
            url = region_data.get("url") or region_data.get("pbf_url")
            print(f"\nRegion: {region_display_name}")
            print(f"Size: {size_str}")

            response = input("\nSelect this region? (y/n, or 0 to go back): ").strip().lower()
            if response == 'y':
                return [(continent, region_path)], False
            elif response == '0':
                return [], True
        return [], False

    # Main loop - reprint menu when user goes back from a subregion
    while True:
        # Has subregions - display menu
        CYAN = '\033[36m'
        RESET = '\033[0m'
        print()
        print(f"{CYAN}┌" + "─" * 78 + f"┐{RESET}")
        header_text = f"{region_display_name} - Select Region(s)"
        print(f"{CYAN}│{RESET} {header_text:<76} {CYAN}│{RESET}")
        print(f"{CYAN}└" + "─" * 78 + f"┘{RESET}")

        subregion_list = sorted(subregions.items())

        # Prepare items for display
        menu_items = []
        for i, (subregion_name, subregion_data) in enumerate(subregion_list, 1):
            size_str = format_size(subregion_data.get("size", 0))
            # Check if has children by looking for subregion key
            has_children = "subregions" in subregion_data and subregion_data["subregions"]
            children_marker = " ▶" if has_children else ""
            # Clean up display name - remove path prefix
            display_name = format_region_name(subregion_name.split('/')[-1])
            menu_items.append((i, display_name, size_str, children_marker))

        # Print in one or two columns based on length
        print_two_column_menu(menu_items, use_columns=len(menu_items) > 20)

        # Calculate total size if selecting all
        total_size = sum(s.get("size", 0) for s in subregions.values())
        if allow_multiple:
            print(f"\n  {'All':>3}. Download All Regions ({format_size(total_size)})")
        print(f"  {'0':>3}. Go Back")

        try:
            if allow_multiple:
                choice = input(f"\nEnter choice (number, comma-separated for multiple, 'all', or '0'): ").strip().lower()
            else:
                choice = input(f"\nEnter choice (number or '0'): ").strip().lower()

            if choice == '0':
                return [], True

            if choice == 'all' and allow_multiple:
                # When user selects "All", download the parent region instead of individual subregions
                # This avoids downloading duplicate data and uses the proper Geofabrik parent file
                # For example: California parent file instead of norcal + socal separately
                return [(continent, region_path)], False

            # Parse comma-separated choices
            choices = [c.strip() for c in choice.split(',') if c.strip()]
            selected_indices = []

            for c in choices:
                if c.isdigit():
                    idx = int(c)
                    if 1 <= idx <= len(subregion_list):
                        selected_indices.append(idx)
                    else:
                        print(f"Invalid choice: {c}. Please enter 1-{len(subregion_list)} or '0'")
                        raise ValueError
                else:
                    print(f"Invalid input: {c}. Please enter numbers only.")
                    raise ValueError

            if not selected_indices:
                print("No valid selections made.")
                continue

            # If multiple selections at this level, collect them all as leaf nodes
            if len(selected_indices) > 1 and allow_multiple:
                all_selections = []
                for idx in selected_indices:
                    subregion_name, subregion_data = subregion_list[idx - 1]
                    new_path = region_path + [subregion_name]
                    all_selections.append((continent, new_path))
                return all_selections, False

            # Single selection - navigate deeper if has children
            idx = selected_indices[0]
            subregion_name, subregion_data = subregion_list[idx - 1]
            new_path = region_path + [subregion_name]

            # Check if this subregion has children
            has_children = "subregions" in subregion_data and subregion_data["subregions"]

            if has_children:
                # Navigate deeper
                display_name = format_region_name(subregion_name.split('/')[-1])
                sub_selections, go_back = select_from_subregions(
                    continent,
                    new_path,
                    subregion_data,
                    f"{region_display_name} > {display_name}",
                    allow_multiple=True
                )
                if go_back:
                    # User wants to go back, loop will reprint this menu
                    continue
                return sub_selections, False
            else:
                # Leaf node
                return [(continent, new_path)], False

        except ValueError:
            continue
        except KeyboardInterrupt:
            return [], True


def select_regions_hierarchical() -> Tuple[str, List[Tuple[str, List[str]]], None]:
    """
    Hierarchical region selection: continent -> country -> subregion(s).

    Returns:
        Tuple of ("region", selected_regions, None)
        where selected_regions is a list of (continent, path) tuples
    """
    while True:
        # Display continents
        CYAN = '\033[36m'
        RESET = '\033[0m'
        print()
        print(f"{CYAN}┌" + "─" * 78 + f"┐{RESET}")
        print(f"{CYAN}│{RESET} {'Select Continent':<76} {CYAN}│{RESET}")
        print(f"{CYAN}└" + "─" * 78 + f"┘{RESET}")

        continents = sorted(osm_pbf_urls.keys())
        for i, continent in enumerate(continents, 1):
            continent_data = osm_pbf_urls[continent]
            region_count = count_subregions(continent_data)
            # Size is at the top level, not in region_url
            size_str = format_size(continent_data.get("size", 0))
            display_name = format_region_name(continent)
            print(f"  {i}. {display_name:<25} ({region_count} regions, {size_str})")

        print(f"  0. Cancel")

        try:
            choice = input(f"\nEnter continent number (1-{len(continents)}, or 0 to cancel): ").strip()

            if choice == '0':
                return "region", [], None

            idx = int(choice)
            if 1 <= idx <= len(continents):
                selected_continent = continents[idx - 1]
                continent_display = format_region_name(selected_continent)

                # Navigate through the selected continent
                continent_data = osm_pbf_urls[selected_continent]

                selections, go_back = select_from_subregions(
                    selected_continent,
                    [],
                    continent_data,
                    continent_display,
                    allow_multiple=True
                )

                if go_back:
                    # User wants to go back to continent selection
                    continue

                if selections:
                    return "region", selections, None
                else:
                    # No selections made, go back to continent menu
                    continue
            else:
                print(f"Invalid choice. Please enter 1-{len(continents)} or 0")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            return "region", [], None


def select_address() -> Tuple[str, str, None]:
    """
    Get address from user for geocoding.

    Returns:
        Tuple of ("address", address_string, None)
    """
    print("\nAddress-based Selection:")
    print("-" * 80)
    print("Enter an address to analyze climbs nearby.")
    print("Example: '123 Main St, Boulder, CO' or 'Yosemite Valley, CA'")
    print()

    address = input("Enter address (or 0 to go back): ").strip()

    if address == '0' or not address:
        return "cancelled", "", None

    return "address", address, None


def select_region_or_address() -> Tuple[str, Any, None]:
    """
    Main entry point: Choose between address or region selection.

    Returns:
        Tuple of (selection_type, selection_data, None)
        - For address: ("address", address_string, None)
        - For region: ("region", [(continent, path), ...], None)
        - For cancel: ("cancelled", None, None)
    """
    while True:
        print("\nClimb Analyzer - Area Selection")
        print("=" * 80)
        print("\nHow would you like to specify your area of interest?")
        print()
        print("  1. Address (geocode a specific location)")
        print("  2. Select Region(s) (continent -> country -> subregion)")
        print("  0. Cancel")
        print()

        try:
            choice = input("Enter choice (1-2, or 0 to cancel): ").strip()

            if choice == '0':
                return "cancelled", None, None

            if choice == '1':
                result_type, result_data, _ = select_address()
                if result_type == "cancelled":
                    continue  # Go back to main menu
                return result_type, result_data, None

            if choice == '2':
                result_type, result_data, _ = select_regions_hierarchical()
                if result_type == "region" and result_data:
                    return result_type, result_data, None
                elif result_type == "region" and not result_data:
                    continue  # Go back to main menu
                else:
                    continue

            print("Invalid choice. Please enter 1, 2, or 0.")

        except KeyboardInterrupt:
            return "cancelled", None, None


# Legacy compatibility functions (for existing code)
def select_countries(country_data_param: Dict = None) -> Tuple[str, List[str], None]:
    """
    Legacy function for backward compatibility.
    Redirects to new region selection system.
    """
    print("\n" + "="*80)
    print("  LEGACY MODE: Redirecting to new region selection")
    print("="*80)

    result_type, result_data, _ = select_regions_hierarchical()

    if result_type == "region" and result_data:
        # Convert region selections to country names (best effort)
        # This is a simplified conversion for backward compatibility
        country_names = []
        for continent, path in result_data:
            if path:
                country_names.append(path[0])  # First element is typically the country

        return "country", list(set(country_names)), None

    return "country", [], None


def select_us_states(state_data_param: Dict = None) -> Tuple[str, List[str], None]:
    """
    Legacy function for backward compatibility.
    """
    print("\n" + "="*80)
    print("  LEGACY MODE: US States selection")
    print("  Please use the new region selection: north-america -> us -> state")
    print("="*80)

    # For now, just redirect to hierarchical selection
    result_type, result_data, _ = select_regions_hierarchical()

    if result_type == "region" and result_data:
        state_names = []
        for continent, path in result_data:
            if continent == "north-america" and len(path) >= 2 and "us" in path[0].lower():
                if len(path) >= 2:
                    state_names.append(path[1])

        return "state", state_names, None

    return "state", [], None
