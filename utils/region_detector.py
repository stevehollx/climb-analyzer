#!/usr/bin/env python3
"""
Region Detection Module for Climb Analyzer

Automatically detects if a region is a US state, country, province, or other
geographic entity using geo_lookup.py functions.
"""

import sys
from difflib import get_close_matches
from typing import List, Tuple, Dict, Optional
from climb_analyzer.data.geo_lookup import find_region, is_us_state, get_all_regions, osm_pbf_urls


def get_all_searchable_regions() -> List[Tuple[str, str, str]]:
    """
    Build a list of all searchable regions from osm_pbf_urls.

    Returns:
        List of (region_type, canonical_name, searchable_name) tuples
    """
    all_regions = []

    # Get all regions from osm_pbf_urls
    regions_with_bounds = get_all_regions(include_bounds=True)

    for path_or_tuple in regions_with_bounds:
        if isinstance(path_or_tuple, tuple):
            path = path_or_tuple[0]
        else:
            path = path_or_tuple

        # Extract region name from path
        if '/' in path:
            region_name = path.split('/')[-1]
        else:
            region_name = path

        # Determine region type
        if '/us/' in path.lower() or path.startswith('us/'):
            region_type = "state"
        elif path.count('/') == 0 or path.count('/') == 1:
            # Top level or one level down = continent or country
            region_type = "country" if path.count('/') == 1 else "continent"
        else:
            region_type = "subregion"

        # Create searchable name (lowercase, no hyphens)
        searchable = region_name.lower().replace('-', ' ')

        all_regions.append((region_type, path, searchable))

    return all_regions


def find_fuzzy_matches(region_name: str, interactive: bool = True) -> Optional[Tuple[str, str]]:
    """
    Find fuzzy matches for an unknown region name and optionally prompt user.

    Args:
        region_name: The region name that didn't match exactly
        interactive: If True, prompt user to confirm match. If False, return best match without prompting.

    Returns:
        Tuple of (region_type, canonical_name) if match confirmed/found, None otherwise
    """
    region_lower = region_name.lower().strip()

    # Build list of all possible region names
    all_regions = get_all_searchable_regions()

    # Get list of searchable names
    searchable_names = [r[2] for r in all_regions]

    # Find close matches (cutoff of 0.6 means 60% similarity)
    matches = get_close_matches(region_lower, searchable_names, n=5, cutoff=0.6)

    if not matches:
        return None

    # Get the full region info for matches
    matched_regions = []
    for match in matches:
        for region_type, canonical_name, searchable in all_regions:
            if searchable == match:
                matched_regions.append((region_type, canonical_name, match))
                break

    if not matched_regions:
        return None

    # Deduplicate by display name - prefer country over subregion
    seen_display_names = {}
    deduped_regions = []
    priority_order = {"country": 1, "state": 2, "subregion": 3, "continent": 4}

    for region_type, canonical_name, searchable in matched_regions:
        # Generate display name
        if '/' in canonical_name:
            display = canonical_name.split('/')[-1].replace('-', ' ').title()
        else:
            display = canonical_name.replace('-', ' ').title()

        display_lower = display.lower()

        # Keep this entry if we haven't seen this display name, or if this is higher priority
        if display_lower not in seen_display_names:
            seen_display_names[display_lower] = (region_type, canonical_name, searchable)
            deduped_regions.append((region_type, canonical_name, searchable))
        else:
            existing_type = seen_display_names[display_lower][0]
            existing_priority = priority_order.get(existing_type, 999)
            current_priority = priority_order.get(region_type, 999)

            # Replace if current has higher priority (lower number)
            if current_priority < existing_priority:
                seen_display_names[display_lower] = (region_type, canonical_name, searchable)
                deduped_regions = [r for r in deduped_regions if r[2] != searchable]
                deduped_regions.append((region_type, canonical_name, searchable))

    matched_regions = deduped_regions

    # If not interactive, return best match
    if not interactive:
        return (matched_regions[0][0], matched_regions[0][1])

    # Interactive mode - prompt user
    print(f"\n❌ Region '{region_name}' not found.")
    print(f"\nDid you mean one of these?")
    for i, (region_type, canonical_name, _) in enumerate(matched_regions, 1):
        # Format display name nicely
        if '/' in canonical_name:
            display = canonical_name.split('/')[-1].replace('-', ' ').title()
        else:
            display = canonical_name.replace('-', ' ').title()
        print(f"  {i}. {display} ({region_type})")
    print(f"  0. None of these - let me type it again")

    while True:
        try:
            choice = input("\nEnter number (or press Ctrl+C to cancel): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                # User wants to type again
                while True:
                    new_input = input("\nEnter region name: ").strip()
                    if new_input:
                        # Try detecting with the new input
                        detected_type, detected_canonical = detect_region_type(new_input)
                        if detected_type != "unknown":
                            return (detected_type, detected_canonical)
                        else:
                            # Recursively call fuzzy match again
                            return find_fuzzy_matches(new_input, interactive=True)
                    else:
                        print("Please enter a region name.")
            elif 1 <= choice_num <= len(matched_regions):
                # User selected a match
                selected = matched_regions[choice_num - 1]
                return (selected[0], selected[1])
            else:
                print(f"Please enter a number between 0 and {len(matched_regions)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n❌ Region selection cancelled")
            sys.exit(1)


def detect_region_type(region_name: str) -> Tuple[str, str]:
    """
    Detect if region is a US state, country, or subregion based on geo_lookup.

    Args:
        region_name: Name of region to detect (e.g., "Colorado", "CO", "Switzerland")

    Returns:
        Tuple of (region_type, canonical_name)
        region_type: "state" | "country" | "subregion" | "unknown"
        canonical_name: Path or name from osm_pbf_urls

    Examples:
        >>> detect_region_type("Colorado")
        ("state", "us/colorado")
        >>> detect_region_type("CO")
        ("state", "us/colorado")
        >>> detect_region_type("Switzerland")
        ("country", "europe/switzerland")
    """
    # US state abbreviation mapping
    US_STATE_ABBREVS = {
        'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
        'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
        'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
        'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
        'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
        'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
        'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
        'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
        'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
        'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
        'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
        'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
        'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
        'pr': 'puerto rico', 'vi': 'us virgin islands'
    }

    region_lower = region_name.lower().strip()

    # Check if input is a US state abbreviation
    if region_lower in US_STATE_ABBREVS:
        full_name = US_STATE_ABBREVS[region_lower]
        # Look up the full state name to get canonical path
        region_info = find_region(full_name)
        if region_info:
            return ("state", region_info.get("path", f"us/{full_name.replace(' ', '-')}"))
        return ("state", f"us/{full_name.replace(' ', '-')}")

    # Handle ambiguous region names - Georgia is both a US state and a country
    if region_lower == "georgia":
        print("\n⚠️  Ambiguous region detected: 'Georgia'")
        print("   1. Georgia (US State)")
        print("   2. Georgia (Country in Europe/Asia)")
        print()

        while True:
            try:
                choice = input("Please select (1 or 2): ").strip()
                if choice == "1":
                    return ("state", "us/georgia")
                elif choice == "2":
                    return ("country", "europe/georgia")
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except (KeyboardInterrupt, EOFError):
                # Default to US state if user cancels
                print("\nDefaulting to Georgia (US State)")
                return ("state", "us/georgia")

    # Check if it's a US state first
    if is_us_state(region_name):
        region_info = find_region(region_name)
        if region_info:
            return ("state", region_info.get("path", region_name))
        return ("state", region_name)

    # Try to find the region using geo_lookup
    region_info = find_region(region_name)

    if region_info:
        path = region_info.get("path", region_name)

        # Determine type based on path depth
        if '/us/' in path.lower() or path.startswith('us/'):
            return ("state", path)
        elif path.count('/') == 0:
            # Top level = continent
            return ("continent", path)
        elif path.count('/') == 1:
            # One level = country
            return ("country", path)
        else:
            # Deeper = subregion
            return ("subregion", path)

    # Unknown region - return as-is and let downstream handle it
    return ("unknown", region_name)


def parse_regions(region_string: str, interactive: bool = True) -> List[Dict[str, str]]:
    """
    Parse comma-separated region string and detect types for each.

    Args:
        region_string: Comma-separated regions like "Colorado,Vermont" or "Switzerland,Austria"
        interactive: If True, prompt user for fuzzy matches when region not found

    Returns:
        List of dicts with keys:
            - input_name: Original input string
            - type: Region type (state/country/subregion/unknown)
            - canonical_name: Path or standardized name from geo_lookup
    """
    # Split by comma and strip whitespace
    regions = [r.strip() for r in region_string.split(',') if r.strip()]

    parsed_regions = []
    for region in regions:
        region_type, canonical_name = detect_region_type(region)

        # If unknown and interactive mode, try fuzzy matching
        if region_type == "unknown" and interactive:
            fuzzy_result = find_fuzzy_matches(region, interactive=True)
            if fuzzy_result:
                region_type, canonical_name = fuzzy_result

        parsed_regions.append({
            'input_name': region,
            'type': region_type,
            'canonical_name': canonical_name
        })

    return parsed_regions


def validate_regions(regions: List[Dict[str, str]], allow_mixed_types: bool = False) -> Tuple[bool, str]:
    """
    Validate a list of parsed regions.

    Args:
        regions: List of region dicts from parse_regions()
        allow_mixed_types: Whether to allow mixing states, countries, etc.

    Returns:
        Tuple of (is_valid, error_message)
        is_valid: True if all regions are valid
        error_message: Error description if invalid, empty string if valid
    """
    if not regions:
        return (False, "No regions provided")

    # Check for unknown regions
    unknown_regions = [r for r in regions if r['type'] == 'unknown']
    if unknown_regions:
        unknown_names = [r['input_name'] for r in unknown_regions]
        return (False, f"Unknown region(s): {', '.join(unknown_names)}")

    # Check for mixed types if not allowed
    if not allow_mixed_types:
        region_types = set(r['type'] for r in regions)
        if len(region_types) > 1:
            types_str = ', '.join(sorted(region_types))
            return (False, f"Mixed region types detected: {types_str}")

    return (True, "")


def print_unknown_region_error(unknown_regions: List[Dict[str, str]]) -> None:
    """
    Print helpful error message for unrecognized regions.

    Args:
        unknown_regions: List of region dicts with type="unknown"
    """
    print()

    for region in unknown_regions:
        name = region.get("input_name", region.get("canonical_name", "unknown"))
        print(f"[X] Region not recognized: '{name}'")

        # Try to find fuzzy matches (non-interactive mode)
        try:
            fuzzy_result = find_fuzzy_matches(name, interactive=False)
            if fuzzy_result:
                region_type, canonical = fuzzy_result
                # Format display name
                if '/' in canonical:
                    display = canonical.split('/')[-1].replace('-', ' ').title()
                else:
                    display = canonical.replace('-', ' ').title()
                print(f"    Did you mean: {display} ({region_type})?")
        except Exception:
            pass  # Silently skip if fuzzy matching fails

    print()
    print("[!] The -r flag only accepts known Geofabrik regions:")
    print("    - US States: 'Vermont', 'California', 'North Carolina' (or abbreviations: 'VT', 'CA', 'NC')")
    print("    - Countries: 'France', 'Switzerland', 'Japan'")
    print("    - Subregions: 'Bristol', 'Bayern', 'Bretagne'")
    print()
    print("    For city/address searches, use: ./climb-analyzer -a \"<address>\"")
    print("    Example: ./climb-analyzer -a \"Bryson City, NC\" --distance 25")
    print()
    print("    To see all available regions: ./climb-analyzer --list-regions")
    print("    Full region list: https://download.geofabrik.de/")
    print()


def get_region_summary(regions: List[Dict[str, str]]) -> str:
    """
    Get a human-readable summary of regions.

    Args:
        regions: List of region dicts from parse_regions()

    Returns:
        Summary string
    """
    if not regions:
        return "No regions"

    region_types = {}
    for r in regions:
        region_type = r['type']
        if region_type not in region_types:
            region_types[region_type] = []
        # Extract display name from canonical path
        canonical = r['canonical_name']
        if '/' in canonical:
            display_name = canonical.split('/')[-1].replace('-', ' ').title()
        else:
            display_name = canonical.replace('-', ' ').title()
        region_types[region_type].append(display_name)

    summary_parts = []
    type_labels = {
        'state': 'US state' if len(region_types.get('state', [])) == 1 else 'US states',
        'country': 'country' if len(region_types.get('country', [])) == 1 else 'countries',
        'subregion': 'subregion' if len(region_types.get('subregion', [])) == 1 else 'subregions',
        'continent': 'continent' if len(region_types.get('continent', [])) == 1 else 'continents',
        'unknown': 'unknown region' if len(region_types.get('unknown', [])) == 1 else 'unknown regions'
    }

    for region_type, names in region_types.items():
        label = type_labels.get(region_type, region_type)
        name_str = ', '.join(names)
        summary_parts.append(f"{len(names)} {label}: {name_str}")

    return '; '.join(summary_parts)


# Test/demo code
if __name__ == "__main__":
    # Test region detection
    test_cases = [
        "Colorado",
        "CO",           # State abbreviation
        "Vermont",
        "VT",           # State abbreviation
        "NC",           # State abbreviation
        "north carolina",  # Full name lowercase
        "Switzerland",
        "Bristol",
        "Unknown Region"
    ]

    print("=" * 80)
    print("REGION DETECTION TESTS")
    print("=" * 80)

    for test in test_cases:
        region_type, canonical = detect_region_type(test)
        print(f"Input: {test:20} -> Type: {region_type:10} Canonical: {canonical}")

    print("\n" + "=" * 80)
    print("BATCH REGION PARSING")
    print("=" * 80)

    batch_tests = [
        "Colorado,Vermont",
        "Switzerland,Austria,Italy",
        "Colorado,Switzerland"  # Mixed types
    ]

    for batch in batch_tests:
        print(f"\nInput: {batch}")
        regions = parse_regions(batch, interactive=False)
        for r in regions:
            print(f"  - {r['input_name']:15} -> {r['type']:10} ({r['canonical_name']})")

        is_valid, error = validate_regions(regions, allow_mixed_types=False)
        summary = get_region_summary(regions)
        print(f"  Valid: {is_valid}, Summary: {summary}")
        if not is_valid:
            print(f"  Error: {error}")
