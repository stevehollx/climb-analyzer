#!/usr/bin/env python3
"""
Index XLSX Files Script

This script scans the repository for XLSX files and creates an index.json file
containing region paths and download URLs for all climb data files.

The script handles split files (e.g., file-1.xlsx, file-2.xlsx) and groups them
by region name (the word before "_climbs" in the filename).
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import quote


def extract_region_name(filename: str) -> str:
    """
    Extract the region name from a filename.
    The region name is the word before "_climbs".

    Examples:
        Luxembourg_climbs_all_basic_2025-11-01_v2.0.0_e0000.xlsx -> Luxembourg
        United-States_climbs_all_basic_2025-11-01_v2.0.0_e0000-1.xlsx -> United-States
    """
    # Remove any split file suffix (-1, -2, etc.) before .xlsx
    filename_cleaned = re.sub(r'-\d+\.xlsx$', '.xlsx', filename)

    # Extract the region name (everything before _climbs)
    # Use non-greedy match to handle region names with underscores (e.g., New_York)
    match = re.match(r'^(.+?)_climbs', filename)
    if match:
        return match.group(1)
    return None


def is_split_file(filename: str) -> bool:
    """Check if a file is a split file (ends with -N.xlsx)."""
    return bool(re.search(r'-\d+\.xlsx$', filename))


def get_base_filename(filename: str) -> str:
    """Get the base filename without the split suffix."""
    return re.sub(r'-\d+\.xlsx$', '.xlsx', filename)


def generate_download_url(repo_name: str, file_path: str) -> str:
    """
    Generate a GitHub raw content download URL for a file.

    Args:
        repo_name: GitHub repository name (e.g., "username/repo")
        file_path: Path to the file relative to repo root

    Returns:
        The raw GitHub URL for downloading the file
    """
    # URL encode the path components to handle special characters
    encoded_path = '/'.join(quote(part) for part in file_path.split('/'))
    return f"https://raw.githubusercontent.com/{repo_name}/main/{encoded_path}"


def scan_repository(repo_path: str = ".") -> Dict[str, Dict[str, List[str]]]:
    """
    Scan the repository for XLSX files and organize them by region.

    Returns:
        Dictionary with region paths as keys and file information as values.
        Structure:
        {
            "continent/country": {
                "region_name": "Country-Name",
                "files": ["file1.xlsx", "file2.xlsx"],
                "download_urls": ["url1", "url2"]
            }
        }
    """
    repo_name = os.environ.get('REPO_NAME', 'stevehollx/global-road-and-trail-climbs')
    index_data = {}

    # Walk through all directories
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and the .git directory
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        # Find all XLSX files in this directory
        xlsx_files = [f for f in files if f.endswith('.xlsx')]

        if xlsx_files:
            # Get the relative path from repo root
            rel_path = os.path.relpath(root, repo_path)
            if rel_path == '.':
                continue  # Skip root directory files

            # Group files by region name
            region_files = {}

            for filename in xlsx_files:
                region_name = extract_region_name(filename)
                if region_name:
                    if region_name not in region_files:
                        region_files[region_name] = set()
                    region_files[region_name].add(filename)

            # Process each region found in this directory
            for region_name, file_set in region_files.items():
                # Sort files to ensure consistent ordering
                # This will put base files before split files, and split files in numerical order
                sorted_files = sorted(file_set, key=lambda x: (
                    get_base_filename(x),
                    0 if not is_split_file(x) else int(re.search(r'-(\d+)\.xlsx$', x).group(1))
                ))

                # Generate download URLs
                download_urls = [
                    generate_download_url(repo_name, os.path.join(rel_path, f))
                    for f in sorted_files
                ]

                # Get file sizes in bytes
                file_sizes = [
                    os.path.getsize(os.path.join(root, f))
                    for f in sorted_files
                ]

                # Store in index
                index_data[rel_path] = {
                    "region_name": region_name,
                    "files": sorted_files,
                    "file_sizes": file_sizes,
                    "download_urls": download_urls,
                    "file_count": len(sorted_files),
                    "total_size": sum(file_sizes),
                    "has_split_files": any(is_split_file(f) for f in sorted_files)
                }

    return index_data


def create_summary_stats(index_data: Dict) -> Dict:
    """Create summary statistics for the index."""
    total_regions = len(index_data)
    total_files = sum(region["file_count"] for region in index_data.values())
    total_size_bytes = sum(region.get("total_size", 0) for region in index_data.values())
    regions_with_splits = sum(1 for region in index_data.values() if region["has_split_files"])

    # Group by continent
    by_continent = {}
    for path in index_data.keys():
        continent = path.split('/')[0]
        if continent not in by_continent:
            by_continent[continent] = 0
        by_continent[continent] += 1

    return {
        "total_regions": total_regions,
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 1),
        "regions_with_split_files": regions_with_splits,
        "by_continent": by_continent
    }


def main():
    """Main function to run the indexer."""
    print("Starting XLSX file indexing...")

    # Scan the repository
    index_data = scan_repository()

    # Create the final index structure
    final_index = {
        "version": "1.0.0",
        "generated_at": __import__('datetime').datetime.utcnow().isoformat() + "Z",
        "repository": os.environ.get('REPO_NAME', 'stevehollx/global-road-and-trail-climbs'),
        "summary": create_summary_stats(index_data),
        "regions": index_data
    }

    # Write to index.json in the root directory
    output_path = Path("index.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_index, f, indent=2, ensure_ascii=False)

    print(f"✓ Index created successfully at {output_path}")
    print(f"  - Found {final_index['summary']['total_regions']} regions")
    print(f"  - Total files: {final_index['summary']['total_files']}")

    if final_index['summary']['total_regions'] == 0:
        print("\n⚠️  No XLSX files found in the repository.")
        print("   The index.json file has been created but is empty.")
        print("   It will be populated as XLSX files are added to the repository.")
    else:
        print("\nRegions by continent:")
        for continent, count in final_index['summary']['by_continent'].items():
            print(f"  - {continent}: {count} regions")


if __name__ == "__main__":
    main()
