#!/usr/bin/env python3
"""
Data Indexer for Climb Analyzer.

Scans data directories and output files to build an index for the Available Data card.
Updates config.yaml with the discovered files and regions.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set
import yaml


class DataIndexer:
    """Indexes available data files for the Available Data card."""

    # Ordered list of elevation datasets (preferred order for display)
    DATASET_ORDER = ['ned10m', 'srtm30m', 'arctic32m', 'rema32m', 'aw3d30', 'aster30m']

    def __init__(self, base_path: Path = None):
        """
        Initialize data indexer.

        Args:
            base_path: Base path of the climb analyzer (defaults to current directory)
        """
        if base_path is None:
            # Assume script is in utils/, so base is parent
            base_path = Path(__file__).parent.parent

        self.base_path = Path(base_path)
        self.planet_osm_data_dir = self.base_path / 'data' / 'planet_osm_data'
        self.osm_indexes_dir = self.base_path / 'osm_indexes'
        self.elevation_data_dir = self.base_path / 'data' / 'elevation_data'
        self.output_dir = self.base_path / 'output'
        self.config_path = self.base_path / 'config.yaml'

    @staticmethod
    def _should_exclude_file(filename: str) -> bool:
        """
        Check if a file should be excluded from indexing.

        Args:
            filename: Name of the file to check

        Returns:
            True if file should be excluded (e.g., macOS metadata files)
        """
        # Exclude macOS metadata files (._*)
        if filename.startswith('._'):
            return True
        # Exclude hidden files
        if filename.startswith('.') and filename not in ['.gitkeep']:
            return True
        return False

    def scan_planet_osm_data(self) -> List[str]:
        """
        Scan data/planet_osm_data for OSM planet files.

        Returns:
            List of OSM planet filenames
        """
        if not self.planet_osm_data_dir.exists():
            return []

        planet_files = []
        for file in self.planet_osm_data_dir.iterdir():
            if file.is_file() and file.suffix in ['.pbf', '.osm']:
                if not self._should_exclude_file(file.name):
                    planet_files.append(file.name)

        return sorted(planet_files)

    def scan_osm_indexes(self) -> List[str]:
        """
        Scan osm_indexes directory for index files.

        Returns:
            List of index names (without extensions)
        """
        if not self.osm_indexes_dir.exists():
            return []

        indexes = set()
        for file in self.osm_indexes_dir.iterdir():
            if file.is_file() and file.suffix in ['.idx', '.bin', '.db']:
                if not self._should_exclude_file(file.name):
                    # Remove extension to get index name
                    indexes.add(file.stem)

        return sorted(list(indexes))

    def scan_elevation_data(self) -> Dict[str, List[str]]:
        """
        Scan data/elevation_data for elevation files grouped by dataset type.

        Returns:
            Dict mapping dataset type to list of regions
            Example: {'aw3d30': ['europe/monaco', 'north-america/usa/california']}
        """
        if not self.elevation_data_dir.exists():
            return {}

        datasets = {}

        # Scan each dataset directory
        for dataset_dir in self.elevation_data_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            regions = set()

            # Recursively scan for region directories with elevation files
            for root, dirs, files in os.walk(dataset_dir):
                # Check if this directory contains elevation files (excluding metadata files)
                has_elevation_files = any(
                    f.endswith(('.tif', '.hgt', '.vrt', '.nc')) and not self._should_exclude_file(f)
                    for f in files
                )

                if has_elevation_files:
                    # Extract region path relative to dataset directory
                    region_path = Path(root).relative_to(dataset_dir)
                    if str(region_path) != '.':
                        regions.add(str(region_path))

            if regions:
                datasets[dataset_name] = sorted(list(regions))

        # Return datasets in preferred order
        ordered_datasets = {}
        for dataset in self.DATASET_ORDER:
            if dataset in datasets:
                ordered_datasets[dataset] = datasets[dataset]

        # Add any datasets not in the preferred order
        for dataset, regions in sorted(datasets.items()):
            if dataset not in ordered_datasets:
                ordered_datasets[dataset] = regions

        return ordered_datasets

    def scan_analyzed_reports(self) -> List[Dict[str, str]]:
        """
        Scan ./output for analyzed climb report files.

        Returns:
            List of dicts with report info:
            [
                {
                    'filename': 'monaco_climbs_all_2025-11-05_v2.0.0_e0000.xlsx',
                    'region': 'monaco',
                    'canonical_path': 'europe/monaco'
                }
            ]
        """
        if not self.output_dir.exists():
            return []

        reports = []

        # Pattern to match climb report files
        # Format: {region}_climbs_{surface}_{date}_v{version}_e{errors}[-{part}].xlsx
        pattern = re.compile(
            r'(.+)_climbs_(?:all|paved|gravel|dirt)_\d{4}-\d{2}-\d{2}(?:_v[\d.]+)?(?:_e\d+)?(?:-\d+)?\.xlsx'
        )

        seen_regions = set()
        for file in self.output_dir.iterdir():
            if file.is_file() and file.suffix == '.xlsx':
                if not self._should_exclude_file(file.name):
                    match = pattern.match(file.name)
                    if match:
                        region = match.group(1)
                        # Deduplicate - only include each region once
                        if region not in seen_regions:
                            seen_regions.add(region)
                            reports.append({
                                'filename': file.name,
                                'region': region,
                                'canonical_path': self._guess_canonical_path(region)
                            })

        return sorted(reports, key=lambda x: x['region'])

    def _guess_canonical_path(self, region: str) -> str:
        """
        Guess canonical path for a region based on available data.

        Args:
            region: Region name (e.g., 'monaco', 'california')

        Returns:
            Canonical path (e.g., 'europe/monaco', 'north-america/usa/california')
        """
        # Try to find in elevation datasets
        for dataset, regions in self.scan_elevation_data().items():
            for canonical_region in regions:
                if canonical_region.endswith(f'/{region}') or canonical_region == region:
                    return canonical_region

        # If not found, just return the region name
        return region

    def index_all_data(self) -> Dict:
        """
        Index all data directories and return complete index.

        Returns:
            Dict with all indexed data
        """
        return {
            'OSM_PLANET_DATA': self.scan_planet_osm_data(),
            'OSM_INDEXES': self.scan_osm_indexes(),
            'ELEVATION_DATASETS': self.scan_elevation_data(),
            'ANALYZED_REPORTS': self.scan_analyzed_reports()
        }

    def update_config(self):
        """
        Update config.yaml with indexed data.
        Preserves existing config values, only updates data index fields.
        """
        # Load existing config
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # Index all data
        index = self.index_all_data()

        # Update config with indexed data
        config['OSM_PLANET_DATA'] = index['OSM_PLANET_DATA']
        config['OSM_INDEXES'] = index['OSM_INDEXES']
        config['ELEVATION_DATASETS'] = index['ELEVATION_DATASETS']
        config['ANALYZED_REPORTS'] = index['ANALYZED_REPORTS']

        # Also update ELEVATION_DATA as list of dataset types
        config['ELEVATION_DATA'] = list(index['ELEVATION_DATASETS'].keys())

        # Write updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return index


def reindex_data(base_path: Path = None) -> Dict:
    """
    Reindex all data and update config.yaml.

    Args:
        base_path: Base path of climb analyzer (defaults to current directory)

    Returns:
        Dict with indexed data
    """
    indexer = DataIndexer(base_path)
    return indexer.update_config()


if __name__ == '__main__':
    # Run indexer
    index = reindex_data()
    print("Data index updated:")
    print(f"  OSM Planet Files: {len(index['OSM_PLANET_DATA'])}")
    print(f"  OSM Indexes: {len(index['OSM_INDEXES'])}")
    print(f"  Elevation Datasets: {len(index['ELEVATION_DATASETS'])}")
    print(f"  Analyzed Reports: {len(index['ANALYZED_REPORTS'])}")
