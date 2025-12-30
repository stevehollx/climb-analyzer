"""
Elevation data fetching with local GeoTIFF support.

This module provides parallel elevation fetching with automatic fallback between
different elevation datasets and adaptive batch sizing.
"""

import concurrent.futures
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# Import configuration - these will need to be available when the module is used
try:
    from utils.config_loader import (
        ELEVATION_BACKOFF_FACTOR,
        ELEVATION_BATCH_SIZE,
        ELEVATION_MAX_CONCURRENT,
        ELEVATION_MAX_RETRIES,
        ELEVATION_REQUEST_TIMEOUT_SEC,
    )
except ImportError:
    # Fallback defaults
    ELEVATION_BATCH_SIZE = 500
    ELEVATION_MAX_CONCURRENT = 4
    ELEVATION_REQUEST_TIMEOUT_SEC = 30
    ELEVATION_MAX_RETRIES = 3
    ELEVATION_BACKOFF_FACTOR = 2


def get_dataset_priority_for_region(region_name: str = None, lat: float = None) -> List[str]:
    """
    Determine dataset priority order based on region and latitude.

    Priority rules:
    1. REMA (32m) - Antarctica only (lat < -60°)
    2. ArcticDEM (32m) - Arctic only (lat > 60°N)
    3. NED 10m - USA only
    4. SRTM 30m - Global (-60° to 60°)
    5. AW3D30 - Global fallback
    6. ASTER 30m - Global last resort

    Args:
        region_name: Region name (e.g., "Alaska", "Antarctica")
        lat: Center latitude for the region

    Returns:
        List of dataset names in priority order
    """
    datasets = []

    # Determine latitude if not provided
    if lat is None and region_name:
        # Try to get latitude from region name
        region_lower = region_name.lower()
        if 'antarctica' in region_lower or 'antarctic' in region_lower:
            lat = -75.0  # Center of Antarctica
        elif 'arctic' in region_lower or 'greenland' in region_lower or 'svalbard' in region_lower:
            lat = 75.0  # Arctic region
        # Default to 40°N if unknown (mid-latitude)
        elif lat is None:
            lat = 40.0

    # Antarctica: REMA is best
    if lat is not None and lat < -60:
        datasets.append('rema32m')
        datasets.append('aw3d30')
        datasets.append('aster30m')
        return datasets

    # Arctic: ArcticDEM is best
    if lat is not None and lat > 60:
        datasets.append('arctic32m')
        datasets.append('aw3d30')
        datasets.append('aster30m')
        return datasets

    # USA: NED 10m is best where available
    if region_name:
        region_lower = region_name.lower()
        usa_regions = [
            'alaska', 'hawaii', 'alabama', 'arizona', 'arkansas', 'california',
            'colorado', 'connecticut', 'delaware', 'florida', 'georgia', 'idaho',
            'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
            'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
            'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'new hampshire', 'new jersey', 'new mexico', 'new york',
            'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
            'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
            'west virginia', 'wisconsin', 'wyoming'
        ]

        if any(state in region_lower for state in usa_regions):
            datasets.append('ned10m')

    # Global standard priority: SRTM → AW3D30 → ASTER
    if lat is not None and -60 <= lat <= 60:
        datasets.append('srtm30m')

    datasets.append('aw3d30')
    datasets.append('aster30m')

    return datasets


class ElevationFetchLog:
    """Track detailed information about elevation fetching attempts."""

    def __init__(self):
        self.failed_fetches = []  # List of failed coordinate fetches
        self.fallback_successes = []  # List of coordinates that succeeded via fallback
        self.total_requested = 0
        self.total_successful = 0
        self.total_failed = 0
        self.total_fallback_success = 0

    def add_failed_fetch(self, coord: Tuple[float, float], way_id: str, street_name: str, datasets_tried: List[str]):
        """Record a failed elevation fetch with metadata."""
        self.failed_fetches.append({
            'coordinate': coord,
            'way_id': way_id,
            'street_name': street_name,
            'datasets_tried': datasets_tried
        })
        self.total_failed += 1

    def add_successful_fetch(self, used_fallback: bool = False, dataset: str = None):
        """
        Record a successful elevation fetch.

        Args:
            used_fallback: True if this was retrieved from a fallback dataset
            dataset: Name of the dataset that provided the elevation
        """
        self.total_successful += 1
        if used_fallback:
            self.total_fallback_success += 1

    def print_summary(self):
        """Print comprehensive elevation fetch summary."""
        print("\n" + "=" * 80)
        print("ELEVATION FETCH SUMMARY")
        print("=" * 80)
        print(f"Total coordinates requested:  {self.total_requested:,}")
        print(f"Successfully fetched:         {self.total_successful:,} ({self.total_successful / max(self.total_requested, 1) * 100:.1f}%)")
        print(f"Failed to fetch:              {self.total_failed:,} ({self.total_failed / max(self.total_requested, 1) * 100:.1f}%)")

        if self.failed_fetches:
            print("\n" + "-" * 80)
            print("FAILED ELEVATION FETCHES (detailed log):")
            print("-" * 80)
            for entry in self.failed_fetches[:100]:  # Limit to first 100 for readability
                lat, lon = entry['coordinate']
                print(f"\nCoordinate:   ({lat:.6f}, {lon:.6f})")
                print(f"OSM Way ID:   {entry['way_id']}")
                print(f"Street Name:  {entry['street_name']}")
                print("API URLs tried:")
                for url in entry['api_urls_tried']:
                    print(f"  - {url}")

            if len(self.failed_fetches) > 100:
                print(f"\n... and {len(self.failed_fetches) - 100:,} more failed fetches (omitted for brevity)")

        print("=" * 80 + "\n")


def build_elevation_url(dataset_name: str) -> Optional[str]:
    """
    Build full elevation API URL from base URL and dataset name.

    Args:
        dataset_name: Dataset name like 'srtm30m', 'ned10m', 'aster30m'

    Returns:
        Full URL like 'http://localhost:5000/v1/srtm30m' or None
    """
    try:
        from utils.config_loader import TOPO_API_BASE_URL
    except ImportError:
        return None

    if not TOPO_API_BASE_URL:
        return None

    # Remove trailing slash if present
    base = TOPO_API_BASE_URL.rstrip("/")

    # Add dataset name
    return f"{base}/{dataset_name}"


class FastElevationFetcher:
    """
    Simplified parallel elevation fetcher with local GeoTIFF support.

    This class handles fetching elevation data from a local elevation API,
    with automatic fallback between datasets and adaptive batch sizing to
    handle server load.
    """

    def __init__(self, primary_dataset: str = None, region_name: str = None, lat: float = None, error_logger=None):
        """
        Initialize elevation fetcher.

        Args:
            primary_dataset: Primary dataset to use ('srtm30m', 'ned10m', 'aster30m', 'rema32m', 'arctic32m')
                           If None, will be auto-selected based on region_name/lat
            region_name: Region name for automatic dataset selection (e.g., "Alaska", "Antarctica")
            lat: Center latitude for automatic dataset selection
            error_logger: Optional ErrorLogger instance for coordinate-level CSV logging
        """
        # Auto-select dataset based on region if not explicitly provided
        if primary_dataset is None:
            dataset_priority = get_dataset_priority_for_region(region_name, lat)
            primary_dataset = dataset_priority[0] if dataset_priority else "srtm30m"
            self.dataset_priority = dataset_priority
        else:
            # If explicitly provided, use it as sole priority
            self.dataset_priority = [primary_dataset]

        self.primary_url = build_elevation_url(primary_dataset)
        self.primary_dataset = primary_dataset
        self.region_name = region_name
        self.optimal_batch_size = ELEVATION_BATCH_SIZE
        self.batch_size_adapted = False  # Track if we've found optimal size
        self.min_batch_size = 99  # Minimum acceptable batch size before erroring out
        self.error_logger = error_logger  # Optional ErrorLogger for CSV logging

        # Track 429 rate limit errors
        self.rate_limit_429_count = 0
        self.rate_limit_429_shown_full_msg = False  # Track if we've shown the long message

        # Track unavailable datasets (only for current session, not persisted)
        # Datasets that return 400 "not in config" are added here to avoid repeated failures
        self.unavailable_datasets = set()

        # Configure fallback endpoints for cascading elevation fetching
        self.fallback_urls = self._configure_fallback_endpoints()

    def _configure_fallback_endpoints(self) -> List[str]:
        """
        Configure cascading fallback endpoints based on dataset priority.

        Uses the dataset priority list from region-based selection,
        or falls back to standard sequence if not available.
        Only includes datasets that are available in OpenTopoData.

        Returns:
            List of fallback endpoint URLs
        """
        fallback_urls = []

        # Use dataset priority if available (from region-based selection)
        if hasattr(self, 'dataset_priority') and self.dataset_priority:
            # Skip the primary dataset (already set), use the rest as fallbacks
            fallback_datasets = self.dataset_priority[1:]
        else:
            # Standard fallback sequence for backward compatibility
            fallback_datasets = ["ned10m", "srtm30m", "aw3d30", "aster30m"]

        # Get list of available datasets from OpenTopoData
        available_datasets = self._get_available_datasets()

        for dataset in fallback_datasets:
            # Skip datasets that returned 400 "not in config" earlier in this session
            if dataset in self.unavailable_datasets:
                continue

            # Skip datasets that aren't available in OpenTopoData
            if available_datasets and dataset not in available_datasets:
                continue

            endpoint = build_elevation_url(dataset)
            if (
                endpoint
                and endpoint != self.primary_url
                and endpoint not in fallback_urls
            ):
                fallback_urls.append(endpoint)

        return fallback_urls

    def _get_available_datasets(self) -> List[str]:
        """
        Query OpenTopoData to get list of available datasets.

        Returns:
            List of available dataset names, or None if query fails
        """
        try:
            # Get base URL without dataset name
            base_url = os.getenv("TOPO_API_BASE_URL", "http://opentopodata-server:5000/v1")
            base_url = base_url.rstrip('/')

            # Try to query a test endpoint to see what datasets exist
            # OpenTopoData returns 400 with available datasets in error message
            # We'll parse the error to get the list

            # For now, return None to skip availability check
            # A full implementation would query the OpenTopoData API or config
            return None
        except Exception:
            return None

    def fetch_elevations_for_coordinates(
        self,
        coordinates: List[Tuple[float, float]],
        persistence,
        progress_desc: str = "Fetching elevation",
        coord_metadata: Optional[Dict[Tuple[float, float], Dict[str, str]]] = None,
        fetch_log: Optional[ElevationFetchLog] = None,
    ) -> List[Optional[float]]:
        """
        Main entry point - uses parallel processing for optimal performance.

        Args:
            coordinates: List of (lat, lon) tuples
            persistence: Persistence manager for checkpointing
            progress_desc: Description for progress bar
            coord_metadata: Optional dict mapping coordinates to metadata (way_id, street_name)
            fetch_log: Optional ElevationFetchLog to track detailed fetch information

        Returns:
            List of elevation values (None for failed lookups)
        """
        return self._fetch_elevations_parallel(
            coordinates, persistence, progress_desc, coord_metadata, fetch_log
        )

    def _save_optimal_batch_size_to_config(self):
        """Save the optimal batch size back to config.yaml."""
        try:
            import yaml

            config_path = Path("config.yaml")

            config = {}
            if config_path.exists():
                # Load existing config
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}

            # Update the batch size
            old_batch_size = config.get("ELEVATION_BATCH_SIZE", ELEVATION_BATCH_SIZE)
            config["ELEVATION_BATCH_SIZE"] = self.optimal_batch_size

            # Save back to file
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=True)

            print(
                f"Saved optimal batch size to config.yaml: {old_batch_size} → {self.optimal_batch_size}"
            )

        except Exception as e:
            print(f"Warning: Could not save optimal batch size to config.yaml: {e}")

    def _wait_with_backoff(self, wait_seconds: float, reason: str = ""):
        """
        Wait for server recovery with progress indication.

        Args:
            wait_seconds: Number of seconds to wait
            reason: Optional reason for the wait
        """
        if wait_seconds > 0:
            print(
                f"Waiting {wait_seconds}s for server recovery{' (' + reason + ')' if reason else ''}..."
            )
            time.sleep(wait_seconds)

    def _reduce_batch_size(self, reason: str = "HTTP error", with_backoff: bool = False) -> bool:
        """
        Conservative batch size reduction.

        Args:
            reason: Reason for batch size reduction
            with_backoff: Whether to apply backoff wait (deprecated)

        Returns:
            True if batch size was reduced, False if already at minimum
        """
        # Reduce by 10% each time to find optimal batch size gradually
        new_size = max(99, int(self.optimal_batch_size * 0.9))

        if new_size != self.optimal_batch_size:
            print(
                f"{reason} - reducing batch size conservatively: {self.optimal_batch_size} → {new_size}"
            )
            self.optimal_batch_size = new_size
            self.batch_size_adapted = True
            return True
        return False

    def _fetch_elevations_parallel(
        self,
        coordinates: List[Tuple[float, float]],
        persistence,
        progress_desc: str,
        coord_metadata: Optional[Dict[Tuple[float, float], Dict[str, str]]] = None,
        fetch_log: Optional[ElevationFetchLog] = None,
        silent_mode: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> List[Optional[float]]:
        """
        Parallel elevation fetching with smart checkpointing and adaptive batch sizing.

        Args:
            coordinates: List of (lat, lon) tuples
            persistence: Persistence manager for checkpointing
            progress_desc: Description for progress bar
            coord_metadata: Optional dict mapping coordinates to metadata (way_id, street_name)
            fetch_log: Optional ElevationFetchLog to track detailed fetch information
            progress_callback: Optional callback function(n) to update external progress bar

        Returns:
            List of elevation values (None for failed lookups)
        """
        # Import dependencies that may not be available at module level
        try:
            from climb_analyzer.processing.checkpoint import SmartCheckpointer
            from climb_analyzer.utils.helpers import check_and_cleanup_memory
        except ImportError:
            # Fallback if refactored modules not available yet
            from climb_analyzer import SmartCheckpointer, check_and_cleanup_memory

        # Import signal handler
        try:
            # In refactored version, this would be injected
            signal_handler = None
        except Exception:
            signal_handler = None

        # Existing coordinate deduplication logic
        unique_coords = []
        coord_to_index = {}
        original_to_unique = []

        for i, coord in enumerate(coordinates):
            rounded_coord = (round(coord[0], 6), round(coord[1], 6))

            if rounded_coord not in coord_to_index:
                coord_to_index[rounded_coord] = len(unique_coords)
                unique_coords.append(rounded_coord)

            original_to_unique.append(coord_to_index[rounded_coord])

        total_unique = len(unique_coords)
        total_original = len(coordinates)

        # Silent mode: no progress bar, no error messages during execution
        # All errors will be collected and reported at the end
        if progress_desc is None:
            silent_mode = True

        # Process unique coordinates in batches with parallel execution
        all_elevations = [None] * total_unique
        coordinate_mapping = {}

        # Track elevation fetch statistics
        total_coords_requested = 0
        total_coords_failed = 0
        failed_batch_count = 0  # Track completely failed batches
        first_connection_error = None  # Track first error for diagnostics

        # Initialize smart checkpointer with dynamic total calculation
        def calculate_total_batches():
            return (
                total_unique + self.optimal_batch_size - 1
            ) // self.optimal_batch_size

        total_batches = calculate_total_batches()
        checkpointer = SmartCheckpointer(total_batches, "Elevation Fetching")

        # Prepare all batches upfront
        all_batches = []
        coord_idx = 0
        while coord_idx < total_unique:
            current_batch_size = min(self.optimal_batch_size, total_unique - coord_idx)
            batch_coords = unique_coords[coord_idx : coord_idx + current_batch_size]
            all_batches.append((len(all_batches), coord_idx, batch_coords))
            coord_idx += current_batch_size

        # Create progress bar only if description provided (silent mode when None)
        use_progress_bar = progress_desc is not None

        # Create dummy progress bar class for silent mode
        class DummyProgressBar:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_postfix(self, *args, **kwargs):
                pass

        if use_progress_bar:
            pbar = tqdm(
                total=total_unique,
                desc=progress_desc,
                unit="coords",
                mininterval=1.0,
                dynamic_ncols=True,
                file=sys.stderr,
            )
        else:
            pbar = DummyProgressBar()

        with pbar:

            # Direct parallel processing - submit all batch requests immediately
            def make_direct_request(batch_info):
                """Make direct API request without wrapper overhead."""
                batch_num, start_idx, batch_coords = batch_info
                try:
                    # Direct API call with minimal overhead
                    elevations, api_urls_tried, coord_dataset_sources = self._fetch_single_batch(
                        batch_coords, max_retries=5, base_delay=1.0, silent_mode=silent_mode
                    )
                    return (start_idx, batch_coords, elevations, api_urls_tried, coord_dataset_sources, batch_num)
                except Exception as e:
                    if not silent_mode:
                        print(f"Direct request error for batch {batch_num}: {e}")
                    return (
                        start_idx,
                        batch_coords,
                        [None] * len(batch_coords),
                        [],
                        {},
                        batch_num,
                    )

            # Submit ALL batches to ThreadPoolExecutor at once for true parallelism
            with ThreadPoolExecutor(max_workers=ELEVATION_MAX_CONCURRENT) as executor:
                completed_batches = 0

                # Submit ALL requests immediately
                future_to_batch = {
                    executor.submit(make_direct_request, batch_info): batch_info
                    for batch_info in all_batches
                }

                # Process results as they complete with periodic interrupt checking
                remaining_futures = set(future_to_batch.keys())
                while remaining_futures:
                    # Check for shutdown signal
                    if signal_handler and signal_handler.kill_now:
                        print("\nCanceling remaining elevation requests...")
                        for f in remaining_futures:
                            f.cancel()
                        batch_info_dict = {
                            "completed_batches": completed_batches,
                            "total_batches": len(all_batches),
                            "last_batch_index": 0,
                            "total_unique": total_unique,
                            "optimal_batch_size": self.optimal_batch_size,
                        }
                        persistence.save_elevation_progress(
                            coordinate_mapping, batch_info_dict
                        )
                        print("Elevation progress saved. Analysis can be resumed.")
                        os._exit(0)

                    # Wait for next future with timeout to allow periodic signal checking
                    try:
                        done, remaining_futures = concurrent.futures.wait(
                            remaining_futures,
                            timeout=0.5,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                    except Exception:
                        continue

                    # Process completed futures
                    for future in done:
                        batch_info = future_to_batch[future]
                        batch_num = batch_info[0]

                        try:
                            (
                                start_idx,
                                batch_coords,
                                batch_elevations,
                                api_urls_tried,
                                coord_dataset_sources,
                                returned_batch_num,
                            ) = future.result()

                            # Store results in the main arrays and track failures
                            batch_failed_count = 0
                            for j, elevation in enumerate(batch_elevations):
                                actual_coord_idx = start_idx + j
                                if actual_coord_idx < total_unique:
                                    coord = unique_coords[actual_coord_idx]
                                    all_elevations[actual_coord_idx] = elevation
                                    total_coords_requested += 1
                                    if elevation is not None:
                                        coordinate_mapping[coord] = elevation
                                        if fetch_log:
                                            # Determine if this was a fallback success
                                            datasets_used = coord_dataset_sources.get(j, [])
                                            primary_dataset = self.primary_url.split('/')[-1] if self.primary_url else 'unknown'
                                            used_fallback = datasets_used and datasets_used[0] != primary_dataset

                                            fetch_log.add_successful_fetch(used_fallback=used_fallback, dataset=datasets_used[0] if datasets_used else None)

                                        # Log to CSV error logger if fallback was used
                                        if self.error_logger and coord_metadata and coord in coord_metadata:
                                            datasets_used = coord_dataset_sources.get(j, [])
                                            primary_dataset_name = self.primary_dataset
                                            used_fallback = datasets_used and datasets_used[0] != primary_dataset_name

                                            if used_fallback:
                                                # Log INFO for fallback success
                                                metadata = coord_metadata[coord]
                                                datasets_tried = [url.split('/')[-1] for url in api_urls_tried]
                                                self.error_logger.log_coordinate_failure(
                                                    coordinate=coord,
                                                    street_name=metadata.get('street_name', 'Unknown'),
                                                    osm_way_id=metadata.get('way_id', 'Unknown'),
                                                    datasets_tried=datasets_tried,
                                                    primary_dataset=primary_dataset_name,
                                                    successful_dataset=datasets_used[0] if datasets_used else None,
                                                    level="INFO"
                                                )
                                    else:
                                        batch_failed_count += 1
                                        total_coords_failed += 1
                                        # Log failed fetch with metadata if available
                                        if fetch_log and coord_metadata and coord in coord_metadata:
                                            metadata = coord_metadata[coord]
                                            # Extract dataset names from URLs
                                            datasets_tried = [url.split('/')[-1] for url in api_urls_tried]
                                            fetch_log.add_failed_fetch(
                                                coord,
                                                metadata.get('way_id', 'Unknown'),
                                                metadata.get('street_name', 'Unknown'),
                                                datasets_tried
                                            )

                                        # Log ERROR to CSV error logger
                                        if self.error_logger and coord_metadata and coord in coord_metadata:
                                            metadata = coord_metadata[coord]
                                            datasets_tried = [url.split('/')[-1] for url in api_urls_tried]
                                            self.error_logger.log_coordinate_failure(
                                                coordinate=coord,
                                                street_name=metadata.get('street_name', 'Unknown'),
                                                osm_way_id=metadata.get('way_id', 'Unknown'),
                                                datasets_tried=datasets_tried,
                                                primary_dataset=self.primary_dataset,
                                                successful_dataset=None,
                                                level="ERROR"
                                            )

                            # Track if entire batch failed (but don't print - will show in summary)
                            if (
                                batch_failed_count == len(batch_elevations)
                                and len(batch_elevations) > 0
                            ):
                                failed_batch_count += 1

                            completed_batches += 1

                            # Update progress bar by number of coordinates processed in this batch
                            pbar.update(len(batch_coords))
                            pbar.set_postfix(
                                {
                                    "completed": completed_batches,
                                    "total": len(all_batches),
                                    "batch_size": self.optimal_batch_size,
                                    "failed": total_coords_failed,
                                }
                            )

                            # Call external progress callback if provided (for parent progress bar)
                            if progress_callback:
                                progress_callback(len(batch_coords))

                            # Checkpoint every 50 batches to reduce overhead
                            if (
                                completed_batches % 50 == 0
                                and checkpointer.should_checkpoint(
                                    completed_batches - 1
                                )
                            ):
                                batch_checkpoint_info = {
                                    "completed_batches": completed_batches,
                                    "total_batches": len(all_batches),
                                    "last_batch_index": start_idx,
                                    "total_unique": total_unique,
                                    "optimal_batch_size": self.optimal_batch_size,
                                }
                                persistence.save_elevation_progress(
                                    coordinate_mapping, batch_checkpoint_info
                                )

                        except Exception as e:
                            if not silent_mode:
                                print(f"Error processing result for batch {batch_num}: {e}")
                            completed_batches += 1
                            # Update by batch size estimate
                            pbar.update(self.optimal_batch_size)

                            # Call external progress callback if provided
                            if progress_callback:
                                progress_callback(self.optimal_batch_size)

        # Save optimal batch size to config if it was adapted
        if self.batch_size_adapted and self.optimal_batch_size != ELEVATION_BATCH_SIZE:
            self._save_optimal_batch_size_to_config()

        # Map back to original coordinate order and cleanup
        result_elevations = [
            all_elevations[unique_idx] for unique_idx in original_to_unique
        ]

        successful_total = sum(1 for e in result_elevations if e is not None)

        # Update fetch_log if provided (summary will be printed by caller)
        if fetch_log:
            fetch_log.total_requested = total_original
            # Don't print summary here - caller will use fetch_log.print_summary()

        # DEBUG: Save sample of failed coordinates for debugging
        if not silent_mode and total_coords_failed > 0:
            failed_coords = [
                unique_coords[i] for i in range(total_unique)
                if all_elevations[i] is None
            ]
            if failed_coords:
                # Save first 100 failed coordinates to a debug file
                try:
                    import json
                    debug_file = Path("debug_failed_coordinates.json")
                    sample = failed_coords[:100]
                    with open(debug_file, "w") as f:
                        json.dump({
                            "total_failed": len(failed_coords),
                            "sample_coordinates": sample,
                            "primary_url": self.primary_url,
                            "fallback_urls": self.fallback_urls
                        }, f, indent=2)
                    print(f"\n📝 Saved {len(sample)} sample failed coordinates to {debug_file}")
                except Exception:
                    pass  # Don't fail if debug logging fails

        persistence.clear_elevation_progress()
        check_and_cleanup_memory(force_cleanup=True)

        return result_elevations

    def _fetch_single_batch(
        self,
        coordinates: List[Tuple[float, float]],
        max_retries: int,
        base_delay: float,
        silent_mode: bool = False,
    ) -> Tuple[List[Optional[float]], List[str], Dict[int, List[str]]]:
        """
        Fetch elevation data for a single batch with cascading fallback logic.

        Args:
            coordinates: List of (lat, lon) tuples
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            silent_mode: If True, suppress error messages

        Returns:
            Tuple of:
            - elevation values list
            - list of API URLs tried (for all coords)
            - dict mapping coord index to list of datasets that returned the value
              (for tracking which dataset provided each elevation)
        """
        api_urls_tried = []

        # Track which dataset provided each coordinate's elevation
        coord_dataset_sources = {}  # {coord_index: [dataset_name, ...]}

        # Try primary endpoint first
        api_urls_tried.append(self.primary_url)
        result = self._fetch_single_batch_from_endpoint(
            coordinates, self.primary_url, max_retries, base_delay, silent_mode
        )

        # Track which coordinates got values from primary
        if result is not None:
            primary_dataset = self.primary_url.split('/')[-1]  # Extract dataset name from URL
            for i, elev in enumerate(result):
                if elev is not None:
                    coord_dataset_sources[i] = [primary_dataset]

        # Always try fallback endpoints for any coordinates with missing data
        # This ensures we get the best coverage even if primary returned some data
        missing_indices = [i for i, e in enumerate(result or [None] * len(coordinates)) if e is None]

        if missing_indices:
            # Try fallback endpoints for coordinates with missing data
            for fallback_url in self.fallback_urls:
                api_urls_tried.append(fallback_url)
                fallback_result = self._fetch_single_batch_from_endpoint(
                    coordinates, fallback_url, max_retries=5, base_delay=base_delay, silent_mode=silent_mode
                )

                if fallback_result is not None:
                    fallback_dataset = fallback_url.split('/')[-1]

                    # Combine results - use fallback data where primary failed
                    if result is None:
                        result = fallback_result
                        # Track sources for all non-None values from this fallback
                        for i, elev in enumerate(result):
                            if elev is not None:
                                coord_dataset_sources[i] = [fallback_dataset]
                    else:
                        # Merge results - use fallback where primary returned None
                        for i, (primary_val, fallback_val) in enumerate(
                            zip(result, fallback_result)
                        ):
                            if primary_val is None and fallback_val is not None:
                                result[i] = fallback_val
                                coord_dataset_sources[i] = [fallback_dataset]

                    # Check if we now have all data
                    valid_count = sum(1 for e in result if e is not None)
                    missing_count = len(coordinates) - valid_count
                    if missing_count == 0:
                        break

        # Return result or all None if all endpoints failed
        return (
            result if result is not None else [None] * len(coordinates),
            api_urls_tried,
            coord_dataset_sources
        )

    def _fetch_single_batch_from_endpoint(
        self,
        coordinates: List[Tuple[float, float]],
        endpoint_url: str,
        max_retries: int,
        base_delay: float,
        silent_mode: bool = False,
    ) -> Optional[List[Optional[float]]]:
        """
        Fetch elevation data from a specific endpoint with retry logic.

        Args:
            coordinates: List of (lat, lon) tuples
            endpoint_url: URL of the elevation API endpoint
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff

        Returns:
            List of elevation values or None if all retries failed
        """
        locations_str = "|".join(
            [f"{round(lat, 5)},{round(lon, 5)}" for lat, lon in coordinates]
        )
        params = {"locations": locations_str}

        for attempt in range(max_retries + 1):
            try:
                response = requests.get(
                    endpoint_url,
                    params=params,
                    timeout=ELEVATION_REQUEST_TIMEOUT_SEC,
                    headers={"Connection": "close"},
                )

                # Check response before raising for status to get better error info
                if not response.ok:
                    # Extract base URL without parameters
                    base_url = endpoint_url.split("?")[0]

                    # Special handling for 429 rate limiting
                    if response.status_code == 429:
                        if attempt < max_retries:
                            # Silently retry with exponential backoff
                            backoff_delay = base_delay * (4**attempt)
                            time.sleep(backoff_delay)
                            continue
                        else:
                            # Last attempt failed - increment counter
                            self.rate_limit_429_count += 1

                            if not silent_mode:
                                # Show full message only once, then compact warnings
                                if not self.rate_limit_429_shown_full_msg:
                                    print("\n" + "=" * 70)
                                    print("⚠️  ELEVATION API RATE LIMITED (Transient)")
                                    print("=" * 70)
                                    print(
                                        f"HTTP 429 from {base_url} after {max_retries + 1} attempts"
                                    )
                                    print(
                                        "\nThis is likely a temporary rate limit, not the daily quota."
                                    )
                                    print("The script will continue processing other batches.")
                                    print("\nIf you see many of these:")
                                    print(
                                        "  • Daily limit may be approaching (100,000 coords/day)"
                                    )
                                    print(
                                        "  • Consider reducing ELEVATION_MAX_CONCURRENT in config.yaml"
                                    )
                                    print("\n✓ Progress is automatically saved")
                                    print("=" * 70 + "\n")
                                    self.rate_limit_429_shown_full_msg = True
                                else:
                                    # Compact warning for subsequent 429 errors
                                    print(f"⚠️  HTTP 429 from {base_url} [{self.rate_limit_429_count} times]")
                            return None

                    # Special handling for 504 gateway timeout
                    elif response.status_code == 504:
                        if attempt < max_retries:
                            backoff_delay = base_delay * (2**attempt)
                            time.sleep(backoff_delay)
                            continue
                        else:
                            if not silent_mode:
                                print(
                                    f"⚠️  HTTP 504 (gateway timeout) from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                                )
                            return None

                    # Special handling for 400 Bad Request (dataset not available)
                    elif response.status_code == 400:
                        # Check if error is "Dataset not in config"
                        error_msg = None
                        try:
                            error_data = response.json()
                            error_msg = error_data.get('error', '')
                        except Exception:
                            pass

                        # If dataset not available, don't retry - immediately try fallback
                        if error_msg and 'not in config' in error_msg.lower():
                            # Extract dataset name from URL
                            dataset_name = base_url.split('/')[-1]

                            # Add to unavailable datasets list for this session
                            if dataset_name not in self.unavailable_datasets:
                                self.unavailable_datasets.add(dataset_name)

                                # Rebuild fallback URLs to exclude this dataset
                                self.fallback_urls = self._configure_fallback_endpoints()

                                if not silent_mode:
                                    print(
                                        f"   Dataset '{dataset_name}' not available in OpenTopoData, skipping for rest of analysis"
                                    )
                                    if self.fallback_urls:
                                        fallback_names = [url.split('/')[-1] for url in self.fallback_urls]
                                        print(f"   Fallback datasets: {', '.join(fallback_names)}")

                            return None  # Trigger fallback immediately
                        else:
                            # Other 400 errors - retry
                            if attempt < max_retries:
                                backoff_delay = base_delay * (2**attempt)
                                time.sleep(backoff_delay)
                                continue
                            else:
                                if not silent_mode:
                                    print(
                                        f"⚠️  HTTP 400 error from {base_url} persisted after {max_retries + 1} attempts"
                                    )
                                    if response.text and not response.text.strip().startswith("<"):
                                        print(f"  Response: {response.text[:200]}")
                                return None

                    # Other errors - retry with backoff
                    else:
                        if attempt < max_retries:
                            backoff_delay = base_delay * (2**attempt)
                            time.sleep(backoff_delay)
                            continue
                        else:
                            if not silent_mode:
                                print(
                                    f"⚠️  HTTP {response.status_code} error from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                                )
                                if response.text and not response.text.strip().startswith(
                                    "<"
                                ):
                                    print(f"  Response: {response.text[:200]}")
                            return None

                response.raise_for_status()
                data = response.json()

                if data.get("status") != "OK":
                    if not silent_mode:
                        base_url = endpoint_url.split("?")[0]
                        print(f"API returned status: {data.get('status')} from {base_url}")
                    return [None] * len(coordinates)

                elevations = []
                results = data.get("results", [])

                for result in results:
                    if (
                        result
                        and "elevation" in result
                        and result["elevation"] is not None
                    ):
                        elevations.append(float(result["elevation"]))
                    else:
                        elevations.append(None)

                # Ensure the number of elevations matches the number of coordinates
                while len(elevations) < len(coordinates):
                    elevations.append(None)

                return elevations[: len(coordinates)]

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else "Unknown"
                base_url = endpoint_url.split("?")[0]

                if attempt >= max_retries:
                    if not silent_mode:
                        if status_code == "Unknown":
                            print(
                                f"⚠️  HTTP error with no status code from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                            )
                            print(f"  Error type: {type(e).__name__}, Details: {str(e)}")
                        else:
                            print(
                                f"⚠️  HTTP {status_code} error from {base_url} persisted after {max_retries + 1} attempts (batch size: {len(coordinates)})"
                            )
                    return None

                # Retry silently
                delay = (
                    base_delay * (4**attempt)
                    if status_code == 429
                    else base_delay * (2**attempt)
                )
                time.sleep(delay)
                continue

            except requests.exceptions.RequestException as e:
                base_url = endpoint_url.split("?")[0]

                # Store first error for diagnostic purposes (without printing spam)
                # This will be shown in the summary if all requests fail
                error_info = {
                    'type': type(e).__name__,
                    'message': str(e),
                    'url': base_url
                }

                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                else:
                    # Clean error message - strip query parameters
                    error_str = str(e)
                    import re
                    # Remove everything after ? in URLs (including multiline query params)
                    error_str = re.sub(r'/v1/[^?]+\?[^)]+', lambda m: m.group(0).split('?')[0], error_str)
                    # Also clean up any remaining ? query params
                    error_str = re.sub(r'\?locations=[^\s)]+', '', error_str)

                    # Only print first 10 connection errors to avoid spam
                    # (actual count is tracked and shown in summary)
                    if not silent_mode:
                        print(
                            f"⚠️  Connection error after {max_retries + 1} attempts from {base_url}"
                        )
                        print(f"  {type(e).__name__}: {error_str}")
                    return None

            except (ValueError, KeyError) as e:
                if not silent_mode:
                    base_url = endpoint_url.split("?")[0]
                    print(f"Failed to parse elevation response from {base_url}: {e}")
                return [None] * len(coordinates)

        return None
