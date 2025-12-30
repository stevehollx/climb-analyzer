"""
Reverse geocoding functionality for climb analysis.

This module provides reverse geocoding using the offline reverse_geocoder library.
"""

import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

# NOTE: reverse_geocoder is imported locally in functions to avoid loading
# the large spatial index (~8-10GB) at module import time
from tqdm import tqdm


class ReverseGeocoder:
    """
    Reverse geocoder using offline reverse_geocoder module.

    This class provides offline reverse geocoding with spatial
    clustering to minimize redundant lookups.
    """

    def __init__(self, max_concurrent: int = None):
        """
        Initialize the reverse geocoder.

        Args:
            max_concurrent: Ignored since reverse_geocoder is synchronous
        """
        # max_concurrent is ignored since reverse_geocoder is synchronous and fast
        print(".")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    async def reverse_geocode_parallel(
        self,
        coordinates: List[Tuple[float, float]],
        persistence,
        progress_desc: str = "Geocoding",
    ) -> Dict:
        """
        Main parallel reverse geocoding function using offline reverse_geocoder.

        Args:
            coordinates: List of (lat, lon) tuples
            persistence: Persistence manager for checkpointing
            progress_desc: Description for progress bar

        Returns:
            Dictionary mapping coordinate indices to location info
        """
        if not coordinates:
            return {}

        # Aggressive coordinate deduplication with spatial clustering
        unique_coords, coord_mapping = self._deduplicate_with_clustering(coordinates)

        print(f"Geocoding {len(unique_coords)} unique locations using offline data")

        # Import reverse_geocoder locally to defer loading large spatial index (~8-10GB)
        # until it's actually needed (not at module import time)
        import reverse_geocoder as rg

        # Process coordinates with reverse_geocoder (much faster than API calls)
        coord_to_location = {}

        # Import signal handler - this is a cross-cutting concern
        # In a production refactor, this would be injected as a dependency
        try:
            from climb_analyzer.utils.graceful_killer import GracefulKiller
            # Note: In the original code, signal_handler is a global instance
            # For now, we'll need to import it from the main module when refactored
            # This is a temporary workaround
            signal_handler = None
        except ImportError:
            signal_handler = None

        with tqdm(
            total=len(unique_coords),
            desc=progress_desc,
            unit="coord",
            mininterval=0.1,
            dynamic_ncols=True,
        ) as pbar:
            # Process in batches for better progress indication
            batch_size = 1700  # Large batches since it's offline
            for i in range(0, len(unique_coords), batch_size):
                batch_coords = unique_coords[i : i + batch_size]

                # Signal handler check
                if signal_handler:
                    signal_handler.set_operation(
                        "geocoding",
                        {
                            "completed_count": i,
                            "total_count": len(unique_coords),
                            "coord_to_location": coord_to_location,
                            "coord_mapping": coord_mapping,
                            "timestamp": time.time(),
                        },
                    )

                    if signal_handler.kill_now:
                        self._save_geocoding_checkpoint(
                            persistence,
                            coord_to_location,
                            coord_mapping,
                            i,
                            len(unique_coords),
                        )
                        print("Geocoding progress saved. Analysis can be resumed.")
                        sys.exit(0)

                try:
                    # Use reverse_geocoder for batch lookup
                    results = rg.search(batch_coords)

                    # Process results
                    for j, (coord, result) in enumerate(zip(batch_coords, results)):
                        if result:
                            # Extract city and state from reverse_geocoder result
                            city = result.get("name", "Unknown")

                            # reverse_geocoder uses 'admin1' for state/province
                            state = result.get("admin1", "Unknown")

                            # Build full address
                            country = result.get("cc", "")  # Country code
                            full_address_parts = [city]
                            if state and state != "Unknown":
                                full_address_parts.append(state)
                            if country:
                                full_address_parts.append(country)
                            full_address = ", ".join(full_address_parts)

                            coord_to_location[coord] = {
                                "city": city,
                                "state": state,
                                "full_address": full_address,
                            }
                        else:
                            coord_to_location[coord] = {
                                "city": "Unknown",
                                "state": "Unknown",
                                "full_address": "Not found",
                            }

                except Exception as e:
                    print(f"Error in reverse geocoding batch: {e}")
                    # Fill with fallback data
                    for coord in batch_coords:
                        coord_to_location[coord] = {
                            "city": "Lookup Failed",
                            "state": "Lookup Failed",
                            "full_address": f"Error: {str(e)[:50]}",
                        }

                # Update progress
                completed = min(i + batch_size, len(unique_coords))
                pbar.n = completed
                pbar.set_postfix(
                    {
                        "success_rate": f"{len([v for v in coord_to_location.values() if 'Error' not in v.get('city', '')])}/{completed}"
                    }
                )
                pbar.refresh()

        # Map results back to all original coordinates
        full_results = {}
        for i, original_coord in enumerate(coordinates):
            clustered_coord = coord_mapping.get(original_coord, original_coord)
            if clustered_coord in coord_to_location:
                full_results[i] = coord_to_location[clustered_coord]
            else:
                full_results[i] = {
                    "city": "Unknown",
                    "state": "Unknown",
                    "full_address": "Not found",
                }

        successful_lookups = len(
            [v for v in coord_to_location.values() if "Error" not in v.get("city", "")]
        )
        print(
            f"Successfully geocoded {successful_lookups}/{len(unique_coords)} unique locations using offline data"
        )

        return full_results

    def _deduplicate_with_clustering(
        self, coordinates: List[Tuple[float, float]], cluster_radius_deg: float = 0.01
    ) -> Tuple[List[Tuple[float, float]], Dict]:
        """
        Advanced deduplication using spatial clustering to reduce lookups.

        Args:
            coordinates: List of (lat, lon) tuples
            cluster_radius_deg: Cluster radius in degrees (default ~1km at equator)

        Returns:
            Tuple of (unique_coords, coord_mapping)
        """
        if not coordinates:
            return [], {}

        # Simple grid-based clustering - coordinates within same grid cell share lookup
        cluster_map = defaultdict(list)
        coord_mapping = {}

        for coord in coordinates:
            # Round to grid cell (approximately 1km at equator)
            cluster_key = (
                round(coord[0] / cluster_radius_deg) * cluster_radius_deg,
                round(coord[1] / cluster_radius_deg) * cluster_radius_deg,
            )
            cluster_map[cluster_key].append(coord)

        # Use cluster center as representative coordinate
        unique_coords = []
        for cluster_key, coords_in_cluster in cluster_map.items():
            if coords_in_cluster:
                # Use the first coordinate in cluster as representative
                representative = coords_in_cluster[0]
                unique_coords.append(representative)

                # Map all coordinates in cluster to the representative
                for coord in coords_in_cluster:
                    coord_mapping[coord] = representative

        return unique_coords, coord_mapping

    def _save_geocoding_checkpoint(
        self,
        persistence,
        coord_to_location: Dict,
        coord_mapping: Dict,
        completed: int,
        total: int,
    ):
        """
        Save geocoding checkpoint.

        Args:
            persistence: Persistence manager
            coord_to_location: Mapping of coordinates to location info
            coord_mapping: Mapping of original coords to clustered coords
            completed: Number of completed geocoding operations
            total: Total number of geocoding operations
        """
        checkpoint_data = {
            "coord_to_location": coord_to_location,
            "coord_mapping": coord_mapping,
            "completed_count": completed,
            "total_count": total,
            "timestamp": time.time(),
        }

        try:
            geocoding_checkpoint_file = (
                persistence.analysis_dir / "geocoding_parallel_progress.pkl"
            )
            temp_file = persistence.analysis_dir / "geocoding_parallel_progress.tmp"

            with open(temp_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            temp_file.rename(geocoding_checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not save geocoding checkpoint: {e}")
