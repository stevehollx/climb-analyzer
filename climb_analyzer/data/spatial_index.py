"""
Spatial indexing for OSM data.

This module provides classes for creating and querying spatial indexes of OSM ways
and node locations for efficient geographic queries.
"""

import json
from pathlib import Path
from typing import List, Optional


class LazyWayMetadataDict:
    """
    Memory-efficient lazy-loading dictionary for way metadata.

    Instead of loading all 4M ways into memory, this builds an index of file positions
    and loads way data on-demand when accessed. Uses LRU cache for recently accessed ways.
    """

    def __init__(self, jsonl_file_path: Path, silent: bool = False, cache_size: int = 50000):
        """Initialize with JSONL file path and build/load position index."""
        self.jsonl_file = jsonl_file_path
        self.silent = silent
        self.cache_size = cache_size
        self.cache = {}  # LRU cache for recently accessed ways
        self.cache_order = []  # Track access order for LRU eviction
        self.way_positions = {}  # Maps way_id -> file_position
        self.metadata_header = {}

        # Cache file for position index (stored alongside JSONL file)
        self.index_cache_file = self.jsonl_file.with_suffix('.jsonl.idx_cache')

        # Try to load cached index, otherwise build it
        if not self._load_cached_index():
            self._build_position_index()
            self._save_cached_index()

    def _load_cached_index(self) -> bool:
        """
        Load position index from cache file if it exists and is up-to-date.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not self.index_cache_file.exists():
            return False

        # Check if cache is newer than the JSONL file
        jsonl_mtime = self.jsonl_file.stat().st_mtime
        cache_mtime = self.index_cache_file.stat().st_mtime

        if cache_mtime < jsonl_mtime:
            # Cache is outdated, rebuild
            return False

        try:
            import pickle

            with open(self.index_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.way_positions = cache_data['way_positions']
                self.metadata_header = cache_data['metadata_header']

            if not self.silent:
                print(f"  Loaded cached position index: {len(self.way_positions):,} ways ({len(self.way_positions) * 16 / 1024 / 1024:.1f} MB in memory)")

            return True

        except Exception as e:
            if not self.silent:
                print(f"  Warning: Could not load cached index: {e}")
            return False

    def _save_cached_index(self):
        """Save position index to cache file."""
        try:
            import pickle

            cache_data = {
                'way_positions': self.way_positions,
                'metadata_header': self.metadata_header
            }

            with open(self.index_cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            if not self.silent:
                print(f"  Warning: Could not save index cache: {e}")

    def _build_position_index(self):
        """Build index of file positions for each way ID."""
        if not self.silent:
            print(f"Building position index from: {self.jsonl_file.name}")

        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            # First line is metadata header
            line = f.readline()
            header = json.loads(line)
            if "_index_metadata" in header:
                self.metadata_header = header["_index_metadata"]

            total_ways = self.metadata_header.get("total_ways", 0)

            # Build position index with progress bar
            try:
                from tqdm import tqdm
                use_tqdm = not self.silent and total_ways > 0
            except ImportError:
                use_tqdm = False

            if use_tqdm:
                pbar = tqdm(total=total_ways, desc="  Indexing ways", unit=" ways",
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                           ascii=" █")

            # Read lines and track positions
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break

                if line.strip():
                    # Parse just the ID without loading full data
                    entry = json.loads(line)
                    way_id = entry["way_id"]
                    self.way_positions[way_id] = pos

                    if use_tqdm:
                        pbar.update(1)

            if use_tqdm:
                pbar.close()
                print()  # Clean separation

            if not self.silent:
                print(f"  Indexed {len(self.way_positions):,} way positions ({len(self.way_positions) * 16 / 1024 / 1024:.1f} MB in memory)")

    def __getitem__(self, way_id):
        """Load and return way data on-demand, with LRU caching."""
        if way_id == "_index_metadata":
            return self.metadata_header

        # Check cache first
        if way_id in self.cache:
            # Move to end of cache order (most recently used)
            self.cache_order.remove(way_id)
            self.cache_order.append(way_id)
            return self.cache[way_id]

        # Not in cache, load from file
        if way_id not in self.way_positions:
            raise KeyError(way_id)

        pos = self.way_positions[way_id]

        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            f.seek(pos)
            line = f.readline()
            entry = json.loads(line)
            # New flat format: {"way_id": ..., "tags": ..., "nodes": ..., "bbox": ...}
            data = {
                "tags": entry["tags"],
                "nodes": entry["nodes"],
                "bbox": entry["bbox"]
            }

        # Add to cache
        self.cache[way_id] = data
        self.cache_order.append(way_id)

        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        return data

    def __contains__(self, way_id):
        """Check if way_id exists."""
        return way_id in self.way_positions or way_id == "_index_metadata"

    def get(self, way_id, default=None):
        """Get way data with default fallback."""
        try:
            return self[way_id]
        except KeyError:
            return default

    def __len__(self):
        """Return number of ways indexed."""
        return len(self.way_positions)

    def get_batch(self, way_ids):
        """
        Efficiently load multiple ways at once.

        This is much faster than individual lookups when querying many ways,
        as it reads the file sequentially for nearby positions.

        Args:
            way_ids: List of way IDs to load

        Returns:
            Dictionary mapping way_id -> way_data
        """
        results = {}

        # Separate cached and uncached IDs
        uncached_ids = []
        for way_id in way_ids:
            if way_id in self.cache:
                results[way_id] = self.cache[way_id]
                # Update LRU order
                self.cache_order.remove(way_id)
                self.cache_order.append(way_id)
            elif way_id in self.way_positions:
                uncached_ids.append(way_id)

        # If all were cached, return immediately
        if not uncached_ids:
            return results

        # Sort uncached IDs by file position for sequential reading
        uncached_with_pos = [(way_id, self.way_positions[way_id])
                             for way_id in uncached_ids]
        uncached_with_pos.sort(key=lambda x: x[1])  # Sort by position

        # Read from file in order
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for way_id, pos in uncached_with_pos:
                f.seek(pos)
                line = f.readline()
                entry = json.loads(line)
                # New flat format: {"way_id": ..., "tags": ..., "nodes": ..., "bbox": ...}
                data = {
                    "tags": entry["tags"],
                    "nodes": entry["nodes"],
                    "bbox": entry["bbox"]
                }

                results[way_id] = data

                # Add to cache
                self.cache[way_id] = data
                self.cache_order.append(way_id)

                # Evict oldest if cache is full
                if len(self.cache) > self.cache_size:
                    oldest = self.cache_order.pop(0)
                    del self.cache[oldest]

        return results


class SpatialIndexManager:
    """
    Manages spatial indexing for OSM ways.

    This class provides an interface to load and query pre-built spatial indexes
    for efficient bounding box queries on OSM way data.
    """

    def __init__(self, osm_file_path: str, cache_dir: str = "data/osm_indexes", silent: bool = False, progress_counter=None):
        """
        Initialize the spatial index manager.

        Args:
            osm_file_path: Path to the OSM file
            cache_dir: Directory where index files are stored
            silent: If True, suppress progress messages during loading
            progress_counter: Shared multiprocessing counter for tracking ways loaded
        """
        self.osm_file_path = osm_file_path
        self.cache_dir = Path(cache_dir)
        self.silent = silent  # Store for use in load_index
        self.progress_counter = progress_counter  # Store for use in load_index

        # Calculate expected index file paths to match setup_wizard naming
        osm_file = Path(osm_file_path)

        # For "georgia-latest.osm.pbf", we want "georgia-latest.osm_spatial"
        # osm_file.stem gives us "georgia-latest.osm"
        base_name = osm_file.stem  # This should be "georgia-latest.osm"

        self.idx_file = self.cache_dir / f"{base_name}_spatial"
        self.metadata_file = self.cache_dir / f"{base_name}_metadata.jsonl"

        self.spatial_idx = None
        self.way_metadata = {}

    def create_index(self, surface_filter="all", cycling_only=True):
        """
        Index creation is not supported in this module.

        Index creation happens in setup_wizard.py.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError("Index creation happens in setup_wizard.py")

    def load_index(self, silent=False):
        """
        Load existing spatial index.

        Uses lazy-loading for way metadata to minimize memory usage.
        Instead of loading all 4M ways into memory, only file positions are indexed.

        Args:
            silent: If True, suppress progress messages (useful for multiprocessing workers)

        Raises:
            FileNotFoundError: If spatial index files not found
            ImportError: If rtree library not available
            Exception: If error occurs during loading
        """
        if not self.exists():
            raise FileNotFoundError(f"Spatial index not found: {self.idx_file}")

        try:
            # Import rtree here to avoid import errors in cloud deployment
            from rtree import index

            # Load the rtree spatial index
            self.spatial_idx = index.Index(str(self.idx_file))

            # Check metadata file exists
            if not self.metadata_file.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {self.metadata_file}. "
                    "Run setup wizard to create index."
                )

            # Use lazy-loading dictionary for way metadata
            # This only indexes file positions (~64 MB for 4M ways)
            # instead of loading all way data (~4+ GB)
            self.way_metadata = LazyWayMetadataDict(
                self.metadata_file, silent=silent
            )

            if not silent:
                print(f"  Spatial index loaded: {len(self.way_metadata):,} ways indexed")

        except ImportError:
            raise ImportError("rtree library required for spatial indexing")
        except Exception as e:
            raise Exception(f"Error loading spatial index: {e}")

    def exists(self) -> bool:
        """
        Check if index files exist AND are valid (non-empty).

        Returns:
            True if all required index files exist and have valid sizes
        """
        # Don't use with_suffix() - it replaces the entire suffix
        # Instead, append the extensions directly
        idx_file_path = Path(str(self.idx_file) + ".idx")
        dat_file_path = Path(str(self.idx_file) + ".dat")

        # Check files exist
        if not (idx_file_path.exists() and dat_file_path.exists()):
            return False
        if not self.metadata_file.exists():
            return False

        # Validate minimum file sizes (catch corrupted/incomplete builds)
        # Valid metadata has header + at least some ways (> 100 bytes)
        # Valid rtree index files are larger than empty stubs
        MIN_METADATA_SIZE = 100
        MIN_IDX_SIZE = 100
        MIN_DAT_SIZE = 1000

        try:
            if self.metadata_file.stat().st_size < MIN_METADATA_SIZE:
                return False
            if idx_file_path.stat().st_size < MIN_IDX_SIZE:
                return False
            if dat_file_path.stat().st_size < MIN_DAT_SIZE:
                return False
        except OSError:
            return False

        return True

    def reload_index(self):
        """Reload the spatial index to clear internal state and prevent performance degradation."""
        if self.spatial_idx is not None:
            # Close the existing index
            try:
                del self.spatial_idx
            except:
                pass

            # Reload it
            from rtree import index
            self.spatial_idx = index.Index(str(self.idx_file))

    def query_bbox(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> List[dict]:
        """
        Query ways within bounding box.

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude

        Returns:
            List of way dictionaries with metadata
        """
        if self.spatial_idx is None:
            self.load_index(silent=self.silent)

        try:
            # Query spatial index (note: lon/lat order for x/y)
            bbox = (min_lon, min_lat, max_lon, max_lat)
            # Use intersection directly without converting to list first - more memory efficient
            way_ids = self.spatial_idx.intersection(bbox, objects=False)

            # Return way metadata
            ways = []
            for way_id in way_ids:
                if way_id in self.way_metadata and way_id != "_index_metadata":
                    way_data = self.way_metadata[way_id]
                    # Validate way data before returning
                    if (
                        way_data.get("bbox")
                        and way_data.get("nodes")
                        and way_data.get("tags")
                        and len(way_data["nodes"]) >= 2
                    ):
                        ways.append({"id": way_id, **way_data})

            return ways

        except Exception as e:
            print(f"Error querying spatial index: {e}")
            return []

    def query_bbox_batched(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, batch_size: int = 50000
    ):
        """
        Query ways within bounding box, yielding results in batches.

        This is memory-efficient for large regions as it processes ways in chunks
        rather than loading all results into memory at once. Uses efficient batch
        loading from the lazy-loading metadata dictionary.

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude
            batch_size: Number of ways to yield per batch (default 50,000)

        Yields:
            Batches of way dictionaries with metadata
        """
        if self.spatial_idx is None:
            self.load_index(silent=self.silent)

        try:
            # Query spatial index (note: lon/lat order for x/y)
            bbox = (min_lon, min_lat, max_lon, max_lat)
            way_ids = self.spatial_idx.intersection(bbox, objects=False)

            # Collect way IDs in batches for efficient loading
            way_id_batch = []
            for way_id in way_ids:
                if way_id in self.way_metadata and way_id != "_index_metadata":
                    way_id_batch.append(way_id)

                    # When we have a batch, load all at once
                    if len(way_id_batch) >= batch_size:
                        # Use batch loading for efficiency
                        if hasattr(self.way_metadata, 'get_batch'):
                            way_data_map = self.way_metadata.get_batch(way_id_batch)
                        else:
                            # Fallback for non-lazy dict
                            way_data_map = {wid: self.way_metadata[wid]
                                           for wid in way_id_batch}

                        # Validate and yield batch
                        batch = []
                        for wid, way_data in way_data_map.items():
                            if (
                                way_data.get("bbox")
                                and way_data.get("nodes")
                                and way_data.get("tags")
                                and len(way_data["nodes"]) >= 2
                            ):
                                batch.append({"id": wid, **way_data})

                        if batch:
                            yield batch

                        way_id_batch = []  # Clear for next batch

            # Process any remaining IDs
            if way_id_batch:
                if hasattr(self.way_metadata, 'get_batch'):
                    way_data_map = self.way_metadata.get_batch(way_id_batch)
                else:
                    way_data_map = {wid: self.way_metadata[wid]
                                   for wid in way_id_batch}

                batch = []
                for wid, way_data in way_data_map.items():
                    if (
                        way_data.get("bbox")
                        and way_data.get("nodes")
                        and way_data.get("tags")
                        and len(way_data["nodes"]) >= 2
                    ):
                        batch.append({"id": wid, **way_data})

                if batch:
                    yield batch

        except Exception as e:
            print(f"Error querying spatial index in batches: {e}")
            return


class LocationIndex:
    """
    Persistent location index for OSM nodes.

    This class manages a persistent index of node locations for fast coordinate lookups.
    """

    def __init__(self, osm_file_path: str, index_dir: str = "data/osm_indexes"):
        """
        Initialize the location index.

        Args:
            osm_file_path: Path to the OSM file
            index_dir: Directory where index files are stored
        """
        self.osm_file_path = osm_file_path
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.index_file = self.index_dir / f"{Path(osm_file_path).stem}_locations.idx"

    def create_index(self):
        """
        Create node location index from OSM file.

        This uses osmium's node location storage to create a memory-mapped
        file for fast access to node coordinates.

        Raises:
            ImportError: If osmium library not available
        """
        try:
            import osmium
        except ImportError:
            raise ImportError("osmium library required for location indexing")

        print(
            "Creating location index (this may take several minutes for large files)..."
        )

        # Use osmium's node location storage
        # This creates a memory-mapped file for fast access
        handler = osmium.SimpleHandler()

        # Apply with location storage
        handler.apply_file(
            self.osm_file_path,
            locations=True,
            idx="sparse_file_array," + str(self.index_file),
        )

        print(f"Location index created: {self.index_file}")

    def exists(self) -> bool:
        """
        Check if index exists.

        Returns:
            True if index file exists
        """
        return self.index_file.exists()
