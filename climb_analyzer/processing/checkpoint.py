"""
Checkpoint and persistence management for climb analysis.

This module provides classes and functions for managing analysis checkpoints,
allowing long-running analyses to be paused and resumed.
"""

import json
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# Custom unpickler to handle module remapping for classes serialized from __main__
class _CheckpointUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps classes serialized from __main__ to correct modules."""

    def find_class(self, module, name):
        if module == "__main__":
            if name in ("ClimbMetrics", "ClimbIdentifier", "SimpleClimbNode", "ElevationProfile"):
                module = "climb_analyzer.engine"
            elif name == "ErrorLogEntry":
                module = "climb_analyzer.data.elevation"
        return super().find_class(module, name)


def _safe_pickle_load(file_handle):
    """Load pickle data using custom unpickler that handles __main__ class remapping."""
    return _CheckpointUnpickler(file_handle).load()


try:
    from utils.config_loader import (
        CHECKPOINT_INTERVAL_MIN,
        CHECKPOINT_MILESTONES_PERC,
    )
except ImportError:
    CHECKPOINT_INTERVAL_MIN = 15.0
    CHECKPOINT_MILESTONES_PERC = [25, 50, 75, 100]

try:
    from utils.data_paths import CHECKPOINT_DIR
except ImportError:
    CHECKPOINT_DIR = Path("data/checkpoint_data")


@dataclass
class CheckpointConfig:
    """
    Global configuration for checkpoint saving.

    Attributes:
        time_interval_minutes: Time interval in minutes between checkpoints
        progress_milestones: List of progress percentages to trigger checkpoints
        save_at_completion: Whether to save a checkpoint at 100% completion
    """

    time_interval_minutes: float = CHECKPOINT_INTERVAL_MIN
    progress_milestones: list = None
    save_at_completion: bool = True

    def __post_init__(self):
        """Initialize progress milestones if not provided."""
        if self.progress_milestones is None:
            self.progress_milestones = CHECKPOINT_MILESTONES_PERC


class SmartCheckpointer:
    """
    Intelligent checkpoint manager with time-based and milestone-based saving.

    This class determines when to save checkpoints based on both elapsed time
    and progress milestones.
    """

    def __init__(self, total_items: int, operation_name: str = "Processing"):
        """
        Initialize the checkpointer.

        Args:
            total_items: Total number of items to process
            operation_name: Name of the operation being checkpointed
        """
        self.total_items = total_items
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_progress = 0
        self.milestone_index = 0

        # Pre-calculate next milestone
        self.checkpoint_config = CheckpointConfig()
        self.next_milestone = (
            self.checkpoint_config.progress_milestones[0]
            if self.checkpoint_config.progress_milestones
            else 100
        )

    def should_checkpoint(self, current_item: int, force_check: bool = False) -> bool:
        """
        Determine if we should save a checkpoint now.

        Args:
            current_item: Current item being processed (0-based)
            force_check: Force a time check even if not at a milestone

        Returns:
            True if checkpoint should be saved
        """
        current_time = time.time()
        current_progress = ((current_item + 1) / self.total_items) * 100

        # Time-based checkpoint check
        time_elapsed = (current_time - self.last_checkpoint_time) / 60.0  # minutes
        time_trigger = time_elapsed >= self.checkpoint_config.time_interval_minutes

        # Progress milestone check
        milestone_trigger = current_progress >= self.next_milestone

        # Force completion checkpoint
        completion_trigger = (
            current_item + 1 >= self.total_items
            and self.checkpoint_config.save_at_completion
        )

        # Decide whether to checkpoint
        should_save = time_trigger or milestone_trigger or completion_trigger

        if should_save:
            self._update_after_checkpoint(current_time, current_progress)

        return should_save

    def _update_after_checkpoint(self, current_time: float, current_progress: float):
        """
        Update internal state after checkpoint.

        Args:
            current_time: Current timestamp
            current_progress: Current progress percentage
        """
        self.last_checkpoint_time = current_time
        self.last_checkpoint_progress = current_progress

        # Move to next milestone
        if (
            current_progress >= self.next_milestone
            and self.milestone_index < len(self.checkpoint_config.progress_milestones) - 1
        ):
            self.milestone_index += 1
            self.next_milestone = self.checkpoint_config.progress_milestones[
                self.milestone_index
            ]

    def get_checkpoint_info(self, current_item: int) -> Dict[str, Any]:
        """
        Get info about current checkpoint status.

        Args:
            current_item: Current item being processed (0-based)

        Returns:
            Dictionary with checkpoint status information
        """
        current_progress = ((current_item + 1) / self.total_items) * 100
        time_since_last = (time.time() - self.last_checkpoint_time) / 60.0

        return {
            "progress_pct": current_progress,
            "next_milestone": self.next_milestone,
            "time_since_last_min": time_since_last,
            "time_until_next_min": max(
                0, self.checkpoint_config.time_interval_minutes - time_since_last
            ),
        }


class ChunkPersistenceManager:
    """
    Manages persistent storage of chunk processing progress and data.

    This class handles saving and loading of analysis checkpoints to disk,
    allowing long-running analyses to be resumed.
    """

    def __init__(self, analysis_id: str):
        """
        Initialize persistence manager with analysis ID.

        Args:
            analysis_id: Unique identifier for this analysis session
        """
        self.analysis_id = analysis_id
        self.base_dir = CHECKPOINT_DIR
        self.analysis_dir = self.base_dir / analysis_id
        self.chunk_dir = self.analysis_dir / "chunks"
        self.elevation_dir = self.analysis_dir / "elevations"
        self.progress_file = self.analysis_dir / "progress.pkl"
        self.metadata_file = self.analysis_dir / "metadata.pkl"
        self.elevation_progress_file = self.analysis_dir / "elevation_progress.pkl"

        # Create directories if they don't exist (with permissive mode for Docker compatibility)
        self.chunk_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        self.elevation_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

    def save_completed_elevations(self, node_elevations):
        """
        Save completed elevation data permanently (separate from checkpoints).

        Args:
            node_elevations: Dictionary mapping node coordinates to elevations,
                           OR Path to a shelve database file
        """
        elevation_complete_file = self.analysis_dir / "elevation_complete.pkl"
        temp_file = self.analysis_dir / "elevation_complete.tmp"

        try:
            # Handle both dict and Path (SQLite database)
            if isinstance(node_elevations, (Path, str)):
                # It's a path to a SQLite database - query directly
                import sqlite3
                conn = sqlite3.connect(str(node_elevations))
                cursor = conn.execute("SELECT COUNT(*) FROM elevations")
                total_nodes = cursor.fetchone()[0]
                conn.close()

                elevation_data = {
                    "elevation_db_path": str(node_elevations),
                    "timestamp": time.time(),
                    "total_nodes": total_nodes,
                }
            else:
                # It's a dictionary - store the dict
                elevation_data = {
                    "node_elevations": node_elevations,
                    "timestamp": time.time(),
                    "total_nodes": len(node_elevations),
                }

            with open(temp_file, "wb") as f:
                pickle.dump(elevation_data, f)
            temp_file.rename(elevation_complete_file)
            print(f"Saved completed elevation data for {elevation_data['total_nodes']:,} nodes")

        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            print(f"Error saving completed elevations: {e}")

    def load_completed_elevations(self) -> Optional[Dict]:
        """
        Load completed elevation data.

        Returns:
            ElevationDatabase object or dictionary of node elevations, or None if not found
        """
        elevation_complete_file = self.analysis_dir / "elevation_complete.pkl"

        if not elevation_complete_file.exists():
            return None

        try:
            with open(elevation_complete_file, "rb") as f:
                elevation_data = _safe_pickle_load(f)

            # Check if this is a path to SQLite database (new format)
            if "elevation_db_path" in elevation_data:
                from pathlib import Path
                import sqlite3

                db_path = Path(elevation_data["elevation_db_path"])
                if db_path.exists():
                    # Return a wrapper that provides dict-like access to the database
                    class ElevationDatabaseWrapper:
                        """Wrapper for SQLite elevation database that supports dict-like access."""
                        def __init__(self, db_path):
                            self.db_path = db_path
                            self.conn = sqlite3.connect(str(db_path))

                        def get(self, key, default=None):
                            """Get elevation for a node_id or coord key."""
                            try:
                                cursor = self.conn.execute(
                                    "SELECT elevation FROM elevations WHERE node_id = ?",
                                    (key,)
                                )
                                result = cursor.fetchone()
                                return result[0] if result else default
                            except Exception:
                                return default

                        def __getitem__(self, key):
                            """Get elevation using dict-like access db[key]."""
                            cursor = self.conn.execute(
                                "SELECT elevation FROM elevations WHERE node_id = ?",
                                (key,)
                            )
                            result = cursor.fetchone()
                            if result is None:
                                raise KeyError(key)
                            return result[0]

                        def __contains__(self, key):
                            """Check if key exists - supports 'if key in db' syntax."""
                            cursor = self.conn.execute(
                                "SELECT 1 FROM elevations WHERE node_id = ? LIMIT 1",
                                (key,)
                            )
                            return cursor.fetchone() is not None

                        def __len__(self):
                            cursor = self.conn.execute("SELECT COUNT(*) FROM elevations")
                            return cursor.fetchone()[0]

                        def keys(self):
                            """Iterate over all keys."""
                            cursor = self.conn.execute("SELECT node_id FROM elevations")
                            return (row[0] for row in cursor)

                        def close(self):
                            if hasattr(self, 'conn'):
                                self.conn.close()

                        def __del__(self):
                            self.close()

                    return ElevationDatabaseWrapper(db_path)
                else:
                    print(f"Warning: Elevation database not found at {db_path}")
                    return None
            else:
                # Old format - return the dict directly
                return elevation_data.get("node_elevations", {})

        except Exception as e:
            print(f"Error loading completed elevations: {e}")
            import traceback
            traceback.print_exc()
            return None

    def has_completed_elevations(self) -> bool:
        """
        Check if completed elevation data exists.

        Returns:
            True if elevation data is available
        """
        elevation_complete_file = self.analysis_dir / "elevation_complete.pkl"
        return elevation_complete_file.exists()

    def load_progress(self) -> Tuple[List[int], int, Dict]:
        """
        Load progress and metadata from disk.

        Returns:
            Tuple of (processed_chunks, total_chunks, metadata)
        """
        processed_chunks = []
        total_chunks = 0
        metadata = {}

        try:
            if self.progress_file.exists():
                with open(self.progress_file, "rb") as f:
                    progress_data = _safe_pickle_load(f)
                processed_chunks = progress_data.get("processed_chunks", [])
                total_chunks = progress_data.get("total_chunks", 0)

            if self.metadata_file.exists():
                with open(self.metadata_file, "rb") as f:
                    metadata = _safe_pickle_load(f)

        except Exception as e:
            print(f"Error loading progress: {e}")

        return processed_chunks, total_chunks, metadata

    def save_chunk(
        self,
        chunk_index: int,
        chunk_data: List[Dict],
        chunk_info: Tuple[float, float, float],
    ):
        """
        Save chunk data to disk.

        Args:
            chunk_index: Index of the chunk
            chunk_data: List of road segments in the chunk
            chunk_info: Tuple of (center_lat, center_lon, chunk_size_km)
        """
        chunk_file = self.chunk_dir / f"chunk_{chunk_index:04d}.pkl"
        temp_file = self.chunk_dir / f"chunk_{chunk_index:04d}.tmp"

        try:
            with open(temp_file, "wb") as f:
                pickle.dump(
                    {
                        "chunk_index": chunk_index,
                        "chunk_data": chunk_data,
                        "chunk_info": chunk_info,
                        "timestamp": time.time(),
                    },
                    f,
                )
            temp_file.rename(chunk_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            print(f"Error saving chunk {chunk_index}: {e}")

    def load_chunk(self, chunk_index: int) -> Optional[List[Dict]]:
        """
        Load chunk data from disk.

        Args:
            chunk_index: Index of the chunk to load

        Returns:
            List of road segments or None if chunk doesn't exist
        """
        chunk_file = self.chunk_dir / f"chunk_{chunk_index:04d}.pkl"

        if not chunk_file.exists():
            return None

        try:
            with open(chunk_file, "rb") as f:
                chunk_data = _safe_pickle_load(f)
            return chunk_data.get("chunk_data", [])
        except Exception as e:
            print(f"Error loading chunk {chunk_index}: {e}")
            return None

    def save_progress(
        self, processed_chunks: List[int], total_chunks: int, metadata: Dict
    ):
        """
        Save progress and metadata to disk.

        Args:
            processed_chunks: List of completed chunk indices
            total_chunks: Total number of chunks
            metadata: Dictionary of analysis metadata
        """
        temp_progress = self.analysis_dir / "progress.tmp"
        temp_metadata = self.analysis_dir / "metadata.tmp"

        try:
            progress_data = {
                "processed_chunks": processed_chunks,
                "total_chunks": total_chunks,
                "timestamp": time.time(),
            }

            with open(temp_progress, "wb") as f:
                pickle.dump(progress_data, f)
            temp_progress.rename(self.progress_file)

            with open(temp_metadata, "wb") as f:
                pickle.dump(metadata, f)
            temp_metadata.rename(self.metadata_file)

        except Exception as e:
            for temp_file in [temp_progress, temp_metadata]:
                if temp_file.exists():
                    temp_file.unlink()
            print(f"Error saving progress: {e}")

    def cleanup(self):
        """Remove all checkpoint files for this analysis."""
        try:
            if self.analysis_dir.exists():
                shutil.rmtree(self.analysis_dir)
                print(f"Cleaned up checkpoint files for analysis: {self.analysis_id}")
        except Exception as e:
            print(f"Error cleaning up files: {e}")

    def save_elevation_progress(self, coordinate_mapping: Dict, batch_info: Dict):
        """
        Save elevation fetching progress.

        Args:
            coordinate_mapping: Dictionary mapping coordinates to elevations
            batch_info: Dictionary with batch processing information
        """
        temp_file = self.analysis_dir / "elevation_progress.tmp"

        try:
            elevation_data = {
                "coordinate_mapping": coordinate_mapping,
                "batch_info": batch_info,
                "timestamp": time.time(),
            }

            with open(temp_file, "wb") as f:
                pickle.dump(elevation_data, f)
            temp_file.rename(self.elevation_progress_file)

        except Exception as e:
            print(f"Error saving elevation progress: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def load_elevation_progress(self) -> Tuple[Dict, Dict]:
        """
        Load elevation progress from disk.

        Returns:
            Tuple of (coordinate_mapping, batch_info)
        """
        if not self.elevation_progress_file.exists():
            return {}, {}

        try:
            with open(self.elevation_progress_file, "rb") as f:
                elevation_data = _safe_pickle_load(f)

            coord_mapping = elevation_data.get("coordinate_mapping", {})
            batch_info = elevation_data.get("batch_info", {})

            return coord_mapping, batch_info
        except Exception as e:
            print(f"Error loading elevation progress: {e}")
            return {}, {}

    def clear_elevation_progress(self):
        """Clear elevation progress after successful completion."""
        try:
            if self.elevation_progress_file.exists():
                self.elevation_progress_file.unlink()
        except Exception as e:
            print(f"Error clearing elevation progress: {e}")

    def save_deduplication_progress(self, dedupe_data: Dict):
        """
        Save deduplication progress.

        Args:
            dedupe_data: Dictionary with deduplication state
        """
        dedupe_file = self.analysis_dir / "deduplication_progress.pkl"
        temp_file = self.analysis_dir / "deduplication_progress.tmp"

        try:
            with open(temp_file, "wb") as f:
                pickle.dump(dedupe_data, f)
            temp_file.rename(dedupe_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            print(f"Error saving deduplication progress: {e}")

    def load_deduplication_progress(self) -> Dict:
        """
        Load deduplication progress from disk.

        Returns:
            Dictionary with deduplication state or empty dict if not found
        """
        dedupe_file = self.analysis_dir / "deduplication_progress.pkl"

        if not dedupe_file.exists():
            return {}

        try:
            with open(dedupe_file, "rb") as f:
                return _safe_pickle_load(f)
        except Exception as e:
            print(f"Error loading deduplication progress: {e}")
            return {}

    def save_boundary_merge_progress(self, progress_data: Dict):
        """
        Save boundary merge progress.

        Args:
            progress_data: Dictionary with boundary merge state
        """
        boundary_merge_file = self.analysis_dir / "boundary_merge_progress.pkl"
        temp_file = self.analysis_dir / "boundary_merge_progress.tmp"

        try:
            with open(temp_file, "wb") as f:
                pickle.dump(progress_data, f)
            temp_file.rename(boundary_merge_file)
            # Confirm checkpoint saved successfully
            streets_done = progress_data.get('streets_processed', 0)
            streets_total = progress_data.get('total_streets', 0)
            print(f"✓ Boundary merge checkpoint saved ({streets_done}/{streets_total} streets)")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            print(f"Error saving boundary merge progress: {e}")

    def load_boundary_merge_progress(self) -> Dict:
        """
        Load boundary merge progress from disk.

        Returns:
            Dictionary with boundary merge state or empty dict if not found
        """
        boundary_merge_file = self.analysis_dir / "boundary_merge_progress.pkl"

        if not boundary_merge_file.exists():
            return {}

        try:
            with open(boundary_merge_file, "rb") as f:
                return _safe_pickle_load(f)
        except Exception as e:
            print(f"Error loading boundary merge progress: {e}")
            return {}

    def clear_boundary_merge_progress(self):
        """Clear boundary merge progress after successful completion."""
        boundary_merge_file = self.analysis_dir / "boundary_merge_progress.pkl"
        try:
            if boundary_merge_file.exists():
                boundary_merge_file.unlink()
        except Exception as e:
            print(f"Error clearing boundary merge progress: {e}")

    # ==================== JSONL Streaming Checkpoint Methods ====================
    # These methods support memory-efficient processing of large regions (e.g., California)
    # by streaming data to/from disk in JSONL format

    def init_filtered_ways_checkpoint(self) -> Path:
        """
        Initialize a new filtered ways checkpoint file.

        Returns:
            Path to the checkpoint file
        """
        checkpoint_file = self.analysis_dir / "filtered_ways.jsonl"
        # Clear any existing file
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        return checkpoint_file

    def append_filtered_ways_batch(self, ways: List[Dict], checkpoint_file: Path):
        """
        Append a batch of filtered ways to the checkpoint file.

        Args:
            ways: List of way dictionaries to append
            checkpoint_file: Path to the checkpoint file
        """
        try:
            with open(checkpoint_file, 'a') as f:
                for way in ways:
                    # Convert way object to dictionary if needed
                    if hasattr(way, '__dict__'):
                        way_dict = self._way_to_dict(way)
                    else:
                        way_dict = way
                    f.write(json.dumps(way_dict) + '\n')
        except Exception as e:
            print(f"[CHECKPOINT ERROR] Failed to append ways batch: {e}")
            import traceback
            traceback.print_exc()

    def _way_to_dict(self, way) -> Dict:
        """
        Convert a SimpleWay object to a dictionary for serialization.

        Args:
            way: SimpleWay object

        Returns:
            Dictionary representation of the way
        """
        return {
            'id': way.id,
            'nodes': [
                {
                    'lat': n.lat,
                    'lon': n.lon,
                    'id': n.ref if hasattr(n, 'ref') else (n.id if hasattr(n, 'id') else None)
                }
                for n in way.nodes
            ],
            'tags': way.tags if hasattr(way, 'tags') else {}
        }

    def _dict_to_way(self, way_dict: Dict):
        """
        Convert a dictionary back to a SimpleWay-like object.

        Args:
            way_dict: Dictionary representation of a way

        Returns:
            SimpleWay-like object
        """
        from types import SimpleNamespace

        # Convert nodes list to objects, preserving OSM node IDs for proper merging
        nodes = [
            SimpleNamespace(
                lat=n['lat'],
                lon=n['lon'],
                id=n.get('id'),     # Node ID for connection checking
                ref=n.get('id')     # Some code checks 'ref', some checks 'id'
            )
            for n in way_dict['nodes']
        ]

        # Create way object
        way = SimpleNamespace(
            id=way_dict['id'],
            nodes=nodes,
            tags=way_dict.get('tags', {})
        )
        return way

    def read_filtered_ways_batched(
        self, checkpoint_file: Path, batch_size: int = 10000
    ) -> Iterator[List]:
        """
        Read filtered ways from checkpoint file in batches.

        Args:
            checkpoint_file: Path to the checkpoint file
            batch_size: Number of ways to yield per batch

        Yields:
            Batches of way objects
        """
        if not checkpoint_file.exists():
            print(f"[CHECKPOINT] No checkpoint file found at {checkpoint_file}")
            return

        try:
            batch = []
            with open(checkpoint_file, 'r') as f:
                for line in f:
                    if line.strip():
                        way_dict = json.loads(line)
                        way = self._dict_to_way(way_dict)
                        batch.append(way)

                        if len(batch) >= batch_size:
                            yield batch
                            batch = []

            # Yield remaining ways
            if batch:
                yield batch

        except Exception as e:
            print(f"[CHECKPOINT ERROR] Failed to read ways from checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def count_filtered_ways(self, checkpoint_file: Path) -> int:
        """
        Count the number of ways in the checkpoint file.

        Args:
            checkpoint_file: Path to the checkpoint file

        Returns:
            Number of ways in the file
        """
        if not checkpoint_file.exists():
            return 0

        try:
            with open(checkpoint_file, 'r') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            print(f"[CHECKPOINT ERROR] Failed to count ways: {e}")
            return 0

    def has_filtered_ways_checkpoint(self) -> bool:
        """
        Check if a filtered ways checkpoint exists.

        Returns:
            True if checkpoint file exists and is not empty
        """
        checkpoint_file = self.analysis_dir / "filtered_ways.jsonl"
        if not checkpoint_file.exists():
            return False
        return checkpoint_file.stat().st_size > 0

    # ==================== Segment Streaming Methods ====================

    def stream_segments_to_checkpoint(self, segments: Iterator[Dict], checkpoint_name: str = "segments") -> Path:
        """
        Stream segments to a checkpoint file.

        Args:
            segments: Iterator of segment dictionaries
            checkpoint_name: Name for the checkpoint file (without extension)

        Returns:
            Path to the checkpoint file
        """
        checkpoint_file = self.analysis_dir / f"{checkpoint_name}.jsonl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        count = 0
        with open(checkpoint_file, 'w') as f:
            for segment in segments:
                f.write(json.dumps(segment) + '\n')
                count += 1

        return checkpoint_file

    def read_segments_batched(self, checkpoint_name: str = "segments", batch_size: int = 10000) -> Iterator[List[Dict]]:
        """
        Read segments from checkpoint file in batches.

        Args:
            checkpoint_name: Name of the checkpoint file (without extension)
            batch_size: Number of segments to yield per batch

        Yields:
            Batches of segment dictionaries
        """
        checkpoint_file = self.analysis_dir / f"{checkpoint_name}.jsonl"
        if not checkpoint_file.exists():
            return

        batch = []
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    segment = json.loads(line)
                    batch.append(segment)

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

        if batch:
            yield batch

    def count_segments(self, checkpoint_name: str = "segments") -> int:
        """
        Count segments in checkpoint file.

        Args:
            checkpoint_name: Name of the checkpoint file (without extension)

        Returns:
            Number of segments
        """
        checkpoint_file = self.analysis_dir / f"{checkpoint_name}.jsonl"
        if not checkpoint_file.exists():
            return 0

        with open(checkpoint_file, 'r') as f:
            return sum(1 for line in f if line.strip())


def configure_checkpoints(
    time_minutes: float = 15.0,
    milestones: list = None,
    save_at_completion: bool = True
):
    """
    Configure global checkpoint settings.

    Args:
        time_minutes: Time interval in minutes between checkpoints
        milestones: List of progress percentages to trigger checkpoints
        save_at_completion: Whether to save a checkpoint at completion

    Note:
        This function modifies the global CHECKPOINT_CONFIG object.
        In the refactored version, this would be better handled through
        dependency injection or configuration objects.
    """
    from climb_analyzer.utils.formatting import print_dim

    print_dim(
        f"Checkpoint config: every {time_minutes}min, milestones at {milestones or [25, 50, 75, 100]}%"
    )
