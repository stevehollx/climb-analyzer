"""
Graceful shutdown handler for climb analysis operations.

This module provides signal handling for graceful shutdown with checkpoint saving.
"""

import atexit
import os
import signal
import threading
from typing import Dict, Optional


class GracefulKiller:
    """
    Handles graceful shutdown with checkpoint saving.

    This class intercepts SIGINT and SIGTERM signals to allow the application
    to save checkpoints before exiting.
    """

    def __init__(self):
        """Initialize the graceful killer with signal handlers."""
        self.kill_now = False
        self.persistence_manager = None
        self.current_operation = "unknown"
        self.checkpoint_data = {}
        self._lock = threading.Lock()

        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
        atexit.register(self._cleanup)

    def set_persistence_manager(self, persistence_manager):
        """
        Set the persistence manager for checkpoint saving.

        Args:
            persistence_manager: ChunkPersistenceManager instance
        """
        with self._lock:
            self.persistence_manager = persistence_manager

    def set_operation(self, operation: str, checkpoint_data: Optional[Dict] = None):
        """
        Set the current operation and checkpoint data.

        Args:
            operation: Name of the current operation
            checkpoint_data: Dictionary containing checkpoint data
        """
        with self._lock:
            self.current_operation = operation
            if checkpoint_data:
                self.checkpoint_data = checkpoint_data.copy()

    def _exit_gracefully(self, signum, frame):
        """
        Handle shutdown signals gracefully.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        with self._lock:
            if self.kill_now:
                print("\nForced shutdown. Exiting immediately...")
                os._exit(1)  # Force exit without cleanup

            self.kill_now = True
            print(f"\nGraceful shutdown during: {self.current_operation}")

            if self.persistence_manager and self.checkpoint_data:
                try:
                    print("Saving emergency checkpoint...")
                    self._save_emergency_checkpoint()
                    print("Emergency checkpoint saved. Analysis can be resumed.")
                except Exception as e:
                    print(f"Failed to save emergency checkpoint: {e}")

            os._exit(0)

    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint based on current operation."""
        if self.current_operation == "chunk_processing":
            self.persistence_manager.save_progress(
                self.checkpoint_data.get("processed_chunks", []),
                self.checkpoint_data.get("total_chunks", 0),
                self.checkpoint_data.get("metadata", {}),
            )
        elif self.current_operation == "elevation_fetching":
            coord_map = self.checkpoint_data.get("coordinate_mapping", {})
            batch_info = self.checkpoint_data.get("batch_info", {})
            self.persistence_manager.save_elevation_progress(coord_map, batch_info)
        elif self.current_operation == "region_elevation_fetching":
            # State/country mode elevation fetching with batch processing
            # CRITICAL: Sync elevation database to disk before saving checkpoint
            elevation_db = self.checkpoint_data.get("elevation_db")
            # print(f"[DEBUG] Signal handler: elevation_db = {elevation_db is not None}")
            if elevation_db:
                try:
                    db_size = len(elevation_db)
                    # print(f"[DEBUG] Database has {db_size:,} entries before sync")
                    elevation_db.sync()
                    print(f"  ✓ Synced {db_size:,} elevations to disk database")
                except Exception as e:
                    print(f"  ⚠️  Warning: Failed to sync elevation database: {e}")
            else:
                print(f"  ⚠️  WARNING: No elevation_db reference in checkpoint_data!")

            node_elevations = self.checkpoint_data.get("node_elevations", {})
            checkpoint_info = {
                "batch_idx": self.checkpoint_data.get("batch_idx", 0),
                "coords_seen": self.checkpoint_data.get("coords_seen", []),
                "elevations_fetched": self.checkpoint_data.get("elevations_fetched", 0),
                "elevations_failed": self.checkpoint_data.get("elevations_failed", 0),
            }
            self.persistence_manager.save_elevation_progress(node_elevations, checkpoint_info)
        elif self.current_operation == "deduplication":
            self.persistence_manager.save_deduplication_progress(self.checkpoint_data)
        elif self.current_operation == "boundary_merge":
            self.persistence_manager.save_boundary_merge_progress(self.checkpoint_data)

    def _cleanup(self):
        """Cleanup handler called on normal exit."""
        pass
