#!/usr/bin/env python3
"""
Batch Progress Tracker

Tracks progress when processing multiple locations (states/countries) in batch mode.
Allows resuming a batch analysis from where it was interrupted, skipping already-completed locations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


class BatchProgressTracker:
    """Tracks progress for multi-location batch processing."""

    def __init__(self, progress_file: Path = None):
        """
        Initialize batch progress tracker.

        Args:
            progress_file: Path to progress file (default: checkpoints/batch_progress.json)
        """
        if progress_file is None:
            progress_file = Path("checkpoints") / "batch_progress.json"

        self.progress_file = Path(progress_file)
        # Don't create directory until actually needed (lazy creation)

        self._progress_data = {
            "batch_id": None,
            "locations": [],
            "completed": [],
            "failed": [],
            "current": None,
            "started_at": None,
            "last_updated": None,
            "metadata": {}
        }

    def start_batch(
        self,
        locations: List[str],
        batch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start a new batch or load existing progress.

        Args:
            locations: List of location names to process
            batch_id: Optional batch identifier (auto-generated if None)
            metadata: Optional metadata (surface_filter, unit_system, etc.)
        """
        # Check if there's existing progress for this batch
        existing_progress = self.load_progress()

        if existing_progress and set(existing_progress.get("locations", [])) == set(locations):
            # Resume existing batch
            self._progress_data = existing_progress
            print(f"\n✓ Resuming existing batch: {self._progress_data['batch_id']}")
            print(f"  Started: {self._progress_data['started_at']}")
            print(f"  Completed: {len(self._progress_data['completed'])}/{len(locations)}")
            if self._progress_data['failed']:
                print(f"  Failed: {len(self._progress_data['failed'])}")
        else:
            # Start new batch
            if batch_id is None:
                batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self._progress_data = {
                "batch_id": batch_id,
                "locations": locations,
                "completed": [],
                "failed": [],
                "current": None,
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            self.save_progress()
            print(f"\n✓ Starting new batch: {batch_id}")
            print(f"  Locations: {len(locations)}")

    def is_completed(self, location: str) -> bool:
        """
        Check if a location has been completed.

        Args:
            location: Location name to check

        Returns:
            True if location has been completed
        """
        return location in self._progress_data.get("completed", [])

    def is_failed(self, location: str) -> bool:
        """
        Check if a location has failed.

        Args:
            location: Location name to check

        Returns:
            True if location has failed
        """
        return location in self._progress_data.get("failed", [])

    def mark_started(self, location: str) -> None:
        """
        Mark a location as currently being processed.

        Args:
            location: Location name
        """
        self._progress_data["current"] = location
        self._progress_data["last_updated"] = datetime.now().isoformat()
        self.save_progress()

    def mark_completed(self, location: str) -> None:
        """
        Mark a location as completed.

        Args:
            location: Location name
        """
        if location not in self._progress_data["completed"]:
            self._progress_data["completed"].append(location)

        # Remove from failed if it was there (retry succeeded)
        if location in self._progress_data.get("failed", []):
            self._progress_data["failed"].remove(location)

        self._progress_data["current"] = None
        self._progress_data["last_updated"] = datetime.now().isoformat()
        self.save_progress()

    def mark_failed(self, location: str, error: Optional[str] = None) -> None:
        """
        Mark a location as failed.

        Args:
            location: Location name
            error: Optional error message
        """
        if location not in self._progress_data["failed"]:
            self._progress_data["failed"].append(location)

        self._progress_data["current"] = None
        self._progress_data["last_updated"] = datetime.now().isoformat()

        # Store error details in metadata
        if error:
            if "errors" not in self._progress_data:
                self._progress_data["errors"] = {}
            self._progress_data["errors"][location] = error

        self.save_progress()

    def get_pending_locations(self) -> List[str]:
        """
        Get list of locations that haven't been completed yet.

        Returns:
            List of pending location names
        """
        all_locations = self._progress_data.get("locations", [])
        completed = set(self._progress_data.get("completed", []))

        return [loc for loc in all_locations if loc not in completed]

    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get summary of batch progress.

        Returns:
            Dictionary with progress statistics
        """
        total = len(self._progress_data.get("locations", []))
        completed = len(self._progress_data.get("completed", []))
        failed = len(self._progress_data.get("failed", []))
        pending = total - completed

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "percent_complete": (completed / total * 100) if total > 0 else 0,
            "locations": {
                "all": self._progress_data.get("locations", []),
                "completed": self._progress_data.get("completed", []),
                "failed": self._progress_data.get("failed", []),
                "pending": self.get_pending_locations()
            }
        }

    def save_progress(self) -> None:
        """Save progress to JSON file."""
        try:
            # Create directory only when actually saving (lazy creation)
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self._progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save batch progress: {e}")

    def load_progress(self) -> Optional[Dict[str, Any]]:
        """
        Load progress from JSON file.

        Returns:
            Progress data dict or None if file doesn't exist
        """
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load batch progress: {e}")
            return None

    def clear_progress(self) -> None:
        """Delete progress file (start fresh)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            print(f"✓ Cleared batch progress file: {self.progress_file}")

    def print_progress(self) -> None:
        """Print formatted progress summary."""
        from climb_analyzer.utils.formatting import (
            print_banner,
            print_success,
            print_error,
            print_separator,
        )

        summary = self.get_progress_summary()

        print_banner("Batch Progress", spacing_before=1)

        print(f"Batch ID: {self._progress_data.get('batch_id', 'N/A')}")
        print(f"Started:  {self._progress_data.get('started_at', 'N/A')}")
        print(f"Updated:  {self._progress_data.get('last_updated', 'N/A')}")
        print()
        print(f"Total Locations:     {summary['total']}")
        print_success(f"Completed:         {summary['completed']}")
        if summary['failed'] > 0:
            print_error(f"Failed:            {summary['failed']}")
        print(f"Pending:             {summary['pending']}")
        print(f"Progress:            {summary['percent_complete']:.1f}%")

        if self._progress_data.get('current'):
            print(f"\nCurrently processing: {self._progress_data['current']}")

        if summary['completed'] > 0:
            print(f"\nCompleted locations:")
            for loc in summary['locations']['completed']:
                print_success(loc, indent=2)

        if summary['failed'] > 0:
            print(f"\nFailed locations:")
            for loc in summary['locations']['failed']:
                error = self._progress_data.get('errors', {}).get(loc, 'Unknown error')
                print_error(f"{loc}: {error}", indent=2)

        if summary['pending'] > 0:
            print(f"\nPending locations:")
            for loc in summary['locations']['pending'][:5]:
                print(f"  {loc}")
            if summary['pending'] > 5:
                print(f"  ... and {summary['pending'] - 5} more")

        print()
        print_separator()


# Convenience functions for use in climb_analyzer.py

def create_batch_tracker() -> BatchProgressTracker:
    """Create and return a batch progress tracker."""
    return BatchProgressTracker()


def should_process_location(tracker: BatchProgressTracker, location: str) -> bool:
    """
    Check if a location should be processed.

    Args:
        tracker: BatchProgressTracker instance
        location: Location name

    Returns:
        True if location should be processed (not completed)
    """
    return not tracker.is_completed(location)
