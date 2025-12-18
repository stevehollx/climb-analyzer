#!/usr/bin/env python3
"""
Global elevation statistics collector.

Provides a singleton pattern for collecting elevation fetch statistics
from all elevation fetcher instances throughout the application.
"""

from typing import Dict, Any, Optional


class ElevationStatsCollector:
    """
    Singleton collector for elevation fetch statistics.

    This allows any part of the application to report and retrieve
    elevation fetch statistics without tight coupling.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.reset()

    def reset(self):
        """Reset all statistics to zero."""
        self.total_coords_requested = 0
        self.total_coords_failed = 0
        self.total_unique_coords = 0
        self.runs_count = 0

        # Per-way failure tracking
        # Structure: {way_id: {'name': str, 'total_coords': int, 'failed_coords': int, 'failed_indices': list}}
        self.way_failures = {}

    def record_fetch(self, total_requested: int, failed: int, unique: int):
        """
        Record statistics from an elevation fetch operation.

        Args:
            total_requested: Total number of coordinates requested
            failed: Number of coordinates that failed to fetch
            unique: Total unique coordinates processed
        """
        self.total_coords_requested += total_requested
        self.total_coords_failed += failed
        self.total_unique_coords += unique
        self.runs_count += 1

    def record_way_failure(self, way_id: str, way_name: str, coord_index: int, total_coords: int):
        """
        Record a coordinate failure for a specific way.

        Args:
            way_id: Unique identifier for the way
            way_name: Display name of the way
            coord_index: Index of the failed coordinate in the way
            total_coords: Total number of coordinates in this way
        """
        if way_id not in self.way_failures:
            self.way_failures[way_id] = {
                'name': way_name,
                'total_coords': total_coords,
                'failed_coords': 0,
                'failed_indices': []
            }
        else:
            # Update total_coords to the maximum seen (handles merged ways)
            self.way_failures[way_id]['total_coords'] = max(
                self.way_failures[way_id]['total_coords'],
                total_coords
            )

        self.way_failures[way_id]['failed_coords'] += 1
        self.way_failures[way_id]['failed_indices'].append(coord_index)

    def get_way_failures(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-way failure statistics.

        Returns:
            Dictionary of way failures with percentages calculated
        """
        result = {}

        for way_id, data in self.way_failures.items():
            total = data['total_coords']
            failed = data['failed_coords']
            failure_pct = (failed / total * 100) if total > 0 else 0

            result[way_id] = {
                'name': data['name'],
                'total_coords': total,
                'failed_coords': failed,
                'failure_percentage': failure_pct,
                'failed_indices': data['failed_indices']
            }

        # Sort by failure count (highest first)
        result = dict(sorted(result.items(), key=lambda x: x[1]['failed_coords'], reverse=True))

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with current statistics
        """
        return {
            'total_coords_requested': self.total_coords_requested,
            'total_coords_failed': self.total_coords_failed,
            'total_unique_coords': self.total_unique_coords,
            'runs_count': self.runs_count,
            'success_rate': self._calculate_success_rate(),
            'failure_rate': self._calculate_failure_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_coords_requested == 0:
            return 100.0

        successful = self.total_coords_requested - self.total_coords_failed
        return (successful / self.total_coords_requested) * 100.0

    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_coords_requested == 0:
            return 0.0

        return (self.total_coords_failed / self.total_coords_requested) * 100.0

    def has_data(self) -> bool:
        """Check if any statistics have been recorded."""
        return self.total_coords_requested > 0

    def get_summary_text(self) -> str:
        """
        Get formatted summary text for display.

        Returns:
            Multi-line summary string
        """
        if not self.has_data():
            return "No elevation fetch statistics available."

        stats = self.get_stats()

        lines = [
            "Elevation Fetch Statistics:",
            f"  Total Coordinates Requested:  {stats['total_coords_requested']:,}",
            f"  Unique Coordinates Processed: {stats['total_unique_coords']:,}",
            f"  Failed Coordinates:           {stats['total_coords_failed']:,}",
            f"  Success Rate:                 {stats['success_rate']:.1f}%",
            f"  Failure Rate:                 {stats['failure_rate']:.1f}%",
        ]

        return '\n'.join(lines)


# Global singleton instance
_stats_collector = ElevationStatsCollector()


def get_stats_collector() -> ElevationStatsCollector:
    """
    Get the global elevation statistics collector instance.

    Returns:
        The singleton ElevationStatsCollector instance
    """
    return _stats_collector


def reset_stats():
    """Reset global statistics."""
    _stats_collector.reset()


def record_elevation_fetch(total_requested: int, failed: int, unique: int):
    """
    Convenience function to record elevation fetch statistics.

    Args:
        total_requested: Total coordinates requested
        failed: Number that failed
        unique: Total unique coordinates
    """
    _stats_collector.record_fetch(total_requested, failed, unique)


def get_elevation_stats() -> Dict[str, Any]:
    """
    Convenience function to get current statistics.

    Returns:
        Dictionary with current statistics
    """
    return _stats_collector.get_stats()


def has_elevation_stats() -> bool:
    """
    Check if any elevation statistics are available.

    Returns:
        True if statistics have been recorded
    """
    return _stats_collector.has_data()


def record_way_failure(way_id: str, way_name: str, coord_index: int, total_coords: int):
    """
    Record a coordinate failure for a specific way.

    Args:
        way_id: Unique identifier for the way
        way_name: Display name of the way
        coord_index: Index of the failed coordinate
        total_coords: Total coordinates in the way
    """
    _stats_collector.record_way_failure(way_id, way_name, coord_index, total_coords)


def get_way_failures() -> Dict[str, Dict[str, Any]]:
    """
    Get per-way failure statistics.

    Returns:
        Dictionary of way failures sorted by failure count
    """
    return _stats_collector.get_way_failures()
