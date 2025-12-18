"""
Configuration constants for the climb analyzer.

This module contains global configuration values used throughout the application.
"""

import os

# Parallel street merging configuration
MERGE_PARALLEL_ENABLED = True  # Enable parallel processing for street merging
MERGE_MAX_WORKERS = min(16, os.cpu_count() or 1)  # Max workers for parallel merge
MERGE_BATCH_SIZE = 500  # Streets per batch for parallel processing

# Deduplication configuration
# IMPORTANT: Spatial deduplication is VERY slow (O(n×k)) and often unnecessary
# because street merging already handles overlapping segments.
# Only enable if you have true duplicates that street merging won't catch.
SKIP_SPATIAL_DEDUPLICATION = True  # Skip slow spatial dedup (recommended for large datasets)
SPATIAL_DEDUP_THRESHOLD = 10000  # Only use spatial dedup if < this many segments

# Way ID deduplication configuration
# Even the "fast" way_id dedup can cause memory issues with 1.5M segments
# The merger handles duplicates anyway, so this is optional
SKIP_ALL_DEDUPLICATION = True  # Skip ALL dedup phases (recommended for large datasets)
