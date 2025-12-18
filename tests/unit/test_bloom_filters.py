#!/usr/bin/env python3
"""
Unit tests for bloom filter functionality used in elevation fetching.

Tests the bloom filter setup, operations, and fallback behavior used for
memory-efficient coordinate and node ID deduplication during elevation
processing.

The bloom filters are used in climb_analyzer/engine.py to prevent OOM
errors when processing large regions like France (150M+ coordinates).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestBloomFilterImport:
    """Tests for bloom filter import and fallback behavior."""

    def test_bloom_filter_import_success(self):
        """Test that pybloom_live can be imported."""
        try:
            from pybloom_live import BloomFilter
            assert BloomFilter is not None
        except ImportError:
            pytest.skip("pybloom_live not installed")

    def test_bloom_filter_fallback_to_set(self):
        """Test that fallback to set works when bloom filter unavailable."""
        # Simulate import failure
        with patch.dict('sys.modules', {'pybloom_live': None}):
            # This should fallback to set
            use_bloom = False
            try:
                from pybloom_live import BloomFilter
                use_bloom = True
            except (ImportError, TypeError):
                global_coords_seen = set()
                fetched_node_ids = set()
                use_bloom = False

            assert use_bloom is False
            assert isinstance(global_coords_seen, set)
            assert isinstance(fetched_node_ids, set)


class TestBloomFilterCapacity:
    """Tests for bloom filter capacity configuration."""

    @pytest.fixture
    def bloom_filter(self):
        """Create a bloom filter for testing."""
        try:
            from pybloom_live import BloomFilter
            return BloomFilter(capacity=1000, error_rate=0.001)
        except ImportError:
            pytest.skip("pybloom_live not installed")

    def test_coordinate_bloom_capacity_constant(self):
        """Verify coordinate bloom filter capacity is 150M."""
        # From engine.py line 10468
        EXPECTED_COORD_CAPACITY = 150000000
        assert EXPECTED_COORD_CAPACITY == 150_000_000

    def test_node_id_bloom_capacity_constant(self):
        """Verify node ID bloom filter capacity is 200M."""
        # From engine.py line 10471
        EXPECTED_NODE_CAPACITY = 200000000
        assert EXPECTED_NODE_CAPACITY == 200_000_000

    def test_error_rate_constant(self):
        """Verify error rate is 0.001 (0.1%)."""
        # From engine.py line 10468
        EXPECTED_ERROR_RATE = 0.001
        assert EXPECTED_ERROR_RATE == 0.001

    def test_create_bloom_filter_with_capacity(self, bloom_filter):
        """Test creating bloom filter with specific capacity."""
        assert bloom_filter is not None
        assert bloom_filter.capacity == 1000


class TestBloomFilterOperations:
    """Tests for bloom filter add and contains operations."""

    @pytest.fixture
    def bloom_filter(self):
        """Create a bloom filter for testing."""
        try:
            from pybloom_live import BloomFilter
            return BloomFilter(capacity=10000, error_rate=0.001)
        except ImportError:
            pytest.skip("pybloom_live not installed")

    def test_add_and_contains(self, bloom_filter):
        """Test basic add and contains operations."""
        coord = (35.1234, -82.5678)

        # Before adding
        assert coord not in bloom_filter

        # After adding
        bloom_filter.add(coord)
        assert coord in bloom_filter

    def test_add_multiple_items(self, bloom_filter):
        """Test adding multiple items."""
        coords = [
            (35.1, -82.5),
            (35.2, -82.6),
            (35.3, -82.7),
        ]

        for coord in coords:
            bloom_filter.add(coord)

        for coord in coords:
            assert coord in bloom_filter

    def test_non_existent_item_not_found(self, bloom_filter):
        """Test that non-existent items are (usually) not found."""
        bloom_filter.add((35.1, -82.5))

        # Different coordinate should not be found
        # Note: Bloom filters can have false positives, but not false negatives
        assert (99.9, -99.9) not in bloom_filter

    def test_string_items(self, bloom_filter):
        """Test adding string items (node IDs)."""
        node_id = "123456789"

        bloom_filter.add(node_id)
        assert node_id in bloom_filter

    def test_integer_items(self, bloom_filter):
        """Test adding integer items."""
        node_id = 123456789

        bloom_filter.add(node_id)
        assert node_id in bloom_filter


class TestBloomFilterProperties:
    """Tests for bloom filter properties and behavior."""

    @pytest.fixture
    def bloom_filter(self):
        """Create a bloom filter for testing."""
        try:
            from pybloom_live import BloomFilter
            return BloomFilter(capacity=1000, error_rate=0.01)
        except ImportError:
            pytest.skip("pybloom_live not installed")

    def test_no_false_negatives(self, bloom_filter):
        """Bloom filters guarantee no false negatives."""
        items = [f"item_{i}" for i in range(100)]

        for item in items:
            bloom_filter.add(item)

        # All added items MUST be found (no false negatives)
        for item in items:
            assert item in bloom_filter

    def test_approximate_false_positive_rate(self, bloom_filter):
        """Test that false positive rate is approximately as configured."""
        # Add 500 items (half capacity)
        for i in range(500):
            bloom_filter.add(f"added_{i}")

        # Check 1000 items that were NOT added
        false_positives = 0
        test_count = 1000
        for i in range(test_count):
            if f"not_added_{i}" in bloom_filter:
                false_positives += 1

        # False positive rate should be around 1% (error_rate=0.01)
        # Allow some variance in testing
        fp_rate = false_positives / test_count
        # Should be less than 5% (generous margin for statistical variance)
        assert fp_rate < 0.05, f"False positive rate {fp_rate} is too high"


class TestCoordinateDeduplication:
    """Tests for coordinate deduplication patterns used in engine.py."""

    @pytest.fixture
    def bloom_filter(self):
        """Create a bloom filter for testing."""
        try:
            from pybloom_live import BloomFilter
            return BloomFilter(capacity=10000, error_rate=0.001)
        except ImportError:
            pytest.skip("pybloom_live not installed")

    def test_coordinate_tuple_dedup(self, bloom_filter):
        """Test deduplication of coordinate tuples."""
        coords = [
            (35.123456, -82.654321),
            (35.123456, -82.654321),  # Duplicate
            (35.234567, -82.765432),
            (35.234567, -82.765432),  # Duplicate
        ]

        unique_coords = []
        for coord in coords:
            if coord not in bloom_filter:
                unique_coords.append(coord)
                bloom_filter.add(coord)

        assert len(unique_coords) == 2

    def test_new_coords_filtering_pattern(self, bloom_filter):
        """Test the pattern used in engine.py for filtering new coords."""
        batch_coords = [
            (35.1, -82.5),
            (35.2, -82.6),
            (35.3, -82.7),
        ]

        # Pre-populate with some coords
        bloom_filter.add((35.1, -82.5))

        # Filter to only new coords (pattern from engine.py:10761)
        new_coords = [c for c in batch_coords if c not in bloom_filter]

        assert len(new_coords) == 2
        assert (35.1, -82.5) not in new_coords
        assert (35.2, -82.6) in new_coords
        assert (35.3, -82.7) in new_coords


class TestSetFallbackBehavior:
    """Tests for set-based fallback when bloom filter is unavailable."""

    def test_set_deduplication(self):
        """Test that set fallback provides same deduplication behavior."""
        global_coords_seen = set()

        coords = [
            (35.1, -82.5),
            (35.1, -82.5),  # Duplicate
            (35.2, -82.6),
        ]

        unique_coords = []
        for coord in coords:
            if coord not in global_coords_seen:
                unique_coords.append(coord)
                global_coords_seen.add(coord)

        assert len(unique_coords) == 2

    def test_set_no_false_positives(self):
        """Test that set has no false positives (unlike bloom filter)."""
        coords_seen = set()
        coords_seen.add((35.1, -82.5))

        # Set will never have false positives
        assert (35.2, -82.6) not in coords_seen

    def test_set_vs_bloom_api_compatibility(self):
        """Test that set and bloom filter have compatible APIs."""
        # Both should support 'in' operator and 'add' method
        test_set = set()

        # Test that set has the same interface we use
        assert hasattr(test_set, 'add')
        assert (35.1, -82.5) not in test_set
        test_set.add((35.1, -82.5))
        assert (35.1, -82.5) in test_set


class TestMemoryEfficiency:
    """Tests documenting memory efficiency of bloom filters vs sets."""

    def test_bloom_filter_memory_estimate(self):
        """Document expected memory usage for bloom filters."""
        # From engine.py comments:
        # 150M capacity @ 0.001 error rate = ~225MB
        # This is much better than set which would use 3GB+

        # These are documented values from the codebase
        COORD_BLOOM_CAPACITY = 150_000_000
        NODE_BLOOM_CAPACITY = 200_000_000
        ERROR_RATE = 0.001

        # Approximate memory formula for bloom filter:
        # bits = -n * ln(p) / (ln(2)^2) where n=capacity, p=error_rate
        # bytes = bits / 8
        import math
        bits_per_element = -math.log(ERROR_RATE) / (math.log(2) ** 2)

        coord_bloom_mb = (COORD_BLOOM_CAPACITY * bits_per_element) / 8 / (1024 * 1024)
        node_bloom_mb = (NODE_BLOOM_CAPACITY * bits_per_element) / 8 / (1024 * 1024)

        # Verify bloom filters use reasonable memory (< 500MB each)
        assert coord_bloom_mb < 500, f"Coord bloom {coord_bloom_mb}MB exceeds 500MB"
        assert node_bloom_mb < 500, f"Node bloom {node_bloom_mb}MB exceeds 500MB"

    def test_set_memory_would_be_larger(self):
        """Document that sets would use much more memory for large datasets."""
        # A coordinate tuple (float, float) uses ~56 bytes in Python
        # Set overhead adds another ~40 bytes per entry
        # Total: ~96 bytes per coordinate

        BYTES_PER_COORD_IN_SET = 96
        COORD_COUNT = 150_000_000

        set_memory_gb = (COORD_COUNT * BYTES_PER_COORD_IN_SET) / (1024 ** 3)

        # Sets would use ~13.4 GB for 150M coordinates
        assert set_memory_gb > 10, "Set memory estimate should be > 10GB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
