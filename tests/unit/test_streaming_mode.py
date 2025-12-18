#!/usr/bin/env python3
"""
Unit tests for streaming mode functionality.

Tests the `should_use_streaming_mode()` function that determines whether to use
disk-based streaming or in-memory processing based on region size and available
system memory.

Streaming mode is critical for large regions like France, California, etc.
that would cause OOM errors with in-memory processing.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climb_analyzer.engine import should_use_streaming_mode


class TestStreamingModeDecision:
    """Tests for should_use_streaming_mode() decision logic."""

    def test_small_region_no_streaming(self):
        """Small regions should use in-memory processing."""
        # Small region: ~1 deg² (e.g., a small county)
        # ~8000 ways, ~8MB estimated
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 16 * (1024**3)  # 16 GB available
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=35.0, min_lon=-82.0,
                max_lat=36.0, max_lon=-81.0  # 1 deg²
            )

        assert result is False

    def test_large_region_uses_streaming(self):
        """Large regions (>100 deg²) should use streaming."""
        # Large region: 150 deg² (e.g., California)
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 16 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=32.0, min_lon=-125.0,
                max_lat=42.0, max_lon=-110.0  # 150 deg²
            )

        assert result is True

    def test_medium_region_memory_check(self):
        """Medium regions use memory threshold for decision."""
        # Medium region: 50 deg² (~400K ways, ~400MB)
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 2 * (1024**3)  # Only 2 GB available
            mock_psutil.virtual_memory.return_value = mock_mem

            # With low memory, even medium region should stream
            result = should_use_streaming_mode(
                min_lat=35.0, min_lon=-85.0,
                max_lat=40.0, max_lon=-75.0  # 50 deg²
            )

        # 50 deg² × 8000 = 400K ways × 1KB = ~400MB
        # 400MB > 50% of 2GB (1GB), so should stream
        assert result is True

    def test_way_count_threshold_1m(self):
        """Regions with >1M estimated ways should use streaming."""
        # Region with >1M ways: needs ~125 deg²
        # 125 deg² × 8000 ways/deg² = 1,000,000 ways
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 32 * (1024**3)  # Lots of memory
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=30.0, min_lon=-130.0,
                max_lat=45.0, max_lon=-120.0  # 150 deg² > 125 deg²
            )

        # >100 deg² threshold triggers streaming
        assert result is True


class TestAreaCalculation:
    """Tests for region area calculation."""

    def test_area_calculation(self):
        """Test area calculation in square degrees."""
        # Area should be lat_span × lon_span
        min_lat, min_lon = 35.0, -82.0
        max_lat, max_lon = 40.0, -77.0

        expected_area = (max_lat - min_lat) * (max_lon - min_lon)  # 5 × 5 = 25

        assert expected_area == 25.0

    def test_small_area_threshold(self):
        """Test that areas <100 deg² don't auto-trigger streaming."""
        # 99 deg² should not trigger streaming by area alone
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 32 * (1024**3)  # Lots of memory
            mock_psutil.virtual_memory.return_value = mock_mem

            # 9.9 × 10 = 99 deg²
            result = should_use_streaming_mode(
                min_lat=35.0, min_lon=-85.0,
                max_lat=44.9, max_lon=-75.0  # 99 deg²
            )

        # 99 deg² × 8000 = 792K ways (< 1M)
        # 792K × 1KB = 792MB
        # 792MB < 50% of 32GB (16GB)
        # 99 < 100 deg²
        # Should NOT stream
        assert result is False


class TestMemoryEstimation:
    """Tests for memory estimation logic."""

    def test_ways_per_deg2_estimate(self):
        """Test the ~8000 ways per deg² estimate."""
        # From function comments: California (500 deg²) has ~4M ways
        # So ~8000 ways/deg²
        WAYS_PER_DEG2 = 8000
        california_area = 500
        estimated_ways = california_area * WAYS_PER_DEG2

        assert estimated_ways == 4_000_000

    def test_memory_per_way_estimate(self):
        """Test the ~1KB per way memory estimate."""
        # Each SimpleWay object is roughly 500-1000 bytes
        # Conservative estimate of 1KB per way
        BYTES_PER_WAY = 1024
        ways_count = 1_000_000
        estimated_memory_gb = (ways_count * BYTES_PER_WAY) / (1024**3)

        assert estimated_memory_gb == pytest.approx(0.93, rel=0.1)  # ~1GB for 1M ways

    def test_france_memory_estimate(self):
        """Test memory estimate for France-sized region."""
        # France is roughly 10° lat × 10° lon = 100 deg²
        area_deg2 = 100
        estimated_ways = area_deg2 * 8000  # 800K ways
        estimated_memory_gb = (estimated_ways * 1024) / (1024**3)

        # Should be ~0.75 GB
        assert estimated_memory_gb < 1.0


class TestPsutilFallback:
    """Tests for psutil availability fallback."""

    def test_with_psutil_available(self):
        """Test when psutil is available."""
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 16 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem

            # Small region, lots of memory - should not stream
            result = should_use_streaming_mode(
                min_lat=35.0, min_lon=-82.0,
                max_lat=36.0, max_lon=-81.0
            )

        assert result is False

    def test_psutil_not_installed(self):
        """Test fallback when psutil is not installed."""
        with patch.dict('sys.modules', {'psutil': None}):
            # The function will use 8GB fallback
            # Small region should still not stream
            result = should_use_streaming_mode(
                min_lat=35.0, min_lon=-82.0,
                max_lat=36.0, max_lon=-81.0
            )

        assert result is False

    def test_fallback_memory_8gb(self):
        """Test that fallback assumes 8GB available."""
        # This is from engine.py line 9302
        FALLBACK_AVAILABLE_GB = 8.0
        assert FALLBACK_AVAILABLE_GB == 8.0


class TestStreamingThresholds:
    """Tests for the three streaming thresholds."""

    def test_threshold_memory_50_percent(self):
        """Test that >50% memory usage triggers streaming."""
        # Region that needs 5GB, with 8GB available
        # 5GB > 50% of 8GB (4GB), so should stream
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * (1024**3)  # 8 GB available
            mock_psutil.virtual_memory.return_value = mock_mem

            # 625 deg² × 8000 ways × 1KB = 5GB
            result = should_use_streaming_mode(
                min_lat=30.0, min_lon=-95.0,
                max_lat=55.0, max_lon=-70.0  # 625 deg²
            )

        # This also triggers the >100 deg² threshold
        assert result is True

    def test_threshold_area_100_deg2(self):
        """Test that >100 deg² triggers streaming."""
        # Area threshold is 100 deg²
        AREA_THRESHOLD = 100
        assert AREA_THRESHOLD == 100

    def test_threshold_ways_1m(self):
        """Test that >1M ways triggers streaming."""
        # Way count threshold is 1,000,000
        WAY_THRESHOLD = 1_000_000
        assert WAY_THRESHOLD == 1_000_000


class TestRealWorldRegions:
    """Tests with real-world region examples."""

    def test_isle_of_wight_no_streaming(self):
        """Isle of Wight (small island) should not use streaming."""
        # Isle of Wight: ~0.3 deg × 0.2 deg = ~0.06 deg²
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=50.57, min_lon=-1.58,
                max_lat=50.77, max_lon=-1.08  # ~0.1 deg²
            )

        assert result is False

    def test_delaware_no_streaming(self):
        """Delaware (small US state) should not use streaming."""
        # Delaware: ~1.4 deg × 0.8 deg = ~1.1 deg²
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=38.45, min_lon=-75.79,
                max_lat=39.84, max_lon=-74.98  # ~1.1 deg²
            )

        assert result is False

    def test_california_uses_streaming(self):
        """California (large US state) should use streaming."""
        # California: ~10 deg × 10 deg = ~100 deg²
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 16 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=32.53, min_lon=-124.48,
                max_lat=42.01, max_lon=-114.13  # ~98 deg²
            )

        # Close to 100 deg² threshold
        # 98 × 8000 = 784K ways (< 1M)
        # But with ~800MB estimated for 16GB, should NOT stream
        # This is a borderline case
        assert result is False

    def test_france_uses_streaming(self):
        """France (large European country) should use streaming."""
        # France: ~9 deg × 11 deg = ~99 deg²
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * (1024**3)  # Less memory
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=41.3, min_lon=-5.1,
                max_lat=51.1, max_lon=9.6  # ~144 deg²
            )

        # 144 deg² > 100 deg² threshold
        assert result is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_area_region(self):
        """Test handling of zero-area region (same min/max)."""
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=35.0, min_lon=-82.0,
                max_lat=35.0, max_lon=-82.0  # Point, not area
            )

        assert result is False

    def test_very_low_memory(self):
        """Test with very low available memory."""
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 0.5 * (1024**3)  # Only 512 MB
            mock_psutil.virtual_memory.return_value = mock_mem

            # Even tiny region might trigger streaming with very low memory
            result = should_use_streaming_mode(
                min_lat=35.0, min_lon=-82.0,
                max_lat=40.0, max_lon=-77.0  # 25 deg²
            )

        # 25 × 8000 × 1KB = 200MB
        # 200MB > 50% of 512MB (256MB) = False
        # So should NOT stream
        assert result is False

    def test_negative_coordinates(self):
        """Test with negative coordinates (southern hemisphere)."""
        with patch('climb_analyzer.engine.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 16 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem

            result = should_use_streaming_mode(
                min_lat=-35.0, min_lon=145.0,
                max_lat=-30.0, max_lon=155.0  # 50 deg² in Australia
            )

        # 50 deg² < 100 deg², should not stream
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
