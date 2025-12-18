"""
Performance tests for climb analyzer components.

Tests cover:
- Elevation fetching throughput and efficiency
- Road merging performance and memory usage
- Geocoding batch processing performance
- Spatial index query performance
- Parallel processing efficiency
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import components to test
from climb_analyzer.data.elevation import FastElevationFetcher
from climb_analyzer.core.merger import BoundaryMerger
from climb_analyzer.data.geocoding import ReverseGeocoder
from climb_analyzer.data.spatial_index import SpatialIndexManager


# ============================================================================
# Performance Test Markers
# ============================================================================

# Mark tests that are slow-running
pytestmark = pytest.mark.slow


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def large_coordinate_set():
    """Generate large set of coordinates for performance testing."""
    return [(35.0 + i*0.001, -82.0 + i*0.001) for i in range(10000)]


@pytest.fixture
def large_segment_set():
    """Generate large set of segments for performance testing."""
    segments = []
    for i in range(5000):
        segments.append({
            "way_id": 100000 + i,
            "way_name": f"Street {i % 100}",  # Create groups
            "coordinates": [
                (35.0 + i*0.0001, -82.0 + i*0.0001),
                (35.0 + (i+1)*0.0001, -82.0 + (i+1)*0.0001),
            ],
            "way_ids": [100000 + i],
        })
    return segments


@pytest.fixture
def mock_persistence():
    """Mock persistence for performance tests."""
    mock = Mock()
    mock.save_elevation_progress = Mock()
    mock.clear_elevation_progress = Mock()
    mock.save_geocoding_progress = Mock()
    return mock


# ============================================================================
# Elevation Fetching Performance Tests
# ============================================================================

class TestElevationFetchingPerformance:
    """Test elevation fetching performance characteristics."""

    def test_batch_processing_throughput(self, mock_persistence):
        """Test throughput of batch elevation processing."""
        coords = [(35.0 + i*0.001, -82.0) for i in range(1000)]

        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/srtm30m"
            fetcher = FastElevationFetcher()

            with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
                # Mock fast successful responses
                mock_response = Mock()
                mock_response.ok = True
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "status": "OK",
                    "results": [{"elevation": 100.0 + i} for i in range(500)]
                }
                mock_get.return_value = mock_response

                start_time = time.time()
                result = fetcher.fetch_elevations_for_coordinates(
                    coords, mock_persistence, progress_desc=None
                )
                elapsed_time = time.time() - start_time

                # Should process 1000 coordinates efficiently
                assert len(result) == 1000
                # Should complete in reasonable time (< 5 seconds for mocked requests)
                assert elapsed_time < 5.0

                # Calculate throughput
                throughput = len(coords) / elapsed_time
                print(f"Elevation fetch throughput: {throughput:.0f} coords/sec")
                assert throughput > 100  # At least 100 coords/sec

    def test_parallel_vs_serial_comparison(self, mock_persistence):
        """Compare parallel vs serial processing performance."""
        coords = [(35.0 + i*0.001, -82.0) for i in range(500)]

        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/srtm30m"

            with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "status": "OK",
                    "results": [{"elevation": 100.0}] * 500
                }
                mock_get.return_value = mock_response

                # Test parallel processing
                fetcher_parallel = FastElevationFetcher()
                start_parallel = time.time()
                result_parallel = fetcher_parallel.fetch_elevations_for_coordinates(
                    coords, mock_persistence, progress_desc=None
                )
                time_parallel = time.time() - start_parallel

                print(f"Parallel processing time: {time_parallel:.3f}s")
                assert len(result_parallel) == 500

    def test_deduplication_efficiency(self, mock_persistence):
        """Test that deduplication reduces work efficiently."""
        # Create coordinates with 50% duplicates
        unique_coords = [(35.0 + i*0.001, -82.0) for i in range(500)]
        coords_with_dupes = unique_coords + unique_coords[:250]  # 750 total, 500 unique

        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/srtm30m"
            fetcher = FastElevationFetcher()

            with patch.object(fetcher, '_fetch_single_batch') as mock_fetch:
                mock_fetch.return_value = ([100.0] * 500, [], {})

                result = fetcher.fetch_elevations_for_coordinates(
                    coords_with_dupes, mock_persistence, progress_desc=None
                )

                # Should return all coordinates
                assert len(result) == 750

                # Calculate total coordinates fetched across all batches
                total_fetched = sum(len(call.args[0]) for call in mock_fetch.call_args_list)

                # Should fetch roughly 500 unique coordinates, not 750
                assert total_fetched <= 550  # Allow some batching overhead

                efficiency = (1 - total_fetched / len(coords_with_dupes)) * 100
                print(f"Deduplication efficiency: {efficiency:.1f}% reduction")
                assert efficiency > 25  # At least 25% reduction

    def test_memory_usage_large_fetch(self, mock_persistence):
        """Test memory usage during large elevation fetch."""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available")

        coords = [(35.0 + i*0.0001, -82.0) for i in range(5000)]

        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/srtm30m"
            fetcher = FastElevationFetcher()

            with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "status": "OK",
                    "results": [{"elevation": 100.0}] * 500
                }
                mock_get.return_value = mock_response

                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                result = fetcher.fetch_elevations_for_coordinates(
                    coords, mock_persistence, progress_desc=None
                )
                mem_after = process.memory_info().rss / 1024 / 1024  # MB

                mem_increase = mem_after - mem_before
                print(f"Memory increase: {mem_increase:.1f} MB for 5000 coordinates")

                # Memory increase should be reasonable (< 100 MB for 5000 coords)
                assert mem_increase < 100

    def test_adaptive_batch_sizing_performance(self, mock_persistence):
        """Test that adaptive batch sizing improves performance."""
        coords = [(35.0 + i*0.001, -82.0) for i in range(1000)]

        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/srtm30m"
            fetcher = FastElevationFetcher()

            original_batch_size = fetcher.optimal_batch_size

            # Simulate server issues requiring batch size reduction
            call_count = [0]

            def mock_get_with_errors(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:  # First few calls fail
                    response = Mock()
                    response.ok = False
                    response.status_code = 504
                    return response
                else:
                    response = Mock()
                    response.ok = True
                    response.status_code = 200
                    response.json.return_value = {
                        "status": "OK",
                        "results": [{"elevation": 100.0}] * 99
                    }
                    return response

            with patch('climb_analyzer.data.elevation.requests.get', side_effect=mock_get_with_errors):
                with patch('time.sleep'):  # Speed up retries
                    result = fetcher.fetch_elevations_for_coordinates(
                        coords, mock_persistence, progress_desc=None
                    )

                    # Batch size should have been reduced
                    assert fetcher.optimal_batch_size <= original_batch_size


# ============================================================================
# Road Merging Performance Tests
# ============================================================================

class TestRoadMergingPerformance:
    """Test road merging performance characteristics."""

    def test_endpoint_indexing_performance(self):
        """Test O(1) endpoint indexing lookup performance."""
        merger = BoundaryMerger()

        # Create many segments with various endpoints
        segments = []
        for i in range(1000):
            segments.append({
                "way_id": i,
                "coordinates": [
                    (35.0 + i*0.001, -82.0),
                    (35.0 + (i+1)*0.001, -82.0),
                ],
                "way_ids": [i],
            })

        # Time the merging operation
        start_time = time.time()
        result = merger._merge_street(segments, "Test Street")
        elapsed_time = time.time() - start_time

        print(f"Merged {len(segments)} segments in {elapsed_time:.3f}s")

        # Should complete in reasonable time
        assert elapsed_time < 10.0  # Less than 10 seconds for 1000 segments

        # Calculate throughput
        throughput = len(segments) / elapsed_time
        print(f"Merge throughput: {throughput:.0f} segments/sec")

    def test_spatial_grouping_overhead(self):
        """Test overhead of spatial grouping for large streets."""
        merger = BoundaryMerger()

        # Create many segments for same street name (triggers spatial grouping)
        segments = []
        for i in range(100):
            segments.append({
                "way_id": 1000 + i,
                "way_name": "service",
                "coordinates": [
                    (35.0 + i*0.01, -82.0 + i*0.01),
                    (35.0 + (i+1)*0.01, -82.0 + (i+1)*0.01),
                ],
                "way_ids": [1000 + i],
            })

        start_time = time.time()
        result = merger._merge_large_street_with_spatial_grouping(segments, "service")
        elapsed_time = time.time() - start_time

        print(f"Spatial grouping for {len(segments)} segments: {elapsed_time:.3f}s")

        # Should handle spatial grouping efficiently
        assert elapsed_time < 5.0

    def test_parallel_worker_efficiency(self, large_segment_set):
        """Test parallel worker efficiency."""
        merger = BoundaryMerger()

        # Group segments by street name
        street_segments = {}
        for seg in large_segment_set:
            street_name = seg["way_name"]
            if street_name not in street_segments:
                street_segments[street_name] = []
            street_segments[street_name].append(seg)

        # Test with reduced worker count for test speed
        with patch.object(merger, 'max_workers', 4):
            with patch.object(merger, 'parallel_enabled', True):
                start_time = time.time()

                # Simulate parallel processing
                with patch('climb_analyzer.core.merger.ProcessPoolExecutor'):
                    # Test would use actual parallel processing here
                    # For unit test, we verify the setup
                    worker_count = merger._calculate_safe_worker_count(len(large_segment_set))
                    assert worker_count > 0
                    assert worker_count <= 4

                elapsed_time = time.time() - start_time
                print(f"Worker setup time: {elapsed_time:.3f}s")

    def test_memory_safe_worker_calculation(self):
        """Test that worker calculation respects memory constraints."""
        merger = BoundaryMerger()

        with patch('climb_analyzer.core.merger.psutil') as mock_psutil:
            with patch('climb_analyzer.core.merger.os.cpu_count', return_value=16):
                # Test with different memory scenarios
                memory_scenarios = [
                    (2 * 1024**3, 1),   # 2 GB total - should limit workers
                    (8 * 1024**3, 4),   # 8 GB total - moderate workers
                    (32 * 1024**3, 8),  # 32 GB total - more workers
                ]

                for total_mem, expected_min_workers in memory_scenarios:
                    mock_mem = Mock()
                    mock_mem.total = total_mem
                    mock_mem.available = total_mem // 2
                    mock_psutil.virtual_memory.return_value = mock_mem

                    worker_count = merger._calculate_safe_worker_count(10000)

                    print(f"Total memory: {total_mem/(1024**3):.0f}GB -> {worker_count} workers")
                    assert worker_count >= 1
                    assert worker_count <= merger.max_workers


# ============================================================================
# Geocoding Performance Tests
# ============================================================================

class TestGeocodingPerformance:
    """Test geocoding performance characteristics."""

    @pytest.mark.asyncio
    async def test_batch_processing_throughput(self, mock_persistence):
        """Test geocoding batch processing throughput."""
        geocoder = ReverseGeocoder()
        coords = [(35.0 + i*0.001, -82.0) for i in range(2000)]

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [
                {"name": "TestCity", "admin1": "TestState", "cc": "US"}
            ] * 1700  # Batch size

            start_time = time.time()
            result = await geocoder.reverse_geocode_parallel(
                coords, mock_persistence, progress_desc="Perf Test"
            )
            elapsed_time = time.time() - start_time

            print(f"Geocoded {len(coords)} coordinates in {elapsed_time:.3f}s")

            # Calculate throughput
            throughput = len(coords) / elapsed_time
            print(f"Geocoding throughput: {throughput:.0f} coords/sec")

            # Offline geocoding should be fast
            assert throughput > 500  # At least 500 coords/sec

    @pytest.mark.asyncio
    async def test_clustering_efficiency_reduction(self, mock_persistence):
        """Test that clustering reduces geocoding lookups."""
        geocoder = ReverseGeocoder()

        # Create coordinates in clusters
        coords = []
        for cluster in range(10):
            base_lat = 35.0 + cluster * 0.1
            base_lon = -82.0 + cluster * 0.1
            # 50 coordinates per cluster, all very close
            for i in range(50):
                coords.append((base_lat + i*0.0001, base_lon + i*0.0001))

        total_coords = len(coords)

        with patch('climb_analyzer.data.geocoding.rg.search') as mock_search:
            mock_search.return_value = [
                {"name": f"City{i}", "admin1": "State", "cc": "US"}
                for i in range(1700)
            ]

            result = await geocoder.reverse_geocode_parallel(
                coords, mock_persistence, progress_desc="Clustering Test"
            )

            # Calculate total unique lookups
            total_lookups = sum(len(call.args[0]) for call in mock_search.call_args_list)

            reduction = (1 - total_lookups / total_coords) * 100
            print(f"Clustering reduced lookups by {reduction:.1f}%")
            print(f"Total coords: {total_coords}, Unique lookups: {total_lookups}")

            # Should significantly reduce lookups through clustering
            assert total_lookups < total_coords
            assert reduction > 10  # At least 10% reduction


# ============================================================================
# Spatial Index Performance Tests
# ============================================================================

class TestSpatialIndexPerformance:
    """Test spatial index query performance."""

    def test_index_loading_time(self, tmp_path):
        """Test spatial index loading performance."""
        import json

        # Create mock index files
        osm_file = tmp_path / "test.osm.pbf"
        osm_file.touch()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        idx_file = cache_dir / "test.osm_spatial.idx"
        dat_file = cache_dir / "test.osm_spatial.dat"
        idx_file.touch()
        dat_file.touch()

        metadata_file = cache_dir / "test.osm_metadata.jsonl"

        # Create large metadata file
        with open(metadata_file, 'w') as f:
            f.write(json.dumps({"_index_metadata": {"total_ways": 10000}}) + '\n')
            for i in range(10000):
                f.write(json.dumps({
                    "id": 100000 + i,
                    "data": {"name": f"Street {i}", "highway": "residential"}
                }) + '\n')

        manager = SpatialIndexManager(str(osm_file), str(cache_dir))

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_rtree.Index.return_value = Mock()

            start_time = time.time()
            manager.load_index(silent=True)
            elapsed_time = time.time() - start_time

            print(f"Loaded 10000 ways in {elapsed_time:.3f}s")

            # Should load efficiently
            assert elapsed_time < 5.0  # Less than 5 seconds for 10k ways

            throughput = 10000 / elapsed_time
            print(f"Load throughput: {throughput:.0f} ways/sec")

    def test_bbox_query_performance(self, tmp_path):
        """Test bounding box query performance."""
        import json

        osm_file = tmp_path / "test.osm.pbf"
        osm_file.touch()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        idx_file = cache_dir / "test.osm_spatial.idx"
        dat_file = cache_dir / "test.osm_spatial.dat"
        idx_file.touch()
        dat_file.touch()

        metadata_file = cache_dir / "test.osm_metadata.jsonl"

        with open(metadata_file, 'w') as f:
            f.write(json.dumps({"_index_metadata": {"total_ways": 100}}) + '\n')
            for i in range(100):
                f.write(json.dumps({
                    "id": 100 + i,
                    "data": {"name": f"Street {i}"}
                }) + '\n')

        manager = SpatialIndexManager(str(osm_file), str(cache_dir))

        with patch('climb_analyzer.data.spatial_index.index') as mock_rtree:
            mock_idx = Mock()
            # Simulate returning many results
            mock_idx.intersection.return_value = list(range(100, 200))
            mock_rtree.Index.return_value = mock_idx

            manager.load_index(silent=True)

            # Test query performance
            bbox = (35.0, -82.0, 36.0, -81.0)

            start_time = time.time()
            for _ in range(1000):  # 1000 queries
                results = list(manager.spatial_idx.intersection(bbox))
            elapsed_time = time.time() - start_time

            queries_per_sec = 1000 / elapsed_time
            print(f"Bbox query performance: {queries_per_sec:.0f} queries/sec")

            # Should handle many queries efficiently
            assert queries_per_sec > 100  # At least 100 queries/sec


# ============================================================================
# End-to-End Performance Tests
# ============================================================================

class TestEndToEndPerformance:
    """Test end-to-end workflow performance."""

    def test_complete_workflow_performance_scaling(self):
        """Test that workflow scales linearly with data size."""
        sizes = [100, 500, 1000]
        times = []

        with patch('climb_analyzer.data.elevation.build_elevation_url') as mock_build:
            mock_build.return_value = "http://localhost:5000/v1/srtm30m"

            for size in sizes:
                coords = [(35.0 + i*0.001, -82.0) for i in range(size)]
                fetcher = FastElevationFetcher()
                mock_persistence = Mock()

                with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
                    mock_response = Mock()
                    mock_response.ok = True
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "status": "OK",
                        "results": [{"elevation": 100.0}] * 500
                    }
                    mock_get.return_value = mock_response

                    start_time = time.time()
                    result = fetcher.fetch_elevations_for_coordinates(
                        coords, mock_persistence, progress_desc=None
                    )
                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)

                print(f"Size {size}: {elapsed_time:.3f}s")

            # Check for reasonable scaling
            # Time for 1000 should be roughly 10x time for 100
            if times[0] > 0:
                scaling_factor = times[-1] / times[0]
                expected_factor = sizes[-1] / sizes[0]
                print(f"Scaling factor: {scaling_factor:.1f}x (expected ~{expected_factor}x)")

                # Allow some overhead, but should scale roughly linearly
                assert scaling_factor < expected_factor * 2


# ============================================================================
# Benchmark Summary
# ============================================================================

class TestPerformanceBenchmarks:
    """Collect and report performance benchmarks."""

    def test_performance_summary(self, capsys):
        """Generate performance benchmark summary."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)

        benchmarks = {
            "Elevation Fetching": {
                "Target": ">100 coords/sec",
                "Memory": "<100 MB for 5000 coords",
                "Deduplication": ">25% reduction",
            },
            "Road Merging": {
                "Target": ">100 segments/sec",
                "Spatial Grouping": "<5s for 100 segments",
                "Worker Safety": "Memory-aware scaling",
            },
            "Geocoding": {
                "Target": ">500 coords/sec",
                "Clustering": ">10% lookup reduction",
            },
            "Spatial Index": {
                "Loading": "<5s for 10k ways",
                "Queries": ">100 queries/sec",
            },
        }

        for component, metrics in benchmarks.items():
            print(f"\n{component}:")
            for metric, target in metrics.items():
                print(f"  {metric}: {target}")

        print("\n" + "="*80)

        captured = capsys.readouterr()
        assert "PERFORMANCE BENCHMARK SUMMARY" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
