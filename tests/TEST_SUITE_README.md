# Climb Analyzer Test Suite

This document describes the comprehensive unit test suite for the climb analyzer project.

## Test Files Created

### 1. Elevation Data Fetching Tests
**File:** `tests/unit/data/test_elevation_fetcher.py` (24KB, ~750 lines)

Tests for the `FastElevationFetcher` class and elevation data fetching functionality.

#### Coverage:
- ✅ **ElevationFetchLog tracking**
  - Failed fetch recording with metadata
  - Successful fetch tracking (primary and fallback)
  - Summary generation

- ✅ **URL building and configuration**
  - Build elevation URLs from dataset names
  - Handle trailing slashes
  - Handle missing base URLs

- ✅ **Initialization and setup**
  - Default and custom dataset configuration
  - Fallback endpoint configuration
  - Error logger integration

- ✅ **Coordinate deduplication**
  - Exact duplicate detection
  - Coordinate rounding to 6 decimal places
  - Mapping back to original order

- ✅ **Batch fetching**
  - Successful batch requests
  - Handling null/missing elevations
  - Batch size adaptation

- ✅ **Fallback logic**
  - Cascading fallback (NED10m → SRTM30m → ASTER30m)
  - Partial data filling from fallback
  - Dataset source tracking

- ✅ **Rate limiting and retries**
  - HTTP 429 rate limit handling
  - HTTP 504 timeout retry
  - Exponential backoff
  - Max retry limits

- ✅ **Adaptive batch sizing**
  - Batch size reduction on errors
  - Minimum batch size enforcement
  - Progressive reduction

- ✅ **Error handling**
  - Connection errors
  - Timeout errors
  - Invalid JSON responses
  - Empty coordinate lists

- ✅ **Parallel processing**
  - ThreadPoolExecutor usage
  - Concurrent batch processing

- ✅ **End-to-end workflows**
  - Complete fetch workflow
  - Integration with metadata and logging

#### Key Test Classes:
- `TestElevationFetchLog` - Log tracking functionality
- `TestBuildElevationUrl` - URL construction
- `TestFastElevationFetcherInit` - Initialization
- `TestCoordinateMatching` - Deduplication logic
- `TestBatchFetching` - Batch API interaction
- `TestFallbackLogic` - Multi-dataset fallback
- `TestRateLimiting` - Retry and backoff
- `TestAdaptiveBatchSizing` - Dynamic batch sizing
- `TestErrorHandling` - Error scenarios
- `TestParallelProcessing` - Concurrency
- `TestEndToEndFetching` - Integration tests

---

### 2. Boundary Road Merging Tests
**File:** `tests/unit/core/test_road_merger.py` (22KB, ~700 lines)

Tests for the `BoundaryMerger` class and segment merging algorithms.

#### Coverage:
- ✅ **Distance calculations**
  - Haversine formula implementation
  - Zero distance (identical points)
  - Known distance validation
  - Symmetry verification
  - Small delta handling

- ✅ **Coordinate matching**
  - Exact coordinate matching
  - Matching within tolerance
  - Distance-based matching
  - Tolerance threshold enforcement

- ✅ **Duplicate detection**
  - Identical segment detection
  - Reversed segment detection
  - Tolerance-based matching
  - Different length rejection
  - Different coordinate rejection

- ✅ **Segment connection**
  - End-to-start connections
  - End-to-end connections
  - Start-to-start connections
  - No connection handling
  - Way ID consolidation

- ✅ **Spatial grouping**
  - Location-based grouping
  - Grid-based clustering
  - Large street optimization
  - Hierarchical sub-grouping

- ✅ **Street merging algorithm**
  - Single segment handling
  - Duplicate removal
  - Chain connection
  - Disconnected segment preservation
  - Empty list handling
  - Complex chain merging

- ✅ **Parallel processing**
  - Worker count calculation
  - Memory-based worker limiting
  - Low memory scenarios
  - Error handling in workers
  - Top-level worker function

- ✅ **Integration workflows**
  - Serial merging
  - Checkpoint support

- ✅ **Edge cases**
  - Single coordinate segments
  - Very long segments (1000+ coordinates)
  - Segments with gaps
  - High-precision coordinates

#### Key Test Classes:
- `TestDistanceCalculation` - Haversine distance
- `TestCoordinateMatching` - Tolerance matching
- `TestDuplicateDetection` - Duplicate finding
- `TestSegmentConnection` - Connection logic
- `TestSpatialGrouping` - Spatial optimization
- `TestStreetMerging` - Main merge algorithm
- `TestParallelProcessing` - Concurrency and memory
- `TestMergeIntegration` - End-to-end workflows
- `TestEdgeCases` - Boundary conditions

---

### 3. Geocoding Tests
**File:** `tests/unit/data/test_geocoding.py` (20KB, ~650 lines)

Tests for the `ReverseGeocoder` class and offline geocoding functionality.

#### Coverage:
- ✅ **Initialization**
  - Basic initialization
  - Default parameters
  - Async context manager protocol

- ✅ **Coordinate deduplication**
  - Identical coordinate clustering
  - Nearby coordinate clustering
  - No duplicate scenarios
  - Empty list handling
  - Single coordinate handling

- ✅ **Reverse geocoding**
  - Empty coordinate list
  - Basic geocoding workflow
  - Batch processing (1700+ coords)
  - Result format validation
  - Unknown location handling
  - Partial data handling

- ✅ **Checkpointing**
  - Checkpoint saving
  - Signal handler integration
  - Resume functionality

- ✅ **Coordinate mapping**
  - Order preservation
  - Duplicate mapping
  - Clustering with mapping

- ✅ **Error handling**
  - reverse_geocoder errors
  - Invalid coordinates

- ✅ **Performance characteristics**
  - Large coordinate sets (5000+)
  - Deduplication efficiency

- ✅ **Integration workflows**
  - Complete geocoding workflow
  - Mixed success/failure results

#### Key Test Classes:
- `TestReverseGeocoderInit` - Initialization
- `TestCoordinateDeduplication` - Clustering logic
- `TestReverseGeocoding` - Main geocoding
- `TestCheckpointing` - Save/resume
- `TestCoordinateMapping` - Result mapping
- `TestErrorHandling` - Error scenarios
- `TestPerformance` - Performance tests
- `TestGeocodingIntegration` - End-to-end

---

### 4. OSM Data Parsing Tests
**File:** `tests/unit/data/test_osm_parsing.py` (22KB, ~700 lines)

Tests for the `SpatialIndexManager` class and OSM data handling.

#### Coverage:
- ✅ **Initialization**
  - Basic initialization
  - Silent mode
  - File path calculation

- ✅ **Index existence checking**
  - All files present
  - Missing .idx file
  - Missing .dat file
  - Missing metadata file
  - Nonexistent directory

- ✅ **Metadata loading**
  - JSONL parsing
  - Way entry loading
  - Header metadata
  - Empty file handling
  - Corrupted JSON handling

- ✅ **Index loading**
  - Successful rtree loading
  - Missing file handling
  - Missing metadata
  - rtree unavailable
  - Verbose vs silent modes

- ✅ **Bounding box queries**
  - Basic bbox queries
  - Empty results
  - Metadata lookup integration

- ✅ **Index reloading**
  - Reload on loaded index

- ✅ **Index creation**
  - NotImplementedError verification

- ✅ **Edge cases**
  - Non-existent way IDs
  - Missing metadata fields
  - Large metadata files (10k+ ways)
  - Unicode in metadata

- ✅ **Integration workflows**
  - Complete load and query
  - Multiple queries on same index

#### Key Test Classes:
- `TestSpatialIndexManagerInit` - Initialization
- `TestIndexExistence` - File validation
- `TestMetadataLoading` - JSONL parsing
- `TestIndexLoading` - rtree loading
- `TestIndexQueries` - Bbox queries
- `TestIndexReload` - Reload functionality
- `TestIndexCreation` - Creation error
- `TestEdgeCases` - Boundary conditions
- `TestSpatialIndexIntegration` - End-to-end

---

### 5. Performance Tests
**File:** `tests/unit/test_performance.py` (24KB, ~700 lines)

Comprehensive performance and benchmarking tests for all components.

#### Coverage:
- ✅ **Elevation fetching performance**
  - Batch processing throughput (>100 coords/sec)
  - Parallel vs serial comparison
  - Deduplication efficiency (>25% reduction)
  - Memory usage (<100 MB for 5000 coords)
  - Adaptive batch sizing impact

- ✅ **Road merging performance**
  - Endpoint indexing (O(1) lookups)
  - Spatial grouping overhead (<5s for 100 segments)
  - Parallel worker efficiency
  - Memory-safe worker calculation

- ✅ **Geocoding performance**
  - Batch throughput (>500 coords/sec)
  - Clustering efficiency (>10% reduction)

- ✅ **Spatial index performance**
  - Index loading time (<5s for 10k ways)
  - Bbox query performance (>100 queries/sec)

- ✅ **End-to-end performance**
  - Workflow scaling (linear with data size)

- ✅ **Performance benchmarks**
  - Summary reporting
  - Benchmark targets

#### Performance Targets:
- **Elevation Fetching:** >100 coords/sec, <100 MB memory
- **Road Merging:** >100 segments/sec
- **Geocoding:** >500 coords/sec offline
- **Spatial Index:** >100 queries/sec
- **Deduplication:** >25% reduction in API calls

#### Key Test Classes:
- `TestElevationFetchingPerformance` - Elevation performance
- `TestRoadMergingPerformance` - Merge performance
- `TestGeocodingPerformance` - Geocoding performance
- `TestSpatialIndexPerformance` - Index performance
- `TestEndToEndPerformance` - Integration performance
- `TestPerformanceBenchmarks` - Benchmark reporting

---

## Running the Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
# Elevation tests
pytest tests/unit/data/test_elevation_fetcher.py -v

# Road merging tests
pytest tests/unit/core/test_road_merger.py -v

# Geocoding tests
pytest tests/unit/data/test_geocoding.py -v

# OSM parsing tests
pytest tests/unit/data/test_osm_parsing.py -v

# Performance tests
pytest tests/unit/test_performance.py -v
```

### Run Tests by Type
```bash
# Run only unit tests
pytest tests/unit/ -m unit -v

# Run only performance tests (slow)
pytest tests/unit/test_performance.py -m slow -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

### Run Specific Test Class
```bash
pytest tests/unit/data/test_elevation_fetcher.py::TestFallbackLogic -v
```

### Run with Coverage
```bash
pytest tests/ --cov=climb_analyzer --cov-report=html
```

---

## Test Statistics

### Total Test Coverage

| Component | Test File | Test Classes | Approx Tests | Lines |
|-----------|-----------|--------------|--------------|-------|
| Elevation Fetching | `test_elevation_fetcher.py` | 11 | ~80 | 750 |
| Road Merging | `test_road_merger.py` | 9 | ~60 | 700 |
| Geocoding | `test_geocoding.py` | 8 | ~50 | 650 |
| OSM Parsing | `test_osm_parsing.py` | 9 | ~55 | 700 |
| Performance | `test_performance.py` | 6 | ~25 | 700 |
| **Total** | **5 files** | **43 classes** | **~270 tests** | **~3500 lines** |

---

## Test Fixtures

Common fixtures available in `tests/conftest.py`:

- **Coordinate fixtures:** `sample_coordinates`, `sample_elevation_data`
- **Profile fixtures:** `sample_climb_profile`, `sample_steep_climb_profile`
- **OSM fixtures:** `sample_way_data`, `sample_climb_segment`
- **File fixtures:** `temp_dir`, `temp_osm_file`, `temp_elevation_file`, `temp_output_dir`
- **Mock fixtures:** `mock_elevation_fetcher`, `mock_error_logger`, `mock_elevation_stats_collector`
- **Config fixtures:** `test_config`, `test_checkpoint_config`

---

## Dependencies

Required for running tests:
- `pytest >= 7.0`
- `pytest-asyncio` (for async geocoding tests)
- `pytest-mock` (optional, for advanced mocking)
- `pytest-cov` (for coverage reports)

All application dependencies should already be installed from `requirements.txt`.

---

## Test Design Principles

### 1. **Isolation**
- Each test is independent
- Uses mocks to avoid external dependencies
- Temporary files cleaned up automatically

### 2. **Comprehensive Coverage**
- Happy path scenarios
- Error conditions
- Edge cases and boundary conditions
- Performance characteristics

### 3. **Fast Execution**
- Mocked external calls (HTTP, file I/O)
- Minimal setup/teardown
- Parallel test execution supported

### 4. **Maintainability**
- Clear test names describing what is tested
- Organized by component and functionality
- Well-documented with docstrings

### 5. **Performance Testing**
- Marked with `@pytest.mark.slow`
- Validates throughput and efficiency
- Memory usage monitoring
- Scaling characteristics

---

## Key Testing Patterns Used

### 1. **Mock External Dependencies**
```python
with patch('climb_analyzer.data.elevation.requests.get') as mock_get:
    mock_response = Mock()
    mock_response.ok = True
    mock_get.return_value = mock_response
```

### 2. **Async Test Support**
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await geocoder.reverse_geocode_parallel(...)
```

### 3. **Parametrized Tests**
```python
@pytest.fixture(params=[100, 500, 1000])
def data_size(request):
    return request.param
```

### 4. **Temporary File Handling**
```python
@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)
```

---

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pytest tests/unit/ -v --cov=climb_analyzer
    pytest tests/integration/ -v --cov-append
```

---

## Future Enhancements

Potential additions to the test suite:

1. **Integration Tests**
   - Full workflow tests with real data
   - Database integration tests
   - API endpoint tests

2. **Stress Tests**
   - Very large datasets (100k+ coordinates)
   - Memory leak detection
   - Concurrency stress testing

3. **Property-Based Tests**
   - Using Hypothesis for property testing
   - Generative test data

4. **Regression Tests**
   - Known bug scenarios
   - Historical performance benchmarks

---

## Contributing

When adding new tests:

1. Follow existing naming conventions: `test_<functionality>.py`
2. Organize by component: `tests/unit/<component>/`
3. Add docstrings explaining what is tested
4. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.slow`)
5. Update this README with new test descriptions

---

**Generated:** 2025-10-18
**Test Suite Version:** 1.0
**Total Test Files:** 5
**Estimated Test Count:** ~270
**Total Lines of Test Code:** ~3500
