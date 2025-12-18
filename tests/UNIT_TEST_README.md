# Climb Analyzer Test Suite

Comprehensive test suite for the climb analyzer project, including unit tests, integration tests, and test fixtures.

## Directory Structure

```
tests/
├── README.md                          # This file
├── conftest.py                        # Shared pytest configuration and fixtures
├── fixtures/                          # Test data fixtures
├── unit/                              # Unit tests
│   ├── core/                          # Core functionality tests
│   │   ├── test_climb_detection.py   # Climb detection logic
│   │   ├── test_climb_scoring.py     # Scoring algorithms (Basic, FIETS, PDI)
│   │   └── test_climb_data_structures.py  # ClimbSegment and ClimbMetrics
│   ├── data/                          # Data fetching tests
│   ├── utils/                         # Utility function tests
│   │   ├── test_error_logger.py      # Error logging functionality
│   │   └── test_elevation_stats_collector.py  # Statistics collection
│   └── processing/                    # Processing pipeline tests
└── integration/                       # Integration tests
    ├── test_complete_climb_workflow.py      # End-to-end climb analysis
    └── test_data_preparation_workflow.py    # Data setup workflows
```

## Installation

### Prerequisites

1. **Python 3.8+** is required
2. **pytest** and **pytest-cov** (for coverage reporting)

```bash
pip install pytest pytest-cov
```

### Optional Dependencies

For enhanced testing capabilities:

```bash
pip install pytest-xdist  # Parallel test execution
pip install pytest-timeout  # Test timeouts
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest

# Or explicitly specify test directory
pytest tests/
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run tests for specific component
pytest tests/unit/core/test_climb_scoring.py
```

### Run Tests by Marker

```bash
# Run only climb detection tests
pytest -m climb_detection

# Run only climb scoring tests
pytest -m climb_scoring

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run only tests that require data
pytest -m requires_data
```

### Run Tests with Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows

# Generate terminal coverage report
pytest --cov=. --cov-report=term-missing
```

### Run Tests in Parallel

```bash
# Use all CPU cores
pytest -n auto

# Use specific number of cores
pytest -n 4
```

### Verbose Output

```bash
# Show detailed test output
pytest -v

# Show even more detail (print statements, etc.)
pytest -vv

# Show captured stdout
pytest -s
```

## Test Markers

Tests are organized using pytest markers. Available markers:

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for complete workflows |
| `slow` | Tests that take significant time to run |
| `requires_data` | Tests that require external data files |
| `requires_network` | Tests that require network access |
| `climb_detection` | Tests related to climb detection logic |
| `climb_scoring` | Tests related to climb scoring algorithms |
| `elevation` | Tests related to elevation data fetching |
| `geocoding` | Tests related to geocoding functionality |
| `error_handling` | Tests related to error logging and handling |
| `data_setup` | Tests related to data setup and preparation |

## Test Configuration

Configuration is managed through [`pytest.ini`](../pytest.ini) in the project root.

### Key Configuration Options

```ini
[pytest]
# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Output options
addopts = -v --strict-markers --tb=short
```

## Writing New Tests

### Unit Test Example

```python
import pytest

@pytest.mark.unit
@pytest.mark.climb_detection
def test_climb_detection():
    """Test that valid climb is detected."""
    elevation_gain = 50.0  # meters
    distance = 1000.0      # meters
    avg_grade = 5.0        # percent

    # Your test logic here
    assert is_valid_climb(elevation_gain, distance, avg_grade) is True
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
def test_complete_workflow(temp_dir):
    """Test complete climb analysis workflow."""
    # Setup test data
    road = create_mock_road()

    # Run analysis
    climbs = analyze_climbs(road)

    # Verify results
    assert len(climbs) > 0
```

### Using Fixtures

```python
def test_with_fixture(sample_climb_profile):
    """Test using a shared fixture from conftest.py."""
    distances = sample_climb_profile["distances"]
    elevations = sample_climb_profile["elevations"]

    # Use fixture data in test
    assert len(distances) == len(elevations)
```

## Available Fixtures

Fixtures are defined in [`conftest.py`](conftest.py). Key fixtures include:

### Test Data Fixtures

- `sample_coordinates` - Sample GPS coordinates
- `sample_elevation_data` - Sample elevation data points
- `sample_climb_profile` - Moderate climb elevation profile
- `sample_steep_climb_profile` - Steep climb elevation profile
- `sample_way_data` - Sample OSM way data
- `sample_climb_segment` - Sample ClimbSegment instance

### File System Fixtures

- `temp_dir` - Temporary directory (auto-cleanup)
- `temp_osm_file` - Temporary OSM file
- `temp_elevation_file` - Temporary elevation data file
- `temp_output_dir` - Temporary output directory

### Mock Fixtures

- `mock_elevation_fetcher` - Mock elevation data fetcher
- `mock_error_logger` - Mock error logger
- `mock_elevation_stats_collector` - Mock statistics collector

### Configuration Fixtures

- `test_config` - Test configuration dictionary
- `test_checkpoint_config` - Checkpoint configuration

## Test Coverage Goals

### Current Coverage

Run coverage report to see current status:

```bash
pytest --cov=. --cov-report=term-missing
```

### Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Core climb detection | 90%+ |
| Scoring algorithms | 95%+ |
| Error handling | 85%+ |
| Data structures | 90%+ |
| Integration workflows | 75%+ |

## Continuous Integration

### GitHub Actions (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: pytest --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Debugging Tests

### Run Single Test

```bash
# Run specific test by name
pytest tests/unit/core/test_climb_scoring.py::TestBasicScore::test_basic_score_moderate_climb
```

### Drop into Debugger on Failure

```bash
# Use pytest's built-in debugger
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb
```

### Print Debug Information

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l
```

## Common Issues

### Import Errors

If you encounter import errors:

```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:/path/to/ca9"

# Or run from project root
cd /Volumes/usb1-drive/ca9
pytest
```

### Fixture Not Found

If fixture is not found:

1. Check that `conftest.py` is in the correct location
2. Ensure fixture is properly defined with `@pytest.fixture` decorator
3. Check that test is in the correct directory structure

### Test Discovery Issues

If tests aren't being discovered:

1. Ensure test files start with `test_` or end with `_test.py`
2. Ensure test functions start with `test_`
3. Check `pytest.ini` configuration

## Performance Optimization

### Skip Slow Tests During Development

```bash
# Skip slow tests
pytest -m "not slow"

# Run only fast unit tests
pytest -m "unit and not slow"
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Use Fixtures**: Leverage fixtures for common test data and setup
3. **Clear Assertions**: Use descriptive assertion messages
4. **Test Edge Cases**: Include tests for boundary conditions and edge cases
5. **Mock External Dependencies**: Use mocks for external services and file I/O
6. **Meaningful Names**: Use descriptive test function names that explain what is being tested
7. **Documentation**: Add docstrings to complex tests
8. **Cleanup**: Use fixtures for automatic cleanup of temporary files

## Contributing

When adding new tests:

1. Place unit tests in `tests/unit/<component>/`
2. Place integration tests in `tests/integration/`
3. Add appropriate markers (e.g., `@pytest.mark.unit`)
4. Use existing fixtures when possible
5. Add new fixtures to `conftest.py` if they'll be reused
6. Update this README if adding new test categories

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [Coverage.py](https://coverage.readthedocs.io/)

## Support

For questions or issues with tests, please:

1. Check this README
2. Review existing tests for examples
3. Check pytest documentation
4. Open an issue in the project repository
