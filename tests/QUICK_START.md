# Quick Start Guide - Testing

Get started with running tests in under 5 minutes!

## 1. Install Dependencies

```bash
pip install pytest pytest-cov
```

## 2. Run All Tests

```bash
# From project root
cd /Volumes/usb1-drive/ca9
pytest
```

## 3. Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# High-priority climb detection/scoring tests
pytest -m "climb_detection or climb_scoring"
```

## 4. Quick Commands

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest -v` | Verbose output |
| `pytest -x` | Stop on first failure |
| `pytest -k "climb"` | Run tests matching "climb" |
| `pytest --lf` | Run last failed tests |
| `pytest --cov` | Show coverage report |

## 5. Test What Matters

### For Climb Detection Development

```bash
pytest tests/unit/core/test_climb_detection.py -v
```

### For Scoring Algorithm Development

```bash
pytest tests/unit/core/test_climb_scoring.py -v
```

### For Error Handling Development

```bash
pytest tests/unit/utils/ -v
```

## 6. Common Test Scenarios

### Running Single Test

```bash
pytest tests/unit/core/test_climb_scoring.py::TestBasicScore::test_basic_score_moderate_climb
```

### Running Tests in Parallel (Faster!)

```bash
pip install pytest-xdist
pytest -n auto
```

### Getting Coverage Report

```bash
pytest --cov=climb_analyzer --cov-report=html
open htmlcov/index.html
```

## 7. Interpreting Results

### Success ✓

```
tests/unit/core/test_climb_detection.py::TestDistanceCalculation::test_same_point_distance PASSED
```

### Failure ✗

```
tests/unit/core/test_climb_detection.py::TestDistanceCalculation::test_known_distance FAILED

def test_known_distance(self):
    distance = self.calculate_distance(35.0, -82.0, 36.0, -82.0)
>   assert 110 < distance < 112
E   assert 111.195 < 112

FAILED - AssertionError
```

## 8. Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [conftest.py](conftest.py) for available fixtures
- Review existing tests for examples
- Add your own tests!

## Need Help?

**Common Issues:**

1. **Import errors**: Make sure you're running from project root
2. **Fixture not found**: Check that `conftest.py` exists in `tests/` directory
3. **Tests not discovered**: Ensure test files start with `test_`

**Quick Fixes:**

```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Clear pytest cache
pytest --cache-clear

# Verbose output for debugging
pytest -vv -s
```
