# How to Contribute

Thank you for your interest in contributing to Climb Analyzer!

## Ways to Contribute

### 1. Contribute Climb Data

The easiest way to help is by running analyses and contributing to the cloud cache:

```bash
# Run a clean analysis for a region not yet in the cache
./climb-analyzer -r "Region Name"

# When prompted, choose to contribute
> This is a clean analysis - contribute to cloud cache? (yes/no): yes
```

See [Cloud Cache Contributions](cloud-cache-contributions.md) for details.

### 2. Report Bugs

Found a problem? Open an issue:

1. Go to [GitHub Issues](https://github.com/stevehollx/climb-analyzer/issues)
2. Click "New Issue"
3. Include:
   - What you were trying to do
   - What happened
   - Error messages (if any)
   - Your environment (OS, Docker version)

### 3. Suggest Features

Have an idea? We'd love to hear it:

1. Open an issue with "Feature Request" label
2. Describe the feature
3. Explain the use case

### 4. Contribute Code

#### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer

# Run setup
./climb-analyzer setup

# Make changes
# ...

# Test your changes
./climb-analyzer -r "Rhode Island"  # Small, fast test
```

#### Code Guidelines

- **Python**: Follow PEP 8
- **TypeScript**: Use existing patterns in `gui/`
- **Comments**: Explain "why", not "what"
- **Tests**: Add tests for new functionality

#### Submit a Pull Request

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Commit: `git commit -m "Add my feature"`
5. Push: `git push origin feature/my-feature`
6. Open a Pull Request

### 5. Improve Documentation

Documentation improvements are always welcome:

- Fix typos
- Clarify confusing sections
- Add examples
- Translate to other languages

## Development Areas

### Core Analysis Engine

`climb_analyzer/` - Python climb detection and analysis

### Web GUI

`gui/` - Next.js/TypeScript web interface

### Elevation System

`opentopodata/`, `dem_downloaders.py` - Elevation data handling

### Cloud Cache

`utils/cloud_cache.py` - GitHub integration for data sharing

## Testing

### Run Tests

```bash
# Python tests
pytest tests/

# GUI tests
cd gui && npm test
```

### Test a Small Region

```bash
# Quick test with Rhode Island
./climb-analyzer -r "Rhode Island"
```

## Code of Conduct

- Be respectful
- Be constructive
- Help others learn

## Questions?

- Open an issue for questions
- Check existing issues first
- Be patient - this is a volunteer project

---

Next: [Cloud Cache Contributions](cloud-cache-contributions.md)
