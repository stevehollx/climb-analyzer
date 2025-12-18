# Contributing to Climb Analyzer

Thank you for your interest in contributing to Climb Analyzer! This document provides guidelines for contributing to the project.

## Main Author
Steve Holl - smholl+ca@gmail.com

## Getting Started

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/climb-analyzer.git
   cd climb-analyzer
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

## How to Contribute

### Reporting Issues
- Check existing issues before creating a new one
- Use issue templates when available
- Provide detailed reproduction steps
- Include system information (OS, Python version, etc.)

### Submitting Changes
1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes following our code style guidelines
3. Add or update tests as needed
4. Update documentation if applicable
5. Commit with clear, descriptive messages:
   ```bash
   git commit -m "feat: add support for new elevation dataset"
   ```
6. Push to your fork and submit a pull request

### Pull Request Guidelines
- PRs should be focused on a single feature or fix
- Include a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Update CHANGELOG.md if appropriate

## Code Style

### Python Code
- Follow PEP 8 guidelines
- Use Black formatter with line length 100:
  ```bash
  black --line-length 100 .
  ```
- Run Ruff linter:
  ```bash
  ruff check .
  ```
- Type hints are encouraged for new code

### Commit Messages
Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks

## Testing

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires Docker)
pytest tests/integration/

# All tests
pytest
```

### Writing Tests
- Add unit tests for new functions/methods
- Include integration tests for new features
- Test edge cases and error conditions
- Maintain test coverage above 80%

## Documentation

### Code Documentation
- Add docstrings to all public functions/classes
- Use Google-style docstrings:
  ```python
  def calculate_grade(elevation_gain: float, distance: float) -> float:
      """Calculate climb grade percentage.

      Args:
          elevation_gain: Elevation gain in meters
          distance: Horizontal distance in meters

      Returns:
          Grade as a percentage

      Raises:
          ValueError: If distance is zero
      """
  ```

### User Documentation
- Update README.md for significant changes
- Add/update markdown files in docs/ for new features
- Include examples for complex features

## Project Structure

```
climb-analyzer/
├── climb_analyzer/        # Core application code
│   ├── core/              # Core processing logic
│   ├── data/              # Data management
│   ├── processing/        # Analysis algorithms
│   └── utils/             # Utility functions
├── utils/                 # Standalone utilities
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── docs/                  # Documentation
├── gui/                   # Web interface (Next.js)
└── scripts/               # Utility scripts
```

## Development Workflow

### Local Mode Development
For changes to local OSM processing:
1. Download test OSM data (Luxembourg is small and good for testing)
2. Test with various regions and surface filters
3. Verify checkpoint/resume functionality

### Cloud Mode Development
For changes to Overpass API integration:
1. Test with small bounding boxes first
2. Respect API rate limits
3. Handle timeouts gracefully

### GUI Development
For web interface changes:
1. Navigate to `gui/` directory
2. Install dependencies: `npm install`
3. Run development server: `npm run dev`
4. Test on multiple browsers

## Security

### Important Security Notes
- **NEVER** commit secrets, API keys, or credentials
- Use environment variables for sensitive configuration
- Review `.gitignore` before committing
- Report security vulnerabilities privately to smholl+ca@gmail.com

### Removed Hardcoded Credentials
The project previously had hardcoded GitHub App credentials. These have been removed.
To use cloud cache features:
1. Copy `.env.example` to `.env`
2. Add your GitHub App credentials
3. Never commit the `.env` file

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release branch
4. Run full test suite
5. Build Docker images
6. Tag release and push

## Questions?

- Open a discussion in GitHub Discussions
- Email the maintainer at smholl+ca@gmail.com
- Check existing documentation in docs/

## License

By contributing, you agree that your contributions will be licensed under the MIT License.