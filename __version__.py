#!/usr/bin/env python3
"""
Version information for Climb Analyzer.

The canonical version is defined in pyproject.toml and should be updated there.
This file reads it at runtime to make it available to the application.
"""

import sys
from pathlib import Path

# Use tomllib (built-in Python 3.11+) or tomli (external package for <3.11)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def get_version() -> str:
    """
    Read version from pyproject.toml.

    Returns:
        Version string (e.g., "2.2.3")
    """
    if tomllib is None:
        # If tomli is not installed and Python < 3.11, return fallback
        return "2.3.0"

    try:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        return pyproject_data["project"]["version"]
    except Exception:
        # Fallback if pyproject.toml can't be read
        return "2.3.0"


__version__ = get_version()
__version_info__ = tuple(int(x) for x in __version__.split(".") if x.isdigit())
