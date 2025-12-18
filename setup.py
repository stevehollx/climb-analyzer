#!/usr/bin/env python3
"""
Minimal setup.py shim for backward compatibility.

Modern Python packaging uses pyproject.toml for configuration.
This file exists only for compatibility with older tools and workflows.

For new installations, use:
    pip install .

For development:
    pip install -e .[dev]
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This is just a shim for backward compatibility
setup()
