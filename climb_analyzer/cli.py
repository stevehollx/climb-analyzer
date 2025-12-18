#!/usr/bin/env python3
"""
Command-line interface for Climb Analyzer.

This module provides the main CLI entry point, importing from the core
analysis engine.

Future: Extract CLI parsing into this module for better separation of concerns.
"""

# Import main entry point from core analysis engine
from climb_analyzer.engine import main

# Re-export main for backwards compatibility
__all__ = ['main']

if __name__ == '__main__':
    main()
