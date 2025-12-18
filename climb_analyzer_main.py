#!/usr/bin/env python3
"""
Climb Analyzer Main Entry Point

Backwards-compatible wrapper that imports from the refactored package structure.

Usage:
    python climb_analyzer_main.py
    python climb_analyzer_main.py --help
"""

from climb_analyzer.cli import main

if __name__ == "__main__":
    main()
