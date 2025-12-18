#!/usr/bin/env python3
"""
Entry point for running climb_analyzer as a module.

Usage:
    python -m climb_analyzer
    python -m climb_analyzer --help
    python -m climb_analyzer -r Vermont
"""

from climb_analyzer.cli import main

if __name__ == '__main__':
    main()
