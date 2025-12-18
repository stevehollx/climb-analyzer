#!/usr/bin/env python3
"""
Terminal output formatting utilities for Climb Analyzer.

Provides consistent, professional formatting for all terminal output with:
- Box-drawing characters for clean section headers
- Subtle color support (can be disabled)
- Standardized icons (✓, ✗, ⚠️)
- Consistent spacing and width
"""

import os
import sys
from typing import Optional


# Terminal width constants
STANDARD_WIDTH = 80  # Standard output width
TABLE_WIDTH = None   # No limit for data tables (use terminal width)

# Color codes (ANSI escape sequences)
class Colors:
    """ANSI color codes for terminal output."""
    # Check if colors are supported
    COLORS_ENABLED = (
        sys.stdout.isatty() and
        os.environ.get('TERM') != 'dumb' and
        os.environ.get('NO_COLOR') is None
    )

    if COLORS_ENABLED:
        RESET = '\033[0m'
        BOLD = '\033[1m'
        DIM = '\033[2m'

        # Subtle colors
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        RED = '\033[31m'
        BLUE = '\033[34m'
        CYAN = '\033[36m'
        GRAY = '\033[90m'

        # Bright variants (more subtle)
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_RED = '\033[91m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_CYAN = '\033[96m'
    else:
        # No colors if not supported
        RESET = BOLD = DIM = ''
        GREEN = YELLOW = RED = BLUE = CYAN = GRAY = ''
        BRIGHT_GREEN = BRIGHT_YELLOW = BRIGHT_RED = BRIGHT_BLUE = BRIGHT_CYAN = ''


# Box-drawing characters
class Box:
    """Box-drawing characters for headers."""
    HORIZONTAL = '─'
    VERTICAL = '│'
    TOP_LEFT = '┌'
    TOP_RIGHT = '┐'
    BOTTOM_LEFT = '└'
    BOTTOM_RIGHT = '┘'


def print_banner(text: str, width: int = STANDARD_WIDTH, spacing_before: int = 2) -> None:
    """
    Print a major section banner with box-drawing characters.

    Used for main application sections (highest level).

    Args:
        text: Banner text to display (will be converted to uppercase)
        width: Width of the banner (default: STANDARD_WIDTH)
        spacing_before: Number of blank lines before banner (default: 2)

    Example:
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ CLIMB ANALYZER - SETUP WIZARD                                                │
        └──────────────────────────────────────────────────────────────────────────────┘
    """
    print('\n' * spacing_before, end='')

    # Create the box
    top_line = Box.TOP_LEFT + Box.HORIZONTAL * (width - 2) + Box.TOP_RIGHT
    bottom_line = Box.BOTTOM_LEFT + Box.HORIZONTAL * (width - 2) + Box.BOTTOM_RIGHT

    # Center and uppercase the text
    text = text.upper()
    text_padded = f" {text} ".ljust(width - 2)
    middle_line = Colors.CYAN + Box.VERTICAL + Colors.RESET + Colors.BOLD + text_padded + Colors.RESET + Colors.CYAN + Box.VERTICAL + Colors.RESET

    print(Colors.CYAN + top_line + Colors.RESET)
    print(middle_line)
    print(Colors.CYAN + bottom_line + Colors.RESET)
    print()


def print_header(text: str, width: int = STANDARD_WIDTH, spacing_before: int = 1) -> None:
    """
    Print a subsection header with box-drawing characters.

    Used for subsections within a workflow (second level).

    Args:
        text: Header text to display
        width: Width of the header (default: STANDARD_WIDTH)
        spacing_before: Number of blank lines before header (default: 1)

    Example:
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ Checking Docker Dependencies                                                 │
        └──────────────────────────────────────────────────────────────────────────────┘
    """
    print('\n' * spacing_before, end='')

    # Create the box
    top_line = Box.TOP_LEFT + Box.HORIZONTAL * (width - 2) + Box.TOP_RIGHT
    bottom_line = Box.BOTTOM_LEFT + Box.HORIZONTAL * (width - 2) + Box.BOTTOM_RIGHT

    # Pad the text
    text_padded = f" {text} ".ljust(width - 2)
    middle_line = Colors.BLUE + Box.VERTICAL + Colors.RESET + text_padded + Colors.BLUE + Box.VERTICAL + Colors.RESET

    print(Colors.BLUE + top_line + Colors.RESET)
    print(middle_line)
    print(Colors.BLUE + bottom_line + Colors.RESET)
    print()


def print_section_simple(text: str, spacing_before: int = 1) -> None:
    """
    Print a simple section label (no box).

    Used for minor subsections or labels.

    Args:
        text: Section text to display
        spacing_before: Number of blank lines before section (default: 1)

    Example:
        Step 1: Extracting all road ways from bounding box...
    """
    print('\n' * spacing_before, end='')
    print(f"{Colors.BOLD}{text}{Colors.RESET}")


def print_success(text: str, indent: int = 0) -> None:
    """
    Print a success message with checkmark.

    Args:
        text: Success message
        indent: Number of spaces to indent (default: 0)

    Example:
        ✓ Docker found: Docker version 28.5.0
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{Colors.GREEN}✓{Colors.RESET} {text}")


def print_warning(text: str, indent: int = 0) -> None:
    """
    Print a warning message with warning symbol.

    Args:
        text: Warning message
        indent: Number of spaces to indent (default: 0)

    Example:
        ⚠️  Low disk space available
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{Colors.YELLOW}⚠️ {Colors.RESET} {text}")


def print_error(text: str, indent: int = 0) -> None:
    """
    Print an error message with X symbol.

    Args:
        text: Error message
        indent: Number of spaces to indent (default: 0)

    Example:
        ✗ Docker NOT found
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str, indent: int = 0) -> None:
    """
    Print an informational message (no icon).

    Args:
        text: Info message
        indent: Number of spaces to indent (default: 0)

    Example:
        Launching setup wizard...
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{Colors.GRAY}{text}{Colors.RESET}")


def print_dim(text: str, indent: int = 0) -> None:
    """
    Print technical/verbose info in dimmed text.

    Use for non-essential output like:
    - Configuration details (checkpoint config, filters, etc.)
    - Technical parameters (batch sizes, memory settings)
    - Debug messages
    - Internal file paths and temp file names

    Args:
        text: Technical/verbose message
        indent: Number of spaces to indent (default: 0)

    Example:
        Using 30.0km chunks for memory efficiency
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{Colors.DIM}{text}{Colors.RESET}")


def print_separator(width: int = STANDARD_WIDTH, char: str = '─') -> None:
    """
    Print a horizontal separator line.

    Args:
        width: Width of separator (default: STANDARD_WIDTH)
        char: Character to use for separator (default: '─')

    Example:
        ────────────────────────────────────────────────────────────────────────────────
    """
    print(Colors.GRAY + char * width + Colors.RESET)


def print_key_value(key: str, value: str, indent: int = 0, key_width: int = 25) -> None:
    """
    Print a key-value pair with aligned values.

    Args:
        key: Key name
        value: Value to display
        indent: Number of spaces to indent (default: 0)
        key_width: Width to align keys to (default: 25)

    Example:
        Available disk space:     1180.4 GB / 1832.7 GB
    """
    indent_str = ' ' * indent
    key_padded = f"{key}:".ljust(key_width)
    print(f"{indent_str}{key_padded} {value}")


def print_list_item(text: str, indent: int = 2, bullet: str = '•') -> None:
    """
    Print a list item with bullet point.

    Args:
        text: List item text
        indent: Number of spaces to indent (default: 2)
        bullet: Bullet character (default: '•')

    Example:
        • OSM data: Queried from Overpass API on-demand
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{bullet} {text}")


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "2.5 GB", "150 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def disable_colors() -> None:
    """Disable color output (useful for logging to files)."""
    Colors.COLORS_ENABLED = False
    Colors.RESET = Colors.BOLD = Colors.DIM = ''
    Colors.GREEN = Colors.YELLOW = Colors.RED = Colors.BLUE = Colors.CYAN = Colors.GRAY = ''
    Colors.BRIGHT_GREEN = Colors.BRIGHT_YELLOW = Colors.BRIGHT_RED = ''
    Colors.BRIGHT_BLUE = Colors.BRIGHT_CYAN = ''


def enable_colors() -> None:
    """Re-enable color output."""
    if sys.stdout.isatty() and os.environ.get('TERM') != 'dumb':
        Colors.COLORS_ENABLED = True
        Colors.RESET = '\033[0m'
        Colors.BOLD = '\033[1m'
        Colors.DIM = '\033[2m'
        Colors.GREEN = '\033[32m'
        Colors.YELLOW = '\033[33m'
        Colors.RED = '\033[31m'
        Colors.BLUE = '\033[34m'
        Colors.CYAN = '\033[36m'
        Colors.GRAY = '\033[90m'
        Colors.BRIGHT_GREEN = '\033[92m'
        Colors.BRIGHT_YELLOW = '\033[93m'
        Colors.BRIGHT_RED = '\033[91m'
        Colors.BRIGHT_BLUE = '\033[94m'
        Colors.BRIGHT_CYAN = '\033[96m'
