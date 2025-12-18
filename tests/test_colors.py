#!/usr/bin/env python3
"""
ANSI color utilities for test output formatting.
"""


class Colors:
    """ANSI escape codes for terminal colors."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    # Box drawing characters
    BOX_TL = '\u2554'  # ╔
    BOX_TR = '\u2557'  # ╗
    BOX_BL = '\u255a'  # ╚
    BOX_BR = '\u255d'  # ╝
    BOX_H = '\u2550'   # ═
    BOX_V = '\u2551'   # ║

    # Light box for inner sections
    LIGHT_TL = '\u250c'  # ┌
    LIGHT_TR = '\u2510'  # ┐
    LIGHT_BL = '\u2514'  # └
    LIGHT_BR = '\u2518'  # ┘
    LIGHT_H = '\u2500'   # ─
    LIGHT_V = '\u2502'   # │
    LIGHT_ML = '\u251c'  # ├
    LIGHT_MR = '\u2524'  # ┤

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.GREEN = ''
        cls.RED = ''
        cls.YELLOW = ''
        cls.CYAN = ''
        cls.BLUE = ''
        cls.MAGENTA = ''
        cls.WHITE = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls.UNDERLINE = ''
        cls.END = ''


def colorize(text: str, *codes: str) -> str:
    """Apply color codes to text."""
    return f"{''.join(codes)}{text}{Colors.END}"


def green(text: str) -> str:
    return colorize(text, Colors.GREEN)


def red(text: str) -> str:
    return colorize(text, Colors.RED)


def yellow(text: str) -> str:
    return colorize(text, Colors.YELLOW)


def cyan(text: str) -> str:
    return colorize(text, Colors.CYAN)


def blue(text: str) -> str:
    return colorize(text, Colors.BLUE)


def bold(text: str) -> str:
    return colorize(text, Colors.BOLD)


def dim(text: str) -> str:
    return colorize(text, Colors.DIM)


def print_pass(msg: str) -> None:
    """Print a passing test message in green."""
    print(f"{Colors.GREEN}[✓ PASS]{Colors.END} {msg}")


def print_fail(msg: str) -> None:
    """Print a failing test message in red."""
    print(f"{Colors.RED}[✗ FAIL]{Colors.END} {msg}")


def print_skip(msg: str) -> None:
    """Print a skipped test message in yellow."""
    print(f"{Colors.YELLOW}[○ SKIP]{Colors.END} {msg}")


def print_info(msg: str) -> None:
    """Print an info message in cyan."""
    print(f"{Colors.CYAN}[ℹ INFO]{Colors.END} {msg}")


def print_warn(msg: str) -> None:
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}[⚠ WARN]{Colors.END} {msg}")


def print_error(msg: str) -> None:
    """Print an error message in red."""
    print(f"{Colors.RED}[✗ ERROR]{Colors.END} {msg}")


def print_running(msg: str) -> None:
    """Print a running test message in blue."""
    print(f"{Colors.BLUE}[▶ RUN ]{Colors.END} {msg}")


def box_header(title: str, width: int = 64) -> str:
    """Create a double-line box header."""
    c = Colors
    inner_width = width - 2
    title_line = f"{c.BOX_V}  {title.center(inner_width - 2)}  {c.BOX_V}"
    return '\n'.join([
        f"{c.CYAN}{c.BOX_TL}{c.BOX_H * inner_width}{c.BOX_TR}{c.END}",
        f"{c.CYAN}{title_line}{c.END}",
        f"{c.CYAN}{c.BOX_BL}{c.BOX_H * inner_width}{c.BOX_BR}{c.END}",
    ])


def light_box(lines: list, width: int = 64) -> str:
    """Create a light single-line box with content."""
    c = Colors
    inner_width = width - 2
    result = [f"{c.LIGHT_TL}{c.LIGHT_H * inner_width}{c.LIGHT_TR}"]

    for i, line in enumerate(lines):
        if line.startswith('---'):
            result.append(f"{c.LIGHT_ML}{c.LIGHT_H * inner_width}{c.LIGHT_MR}")
        else:
            padded = line.ljust(inner_width)[:inner_width]
            result.append(f"{c.LIGHT_V}{padded}{c.LIGHT_V}")

    result.append(f"{c.LIGHT_BL}{c.LIGHT_H * inner_width}{c.LIGHT_BR}")
    return '\n'.join(result)


def divider(char: str = '─', width: int = 64) -> str:
    """Create a divider line."""
    return char * width


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_percentage_diff(actual: float, expected: float) -> str:
    """Format the percentage difference between actual and expected values."""
    if expected == 0:
        return "N/A"
    diff = abs(actual - expected) / expected * 100
    return f"{diff:.1f}%"
