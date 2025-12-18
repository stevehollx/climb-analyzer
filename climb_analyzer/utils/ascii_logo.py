#!/usr/bin/env python3
"""
80s Retro ASCII Logo for Climb Analyzer
Features 3D text, mountains, and a bike climbing
"""

from climb_analyzer.utils.formatting import Colors

# Import version information
try:
    from __version__ import __version__
except ImportError:
    __version__ = "unknown"


def print_logo():
    """Print the 80s retro ASCII logo with mountains and bike."""

    # Color scheme: Cyan for borders, Cyan/Yellow for text
    # Box width: 69 chars content + 2 for ║ = 71, + 10 padding = 81 total
    logo = f"""
{Colors.CYAN}          ╔═════════════════════════════════════════════════════════════════════╗
          ║{Colors.RESET}                                                                     {Colors.CYAN}║
          ║{Colors.BRIGHT_CYAN}    ██████╗██╗     ██╗███╗   ███╗██████╗                             {Colors.CYAN}║
          ║{Colors.BRIGHT_CYAN}   ██╔════╝██║     ██║████╗ ████║██╔══██╗                            {Colors.CYAN}║
          ║{Colors.BRIGHT_CYAN}   ██║     ██║     ██║██╔████╔██║██████╔╝                            {Colors.CYAN}║
          ║{Colors.BRIGHT_CYAN}   ██║     ██║     ██║██║╚██╔╝██║██╔══██╗                            {Colors.CYAN}║
          ║{Colors.BRIGHT_CYAN}   ╚██████╗███████╗██║██║ ╚═╝ ██║██████╔╝                            {Colors.CYAN}║
          ║{Colors.BRIGHT_CYAN}    ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝╚═════╝                             {Colors.CYAN}║
          ║{Colors.RESET}                                                                     {Colors.CYAN}║
          ║{Colors.YELLOW}    █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗██████╗   {Colors.CYAN}║
          ║{Colors.YELLOW}   ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗  {Colors.CYAN}║
          ║{Colors.YELLOW}   ███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  ██████╔╝  {Colors.CYAN}║
          ║{Colors.YELLOW}   ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗  {Colors.CYAN}║
          ║{Colors.YELLOW}   ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗██║  ██║  {Colors.CYAN}║
          ║{Colors.YELLOW}   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝  {Colors.CYAN}║
          ║{Colors.RESET}                                                                     {Colors.CYAN}║
          ║                Find Epic Climbs in OpenStreetMap                    {Colors.CYAN}║
          ║{Colors.GRAY}                        Version {__version__}{' ' * (25 - len(__version__))}            {Colors.CYAN}║
          ╚═════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.BRIGHT_CYAN}                                     /\\                    /\\
                                    /  \\                  /  \\
                                   /    \\                /    \\
{Colors.CYAN}                                  /      \\              /      \\
                                 /        \\            /        \\
{Colors.GRAY}                                /          \\          /          \\
                               /            \\________/            \\{Colors.RESET}
{Colors.DIM}                           ___/____________________________________\\
{Colors.YELLOW}                                  __o
                                _ \\<_
                               (_)/(_)
{Colors.DIM}                       ════════════════════════════════════════════════{Colors.RESET}
"""
    print(logo)


def print_small_logo():
    """Print a compact version of the logo for smaller displays."""

    # Build logo with proper escape handling
    logo = f"""{Colors.BRIGHT_CYAN}   ____ _ _           _
  / ___| (_)_ __ ___ | |__
 | |   | | | '_ ` _ \\| '_ \\
 | |___| | | | | | | | |_) |
  \\____|_|_|_| |_| |_|_.__/
{Colors.YELLOW}    / \\   _ __   __ _| |_   _ _______ _ __
   / _ \\ | '_ \\ / _` | | | | |_  / _ \\ '__|
  / ___ \\| | | | (_| | | |_| |/ /  __/ |
 /_/   \\_\\_| |_|\\__,_|_|\\__, /___\\___|_|
                        |___/
{Colors.RESET}
  Find Epic Climbs in OpenStreetMap
{Colors.GRAY}  Version {__version__}{Colors.RESET}
"""
    print(logo)


if __name__ == "__main__":
    # Test the logos
    print("Full Logo:")
    print_logo()

    print("\n\nSmall Logo:")
    print_small_logo()
