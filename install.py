#!/usr/bin/env python3
"""
Climb Analyzer - Guided Installation

This script helps you choose and install the correct dependencies for your deployment mode.
Run this ONCE when setting up the climb analyzer for the first time.

For data setup (OSM files, elevation data), run: python data_setup.py
"""

import subprocess
import sys
from pathlib import Path


def print_header():
    """Print welcome header."""
    print("=" * 80)
    print("CLIMB ANALYZER - INSTALLATION WIZARD")
    print("=" * 80)
    print()
    print("This wizard will help you install the right dependencies for your use case.")
    print("This only needs to be run ONCE when first setting up the climb analyzer.")
    print()


def print_deployment_modes():
    """Print detailed information about deployment modes."""
    print("=" * 80)
    print("DEPLOYMENT MODE OPTIONS")
    print("=" * 80)
    print()

    print("📦 LOCAL MODE")
    print("-" * 40)
    print("Process OpenStreetMap data from local .pbf files")
    print()
    print("Best for:")
    print("  ✓ Large-scale analysis (entire states, countries)")
    print("  ✓ Offline processing")
    print("  ✓ Repeated analysis of same regions")
    print("  ✓ Complete control over data")
    print()
    print("Requirements:")
    print("  • Download planet OSM file (~70GB compressed, ~1.5TB uncompressed)")
    print("  • Create spatial index (~50GB)")
    print("  • ~200GB+ free disk space")
    print("  • C++ libraries: libosmium, libspatialindex")
    print()
    print("Dependencies installed:")
    print("  • osmium (Python bindings for libosmium C++ library)")
    print("  • rtree (spatial indexing with libspatialindex)")
    print()
    print("After installation, run: python data_setup.py")
    print("  to download OSM files and create spatial index")
    print()

    print("=" * 80)
    print()

    print("☁️  CLOUD MODE")
    print("-" * 40)
    print("Query data from Overpass API on-demand")
    print()
    print("Best for:")
    print("  ✓ Small regions (cities, counties)")
    print("  ✓ Quick one-off analysis")
    print("  ✓ No local storage requirements")
    print("  ✓ Always up-to-date OSM data")
    print()
    print("Requirements:")
    print("  • Active internet connection")
    print("  • Overpass API access (free, public)")
    print("  • Minimal disk space (~1GB for elevation cache)")
    print()
    print("Dependencies installed:")
    print("  • overpy (Overpass API client)")
    print()
    print("Limitations:")
    print("  • API rate limits (slower for large areas)")
    print("  • Network latency")
    print("  • May timeout on very large regions")
    print()

    print("=" * 80)
    print()


def check_system_libraries():
    """Check if required system libraries are installed (for local mode)."""
    print("\nChecking system libraries for LOCAL mode...")
    print("-" * 40)

    libraries_ok = True

    # Check for libosmium
    print("Checking for libosmium...")
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", "libosmium"],
            capture_output=True,
            check=False
        )
        if result.returncode == 0:
            print("  ✓ libosmium found")
        else:
            print("  ✗ libosmium NOT found")
            libraries_ok = False
    except FileNotFoundError:
        print("  ⚠️  pkg-config not found, cannot check for libosmium")
        libraries_ok = False

    # Check for libspatialindex
    print("Checking for libspatialindex...")
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", "spatialindex"],
            capture_output=True,
            check=False
        )
        if result.returncode == 0:
            print("  ✓ libspatialindex found")
        else:
            print("  ✗ libspatialindex NOT found")
            libraries_ok = False
    except FileNotFoundError:
        print("  ⚠️  pkg-config not found, cannot check for libspatialindex")

    print()

    if not libraries_ok:
        print("⚠️  System libraries missing for LOCAL mode")
        print()
        print("To install system libraries:")
        print()
        print("macOS (Homebrew):")
        print("  brew install libosmium spatialindex")
        print()
        print("Ubuntu/Debian:")
        print("  sudo apt-get update")
        print("  sudo apt-get install libosmium-dev libspatialindex-dev")
        print()
        print("Fedora/RHEL:")
        print("  sudo dnf install libosmium-devel spatialindex-devel")
        print()
        print("Windows:")
        print("  Consider using WSL2 with Ubuntu")
        print("  Or install via conda: conda install -c conda-forge libosmium rtree")
        print()

    return libraries_ok


def get_deployment_choice():
    """Get deployment mode choice from user."""
    while True:
        print("\nWhich deployment mode do you want to use?")
        print()
        print("  [1] LOCAL  - Process local OSM files (large-scale, offline)")
        print("  [2] CLOUD  - Query Overpass API (small-scale, online)")
        print()
        print("  [?] Show detailed comparison")
        print("  [q] Quit")
        print()

        choice = input("Enter choice [1/2/?/q]: ").strip().lower()

        if choice == "1":
            return "local"
        elif choice == "2":
            return "cloud"
        elif choice == "?":
            print()
            print_deployment_modes()
        elif choice == "q":
            print("\nInstallation cancelled.")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, ?, or q")


def install_dependencies(mode: str, check_libs: bool = True):
    """
    Install dependencies for the selected mode.

    Args:
        mode: Deployment mode ('local', 'cloud', or 'all')
        check_libs: Whether to check system libraries first
    """
    print()
    print("=" * 80)
    print(f"INSTALLING DEPENDENCIES - {mode.upper()} MODE")
    print("=" * 80)
    print()

    # Check system libraries for local mode
    if mode in ["local", "all"] and check_libs:
        libs_ok = check_system_libraries()
        if not libs_ok:
            print()
            proceed = input("System libraries missing. Continue anyway? [y/N]: ").strip().lower()
            if proceed != 'y':
                print("\nPlease install system libraries and run this script again.")
                print("Or choose CLOUD mode if you don't need local file processing.")
                sys.exit(1)

    # Prepare pip install command
    print(f"Installing climb-analyzer with [{mode}] dependencies...\n")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        f".[{mode}]"
    ]

    print(f"Running: {' '.join(cmd)}\n")
    print("-" * 80)

    try:
        result = subprocess.run(cmd, check=True)

        print()
        print("-" * 80)
        print()
        print("✓ Installation successful!")
        print()

        # Print next steps
        print_next_steps(mode)

        return True

    except subprocess.CalledProcessError as e:
        print()
        print("-" * 80)
        print()
        print(f"✗ Installation failed with error code {e.returncode}")
        print()
        print("Common issues:")
        print("  • Missing system libraries (for LOCAL mode)")
        print("  • Python version < 3.9")
        print("  • pip not installed or outdated")
        print()
        print("Try:")
        print("  python3 -m pip install --upgrade pip")
        print("  python3 install.py")
        print()
        return False


def print_next_steps(mode: str):
    """Print next steps after installation."""
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    if mode == "local":
        print("📦 LOCAL MODE SETUP")
        print("-" * 40)
        print("1. Run the data setup wizard to download OSM data:")
        print("     python data_setup.py")
        print()
        print("2. Download planet OSM file (or regional extract)")
        print("3. Create spatial index")
        print("4. Download elevation data (optional)")
        print()

    if mode == "cloud":
        print("☁️  CLOUD MODE SETUP")
        print("-" * 40)
        print("1. Run the data setup wizard to configure:")
        print("     python data_setup.py")
        print()
        print("2. Optionally download elevation data for better performance")
        print()

    print("=" * 80)
    print("RUNNING THE ANALYZER")
    print("=" * 80)
    print()
    print("Interactive mode:")
    print("  python climb_analyzer.py")
    print()
    print("Batch mode (using config.yaml):")
    print("  python climb_analyzer.py --batch --scope state")
    print()
    print("For more options:")
    print("  python climb_analyzer.py --help")
    print()


def verify_installation(mode: str):
    """Verify that the installation was successful."""
    print()
    print("Verifying installation...")
    print("-" * 40)

    # Check core imports
    print("Checking core dependencies...")
    try:
        import pandas
        import numpy
        import geopy
        import requests
        import tqdm
        print("  ✓ Core dependencies OK")
    except ImportError as e:
        print(f"  ✗ Missing core dependency: {e}")
        return False

    # Check mode-specific imports
    if mode == "local":
        print("Checking LOCAL mode dependencies...")
        try:
            import osmium
            import rtree
            print("  ✓ LOCAL mode dependencies OK")
        except ImportError as e:
            print(f"  ✗ Missing LOCAL dependency: {e}")
            if mode == "local":
                return False

    if mode == "cloud":
        print("Checking CLOUD mode dependencies...")
        try:
            import overpy
            print("  ✓ CLOUD mode dependencies OK")
        except ImportError as e:
            print(f"  ✗ Missing CLOUD dependency: {e}")
            if mode == "cloud":
                return False

    print()
    print("✓ All dependencies verified successfully!")
    return True


def main():
    """Main installation wizard."""
    print_header()

    # Check if already installed
    try:
        import climb_analyzer
        print("⚠️  climb-analyzer appears to be already installed.")
        print()
        reinstall = input("Reinstall/change deployment mode? [y/N]: ").strip().lower()
        if reinstall != 'y':
            print("\nInstallation cancelled.")
            print("To reconfigure data sources, run: python data_setup.py")
            return
        print()
    except ImportError:
        pass

    # Show deployment modes
    print_deployment_modes()

    # Get user choice
    mode = get_deployment_choice()

    # Install dependencies
    success = install_dependencies(mode)

    if success:
        # Verify installation
        verify_installation(mode)

        # Create default config.yaml with deployment type
        create_default_config(mode)

        print()
        print("=" * 80)
        print("INSTALLATION COMPLETE!")
        print("=" * 80)
        print()
        print(f"Deployment mode: {mode.upper()}")
        print()
        print("Next: Run the data setup wizard")
        print("  python data_setup.py")
        print()


def create_default_config(mode: str):
    """Create default config.yaml with deployment type."""
    config_file = Path("config.yaml")

    # Don't overwrite existing config
    if config_file.exists():
        print(f"\n✓ config.yaml already exists")
        return

    # Use the selected mode as deployment type
    deployment_type = mode

    config_content = f"""# Climb Analyzer Configuration
# Generated by install.py

# Deployment type: 'local' or 'cloud'
# local: Process local OSM .pbf files (requires planet file + spatial index)
# cloud: Query Overpass API on-demand (requires internet)
DEPLOYMENT_TYPE: {deployment_type}

# Overpass API settings (cloud mode only)
OVERPASS_API_URL: https://overpass-api.de/api/interpreter
OVERPASS_API_DELAY_SEC: 0.1

# Elevation API settings
TOPO_API_BASE_URL: https://api.opentopodata.org/v1/aster30m
ELEVATION_MAX_CONCURRENT: 2
ELEVATION_BATCH_SIZE: 100
ELEVATION_REQUEST_TIMEOUT_SEC: 45
ELEVATION_MAX_RETRIES: 3
ELEVATION_BACKOFF_FACTOR: 2.0

# Checkpoint settings
CHECKPOINT_INTERVAL_MIN: 5.0
CHECKPOINT_MILESTONES_PERC: [25, 50, 75, 100]

# Geocoding settings
GEOCODING_MAX_CONCURRENT: 8
GEOCODING_RETRY_ATTEMPTS: 2

# OSM Coverage (for batch mode)
# List of states or countries to analyze
# Examples:
#   OSM_COVERAGE: ["California", "Oregon", "Washington"]
#   OSM_COVERAGE: ["Switzerland", "Austria"]
OSM_COVERAGE: []

# Planet file path (local mode only)
# Path to your planet OSM .pbf file
# Example: planet-osm/planet-latest.osm.pbf
"""

    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"\n✓ Created default config.yaml (deployment type: {deployment_type})")
        print(f"  Edit {config_file} to customize settings")
    except Exception as e:
        print(f"\n⚠️  Could not create config.yaml: {e}")
        print(f"  You can create it manually later")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
