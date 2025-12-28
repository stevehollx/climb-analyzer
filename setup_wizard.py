#!/usr/bin/env python3
"""
Climb Analyzer - Master Setup Wizard

This is the MAIN entry point for first-time setup.
Checks all prerequisites, sets up Docker containers, and guides through data configuration.

Run this ONCE when first setting up the climb analyzer:
    python setup_wizard.py
"""

import os
import shutil
import subprocess
import sys

# Import centralized formatting utilities
from climb_analyzer.utils.formatting import (
    print_banner,
    print_error,
    print_header,
    print_info,
    print_list_item,
    print_section_simple,
    print_success,
    print_warning,
)

# REMOVED: NASA Earthdata credential functions
# As of December 2025, NASA LP DAAC Data Pool was retired.
# SRTM now uses OpenTopography S3 (public, no auth needed)
# AW3D30 uses JAXA public FTP (no auth needed)
# All datasets are now public - no credentials required!


def check_command(cmd: str) -> bool:
    """Check if a command exists."""
    return shutil.which(cmd) is not None


def get_command_version(cmd: list) -> str:
    """Get version of a command."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip() or result.stderr.strip()
    except Exception:
        return "unknown"


def check_docker_dependencies():
    """
    Check for Docker and Docker Compose on the host system.

    Returns:
        tuple: (docker_ok, compose_ok, messages)
    """
    print_header("Checking Docker Dependencies on Host System")

    messages = []
    docker_ok = False
    compose_ok = False

    # Check for Docker
    print("Checking for Docker...")
    if check_command("docker"):
        version = get_command_version(["docker", "--version"])
        print_success(f"Docker found: {version}", indent=2)
        docker_ok = True

        # Check if Docker daemon is running
        result = subprocess.run(["docker", "info"], capture_output=True, check=False)
        if result.returncode != 0:
            print_warning("Docker is installed but daemon is not running", indent=2)
            messages.append("Start Docker: sudo systemctl start docker")
            docker_ok = False
    else:
        print_error("Docker NOT found", indent=2)
        messages.append("Install Docker: https://docs.docker.com/get-docker/")

    # Check for Docker Compose
    print("\nChecking for Docker Compose...")

    # Try docker compose (v2, plugin)
    result = subprocess.run(
        ["docker", "compose", "version"], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        version = result.stdout.strip()
        print_success(f"Docker Compose found: {version}", indent=2)
        compose_ok = True
    # Try docker-compose (v1, standalone)
    elif check_command("docker-compose"):
        version = get_command_version(["docker-compose", "--version"])
        print_success(f"Docker Compose found: {version}", indent=2)
        compose_ok = True
    else:
        print_error("Docker Compose NOT found", indent=2)
        messages.append("Install Docker Compose: https://docs.docker.com/compose/install/")

    return docker_ok, compose_ok, messages


def check_disk_space():
    """Check available disk space."""
    print_header("Checking Disk Space")

    try:
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)

        print(f"Available disk space: {free_gb:.1f} GB / {total_gb:.1f} GB")

        if free_gb < 5:
            print_warning("Warning: Less than 5 GB free", indent=2)
            print("  Recommended: At least 20 GB for local mode with state-level analysis")
            return False
        elif free_gb < 20:
            print_warning("Low disk space for large-scale analysis", indent=2)
            print("  Recommended: At least 20 GB for state-level analysis")
            return True
        else:
            print_success("Sufficient disk space available", indent=2)
            return True
    except Exception as e:
        print_warning(f"Could not check disk space: {e}", indent=2)
        return True


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Python version: {version_str}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error("Python 3.9+ required", indent=2)
        return False
    else:
        print_success("Python version OK", indent=2)
        return True


def get_docker_compose_command():
    """
    Detect which docker compose command to use.

    Returns:
        list: Command to use (e.g., ['docker', 'compose'] or ['docker-compose'])
    """
    # Try docker compose (v2, recommended)
    result = subprocess.run(["docker", "compose", "version"], capture_output=True, check=False)
    if result.returncode == 0:
        return ["docker", "compose"]

    # Fall back to docker-compose (v1)
    if check_command("docker-compose"):
        return ["docker-compose"]

    return None


def stop_existing_opentopodata():
    """Stop any existing opentopodata containers to avoid port conflicts."""
    print("Checking for existing opentopodata containers...")

    try:
        # Find containers using port 5000
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "publish=5000",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            containers = result.stdout.strip().split("\n")
            for container in containers:
                if container:
                    print(f"  Stopping existing container: {container}")
                    subprocess.run(["docker", "stop", container], capture_output=True, check=False)
                    subprocess.run(["docker", "rm", container], capture_output=True, check=False)
            print_success("Stopped existing containers\n", indent=2)
        else:
            print("  No existing containers on port 5000\n")

    except Exception as e:
        print(f"  ⚠️  Could not check for existing containers: {e}\n")


def build_docker_images(deployment_mode="local", containers=None):
    """
    Build Docker images based on deployment mode.

    Args:
        deployment_mode: "local" or "cloud"
        containers: Optional list of specific containers to build. If None, builds based on mode.
    """
    print_header("Building Climb Analyzer Docker Image")

    # Determine what to build
    if containers:
        build_list = containers
    elif deployment_mode == "cloud":
        build_list = ["climb-analyzer"]
    else:
        # For LOCAL, only build climb-analyzer initially
        # opentopodata will be built later after elevation data is ready
        build_list = ["climb-analyzer"]

    # Check if image already exists
    compose_cmd = get_docker_compose_command()
    if compose_cmd:
        try:
            result = subprocess.run(
                ["docker", "images", "-q", "climb-analyzer"],
                capture_output=True,
                text=True,
                check=False,
            )
            image_exists = bool(result.stdout.strip())

            if image_exists:
                print_success("climb-analyzer image already exists", indent=0)
                response = input("Rebuild image? [y/N]: ").strip().lower()
                if response != "y":
                    print("Using existing image")
                    return True
                # If user said yes to rebuild, skip the second prompt and go straight to building
                print()
            else:
                # Image doesn't exist, ask if they want to build
                print()
                response = input("Build climb-analyzer image now? [Y/n]: ").strip().lower()
                if response and response != "y":
                    print("Skipping Docker build. You can build later with:")
                    print(f"  docker compose build {' '.join(build_list)}")
                    return False
        except Exception:
            # On error, ask if they want to build
            print()
            response = input("Build climb-analyzer image now? [Y/n]: ").strip().lower()
            if response and response != "y":
                print("Skipping Docker build. You can build later with:")
                print(f"  docker compose build {' '.join(build_list)}")
                return False

    print("\nBuilding images (this may take 5-10 minutes)...\n")

    # Get the right docker compose command
    compose_cmd = get_docker_compose_command()
    if not compose_cmd:
        print_error("Could not find docker compose command", indent=0)
        return False

    try:
        # Build the specified containers
        result = subprocess.run(compose_cmd + ["build"] + build_list, check=False)

        if result.returncode == 0:
            print("\n✓ Docker images built successfully!")
            return True
        else:
            print("\n✗ Docker build failed")
            print("\nTry building manually with:")
            print(f"  {' '.join(compose_cmd)} build")
            return False

    except Exception as e:
        print(f"\n✗ Error building images: {e}")
        return False


def build_opentopodata_container():
    """Build the opentopodata container after elevation data is ready."""
    print_header("Building OpenTopoData Container")

    print("Now that elevation data is ready, building opentopodata server...")
    print()

    compose_cmd = get_docker_compose_command()
    if not compose_cmd:
        print_error("Could not find docker compose command", indent=0)
        return False

    try:
        result = subprocess.run(compose_cmd + ["build", "opentopodata"], check=False)

        if result.returncode == 0:
            print("\n✓ OpenTopoData container built successfully!")
            return True
        else:
            print("\n✗ OpenTopoData build failed")
            print("\nTry building manually with:")
            print("  docker compose build opentopodata")
            return False

    except Exception as e:
        print(f"\n✗ Error building opentopodata: {e}")
        return False


def clone_opentopodata():
    """Clone OpenTopoData repository if it doesn't exist."""
    import os

    if os.path.exists("opentopodata"):
        # Verify it's a valid clone with Makefile
        if os.path.exists("opentopodata/Makefile"):
            print_success("OpenTopoData repository already exists", indent=0)
            return True
        else:
            print("⚠️  opentopodata directory exists but appears incomplete (no Makefile)")
            print("   Removing and re-cloning...")
            import shutil

            try:
                shutil.rmtree("opentopodata")
            except Exception as e:
                print(f"✗ Could not remove incomplete directory: {e}")
                return False

    print("Cloning OpenTopoData repository...")
    try:
        result = subprocess.run(
            ["git", "clone", "https://github.com/ajnisbet/opentopodata.git"],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print_success("OpenTopoData cloned successfully", indent=0)
            # Verify Makefile exists after clone
            if os.path.exists("opentopodata/Makefile"):
                return True
            else:
                print("✗ Clone succeeded but Makefile not found")
                print(f"   Directory contents: {os.listdir('opentopodata')[:10]}")
                return False
        else:
            print(f"✗ Failed to clone OpenTopoData: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error cloning OpenTopoData: {e}")
        return False


def build_and_start_opentopodata():
    """Build and start the opentopodata container using make commands."""
    import os
    import time

    # Check if opentopodata-server container exists (running or stopped)
    check_result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",  # Check all containers, not just running ones
            "--filter",
            "name=opentopodata-server",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if check_result.stdout.strip():
        # Check if it's running or stopped
        running_check = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=opentopodata-server",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if running_check.stdout.strip():
            print_success("OpenTopoData server is already running", indent=0)
        else:
            print_warning("OpenTopoData server container exists but is stopped", indent=0)

        print()
        response = input("Would you like to restart the server? [y/N]: ").strip().lower()

        if response == "y":
            print("\nStopping and removing existing OpenTopoData container...")
            subprocess.run(
                ["docker", "stop", "opentopodata-server"],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["docker", "rm", "opentopodata-server"],
                capture_output=True,
                check=False,
            )
            print_success("Removed existing container", indent=0)
            print()
        else:
            print("Skipping OpenTopoData setup")
            return True

    print("Building and starting OpenTopoData container...")
    print()

    # First, ensure the repository is cloned
    if not clone_opentopodata():
        return False

    # Skip config scanning during initial setup - will be done automatically when data is downloaded
    # The config will be created with test data or updated later
    original_dir = os.getcwd()

    # Change to opentopodata directory
    try:
        os.chdir("opentopodata")

        # Check platform for correct build target
        import platform

        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        build_target = "build-m1" if is_apple_silicon else "build"

        print(f"Building OpenTopoData ({build_target})...")
        result = subprocess.run(
            ["make", build_target],
            check=False,
            capture_output=False,  # Show output for debugging
        )

        if result.returncode != 0:
            print("\n✗ OpenTopoData build failed")
            print("\nTry manually with:")
            print(f"  cd opentopodata && make {build_target}")
            return False

        print("\n✓ OpenTopoData built successfully!")

        # Read version from VERSION file
        version = "latest"
        if os.path.exists("VERSION"):
            with open("VERSION") as f:
                version = f.read().strip()

        # Start in daemon mode with network configuration for climb-analyzer communication
        print("\nStarting OpenTopoData daemon...")

        # First, ensure the climb-network exists (create if needed)
        subprocess.run(
            ["docker", "network", "create", "climb-network"],
            capture_output=True,
            check=False,
        )

        # Mount elevation data from parent directory and connect to network
        elevation_data_path = os.path.join(original_dir, "data/elevation_data")
        config_path = os.path.join(original_dir, "opentopodata-config.yaml")

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-itd",
            "--name",
            "opentopodata-server",
            "--network",
            "climb-network",
            "--volume",
            f"{elevation_data_path}:/app/data:ro",
            "--volume",
            f"{config_path}:/app/config.yaml:ro",
            "-p",
            "5000:5000",
            f"opentopodata:{version}",
        ]

        result = subprocess.run(
            docker_cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print_success("OpenTopoData container started!", indent=0)

            # Wait for container to initialize
            print("\nWaiting for server to initialize...")
            time.sleep(3)

            # Check if container is running
            check_result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=opentopodata-server",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if check_result.stdout.strip():
                print_success("Container is running and connected to climb-network", indent=0)
                return True
            else:
                print_warning("Container may not be running", indent=0)
                print("Check status with: docker ps")
                return True
        else:
            print("\n✗ OpenTopoData start failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            print("\nTry manually with:")
            print(
                f"  cd opentopodata && docker run --rm -itd --name opentopodata-server --network climb-network --volume {elevation_data_path}:/app/data:ro -p 5000:5000 opentopodata:{version}"
            )
            return False

    except Exception as e:
        print(f"\n✗ Error building/starting opentopodata: {e}")
        return False
    finally:
        os.chdir(original_dir)


def print_deployment_mode_comparison():
    """Print detailed deployment mode comparison."""
    print_banner("Deployment Mode Options")

    print_section_simple("LOCAL MODE")
    print("-" * 80)
    print("Process OpenStreetMap data from local .pbf files\n")
    print("Best for:")
    print_success("Large-scale analysis (entire states, countries)", indent=2)
    print_success("Offline processing", indent=2)
    print_success("Repeated analysis of same regions", indent=2)
    print_success("Complete control over data\n", indent=2)
    print("Requirements:")
    print("  • Download OSM .pbf file for your region (~150MB - 10GB)")
    print("  • Build spatial index (auto-generated)")
    print("  • Download elevation data (auto or manual)")
    print("  • Docker with climb-analyzer and opentopodata containers\n")

    print()

    print_section_simple("CLOUD MODE")
    print("-" * 80)
    print("Query data from Overpass API on-demand\n")
    print("Best for:")
    print_success("Small regions (cities, counties)", indent=2)
    print_success("Quick one-off analysis", indent=2)
    print_success("No local storage requirements", indent=2)
    print_success("Always up-to-date OSM data\n", indent=2)
    print("Requirements:")
    print("  • Active internet connection")
    print("  • Overpass API access (free, public)")
    print("  • Minimal disk space (~1GB for elevation cache)\n")
    print("Limitations:")
    print("  • API rate limits (slower for large areas)")
    print("  • May timeout on very large regions (states, countries)\n")


def create_default_config(deployment_mode: str) -> dict:
    """
    Create default configuration based on deployment mode.

    Args:
        deployment_mode: 'local' or 'cloud'

    Returns:
        Config dictionary
    """
    # Detect CPU cores for elevation concurrency default
    try:
        import multiprocessing

        default_elevation_workers = multiprocessing.cpu_count()
    except Exception:
        default_elevation_workers = 8  # Fallback if detection fails

    if deployment_mode == "cloud":
        return {
            # === DEPLOYMENT ===
            "DEPLOYMENT_TYPE": "cloud",
            # === OSM DATA SOURCE ===
            "OVERPASS_API_URL": "https://overpass-api.de/api/interpreter",
            "OVERPASS_API_DELAY_SEC": 2.0,  # 2 second delay to stay well under 5 req/sec limit
            # === ELEVATION DATA SOURCE ===
            "TOPO_API_BASE_URL": "https://api.opentopodata.org/v1",
            # === CHECKPOINT SETTINGS ===
            "CHECKPOINT_INTERVAL_MIN": 15.0,
            "CHECKPOINT_MILESTONES_PERC": [25, 50, 75, 100],
            # === GEOCODING API ===
            "GEOCODING_MAX_CONCURRENT": 8,
            "GEOCODING_DELAY_BETWEEN_BATCHES_SEC": 0.1,
            "GEOCODING_REQUEST_TIMEOUT_SEC": 15,
            "GEOCODING_CONNECT_TIMEOUT_SEC": 5,
            "GEOCODING_RETRY_ATTEMPTS": 2,
            # === ELEVATION API ===
            "ELEVATION_MAX_CONCURRENT": 2,
            "ELEVATION_BATCH_SIZE": 100,
            "ELEVATION_DELAY_BETWEEN_BATCHES_SEC": 2.0,
            "ELEVATION_REQUEST_TIMEOUT_SEC": 45,
            "ELEVATION_CONNECT_TIMEOUT_SEC": 10,
            "ELEVATION_MAX_RETRIES": 3,
            "ELEVATION_BACKOFF_FACTOR": 2.0,
            # === DATA TRACKING ===
            "OSM_COVERAGE": [],
            "ELEVATION_DATASETS": {},
        }
    else:  # local mode
        return {
            # === DEPLOYMENT ===
            "DEPLOYMENT_TYPE": "local",
            # === OSM DATA SOURCE ===
            "OVERPASS_API_URL": None,
            "OVERPASS_API_DELAY_SEC": 0.0,
            # === ELEVATION DATA SOURCE ===
            "TOPO_API_BASE_URL": "http://opentopodata-server:5000/v1",
            # === CHECKPOINT SETTINGS ===
            "CHECKPOINT_INTERVAL_MIN": 15.0,
            "CHECKPOINT_MILESTONES_PERC": [25, 50, 75, 100],
            # === GEOCODING API ===
            "GEOCODING_MAX_CONCURRENT": 8,
            "GEOCODING_DELAY_BETWEEN_BATCHES_SEC": 0.1,
            "GEOCODING_REQUEST_TIMEOUT_SEC": 15,
            "GEOCODING_CONNECT_TIMEOUT_SEC": 5,
            "GEOCODING_RETRY_ATTEMPTS": 2,
            # === ELEVATION API ===
            "ELEVATION_MAX_CONCURRENT": default_elevation_workers,  # Matches CPU cores (auto-detected)
            "ELEVATION_BATCH_SIZE": 100,  # Benchmarks show parallel 100s fastest. Instability with large ~1000 batches.
            "ELEVATION_DELAY_BETWEEN_BATCHES_SEC": 0.0,
            "ELEVATION_REQUEST_TIMEOUT_SEC": 45,
            "ELEVATION_CONNECT_TIMEOUT_SEC": 10,
            "ELEVATION_MAX_RETRIES": 3,
            "ELEVATION_BACKOFF_FACTOR": 2.0,
            # === ELEVATION DATASET TIERS ===
            "ELEVATION_DATASET_TIERS": "primary+secondary",  # Options: "primary", "primary+secondary"
            # === DATA TRACKING ===
            "OSM_COVERAGE": [],
            "ELEVATION_DATASETS": {},
        }


def run_deployment_mode_wizard():
    """
    Run deployment mode configuration wizard.

    Returns:
        tuple: (success: bool, deployment_mode: str) where deployment_mode is 'local' or 'cloud'
    """
    print_header("Deployment Mode Configuration")

    # Check if deployment mode is already configured
    current_mode = None
    config_exists = False
    try:
        import yaml

        config_path = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(config_path):
            config_exists = True
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if config and "DEPLOYMENT_TYPE" in config:
                    current_mode = config.get("DEPLOYMENT_TYPE", "").upper()
    except Exception:
        pass

    print("Choose how to process OpenStreetMap data:\n")
    print(
        "  1. LOCAL  - Process local map and elevation files (For large state/country analysis, faster)"
    )
    print(
        "  2. CLOUD  - Query Overpass API (for small scale analysis, slower, simpler, no data downloads)"
    )
    print("\n  ?. INFO   - More information on the difference between the deployment types\n")

    if current_mode:
        print(f"Current deployment mode: {current_mode}\n")
        print(f"  0. Keep current mode ({current_mode})")
        print("  ?. INFO   - More information\n")

        while True:
            response = input("Select option [0,1,2,?]: ").strip().lower()
            if response == "?":
                print_deployment_mode_comparison()
                print(f"\nCurrent deployment mode: {current_mode}\n")
            elif response == "" or response == "0" or response == "keep":
                print(f"\n✓ Keeping current deployment mode: {current_mode}")
                return (True, current_mode.lower())
            elif response == "1" or response == "local":
                deployment_mode = "local"
                break
            elif response == "2" or response == "cloud":
                deployment_mode = "cloud"
                break
            else:
                print("Invalid choice. Please enter 0, 1, 2, or ?")
    else:
        print("Default: CLOUD mode (recommended for first-time setup)\n")
        while True:
            response = (
                input("Select deployment mode [1=LOCAL / 2=CLOUD / ?=help]: ")
                .strip()
                .lower()
            )
            if response == "?":
                print_deployment_mode_comparison()
                print("Default: CLOUD mode (recommended for first-time setup)\n")
            elif response == "1" or response == "local":
                deployment_mode = "local"
                break
            elif response == "" or response == "2" or response == "cloud":
                deployment_mode = "cloud"
                break
            else:
                print("Invalid choice. Please enter 1, 2, or ?")
                continue

    # Create/update config
    try:
        import yaml

        config_path = os.path.join(os.getcwd(), "config.yaml")

        # If config exists and we're just changing mode, preserve other settings
        if config_exists and os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            # Update deployment type
            config["DEPLOYMENT_TYPE"] = deployment_mode

            # Update mode-specific settings
            default_config = create_default_config(deployment_mode)
            for key in [
                "OVERPASS_API_URL",
                "OVERPASS_API_DELAY_SEC",
                "TOPO_API_BASE_URL",
                "ELEVATION_MAX_CONCURRENT",
                "ELEVATION_DELAY_BETWEEN_BATCHES_SEC",
            ]:
                config[key] = default_config[key]
        else:
            # Create new config from scratch
            config = create_default_config(deployment_mode)

        # Write config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Deployment mode set to: {deployment_mode.upper()}\n")

        return (True, deployment_mode)

    except Exception as e:
        print(f"\n✗ Error writing config: {e}")
        print(f"Continuing with {deployment_mode} mode in memory...")
        return (True, deployment_mode)


def explain_data_setup(deployment_mode: str):
    """
    Explain data setup process to user.

    Args:
        deployment_mode: 'local' or 'cloud'
    """
    print_header("Data Setup Information")

    print("How data is handled:\n")

    if deployment_mode == "cloud":
        print("  CLOUD MODE:")
        print("  • OSM data: Queried from Overpass API on-demand (no download needed)")
        print("  • Elevation: Queried from api.opentopodata.org (no download needed)")
        print("  • First analysis: Ready to go immediately!")
        print("  • Note: Subject to API rate limits, best for small regions\n")
    elif deployment_mode == "local":
        print("  LOCAL MODE:")
        print("  • OSM data: Downloaded automatically when you select a region")
        print("  • Elevation: Downloaded automatically for the region's coordinates")
        print("  • First analysis: Will download data as needed (20-60 minutes)")
        print("  • Subsequent analyses: Reuses existing data (fast!)")
        print("  • Storage: ~100MB - 400GB depending on region size and amount stored\n")
    else:  # both
        print("  BOTH MODES:")
        print("  • You can switch between cloud and local mode anytime")
        print("  • When using local mode, data downloads automatically as needed")
        print("  • When using cloud mode, uses public APIs (no downloads)\n")

    print("💡 No manual data setup required!")
    print("   The analyzer will automatically download any missing data")
    if deployment_mode == "local":
        print(
            "   To download data separately without running analysis: ./climb-analyzer data_download\n"
        )
    else:
        print()

    input("Press Enter to continue...")


def verify_docker_setup():
    """Verify Docker container dependencies."""
    print_header("Verifying Container Dependencies")

    print("Checking Python dependencies inside climb-analyzer container...\n")

    compose_cmd = get_docker_compose_command()
    if not compose_cmd:
        print_error("Could not find docker compose command", indent=0)
        return False

    try:
        # Check core dependencies using direct docker run (avoids compose network issues)
        # Get the image name from docker images
        result = subprocess.run(
            ["docker", "images", "-q", "climb-analyzer"],
            capture_output=True,
            text=True,
            check=False,
        )

        if not result.stdout.strip():
            print_error("climb-analyzer image not found", indent=0)
            return False

        # Run verification directly with docker run (bypasses compose network requirements)
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "climb-analyzer",
                "-c",
                "import pandas, numpy, geopy, requests; print('✓ Core dependencies OK')",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print_error("Core dependencies missing", indent=0)
            if result.stderr:
                print(f"\nError output:\n{result.stderr}")
            if result.stdout:
                print(f"\nStdout:\n{result.stdout}")
            return False

        # Check deployment-specific dependencies using direct docker run
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "climb-analyzer",
                "-c",
                "import osmium, rtree, overpy; print('✓ Deployment dependencies OK')",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print_warning(
                "Some deployment dependencies missing (this is OK if you only use one mode)",
                indent=0,
            )

        print()
        return True

    except Exception as e:
        print(f"✗ Error verifying dependencies: {e}")
        return False


def configure_elevation_dataset_tiers(deployment_mode: str):
    """
    Configure elevation dataset tier preference (local mode only).

    Args:
        deployment_mode: 'local' or 'cloud'
    """
    # Only prompt for local mode (case-insensitive check)
    mode_lower = deployment_mode.lower() if deployment_mode else ""
    if mode_lower != "local" and mode_lower != "both":
        return

    print_header("Elevation Data Accuracy Configuration")

    print("Choose how many elevation datasets to use for better accuracy:\n")
    print("  1. PRIMARY ONLY")
    print("     • Uses only the best available dataset for your region")
    print("     • Minimal disk storage (~5-50 GB per region)")
    print("     • Fastest downloads")
    print("     • May have some gaps in coverage\n")

    print("  2. PRIMARY + SECONDARY (Recommended)")
    print("     • Uses primary dataset with secondary as fallback")
    print("     • Moderate disk storage (~10-100 GB per region)")
    print("     • Better coverage and gap-filling")
    print("     • Maximum accuracy with all available datasets\n")

    print("Default: PRIMARY + SECONDARY (recommended)\n")

    while True:
        response = input("Select option [1/2]; default 2: ").strip()

        if response == "1":
            tier_setting = "primary"
            break
        elif response == "" or response == "2":
            tier_setting = "primary+secondary"
            break
        else:
            print("Invalid choice. Please enter 1 or 2")
            continue

    # Update config
    try:
        import yaml

        config_path = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = create_default_config(deployment_mode)

        config["ELEVATION_DATASET_TIERS"] = tier_setting

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Elevation dataset tier set to: {tier_setting.upper()}")

        # Show storage estimates
        if tier_setting == "primary":
            print("  Expected storage per region: ~5-50 GB")
        else:
            print("  Expected storage per region: ~10-100 GB")

        print()

    except Exception as e:
        print(f"\n⚠️  Error saving elevation tier setting: {e}")
        print("Using default: primary+secondary\n")


def configure_advanced_settings(deployment_mode: str):
    """
    Configure advanced settings interactively.

    Args:
        deployment_mode: 'local' or 'cloud'
    """
    print_header("Advanced Configuration (Optional)")

    response = input("Would you like to configure advanced settings? (y/N): ")

    if response.lower() != "y" and response.lower() != "yes":
        print("\nUsing default settings")
        return

    try:
        import yaml

        config_path = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = create_default_config(deployment_mode)

        # 1. Country/state boundary update
        print_banner("Geographic Definitions Update", spacing_before=1)
        print("Update country boundaries, state boundaries, and OSM region data?")
        print("This downloads the latest geographic data and generates bounding boxes.")
        print("Recommendation: Only needed if boundaries are outdated or incorrect")
        response = input("Update geographic definitions? [y/N]: ").strip().lower()
        if response == "y":
            print("\nUpdating geographic definitions...")
            print("This may take 5-10 minutes (crawls Geofabrik for planet subregion info)")
            try:
                result = subprocess.run(
                    ["python3", "utils/update_geo_definitions.py"],
                    check=False,
                )
                if result.returncode == 0:
                    print_success("Geographic definitions updated successfully", indent=0)
                    print("  - Country boundaries: ✓")
                    print("  - State boundaries: ✓")
                    print("  - OSM region hierarchy: ✓")
                    print("  - Region bounding boxes: ✓")
                else:
                    print(f"⚠️  Update failed with return code: {result.returncode}")
            except Exception as e:
                print(f"⚠️  Error updating definitions: {e}")

        # 2. Checkpoint settings
        print_banner("Checkpoint Settings", spacing_before=1)
        print("Checkpoints save your progress during analysis so you can resume if interrupted.")
        print(f"Current checkpoint interval: {config.get('CHECKPOINT_INTERVAL_MIN', 15.0)} minutes")
        print(
            "Checkpoints are also saved at 25%, 50%, 75%, and on completion of long running tasks."
        )
        print("Recommendation: 15 minutes (more frequent = more disk I/O, less data loss on crash)")

        response = input(
            f"Enter checkpoint interval in minutes [default: {config.get('CHECKPOINT_INTERVAL_MIN', 15.0)}]: "
        ).strip()
        if response:
            try:
                interval = float(response)
                if interval > 0 and interval <= 60:
                    config["CHECKPOINT_INTERVAL_MIN"] = interval
                    print(f"✓ Set checkpoint interval to {interval} minutes")
                else:
                    print_warning(
                        "Value must be between 0 and 60, keeping current setting", indent=0
                    )
            except ValueError:
                print_warning("Invalid value, keeping current setting", indent=0)

        # 3. Max elevation threads (for local mode only)
        if deployment_mode == "local":
            print_banner("Elevation Fetching Threads", spacing_before=1)

            # Try to detect CPU cores
            try:
                import multiprocessing

                cpu_count = multiprocessing.cpu_count()
                default_workers = config.get("ELEVATION_MAX_CONCURRENT", cpu_count)
                print(f"Current: {default_workers} parallel workers")
                print(f"Detected CPU cores: {cpu_count}")
                print(f"Recommendation: Use {cpu_count} workers (matches CPU cores)")
            except Exception:
                default_workers = config.get("ELEVATION_MAX_CONCURRENT", 8)
                print(f"Current: {default_workers} parallel workers")
                print("Recommendation: 8 workers for most systems, 16 for high-end systems")

            print("More workers = faster processing, but requires more CPU and memory")
            response = input(f"Enter max elevation threads [default: {default_workers}]: ").strip()
            if response:
                try:
                    threads = int(response)
                    if threads > 0 and threads <= 32:
                        config["ELEVATION_MAX_CONCURRENT"] = threads
                        print(f"✓ Set max elevation threads to {threads}")
                    else:
                        print_warning(
                            "Value must be between 1 and 32, keeping current setting", indent=0
                        )
                except ValueError:
                    print_warning("Invalid value, keeping current setting", indent=0)

        # Save config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print("\n✓ Advanced settings saved to config.yaml")
        print("\nNote: Additional settings like milestone percentages and API timeouts")
        print("      can be configured by editing config.yaml directly")

    except Exception as e:
        print(f"\n⚠️  Error configuring advanced settings: {e}")
        print("You can manually edit config.yaml later")


def print_next_steps():
    """Print next steps after setup."""
    print_header("Setup Complete!")

    print("Your climb analyzer is ready to use!\n")

    print("Quick Start:\n")

    print("1. Web GUI (Recommended for new users):")
    print("   ./climb-analyzer -g\n")

    print("2. Interactive mode:")
    print("   ./climb-analyzer\n")

    print("3. Address analysis:")
    print("   ./climb-analyzer -a 'Boulder, CO' --distance 25\n")

    print("4. Single region:")
    print("   ./climb-analyzer -r Colorado -s paved\n")

    print("5. Batch regions:")
    print("   ./climb-analyzer -r 'Vermont,New Hampshire,Maine' -s paved\n")

    print("6. Data management:")
    print("   ./climb-analyzer -D -r Vermont  # Download data only")
    print("   ./climb-analyzer -U              # Update boundaries")
    print("   ./climb-analyzer -P              # Delete planet/OSM data")
    print("   ./climb-analyzer -E              # Delete elevation data")
    print("   ./climb-analyzer -A              # Delete all data\n")

    print("7. Useful options:")
    print("   --list-regions                   # List available regions")
    print("   --ignore-checkpoints             # Start fresh (ignore saved progress)")
    print("   -v, --verbose                    # Show detailed output\n")

    print("8. Help:")
    print("   ./climb-analyzer --help          # Show help")
    print("   ./climb-analyzer --help-extended # Extended help with examples\n")

    print("9. Other commands:")
    print("   ./climb-analyzer shell           # Open shell in container")
    print("   ./climb-analyzer logs            # View recent logs")
    print("   ./climb-analyzer setup           # Re-run setup wizard")
    print("   ./climb-analyzer build           # Rebuild Docker image\n")

    print("Documentation:")
    print("   • INSTALLATION.md - Full installation guide")
    print("   • DOCKER_SETUP.md - Docker usage guide\n")


def install_gui_server():
    """
    Install and set up the web GUI server inside the Docker container.

    Returns:
        tuple: (success: bool, user_wants_gui: bool)
            - success: True if installation successful or skipped, False on error
            - user_wants_gui: True if user chose to install GUI, False if declined
    """
    print_header("Web GUI and Climb Map Visualizer Setup")

    print("The Climb Analyzer includes an optional web GUI for visualizing results.")
    print("The GUI provides interactive maps and data exploration features.\n")

    print("The GUI will be installed inside the Docker container.")
    print("Installation downloads ~200MB of dependencies and may take 2-5 minutes.\n")

    response = input("Install web GUI dependencies now? [Y/n]: ").strip().lower()

    if response == "n":
        print("Skipping GUI installation.")
        print("You can install it later with: ./climb-analyzer -g\n")
        return (True, False)  # (success, user_wants_gui)

    # Get the compose command
    compose_cmd = get_docker_compose_command()
    if not compose_cmd:
        print_error("Could not find docker compose command", indent=0)
        return (True, True)  # Don't fail setup, allow GUI prompt if already installed

    # Check if gui directory exists
    ui_dir = os.path.join(os.getcwd(), "gui")
    if not os.path.exists(ui_dir):
        print_error("GUI directory not found", indent=0)
        print(f"Expected to find GUI at: {ui_dir}\n")
        return (True, True)  # Don't fail setup, allow GUI prompt if already installed

    # Check if Node.js is installed in the container
    print("Checking if Node.js is installed in container...")
    node_check = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "node",
            "climb-analyzer",
            "--version",
        ],
        capture_output=True,
        check=False,
    )

    if node_check.returncode != 0:
        print_warning("Node.js not found in container - need to rebuild image", indent=0)
        print("\nThe Docker image needs to be rebuilt to include Node.js.")
        print("This will take about 5-10 minutes.\n")

        rebuild = input("Rebuild Docker image with Node.js support? [Y/n]: ").strip().lower()
        if rebuild == "n":
            print("\nSkipping GUI installation.")
            print("To install GUI later, rebuild the image with: ./climb-analyzer build")
            print("Then run: ./climb-analyzer -g\n")
            return (True, False)  # (success, user_wants_gui)

        print("\nRebuilding Docker image with Node.js...")
        print("This may take 5-10 minutes...\n")

        rebuild_result = subprocess.run(
            compose_cmd + ["build", "climb-analyzer"],
            check=False,
        )

        if rebuild_result.returncode != 0:
            print_error("\nDocker rebuild failed", indent=0)
            print("You can try rebuilding manually with: ./climb-analyzer build\n")
            return (True, True)  # (success, user_wants_gui) - don't fail setup

        print_success("\nDocker image rebuilt successfully!", indent=0)
        print()

    print("Installing GUI dependencies inside Docker container...")
    print("This may take a few minutes...\n")

    try:
        # Run npm install inside the container (bypass entrypoint to run npm directly)
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "npm",
                "-v",
                f"{ui_dir}:/app/gui",
                "-w",
                "/app/gui",
                "climb-analyzer",
                "install",
            ],
            check=False,
        )

        if result.returncode == 0:
            print_success("\nGUI dependencies installed successfully!", indent=0)
            print("\nThe GUI is now available!")
            print("To start it, run: ./climb-analyzer -g\n")

            # Verify installation worked by checking for node_modules
            node_modules_check = os.path.join(ui_dir, "node_modules", "next")
            if not os.path.exists(node_modules_check):
                print_warning("Warning: GUI installation may have issues", indent=0)
                print("If the GUI doesn't start, try reinstalling:")
                print("  ./climb-analyzer shell")
                print("  cd gui && rm -rf node_modules && npm install\n")

            return (True, True)  # (success, user_wants_gui)
        else:
            print_error("\nGUI installation failed", indent=0)
            print("\nTo install manually:")
            print("  1. Run: ./climb-analyzer shell")
            print("  2. Inside container: cd gui && npm install")
            print("  3. Exit shell and run: ./climb-analyzer -g\n")
            return (True, True)  # (success, user_wants_gui) - don't fail setup

    except Exception as e:
        print_error(f"\nError installing GUI: {e}", indent=0)
        print("You can try installing manually with:")
        print("  ./climb-analyzer shell")
        print("  cd gui && npm install\n")
        return (True, True)  # (success, user_wants_gui) - don't fail setup


def main():
    """Main setup wizard flow."""
    # Print the logo first
    try:
        from climb_analyzer.utils.ascii_logo import print_logo

        print_logo()
    except Exception:
        # Fallback if logo can't be printed
        print_banner("Climb Analyzer - Setup Wizard")

    print("Welcome to the Climb Analyzer setup wizard!")
    print("This wizard will guide you through the complete setup process.\n")

    print("This wizard will:")
    print_list_item("Check Docker and system requirements")
    print_list_item("Configure deployment mode (LOCAL or CLOUD)")
    print_list_item("Build Docker containers")
    print_list_item("Configure data source types")
    print_list_item("Optionally install web GUI server")
    print_list_item("Verify the installation")
    print()

    # Note: NASA Earthdata credentials are no longer needed (December 2025)
    # SRTM now uses OpenTopography S3, AW3D30 uses JAXA FTP - all public!

    # Check Python version (for this script)
    if not check_python_version():
        print("\n✗ Python 3.9+ required to run this setup wizard")
        print("Please upgrade Python and try again")
        sys.exit(1)

    # Check Docker dependencies
    docker_ok, compose_ok, messages = check_docker_dependencies()

    if not docker_ok or not compose_ok:
        print_error("Docker prerequisites not met\n", indent=0)
        print("Please install the missing components:\n")
        for msg in messages:
            print(f"  • {msg}")
        print()
        print("After installing, run this wizard again:")
        print("  python setup_wizard.py")
        sys.exit(1)

    print_success("All Docker prerequisites met!\n", indent=0)

    # Check disk space
    check_disk_space()

    # Configure deployment mode FIRST (before building containers)
    mode_ok, deployment_mode = run_deployment_mode_wizard()

    # Explain data setup (only if deployment mode was configured)
    if mode_ok:
        explain_data_setup(deployment_mode)

    # Note: NASA Earthdata credentials are no longer needed (December 2025)
    # All elevation datasets now use public sources:
    # - SRTM: OpenTopography S3 (public)
    # - AW3D30: JAXA FTP (public)
    # - NED, ArcticDEM, REMA: AWS S3 (public)

    # The wizard returns the user's choice and has already written it to config
    # No need to re-read; trust the wizard's return value
    print(f"\nDetected deployment mode: {deployment_mode.upper()}\n")

    # Check if climb-analyzer container exists - if not, build it
    needs_build = True
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "climb-analyzer"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            print_success("climb-analyzer container already exists\n", indent=0)
            needs_build = False
    except Exception:
        pass

    # Build Docker images if needed
    if needs_build:
        if not build_docker_images(deployment_mode):
            print("\n⚠️  Docker images not built")
            if deployment_mode == "cloud":
                print("You can build them later with: docker compose build climb-analyzer")
            else:
                print("You can build them later with: docker compose build")
            print("Then run: docker compose run --rm climb-analyzer python data_setup.py")
            sys.exit(1)

        # Verify container after build
        if deployment_mode == "local":
            if not verify_docker_setup():
                print("\n⚠️  Container verification had issues")
                print("You may need to rebuild: docker-compose build")
                sys.exit(1)

    # Deployment-specific setup
    if mode_ok:

        # If LOCAL mode, automatically build and start opentopodata container
        if deployment_mode == "local":
            print_header("OpenTopoData Server Setup")
            print("OpenTopoData serves elevation data for local mode.")

            # Check for existing elevation data
            from pathlib import Path
            elevation_data_dir = Path("data/elevation_data")
            has_elevation_data = (
                elevation_data_dir.exists() and
                (any(elevation_data_dir.glob("*/*.tif")) or any(elevation_data_dir.glob("*/*.hgt")))
            )

            try:
                config_created = False

                if has_elevation_data:
                    # Generate config from existing elevation data
                    print("\nDetected existing elevation data, generating config...")
                    result = subprocess.run(
                        ["python3", "utils/generate_opentopodata_config.py"],
                        check=False,
                    )
                    config_created = result.returncode == 0
                    if config_created:
                        print_success("Config generated from existing elevation data", indent=0)
                else:
                    # No elevation data - create test config for initial setup
                    print("\nSetting up OpenTopoData server with test dataset...")
                    print("(This will be updated with actual elevation data after download)\n")
                    sys.path.insert(0, os.getcwd())
                    from climb_analyzer.data.dem_downloaders import create_test_dataset_config
                    config_created = create_test_dataset_config(Path("opentopodata-config.yaml"))

                if config_created:
                    # Build and start the container
                    opentopodata_ok = build_and_start_opentopodata()
                    if opentopodata_ok:
                        if has_elevation_data:
                            print_success("OpenTopoData server is running with elevation data!", indent=0)
                        else:
                            print_success("OpenTopoData server is running with test dataset!", indent=0)
                            print_info(
                                "\nThe config will be updated automatically when you download elevation data"
                            )
                        print_success("Server available at: http://localhost:5000", indent=0)
                    else:
                        print("\n⚠️  OpenTopoData build/start had issues")
                        print("You can start it manually later with:")
                        print("  docker compose up -d --build opentopodata")
                else:
                    print("\n⚠️  Could not create config")
            except Exception as e:
                print(f"\n⚠️  Error setting up OpenTopoData: {e}")
                print("You can set it up manually later with:")
                print("  docker compose up -d --build opentopodata")

        # Configure elevation dataset tiers (local mode only)
        # Normalize deployment_mode to lowercase for consistency
        normalized_mode = (
            deployment_mode.lower() if isinstance(deployment_mode, str) else deployment_mode
        )
        configure_elevation_dataset_tiers(normalized_mode)

        # Configure advanced settings
        configure_advanced_settings(deployment_mode)

    # Optional: Install web GUI server (independent of deployment mode)
    gui_success, user_wants_gui = install_gui_server()

    # Prompt to launch GUI (only if it's already installed) - BEFORE printing "Setup Complete!"
    # Show prompt if GUI exists, regardless of whether user chose to install in this session
    gui_dir = os.path.join(os.getcwd(), "gui")
    node_modules = os.path.join(gui_dir, "node_modules")
    if os.path.exists(node_modules):
        # Check if GUI is already running
        check_running = subprocess.run(
            ["docker", "ps", "--filter", "publish=3000", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )

        if check_running.stdout.strip():
            container_name = check_running.stdout.strip()
            print_header("Web GUI Status")
            print("✓ Web GUI is already running: http://localhost:3000\n")
            print(f"Container: {container_name}\n")

            response = input("Would you like to restart the GUI? [y/N]: ").strip().lower()

            if response == "y":
                print("\nRestarting GUI...")
                # Stop the existing container
                print(f"Stopping container {container_name}...")
                subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
                subprocess.run(["docker", "rm", container_name], capture_output=True, check=False)

                # Start new GUI (runs in background)
                print("Starting fresh GUI instance...\n")
                result = subprocess.run(["./climb-analyzer", "-g"], check=False)
                if result.returncode != 0:
                    print("\n⚠️  Error launching GUI")
                    print("You can start it manually with: ./climb-analyzer -g")
            else:
                print("\nGUI is already running at: http://localhost:3000")
                print("You can stop it with: docker stop " + container_name + "\n")
        else:
            print_header("Launch Web GUI?")
            print("Would you like to start the web GUI now?\n")
            print("The GUI will be available at: http://localhost:3000\n")

            response = input("Start GUI now? [y/N]: ").strip().lower()

            if response == "y":
                print("\nLaunching GUI...\n")
                result = subprocess.run(["./climb-analyzer", "-g"], check=False)
                if result.returncode != 0:
                    print("\n⚠️  Error launching GUI")
                    print("You can start it manually with: ./climb-analyzer -g")
            else:
                print("\nYou can start the GUI later with: ./climb-analyzer -g\n")

    # Print next steps and "Setup Complete!" message AFTER GUI prompt
    print_next_steps()

    if not mode_ok:
        print_warning("Note: Deployment mode was not configured", indent=0)
        print("Configure it manually by re-running: python setup_wizard.py\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Setup failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
