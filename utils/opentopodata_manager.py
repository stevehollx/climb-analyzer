#!/usr/bin/env python3
"""
OpenTopoData Server Manager

Automates OpenTopoData configuration, build, and deployment:
- Updates config.yaml based on available elevation datasets
- Builds Docker image for current platform (macOS Intel/M1, Linux)
- Manages Docker container lifecycle (stop, start, restart)
- Validates server health and readiness
"""

import os
import platform
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import yaml

# Tile reconciliation removed - keeping all datasets for complete coverage and fallback support


def check_dataset_has_files(dataset_dir: Path) -> bool:
    """
    Check if a dataset directory has any elevation data files.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        True if directory has data files, False otherwise
    """
    if not dataset_dir.exists():
        return False

    patterns = ['*.tif', '*.hgt', '*.img', '*.vrt']
    for pattern in patterns:
        if list(dataset_dir.glob(pattern)):
            return True

    return False


def detect_platform() -> Tuple[str, str]:
    """
    Detect operating system and architecture.

    Returns:
        Tuple of (os_type, build_target)
        - os_type: "macos", "linux", or "unknown"
        - build_target: "build-m1", "build", or "build"
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        # macOS
        if machine in ["arm64", "aarch64"]:
            return "macos", "build-m1"
        else:
            return "macos", "build"
    elif system == "linux":
        return "linux", "build"
    else:
        return "unknown", "build"


def is_running_in_docker() -> bool:
    """Check if running inside a Docker container."""
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def get_base_directory() -> Path:
    """
    Get base directory for project.

    Returns correct path whether running on host or inside Docker container.
    Handles being run from project root or from utils/ subdirectory.
    """
    # Use the script's location to find project root reliably
    # This avoids issues with cwd being in unexpected locations
    script_path = Path(__file__).resolve()

    # If this script is in utils/, go up to project root
    if script_path.parent.name == 'utils':
        return script_path.parent.parent

    # Otherwise, assume script is in project root
    return script_path.parent


def get_host_path(container_path: Path) -> Path:
    """
    Convert container path to host path for Docker volume mounts.

    When running inside Docker, paths like /app need to be converted
    to their host equivalents for mounting into other containers.

    Args:
        container_path: Path as seen inside the container

    Returns:
        Host path suitable for Docker volume mounts

    Raises:
        RuntimeError: If running in Docker but HOST_PROJECT_DIR is not set
    """
    if not is_running_in_docker():
        # On host - use path as-is
        return container_path.resolve()

    # Running inside Docker - MUST have HOST_PROJECT_DIR set
    host_base_str = os.environ.get('HOST_PROJECT_DIR')

    if not host_base_str:
        raise RuntimeError(
            "HOST_PROJECT_DIR environment variable must be set when running inside Docker.\n"
            "This should be set automatically by docker-compose.yml or when starting the container."
        )

    # Running inside Docker - need to translate paths
    container_path_str = str(container_path.resolve())

    # Check if path starts with /app (typical container working directory)
    if container_path_str.startswith('/app'):
        host_base = Path(host_base_str)

        # Replace /app with host base
        relative = container_path_str[4:]  # Remove '/app'
        if relative.startswith('/'):
            relative = relative[1:]  # Remove leading slash

        host_path = host_base / relative if relative else host_base
        return host_path

    # If not /app, assume it's already a mounted path and use as-is
    return container_path.resolve()


def scan_elevation_datasets(elevation_data_dir: Path, skip_empty: bool = True) -> List[dict]:
    """
    Scan elevation_data directory and detect available datasets.

    Args:
        elevation_data_dir: Path to elevation_data directory
        skip_empty: If True, skip datasets with no data files (default: True)

    Returns:
        List of dataset configuration dictionaries
    """
    datasets = []
    skipped_empty = []

    # Check for NED (10m resolution)
    ned_dir = elevation_data_dir / "ned10m"
    if ned_dir.exists():
        ned_files = list(ned_dir.glob("USGS_13_*.tif")) + list(ned_dir.glob("*.tif"))
        if ned_files:
            print(f"  ✓ Found {len(ned_files)} NED tiles")
            datasets.append({
                "name": "ned10m",
                "path": "data/ned10m/",
            })
        elif skip_empty:
            skipped_empty.append("ned10m")

    # Check for SRTM (30m resolution)
    # SRTM files can be either .hgt or .tif (newer GeoTIFF format)
    srtm_dir = elevation_data_dir / "srtm30m"
    if srtm_dir.exists():
        srtm_files = list(srtm_dir.glob("*.hgt")) + list(srtm_dir.glob("*.tif"))
        if srtm_files:
            print(f"  ✓ Found {len(srtm_files)} SRTM tiles")
            datasets.append({
                "name": "srtm30m",
                "path": "data/srtm30m/",
            })
        elif skip_empty:
            skipped_empty.append("srtm30m")

    # NOTE: ASTER is deprecated (December 2025). AW3D30 provides better coverage and accuracy.
    # Existing ASTER data will still work but new downloads are not supported.

    # Check for AW3D30 (30m resolution)
    aw3d30_dir = elevation_data_dir / "aw3d30"
    if aw3d30_dir.exists():
        # Check for flattened structure (N*.tif files)
        aw3d30_flat_files = list(aw3d30_dir.glob("[NS][0-9][0-9][0-9][EW][0-9][0-9][0-9].tif"))
        # Check for nested structure (ALPSMLC30_*.tif in subdirectories)
        aw3d30_nested_files = list(aw3d30_dir.rglob("ALPSMLC30_*_DSM.tif"))

        aw3d30_tiles = len(aw3d30_flat_files) + len(aw3d30_nested_files)
        if aw3d30_tiles > 0:
            print(f"  ✓ Found {aw3d30_tiles} AW3D30 tiles")
            datasets.append({
                "name": "aw3d30",
                "path": "data/aw3d30/",
                # No filename_regex needed - files are now flattened to simple format (N043E007.tif)
            })
        elif skip_empty:
            skipped_empty.append("aw3d30")

    # Check for ArcticDEM (32m resolution)
    # Requires VRT file - raw tiles use non-standard naming that OpenTopoData can't parse
    arctic_vrt = elevation_data_dir / "arctic32m-vrt" / "arctic32m.vrt"
    arctic_dir = elevation_data_dir / "arctic32m"
    if arctic_vrt.exists():
        print("  ✓ Found ArcticDEM VRT file")
        datasets.append({
            "name": "arctic32m",
            "path": "data/arctic32m-vrt/arctic32m.vrt",
        })
    elif arctic_dir.exists():
        arctic_files = list(arctic_dir.glob("**/*.tif"))
        if arctic_files:
            # Tiles exist but no VRT - can't be used directly by OpenTopoData
            print(f"  ⚠️  Found {len(arctic_files)} ArcticDEM tiles but no VRT file (skipping)")
            print(f"       Run: python scripts/manage_arctic_vrt.py --dataset arctic32m --rebuild")
        elif skip_empty:
            skipped_empty.append("arctic32m")
    elif skip_empty:
        skipped_empty.append("arctic32m")

    # Check for REMA (32m resolution, Antarctic)
    # Requires VRT file - raw tiles use non-standard naming that OpenTopoData can't parse
    rema_vrt = elevation_data_dir / "rema32m-vrt" / "rema32m.vrt"
    rema_dir = elevation_data_dir / "rema32m"
    if rema_vrt.exists():
        print("  ✓ Found REMA VRT file")
        datasets.append({
            "name": "rema32m",
            "path": "data/rema32m-vrt/rema32m.vrt",
        })
    elif rema_dir.exists():
        rema_files = list(rema_dir.glob("**/*.tif"))
        if rema_files:
            # Tiles exist but no VRT - can't be used directly by OpenTopoData
            print(f"  ⚠️  Found {len(rema_files)} REMA tiles but no VRT file (skipping)")
            print(f"       Run: python scripts/manage_arctic_vrt.py --dataset rema32m --rebuild")
        elif skip_empty:
            skipped_empty.append("rema32m")
    elif skip_empty:
        skipped_empty.append("rema32m")

    # Report skipped datasets
    if skipped_empty:
        print(f"  ⚠️  Skipped {len(skipped_empty)} empty dataset(s): {', '.join(skipped_empty)}")

    return datasets


def update_config(
    elevation_data_dir: Path,
    config_path: Path,
    preserve_existing: bool = True,
) -> bool:
    """
    Update OpenTopoData config.yaml based on available datasets.

    Args:
        elevation_data_dir: Path to elevation_data directory
        config_path: Path to config.yaml file
        preserve_existing: If True, keep datasets not found by scan

    Returns:
        True if successful
    """
    from climb_analyzer.utils.formatting import print_banner
    print_banner("Updating OpenTopoData Configuration", spacing_before=1)

    # Note: Tile reconciliation has been removed to preserve all datasets for complete coverage
    # This allows querying secondary/tertiary datasets when primary has NODATA
    print("   Preserving all datasets for complete coverage and fallback support\n")

    # Scan for datasets
    print("Scanning elevation data directory...")
    datasets = scan_elevation_datasets(elevation_data_dir, skip_empty=True)

    if not datasets:
        print("\n  ⚠️  No elevation datasets found")
        print(f"  📁 Checked: {elevation_data_dir}")
        return False

    # Read existing config to preserve other settings
    existing_datasets = []
    if preserve_existing and config_path.exists():
        try:
            with open(config_path) as f:
                existing_config = yaml.safe_load(f)
                if existing_config and "datasets" in existing_config:
                    # Keep datasets that aren't in our new list AND that have data files
                    new_dataset_names = {d["name"] for d in datasets}

                    for dataset in existing_config["datasets"]:
                        dataset_name = dataset.get("name")
                        if dataset_name not in new_dataset_names:
                            # Check if this preserved dataset actually has files
                            dataset_dir = elevation_data_dir / dataset_name
                            if check_dataset_has_files(dataset_dir):
                                existing_datasets.append(dataset)
                            else:
                                print(f"  ⚠️  Skipping empty preserved dataset: {dataset_name}")

                    if existing_datasets:
                        print(f"     Preserving {len(existing_datasets)} existing dataset(s)")
        except Exception as e:
            print(f"  ⚠️  Could not read existing config: {e}")

    # Combine existing and new datasets
    # No MultiDataset parent - each dataset is independently queryable
    # This allows users to query secondary/tertiary datasets if primary has NODATA
    all_datasets = datasets + existing_datasets

    # Create config
    config = {
        "datasets": all_datasets,
        "max_locations_per_request": 500,
        "access_control_allow_origin": "*",
    }

    # Write config
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Updated configuration: {config_path}")
        print(f"   Configured {len(all_datasets)} dataset(s):")
        for ds in all_datasets:
            print(f"     • {ds['name']}")

        return True

    except Exception as e:
        print(f"\n❌ Failed to write config: {e}")
        return False


def is_docker_running(verbose: bool = False) -> bool:
    """
    Check if Docker daemon is running.

    Args:
        verbose: If True, print debug info about Docker check

    Returns:
        True if Docker is accessible
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=False,
            timeout=5
        )
        is_running = result.returncode == 0
        if verbose and not is_running:
            stderr = result.stderr.decode()[:200] if result.stderr else "no stderr"
            # print(f"  [DEBUG] docker info returned code {result.returncode}: {stderr}")
        return is_running
    except subprocess.TimeoutExpired:
        if verbose:
            pass  # Debug output disabled
        return False
    except FileNotFoundError:
        if verbose:
            pass  # Debug output disabled
        return False
    except Exception as e:
        if verbose:
            pass  # Debug output disabled
        return False


def ensure_docker_network(network_name: str = "climb-network") -> bool:
    """
    Ensure Docker network exists for container communication.

    Args:
        network_name: Name of Docker network to create/verify

    Returns:
        True if network exists or was created successfully
    """
    # First check if Docker is running
    if not is_docker_running(verbose=True):
        print("  ❌ Docker daemon is not running or not accessible")
        print("\n💡 Possible causes:")
        print("     1. Docker not started: Open Docker Desktop (macOS) or run: sudo systemctl start docker (Linux)")
        print("     2. Permission denied: Docker socket not mounted in container")
        print("     3. Running inside container: Rebuild from host instead")
        return False

    try:
        # Check if network exists
        result = subprocess.run(
            ["docker", "network", "ls", "--filter", f"name={network_name}", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            check=False
        )

        # Check for exact match (filter does partial match, so verify exact)
        existing_networks = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
        if network_name in existing_networks:
            print(f"  ✓ Network already exists: {network_name}")
            return True

        # Create network
        print(f"  ⏳ Creating Docker network: {network_name}")
        result = subprocess.run(
            ["docker", "network", "create", network_name],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            print(f"  ✓ Created Docker network: {network_name}")
            return True
        else:
            error_msg = result.stderr.strip()
            print(f"  ❌ Could not create network: {error_msg}")
            return False

    except Exception as e:
        print(f"  ❌ Error managing Docker network: {e}")
        return False


def stop_container(container_name: str = "opentopodata-server") -> bool:
    """
    Stop and remove existing OpenTopoData container.

    Args:
        container_name: Name of container to stop

    Returns:
        True if successful (or container doesn't exist)
    """
    try:
        # Stop container
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            check=False,
            timeout=30
        )

        # Remove container
        subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True,
            check=False,
            timeout=10
        )

        return True

    except Exception as e:
        print(f"  ⚠️  Error stopping container: {e}")
        return False


def build_image(opentopodata_dir: Path, build_target: str = "build") -> Optional[str]:
    """
    Build OpenTopoData Docker image using Makefile.

    Args:
        opentopodata_dir: Path to opentopodata directory
        build_target: Make target ("build" or "build-m1")

    Returns:
        Version string if successful, None otherwise
    """
    print("\nBuilding OpenTopoData Docker image...")
    print(f"  Target: {build_target}")

    try:
        # Change to opentopodata directory and run make
        result = subprocess.run(
            ["make", build_target],
            cwd=opentopodata_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"  ❌ Build failed: {result.stderr}")
            return None

        # Read version
        version_file = opentopodata_dir / "VERSION"
        if version_file.exists():
            version = version_file.read_text().strip()
            print(f"  ✓ Built opentopodata:{version}")
            return version
        else:
            print("  ⚠️  Version file not found, using 'latest'")
            return "latest"

    except subprocess.TimeoutExpired:
        print("  ❌ Build timed out (>5 minutes)")
        return None
    except Exception as e:
        print(f"  ❌ Build error: {e}")
        return None


def start_container(
    base_dir: Path,
    version: str,
    config_path: Optional[Path] = None,
    container_name: str = "opentopodata-server",
    network_name: str = "climb-network",
    port: int = 5000
) -> bool:
    """
    Start OpenTopoData Docker container.

    Args:
        base_dir: Base directory containing elevation_data
        version: Docker image version tag
        config_path: Optional path to config.yaml (will be mounted)
        container_name: Name for the container
        network_name: Docker network to connect to
        port: Port to expose

    Returns:
        True if successful
    """
    print("\nStarting OpenTopoData container...")

    try:
        # Convert paths to host paths (important when running inside Docker)
        # Use data/elevation_data as per data_paths.py configuration
        elevation_data_host = get_host_path(base_dir / "data" / "elevation_data")

        # Build docker run command
        cmd = [
            "docker", "run",
            "--rm", "-d",
            "--name", container_name,
            "--network", network_name,
            "--volume", f"{elevation_data_host}:/app/data:ro",
        ]

        # Show network info
        if is_running_in_docker():
            print(f"  Docker network: {network_name} (for inter-container communication)")
        else:
            print(f"  Docker network: {network_name}")

        # Show user-friendly paths
        try:
            elev_rel = elevation_data_host.relative_to(Path.cwd())
            print(f"  Volume mount: ./{elev_rel} → /app/data (read-only)")
        except ValueError:
            print(f"  Volume mount: {elevation_data_host} → /app/data (read-only)")

        # Add config volume if provided
        if config_path and config_path.exists():
            config_path_host = get_host_path(config_path)
            cmd.extend(["--volume", f"{config_path_host}:/app/config.yaml:ro"])

            try:
                config_rel = config_path_host.relative_to(Path.cwd())
                print(f"  Config mount: ./{config_rel} → /app/config.yaml")
            except ValueError:
                print(f"  Config mount: {config_path_host} → /app/config.yaml")

        # Add port mapping
        cmd.extend(["-p", f"{port}:{port}"])

        # Add image
        cmd.append(f"opentopodata:{version}")

        # Run container
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )

        if result.returncode == 0:
            print(f"  ✓ Container started: {container_name}")
            print(f"  ✓ Server available at: http://localhost:{port}")
            return True
        else:
            print(f"  ❌ Failed to start container: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ❌ Error starting container: {e}")
        return False


def get_configured_datasets(
    base_url: str = "http://localhost:5000",
    container_name: str = "opentopodata-server"
) -> set:
    """
    Query OpenTopoData server for currently configured datasets.

    Args:
        base_url: Base URL of OpenTopoData server
        container_name: Container name for Docker network access

    Returns:
        Set of dataset names configured on the server, or empty set if query fails
    """
    # Determine if running inside Docker container
    in_docker = Path("/.dockerenv").exists()

    # Build list of URLs to try
    urls_to_try = []

    if in_docker:
        # Inside Docker - try container name first
        urls_to_try.append(f"http://{container_name}:5000/datasets")
        urls_to_try.append(f"{base_url}/datasets")
    else:
        # On host - try localhost first
        urls_to_try.append(f"{base_url}/datasets")

    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Response format: {"results": [{"name": "ned10m", ...}, ...]}
                if "results" in data and isinstance(data["results"], list):
                    return {ds["name"] for ds in data["results"] if "name" in ds}
        except Exception:
            continue  # Try next URL

    # If all attempts failed, return empty set
    return set()


def check_server_health(
    base_url: str = "http://localhost:5000",
    max_retries: int = 20,
    retry_delay: float = 2.0,
    container_name: str = "opentopodata-server",
    quiet: bool = False
) -> bool:
    """
    Check if OpenTopoData server is healthy and responding.

    Args:
        base_url: Base URL of OpenTopoData server (default: http://localhost:5000)
        max_retries: Maximum number of health check attempts (default: 20)
        retry_delay: Seconds to wait between retries (default: 2.0)
        container_name: Name of OpenTopoData container (for Docker network access)
        quiet: If True, suppress progress output (for quick checks)

    Returns:
        True if server is healthy

    Note:
        Default settings allow up to 40 seconds for server startup,
        which is appropriate for Docker container initialization.

        When running inside Docker, automatically uses container name
        instead of localhost for proper inter-container communication.
    """
    # If running inside Docker, use container name instead of localhost
    if is_running_in_docker():
        # Use container name for Docker network communication
        base_url = f"http://{container_name}:5000"
        if not quiet:
            from climb_analyzer.utils.formatting import print_dim

            print(f"\nWaiting for server to be ready (max {max_retries * retry_delay:.0f}s)...")
            print_dim(f"  Using Docker network: {base_url}")
    else:
        if not quiet:
            print(f"\nWaiting for server to be ready (max {max_retries * retry_delay:.0f}s)...")

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(f"{base_url}/health", timeout=3)
            if response.status_code == 200:
                if not quiet:
                    print(f"  ✓ Server is healthy (ready after {attempt * retry_delay:.0f}s)")
                return True
            elif response.status_code == 500:
                # Server returned 500 Internal Server Error - fail immediately
                if not quiet:
                    print(f"\n  ❌ Server returned 500 Internal Server Error")
                    print(f"     This indicates a configuration or data loading problem")
                    print(f"     Check logs: docker logs opentopodata-server")
                return False
            else:
                # Server responded but not ready yet (e.g., 503 Service Unavailable)
                if attempt < max_retries:
                    if not quiet:
                        print(f"  ⏳ Waiting for server ({attempt * retry_delay:.0f}s elapsed, status: {response.status_code})...")
                    time.sleep(retry_delay)
                else:
                    if not quiet:
                        print(f"\n  ❌ Server not healthy after {max_retries * retry_delay:.0f}s (status: {response.status_code})")
                    return False
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                if not quiet:
                    print(f"  ⏳ Waiting for server ({attempt * retry_delay:.0f}s elapsed)...")
                time.sleep(retry_delay)
            else:
                if not quiet:
                    print(f"\n  ❌ Server not responding after {max_retries * retry_delay:.0f}s")
                return False

    return False


def clone_opentopodata(target_dir: Path) -> bool:
    """
    Clone OpenTopoData repository if it doesn't exist.

    Args:
        target_dir: Directory where opentopodata should be cloned

    Returns:
        True if successful or already exists, False on error
    """
    if target_dir.exists():
        print(f"     OpenTopoData repository already exists at {target_dir}")
        return True

    print(f"\n  Cloning OpenTopoData repository to {target_dir}...")

    try:
        # Clone into parent directory, which will create the opentopodata subdirectory
        parent_dir = target_dir.parent
        result = subprocess.run(
            ["git", "clone", "https://github.com/ajnisbet/opentopodata.git", str(target_dir.name)],
            cwd=str(parent_dir),
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("  ✓ OpenTopoData cloned successfully")
            return True
        else:
            print(f"  ❌ Failed to clone OpenTopoData: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ❌ Error cloning OpenTopoData: {e}")
        return False


def rebuild_and_restart(
    elevation_data_dir: Optional[Path] = None,
    opentopodata_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    auto_update_config: bool = True,
    validate_health: bool = True
) -> bool:
    """
    Complete workflow: update config, rebuild image, restart container.

    Args:
        elevation_data_dir: Path to elevation_data (default: ./elevation_data)
        opentopodata_dir: Path to opentopodata repo (default: ./opentopodata)
        config_path: Path to config.yaml (default: ./opentopodata/config.yaml)
        auto_update_config: If True, scan and update config automatically
        validate_health: If True, wait for server health check

    Returns:
        True if successful
    """
    from climb_analyzer.utils.formatting import print_banner
    print_banner("Rebuilding OpenTopoData Server")

    # Check if running inside Docker
    if is_running_in_docker():
        print("\nRunning inside Docker container - using Docker socket for rebuild")

        # Get paths
        base_dir = get_base_directory()
        if elevation_data_dir is None:
            elevation_data_dir = base_dir / "data" / "elevation_data"
        if opentopodata_dir is None:
            opentopodata_dir = base_dir / "opentopodata"
        if config_path is None:
            config_path = opentopodata_dir / "config.yaml"

        # Update config if requested
        if auto_update_config:
            if not update_config(elevation_data_dir, config_path):
                print("\n⚠️  Config update failed")
                return False

        # Check if Docker socket is accessible
        if not is_docker_running(verbose=True):
            print("\n⚠️  Docker socket not accessible from inside container")
            print("     This is likely a permission issue with DOCKER_GID")
            print("\nPossible fixes:")
            print("  1. Exit and restart: ./climb-analyzer stop && ./climb-analyzer run")
            print("     (This re-detects DOCKER_GID and recreates the container)")
            print("  2. Manually check docker group: getent group docker")
            print("     Then set DOCKER_GID in .env to match")
            print("  3. Verify socket is mounted: docker-compose.yml should have:")
            print("     - /var/run/docker.sock:/var/run/docker.sock")
            return False

        print("\n✓ Docker socket accessible - proceeding with rebuild")
        # Continue with normal rebuild process using Docker commands through the socket

    # Detect platform
    os_type, build_target = detect_platform()
    print(f"Platform: {os_type} ({build_target})")

    # Get paths (use relative paths first)
    base_dir = get_base_directory()
    if elevation_data_dir is None:
        # Use data/elevation_data as per data_paths.py configuration
        elevation_data_dir = base_dir / "data" / "elevation_data"
    if opentopodata_dir is None:
        opentopodata_dir = base_dir / "opentopodata"
    if config_path is None:
        config_path = opentopodata_dir / "config.yaml"

    # Display paths (relative to cwd for clarity)
    try:
        base_rel = base_dir.relative_to(Path.cwd())
        print(f"Base directory: ./{base_rel}" if str(base_rel) != "." else f"Base directory: {base_dir}")
    except ValueError:
        print(f"Base directory: {base_dir}")

    try:
        elev_rel = elevation_data_dir.relative_to(Path.cwd())
        print(f"Elevation data: ./{elev_rel}")
    except ValueError:
        print(f"Elevation data: {elevation_data_dir}")

    try:
        otd_rel = opentopodata_dir.relative_to(Path.cwd())
        print(f"OpenTopoData: ./{otd_rel}")
    except ValueError:
        print(f"OpenTopoData: {opentopodata_dir}")

    # Clone OpenTopoData if it doesn't exist
    if not opentopodata_dir.exists():
        print(f"\n  OpenTopoData directory not found at {opentopodata_dir}")
        if not clone_opentopodata(opentopodata_dir):
            print("\n❌ Failed to clone OpenTopoData repository")
            return False

    # Update config if requested
    if auto_update_config:
        if not update_config(elevation_data_dir, config_path):
            print("\n⚠️  Config update failed, but continuing with rebuild...")

    # Stop existing container
    print("\nStopping existing container...")
    stop_container()

    # Ensure Docker network exists
    print("\nChecking Docker network...")
    if not ensure_docker_network():
        print("\n❌ Failed to create Docker network")
        return False

    # Build image
    version = build_image(opentopodata_dir, build_target)
    if not version:
        print("\n❌ Build failed")
        return False

    # Start container
    if not start_container(base_dir, version, config_path):
        print("\n❌ Failed to start container")
        return False

    # Validate health (generous timeout for large datasets)
    if validate_health:
        if not check_server_health(max_retries=60, retry_delay=2.0):
            print("\n⚠️  Server started but health check failed after 120 seconds")
            print("    The server may still be loading datasets")
            print("    Check logs: docker logs opentopodata-server")
            return False

    from climb_analyzer.utils.formatting import print_banner
    print_banner("OpenTopoData Server Ready", spacing_before=1)

    return True


def quick_restart(container_name: str = "opentopodata-server") -> bool:
    """
    Quick restart of existing container (no rebuild).

    Args:
        container_name: Name of container to restart

    Returns:
        True if successful
    """
    print("\n" + "=" * 70)
    print("  RESTARTING OPENTOPODATA CONTAINER")
    print("=" * 70 + "\n")

    try:
        # Check if container exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False
        )

        if container_name not in result.stdout:
            print(f"  ❌ Container '{container_name}' not found")
            print("     Run rebuild_and_restart() to create it")
            return False

        # Restart container
        result = subprocess.run(
            ["docker", "restart", container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )

        if result.returncode == 0:
            print(f"  ✓ Container restarted: {container_name}")

            # Validate health
            if check_server_health():
                print("\n✓ Server ready")
                return True
            else:
                print("\n⚠️  Server restarted but health check failed")
                return False
        else:
            print(f"  ❌ Restart failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ❌ Error restarting container: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "rebuild":
            success = rebuild_and_restart()
            sys.exit(0 if success else 1)

        elif command == "restart":
            success = quick_restart()
            sys.exit(0 if success else 1)

        elif command == "update-config":
            base_dir = get_base_directory()
            elevation_dir = base_dir / "elevation_data"
            config_path = base_dir / "opentopodata" / "config.yaml"
            success = update_config(elevation_dir, config_path)
            sys.exit(0 if success else 1)

        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python opentopodata_manager.py rebuild       # Rebuild and restart")
            print("  python opentopodata_manager.py restart       # Quick restart")
            print("  python opentopodata_manager.py update-config # Update config only")
            sys.exit(1)
    else:
        print("OpenTopoData Server Manager")
        print("\nUsage:")
        print("  python opentopodata_manager.py rebuild       # Full rebuild and restart")
        print("  python opentopodata_manager.py restart       # Quick restart")
        print("  python opentopodata_manager.py update-config # Update config only")
        print("\nOr import and use programmatically:")
        print("  from opentopodata_manager import rebuild_and_restart")
        print("  rebuild_and_restart()")
