"""
Version checker utility for climb-analyzer.

Checks GitHub for newer versions and prompts user to update.
"""

import sys
import re
from pathlib import Path
from typing import Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

import requests


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """
    Parse semantic version string into tuple of integers.

    Args:
        version_str: Version string like "2.1.0"

    Returns:
        Tuple of (major, minor, patch)
    """
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    return tuple(map(int, match.groups()))


def compare_versions(local_version: str, remote_version: str) -> int:
    """
    Compare two semantic versions.

    Args:
        local_version: Local version string
        remote_version: Remote version string

    Returns:
        -1 if local < remote (update available)
         0 if local == remote (up to date)
         1 if local > remote (ahead of remote)
    """
    local = parse_version(local_version)
    remote = parse_version(remote_version)

    if local < remote:
        return -1
    elif local > remote:
        return 1
    else:
        return 0


def get_local_version() -> Optional[str]:
    """
    Read version from local pyproject.toml.

    Returns:
        Version string or None if not found
    """
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        return data.get("project", {}).get("version")
    except Exception as e:
        print(f"Warning: Could not read local version: {e}")
        return None


def get_remote_version(timeout: int = 5) -> Optional[str]:
    """
    Fetch version from GitHub pyproject.toml.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Version string or None if fetch failed
    """
    try:
        # Use raw GitHub URL to get the file directly
        url = "https://raw.githubusercontent.com/stevehollx/climb-analyzer/main/pyproject.toml"

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Parse TOML content
        data = tomllib.loads(response.text)
        return data.get("project", {}).get("version")

    except requests.exceptions.RequestException as e:
        # Network errors are expected (offline, etc.) - fail silently
        return None
    except Exception as e:
        print(f"Warning: Could not parse remote version: {e}")
        return None


def check_for_updates(silent: bool = False) -> bool:
    """
    Check for newer version on GitHub and prompt user to update.

    Args:
        silent: If True, don't print anything unless update available

    Returns:
        True if update is available, False otherwise
    """
    local_version = get_local_version()
    if not local_version:
        return False

    remote_version = get_remote_version()
    if not remote_version:
        # Couldn't fetch remote version - fail silently (offline, network issues, etc.)
        return False

    try:
        comparison = compare_versions(local_version, remote_version)

        if comparison < 0:
            # Update available
            print("\n" + "=" * 70)
            print(f"UPDATE AVAILABLE: v{local_version} → v{remote_version}")
            print("=" * 70)
            print(f"Current version:  {local_version}")
            print(f"Latest version:   {remote_version}")
            print("\nTo update, run:")
            print("  git pull origin main")
            print("  pip install -e .")
            print("\nOr clone fresh:")
            print("  git clone https://github.com/stevehollx/climb-analyzer.git")
            print("=" * 70 + "\n")
            return True

        elif comparison > 0:
            # Local version ahead of remote (development version)
            if not silent:
                print(f"Running development version {local_version} (ahead of release {remote_version})")
            return False

        else:
            # Up to date
            if not silent:
                print(f"✓ climb-analyzer v{local_version} (up to date)")
            return False

    except Exception as e:
        if not silent:
            print(f"Warning: Version check failed: {e}")
        return False


def main():
    """CLI entry point for testing."""
    check_for_updates(silent=False)


if __name__ == "__main__":
    main()
