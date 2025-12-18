"""
Minimal batch cleanup utilities for climb-analyzer.
Provides:
 - get_cleanup_targets() -> dict
 - perform_cleanup(targets, verbose=False) -> None
 - prompt_cleanup(batch_mode=False, default_yes=False) -> bool

This file is intentionally conservative and safe: it only targets known
directories/files used by the application and asks for confirmation before
removing anything unless default_yes=True.
"""
import shutil
from pathlib import Path
from typing import Dict, List


def get_cleanup_targets() -> Dict[str, List[Path]]:
    """Scan the workspace and return a dictionary of cleanup targets.

    Returns a dict with keys: 'planet_files', 'indices', 'elevation', 'checkpoints',
    each mapping to a list of Path objects that would be removed.
    """
    targets = {
        "planet_files": [],
        "indices": [],
        "elevation": [],
        "checkpoints": [],
        "other": [],
    }

    root = Path("")

    # Planet OSM .pbf files
    planet_dir = root / "data/planet-osm"
    if planet_dir.exists() and planet_dir.is_dir():
        for p in planet_dir.glob("*.pbf"):
            targets["planet_files"].append(p)
            # index files
            idx = p.with_suffix(".pbf.idx")
            if idx.exists():
                targets["indices"].append(idx)
            # potential rtree dir
            idxdir = p.with_suffix("")
            if idxdir.exists() and idxdir.is_dir():
                targets["indices"].append(idxdir)

    # Elevation data directory
    elevation_dir = root / "data/elevation_data"
    if elevation_dir.exists() and elevation_dir.is_dir():
        for child in elevation_dir.rglob("*"):
            targets["elevation"].append(child)

    # Checkpoints
    checkpoints_dir = root / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        for child in checkpoints_dir.rglob("*"):
            targets["checkpoints"].append(child)

    # Output files that look like analysis results
    output_dir = root / "output"
    if output_dir.exists() and output_dir.is_dir():
        for child in output_dir.iterdir():
            targets["other"].append(child)

    return targets


def _safe_remove_path(p: Path, verbose: bool = False):
    try:
        if p.is_dir():
            shutil.rmtree(p)
            if verbose:
                print(f"Removed directory: {p}")
        else:
            p.unlink()
            if verbose:
                print(f"Removed file: {p}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not remove {p}: {e}")


def perform_cleanup(targets: Dict[str, List[Path]], verbose: bool = False) -> None:
    """Perform cleanup of the provided targets.

    This will remove files and directories listed in the targets dict. Be careful
    when calling this function; it performs irreversible deletions.
    """
    # Flatten all paths
    all_paths = []
    for lst in targets.values():
        all_paths.extend(lst)

    # Sort paths so files are removed before directories (safe order)
    all_paths_sorted = sorted(all_paths, key=lambda p: (p.exists() and p.is_dir(), str(p)))

    for p in all_paths_sorted:
        if not p.exists():
            if verbose:
                print(f"Skipping missing: {p}")
            continue
        _safe_remove_path(p, verbose=verbose)


def prompt_cleanup(batch_mode: bool = False, default_yes: bool = False) -> bool:
    """Prompt the user to confirm cleanup.

    If batch_mode is True and default_yes is True, return True without prompting.
    """
    targets = get_cleanup_targets()
    total_items = sum(len(v) for v in targets.values())

    if total_items == 0:
        print("   No cleanup targets found")
        return False

    if batch_mode and default_yes:
        return True

    print("Cleanup targets found:")
    for k, v in targets.items():
        if v:
            print(f"  {k}: {len(v)} item(s)")

    if default_yes:
        confirm = input("Delete found files and directories? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            return True
        return False
    else:
        confirm = input("Delete found files and directories? [y/N]: ").strip().lower()
        if confirm in ("y", "yes"):
            return True
        return False
