#!/usr/bin/env python3
"""
Upload local sqlite files to GitHub releases as gzipped sqlite.

For each US state release that has xlsx but no sqlite.gz:
  1. Find the latest matching local .sqlite file in output/
  2. Gzip it streaming (~3x smaller)
  3. If gz exceeds 1.95 GB, split into numbered chunks with sha256
  4. Upload gz (or chunks) to the existing GitHub release
  5. Skip states that already have sqlite.gz on GitHub

Usage:
    python scripts/upload_sqlite_gz_to_releases.py --dry-run
    python scripts/upload_sqlite_gz_to_releases.py --state idaho
    python scripts/upload_sqlite_gz_to_releases.py --all
    python scripts/upload_sqlite_gz_to_releases.py --all --yes

Notes:
  - Requires GITHUB_TOKEN or 'gh' CLI logged in for upload
  - Output directory is read from ./output/
  - Compressed files are created in /tmp/sqlite_gz_upload/ (cleaned up after)
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = "stevehollx/global-road-and-trail-climbs"
OUTPUT_DIR = Path("output")
WORK_DIR = Path("/tmp/sqlite_gz_upload")
SPLIT_THRESHOLD = 1_950_000_000   # 1.95 GB
CHUNK_SIZE = 1_900_000_000        # 1.9 GB per chunk
APP_VERSION = "2.4.0"


# ---------- GitHub helpers ----------

def _gh(args: List[str], capture: bool = True) -> str:
    result = subprocess.run(
        ["gh"] + args, capture_output=capture, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def list_releases() -> List[Dict]:
    out = _gh(["api", f"repos/{REPO}/releases?per_page=100"])
    raw = json.loads(out)
    releases = []
    for r in raw:
        releases.append({
            "id": r["id"],
            "tag_name": r.get("tag_name") or "",
            "name": r.get("name") or "",
            "draft": r.get("draft", False),
            "assets": [
                {"id": a["id"], "name": a["name"], "size": a.get("size", 0)}
                for a in r.get("assets", [])
            ],
        })
    return releases


def has_sqlite_gz(release: Dict) -> bool:
    for a in release.get("assets", []):
        if a["name"].endswith(".sqlite.gz") or re.match(r".*\.sqlite\.gz\.\d{3}$", a["name"]):
            return True
    return False


def release_tag_to_state_key(tag: str) -> str:
    """Convert 'idaho-v2.4.0' to 'idaho'."""
    return re.sub(r"-v\d+\.\d+\.\d+$", "", tag)


def state_key_to_filename_prefix(state_key: str) -> str:
    """Convert 'new-york' to 'New_York' for filename matching."""
    return "_".join(w.capitalize() for w in state_key.split("-"))


# ---------- File helpers ----------

def find_latest_sqlite(state_prefix: str) -> Optional[Path]:
    """Find the most recent single-file .sqlite in output/ for a state.

    Prefers v2.4.0 over v2.3.0; prefers a single full file over partitioned
    files (those have _northeast, _northwest, etc. suffixes).
    """
    candidates = sorted(
        OUTPUT_DIR.glob(f"{state_prefix}_climbs_*.sqlite"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Exclude partition files (end with known partition suffixes like _northeast.sqlite)
    PARTITION_SUFFIXES = ("_northeast", "_northwest", "_southeast", "_southwest",
                          "_north", "_south", "_east", "_west", "_other",
                          "_norcal", "_socal")
    single_file_candidates = []
    for p in candidates:
        stem = p.stem  # no extension
        is_partition = any(stem.endswith(suf) for suf in PARTITION_SUFFIXES)
        if not is_partition:
            single_file_candidates.append(p)

    if not single_file_candidates:
        return None

    # Prefer v2.4.0 over v2.3.0 over others
    for version in ("v2.4.0", "v2.3.0"):
        for p in single_file_candidates:
            if version in p.name:
                return p
    # Fallback: most recent
    return single_file_candidates[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_sqlite(src: Path, dest_dir: Path) -> Path:
    """Stream-gzip src into dest_dir/<src.name>.gz. Returns gz path."""
    gz_path = dest_dir / (src.name + ".gz")
    src_gb = src.stat().st_size / (1024 ** 3)
    print(f"    Compressing {src.name} ({src_gb:.2f} GB)...")
    with open(src, "rb") as r, gzip.open(gz_path, "wb", compresslevel=6) as w:
        shutil.copyfileobj(r, w, length=8 * 1024 * 1024)
    gz_gb = gz_path.stat().st_size / (1024 ** 3)
    ratio = src.stat().st_size / gz_path.stat().st_size
    print(f"    Compressed to {gz_gb:.2f} GB ({ratio:.1f}x)")
    return gz_path


def split_if_needed(gz_path: Path) -> Tuple[List[Path], Optional[Path]]:
    """Split gz_path into chunks if > 1.95 GB. Returns (files_to_upload, sha_path)."""
    if gz_path.stat().st_size < SPLIT_THRESHOLD:
        return [gz_path], None

    print(f"    File exceeds 1.95 GB — splitting into chunks")
    chunks = []
    sums = {}
    with open(gz_path, "rb") as f:
        idx = 1
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            chunk_name = gz_path.parent / f"{gz_path.name}.{idx:03d}"
            with open(chunk_name, "wb") as out:
                out.write(data)
            sums[chunk_name.name] = hashlib.sha256(data).hexdigest()
            chunks.append(chunk_name)
            print(f"      Created {chunk_name.name} ({len(data) / (1024**3):.2f} GB)")
            idx += 1

    sha_path = gz_path.parent / f"{gz_path.name}.sha256"
    with open(sha_path, "w") as sf:
        for name, digest in sorted(sums.items()):
            sf.write(f"{digest}  {name}\n")

    gz_path.unlink()  # Free space
    return chunks, sha_path


def upload_files(tag: str, files: List[Path], sha_path: Optional[Path]) -> bool:
    """Upload files to a GitHub release. Returns True on success."""
    all_uploads = list(files)
    if sha_path:
        all_uploads.append(sha_path)

    for f in all_uploads:
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"    Uploading {f.name} ({size_mb:.1f} MB)...")
        try:
            _gh([
                "release", "upload",
                "--repo", REPO,
                "--clobber",
                tag,
                str(f),
            ], capture=False)
            print(f"    ✓ Uploaded {f.name}")
        except RuntimeError as e:
            print(f"    ✗ Failed to upload {f.name}: {e}")
            return False
    return True


# ---------- Core logic ----------

def process_state(release: Dict, output_dir: Path, work_dir: Path, dry_run: bool) -> bool:
    """Process a single release. Returns True on success."""
    tag = release["tag_name"]
    state_key = release_tag_to_state_key(tag)
    state_prefix = state_key_to_filename_prefix(state_key)

    print(f"\n=== {state_key} (tag: {tag}) ===")

    # Find the local sqlite
    sqlite_path = find_latest_sqlite(state_prefix)
    if not sqlite_path:
        print(f"  ⚠️  No local sqlite found for prefix '{state_prefix}' — skipping")
        return False

    sqlite_gb = sqlite_path.stat().st_size / (1024 ** 3)
    print(f"  Local sqlite: {sqlite_path.name} ({sqlite_gb:.2f} GB)")

    if dry_run:
        # Estimate compressed size
        estimated_gz_gb = sqlite_gb * 0.4  # ~40% of original after gzip
        needs_split = (sqlite_gb * 0.4 * 1024**3) > SPLIT_THRESHOLD
        print(f"  [dry-run] Would gzip to ~{estimated_gz_gb:.2f} GB (estimated)")
        if needs_split:
            print(f"  [dry-run] Would split into multiple chunks")
        print(f"  [dry-run] Would upload to {tag}")
        return True

    state_work = work_dir / state_key
    state_work.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Gzip
        gz_path = gzip_sqlite(sqlite_path, state_work)

        # 2. Split if needed
        upload_files_list, sha_path = split_if_needed(gz_path)

        # 3. Upload
        print(f"  Uploading {len(upload_files_list)} file(s) to {tag}...")
        success = upload_files(tag, upload_files_list, sha_path)

        if success:
            print(f"  ✓ Done: {state_key}")
        else:
            print(f"  ✗ Upload failed for {state_key}")

        return success

    finally:
        if state_work.exists():
            shutil.rmtree(state_work, ignore_errors=True)


# ---------- US state name lists ----------

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new-hampshire", "new-jersey", "new-mexico", "new-york", "north-carolina",
    "north-dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode-island", "south-carolina", "south-dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west-virginia",
    "wisconsin", "wyoming",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Process all US state releases missing sqlite.gz")
    parser.add_argument("--state", help="Process only this state (e.g. 'idaho' or 'new-york')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without uploading")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Local output directory (default: output/)")
    args = parser.parse_args()

    if not args.all and not args.state:
        parser.error("Specify --all or --state <name>")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {WORK_DIR}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dry-run: {args.dry_run}")

    print(f"\nFetching releases from {REPO}...")
    releases = list_releases()
    print(f"Found {len(releases)} releases")

    # Filter to US state releases that are missing sqlite.gz
    to_process = []
    already_done = []
    skipped_no_state = []

    for r in releases:
        state_key = release_tag_to_state_key(r["tag_name"])
        if state_key not in US_STATES:
            skipped_no_state.append(state_key)
            continue

        if args.state and state_key != args.state.lower().replace(" ", "-").replace("_", "-"):
            continue

        if has_sqlite_gz(r):
            already_done.append(state_key)
        else:
            to_process.append(r)

    print(f"\nStates already with sqlite.gz ({len(already_done)}): {', '.join(sorted(already_done))}")
    print(f"\nStates to process ({len(to_process)}):")
    for r in to_process:
        state_key = release_tag_to_state_key(r["tag_name"])
        state_prefix = state_key_to_filename_prefix(state_key)
        sqlite_path = find_latest_sqlite(state_prefix)
        if sqlite_path:
            size_gb = sqlite_path.stat().st_size / (1024**3)
            print(f"  - {state_key}: {sqlite_path.name} ({size_gb:.2f} GB)")
        else:
            print(f"  - {state_key}: NO LOCAL SQLITE FOUND")

    if not to_process:
        print("\nNothing to do.")
        return

    if not args.dry_run and not args.yes:
        try:
            resp = input("\nProceed? [y/N]: ").strip().lower()
        except EOFError:
            print("\n(no TTY — use --yes to skip confirmation)")
            return
        if resp != "y":
            print("Aborted.")
            return

    succeeded = 0
    failed = 0
    for r in to_process:
        try:
            ok = process_state(r, args.output_dir, WORK_DIR, args.dry_run)
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n=== Summary ===")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed/skipped: {failed}")
    if succeeded and not args.dry_run:
        print()
        print("Next steps:")
        print(f"  Trigger index rebuild: gh workflow run 'Index Release Assets' --repo {REPO}")


if __name__ == "__main__":
    main()
