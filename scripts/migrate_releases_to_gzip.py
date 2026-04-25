#!/usr/bin/env python3
"""
Migrate existing global-road-and-trail-climbs GitHub releases to gzipped SQLite.

For each per-region release:
  1. Download the raw SQLite assets (single .sqlite or split .sqlite.001/.002/...)
  2. If split, concatenate into a single .sqlite
  3. Gzip the .sqlite to .sqlite.gz (streaming, ~3x smaller)
  4. If the .sqlite.gz exceeds 1.95 GB, split into .sqlite.gz.001/.002/...
  5. Upload the new gz asset(s) to the release
  6. Delete the old raw .sqlite asset(s) and their .sha256
  7. Update the region README.md in the repo folder tree to reference
     the new gz asset(s)

After all regions are migrated, run the "Index Release Assets" workflow
on the repo to rebuild index.json.

Usage:
    python scripts/migrate_releases_to_gzip.py --dry-run
    python scripts/migrate_releases_to_gzip.py --region california
    python scripts/migrate_releases_to_gzip.py --all
    python scripts/migrate_releases_to_gzip.py --all --work-dir /tmp/migrate

Safety:
  - Dry-run mode shows all actions without making changes
  - New assets are uploaded BEFORE old assets are deleted, so the release
    is never without a SQLite at any point
  - SHA256 verification after download and after reassembly
  - Idempotent: skips regions that already have .sqlite.gz assets
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

REPO = "stevehollx/global-road-and-trail-climbs"
SPLIT_THRESHOLD = 1_950_000_000  # 1.95 GB
CHUNK_SIZE = 1_900_000_000       # 1.9 GB per chunk
DEFAULT_WORK_DIR = Path("/tmp/gz_migrate")


# ---------- GitHub API helpers ----------

def _gh(args: List[str], capture: bool = True) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=capture,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh {' '.join(args)} failed ({result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def list_releases() -> List[Dict]:
    """Return all releases for the repo (including drafts)."""
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
                {
                    "id": a["id"],
                    "name": a["name"],
                    "size": a.get("size", 0),
                    "url": a.get("browser_download_url", ""),
                }
                for a in r.get("assets", [])
            ],
        })
    return releases


def delete_asset(asset_id: int) -> None:
    """Delete a release asset by ID."""
    _gh(["api", "-X", "DELETE", f"repos/{REPO}/releases/assets/{asset_id}"])


def upload_asset(release_id: int, local_file: Path) -> None:
    """Upload a file as a release asset."""
    _gh([
        "release", "upload",
        f"--repo", REPO,
        "--clobber",
        get_release_tag_from_id(release_id),
        str(local_file),
    ], capture=False)


def get_release_tag_from_id(release_id: int) -> str:
    """Look up a release's tag name from its numeric ID."""
    out = _gh([
        "api", f"repos/{REPO}/releases/{release_id}",
        "-q", ".tag_name",
    ])
    return out.strip()


def get_release_body(release_id: int) -> str:
    """Fetch the current release body markdown."""
    out = _gh([
        "api", f"repos/{REPO}/releases/{release_id}",
        "-q", ".body",
    ])
    return out.rstrip("\n")


def update_release_body(release_id: int, new_body: str) -> None:
    """PATCH the release body."""
    _gh([
        "api", "-X", "PATCH", f"repos/{REPO}/releases/{release_id}",
        "-f", f"body={new_body}",
    ])


# ---------- Download helpers ----------

def download(url: str, dest: Path) -> None:
    """Download a URL to a local file, streaming with progress."""
    print(f"    Downloading {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "migrate-gzip/1.0"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -1
        while True:
            chunk = response.read(8 * 1024 * 1024)  # 8 MB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                if pct != last_pct and pct % 10 == 0:
                    mb = downloaded / (1024**2)
                    total_mb = total / (1024**2)
                    print(f"      {pct}% ({mb:.0f}/{total_mb:.0f} MB)")
                    last_pct = pct


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_expected_checksums(sha256_url: str) -> Dict[str, str]:
    """Fetch and parse a sha256sum-format file from a URL.

    Returns a dict of {filename: hex_digest}. Each line is expected to be in
    the format '<hex_digest>  <filename>' (standard sha256sum output).
    """
    try:
        req = urllib.request.Request(sha256_url, headers={"User-Agent": "migrate-gzip/1.0"})
        with urllib.request.urlopen(req) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    ⚠️  Could not fetch checksum file: {e}")
        return {}

    result = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "<hex>  <filename>" (two spaces, standard) or "<hex> <filename>"
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[1].strip()] = parts[0].strip().lower()
    return result


def verify_or_raise(path: Path, expected_hex: str) -> None:
    """Compute sha256 of a file and raise if it doesn't match the expected value."""
    actual = sha256_file(path).lower()
    expected = expected_hex.lower()
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}. "
            "Download is corrupt — aborting migration of this release."
        )
    print(f"    ✓ Checksum verified: {path.name}")


# ---------- Core operations ----------

def identify_sqlite_assets(assets: List[Dict]) -> Dict:
    """Classify release assets into raw, split-raw, gz, split-gz, and sha256 buckets."""
    single_sqlite = None
    split_raw = []     # [(name, url, size)] for .sqlite.001/.002/...
    single_gz = None
    split_gz = []      # for .sqlite.gz.001/.002/...
    raw_sha256 = None
    gz_sha256 = None

    for a in assets:
        name = a["name"]
        if re.match(r".*\.sqlite\.gz\.\d{3}$", name):
            split_gz.append(a)
        elif name.endswith(".sqlite.gz"):
            single_gz = a
        elif name.endswith(".sqlite.gz.sha256"):
            gz_sha256 = a
        elif re.match(r".*\.sqlite\.\d{3}$", name):
            split_raw.append(a)
        elif name.endswith(".sqlite"):
            single_sqlite = a
        elif name.endswith(".sqlite.sha256"):
            raw_sha256 = a

    split_raw.sort(key=lambda a: a["name"])
    split_gz.sort(key=lambda a: a["name"])

    return {
        "single_sqlite": single_sqlite,
        "split_raw": split_raw,
        "single_gz": single_gz,
        "split_gz": split_gz,
        "raw_sha256": raw_sha256,
        "gz_sha256": gz_sha256,
    }


def assemble_sqlite(
    single_sqlite: Optional[Dict],
    split_raw: List[Dict],
    raw_sha256_asset: Optional[Dict],
    work_dir: Path,
) -> Tuple[Path, str]:
    """Download and reassemble the raw SQLite for a release, verifying checksums
    against the uploaded .sha256 sidecar before compression.

    Returns (path_to_assembled_sqlite, base_name).

    Raises RuntimeError if any chunk fails checksum verification.
    """
    expected_checksums: Dict[str, str] = {}
    if raw_sha256_asset:
        print(f"    Fetching expected checksums from {raw_sha256_asset['name']}")
        expected_checksums = fetch_expected_checksums(raw_sha256_asset["url"])
        if expected_checksums:
            print(f"    Loaded {len(expected_checksums)} checksum(s)")
        else:
            print("    ⚠️  Checksum file was empty or unparseable — will rely on size check only")

    def _download_and_verify(asset: Dict, local_path: Path) -> None:
        # Skip re-download if size already matches (resumes across runs)
        if not local_path.exists() or local_path.stat().st_size != asset["size"]:
            download(asset["url"], local_path)
        # Verify against known checksum if we have one
        if asset["name"] in expected_checksums:
            verify_or_raise(local_path, expected_checksums[asset["name"]])
        elif expected_checksums:
            print(f"    ⚠️  No expected checksum for {asset['name']} — skipping verification")

    if single_sqlite:
        out = work_dir / single_sqlite["name"]
        _download_and_verify(single_sqlite, out)
        return out, single_sqlite["name"]

    if not split_raw:
        raise RuntimeError("No raw SQLite assets found in release")

    # Download + verify each chunk, then concatenate
    chunk_paths = []
    for chunk_asset in split_raw:
        chunk_path = work_dir / chunk_asset["name"]
        _download_and_verify(chunk_asset, chunk_path)
        chunk_paths.append(chunk_path)

    # Derive base name: "California_....sqlite.001" -> "California_....sqlite"
    base_name = split_raw[0]["name"].rsplit(".", 1)[0]
    assembled = work_dir / base_name
    print(f"    Assembling {len(chunk_paths)} verified chunks -> {assembled.name}")
    with open(assembled, "wb") as out:
        for cp in chunk_paths:
            with open(cp, "rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    return assembled, base_name


def gzip_file(src: Path) -> Path:
    """Stream-compress src to src + '.gz'. Returns the gz path."""
    import gzip
    gz_path = src.with_name(src.name + ".gz")
    src_gb = src.stat().st_size / (1024**3)
    print(f"    Compressing {src.name} ({src_gb:.2f} GB) -> {gz_path.name}")
    with open(src, "rb") as r, gzip.open(gz_path, "wb", compresslevel=6) as w:
        shutil.copyfileobj(r, w, length=1024 * 1024)
    gz_gb = gz_path.stat().st_size / (1024**3)
    ratio = src.stat().st_size / gz_path.stat().st_size
    print(f"    ✓ Compressed to {gz_gb:.2f} GB ({ratio:.1f}x)")
    return gz_path


def split_if_needed(gz_path: Path) -> Tuple[List[Path], Optional[Path]]:
    """Split gz_path into .001/.002/... chunks if it exceeds SPLIT_THRESHOLD.

    Returns (list_of_paths_to_upload, optional_sha256_file).
    If no split needed, returns ([gz_path], None).
    """
    if gz_path.stat().st_size < SPLIT_THRESHOLD:
        return [gz_path], None

    print(f"    Gzipped file exceeds 1.95 GB — splitting into chunks")
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

    # Write a .sha256 sidecar alongside the gz
    sha_path = gz_path.parent / f"{gz_path.name}.sha256"
    with open(sha_path, "w") as sf:
        for name, digest in sorted(sums.items()):
            sf.write(f"{digest}  {name}\n")

    # Delete the monolithic gz to free disk
    gz_path.unlink()
    return chunks, sha_path


def get_region_readme_path(release_tag: str) -> Optional[str]:
    """Map a release tag (e.g. 'california-v2.4.0') to its README path in the repo."""
    # Strip "-vX.Y.Z" suffix
    m = re.match(r"^(.*?)-v\d+\.\d+\.\d+$", release_tag)
    if not m:
        return None
    name = m.group(1)

    # US states follow: north-america/united-states-of-america/<state>/README.md
    us_states = {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "district-of-columbia", "florida", "georgia",
        "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
        "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada", "new-hampshire",
        "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
        "ohio", "oklahoma", "oregon", "pennsylvania", "rhode-island", "south-carolina",
        "south-dakota", "tennessee", "texas", "utah", "vermont", "virginia",
        "washington", "west-virginia", "wisconsin", "wyoming",
    }
    if name in us_states:
        return f"north-america/united-states-of-america/{name}/README.md"

    # Other regions - caller will have to specify
    return None


def read_repo_file(path: str) -> Tuple[str, str]:
    """Read a file from the repo. Returns (content, sha)."""
    out = _gh(["api", f"repos/{REPO}/contents/{path}", "-q", "{content: .content, sha: .sha}"])
    data = json.loads(out)
    import base64
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def write_repo_file(path: str, content: str, message: str, sha: str) -> None:
    """Update a file in the repo via the API."""
    import base64
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    _gh([
        "api", "-X", "PUT",
        f"repos/{REPO}/contents/{path}",
        "-f", f"message={message}",
        "-f", f"content={encoded}",
        "-f", f"sha={sha}",
    ])


def rewrite_release_body(body: str, old_asset_names: List[str], new_assets: List[Tuple[str, int, str]]) -> str:
    """Rewrite a GitHub release body to reflect the new gzipped assets.

    Removes any lines/table-rows that reference old raw SQLite filenames and
    appends a fresh "Gzipped SQLite" section. Leaves XLSX entries and other
    content intact.

    Args:
        body: Current release body markdown.
        old_asset_names: Filenames of raw SQLite (and sha256) assets that
                         were just deleted - any line mentioning these is
                         removed from the body.
        new_assets: [(filename, size_bytes, download_url), ...] for the new
                    gzipped assets.

    Returns:
        The updated release body markdown.
    """
    # Remove any line referencing the old raw filenames
    kept_lines = []
    for line in body.split("\n"):
        if any(old in line for old in old_asset_names):
            continue
        kept_lines.append(line)
    updated = "\n".join(kept_lines)

    # Strip any existing "Split Database Files" section since it referenced
    # the old raw chunks. Find it and remove until the next top-level heading
    # or end of document.
    lines = updated.split("\n")
    cleaned = []
    skip = False
    for line in lines:
        if re.match(r"^#+\s*Split Database Files", line, re.IGNORECASE):
            skip = True
            continue
        if skip:
            # Stop skipping at the next markdown heading or separator
            if line.startswith("#") or line.startswith("---"):
                skip = False
                cleaned.append(line)
            continue
        cleaned.append(line)
    updated = "\n".join(cleaned)

    # Append a new "Gzipped SQLite (preferred)" section just before any
    # trailing "---" horizontal rule / footer, or at the end of the doc.
    new_section_lines = [
        "",
        "### Gzipped SQLite Database (preferred)",
        "",
        "Download this smaller compressed file and decompress on device:",
        "",
        "| File | Size | Format |",
        "|------|------|--------|",
    ]
    for fname, size, url in new_assets:
        size_mb = size / (1024**2)
        new_section_lines.append(f"| [{fname}]({url}) | {size_mb:.1f} MB | gzip |")
    new_section_lines.append("")
    new_section = "\n".join(new_section_lines)

    # Try to insert before the footer divider
    footer_match = re.search(r"\n---\s*\n", updated)
    if footer_match:
        insert_pos = footer_match.start()
        updated = updated[:insert_pos] + new_section + updated[insert_pos:]
    else:
        updated = updated.rstrip() + "\n" + new_section + "\n"

    return updated


def rewrite_readme_for_gz(content: str, old_asset_names: List[str], new_assets: List[Tuple[str, int, str]]) -> str:
    """Replace references to old raw sqlite assets with the new gz assets.

    new_assets: list of (filename, size_bytes, download_url)
    Returns the updated README content.
    """
    updated = content

    # Remove lines that reference any old raw sqlite file or its checksum
    filtered_lines = []
    for line in updated.split("\n"):
        skip = False
        for old_name in old_asset_names:
            if old_name in line:
                skip = True
                break
        if not skip:
            filtered_lines.append(line)
    updated = "\n".join(filtered_lines)

    # Append a new section listing the gz assets (after a "Files" heading if
    # present, or at the end of the file as a fallback).
    new_section = ["", "### Gzipped SQLite (preferred)", "", "| File | Size | Format |", "|------|------|--------|"]
    for fname, size, url in new_assets:
        size_mb = size / (1024**2)
        new_section.append(f"| [{fname}]({url}) | {size_mb:.1f} MB | gzip |")
    new_section.append("")
    updated = updated.rstrip() + "\n" + "\n".join(new_section) + "\n"
    return updated


# ---------- Per-region migration ----------

def migrate_release(release: Dict, work_dir: Path, dry_run: bool) -> bool:
    """Migrate a single release from raw sqlite to gzipped sqlite.

    Returns True on success, False on skip/failure.
    """
    tag = release["tag_name"]
    release_id = release["id"]
    print(f"\n=== {tag} (release_id={release_id}) ===")

    assets = release.get("assets") or []
    if not assets:
        print("  No assets found, skipping")
        return False

    classified = identify_sqlite_assets(assets)

    # Skip if already migrated (has .gz assets)
    if classified["single_gz"] or classified["split_gz"]:
        print("  ✓ Already has gzipped assets, skipping")
        return False

    # Must have at least one raw sqlite to migrate
    if not classified["single_sqlite"] and not classified["split_raw"]:
        print("  ⚠️  No raw SQLite found, nothing to migrate")
        return False

    region_work = work_dir / tag
    region_work.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download + assemble raw sqlite, verifying checksums against
        #    the uploaded .sha256 sidecar before proceeding. A checksum
        #    mismatch here aborts this release's migration so we don't
        #    propagate a corrupt download into the gzipped files.
        print("  → Downloading raw SQLite")
        sqlite_path, base_sqlite_name = assemble_sqlite(
            classified["single_sqlite"],
            classified["split_raw"],
            classified["raw_sha256"],
            region_work,
        )
        print(f"    Raw SQLite: {sqlite_path.name} ({sqlite_path.stat().st_size / (1024**3):.2f} GB)")

        # 2. Gzip
        print("  → Compressing with gzip")
        gz_path = gzip_file(sqlite_path)

        # Free disk: raw sqlite no longer needed
        sqlite_path.unlink()

        # 3. Split gz if needed
        print("  → Checking if gz split is needed")
        upload_paths, sha_path = split_if_needed(gz_path)
        for p in upload_paths:
            print(f"    Will upload: {p.name} ({p.stat().st_size / (1024**2):.1f} MB)")

        if dry_run:
            print("  [dry-run] would upload new gz assets + delete old raw assets + update README")
            return True

        # 4. Upload new assets FIRST (before deleting anything)
        print("  → Uploading new gz assets")
        for p in upload_paths:
            upload_asset(release_id, p)
        if sha_path:
            upload_asset(release_id, sha_path)

        # 5. Delete old raw sqlite assets (ordering: delete after upload)
        print("  → Deleting old raw SQLite assets")
        old_asset_names = []
        to_delete = []
        if classified["single_sqlite"]:
            to_delete.append(classified["single_sqlite"])
        to_delete.extend(classified["split_raw"])
        if classified["raw_sha256"]:
            to_delete.append(classified["raw_sha256"])
        for asset in to_delete:
            print(f"    Deleting {asset['name']}")
            delete_asset(asset["id"])
            old_asset_names.append(asset["name"])

        # 6. Re-fetch the release to get canonical URLs for the newly uploaded
        #    assets. We'll use these for both the README update and the
        #    release body update.
        try:
            fresh_release_json = _gh([
                "api", f"repos/{REPO}/releases/{release_id}",
                "-q", ".assets",
            ])
            fresh_assets = json.loads(fresh_release_json)
        except Exception as e:
            print(f"    ⚠️  Failed to re-fetch release assets: {e}")
            fresh_assets = []

        new_names = {p.name for p in upload_paths}
        new_assets_info: List[Tuple[str, int, str]] = []
        for a in fresh_assets:
            if a["name"] in new_names:
                new_assets_info.append((a["name"], a["size"], a["browser_download_url"]))

        # 7. Update the GitHub release body to reflect the new gz assets.
        if new_assets_info:
            print("  → Updating release body")
            try:
                current_body = get_release_body(release_id)
                new_body = rewrite_release_body(current_body, old_asset_names, new_assets_info)
                if new_body != current_body:
                    update_release_body(release_id, new_body)
                    print("    ✓ Release body updated")
                else:
                    print("    (release body unchanged, skipping)")
            except Exception as e:
                print(f"    ⚠️  Failed to update release body: {e}")

        # 8. Update the README.md in the folder tree (best-effort).
        readme_path = get_region_readme_path(tag)
        if readme_path and new_assets_info:
            print(f"  → Updating {readme_path}")
            try:
                content, sha = read_repo_file(readme_path)
                new_content = rewrite_readme_for_gz(content, old_asset_names, new_assets_info)
                if new_content != content:
                    write_repo_file(
                        readme_path,
                        new_content,
                        f"Migrate {tag} to gzipped SQLite",
                        sha,
                    )
                    print("    ✓ README updated")
                else:
                    print("    (README content unchanged, skipping write)")
            except Exception as e:
                print(f"    ⚠️  Failed to update README: {e}")

        print(f"  ✓ Migrated {tag}")
        return True

    finally:
        # Always clean up the working directory to free disk space
        if region_work.exists():
            shutil.rmtree(region_work, ignore_errors=True)


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Migrate all releases with raw SQLite assets")
    parser.add_argument("--region", help="Migrate only this region (by tag name, e.g. 'california-v2.4.0' or just 'california')")
    parser.add_argument("--dry-run", action="store_true", help="Do the full migration work locally (download + compress + split) but skip all uploads/deletes/README updates")
    parser.add_argument("--list-only", action="store_true", help="Just print what would be migrated without downloading anything")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help=f"Working directory for downloads (default: {DEFAULT_WORK_DIR})")
    args = parser.parse_args()

    if not args.all and not args.region:
        parser.error("Must specify either --all or --region")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {args.work_dir}")
    print(f"Dry-run: {args.dry_run}")

    releases = list_releases()
    print(f"Fetched {len(releases)} releases from {REPO}")

    # Filter by region if specified
    if args.region:
        region_lower = args.region.lower()
        releases = [
            r for r in releases
            if r["tag_name"].lower() == region_lower
            or r["tag_name"].lower().startswith(region_lower + "-v")
        ]
        if not releases:
            print(f"No release matching '{args.region}'")
            sys.exit(1)

    # Filter to releases that have raw sqlite (i.e. still need migrating)
    releases_to_process = []
    for r in releases:
        classified = identify_sqlite_assets(r.get("assets") or [])
        has_raw = classified["single_sqlite"] is not None or bool(classified["split_raw"])
        has_gz = classified["single_gz"] is not None or bool(classified["split_gz"])
        if has_raw and not has_gz:
            releases_to_process.append(r)

    print(f"\n{len(releases_to_process)} release(s) will be migrated:")
    for r in releases_to_process:
        # Summarize what we're going to do per release
        c = identify_sqlite_assets(r.get("assets") or [])
        if c["single_sqlite"]:
            raw_size_mb = c["single_sqlite"]["size"] / (1024**2)
            raw_desc = f"single .sqlite ({raw_size_mb:.0f} MB)"
        elif c["split_raw"]:
            total_mb = sum(a["size"] for a in c["split_raw"]) / (1024**2)
            raw_desc = f"{len(c['split_raw'])} split chunks ({total_mb:.0f} MB total)"
        else:
            raw_desc = "?"
        print(f"  - {r['tag_name']}: {raw_desc}")

    if not releases_to_process:
        print("Nothing to do.")
        return

    if args.list_only:
        print("\n(list-only mode, exiting without downloading)")
        return

    if not args.dry_run and not args.yes:
        try:
            resp = input("\nProceed? [y/N]: ").strip().lower()
        except EOFError:
            print("\n(no TTY, aborting — use --yes to skip confirmation)")
            return
        if resp != "y":
            print("Aborted.")
            return

    migrated = 0
    failed = 0
    for r in releases_to_process:
        try:
            if migrate_release(r, args.work_dir, args.dry_run):
                migrated += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Summary ===")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped/failed: {failed}")
    if migrated > 0 and not args.dry_run:
        print()
        print("Next steps:")
        print(f"  1. Trigger the index rebuild workflow:")
        print(f"     gh workflow run 'Index Release Assets' --repo {REPO}")
        print(f"  2. Verify index.json contains the new gz fields")


if __name__ == "__main__":
    main()
