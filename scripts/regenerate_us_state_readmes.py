#!/usr/bin/env python3
"""
Regenerate stale US state README files in the main branch.

Problem: `scripts/upload_sqlite_gz_to_releases.py` added sqlite.gz assets to
existing GitHub releases but never updated the README files under
`north-america/united-states-of-america/<state>/README.md`. The release has
the files; the index looks like it's missing them.

Fix: for each US state, fetch the release's current asset list, rebuild the
README markdown in the same format as `cloud_cache._build_release_markdown`,
and commit updates directly to `main` in one atomic commit.

Usage:
    python3 scripts/regenerate_us_state_readmes.py --dry-run
    python3 scripts/regenerate_us_state_readmes.py --apply
"""

import argparse
import base64
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = "stevehollx/global-road-and-trail-climbs"

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "district-of-columbia", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "new-hampshire",
    "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "puerto-rico", "rhode-island",
    "south-carolina", "south-dakota", "tennessee", "texas", "us-virgin-islands",
    "utah", "vermont", "virginia", "washington", "west-virginia", "wisconsin",
    "wyoming",
}


def gh(args: List[str], capture: bool = True) -> str:
    r = subprocess.run(["gh"] + args, capture_output=capture, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def load_datasets_info(state_key: str) -> Dict[str, List[str]]:
    """Return {'priority':[...], 'actually_used':[...]} from checkpoint if available."""
    from pathlib import Path
    cpbase = Path("data/checkpoint_data")
    if not cpbase.exists():
        return {"priority": [], "actually_used": []}

    # Match state prefix in checkpoint dir names (e.g. Arkansas_all_region_...)
    prefix_candidates = [
        "_".join(w.capitalize() for w in state_key.split("-")),  # Arkansas, DistrictOfColumbia
        state_key.replace("-", ""),                               # arkansas, districtofcolumbia
    ]
    matches = []
    for d in cpbase.iterdir():
        if not d.is_dir():
            continue
        dname = d.name.lower()
        for p in prefix_candidates:
            if dname.startswith(p.lower() + "_"):
                matches.append(d)
                break
    if not matches:
        return {"priority": [], "actually_used": []}
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    dsfile = matches[0] / "datasets_used.json"
    if not dsfile.exists():
        return {"priority": [], "actually_used": []}
    try:
        data = json.loads(dsfile.read_text())
        if isinstance(data, list):
            return {"priority": [], "actually_used": data}
        return {
            "priority": list(data.get("priority", [])),
            "actually_used": list(data.get("actually_used", [])),
        }
    except Exception:
        return {"priority": [], "actually_used": []}


def fmt_size(n: int) -> str:
    return f"{n / (1024 ** 2):.1f} MB"


def parse_version_and_errors(filename: str) -> Tuple[Optional[str], Optional[int]]:
    m = re.search(r"_v(\d+\.\d+\.\d+)_e(\d{4})", filename)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def build_readme(state_key: str, release: Dict) -> str:
    """Rebuild the README markdown from current release assets."""
    tag = release["tag_name"]
    location_name = state_key.replace("-", " ").title()
    date_str = datetime.now().strftime("%Y-%m-%d")
    assets = release.get("assets", [])

    # Parse version/errors from any xlsx filename
    version = "2.4.0"
    errors = 0
    for a in assets:
        if a["name"].endswith(".xlsx"):
            v, e = parse_version_and_errors(a["name"])
            if v:
                version = v
            if e is not None:
                errors = e
            break

    # Guess climb count — we don't have it from the release, so infer from xlsx count
    xlsx_count = sum(1 for a in assets if a["name"].endswith(".xlsx"))
    sqlite_count = sum(
        1 for a in assets
        if a["name"].endswith(".sqlite")
        or re.match(r".*\.sqlite\.\d{3}$", a["name"])
    )
    sqlite_gz_count = sum(
        1 for a in assets
        if a["name"].endswith(".sqlite.gz")
        or re.match(r".*\.sqlite\.gz\.\d{3}$", a["name"])
    )
    log_count = sum(1 for a in assets if "error" in a["name"].lower() or a["name"].endswith(".log"))
    sha_count = sum(1 for a in assets if a["name"].endswith(".sha256"))
    total_size = sum(a.get("size", 0) for a in assets)

    file_parts = []
    if xlsx_count:
        file_parts.append(f"{xlsx_count} Excel file(s)")
    if sqlite_gz_count:
        file_parts.append(f"{sqlite_gz_count} gzipped SQLite")
    elif sqlite_count:
        file_parts.append(f"{sqlite_count} SQLite database(s)")
    if log_count:
        file_parts.append("error log")
    if sha_count:
        file_parts.append("sha256 checksum")
    files_desc = " + ".join(file_parts) if file_parts else "No files"

    # Build body
    md = f"""# {location_name} Climb Analysis

## Info
* Date: {date_str}
* Version: {version}
* Elevation Errors: {errors}
* Files: {files_desc}
* Total Size: {total_size / (1024 ** 2):.1f} MB
* Release Tag: `{tag}`

"""

    # Elevation datasets section
    ds_info = load_datasets_info(state_key)
    priority = ds_info["priority"]
    actually_used = ds_info["actually_used"]
    if priority and actually_used and set(priority) != set(actually_used):
        md += "## Elevation datasets (configured cascade)\n"
        for i, ds in enumerate(priority, 1):
            marker = " — supplied data" if ds in actually_used else ""
            md += f"{i}. {ds}{marker}\n"
        md += f"\n**Datasets that actually supplied elevations:** {', '.join(actually_used)}\n\n"
    elif priority:
        md += "## Elevation datasets (configured cascade)\n"
        for i, ds in enumerate(priority, 1):
            md += f"{i}. {ds}\n"
        md += "\n"
    elif actually_used:
        md += "## Elevation datasets used\n"
        for i, ds in enumerate(actually_used, 1):
            md += f"{i}. {ds}\n"
        md += "\n"

    # Files table — sort: xlsx, sqlite.gz, sqlite, others
    def sort_key(a):
        n = a["name"]
        if n.endswith(".xlsx"):           return (0, n)
        if n.endswith(".sqlite.gz"):      return (1, n)
        if re.match(r".*\.sqlite\.gz\.\d{3}$", n):  return (2, n)
        if n.endswith(".sqlite"):         return (3, n)
        if re.match(r".*\.sqlite\.\d{3}$", n):      return (4, n)
        if n.endswith(".sha256"):         return (5, n)
        return (6, n)

    md += "## Files\n\n"
    md += "| File | Size | Format |\n"
    md += "|------|------|--------|\n"
    for a in sorted(assets, key=sort_key):
        name = a["name"]
        size_mb = a.get("size", 0) / (1024 ** 2)
        url = a.get("browser_download_url", "")
        if name.endswith(".xlsx"):
            fmt = "Excel"
        elif name.endswith(".sqlite.gz") or re.match(r".*\.sqlite\.gz\.\d{3}$", name):
            fmt = "SQLite (gzipped)"
        elif name.endswith(".sqlite") or re.match(r".*\.sqlite\.\d{3}$", name):
            fmt = "SQLite"
        elif name.endswith(".sha256"):
            fmt = "SHA256 checksum"
        elif "error" in name.lower() or name.endswith(".log"):
            fmt = "Log"
        else:
            fmt = "File"
        md += f"| [{name}]({url}) | {size_mb:.1f} MB | {fmt} |\n"

    md += f"\n## Release\n\n[View Release](https://github.com/{REPO}/releases/tag/{tag})\n"
    md += "\n---\n\n*This README was automatically regenerated by `scripts/regenerate_us_state_readmes.py`.*\n"
    return md


def fetch_current_readme(repo_path: str) -> Optional[Tuple[str, str]]:
    """Return (sha, decoded content) for existing README, or None if missing."""
    try:
        out = gh(["api", f"repos/{REPO}/contents/{repo_path}"])
    except RuntimeError:
        return None
    data = json.loads(out)
    return data["sha"], base64.b64decode(data["content"]).decode("utf-8")


def update_readme(repo_path: str, new_content: str, sha: Optional[str], state_key: str) -> bool:
    payload = {
        "message": f"Regenerate {state_key} README from current release assets",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("ascii"),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha

    tmp = Path("/tmp/readme_update.json")
    tmp.write_text(json.dumps(payload))
    try:
        gh(["api", f"repos/{REPO}/contents/{repo_path}", "--method", "PUT",
            "--input", str(tmp)])
        return True
    except RuntimeError as e:
        print(f"  ✗ Update failed: {e}")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Show planned changes, don't commit")
    ap.add_argument("--apply", action="store_true", help="Commit updates to main")
    ap.add_argument("--state", help="Process only this state (e.g. 'kansas')")
    args = ap.parse_args()
    if not args.dry_run and not args.apply:
        ap.error("specify --dry-run or --apply")

    print(f"Fetching releases from {REPO}...")
    releases = json.loads(gh(["api", f"repos/{REPO}/releases?per_page=100"]))

    # Best release per state: prefer published over draft, and newest modified
    by_state: Dict[str, Dict] = {}
    for r in releases:
        tag = r.get("tag_name", "")
        state = re.sub(r"-v\d+\.\d+\.\d+$", "", tag)
        if state not in US_STATES:
            continue
        if args.state and state != args.state:
            continue
        cur = by_state.get(state)
        if cur is None:
            by_state[state] = r
        elif cur.get("draft") and not r.get("draft"):
            by_state[state] = r

    print(f"Evaluating {len(by_state)} US state releases")
    updated = []
    unchanged = []
    failed = []

    for state in sorted(by_state.keys()):
        release = by_state[state]
        repo_path = f"north-america/united-states-of-america/{state}/README.md"
        new_content = build_readme(state, release)
        existing = fetch_current_readme(repo_path)
        existing_sha = existing[0] if existing else None
        existing_content = existing[1] if existing else None

        # Compare after normalizing the auto-generated date line (avoid churn
        # when only today's date changes and nothing else)
        def strip_date(s):
            return re.sub(r"\* Date: \d{4}-\d{2}-\d{2}", "* Date: <date>", s or "")

        if strip_date(new_content) == strip_date(existing_content):
            unchanged.append(state)
            continue

        assets = len(release.get("assets", []))
        print(f"\n  {state}: would update ({assets} assets)")
        if args.dry_run:
            # Diff hint: show which filenames are new
            old_files = set(re.findall(r"\| \[([^\]]+)\]\(", existing_content or ""))
            new_files = set(re.findall(r"\| \[([^\]]+)\]\(", new_content))
            added = new_files - old_files
            if added:
                print(f"    + {len(added)} file(s) added to README:")
                for f in sorted(added):
                    print(f"      {f}")
            removed = old_files - new_files
            if removed:
                print(f"    - {len(removed)} file(s) no longer in release:")
                for f in sorted(removed):
                    print(f"      {f}")
        else:
            ok = update_readme(repo_path, new_content, existing_sha, state)
            if ok:
                updated.append(state)
                print(f"    ✓ committed to main")
            else:
                failed.append(state)

    print()
    print(f"=== Summary ===")
    print(f"  Unchanged: {len(unchanged)}")
    print(f"  Would update / updated: {len([s for s in by_state if s not in unchanged])}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    missing = sorted(US_STATES - set(by_state.keys()))
    if missing:
        print(f"\nNo release found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
