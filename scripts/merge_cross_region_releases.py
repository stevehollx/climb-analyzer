#!/usr/bin/env python3
"""Cross-region climb merger — top-level orchestrator.

Downloads each region's latest v2.4.0 release assets, finds cross-border climbs
between adjacent regions, regenerates elevation profiles via osmium+opentopodata,
writes v2.4.1 xlsx+sqlite.gz for each affected region, and (in --apply mode)
uploads the new release tag, deletes the v2.4.0 tag+assets, regenerates the
repo READMEs, writes a markdown merge report, and opens a PR.

Default is --dry-run: no writes to releases, PRs, or main repo.

Usage:
    scripts/merge_cross_region_releases.py --region-set us-states
    scripts/merge_cross_region_releases.py --region-set us-states --states kansas,colorado
    scripts/merge_cross_region_releases.py --region-set us-states --apply
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from climb_analyzer.core.cross_region_merger import (  # noqa: E402
    CANDIDATE_DETECTION_COLUMNS,
    MergeResult,
    RegionInput,
    candidate_detection_columns,
    col_gain,
    col_length,
    col_profile,
    detect_units,
    find_adjacent_region_pairs,
    find_merge_candidates,
    load_region,
    load_region_bbox_only,
    load_region_df,
    merge_candidate,
)
from climb_analyzer.core.elevation_recompute import (  # noqa: E402
    extract_ways_batch,
    climbs_share_nodes,
)
from climb_analyzer.data.sqlite_export import SQLiteExporter  # noqa: E402

logger = logging.getLogger("cross_region_merge")

RELEASES_REPO = "stevehollx/global-road-and-trail-climbs"
CURRENT_VERSION = "2.4.0"
NEW_VERSION = "2.4.1"
XLSX_ROWS_PER_PART = 200_000
SPLIT_THRESHOLD = 1_950_000_000
CHUNK_SIZE = 1_900_000_000


# --- Region sets ---

US_STATES = [
    ("Alabama", "alabama", "alabama-latest.osm.pbf"),
    ("Alaska", "alaska", "alaska-latest.osm.pbf"),
    ("Arizona", "arizona", "arizona-latest.osm.pbf"),
    ("Arkansas", "arkansas", "arkansas-latest.osm.pbf"),
    ("California", "california", "california-latest.osm.pbf"),
    ("Colorado", "colorado", "colorado-latest.osm.pbf"),
    ("Connecticut", "connecticut", "connecticut-latest.osm.pbf"),
    ("Delaware", "delaware", "delaware-latest.osm.pbf"),
    ("Florida", "florida", "florida-latest.osm.pbf"),
    ("Georgia", "georgia", "us_georgia-latest.osm.pbf"),
    ("Hawaii", "hawaii", "hawaii-latest.osm.pbf"),
    ("Idaho", "idaho", "idaho-latest.osm.pbf"),
    ("Illinois", "illinois", "illinois-latest.osm.pbf"),
    ("Indiana", "indiana", "indiana-latest.osm.pbf"),
    ("Iowa", "iowa", "iowa-latest.osm.pbf"),
    ("Kansas", "kansas", "kansas-latest.osm.pbf"),
    ("Kentucky", "kentucky", "kentucky-latest.osm.pbf"),
    ("Louisiana", "louisiana", "louisiana-latest.osm.pbf"),
    ("Maine", "maine", "maine-latest.osm.pbf"),
    ("Maryland", "maryland", "maryland-latest.osm.pbf"),
    ("Massachusetts", "massachusetts", "massachusetts-latest.osm.pbf"),
    ("Michigan", "michigan", "michigan-latest.osm.pbf"),
    ("Minnesota", "minnesota", "minnesota-latest.osm.pbf"),
    ("Mississippi", "mississippi", "mississippi-latest.osm.pbf"),
    ("Missouri", "missouri", "missouri-latest.osm.pbf"),
    ("Montana", "montana", "montana-latest.osm.pbf"),
    ("Nebraska", "nebraska", "nebraska-latest.osm.pbf"),
    ("Nevada", "nevada", "nevada-latest.osm.pbf"),
    ("New Hampshire", "new-hampshire", "new-hampshire-latest.osm.pbf"),
    ("New Jersey", "new-jersey", "new-jersey-latest.osm.pbf"),
    ("New Mexico", "new-mexico", "new-mexico-latest.osm.pbf"),
    ("New York", "new-york", "new-york-latest.osm.pbf"),
    ("North Carolina", "north-carolina", "north-carolina-latest.osm.pbf"),
    ("North Dakota", "north-dakota", "north-dakota-latest.osm.pbf"),
    ("Ohio", "ohio", "ohio-latest.osm.pbf"),
    ("Oklahoma", "oklahoma", "oklahoma-latest.osm.pbf"),
    ("Oregon", "oregon", "oregon-latest.osm.pbf"),
    ("Pennsylvania", "pennsylvania", "pennsylvania-latest.osm.pbf"),
    ("Rhode Island", "rhode-island", "rhode-island-latest.osm.pbf"),
    ("South Carolina", "south-carolina", "south-carolina-latest.osm.pbf"),
    ("South Dakota", "south-dakota", "south-dakota-latest.osm.pbf"),
    ("Tennessee", "tennessee", "tennessee-latest.osm.pbf"),
    ("Texas", "texas", "texas-latest.osm.pbf"),
    ("Utah", "utah", "utah-latest.osm.pbf"),
    ("Vermont", "vermont", "vermont-latest.osm.pbf"),
    ("Virginia", "virginia", "virginia-latest.osm.pbf"),
    ("Washington", "washington", "washington-latest.osm.pbf"),
    ("West Virginia", "west-virginia", "west-virginia-latest.osm.pbf"),
    ("Wisconsin", "wisconsin", "wisconsin-latest.osm.pbf"),
    ("Wyoming", "wyoming", "wyoming-latest.osm.pbf"),
]

# Canadian provinces and territories. Slug includes "canada-" prefix to match
# the release-tag format on stevehollx/global-road-and-trail-climbs
# (e.g. canada-british-columbia-v2.4.0).
CA_PROVINCES = [
    ("Alberta", "canada-alberta", "alberta-latest.osm.pbf"),
    ("British Columbia", "canada-british-columbia", "british-columbia-latest.osm.pbf"),
    ("Manitoba", "canada-manitoba", "manitoba-latest.osm.pbf"),
    ("New Brunswick", "canada-new-brunswick", "new-brunswick-latest.osm.pbf"),
    ("Newfoundland and Labrador", "canada-newfoundland-and-labrador", "newfoundland-and-labrador-latest.osm.pbf"),
    ("Northwest Territories", "canada-northwest-territories", "northwest-territories-latest.osm.pbf"),
    ("Nova Scotia", "canada-nova-scotia", "nova-scotia-latest.osm.pbf"),
    ("Nunavut", "canada-nunavut", "nunavut-latest.osm.pbf"),
    ("Ontario", "canada-ontario", "ontario-latest.osm.pbf"),
    ("Prince Edward Island", "canada-prince-edward-island", "prince-edward-island-latest.osm.pbf"),
    ("Quebec", "canada-quebec", "quebec-latest.osm.pbf"),
    ("Saskatchewan", "canada-saskatchewan", "saskatchewan-latest.osm.pbf"),
    ("Yukon", "canada-yukon", "yukon-latest.osm.pbf"),
]


@dataclass
class LoadedRegion:
    display_name: str
    slug: str
    pbf_path: Path
    release_tag: str
    release_date: str
    xlsx_parts: List[Path]
    sqlite_gz: Optional[Path]
    region_input: RegionInput


# --- gh helpers ---

def gh_json(args: List[str]) -> dict:
    p = subprocess.run(["gh"] + args, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{p.stderr}")
    return json.loads(p.stdout)


def gh_run(args: List[str], capture: bool = True) -> str:
    p = subprocess.run(["gh"] + args, capture_output=capture, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{p.stderr}")
    return p.stdout


def latest_state_release(slug: str, version: str) -> Optional[dict]:
    try:
        tag = f"{slug}-v{version}"
        return gh_json(["release", "view", tag, "-R", RELEASES_REPO, "--json",
                        "tagName,name,publishedAt,assets"])
    except RuntimeError:
        return None


def download_assets(slug: str, version: str, dest: Path) -> Tuple[List[Path], Optional[Path]]:
    """Download xlsx parts + errors.txt only. sqlite.gz is rebuilt from merged df."""
    tag = f"{slug}-v{version}"
    dest.mkdir(parents=True, exist_ok=True)
    patterns = ["*_climbs_*.xlsx", "*_errors_*.txt"]
    args = ["release", "download", tag, "-R", RELEASES_REPO, "--dir", str(dest)]
    for pat in patterns:
        args.extend(["--pattern", pat])
    subprocess.run(["gh"] + args, capture_output=True, text=True, check=False)
    xlsx_parts = sorted(dest.glob("*_climbs_*.xlsx"))
    return xlsx_parts, None


def extract_date_from_filename(xlsx: Path) -> str:
    parts = xlsx.stem.split("_")
    for p in parts:
        if len(p) == 10 and p[4] == "-" and p[7] == "-":
            return p
    return date.today().isoformat()


def extract_errors_from_filename(xlsx: Path) -> int:
    for p in xlsx.stem.split("_"):
        if p.startswith("e") and len(p) >= 5 and p[1:5].isdigit():
            return int(p[1:5])
    return 0


def _way_ids_by_climb(df: pd.DataFrame) -> Dict[int, List[int]]:
    """Map df index -> list of way_ids parsed from the 'All Way IDs' column."""
    result: Dict[int, List[int]] = {}
    for idx, val in df["All Way IDs"].items():
        if val is None:
            result[idx] = []
            continue
        s = str(val)
        ids: List[int] = []
        for part in s.replace(" ", ",").split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
        result[idx] = ids
    return result


# --- output writing ---

def write_split_xlsx(df: pd.DataFrame, base_no_ext: Path, errors: int) -> List[Path]:
    """Write df as one or more xlsx files with the *_e####-N.xlsx naming.

    Sort by Basic Score desc (matches engine output) so hardest go into part 1.
    Prefers xlsxwriter engine (3-5x faster than openpyxl for writing).
    """
    if "Basic Score" in df.columns:
        df = df.sort_values("Basic Score", ascending=False, kind="stable").reset_index(drop=True)

    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"

    n_rows = len(df)
    n_parts = max(1, (n_rows + XLSX_ROWS_PER_PART - 1) // XLSX_ROWS_PER_PART)
    paths: List[Path] = []
    if n_parts == 1:
        out = Path(f"{base_no_ext}_e{errors:04d}.xlsx")
        df.to_excel(out, index=False, engine=engine)
        paths.append(out)
    else:
        for i in range(n_parts):
            lo = i * XLSX_ROWS_PER_PART
            hi = min(lo + XLSX_ROWS_PER_PART, n_rows)
            part = df.iloc[lo:hi]
            out = Path(f"{base_no_ext}_e{errors:04d}-{i + 1}.xlsx")
            part.to_excel(out, index=False, engine=engine)
            paths.append(out)
    return paths


def build_sqlite_gz(df: pd.DataFrame, base_no_ext: Path, errors: int,
                    file_id: Optional[str] = None, units: str = "ft") -> Tuple[Path, Optional[Path]]:
    """Write df to a single sqlite via SQLiteExporter, gzip, split if needed."""
    sqlite_path = Path(f"{base_no_ext}_e{errors:04d}.sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()

    fid = file_id or (sqlite_path.stem + ".db")
    with SQLiteExporter(sqlite_path, file_id=fid, units=units) as exporter:
        rows = df.to_dict(orient="records")
        exporter.write_rows(rows)

    gz_path = Path(f"{sqlite_path}.gz")
    with open(sqlite_path, "rb") as r, gzip.open(gz_path, "wb", compresslevel=6) as w:
        shutil.copyfileobj(r, w, length=8 * 1024 * 1024)
    sqlite_path.unlink()

    if gz_path.stat().st_size < SPLIT_THRESHOLD:
        return gz_path, None

    # split large gz
    sums: Dict[str, str] = {}
    chunks: List[Path] = []
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
            idx += 1
    sha_path = gz_path.parent / f"{gz_path.name}.sha256"
    with open(sha_path, "w") as sf:
        for name, digest in sorted(sums.items()):
            sf.write(f"{digest}  {name}\n")
    gz_path.unlink()
    # Return the sha file alongside chunks
    return chunks[0] if chunks else gz_path, sha_path


# --- release ops ---

def delete_release_and_tag(tag: str) -> None:
    try:
        gh_run(["release", "delete", tag, "-R", RELEASES_REPO], capture=False)
    except RuntimeError as e:
        logger.warning(f"release delete {tag} failed: {e}")
    try:
        subprocess.run(
            ["gh", "api", "-X", "DELETE", f"/repos/{RELEASES_REPO}/git/refs/tags/{tag}"],
            capture_output=True, text=True, check=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"tag delete {tag} failed: {e}")


def create_release(tag: str, title: str, notes: str) -> None:
    gh_run([
        "release", "create", tag,
        "-R", RELEASES_REPO,
        "--title", title,
        "--notes", notes,
    ], capture=False)


def upload_assets(tag: str, files: List[Path]) -> None:
    for f in files:
        gh_run(["release", "upload", tag, str(f), "-R", RELEASES_REPO], capture=False)


# --- markdown report ---

def write_markdown_report(merges: List[MergeResult], out_path: Path, region_set: str) -> None:
    by_pair: Dict[Tuple[str, str], List[MergeResult]] = {}
    for m in merges:
        key = tuple(sorted([m.candidate.region_a, m.candidate.region_b]))
        by_pair.setdefault(key, []).append(m)

    lines: List[str] = []
    lines.append(f"# Cross-Region Climb Merge Report — {region_set}")
    lines.append("")
    lines.append(f"Date: {date.today().isoformat()}")
    lines.append(f"Version bump: v{CURRENT_VERSION} -> v{NEW_VERSION}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    affected = set()
    for m in merges:
        affected.add(m.candidate.region_a)
        affected.add(m.candidate.region_b)
    lines.append(f"- Cross-border climbs found: **{len(merges)}**")
    lines.append(f"- Adjacent pairs with merges: **{len(by_pair)}**")
    lines.append(f"- Regions affected: **{len(affected)}**")
    lines.append("")
    lines.append("## By Region Pair")
    lines.append("")
    for (a, b), items in sorted(by_pair.items()):
        lines.append(f"### {a} ↔ {b}")
        lines.append("")
        # Detect units from the first merged row's columns (imperial/metric)
        sample_row = items[0].merged_row_a if items else {}
        if "Length (mi)" in sample_row:
            length_label, gain_label = "Length (mi)", "Elev Gain (ft)"
            len_key, gain_key = "Length (mi)", "Elev Gain (ft)"
        else:
            length_label, gain_label = "Length (km)", "Elev Gain (m)"
            len_key, gain_key = "Length (km)", "Elev Gain (m)"
        lines.append(f"| Street Name | {length_label} | {gain_label} | Avg Grade | Max Grade | Profile | Way IDs |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for m in items:
            row = m.merged_row_a
            way_ids = row.get("All Way IDs", "")
            way_list = str(way_ids).split(",")
            way_short = ", ".join(w.strip() for w in way_list[:4])
            if len(way_list) > 4:
                way_short += f" (+{len(way_list) - 4} more)"
            lines.append(
                f"| {m.candidate.street_name} "
                f"| {row.get(len_key, 0):.2f} "
                f"| {row.get(gain_key, 0):.0f} "
                f"| {row.get('Avg Grade (%)', 0):.2f}% "
                f"| {row.get('Max Grade (%)', 0):.2f}% "
                f"| {m.profile_source} "
                f"| {way_short} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines))


# --- main orchestration ---

def load_all_regions(region_defs: List[Tuple[str, str, str]], work_dir: Path,
                     pbf_dir: Path, only: Optional[List[str]] = None,
                     lean: bool = True) -> List[LoadedRegion]:
    """Download each region's v2.4.0 assets and return bbox-only skeletons.

    First run: does a full lean xlsx parse per region to build the pickle cache,
    records the bbox, then drops the df. Subsequent runs reuse the cache.
    Peak memory stays at ~1 state's worth because dfs are dropped as we go.
    """
    loaded: List[LoadedRegion] = []
    cache_dir = work_dir / "_parquet_cache"
    for display_name, slug, pbf_name in region_defs:
        if only and slug not in only:
            continue
        rel = latest_state_release(slug, CURRENT_VERSION)
        if rel is None:
            logger.info(f"  [skip] {slug}: no v{CURRENT_VERSION} release")
            continue
        dest = work_dir / slug
        xlsx_parts, sqlite_gz = download_assets(slug, CURRENT_VERSION, dest)
        if not xlsx_parts:
            logger.warning(f"  [skip] {slug}: no xlsx parts downloaded")
            continue
        release_date = extract_date_from_filename(xlsx_parts[0])
        pbf = pbf_dir / pbf_name
        if not pbf.exists():
            logger.warning(f"  [skip] {slug}: pbf not found at {pbf}")
            continue
        # Detect units from xlsx header so we read the right lean column set
        sample_cols = list(pd.read_excel(xlsx_parts[0], nrows=0).columns)
        units = detect_units(sample_cols)
        columns = candidate_detection_columns(units) if lean else None
        ri = load_region_bbox_only(display_name, xlsx_parts, pbf, cache_dir,
                                   columns=columns)
        loaded.append(LoadedRegion(
            display_name=display_name, slug=slug, pbf_path=pbf,
            release_tag=rel["tagName"], release_date=release_date,
            xlsx_parts=xlsx_parts, sqlite_gz=sqlite_gz,
            region_input=ri,
        ))
        logger.info(f"  [loaded bbox] {slug}: bbox={ri.bbox} from {len(xlsx_parts)} xlsx part(s)")
    return loaded


def reload_region_full(region: LoadedRegion, cache_dir: Optional[Path] = None) -> LoadedRegion:
    """Re-load a region with all columns for output writing. Caches in pickle."""
    full = load_region(region.display_name, region.xlsx_parts, region.pbf_path,
                       columns=None, cache_dir=cache_dir)
    return LoadedRegion(
        display_name=region.display_name, slug=region.slug, pbf_path=region.pbf_path,
        release_tag=region.release_tag, release_date=region.release_date,
        xlsx_parts=region.xlsx_parts, sqlite_gz=region.sqlite_gz,
        region_input=full,
    )


def free_region(region: LoadedRegion) -> None:
    """Drop the in-memory df to free RAM."""
    if region.region_input is not None:
        region.region_input.df = None
    import gc
    gc.collect()


def run_merges(regions: List[LoadedRegion], work_dir: Path) -> Tuple[List[MergeResult], Dict[str, List[MergeResult]]]:
    cache_dir = work_dir / "_parquet_cache"
    pairs = find_adjacent_region_pairs([r.region_input for r in regions], buffer_km=10.0)
    logger.info(f"Adjacent pairs: {len(pairs)}")

    name_to_loaded = {r.region_input.name: r for r in regions}
    all_merges: List[MergeResult] = []
    by_region: Dict[str, List[MergeResult]] = {r.region_input.name: [] for r in regions}
    consumed: Dict[str, set] = {r.region_input.name: set() for r in regions}

    for pair_i, (a_skel, b_skel, _dist) in enumerate(pairs, 1):
        # Lazy-load just this pair's dfs from pickle cache
        lr_a = name_to_loaded[a_skel.name]
        lr_b = name_to_loaded[b_skel.name]
        cols_a = candidate_detection_columns(a_skel.units)
        cols_b = candidate_detection_columns(b_skel.units)
        df_a = load_region_df(a_skel, lr_a.xlsx_parts, cache_dir, columns=cols_a)
        df_b = load_region_df(b_skel, lr_b.xlsx_parts, cache_dir, columns=cols_b)
        a = RegionInput(name=a_skel.name, df=df_a, bbox=a_skel.bbox, pbf_path=a_skel.pbf_path, units=a_skel.units)
        b = RegionInput(name=b_skel.name, df=df_b, bbox=b_skel.bbox, pbf_path=b_skel.pbf_path, units=b_skel.units)
        cands = find_merge_candidates(a, b)
        if not cands:
            continue
        # Keep only candidates whose climbs haven't been consumed by earlier pairs
        cands = [c for c in cands if c.idx_a not in consumed[a.name]
                 and c.idx_b not in consumed[b.name]]
        if not cands:
            logger.info(f"  [{pair_i}/{len(pairs)}] {a.name} <-> {b.name}: "
                        f"all candidates already consumed")
            continue

        # Batch-extract all way geometries for this pair — one osmium call per PBF
        all_way_ids = {w for c in cands for w in c.way_ids_union}
        logger.info(f"  [{pair_i}/{len(pairs)}] {a.name} <-> {b.name}: "
                    f"{len(cands)} candidate(s) after name/proximity, "
                    f"extracting {len(all_way_ids)} ways...")
        try:
            geom = extract_ways_batch(sorted(all_way_ids), [a.pbf_path, b.pbf_path])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"    batch extract failed: {exc}; falling back to splice")
            geom = ({}, {})

        _nodes_map, ways_map = geom
        # Shared-node filter: a candidate is only a real cross-border climb if
        # its two sides share at least one OSM node (typically a border node).
        # Unrelated same-name roads in adjacent states share zero nodes.
        connected_cands = []
        a_way_lookup = _way_ids_by_climb(a.df)
        b_way_lookup = _way_ids_by_climb(b.df)
        for c in cands:
            if climbs_share_nodes(
                a_way_lookup.get(c.idx_a, []),
                b_way_lookup.get(c.idx_b, []),
                ways_map,
            ):
                connected_cands.append(c)
        logger.info(f"    shared-node check: {len(connected_cands)}/{len(cands)} pass")

        merged_count = 0
        for c in connected_cands:
            result = merge_candidate(c, a, b, do_recompute=True, precomputed_geom=geom)
            all_merges.append(result)
            by_region[a.name].append(result)
            by_region[b.name].append(result)
            consumed[a.name].add(c.idx_a)
            consumed[b.name].add(c.idx_b)
            merged_count += 1
        logger.info(f"    merged {merged_count}")

        # release memory before next pair
        del df_a, df_b, a, b, a_way_lookup, b_way_lookup, cands, connected_cands
        import gc
        gc.collect()
    return all_merges, by_region


def build_outputs_streaming(region: LoadedRegion, merges: List[MergeResult],
                            work_dir: Path,
                            batch_size: int = XLSX_ROWS_PER_PART) -> Tuple[List[Path], Path, Optional[Path]]:
    """Memory-efficient build via temp SQLite as sort buffer.

    Reads source xlsx parts one at a time into a temp sqlite, drops merged-out
    rows on the fly, appends merged rows, then streams ORDER BY "Basic Score" DESC
    out in `batch_size` chunks — each chunk becomes one output xlsx part and
    feeds into the iOS SQLiteExporter incrementally. Memory is bounded to ~one
    xlsx part at a time (≤200k rows).
    """
    import sqlite3
    import gc

    region_name = region.region_input.name
    drop_idxs: set = set()
    add_rows: List[Dict] = []
    for m in merges:
        if m.candidate.region_a == region_name:
            drop_idxs.add(m.candidate.idx_a)
            add_rows.append(m.merged_row_a)
        elif m.candidate.region_b == region_name:
            drop_idxs.add(m.candidate.idx_b)
            add_rows.append(m.merged_row_b)

    out_dir = work_dir / "out" / region.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    units_label = "imperial" if (region.region_input is not None and region.region_input.units == "imperial") else "metric"
    base = out_dir / f"{region.display_name.replace(' ', '_')}_climbs_all-surfaces_all-access_{units_label}_{region.release_date}_v{NEW_VERSION}"
    errors = extract_errors_from_filename(region.xlsx_parts[0])

    # Phase 1: stream source xlsx parts → temp sort-buffer sqlite
    work_db_path = out_dir / "_sort_buffer.sqlite"
    if work_db_path.exists():
        work_db_path.unlink()

    conn = sqlite3.connect(str(work_db_path))
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")  # 200 MB

    columns: Optional[List[str]] = None
    insert_sql: Optional[str] = None
    global_idx = 0
    inserted_src = 0

    for part_path in region.xlsx_parts:
        chunk = pd.read_excel(part_path)
        if columns is None:
            columns = list(chunk.columns)
            quoted = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(f'CREATE TABLE rows ({quoted})')
            insert_sql = f"INSERT INTO rows VALUES ({placeholders})"

        chunk_records = chunk.values.tolist()
        rows_to_insert = []
        for rec in chunk_records:
            if global_idx not in drop_idxs:
                rows_to_insert.append(tuple(rec))
            global_idx += 1

        if rows_to_insert:
            conn.executemany(insert_sql, rows_to_insert)
            conn.commit()
            inserted_src += len(rows_to_insert)
        del chunk, chunk_records, rows_to_insert
        gc.collect()

    inserted_merged = 0
    if add_rows and columns:
        merge_tuples = [tuple(row.get(c) for c in columns) for row in add_rows]
        conn.executemany(insert_sql, merge_tuples)
        conn.commit()
        inserted_merged = len(merge_tuples)
        del merge_tuples

    total_rows = inserted_src + inserted_merged
    logger.info(f"    sort-buffer: {inserted_src} source + {inserted_merged} merged = {total_rows} rows")

    if "Basic Score" in columns:
        conn.execute('CREATE INDEX idx_score ON rows("Basic Score" DESC)')

    # Phase 2: stream sorted rows → split xlsx parts + iOS sqlite
    final_sqlite_path = Path(f"{base}_e{errors:04d}.sqlite")
    if final_sqlite_path.exists():
        final_sqlite_path.unlink()
    fid = final_sqlite_path.stem + ".db"

    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"

    select_sql = 'SELECT * FROM rows ORDER BY "Basic Score" DESC' if "Basic Score" in columns else "SELECT * FROM rows"

    n_parts = max(1, (total_rows + batch_size - 1) // batch_size)
    paths: List[Path] = []

    units = region.region_input.units if region.region_input is not None else "imperial"
    sqlite_units = "ft" if units == "imperial" else "m"
    with SQLiteExporter(final_sqlite_path, file_id=fid, units=sqlite_units) as exporter:
        cursor = conn.execute(select_sql)
        part_idx = 1
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            part_df = pd.DataFrame(batch, columns=columns)
            if n_parts == 1:
                out_xlsx = Path(f"{base}_e{errors:04d}.xlsx")
            else:
                out_xlsx = Path(f"{base}_e{errors:04d}-{part_idx}.xlsx")
            part_df.to_excel(out_xlsx, index=False, engine=engine)
            paths.append(out_xlsx)

            # Feed iOS exporter in 5k-row sub-batches to keep memory steady
            records = part_df.to_dict(orient="records")
            sub = 5000
            for i in range(0, len(records), sub):
                exporter.write_rows(records[i:i + sub])

            del part_df, batch, records
            gc.collect()
            part_idx += 1

    conn.close()
    work_db_path.unlink(missing_ok=True)

    # Gzip + optional split — same as build_sqlite_gz
    gz_path = Path(f"{final_sqlite_path}.gz")
    with open(final_sqlite_path, "rb") as r, gzip.open(gz_path, "wb", compresslevel=6) as w:
        shutil.copyfileobj(r, w, length=8 * 1024 * 1024)
    final_sqlite_path.unlink()

    if gz_path.stat().st_size < SPLIT_THRESHOLD:
        return paths, gz_path, None

    sums: Dict[str, str] = {}
    chunks: List[Path] = []
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
            idx += 1
    sha_path = gz_path.parent / f"{gz_path.name}.sha256"
    with open(sha_path, "w") as sf:
        for name, digest in sorted(sums.items()):
            sf.write(f"{digest}  {name}\n")
    gz_path.unlink()
    return paths, chunks[0] if chunks else gz_path, sha_path


def build_outputs_for_region(region: LoadedRegion, merges: List[MergeResult],
                             work_dir: Path) -> Tuple[List[Path], Path, Optional[Path]]:
    """Streaming build — bounded memory, works for any state size."""
    return build_outputs_streaming(region, merges, work_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-set", choices=["us-states", "ca-provinces"], default="us-states")
    parser.add_argument("--states", help="Comma-separated slugs to limit to (e.g. kansas,colorado)")
    parser.add_argument("--skip-states",
                        help="Comma-separated slugs to skip during build phase "
                             "(use for huge states that OOM during in-memory build)")
    parser.add_argument("--work-dir", default="/tmp/cross_region_merge")
    parser.add_argument("--pbf-dir", default="data/planet_osm_data")
    parser.add_argument("--apply", action="store_true",
                        help="Actually upload releases, delete v2.4.0, regen READMEs, open PR")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pbf_dir = Path(args.pbf_dir)

    if args.region_set == "us-states":
        region_defs = US_STATES
    elif args.region_set == "ca-provinces":
        region_defs = CA_PROVINCES
    else:
        raise ValueError(f"unknown region-set {args.region_set}")
    only = args.states.split(",") if args.states else None

    logger.info(f"=== Cross-region merge ({args.region_set}) — "
                f"{'APPLY' if args.apply else 'DRY-RUN'} ===")
    regions = load_all_regions(region_defs, work_dir, pbf_dir, only=only)
    logger.info(f"Loaded {len(regions)} region(s)")

    merges_pkl = work_dir / "_merges.pkl"
    if merges_pkl.exists():
        import pickle
        logger.info(f"Resuming from cached merges: {merges_pkl}")
        with open(merges_pkl, "rb") as f:
            all_merges, by_region = pickle.load(f)
    else:
        all_merges, by_region = run_merges(regions, work_dir)
        import pickle
        with open(merges_pkl, "wb") as f:
            pickle.dump((all_merges, by_region), f)
        logger.info(f"Persisted merges to {merges_pkl}")
    logger.info(f"Total cross-border climbs found: {len(all_merges)}")

    report_path = work_dir / f"cross_region_merges_{args.region_set}_{date.today().isoformat()}.md"
    write_markdown_report(all_merges, report_path, args.region_set)
    logger.info(f"Report written: {report_path}")

    affected_regions = [r for r in regions if by_region[r.region_input.name]]
    logger.info(f"Affected regions: {len(affected_regions)}")
    for r in affected_regions:
        n = len(by_region[r.region_input.name])
        logger.info(f"  {r.slug}: {n} merge(s) -> rebuild xlsx + sqlite.gz -> v{NEW_VERSION}")

    if not affected_regions:
        logger.info("No cross-border climbs found — nothing to publish.")
        return 0

    logger.info("Regenerating outputs for affected regions (reloading full columns)...")
    # Process in ascending size order so a memory failure on the largest state
    # doesn't waste work on smaller ones. Size proxy = total bytes of source xlsx parts.
    affected_regions = sorted(affected_regions,
                              key=lambda r: sum(p.stat().st_size for p in r.xlsx_parts))
    skip_set = set((args.skip_states or "").split(",")) if args.skip_states else set()
    if skip_set:
        before = len(affected_regions)
        affected_regions = [r for r in affected_regions if r.slug not in skip_set]
        deferred = sorted(skip_set & {r.slug for r in regions})
        logger.warning(f"Skipping build for {before - len(affected_regions)} state(s) "
                       f"per --skip-states: {deferred}")
    built: Dict[str, Tuple[List[Path], Path, Optional[Path]]] = {}
    for r in affected_regions:
        out_dir = work_dir / "out" / r.slug
        existing_xlsx = sorted(out_dir.glob(f"*_v{NEW_VERSION}_*.xlsx")) if out_dir.exists() else []
        existing_gz = sorted(out_dir.glob(f"*_v{NEW_VERSION}_*.sqlite.gz*")) if out_dir.exists() else []
        if existing_xlsx and existing_gz:
            sha_files = sorted(out_dir.glob(f"*_v{NEW_VERSION}_*.sha256"))
            built[r.slug] = (existing_xlsx, existing_gz[0], sha_files[0] if sha_files else None)
            logger.info(f"  [skip] {r.slug}: already built ({len(existing_xlsx)} xlsx, {len(existing_gz)} gz)")
            continue
        merges_for_r = by_region[r.region_input.name]
        # Streaming build reads source xlsx parts directly — no full-df reload needed
        try:
            xlsx_parts, gz_or_first_chunk, sha_path = build_outputs_for_region(r, merges_for_r, work_dir)
        except MemoryError as e:
            logger.error(f"  [build-failed] {r.slug}: out of memory ({e}); skipping, restart to retry")
            continue
        built[r.slug] = (xlsx_parts, gz_or_first_chunk, sha_path)
        logger.info(f"  [built] {r.slug}: {len(xlsx_parts)} xlsx part(s), sqlite.gz at {gz_or_first_chunk.name}")
        import gc
        gc.collect()

    if not args.apply:
        logger.info(f"DRY-RUN complete. Outputs in {work_dir}/out/. Run with --apply to publish.")
        return 0

    logger.info("APPLY mode — publishing releases")
    for r in affected_regions:
        xlsx_parts, gz_first, sha = built[r.slug]
        new_tag = f"{r.slug}-v{NEW_VERSION}"
        old_tag = f"{r.slug}-v{CURRENT_VERSION}"
        notes = (
            f"Cross-region climb merges applied. See "
            f"`cross_region_merges_{args.region_set}_{date.today().isoformat()}.md` for details."
        )
        try:
            create_release(new_tag, f"{r.display_name} v{NEW_VERSION}", notes)
        except RuntimeError as e:
            logger.error(f"create_release {new_tag}: {e}")
            continue

        upload_files = list(xlsx_parts)
        # Find all gz chunk files in the same dir for upload
        gz_parent = gz_first.parent
        gz_chunks = sorted(gz_parent.glob(f"*_v{NEW_VERSION}_*.sqlite.gz*"))
        upload_files.extend(gz_chunks)
        if sha and sha.exists():
            upload_files.append(sha)
        # Also upload errors file if present in downloaded v2.4.0 set
        errors_txt = list(Path(work_dir / r.slug).glob("*_errors_*.txt"))
        if errors_txt:
            upload_files.extend(errors_txt)
        try:
            upload_assets(new_tag, upload_files)
        except RuntimeError as e:
            logger.error(f"upload {new_tag}: {e}")
            continue

        # Delete old release and tag
        delete_release_and_tag(old_tag)
        logger.info(f"  [published] {new_tag}, deleted {old_tag}")

    # Regenerate READMEs (commits directly to main on releases repo)
    logger.info("Regenerating READMEs on releases repo...")
    subprocess.run([
        "python3", str(_REPO_ROOT / "scripts" / "regenerate_us_state_readmes.py"), "--apply"
    ], check=False)

    # Open a PR adding the markdown report under reports/
    try:
        pr_url = open_report_pr(report_path, args.region_set)
        logger.info(f"PR opened: {pr_url}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"PR creation failed ({exc}); report still at {report_path}")

    logger.info(f"Report: {report_path}")
    return 0


def open_report_pr(report_path: Path, region_set: str) -> str:
    """Clone releases repo to tmp, add report under reports/, push branch, open PR."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        subprocess.run(
            ["gh", "repo", "clone", RELEASES_REPO, str(tdp / "repo"), "--", "--depth", "5"],
            check=True, capture_output=True, text=True,
        )
        repo_dir = tdp / "repo"
        branch = f"cross-region-merge-{region_set}-{date.today().isoformat()}"
        subprocess.run(["git", "-C", str(repo_dir), "checkout", "-b", branch], check=True)

        dest = repo_dir / "reports" / report_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(report_path, dest)
        subprocess.run(["git", "-C", str(repo_dir), "add", f"reports/{report_path.name}"], check=True)
        msg = f"Add cross-region merge report ({region_set}, {date.today().isoformat()})"
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", msg], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "push", "-u", "origin", branch],
            check=True, capture_output=True, text=True,
        )
        body = (
            f"Cross-region climb merges applied to {region_set}. Release assets updated "
            f"to v{NEW_VERSION} for affected states. See attached report for per-pair details."
        )
        pr_out = subprocess.run(
            ["gh", "pr", "create", "-R", RELEASES_REPO,
             "--title", f"Cross-region merge report ({region_set}, {date.today().isoformat()})",
             "--body", body, "--head", branch, "--base", "main"],
            check=True, capture_output=True, text=True,
        )
        return pr_out.stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
