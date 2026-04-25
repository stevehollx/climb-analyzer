"""Cross-region climb merger — core logic.

Produces a schema-clean merged row (matches the 26-column XLSX layout exactly)
that can be written back into BOTH region dataframes so the same physical climb
appears in each. Elevation/length/profile are recomputed from reconstructed
geometry via climb_analyzer.core.elevation_recompute.

Adjacency and candidate detection reuse the existing
scripts/merge_cross_region_climbs.py functions to avoid duplication.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False

# Reuse existing primitives; scripts/ isn't a package so bolt it on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from merge_cross_region_climbs import (  # noqa: E402
    calculate_region_bbox,
    bbox_distance_km,
    regions_are_adjacent,
    haversine_distance as haversine_km,
    extract_way_ids,
)

from climb_analyzer.core.elevation_recompute import (  # noqa: E402
    RecomputedClimb,
    extract_ways_batch,
    recompute_climb_geometry,
)

logger = logging.getLogger(__name__)

FT_PER_M = 3.28084
MI_PER_KM = 0.621371


# Engine output column names differ by units. Imperial xlsx (US) uses (mi/ft),
# metric xlsx (CA, EU) uses (km/m). The Latitude/Longitude/Score columns are
# unit-agnostic. Helpers below let the merger work with either.

def col_length(units: str) -> str:
    return "Length (mi)" if units == "imperial" else "Length (km)"

def col_gain(units: str) -> str:
    return "Elev Gain (ft)" if units == "imperial" else "Elev Gain (m)"

def col_height(units: str) -> str:
    return "Height (ft)" if units == "imperial" else "Height (m)"

def col_prominence(units: str) -> str:
    return "Prominence (ft)" if units == "imperial" else "Prominence (m)"

def col_profile(units: str) -> str:
    return "Elevation Profile (ft)" if units == "imperial" else "Elevation Profile (m)"

def col_from_center(units: str) -> str:
    return "From Center (mi)" if units == "imperial" else "From Center (km)"


def detect_units(columns: Sequence[str]) -> str:
    """Detect imperial/metric from a column list. Returns 'imperial' or 'metric'."""
    cols = set(columns)
    if "Length (mi)" in cols:
        return "imperial"
    if "Length (km)" in cols:
        return "metric"
    raise ValueError("Cannot detect units: neither 'Length (mi)' nor 'Length (km)' found")


def xlsx_columns(units: str) -> List[str]:
    return [
        "Street Name", "City", "State", "Country", col_from_center(units),
        "Latitude", "Longitude", "Cycling", "Category", "Basic Score",
        "FIETS Score", "PDI Score", col_gain(units), col_height(units),
        col_prominence(units), col_length(units), "Avg Grade (%)", "Max Grade (%)",
        "Highway Type", "Surface", "Tracktype", "Start Way ID", "OSM Link",
        "All Way IDs", "Connected Climbs", col_profile(units),
    ]


# Backwards-compat alias for imperial
XLSX_COLUMNS = xlsx_columns("imperial")

# Generic / placeholder street names — matching these across regions produces
# meaningless matches (any two unnamed roads near a border would pair up).
GENERIC_NAMES = {"Unnamed Road", "Track", "Road", "Path", "Trail", "Unknown",
                 "nan", "None", ""}


@dataclass
class RegionInput:
    name: str
    df: Optional[pd.DataFrame]  # may be None if bbox-only (for memory-lean adjacency pass)
    bbox: Tuple[float, float, float, float]
    pbf_path: Path
    units: str = "imperial"  # "imperial" (US) or "metric" (CA, EU)


@dataclass
class MergeCandidate:
    region_a: str
    region_b: str
    idx_a: int
    idx_b: int
    street_name: str
    endpoint_distance_km: float
    length_a_mi: float
    length_b_mi: float
    way_ids_union: List[int] = field(default_factory=list)


@dataclass
class MergeResult:
    candidate: MergeCandidate
    merged_row_a: Dict
    merged_row_b: Dict
    recomputed: Optional[RecomputedClimb]
    profile_source: str


def find_adjacent_region_pairs(
    regions: Sequence[RegionInput], buffer_km: float = 10.0
) -> List[Tuple[RegionInput, RegionInput, float]]:
    pairs: List[Tuple[RegionInput, RegionInput, float]] = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = regions[i], regions[j]
            d = bbox_distance_km(a.bbox, b.bbox)
            if regions_are_adjacent(a.bbox, b.bbox, buffer_km=buffer_km):
                pairs.append((a, b, d))
    return pairs


def find_merge_candidates(
    region_a: RegionInput,
    region_b: RegionInput,
    distance_threshold_km: float = 0.5,
    length_tolerance_pct: float = 5.0,
) -> List[MergeCandidate]:
    """Pairs of climbs in a and b that look like one climb truncated at the border.

    Same street name, start-points within threshold, lengths differ by at least
    tolerance (identical lengths = duplicate, not a border truncation).

    Uses a kd-tree on region_b's coordinates to find spatial matches in O(n log m)
    instead of O(n*m) via brute-force groupby.
    """
    if not HAS_KDTREE:
        raise RuntimeError("scipy.spatial.cKDTree is required for cross-region merging")

    df_a = region_a.df
    df_b = region_b.df

    if df_a.empty or df_b.empty:
        return []

    a_names = df_a["Street Name"].astype(str)
    b_names = df_b["Street Name"].astype(str)

    # KDTree over region_b — lat/lon in degrees. Conservative convert km->deg
    # at 111 km/deg (overestimates longitude distance near equator; fine here).
    b_coords = np.column_stack([df_b["Latitude"].values, df_b["Longitude"].values])
    tree = cKDTree(b_coords)
    radius_deg = distance_threshold_km / 111.0

    # Map index -> row idx (since df indexes may not be 0..N-1 after earlier ops)
    a_index_arr = df_a.index.to_numpy()
    b_index_arr = df_b.index.to_numpy()

    used_a: set = set()
    used_b: set = set()
    candidates: List[MergeCandidate] = []

    # Group region B by name for fast name equality check
    b_name_to_positions: Dict[str, List[int]] = {}
    for pos, nm in enumerate(b_names.values):
        b_name_to_positions.setdefault(nm, []).append(pos)

    units = region_a.units
    length_col = col_length(units)

    a_lats = df_a["Latitude"].values
    a_lons = df_a["Longitude"].values
    a_lens = df_a[length_col].values

    for pos_a in range(len(df_a)):
        name_a = a_names.iloc[pos_a]
        if name_a in GENERIC_NAMES or not name_a.strip() or name_a == "nan":
            continue
        same_name_b = b_name_to_positions.get(name_a)
        if not same_name_b:
            continue
        idx_a = a_index_arr[pos_a]
        if idx_a in used_a:
            continue

        la = float(a_lens[pos_a])
        if la <= 0:
            continue

        # Query tree for B points near A's start
        near = tree.query_ball_point([a_lats[pos_a], a_lons[pos_a]], radius_deg)
        if not near:
            continue
        near_same_name = [p for p in near if p in set(same_name_b)]
        if not near_same_name:
            continue

        best: Optional[Tuple[float, int]] = None
        for pos_b in near_same_name:
            idx_b = b_index_arr[pos_b]
            if idx_b in used_b:
                continue
            d_km = haversine_km(
                float(a_lats[pos_a]), float(a_lons[pos_a]),
                float(df_b["Latitude"].iat[pos_b]), float(df_b["Longitude"].iat[pos_b]),
            )
            if d_km >= distance_threshold_km:
                continue
            lb = float(df_b[length_col].iat[pos_b])
            if lb <= 0:
                continue
            diff_pct = abs(la - lb) / max(la, lb) * 100.0
            if diff_pct < length_tolerance_pct:
                continue
            if best is None or d_km < best[0]:
                best = (d_km, pos_b)

        if best is None:
            continue
        d_km, pos_b = best
        idx_b = b_index_arr[pos_b]
        ra = df_a.iloc[pos_a]
        rb = df_b.iloc[pos_b]
        way_ids = sorted(set(
            extract_way_ids(ra.get("All Way IDs", ""))
            + extract_way_ids(rb.get("All Way IDs", ""))
        ))
        candidates.append(MergeCandidate(
            region_a=region_a.name,
            region_b=region_b.name,
            idx_a=int(idx_a),
            idx_b=int(idx_b),
            street_name=str(name_a),
            endpoint_distance_km=d_km,
            length_a_mi=float(la),
            length_b_mi=float(df_b[length_col].iat[pos_b]),
            way_ids_union=way_ids,
        ))
        used_a.add(idx_a)
        used_b.add(idx_b)

    return candidates


def _filter_near_border(df: pd.DataFrame, self_bbox: Tuple[float, float, float, float],
                        other_bbox: Tuple[float, float, float, float],
                        threshold_km: float) -> pd.DataFrame:
    """Filter df to rows whose start point is within threshold_km of the shared border.

    Strategy: the shared border is the axis where the two bboxes meet (or nearly
    so). For each axis (lat, lon), compute the gap between the two bboxes. The
    border line is on the axis with the smallest gap. Filter df to rows within
    threshold on that axis AND within the union of both bboxes' range on the
    other axis.
    """
    s_min_lat, s_min_lon, s_max_lat, s_max_lon = self_bbox
    o_min_lat, o_min_lon, o_max_lat, o_max_lon = other_bbox

    deg = threshold_km / 111.0

    # Gap on each axis (negative = overlap)
    lat_gap = max(s_min_lat - o_max_lat, o_min_lat - s_max_lat)
    lon_gap = max(s_min_lon - o_max_lon, o_min_lon - s_max_lon)

    lat = df["Latitude"]
    lon = df["Longitude"]

    # The border axis is the one with the LARGER gap (smaller overlap = where
    # data is most separated = where the state line runs). For Kansas-Missouri:
    # lon_gap ~= -1 (small overlap), lat_gap ~= -4 (big overlap) -> border is lon.
    if lon_gap > lat_gap:
        # Shared border runs N-S (different longitudes meet). Filter on lon.
        if s_max_lon <= o_min_lon:
            # self is west of other; border at ~s_max_lon
            border_lon = (s_max_lon + o_min_lon) / 2
        elif o_max_lon <= s_min_lon:
            border_lon = (s_min_lon + o_max_lon) / 2
        else:
            border_lon = (max(s_min_lon, o_min_lon) + min(s_max_lon, o_max_lon)) / 2
        lon_mask = (lon >= border_lon - deg) & (lon <= border_lon + deg)
        lat_lo = max(s_min_lat, o_min_lat) - deg
        lat_hi = min(s_max_lat, o_max_lat) + deg
        lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
        return df[lon_mask & lat_mask]
    else:
        # Shared border runs E-W. Filter on lat.
        if s_max_lat <= o_min_lat:
            border_lat = (s_max_lat + o_min_lat) / 2
        elif o_max_lat <= s_min_lat:
            border_lat = (s_min_lat + o_max_lat) / 2
        else:
            border_lat = (max(s_min_lat, o_min_lat) + min(s_max_lat, o_max_lat)) / 2
        lat_mask = (lat >= border_lat - deg) & (lat <= border_lat + deg)
        lon_lo = max(s_min_lon, o_min_lon) - deg
        lon_hi = min(s_max_lon, o_max_lon) + deg
        lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
        return df[lat_mask & lon_mask]


def _fallback_metrics(ra: pd.Series, rb: pd.Series, units: str = "imperial") -> Dict[str, float]:
    """Summed metrics when recompute fails. Profile is left for caller.

    For imperial: length_ft = length_mi * 5280.
    For metric: length_m = length_km * 1000.
    Avg grade math is unit-consistent: gain/length * 100 in matching units.
    """
    length_c = col_length(units)
    gain_c = col_gain(units)
    length_total = float(ra[length_c]) + float(rb[length_c])
    gain_total = float(ra[gain_c]) + float(rb[gain_c])
    max_grade = max(float(ra["Max Grade (%)"]), float(rb["Max Grade (%)"]))
    if units == "imperial":
        length_in_gain_units = length_total * 5280.0  # mi -> ft
    else:
        length_in_gain_units = length_total * 1000.0  # km -> m
    avg_grade = (gain_total / length_in_gain_units * 100.0) if length_in_gain_units > 0 else 0.0
    return {
        length_c: round(length_total, 3),
        gain_c: round(gain_total, 0),
        "Avg Grade (%)": round(avg_grade, 2),
        "Max Grade (%)": round(max_grade, 2),
    }


def _convert_profile_m_to_ft(profile_m: str) -> str:
    """Convert an elevation profile's elevation values from meters to feet.

    Profile format: 'dist,ele,grade|dist,ele,grade|...'. Only the elevation
    column changes units; distance and grade are unit-independent.
    """
    if not profile_m:
        return ""
    parts = profile_m.split("|")
    out = []
    for p in parts:
        toks = p.split(",")
        if len(toks) < 3:
            continue
        try:
            ele_ft = float(toks[1]) * FT_PER_M
        except ValueError:
            continue
        out.append(f"{toks[0]},{int(round(ele_ft))},{toks[2]}")
    return "|".join(out)


def _splice_profile(profile_a: str, profile_b: str) -> str:
    """Append profile_b onto profile_a, shifting b's distances to continue from a.

    Profile format: 'dist,ele,grade|dist,ele,grade|...'. Best-effort — marked
    approximate by the caller via Connected Climbs column.
    """
    if not profile_a or profile_a == "nan":
        return profile_b or ""
    if not profile_b or profile_b == "nan":
        return profile_a

    a_points = profile_a.split("|")
    b_points = profile_b.split("|")
    if not a_points or not b_points:
        return profile_a + "|" + profile_b

    try:
        a_last_dist = float(a_points[-1].split(",")[0])
    except (IndexError, ValueError):
        return profile_a + "|" + profile_b

    shifted: List[str] = []
    for p in b_points:
        parts = p.split(",")
        if len(parts) < 3:
            continue
        try:
            d = float(parts[0]) + a_last_dist
        except ValueError:
            continue
        shifted.append(f"{int(round(d))},{parts[1]},{parts[2]}")

    return profile_a + "|" + "|".join(shifted)


def _build_base_row(base: pd.Series, way_ids_union: List[int], other_region: str,
                    units: str = "imperial") -> Dict:
    """Start from a region's own truncated row; keep columns that are region-specific
    (City, Country, From Center, Cycling, Category/scores, Highway Type, Surface,
    Tracktype, OSM Link). The caller overwrites the computed metric columns.
    """
    row = {c: base.get(c) for c in xlsx_columns(units) if c in base.index}
    row["All Way IDs"] = ", ".join(str(w) for w in way_ids_union)
    if way_ids_union:
        row["Start Way ID"] = way_ids_union[0]
        row["OSM Link"] = f"[{way_ids_union[0]}](https://www.openstreetmap.org/way/{way_ids_union[0]})"
    existing = str(base.get("Connected Climbs", "") or "")
    marker = f"Cross-region: {other_region}"
    if existing and existing != "nan":
        row["Connected Climbs"] = f"{existing}; {marker}"
    else:
        row["Connected Climbs"] = marker
    return row


def merge_candidate(
    candidate: MergeCandidate,
    region_a: RegionInput,
    region_b: RegionInput,
    do_recompute: bool = True,
    precomputed_geom: Optional[Tuple[Dict[int, Tuple[float, float]], Dict[int, List[int]]]] = None,
) -> MergeResult:
    ra = region_a.df.loc[candidate.idx_a]
    rb = region_b.df.loc[candidate.idx_b]

    recomputed: Optional[RecomputedClimb] = None
    if do_recompute:
        try:
            recomputed = recompute_climb_geometry(
                way_ids=candidate.way_ids_union,
                pbf_paths=[region_a.pbf_path, region_b.pbf_path],
                precomputed_geom=precomputed_geom,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"recompute failed for '{candidate.street_name}': {exc}")

    units = region_a.units
    profile_col = col_profile(units)

    if recomputed is not None:
        # recompute returns length_m, elev_gain_m, and a profile string with
        # elevation values in meters. Convert to imperial when needed.
        if units == "imperial":
            length_val = (recomputed.length_m / 1000.0) * MI_PER_KM
            gain_val = recomputed.elev_gain_m * FT_PER_M
            profile = _convert_profile_m_to_ft(recomputed.profile)
        else:
            length_val = recomputed.length_m / 1000.0  # m -> km
            gain_val = recomputed.elev_gain_m  # already meters
            profile = recomputed.profile  # already meters
        metrics = {
            col_length(units): round(length_val, 3),
            col_gain(units): round(gain_val, 0),
            "Avg Grade (%)": round(recomputed.avg_grade_pct, 2),
            "Max Grade (%)": round(recomputed.max_grade_pct, 2),
        }
        profile_source = "recomputed"
    else:
        metrics = _fallback_metrics(ra, rb, units)
        profile = _splice_profile(
            str(ra.get(profile_col, "") or ""),
            str(rb.get(profile_col, "") or ""),
        )
        profile_source = "spliced"

    row_a = _build_base_row(ra, candidate.way_ids_union, region_b.name, units)
    row_b = _build_base_row(rb, candidate.way_ids_union, region_a.name, units)
    for row in (row_a, row_b):
        row.update(metrics)
        row[profile_col] = profile
        if profile_source == "spliced":
            row["Connected Climbs"] = f"{row['Connected Climbs']} (profile approximate)"

    return MergeResult(
        candidate=candidate,
        merged_row_a=row_a,
        merged_row_b=row_b,
        recomputed=recomputed,
        profile_source=profile_source,
    )


def apply_merges_to_dataframe(
    df: pd.DataFrame, merges: List[MergeResult], side: str
) -> pd.DataFrame:
    """Return a new df where truncated rows are removed and merged rows inserted.

    side must be 'a' or 'b' to pick which merged row variant to insert.
    """
    assert side in ("a", "b")
    drop_idxs: List[int] = []
    add_rows: List[Dict] = []
    for m in merges:
        if side == "a":
            drop_idxs.append(m.candidate.idx_a)
            add_rows.append(m.merged_row_a)
        else:
            drop_idxs.append(m.candidate.idx_b)
            add_rows.append(m.merged_row_b)

    keep = df.drop(index=drop_idxs)
    if add_rows:
        add_df = pd.DataFrame(add_rows, columns=df.columns)
        out = pd.concat([keep, add_df], ignore_index=True)
    else:
        out = keep.reset_index(drop=True)
    return out


def cache_path_for(xlsx_parts: Sequence[Path], cache_dir: Path, columns: Optional[List[str]]) -> Path:
    tag = "all" if not columns else "lean"
    first_stem = Path(xlsx_parts[0]).stem
    return cache_dir / f"{first_stem}_{tag}.pkl"


def _detect_units_from_xlsx_header(xlsx_path: Path) -> str:
    """Peek at column headers without loading the body."""
    df = pd.read_excel(xlsx_path, nrows=0)
    return detect_units(list(df.columns))


def load_region(name: str, xlsx_parts: Sequence[Path], pbf_path: Path,
                columns: Optional[List[str]] = None,
                cache_dir: Optional[Path] = None,
                units: Optional[str] = None) -> RegionInput:
    """Load one region's split xlsx parts into a single dataframe.

    If `columns` is provided, only loads those columns — much faster for
    candidate-detection passes.

    If `cache_dir` is provided, caches the loaded df as a pickle keyed on
    (xlsx filenames + columns) so subsequent runs skip the slow openpyxl parse.

    If `units` is None, auto-detects from the first xlsx header.
    """
    if units is None:
        units = _detect_units_from_xlsx_header(xlsx_parts[0])

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_path_for(xlsx_parts, cache_dir, columns)
        if cache_path.exists() and all(Path(p).stat().st_mtime <= cache_path.stat().st_mtime for p in xlsx_parts):
            df = pd.read_pickle(cache_path)
            bbox = calculate_region_bbox(df)
            return RegionInput(name=name, df=df, bbox=bbox, pbf_path=pbf_path, units=units)

    read_kwargs = {"usecols": columns} if columns else {}
    frames = [pd.read_excel(p, **read_kwargs) for p in xlsx_parts]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    bbox = calculate_region_bbox(df)

    if cache_path is not None:
        try:
            df.to_pickle(cache_path)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"pickle cache write failed for {name}: {e}")

    return RegionInput(name=name, df=df, bbox=bbox, pbf_path=pbf_path, units=units)


def load_region_bbox_only(name: str, xlsx_parts: Sequence[Path], pbf_path: Path,
                          cache_dir: Path,
                          columns: Optional[List[str]] = None) -> RegionInput:
    """Return a bbox-only RegionInput (df=None) without keeping data in memory.

    Uses the pickle cache as the source of truth. If no cache exists yet for
    this region, does a full lean load, writes the cache, captures the bbox,
    then drops the df — subsequent pair processing re-reads from the cache.
    """
    units = _detect_units_from_xlsx_header(xlsx_parts[0])
    cache_path = cache_path_for(xlsx_parts, cache_dir, columns)
    if cache_path.exists() and all(Path(p).stat().st_mtime <= cache_path.stat().st_mtime for p in xlsx_parts):
        # Read only the lat/lon columns to compute bbox quickly
        df_min = pd.read_pickle(cache_path)
        bbox = calculate_region_bbox(df_min)
        del df_min
        return RegionInput(name=name, df=None, bbox=bbox, pbf_path=pbf_path, units=units)

    # First time: do a lean load (also primes the pickle cache for this region)
    ri = load_region(name, xlsx_parts, pbf_path, columns=columns, cache_dir=cache_dir, units=units)
    bbox = ri.bbox
    ri.df = None  # drop df to free memory
    return RegionInput(name=name, df=None, bbox=bbox, pbf_path=pbf_path, units=units)


def load_region_df(region: RegionInput, xlsx_parts: Sequence[Path], cache_dir: Path,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a region's df from the pickle cache (must already exist)."""
    cache_path = cache_path_for(xlsx_parts, cache_dir, columns)
    if not cache_path.exists():
        raise RuntimeError(f"pickle cache missing for {region.name}: {cache_path}")
    return pd.read_pickle(cache_path)


def candidate_detection_columns(units: str) -> List[str]:
    return [
        "Street Name", "Latitude", "Longitude", col_length(units),
        col_gain(units), "Max Grade (%)", "All Way IDs", col_profile(units),
        "Connected Climbs",
    ]


# Backwards-compat alias for imperial
CANDIDATE_DETECTION_COLUMNS = candidate_detection_columns("imperial")
