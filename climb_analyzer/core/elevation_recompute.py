"""Recompute climb geometry and elevation profile from OSM way IDs.

Used by the cross-region merger: after stitching two truncated climbs from
adjacent regions into one, we regenerate its profile from the underlying OSM
geometry rather than splicing the two truncated profiles.

Pipeline:
    way_ids + pbf_paths
        -> osmium getid -r (extract geometry for the requested ways)
        -> pyosmium parse (build node map + way->node lists)
        -> traversal reconstruction (order ways into a linear path)
        -> opentopodata batch elevation lookups
        -> climb_analyzer.data.elevation_profile.generate_elevation_profile
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import osmium
import requests

from climb_analyzer.data.elevation_profile import (
    determine_segment_interval,
    generate_elevation_profile,
)

logger = logging.getLogger(__name__)

OPENTOPODATA_URL = "http://localhost:5000"
OPENTOPODATA_BATCH = 100


@dataclass
class RecomputedClimb:
    nodes: List[Dict[str, float]]
    elevations_m: List[float]
    length_m: float
    elev_gain_m: float
    avg_grade_pct: float
    max_grade_pct: float
    profile: str


class _NodeWayCollector(osmium.SimpleHandler):
    def __init__(self, wanted_way_ids: set) -> None:
        super().__init__()
        self.wanted = wanted_way_ids
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.ways: Dict[int, List[int]] = {}

    def node(self, n) -> None:
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w) -> None:
        if w.id in self.wanted:
            self.ways[w.id] = [ref.ref for ref in w.nodes]


def _extract_ways(way_ids: Sequence[int], pbf_paths: Sequence[Path]) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, List[int]]]:
    """Run osmium getid -r over each PBF until all requested ways are found.

    Returns (nodes, ways) where nodes maps node_id -> (lat, lon) and ways maps
    way_id -> ordered list of node_ids. Missing ways are omitted from the result.
    """
    remaining = set(way_ids)
    nodes: Dict[int, Tuple[float, float]] = {}
    ways: Dict[int, List[int]] = {}

    for pbf in pbf_paths:
        if not remaining:
            break
        if not pbf.exists():
            logger.warning(f"PBF not found: {pbf}")
            continue

        with tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as idf:
            for wid in remaining:
                idf.write(f"w{wid}\n")
            id_file_path = Path(idf.name)

        try:
            cmd = [
                "osmium", "getid",
                "--add-referenced",
                "--overwrite",
                "--output", str(tmp_path),
                "--id-file", str(id_file_path),
                str(pbf),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            # osmium 1.14.x returns rc=1 when SOME requested ids weren't found,
            # even though the output file is written. Treat as success if the
            # output exists with non-trivial size.
            output_ok = tmp_path.exists() and tmp_path.stat().st_size > 100
            if result.returncode not in (0, 1) or not output_ok:
                err = (result.stderr or result.stdout or "").strip() or f"rc={result.returncode}"
                logger.warning(f"osmium getid failed on {pbf.name}: {err}")
                continue

            handler = _NodeWayCollector(remaining)
            handler.apply_file(str(tmp_path), locations=False)

            nodes.update(handler.nodes)
            ways.update(handler.ways)
            remaining -= set(handler.ways.keys())
        finally:
            tmp_path.unlink(missing_ok=True)
            id_file_path.unlink(missing_ok=True)

    if remaining:
        logger.warning(f"Could not locate {len(remaining)} way_id(s) across {len(pbf_paths)} PBF(s): {sorted(remaining)[:5]}...")

    return nodes, ways


def _order_ways(ways: Dict[int, List[int]]) -> List[int]:
    """Return a linear traversal of node_ids by chaining ways on shared endpoints.

    Assumes the ways form a simple path. If they don't (branching or disjoint),
    falls back to in-order concatenation of ways as given. Either way, at each
    seam a duplicate node is dropped.
    """
    if not ways:
        return []
    if len(ways) == 1:
        return list(next(iter(ways.values())))

    endpoint_to_ways: Dict[int, List[int]] = {}
    for wid, nds in ways.items():
        if len(nds) < 2:
            continue
        endpoint_to_ways.setdefault(nds[0], []).append(wid)
        endpoint_to_ways.setdefault(nds[-1], []).append(wid)

    terminals = [n for n, ws in endpoint_to_ways.items() if len(ws) == 1]
    if not terminals:
        logger.debug("No terminal endpoints found; falling back to in-order concat")
        return _concat_in_order(ways)

    start_node = terminals[0]
    current_way_id = endpoint_to_ways[start_node][0]
    visited_ways = {current_way_id}

    first_way_nodes = list(ways[current_way_id])
    if first_way_nodes[0] != start_node:
        first_way_nodes.reverse()
    path: List[int] = list(first_way_nodes)

    while len(visited_ways) < len(ways):
        tail = path[-1]
        candidates = [w for w in endpoint_to_ways.get(tail, []) if w not in visited_ways]
        if not candidates:
            logger.debug(f"Path broken at node {tail}; {len(ways) - len(visited_ways)} way(s) unvisited")
            break
        next_way_id = candidates[0]
        nxt = list(ways[next_way_id])
        if nxt[0] != tail:
            nxt.reverse()
        path.extend(nxt[1:])
        visited_ways.add(next_way_id)

    if len(visited_ways) < len(ways):
        leftover = [w for w in ways if w not in visited_ways]
        logger.debug(f"Appending {len(leftover)} orphan ways in-order")
        for wid in leftover:
            nxt = ways[wid]
            if path and nxt and nxt[0] == path[-1]:
                path.extend(nxt[1:])
            else:
                path.extend(nxt)

    return path


def _concat_in_order(ways: Dict[int, List[int]]) -> List[int]:
    path: List[int] = []
    for nds in ways.values():
        if path and nds and nds[0] == path[-1]:
            path.extend(nds[1:])
        else:
            path.extend(nds)
    return path


def _haversine_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _resample_path(coords: List[Tuple[float, float]], interval_m: float) -> List[Tuple[float, float]]:
    """Interpolate coords at a fixed linear interval. Keeps first and last."""
    if len(coords) < 2:
        return list(coords)

    cumulative = [0.0]
    for i in range(1, len(coords)):
        cumulative.append(cumulative[-1] + _haversine_m(*coords[i - 1], *coords[i]))
    total = cumulative[-1]
    if total <= 0:
        return [coords[0]]

    out: List[Tuple[float, float]] = []
    target = 0.0
    j = 1
    while target < total:
        while j < len(cumulative) and cumulative[j] < target:
            j += 1
        if j >= len(cumulative):
            break
        seg_len = cumulative[j] - cumulative[j - 1]
        if seg_len <= 0:
            out.append(coords[j])
        else:
            t = (target - cumulative[j - 1]) / seg_len
            lat = coords[j - 1][0] + t * (coords[j][0] - coords[j - 1][0])
            lon = coords[j - 1][1] + t * (coords[j][1] - coords[j - 1][1])
            out.append((lat, lon))
        target += interval_m

    out.append(coords[-1])
    return out


def _fetch_elevations(coords: List[Tuple[float, float]], datasets: str) -> List[Optional[float]]:
    elevations: List[Optional[float]] = []
    for i in range(0, len(coords), OPENTOPODATA_BATCH):
        batch = coords[i : i + OPENTOPODATA_BATCH]
        locs = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in batch)
        url = f"{OPENTOPODATA_URL}/v1/{datasets}"
        resp = requests.post(url, data={"locations": locs}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            raise RuntimeError(f"opentopodata error: {data}")
        for r in data["results"]:
            elevations.append(r.get("elevation"))
    return elevations


def extract_ways_batch(
    way_ids: Sequence[int], pbf_paths: Sequence[Path]
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, List[int]]]:
    """Public batched extraction. One osmium call per PBF for all way_ids."""
    return _extract_ways(way_ids, [Path(p) for p in pbf_paths])


def climb_terminals(
    way_ids: Sequence[int],
    nodes_map: Dict[int, Tuple[float, float]],
    ways_map: Dict[int, List[int]],
) -> List[Tuple[int, float, float]]:
    """Return the 2 terminal nodes of a climb's ordered traversal.

    Each terminal is (node_id, lat, lon). If ways form a non-simple graph,
    returns whatever _order_ways produces (first and last of the ordered path).
    Empty list if no geometry could be reconstructed.
    """
    ways = {wid: ways_map[wid] for wid in way_ids if wid in ways_map}
    if not ways:
        return []
    ordered = _order_ways(ways)
    if len(ordered) < 2:
        return []
    first, last = ordered[0], ordered[-1]
    out: List[Tuple[int, float, float]] = []
    if first in nodes_map:
        out.append((first, *nodes_map[first]))
    if last in nodes_map and last != first:
        out.append((last, *nodes_map[last]))
    return out


def climbs_share_nodes(
    way_ids_a: Sequence[int],
    way_ids_b: Sequence[int],
    ways_map: Dict[int, List[int]],
) -> bool:
    """True if the two climbs share at least one OSM node.

    A true cross-border climb is one physical road whose data was split into
    two state PBFs — their node IDs are consistent across the split and they
    share at least the border node (often 1-2 border nodes). Two unrelated
    same-name roads in adjacent states share zero nodes. This is a strict and
    reliable discriminator.
    """
    nodes_a = set()
    for wid in way_ids_a:
        if wid in ways_map:
            nodes_a.update(ways_map[wid])
    if not nodes_a:
        return False
    for wid in way_ids_b:
        if wid in ways_map:
            for nid in ways_map[wid]:
                if nid in nodes_a:
                    return True
    return False


def recompute_climb_geometry(
    way_ids: Sequence[int],
    pbf_paths: Sequence[Path],
    dataset_cascade: Sequence[str] = ("ned10m", "srtm30m", "aw3d30"),
    sample_interval_m: Optional[float] = None,
    precomputed_geom: Optional[Tuple[Dict[int, Tuple[float, float]], Dict[int, List[int]]]] = None,
) -> Optional[RecomputedClimb]:
    """Reconstruct a climb's geometry and elevation profile from its way IDs.

    Returns None if no geometry could be reconstructed. Nodes with null elevations
    are interpolated linearly from neighbors where possible; if elevation coverage
    is <50%, returns None.

    If sample_interval_m is None, uses determine_segment_interval() to match the
    engine's adaptive sampling — critical to keep elev_gain comparable to stored
    values (denser sampling inflates gain via noise accumulation).

    If precomputed_geom is supplied, skip osmium extraction and select the
    requested way_ids from the shared pre-extracted maps.
    """
    if precomputed_geom is not None:
        nodes_all, ways_all = precomputed_geom
        ways = {wid: ways_all[wid] for wid in way_ids if wid in ways_all}
        needed_nodes = {nid for nds in ways.values() for nid in nds}
        nodes_map = {nid: nodes_all[nid] for nid in needed_nodes if nid in nodes_all}
    else:
        nodes_map, ways = _extract_ways(way_ids, [Path(p) for p in pbf_paths])
    if not ways:
        return None

    ordered_node_ids = _order_ways(ways)
    coords = [nodes_map[n] for n in ordered_node_ids if n in nodes_map]
    if len(coords) < 2:
        return None

    raw_total_m = 0.0
    for i in range(1, len(coords)):
        raw_total_m += _haversine_m(*coords[i - 1], *coords[i])
    interval = sample_interval_m if sample_interval_m is not None else determine_segment_interval(raw_total_m)
    sampled = _resample_path(coords, interval)
    datasets_str = ",".join(dataset_cascade)
    raw_elevs = _fetch_elevations(sampled, datasets_str)

    known = [(i, e) for i, e in enumerate(raw_elevs) if e is not None]
    if len(known) < max(2, len(raw_elevs) // 2):
        logger.warning(
            f"Insufficient elevation coverage ({len(known)}/{len(raw_elevs)}); skipping recompute"
        )
        return None

    elevations: List[float] = []
    for i, e in enumerate(raw_elevs):
        if e is not None:
            elevations.append(e)
            continue
        left = next((idx for idx in range(i - 1, -1, -1) if raw_elevs[idx] is not None), None)
        right = next((idx for idx in range(i + 1, len(raw_elevs)) if raw_elevs[idx] is not None), None)
        if left is not None and right is not None:
            t = (i - left) / (right - left)
            elevations.append(raw_elevs[left] + t * (raw_elevs[right] - raw_elevs[left]))
        elif left is not None:
            elevations.append(raw_elevs[left])
        elif right is not None:
            elevations.append(raw_elevs[right])
        else:
            elevations.append(0.0)

    node_dicts = [{"lat": lat, "lon": lon} for lat, lon in sampled]
    profile = generate_elevation_profile(node_dicts, elevations)

    length_m = 0.0
    for i in range(1, len(sampled)):
        length_m += _haversine_m(*sampled[i - 1], *sampled[i])

    gain_m = 0.0
    for i in range(1, len(elevations)):
        d = elevations[i] - elevations[i - 1]
        if d > 0:
            gain_m += d

    avg_grade = (gain_m / length_m * 100.0) if length_m > 0 else 0.0

    max_grade = 0.0
    window_m = 100.0
    i = 0
    while i < len(sampled) - 1:
        acc = 0.0
        j = i + 1
        while j < len(sampled) and acc < window_m:
            acc += _haversine_m(*sampled[j - 1], *sampled[j])
            j += 1
        if acc > 0:
            de = elevations[min(j - 1, len(elevations) - 1)] - elevations[i]
            g = de / acc * 100.0
            if g > max_grade:
                max_grade = g
        i += max(1, (j - i) // 2)

    return RecomputedClimb(
        nodes=node_dicts,
        elevations_m=elevations,
        length_m=length_m,
        elev_gain_m=gain_m,
        avg_grade_pct=avg_grade,
        max_grade_pct=max_grade,
        profile=profile,
    )
