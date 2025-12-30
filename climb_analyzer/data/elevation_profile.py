"""
Elevation profile generation for climbs with dynamic resolution.

This module generates compact elevation profiles for climbs with smart segmentation
based on climb length, descent detection, and grade changes.
"""

from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Tuple


def determine_segment_interval(total_distance_m: float) -> float:
    """
    Determine optimal segment interval based on climb length.

    Resolution strategy - preserves detail for short climbs, scales for long trails:
    - Short climbs (<1km): 25m intervals for fine detail
    - Standard climbs (1-5km): 50m intervals
    - Medium climbs (5-10km): 75m intervals
    - Long climbs (>10km): Dynamically scaled to target max 2500 segments
      This ensures profiles fit within Excel's 32,767 char limit even for
      ultra-long trails like the Great Himalaya Trail (1700km) or PCT (4265km)

    Args:
        total_distance_m: Total climb distance in meters

    Returns:
        Optimal segment interval in meters
    """
    if total_distance_m < 1000:
        return 25
    elif total_distance_m < 5000:
        return 50
    elif total_distance_m < 10000:
        return 75
    else:
        # Scale interval to target max 1500 evenly-spaced segments
        # Critical points (peaks, valleys) add ~10-20% more segments
        # With ~15 chars/segment, 1800 total segments = 27,000 chars
        # This fits within Excel's 32,767 char cell limit with margin
        return max(100, total_distance_m / 1500)


def haversine_distance(node1: Dict, node2: Dict) -> float:
    """
    Calculate distance in meters between two lat/lon points using Haversine formula.

    Args:
        node1: Dictionary with 'lat' and 'lon' keys, or object with .lat/.lon attributes
        node2: Dictionary with 'lat' and 'lon' keys, or object with .lat/.lon attributes

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    # Handle both dict nodes and object nodes with .lat/.lon attributes
    if isinstance(node1, dict):
        lat1, lon1 = radians(node1['lat']), radians(node1['lon'])
    else:
        lat1, lon1 = radians(node1.lat), radians(node1.lon)

    if isinstance(node2, dict):
        lat2, lon2 = radians(node2['lat']), radians(node2['lon'])
    else:
        lat2, lon2 = radians(node2.lat), radians(node2.lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def calc_grade(dist1: float, ele1: float, dist2: float, ele2: float) -> float:
    """
    Calculate grade percentage between two points.

    Args:
        dist1: Distance at first point (meters)
        ele1: Elevation at first point (meters)
        dist2: Distance at second point (meters)
        ele2: Elevation at second point (meters)

    Returns:
        Grade as percentage (positive = uphill, negative = downhill)
    """
    dist_delta = dist2 - dist1
    ele_delta = ele2 - ele1
    return (ele_delta / dist_delta) * 100 if dist_delta > 0 else 0.0


def generate_elevation_profile(nodes: List[Dict], elevations: List[float]) -> str:
    """
    Generate compact elevation profile with dynamic resolution and smart segmentation.

    Features:
    - Dynamic segment interval based on climb length
    - Descent detection and isolation
    - Significant grade change detection (>3% delta)
    - Compact pipe-delimited format: "dist,ele,grade|dist,ele,grade|..."

    Args:
        nodes: List of node dictionaries with 'lat' and 'lon' keys
        elevations: List of elevation values corresponding to nodes

    Returns:
        Pipe-delimited string: "0.0,100.0,4.5|145.2,106.5,4.8|..."
        Empty string if insufficient data
    """
    # Validate input
    if not nodes or not elevations or len(nodes) < 2 or len(elevations) < 2:
        return ""

    if len(nodes) != len(elevations):
        # Handle mismatch by truncating to shorter length
        min_len = min(len(nodes), len(elevations))
        nodes = nodes[:min_len]
        elevations = elevations[:min_len]

        if min_len < 2:
            return ""

    # Safety check: Ensure elevations go from LOW to HIGH (ascending profile)
    # If first elevation is higher than last, reverse both nodes and elevations
    if elevations[0] > elevations[-1]:
        nodes = list(reversed(nodes))
        elevations = list(reversed(elevations))

    # Step 1: Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(nodes)):
        try:
            dist = haversine_distance(nodes[i-1], nodes[i])
            distances.append(distances[-1] + dist)
        except (KeyError, TypeError, ValueError):
            # Handle malformed node data
            return ""

    total_distance = distances[-1]

    # Handle edge case of zero-distance climb
    if total_distance < 1.0:
        return ""

    # Step 2: Determine optimal segment interval based on climb length
    segment_interval = determine_segment_interval(total_distance)

    # Step 3: Identify critical points (start, end, descents, grade changes)
    critical_indices = set([0, len(nodes)-1])  # Always include start and end

    # Add descent boundaries (grade transitions from positive to negative or vice versa)
    for i in range(1, len(elevations) - 1):
        try:
            prev_grade = calc_grade(distances[i-1], elevations[i-1],
                                    distances[i], elevations[i])
            next_grade = calc_grade(distances[i], elevations[i],
                                    distances[i+1], elevations[i+1])

            # Detect start of descent (positive to negative)
            if prev_grade >= 0 and next_grade < -0.5:  # -0.5% threshold to avoid noise
                critical_indices.add(i)
            # Detect end of descent (negative to positive)
            elif prev_grade < -0.5 and next_grade >= 0:
                critical_indices.add(i)
        except (ZeroDivisionError, ValueError):
            continue

    # Add significant grade change boundaries (only for non-descent sections)
    # Look ahead/behind 2-4 nodes to smooth out noise
    look_ahead = min(4, len(elevations) // 20)  # Adaptive based on climb length
    if look_ahead < 2:
        look_ahead = 2

    for i in range(look_ahead, len(elevations) - look_ahead):
        if i in critical_indices:
            continue

        try:
            # Calculate average grade before and after this point
            prev_grade = calc_grade(distances[i-look_ahead], elevations[i-look_ahead],
                                    distances[i], elevations[i])
            next_grade = calc_grade(distances[i], elevations[i],
                                    distances[i+look_ahead], elevations[i+look_ahead])

            # Skip if either section is a descent
            if prev_grade < -0.5 or next_grade < -0.5:
                continue

            # Add boundary if significant grade change (>3% delta)
            if abs(next_grade - prev_grade) > 3.0:
                critical_indices.add(i)
        except (ZeroDivisionError, ValueError):
            continue

    # Step 4: Add evenly-spaced points based on segment interval
    current_dist = segment_interval
    while current_dist < total_distance:
        # Find closest node to this distance
        closest_idx = min(range(len(distances)),
                          key=lambda i: abs(distances[i] - current_dist))
        critical_indices.add(closest_idx)
        current_dist += segment_interval

    # Step 5: Sort indices and remove points that are too close together
    sorted_indices = sorted(list(critical_indices))

    # Filter out points that are too close (< 10m apart) unless they're descent boundaries
    filtered_indices = [sorted_indices[0]]  # Always keep first point

    for i in range(1, len(sorted_indices)):
        idx = sorted_indices[i]
        prev_idx = filtered_indices[-1]

        # Calculate distance between points
        dist_between = distances[idx] - distances[prev_idx]

        # Keep point if it's far enough away OR if it's a descent boundary
        if dist_between >= 10.0:
            filtered_indices.append(idx)
        else:
            # Check if this is a descent boundary (grade sign change)
            try:
                if prev_idx > 0 and idx < len(elevations) - 1:
                    grade_before = calc_grade(distances[prev_idx-1], elevations[prev_idx-1],
                                              distances[prev_idx], elevations[prev_idx])
                    grade_after = calc_grade(distances[idx], elevations[idx],
                                             distances[idx+1], elevations[idx+1])

                    # Keep if sign change (descent boundary)
                    if (grade_before >= 0 and grade_after < 0) or (grade_before < 0 and grade_after >= 0):
                        filtered_indices.append(idx)
            except (ZeroDivisionError, ValueError, IndexError):
                pass

    # Always include last point
    if filtered_indices[-1] != sorted_indices[-1]:
        filtered_indices.append(sorted_indices[-1])

    # Step 6: Build profile segments in compact pipe-delimited format
    profile_parts = []

    for i in range(len(filtered_indices) - 1):
        idx = filtered_indices[i]
        next_idx = filtered_indices[i + 1]

        # Calculate average grade to next segment
        dist_delta = distances[next_idx] - distances[idx]
        elev_delta = elevations[next_idx] - elevations[idx]
        grade = (elev_delta / dist_delta) * 100 if dist_delta > 0 else 0.0

        # Sanity check: cap grade at reasonable max for visualization
        # This matches the max_grade smoothing logic (35% base + 50% allowance for steep climbs)
        # Extremely steep roads can reach 35-40%, so we allow up to 50% as final sanity check
        # Grade > 100% is physically impossible for roads/trails
        if grade > 50.0:
            grade = 50.0
        elif grade < -50.0:
            grade = -50.0

        # Format: "dist,ele,grade" (compact format for long trails)
        # dist: integer meters, ele: integer meters, grade: 1 decimal
        # This reduces ~20 chars/segment to ~12 chars/segment
        segment_str = f"{int(distances[idx])},{int(elevations[idx])},{grade:.1f}"
        profile_parts.append(segment_str)

    # Add final point (grade = 0 for last segment since there's no "next" point)
    last_idx = filtered_indices[-1]
    profile_parts.append(f"{int(distances[last_idx])},{int(elevations[last_idx])},0.0")

    # Step 7: Join with pipe delimiter
    return "|".join(profile_parts)


def downsample_profile(profile_str: str, max_chars: int = 30000) -> str:
    """
    Downsample elevation profile to fit within character limit (e.g., Excel cells).

    Preserves:
    - Start and end points
    - Local maxima (peaks) and minima (valleys)
    - Grade sign changes (descent boundaries)
    - Evenly samples remaining points

    Args:
        profile_str: Original pipe-delimited profile
        max_chars: Maximum character limit (default 30k for Excel safety margin)

    Returns:
        Downsampled profile string fitting within limit
    """
    if not profile_str or len(profile_str) <= max_chars:
        return profile_str

    segments = profile_str.split("|")
    if len(segments) <= 3:
        return profile_str

    # Parse all segments
    parsed = []
    for seg in segments:
        parts = seg.split(",")
        if len(parts) == 3:
            try:
                parsed.append({
                    "dist": float(parts[0]),
                    "elev": float(parts[1]),
                    "grade": float(parts[2]),
                    "original": seg,
                    "critical": False
                })
            except ValueError:
                continue

    if len(parsed) <= 3:
        return profile_str

    # Mark critical points - always preserve start and end
    parsed[0]["critical"] = True
    parsed[-1]["critical"] = True

    # Mark local maxima, minima, and grade sign changes
    for i in range(1, len(parsed) - 1):
        prev_elev = parsed[i - 1]["elev"]
        curr_elev = parsed[i]["elev"]
        next_elev = parsed[i + 1]["elev"]
        prev_grade = parsed[i - 1]["grade"]
        curr_grade = parsed[i]["grade"]

        # Local maxima (peaks)
        if curr_elev > prev_elev and curr_elev > next_elev:
            parsed[i]["critical"] = True

        # Local minima (valleys)
        elif curr_elev < prev_elev and curr_elev < next_elev:
            parsed[i]["critical"] = True

        # Grade sign changes (descent start/end)
        if (prev_grade >= 0 and curr_grade < -1) or (prev_grade < -1 and curr_grade >= 0):
            parsed[i]["critical"] = True

    # Calculate target number of segments to fit within limit
    avg_seg_len = len(profile_str) / len(segments)
    target_segments = int(max_chars / avg_seg_len) - 10  # Safety margin

    critical_indices = [i for i, p in enumerate(parsed) if p["critical"]]
    non_critical_indices = [i for i, p in enumerate(parsed) if not p["critical"]]

    # Sample non-critical points evenly to fill remaining slots
    remaining_slots = target_segments - len(critical_indices)
    if remaining_slots > 0 and non_critical_indices:
        step = max(1, len(non_critical_indices) // remaining_slots)
        sampled_indices = non_critical_indices[::step][:remaining_slots]
    else:
        sampled_indices = []

    # Combine critical and sampled indices, sort by position
    keep_indices = sorted(set(critical_indices + sampled_indices))

    # Build result using compact format (int dist, int elev, 1 decimal grade)
    # This ensures consistent output regardless of input format
    result_parts = []
    for i in keep_indices:
        p = parsed[i]
        result_parts.append(f"{int(p['dist'])},{int(p['elev'])},{p['grade']:.1f}")
    return "|".join(result_parts)


def parse_elevation_profile(profile_str: str) -> List[Dict[str, float]]:
    """
    Parse pipe-delimited elevation profile string into structured data.

    This is a helper function for testing and validation. Mobile apps
    will implement their own parsers.

    Args:
        profile_str: Pipe-delimited string "dist,ele,grade|dist,ele,grade|..."

    Returns:
        List of dictionaries with keys: 'distance_m', 'elevation_m', 'grade_pct'
    """
    if not profile_str:
        return []

    segments = []
    for seg in profile_str.split('|'):
        try:
            parts = seg.split(',')
            if len(parts) == 3:
                segments.append({
                    'distance_m': float(parts[0]),
                    'elevation_m': float(parts[1]),
                    'grade_pct': float(parts[2])
                })
        except (ValueError, IndexError):
            continue

    return segments


def get_profile_stats(profile_str: str) -> Dict[str, float]:
    """
    Calculate statistics from an elevation profile for validation.

    Args:
        profile_str: Pipe-delimited elevation profile string

    Returns:
        Dictionary with keys:
        - segment_count: Number of segments
        - total_distance_m: Total distance
        - elevation_gain_m: Total elevation gain (positive only)
        - elevation_loss_m: Total elevation loss (negative only)
        - avg_grade_pct: Average grade (positive sections only)
        - max_grade_pct: Maximum grade
        - min_grade_pct: Minimum grade (most negative)
    """
    segments = parse_elevation_profile(profile_str)

    if not segments:
        return {
            'segment_count': 0,
            'total_distance_m': 0,
            'elevation_gain_m': 0,
            'elevation_loss_m': 0,
            'avg_grade_pct': 0,
            'max_grade_pct': 0,
            'min_grade_pct': 0
        }

    segment_count = len(segments)
    total_distance = segments[-1]['distance_m']

    # Calculate elevation changes
    elevation_gain = 0.0
    elevation_loss = 0.0
    positive_grades = []
    all_grades = []

    for i in range(len(segments) - 1):
        elev_delta = segments[i+1]['elevation_m'] - segments[i]['elevation_m']
        grade = segments[i]['grade_pct']

        if elev_delta > 0:
            elevation_gain += elev_delta
        else:
            elevation_loss += abs(elev_delta)

        if grade > 0:
            positive_grades.append(grade)

        all_grades.append(grade)

    avg_grade = sum(positive_grades) / len(positive_grades) if positive_grades else 0
    max_grade = max(all_grades) if all_grades else 0
    min_grade = min(all_grades) if all_grades else 0

    return {
        'segment_count': segment_count,
        'total_distance_m': total_distance,
        'elevation_gain_m': elevation_gain,
        'elevation_loss_m': elevation_loss,
        'avg_grade_pct': avg_grade,
        'max_grade_pct': max_grade,
        'min_grade_pct': min_grade
    }
