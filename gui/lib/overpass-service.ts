/**
 * Client-side Overpass service for fetching road geometry
 * Uses the API route to proxy requests and avoid CORS issues
 */

// In-memory cache for route geometry
const routeCache = new Map<string, [number, number][]>();

/**
 * Fetch route geometry for a climb using its way IDs
 * @param wayId Primary way ID (may be a single ID or "See All Way IDs")
 * @param allWayIds Comma/space-separated list of all way IDs
 * @returns Array of [lon, lat] coordinates for the route
 */
export async function fetchRouteGeometry(
  wayId: string,
  allWayIds?: string
): Promise<[number, number][]> {
  // Determine which way IDs to use
  const wayIdsString = allWayIds || wayId;

  // Check cache first
  if (routeCache.has(wayIdsString)) {
    return routeCache.get(wayIdsString)!;
  }

  // Parse way IDs from string
  const wayIds = parseWayIds(wayIdsString);

  if (wayIds.length === 0) {
    return [];
  }

  try {
    const response = await fetch('/api/overpass', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ wayIds }),
    });

    if (!response.ok) {
      console.error('Overpass API error:', response.status);
      return [];
    }

    const data = await response.json();
    const coordinates = data.coordinates || [];

    // Cache the result
    routeCache.set(wayIdsString, coordinates);

    return coordinates;
  } catch (error) {
    console.error('Failed to fetch route geometry:', error);
    return [];
  }
}

/**
 * Parse way IDs from various string formats
 * Handles: comma-separated, space-separated, or "See All Way IDs" placeholder
 */
function parseWayIds(wayIdsString: string): string[] {
  if (!wayIdsString || wayIdsString === 'See All Way IDs') {
    return [];
  }

  return wayIdsString
    .replace(/,/g, ' ')
    .split(' ')
    .map((id) => id.trim())
    .filter((id) => id && /^\d+$/.test(id)); // Only valid numeric IDs
}

/**
 * Clear the route geometry cache
 */
export function clearRouteCache(): void {
  routeCache.clear();
}

/**
 * Get cache size for debugging
 */
export function getRouteCacheSize(): number {
  return routeCache.size;
}

/**
 * Convert route coordinates to GeoJSON LineString
 */
export function routeToGeoJSON(
  coordinates: [number, number][],
  properties: Record<string, unknown> = {}
): GeoJSON.Feature<GeoJSON.LineString> {
  return {
    type: 'Feature',
    properties,
    geometry: {
      type: 'LineString',
      coordinates,
    },
  };
}
