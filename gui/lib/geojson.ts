import { Climb, ClimbGeoJSON } from '@/types/climb';
import { getCategoryColor, getCategoryOpacity } from './csv-parser';

/**
 * Convert climbs to GeoJSON FeatureCollection
 * Note: Since we only have start/end points (lat/lon) from CSV,
 * we'll create point features. In the future, we could fetch full
 * geometry from OSM using the Way ID.
 */
export function climbsToGeoJSON(climbs: Climb[]): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = climbs.map(climb => {
    // For now, create a point feature at the climb location
    // TODO: Fetch full LineString geometry from OSM API using wayId
    return {
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [climb.lon, climb.lat],
      },
      properties: {
        ...climb,
        color: getCategoryColor(climb.category),
        opacity: getCategoryOpacity(climb.category),
      },
    };
  });

  return {
    type: 'FeatureCollection',
    features,
  };
}

/**
 * Calculate bounds for a set of climbs
 */
export function getClimbsBounds(climbs: Climb[]): [[number, number], [number, number]] | null {
  if (climbs.length === 0) return null;

  let minLng = Infinity;
  let maxLng = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;

  climbs.forEach(climb => {
    minLng = Math.min(minLng, climb.lon);
    maxLng = Math.max(maxLng, climb.lon);
    minLat = Math.min(minLat, climb.lat);
    maxLat = Math.max(maxLat, climb.lat);
  });

  // Add 10% padding
  const lngPadding = (maxLng - minLng) * 0.1;
  const latPadding = (maxLat - minLat) * 0.1;

  return [
    [minLng - lngPadding, minLat - latPadding],
    [maxLng + lngPadding, maxLat + latPadding],
  ];
}

/**
 * Fetch OSM way geometry for a climb
 * This will be used in a future enhancement to show actual route geometry
 */
export async function fetchOSMWayGeometry(wayId: string): Promise<GeoJSON.LineString | null> {
  try {
    const response = await fetch(`https://www.openstreetmap.org/api/0.6/way/${wayId}/full.json`);
    if (!response.ok) return null;

    const data = await response.json();
    const way = data.elements.find((el: any) => el.type === 'way');
    if (!way) return null;

    const nodes = data.elements.filter((el: any) => el.type === 'node');
    const coordinates = way.nodes.map((nodeId: number) => {
      const node = nodes.find((n: any) => n.id === nodeId);
      return node ? [node.lon, node.lat] : null;
    }).filter(Boolean);

    if (coordinates.length < 2) return null;

    return {
      type: 'LineString',
      coordinates,
    };
  } catch (error) {
    console.error('Failed to fetch OSM way geometry:', error);
    return null;
  }
}
