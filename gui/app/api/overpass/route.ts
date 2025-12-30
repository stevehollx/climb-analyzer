import { NextRequest, NextResponse } from 'next/server';

/**
 * API route to proxy Overpass API requests
 * This avoids CORS issues when fetching OSM data from the browser
 */

const OVERPASS_URL = 'https://overpass-api.de/api/interpreter';

interface OverpassNode {
  lat: number;
  lon: number;
}

interface OverpassWay {
  id: number;
  geometry?: OverpassNode[];
}

interface OverpassResponse {
  elements: OverpassWay[];
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { wayIds } = body;

    if (!wayIds || !Array.isArray(wayIds) || wayIds.length === 0) {
      return NextResponse.json(
        { error: 'wayIds array is required' },
        { status: 400 }
      );
    }

    // Build batched Overpass query for all ways
    // Format: [out:json];(way(ID1);way(ID2);way(ID3););out geom;
    const wayQueries = wayIds.map((id: string) => `way(${id})`).join(';');
    const query = `[out:json];(${wayQueries};);out geom;`;

    // Make request to Overpass API
    const response = await fetch(OVERPASS_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `data=${encodeURIComponent(query)}`,
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Overpass API error: ${response.status}` },
        { status: response.status }
      );
    }

    const data: OverpassResponse = await response.json();

    // Process and combine all coordinates
    const coordinates = processWayGeometry(data.elements, wayIds);

    return NextResponse.json({ coordinates });
  } catch (error) {
    console.error('Overpass API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch route geometry' },
      { status: 500 }
    );
  }
}

/**
 * Process way geometry from Overpass response
 * - Combines multiple ways in correct order
 * - Reverses segments if needed for proper connection
 * - Removes duplicate coordinates at segment boundaries
 * - Removes circular loop coordinates
 */
function processWayGeometry(
  elements: OverpassWay[],
  wayIds: string[]
): [number, number][] {
  const allCoordinates: [number, number][] = [];

  for (const wayId of wayIds) {
    const way = elements.find((e) => e.id === parseInt(wayId));
    if (!way?.geometry) continue;

    let coordinates: [number, number][] = way.geometry.map((node) => [
      node.lon,
      node.lat,
    ]);

    // Check if this segment connects better when reversed
    if (allCoordinates.length > 0) {
      const lastCoord = allCoordinates[allCoordinates.length - 1];
      const firstCoord = coordinates[0];
      const lastNewCoord = coordinates[coordinates.length - 1];

      const distance = calculateDistance(lastCoord, firstCoord);
      const reverseDistance = calculateDistance(lastCoord, lastNewCoord);

      if (reverseDistance < distance) {
        coordinates.reverse();
      }
    }

    // Skip duplicate coordinates at segment boundaries
    if (allCoordinates.length > 0) {
      const lastCoord = allCoordinates[allCoordinates.length - 1];
      const firstCoord = coordinates[0];
      const distance = calculateDistance(lastCoord, firstCoord);

      if (distance < 0.1) {
        // Very close, likely duplicate
        coordinates = coordinates.slice(1);
      }
    }

    allCoordinates.push(...coordinates);
  }

  // Remove circular loop coordinates (end points close to start)
  if (allCoordinates.length > 2) {
    const firstCoord = allCoordinates[0];
    let lastCoord = allCoordinates[allCoordinates.length - 1];
    const endToStartDistance = calculateDistance(lastCoord, firstCoord);

    // 500m threshold to catch most loops
    if (endToStartDistance < 500) {
      while (allCoordinates.length > 2) {
        lastCoord = allCoordinates[allCoordinates.length - 1];
        const distanceToStart = calculateDistance(lastCoord, firstCoord);
        if (distanceToStart < 500) {
          allCoordinates.pop();
        } else {
          break;
        }
      }
    }
  }

  return allCoordinates;
}

/**
 * Calculate distance between two coordinates in meters using Haversine formula
 */
function calculateDistance(
  coord1: [number, number],
  coord2: [number, number]
): number {
  const R = 6371000; // Earth's radius in meters
  const lat1 = (coord1[1] * Math.PI) / 180;
  const lat2 = (coord2[1] * Math.PI) / 180;
  const dLat = ((coord2[1] - coord1[1]) * Math.PI) / 180;
  const dLon = ((coord2[0] - coord1[0]) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}
