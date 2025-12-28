'use client';

import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Trophy } from 'lucide-react';
import { Climb } from '@/types/climb';
import { getCategoryColor } from '@/lib/csv-parser';

// Category priority for sorting overlapping pins (HC first)
const CATEGORY_ORDER: Record<string, number> = {
  'HC': 0,
  'Cat 1': 1,
  '1': 1,
  'Cat 2': 2,
  '2': 2,
  'Cat 3': 3,
  '3': 3,
  'Cat 4': 4,
  '4': 4,
  'Uncategorized': 5,
  'N/A': 5,
};

// Cluster interface
interface ClimbCluster {
  id: string;
  lat: number;
  lon: number;
  climbs: Climb[];
  highestCategory: string;
}

// Interface for climb with display position
interface ClimbWithDisplayPos extends Climb {
  displayLat?: number;
  displayLon?: number;
}

// Cluster climbs based on proximity at current zoom level
function clusterClimbs(
  climbs: Climb[],
  viewportBounds: { north: number; south: number; east: number; west: number },
  zoom: number
): { clusters: ClimbCluster[]; unclustered: Climb[] } {
  // At high zoom (>= 12), don't cluster - show individual pins
  if (zoom >= 12) {
    return { clusters: [], unclustered: climbs };
  }

  // Calculate cluster radius based on zoom level
  // Lower zoom = larger clusters
  const latSpan = viewportBounds.north - viewportBounds.south;
  const clusterRadius = latSpan * (0.08 - (zoom * 0.005)); // Decreases with zoom

  const clusters: ClimbCluster[] = [];
  const assigned = new Set<number>();

  climbs.forEach((climb, i) => {
    if (assigned.has(i)) return;

    const clusterClimbs: Climb[] = [climb];
    assigned.add(i);

    // Find all nearby climbs
    climbs.forEach((other, j) => {
      if (i === j || assigned.has(j)) return;

      const latDiff = Math.abs(climb.lat - other.lat);
      const lonDiff = Math.abs(climb.lon - other.lon);

      if (latDiff < clusterRadius && lonDiff < clusterRadius) {
        clusterClimbs.push(other);
        assigned.add(j);
      }
    });

    // Only create cluster if 2+ climbs
    if (clusterClimbs.length >= 2) {
      // Calculate center
      const centerLat = clusterClimbs.reduce((sum, c) => sum + c.lat, 0) / clusterClimbs.length;
      const centerLon = clusterClimbs.reduce((sum, c) => sum + c.lon, 0) / clusterClimbs.length;

      // Find highest category in cluster
      const highestCategory = clusterClimbs.reduce((best, c) => {
        const bestOrder = CATEGORY_ORDER[best] ?? 5;
        const cOrder = CATEGORY_ORDER[c.category] ?? 5;
        return cOrder < bestOrder ? c.category : best;
      }, 'Uncategorized');

      clusters.push({
        id: `cluster-${i}`,
        lat: centerLat,
        lon: centerLon,
        climbs: clusterClimbs,
        highestCategory,
      });
    }
  });

  // Get unclustered climbs (single climbs that weren't grouped)
  const clusteredIndices = new Set<number>();
  clusters.forEach(cluster => {
    cluster.climbs.forEach(c => {
      const idx = climbs.findIndex(climb => climb.wayId === c.wayId && climb.lat === c.lat);
      if (idx !== -1) clusteredIndices.add(idx);
    });
  });

  const unclustered = climbs.filter((_, i) => !clusteredIndices.has(i));

  return { clusters, unclustered };
}

// Function to stagger overlapping pins
function staggerOverlappingPins(
  climbs: Climb[],
  viewportBounds: { north: number; south: number; east: number; west: number },
  zoom: number
): ClimbWithDisplayPos[] {
  // Only stagger at high zoom levels (12+)
  if (zoom < 12) {
    return climbs.map(c => ({ ...c, displayLat: c.lat, displayLon: c.lon }));
  }

  const span = viewportBounds.north - viewportBounds.south;
  const overlapThreshold = span * 0.015; // 1.5% of visible span
  const offsetDistance = span * 0.012;   // 1.2% of visible span

  // Group climbs by proximity
  const groups: Climb[][] = [];
  const assigned = new Set<number>();

  climbs.forEach((climb, i) => {
    if (assigned.has(i)) return;

    const group: Climb[] = [climb];
    assigned.add(i);

    climbs.forEach((other, j) => {
      if (i === j || assigned.has(j)) return;

      const latDiff = Math.abs(climb.lat - other.lat);
      const lonDiff = Math.abs(climb.lon - other.lon);

      if (latDiff < overlapThreshold && lonDiff < overlapThreshold) {
        group.push(other);
        assigned.add(j);
      }
    });

    groups.push(group);
  });

  // Process groups and assign display positions
  const result: ClimbWithDisplayPos[] = [];

  groups.forEach(group => {
    if (group.length === 1) {
      result.push({ ...group[0], displayLat: group[0].lat, displayLon: group[0].lon });
      return;
    }

    // Sort by category (HC first)
    group.sort((a, b) => {
      const orderA = CATEGORY_ORDER[a.category] ?? 5;
      const orderB = CATEGORY_ORDER[b.category] ?? 5;
      return orderA - orderB;
    });

    // Calculate center of group
    const centerLat = group.reduce((sum, c) => sum + c.lat, 0) / group.length;
    const centerLon = group.reduce((sum, c) => sum + c.lon, 0) / group.length;

    // Arrange in circle around center
    const angleStep = (2 * Math.PI) / group.length;

    group.forEach((climb, i) => {
      const angle = i * angleStep - Math.PI / 2; // Start from top
      const displayLat = centerLat + offsetDistance * Math.cos(angle);
      // Adjust for longitude distortion at latitude
      const displayLon = centerLon + offsetDistance * Math.sin(angle) / Math.cos(centerLat * Math.PI / 180);

      result.push({ ...climb, displayLat, displayLon });
    });
  });

  return result;
}

interface ClimbMapProps {
  climbs: Climb[];
  allClimbs?: Climb[]; // All climbs before filtering, for finding connected climbs
  bounds?: {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
  };
  showAllRoutes?: boolean;
  scoreType?: 'basic' | 'fiets' | 'pdi'; // Score type for "Find Best Climb" feature
  onClimbClick?: (climb: Climb) => void; // Callback when a climb is clicked
}

export function ClimbMap({ climbs, allClimbs, bounds, showAllRoutes = false, scoreType = 'basic', onClimbClick }: ClimbMapProps) {
  // Use allClimbs for lookups if provided, otherwise fall back to climbs
  const climbsForLookup = allClimbs || climbs;
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [viewportBounds, setViewportBounds] = useState<{
    north: number;
    south: number;
    east: number;
    west: number;
  } | null>(null);
  const [currentZoom, setCurrentZoom] = useState(10);
  const lastClusterZoomRef = useRef<number>(10); // Track last zoom level for cluster buffer
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const clusterMarkersRef = useRef<maplibregl.Marker[]>([]); // Separate ref for cluster markers
  const hasInitialFit = useRef(false);
  const [highlightedClimbIndex, setHighlightedClimbIndex] = useState<number | null>(null);
  const [clusteringEnabled, setClusteringEnabled] = useState(true);
  const [highlightedWayIds, setHighlightedWayIds] = useState<Set<string>>(new Set());
  const [activeClimbId, setActiveClimbId] = useState<string | null>(null); // Most recently clicked climb
  const routeDataCache = useRef<Map<string, any>>(new Map());
  const [isRenderingAllRoutes, setIsRenderingAllRoutes] = useState(false);
  const [renderProgress, setRenderProgress] = useState({ current: 0, total: 0 });

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    // Calculate center point
    const centerLat = bounds
      ? (bounds.minLat + bounds.maxLat) / 2
      : climbs[0]?.lat || 0;
    const centerLon = bounds
      ? (bounds.minLon + bounds.maxLon) / 2
      : climbs[0]?.lon || 0;

    // Initialize map with terrain/elevation tiles
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'raster-tiles': {
            type: 'raster',
            tiles: [
              'https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
              'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',
              'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'
            ],
            tileSize: 256,
            attribution: 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
          }
        },
        layers: [
          {
            id: 'simple-tiles',
            type: 'raster',
            source: 'raster-tiles',
            minzoom: 0,
            maxzoom: 17
          }
        ]
      },
      center: [centerLon, centerLat],
      zoom: 10,
    });

    // Add navigation controls
    map.current.addControl(new maplibregl.NavigationControl(), 'top-right');

    map.current.on('load', () => {
      setMapLoaded(true);
      // Set initial viewport bounds
      updateViewportBounds();
    });

    // Update viewport bounds and zoom when map moves
    const updateViewportBounds = () => {
      if (!map.current) return;
      const bounds = map.current.getBounds();
      setViewportBounds({
        north: bounds.getNorth(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        west: bounds.getWest(),
      });

      const newZoom = map.current.getZoom();
      const lastClusterZoom = lastClusterZoomRef.current;

      // Zoom buffer logic: prevent flickering when slightly zooming out
      // Only re-cluster if zooming IN or zooming OUT significantly (> 0.5 levels)
      const shouldUpdateClusters = newZoom > lastClusterZoom || newZoom < lastClusterZoom - 0.5;

      if (shouldUpdateClusters) {
        lastClusterZoomRef.current = newZoom;
      }

      setCurrentZoom(newZoom);
    };

    map.current.on('moveend', updateViewportBounds);

    // Fit to bounds if provided
    if (bounds && map.current) {
      map.current.fitBounds(
        [
          [bounds.minLon, bounds.minLat],
          [bounds.maxLon, bounds.maxLat],
        ],
        { padding: 50, duration: 0 }
      );
    }

    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, []);

  // Add climb markers and routes when map is loaded or climbs change or viewport changes
  useEffect(() => {
    if (!map.current || !mapLoaded || climbs.length === 0) return;

    console.log('=== ClimbMap useEffect: Re-rendering markers ===');
    console.log('Currently highlighted wayIds:', Array.from(highlightedWayIds));

    // Find any marker with an open popup and track which climb it belongs to
    let openPopupMarker: maplibregl.Marker | null = null;
    let openPopupClimbWayId: string | null = null;

    markersRef.current.forEach(marker => {
      const popup = marker.getPopup();
      if (popup && popup.isOpen()) {
        openPopupMarker = marker;
        // Extract wayId from the marker's element dataset (we'll add this below)
        const el = marker.getElement();
        openPopupClimbWayId = el.dataset.wayId || null;
        console.log('  - Found open popup for climb wayId:', openPopupClimbWayId);
      }
    });

    // Remove all markers EXCEPT the one with open popup
    markersRef.current.forEach(marker => {
      if (marker !== openPopupMarker) {
        marker.remove();
      }
    });

    // Remove all cluster markers
    clusterMarkersRef.current.forEach(marker => marker.remove());
    clusterMarkersRef.current = [];

    // Start fresh array, will add back the open popup marker if it still applies
    markersRef.current = [];

    // DON'T remove highlighted routes that are still needed
    // Only remove old index-based layers if they exist
    climbs.forEach((_, index) => {
      const layerId = `climb-route-${index}`;
      if (map.current!.getLayer(layerId)) {
        console.log('Removing old index-based layer:', layerId);
        map.current!.removeLayer(layerId);
      }
      if (map.current!.getSource(layerId)) {
        map.current!.removeSource(layerId);
      }
    });

    // Filter climbs to only those in viewport (if viewport bounds are set)
    let visibleClimbs: ClimbWithDisplayPos[] = climbs.map(c => ({ ...c, displayLat: c.lat, displayLon: c.lon }));
    let clusters: ClimbCluster[] = [];

    if (viewportBounds) {
      const filteredClimbs = climbs.filter(climb =>
        climb.lat >= viewportBounds.south &&
        climb.lat <= viewportBounds.north &&
        climb.lon >= viewportBounds.west &&
        climb.lon <= viewportBounds.east
      );

      // Apply clustering if enabled and at low zoom
      if (clusteringEnabled && currentZoom < 12) {
        const result = clusterClimbs(filteredClimbs, viewportBounds, currentZoom);
        clusters = result.clusters;

        // Apply staggering to unclustered climbs
        visibleClimbs = staggerOverlappingPins(result.unclustered, viewportBounds, currentZoom);
        console.log(`Viewport: ${climbs.length} total, ${filteredClimbs.length} in view, ${clusters.length} clusters, ${visibleClimbs.length} individual (zoom: ${currentZoom.toFixed(1)})`);
      } else {
        // No clustering - just apply staggering
        visibleClimbs = staggerOverlappingPins(filteredClimbs, viewportBounds, currentZoom);
        console.log(`Viewport filter: ${climbs.length} total climbs, ${visibleClimbs.length} visible in viewport (zoom: ${currentZoom.toFixed(1)})`);
      }
    }

    // Add cluster markers
    clusters.forEach(cluster => {
      const color = getCategoryColor(cluster.highestCategory as any);

      // Create cluster marker element
      const el = document.createElement('div');
      el.className = 'cluster-marker';
      el.style.backgroundColor = color;
      el.style.color = 'white';
      el.style.borderRadius = '50%';
      el.style.border = '3px solid white';
      el.style.boxShadow = '0 3px 8px rgba(0,0,0,0.5)';
      el.style.cursor = 'pointer';
      el.style.display = 'flex';
      el.style.alignItems = 'center';
      el.style.justifyContent = 'center';
      el.style.fontWeight = 'bold';
      el.style.transition = 'transform 0.2s';

      // Size based on cluster count
      const count = cluster.climbs.length;
      const size = count < 10 ? 32 : count < 50 ? 40 : count < 100 ? 48 : 56;
      el.style.width = `${size}px`;
      el.style.height = `${size}px`;
      el.style.fontSize = count < 10 ? '12px' : count < 100 ? '14px' : '16px';

      el.textContent = count.toString();

      // Hover effect
      el.addEventListener('mouseenter', () => {
        el.style.transform = 'scale(1.15)';
      });
      el.addEventListener('mouseleave', () => {
        el.style.transform = 'scale(1)';
      });

      // Click to zoom in and expand cluster
      el.addEventListener('click', () => {
        if (!map.current) return;

        // Calculate bounds of cluster climbs
        const lats = cluster.climbs.map(c => c.lat);
        const lons = cluster.climbs.map(c => c.lon);
        const minLat = Math.min(...lats);
        const maxLat = Math.max(...lats);
        const minLon = Math.min(...lons);
        const maxLon = Math.max(...lons);

        // Zoom to fit cluster with padding
        map.current.fitBounds(
          [[minLon, minLat], [maxLon, maxLat]],
          { padding: 50, duration: 500, maxZoom: 14 }
        );
      });

      // Create popup with cluster info
      const popup = new maplibregl.Popup({
        offset: size / 2 + 5,
        closeButton: false,
        closeOnClick: true
      }).setHTML(`
        <div style="font-family: system-ui; text-align: center; padding: 4px;">
          <div style="font-weight: 600; margin-bottom: 4px;">${count} Climbs</div>
          <div style="font-size: 11px; color: #666;">Click to zoom in</div>
        </div>
      `);

      const marker = new maplibregl.Marker({
        element: el,
        anchor: 'center'
      })
        .setLngLat([cluster.lon, cluster.lat])
        .setPopup(popup)
        .addTo(map.current!);

      clusterMarkersRef.current.push(marker);
    });

    // Add markers only for visible climbs (using display positions for staggered pins)
    visibleClimbs.forEach((climb, index) => {
      // If this climb already has an open popup marker, reuse it instead of creating new
      if (openPopupMarker && openPopupClimbWayId === climb.wayId) {
        console.log('  - Reusing existing marker with open popup for climb:', climb.streetName);
        markersRef.current.push(openPopupMarker);
        return; // Skip creating a new marker for this climb
      }

      const color = getCategoryColor(climb.category);

      // Create marker element - larger and more visible
      const el = document.createElement('div');
      el.className = 'climb-marker';
      el.dataset.wayId = climb.wayId || ''; // Store wayId for tracking
      el.style.backgroundColor = color;
      el.style.width = '20px';
      el.style.height = '20px';
      el.style.borderRadius = '50%';
      el.style.border = '3px solid white';
      el.style.boxShadow = '0 3px 6px rgba(0,0,0,0.4)';
      el.style.cursor = 'pointer';
      el.style.transition = 'box-shadow 0.2s, border-width 0.2s';

      // Add hover effect using border and shadow to avoid repositioning
      el.addEventListener('mouseenter', () => {
        el.style.borderWidth = '4px';
        el.style.boxShadow = '0 4px 10px rgba(0,0,0,0.6)';
      });
      el.addEventListener('mouseleave', () => {
        el.style.borderWidth = '3px';
        el.style.boxShadow = '0 3px 6px rgba(0,0,0,0.4)';
      });

      // Parse connected climbs (filter out empty strings and "None")
      const connectedClimbs = climb.connectedClimbs
        ? climb.connectedClimbs.split(',').map(w => w.trim()).filter(w => w && w !== '' && w.toLowerCase() !== 'none')
        : [];

      console.log(`Climb "${climb.streetName}" has ${connectedClimbs.length} connected climbs:`, connectedClimbs);

      // Create popup content with prominent category display
      let popupContent = `
        <div style="font-family: system-ui; min-width: 200px;">
          <h4 style="margin: 0 0 4px 0; font-weight: 600;">
            ${climb.streetName || 'Unnamed Climb'}
          </h4>
          <div style="display: inline-block; background: ${color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-bottom: 8px;">
            ${climb.category}
          </div>
          <p style="margin: 0 0 4px 0; font-size: 12px; color: #666;">
            ${climb.city ? `${climb.city}, ` : ''}${climb.state || ''}${climb.country ? `, ${climb.country}` : ''}
          </p>
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 12px;">
            <div style="margin-bottom: 4px;">
              <strong>Elevation Gain:</strong> ${climb.elevationGain.toFixed(0)} ft
            </div>
            <div style="margin-bottom: 4px;">
              <strong>Length:</strong> ${climb.length.toFixed(2)} mi
            </div>
            <div style="margin-bottom: 4px;">
              <strong>Avg Grade:</strong> ${climb.avgGrade.toFixed(1)}%
            </div>
            <div style="margin-bottom: 4px;">
              <strong>Max Grade:</strong> ${climb.maxGrade.toFixed(1)}%
            </div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
              <strong>Basic Score:</strong> ${climb.basicScore.toFixed(1)}
            </div>
          </div>
      `;

      // Add connected climbs section if there are any
      if (connectedClimbs.length > 0) {
        popupContent += `
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 12px;">
            <div style="margin-bottom: 4px;"><strong>Connected Climbs:</strong></div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
              ${connectedClimbs.map((connectedName, idx) => {
                // Connected climbs are stored as street names in the data
                return `<a href="#" class="connected-climb-link-${index}-${idx}" data-name="${connectedName}" style="color: #2563eb; text-decoration: underline; cursor: pointer; font-size: 11px;">${connectedName}</a>`;
              }).join(', ')}
            </div>
          </div>
        `;
      }

      popupContent += `</div>`;

      const popup = new maplibregl.Popup({
        offset: 15,
        closeButton: true,
        closeOnClick: true
      }).setHTML(popupContent);

      // Add event listeners for connected climb links after popup opens
      popup.on('open', () => {
        console.log(`Popup opened for "${climb.streetName}", setting up ${connectedClimbs.length} connected climb link(s)`);
        connectedClimbs.forEach((connectedName, idx) => {
          const link = document.querySelector(`.connected-climb-link-${index}-${idx}`);
          console.log(`Looking for link .connected-climb-link-${index}-${idx} for climb "${connectedName}":`, link ? 'FOUND' : 'NOT FOUND');
          if (link) {
            link.addEventListener('click', (e) => {
              e.preventDefault();
              console.log('=== CONNECTED CLIMB LINK CLICKED ===');
              console.log('Clicked climb name:', connectedName);
              console.log('Current highlighted wayIds before click:', Array.from(highlightedWayIds));
              // Find the climb with this street name in all climbs (not just filtered)
              // Note: Connected climbs are stored as street names, not wayIds
              const targetClimb = climbsForLookup.find(c => c.streetName.trim() === connectedName.trim());
              console.log('Target climb found:', targetClimb ? `"${targetClimb.streetName}" (wayId: ${targetClimb.wayId}, category: ${targetClimb.category})` : 'NOT FOUND');
              if (targetClimb) {
                // Find the index in the full array for consistency
                const targetClimbIndex = climbsForLookup.findIndex(c => c.streetName.trim() === connectedName.trim());
                console.log('Target climb index:', targetClimbIndex);
                console.log('Target climb wayId:', targetClimb.wayId);
                // Highlight this route
                setHighlightedClimbIndex(targetClimbIndex);
                console.log('Calling showRouteForClimb with wayId:', targetClimb.wayId);
                showRouteForClimb(targetClimb.wayId, targetClimb.category, targetClimbIndex, true);
              } else {
                console.warn('Connected climb not found in dataset:', connectedName);
                console.log('Available climbs:', climbsForLookup.map(c => c.streetName));
              }
            });
          } else {
            console.warn(`Link element not found in DOM: .connected-climb-link-${index}-${idx}`);
          }
        });
      });

      const marker = new maplibregl.Marker({
        element: el,
        anchor: 'center'
      })
        // Use display position for staggered pins, fall back to actual position
        .setLngLat([climb.displayLon ?? climb.lon, climb.displayLat ?? climb.lat])
        .addTo(map.current!);

      // Attach popup but don't let MapLibre handle the click behavior
      marker.setPopup(popup);

      // Custom click handler - supports Shift+click for multi-selection
      el.addEventListener('click', (e) => {
        const isMultiSelect = e.shiftKey;
        console.log('🔵 MARKER CLICKED:', climb.streetName, isMultiSelect ? '(MULTI-SELECT)' : '');

        e.stopPropagation();
        e.preventDefault();

        const climbIndex = climbs.findIndex(c => c.wayId === climb.wayId && c.lat === climb.lat && c.lon === climb.lon);

        if (climbIndex !== -1) {
          // Set this climb as the active climb
          setActiveClimbId(climb.wayId);
          setHighlightedClimbIndex(climbIndex);

          if (isMultiSelect) {
            // Shift+click: Add to multi-selection (don't clear previous)
            console.log('  - Adding to multi-selection');
            if (climb.wayId && climb.wayId !== '') {
              showRouteForClimb(climb.wayId, climb.category, climbIndex, true);
            }
          } else {
            // Normal click: Clear previous and select only this one
            console.log('  - Single selection (clearing previous)');

            // Clear other highlighted routes but keep this one
            highlightedWayIds.forEach(wayId => {
              if (wayId !== climb.wayId) {
                const layerId = `climb-route-${wayId}`;
                const highlightLayerId = `climb-route-highlight-${wayId}`;
                [layerId, highlightLayerId].forEach(id => {
                  if (map.current!.getLayer(id)) {
                    map.current!.removeLayer(id);
                  }
                  if (map.current!.getSource(id)) {
                    map.current!.removeSource(id);
                  }
                });
              }
            });
            setHighlightedWayIds(new Set());

            // Show route for this climb
            if (climb.wayId && climb.wayId !== '') {
              showRouteForClimb(climb.wayId, climb.category, climbIndex, true);
            }
          }

          // Open popup
          if (!marker.getPopup().isOpen()) {
            marker.togglePopup();
          }

          // Center the climb in the map view
          map.current!.flyTo({
            center: [climb.lon, climb.lat],
            duration: 500
          });

          // Notify parent
          setTimeout(() => {
            if (onClimbClick) {
              onClimbClick(climb);
            }
          }, 100);
        }
      });

      // Store marker reference for cleanup
      markersRef.current.push(marker);

      // Fetch and display route if showAllRoutes is enabled
      if (showAllRoutes && climb.wayId && climb.wayId !== '') {
        showRouteForClimb(climb.wayId, climb.category, index, false);
      }
    });

    // Fit to bounds only on first load (not on every viewport change)
    if (bounds && map.current && !hasInitialFit.current && climbs.length > 0) {
      map.current.fitBounds(
        [
          [bounds.minLon, bounds.minLat],
          [bounds.maxLon, bounds.maxLat],
        ],
        { padding: 50 }
      );
      hasInitialFit.current = true;
    }
  }, [mapLoaded, climbs, viewportBounds, currentZoom, clusteringEnabled]);

  // Function to clear all highlighted routes
  const clearHighlightedRoutes = () => {
    if (!map.current) return;

    console.log('=== Clearing highlighted routes ===');
    console.log('Current highlighted wayIds:', Array.from(highlightedWayIds));

    // Remove all highlighted route layers and sources
    highlightedWayIds.forEach(wayId => {
      const layerId = `climb-route-${wayId}`;
      const highlightLayerId = `climb-route-highlight-${wayId}`;

      [layerId, highlightLayerId].forEach(id => {
        if (map.current!.getLayer(id)) {
          console.log('  Removing layer:', id);
          map.current!.removeLayer(id);
        }
        if (map.current!.getSource(id)) {
          console.log('  Removing source:', id);
          map.current!.removeSource(id);
        }
      });
    });

    // Clear all state
    setHighlightedWayIds(new Set());
    setHighlightedClimbIndex(null);
    setActiveClimbId(null);
    console.log('All highlighted routes cleared');
  };

  // Function to render all routes with progress tracking
  const renderAllRoutes = async () => {
    if (!map.current || isRenderingAllRoutes) return;

    console.log('=== Rendering all routes ===');
    setIsRenderingAllRoutes(true);
    setRenderProgress({ current: 0, total: climbs.length });

    // Filter climbs that have valid wayIds
    const climbsWithWayIds = climbs.filter(c => c.wayId && c.wayId !== '');
    setRenderProgress({ current: 0, total: climbsWithWayIds.length });

    console.log(`Rendering ${climbsWithWayIds.length} routes`);

    // Process climbs in batches to avoid blocking the UI
    const batchSize = 5;
    for (let i = 0; i < climbsWithWayIds.length; i += batchSize) {
      const batch = climbsWithWayIds.slice(i, Math.min(i + batchSize, climbsWithWayIds.length));

      // Process batch in parallel
      await Promise.all(
        batch.map(async (climb, batchIndex) => {
          const actualIndex = i + batchIndex;
          try {
            await showRouteForClimb(climb.wayId, climb.category, actualIndex, true);
          } catch (error) {
            console.error(`Failed to render route for ${climb.streetName}:`, error);
          }
        })
      );

      // Update progress
      setRenderProgress({ current: Math.min(i + batchSize, climbsWithWayIds.length), total: climbsWithWayIds.length });

      // Small delay to allow UI to update
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    console.log('All routes rendered');
    setIsRenderingAllRoutes(false);
    setRenderProgress({ current: 0, total: 0 });
  };

  // Function to find and highlight the best climb in the current viewport
  const findBestClimbInView = () => {
    if (!map.current || !viewportBounds || climbs.length === 0) return;

    // Filter climbs in current viewport
    const visibleClimbs = climbs.filter(climb =>
      climb.lat >= viewportBounds.south &&
      climb.lat <= viewportBounds.north &&
      climb.lon >= viewportBounds.west &&
      climb.lon <= viewportBounds.east
    );

    if (visibleClimbs.length === 0) {
      console.log('No climbs visible in current viewport');
      return;
    }

    // Find best climb by current score type
    const getScore = (climb: Climb): number => {
      switch (scoreType) {
        case 'pdi': return climb.pdiScore || 0;
        case 'fiets': return climb.fietsScore || 0;
        case 'basic':
        default: return climb.basicScore || 0;
      }
    };

    const bestClimb = visibleClimbs.reduce((best, climb) =>
      getScore(climb) > getScore(best) ? climb : best
    );

    console.log(`Found best climb: ${bestClimb.streetName} with ${scoreType} score: ${getScore(bestClimb)}`);

    // Find the index of the best climb
    const climbIndex = climbs.findIndex(c => c.wayId === bestClimb.wayId);
    if (climbIndex !== -1) {
      setHighlightedClimbIndex(climbIndex);

      // Show route for this climb
      if (bestClimb.wayId && bestClimb.wayId !== '') {
        showRouteForClimb(bestClimb.wayId, bestClimb.category, climbIndex, true);
      }

      // Fly to the climb
      map.current.flyTo({
        center: [bestClimb.lon, bestClimb.lat],
        zoom: 14,
        duration: 1000
      });

      // Find the marker for this climb and open its popup
      const marker = markersRef.current.find(m => {
        const el = m.getElement();
        return el.dataset.wayId === bestClimb.wayId;
      });

      if (marker) {
        // Small delay to ensure fly animation is started
        setTimeout(() => {
          const popup = marker.getPopup();
          if (popup && !popup.isOpen()) {
            marker.togglePopup();
          }
        }, 500);
      }

      // Notify parent component
      if (onClimbClick) {
        setTimeout(() => onClimbClick(bestClimb), 600);
      }
    }
  };

  // Function to show route for a climb (either highlighted or as part of showAllRoutes)
  const showRouteForClimb = async (wayId: string, category: string, index: number, isHighlighted: boolean) => {
    if (!map.current) return;

    console.log(`=== showRouteForClimb called ===`);
    console.log('  wayId:', wayId);
    console.log('  category:', category);
    console.log('  index:', index);
    console.log('  isHighlighted:', isHighlighted);

    // Check cache first
    let geojson = routeDataCache.current.get(wayId);
    console.log('  Cached geojson:', geojson ? 'YES' : 'NO');

    if (!geojson) {
      // Fetch from Overpass API
      try {
        const query = `[out:json];way(${wayId});out geom;`;
        const response = await fetch(`https://overpass-api.de/api/interpreter`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: `data=${encodeURIComponent(query)}`
        });

        if (!response.ok) {
          console.error(`Failed to fetch route for way ${wayId}: HTTP ${response.status}`);
          return;
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          const text = await response.text();
          console.error(`Expected JSON but got ${contentType} for way ${wayId}:`, text.substring(0, 200));
          return;
        }

        const data = await response.json();

        if (data.elements && data.elements.length > 0) {
          const way = data.elements[0];
          if (way.geometry && way.geometry.length > 0) {
            const coordinates = way.geometry.map((node: any) => [node.lon, node.lat]);
            geojson = {
              type: 'Feature' as const,
              properties: {},
              geometry: {
                type: 'LineString' as const,
                coordinates: coordinates,
              },
            };
            // Cache it
            routeDataCache.current.set(wayId, geojson);
          }
        }
      } catch (error) {
        console.error(`Failed to fetch route geometry for way ${wayId}:`, error);
        return;
      }
    }

    if (!geojson) {
      console.log('  No geojson available, aborting');
      return;
    }

    const color = getCategoryColor(category as any);
    // Use wayId for layer naming to ensure consistency across clicks
    const layerId = `climb-route-${wayId}`;
    const highlightLayerId = `climb-route-highlight-${wayId}`;

    console.log('  Using layer IDs:', { layerId, highlightLayerId });
    console.log('  Color for category:', color);

    // Only remove layers for THIS specific climb (not all highlighted climbs)
    [layerId, highlightLayerId].forEach(id => {
      const hasLayer = map.current!.getLayer(id);
      const hasSource = map.current!.getSource(id);
      if (hasLayer) {
        console.log('  Removing existing layer:', id);
        map.current!.removeLayer(id);
      }
      if (hasSource) {
        console.log('  Removing existing source:', id);
        map.current!.removeSource(id);
      }
    });

    if (isHighlighted) {
      console.log('  Adding HIGHLIGHTED route');
      // Add highlighted route (wider, more opaque)
      if (!map.current!.getSource(highlightLayerId)) {
        console.log('  Creating source:', highlightLayerId);
        map.current!.addSource(highlightLayerId, {
          type: 'geojson',
          data: geojson,
        });
      } else {
        console.log('  Source already exists:', highlightLayerId);
      }

      if (!map.current!.getLayer(highlightLayerId)) {
        console.log('  Creating layer:', highlightLayerId);
        map.current!.addLayer({
          id: highlightLayerId,
          type: 'line',
          source: highlightLayerId,
          layout: {
            'line-join': 'round',
            'line-cap': 'round',
          },
          paint: {
            'line-color': color,
            'line-width': 8,
            'line-opacity': 1.0,
          },
        });
      } else {
        console.log('  Layer already exists:', highlightLayerId);
      }

      // Track this highlighted route
      setHighlightedWayIds(prev => {
        const newSet = new Set([...prev, wayId]);
        console.log('  Updated highlightedWayIds:', Array.from(newSet));
        return newSet;
      });
    } else {
      // Add normal route
      if (!map.current!.getSource(layerId)) {
        map.current!.addSource(layerId, {
          type: 'geojson',
          data: geojson,
        });
      }

      if (!map.current!.getLayer(layerId)) {
        map.current!.addLayer({
          id: layerId,
          type: 'line',
          source: layerId,
          layout: {
            'line-join': 'round',
            'line-cap': 'round',
          },
          paint: {
            'line-color': color,
            'line-width': 4,
            'line-opacity': 0.6,
          },
        });
      }
    }
  };

  if (!climbs || climbs.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-100 rounded-lg">
        <p className="text-gray-500">No climbs to display</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      {!mapLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg z-10">
          <p className="text-gray-600">Loading map...</p>
        </div>
      )}

      {/* Control buttons - top left */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
        {/* Clear routes button - only show if there are highlighted routes */}
        {highlightedWayIds.size > 0 && (
          <button
            onClick={clearHighlightedRoutes}
            className="bg-white hover:bg-gray-100 text-gray-800 font-semibold py-2 px-4 border border-gray-300 rounded shadow-md transition-colors"
            title="Clear all highlighted routes"
          >
            Clear {highlightedWayIds.size} Route{highlightedWayIds.size > 1 ? 's' : ''}
          </button>
        )}

        {/* Multi-select hint */}
        {highlightedWayIds.size === 0 && (
          <div className="bg-white/80 text-xs text-gray-600 py-1 px-2 rounded shadow-sm">
            Shift+click to select multiple
          </div>
        )}

        {/* Show all routes button */}
        <button
          onClick={renderAllRoutes}
          disabled={isRenderingAllRoutes}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 border border-blue-700 rounded shadow-md transition-colors"
        >
          {isRenderingAllRoutes ? 'Rendering...' : 'Show All Routes'}
        </button>

        {/* Find Best Climb button */}
        <button
          onClick={findBestClimbInView}
          className="bg-amber-500 hover:bg-amber-600 text-white font-semibold py-2 px-4 border border-amber-600 rounded shadow-md transition-colors flex items-center gap-2"
          title={`Find highest ${scoreType} score climb in current view`}
        >
          <Trophy className="h-4 w-4" />
          Find Best
        </button>

        {/* Clustering toggle */}
        <button
          onClick={() => setClusteringEnabled(!clusteringEnabled)}
          className={`font-semibold py-2 px-4 border rounded shadow-md transition-colors ${
            clusteringEnabled
              ? 'bg-purple-600 hover:bg-purple-700 text-white border-purple-700'
              : 'bg-white hover:bg-gray-100 text-gray-800 border-gray-300'
          }`}
          title={clusteringEnabled ? 'Clustering enabled - click to disable' : 'Clustering disabled - click to enable'}
        >
          {clusteringEnabled ? 'Clusters On' : 'Clusters Off'}
        </button>
      </div>

      {/* Progress bar - bottom center, above map controls */}
      {isRenderingAllRoutes && renderProgress.total > 0 && (
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 z-10 bg-white rounded-lg shadow-lg p-4 min-w-[300px]">
          <div className="text-sm text-gray-700 mb-2 text-center">
            Rendering routes: {renderProgress.current} / {renderProgress.total}
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${(renderProgress.current / renderProgress.total) * 100}%` }}
            ></div>
          </div>
          <div className="text-xs text-gray-500 mt-1 text-center">
            {Math.round((renderProgress.current / renderProgress.total) * 100)}%
          </div>
        </div>
      )}

      <div
        ref={mapContainer}
        className="w-full h-full rounded-lg"
        style={{ minHeight: '400px' }}
      />
    </div>
  );
}
