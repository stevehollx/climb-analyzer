'use client';

import { useEffect, useRef, useState } from 'react';
import { Climb } from '@/types/climb';

interface ElevationProfileProps {
  climb: Climb;
  onClose: () => void;
  hideHeader?: boolean;
  units?: 'imperial' | 'metric';  // Optional units override
}

interface ProfilePoint {
  distance: number;  // in miles (imperial) or km (metric)
  elevation: number; // in feet (imperial) or meters (metric)
  grade: number;     // percentage
}

interface TooltipData {
  x: number;
  y: number;
  distance: number;
  elevation: number;
  grade: number;
}

// Grade color mapping based on cycling grade standards
const getGradeColor = (grade: number): string => {
  if (grade < 0) return '#808080';             // Descents are gray
  const absGrade = Math.abs(grade);
  if (absGrade <= 2) return '#00b050';        // 0-2% green
  if (absGrade <= 5) return '#92d050';        // 3-5% yellow-green
  if (absGrade <= 8) return '#ffc000';        // 6-8% yellow-orange
  if (absGrade <= 11) return '#ff8000';       // 9-11% orange
  if (absGrade <= 15) return '#ff0000';       // 12-15% red
  if (absGrade <= 20) return '#8B0000';       // 16-20% dark red/maroon
  return '#7030a0';                            // >20% purple
};

const getGradeLabel = (grade: number): string => {
  const absGrade = Math.abs(grade);
  if (absGrade <= 2) return 'Gentle';
  if (absGrade <= 5) return 'Moderate';
  if (absGrade <= 8) return 'Tough';
  if (absGrade <= 11) return 'Hard';
  if (absGrade <= 15) return 'Very Steep';
  if (absGrade <= 20) return 'Brutal';
  return 'Extreme';
};

/**
 * LTTB (Largest-Triangle-Three-Buckets) downsampling algorithm
 * Preserves visual fidelity while reducing point count
 * Also preserves steep sections (>10% grade) and peaks/valleys
 */
function lttbDownsample(data: ProfilePoint[], threshold: number): ProfilePoint[] {
  if (data.length <= threshold) return data;

  const sampled: ProfilePoint[] = [];

  // Always keep first and last points
  sampled.push(data[0]);

  // Identify critical points to always preserve
  const criticalIndices = new Set<number>();
  criticalIndices.add(0);
  criticalIndices.add(data.length - 1);

  // Preserve steep sections (>10% grade)
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i].grade) > 10) {
      criticalIndices.add(i);
    }
  }

  // Preserve peaks and valleys
  for (let i = 1; i < data.length - 1; i++) {
    const prev = data[i - 1].elevation;
    const curr = data[i].elevation;
    const next = data[i + 1].elevation;

    // Local maximum (peak)
    if (curr > prev && curr > next) {
      criticalIndices.add(i);
    }
    // Local minimum (valley)
    if (curr < prev && curr < next) {
      criticalIndices.add(i);
    }
    // Grade sign change (descent boundary)
    if ((data[i - 1].grade >= 0 && data[i].grade < -1) ||
        (data[i - 1].grade < -1 && data[i].grade >= 0)) {
      criticalIndices.add(i);
    }
  }

  // Calculate remaining slots after critical points
  const criticalArray = Array.from(criticalIndices).sort((a, b) => a - b);
  const remainingSlots = threshold - criticalArray.length;

  if (remainingSlots <= 0) {
    // Just return critical points
    return criticalArray.map(i => data[i]);
  }

  // LTTB for non-critical points
  const bucketSize = (data.length - 2) / (remainingSlots + 1);
  let selectedIndex = 0;

  for (let i = 0; i < remainingSlots; i++) {
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.floor((i + 2) * bucketSize) + 1;

    // Calculate average point of next bucket (for triangle calculation)
    let avgX = 0, avgY = 0;
    const nextBucketStart = Math.min(bucketEnd, data.length - 1);
    const nextBucketEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, data.length);
    const nextBucketSize = nextBucketEnd - nextBucketStart;

    if (nextBucketSize > 0) {
      for (let j = nextBucketStart; j < nextBucketEnd; j++) {
        avgX += data[j].distance;
        avgY += data[j].elevation;
      }
      avgX /= nextBucketSize;
      avgY /= nextBucketSize;
    }

    // Find point that creates largest triangle
    let maxArea = -1;
    let maxIndex = bucketStart;

    for (let j = bucketStart; j < bucketEnd && j < data.length - 1; j++) {
      // Skip if already a critical point
      if (criticalIndices.has(j)) continue;

      // Calculate triangle area
      const area = Math.abs(
        (data[selectedIndex].distance - avgX) * (data[j].elevation - data[selectedIndex].elevation) -
        (data[selectedIndex].distance - data[j].distance) * (avgY - data[selectedIndex].elevation)
      );

      if (area > maxArea) {
        maxArea = area;
        maxIndex = j;
      }
    }

    if (!criticalIndices.has(maxIndex)) {
      criticalIndices.add(maxIndex);
    }
    selectedIndex = maxIndex;
  }

  // Combine and sort all selected indices
  const allIndices = Array.from(criticalIndices).sort((a, b) => a - b);
  return allIndices.map(i => data[i]);
}

/**
 * Parse, process, and prepare elevation profile data
 * - Converts units from source format (meters for distance, configurable for elevation)
 * - Detects and reverses descending profiles
 * - Recalculates grades from actual elevation/distance data
 * - Applies LTTB downsampling for performance
 */
function processElevationProfile(
  profileString: string,
  units: 'imperial' | 'metric'
): ProfilePoint[] {
  if (!profileString) return [];

  const segments = profileString.split('|');
  if (segments.length < 2) return [];

  // Parse raw data - source format is: dist(meters),elevation(feet or meters),grade
  // Distance is ALWAYS in meters from Python source
  // Elevation is in feet if imperial, meters if metric (based on --units flag during analysis)
  let rawData = segments.map(segment => {
    const parts = segment.split(',').map(parseFloat);
    if (parts.length !== 3 || parts.some(isNaN)) return null;

    const distMeters = parts[0];
    const elevation = parts[1];  // Already in correct units from source

    return {
      distMeters,
      elevation,
      grade: 0  // Will be recalculated
    };
  }).filter((p): p is { distMeters: number; elevation: number; grade: number } => p !== null);

  if (rawData.length < 2) return [];

  // Detect descending profile (first elevation > last elevation)
  // If so, reverse to show ascending from left to right
  const firstEle = rawData[0].elevation;
  const lastEle = rawData[rawData.length - 1].elevation;

  if (firstEle > lastEle) {
    // Reverse the profile
    rawData = rawData.reverse();

    // Recalculate distances from start
    const maxDist = rawData[rawData.length - 1].distMeters;
    rawData = rawData.map(p => ({
      ...p,
      distMeters: maxDist - p.distMeters
    }));

    // Sort by distance (should already be sorted after reversal, but ensure it)
    rawData.sort((a, b) => a.distMeters - b.distMeters);
  }

  // Recalculate grades from actual elevation/distance data
  // This is defensive - handles any inaccuracies in source data
  for (let i = 0; i < rawData.length; i++) {
    if (i + 1 < rawData.length) {
      const elevDelta = rawData[i + 1].elevation - rawData[i].elevation;
      const distDelta = rawData[i + 1].distMeters - rawData[i].distMeters;

      // Convert elevation delta to meters if in feet for grade calculation
      const elevDeltaMeters = units === 'imperial' ? elevDelta / 3.28084 : elevDelta;

      if (distDelta > 0) {
        rawData[i].grade = (elevDeltaMeters / distDelta) * 100;
        // Cap at reasonable max (roads rarely exceed 35%)
        if (rawData[i].grade > 50) rawData[i].grade = 50;
        if (rawData[i].grade < -50) rawData[i].grade = -50;
      }
    } else {
      rawData[i].grade = 0;  // Last point
    }
  }

  // Convert to display units
  const points: ProfilePoint[] = rawData.map(p => ({
    distance: units === 'imperial' ? p.distMeters / 1609.34 : p.distMeters / 1000,  // miles or km
    elevation: p.elevation,  // Already in correct units
    grade: p.grade
  }));

  // Apply LTTB downsampling if too many points (limit to ~50 for smooth rendering)
  const MAX_POINTS = 50;
  if (points.length > MAX_POINTS) {
    return lttbDownsample(points, MAX_POINTS);
  }

  return points;
}

export function ElevationProfile({ climb, onClose, hideHeader = false, units = 'imperial' }: ElevationProfileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [points, setPoints] = useState<ProfilePoint[]>([]);
  const [scales, setScales] = useState<{
    xScale: (dist: number) => number;
    yScale: (ele: number) => number;
    padding: { top: number; right: number; bottom: number; left: number };
  } | null>(null);

  // Unit labels
  const distanceUnit = units === 'imperial' ? 'mi' : 'km';
  const elevationUnit = units === 'imperial' ? 'ft' : 'm';

  useEffect(() => {
    if (!canvasRef.current || !climb.elevationProfile) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Process elevation profile with all improvements
    const processedPoints = processElevationProfile(climb.elevationProfile, units);

    if (processedPoints.length === 0) return;

    // Get actual display size and device pixel ratio for crisp rendering
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Set canvas internal size (accounting for pixel ratio)
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Scale context to match device pixel ratio
    ctx.scale(dpr, dpr);

    // Use display size for calculations
    const width = rect.width;
    const height = rect.height;
    const padding = { top: 40, right: 40, bottom: 30, left: 80 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Find data ranges
    const maxDist = Math.max(...processedPoints.map(p => p.distance));
    const minEle = Math.min(...processedPoints.map(p => p.elevation));
    const maxEle = Math.max(...processedPoints.map(p => p.elevation));
    const eleRange = maxEle - minEle || 1;  // Avoid division by zero

    // Scale functions
    const xScale = (dist: number) => padding.left + (dist / maxDist) * chartWidth;
    const yScale = (ele: number) => padding.top + chartHeight - ((ele - minEle) / eleRange) * chartHeight;

    // Save points and scales for mouse interaction
    setPoints(processedPoints);
    setScales({ xScale, yScale, padding });

    // Draw grid lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;

    // Horizontal grid lines (elevation)
    const numHGridLines = 5;
    for (let i = 0; i <= numHGridLines; i++) {
      const ele = minEle + (eleRange * i / numHGridLines);
      const y = yScale(ele);

      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${Math.round(ele)} ${elevationUnit}`, padding.left - 10, y);
    }

    // Vertical grid lines (distance)
    const numVGridLines = 5;
    for (let i = 0; i <= numVGridLines; i++) {
      const dist = maxDist * i / numVGridLines;
      const x = xScale(dist);

      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();

      // X-axis labels
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${dist.toFixed(1)} ${distanceUnit}`, x, padding.top + chartHeight + 10);
    }

    // Draw elevation profile with color-coded filled areas
    const baselineY = padding.top + chartHeight;

    for (let i = 0; i < processedPoints.length - 1; i++) {
      const p1 = processedPoints[i];
      const p2 = processedPoints[i + 1];

      // Use the grade of the segment for coloring
      const avgGrade = (p1.grade + p2.grade) / 2;
      const color = getGradeColor(avgGrade);

      // Fill the area under this segment
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.6;  // Semi-transparent fill
      ctx.beginPath();
      ctx.moveTo(xScale(p1.distance), yScale(p1.elevation));
      ctx.lineTo(xScale(p2.distance), yScale(p2.elevation));
      ctx.lineTo(xScale(p2.distance), baselineY);
      ctx.lineTo(xScale(p1.distance), baselineY);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1.0;  // Reset alpha

      // Draw the outline stroke
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xScale(p1.distance), yScale(p1.elevation));
      ctx.lineTo(xScale(p2.distance), yScale(p2.elevation));
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#111827';
    ctx.font = 'bold 14px sans-serif';

    // Y-axis label
    ctx.save();
    ctx.translate(15, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(`Elevation (${elevationUnit})`, 0, 0);
    ctx.restore();

    // Title
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(`${climb.streetName} - Elevation Profile`, width / 2, 20);

  }, [climb, units]);

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !points.length || !scales) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    // Get mouse position relative to canvas
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Check if mouse is within chart area
    const { xScale, yScale, padding } = scales;
    const chartWidth = rect.width - padding.left - padding.right;
    const chartHeight = rect.height - padding.top - padding.bottom;

    if (mouseX < padding.left || mouseX > padding.left + chartWidth ||
        mouseY < padding.top || mouseY > padding.top + chartHeight) {
      setTooltip(null);
      return;
    }

    // Find the closest point based on x-position
    const maxDist = points[points.length - 1].distance;
    const chartX = mouseX - padding.left;
    const mouseDistance = (chartX / chartWidth) * maxDist;

    // Find nearest point
    let nearestIndex = 0;
    let minDistDiff = Math.abs(points[0].distance - mouseDistance);

    for (let i = 1; i < points.length; i++) {
      const distDiff = Math.abs(points[i].distance - mouseDistance);
      if (distDiff < minDistDiff) {
        minDistDiff = distDiff;
        nearestIndex = i;
      }
    }

    const point = points[nearestIndex];

    // Calculate tooltip position (offset slightly to avoid cursor overlap)
    const tooltipX = xScale(point.distance) + 15;
    const tooltipY = yScale(point.elevation) - 10;

    setTooltip({
      x: tooltipX,
      y: tooltipY,
      distance: point.distance,
      elevation: point.elevation,
      grade: point.grade
    });
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  // Unit labels for tooltip
  const distanceUnitLabel = units === 'imperial' ? 'mi' : 'km';
  const elevationUnitLabel = units === 'imperial' ? 'ft' : 'm';

  if (!climb.elevationProfile) {
    return null;
  }

  return (
    <div className="w-full">
      {/* Elevation Profile Chart */}
      <div className="w-full bg-white border-t-2 border-gray-300 shadow-2xl pb-4" style={{ height: '24vh' }}>
        <div className="relative w-full h-full pt-4 px-4 pb-0">
          {/* Close button */}
          {!hideHeader && (
            <button
              onClick={onClose}
              className="absolute top-2 right-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-full w-8 h-8 flex items-center justify-center font-bold z-10"
            >
              ×
            </button>
          )}

          {/* Canvas for chart */}
          <canvas
            ref={canvasRef}
            className="w-full h-full cursor-crosshair"
            style={{ display: 'block' }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />

          {/* Tooltip */}
          {tooltip && (
            <div
              className="absolute bg-black/90 text-white px-3 py-2 rounded-md text-sm pointer-events-none shadow-lg z-20"
              style={{
                left: `${tooltip.x}px`,
                top: `${tooltip.y}px`,
                transform: 'translate(-50%, -100%)'
              }}
            >
              <div className="font-semibold mb-1">Climb Data</div>
              <div className="space-y-0.5">
                <div>Distance: <span className="font-mono">{tooltip.distance.toFixed(2)} {distanceUnitLabel}</span></div>
                <div>Elevation: <span className="font-mono">{Math.round(tooltip.elevation)} {elevationUnitLabel}</span></div>
                <div>Grade: <span className="font-mono" style={{ color: getGradeColor(tooltip.grade) }}>{tooltip.grade.toFixed(1)}%</span></div>
              </div>
            </div>
          )}
        </div>

        {/* X-axis label as HTML text */}
        <div className="text-center text-sm font-bold text-gray-900 py-2 pb-4 bg-white">
          Distance ({distanceUnitLabel})
        </div>
      </div>

      {/* Grade Legend - Separate Panel Below */}
      <div className="w-full bg-white border-t border-gray-200 shadow-md p-3 mt-4">
        <div className="flex items-center justify-center gap-4 flex-wrap text-xs">
          <span className="font-semibold">Grade Legend</span>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#808080' }}></div>
            <span>Descent</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#00b050' }}></div>
            <span>0-2%</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#92d050' }}></div>
            <span>3-5%</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#ffc000' }}></div>
            <span>6-8%</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#ff8000' }}></div>
            <span>9-11%</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#ff0000' }}></div>
            <span>12-15%</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#8B0000' }}></div>
            <span>16-20%</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, backgroundColor: '#7030a0' }}></div>
            <span>&gt;20%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
