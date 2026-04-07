'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { X, ChevronUp, ChevronDown, ExternalLink, Mountain, MapPin, TrendingUp, Ruler, ArrowUp, Link2 } from 'lucide-react';
import { Climb } from '@/types/climb';
import { getCategoryColor } from '@/lib/csv-parser';
import { ElevationProfile } from '@/components/ElevationProfile';

type DrawerState = 'collapsed' | 'half' | 'full';

interface ClimbDetailDrawerProps {
  climb: Climb | null;
  allClimbs?: Climb[];
  onClose: () => void;
  onClimbSelect?: (climb: Climb) => void;
}

export function ClimbDetailDrawer({ climb, allClimbs = [], onClose, onClimbSelect }: ClimbDetailDrawerProps) {
  const [drawerState, setDrawerState] = useState<DrawerState>('half');
  const [isDragging, setIsDragging] = useState(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);
  const drawerRef = useRef<HTMLDivElement>(null);

  // Get connected climbs
  const connectedClimbs = climb?.connectedClimbs
    ? climb.connectedClimbs
        .split(',')
        .map(name => name.trim())
        .filter(name => name && name !== '' && name.toLowerCase() !== 'none')
        .map(name => allClimbs.find(c => c.streetName.trim() === name))
        .filter((c): c is Climb => c !== undefined)
    : [];

  // Get drawer height based on state
  const getDrawerHeight = useCallback((state: DrawerState): string => {
    switch (state) {
      case 'collapsed': return '80px';
      case 'half': return '45vh';
      case 'full': return '85vh';
    }
  }, []);

  // Cycle through drawer states
  const cycleDrawerState = () => {
    setDrawerState(prev => {
      switch (prev) {
        case 'collapsed': return 'half';
        case 'half': return 'full';
        case 'full': return 'collapsed';
      }
    });
  };

  // Handle drag start
  const handleDragStart = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDragging(true);
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    dragStartY.current = clientY;
    if (drawerRef.current) {
      dragStartHeight.current = drawerRef.current.offsetHeight;
    }
    e.preventDefault();
  };

  // Handle drag move
  useEffect(() => {
    if (!isDragging) return;

    const handleMove = (e: MouseEvent | TouchEvent) => {
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
      const delta = dragStartY.current - clientY;
      const newHeight = dragStartHeight.current + delta;
      const windowHeight = window.innerHeight;

      // Determine new state based on height
      if (newHeight < windowHeight * 0.15) {
        setDrawerState('collapsed');
      } else if (newHeight < windowHeight * 0.6) {
        setDrawerState('half');
      } else {
        setDrawerState('full');
      }
    };

    const handleEnd = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleEnd);
    document.addEventListener('touchmove', handleMove);
    document.addEventListener('touchend', handleEnd);

    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleMove);
      document.removeEventListener('touchend', handleEnd);
    };
  }, [isDragging]);

  // Close drawer when climb is null
  if (!climb) {
    return null;
  }

  const categoryColor = getCategoryColor(climb.category);
  const osmUrl = climb.wayId ? `https://www.openstreetmap.org/way/${climb.wayId}` : null;

  return (
    <div
      ref={drawerRef}
      className={`fixed bottom-0 left-0 right-0 bg-white rounded-t-2xl shadow-2xl z-50 transition-all duration-300 ease-out ${
        isDragging ? 'transition-none' : ''
      }`}
      style={{
        height: getDrawerHeight(drawerState),
        maxHeight: '90vh',
      }}
    >
      {/* Drag handle */}
      <div
        className="flex justify-center py-3 cursor-grab active:cursor-grabbing"
        onMouseDown={handleDragStart}
        onTouchStart={handleDragStart}
        onClick={cycleDrawerState}
      >
        <div className="w-12 h-1.5 bg-gray-300 rounded-full" />
      </div>

      {/* Header */}
      <div className="px-4 pb-3 border-b flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-gray-900 truncate">
              {climb.streetName || 'Unnamed Climb'}
            </h2>
            <div
              className="px-2 py-0.5 rounded text-xs font-bold text-white flex-shrink-0"
              style={{ backgroundColor: categoryColor }}
            >
              {climb.category}
            </div>
          </div>
          <p className="text-sm text-gray-600 truncate">
            {[climb.city, climb.state, climb.country].filter(Boolean).join(', ')}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={cycleDrawerState}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            title={drawerState === 'full' ? 'Collapse' : 'Expand'}
          >
            {drawerState === 'full' ? (
              <ChevronDown className="h-5 w-5 text-gray-600" />
            ) : (
              <ChevronUp className="h-5 w-5 text-gray-600" />
            )}
          </button>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            title="Close"
          >
            <X className="h-5 w-5 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Scrollable content */}
      <div
        className={`overflow-y-auto px-4 py-3 ${
          drawerState === 'collapsed' ? 'hidden' : ''
        }`}
        style={{
          height: drawerState === 'collapsed' ? 0 : `calc(${getDrawerHeight(drawerState)} - 100px)`,
        }}
      >
        {/* Stats Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <StatCard
            icon={<ArrowUp className="h-4 w-4" />}
            label="Elevation Gain"
            value={`${climb.elevationGain?.toFixed(0) || '—'} ft`}
          />
          <StatCard
            icon={<Ruler className="h-4 w-4" />}
            label="Length"
            value={`${climb.length?.toFixed(2) || '—'} mi`}
          />
          <StatCard
            icon={<TrendingUp className="h-4 w-4" />}
            label="Avg Grade"
            value={`${climb.avgGrade?.toFixed(1) || '—'}%`}
          />
          <StatCard
            icon={<TrendingUp className="h-4 w-4 text-red-500" />}
            label="Max Grade"
            value={`${climb.maxGrade?.toFixed(1) || '—'}%`}
          />
        </div>

        {/* Additional Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <StatCard
            icon={<Mountain className="h-4 w-4" />}
            label="Height"
            value={`${climb.height?.toFixed(0) || '—'} ft`}
          />
          <StatCard
            icon={<Mountain className="h-4 w-4" />}
            label="Prominence"
            value={`${climb.prominence?.toFixed(0) || '—'} ft`}
          />
          <StatCard
            icon={<MapPin className="h-4 w-4" />}
            label="Coordinates"
            value={`${climb.lat?.toFixed(4)}, ${climb.lon?.toFixed(4)}`}
            small
          />
          <StatCard
            icon={<Link2 className="h-4 w-4" />}
            label="Way ID"
            value={climb.wayId || '—'}
            small
          />
        </div>

        {/* Scores */}
        <div className="bg-gray-50 rounded-lg p-3 mb-4">
          <h3 className="font-semibold text-gray-900 mb-2">Scores</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {climb.basicScore?.toFixed(1) || '—'}
              </div>
              <div className="text-xs text-gray-500">Basic</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {climb.fietsScore?.toFixed(1) || '—'}
              </div>
              <div className="text-xs text-gray-500">FIETS</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {climb.pdiScore?.toFixed(1) || '—'}
              </div>
              <div className="text-xs text-gray-500">PDI</div>
            </div>
          </div>
        </div>

        {/* Road Info */}
        {(climb.surface || climb.highwayType || climb.cyclingAccess || climb.tracktype) && (
          <div className="bg-gray-50 rounded-lg p-3 mb-4">
            <h3 className="font-semibold text-gray-900 mb-2">Road Details</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
              {climb.surface && (
                <div>
                  <div className="text-gray-500">Surface</div>
                  <div className="font-medium capitalize">{climb.surface}</div>
                </div>
              )}
              {climb.highwayType && (
                <div>
                  <div className="text-gray-500">Highway Type</div>
                  <div className="font-medium capitalize">{climb.highwayType}</div>
                </div>
              )}
              {climb.tracktype && (
                <div>
                  <div className="text-gray-500">Track Type</div>
                  <div className="font-medium">{climb.tracktype}</div>
                </div>
              )}
              {climb.cyclingAccess && (
                <div>
                  <div className="text-gray-500">Cycling Access</div>
                  <div className="font-medium capitalize">{climb.cyclingAccess}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Connected Climbs */}
        {connectedClimbs.length > 0 && (
          <div className="bg-gray-50 rounded-lg p-3 mb-4">
            <h3 className="font-semibold text-gray-900 mb-2">
              Connected Climbs ({connectedClimbs.length})
            </h3>
            <div className="space-y-2">
              {connectedClimbs.map((connected, idx) => (
                <button
                  key={idx}
                  onClick={() => onClimbSelect?.(connected)}
                  className="w-full flex items-center justify-between p-2 bg-white rounded-lg hover:bg-blue-50 transition-colors text-left"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getCategoryColor(connected.category) }}
                    />
                    <span className="font-medium text-sm">{connected.streetName}</span>
                  </div>
                  <span className="text-xs text-gray-500">
                    {connected.elevationGain?.toFixed(0)} ft • {connected.avgGrade?.toFixed(1)}%
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Elevation Profile */}
        {drawerState === 'full' && (
          <div className="mb-4">
            <h3 className="font-semibold text-gray-900 mb-2">Elevation Profile</h3>
            <div className="bg-gray-50 rounded-lg overflow-hidden">
              <ElevationProfile climb={climb} onClose={() => {}} hideHeader />
            </div>
          </div>
        )}

        {/* All Way IDs (if multiple) */}
        {climb.allWayIds && climb.allWayIds.includes(',') && (
          <div className="bg-gray-50 rounded-lg p-3 mb-4">
            <h3 className="font-semibold text-gray-900 mb-2">All Way IDs</h3>
            <div className="text-xs text-gray-600 font-mono break-all">
              {climb.allWayIds}
            </div>
          </div>
        )}

        {/* OSM Link */}
        {osmUrl && (
          <a
            href={osmUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            <ExternalLink className="h-4 w-4" />
            View on OpenStreetMap
          </a>
        )}
      </div>
    </div>
  );
}

// Stat card component
function StatCard({
  icon,
  label,
  value,
  small = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  small?: boolean;
}) {
  return (
    <div className="bg-gray-50 rounded-lg p-2">
      <div className="flex items-center gap-1 text-gray-500 mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <div className={`font-semibold text-gray-900 ${small ? 'text-xs' : 'text-sm'}`}>
        {value}
      </div>
    </div>
  );
}
