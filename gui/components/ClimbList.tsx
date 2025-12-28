'use client';

import { useState, useMemo, useCallback, useRef, useEffect, ReactElement, CSSProperties } from 'react';
import { List } from 'react-window';
import { Search, ChevronUp, ChevronDown, ArrowUpDown } from 'lucide-react';
import { Climb } from '@/types/climb';
import { getCategoryColor } from '@/lib/csv-parser';

type SortField = 'pdiScore' | 'basicScore' | 'fietsScore' | 'elevationGain' | 'prominence' | 'height' | 'length' | 'avgGrade' | 'maxGrade';
type SortDirection = 'asc' | 'desc';

interface ClimbListProps {
  climbs: Climb[];
  scoreType: 'basic' | 'fiets' | 'pdi';
  selectedClimb: Climb | null;
  onClimbClick: (climb: Climb) => void;
}

const SORT_OPTIONS: { field: SortField; label: string; defaultDir: SortDirection }[] = [
  { field: 'pdiScore', label: 'PDI Score', defaultDir: 'desc' },
  { field: 'basicScore', label: 'Basic Score', defaultDir: 'desc' },
  { field: 'fietsScore', label: 'FIETS Score', defaultDir: 'desc' },
  { field: 'elevationGain', label: 'Elevation Gain', defaultDir: 'desc' },
  { field: 'prominence', label: 'Prominence', defaultDir: 'desc' },
  { field: 'height', label: 'Height', defaultDir: 'desc' },
  { field: 'length', label: 'Distance', defaultDir: 'desc' },
  { field: 'avgGrade', label: 'Avg Grade', defaultDir: 'desc' },
  { field: 'maxGrade', label: 'Max Grade', defaultDir: 'desc' },
];

// Map scoreType prop to default sort field
const SCORE_TYPE_TO_FIELD: Record<string, SortField> = {
  'pdi': 'pdiScore',
  'basic': 'basicScore',
  'fiets': 'fietsScore',
};

// Row component props for react-window 2.x
interface ClimbRowProps {
  climb: Climb;
  rank: number;
  isSelected: boolean;
  sortField: SortField;
  onClimbClick: (climb: Climb) => void;
}

// Row component for the list
function ClimbRow({
  ariaAttributes,
  index,
  style,
  climb,
  rank,
  isSelected,
  sortField,
  onClimbClick,
}: {
  ariaAttributes: { 'aria-posinset': number; 'aria-setsize': number; role: 'listitem' };
  index: number;
  style: CSSProperties;
} & ClimbRowProps): ReactElement {
  const categoryColor = getCategoryColor(climb.category);

  // Get the score value to display based on current sort field
  const getScoreValue = (): number => {
    switch (sortField) {
      case 'pdiScore': return climb.pdiScore;
      case 'basicScore': return climb.basicScore;
      case 'fietsScore': return climb.fietsScore;
      default: return climb[sortField] ?? 0;
    }
  };

  const getSortLabel = (): string => {
    return SORT_OPTIONS.find(o => o.field === sortField)?.label.toLowerCase().replace(' score', '') || '';
  };

  return (
    <div
      {...ariaAttributes}
      style={style}
      className={`flex items-center px-4 py-2 border-b border-gray-100 cursor-pointer transition-colors ${
        isSelected ? 'bg-blue-50 border-l-4 border-l-blue-500' : 'hover:bg-gray-50'
      }`}
      onClick={() => onClimbClick(climb)}
    >
      {/* Rank */}
      <div className="w-12 flex-shrink-0 text-sm font-medium text-gray-500">
        #{rank}
      </div>

      {/* Category indicator */}
      <div
        className="w-3 h-3 rounded-full flex-shrink-0 mr-3"
        style={{ backgroundColor: categoryColor }}
        title={climb.category}
      />

      {/* Main info */}
      <div className="flex-1 min-w-0">
        <div className="font-medium text-gray-900 truncate">
          {climb.streetName || 'Unnamed'}
        </div>
        <div className="text-xs text-gray-500 truncate">
          {[climb.city, climb.state].filter(Boolean).join(', ')}
        </div>
      </div>

      {/* Metrics */}
      <div className="flex items-center gap-4 text-sm flex-shrink-0">
        <div className="text-right w-20">
          <div className="font-medium text-gray-900">
            {climb.elevationGain?.toFixed(0) || '—'} ft
          </div>
          <div className="text-xs text-gray-500">gain</div>
        </div>
        <div className="text-right w-16">
          <div className="font-medium text-gray-900">
            {climb.length?.toFixed(1) || '—'} mi
          </div>
          <div className="text-xs text-gray-500">length</div>
        </div>
        <div className="text-right w-16">
          <div className="font-medium text-gray-900">
            {climb.avgGrade?.toFixed(1) || '—'}%
          </div>
          <div className="text-xs text-gray-500">avg</div>
        </div>
        <div className="text-right w-20">
          <div className="font-semibold text-blue-600">
            {getScoreValue().toFixed(1)}
          </div>
          <div className="text-xs text-gray-500">
            {getSortLabel()}
          </div>
        </div>
      </div>
    </div>
  );
}

export function ClimbList({ climbs, scoreType, selectedClimb, onClimbClick }: ClimbListProps) {
  const [sortField, setSortField] = useState<SortField>(SCORE_TYPE_TO_FIELD[scoreType] || 'pdiScore');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const [listHeight, setListHeight] = useState(400);

  // Update sort field when scoreType changes
  useEffect(() => {
    const newSortField = SCORE_TYPE_TO_FIELD[scoreType];
    if (newSortField) {
      setSortField(newSortField);
    }
  }, [scoreType]);

  // Calculate list height based on container
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        // Account for header (search + sort controls)
        const headerHeight = 100;
        const containerHeight = containerRef.current.clientHeight;
        setListHeight(Math.max(200, containerHeight - headerHeight));
      }
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  // Filter climbs by search query
  const filteredClimbs = useMemo(() => {
    if (!searchQuery.trim()) return climbs;

    const query = searchQuery.toLowerCase().trim();

    return climbs.filter(climb => {
      const name = climb.streetName?.toLowerCase() || '';
      const city = climb.city?.toLowerCase() || '';
      const state = climb.state?.toLowerCase() || '';

      return name.includes(query) || city.includes(query) || state.includes(query);
    });
  }, [climbs, searchQuery]);

  // Sort climbs
  const sortedClimbs = useMemo(() => {
    const sorted = [...filteredClimbs].sort((a, b) => {
      const aVal = a[sortField] ?? 0;
      const bVal = b[sortField] ?? 0;
      const diff = aVal - bVal;
      return sortDirection === 'desc' ? -diff : diff;
    });
    return sorted;
  }, [filteredClimbs, sortField, sortDirection]);

  // Handle sort change
  const handleSortChange = (field: SortField) => {
    if (field === sortField) {
      // Toggle direction
      setSortDirection(prev => prev === 'desc' ? 'asc' : 'desc');
    } else {
      // New field - use default direction
      const option = SORT_OPTIONS.find(o => o.field === field);
      setSortField(field);
      setSortDirection(option?.defaultDir || 'desc');
    }
  };

  // Create row component with closure over data
  const rowComponent = useCallback(({
    ariaAttributes,
    index,
    style,
  }: {
    ariaAttributes: { 'aria-posinset': number; 'aria-setsize': number; role: 'listitem' };
    index: number;
    style: CSSProperties;
  }): ReactElement => {
    const climb = sortedClimbs[index];
    const isSelected = selectedClimb?.wayId === climb.wayId;
    const rank = index + 1;

    return (
      <ClimbRow
        ariaAttributes={ariaAttributes}
        index={index}
        style={style}
        climb={climb}
        rank={rank}
        isSelected={isSelected}
        sortField={sortField}
        onClimbClick={onClimbClick}
      />
    );
  }, [sortedClimbs, selectedClimb, sortField, onClimbClick]);

  return (
    <div ref={containerRef} className="h-full flex flex-col bg-white">
      {/* Search and Sort Controls */}
      <div className="p-3 border-b space-y-2">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search by name, city, or state..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-gray-500 flex items-center gap-1">
            <ArrowUpDown className="h-3 w-3" />
            Sort:
          </span>
          <select
            value={sortField}
            onChange={(e) => handleSortChange(e.target.value as SortField)}
            className="text-sm border rounded px-2 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {SORT_OPTIONS.map(option => (
              <option key={option.field} value={option.field}>
                {option.label}
              </option>
            ))}
          </select>
          <button
            onClick={() => setSortDirection(prev => prev === 'desc' ? 'asc' : 'desc')}
            className="flex items-center gap-1 text-sm px-2 py-1 border rounded hover:bg-gray-50"
            title={sortDirection === 'desc' ? 'Sorted descending (highest first)' : 'Sorted ascending (lowest first)'}
          >
            {sortDirection === 'desc' ? (
              <>
                <ChevronDown className="h-4 w-4" />
                <span className="text-xs text-gray-600">High to Low</span>
              </>
            ) : (
              <>
                <ChevronUp className="h-4 w-4" />
                <span className="text-xs text-gray-600">Low to High</span>
              </>
            )}
          </button>
          <span className="text-xs text-gray-400 ml-auto">
            {sortedClimbs.length} climbs
          </span>
        </div>
      </div>

      {/* Virtual List */}
      <div className="flex-1">
        {sortedClimbs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            {searchQuery ? 'No climbs match your search' : 'No climbs to display'}
          </div>
        ) : (
          <List
            rowComponent={rowComponent}
            rowCount={sortedClimbs.length}
            rowHeight={64}
            defaultHeight={listHeight}
            overscanCount={5}
            rowProps={{}}
            style={{ height: listHeight }}
          />
        )}
      </div>
    </div>
  );
}
