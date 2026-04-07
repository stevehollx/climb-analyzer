'use client';

import { useState, useEffect, useCallback } from 'react';
import { Search, X } from 'lucide-react';

interface ClimbSearchProps {
  onSearchChange: (query: string) => void;
  placeholder?: string;
  className?: string;
}

/**
 * Search component for filtering climbs by name, city, state, or wayId.
 * Matches iOS app search functionality with debounced input.
 */
export function ClimbSearch({
  onSearchChange,
  placeholder = 'Search by name, city, state, or wayId...',
  className = '',
}: ClimbSearchProps) {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');

  // Debounce search input (300ms like iOS)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  // Notify parent of debounced query changes
  useEffect(() => {
    onSearchChange(debouncedQuery);
  }, [debouncedQuery, onSearchChange]);

  const handleClear = useCallback(() => {
    setQuery('');
    setDebouncedQuery('');
    onSearchChange('');
  }, [onSearchChange]);

  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                     text-sm placeholder-gray-400"
        />
        {query && (
          <button
            onClick={handleClear}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-1
                       hover:bg-gray-100 rounded-full transition-colors"
            aria-label="Clear search"
          >
            <X className="h-4 w-4 text-gray-400" />
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * Filter climbs based on search query.
 * Matches iOS search behavior: searches name, city, state, and wayId.
 *
 * @param climbs - Array of climbs to filter
 * @param query - Search query string
 * @returns Filtered array of climbs
 */
export function filterClimbsBySearch<T extends {
  streetName: string;
  city: string;
  state: string;
  wayId: string;
}>(climbs: T[], query: string): T[] {
  if (!query.trim()) {
    return climbs;
  }

  const lowerQuery = query.toLowerCase().trim();

  return climbs.filter((climb) => {
    // Search in street name
    if (climb.streetName?.toLowerCase().includes(lowerQuery)) {
      return true;
    }
    // Search in city
    if (climb.city?.toLowerCase().includes(lowerQuery)) {
      return true;
    }
    // Search in state
    if (climb.state?.toLowerCase().includes(lowerQuery)) {
      return true;
    }
    // Search in wayId
    if (climb.wayId?.toLowerCase().includes(lowerQuery)) {
      return true;
    }
    return false;
  });
}
