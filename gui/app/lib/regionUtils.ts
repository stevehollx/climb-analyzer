/**
 * Utility functions for handling region names and continent-based styling.
 */

/**
 * Convert canonical region path to friendly name.
 * @param canonicalPath - Path like "europe/monaco" or "north-america/usa/california"
 * @returns Friendly name like "Monaco" or "California"
 */
export function getFriendlyName(canonicalPath: string): string {
  if (!canonicalPath) return '';

  // Split by / and get the last part
  const parts = canonicalPath.split('/');
  const lastPart = parts[parts.length - 1];

  // Convert from kebab-case to Title Case
  return lastPart
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Extract continent from canonical path.
 * @param canonicalPath - Path like "europe/monaco" or "north-america/usa/california"
 * @returns Continent name like "europe" or "north-america"
 */
export function getContinent(canonicalPath: string): string {
  if (!canonicalPath) return 'unknown';

  const parts = canonicalPath.split('/');
  return parts[0];
}

/**
 * Continent color mapping for consistent color coding across the app.
 */
const CONTINENT_COLORS = {
  'africa': {
    bg: 'bg-yellow-100',
    text: 'text-yellow-800',
    border: 'border-yellow-200'
  },
  'asia': {
    bg: 'bg-red-100',
    text: 'text-red-800',
    border: 'border-red-200'
  },
  'europe': {
    bg: 'bg-blue-100',
    text: 'text-blue-800',
    border: 'border-blue-200'
  },
  'north-america': {
    bg: 'bg-green-100',
    text: 'text-green-800',
    border: 'border-green-200'
  },
  'south-america': {
    bg: 'bg-purple-100',
    text: 'text-purple-800',
    border: 'border-purple-200'
  },
  'oceania': {
    bg: 'bg-teal-100',
    text: 'text-teal-800',
    border: 'border-teal-200'
  },
  'antarctica': {
    bg: 'bg-cyan-100',
    text: 'text-cyan-800',
    border: 'border-cyan-200'
  },
  'unknown': {
    bg: 'bg-gray-100',
    text: 'text-gray-800',
    border: 'border-gray-200'
  }
};

/**
 * Get Tailwind CSS classes for continent-based color coding.
 * @param canonicalPath - Path like "europe/monaco"
 * @returns Object with bg, text, and border classes
 */
export function getContinentColors(canonicalPath: string) {
  const continent = getContinent(canonicalPath);
  return CONTINENT_COLORS[continent as keyof typeof CONTINENT_COLORS] || CONTINENT_COLORS.unknown;
}

/**
 * Get friendly continent name.
 * @param continent - Continent slug like "north-america"
 * @returns Friendly name like "North America"
 */
export function getFriendlyContinentName(continent: string): string {
  return continent
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
