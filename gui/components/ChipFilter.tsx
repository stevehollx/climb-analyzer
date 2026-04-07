'use client';

import { useState, useCallback } from 'react';

interface ChipOption {
  value: string;
  label: string;
  color?: string;
}

interface ChipFilterProps {
  label: string;
  options: ChipOption[];
  selected: Set<string>;
  onChange: (selected: Set<string>) => void;
  allowMultiple?: boolean;
  showAll?: boolean;
  className?: string;
}

/**
 * Chip-style multi-select filter component matching iOS design.
 * Supports toggle selection with optional "All" chip.
 */
export function ChipFilter({
  label,
  options,
  selected,
  onChange,
  allowMultiple = true,
  showAll = true,
  className = '',
}: ChipFilterProps) {
  const isAllSelected = selected.has('all') || selected.size === 0;

  const handleChipClick = useCallback(
    (value: string) => {
      const newSelected = new Set(selected);

      if (value === 'all') {
        // Clicking "All" clears other selections
        onChange(new Set(['all']));
        return;
      }

      // Remove 'all' if it was selected
      newSelected.delete('all');

      if (allowMultiple) {
        // Toggle the chip
        if (newSelected.has(value)) {
          newSelected.delete(value);
          // If nothing selected, revert to 'all'
          if (newSelected.size === 0) {
            onChange(new Set(['all']));
            return;
          }
        } else {
          newSelected.add(value);
        }
      } else {
        // Single select mode
        if (newSelected.has(value)) {
          onChange(new Set(['all']));
          return;
        } else {
          onChange(new Set([value]));
          return;
        }
      }

      onChange(newSelected);
    },
    [selected, onChange, allowMultiple]
  );

  return (
    <div className={className}>
      <label className="text-sm font-medium text-gray-700 block mb-2">
        {label}
      </label>
      <div className="flex flex-wrap gap-1.5">
        {showAll && (
          <Chip
            label="All"
            isSelected={isAllSelected}
            onClick={() => handleChipClick('all')}
          />
        )}
        {options.map((option) => (
          <Chip
            key={option.value}
            label={option.label}
            color={option.color}
            isSelected={selected.has(option.value) && !isAllSelected}
            onClick={() => handleChipClick(option.value)}
          />
        ))}
      </div>
    </div>
  );
}

interface ChipProps {
  label: string;
  isSelected: boolean;
  color?: string;
  onClick: () => void;
}

function Chip({ label, isSelected, color, onClick }: ChipProps) {
  return (
    <button
      onClick={onClick}
      className={`
        px-3 py-1.5 rounded-full text-xs font-medium
        transition-all duration-150 ease-in-out
        border
        ${
          isSelected
            ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
            : 'bg-white text-gray-700 border-gray-300 hover:border-gray-400 hover:bg-gray-50'
        }
      `}
      style={
        isSelected && color
          ? { backgroundColor: color, borderColor: color }
          : undefined
      }
    >
      {label}
    </button>
  );
}

// Pre-defined filter options matching iOS app

export const SURFACE_OPTIONS: ChipOption[] = [
  { value: 'paved', label: 'Paved' },
  { value: 'asphalt', label: 'Asphalt' },
  { value: 'concrete', label: 'Concrete' },
  { value: 'gravel', label: 'Gravel' },
  { value: 'dirt', label: 'Dirt' },
  { value: 'unpaved', label: 'Unpaved' },
  { value: 'ground', label: 'Ground' },
];

export const HIGHWAY_TYPE_OPTIONS: ChipOption[] = [
  { value: 'primary', label: 'Primary' },
  { value: 'secondary', label: 'Secondary' },
  { value: 'tertiary', label: 'Tertiary' },
  { value: 'residential', label: 'Residential' },
  { value: 'unclassified', label: 'Unclassified' },
  { value: 'track', label: 'Track' },
  { value: 'path', label: 'Path' },
  { value: 'cycleway', label: 'Cycleway' },
  { value: 'footway', label: 'Footway' },
];

export const CYCLING_ACCESS_OPTIONS: ChipOption[] = [
  { value: 'yes', label: 'Yes' },
  { value: 'designated', label: 'Designated' },
  { value: 'permissive', label: 'Permissive' },
  { value: 'destination', label: 'Destination' },
  { value: 'no', label: 'No' },
];

export const TRACKTYPE_OPTIONS: ChipOption[] = [
  { value: 'grade1', label: 'Grade 1 (paved)' },
  { value: 'grade2', label: 'Grade 2 (compacted)' },
  { value: 'grade3', label: 'Grade 3 (soft)' },
  { value: 'grade4', label: 'Grade 4 (rough)' },
  { value: 'grade5', label: 'Grade 5 (very rough)' },
];
