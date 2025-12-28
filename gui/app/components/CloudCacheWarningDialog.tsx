'use client';

import { useState, useEffect } from 'react';
import { Cloud, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface CloudCacheWarningDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onUseCache: () => void;
  onContinueAnalysis: () => void;
  region: string;
}

interface CacheMatch {
  exists: boolean;
  fileCount: number;
  regionPath: string;
  regionName: string;
}

export default function CloudCacheWarningDialog({
  isOpen,
  onClose,
  onUseCache,
  onContinueAnalysis,
  region,
}: CloudCacheWarningDialogProps) {
  const [checking, setChecking] = useState(true);
  const [cacheMatch, setCacheMatch] = useState<CacheMatch | null>(null);

  useEffect(() => {
    if (isOpen && region) {
      checkForCloudCache();
    }
  }, [isOpen, region]);

  const checkForCloudCache = async () => {
    try {
      setChecking(true);

      // Fetch cloud cache regions
      const response = await fetch('/api/cloud-cache?action=list');
      if (!response.ok) {
        setCacheMatch(null);
        setChecking(false);
        return;
      }

      const data = await response.json();
      if (!data.regions || data.regions.length === 0) {
        setCacheMatch(null);
        setChecking(false);
        return;
      }

      // Check if our region matches any cloud cache data
      const normalizedRegion = region.toLowerCase().trim();
      const match = findMatchingRegion(data.regions, normalizedRegion);

      if (match) {
        setCacheMatch({
          exists: true,
          fileCount: match.fileCount,
          regionPath: match.path,
          regionName: match.name,
        });
      } else {
        setCacheMatch(null);
      }

      setChecking(false);
    } catch (error) {
      console.error('Failed to check cloud cache:', error);
      setCacheMatch(null);
      setChecking(false);
    }
  };

  const findMatchingRegion = (regions: any[], searchRegion: string): any | null => {
    for (const continent of regions) {
      // Check continent name
      if (continent.name.toLowerCase().includes(searchRegion) ||
          searchRegion.includes(continent.name.toLowerCase())) {
        if (continent.fileCount > 0) return continent;
      }

      // Check countries
      if (continent.children) {
        for (const country of continent.children) {
          if (country.name.toLowerCase().includes(searchRegion) ||
              searchRegion.includes(country.name.toLowerCase()) ||
              country.path.toLowerCase().includes(searchRegion)) {
            if (country.fileCount > 0) return country;
          }

          // Check states/subregions
          if (country.children) {
            for (const state of country.children) {
              if (state.name.toLowerCase().includes(searchRegion) ||
                  searchRegion.includes(state.name.toLowerCase()) ||
                  state.path.toLowerCase().includes(searchRegion)) {
                if (state.fileCount > 0) return state;
              }
            }
          }
        }
      }
    }
    return null;
  };

  // If checking or no match found, automatically proceed
  useEffect(() => {
    if (!checking && !cacheMatch && isOpen) {
      onContinueAnalysis();
    }
  }, [checking, cacheMatch, isOpen]);

  // Don't show dialog if no cache match found
  if (!cacheMatch || checking || !isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <Card className="max-w-lg w-full mx-4 p-6 space-y-4">
        <div className="flex items-center gap-2">
          <Cloud className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-bold">Cloud Cache Match Found!</h2>
        </div>

        <div className="space-y-3">
          <p className="text-sm text-gray-700">
            We found pre-analyzed climb data for <strong>{cacheMatch.regionName}</strong> in the cloud cache:
          </p>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-sm text-blue-900">
              <strong>{cacheMatch.fileCount}</strong> analysis file(s) available
            </p>
            <p className="text-xs text-blue-700 mt-1">
              Path: {cacheMatch.regionPath}
            </p>
          </div>

          <p className="text-sm text-gray-700">
            You can download the pre-analyzed data instantly instead of running a new analysis.
            This saves time and computational resources.
          </p>

          <p className="text-sm font-medium text-gray-900">
            What would you like to do?
          </p>
        </div>

        <div className="flex gap-3 justify-end pt-4">
          <Button variant="outline" onClick={onContinueAnalysis}>
            Run New Analysis
          </Button>
          <Button onClick={onUseCache} className="bg-blue-600 hover:bg-blue-700">
            <Cloud className="mr-2 h-4 w-4" />
            Use Cloud Cache
          </Button>
        </div>
      </Card>
    </div>
  );
}
