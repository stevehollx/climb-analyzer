'use client';

import { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Cloud, Download, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface Region {
  name: string;
  path: string;
  fileCount: number;
  children?: Region[];
}

interface CloudCacheTreeProps {
  showDownload?: boolean;
  onDownload?: (regionPath: string, regionName: string) => void;
}

export default function CloudCacheTree({ showDownload = false, onDownload }: CloudCacheTreeProps) {
  const [regions, setRegions] = useState<Region[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());
  const [downloadingPaths, setDownloadingPaths] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchCloudCacheRegions();
  }, []);

  const fetchCloudCacheRegions = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/cloud-cache?action=list');
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else if (data.regions && data.regions.length > 0) {
        setRegions(data.regions);
        setError(null);
      } else {
        setError('No cloud cache data available');
      }
    } catch (err) {
      setError('Failed to load cloud cache data');
      console.error('Error fetching cloud cache:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleExpanded = (path: string) => {
    const newExpanded = new Set(expandedPaths);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedPaths(newExpanded);
  };

  const handleDownload = async (regionPath: string, regionName: string) => {
    if (onDownload) {
      setDownloadingPaths(new Set(downloadingPaths).add(regionPath));
      try {
        await onDownload(regionPath, regionName);
      } finally {
        const newDownloading = new Set(downloadingPaths);
        newDownloading.delete(regionPath);
        setDownloadingPaths(newDownloading);
      }
    }
  };

  const renderRegion = (region: Region, level: number = 0) => {
    const hasChildren = region.children && region.children.length > 0;
    const isExpanded = expandedPaths.has(region.path);
    const isDownloading = downloadingPaths.has(region.path);
    const paddingLeft = level * 24;

    return (
      <div key={region.path} className="border-l-2 border-gray-200">
        <div
          className={`flex items-center justify-between py-2 px-3 hover:bg-gray-50 ${
            level === 0 ? 'bg-gray-100 font-semibold' : ''
          }`}
          style={{ paddingLeft: `${paddingLeft + 12}px` }}
        >
          <div className="flex items-center flex-1">
            {hasChildren ? (
              <button
                onClick={() => toggleExpanded(region.path)}
                className="mr-2 focus:outline-none"
              >
                {isExpanded ? (
                  <ChevronDown className="h-4 w-4 text-gray-600" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-gray-600" />
                )}
              </button>
            ) : (
              <div className="w-6" />
            )}
            <Cloud className="h-4 w-4 mr-2 text-blue-500" />
            <span className="text-sm">
              {region.name} <span className="text-gray-500">({region.fileCount})</span>
            </span>
          </div>

          {showDownload && region.fileCount > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleDownload(region.path, region.name)}
              disabled={isDownloading}
              className="ml-2"
            >
              {isDownloading ? (
                <>
                  <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                  Downloading...
                </>
              ) : (
                <>
                  <Download className="mr-2 h-3 w-3" />
                  Download
                </>
              )}
            </Button>
          )}
        </div>

        {hasChildren && isExpanded && (
          <div>
            {region.children!.map((child) => renderRegion(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center">
          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
          <span>Loading cloud cache data...</span>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="text-center text-gray-500">
          <Cloud className="h-12 w-12 mx-auto mb-3 text-gray-300" />
          <p>{error}</p>
        </div>
      </Card>
    );
  }

  if (regions.length === 0) {
    return (
      <Card className="p-6">
        <div className="text-center text-gray-500">
          <Cloud className="h-12 w-12 mx-auto mb-3 text-gray-300" />
          <p>No cloud cache data available</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      <div className="space-y-1">
        {regions.map((region) => renderRegion(region, 0))}
      </div>
    </Card>
  );
}
