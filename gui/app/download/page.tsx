'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Download, Loader2, CheckCircle2, XCircle, ChevronRight, ChevronDown, Cloud } from 'lucide-react';
import CloudCacheTree from '../components/CloudCacheTree';
import AvailableDataCard from '../components/AvailableDataCard';

interface Region {
  name: string;
  path: string;
  type: 'continent' | 'country' | 'state' | 'subregion';
  children?: Region[];
}

interface Progress {
  phase: string;
  phaseNumber: number;
  totalPhases: number;
  stepProgress: number;
  stepDescription: string;
  fileProgress: number;
  currentFile: string;
  filesCompleted: number;
  filesTotal: number;
  estimatedTimeRemaining: string;
}

export default function DownloadPage() {
  const [regions, setRegions] = useState('');
  const [isDownloading, setIsDownloading] = useState(false);
  const [message, setMessage] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [jobStatus, setJobStatus] = useState<string>('');
  const [progress, setProgress] = useState<Progress | null>(null);
  const [showTerminal, setShowTerminal] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isResyncing, setIsResyncing] = useState(false);
  const [resyncMessage, setResyncMessage] = useState('');
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Region tree
  const [regionTree, setRegionTree] = useState<Region[]>([]);
  const [isLoadingRegions, setIsLoadingRegions] = useState(true);
  const [selectedRegions, setSelectedRegions] = useState<Set<string>>(new Set());
  const [expandedContinents, setExpandedContinents] = useState<Set<string>>(new Set());
  const [showRegionSelector, setShowRegionSelector] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Load region tree
  useEffect(() => {
    fetch('/api/regions')
      .then(res => res.json())
      .then(data => {
        setRegionTree(data.regions || []);
        setIsLoadingRegions(false);
      })
      .catch(err => {
        console.error('Failed to fetch regions:', err);
        setIsLoadingRegions(false);
      });
  }, []);

  // Poll for job status
  useEffect(() => {
    if (jobId && isDownloading) {
      const pollStatus = async () => {
        try {
          const response = await fetch(`/api/download?jobId=${jobId}`);
          if (response.ok) {
            const data = await response.json();
            setLogs(data.logs || []);
            setJobStatus(data.status || '');
            setProgress(data.progress || null);

            // Stop polling if job is complete
            if (data.status === 'completed' || data.status === 'failed') {
              setIsDownloading(false);
              if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
              }
              if (data.status === 'completed') {
                setMessage('✓ Download completed successfully!');
              } else {
                setMessage('❌ Download failed. Check logs for details.');
              }
            }
          }
        } catch (error) {
          console.error('Failed to poll job status:', error);
        }
      };

      // Poll every 2 seconds
      pollIntervalRef.current = setInterval(pollStatus, 2000);
      pollStatus(); // Initial poll

      return () => {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
        }
      };
    }
  }, [jobId, isDownloading]);

  const handleDownload = async () => {
    const selectedList = Array.from(selectedRegions);
    const textRegions = regions.split(',').map(r => r.trim()).filter(r => r);
    const allRegions = [...selectedList, ...textRegions];

    if (allRegions.length === 0) {
      setMessage('Please select or enter at least one region');
      return;
    }

    setIsDownloading(true);
    setMessage('Starting download...');
    setLogs([]);
    setJobStatus('');
    setJobId(null);
    setIsCancelling(false);

    try {
      const response = await fetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ regions: allRegions }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to start download');
      }

      const { jobId: newJobId, message: successMessage } = await response.json();
      setJobId(newJobId);
      setMessage(successMessage);
    } catch (error) {
      console.error('Failed to start download:', error);
      setMessage(`❌ Error: ${error}`);
      setIsDownloading(false);
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;

    setIsCancelling(true);
    try {
      const response = await fetch(`/api/download?jobId=${jobId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to cancel download');
      }

      setMessage('❌ Download cancelled by user');
      setIsDownloading(false);
      setJobStatus('cancelled');

      // Stop polling
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    } catch (error) {
      console.error('Failed to cancel download:', error);
      setMessage(`❌ Error cancelling: ${error}`);
    } finally {
      setIsCancelling(false);
    }
  };

  const handleResync = async () => {
    setIsResyncing(true);
    setResyncMessage('Scanning data directories and reindexing...');

    try {
      // First, call the data indexer to scan all directories
      const reindexResponse = await fetch('/api/reindex-data', {
        method: 'POST',
      });

      if (!reindexResponse.ok) {
        const error = await reindexResponse.json();
        throw new Error(error.error || 'Failed to reindex data');
      }

      // Then call resync-config (for backward compatibility)
      const response = await fetch('/api/resync-config', {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to resync config');
      }

      const result = await response.json();
      const { details } = result;

      if (details.changes && details.changes.length > 0) {
        setResyncMessage(
          `✓ Data reindexed! Found ${details.osmPlanetFiles} planet files, ` +
          `${details.osmIndexes} indexes, ${details.elevationDatasets} elevation datasets. ${details.changes.length} changes applied.`
        );
      } else {
        setResyncMessage(`✓ Data index updated (${details.osmPlanetFiles} planet files, ${details.osmIndexes} indexes, ${details.elevationDatasets} datasets)`);
      }

    } catch (error) {
      console.error('Failed to reindex data:', error);
      setResyncMessage(`❌ Error: ${error}`);
    } finally {
      setIsResyncing(false);
      // Clear message after 5 seconds
      setTimeout(() => setResyncMessage(''), 5000);
    }
  };

  const handleCloudCacheDownload = async (regionPath: string, regionName: string) => {
    try {
      setMessage(`Downloading ${regionName} from cloud cache...`);
      const response = await fetch(`/api/cloud-cache?action=download&region=${encodeURIComponent(regionPath)}`);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to download from cloud cache');
      }

      const result = await response.json();
      setMessage(`✓ Downloaded ${regionName} successfully!`);

      // Reindex data to include the newly downloaded files
      await fetch('/api/reindex-data', { method: 'POST' });

      setTimeout(() => setMessage(''), 5000);
    } catch (error) {
      console.error('Failed to download from cloud cache:', error);
      setMessage(`❌ Error downloading ${regionName}: ${error}`);
      setTimeout(() => setMessage(''), 5000);
    }
  };

  const toggleRegion = (path: string) => {
    const newSelected = new Set(selectedRegions);
    if (newSelected.has(path)) {
      newSelected.delete(path);
    } else {
      newSelected.add(path);

      // When selecting a child region, deselect any parent regions that might cover it
      // For example, if "europe/france" is selected and user selects "europe/france/alsace",
      // we should deselect "europe/france"
      const pathParts = path.split('/');
      for (let i = 1; i < pathParts.length; i++) {
        const parentPath = pathParts.slice(0, i).join('/');
        if (newSelected.has(parentPath)) {
          newSelected.delete(parentPath);
        }
      }
    }
    setSelectedRegions(newSelected);
  };

  const toggleContinent = (continentPath: string) => {
    const newExpanded = new Set(expandedContinents);
    if (newExpanded.has(continentPath)) {
      newExpanded.delete(continentPath);
    } else {
      newExpanded.add(continentPath);
    }
    setExpandedContinents(newExpanded);
  };

  const toggleAllChildren = (region: Region) => {
    // When selecting "All", we select the parent region itself, not all children
    // This downloads the full region PBF instead of individual subregions
    const newSelected = new Set(selectedRegions);

    if (selectedRegions.has(region.path)) {
      // Already selected, deselect it
      newSelected.delete(region.path);
    } else {
      // Select the parent region and deselect any individual children
      newSelected.add(region.path);

      // Remove any individually selected children since parent now covers them
      const removeChildPaths = (r: Region) => {
        if (r.children) {
          r.children.forEach(child => {
            newSelected.delete(child.path);
            removeChildPaths(child);
          });
        }
      };
      removeChildPaths(region);
    }

    setSelectedRegions(newSelected);
  };

  const hasChildrenSelected = (region: Region): boolean => {
    // Check if any children (at any depth) are individually selected
    if (!region.children) return false;

    const checkAnyChild = (r: Region): boolean => {
      if (selectedRegions.has(r.path)) return true;
      if (r.children) {
        return r.children.some(child => checkAnyChild(child));
      }
      return false;
    };

    return region.children.some(child => checkAnyChild(child));
  };

  const getStatusIcon = () => {
    if (jobStatus === 'completed') {
      return <CheckCircle2 className="h-5 w-5 text-green-600" />;
    } else if (jobStatus === 'failed') {
      return <XCircle className="h-5 w-5 text-red-600" />;
    } else if (jobStatus === 'running') {
      return <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />;
    }
    return null;
  };

  // Search filtering
  const matchesSearch = (region: Region, query: string): boolean => {
    if (!query) return true;
    const lowerQuery = query.toLowerCase();

    // Check if region name or path matches
    if (region.name.toLowerCase().includes(lowerQuery) ||
        region.path.toLowerCase().includes(lowerQuery)) {
      return true;
    }

    // Check if any children match
    if (region.children) {
      return region.children.some(child => matchesSearch(child, query));
    }

    return false;
  };

  const filterRegionTree = (regions: Region[], query: string): Region[] => {
    if (!query) return regions;

    const filtered: Region[] = [];

    for (const region of regions) {
      const childrenMatch = region.children ? filterRegionTree(region.children, query) : [];
      const selfMatches = region.name.toLowerCase().includes(query.toLowerCase()) ||
                         region.path.toLowerCase().includes(query.toLowerCase());

      if (selfMatches || childrenMatch.length > 0) {
        filtered.push({
          ...region,
          children: childrenMatch.length > 0 ? childrenMatch : region.children
        });
      }
    }

    return filtered;
  };

  // Auto-expand matching results
  useEffect(() => {
    if (searchQuery) {
      const newExpanded = new Set<string>();
      const expandMatching = (regions: Region[]) => {
        regions.forEach(region => {
          if (matchesSearch(region, searchQuery) && region.children) {
            newExpanded.add(region.path);
            expandMatching(region.children);
          }
        });
      };
      expandMatching(regionTree);
      setExpandedContinents(newExpanded);
    }
  }, [searchQuery, regionTree]);

  const renderRegionTree = (region: Region, depth: number = 0, parentSelected: boolean = false) => {
    const hasChildren = region.children && region.children.length > 0;
    const isExpanded = expandedContinents.has(region.path);
    const isSelected = selectedRegions.has(region.path);
    const childrenSelected = hasChildren ? hasChildrenSelected(region) : false;

    // If parent is selected, this region is covered and shouldn't be individually selectable
    const isCoveredByParent = parentSelected;

    return (
      <div key={region.path}>
        <div style={{ marginLeft: `${depth * 16}px` }}>
          <div className="flex items-center py-1 hover:bg-gray-50 rounded px-2">
            {hasChildren ? (
              <button
                onClick={() => toggleContinent(region.path)}
                className="mr-2 p-0.5 hover:bg-gray-200 rounded"
              >
                {isExpanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </button>
            ) : (
              <div className="w-6" />
            )}
            <div className="flex items-center gap-2 flex-1">
              {!hasChildren && (
                <Checkbox
                  checked={isSelected || isCoveredByParent}
                  onCheckedChange={() => toggleRegion(region.path)}
                  disabled={isDownloading || isCoveredByParent}
                />
              )}
              <span className={`text-sm ${hasChildren ? 'font-semibold' : ''} ${isCoveredByParent ? 'text-gray-500' : ''}`}>
                {region.name}
              </span>
              {isCoveredByParent && (
                <span className="text-xs text-gray-500">(included in parent)</span>
              )}
              {region.type === 'state' && !isCoveredByParent && (
                <span className="text-xs text-gray-500">(US State)</span>
              )}
            </div>
          </div>
        </div>
        {hasChildren && isExpanded && (
          <>
            {/* "All" checkbox - selects parent region instead of individual children */}
            {depth > 0 && (
              <div style={{ marginLeft: `${(depth + 1) * 16}px` }}>
                <div className="flex items-center py-1 hover:bg-gray-50 rounded px-2">
                  <div className="w-6" />
                  <div className="flex items-center gap-2 flex-1">
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => toggleAllChildren(region)}
                      disabled={isDownloading || childrenSelected || isCoveredByParent}
                    />
                    <span className="text-sm italic text-gray-600">
                      All {region.name} regions
                    </span>
                    {isSelected && (
                      <span className="text-xs text-green-600">(full region selected)</span>
                    )}
                    {childrenSelected && !isSelected && (
                      <span className="text-xs text-gray-500">(deselect children to enable)</span>
                    )}
                  </div>
                </div>
              </div>
            )}
            {region.children?.map(child => renderRegionTree(child, depth + 1, isSelected || isCoveredByParent))}
          </>
        )}
      </div>
    );
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-2">Download Data</h1>
      <p className="text-gray-600 mb-8">
        Pre-download OSM and elevation data for offline analysis
      </p>

      <div className="max-w-4xl space-y-6">
        {/* Cloud Cache Download Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Cloud className="h-5 w-5 text-blue-500" />
              <CardTitle>Download from Cloud Cache</CardTitle>
            </div>
            <CardDescription>
              Download pre-analyzed climb data from GitHub repository, where available.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <p className="text-sm text-gray-600">
                Cloud cache contains pre-analyzed climb data. Download region climbs directly to <code className="px-1 py-0.5 bg-gray-100 rounded text-xs">./output</code> for visualization without running analysis. If yor region is avaialble, use this data and save hours of processing time. If your region is not available here, then download the map and elevation data needed to compute a region analysis.
              </p>
              <CloudCacheTree showDownload={true} onDownload={handleCloudCacheDownload} />
            </div>
          </CardContent>
        </Card>

        {/* Available Data Card */}
        <AvailableDataCard
          onResync={handleResync}
          isResyncing={isResyncing}
          resyncMessage={resyncMessage}
        />

        {/* Download Manager Card */}
        <Card>
          <CardHeader>
            <CardTitle>OSM + Elevation Data Download Manager</CardTitle>
            <CardDescription>
              Download map and elevation data for a region to compute climbs. Select regions or enter custom names to download.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Region Selector Toggle */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Select from regions</Label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowRegionSelector(!showRegionSelector)}
                  disabled={isDownloading}
                >
                  {showRegionSelector ? 'Hide' : 'Show'} Region Selector
                </Button>
              </div>

              {showRegionSelector && (
                <div className="border rounded-lg p-4 bg-gray-50">
                  {/* Search Input */}
                  <div className="mb-3">
                    <Input
                      placeholder="Search regions... (e.g., 'fra' for France)"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="bg-white"
                    />
                  </div>

                  {/* Region Tree */}
                  <div className="max-h-80 overflow-y-auto">
                    {isLoadingRegions ? (
                      <p className="text-sm text-gray-500">Loading regions...</p>
                    ) : regionTree.length > 0 ? (
                      <div className="space-y-1">
                        {filterRegionTree(regionTree, searchQuery).map(region => renderRegionTree(region))}
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500">No regions available</p>
                    )}
                    {searchQuery && filterRegionTree(regionTree, searchQuery).length === 0 && (
                      <p className="text-sm text-gray-500">No matching regions found</p>
                    )}
                  </div>
                </div>
              )}

              {selectedRegions.size > 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-sm font-medium text-blue-900 mb-2">
                    Selected: {selectedRegions.size} region(s)
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {Array.from(selectedRegions).map(path => (
                      <span key={path} className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs rounded">
                        {path.split('/').pop()?.replace(/-/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Text Input */}
            <div className="space-y-2">
              <Label htmlFor="regions">Or enter region names manually</Label>
              <Input
                id="regions"
                placeholder="e.g., Colorado, Vermont, Switzerland, monaco"
                value={regions}
                onChange={(e) => setRegions(e.target.value)}
                disabled={isDownloading}
              />
              <p className="text-sm text-gray-500">
                Enter comma-separated regions (supports country names, US states, or paths like europe/monaco)
              </p>
            </div>

            <Button
              className="w-full"
              onClick={handleDownload}
              disabled={isDownloading || (selectedRegions.size === 0 && !regions.trim())}
            >
              {isDownloading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Downloading...
                </>
              ) : (
                <>
                  <Download className="mr-2 h-4 w-4" />
                  Download {selectedRegions.size + (regions.trim() ? regions.split(',').length : 0)} Region(s)
                </>
              )}
            </Button>

            {message && (
              <div className={`border rounded-lg p-4 text-sm flex items-center gap-2 ${
                jobStatus === 'completed' ? 'bg-green-50 border-green-200' :
                jobStatus === 'failed' ? 'bg-red-50 border-red-200' :
                'bg-blue-50 border-blue-200'
              }`}>
                {getStatusIcon()}
                <p className={`${
                  jobStatus === 'completed' ? 'text-green-900' :
                  jobStatus === 'failed' ? 'text-red-900' :
                  'text-blue-900'
                }`}>
                  {message}
                </p>
              </div>
            )}

            <div className="bg-gray-50 rounded-lg p-4 text-sm">
              <p className="text-gray-600">
                Downloads include OSM road data and spatial indexes. File sizes vary by region (typically 50MB - 500MB per region).
                Elevation data download is handled separately during analysis.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Progress Card */}
        {jobId && (progress || isDownloading) && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Download Progress</CardTitle>
                <div className="flex items-center gap-2">
                  {isDownloading && jobStatus !== 'cancelled' && (
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleCancel}
                      disabled={isCancelling}
                    >
                      {isCancelling ? (
                        <>
                          <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                          Cancelling...
                        </>
                      ) : (
                        <>
                          <XCircle className="mr-2 h-3 w-3" />
                          Cancel
                        </>
                      )}
                    </Button>
                  )}
                  {getStatusIcon()}
                </div>
              </div>
              <CardDescription>
                Job ID: {jobId}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {!progress ? (
                /* Loading state - no progress data yet */
                <div className="flex items-center justify-center py-8">
                  <div className="text-center">
                    <Loader2 className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-2" />
                    <p className="text-sm text-gray-600">Starting download...</p>
                  </div>
                </div>
              ) : (
                <>
              {/* Overall Progress (Main Progress Bar) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-gray-900">
                      Overall Progress
                    </p>
                    <p className="text-xs text-gray-500">
                      {progress.filesCompleted} of {progress.filesTotal} regions completed
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-blue-600">
                      {progress.filesTotal > 0 ? Math.round((progress.filesCompleted / progress.filesTotal) * 100) : 0}%
                    </p>
                    <p className="text-xs text-gray-500">
                      {progress.estimatedTimeRemaining} remaining
                    </p>
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${progress.filesTotal > 0 ? (progress.filesCompleted / progress.filesTotal) * 100 : 0}%` }}
                  />
                </div>
              </div>

              {/* Step Progress (Sub Progress Bar) */}
              {progress.stepDescription && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-700">
                      {progress.stepDescription}
                    </p>
                    <p className="text-sm font-medium text-green-600">
                      {progress.stepProgress}%
                    </p>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progress.stepProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* File Progress (Individual File Progress Bar) */}
              {progress.currentFile && progress.fileProgress > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-600 truncate max-w-xs">
                      Downloading: {progress.currentFile}
                    </p>
                    <p className="text-sm font-medium text-green-600">
                      {progress.fileProgress}%
                    </p>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className="bg-green-400 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${progress.fileProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Terminal Output Toggle */}
              <div className="pt-4 border-t">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowTerminal(!showTerminal)}
                  className="w-full"
                >
                  {showTerminal ? 'Hide' : 'Show'} Terminal Output
                </Button>
              </div>

              {/* Collapsible Terminal Output */}
              {showTerminal && logs.length > 0 && (
                <div className="bg-gray-900 rounded-lg p-4 max-h-64 overflow-y-auto">
                  <pre className="text-green-400 font-mono text-xs whitespace-pre-wrap">
                    {logs.join('')}
                  </pre>
                </div>
              )}
              </>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
