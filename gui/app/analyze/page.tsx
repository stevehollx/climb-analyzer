'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Play, Loader2, FolderOpen, ChevronDown, ChevronRight, CheckCircle2, XCircle } from 'lucide-react';
import { AnalysisConfig } from '@/types/climb';
import CloudCacheWarningDialog from '../components/CloudCacheWarningDialog';
import { CommandPreview } from '@/components/CommandPreview';

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
  itemsCompleted: number;
  itemsTotal: number;
  estimatedTimeRemaining: string;
  startTime: number;
}

export default function AnalyzePage() {
  const [config, setConfig] = useState<AnalysisConfig>({
    mode: 'address',
    surfaceFilter: 'all',
    cyclingFilter: false,
    units: 'metric', // Default to metric (will be updated based on region)
    minScore: 6000, // Default for address mode (uses basic score)
    geocoding: true,
    deleteDataOnComplete: false,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [isFailed, setIsFailed] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [jobStatus, setJobStatus] = useState<string>('');
  const [progress, setProgress] = useState<Progress | null>(null);
  const [showTerminal, setShowTerminal] = useState(false);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Region selector state
  const [regionTree, setRegionTree] = useState<Region[]>([]);
  const [selectedRegions, setSelectedRegions] = useState<Set<string>>(new Set());
  const [expandedContinents, setExpandedContinents] = useState<Set<string>>(new Set());
  const [showRegionSelector, setShowRegionSelector] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Cloud cache warning dialog state
  const [showCloudCacheDialog, setShowCloudCacheDialog] = useState(false);
  const [pendingAnalysisConfig, setPendingAnalysisConfig] = useState<AnalysisConfig | null>(null);

  // Load region tree on mount
  useEffect(() => {
    fetch('/api/regions')
      .then(res => res.json())
      .then(data => {
        if (data.regions) {
          setRegionTree(data.regions);
        }
      })
      .catch(err => console.error('Failed to load regions:', err));
  }, []);

  // Clear selections and adjust min score when mode changes
  useEffect(() => {
    setSelectedRegions(new Set());
    setShowRegionSelector(false);

    // Set min score to 0 for region mode, 6000 for address mode
    if (config.mode === 'region') {
      setConfig(prev => ({ ...prev, minScore: 0 }));
    } else if (config.mode === 'address') {
      setConfig(prev => ({ ...prev, minScore: 6000 }));
    }
  }, [config.mode]);

  // Restore analysis state from localStorage on mount
  useEffect(() => {
    try {
      const savedState = localStorage.getItem('analysisState');
      if (savedState) {
        const state = JSON.parse(savedState);

        // Only restore if the job was running or recently completed
        if (state.jobId) {
          setJobId(state.jobId);
          setIsRunning(state.isRunning || false);
          setIsComplete(state.isComplete || false);
          setIsFailed(state.isFailed || false);
          setLogs(state.logs || []);
          setJobStatus(state.jobStatus || '');
          setProgress(state.progress || null);

          // If analysis was running, resume polling
          if (state.isRunning && state.jobId) {
            setIsRunning(true);
          }
        }
      }
    } catch (error) {
      console.error('Failed to restore analysis state:', error);
    }
  }, []);

  // Save analysis state to localStorage whenever it changes
  useEffect(() => {
    if (jobId) {
      try {
        const state = {
          jobId,
          isRunning,
          isComplete,
          isFailed,
          logs,
          jobStatus,
          progress,
          timestamp: Date.now(),
        };
        localStorage.setItem('analysisState', JSON.stringify(state));
      } catch (error) {
        console.error('Failed to save analysis state:', error);
      }
    }
  }, [jobId, isRunning, isComplete, isFailed, logs, jobStatus, progress]);

  // Helper function to detect US regions
  const isUSRegion = (regionPath: string): boolean => {
    // Check if path contains "north-america/us/" or "north-america/united-states"
    // or is a top-level US state name (legacy paths)
    const usStateNames = [
      'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
      'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
      'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
      'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
      'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
      'new-hampshire', 'new-jersey', 'new-mexico', 'new-york',
      'north-carolina', 'north-dakota', 'ohio', 'oklahoma', 'oregon',
      'pennsylvania', 'rhode-island', 'south-carolina', 'south-dakota',
      'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
      'west-virginia', 'wisconsin', 'wyoming'
    ];

    const pathLower = regionPath.toLowerCase().replace(/\s+/g, '-');

    // Check if path starts with north-america/us or contains /us/
    if (pathLower.includes('north-america/us') || pathLower.includes('/us/')) {
      return true;
    }

    // Check if it's a top-level US state name
    const lastPart = regionPath.split('/').pop()?.toLowerCase().replace(/\s+/g, '-');
    if (lastPart && usStateNames.includes(lastPart)) {
      return true;
    }

    return false;
  };

  // Update config when regions are selected/deselected
  useEffect(() => {
    const selectedList = Array.from(selectedRegions);
    if (config.mode === 'region' && selectedList.length > 0) {
      // Determine units based on first selected region
      const firstRegion = selectedList[0];
      const shouldUseImperial = isUSRegion(firstRegion);

      setConfig(prev => ({
        ...prev,
        region: selectedList[0],
        regions: selectedList,
        units: shouldUseImperial ? 'imperial' : 'metric'
      }));
    }
  }, [selectedRegions]);

  const toggleRegion = (path: string) => {
    const newSelected = new Set(selectedRegions);
    if (newSelected.has(path)) {
      newSelected.delete(path);
    } else {
      // Region mode: allow multiple selections
      newSelected.add(path);

      // When selecting a child region, deselect any parent regions
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

  // Search filtering
  const matchesSearch = (region: Region, query: string): boolean => {
    if (!query) return true;
    const lowerQuery = query.toLowerCase();

    if (region.name.toLowerCase().includes(lowerQuery) ||
        region.path.toLowerCase().includes(lowerQuery)) {
      return true;
    }

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
                  disabled={isRunning || isCoveredByParent}
                />
              )}
              <span className={`text-sm ${hasChildren ? 'font-semibold' : ''} ${isCoveredByParent ? 'text-gray-500' : ''}`}>
                {region.name}
              </span>
              {isCoveredByParent && (
                <span className="text-xs text-gray-500">(included in parent)</span>
              )}
            </div>
          </div>
        </div>
        {hasChildren && isExpanded && (
          <>
            {depth > 0 && (
              <div style={{ marginLeft: `${(depth + 1) * 16}px` }}>
                <div className="flex items-center py-1 hover:bg-gray-50 rounded px-2">
                  <div className="w-6" />
                  <div className="flex items-center gap-2 flex-1">
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => toggleAllChildren(region)}
                      disabled={isRunning || childrenSelected || isCoveredByParent}
                    />
                    <span className="text-sm italic text-gray-600">
                      All {region.name} regions
                    </span>
                    {isSelected && (
                      <span className="text-xs text-green-600">(full region)</span>
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

  // Poll for job status
  useEffect(() => {
    if (jobId && isRunning) {
      const pollStatus = async () => {
        try {
          const response = await fetch(`/api/analyze?jobId=${jobId}`);
          if (response.ok) {
            const data = await response.json();
            setLogs(data.logs || []);
            setJobStatus(data.status || '');
            setProgress(data.progress || null);

            // Stop polling if job is complete
            if (data.status === 'completed' || data.status === 'failed') {
              setIsRunning(false);
              if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
              }
              if (data.status === 'completed') {
                setIsComplete(true);
                setIsFailed(false);
                // Trigger reindexing to include newly analyzed climb reports
                fetch('/api/reindex-data', { method: 'POST' }).catch(err =>
                  console.error('Failed to reindex data after analysis:', err)
                );
              } else if (data.status === 'failed') {
                setIsComplete(false);
                setIsFailed(true);
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
  }, [jobId, isRunning]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Show cloud cache warning dialog to check for matches
    setPendingAnalysisConfig(config);
    setShowCloudCacheDialog(true);
  };

  const startAnalysis = async (configToUse: AnalysisConfig) => {
    setIsRunning(true);
    setIsComplete(false);
    setIsFailed(false);
    setLogs([]);
    setJobStatus('');
    setJobId(null);
    setProgress(null);

    try {
      // Call the API to start analysis
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configToUse),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to start analysis');
      }

      const { jobId: newJobId } = await response.json();
      setJobId(newJobId);

    } catch (error) {
      console.error('Failed to run analysis:', error);
      setIsRunning(false);
      setLogs([`❌ Error: ${error}`]);
    }
  };

  const handleUseCloudCache = async () => {
    setShowCloudCacheDialog(false);

    // Get the region from config
    let region = pendingAnalysisConfig?.regions || pendingAnalysisConfig?.address || '';

    // If regions is an array, join them
    if (Array.isArray(region)) {
      region = region.join(',');
    }

    // Redirect to download page or show success message
    try {
      const response = await fetch(`/api/cloud-cache?action=download&region=${encodeURIComponent(region)}`);

      if (response.ok) {
        setLogs(['✓ Downloaded from cloud cache successfully!', 'Files saved to ./output directory']);
        setIsComplete(true);
        // Trigger reindexing to include newly downloaded climb reports
        fetch('/api/reindex-data', { method: 'POST' }).catch(err =>
          console.error('Failed to reindex data after cloud cache download:', err)
        );
      } else {
        const error = await response.json();
        setLogs([`❌ Failed to download from cloud cache: ${error.error}`]);
      }
    } catch (error) {
      console.error('Failed to download from cloud cache:', error);
      setLogs([`❌ Error downloading from cloud cache: ${error}`]);
    }

    setPendingAnalysisConfig(null);
  };

  const handleContinueWithAnalysis = () => {
    setShowCloudCacheDialog(false);

    if (pendingAnalysisConfig) {
      startAnalysis(pendingAnalysisConfig);
    }

    setPendingAnalysisConfig(null);
  };

  const handleCloseDialog = () => {
    setShowCloudCacheDialog(false);
    setPendingAnalysisConfig(null);
  };

  const handleStop = async () => {
    if (!jobId) return;

    try {
      const response = await fetch(`/api/analyze?jobId=${jobId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to stop analysis');
      }

      setIsRunning(false);
      setJobStatus('stopped');
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    } catch (error) {
      console.error('Failed to stop analysis:', error);
      alert(`Failed to stop analysis: ${error}`);
    }
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

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Run Analysis</h1>
        <p className="text-gray-600">
          Configure and run climb analysis for an address, region, or multiple regions
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Form */}
        <div className="lg:col-span-1">
          <form onSubmit={handleSubmit}>
            <Card>
              <CardHeader>
                <CardTitle>Analysis Configuration</CardTitle>
                <CardDescription>
                  Set parameters for your climb analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Analysis Mode */}
                <div className="space-y-2">
                  <Label>Analysis Mode</Label>
                  <Select
                    value={config.mode}
                    onValueChange={(value: any) =>
                      setConfig({ ...config, mode: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="address">Address (Radius)</SelectItem>
                      <SelectItem value="region">Region (Full Area)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-gray-500">
                    Address analyzes within a radius; Region analyzes full region(s)
                  </p>
                </div>
                {/* Address Mode Fields */}
                {config.mode === 'address' && (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="address">Address</Label>
                      <Input
                        id="address"
                        placeholder="e.g., Boulder, CO"
                        value={config.address || ''}
                        onChange={(e) =>
                          setConfig({ ...config, address: e.target.value })
                        }
                      />
                      <p className="text-sm text-gray-500">
                        Enter a city, address, or location
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="radius">Radius</Label>
                      <Input
                        id="radius"
                        type="number"
                        min="0"
                        max={config.units === 'imperial' ? '25' : '40'}
                        step="0.1"
                        placeholder="25"
                        value={config.radius || ''}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            radius: parseFloat(e.target.value),
                          })
                        }
                      />
                      <p className="text-sm text-gray-500">
                        Search radius in {config.units === 'imperial' ? 'miles (max 25)' : 'kilometers (max 40)'}
                      </p>
                    </div>
                  </>
                )}

                {/* Region Mode Fields */}
                {config.mode === 'region' && (
                  <div className="space-y-4">
                    <Label>Region Selection - Choose One Method:</Label>

                    {/* Method 1: Region Tree Selector */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-sm font-medium">Method 1: Browse & Select Regions</Label>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => setShowRegionSelector(!showRegionSelector)}
                        >
                          {showRegionSelector ? 'Hide' : 'Show'} Region Tree
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

                          <p className="text-sm text-gray-600 mb-3">
                            Select one or more regions (selecting "All" analyzes the full parent region):
                          </p>

                          {/* Region Tree */}
                          <div className="max-h-80 overflow-y-auto">
                            {regionTree.length > 0 ? (
                              <div className="space-y-1">
                                {filterRegionTree(regionTree, searchQuery).map(region => renderRegionTree(region))}
                              </div>
                            ) : (
                              <p className="text-sm text-gray-500">Loading regions...</p>
                            )}
                            {searchQuery && filterRegionTree(regionTree, searchQuery).length === 0 && (
                              <p className="text-sm text-gray-500">No matching regions found</p>
                            )}
                          </div>
                        </div>
                      )}

                      {selectedRegions.size > 0 && (
                        <div className="text-sm">
                          <span className="font-semibold">Selected ({selectedRegions.size}): </span>
                          <span className="text-blue-600">{Array.from(selectedRegions).join(', ')}</span>
                        </div>
                      )}
                    </div>

                    {/* OR Divider */}
                    <div className="relative">
                      <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-300"></div>
                      </div>
                      <div className="relative flex justify-center text-sm">
                        <span className="px-2 bg-white text-gray-500 font-medium">OR</span>
                      </div>
                    </div>

                    {/* Method 2: Manual Text Input */}
                    <div className="space-y-2">
                      <Label htmlFor="regions" className="text-sm font-medium">Method 2: Enter Region Names Manually</Label>
                      <Input
                        id="regions"
                        placeholder="e.g., Colorado, europe/france, asia/japan (comma-separated for multiple)"
                        value={config.regions?.join(', ') || ''}
                        onChange={(e) => {
                          const inputRegions = e.target.value
                            .split(',')
                            .map((r) => r.trim())
                            .filter(r => r);

                          // Determine units based on first region
                          const shouldUseImperial = inputRegions.length > 0 && isUSRegion(inputRegions[0]);

                          setConfig({
                            ...config,
                            regions: inputRegions,
                            region: inputRegions[0] || '',
                            units: shouldUseImperial ? 'imperial' : 'metric',
                          });
                          // Clear tree selection if typing manually
                          const selectedList = Array.from(selectedRegions).sort().join(',');
                          const inputList = inputRegions.sort().join(',');
                          if (selectedList !== inputList) {
                            setSelectedRegions(new Set());
                          }
                        }}
                      />
                      <p className="text-sm text-gray-500">
                        Type region names separated by commas (e.g., states, countries, or paths like "north-america/us/colorado")
                      </p>
                    </div>
                  </div>
                )}

                {/* Surface Filter */}
                <div className="space-y-2">
                  <Label>Surface Filter</Label>
                  <Select
                    value={config.surfaceFilter}
                    onValueChange={(value: any) =>
                      setConfig({ ...config, surfaceFilter: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Surfaces</SelectItem>
                      <SelectItem value="paved">Paved Only</SelectItem>
                      <SelectItem value="gravel">Gravel Only</SelectItem>
                      <SelectItem value="dirt">Dirt Only</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Units */}
                <div className="space-y-2">
                  <Label>Units</Label>
                  <Select
                    value={config.units}
                    onValueChange={(value: any) =>
                      setConfig({ ...config, units: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="imperial">Imperial</SelectItem>
                      <SelectItem value="metric">Metric</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Min Score */}
                <div className="space-y-2">
                  <Label htmlFor="minScore">Minimum Basic Score (Optional)</Label>
                  <Input
                    id="minScore"
                    type="number"
                    min="0"
                    step="1"
                    placeholder="Default: 6000"
                    value={config.minScore !== undefined ? config.minScore : ''}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        minScore: e.target.value
                          ? parseInt(e.target.value, 10)
                          : undefined,
                      })
                    }
                  />
                  <p className="text-sm text-gray-500">
                    Filter climbs by basic score threshold (all 3 scoring algorithms are always calculated)
                  </p>
                </div>

                {/* Options */}
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="deleteData"
                      checked={config.deleteDataOnComplete}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          deleteDataOnComplete: e.target.checked,
                        })
                      }
                      className="rounded"
                    />
                    <Label htmlFor="deleteData" className="cursor-pointer">
                      Delete Data After Analysis
                    </Label>
                  </div>
                </div>

                {/* CLI command preview - copy and paste to run in your own terminal */}
                <CommandPreview config={config} />

                {/* Submit/Stop Button */}
                {!isRunning ? (
                  <Button
                    type="submit"
                    className="w-full"
                    disabled={isRunning}
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Or Run in Background (Server)
                  </Button>
                ) : (
                  <Button
                    type="button"
                    onClick={handleStop}
                    variant="destructive"
                    className="w-full"
                  >
                    <XCircle className="mr-2 h-4 w-4" />
                    Stop Analysis
                  </Button>
                )}

                {/* View Results Button */}
                <div className="pt-4 border-t">
                  <Button
                    type="button"
                    variant={isComplete ? "default" : isFailed ? "destructive" : "outline"}
                    className={`w-full ${isComplete ? 'bg-green-600 hover:bg-green-700' : isFailed ? 'bg-red-600 hover:bg-red-700' : ''}`}
                    onClick={() => window.location.href = '/visualize'}
                  >
                    <FolderOpen className="mr-2 h-4 w-4" />
                    {isComplete ? 'View Results (Analysis Complete!)' : isFailed ? 'Analysis Could Not Complete' : 'View Results & Output Files'}
                  </Button>
                  {isComplete && (
                    <p className="text-xs text-green-600 mt-2 text-center font-semibold">
                      ✓ Your analysis is ready to view!
                    </p>
                  )}
                  {isFailed && (
                    <p className="text-xs text-red-600 mt-2 text-center font-semibold">
                      ✗ Analysis failed - check logs for details
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </form>
        </div>

        {/* Progress and Logs */}
        <div className="lg:col-span-2 space-y-6">
          {/* Progress Card */}
          {jobId && (progress || isRunning) && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Analysis Progress</CardTitle>
                  {getStatusIcon()}
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
                      <p className="text-sm text-gray-600">Starting analysis...</p>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Main Progress Bar - Overall Phase Progress */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-semibold text-gray-900">
                            Step {progress.phaseNumber} of {progress.totalPhases}: {progress.phase}
                          </p>
                          <p className="text-xs text-gray-500">
                            Overall analysis progress
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-medium text-blue-600">
                            {Math.round((progress.phaseNumber / progress.totalPhases) * 100)}%
                          </p>
                          <p className="text-xs text-gray-500">
                            Est. {progress.estimatedTimeRemaining} remaining
                          </p>
                        </div>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                          style={{ width: `${(progress.phaseNumber / progress.totalPhases) * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Sub Progress Bar - Step Progress */}
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
                        {progress.itemsTotal > 0 && (
                          <p className="text-xs text-gray-500">
                            {progress.itemsCompleted} of {progress.itemsTotal} items processed
                          </p>
                        )}
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

          {/* Status Card - shown when no job is running */}
          {!jobId && !isRunning && (
            <Card>
              <CardHeader>
                <CardTitle>Status</CardTitle>
                <CardDescription>Analysis status</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 text-center py-4">
                  No analysis running. Configure settings and click "Start Analysis" to begin.
                </p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Cloud Cache Warning Dialog */}
        <CloudCacheWarningDialog
          isOpen={showCloudCacheDialog}
          onClose={handleCloseDialog}
          onUseCache={handleUseCloudCache}
          onContinueAnalysis={handleContinueWithAnalysis}
          region={Array.isArray(config.regions) ? config.regions.join(',') : (config.regions || config.address || '')}
        />
      </div>
    </div>
  );
}
