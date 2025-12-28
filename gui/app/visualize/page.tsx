'use client';

import { useState, useEffect, Suspense, useRef } from 'react';
import { useSearchParams } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Map, List, Upload, CheckCircle2, X, ChevronDown, ChevronUp, Trophy } from 'lucide-react';
import { Climb } from '@/types/climb';
import { parseClimbCSV, parseClimbExcel, getCategoryColor, getTopNClimbs, getTopPercentClimbs } from '@/lib/csv-parser';
import { ClimbMap } from '@/components/ClimbMap';
import { ClimbList } from '@/components/ClimbList';
import { ElevationProfile } from '@/components/ElevationProfile';
import { LocationFilter } from '@/components/LocationFilter';
import { ClimbDetailDrawer } from '@/components/ClimbDetailDrawer';

interface LoadedFile {
  name: string;
  climbs: Climb[];
}

// Calculate distance between two coordinates in miles using Haversine formula
function calculateDistanceMiles(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 3959; // Earth's radius in miles
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

interface OutputFile {
  filename: string;
  region: string;
  createdAt: string;
}

function VisualizePageContent() {
  const searchParams = useSearchParams();
  const [loadedFiles, setLoadedFiles] = useState<LoadedFile[]>([]);
  const [availableFiles, setAvailableFiles] = useState<OutputFile[]>([]);
  const [hasAutoLoaded, setHasAutoLoaded] = useState(false);
  // File loading limits - to prevent loading too many climbs and causing slow performance
  const [loadStartN, setLoadStartN] = useState(1);
  const [loadEndN, setLoadEndN] = useState(2000);
  const [loadScoreType, setLoadScoreType] = useState<'basic' | 'fiets' | 'pdi'>('basic');
  const [filterMode, setFilterMode] = useState<'top-n' | 'top-percent'>('top-n');
  const [startN, setStartN] = useState(1);
  const [endN, setEndN] = useState(25);
  const [startPercent, setStartPercent] = useState(0);
  const [endPercent, setEndPercent] = useState(10);
  const [scoreType, setScoreType] = useState<'basic' | 'fiets' | 'pdi'>('basic');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingFileName, setLoadingFileName] = useState('');
  const [loadingStage, setLoadingStage] = useState<'downloading' | 'parsing' | null>(null);
  const [loadingCancelled, setLoadingCancelled] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [showAllRoutes, setShowAllRoutes] = useState(false);
  const [selectedClimb, setSelectedClimb] = useState<Climb | null>(null);
  const [viewMode, setViewMode] = useState<'map' | 'list'>('map');
  const [useDetailDrawer, setUseDetailDrawer] = useState(true); // Use detail drawer instead of simple elevation profile

  // Load available output files on mount
  useEffect(() => {
    const loadAvailableFiles = async () => {
      try {
        const response = await fetch('/api/output-files');
        if (response.ok) {
          const data = await response.json();
          setAvailableFiles(data.outputFiles || []);
        }
      } catch (error) {
        console.error('Failed to load available files:', error);
      }
    };
    loadAvailableFiles();
  }, []);

  // Auto-load file from query parameter
  useEffect(() => {
    const fileParam = searchParams.get('file');
    if (fileParam && !hasAutoLoaded) {
      setHasAutoLoaded(true);
      const loadFileFromServer = async () => {
        try {
          setIsLoading(true);
          setLoadingFileName(fileParam);
          setLoadingProgress(0);

          const response = await fetch(`/api/output-file?filename=${encodeURIComponent(fileParam)}`);
          if (!response.ok) {
            throw new Error(`Failed to load file: ${response.statusText}`);
          }

          const arrayBuffer = await response.arrayBuffer();
          let climbs: Climb[];

          if (fileParam.endsWith('.xlsx') || fileParam.endsWith('.xls')) {
            climbs = parseClimbExcel(arrayBuffer);
          } else {
            const text = new TextDecoder().decode(arrayBuffer);
            climbs = parseClimbCSV(text);
          }

          setLoadedFiles([{ name: fileParam, climbs }]);
          setLoadingProgress(100);
        } catch (error) {
          console.error('Failed to auto-load file:', error);
          alert(`Failed to load ${fileParam}. Please select it manually.`);
        } finally {
          setIsLoading(false);
          setLoadingProgress(0);
          setLoadingFileName('');
        }
      };

      loadFileFromServer();
    }
  }, [searchParams, hasAutoLoaded]);

  // Monitor selectedClimb changes
  useEffect(() => {
  }, [selectedClimb]);

  // Collapse state for filter cards
  const [loadResultsCollapsed, setLoadResultsCollapsed] = useState(false);
  const [filtersCollapsed, setFiltersCollapsed] = useState(false);
  const [legendCollapsed, setLegendCollapsed] = useState(false);
  const [advancedFiltersCollapsed, setAdvancedFiltersCollapsed] = useState(false);

  // Advanced filters
  const [minLength, setMinLength] = useState<number>(0);
  const [maxLength, setMaxLength] = useState<number>(999);
  const [minProminence, setMinProminence] = useState<number>(0);
  const [maxProminence, setMaxProminence] = useState<number>(99999);
  const [minElevGain, setMinElevGain] = useState<number>(0);
  const [maxElevGain, setMaxElevGain] = useState<number>(99999);
  const [minHeight, setMinHeight] = useState<number>(0);
  const [maxHeight, setMaxHeight] = useState<number>(99999);
  const [minAvgGrade, setMinAvgGrade] = useState<number>(0);
  const [maxAvgGrade, setMaxAvgGrade] = useState<number>(100);
  const [minMaxGrade, setMinMaxGrade] = useState<number>(0);
  const [maxMaxGrade, setMaxMaxGrade] = useState<number>(100);
  const [surfaceFilters, setSurfaceFilters] = useState<Set<string>>(new Set(['all']));
  const [cyclingAccessFilters, setCyclingAccessFilters] = useState<Set<string>>(new Set(['all']));
  const [cyclingPermittedOnly, setCyclingPermittedOnly] = useState(false);
  const [highwayTypeFilters, setHighwayTypeFilters] = useState<Set<string>>(new Set(['all']));
  const [tracktypeFilters, setTracktypeFilters] = useState<Set<string>>(new Set(['all']));
  const [locationFilter, setLocationFilter] = useState<{ lat: number; lon: number; radius: number } | null>(null);

  // Category filter state - All categories selected by default including N/A and Uncategorized
  // Support both formats: "Cat 1" and "1", "N/A" and "Uncategorized"
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(
    new Set(['HC', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', '1', '2', '3', '4', 'N/A', 'Uncategorized'])
  );

  const toggleCategory = (category: string) => {
    setSelectedCategories(prev => {
      const newSet = new Set(prev);

      // Map category names to both possible formats
      const categoryMap: { [key: string]: string[] } = {
        'HC': ['HC'],
        'Cat 1': ['Cat 1', '1'],
        'Cat 2': ['Cat 2', '2'],
        'Cat 3': ['Cat 3', '3'],
        'Cat 4': ['Cat 4', '4'],
        'Uncategorized': ['Uncategorized', 'N/A']
      };

      const categoriesToToggle = categoryMap[category] || [category];
      const isCurrentlySelected = categoriesToToggle.some(cat => newSet.has(cat));

      // Toggle all related category formats
      categoriesToToggle.forEach(cat => {
        if (isCurrentlySelected) {
          newSet.delete(cat);
        } else {
          newSet.add(cat);
        }
      });

      return newSet;
    });
  };

  // Get all climbs from all loaded files
  const allClimbs = loadedFiles.flatMap(f => f.climbs);

  // DEBUG: Log categories found in loaded climbs
  useEffect(() => {
  }, [allClimbs.length]);

  // Track previous climb count to detect when new files are added
  const [prevClimbCount, setPrevClimbCount] = useState(0);

  // Update endN when climbs are loaded
  useEffect(() => {
    if (allClimbs.length === 0) return;

    const prevCount = prevClimbCount;
    const currentCount = allClimbs.length;

    // Case 1: User had endN at previous max, expand to new max
    if (prevCount > 0 && currentCount > prevCount && endN === prevCount) {
      setEndN(currentCount);
    }
    // Case 2: endN exceeds new total, reduce it
    else if (currentCount < endN) {
      setEndN(currentCount);
    }

    setPrevClimbCount(currentCount);
  }, [allClimbs.length]);

  // Update filter max values based on loaded climb data
  useEffect(() => {
    if (allClimbs.length > 0) {
      const maxLengthVal = Math.max(...allClimbs.map(c => c.length));
      const maxProminenceVal = Math.max(...allClimbs.map(c => c.prominence));
      const maxElevGainVal = Math.max(...allClimbs.map(c => c.elevationGain));
      const maxHeightVal = Math.max(...allClimbs.map(c => c.height));
      const maxAvgGradeVal = Math.max(...allClimbs.map(c => c.avgGrade));
      const maxMaxGradeVal = Math.max(...allClimbs.map(c => c.maxGrade));

      // Update max values to match data (with a small buffer)
      setMaxLength(Math.ceil(maxLengthVal + 1));
      setMaxProminence(Math.ceil(maxProminenceVal + 100));
      setMaxElevGain(Math.ceil(maxElevGainVal + 100));
      setMaxHeight(Math.ceil(maxHeightVal + 100));
      setMaxAvgGrade(Math.ceil(maxAvgGradeVal + 1));
      setMaxMaxGrade(Math.ceil(maxMaxGradeVal + 1));
    }
  }, [allClimbs]);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;

    if (!files || files.length === 0) {
      return;
    }

    setIsLoading(true);
    setLoadingProgress(0);
    const newFiles: LoadedFile[] = [];
    const fileArray = Array.from(files);
    const totalFiles = fileArray.length;

    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      try {
        setLoadingFileName(file.name);
        setLoadingProgress(((i) / totalFiles) * 100);

        const isExcel = file.name.endsWith('.xlsx') || file.name.endsWith('.xls');
        let climbs: Climb[];

        if (isExcel) {
          // Parse Excel file
          const arrayBuffer = await file.arrayBuffer();
          climbs = parseClimbExcel(arrayBuffer);
        } else {
          // Parse CSV file
          const text = await file.text();
          climbs = parseClimbCSV(text);
        }

        newFiles.push({ name: file.name, climbs });
        setLoadingProgress(((i + 1) / totalFiles) * 100);
      } catch (error) {
        console.error(`Failed to parse ${file.name}:`, error);
        alert(`Failed to load ${file.name}. Please ensure it's a valid climb file (CSV or Excel).`);
      }
    }

    setLoadedFiles(prev => [...prev, ...newFiles]);
    setIsLoading(false);
    setLoadingProgress(0);
    setLoadingFileName('');

    // CRITICAL FIX: Reset the file input so the same file can be selected again
    // This allows re-uploading after removing a file
    event.target.value = '';
  };

  const removeFile = (fileName: string) => {
    setLoadedFiles(prev => prev.filter(f => f.name !== fileName));
  };

  const cancelLoading = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoadingCancelled(true);
    }
  };

  const loadFileFromOutputDir = async (filename: string) => {
    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();
    setLoadingCancelled(false);

    try {
      setIsLoading(true);
      setLoadingFileName(filename);
      setLoadingProgress(0);
      setLoadingStage('downloading');

      const response = await fetch(
        `/api/output-file?filename=${encodeURIComponent(filename)}`,
        { signal: abortControllerRef.current.signal }
      );

      if (!response.ok) {
        throw new Error(`Failed to load file: ${response.statusText}`);
      }

      // Get content length for progress tracking
      const contentLength = response.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength, 10) : 0;

      if (total > 0 && response.body) {
        // Stream the response with progress
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          chunks.push(value);
          received += value.length;
          setLoadingProgress(Math.round((received / total) * 50)); // 0-50% for download
        }

        // Combine chunks into single array buffer
        const allChunks = new Uint8Array(received);
        let position = 0;
        for (const chunk of chunks) {
          allChunks.set(chunk, position);
          position += chunk.length;
        }

        // Switch to parsing stage
        setLoadingStage('parsing');
        setLoadingProgress(50);

        // Small delay to allow UI to update before blocking parse
        await new Promise(resolve => setTimeout(resolve, 50));

        let climbs: Climb[];
        if (filename.endsWith('.xlsx') || filename.endsWith('.xls')) {
          climbs = parseClimbExcel(allChunks.buffer);
        } else {
          const text = new TextDecoder().decode(allChunks);
          climbs = parseClimbCSV(text);
        }

        // Apply loading limits - sort by score and slice
        const sortedClimbs = [...climbs].sort((a, b) => {
          const scoreA = loadScoreType === 'pdi' ? b.pdiScore : loadScoreType === 'fiets' ? b.fietsScore : b.basicScore;
          const scoreB = loadScoreType === 'pdi' ? a.pdiScore : loadScoreType === 'fiets' ? a.fietsScore : a.basicScore;
          return scoreA - scoreB;
        });
        const limitedClimbs = sortedClimbs.slice(loadStartN - 1, loadEndN);

        setLoadingProgress(100);
        setLoadedFiles(prev => [...prev, { name: filename, climbs: limitedClimbs }]);
      } else {
        // Fallback for when content-length is not available
        setLoadingProgress(25);
        const arrayBuffer = await response.arrayBuffer();

        setLoadingStage('parsing');
        setLoadingProgress(50);
        await new Promise(resolve => setTimeout(resolve, 50));

        let climbs: Climb[];
        if (filename.endsWith('.xlsx') || filename.endsWith('.xls')) {
          climbs = parseClimbExcel(arrayBuffer);
        } else {
          const text = new TextDecoder().decode(arrayBuffer);
          climbs = parseClimbCSV(text);
        }

        // Apply loading limits - sort by score and slice
        const sortedClimbs = [...climbs].sort((a, b) => {
          const scoreA = loadScoreType === 'pdi' ? b.pdiScore : loadScoreType === 'fiets' ? b.fietsScore : b.basicScore;
          const scoreB = loadScoreType === 'pdi' ? a.pdiScore : loadScoreType === 'fiets' ? a.fietsScore : a.basicScore;
          return scoreA - scoreB;
        });
        const limitedClimbs = sortedClimbs.slice(loadStartN - 1, loadEndN);

        setLoadingProgress(100);
        setLoadedFiles(prev => [...prev, { name: filename, climbs: limitedClimbs }]);
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Loading cancelled by user');
      } else {
        console.error('Failed to load file:', error);
        alert(`Failed to load ${filename}: ${error}`);
      }
    } finally {
      setIsLoading(false);
      setLoadingProgress(0);
      setLoadingFileName('');
      setLoadingStage(null);
      abortControllerRef.current = null;
    }
  };

  // Apply filtering based on score type and filter mode with range
  let filteredClimbs: Climb[] = [];

  if (allClimbs.length > 0) {
    if (filterMode === 'top-n') {
      // Get top climbs and then slice the range
      const topClimbs = getTopNClimbs(allClimbs, endN, scoreType);
      filteredClimbs = topClimbs.slice(startN - 1, endN); // startN is 1-indexed
    } else {
      // Get top percentage climbs and slice the range
      const topClimbs = getTopPercentClimbs(allClimbs, endPercent, scoreType);
      const startIndex = Math.floor((startPercent / 100) * allClimbs.length);
      const endIndex = Math.floor((endPercent / 100) * allClimbs.length);
      filteredClimbs = topClimbs.slice(startIndex, endIndex);
    }
  }

  // Apply category filter
  filteredClimbs = filteredClimbs.filter(climb => selectedCategories.has(climb.category));

  // Apply advanced filters
  filteredClimbs = filteredClimbs.filter(climb => {
    // Length filter
    if (climb.length < minLength || climb.length > maxLength) return false;

    // Prominence filter
    if (climb.prominence < minProminence || climb.prominence > maxProminence) return false;

    // Elevation gain filter
    if (climb.elevationGain < minElevGain || climb.elevationGain > maxElevGain) return false;

    // Height filter
    if (climb.height < minHeight || climb.height > maxHeight) return false;

    // Average grade filter
    if (climb.avgGrade < minAvgGrade || climb.avgGrade > maxAvgGrade) return false;

    // Max grade filter
    if (climb.maxGrade < minMaxGrade || climb.maxGrade > maxMaxGrade) return false;

    // Surface filter (skip if 'all' is selected)
    if (!surfaceFilters.has('all') && climb.surface && !surfaceFilters.has(climb.surface.toLowerCase())) return false;

    // Cycling access filter (skip if 'all' is selected)
    if (!cyclingAccessFilters.has('all') && climb.cyclingAccess && !cyclingAccessFilters.has(climb.cyclingAccess.toLowerCase())) return false;

    // Cycling permitted only filter
    if (cyclingPermittedOnly && climb.cyclingAccess && climb.cyclingAccess.toLowerCase() === 'no') return false;

    // Highway type filter (skip if 'all' is selected)
    if (!highwayTypeFilters.has('all') && climb.highwayType && !highwayTypeFilters.has(climb.highwayType.toLowerCase())) return false;

    // Tracktype filter (skip if 'all' is selected)
    if (!tracktypeFilters.has('all') && climb.tracktype && !tracktypeFilters.has(climb.tracktype.toLowerCase())) return false;

    // Location filter (distance from point)
    if (locationFilter) {
      const distance = calculateDistanceMiles(
        locationFilter.lat,
        locationFilter.lon,
        climb.lat,
        climb.lon
      );
      if (distance > locationFilter.radius) return false;
    }

    return true;
  });

  // Calculate map bounds if we have climbs
  const mapBounds = filteredClimbs.length > 0 ? {
    minLat: Math.min(...filteredClimbs.map(c => c.lat)),
    maxLat: Math.max(...filteredClimbs.map(c => c.lat)),
    minLon: Math.min(...filteredClimbs.map(c => c.lon)),
    maxLon: Math.max(...filteredClimbs.map(c => c.lon)),
  } : null;

  return (
    <div className="h-full flex flex-col">
      {/* Main content area - scrollable */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Visualize Climbs
          </h1>
          <p className="text-gray-600">
            Load and explore climb analysis results on an interactive map
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls - left column */}
          <div className="space-y-6">
          <Card>
            <CardHeader className="cursor-pointer" onClick={() => setLoadResultsCollapsed(!loadResultsCollapsed)}>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Load Results</CardTitle>
                  <CardDescription>Select analysis files to visualize</CardDescription>
                </div>
                <button className="hover:bg-gray-100 p-1 rounded">
                  {loadResultsCollapsed ? <ChevronDown className="h-5 w-5" /> : <ChevronUp className="h-5 w-5" />}
                </button>
              </div>
            </CardHeader>
            {!loadResultsCollapsed && (
            <CardContent className="space-y-4">
              {/* Available Output Files */}
              {availableFiles.length > 0 && (
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Available Results (./output)</Label>
                  <div className="max-h-48 overflow-y-auto border rounded-lg p-2 bg-gray-50 space-y-1">
                    {availableFiles.map((file) => {
                      const isLoaded = loadedFiles.some(f => f.name === file.filename);
                      const date = new Date(file.createdAt);
                      const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

                      return (
                        <button
                          key={file.filename}
                          onClick={() => !isLoaded && !isLoading && loadFileFromOutputDir(file.filename)}
                          disabled={isLoaded || isLoading}
                          className={`w-full text-left p-2 rounded text-sm transition-colors ${
                            isLoaded
                              ? 'bg-green-100 border border-green-300 text-green-700 cursor-default'
                              : 'bg-white border border-gray-200 hover:bg-blue-50 hover:border-blue-300'
                          }`}
                        >
                          <div className="font-medium truncate">
                            {file.filename}
                            {isLoaded && <span className="ml-2 text-xs">(loaded)</span>}
                          </div>
                          <div className="text-xs text-gray-500 mt-0.5">
                            {file.region && <span className="mr-2">Region: {file.region}</span>}
                            <span>{dateStr}</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Loading Limits */}
              <div className="space-y-3 p-3 bg-gray-50 rounded-lg border">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">Load Limit (prevents slow loading)</Label>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-gray-500 w-16">Sort by:</Label>
                    <Select
                      value={loadScoreType}
                      onValueChange={(value: 'basic' | 'fiets' | 'pdi') => setLoadScoreType(value)}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="basic">Basic</SelectItem>
                        <SelectItem value="fiets">FIETS</SelectItem>
                        <SelectItem value="pdi">PDI</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-gray-500">From #</label>
                      <input
                        type="number"
                        value={loadStartN}
                        onChange={(e) => setLoadStartN(Math.max(1, parseInt(e.target.value) || 1))}
                        className="w-full px-2 py-1 text-sm border rounded"
                        min={1}
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">To #</label>
                      <input
                        type="number"
                        value={loadEndN}
                        onChange={(e) => setLoadEndN(Math.max(1, parseInt(e.target.value) || 1))}
                        className="w-full px-2 py-1 text-sm border rounded"
                        min={1}
                      />
                    </div>
                  </div>
                  <Slider
                    value={[loadStartN, loadEndN]}
                    onValueChange={(values) => {
                      setLoadStartN(values[0]);
                      setLoadEndN(values[1]);
                    }}
                    min={1}
                    max={10000}
                    step={100}
                    minStepsBetweenThumbs={1}
                  />
                  <div className="text-xs text-gray-500 text-center">
                    Load climbs #{loadStartN} to #{loadEndN} ({loadEndN - loadStartN + 1} max)
                  </div>
                </div>
              </div>

              <input
                type="file"
                id="file-input"
                className="hidden"
                accept=".csv,.xlsx"
                multiple
                onChange={handleFileSelect}
              />
              <Button
                className="w-full"
                variant="outline"
                onClick={() => document.getElementById('file-input')?.click()}
                disabled={isLoading}
              >
                <Upload className="mr-2 h-4 w-4" />
                {isLoading ? 'Loading...' : 'Select Files from Computer'}
              </Button>

              {/* Loading Progress Bar */}
              {isLoading && (
                <div className="space-y-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-medium text-blue-900">
                      {loadingStage === 'downloading' ? 'Downloading' : 'Parsing'}: {loadingFileName}
                    </div>
                    <button
                      onClick={cancelLoading}
                      className="text-xs px-2 py-1 bg-red-100 hover:bg-red-200 text-red-700 rounded transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                  <div className="w-full bg-blue-200 rounded-full h-3 overflow-hidden">
                    {loadingProgress === 0 ? (
                      /* Animated indeterminate progress bar */
                      <div
                        className="h-3 rounded-full bg-gradient-to-r from-blue-400 via-blue-600 to-blue-400 animate-pulse"
                        style={{
                          width: '100%',
                          backgroundSize: '200% 100%',
                          animation: 'shimmer 1.5s infinite linear',
                        }}
                      />
                    ) : (
                      <div
                        className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${loadingProgress}%` }}
                      />
                    )}
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-blue-700">
                      {loadingStage === 'downloading' && loadingProgress > 0
                        ? `Downloading... ${Math.round(loadingProgress * 2)}%`
                        : loadingStage === 'parsing'
                        ? 'Parsing data (this may take a moment for large files)...'
                        : 'Starting download...'}
                    </span>
                    <span className="text-blue-600 font-medium">
                      {Math.round(loadingProgress)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Add shimmer keyframes via style tag */}
              <style jsx>{`
                @keyframes shimmer {
                  0% { background-position: -200% 0; }
                  100% { background-position: 200% 0; }
                }
              `}</style>

              {/* Loaded Files List */}
              {loadedFiles.length > 0 && (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {loadedFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-2 p-2 bg-green-50 border border-green-200 rounded text-sm"
                    >
                      <CheckCircle2 className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-green-900 truncate">
                          {file.name}
                        </div>
                        <div className="text-xs text-green-700">
                          {file.climbs.length} climbs loaded
                        </div>
                      </div>
                      <button
                        onClick={() => removeFile(file.name)}
                        className="flex-shrink-0 text-green-600 hover:text-green-800"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="text-sm text-gray-500">
                {loadedFiles.length === 0 ? (
                  'No files loaded'
                ) : (
                  `${loadedFiles.length} file(s) • ${allClimbs.length} total climbs`
                )}
              </div>
            </CardContent>
            )}
          </Card>

          <Card>
            <CardHeader className="cursor-pointer" onClick={() => setFiltersCollapsed(!filtersCollapsed)}>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Filters</CardTitle>
                  <CardDescription>Adjust climb display settings</CardDescription>
                </div>
                <button className="hover:bg-gray-100 p-1 rounded">
                  {filtersCollapsed ? <ChevronDown className="h-5 w-5" /> : <ChevronUp className="h-5 w-5" />}
                </button>
              </div>
            </CardHeader>
            {!filtersCollapsed && (
            <CardContent className="space-y-4">
              {/* Score Type */}
              <div className="space-y-2">
                <Label>Score Type</Label>
                <Select
                  value={scoreType}
                  onValueChange={(value: any) => setScoreType(value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="basic">Basic</SelectItem>
                    <SelectItem value="fiets">FIETS</SelectItem>
                    <SelectItem value="pdi">PDI</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Cycling Permitted Filter */}
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="cyclingPermitted"
                  checked={cyclingPermittedOnly}
                  onChange={(e) => setCyclingPermittedOnly(e.target.checked)}
                  className="rounded w-4 h-4"
                />
                <Label htmlFor="cyclingPermitted" className="cursor-pointer text-sm">
                  Cycling Permitted Only
                </Label>
              </div>

              {/* Detail Drawer Toggle */}
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="useDetailDrawer"
                  checked={useDetailDrawer}
                  onChange={(e) => setUseDetailDrawer(e.target.checked)}
                  className="rounded w-4 h-4"
                />
                <Label htmlFor="useDetailDrawer" className="cursor-pointer text-sm">
                  Use Detail Drawer (vs inline profile)
                </Label>
              </div>

              {/* Filter Mode */}
              <div className="space-y-2">
                <Label>Show</Label>
                <Select
                  value={filterMode}
                  onValueChange={(value: any) => setFilterMode(value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="top-n">Top N Climbs</SelectItem>
                    <SelectItem value="top-percent">Top Percentage</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Range Sliders with Text Inputs */}
              {filterMode === 'top-n' ? (
                <div className="space-y-3">
                  <Label>Climb Range</Label>

                  {/* Text Inputs */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-gray-500">Start #</label>
                      <input
                        type="number"
                        value={startN}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 1;
                          setStartN(Math.max(1, Math.min(val, allClimbs.length || 500)));
                        }}
                        className="w-full px-2 py-1 text-sm border rounded"
                        min={1}
                        max={allClimbs.length || 500}
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">End #</label>
                      <input
                        type="number"
                        value={endN}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 1;
                          setEndN(Math.max(1, Math.min(val, allClimbs.length || 500)));
                        }}
                        className="w-full px-2 py-1 text-sm border rounded"
                        min={1}
                        max={allClimbs.length || 500}
                      />
                    </div>
                  </div>

                  {/* Dual-handle Range Slider */}
                  <Slider
                    value={[startN, endN]}
                    onValueChange={(values) => {
                      setStartN(values[0]);
                      setEndN(values[1]);
                    }}
                    min={1}
                    max={allClimbs.length || 500}
                    step={1}
                    minStepsBetweenThumbs={1}
                  />

                  <div className="text-xs text-gray-500 text-center">
                    Showing climbs #{startN} to #{endN} ({endN - startN + 1} climbs)
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <Label>Percentage Range</Label>

                  {/* Text Inputs */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-gray-500">Start %</label>
                      <input
                        type="number"
                        value={startPercent}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 0;
                          setStartPercent(Math.max(0, Math.min(val, 100)));
                        }}
                        className="w-full px-2 py-1 text-sm border rounded"
                        min={0}
                        max={100}
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">End %</label>
                      <input
                        type="number"
                        value={endPercent}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 0;
                          setEndPercent(Math.max(0, Math.min(val, 100)));
                        }}
                        className="w-full px-2 py-1 text-sm border rounded"
                        min={0}
                        max={100}
                      />
                    </div>
                  </div>

                  {/* Dual-handle Range Slider */}
                  <Slider
                    value={[startPercent, endPercent]}
                    onValueChange={(values) => {
                      setStartPercent(values[0]);
                      setEndPercent(values[1]);
                    }}
                    min={0}
                    max={100}
                    step={1}
                    minStepsBetweenThumbs={1}
                  />

                  <div className="text-xs text-gray-500 text-center">
                    Showing top {startPercent}% to {endPercent}%
                  </div>
                </div>
              )}
            </CardContent>
            )}
          </Card>

          <Card>
            <CardHeader className="cursor-pointer" onClick={() => setLegendCollapsed(!legendCollapsed)}>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Legend</CardTitle>
                  <CardDescription>Select categories to display</CardDescription>
                </div>
                <button className="hover:bg-gray-100 p-1 rounded">
                  {legendCollapsed ? <ChevronDown className="h-5 w-5" /> : <ChevronUp className="h-5 w-5" />}
                </button>
              </div>
            </CardHeader>
            {!legendCollapsed && (
            <CardContent className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedCategories.has('HC')}
                  onChange={() => toggleCategory('HC')}
                  className="w-4 h-4"
                />
                <div className="w-4 h-4 bg-red-900 rounded"></div>
                <span className="text-sm">HC (Hors Catégorie)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedCategories.has('Cat 1') || selectedCategories.has('1')}
                  onChange={() => toggleCategory('Cat 1')}
                  className="w-4 h-4"
                />
                <div className="w-4 h-4 bg-red-600 rounded"></div>
                <span className="text-sm">Category 1</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedCategories.has('Cat 2') || selectedCategories.has('2')}
                  onChange={() => toggleCategory('Cat 2')}
                  className="w-4 h-4"
                />
                <div className="w-4 h-4 bg-orange-500 rounded"></div>
                <span className="text-sm">Category 2</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedCategories.has('Cat 3') || selectedCategories.has('3')}
                  onChange={() => toggleCategory('Cat 3')}
                  className="w-4 h-4"
                />
                <div className="w-4 h-4 bg-orange-400 rounded"></div>
                <span className="text-sm">Category 3</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedCategories.has('Cat 4') || selectedCategories.has('4')}
                  onChange={() => toggleCategory('Cat 4')}
                  className="w-4 h-4"
                />
                <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                <span className="text-sm">Category 4</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedCategories.has('Uncategorized') || selectedCategories.has('N/A')}
                  onChange={() => toggleCategory('Uncategorized')}
                  className="w-4 h-4"
                />
                <div className="w-4 h-4 bg-gray-500 rounded"></div>
                <span className="text-sm">Uncategorized</span>
              </label>
            </CardContent>
            )}
          </Card>

          <Card>
            <CardHeader className="cursor-pointer" onClick={() => setAdvancedFiltersCollapsed(!advancedFiltersCollapsed)}>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Advanced Filters</CardTitle>
                  <CardDescription>Filter by climb properties</CardDescription>
                </div>
                <button className="hover:bg-gray-100 p-1 rounded">
                  {advancedFiltersCollapsed ? <ChevronDown className="h-5 w-5" /> : <ChevronUp className="h-5 w-5" />}
                </button>
              </div>
            </CardHeader>
            {!advancedFiltersCollapsed && (
            <CardContent className="space-y-4">
              {/* Location filter */}
              <div className="space-y-2">
                <Label>Location Filter</Label>
                <LocationFilter
                  onLocationChange={setLocationFilter}
                  initialRadius={25}
                />
              </div>

              <div className="border-t pt-4" />

              {/* Length filter */}
              <div className="space-y-2">
                <Label>Length (mi)</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Min</label>
                    <input
                      type="number"
                      value={minLength}
                      onChange={(e) => setMinLength(parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="0.1"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Max</label>
                    <input
                      type="number"
                      value={maxLength}
                      onChange={(e) => setMaxLength(parseFloat(e.target.value) || 999)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="0.1"
                    />
                  </div>
                </div>
              </div>

              {/* Elevation Gain filter */}
              <div className="space-y-2">
                <Label>Elevation Gain (ft)</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Min</label>
                    <input
                      type="number"
                      value={minElevGain}
                      onChange={(e) => setMinElevGain(parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="10"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Max</label>
                    <input
                      type="number"
                      value={maxElevGain}
                      onChange={(e) => setMaxElevGain(parseFloat(e.target.value) || 99999)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="10"
                    />
                  </div>
                </div>
              </div>

              {/* Prominence filter */}
              <div className="space-y-2">
                <Label>Prominence (ft)</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Min</label>
                    <input
                      type="number"
                      value={minProminence}
                      onChange={(e) => setMinProminence(parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="10"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Max</label>
                    <input
                      type="number"
                      value={maxProminence}
                      onChange={(e) => setMaxProminence(parseFloat(e.target.value) || 99999)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="10"
                    />
                  </div>
                </div>
                <Slider
                  value={[minProminence, maxProminence]}
                  onValueChange={(values) => {
                    setMinProminence(values[0]);
                    setMaxProminence(values[1]);
                  }}
                  min={0}
                  max={10000}
                  step={10}
                  minStepsBetweenThumbs={10}
                />
              </div>

              {/* Height filter */}
              <div className="space-y-2">
                <Label>Height (ft)</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Min</label>
                    <input
                      type="number"
                      value={minHeight}
                      onChange={(e) => setMinHeight(parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="10"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Max</label>
                    <input
                      type="number"
                      value={maxHeight}
                      onChange={(e) => setMaxHeight(parseFloat(e.target.value) || 99999)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="10"
                    />
                  </div>
                </div>
                <Slider
                  value={[minHeight, maxHeight]}
                  onValueChange={(values) => {
                    setMinHeight(values[0]);
                    setMaxHeight(values[1]);
                  }}
                  min={0}
                  max={15000}
                  step={10}
                  minStepsBetweenThumbs={10}
                />
              </div>

              {/* Average Grade filter */}
              <div className="space-y-2">
                <Label>Avg Grade (%)</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Min</label>
                    <input
                      type="number"
                      value={minAvgGrade}
                      onChange={(e) => setMinAvgGrade(parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="0.1"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Max</label>
                    <input
                      type="number"
                      value={maxAvgGrade}
                      onChange={(e) => setMaxAvgGrade(parseFloat(e.target.value) || 100)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="0.1"
                    />
                  </div>
                </div>
                <Slider
                  value={[minAvgGrade, maxAvgGrade]}
                  onValueChange={(values) => {
                    setMinAvgGrade(values[0]);
                    setMaxAvgGrade(values[1]);
                  }}
                  min={0}
                  max={30}
                  step={0.1}
                  minStepsBetweenThumbs={0.1}
                />
              </div>

              {/* Max Grade filter */}
              <div className="space-y-2">
                <Label>Max Grade (%)</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Min</label>
                    <input
                      type="number"
                      value={minMaxGrade}
                      onChange={(e) => setMinMaxGrade(parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="0.1"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Max</label>
                    <input
                      type="number"
                      value={maxMaxGrade}
                      onChange={(e) => setMaxMaxGrade(parseFloat(e.target.value) || 100)}
                      className="w-full px-2 py-1 text-sm border rounded"
                      step="0.1"
                    />
                  </div>
                </div>
                <Slider
                  value={[minMaxGrade, maxMaxGrade]}
                  onValueChange={(values) => {
                    setMinMaxGrade(values[0]);
                    setMaxMaxGrade(values[1]);
                  }}
                  min={0}
                  max={50}
                  step={0.1}
                  minStepsBetweenThumbs={0.1}
                />
              </div>
            </CardContent>
            )}
          </Card>
        </div>

        {/* Map */}
        <div className="lg:col-span-3">
          <Card style={{ height: selectedClimb ? 'calc(65vh - 9rem)' : 'calc(100vh - 12rem)' }}>
            <CardContent className="p-0 h-full">
              {allClimbs.length === 0 ? (
                <div className="w-full h-full bg-gray-200 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <Map className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 mb-2">Map will be displayed here</p>
                    <p className="text-sm text-gray-500">
                      Select analysis files to visualize climbs
                    </p>
                    <p className="text-xs text-gray-400 mt-4">
                      Using MapLibre GL JS for interactive mapping
                    </p>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full flex flex-col">
                  {/* Header with view toggle */}
                  <div className="p-4 bg-white border-b">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-semibold text-gray-900">
                          Showing {filteredClimbs.length} of {allClimbs.length} climbs
                          {filterMode === 'top-n'
                            ? ` (Climbs #${startN}-${endN} by ${scoreType} score)`
                            : ` (Top ${startPercent}%-${endPercent}% by ${scoreType} score)`
                          }
                        </h3>
                        {viewMode === 'map' && mapBounds && (
                          <p className="text-sm text-gray-600">
                            Area: {mapBounds.minLat.toFixed(4)}° to {mapBounds.maxLat.toFixed(4)}°N,
                            {' '}{mapBounds.minLon.toFixed(4)}° to {mapBounds.maxLon.toFixed(4)}°E
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-4">
                        {/* View Toggle */}
                        <div className="flex items-center bg-gray-100 rounded-lg p-1">
                          <button
                            onClick={() => setViewMode('map')}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                              viewMode === 'map'
                                ? 'bg-white text-gray-900 shadow-sm'
                                : 'text-gray-600 hover:text-gray-900'
                            }`}
                          >
                            <Map className="h-4 w-4" />
                            Map
                          </button>
                          <button
                            onClick={() => setViewMode('list')}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                              viewMode === 'list'
                                ? 'bg-white text-gray-900 shadow-sm'
                                : 'text-gray-600 hover:text-gray-900'
                            }`}
                          >
                            <List className="h-4 w-4" />
                            List
                          </button>
                        </div>
                        {/* Map-specific controls */}
                        {viewMode === 'map' && (
                          <label className="flex items-center gap-2 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={showAllRoutes}
                              onChange={(e) => setShowAllRoutes(e.target.checked)}
                              className="w-4 h-4"
                            />
                            <span className="text-sm text-gray-700">Show all routes</span>
                          </label>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Content area - Map or List */}
                  <div className="flex-1">
                    {viewMode === 'map' ? (
                      <ClimbMap
                        climbs={filteredClimbs}
                        allClimbs={allClimbs}
                        bounds={mapBounds || undefined}
                        showAllRoutes={showAllRoutes}
                        scoreType={scoreType}
                        onClimbClick={setSelectedClimb}
                      />
                    ) : (
                      <ClimbList
                        climbs={filteredClimbs}
                        scoreType={scoreType}
                        selectedClimb={selectedClimb}
                        onClimbClick={setSelectedClimb}
                      />
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Elevation Profile - directly below map, same column (only when not using drawer) */}
          {selectedClimb && !useDetailDrawer && (
            <div className="mt-6">
              <ElevationProfile
                climb={selectedClimb}
                onClose={() => setSelectedClimb(null)}
              />
            </div>
          )}
        </div>
        </div>
      </div>

      {/* Detail Drawer - fixed at bottom (when enabled) */}
      {useDetailDrawer && (
        <ClimbDetailDrawer
          climb={selectedClimb}
          allClimbs={allClimbs}
          onClose={() => setSelectedClimb(null)}
          onClimbSelect={setSelectedClimb}
        />
      )}
    </div>
  );
}

// Wrap with Suspense for useSearchParams
export default function VisualizePage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-full">Loading...</div>}>
      <VisualizePageContent />
    </Suspense>
  );
}
