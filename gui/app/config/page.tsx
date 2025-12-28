'use client';

import { useState, useEffect } from 'react';
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
import { Save, Settings as SettingsIcon, Loader2, RefreshCw, Key, Trash2 } from 'lucide-react';
import { Config, DeploymentType } from '@/types/climb';

export default function ConfigPage() {
  const [config, setConfig] = useState<Config>({
    deploymentType: 'cloud',
    elevationBatchSize: 100,
    elevationMaxConcurrent: 2,
    checkpointIntervalMin: 15,
    minClimbLengthFt: 100,
    elevationDelayBetweenBatchesSec: 0,
    elevationRequestTimeoutSec: 45,
    elevationMaxRetries: 3,
    elevationBackoffFactor: 2.0,
    elevationDatasetTiers: 'primary+secondary+tertiary',
    geocodingMaxConcurrent: 8,
    geocodingRetryAttempts: 2,
    osmChunkSizeKm: 30.0,
    cloudCacheEnabled: true,
  });

  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isUpdatingGeo, setIsUpdatingGeo] = useState(false);
  const [geoUpdateMessage, setGeoUpdateMessage] = useState('');
  const [geoBoundariesDate, setGeoBoundariesDate] = useState<string>('Loading...');

  // OpenTopoData rebuild/restart state
  const [isRebuildingOTD, setIsRebuildingOTD] = useState(false);
  const [isRestartingOTD, setIsRestartingOTD] = useState(false);
  const [otdMessage, setOtdMessage] = useState('');

  // EarthData credentials state
  const [earthdataUsername, setEarthdataUsername] = useState('');
  const [earthdataPassword, setEarthdataPassword] = useState('');
  const [hasEarthdataCredentials, setHasEarthdataCredentials] = useState(false);
  const [isSavingCredentials, setIsSavingCredentials] = useState(false);

  // Load config from API on mount
  useEffect(() => {
    fetch('/api/config')
      .then(res => res.json())
      .then(data => {
        setConfig(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to load config:', err);
        setIsLoading(false);
      });

    // Fetch geo boundaries date
    fetch('/api/geo-boundaries-info')
      .then(res => res.json())
      .then(data => {
        setGeoBoundariesDate(data.date || 'Unknown');
      })
      .catch(err => {
        console.error('Failed to fetch geo boundaries date:', err);
        setGeoBoundariesDate('Unknown');
      });

    // Load EarthData credentials
    fetch('/api/earthdata-credentials')
      .then(res => res.json())
      .then(data => {
        if (data.hasCredentials) {
          setHasEarthdataCredentials(true);
          setEarthdataUsername(data.username || '');
          setEarthdataPassword(data.password || '');
        }
      })
      .catch(err => {
        console.error('Failed to load EarthData credentials:', err);
      });
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const response = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error('Failed to save configuration');
      }

      const result = await response.json();
      alert(result.message || 'Configuration saved successfully!');
    } catch (error) {
      console.error('Failed to save config:', error);
      alert('Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  };

  const handleUpdateGeoBoundaries = async () => {
    if (!confirm('Update geographic boundary data from sources? This may take a few minutes.')) {
      return;
    }

    setIsUpdatingGeo(true);
    setGeoUpdateMessage('Updating geographic boundaries...');

    try {
      const response = await fetch('/api/update-geo', {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to update geographic boundaries');
      }

      const result = await response.json();
      setGeoUpdateMessage('✓ ' + result.message);

      // Refresh the geo boundaries date
      fetch('/api/geo-boundaries-info')
        .then(res => res.json())
        .then(data => {
          setGeoBoundariesDate(data.date || 'Unknown');
        })
        .catch(err => console.error('Failed to refresh date:', err));

      setTimeout(() => setGeoUpdateMessage(''), 5000);
    } catch (error) {
      console.error('Failed to update geo boundaries:', error);
      setGeoUpdateMessage('❌ Error: ' + String(error));
    } finally {
      setIsUpdatingGeo(false);
    }
  };

  const handleSaveEarthdataCredentials = async () => {
    if (!earthdataUsername || !earthdataPassword) {
      alert('Please enter both username and password');
      return;
    }

    setIsSavingCredentials(true);
    try {
      const response = await fetch('/api/earthdata-credentials', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: earthdataUsername,
          password: earthdataPassword,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to save credentials');
      }

      setHasEarthdataCredentials(true);
      alert('EarthData credentials saved successfully!');
    } catch (error) {
      console.error('Failed to save credentials:', error);
      alert('Failed to save EarthData credentials: ' + String(error));
    } finally {
      setIsSavingCredentials(false);
    }
  };

  const handleRemoveEarthdataCredentials = async () => {
    if (!confirm('Remove EarthData credentials from .netrc file?')) {
      return;
    }

    try {
      const response = await fetch('/api/earthdata-credentials', {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to remove credentials');
      }

      setHasEarthdataCredentials(false);
      setEarthdataUsername('');
      setEarthdataPassword('');
      alert('EarthData credentials removed successfully!');
    } catch (error) {
      console.error('Failed to remove credentials:', error);
      alert('Failed to remove EarthData credentials');
    }
  };

  const handleRebuildOpenTopoData = async () => {
    if (!confirm('Rebuild OpenTopoData server? This will rebuild the Docker image and may take several minutes.')) {
      return;
    }

    setIsRebuildingOTD(true);
    setOtdMessage('Rebuilding OpenTopoData server...');

    try {
      const response = await fetch('/api/rebuild-opentopodata', {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to rebuild OpenTopoData');
      }

      const result = await response.json();
      setOtdMessage('✓ ' + result.message);
      setTimeout(() => setOtdMessage(''), 5000);
    } catch (error) {
      console.error('Failed to rebuild OpenTopoData:', error);
      setOtdMessage('❌ Error: ' + String(error));
    } finally {
      setIsRebuildingOTD(false);
    }
  };

  const handleRestartOpenTopoData = async () => {
    if (!confirm('Restart OpenTopoData server? This will quickly restart the container without rebuilding.')) {
      return;
    }

    setIsRestartingOTD(true);
    setOtdMessage('Restarting OpenTopoData server...');

    try {
      const response = await fetch('/api/rebuild-opentopodata', {
        method: 'PUT',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to restart OpenTopoData');
      }

      const result = await response.json();
      setOtdMessage('✓ ' + result.message);
      setTimeout(() => setOtdMessage(''), 5000);
    } catch (error) {
      console.error('Failed to restart OpenTopoData:', error);
      setOtdMessage('❌ Error: ' + String(error));
    } finally {
      setIsRestartingOTD(false);
    }
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Configuration</h1>
        <p className="text-gray-600">
          Manage deployment mode and analysis settings
        </p>
      </div>

      <div className="max-w-2xl space-y-6">
        {/* Main Configuration Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <SettingsIcon className="h-5 w-5 text-gray-600" />
              <CardTitle>System Configuration</CardTitle>
            </div>
            <CardDescription>
              These settings control how the climb analyzer operates
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Deployment Type */}
            <div className="space-y-2">
              <Label>Deployment Mode</Label>
              <Select
                value={config.deploymentType}
                onValueChange={(value: DeploymentType) => {
                  const newConfig = { ...config, deploymentType: value };
                  // When switching to local mode, auto-set max concurrent to CPU cores
                  if (value === 'local' && config.cpuCores) {
                    newConfig.elevationMaxConcurrent = config.cpuCores;
                  }
                  // When switching to cloud mode, reset to 2
                  if (value === 'cloud') {
                    newConfig.elevationMaxConcurrent = 2;
                  }
                  setConfig(newConfig);
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cloud">Cloud (Overpass API)</SelectItem>
                  <SelectItem value="local">Local (OSM + DEM Files)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-sm text-gray-500">
                {config.deploymentType === 'cloud'
                  ? 'Uses Overpass API for OSM data - no local planet files required'
                  : `Uses local planet files for OSM data - faster for large regions${config.cpuCores ? ` (${config.cpuCores} CPU cores detected)` : ''}`}
              </p>
            </div>

            {/* Elevation Dataset Tiers (for local mode) */}
            {config.deploymentType === 'local' && (
              <div className="space-y-2">
                <Label>Elevation Dataset Tiers</Label>
                <Select
                  value={config.elevationDatasetTiers}
                  onValueChange={(value: string) =>
                    setConfig({ ...config, elevationDatasetTiers: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="primary">Primary Only (~5-50 GB)</SelectItem>
                    <SelectItem value="primary+secondary">Primary + Secondary (~10-100 GB)</SelectItem>
                    <SelectItem value="primary+secondary+tertiary">Primary + Secondary + Tertiary (~15-150 GB)</SelectItem>
                  </SelectContent>
                </Select>
                <div className="text-sm text-gray-500 space-y-1">
                  <p>• <strong>Primary:</strong> SRTM 30m (OpenTopography S3, best accuracy for 60°N to 56°S)</p>
                  <p>• <strong>Secondary:</strong> AW3D30 (JAXA FTP, global coverage 84°N to 84°S)</p>
                  <p className="mt-2 text-xs text-green-600">
                    ✓ All datasets use public sources - no authentication required
                  </p>
                </div>
              </div>
            )}

            {/* Elevation Batch Size */}
            <div className="space-y-2">
              <Label htmlFor="batchSize">Elevation Batch Size</Label>
              <Input
                id="batchSize"
                type="number"
                value={config.elevationBatchSize}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    elevationBatchSize: parseInt(e.target.value),
                  })
                }
              />
              <p className="text-sm text-gray-500">
                Number of elevation points to fetch per request (default: 100)
              </p>
            </div>

            {/* Max Concurrent */}
            <div className="space-y-2">
              <Label htmlFor="maxConcurrent">Max Concurrent Requests</Label>
              <Input
                id="maxConcurrent"
                type="number"
                value={config.elevationMaxConcurrent}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    elevationMaxConcurrent: parseInt(e.target.value),
                  })
                }
              />
              <p className="text-sm text-gray-500">
                Maximum concurrent elevation requests
                {config.deploymentType === 'cloud'
                  ? ' (recommended: 1-2 for cloud)'
                  : ' (can use higher values for local)'}
              </p>
            </div>

            {/* Checkpoint Interval */}
            <div className="space-y-2">
              <Label htmlFor="checkpointInterval">
                Checkpoint Interval (minutes)
              </Label>
              <Input
                id="checkpointInterval"
                type="number"
                value={config.checkpointIntervalMin}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    checkpointIntervalMin: parseInt(e.target.value),
                  })
                }
              />
              <p className="text-sm text-gray-500">
                How often to save analysis progress (default: 15 minutes)
              </p>
            </div>

            {/* Info Box */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-900 mb-2">
                Configuration Notes
              </h4>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Cloud mode is easier to get running (no setup required)</li>
                <li>
                  • Local mode is recommended for analyzing more than 25 mi (40 km) per day and is much faster
                </li>
                <li>
                  • Local mode requires downloading planet files first
                </li>
                <li>
                  • Lower concurrent requests if you experience rate limiting
                </li>
                <li>• Checkpoints allow resuming interrupted analyses</li>
              </ul>
            </div>

            {/* Update Geographic Boundaries Button */}
            <div className="space-y-2">
              <Button
                onClick={handleUpdateGeoBoundaries}
                disabled={isUpdatingGeo}
                variant="outline"
                className="w-full"
              >
                {isUpdatingGeo ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Updating Geographic Boundaries...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Update Geographic Boundaries
                  </>
                )}
              </Button>
              {geoUpdateMessage && (
                <p className="text-sm text-center text-gray-700">
                  {geoUpdateMessage}
                </p>
              )}
              <p className="text-xs text-gray-500">
                Updates country and state boundary data from OSM sources
              </p>
              <p className="text-xs text-gray-600 text-center">
                Region boundaries last fetched: <span className="font-medium">{geoBoundariesDate}</span>
              </p>
            </div>

            {/* Save Button */}
            <Button
              onClick={handleSave}
              disabled={isSaving}
              className="w-full"
            >
              {isSaving ? (
                'Saving...'
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4" />
                  Save Configuration
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Elevation Data Sources Card (replaced EarthData Credentials) */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Key className="h-5 w-5 text-green-600" />
              <CardTitle>Elevation Data Sources</CardTitle>
            </div>
            <CardDescription>
              All elevation datasets now use public sources - no credentials required!
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-sm text-green-800 font-medium mb-2">
                ✓ No authentication required
              </p>
              <p className="text-xs text-green-700">
                As of December 2025, all elevation datasets use public sources:
              </p>
              <ul className="text-xs text-green-700 mt-2 space-y-1 ml-4 list-disc">
                <li><strong>SRTM 30m</strong> - OpenTopography S3 (public)</li>
                <li><strong>AW3D30</strong> - JAXA FTP (public)</li>
                <li><strong>NED 10m</strong> - AWS S3 (public)</li>
                <li><strong>ArcticDEM</strong> - AWS S3 (public)</li>
                <li><strong>REMA</strong> - AWS S3 (public)</li>
              </ul>
            </div>

            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="text-xs text-gray-600">
                <strong>Note:</strong> NASA LP DAAC Data Pool was retired in December 2025.
                SRTM data is now served from OpenTopography&apos;s public S3 bucket.
                ASTER has been deprecated (AW3D30 provides better coverage and accuracy).
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Cloud Cache Settings Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <SettingsIcon className="h-5 w-5 text-gray-600" />
              <CardTitle>Cloud Cache Settings</CardTitle>
            </div>
            <CardDescription>
              Share your analyses with the community via GitHub pull requests
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="cloudCacheEnabled"
                checked={config.cloudCacheEnabled}
                onChange={(e) =>
                  setConfig({ ...config, cloudCacheEnabled: e.target.checked })
                }
                className="rounded"
              />
              <Label htmlFor="cloudCacheEnabled" className="cursor-pointer">
                Enable Cloud Cache Upload
              </Label>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-sm text-blue-900 font-medium mb-2">
                How Cloud Cache Works
              </p>
              <ul className="text-xs text-blue-800 space-y-1">
                <li>• When enabled, clean analyses are automatically uploaded to the cloud cache</li>
                <li>• Creates a pull request to share your analysis with the community</li>
                <li>• Other users can download your pre-computed analyses</li>
                <li>• Only "clean" analyses (no filters, default settings) are uploaded</li>
                <li>• You can disable this per-analysis on the Run Analysis page</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* OpenTopoData Management Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <RefreshCw className="h-5 w-5 text-gray-600" />
              <CardTitle>OpenTopoData Server</CardTitle>
            </div>
            <CardDescription>
              Manage the elevation data server (rebuild after downloading new DEM files)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-xs text-blue-800">
                <strong>Rebuild:</strong> Rebuilds Docker image with latest elevation data (use after downloading new DEM files)
              </p>
              <p className="text-xs text-blue-800 mt-1">
                <strong>Restart:</strong> Quick restart of the server without rebuilding
              </p>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={handleRebuildOpenTopoData}
                disabled={isRebuildingOTD || isRestartingOTD}
                variant="default"
                className="flex-1"
              >
                {isRebuildingOTD ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Rebuilding...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Rebuild Server
                  </>
                )}
              </Button>

              <Button
                onClick={handleRestartOpenTopoData}
                disabled={isRebuildingOTD || isRestartingOTD}
                variant="outline"
                className="flex-1"
              >
                {isRestartingOTD ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Restarting...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Quick Restart
                  </>
                )}
              </Button>
            </div>

            {otdMessage && (
              <p className="text-sm text-center text-gray-700">
                {otdMessage}
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
