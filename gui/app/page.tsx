'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Play, Map, Download, Settings, FileText, Cloud } from 'lucide-react';
import CloudCacheTree from './components/CloudCacheTree';
import AvailableDataCard from './components/AvailableDataCard';

interface OutputFile {
  filename: string;
  region: string;
  createdAt: string;
}

export default function Home() {
  const [deploymentMode, setDeploymentMode] = useState<string>('Loading...');
  const [outputFiles, setOutputFiles] = useState<OutputFile[]>([]);
  const [isLoadingOutputFiles, setIsLoadingOutputFiles] = useState(true);

  useEffect(() => {
    // Fetch deployment mode from config
    fetch('/api/config')
      .then(res => res.json())
      .then(data => {
        const mode = data.deploymentType === 'local' ? 'Local' : 'Cloud';
        setDeploymentMode(mode);
      })
      .catch(err => {
        console.error('Failed to fetch config:', err);
        setDeploymentMode('Unknown');
      });

    // Fetch output files separately (slower - scans directory)
    fetch('/api/output-files')
      .then(res => res.json())
      .then(data => {
        setOutputFiles(data.outputFiles || []);
        setIsLoadingOutputFiles(false);
      })
      .catch(err => {
        console.error('Failed to fetch output files:', err);
        setIsLoadingOutputFiles(false);
      });
  }, []);

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Welcome to Climb Analyzer
        </h1>
        <p className="text-gray-600">
          Analyze and visualize road climbs from OpenStreetMap data
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-center gap-2">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Play className="h-6 w-6 text-blue-600" />
              </div>
              <CardTitle>Run Analysis</CardTitle>
            </div>
            <CardDescription>
              Start a new climb analysis for an address or region
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/analyze">
              <Button className="w-full">Get Started</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-center gap-2">
              <div className="p-2 bg-green-100 rounded-lg">
                <Map className="h-6 w-6 text-green-600" />
              </div>
              <CardTitle>Visualize Climbs</CardTitle>
            </div>
            <CardDescription>
              View and explore your climb analysis results on a map
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/visualize">
              <Button variant="outline" className="w-full">View Map</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-center gap-2">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Download className="h-6 w-6 text-purple-600" />
              </div>
              <CardTitle>Download Data</CardTitle>
            </div>
            <CardDescription>
              Download OSM and elevation data for offline analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/download">
              <Button variant="outline" className="w-full">Download</Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* Stats */}
      <div className="max-w-sm mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Deployment Mode
            </CardTitle>
            <Settings className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{deploymentMode}</div>
            <p className="text-xs text-gray-500 mt-1">
              <Link href="/config" className="text-blue-600 hover:underline">
                Change settings
              </Link>
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Data Coverage */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <AvailableDataCard />

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-green-600" />
              <CardTitle>Analysis Results</CardTitle>
            </div>
            <CardDescription>
              Completed analyses in output folder
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingOutputFiles ? (
              <p className="text-sm text-gray-500">Loading...</p>
            ) : outputFiles.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {outputFiles.map((file, idx) => (
                  <Link
                    key={idx}
                    href={`/visualize?file=${encodeURIComponent(file.filename)}`}
                    className="flex justify-between items-center p-2 bg-gray-50 hover:bg-gray-100 rounded text-xs cursor-pointer transition-colors"
                  >
                    <div>
                      <p className="font-medium text-gray-800">{file.region}</p>
                      <p className="text-gray-500">{file.filename}</p>
                    </div>
                    <p className="text-gray-400 text-xs">
                      {new Date(file.createdAt).toLocaleDateString()}
                    </p>
                  </Link>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No analysis results yet</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Cloud className="h-5 w-5 text-blue-500" />
              <CardTitle>Cloud Cache Data</CardTitle>
            </div>
            <CardDescription>
              Pre-analyzed climb data from GitHub
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CloudCacheTree showDownload={false} />
          </CardContent>
        </Card>
      </div>

      {/* Getting Started */}
      <Card>
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>
            Quick guide to using Climb Analyzer
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Link href="/config" className="flex gap-4 hover:bg-gray-50 p-3 rounded-lg transition-colors">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
              1
            </div>
            <div>
              <h3 className="font-semibold mb-1">Configure Your Setup</h3>
              <p className="text-sm text-gray-600">
                Choose between Cloud (Overpass API) or Local (planet files) deployment mode in Configuration.
              </p>
            </div>
          </Link>

          <Link href="/download" className="flex gap-4 hover:bg-gray-50 p-3 rounded-lg transition-colors">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
              2
            </div>
            <div>
              <h3 className="font-semibold mb-1">Download Data (Optional)</h3>
              <p className="text-sm text-gray-600">
                Pre-download OSM and elevation data for faster offline analysis, or let the analyzer download it automatically.
              </p>
            </div>
          </Link>

          <Link href="/analyze" className="flex gap-4 hover:bg-gray-50 p-3 rounded-lg transition-colors">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
              3
            </div>
            <div>
              <h3 className="font-semibold mb-1">Run Your First Analysis</h3>
              <p className="text-sm text-gray-600">
                Go to Run Analysis, enter an address or region, choose your parameters, and let the analyzer discover climbs.
              </p>
            </div>
          </Link>

          <Link href="/visualize" className="flex gap-4 hover:bg-gray-50 p-3 rounded-lg transition-colors">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
              4
            </div>
            <div>
              <h3 className="font-semibold mb-1">Visualize Results</h3>
              <p className="text-sm text-gray-600">
                View your climbs on an interactive map, filter by category, and explore detailed climb attributes.
              </p>
            </div>
          </Link>

          <div className="flex gap-4 hover:bg-gray-50 p-3 rounded-lg transition-colors">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
              5
            </div>
            <div>
              <h3 className="font-semibold mb-1">View Climbs On The Go</h3>
              <p className="text-sm text-gray-600">
                Download the iOS Climb Analyzer app to view and navigate your climbs on your iPhone or iPad while riding.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
