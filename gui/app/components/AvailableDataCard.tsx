'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Database, Loader2, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import { getFriendlyName, getContinentColors } from '@/app/lib/regionUtils';

interface DataInfo {
  osmPlanetData: string[];
  osmIndexes: string[];
  elevationData: string[];
  elevationDatasets: { [key: string]: string[] };
  analyzedReports: Array<{
    filename: string;
    region: string;
    canonical_path: string;
  }>;
}

interface AvailableDataCardProps {
  onResync?: () => void;
  isResyncing?: boolean;
  resyncMessage?: string;
}

export default function AvailableDataCard({ onResync, isResyncing, resyncMessage }: AvailableDataCardProps) {
  const [dataInfo, setDataInfo] = useState<DataInfo | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(true);

  // Dataset order for elevation data
  const DATASET_ORDER = ['ned10m', 'srtm30m', 'arctic32m', 'rema32m', 'aw3d30', 'aster30m'];

  useEffect(() => {
    loadDataInfo();
  }, []);

  const loadDataInfo = () => {
    setIsLoadingData(true);
    fetch('/api/data-info')
      .then(res => res.json())
      .then(data => {
        setDataInfo(data);
        setIsLoadingData(false);
      })
      .catch(err => {
        console.error('Failed to fetch data info:', err);
        setIsLoadingData(false);
      });
  };

  // Reload data when resync completes
  useEffect(() => {
    if (!isResyncing && resyncMessage?.includes('✓')) {
      loadDataInfo();
    }
  }, [isResyncing, resyncMessage]);

  const handleResync = async () => {
    if (onResync) {
      await onResync();
      // Reload data after resyncing
      loadDataInfo();
    }
  };

  // Sort elevation datasets by preferred order
  const getSortedDatasets = () => {
    if (!dataInfo?.elevationDatasets) return [];

    const datasets = Object.entries(dataInfo.elevationDatasets);
    return datasets.sort((a, b) => {
      const indexA = DATASET_ORDER.indexOf(a[0]);
      const indexB = DATASET_ORDER.indexOf(b[0]);

      // If both are in the preferred order, sort by index
      if (indexA !== -1 && indexB !== -1) {
        return indexA - indexB;
      }

      // If only one is in the preferred order, it comes first
      if (indexA !== -1) return -1;
      if (indexB !== -1) return 1;

      // Otherwise, sort alphabetically
      return a[0].localeCompare(b[0]);
    });
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-blue-600" />
            <CardTitle>Available Data</CardTitle>
          </div>
          {onResync && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleResync}
              disabled={isResyncing}
            >
              {isResyncing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Syncing...
                </>
              ) : (
                'Resync Config'
              )}
            </Button>
          )}
        </div>
        <CardDescription>
          Downloaded data files and analyzed climb reports
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {resyncMessage && (
          <div className={`text-sm p-3 rounded-lg ${
            resyncMessage.includes('✓') ? 'bg-green-50 text-green-800 border border-green-200' :
            resyncMessage.includes('❌') ? 'bg-red-50 text-red-800 border border-red-200' :
            'bg-blue-50 text-blue-800 border border-blue-200'
          }`}>
            {resyncMessage}
          </div>
        )}

        {isLoadingData ? (
          <p className="text-sm text-gray-500">Loading...</p>
        ) : dataInfo ? (
          <div className="space-y-4">
            {/* Analyzed Climb Reports */}
            <div>
              <h4 className="text-xs font-semibold text-gray-600 uppercase mb-2">Analyzed Climb Reports</h4>
              {dataInfo.analyzedReports && dataInfo.analyzedReports.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {dataInfo.analyzedReports.map(report => {
                    const colors = getContinentColors(report.canonical_path);
                    const friendlyName = getFriendlyName(report.canonical_path);

                    return (
                      <Link
                        key={report.filename}
                        href={`/visualize?file=${encodeURIComponent(report.filename)}`}
                        className="group"
                      >
                        <span className={`flex items-center gap-1 px-2 py-1 ${colors.bg} ${colors.text} text-xs rounded hover:opacity-80 transition-opacity`}>
                          {friendlyName}
                          <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </span>
                      </Link>
                    );
                  })}
                </div>
              ) : (
                <p className="text-xs text-gray-500">None</p>
              )}
            </div>

            {/* OSM Planet Data */}
            <div>
              <h4 className="text-xs font-semibold text-gray-600 uppercase mb-2">OSM Planet Files</h4>
              {dataInfo.osmPlanetData.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {dataInfo.osmPlanetData.map(file => {
                    const friendlyName = file.replace('.osm.pbf', '').replace('-latest', '');
                    const colors = getContinentColors('unknown'); // OSM files don't have canonical paths

                    return (
                      <span key={file} className={`px-2 py-1 ${colors.bg} ${colors.text} text-xs rounded`}>
                        {getFriendlyName(friendlyName)}
                      </span>
                    );
                  })}
                </div>
              ) : (
                <p className="text-xs text-gray-500">None</p>
              )}
            </div>

            {/* OSM Indexes */}
            <div>
              <h4 className="text-xs font-semibold text-gray-600 uppercase mb-2">OSM Indexes</h4>
              {dataInfo.osmIndexes.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {dataInfo.osmIndexes.map(index => {
                    const friendlyName = index.replace('-latest', '');
                    const colors = getContinentColors('unknown'); // OSM indexes don't have canonical paths

                    return (
                      <span key={index} className={`px-2 py-1 ${colors.bg} ${colors.text} text-xs rounded`}>
                        {getFriendlyName(friendlyName)}
                      </span>
                    );
                  })}
                </div>
              ) : (
                <p className="text-xs text-gray-500">None</p>
              )}
            </div>

            {/* Elevation Data */}
            <div>
              <h4 className="text-xs font-semibold text-gray-600 uppercase mb-2">Elevation Data</h4>
              {getSortedDatasets().length > 0 ? (
                <div className="space-y-2">
                  {getSortedDatasets().map(([dataset, regions]) => (
                    <div key={dataset}>
                      <p className="text-xs font-medium text-gray-600 mb-1">{dataset}:</p>
                      <div className="flex flex-wrap gap-1">
                        {(regions as string[]).map(region => {
                          const colors = getContinentColors(region);
                          const friendlyName = getFriendlyName(region);

                          return (
                            <span key={`${dataset}-${region}`} className={`px-2 py-1 ${colors.bg} ${colors.text} text-xs rounded`}>
                              {friendlyName}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-gray-500">None</p>
              )}
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-500">No data available</p>
        )}
      </CardContent>
    </Card>
  );
}
