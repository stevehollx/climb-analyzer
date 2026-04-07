'use client';

import { useState } from 'react';
import { Partition } from '@/types/climb';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, Download, CheckCircle2, MapPin } from 'lucide-react';

interface PartitionSelectorProps {
  regionName: string;
  partitions: Partition[];
  loadedPartitions: string[];  // List of partition_ids already loaded
  onLoadPartition: (partition: Partition) => Promise<void>;
  onLoadAll?: () => Promise<void>;
  isLoading?: boolean;
  loadingPartitionId?: string | null;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function PartitionSelector({
  regionName,
  partitions,
  loadedPartitions,
  onLoadPartition,
  onLoadAll,
  isLoading = false,
  loadingPartitionId = null,
}: PartitionSelectorProps) {
  const totalSize = partitions.reduce((sum, p) => sum + p.database_size, 0);
  const loadedCount = loadedPartitions.length;
  const totalCount = partitions.length;

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <MapPin className="h-5 w-5" />
          {regionName} - Select Partitions
        </CardTitle>
        <CardDescription>
          This region is split into {totalCount} geographic partitions.
          Select the areas you want to explore.
          <br />
          <span className="text-xs text-gray-500">
            Total: {formatBytes(totalSize)} | Loaded: {loadedCount}/{totalCount}
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Partition list */}
        <div className="grid gap-2">
          {partitions.map((partition) => {
            const isLoaded = loadedPartitions.includes(partition.partition_id);
            const isLoadingThis = loadingPartitionId === partition.partition_id;

            return (
              <button
                key={partition.partition_id}
                onClick={() => !isLoaded && !isLoading && onLoadPartition(partition)}
                disabled={isLoaded || isLoading}
                className={`w-full text-left p-3 rounded-lg border transition-colors ${
                  isLoaded
                    ? 'bg-green-50 border-green-200 cursor-default'
                    : isLoadingThis
                    ? 'bg-blue-50 border-blue-300'
                    : 'bg-white border-gray-200 hover:bg-blue-50 hover:border-blue-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 flex items-center gap-2">
                      {partition.display_name}
                      {isLoaded && (
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      )}
                    </div>
                    <div className="text-sm text-gray-500 mt-0.5">
                      {formatBytes(partition.database_size)}
                      {partition.climb_count && (
                        <span className="ml-2">
                          ({partition.climb_count.toLocaleString()} climbs)
                        </span>
                      )}
                    </div>
                  </div>
                  {isLoadingThis && (
                    <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
                  )}
                  {!isLoaded && !isLoadingThis && (
                    <Download className="h-5 w-5 text-gray-400" />
                  )}
                </div>
              </button>
            );
          })}
        </div>

        {/* Load All button */}
        {onLoadAll && loadedCount < totalCount && (
          <div className="pt-2 border-t">
            <Button
              variant="outline"
              className="w-full"
              onClick={onLoadAll}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <Download className="mr-2 h-4 w-4" />
                  Load All Partitions ({formatBytes(totalSize)})
                </>
              )}
            </Button>
            <p className="text-xs text-gray-500 text-center mt-2">
              Downloads all partitions sequentially. May take a while for large regions.
            </p>
          </div>
        )}

        {/* All loaded message */}
        {loadedCount === totalCount && (
          <div className="pt-2 border-t">
            <div className="flex items-center justify-center gap-2 text-green-700 bg-green-50 p-2 rounded-lg">
              <CheckCircle2 className="h-4 w-4" />
              <span className="text-sm font-medium">All partitions loaded</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
