'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Trash2, AlertTriangle, Loader2, RefreshCw } from 'lucide-react';

export default function ManagePage() {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isResyncing, setIsResyncing] = useState(false);
  const [resyncMessage, setResyncMessage] = useState('');

  const handleDeleteCheckpoints = async () => {
    if (!confirm('Are you sure you want to delete all checkpoints?')) return;

    setIsLoading(true);
    setMessage('Deleting checkpoints...');

    try {
      const response = await fetch('/api/data/checkpoints', {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to delete checkpoints');
      }

      const result = await response.json();
      setMessage(`✓ ${result.message}`);

      // Auto-resync config after deletion
      await handleResync();
    } catch (error) {
      console.error('Failed to delete checkpoints:', error);
      setMessage(`❌ Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteOSM = async () => {
    if (!confirm('Are you sure you want to delete all OSM data?')) return;

    setIsLoading(true);
    setMessage('Deleting OSM data...');

    try {
      const response = await fetch('/api/data/osm', {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to delete OSM data');
      }

      const result = await response.json();
      setMessage(`✓ ${result.message}`);

      // Auto-resync config after deletion
      await handleResync();
    } catch (error) {
      console.error('Failed to delete OSM data:', error);
      setMessage(`❌ Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteElevation = async () => {
    if (!confirm('Are you sure you want to delete all elevation data?')) return;

    setIsLoading(true);
    setMessage('Deleting elevation data...');

    try {
      const response = await fetch('/api/data/elevation', {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to delete elevation data');
      }

      const result = await response.json();
      setMessage(`✓ ${result.message}`);

      // Auto-resync config after deletion
      await handleResync();
    } catch (error) {
      console.error('Failed to delete elevation data:', error);
      setMessage(`❌ Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteIndexes = async () => {
    if (!confirm('Are you sure you want to delete all OSM indexes?')) return;

    setIsLoading(true);
    setMessage('Deleting OSM indexes...');

    try {
      const response = await fetch('/api/data/indexes', {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to delete OSM indexes');
      }

      const result = await response.json();
      setMessage(`✓ ${result.message}`);

      // Auto-resync config after deletion
      await handleResync();
    } catch (error) {
      console.error('Failed to delete OSM indexes:', error);
      setMessage(`❌ Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    const confirmText = 'DELETE ALL';
    const userInput = prompt(
      `⚠️ WARNING: This will delete ALL data:\n` +
      `  • All OSM planet files and indexes\n` +
      `  • All elevation datasets\n` +
      `  • All checkpoints\n\n` +
      `This action CANNOT be undone!\n\n` +
      `Type "${confirmText}" to confirm:`
    );

    if (userInput !== confirmText) {
      setMessage('Delete all data cancelled.');
      return;
    }

    setIsLoading(true);
    setMessage('Deleting all data (OSM + elevation + indexes + checkpoints)...');

    try {
      const response = await fetch('/api/data/all', {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to delete all data');
      }

      const result = await response.json();
      setMessage(`✓ ${result.message}`);

      // Auto-resync config after deletion
      await handleResync();
    } catch (error) {
      console.error('Failed to delete all data:', error);
      setMessage(`❌ Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResync = async () => {
    setIsResyncing(true);
    setResyncMessage('Scanning data directories...');

    try {
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
          `✓ Config synced! Found ${details.osmPlanetFiles} planet files, ` +
          `${details.osmIndexes} indexes, ${details.elevationDatasets} elevation datasets. ${details.changes.length} changes applied.`
        );
      } else {
        setResyncMessage(`✓ Config already in sync (${details.osmPlanetFiles} planet files, ${details.osmIndexes} indexes, ${details.elevationDatasets} datasets)`);
      }
    } catch (error) {
      console.error('Failed to resync config:', error);
      setResyncMessage(`❌ Error: ${error}`);
    } finally {
      setIsResyncing(false);
      // Clear message after 5 seconds
      setTimeout(() => setResyncMessage(''), 5000);
    }
  };

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-2">
        <h1 className="text-3xl font-bold">Data Management</h1>
        <Button
          variant="outline"
          size="sm"
          onClick={handleResync}
          disabled={isResyncing || isLoading}
        >
          {isResyncing ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Syncing...
            </>
          ) : (
            <>
              <RefreshCw className="mr-2 h-4 w-4" />
              Resync Config
            </>
          )}
        </Button>
      </div>
      <p className="text-gray-600 mb-8">
        Clean up cached data and checkpoints
      </p>

      <div className="max-w-2xl space-y-6">
        {resyncMessage && (
          <div className={`text-sm p-3 rounded-lg ${
            resyncMessage.includes('✓') ? 'bg-green-50 text-green-800 border border-green-200' :
            resyncMessage.includes('❌') ? 'bg-red-50 text-red-800 border border-red-200' :
            'bg-blue-50 text-blue-800 border border-blue-200'
          }`}>
            {resyncMessage}
          </div>
        )}
        {message && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-900 whitespace-pre-wrap">{message}</p>
          </div>
        )}

        {/* Warning Card - Moved to Top */}
        <Card className="border-yellow-500 bg-yellow-50">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-600" />
              <CardTitle className="text-yellow-900">Warning</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-yellow-800">
              Deleting data will require re-downloading for future analyses.
              Checkpoints allow resuming interrupted analyses - only delete if
              you want to restart from scratch or no longer plan to analyze
              that region.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Delete OSM Data</CardTitle>
            <CardDescription>
              Remove all downloaded OpenStreetMap planet files
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="destructive"
              onClick={handleDeleteOSM}
              disabled={isLoading}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete OSM Data
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Delete OSM Indexes</CardTitle>
            <CardDescription>
              Remove all OSM spatial indexes (osm_indexes folder)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="destructive"
              onClick={handleDeleteIndexes}
              disabled={isLoading}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete OSM Indexes
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Delete Elevation Data</CardTitle>
            <CardDescription>
              Remove all downloaded elevation datasets
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="destructive"
              onClick={handleDeleteElevation}
              disabled={isLoading}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Elevation Data
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Delete Checkpoints</CardTitle>
            <CardDescription>
              Remove checkpoint files to start analyses fresh
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="destructive"
              onClick={handleDeleteCheckpoints}
              disabled={isLoading}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Checkpoints
            </Button>
          </CardContent>
        </Card>

        {/* Delete All Data - Most Destructive, Show Last with Strong Warning */}
        <Card className="border-red-500 bg-red-50">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-6 w-6 text-red-600" />
              <CardTitle className="text-red-900">Delete All Data</CardTitle>
            </div>
            <CardDescription className="text-red-700">
              Delete ALL data: OSM planet files, indexes, elevation datasets, and checkpoints
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-red-100 border border-red-300 rounded p-3">
              <p className="text-sm text-red-800 font-semibold mb-2">
                ⚠️ DANGER: This action cannot be undone!
              </p>
              <ul className="text-sm text-red-800 space-y-1 ml-4">
                <li>• All OSM planet files and spatial indexes will be deleted</li>
                <li>• All elevation datasets will be deleted</li>
                <li>• All analysis checkpoints will be deleted</li>
                <li>• You will need to re-download all data for future analyses</li>
              </ul>
            </div>
            <Button
              variant="destructive"
              onClick={handleDeleteAll}
              disabled={isLoading}
              className="bg-red-600 hover:bg-red-700"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete All Data (OSM + indexes + elevation + checkpoints)
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
