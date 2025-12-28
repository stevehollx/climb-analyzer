import { AnalysisConfig, Config, OutputFile } from '@/types/climb';

const API_BASE = '/api';

/**
 * Get configuration from config.yaml
 */
export async function getConfig(): Promise<Config> {
  const response = await fetch(`${API_BASE}/config`);
  if (!response.ok) throw new Error('Failed to fetch config');
  return response.json();
}

/**
 * Update configuration
 */
export async function updateConfig(config: Partial<Config>): Promise<void> {
  const response = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error('Failed to update config');
}

/**
 * Download data for regions
 */
export async function downloadData(regions: string[]): Promise<void> {
  const response = await fetch(`${API_BASE}/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ regions }),
  });
  if (!response.ok) throw new Error('Failed to start download');
}

/**
 * Run analysis
 */
export async function runAnalysis(config: AnalysisConfig): Promise<void> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error('Failed to start analysis');
}

/**
 * Get list of output files
 */
export async function getOutputFiles(): Promise<OutputFile[]> {
  const response = await fetch(`${API_BASE}/results`);
  if (!response.ok) throw new Error('Failed to fetch output files');
  return response.json();
}

/**
 * Load CSV content from output file
 */
export async function loadOutputFile(filename: string): Promise<string> {
  const response = await fetch(`${API_BASE}/results/${encodeURIComponent(filename)}`);
  if (!response.ok) throw new Error('Failed to load output file');
  return response.text();
}

/**
 * Delete checkpoint files
 */
export async function deleteCheckpoints(): Promise<void> {
  const response = await fetch(`${API_BASE}/data/checkpoints`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete checkpoints');
}

/**
 * Delete OSM data
 */
export async function deleteOSMData(): Promise<void> {
  const response = await fetch(`${API_BASE}/data/osm`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete OSM data');
}

/**
 * Delete elevation data
 */
export async function deleteElevationData(): Promise<void> {
  const response = await fetch(`${API_BASE}/data/elevation`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete elevation data');
}

/**
 * Subscribe to progress updates via Server-Sent Events
 */
export function subscribeToProgress(
  jobId: string,
  onProgress: (data: any) => void,
  onError?: (error: Error) => void,
  onComplete?: () => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/progress/${jobId}`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'complete') {
        onComplete?.();
        eventSource.close();
      } else {
        onProgress(data);
      }
    } catch (error) {
      console.error('Failed to parse progress data:', error);
    }
  };

  eventSource.onerror = (error) => {
    onError?.(new Error('Progress stream error'));
    eventSource.close();
  };

  return () => eventSource.close();
}
