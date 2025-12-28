import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import yaml from 'yaml';
import os from 'os';

const CONFIG_PATH = path.join(process.cwd(), '..', 'config.yaml');

// GET - Read current config
export async function GET(): Promise<Response> {
  try {
    const configContent = await fs.readFile(CONFIG_PATH, 'utf-8');
    const config = yaml.parse(configContent);

    // Get CPU core count
    const cpuCores = os.cpus().length;

    // Extract the relevant fields for the GUI
    return NextResponse.json({
      // Basic settings
      deploymentType: config.DEPLOYMENT_TYPE || 'cloud',
      elevationBatchSize: config.ELEVATION_BATCH_SIZE || 100,
      elevationMaxConcurrent: config.ELEVATION_MAX_CONCURRENT || 2,
      checkpointIntervalMin: config.CHECKPOINT_INTERVAL_MIN || 15,
      // System info
      cpuCores: cpuCores,
      // Advanced settings
      minClimbLengthFt: config.MIN_CLIMB_LENGTH_FT || 100,
      elevationDelayBetweenBatchesSec: config.ELEVATION_DELAY_BETWEEN_BATCHES_SEC || 0,
      elevationRequestTimeoutSec: config.ELEVATION_REQUEST_TIMEOUT_SEC || 45,
      elevationMaxRetries: config.ELEVATION_MAX_RETRIES || 3,
      elevationBackoffFactor: config.ELEVATION_BACKOFF_FACTOR || 2.0,
      elevationDatasetTiers: config.ELEVATION_DATASET_TIERS || 'primary+secondary+tertiary',
      geocodingMaxConcurrent: config.GEOCODING_MAX_CONCURRENT || 8,
      geocodingRetryAttempts: config.GEOCODING_RETRY_ATTEMPTS || 2,
      osmChunkSizeKm: config.OSM_CHUNK_SIZE_KM || 30.0,
      cloudCacheEnabled: config.CLOUD_CACHE_ENABLED ?? true,
      // cloudCacheRepo is hardcoded to stevehollx/global-road-and-trail-climbs
    });
  } catch (error) {
    console.error('Failed to read config:', error);

    // Get CPU core count
    const cpuCores = os.cpus().length;

    // Return defaults if config doesn't exist
    return NextResponse.json({
      deploymentType: 'cloud',
      elevationBatchSize: 100,
      elevationMaxConcurrent: 2,
      checkpointIntervalMin: 15,
      cpuCores: cpuCores,
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
      // cloudCacheRepo is hardcoded to stevehollx/global-road-and-trail-climbs
    });
  }
}

// POST - Update config
export async function POST(request: Request): Promise<Response> {
  try {
    const updates = await request.json();

    // Read current config
    let config: any = {};
    try {
      const configContent = await fs.readFile(CONFIG_PATH, 'utf-8');
      config = yaml.parse(configContent);
    } catch (error) {
      // Config doesn't exist, will create new one
      console.log('Config file not found, creating new one');
    }

    // Update only the fields we manage in the GUI
    // Basic settings
    if (updates.deploymentType !== undefined) {
      config.DEPLOYMENT_TYPE = updates.deploymentType;
    }
    if (updates.elevationBatchSize !== undefined) {
      config.ELEVATION_BATCH_SIZE = parseInt(updates.elevationBatchSize, 10);
    }
    if (updates.elevationMaxConcurrent !== undefined) {
      config.ELEVATION_MAX_CONCURRENT = parseInt(updates.elevationMaxConcurrent, 10);
    }
    if (updates.checkpointIntervalMin !== undefined) {
      config.CHECKPOINT_INTERVAL_MIN = parseFloat(updates.checkpointIntervalMin);
    }

    // Advanced settings
    if (updates.minClimbLengthFt !== undefined) {
      config.MIN_CLIMB_LENGTH_FT = parseFloat(updates.minClimbLengthFt);
    }
    if (updates.elevationDelayBetweenBatchesSec !== undefined) {
      config.ELEVATION_DELAY_BETWEEN_BATCHES_SEC = parseFloat(updates.elevationDelayBetweenBatchesSec);
    }
    if (updates.elevationRequestTimeoutSec !== undefined) {
      config.ELEVATION_REQUEST_TIMEOUT_SEC = parseInt(updates.elevationRequestTimeoutSec, 10);
    }
    if (updates.elevationMaxRetries !== undefined) {
      config.ELEVATION_MAX_RETRIES = parseInt(updates.elevationMaxRetries, 10);
    }
    if (updates.elevationBackoffFactor !== undefined) {
      config.ELEVATION_BACKOFF_FACTOR = parseFloat(updates.elevationBackoffFactor);
    }
    if (updates.elevationDatasetTiers !== undefined) {
      config.ELEVATION_DATASET_TIERS = updates.elevationDatasetTiers;
    }
    if (updates.geocodingMaxConcurrent !== undefined) {
      config.GEOCODING_MAX_CONCURRENT = parseInt(updates.geocodingMaxConcurrent, 10);
    }
    if (updates.geocodingRetryAttempts !== undefined) {
      config.GEOCODING_RETRY_ATTEMPTS = parseInt(updates.geocodingRetryAttempts, 10);
    }
    if (updates.osmChunkSizeKm !== undefined) {
      config.OSM_CHUNK_SIZE_KM = parseFloat(updates.osmChunkSizeKm);
    }
    if (updates.cloudCacheEnabled !== undefined) {
      config.CLOUD_CACHE_ENABLED = Boolean(updates.cloudCacheEnabled);
    }
    // cloudCacheRepo is hardcoded - not configurable

    // Write back to file
    const yamlContent = yaml.stringify(config);
    await fs.writeFile(CONFIG_PATH, yamlContent, 'utf-8');

    return NextResponse.json({
      success: true,
      message: 'Configuration updated successfully',
    });
  } catch (error) {
    console.error('Failed to update config:', error);
    return NextResponse.json(
      { error: 'Failed to update configuration', details: String(error) },
      { status: 500 }
    );
  }
}
