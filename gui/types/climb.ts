/**
 * Climb data types based on the climb analyzer CSV output format
 */

export type ClimbCategory = 'HC' | 'Cat 1' | 'Cat 2' | 'Cat 3' | 'Cat 4' | 'Uncategorized';

export type SurfaceType = 'paved' | 'gravel' | 'dirt' | 'all';

export type ScoreType = 'basic' | 'fiets' | 'pdi';

export type Units = 'metric' | 'imperial';

export type AnalysisMode = 'address' | 'region' | 'batch';

export type DeploymentType = 'cloud' | 'local';

export interface Climb {
  // Location data
  streetName: string;
  city: string;
  state: string;
  country: string;
  distanceFromCenter: number;
  lat: number;
  lon: number;

  // Categorization
  category: ClimbCategory;
  cyclingAccess: string;

  // Scores
  basicScore: number;
  fietsScore: number;
  pdiScore: number;

  // Elevation metrics
  elevationGain: number;
  height: number;
  prominence: number;

  // Distance metrics
  length: number;
  avgGrade: number;
  maxGrade: number;

  // Road details
  highwayType: string;
  surface: string;
  tracktype: string;
  wayId: string;
  osmLink: string;
  allWayIds?: string;
  connectedClimbs: string;

  // Elevation profile data (format: "dist,ele,grade|dist,ele,grade|...")
  elevationProfile?: string;
}

export interface ClimbGeoJSON extends GeoJSON.Feature<GeoJSON.LineString> {
  properties: Climb & {
    color: string;
    opacity: number;
  };
}

export interface AnalysisConfig {
  mode: AnalysisMode;

  // Address mode
  address?: string;
  radius?: number;

  // Region mode
  region?: string;
  regions?: string[];

  // Analysis parameters
  surfaceFilter: SurfaceType;
  cyclingFilter: boolean;
  units: Units;
  minScore?: number;  // Uses basic score for filtering
  geocoding: boolean;

  // Data management
  deleteDataOnComplete: boolean;
}

export interface Config {
  // Basic settings
  deploymentType: DeploymentType;
  elevationBatchSize: number;
  elevationMaxConcurrent: number;
  checkpointIntervalMin: number;
  // System info
  cpuCores?: number;
  // Advanced settings
  minClimbLengthFt: number;
  elevationDelayBetweenBatchesSec: number;
  elevationRequestTimeoutSec: number;
  elevationMaxRetries: number;
  elevationBackoffFactor: number;
  elevationDatasetTiers: string;
  geocodingMaxConcurrent: number;
  geocodingRetryAttempts: number;
  osmChunkSizeKm: number;
  cloudCacheEnabled: boolean;
  // cloudCacheRepo is hardcoded to stevehollx/global-road-and-trail-climbs
}

export interface AnalysisProgress {
  stage: string;
  progress: number;
  message: string;
  eta?: string;
}

export interface OutputFile {
  filename: string;
  path: string;
  location: string;
  surface: SurfaceType;
  scoreType: ScoreType;
  scope: string;
  radius?: number;
  climbCount: number;
  createdAt: Date;
  size: number;
}

/**
 * Geographic partition for large regions
 * Used when a region's SQLite database is split into geographic partitions
 * to meet GitHub's 2GB limit and sql.js WASM memory constraints
 */
export interface Partition {
  partition_id: string;        // e.g., "norcal", "socal", "northeast"
  display_name: string;        // e.g., "Northern California"
  database_file: string;
  database_size: number;
  database_url: string;
  bounds?: {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
  } | null;
  climb_count?: number | null;
}

/**
 * Region metadata from index.json
 * Supports both single-file and partitioned regions
 */
export interface RegionMetadata {
  region_name: string;
  version: string;
  release_tag: string;
  release_url: string;
  climb_count: number | null;
  elevation_errors: number | null;
  // XLSX files
  files: string[];
  download_urls: string[];
  file_sizes: number[];
  total_size: number;
  file_count: number;
  has_split_files: boolean;
  // Single database (non-split, non-partitioned)
  database_file: string | null;
  database_size: number | null;
  database_url: string | null;
  // Split database (binary chunks)
  is_split: boolean;
  split_files: string[] | null;
  split_urls: string[] | null;
  split_sizes: number[] | null;
  split_checksums: string[] | null;
  // Partitioned database (geographic partitions)
  is_partitioned: boolean;
  partition_type: 'geofabrik' | 'quadtree' | null;
  partitions: Partition[] | null;
  total_database_size: number | null;
  // Dates
  published_at: string;
  last_updated: string | null;
}
