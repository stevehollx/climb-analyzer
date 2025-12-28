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
