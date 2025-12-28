import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import yaml from 'yaml';

const CONFIG_PATH = path.join(process.cwd(), '..', 'config.yaml');
const PLANET_DIR = path.join(process.cwd(), '..', 'data', 'planet_osm_data');
const INDEXES_DIR = path.join(process.cwd(), '..', 'data', 'osm_indexes');
const OUTPUT_DIR = path.join(process.cwd(), '..', 'output');

interface DataCoverage {
  osmCoverage: string[];
  elevationDatasets: Record<string, string[]>;
  planetFiles: string[];
  indexes: string[];
  outputFiles: Array<{
    filename: string;
    region: string;
    createdAt: string;
  }>;
}

export async function GET(): Promise<Response> {
  try {
    const coverage: DataCoverage = {
      osmCoverage: [],
      elevationDatasets: {},
      planetFiles: [],
      indexes: [],
      outputFiles: [],
    };

    // Read config.yaml - this is fast and has all the data we need
    const configContent = await fs.readFile(CONFIG_PATH, 'utf-8');
    const config = yaml.parse(configContent);

    coverage.elevationDatasets = config.ELEVATION_DATASETS || {};
    coverage.planetFiles = config.OSM_PLANET_DATA || [];

    // Derive canonical region names from elevation datasets for consistency
    // Use the continent/region format (e.g., "europe/monaco") everywhere
    const canonicalRegions = new Set<string>();
    for (const regions of Object.values(coverage.elevationDatasets)) {
      if (Array.isArray(regions)) {
        regions.forEach((r: string) => canonicalRegions.add(r));
      }
    }
    coverage.osmCoverage = Array.from(canonicalRegions).sort();
    coverage.indexes = Array.from(canonicalRegions).sort(); // Same canonical regions for indexes

    return NextResponse.json(coverage);
  } catch (error) {
    console.error('Failed to get data coverage:', error);
    return NextResponse.json(
      { error: 'Failed to get data coverage', details: String(error) },
      { status: 500 }
    );
  }
}
