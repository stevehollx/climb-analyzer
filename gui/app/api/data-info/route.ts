import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import yaml from 'yaml';

export async function GET(): Promise<Response> {
  try {
    // Read config.yaml (1 level up from gui/)
    const configPath = path.join(process.cwd(), '..', 'config.yaml');

    if (!fs.existsSync(configPath)) {
      return NextResponse.json({
        osmPlanetData: [],
        osmIndexes: [],
        elevationData: [],
        elevationDatasets: {},
        analyzedReports: []
      });
    }

    const configContent = fs.readFileSync(configPath, 'utf-8');
    const config = yaml.parse(configContent);

    return NextResponse.json({
      osmPlanetData: config.OSM_PLANET_DATA || [],
      osmIndexes: config.OSM_INDEXES || [],
      elevationData: config.ELEVATION_DATA || [],
      elevationDatasets: config.ELEVATION_DATASETS || {},
      analyzedReports: config.ANALYZED_REPORTS || []
    });
  } catch (error) {
    console.error('Failed to read data info:', error);
    return NextResponse.json(
      { error: 'Failed to read data info', details: String(error) },
      { status: 500 }
    );
  }
}
