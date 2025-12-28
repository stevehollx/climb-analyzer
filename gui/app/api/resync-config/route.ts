import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(): Promise<Response> {
  try {
    // Get the path to the sync script (1 level up from gui/)
    const syncScript = path.join(process.cwd(), '..', 'utils', 'sync_config_data.py');

    return new Promise((resolve) => {
      const pythonProcess = spawn('python3', [syncScript], {
        cwd: path.join(process.cwd(), '..'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
        console.log(data.toString());
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        console.error(data.toString());
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve(NextResponse.json({
            success: true,
            message: 'Config successfully synced with data on disk',
            output,
            details: parseResyncOutput(output)
          }));
        } else {
          resolve(NextResponse.json(
            {
              error: 'Failed to resync config',
              details: errorOutput || output,
              exitCode: code
            },
            { status: 500 }
          ));
        }
      });
    });
  } catch (error) {
    console.error('Failed to resync config:', error);
    return NextResponse.json(
      { error: 'Failed to resync config', details: String(error) },
      { status: 500 }
    );
  }
}

/**
 * Parse the output from sync_config_data.py to extract summary information
 */
function parseResyncOutput(output: string): {
  osmPlanetFiles: number;
  osmIndexes: number;
  elevationDatasets: number;
  checkpointRegions: number;
  changes: string[];
} {
  const lines = output.split('\n');

  const result = {
    osmPlanetFiles: 0,
    osmIndexes: 0,
    elevationDatasets: 0,
    checkpointRegions: 0,
    changes: [] as string[]
  };

  for (const line of lines) {
    // Extract numbers from summary
    const osmMatch = line.match(/OSM Planet Files:\s*(\d+)/);
    if (osmMatch) result.osmPlanetFiles = parseInt(osmMatch[1], 10);

    const indexMatch = line.match(/OSM Indexes:\s*(\d+)/);
    if (indexMatch) result.osmIndexes = parseInt(indexMatch[1], 10);

    const elevMatch = line.match(/Elevation Datasets:\s*(\d+)/);
    if (elevMatch) result.elevationDatasets = parseInt(elevMatch[1], 10);

    const checkMatch = line.match(/Checkpoint Regions:\s*(\d+)/);
    if (checkMatch) result.checkpointRegions = parseInt(checkMatch[1], 10);

    // Capture change lines
    if (line.trim().startsWith('+') || line.trim().startsWith('-') || line.trim().startsWith('•')) {
      result.changes.push(line.trim());
    }
  }

  return result;
}
