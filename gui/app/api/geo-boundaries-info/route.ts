import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET(): Promise<Response> {
  try {
    // Try multiple possible paths
    const possiblePaths = [
      '/app/climb_analyzer/data/geo_definitions.py',  // Absolute path in container
      path.join(process.cwd(), '..', 'climb_analyzer', 'data', 'geo_definitions.py'),  // One level up
      path.join(process.cwd(), '..', '..', 'climb_analyzer', 'data', 'geo_definitions.py'),  // Two levels up
    ];

    let content = '';
    let usedPath = '';

    for (const testPath of possiblePaths) {
      try {
        content = await fs.readFile(testPath, 'utf-8');
        usedPath = testPath;
        break;
      } catch (err) {
        // Try next path
        continue;
      }
    }

    if (!content) {
      console.error('Could not find geo_definitions.py at any path');
      return NextResponse.json({
        date: 'Unknown',
        success: false,
        message: 'File not found',
      });
    }

    const lines = content.split('\n').slice(0, 5); // Read first 5 lines

    // Look for the line with "Generated automatically on"
    const dateLine = lines.find(line => line.includes('Generated automatically on'));

    if (dateLine) {
      // Extract date using regex
      const match = dateLine.match(/Generated automatically on (.+?) UTC/);
      if (match && match[1]) {
        return NextResponse.json({
          date: match[1] + ' UTC',
          success: true,
          path: usedPath,
        });
      }
    }

    return NextResponse.json({
      date: 'Unknown',
      success: false,
      message: 'Date not found in file',
    });
  } catch (error) {
    console.error('Failed to read geo boundaries date:', error);
    return NextResponse.json({
      date: 'Unknown',
      success: false,
      error: String(error)
    });
  }
}
