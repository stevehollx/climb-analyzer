import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const OUTPUT_DIR = path.join(process.cwd(), '../../output');

export async function GET(): Promise<Response> {
  try {
    const files = await fs.readdir(OUTPUT_DIR);
    const xlsxFiles = files.filter(f => f.endsWith('.xlsx') || f.endsWith('.csv'));

    const fileDetails = await Promise.all(
      xlsxFiles.map(async (filename) => {
        const filePath = path.join(OUTPUT_DIR, filename);
        const stats = await fs.stat(filePath);

        // Parse filename to extract metadata
        // Format: climbs_[LOCATION]_[SURFACE]_[SCORE]_[SCOPE]_[RADIUS].xlsx
        const parts = filename.replace('.xlsx', '').replace('.csv', '').split('_');

        return {
          filename,
          path: filePath,
          size: stats.size,
          createdAt: stats.mtime,
          location: parts.slice(1, -3).join('_'),
          surface: parts[parts.length - 3],
          scoreType: parts[parts.length - 2],
          scope: parts[parts.length - 1],
        };
      })
    );

    return NextResponse.json(fileDetails);
  } catch (error) {
    console.error('Failed to read output directory:', error);
    return NextResponse.json(
      { error: 'Failed to read output files' },
      { status: 500 }
    );
  }
}
