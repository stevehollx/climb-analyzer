import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const OUTPUT_DIR = path.join(process.cwd(), '..', 'output');

interface OutputFile {
  filename: string;
  region: string;
  createdAt: string;
}

export async function GET(): Promise<Response> {
  try {
    const outputFiles: OutputFile[] = [];

    // Check output directory for analysis results
    try {
      const files = await fs.readdir(OUTPUT_DIR);
      // Include .xlsx, .csv, and .sqlite files
      const dataFiles = files.filter(f =>
        f.endsWith('.xlsx') || f.endsWith('.csv') || f.endsWith('.sqlite') || f.endsWith('.db')
      );

      for (const filename of dataFiles) {
        const filePath = path.join(OUTPUT_DIR, filename);
        try {
          const stats = await fs.stat(filePath);

          // Parse filename to extract region
          // Format: [REGION]_climbs_[SURFACE]_[SCORE]_[DATE]_[VERSION].xlsx/sqlite
          const baseName = filename
            .replace('.xlsx', '')
            .replace('.csv', '')
            .replace('.sqlite', '')
            .replace('.db', '');
          const parts = baseName.split('_');
          // Find "_climbs" and take everything before it as region
          const climbsIndex = parts.indexOf('climbs');
          const region = climbsIndex > 0 ? parts.slice(0, climbsIndex).join(' ') : parts[0];

          outputFiles.push({
            filename,
            region,
            createdAt: stats.mtime.toISOString(),
          });
        } catch (err) {
          console.error(`Error reading file ${filename}:`, err);
        }
      }

      // Sort by most recent first
      outputFiles.sort((a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );
    } catch (error) {
      console.log('Output directory not found');
    }

    return NextResponse.json({ outputFiles });
  } catch (error) {
    console.error('Failed to get output files:', error);
    return NextResponse.json(
      { error: 'Failed to get output files', details: String(error) },
      { status: 500 }
    );
  }
}
