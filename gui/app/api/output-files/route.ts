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
      const xlsxFiles = files.filter(f => f.endsWith('.xlsx') || f.endsWith('.csv'));

      for (const filename of xlsxFiles) {
        const filePath = path.join(OUTPUT_DIR, filename);
        try {
          const stats = await fs.stat(filePath);

          // Parse filename to extract region
          // Format: climbs_[REGION]_[SURFACE]_[SCORE]_[SCOPE]_[RADIUS].xlsx
          const parts = filename.replace('.xlsx', '').replace('.csv', '').split('_');
          const region = parts.slice(1, -3).join('_');

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
