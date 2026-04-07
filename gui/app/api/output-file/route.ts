import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const OUTPUT_DIR = path.join(process.cwd(), '..', 'output');

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const filename = searchParams.get('filename');

    if (!filename) {
      return NextResponse.json(
        { error: 'Filename parameter is required' },
        { status: 400 }
      );
    }

    // Security: Prevent directory traversal
    const normalizedFilename = path.basename(filename);
    const filePath = path.join(OUTPUT_DIR, normalizedFilename);

    // Check if file exists
    try {
      await fs.access(filePath);
    } catch {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }

    // Read and return the file
    const fileBuffer = await fs.readFile(filePath);
    const ext = path.extname(normalizedFilename).toLowerCase();

    let contentType = 'application/octet-stream';
    if (ext === '.xlsx' || ext === '.xls') {
      contentType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    } else if (ext === '.csv') {
      contentType = 'text/csv';
    } else if (ext === '.sqlite' || ext === '.db') {
      contentType = 'application/x-sqlite3';
    }

    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${normalizedFilename}"`,
      },
    });
  } catch (error) {
    console.error('Failed to serve output file:', error);
    return NextResponse.json(
      { error: 'Failed to load file', details: String(error) },
      { status: 500 }
    );
  }
}
