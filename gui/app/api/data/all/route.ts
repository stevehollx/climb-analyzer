import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

async function deleteDirectoryContents(dirPath: string): Promise<{ deleted: number; errors: string[] }> {
  let deleted = 0;
  const errors: string[] = [];

  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      try {
        if (entry.isDirectory()) {
          const result = await deleteDirectoryContents(fullPath);
          deleted += result.deleted;
          errors.push(...result.errors);
          await fs.rmdir(fullPath);
        } else {
          await fs.unlink(fullPath);
          deleted++;
        }
      } catch (err) {
        errors.push(`Failed to delete ${fullPath}: ${err}`);
      }
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
      errors.push(`Failed to read directory ${dirPath}: ${err}`);
    }
  }

  return { deleted, errors };
}

export async function DELETE() {
  try {
    const baseDir = path.join(process.cwd(), '..', 'data');
    const directories = [
      'planet_osm_data',
      'osm_indexes',
      'elevation_data',
      'checkpoint_data'
    ];

    let totalDeleted = 0;
    const allErrors: string[] = [];
    const results: Record<string, number> = {};

    for (const dir of directories) {
      const dirPath = path.join(baseDir, dir);
      const { deleted, errors } = await deleteDirectoryContents(dirPath);
      totalDeleted += deleted;
      allErrors.push(...errors);
      results[dir] = deleted;
    }

    if (allErrors.length > 0) {
      return NextResponse.json({
        message: `Deleted ${totalDeleted} files with ${allErrors.length} errors`,
        totalDeleted,
        results,
        errors: allErrors
      });
    }

    return NextResponse.json({
      message: `Successfully deleted ${totalDeleted} files from all data directories`,
      totalDeleted,
      results
    });
  } catch (error) {
    console.error('Failed to delete all data:', error);
    return NextResponse.json(
      { error: 'Failed to delete all data', details: String(error) },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const baseDir = path.join(process.cwd(), '..', 'data');
    const directories = [
      { name: 'planet_osm_data', label: 'OSM Data' },
      { name: 'osm_indexes', label: 'OSM Indexes' },
      { name: 'elevation_data', label: 'Elevation Data' },
      { name: 'checkpoint_data', label: 'Checkpoints' }
    ];

    let totalFiles = 0;
    let totalSize = 0;
    const breakdown: Record<string, { fileCount: number; size: number; sizeFormatted: string }> = {};

    for (const dir of directories) {
      const dirPath = path.join(baseDir, dir.name);
      let fileCount = 0;
      let size = 0;

      try {
        const entries = await fs.readdir(dirPath, { withFileTypes: true });

        for (const entry of entries) {
          const fullPath = path.join(dirPath, entry.name);
          if (entry.isDirectory()) {
            const subEntries = await fs.readdir(fullPath);
            fileCount += subEntries.length;
            for (const subEntry of subEntries) {
              try {
                const stat = await fs.stat(path.join(fullPath, subEntry));
                size += stat.size;
              } catch {
                // Ignore stat errors
              }
            }
          } else {
            fileCount++;
            try {
              const stat = await fs.stat(fullPath);
              size += stat.size;
            } catch {
              // Ignore stat errors
            }
          }
        }
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
          console.error(`Error reading ${dirPath}:`, err);
        }
      }

      totalFiles += fileCount;
      totalSize += size;
      breakdown[dir.label] = {
        fileCount,
        size,
        sizeFormatted: formatBytes(size)
      };
    }

    return NextResponse.json({
      totalFiles,
      totalSize,
      totalSizeFormatted: formatBytes(totalSize),
      breakdown
    });
  } catch (error) {
    console.error('Failed to get all data info:', error);
    return NextResponse.json(
      { error: 'Failed to get all data info', details: String(error) },
      { status: 500 }
    );
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
