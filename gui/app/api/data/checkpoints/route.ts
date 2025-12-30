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
          // Recursively delete subdirectory
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
    // Directory doesn't exist or can't be read
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
      errors.push(`Failed to read directory ${dirPath}: ${err}`);
    }
  }

  return { deleted, errors };
}

export async function DELETE() {
  try {
    const dataDir = path.join(process.cwd(), '..', 'data', 'checkpoint_data');

    const { deleted, errors } = await deleteDirectoryContents(dataDir);

    if (errors.length > 0) {
      return NextResponse.json({
        message: `Deleted ${deleted} checkpoint files with ${errors.length} errors`,
        deleted,
        errors
      });
    }

    return NextResponse.json({
      message: `Successfully deleted ${deleted} checkpoint files`,
      deleted
    });
  } catch (error) {
    console.error('Failed to delete checkpoints:', error);
    return NextResponse.json(
      { error: 'Failed to delete checkpoints', details: String(error) },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const dataDir = path.join(process.cwd(), '..', 'data', 'checkpoint_data');

    let fileCount = 0;
    let totalSize = 0;

    try {
      const entries = await fs.readdir(dataDir, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isDirectory()) {
          // Count files in subdirectory
          const subDir = path.join(dataDir, entry.name);
          const subEntries = await fs.readdir(subDir);
          fileCount += subEntries.length;

          for (const subEntry of subEntries) {
            try {
              const stat = await fs.stat(path.join(subDir, subEntry));
              totalSize += stat.size;
            } catch {
              // Ignore stat errors
            }
          }
        } else {
          fileCount++;
          try {
            const stat = await fs.stat(path.join(dataDir, entry.name));
            totalSize += stat.size;
          } catch {
            // Ignore stat errors
          }
        }
      }
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        // Directory doesn't exist
        return NextResponse.json({ fileCount: 0, totalSize: 0, exists: false });
      }
      throw err;
    }

    return NextResponse.json({
      fileCount,
      totalSize,
      totalSizeFormatted: formatBytes(totalSize),
      exists: true
    });
  } catch (error) {
    console.error('Failed to get checkpoint info:', error);
    return NextResponse.json(
      { error: 'Failed to get checkpoint info', details: String(error) },
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
