import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

// Progress tracking interface
interface Progress {
  phase: string;
  phaseNumber: number;
  totalPhases: number;
  stepProgress: number; // 0-100
  stepDescription: string;
  fileProgress: number; // 0-100
  currentFile: string;
  filesCompleted: number;
  filesTotal: number;
  estimatedTimeRemaining: string;
  startTime: number;
}

// Store running jobs and their processes
const activeJobs = new Map<string, {
  process: any;
  logs: string[];
  status: string;
  progress: Progress;
}>();

// Parse output line and update progress
function parseProgressLine(line: string, progress: Progress): Progress {
  const newProgress = { ...progress };

  // Parse PROGRESS_JSON: {...} lines (structured progress from Python script)
  if (line.includes('PROGRESS_JSON:')) {
    try {
      const jsonStr = line.substring(line.indexOf('{'));
      const data = JSON.parse(jsonStr);

      if (data.phase) newProgress.phase = data.phase;
      if (data.phaseNumber !== undefined) newProgress.phaseNumber = data.phaseNumber;
      if (data.totalPhases !== undefined) newProgress.totalPhases = data.totalPhases;
      if (data.stepProgress !== undefined) newProgress.stepProgress = data.stepProgress;
      if (data.stepDescription) newProgress.stepDescription = data.stepDescription;
      if (data.fileProgress !== undefined) newProgress.fileProgress = data.fileProgress;
      if (data.currentFile) newProgress.currentFile = data.currentFile;
      if (data.filesCompleted !== undefined) newProgress.filesCompleted = data.filesCompleted;
      if (data.filesTotal !== undefined) newProgress.filesTotal = data.filesTotal;

      // Calculate estimated time
      const elapsed = Date.now() - newProgress.startTime;
      const totalProgress = ((newProgress.phaseNumber - 1) / newProgress.totalPhases) +
                           (newProgress.stepProgress / 100 / newProgress.totalPhases);
      if (totalProgress > 0) {
        const estimatedTotal = elapsed / totalProgress;
        const remaining = Math.max(0, estimatedTotal - elapsed);
        const minutes = Math.floor(remaining / 60000);
        const seconds = Math.floor((remaining % 60000) / 1000);
        newProgress.estimatedTimeRemaining = minutes > 0 ?
          `${minutes}m ${seconds}s` : `${seconds}s`;
      }
    } catch (e) {
      console.error('Failed to parse progress JSON:', e);
    }
  }

  // Fallback: Parse text-based progress indicators
  else {
    // Region detection - extract current region being processed
    // Format: [1/1] monaco  (lowercase, no capitals or spaces in region name part)
    // NOT matching phase indicators like: [2/4] Building spatial index
    const regionMatch = line.match(/^\[(\d+)\/(\d+)\]\s+([a-z][a-z\-\/]*)\s*$/i);
    if (regionMatch) {
      const currentRegion = parseInt(regionMatch[1], 10);
      const totalRegions = parseInt(regionMatch[2], 10);
      const regionName = regionMatch[3];
      newProgress.filesTotal = totalRegions;
      newProgress.filesCompleted = currentRegion - 1;
      newProgress.stepDescription = `Processing region ${currentRegion} of ${totalRegions}: ${regionName}`;
    }

    // Step detection - more specific than phase
    // Check for destination line to extract filename
    if (line.includes('Destination:') && line.includes('.osm.pbf')) {
      newProgress.phase = 'Downloading OSM Data';
      newProgress.phaseNumber = 1;
      newProgress.totalPhases = 3;
      const fileMatch = line.match(/([a-z\-]+\.osm\.pbf)/i);
      if (fileMatch) {
        newProgress.currentFile = fileMatch[1];
        newProgress.stepDescription = `Downloading ${fileMatch[1]}`;
      }
    } else if (line.includes('Downloading OSM data for')) {
      newProgress.phase = 'Downloading OSM Data';
      newProgress.phaseNumber = 1;
      newProgress.totalPhases = 3;
      // Extract region name from "Downloading OSM data for California"
      const regionMatch = line.match(/Downloading OSM data for (.+)/i);
      if (regionMatch && !newProgress.currentFile) {
        newProgress.stepDescription = `Downloading OSM data for ${regionMatch[1].trim()}`;
      }
    } else if (line.includes('Building spatial index') || line.includes('Building index') || line.includes('[2/4]')) {
      newProgress.phase = 'Building Spatial Indexes';
      newProgress.phaseNumber = 2;
      newProgress.totalPhases = 3;
      newProgress.currentFile = '';
      newProgress.fileProgress = 0;
      // Extract more detail from the line if available
      const indexMatch = line.match(/Building (?:spatial )?index(?: for (.+))?/i);
      if (indexMatch && indexMatch[1]) {
        newProgress.stepDescription = `Building spatial index for ${indexMatch[1]}`;
      } else if (!newProgress.stepDescription.includes('Processing region')) {
        newProgress.stepDescription = 'Building spatial index for OSM data';
      }
    } else if (line.includes('Downloading elevation') || line.includes('DEM') || line.includes('.tif')) {
      newProgress.phase = 'Downloading Elevation Data';
      newProgress.phaseNumber = 3;
      newProgress.totalPhases = 3;
      const fileMatch = line.match(/([a-z\-]+\.tif)/i);
      if (fileMatch) {
        newProgress.currentFile = fileMatch[1];
        newProgress.stepDescription = `Downloading elevation tile: ${fileMatch[1]}`;
      }
    }

    // File progress detection (tqdm-style: XX%|███████)
    const percentMatch = line.match(/(\d+)%\|/);
    if (percentMatch) {
      newProgress.fileProgress = parseInt(percentMatch[1], 10);
    }

    // File completion detection
    if (line.includes('✓') || line.includes('Downloaded:') || line.includes('Index built')) {
      const fileMatch = line.match(/([a-z\-]+\.osm\.pbf|[a-z\-]+\.tif|OSM data)/i);
      if (fileMatch) {
        newProgress.currentFile = '';
        newProgress.fileProgress = 0;
      }
    }

    // Region completion detection - when we see "✓ Elevation data" or "Data download complete"
    if ((line.includes('✓ Elevation data:') && line.includes('downloaded')) ||
        line.includes('✓ Data download complete')) {
      // Increment completed regions
      if (newProgress.filesCompleted < newProgress.filesTotal) {
        newProgress.filesCompleted++;
      }
    }

    // Update step progress based on current phase within region
    // Each region has 3 phases, so stepProgress = (phaseNumber / totalPhases) * 100
    if (newProgress.phaseNumber > 0 && newProgress.totalPhases > 0) {
      newProgress.stepProgress = Math.round((newProgress.phaseNumber / newProgress.totalPhases) * 100);
    }
  }

  return newProgress;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');

    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }

    const job = activeJobs.get(jobId);

    if (!job) {
      return NextResponse.json(
        { error: 'Job not found', jobId },
        { status: 404 }
      );
    }

    return NextResponse.json({
      jobId,
      status: job.status,
      logs: job.logs,
      isRunning: job.status === 'running',
      progress: job.progress,
    });
  } catch (error) {
    console.error('Failed to get job status:', error);
    return NextResponse.json(
      { error: 'Failed to get job status', details: String(error) },
      { status: 500 }
    );
  }
}

export async function POST(request: Request): Promise<Response> {
  try {
    const { regions } = await request.json();

    if (!regions || regions.length === 0) {
      return NextResponse.json(
        { error: 'Regions are required' },
        { status: 400 }
      );
    }

    const jobId = uuidv4();

    // Build command line arguments
    const args: string[] = [
      '--run-region',
      regions.join(','),
      '--data-download'  // Download only, don't analyze
    ];

    // Get the path to the CLI script (1 level up from gui/)
    const cliPath = path.join(process.cwd(), '..', 'climb_analyzer.py');

    // Spawn the Python process
    const pythonProcess = spawn('python3', [cliPath, ...args], {
      cwd: path.join(process.cwd(), '..'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    const logs: string[] = [];

    // Initialize progress
    const initialProgress: Progress = {
      phase: 'Initializing',
      phaseNumber: 0,
      totalPhases: 3,
      stepProgress: 0,
      stepDescription: 'Starting download process...',
      fileProgress: 0,
      currentFile: '',
      filesCompleted: 0,
      filesTotal: regions.length,
      estimatedTimeRemaining: 'Calculating...',
      startTime: Date.now(),
    };

    // Store the job
    activeJobs.set(jobId, {
      process: pythonProcess,
      logs,
      status: 'running',
      progress: initialProgress
    });

    // Filter function to clean up terminal output for GUI display
    const filterTqdmOutput = (text: string): string => {
      // Split into lines
      const lines = text.split(/\r?\n/);
      const cleanedLines: string[] = [];

      for (const line of lines) {
        // Skip tqdm progress bar lines by detecting the tqdm pattern:
        // Format: "  Downloading:  XX%|▉         | 120M/1.19G [00:07<01:00, 19.0MB/s]"
        // Key indicators: percentage followed by pipe, and time brackets with < separator
        if (line.includes('%|') && line.includes('[') && line.includes('<')) {
          continue;
        }
        // Skip empty lines with just carriage returns
        if (line.trim() === '') {
          continue;
        }
        // Clean carriage returns from the line
        const cleaned = line.replace(/\r/g, '');
        if (cleaned.trim()) {
          cleanedLines.push(cleaned);
        }
      }

      return cleanedLines.join('\n') + (cleanedLines.length > 0 ? '\n' : '');
    };

    // Capture stdout
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      const filtered = filterTqdmOutput(output);
      if (filtered) {
        logs.push(filtered);
      }
      console.log(`[${jobId}] ${output}`);

      // Parse progress from output
      const job = activeJobs.get(jobId);
      if (job) {
        const lines = output.split('\n');
        for (const line of lines) {
          if (line.trim()) {
            job.progress = parseProgressLine(line, job.progress);
          }
        }
      }
    });

    // Capture stderr
    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString();
      const filtered = filterTqdmOutput(output);
      if (filtered) {
        logs.push(filtered);
      }
      console.error(`[${jobId}] ${output}`);

      // Also parse progress from stderr (tqdm outputs there)
      const job = activeJobs.get(jobId);
      if (job) {
        const lines = output.split('\n');
        for (const line of lines) {
          if (line.trim()) {
            job.progress = parseProgressLine(line, job.progress);
          }
        }
      }
    });

    // Handle process exit
    pythonProcess.on('close', (code) => {
      console.log(`[${jobId}] Download process exited with code ${code}`);
      logs.push(`Download completed with exit code ${code}`);

      const job = activeJobs.get(jobId);
      if (job) {
        job.status = code === 0 ? 'completed' : 'failed';
        if (code === 0) {
          job.progress.phase = 'Completed';
          job.progress.phaseNumber = job.progress.totalPhases;
          job.progress.stepProgress = 100;
          job.progress.fileProgress = 100;
          job.progress.stepDescription = 'All downloads completed successfully';
          job.progress.estimatedTimeRemaining = '0s';
        }
      }

      // Clean up after some time
      setTimeout(() => {
        activeJobs.delete(jobId);
      }, 300000); // Keep logs for 5 minutes after completion
    });

    return NextResponse.json({
      jobId,
      message: 'Download started successfully',
      regions
    });
  } catch (error) {
    console.error('Failed to start download:', error);
    return NextResponse.json(
      { error: 'Failed to start download', details: String(error) },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');

    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }

    const job = activeJobs.get(jobId);

    if (!job) {
      return NextResponse.json(
        { error: 'Job not found', jobId },
        { status: 404 }
      );
    }

    // Kill the process (sends SIGTERM, equivalent to Ctrl+C)
    if (job.process && !job.process.killed) {
      job.process.kill('SIGTERM');
      job.status = 'cancelled';
      job.logs.push('\n\n=== Download cancelled by user ===\n');

      console.log(`[${jobId}] Download process cancelled by user`);

      return NextResponse.json({
        message: 'Download cancelled successfully',
        jobId
      });
    } else {
      return NextResponse.json(
        { error: 'Process already terminated', jobId },
        { status: 400 }
      );
    }
  } catch (error) {
    console.error('Failed to cancel download:', error);
    return NextResponse.json(
      { error: 'Failed to cancel download', details: String(error) },
      { status: 500 }
    );
  }
}
