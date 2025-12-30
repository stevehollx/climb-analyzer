import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';
import yaml from 'yaml';

// Progress tracking interface
interface Progress {
  phase: string;
  phaseNumber: number;
  totalPhases: number;
  stepProgress: number; // 0-100
  stepDescription: string;
  itemsCompleted: number;
  itemsTotal: number;
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

// Process terminal output to handle carriage returns (\r) like a real terminal
// This makes tqdm progress bars overwrite themselves instead of creating duplicate lines
function processTerminalOutput(output: string, logs: string[]): void {
  // Strip ANSI escape codes (colors, formatting) before processing
  // Regex matches: ESC [ ... m (colors), ESC [ ... K (clear line), etc.
  const ansiRegex = /\x1b\[[0-9;]*[a-zA-Z]/g;
  const cleanOutput = output.replace(ansiRegex, '');

  // Split on carriage return to get segments
  const segments = cleanOutput.split('\r');

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const hasCarriageReturnAfter = i < segments.length - 1;

    if (hasCarriageReturnAfter) {
      // This segment has \r after it - should replace current line
      // Handle case where segment contains newlines
      if (segment.includes('\n')) {
        const lines = segment.split('\n');
        // Add all but last line as new lines
        for (let j = 0; j < lines.length - 1; j++) {
          if (logs.length === 0 || j > 0) {
            logs.push(lines[j] + '\n');
          } else {
            logs[logs.length - 1] = lines[j] + '\n';
          }
        }
        // Last line will be overwritten by next segment (due to \r)
        if (logs.length === 0) {
          logs.push(lines[lines.length - 1]);
        } else {
          logs[logs.length - 1] = lines[lines.length - 1];
        }
      } else {
        // No newlines - just replace last line
        if (logs.length === 0) {
          logs.push(segment);
        } else {
          logs[logs.length - 1] = segment;
        }
      }
    } else {
      // Last segment - no \r after it
      if (segment.includes('\n')) {
        const lines = segment.split('\n');
        // Handle first line
        if (lines[0]) {
          if (logs.length === 0) {
            logs.push(lines[0] + '\n');
          } else {
            // Append to or replace last line based on whether it has \n
            if (logs[logs.length - 1].endsWith('\n')) {
              logs.push(lines[0] + '\n');
            } else {
              logs[logs.length - 1] += lines[0] + '\n';
            }
          }
        }
        // Add remaining lines
        for (let j = 1; j < lines.length; j++) {
          if (j === lines.length - 1 && !segment.endsWith('\n')) {
            // Last line without trailing \n
            logs.push(lines[j]);
          } else {
            logs.push(lines[j] + (j < lines.length - 1 ? '\n' : ''));
          }
        }
      } else if (segment) {
        // No newlines in segment - append to current line
        if (logs.length === 0) {
          logs.push(segment);
        } else {
          logs[logs.length - 1] += segment;
        }
      }
    }
  }
}

// Parse output line and update progress
function parseProgressLine(line: string, progress: Progress): Progress {
  const newProgress = { ...progress };

  // Phase detection - based on actual CLI output
  // Step 0: Preparing - checking servers, downloading data, etc.
  if (line.includes('OpenTopoData') || line.includes('Checking') || line.includes('Downloading') ||
      line.includes('Validating Data') || line.includes('server status')) {
    newProgress.phase = 'Preparing';
    newProgress.phaseNumber = 0;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Checking servers and downloading data...';
  } else if (line.includes('Step 1:') && line.includes('Extracting')) {
    newProgress.phase = 'Extracting Ways';
    newProgress.phaseNumber = 1;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Extracting road ways from OSM data...';
  } else if (line.includes('Step 2:') && line.includes('Converting')) {
    newProgress.phase = 'Converting Segments';
    newProgress.phaseNumber = 2;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Converting ways to segments...';
  } else if (line.includes('Step 3:') && line.includes('Merging')) {
    newProgress.phase = 'Merging Segments';
    newProgress.phaseNumber = 3;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Merging segments by street...';
  } else if (line.includes('Step 4:') && line.includes('elevations')) {
    newProgress.phase = 'Extracting Elevations';
    newProgress.phaseNumber = 4;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Fetching elevation data...';
  } else if (line.includes('Step 5:') && line.includes('climbs')) {
    newProgress.phase = 'Analyzing Climbs';
    newProgress.phaseNumber = 5;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Detecting and scoring climbs...';
  } else if (line.includes('Step 6:') && (line.includes('Post-processing') || line.includes('merge') || line.includes('Boundary'))) {
    newProgress.phase = 'Finalizing';
    newProgress.phaseNumber = 6;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Finalizing and merging climbs...';
  } else if (line.includes('Writing output') || line.includes('Generating results') || line.includes('Saving')) {
    newProgress.phase = 'Generating Output';
    newProgress.phaseNumber = 6;
    newProgress.totalPhases = 6;
    newProgress.stepDescription = 'Saving results...';
  }

  // Progress bar detection from tqdm output
  // Format: 45%|████▌     | 234/520 [00:12<00:15, 18.2it/s]
  const tqdmMatch = line.match(/(\d+)%\|[^\|]+\|\s*(\d+)\/(\d+)\s*\[([^\<]+)<([^,\]]+)/);
  if (tqdmMatch) {
    const percent = parseInt(tqdmMatch[1], 10);
    const completed = parseInt(tqdmMatch[2], 10);
    const total = parseInt(tqdmMatch[3], 10);
    const timeRemaining = tqdmMatch[5].trim();  // Just the time part (e.g., "00:15")

    newProgress.stepProgress = percent;
    newProgress.itemsCompleted = completed;
    newProgress.itemsTotal = total;
    newProgress.estimatedTimeRemaining = timeRemaining;

    // Update description based on current phase
    if (newProgress.phase === 'Extracting Ways') {
      newProgress.stepDescription = `Processing way ${completed} of ${total}`;
    } else if (newProgress.phase === 'Detecting Climbs') {
      newProgress.stepDescription = `Analyzing segment ${completed} of ${total}`;
    } else if (newProgress.phase === 'Enriching Elevation') {
      newProgress.stepDescription = `Enriching climb ${completed} of ${total}`;
    } else if (newProgress.phase === 'Finalizing') {
      // Check if this is "Finding splits" or "Applying merges" from tqdm desc
      if (line.includes('Finding splits')) {
        newProgress.stepDescription = `Finding cross-chunk splits: ${completed} of ${total} streets`;
      } else if (line.includes('Applying merges')) {
        newProgress.stepDescription = `Applying merges: ${completed} of ${total}`;
      } else {
        newProgress.stepDescription = `Finalizing: ${completed} of ${total}`;
      }
    } else {
      newProgress.stepDescription = `Processing item ${completed} of ${total}`;
    }
  }

  // Count-based progress (e.g., "Processing 45/200 climbs")
  const countMatch = line.match(/(\d+)\/(\d+)\s+(ways|climbs|segments|points)/i);
  if (countMatch) {
    const completed = parseInt(countMatch[1], 10);
    const total = parseInt(countMatch[2], 10);
    const itemType = countMatch[3];

    newProgress.itemsCompleted = completed;
    newProgress.itemsTotal = total;
    newProgress.stepProgress = Math.round((completed / total) * 100);
    newProgress.stepDescription = `Processing ${completed} of ${total} ${itemType}`;

    // Calculate estimated time if we have start time
    const elapsed = Date.now() - newProgress.startTime;
    if (completed > 0 && total > 0) {
      const estimatedTotal = (elapsed / completed) * total;
      const remaining = Math.max(0, estimatedTotal - elapsed);
      const minutes = Math.floor(remaining / 60000);
      const seconds = Math.floor((remaining % 60000) / 1000);
      newProgress.estimatedTimeRemaining = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
    }
  }

  // Completion detection
  if (line.includes('Analysis complete') || line.includes('✓ Results saved')) {
    newProgress.phase = 'Complete';
    newProgress.phaseNumber = 6;
    newProgress.totalPhases = 6;
    newProgress.stepProgress = 100;
    newProgress.estimatedTimeRemaining = '0s';
  }

  return newProgress;
}

export async function POST(request: Request): Promise<Response> {
  try {
    const config = await request.json();
    const jobId = uuidv4();

    // Build command line arguments based on config
    const args: string[] = [];

    // Analysis mode
    if (config.mode === 'address') {
      if (!config.address || !config.radius) {
        return NextResponse.json(
          { error: 'Address and radius are required for address mode' },
          { status: 400 }
        );
      }
      args.push('--address', config.address);
      args.push('--distance', config.radius.toString());
    } else if (config.mode === 'region') {
      if (!config.region) {
        return NextResponse.json(
          { error: 'Region is required for region mode' },
          { status: 400 }
        );
      }
      args.push('--run-region', config.region);
    } else if (config.mode === 'batch') {
      if (!config.regions || config.regions.length === 0) {
        return NextResponse.json(
          { error: 'Regions are required for batch mode' },
          { status: 400 }
        );
      }
      args.push('--run-region', config.regions.join(','));
    }

    // Surface filter
    if (config.surfaceFilter && config.surfaceFilter !== 'all') {
      args.push('--surface-filter', config.surfaceFilter);
    }

    // Cycling filter is always disabled - filtering happens in visualization
    // This ensures all climbs are included in the analysis

    // Units
    if (config.units) {
      args.push('--units', config.units);
    }

    // Min score (uses basic scoring for filtering - all 3 scores are always calculated)
    if (config.minScore !== undefined && config.minScore !== null) {
      args.push('--score-type', 'basic');  // Default to basic score for filtering
      args.push('--min-score', config.minScore.toString());
    }

    // Delete data on complete (cleanup all data for this region after analysis)
    if (config.deleteDataOnComplete) {
      args.push('--cleanup-all-data');
    }

    // Cloud cache upload - read from global config
    try {
      const configPath = path.join(process.cwd(), '..', 'config.yaml');
      const configContent = await fs.readFile(configPath, 'utf-8');
      const globalConfig = yaml.parse(configContent);
      const cloudCacheEnabled = globalConfig.CLOUD_CACHE_ENABLED ?? true;

      if (!cloudCacheEnabled) {
        args.push('--no-cloud-upload');
      }
    } catch (error) {
      // If config can't be read, default to enabled (no --no-cloud-upload flag)
      console.log('Could not read cloud cache config, defaulting to enabled');
    }

    // Get the path to the CLI script (1 level up from gui/)
    const cliPath = path.join(process.cwd(), '..', 'climb_analyzer.py');

    // Spawn the Python process
    const pythonProcess = spawn('python3', [cliPath, ...args], {
      cwd: path.join(process.cwd(), '..'),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        FORCE_TQDM: '1',  // Force tqdm to output even when not in a TTY
        NO_COLOR: '1'     // Disable ANSI color codes to prevent newline corruption
      },
    });

    const logs: string[] = [];
    const progress: Progress = {
      phase: 'Starting',
      phaseNumber: 0,
      totalPhases: 6,
      stepProgress: 0,
      stepDescription: 'Initializing analysis...',
      itemsCompleted: 0,
      itemsTotal: 0,
      estimatedTimeRemaining: 'Calculating...',
      startTime: Date.now()
    };

    // Store the job immediately
    activeJobs.set(jobId, {
      process: pythonProcess,
      logs,
      status: 'running',
      progress
    });

    // Capture stdout
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      processTerminalOutput(output, logs);
      console.log(`[${jobId}] ${output}`);

      // Parse progress from output
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.trim()) {
          const job = activeJobs.get(jobId);
          if (job) {
            job.progress = parseProgressLine(line, job.progress);
          }
        }
      }
    });

    // Capture stderr (tqdm outputs here when FORCE_TQDM=1)
    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString();
      processTerminalOutput(output, logs);
      console.error(`[${jobId}] STDERR: ${output}`);

      // Parse progress from stderr too (tqdm outputs here)
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.trim()) {
          const job = activeJobs.get(jobId);
          if (job) {
            job.progress = parseProgressLine(line, job.progress);
          }
        }
      }
    });

    // Handle process exit
    pythonProcess.on('close', (code) => {
      console.log(`[${jobId}] Process exited with code ${code}`);
      logs.push(`Process completed with exit code ${code}`);

      const job = activeJobs.get(jobId);
      if (job) {
        job.status = code === 0 ? 'completed' : 'failed';
        job.progress.phase = code === 0 ? 'Complete' : 'Failed';
        job.progress.stepProgress = 100;
      }

      // Clean up after some time
      setTimeout(() => {
        activeJobs.delete(jobId);
      }, 300000); // Keep logs for 5 minutes after completion
    });

    return NextResponse.json({
      jobId,
      message: 'Analysis started successfully',
      args: args
    });
  } catch (error) {
    console.error('Failed to start analysis:', error);
    return NextResponse.json(
      { error: 'Failed to start analysis', details: String(error) },
      { status: 500 }
    );
  }
}

// DELETE - Stop a running analysis
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
        { error: 'Job not found' },
        { status: 404 }
      );
    }

    // Send SIGTERM to the process
    job.process.kill('SIGTERM');
    job.status = 'stopped';
    job.progress.phase = 'Stopped';
    job.progress.stepDescription = 'Analysis stopped by user';

    console.log(`[${jobId}] Analysis stopped by user`);

    return NextResponse.json({
      success: true,
      message: 'Analysis stopped successfully',
      jobId
    });
  } catch (error) {
    console.error('Failed to stop analysis:', error);
    return NextResponse.json(
      { error: 'Failed to stop analysis', details: String(error) },
      { status: 500 }
    );
  }
}

// Get job status
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const jobId = searchParams.get('jobId');

  if (!jobId) {
    // Return all active jobs
    return NextResponse.json({
      jobs: Array.from(activeJobs.keys())
    });
  }

  const job = activeJobs.get(jobId);
  if (!job) {
    return NextResponse.json(
      { error: 'Job not found' },
      { status: 404 }
    );
  }

  return NextResponse.json({
    jobId,
    status: job.status,
    running: job.process.exitCode === null,
    exitCode: job.process.exitCode,
    logs: job.logs,
    progress: job.progress
  });
}
