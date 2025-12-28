import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(): Promise<Response> {
  try {
    console.log('Rebuilding OpenTopoData server...');

    const pythonScript = `
import sys
sys.path.insert(0, '${path.join(process.cwd(), '..')}')

from utils.opentopodata_manager import rebuild_and_restart

success = rebuild_and_restart(validate_health=True)
if success:
    print("REBUILD_SUCCESS")
else:
    print("REBUILD_FAILED")
`;

    return new Promise((resolve) => {
      const pythonProcess = spawn('python3', ['-c', pythonScript], {
        cwd: path.join(process.cwd(), '..'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        const text = data.toString();
        output += text;
        console.log('[OpenTopoData rebuild]', text);
      });

      pythonProcess.stderr.on('data', (data) => {
        const text = data.toString();
        errorOutput += text;
        console.error('[OpenTopoData rebuild error]', text);
      });

      pythonProcess.on('close', (code) => {
        if (code === 0 && output.includes('REBUILD_SUCCESS')) {
          resolve(NextResponse.json({
            success: true,
            message: 'OpenTopoData server rebuilt and ready',
            output: output,
          }));
        } else {
          resolve(NextResponse.json(
            {
              error: 'OpenTopoData rebuild failed',
              details: errorOutput || output,
              exitCode: code,
            },
            { status: 500 }
          ));
        }
      });

      // Timeout after 10 minutes (rebuild can take a while)
      setTimeout(() => {
        pythonProcess.kill();
        resolve(NextResponse.json(
          {
            error: 'OpenTopoData rebuild timed out',
            details: 'Rebuild took longer than 10 minutes',
          },
          { status: 408 }
        ));
      }, 600000);
    });
  } catch (error) {
    console.error('Failed to rebuild OpenTopoData:', error);
    return NextResponse.json(
      { error: 'Failed to rebuild OpenTopoData', details: String(error) },
      { status: 500 }
    );
  }
}

// Quick restart endpoint (no rebuild)
export async function PUT(): Promise<Response> {
  try {
    console.log('Restarting OpenTopoData server...');

    const pythonScript = `
import sys
sys.path.insert(0, '${path.join(process.cwd(), '..')}')

from utils.opentopodata_manager import quick_restart

success = quick_restart()
if success:
    print("RESTART_SUCCESS")
else:
    print("RESTART_FAILED")
`;

    return new Promise((resolve) => {
      const pythonProcess = spawn('python3', ['-c', pythonScript], {
        cwd: path.join(process.cwd(), '..'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        const text = data.toString();
        output += text;
        console.log('[OpenTopoData restart]', text);
      });

      pythonProcess.stderr.on('data', (data) => {
        const text = data.toString();
        errorOutput += text;
        console.error('[OpenTopoData restart error]', text);
      });

      pythonProcess.on('close', (code) => {
        if (code === 0 && output.includes('RESTART_SUCCESS')) {
          resolve(NextResponse.json({
            success: true,
            message: 'OpenTopoData server restarted',
            output: output,
          }));
        } else {
          resolve(NextResponse.json(
            {
              error: 'OpenTopoData restart failed',
              details: errorOutput || output,
              exitCode: code,
            },
            { status: 500 }
          ));
        }
      });

      // Timeout after 2 minutes for restart
      setTimeout(() => {
        pythonProcess.kill();
        resolve(NextResponse.json(
          {
            error: 'OpenTopoData restart timed out',
            details: 'Restart took longer than 2 minutes',
          },
          { status: 408 }
        ));
      }, 120000);
    });
  } catch (error) {
    console.error('Failed to restart OpenTopoData:', error);
    return NextResponse.json(
      { error: 'Failed to restart OpenTopoData', details: String(error) },
      { status: 500 }
    );
  }
}
