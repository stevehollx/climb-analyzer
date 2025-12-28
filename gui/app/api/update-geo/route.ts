import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(): Promise<Response> {
  try {
    // Get the path to the update script
    const scriptPath = path.join(process.cwd(), '..', '..', 'utils', 'update_geo_definitions.py');

    return new Promise((resolve) => {
      const pythonProcess = spawn('python3', [scriptPath], {
        cwd: path.join(process.cwd(), '..', '..'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
        console.log(data.toString());
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        console.error(data.toString());
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve(NextResponse.json({
            success: true,
            message: 'Geographic boundaries updated successfully',
            output
          }));
        } else {
          resolve(NextResponse.json(
            {
              error: 'Failed to update geographic boundaries',
              details: errorOutput || output,
              exitCode: code
            },
            { status: 500 }
          ));
        }
      });
    });
  } catch (error) {
    console.error('Failed to update geographic boundaries:', error);
    return NextResponse.json(
      { error: 'Failed to update geographic boundaries', details: String(error) },
      { status: 500 }
    );
  }
}
