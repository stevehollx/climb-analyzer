import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(): Promise<Response> {
  try {
    const pythonScript = path.join(process.cwd(), '..', 'utils', 'data_indexer.py');

    return new Promise<Response>((resolve) => {
      const pythonProcess = spawn('python3', [pythonScript], {
        cwd: path.join(process.cwd(), '..'),
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
            message: 'Data index updated successfully',
            output
          }));
        } else {
          resolve(NextResponse.json(
            {
              error: 'Failed to reindex data',
              details: errorOutput || output,
              exitCode: code
            },
            { status: 500 }
          ));
        }
      });
    });
  } catch (error) {
    console.error('Failed to reindex data:', error);
    return NextResponse.json(
      { error: 'Failed to reindex data', details: String(error) },
      { status: 500 }
    );
  }
}
