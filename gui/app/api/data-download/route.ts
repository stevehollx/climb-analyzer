import { NextResponse } from 'next/server';
import { promises as fs, createWriteStream } from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';

const ALLOWED_HOST = 'github.com';
const ALLOWED_OBJECTS_HOST = 'objects.githubusercontent.com';

function safeFilename(name: string): boolean {
  // Reject anything that could traverse or hit hidden files
  return /^[A-Za-z0-9._-]+$/.test(name) && !name.startsWith('.');
}

export async function POST(request: Request): Promise<Response> {
  let body: { url?: string; filename?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { url, filename } = body;
  if (!url || !filename) {
    return NextResponse.json({ error: 'url and filename are required' }, { status: 400 });
  }
  if (!safeFilename(filename)) {
    return NextResponse.json({ error: 'Unsafe filename' }, { status: 400 });
  }

  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return NextResponse.json({ error: 'Invalid URL' }, { status: 400 });
  }
  if (parsed.host !== ALLOWED_HOST && parsed.host !== ALLOWED_OBJECTS_HOST) {
    return NextResponse.json(
      { error: `Only github.com download URLs are allowed (got ${parsed.host})` },
      { status: 400 }
    );
  }

  const outputDir = path.join(process.cwd(), '..', 'output');
  await fs.mkdir(outputDir, { recursive: true });
  const outPath = path.join(outputDir, filename);

  // If already present, treat as success
  try {
    const stat = await fs.stat(outPath);
    if (stat.size > 0) {
      return NextResponse.json({ ok: true, path: outPath, size: stat.size, alreadyExists: true });
    }
  } catch {}

  try {
    const res = await fetch(parsed.toString(), { redirect: 'follow' });
    if (!res.ok || !res.body) {
      return NextResponse.json(
        { error: `Upstream returned ${res.status}` },
        { status: 502 }
      );
    }

    const tmp = outPath + '.part';
    const ws = createWriteStream(tmp);
    // @ts-ignore – Node 18+ supports fromWeb
    await pipeline(Readable.fromWeb(res.body as any), ws);
    await fs.rename(tmp, outPath);

    const stat = await fs.stat(outPath);
    return NextResponse.json({ ok: true, path: outPath, size: stat.size });
  } catch (err) {
    return NextResponse.json(
      { error: `Download failed: ${err instanceof Error ? err.message : err}` },
      { status: 500 }
    );
  }
}
