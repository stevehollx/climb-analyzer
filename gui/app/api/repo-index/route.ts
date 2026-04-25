import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const REPO = 'stevehollx/global-road-and-trail-climbs';
const REMOTE_URL = `https://raw.githubusercontent.com/${REPO}/main/index.json`;

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const forceRemote = url.searchParams.get('remote') === '1';

  // Try local first (the repo lives at /mnt/usb1/ca11/index.json — gui/ is one level deeper)
  if (!forceRemote) {
    try {
      const localPath = path.join(process.cwd(), '..', 'index.json');
      const txt = await fs.readFile(localPath, 'utf-8');
      const json = JSON.parse(txt);
      return NextResponse.json({ source: 'local', path: localPath, ...json });
    } catch {
      // fall through to remote
    }
  }

  try {
    const res = await fetch(REMOTE_URL, { cache: 'no-store' });
    if (!res.ok) {
      return NextResponse.json(
        { error: `GitHub returned ${res.status}` },
        { status: res.status }
      );
    }
    const json = await res.json();
    return NextResponse.json({ source: 'remote', url: REMOTE_URL, ...json });
  } catch (err) {
    return NextResponse.json(
      { error: `Failed to fetch index: ${err instanceof Error ? err.message : err}` },
      { status: 502 }
    );
  }
}
