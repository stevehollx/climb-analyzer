import { NextResponse } from 'next/server';

/**
 * DEPRECATED: NASA Earthdata credentials are no longer needed.
 *
 * As of December 2025, NASA LP DAAC Data Pool was retired.
 * All elevation datasets now use public sources:
 * - SRTM: OpenTopography S3 (public, no auth)
 * - AW3D30: JAXA FTP (public, no auth)
 * - NED, ArcticDEM, REMA: AWS S3 (public, no auth)
 */

// GET - Return deprecated status
export async function GET(): Promise<Response> {
  return NextResponse.json({
    deprecated: true,
    message: 'NASA Earthdata credentials are no longer needed. All datasets now use public sources.',
    hasCredentials: false,
    username: '',
    password: '',
    source: null,
  });
}

// POST - Return deprecated status
export async function POST(): Promise<Response> {
  return NextResponse.json({
    deprecated: true,
    success: false,
    message: 'NASA Earthdata credentials are no longer needed. All datasets now use public sources.',
  });
}

// DELETE - Return success (nothing to delete)
export async function DELETE(): Promise<Response> {
  return NextResponse.json({
    deprecated: true,
    success: true,
    message: 'NASA Earthdata credentials are no longer needed. Nothing to delete.',
  });
}
