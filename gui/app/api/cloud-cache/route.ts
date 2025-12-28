import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function GET(request: Request): Promise<Response> {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');

    if (action === 'list') {
      // List available cloud cache regions by fetching from GitHub
      return listAvailableRegions();
    } else if (action === 'download') {
      const region = searchParams.get('region');
      if (!region) {
        return NextResponse.json(
          { error: 'Region is required' },
          { status: 400 }
        );
      }
      return downloadCloudCache(region);
    }

    return NextResponse.json(
      { error: 'Invalid action. Use ?action=list or ?action=download&region=xyz' },
      { status: 400 }
    );
  } catch (error) {
    console.error('Cloud cache API error:', error);
    return NextResponse.json(
      { error: 'Failed to process request', details: String(error) },
      { status: 500 }
    );
  }
}

async function listAvailableRegions(): Promise<Response> {
  return new Promise<Response>((resolve) => {
    const pythonScript = path.join(process.cwd(), '..', 'utils', 'cloud_cache.py');

    // Run Python script to list available regions with hierarchical structure
    const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.insert(0, '${path.join(process.cwd(), '..')}')
from utils.cloud_cache import CloudCacheManager
import json
from collections import defaultdict

cache = CloudCacheManager()
if not cache.is_token_configured():
    print(json.dumps({'error': 'Cloud cache not configured'}))
    sys.exit(0)

# Get list of available files from GitHub
try:
    # Recursively walk the repository structure: continent/country/state
    all_files = []

    # List continents (top-level directories)
    continents = cache.github.list_directory_files('')

    for continent_item in continents:
        # Skip if it's a file (we only want directories)
        if '.' not in continent_item:
            continent_path = continent_item

            # List countries within this continent
            countries = cache.github.list_directory_files(continent_path)

            for country_item in countries:
                if '.' not in country_item:
                    country_path = f"{continent_path}/{country_item}"

                    # List files/states within this country
                    items = cache.github.list_directory_files(country_path)

                    for item in items:
                        if '.' in item:
                            # It's a file - country-level analysis
                            all_files.append(f"{country_path}/{item}")
                        else:
                            # It's a state directory
                            state_path = f"{country_path}/{item}"
                            state_files = cache.github.list_directory_files(state_path)
                            for state_file in state_files:
                                all_files.append(f"{state_path}/{state_file}")

    # Build hierarchical structure with file counts
    # Structure: { "continent": { "country": { "state": file_count } } }
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for file_path in all_files:
        # Look for .xlsx files
        if file_path.endswith('.xlsx'):
            # Extract region path (e.g., "north-america/united-states-of-america/vermont/file.xlsx")
            parts = file_path.split('/')

            if len(parts) >= 3:
                # State level: continent/country/state/file.xlsx
                continent = parts[0]
                country = parts[1]
                state = parts[2]
                hierarchy[continent][country][state] += 1
            elif len(parts) == 2:
                # Country level: continent/file.xlsx
                continent = parts[0]
                # Use __direct__ to indicate files directly in continent folder
                hierarchy[continent]['__direct__']['__files__'] += 1

    # Convert to proper nested structure for frontend
    def format_name(slug):
        """Convert slug to display name"""
        return slug.replace('-', ' ').title()

    regions = []
    for continent in sorted(hierarchy.keys()):
        continent_data = {
            'name': format_name(continent),
            'path': continent,
            'fileCount': 0,
            'children': []
        }

        for country in sorted(hierarchy[continent].keys()):
            if country == '__direct__':
                # Files directly in continent folder
                continent_data['fileCount'] += hierarchy[continent][country]['__files__']
                continue

            country_data = {
                'name': format_name(country),
                'path': f"{continent}/{country}",
                'fileCount': 0,
                'children': []
            }

            for state in sorted(hierarchy[continent][country].keys()):
                file_count = hierarchy[continent][country][state]
                state_data = {
                    'name': format_name(state),
                    'path': f"{continent}/{country}/{state}",
                    'fileCount': file_count
                }
                country_data['children'].append(state_data)
                country_data['fileCount'] += file_count

            continent_data['children'].append(country_data)
            continent_data['fileCount'] += country_data['fileCount']

        regions.append(continent_data)

    print(json.dumps({'regions': regions}))
except Exception as e:
    import traceback
    print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))
`], {
      cwd: path.join(process.cwd(), '..'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        resolve(NextResponse.json(
          { error: 'Failed to list cloud cache regions', details: errorOutput },
          { status: 500 }
        ));
        return;
      }

      try {
        const result = JSON.parse(output);
        if (result.error) {
          resolve(NextResponse.json(
            { error: result.error, available: false, traceback: result.traceback },
            { status: 200 }
          ));
        } else {
          resolve(NextResponse.json({
            regions: result.regions || [],
            available: true
          }));
        }
      } catch (e) {
        resolve(NextResponse.json(
          { error: 'Failed to parse response', details: String(e), output },
          { status: 500 }
        ));
      }
    });
  });
}

async function downloadCloudCache(region: string): Promise<Response> {
  return new Promise<Response>((resolve) => {
    // Download directly from cloud cache to output directory
    const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.insert(0, '${path.join(process.cwd(), '..')}')
from utils.cloud_cache import CloudCacheManager
from pathlib import Path
import json

cache = CloudCacheManager()
if not cache.is_token_configured():
    print(json.dumps({'error': 'Cloud cache not configured'}))
    sys.exit(1)

try:
    # Parse region path to determine country and state
    region_path = '${region}'
    parts = region_path.split('/')

    country = None
    state = None
    scope_type = 'country'

    if len(parts) == 3:
        # continent/country/state format
        country = parts[1].replace('-', ' ').title()
        state = parts[2].replace('-', ' ').title()
        scope_type = 'state'
    elif len(parts) == 2:
        # continent/country format
        country = parts[1].replace('-', ' ').title()
        scope_type = 'country'
    else:
        print(json.dumps({'error': 'Invalid region path format'}))
        sys.exit(1)

    # Check if data exists in cloud cache
    cache_info = cache.check_cached(country, state, scope_type)

    if not cache_info['exists']:
        print(json.dumps({'error': f'No cloud cache data found for {region_path}'}))
        sys.exit(1)

    # Download to output directory
    output_dir = Path('output')
    result = cache.download_cache(cache_info, output_dir)

    if result['xlsx_files']:
        print(json.dumps({
            'success': True,
            'files': [str(f) for f in result['xlsx_files']],
            'csv_file': str(result['csv_file']) if result['csv_file'] else None,
            'total_parts': result['total_parts']
        }))
    else:
        print(json.dumps({'error': 'No files downloaded'}))
        sys.exit(1)

except Exception as e:
    import traceback
    print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))
    sys.exit(1)
`], {
      cwd: path.join(process.cwd(), '..'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        try {
          const error = JSON.parse(output);
          resolve(NextResponse.json(
            { error: error.error || 'Failed to download from cloud cache', details: error.traceback || errorOutput },
            { status: 500 }
          ));
        } catch (e) {
          resolve(NextResponse.json(
            { error: 'Failed to download from cloud cache', details: errorOutput || output },
            { status: 500 }
          ));
        }
        return;
      }

      try {
        const result = JSON.parse(output);
        resolve(NextResponse.json({
          success: true,
          message: `Successfully downloaded ${result.files.length} file(s) to output directory`,
          region,
          files: result.files,
          totalParts: result.total_parts
        }));
      } catch (e) {
        resolve(NextResponse.json(
          { error: 'Failed to parse download result', details: String(e), output },
          { status: 500 }
        ));
      }
    });
  });
}

export async function POST(request: Request): Promise<Response> {
  try {
    const { region, filePath } = await request.json();

    if (!region || !filePath) {
      return NextResponse.json(
        { error: 'Region and filePath are required' },
        { status: 400 }
      );
    }

    // Upload analysis results to cloud cache
    return new Promise<Response>((resolve) => {
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.insert(0, '${path.join(process.cwd(), '..')}')
from utils.cloud_cache import CloudCacheManager
from pathlib import Path
import json

cache = CloudCacheManager()
if not cache.is_token_configured():
    print(json.dumps({'error': 'Cloud cache not configured'}))
    sys.exit(1)

try:
    # Upload file to cloud cache
    file_path = Path('${filePath}')
    result = cache.upload_cache('${region}', file_path)
    print(json.dumps({'success': True, 'message': 'Uploaded successfully'}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
    sys.exit(1)
`], {
        cwd: path.join(process.cwd(), '..'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          resolve(NextResponse.json(
            { error: 'Failed to upload to cloud cache', details: errorOutput },
            { status: 500 }
          ));
          return;
        }

        try {
          const result = JSON.parse(output);
          if (result.error) {
            resolve(NextResponse.json(
              { error: result.error },
              { status: 500 }
            ));
          } else {
            resolve(NextResponse.json(result));
          }
        } catch (e) {
          resolve(NextResponse.json(
            { error: 'Failed to parse response', details: String(e) },
            { status: 500 }
          ));
        }
      });
    });
  } catch (error) {
    console.error('Failed to upload to cloud cache:', error);
    return NextResponse.json(
      { error: 'Failed to upload to cloud cache', details: String(error) },
      { status: 500 }
    );
  }
}
