# Climb Analyzer - Web GUI Setup Guide

Comprehensive guide for installing, configuring, and using the Climb Analyzer web interface.

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Starting the GUI](#-starting-the-gui)
- [Interface Overview](#-interface-overview)
- [Features](#-features)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Development](#-development)

---

## 🌐 Overview

The Climb Analyzer Web GUI is a modern Next.js application that provides an intuitive interface for:
- Configuring and running climb analyses
- Visualizing climbs on interactive maps
- Managing OSM and elevation data downloads
- Monitoring analysis progress in real-time
- Cleaning up cached data and checkpoints

**Technology Stack:**
- **Framework**: Next.js 14.2.5 (App Router)
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS 3.4
- **UI Components**: Radix UI (via shadcn/ui)
- **Mapping**: MapLibre GL JS 5.9.0
- **Icons**: Lucide React

---

## 📦 Installation

### Automatic Installation (Recommended)

The GUI is installed automatically during the setup wizard:

```bash
./climb-analyzer setup
```

When prompted "Install web GUI? (y/n)", answer **y**.

The wizard will:
1. Check for Node.js in the Docker container
2. Run `npm install` to install dependencies
3. Build production assets with `npm run build`
4. Configure the GUI to communicate with the backend

### Manual Installation

If you skipped GUI installation or need to reinstall:

```bash
# Enter the Docker container
./climb-analyzer shell

# Navigate to GUI directory
cd gui/

# Install dependencies (447 MB)
npm install

# Build for production (creates .next folder, ~76 MB)
npm run build

# Exit container
exit
```

**System Requirements:**
- Node.js 18+ (pre-installed in Docker container)
- 550+ MB disk space (447 MB for node_modules, ~76 MB for build)
- 2 GB RAM during build, 512 MB RAM during runtime

---

## 🚀 Starting the GUI

### Quick Start

```bash
# Start GUI server in background (port 3000)
./climb-analyzer -g

# GUI is now accessible at: http://localhost:3000
```

### Additional Commands

```bash
# Stop GUI server
./climb-analyzer --stop-gui

# Restart GUI server
./climb-analyzer --restart-gui

# View GUI server logs
cat gui/gui-server.log

# Check if GUI is running (manual check)
ps aux | grep "node.*gui" | grep -v grep
# Or check the PID file:
cat gui/gui-server.pid 2>/dev/null && echo "GUI is running" || echo "GUI is not running"
```

### Advanced: Manual Server Control

If you prefer to run the GUI manually:

```bash
# Development server (with hot reload)
./climb-analyzer shell
cd gui/
npm run dev

# Production server (optimized, faster)
./climb-analyzer shell
cd gui/
npm run build
npm start
```

**Port Configuration:**

By default, the GUI runs on port **3000**. To use a different port:

```bash
# Development
PORT=3001 npm run dev

# Production
PORT=3001 npm start
```

---

## 🖥️ Interface Overview

### Dashboard (`/`)

**Main Hub** - Overview of system status and quick actions.

**Features:**
- Recent analyses with status (completed, failed, in progress)
- System statistics (disk usage, available regions, data coverage)
- Quick action buttons:
  - Start New Analysis
  - Download Data
  - View Configuration
  - Clean Up Data
- Data coverage summary (OSM regions, elevation datasets)

**Use Case**: Get a quick overview before starting work.

---

### Analyze (`/analyze`)

**Analysis Configuration** - Configure and execute climb analyses with full CLI option support.

**Configuration Options:**

| Section | Options | Description |
|---------|---------|-------------|
| **Region Selection** | US State, Country, Custom Region | Choose analysis area |
| **Climb Type** | All Climbs, Cycling-Only | Filter by cycling accessibility |
| **Score Threshold** | Basic score minimum (default: 6000) | Minimum climb significance |
| **Score Type** | Basic, FIETS, PDI | Climb scoring algorithm |
| **Units** | Imperial, Metric | Distance and elevation units |
| **Surface Filter** | All, Paved, Gravel, Dirt | Road surface type filter |
| **Geocoding** | Enabled, Disabled | Include city/state names |
| **Output** | Custom directory path | Where to save results |

**Workflow:**
1. Select region (e.g., "Rhode Island", "Luxembourg")
2. Configure options (defaults are sensible)
3. Click "Start Analysis"
4. Monitor progress in real-time with progress bar
5. Download results when complete (automatic link appears)

**Features:**
- Real-time progress tracking via Server-Sent Events (SSE)
- Checkpoint resume (automatically resumes if analysis was interrupted)
- Validation (ensures required data is available before starting)
- Error handling with clear error messages
- Analysis history (view past analyses)

**Use Case**: Run new analyses with custom parameters.

---

### Visualize (`/visualize`)

**Map Visualization** - Interactive map showing all climbs with filtering and category highlighting.

**Map Features:**
- **Climb Points**: Each climb displayed as a colored marker
- **Category Colors**:
  - **HC** (Hors Catégorie): Dark Red (#8B0000) - Hardest climbs
  - **Cat 1**: Crimson (#DC143C) - Very difficult
  - **Cat 2**: Tomato (#FF6347) - Difficult
  - **Cat 3**: Orange (#FFA500) - Moderate
  - **Cat 4**: Gold (#FFD700) - Easy
  - **Uncategorized**: Gray (#A9A9A9) - Below threshold
- **Interactive Tooltips**: Hover over climbs to see details
  - Climb name, length, elevation gain
  - Average grade, max grade
  - Basic score, category
  - Surface type, highway classification
- **Zoom Controls**: Zoom in/out, fit bounds
- **Layer Controls**: Toggle climb layers, base maps

**Filtering Options:**
- **By Rank**: Top N climbs (e.g., top 100)
- **By Percentage**: Top X% of climbs (e.g., top 10%)
- **By Score**: Minimum score threshold
- **By Category**: Show/hide specific categories (HC, Cat1-4)
- **By Score Type**: Rank by Basic, FIETS, or PDI score
- **By Surface**: Paved, gravel, dirt, mixed
- **By Cycling**: Cycling-friendly only

**Elevation Profile:**
- Click any climb to see elevation profile chart
- Shows distance vs elevation
- Highlights steepest sections
- Displays grade percentages at intervals

**Use Case**: Explore analysis results visually, find climbs in specific areas, compare difficulty.

---

### Download (`/download`)

**Data Downloader** - Pre-download OSM and elevation data for offline analysis.

**OSM Data Downloads:**
- **Region Search**: Search for US states, countries, or custom regions
- **Coverage Preview**: See available regions from Geofabrik
- **Download Progress**: Real-time progress bar
- **File Size**: Shows file size before download
- **Auto-Configuration**: Automatically updates `config.yaml` after download

**Supported Regions:**
- US States (50 states + territories)
- Countries (200+ countries from Geofabrik)
- Continents (Africa, Asia, Europe, North America, etc.)
- Custom extracts (if URL provided)

**Elevation Data Downloads:**
- **Dataset Selection**:
  - **SRTM**: 1-arc-second (~30m) for 60°N-56°S
  - **ASTER**: 1-arc-second (~30m) global
  - **AW3D30**: 1-arc-second (~30m) global
  - **NED**: 1-arc-second (~10m) for USA
  - **ARCTICDEM**: Arctic regions (60°N+)
- **Region Coverage**: Select region to download tiles for
- **Tile Preview**: Shows number of tiles to download
- **NASA Credentials**: Prompts for Earthdata credentials if not configured
- **Background Downloads**: Continue using GUI while downloading

**Features:**
- Download queue (multiple downloads at once)
- Pause/resume capability
- Disk space validation before download
- Duplicate detection (skip already downloaded)
- Error handling with retry logic

**Use Case**: Prepare data before traveling offline, batch download multiple regions.

---

### Config (`/config`)

**Configuration Manager** - View and edit `config.yaml` settings.

**Editable Settings:**

| Setting | Description | Values |
|---------|-------------|--------|
| **Deployment Type** | Analysis mode | `local`, `cloud` |
| **Elevation API URL** | OpenTopoData server | URL (default: `http://opentopodata-server:5000/v1`) |
| **Elevation Batch Size** | Coords per request | 1-200 (default: 100) |
| **Max Concurrent** | Parallel workers | 1-32 (default: 16 local, 2 cloud) |
| **Dataset Tiers** | Elevation datasets | `primary`, `primary+secondary`, `primary+secondary+tertiary` |
| **Checkpoint Interval** | Auto-save frequency | Minutes (default: 15) |
| **OSM Chunk Size** | Spatial chunk size | Kilometers (default: 30) |
| **Min Climb Length** | Minimum climb filter | Feet (default: 0) |
| **Cloud Cache Enabled** | Share results | `true`, `false` |
| **Cloud Cache Repo** | GitHub repository | `owner/repo` |

**Features:**
- **Syntax Validation**: Ensures YAML is valid before saving
- **Real-time Preview**: See changes before applying
- **Reset to Defaults**: Restore original settings
- **Export Configuration**: Download `config.yaml` as file
- **Import Configuration**: Upload `config.yaml` from file
- **Help Text**: Detailed descriptions for each setting

**Use Case**: Switch between LOCAL and CLOUD modes, tune performance settings, configure cloud cache.

---

### Manage (`/manage`)

**Data Management** - Clean up cached data and monitor disk usage.

**Disk Usage Overview:**
- **OSM Data**: .pbf files by region (MB/GB)
- **Spatial Indexes**: R-tree indexes (MB/GB)
- **Elevation Data**: DEM tiles by dataset (GB)
- **Checkpoints**: Resume state files (MB/GB)
- **Output Files**: Analysis results (MB)
- **Total Used**: Overall disk usage

**Cleanup Operations:**

| Operation | What Gets Deleted | Recoverable? | Disk Space Saved |
|-----------|-------------------|--------------|------------------|
| **Delete Checkpoints** | Resume state for all analyses | No (must restart analyses) | 1-10 GB per large region |
| **Delete OSM Data** | All .pbf files | Yes (re-download) | 150 MB - 5 GB per region |
| **Delete Elevation Data** | All DEM tiles | Yes (re-download) | 10-350 GB per region |
| **Delete Spatial Indexes** | All .idx files | Yes (rebuild with `./climb-analyzer build-index`) | 2-10x OSM file size |
| **Delete Output Files** | Analysis results (.xlsx) | No (must re-run analyses) | 5-50 MB per analysis |
| **Delete All Data** | Everything except config | Partially (must re-download/re-run) | Entire `data/` folder |

**Features:**
- **Selective Deletion**: Delete specific regions or datasets
- **Confirmation Prompts**: Prevents accidental deletion
- **Disk Space Preview**: Shows how much space will be freed
- **Safe Mode**: Protects output files by default
- **Rebuild Index**: Automatically rebuild spatial indexes if deleted

**Use Case**: Free up disk space, remove outdated data, clean up after failed analyses.

---

### Docs (`/docs`)

**Documentation Viewer** - Built-in help and usage examples.

**Documentation Sections:**
- **Getting Started**: Quick start guide
- **CLI Reference**: Complete command-line options
- **Configuration Guide**: Detailed config.yaml documentation
- **Data Management**: OSM and elevation data setup
- **Elevation System**: How elevation data works
- **Checkpointing**: Resume functionality
- **Cloud Cache**: Community data sharing
- **Troubleshooting**: Common issues and solutions

**Features:**
- **Search**: Find documentation by keyword
- **Table of Contents**: Jump to specific sections
- **Code Examples**: Copy-paste ready commands
- **External Links**: Links to GitHub, Geofabrik, NASA Earthdata

**Use Case**: Learn how to use Climb Analyzer, troubleshoot issues, reference CLI options.

---

## 🔌 API Reference

The GUI communicates with the Python backend via Next.js API routes. All routes accept/return JSON.

### Configuration

#### `GET /api/config`
**Description**: Fetch current `config.yaml` settings.

**Response:**
```json
{
  "DEPLOYMENT_TYPE": "local",
  "TOPO_API_BASE_URL": "http://opentopodata-server:5000/v1",
  "ELEVATION_BATCH_SIZE": 100,
  "ELEVATION_MAX_CONCURRENT": 16,
  "ELEVATION_DATASET_TIERS": "primary+secondary+tertiary",
  "CHECKPOINT_INTERVAL_MIN": 15,
  "CHECKPOINT_MILESTONES_PERC": [10, 25, 50, 75, 90],
  "OSM_CHUNK_SIZE_KM": 30,
  "CLOUD_CACHE_ENABLED": true,
  "CLOUD_CACHE_REPO": "stevehollx/global-road-and-trail-climbs",
  "OSM_COVERAGE": ["Rhode Island", "Luxembourg"],
  "OSM_PLANET_DATA": ["rhode-island-latest.osm.pbf"],
  "ELEVATION_DATASETS": {}
}
```

#### `POST /api/config`
**Description**: Update `config.yaml` settings.

**Request Body:**
```json
{
  "DEPLOYMENT_TYPE": "cloud",
  "ELEVATION_MAX_CONCURRENT": 2
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully"
}
```

---

### Analysis

#### `POST /api/analyze`
**Description**: Start a new climb analysis job.

**Request Body:**
```json
{
  "region": "Rhode Island",
  "cyclingOnly": false,
  "scoreThreshold": 6000,
  "scoreType": "basic",
  "units": "imperial",
  "geocoding": true,
  "outputDir": "/app/output"
}
```

**Response:**
```json
{
  "jobId": "rhode-island-1731868945",
  "status": "started",
  "message": "Analysis started successfully"
}
```

#### `GET /api/progress/[jobId]`
**Description**: Server-Sent Events (SSE) stream for real-time progress updates.

**Response** (SSE format):
```
data: {"percent": 0, "message": "Initializing analysis..."}

data: {"percent": 15, "message": "Processing OSM data... (batch 1/10)"}

data: {"percent": 45, "message": "Fetching elevations... (batch 5/10)"}

data: {"percent": 85, "message": "Analyzing climbs... (3521 found)"}

data: {"percent": 100, "message": "Analysis complete! Output: Rhode_Island_climbs_all_basic_2025-11-17_v2.0.1_e6000.xlsx", "complete": true}
```

---

### Results

#### `GET /api/results`
**Description**: List all analysis output files in `output/` directory.

**Response:**
```json
{
  "files": [
    {
      "filename": "Rhode_Island_climbs_all_basic_2025-11-17_v2.0.1_e6000.xlsx",
      "path": "/app/output/Rhode_Island_climbs_all_basic_2025-11-17_v2.0.1_e6000.xlsx",
      "size": "1.2 MB",
      "modified": "2025-11-17T14:23:15Z",
      "region": "Rhode Island",
      "scoreType": "basic",
      "threshold": 6000,
      "version": "2.0.1"
    }
  ]
}
```

#### `GET /api/output-file?path=/app/output/file.xlsx`
**Description**: Parse and return climb data from an Excel output file.

**Response:**
```json
{
  "climbs": [
    {
      "streetName": "Mount Hope Road",
      "city": "Bristol",
      "state": "Rhode Island",
      "basicScore": 8542,
      "elevGainFt": 234,
      "lengthMi": 1.2,
      "avgGradePct": 3.5,
      "maxGradePct": 8.9,
      "category": "Cat 4",
      "surface": "paved",
      "cycling": "Yes",
      "lat": 41.6789,
      "lon": -71.2890
    }
  ],
  "totalClimbs": 127,
  "region": "Rhode Island"
}
```

---

### Data Management

#### `GET /api/data-info`
**Description**: Get disk usage information for all data types.

**Response:**
```json
{
  "osm": {
    "files": [
      {"name": "rhode-island-latest.osm.pbf", "size": "152 MB", "region": "Rhode Island"}
    ],
    "totalSize": "152 MB"
  },
  "elevation": {
    "datasets": {
      "srtm": {"tiles": 45, "size": "1.8 GB"},
      "aster": {"tiles": 12, "size": "450 MB"}
    },
    "totalSize": "2.25 GB"
  },
  "checkpoints": {
    "files": [
      {"name": "Rhode_Island_all_country_basic_1731868945", "size": "234 MB"}
    ],
    "totalSize": "234 MB"
  },
  "indexes": {
    "files": [
      {"name": "rhode-island.idx", "size": "512 MB"}
    ],
    "totalSize": "512 MB"
  },
  "output": {
    "files": [
      {"name": "Rhode_Island_climbs_all_basic_2025-11-17_v2.0.1_e6000.xlsx", "size": "1.2 MB"}
    ],
    "totalSize": "1.2 MB"
  },
  "totalDiskUsage": "3.15 GB"
}
```

#### `DELETE /api/data/checkpoints`
**Description**: Delete all checkpoint files (resume state).

**Response:**
```json
{
  "success": true,
  "deletedFiles": 3,
  "freedSpace": "1.2 GB"
}
```

#### `DELETE /api/data/osm`
**Description**: Delete all OSM .pbf files.

**Query Parameters:**
- `region` (optional): Specific region to delete (e.g., `Rhode Island`)

**Response:**
```json
{
  "success": true,
  "deletedFiles": 1,
  "freedSpace": "152 MB"
}
```

#### `DELETE /api/data/elevation`
**Description**: Delete elevation data (DEM tiles).

**Query Parameters:**
- `dataset` (optional): Specific dataset to delete (e.g., `srtm`, `aster`)

**Response:**
```json
{
  "success": true,
  "deletedFiles": 45,
  "freedSpace": "1.8 GB"
}
```

#### `DELETE /api/data/indexes`
**Description**: Delete spatial index files.

**Response:**
```json
{
  "success": true,
  "deletedFiles": 1,
  "freedSpace": "512 MB"
}
```

#### `DELETE /api/data/all`
**Description**: Delete ALL data (OSM, elevation, checkpoints, indexes).

**Warning**: Does NOT delete output files (.xlsx results) by default.

**Query Parameters:**
- `includeOutputs` (optional): Set to `true` to also delete output files

**Response:**
```json
{
  "success": true,
  "deletedFiles": 62,
  "freedSpace": "4.5 GB"
}
```

---

### Downloads

#### `POST /api/download`
**Description**: Download OSM or elevation data.

**Request Body (OSM):**
```json
{
  "type": "osm",
  "region": "Rhode Island"
}
```

**Request Body (Elevation):**
```json
{
  "type": "elevation",
  "dataset": "srtm",
  "region": "Rhode Island"
}
```

**Response:**
```json
{
  "jobId": "download-osm-rhode-island-1731868945",
  "status": "started",
  "message": "Download started",
  "estimatedSize": "152 MB",
  "estimatedTime": "2-3 minutes"
}
```

---

### Regions

#### `GET /api/regions`
**Description**: List all available regions (US states, countries).

**Response:**
```json
{
  "usStates": [
    {"name": "Rhode Island", "code": "RI"},
    {"name": "California", "code": "CA"}
  ],
  "countries": [
    {"name": "Luxembourg", "code": "LU"},
    {"name": "Monaco", "code": "MC"}
  ]
}
```

---

### Utilities

#### `GET /api/data-coverage`
**Description**: Check what data is available for a specific region.

**Query Parameters:**
- `region`: Region name (e.g., `Rhode Island`)

**Response:**
```json
{
  "region": "Rhode Island",
  "hasOSM": true,
  "osmFile": "rhode-island-latest.osm.pbf",
  "hasIndex": true,
  "indexFile": "rhode-island.idx",
  "hasElevation": true,
  "elevationDatasets": ["srtm", "aster"],
  "ready": true
}
```

#### `POST /api/rebuild-opentopodata`
**Description**: Rebuild OpenTopoData container (LOCAL mode only).

**Response:**
```json
{
  "success": true,
  "message": "OpenTopoData container rebuilt successfully"
}
```

#### `POST /api/reindex-data`
**Description**: Rebuild spatial indexes for OSM data.

**Request Body:**
```json
{
  "region": "Rhode Island"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Spatial index rebuilt for Rhode Island",
  "indexSize": "512 MB"
}
```

#### `POST /api/resync-config`
**Description**: Resync `config.yaml` with actual data on disk (update tracked files).

**Response:**
```json
{
  "success": true,
  "added": ["rhode-island-latest.osm.pbf"],
  "removed": ["old-file.osm.pbf"]
}
```

#### `GET /api/geo-boundaries-info`
**Description**: Get geographic boundary information for a region.

**Query Parameters:**
- `region`: Region name

**Response:**
```json
{
  "region": "Rhode Island",
  "boundingBox": {
    "north": 42.0188,
    "south": 41.1460,
    "east": -71.1204,
    "west": -71.8903
  },
  "area": "1,214 sq mi",
  "center": {"lat": 41.5828, "lon": -71.5055}
}
```

#### `POST /api/earthdata-credentials`
**Description**: Store NASA Earthdata credentials for elevation downloads.

**Request Body:**
```json
{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Credentials stored in .credentials/netrc"
}
```

#### `POST /api/update-geo`
**Description**: Update geographic definitions (states, countries, boundaries).

**Response:**
```json
{
  "success": true,
  "message": "Geographic definitions updated from Natural Earth data",
  "regionsUpdated": 245
}
```

#### `GET /api/cloud-cache`
**Description**: Get cloud cache status (GitHub repository integration).

**Response:**
```json
{
  "enabled": true,
  "repo": "stevehollx/global-road-and-trail-climbs",
  "authenticated": true,
  "availableRegions": ["Hawaii", "Luxembourg", "Monaco"],
  "canPush": true
}
```

---

## ⚙️ Configuration

### GUI-Specific Settings

The GUI reads settings from `config.yaml` in the parent directory. No GUI-specific configuration file is needed.

### Port Configuration

Default port: **3000**

To change:

```bash
# Edit gui/package.json
"scripts": {
  "start": "next start -p 3001"  # Change to desired port
}

# Or use environment variable
PORT=3001 npm start
```

### API Base URL

The GUI assumes the backend is accessible at the same host. If running GUI and backend on different machines:

```javascript
// Edit gui/lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';
```

---

## 🔧 Troubleshooting

### GUI Won't Start

**Problem**: `./climb-analyzer -g` fails or returns error.

**Solutions:**

```bash
# 1. Check if port 3000 is in use
lsof -i :3000  # macOS/Linux
netstat -ano | findstr :3000  # Windows

# Kill process using port 3000
kill $(lsof -t -i :3000)

# 2. Check if Node.js is installed in container
./climb-analyzer shell
node --version  # Should show v18.x or later
npm --version
exit

# 3. Reinstall GUI dependencies
./climb-analyzer shell
cd gui/
rm -rf node_modules/ .next/
npm install
npm run build
exit

# 4. Check GUI server logs
cat gui/gui-server.log

# 5. Start manually in foreground to see errors
./climb-analyzer shell
cd gui/
npm run dev
```

---

### GUI Shows "Cannot connect to server"

**Problem**: GUI loads but shows connection errors.

**Causes & Solutions:**

1. **Backend not running**:
   ```bash
   docker ps | grep climb-analyzer
   # Should see climb-analyzer container running
   ```

2. **OpenTopoData not running** (LOCAL mode):
   ```bash
   docker ps | grep opentopodata
   docker logs opentopodata-server
   ```

3. **Incorrect API URL**:
   Check `gui/lib/api.ts` for correct backend URL.

4. **Firewall blocking requests**:
   Ensure Docker containers can communicate on `climb-network`.

---

### Analysis Doesn't Start

**Problem**: Click "Start Analysis" but nothing happens.

**Solutions:**

```bash
# 1. Check browser console for errors (F12 → Console)

# 2. Check backend logs
docker logs climb-analyzer

# 3. Verify data is available
ls data/planet_osm_data/
ls data/elevation_data/

# 4. Test analysis manually
./climb-analyzer -r "Rhode Island"

# 5. Check file permissions
ls -la data/
# Files should be owned by your user, not root
```

---

### Map Doesn't Load Climbs

**Problem**: Visualize page shows empty map.

**Solutions:**

1. **No output files**:
   ```bash
   ls output/
   # Should show .xlsx files
   ```

2. **Output file parsing failed**:
   - Check browser console for errors
   - Verify Excel file is not corrupted
   - Re-run analysis to generate new file

3. **Coordinates missing**:
   - Ensure geocoding was enabled during analysis
   - Check if output file has `lat`/`lon` columns

---

### Download Progress Stuck

**Problem**: Data download appears frozen.

**Solutions:**

```bash
# 1. Check internet connection
ping download.geofabrik.de
ping e4ftl01.cr.usgs.gov

# 2. Check download logs
docker logs climb-analyzer | grep -i download

# 3. Cancel and retry download
# (Use "Cancel" button in GUI, then retry)

# 4. Download manually via CLI
./climb-analyzer
# Select download option from menu
```

---

### High Memory Usage

**Problem**: GUI or backend using excessive RAM.

**Solutions:**

```bash
# 1. Check Docker memory limit
docker stats

# 2. Increase Docker memory allocation
# Docker Desktop → Settings → Resources → Memory
# Set to 8 GB minimum, 16 GB recommended

# 3. Close unused browser tabs

# 4. Reduce ELEVATION_MAX_CONCURRENT in config.yaml
# From 16 to 8 or 4
```

---

### Permission Errors

**Problem**: "Permission denied" when accessing files.

**Solution:**

```bash
# Ensure HOST_UID and HOST_GID are correct in .env
echo "HOST_UID=$(id -u)" > .env
echo "HOST_GID=$(id -g)" >> .env

# Fix ownership of existing files
sudo chown -R $(id -u):$(id -g) data/ output/ gui/

# Restart containers
docker compose down
docker compose up -d
```

---

## 🛠️ Development

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer/gui

# Install dependencies
npm install

# Start development server (hot reload enabled)
npm run dev

# GUI available at: http://localhost:3000
```

### Project Structure

```
gui/
├── app/                      # Next.js App Router
│   ├── layout.tsx            # Root layout with sidebar
│   ├── page.tsx              # Dashboard page
│   ├── analyze/              # Analysis configuration page
│   ├── visualize/            # Map visualization page
│   ├── download/             # Data download page
│   ├── config/               # Configuration page
│   ├── manage/               # Data management page
│   ├── docs/                 # Documentation page
│   └── api/                  # API routes (18 endpoints)
├── components/
│   ├── ui/                   # shadcn/ui components (Button, Card, etc.)
│   ├── layout/               # Layout components (Sidebar, Header)
│   ├── forms/                # Form components
│   ├── map/                  # Map components (MapLibre)
│   └── progress/             # Progress indicators
├── lib/
│   ├── api.ts                # API client functions
│   ├── csv-parser.ts         # Parse climb CSV/Excel files
│   ├── geojson.ts            # GeoJSON conversion
│   └── utils.ts              # Utility functions
├── types/
│   └── climb.ts              # TypeScript type definitions
├── public/                   # Static assets (icons, images)
├── package.json              # Dependencies and scripts
├── tsconfig.json             # TypeScript configuration
├── next.config.mjs           # Next.js configuration
└── tailwind.config.ts        # Tailwind CSS configuration
```

### Adding New Pages

1. Create page directory in `app/`:
   ```bash
   mkdir app/mypage
   touch app/mypage/page.tsx
   ```

2. Add page component:
   ```tsx
   export default function MyPage() {
     return (
       <div>
         <h1>My Page</h1>
       </div>
     );
   }
   ```

3. Update sidebar navigation in `components/layout/Sidebar.tsx`:
   ```tsx
   <Link href="/mypage">My Page</Link>
   ```

### Adding UI Components

Use shadcn/ui CLI or manually add to `components/ui/`:

```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
npx shadcn-ui@latest add dialog
```

### Adding API Routes

1. Create API route file:
   ```bash
   mkdir -p app/api/myendpoint
   touch app/api/myendpoint/route.ts
   ```

2. Implement route handlers:
   ```typescript
   import { NextResponse } from 'next/server';

   export async function GET(request: Request) {
     // Logic here
     return NextResponse.json({ data: 'value' });
   }

   export async function POST(request: Request) {
     const body = await request.json();
     // Logic here
     return NextResponse.json({ success: true });
   }
   ```

3. Add client function in `lib/api.ts`:
   ```typescript
   export async function fetchMyData() {
     const response = await fetch('/api/myendpoint');
     return response.json();
   }
   ```

### TypeScript Types

All climb-related types are in `types/climb.ts`. Update when adding features:

```typescript
export interface Climb {
  streetName: string;
  city: string;
  state: string;
  basicScore: number;
  elevGainFt: number;
  lengthMi: number;
  avgGradePct: number;
  maxGradePct: number;
  category: string;
  // Add new fields here
}
```

### Building for Production

```bash
# Build optimized production bundle
npm run build

# Test production build locally
npm start

# Build output is in .next/ directory
```

---

## 📚 Additional Resources

### GUI Documentation
- [gui/README.md](gui/README.md) - GUI architecture and development
- [gui/QUICKSTART.md](gui/QUICKSTART.md) - Quick start guide
- [gui/IMPLEMENTATION_SUMMARY.md](gui/IMPLEMENTATION_SUMMARY.md) - Implementation details

### Related Documentation
- [INSTALLATION.md](INSTALLATION.md) - Main installation guide
- [docs/CLI_ARGUMENTS_SPEC.md](docs/CLI_ARGUMENTS_SPEC.md) - CLI reference
- [docs/ELEVATION_SYSTEM_SUMMARY.md](docs/ELEVATION_SYSTEM_SUMMARY.md) - Elevation data architecture

### External Resources
- **Next.js Documentation**: https://nextjs.org/docs
- **Tailwind CSS Documentation**: https://tailwindcss.com/docs
- **shadcn/ui Components**: https://ui.shadcn.com/
- **MapLibre GL JS**: https://maplibre.org/maplibre-gl-js-docs/

---

## 📄 License

Part of the Climb Analyzer project. See [LICENSE](LICENSE) for details.

---

**Need Help?** Open an issue on GitHub or consult [docs/](docs/) for detailed technical documentation.

Happy climbing! 🚴⛰️
