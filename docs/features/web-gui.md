# Web GUI

The web GUI provides a visual interface for configuring analyses and viewing results on interactive maps.

As of Dec 2025 v2.2.1, this is in beta, and not thoroughly tested yet. Use the iOS app for visualizations. I'll work to true this up to parity with the features of the CLI analyzer and the iOS visualizer in early 2026.

## Quick Start

```bash
# Start the GUI
./climb-analyzer -g

# Open in browser
# http://localhost:3000
```

## Starting and Stopping

```bash
# Start GUI server
./climb-analyzer -g

# Stop GUI server
./climb-analyzer --stop-gui

# Restart GUI server
./climb-analyzer --restart-gui

# Check if running
ps aux | grep "node.*gui"
```

## Pages

### Dashboard (`/`)

Overview of system status and quick actions:

- Recent analyses
- Data coverage summary
- Quick action buttons
- System statistics

### Analyze (`/analyze`)

Configure and run climb analyses:

**Options:**
- Region selection (state, country, custom)
- Surface filter (all, paved, gravel, dirt)
- Score type (basic, FIETS, PDI)
- Units (imperial, metric)
- Cycling filter
- Geocoding

**Features:**
- Real-time progress tracking
- Checkpoint resume
- Error handling with clear messages

### Visualize (`/visualize`)

Interactive map showing climb locations:

**Map Features:**
- Colored markers by category (HC, Cat 1-4)
- Hover tooltips with climb details
- Click for elevation profile
- Zoom and pan controls

**Filtering:**
- By rank (top N climbs)
- By percentage (top X%)
- By score threshold
- By category
- By surface type
- By cycling accessibility

**Category Colors:**
- **HC**: Dark Red
- **Cat 1**: Crimson
- **Cat 2**: Tomato
- **Cat 3**: Orange
- **Cat 4**: Gold
- **Uncategorized**: Gray

### Download (`/download`)

Manage data downloads:

**OSM Data:**
- Search for regions
- Download progress
- File size preview

**Elevation Data:**
- Dataset selection (SRTM, NED, ASTER, etc.)
- Tile preview
- Background downloads

### Config (`/config`)

View and edit configuration:

| Setting | Description |
|---------|-------------|
| Deployment Type | local or cloud |
| Elevation API URL | OpenTopoData server |
| Batch Size | Coords per request |
| Max Concurrent | Parallel workers |
| Cloud Cache | Enable/disable |

### Manage (`/manage`)

Data management and cleanup:

**Disk Usage:**
- OSM data by region
- Elevation data by dataset
- Checkpoints
- Output files

**Cleanup Operations:**
- Delete checkpoints
- Delete OSM data
- Delete elevation data
- Delete all data

## Installation

### Automatic (Recommended)

During setup wizard:

```bash
./climb-analyzer setup
# Answer 'y' when prompted to install GUI
```

### Manual

```bash
# Enter Docker container
./climb-analyzer shell

# Install dependencies
cd gui/
npm install

# Build for production
npm run build

# Exit container
exit
```

## Configuration

### Port

Default: **3000**

To change:

```bash
# Start on different port
PORT=3001 npm start
```

### API URL

The GUI communicates with the backend at the same host. If running separately:

```javascript
// gui/lib/api.ts
const API_BASE_URL = 'http://your-backend-host:3000';
```

## Development

### Development Server

```bash
./climb-analyzer shell
cd gui/
npm run dev
```

Hot reload enabled for development.

### Production Build

```bash
npm run build
npm start
```

### Project Structure

```
gui/
├── app/                 # Next.js pages
│   ├── analyze/
│   ├── visualize/
│   ├── download/
│   ├── config/
│   ├── manage/
│   └── api/            # API routes
├── components/
│   ├── ui/             # UI components
│   ├── map/            # Map components
│   └── forms/          # Form components
├── lib/                # Utilities
└── types/              # TypeScript types
```

## Troubleshooting

### GUI Won't Start

```bash
# Check port 3000
lsof -i :3000

# Kill process using port
kill $(lsof -t -i :3000)

# Reinstall dependencies
./climb-analyzer shell
cd gui/
rm -rf node_modules/ .next/
npm install
npm run build
```

### "Cannot connect to server"

1. Verify Docker container is running:
   ```bash
   docker ps | grep climb-analyzer
   ```

2. Check OpenTopoData (local mode):
   ```bash
   docker ps | grep opentopodata
   ```

### Map Doesn't Load

1. Check output files exist:
   ```bash
   ls output/*.xlsx
   ```

2. Verify file has coordinates (lat/lon columns)

3. Check browser console for errors

### High Memory Usage

```bash
# Reduce concurrent workers in config.yaml
ELEVATION_MAX_CONCURRENT: 4  # From 16
```

## Technology Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS 3.4
- **UI Components**: Radix UI (shadcn/ui)
- **Mapping**: MapLibre GL JS
- **Icons**: Lucide React

---

Next: [Large Country Analysis](large-countries.md)
