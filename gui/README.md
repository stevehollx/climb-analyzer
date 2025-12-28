# Climb Analyzer - Web UI

Modern web interface for the Climb Analyzer built with Next.js, TypeScript, and Tailwind CSS.

## Features

- **Dashboard**: Overview of analyses, statistics, and quick actions
- **Run Analysis**: Configure and execute climb analysis with all CLI options
- **Visualize Climbs**: Interactive map with climb filtering and category highlighting
- **Download Data**: Pre-download OSM and elevation data for offline use
- **Configuration**: Manage deployment mode and system settings
- **Data Management**: Clean up cached data and checkpoints
- **Documentation**: Built-in help and usage examples

## Technology Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Components**: shadcn/ui (Radix UI primitives)
- **Icons**: Lucide React
- **Mapping**: MapLibre GL JS (planned)
- **State**: Zustand (lightweight state management)

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Climb Analyzer Python backend (in parent directory)

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The app will be available at [http://localhost:3000](http://localhost:3000)

## Project Structure

```
ui/
├── app/                    # Next.js app directory
│   ├── layout.tsx          # Root layout with sidebar
│   ├── page.tsx            # Dashboard page
│   ├── analyze/            # Analysis configuration page
│   ├── visualize/          # Map visualization page
│   ├── download/           # Data download page
│   ├── config/             # Configuration page
│   ├── manage/             # Data management page
│   ├── docs/               # Documentation page
│   └── api/                # API routes (backend interface)
│       ├── config/         # Config CRUD
│       ├── analyze/        # Start analysis
│       ├── download/       # Download data
│       ├── results/        # List/load output files
│       ├── progress/       # SSE progress updates
│       └── data/           # Data management operations
├── components/
│   ├── ui/                 # Reusable UI components
│   ├── layout/             # Layout components (Sidebar, etc.)
│   ├── map/                # Map components (TODO)
│   ├── forms/              # Form components (TODO)
│   └── progress/           # Progress indicators (TODO)
├── lib/
│   ├── api.ts              # API client functions
│   ├── csv-parser.ts       # Parse climb CSV files
│   ├── geojson.ts          # GeoJSON conversion
│   └── utils.ts            # Utility functions
├── types/
│   └── climb.ts            # TypeScript type definitions
└── public/                 # Static assets
```

## API Integration

The web UI communicates with the Python CLI via Next.js API routes:

- **GET /api/config**: Fetch config.yaml settings
- **POST /api/config**: Update configuration
- **POST /api/analyze**: Start analysis job
- **POST /api/download**: Download region data
- **GET /api/results**: List output CSV files
- **GET /api/results/[file]**: Load specific CSV file
- **GET /api/progress/[jobId]**: SSE stream for progress updates
- **DELETE /api/data/checkpoints**: Delete checkpoint files
- **DELETE /api/data/osm**: Delete OSM data
- **DELETE /api/data/elevation**: Delete elevation data

## Configuration

The UI reads and writes to `../config.yaml` in the parent directory. Settings include:

- **Deployment Type**: Cloud (Overpass API) or Local (planet files)
- **Elevation Batch Size**: Points per elevation request
- **Max Concurrent Requests**: Concurrent elevation fetches
- **Checkpoint Interval**: Auto-save frequency

## CLI Integration

All CLI options are available in the web UI:

### Analysis Modes
- **Address**: Analyze within radius of an address
- **Region**: Analyze entire state/country
- **Batch**: Process multiple regions

### Parameters
- Surface filter (paved, gravel, dirt, all)
- Cycling accessibility filter
- Units (metric/imperial)
- Score type (basic, FIETS, PDI)
- Minimum score threshold
- Geocoding (yes/no)
- Delete data after analysis

## Map Visualization (Planned)

The visualize page will use MapLibre GL JS to display:

- Climb locations as points (or lines when OSM geometry is fetched)
- Color-coded by category (HC, Cat 1-4, Uncategorized)
- Interactive hover tooltips with climb details
- Filtering by top N or top percentage
- Score type selection for ranking

### Category Colors
- **HC**: Dark Red (#8B0000)
- **Cat 1**: Crimson (#DC143C)
- **Cat 2**: Tomato (#FF6347)
- **Cat 3**: Orange (#FFA500)
- **Cat 4**: Gold (#FFD700)
- **Uncategorized**: Gray (#A9A9A9)

## Development

### Adding New Pages

1. Create page directory in `app/`
2. Add `page.tsx` file
3. Update sidebar navigation in `components/layout/Sidebar.tsx`

### Adding UI Components

Use shadcn/ui CLI or manually add to `components/ui/`:

```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
```

### TypeScript Types

All climb-related types are in `types/climb.ts`. Update as needed when adding features.

## TODO

### High Priority
- [ ] Implement API routes for analysis execution
- [ ] Add SSE progress streaming
- [ ] Integrate MapLibre GL JS in visualize page
- [ ] Load and parse CSV files from ../output
- [ ] Display climbs on map with category colors

### Medium Priority
- [ ] Add real-time log streaming
- [ ] Implement file upload for results
- [ ] Add dashboard statistics (parse output files)
- [ ] Fetch full climb geometry from OSM API
- [ ] Add elevation profile charts

### Low Priority
- [ ] Docker configuration
- [ ] Add user authentication (optional)
- [ ] Store analysis history in database
- [ ] Export filtered climb lists
- [ ] 3D terrain visualization with elevation tiles

## Contributing

This UI is part of the Climb Analyzer project. All development should:

1. Stay within `ca9/ui/` directory
2. Not modify core Python analysis logic
3. Use TypeScript for type safety
4. Follow existing component patterns
5. Update this README for significant changes

## License

Part of the Climb Analyzer project.
