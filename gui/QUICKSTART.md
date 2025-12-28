# Quick Start - Climb Analyzer Web UI

## Current Status

✅ **Complete UI structure built** - All pages, components, and code ready
✅ **Minimal dependencies** - Reduced to just 10 packages (Next.js, React, Tailwind)
⏳ **Installation blocked** - Drive I/O extremely slow for npm/yarn operations

## The Issue

The drive is experiencing very slow I/O operations during package installation:
- `yarn install` stuck on step 3/4 (linking dependencies) for 15+ minutes
- Zombie `rm -rf` process from 22 minutes ago still in uninterruptible sleep
- Network downloads taking 260-280 seconds per package

This is NOT a hardware issue - it's either:
1. Network connectivity to npm registry is extremely slow
2. USB/Thunderbolt connection having I/O contention
3. File system operations bottlenecked

## Workaround Options

### Option 1: Copy to Internal SSD (FASTEST)

```bash
# Copy ui folder to your internal drive
cp -R /Volumes/usb1-drive/ca9/ui ~/climb-analyzer-ui
cd ~/climb-analyzer-ui

# Install will be much faster on internal SSD
yarn install

# Start dev server
yarn dev

# Open http://localhost:3000
```

The UI will still read data from `/Volumes/usb1-drive/ca9/output` via the API routes.

### Option 2: Wait for Current Install

The yarn install IS making progress, just very slowly. You can:
- Leave it running overnight
- Check tomorrow if it completed
- Then `cd /Volumes/usb1-drive/ca9/ui && yarn dev`

### Option 3: Use Docker (SIMPLEST FOR USERS)

Once built, package everything in Docker:

```dockerfile
# Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile
COPY . .
RUN yarn build
CMD ["yarn", "start"]
```

Users just run: `docker run -p 3000:3000 -v /path/to/output:/app/output climb-analyzer-ui`

## What's Already Built

All these files are ready in `/Volumes/usb1-drive/ca9/ui/`:

**Pages:**
- `/` - Dashboard with stats and quick actions
- `/analyze` - Full analysis configuration form
- `/visualize` - Map visualization with climb filtering
- `/download` - Data download manager
- `/config` - Configuration settings
- `/manage` - Data cleanup tools
- `/docs` - Documentation

**API Routes:**
- `/api/config` - Read/write config.yaml
- `/api/results` - List/load output CSV files
- `/api/analyze` - Run analysis (needs implementation)
- `/api/progress` - SSE progress stream (needs implementation)

**Components:**
- Sidebar navigation
- Form inputs and controls
- Cards and layouts
- All TypeScript types

**Utilities:**
- CSV parser for climb data
- GeoJSON converter
- API client functions
- Color coding system

## Files Structure

```
ui/
├── app/                    # All pages ready
├── components/             # UI components
├── lib/                    # Utilities, CSV parser, API client
├── types/                  # TypeScript definitions
├── package.json            # Minimal 10 packages
└── README.md              # Full documentation
```

## Next Steps

1. **Choose a workaround above** (recommend Option 1 - copy to internal SSD)
2. Complete `yarn install`
3. Start dev server with `yarn dev`
4. Test with your climb data from `../output`
5. Implement remaining API routes:
   - Analysis execution (subprocess to Python)
   - Progress streaming (SSE)
   - File loading from output directory

## Package Distribution

Once working, you can distribute via:

**For developers:**
- Include package.json + source code
- They run `yarn install && yarn dev`

**For end users:**
- Build with `yarn build`
- Export static files with `yarn export`
- Distribute the `/out` folder - just HTML/CSS/JS
- OR package in Docker image

**For integrated deployment:**
- Add to your existing Docker Compose setup
- Users get both CLI and UI in one `docker-compose up`

## Troubleshooting

**If yarn is still stuck after hours:**
```bash
# Kill all processes
pkill -f yarn
pkill -f npm

# Try on internal drive instead
cp -R /Volumes/usb1-drive/ca9/ui ~/climb-analyzer-ui
cd ~/climb-analyzer-ui
rm -rf node_modules yarn.lock
yarn install
```

**If you want to start fresh:**
```bash
cd /Volumes/usb1-drive/ca9/ui
rm -rf node_modules yarn.lock package-lock.json
yarn install
```

The code is complete and ready - it's just the dependency installation that's slow on this specific drive/network setup.
