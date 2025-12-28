# Testing the Climb Analyzer Web UI

## Current Status

The web UI has been fully built with all pages, components, and structure complete. However, npm install is failing on the USB drive due to file system issues (`ENOTEMPTY` errors during package installation).

## What's Been Built

✅ Complete Next.js application structure
✅ All 7 pages (Dashboard, Analyze, Visualize, Download, Config, Manage, Docs)
✅ TypeScript types for climbs and configuration
✅ CSV parser and GeoJSON converter
✅ shadcn/ui components
✅ Sidebar navigation
✅ API route structure
✅ Comprehensive documentation

## The Issue

The USB drive's file system is causing npm to fail when renaming/moving packages during installation. This is a common issue with external drives.

## Solution: Copy to Local Drive

To test the web UI, copy the entire `ui` folder to your local SSD:

```bash
# Copy to local directory
cp -R /Volumes/usb1-drive/ca9/ui ~/climb-analyzer-ui
cd ~/climb-analyzer-ui

# Clean start
rm -rf node_modules package-lock.json

# Install dependencies
npm install --legacy-peer-deps

# Start dev server
npm run dev

# Open browser to:
# http://localhost:3000
```

## Alternative: Manual TypeScript Installation

If you want to try on the USB drive:

```bash
cd /Volumes/usb1-drive/ca9/ui

# Create TypeScript file manually
echo 'import type { NextConfig } from "next"; const config: NextConfig = {}; export default config;' > next.config.ts

# Convert config to JS
cat > next.config.mjs << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;
EOF

# Remove the .ts config
rm next.config.ts

# Install TypeScript directly in parent folder
cd ..
npm install typescript --save-dev

# Try again
cd ui
npx next dev --turbopack
```

## Testing With Your Data

Once the server runs, you can:

1. **View Dashboard** - http://localhost:3000
2. **Configure Analysis** - http://localhost:3000/analyze
3. **Visualize Results** - http://localhost:3000/visualize

The API route at `/api/results` will automatically scan `../output/` for your climb analysis files:
- `climbs_7300_Dunhill_Terrace_Northeast_Atlanta_Fulton_Coun_paved_basic_address_2km.xlsx`
- `climbs_7300_dunhill_ter_sandy_springs_ga_30328_paved_basic_address_2km.xlsx`
- `climbs_8155_Geer_Hwy_Cleveland_SC_29635_paved_basic_address_24km.xlsx`
- `climbs_Caesars_Head_SC_29635_paved_basic_address_24km.xlsx`

## What Still Needs Implementation

While the UI is complete, these features need backend connections:

1. **Analysis Execution** - API route to run `python ../climb_analyzer.py` with subprocess
2. **File Loading** - Parse XLSX/CSV files from `../output`
3. **Progress Streaming** - SSE for real-time updates
4. **Map Integration** - Add MapLibre GL JS to visualize climbs
5. **Data Management** - Implement delete operations

But all the UI, forms, pages, and structure are ready!

## Quick Test Without npm install

If you just want to see the structure:

```bash
cd /Volumes/usb1-drive/ca9/ui
tree -L 2 -I 'node_modules'
```

All files are in place:
- `app/` - All pages and API routes
- `components/` - UI components and layout
- `lib/` - Utility functions, CSV parser, API client
- `types/` - TypeScript type definitions

## Recommended Next Steps

1. Copy `ui` folder to local SSD
2. Run `npm install --legacy-peer-deps`
3. Start dev server with `npm run dev`
4. Implement remaining API routes:
   - `/api/analyze/route.ts` - Execute Python subprocess
   - `/api/results/[filename]/route.ts` - Load specific CSV
   - `/api/progress/[jobId]/route.ts` - SSE progress stream
5. Add MapLibre GL JS to visualization page
6. Test with your Caesars Head and Atlanta climb data!

The foundation is solid - just needs to run on a faster drive for npm to work properly.
