#!/bin/bash
# Start the Next.js dev server using npx (no node_modules needed)
cd "$(dirname "$0")"
npx next@15.5.6 dev --turbopack
