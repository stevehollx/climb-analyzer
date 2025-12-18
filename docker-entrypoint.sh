#!/bin/bash
# Climb Analyzer - Docker Entrypoint Script
# Handles startup logic for the climb-analyzer container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print the ASCII logo first
export PYTHONPATH=/app
python3 -c "from climb_analyzer.utils.ascii_logo import print_logo; print_logo()" 2>/dev/null || {
    # Fallback if logo can't be printed
    echo "=========================================="
    echo "  Climb Analyzer - Container Starting"
    echo "=========================================="
}
echo ""

# Function to check if opentopodata is ready
check_opentopodata() {
    echo -n "Checking OpenTopoData server... "

    if python3 -c "import requests; requests.get('${TOPO_API_BASE_URL:-http://opentopodata:5000}/health', timeout=2)" 2>/dev/null; then
        echo -e "${GREEN}✓ Ready${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Not ready${NC}"
        return 1
    fi
}

# OpenTopoData check is now deferred until runtime
# The check happens AFTER region selection when elevation data is confirmed
# This allows the server to be configured with the correct datasets first
if [ "${SKIP_OPENTOPODATA_CHECK}" != "1" ]; then
    # Check deployment mode from config if available
    DEPLOYMENT_TYPE="local"
    if [ -f "/app/config.yaml" ]; then
        DEPLOYMENT_TYPE=$(python3 -c "import yaml; print(yaml.safe_load(open('/app/config.yaml')).get('DEPLOYMENT_TYPE', 'local'))" 2>/dev/null || echo "local")
    fi

    # Quiet startup - mode details shown later by application
fi

# Check for config.yaml
if [ ! -f "/app/config.yaml" ]; then
    echo -e "${YELLOW}⚠ No config.yaml found${NC}"
    echo "A default config.yaml will be created on first run"
fi

# Ensure output, credentials, and data directories exist and are writable
# This is important when volumes are mounted from the host
for dir in /app/output /app/.credentials /app/data /app/data/checkpoint_data /app/data/osm_indexes /app/data/planet_osm_data /app/data/elevation_data; do
    if [ -d "$dir" ]; then
        # Test if directory is writable
        if [ ! -w "$dir" ]; then
            # When running as non-root user, we can't chmod
            # Just warn and continue - the application will handle it
            echo -e "${YELLOW}⚠ $dir is not writable${NC}"
            echo -e "${YELLOW}  If you encounter permission errors, check your HOST_UID/HOST_GID settings${NC}"
        fi
    else
        # Create directory if it doesn't exist
        mkdir -p "$dir" 2>/dev/null || {
            echo -e "${YELLOW}⚠ Could not create $dir${NC}"
        }
    fi
done

# Verify Python dependencies
echo ""
echo "Verifying Python dependencies..."

if python3 -c "import pandas, numpy, geopy, requests, tqdm, yaml" 2>/dev/null; then
    echo -e "${GREEN}✓ Core dependencies OK${NC}"
else
    echo -e "${RED}✗ Core dependencies missing${NC}"
    exit 1
fi

# Check deployment-specific dependencies based on config
if [ -f "/app/config.yaml" ]; then
    DEPLOYMENT_TYPE=$(python3 -c "import yaml; print(yaml.safe_load(open('/app/config.yaml')).get('DEPLOYMENT_TYPE', 'cloud'))" 2>/dev/null || echo "cloud")

    if [ "$DEPLOYMENT_TYPE" = "local" ]; then
        if python3 -c "import osmium, rtree" 2>/dev/null; then
            echo -e "${GREEN}✓ Local mode dependencies OK${NC}"
        else
            echo -e "${RED}✗ Local mode dependencies missing${NC}"
            exit 1
        fi
    elif [ "$DEPLOYMENT_TYPE" = "cloud" ]; then
        if python3 -c "import overpy" 2>/dev/null; then
            echo -e "${GREEN}✓ Cloud mode dependencies OK${NC}"
        else
            echo -e "${YELLOW}⚠ Cloud mode dependency (overpy) missing${NC}"
            echo "Attempting to install..."
            pip install --no-cache-dir overpy
        fi
    fi
fi

# Execute the main command
exec "$@"
