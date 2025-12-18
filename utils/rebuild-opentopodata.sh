#!/bin/bash
# Rebuild and restart OpenTopoData server with updated configuration

set -e

# ANSI color codes
CYAN='\033[36m'
BOLD='\033[1m'
NC='\033[0m'

echo
echo -e "${CYAN}┌──────────────────────────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│${NC}${BOLD} REBUILDING OPENTOPODATA SERVER                                               ${NC}${CYAN}│${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
echo

# Detect if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ "$(uname -m)" == "arm64" ]]; then
        BUILD_TARGET="build-m1"
        echo "Detected: macOS Apple Silicon"
    else
        BUILD_TARGET="build"
        echo "Detected: macOS Intel"
    fi
else
    # Linux
    BUILD_TARGET="build"
    echo "Detected: Linux"
fi

# Determine the base directory (project root)
# This works whether script is run from host or inside Docker container
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If we're in utils/, go up one level to project root
if [[ "$SCRIPT_DIR" == */utils ]]; then
    BASE_DIR="$(dirname "$SCRIPT_DIR")"
else
    BASE_DIR="$SCRIPT_DIR"
fi

echo "Base directory: $BASE_DIR"

# Check if opentopodata directory exists, clone if not
if [ ! -d "$BASE_DIR/opentopodata" ]; then
    echo
    echo "⚠️  OpenTopoData directory not found at $BASE_DIR/opentopodata"
    echo "Cloning OpenTopoData repository..."
    cd "$BASE_DIR"
    if git clone https://github.com/ajnisbet/opentopodata.git; then
        echo "  ✓ OpenTopoData cloned successfully"
    else
        echo "  ❌ Error: Failed to clone OpenTopoData repository"
        exit 1
    fi
fi

# Auto-generate config based on existing datasets (before building)
echo
echo "Auto-generating OpenTopoData config from existing datasets..."
cd "$BASE_DIR"
python3 utils/generate_opentopodata_config.py
if [ $? -eq 0 ]; then
    echo "  ✓ Config generated successfully"

    # Copy config into opentopodata directory for Docker build
    if [ -f "$BASE_DIR/opentopodata-config.yaml" ]; then
        cp "$BASE_DIR/opentopodata-config.yaml" "$BASE_DIR/opentopodata/config.yaml"
        echo "  ✓ Config copied to opentopodata/config.yaml for build"
    fi
else
    echo "  ⚠️  Warning: Config generation failed, will use default config"
fi

# Stop existing container
echo
echo "Stopping existing OpenTopoData container..."
docker stop opentopodata-server 2>/dev/null || echo "  (no running container found)"
docker rm opentopodata-server 2>/dev/null || echo "  (no container to remove)"

# Change to opentopodata directory
cd "$BASE_DIR/opentopodata"

# Prune old opentopodata images to free disk space
echo
echo "Pruning old opentopodata images..."
# Remove dangling images first
docker image prune -f >/dev/null 2>&1 || true
# Remove old opentopodata images (keeps current one until new build completes)
OLD_IMAGES=$(docker images --filter "reference=opentopodata*" --filter "dangling=false" -q 2>/dev/null | tail -n +2)
if [ -n "$OLD_IMAGES" ]; then
    echo "$OLD_IMAGES" | xargs -r docker rmi -f 2>/dev/null || true
    echo "  ✓ Old images pruned"
else
    echo "  ✓ No old images to prune"
fi

# Build
echo
echo "Building OpenTopoData image..."
make $BUILD_TARGET

# Read version
VERSION=$(cat VERSION)

# Ensure network exists
echo
echo "Ensuring Docker network exists..."
docker network create climb-network 2>/dev/null || echo "  (network already exists)"

# Detect if running inside Docker container
if [ -f /.dockerenv ]; then
    # Inside container - use HOST_PROJECT_DIR environment variable
    if [ -n "$HOST_PROJECT_DIR" ]; then
        HOST_BASE="$HOST_PROJECT_DIR"
        echo "  Detected: Running inside Docker container"
        echo "  Using host path: $HOST_BASE"
    else
        echo "  ❌ Error: Running inside Docker but HOST_PROJECT_DIR not set"
        echo "  This environment variable should be set by docker-compose.yml"
        exit 1
    fi
else
    # On host - use current base directory
    HOST_BASE="$BASE_DIR"
    echo "  Detected: Running on host"
fi

# Start daemon with network configuration
echo
echo "Starting OpenTopoData daemon..."

# Build docker run command
# Config is baked into the image, only mount elevation data
CMD="docker run --rm -itd \
    --name opentopodata-server \
    --network climb-network \
    --volume \"$HOST_BASE/data/elevation_data:/app/data:ro\" \
    -p 5000:5000 \
    opentopodata:$VERSION"

# Execute the command
eval $CMD

echo
echo "✓ OpenTopoData server rebuilt and started!"
echo "✓ Server available at: http://localhost:5000"
echo

# Verify the mount
echo "Verifying mounts..."
docker inspect opentopodata-server --format='{{range .Mounts}}Source: {{.Source}} → Dest: {{.Destination}}{{println}}{{end}}'

echo
echo "Check status with: docker ps | grep opentopodata"
echo

# Verification tests
echo -e "${CYAN}┌──────────────────────────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│${NC}${BOLD} VERIFYING OPENTOPODATA SERVER                                                ${NC}${CYAN}│${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
echo

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..10}; do
    if curl -s http://localhost:5000/health >/dev/null 2>&1; then
        echo "  ✓ Server is responding"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "  ⚠️  Warning: Server not responding after 10 seconds"
        echo "  The server may still be starting up. Check logs with: docker logs opentopodata-server"
        exit 0
    fi
    sleep 1
done

# Test health endpoint
echo
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:5000/health)
if echo "$HEALTH_RESPONSE" | grep -q "OK"; then
    echo "  ✓ Health check passed: $HEALTH_RESPONSE"
else
    echo "  ❌ Health check failed: $HEALTH_RESPONSE"
    exit 1
fi

# Get available datasets from config
echo
echo "Available datasets from config:"
if [ -f "$BASE_DIR/opentopodata-config.yaml" ]; then
    # Extract dataset names from YAML (simple grep approach)
    DATASETS=$(grep "^- name:" "$BASE_DIR/opentopodata-config.yaml" | sed 's/^- name: //')
    echo "$DATASETS" | while read -r dataset; do
        if [ -n "$dataset" ]; then
            echo "  • $dataset"
        fi
    done

    # Store dataset names in array for testing
    DATASET_ARRAY=($DATASETS)
else
    echo "  ⚠️  Config file not found at $BASE_DIR/opentopodata-config.yaml"
fi

# Test individual datasets
if [ ${#DATASET_ARRAY[@]} -gt 0 ]; then
    echo
    echo "Testing individual datasets (Paris, France: 48.8566,2.3522)..."
    for dataset in "${DATASET_ARRAY[@]}"; do
        if [ -n "$dataset" ]; then
            RESPONSE=$(curl -s "http://localhost:5000/v1/$dataset?locations=48.8566,2.3522")
            # Check for OK status (allow whitespace in JSON)
            if echo "$RESPONSE" | grep -q '"status".*"OK"'; then
                # Extract elevation value (handle both integer and float)
                ELEVATION=$(echo "$RESPONSE" | sed -n 's/.*"elevation"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)
                if [ -n "$ELEVATION" ] && [ "$ELEVATION" != "null" ]; then
                    echo "  ✓ $dataset: ${ELEVATION}m"
                else
                    echo "  • $dataset: null (no data at this location)"
                fi
            else
                echo "  ❌ $dataset: Error - $RESPONSE"
            fi
        fi
    done
fi

# Test multi-dataset query (small batch)
if [ ${#DATASET_ARRAY[@]} -gt 1 ]; then
    echo
    echo "Testing multi-dataset query (1 coordinate)..."
    # Build comma-separated dataset list
    MULTI_DATASET=$(IFS=,; echo "${DATASET_ARRAY[*]}")
    echo "  Query: /v1/$MULTI_DATASET"

    RESPONSE=$(curl -s "http://localhost:5000/v1/$MULTI_DATASET?locations=48.8566,2.3522")
    # Check for OK status (allow whitespace in JSON)
    if echo "$RESPONSE" | grep -q '"status".*"OK"'; then
        # Extract dataset and elevation using sed (portable)
        DATASET_USED=$(echo "$RESPONSE" | sed -n 's/.*"dataset"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
        ELEVATION=$(echo "$RESPONSE" | sed -n 's/.*"elevation"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)
        if echo "$RESPONSE" | grep -iq "duplicate"; then
            echo "  ❌ DUPLICATE DATASET ERROR detected!"
            echo "  Response: $RESPONSE"
        elif [ -n "$DATASET_USED" ]; then
            echo "  ✓ Multi-dataset query succeeded"
            echo "    Dataset used: $DATASET_USED"
            if [ -n "$ELEVATION" ]; then
                echo "    Elevation: ${ELEVATION}m"
            else
                echo "    Elevation: null (no data at this location)"
            fi
        else
            echo "  • Multi-dataset query returned: $RESPONSE"
        fi
    else
        echo "  ❌ Multi-dataset query failed!"
        echo "  Response: $RESPONSE"
        if echo "$RESPONSE" | grep -iq "duplicate"; then
            echo
            echo "  WARNING: Duplicate datasets detected in multi-dataset query!"
            echo "  Dataset list: $MULTI_DATASET"
            echo "  Check for duplicates in dataset priority configuration."
        fi
    fi

    # Test multi-dataset query (larger batch to verify server handles it)
    echo
    echo "Testing multi-dataset query (5 coordinates)..."
    COORDS="48.8566,2.3522|51.5074,-0.1278|40.7128,-74.0060|35.6762,139.6503|37.7749,-122.4194"
    RESPONSE=$(curl -s "http://localhost:5000/v1/$MULTI_DATASET?locations=$COORDS")
    # Check for OK status (allow whitespace in JSON)
    if echo "$RESPONSE" | grep -q '"status".*"OK"'; then
        RESULT_COUNT=$(echo "$RESPONSE" | grep -o '"elevation"' | wc -l)
        echo "  ✓ Multi-dataset batch query succeeded ($RESULT_COUNT results)"
    else
        echo "  ❌ Multi-dataset batch query failed: $RESPONSE"
    fi
fi

echo
echo -e "${CYAN}┌──────────────────────────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│${NC}${BOLD} VERIFICATION COMPLETE                                                        ${NC}${CYAN}│${NC}"
echo -e "${CYAN}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
echo
