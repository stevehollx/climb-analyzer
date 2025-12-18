#!/bin/bash
# Test script for checkpoint functionality in state/country mode

echo "========================================"
echo "  Checkpoint Testing Script"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Check if checkpoint files exist from previous run
echo "Test 1: Checking for existing checkpoints..."
CHECKPOINT_DIR="data/checkpoint_data"
if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_FILES=$(find "$CHECKPOINT_DIR" -name "elevation_progress.pkl" 2>/dev/null)
    if [ -n "$CHECKPOINT_FILES" ]; then
        echo -e "${YELLOW}Found existing checkpoint files:${NC}"
        for file in $CHECKPOINT_FILES; do
            FILE_SIZE=$(du -h "$file" | cut -f1)
            echo "  - $file (Size: $FILE_SIZE)"
        done
        echo ""
        read -p "Do you want to clear these checkpoints and start fresh? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f $CHECKPOINT_FILES
            echo -e "${GREEN}✓ Checkpoints cleared${NC}"
        else
            echo -e "${YELLOW}⚠️  Will resume from existing checkpoint${NC}"
        fi
    else
        echo -e "${GREEN}✓ No existing checkpoints found${NC}"
    fi
else
    echo -e "${GREEN}✓ No checkpoint directory exists${NC}"
fi
echo ""

# Test 2: Check config.yaml checkpoint settings
echo "Test 2: Checking checkpoint configuration..."
if [ -f "config.yaml" ]; then
    echo "Checkpoint settings in config.yaml:"
    grep -A 5 "CHECKPOINT_INTERVAL_MIN" config.yaml || echo "  (Using defaults)"
    echo ""
else
    echo -e "${RED}✗ config.yaml not found!${NC}"
    echo ""
fi

# Test 3: Verify GracefulKiller has region_elevation_fetching support
echo "Test 3: Verifying signal handler support..."
if grep -q "region_elevation_fetching" climb_analyzer/utils/graceful_killer.py; then
    echo -e "${GREEN}✓ Signal handler supports 'region_elevation_fetching' operation${NC}"
else
    echo -e "${RED}✗ Signal handler missing 'region_elevation_fetching' support!${NC}"
fi
echo ""

# Test 4: Check if SmartCheckpointer is imported in climb_analyzer.py
echo "Test 4: Verifying checkpoint integration..."
if grep -q "from climb_analyzer.processing.checkpoint import SmartCheckpointer" climb_analyzer.py; then
    echo -e "${GREEN}✓ SmartCheckpointer imported in climb_analyzer.py${NC}"
else
    echo -e "${RED}✗ SmartCheckpointer not imported!${NC}"
fi

if grep -q "checkpointer = SmartCheckpointer.*Region Elevation Fetching" climb_analyzer.py; then
    echo -e "${GREEN}✓ Checkpointer initialized for region elevation fetching${NC}"
else
    echo -e "${RED}✗ Checkpointer not initialized!${NC}"
fi

if grep -q "elevation_checkpoint = persistence.load_elevation_progress()" climb_analyzer.py; then
    echo -e "${GREEN}✓ Checkpoint resume logic present${NC}"
else
    echo -e "${RED}✗ Checkpoint resume logic missing!${NC}"
fi
echo ""

# Test 5: Instructions for manual testing
echo "========================================"
echo "  Manual Testing Instructions"
echo "========================================"
echo ""
echo "To test checkpoint functionality:"
echo ""
echo "1. Start a state/country analysis:"
echo "   ${YELLOW}python3 climb_analyzer.py --state Georgia --surface all${NC}"
echo ""
echo "2. Wait for elevation fetching to start (you'll see 'Processing' progress bar)"
echo ""
echo "3. Let it run for 1-2 minutes to process several batches"
echo "   Watch for '💾 Checkpoint saved' messages (based on your config)"
echo ""
echo "4. Press Ctrl+C to interrupt"
echo ""
echo "5. Verify you see:"
echo "   - ${GREEN}'Graceful shutdown during: region_elevation_fetching'${NC} (not 'unknown')"
echo "   - ${GREEN}'✓ Checkpoint saved at batch X/Y'${NC}"
echo "   - ${GREEN}'Elevations saved: N'${NC}"
echo ""
echo "6. Restart the same analysis:"
echo "   ${YELLOW}python3 climb_analyzer.py --state Georgia --surface all${NC}"
echo ""
echo "7. Verify you see:"
echo "   - ${GREEN}'Found existing elevation checkpoint - resuming...'${NC}"
echo "   - ${GREEN}'Resuming from batch X/Y'${NC}"
echo "   - Progress bar continues from where it left off"
echo ""
echo "8. Let it complete (or Ctrl+C again to test multiple resume cycles)"
echo ""
echo "9. On completion, verify checkpoint is cleaned up:"
echo "   - ${GREEN}'✓ Elevation checkpoint cleared (processing complete)'${NC}"
echo ""

# Test 6: Show current checkpoint status
echo "========================================"
echo "  Current Checkpoint Status"
echo "========================================"
echo ""
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint directories:"
    find "$CHECKPOINT_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
        echo "  - $(basename "$dir")"
        if [ -f "$dir/elevation_progress.pkl" ]; then
            SIZE=$(du -h "$dir/elevation_progress.pkl" | cut -f1)
            MODIFIED=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$dir/elevation_progress.pkl" 2>/dev/null || stat -c "%y" "$dir/elevation_progress.pkl" 2>/dev/null | cut -d'.' -f1)
            echo "    Checkpoint: Yes (${SIZE}, modified: ${MODIFIED})"
        else
            echo "    Checkpoint: No"
        fi
    done
else
    echo "No checkpoint directory found (normal for first run)"
fi
echo ""

echo "========================================"
echo "  Summary"
echo "========================================"
echo ""
echo "Checkpointing has been added to state/country mode."
echo "See CHECKPOINTING_ADDED.md for detailed documentation."
echo ""
