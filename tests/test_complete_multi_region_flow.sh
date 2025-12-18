#!/bin/bash
#
# Complete Multi-Region Flow Test
#
# Tests the entire flow for Caesars Head, SC:
# 1. Detection of 3 needed regions
# 2. Verification of existing files
# 3. Merging into single file
# 4. Analysis uses merged file
#

set -e

echo "======================================================================"
echo "  COMPLETE MULTI-REGION FLOW TEST"
echo "======================================================================"
echo ""
echo "Testing: Caesars Head, SC 29635"
echo "Expected: SC + NC + TN regions"
echo ""

# Step 1: Check current files
echo "Step 1: Checking existing OSM files..."
echo ""
ls -lh planet-osm/*.osm.pbf 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'|| echo "  No OSM files found"
echo ""

# Step 2: Test region detection
echo "Step 2: Testing region detection..."
python3 << 'EOF'
from region_mapper import find_regions_for_bbox

bbox = (34.88819422162162, -82.88911663481501, 35.32315097837838, -82.357444965185)
center = (35.1057, -82.6233)

regions = find_regions_for_bbox(bbox, prefer_smallest=True, center_point=center)

print(f"  Regions needed: {len(regions)}")
for i, (path, url, size) in enumerate(regions, 1):
    print(f"    {i}. {path.split('/')[-1]}")

if len(regions) == 3:
    print("  ✓ PASS: Detected 3 regions")
else:
    print(f"  ❌ FAIL: Expected 3 regions, got {len(regions)}")
    exit(1)
EOF

# Step 3: Check if merge is needed
echo ""
echo "Step 3: Checking merge status..."
if ls planet-osm/merged-*.osm.pbf 1> /dev/null 2>&1; then
    echo "  ✓ Merged file exists:"
    ls -lh planet-osm/merged-*.osm.pbf | awk '{print "     " $9 " (" $5 ")"}'
else
    echo "  ⚠️  No merged file found - will be created during validation"
fi

# Step 4: Test find_osm_file_for_region behavior
echo ""
echo "Step 4: Testing OSM file selection..."
python3 << 'EOF'
from pathlib import Path
from data_validator import find_osm_file_for_region

result = find_osm_file_for_region("South Carolina")

if result:
    print(f"  Selected file: {result.name}")
    if result.name.startswith("merged-"):
        print("  ✓ PASS: Using merged file")
    else:
        print("  ⚠️  Using individual file (expected if no merged file exists)")
else:
    print("  ❌ FAIL: No file found")
    exit(1)
EOF

echo ""
echo "======================================================================"
echo "  TEST SUMMARY"
echo "======================================================================"
echo ""
echo "To complete the test:"
echo "  1. Run analysis: python climb_analyzer.py"
echo "  2. Choose: Address-based search"
echo "  3. Enter: Caesars Head, SC 29635"
echo "  4. Verify it creates/uses merged file"
echo ""
echo "Expected behavior:"
echo "  • Validation detects need for merge"
echo "  • Creates: merged-caesars-head-sc-29635-latest.osm.pbf"
echo "  • Analysis uses merged file"
echo "  • Complete coverage across SC/NC/TN"
echo ""
