#!/usr/bin/env bash
# run_final_queue.sh
# Final batch of runs to complete the N America dataset:
#   1. New runs: DC, Puerto Rico, US Virgin Islands
#   2. Kansas (fix release — currently missing sqlite.gz in published version)
#   3. Alaska re-run (fell back to aw3d30 before arctic VRT worked)
#   4. BC/Quebec/Ontario re-run (failed earlier from opentopodata config bug)
#   5. Yukon/NWT re-run (arctic32m now works; they used aw3d30 only)
#   6. Final validation
#
# Preconditions (all handled in earlier fixes, just noted):
#   - .env HOST_UID=1000 (not 501)
#   - utils/file_splitter.py readable by container
#   - dem_downloaders.py / opentopodata_manager.py use "data/arctic32m-vrt/"
#     (directory, not .vrt file)
#   - scripts/manage_arctic_vrt.py writes relative paths correctly
#   - cloud_cache.py uploads only sqlite.gz (no raw .sqlite)
#   - cloud_cache.py _create_release_pr routes non-US paths to
#     north-america/<country>/<region>/README.md
#   - engine.py recovers scope_info from persistence.analysis_id fallback

set -uo pipefail
# Note: no -e. Individual run failures shouldn't halt the whole queue —
# each region's errors are captured in its own log file.
shopt -s nullglob

REPO="stevehollx/global-road-and-trail-climbs"
LOG="/tmp/final_queue.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/data/checkpoint_data"
OUTPUT_DIR="$SCRIPT_DIR/output"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

clean_region() {
    # Delete checkpoint(s) + output files for a region prefix so the run is fresh.
    local prefix="$1"
    # Delete matching checkpoint dirs (any suffix, any timestamp)
    for d in "$CHECKPOINT_DIR"/"${prefix}"_all_region_* \
             "$CHECKPOINT_DIR"/"${prefix}"_all_country_*; do
        [[ -d "$d" ]] && { log "  Deleting checkpoint: $(basename "$d")"; rm -rf "$d"; }
    done
    # Also match lowercase canonical form (e.g. canada_alberta_all_country_*)
    local lower="$(echo "$prefix" | tr '[:upper:]' '[:lower:]')"
    for d in "$CHECKPOINT_DIR"/"${lower}"*_all_country_*; do
        [[ -d "$d" ]] && { log "  Deleting checkpoint: $(basename "$d")"; rm -rf "$d"; }
    done
    # Delete output files
    for pat in "${prefix}_climbs_*" "${prefix}_errors_*" "${prefix}_State_Analysis_errors_*"; do
        for f in "$OUTPUT_DIR"/$pat; do
            [[ -f "$f" ]] && { log "  Deleting output: $(basename "$f")"; rm -f "$f"; }
        done
    done
}

run_region() {
    local region="$1"
    local prefix="$2"
    local region_log="/tmp/final_$(echo "$prefix" | tr '[:upper:]' '[:lower:]').log"
    log ""
    log "===== Starting: $region ====="
    log "  Log: $region_log"
    cd "$SCRIPT_DIR"
    if ./climb-analyzer -r "$region" >> "$region_log" 2>&1; then
        log "  Run completed: $region"
    else
        log "  Run exit=$? for $region (check log; upload often exits 120 even on success)"
    fi
}

log "Final queue starting"

# ---------- Step 1: US territories (new runs) ----------
log ""
log "===== Step 1: US territories ====="
for entry in "District of Columbia|DistrictOfColumbia" "Puerto Rico|PuertoRico" "US Virgin Islands|USVirginIslands"; do
    IFS='|' read -r region prefix <<< "$entry"
    clean_region "$prefix"
    run_region "$region" "$prefix"
done

# ---------- Step 2: Kansas re-upload (missing sqlite.gz on published release) ----------
log ""
log "===== Step 2: Kansas re-run (existing published release missing sqlite.gz) ====="
clean_region "Kansas"
run_region "Kansas" "Kansas"

# ---------- Step 3: Alaska re-run (arctic32m primary) ----------
log ""
log "===== Step 3: Alaska re-run (arctic32m now served) ====="
clean_region "Alaska"
run_region "Alaska" "Alaska"

# ---------- Step 4: Previously-failed Canadian provinces ----------
log ""
log "===== Step 4: BC/Quebec/Ontario re-run (opentopodata crash earlier) ====="
for entry in \
    "British Columbia|BritishColumbia" \
    "Quebec|Quebec" \
    "Ontario|Ontario"; do
    IFS='|' read -r region prefix <<< "$entry"
    clean_region "$prefix"
    # Also match "canada/british-columbia" form in checkpoint naming
    for d in "$CHECKPOINT_DIR"/canada${prefix,,}_all_country_*; do
        [[ -d "$d" ]] && { log "  Deleting checkpoint: $(basename "$d")"; rm -rf "$d"; }
    done
    run_region "$region" "$prefix"
done

# ---------- Step 5: Arctic-primary re-runs (Yukon + NWT used aw3d30 only) ----------
log ""
log "===== Step 5: Yukon + NWT arctic re-run ====="
for entry in \
    "Yukon|Yukon" \
    "Northwest Territories|NorthwestTerritories"; do
    IFS='|' read -r region prefix <<< "$entry"
    clean_region "$prefix"
    run_region "$region" "$prefix"
done

# ---------- Step 6: Final validation ----------
log ""
log "===== Step 6: N America release validation ====="
cd "$SCRIPT_DIR"
python3 scripts/validate_na_releases.py 2>&1 | tee -a "$LOG" || {
    log "  ⚠ Validation found issues - review above"
}

# ---------- Step 7: Regenerate US state README index ----------
log ""
log "===== Step 7: Regenerate US state README indexes ====="
python3 scripts/regenerate_us_state_readmes.py --apply 2>&1 | tee -a "$LOG" || true

log ""
log "===== Final queue complete ====="
log "Manual follow-ups (if desired):"
log "  - Nunavut ran with mixed arctic32m+aw3d30; re-run for pure arctic if needed"
log "  - Alaska's Canadian province region READMEs (alberta/manitoba/etc) still in releases/ — re-run to move them"
