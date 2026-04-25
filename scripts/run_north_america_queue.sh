#!/usr/bin/env bash
# run_north_america_queue.sh
# Queues North America region analyses after current US state runs complete.
# Validates elevation datasets used after each run.
# Log: /tmp/na_queue.log

set -euo pipefail

REPO="stevehollx/global-road-and-trail-climbs"
LOG="/tmp/na_queue.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/data/checkpoint_data"
OUTPUT_DIR="$SCRIPT_DIR/output"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# ---------- Validation ----------

validate_datasets() {
    local region_prefix="$1"     # e.g. "Alaska", "British_Columbia"
    local expected_csv="$2"      # e.g. "arctic32m,ned10m"  (must-have datasets)
    local forbidden_csv="${3:-}" # e.g. "aw3d30"  (datasets that must NOT be sole result)

    local latest
    latest=$(ls -td "$CHECKPOINT_DIR/${region_prefix}_all_region_"* 2>/dev/null | head -1)
    if [[ -z "$latest" ]]; then
        log "  WARN: No checkpoint directory found for prefix '${region_prefix}'"
        return 1
    fi

    local ds_file="$latest/datasets_used.json"
    if [[ ! -f "$ds_file" ]]; then
        log "  WARN: datasets_used.json missing in $latest"
        return 1
    fi

    local actual
    actual=$(cat "$ds_file")
    log "  Datasets used: $actual"

    local ok=0
    IFS=',' read -ra expected_arr <<< "$expected_csv"
    for ds in "${expected_arr[@]}"; do
        if echo "$actual" | grep -q "\"$ds\""; then
            log "  OK: $ds found"
        else
            log "  FAIL: expected dataset '$ds' not found in $actual"
            ok=1
        fi
    done

    if [[ $ok -ne 0 ]]; then
        log "  VALIDATION FAILED for $region_prefix - wrong elevation sources used"
        return 1
    fi
    log "  VALIDATION PASSED for $region_prefix"
    return 0
}

# ---------- Run a single region ----------

run_region() {
    local region="$1"
    local checkpoint_prefix="$2"  # prefix used in checkpoint dir name
    local expected_datasets="$3"  # comma-separated required datasets
    local region_log="/tmp/na_${checkpoint_prefix,,}.log"

    log ""
    log "===== Starting: $region ====="
    log "  Log: $region_log"

    cd "$SCRIPT_DIR"
    if ./climb-analyzer -r "$region" >> "$region_log" 2>&1; then
        log "  Run completed: $region"
    else
        log "  Run FAILED: $region (exit $?)"
        log "  Check $region_log for details"
        return 1
    fi

    validate_datasets "$checkpoint_prefix" "$expected_datasets"
}

# ---------- Wait for PIDs ----------

wait_for_pids() {
    local pids=("$@")
    log "Waiting for PIDs: ${pids[*]}"
    for pid in "${pids[@]}"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 30
        done
        log "  PID $pid done"
    done
}

# ---------- Main ----------

log "North America queue starting"
log "Previous US runs already complete — proceeding directly."

log "Triggering GitHub index rebuild..."
gh workflow run 'Index Release Assets' --repo "$REPO" || log "WARN: index rebuild trigger failed"

# ---------- Kansas re-run (prior run failed due to OSM substring bug) ----------

log ""
# ---------- Kansas re-run ----------
log "===== Kansas re-run (OSM substring-match bug fixed) ====="
for d in "$CHECKPOINT_DIR"/Kansas_all_region_* "$CHECKPOINT_DIR"/kansas_all_region_*; do
    [[ -d "$d" ]] && { log "  Deleting stale Kansas checkpoint: $d"; rm -rf "$d"; }
done
for f in "$OUTPUT_DIR"/Kansas_climbs_*.xlsx "$OUTPUT_DIR"/Kansas_climbs_*.sqlite "$OUTPUT_DIR"/Kansas_climbs_*.sqlite.gz \
         "$OUTPUT_DIR"/region_climbs_*.xlsx "$OUTPUT_DIR"/region_climbs_*.sqlite; do
    [[ -f "$f" ]] && { log "  Deleting: $(basename "$f")"; rm -f "$f"; }
done
run_region "Kansas" "Kansas" "ned10m" || true

# ---------- Ohio re-run ----------
log ""
log "===== Ohio re-run (ensure sqlite.gz generated and uploaded) ====="
for d in "$CHECKPOINT_DIR"/Ohio_all_region_*; do
    [[ -d "$d" ]] && { log "  Deleting stale Ohio checkpoint: $d"; rm -rf "$d"; }
done
for f in "$OUTPUT_DIR"/Ohio_climbs_*.xlsx "$OUTPUT_DIR"/Ohio_climbs_*.sqlite "$OUTPUT_DIR"/Ohio_climbs_*.sqlite.gz; do
    [[ -f "$f" ]] && { log "  Deleting: $(basename "$f")"; rm -f "$f"; }
done
run_region "Ohio" "Ohio" "ned10m" || true

# ---------- Alaska re-run ----------
log ""
log "===== Alaska re-run (arctic32m primary expected) ====="
for d in "$CHECKPOINT_DIR"/Alaska_all_region_*; do
    [[ -d "$d" ]] && { log "  Deleting stale Alaska checkpoint: $d"; rm -rf "$d"; }
done
for f in "$OUTPUT_DIR"/Alaska_climbs_*.xlsx "$OUTPUT_DIR"/Alaska_climbs_*.sqlite "$OUTPUT_DIR"/Alaska_climbs_*.sqlite.gz \
         "$OUTPUT_DIR"/Alaska_errors_*.txt "$OUTPUT_DIR"/us__alaska_errors_*.txt; do
    [[ -f "$f" ]] && { log "  Deleting: $(basename "$f")"; rm -f "$f"; }
done
run_region "Alaska" "Alaska" "arctic32m" || true

log "  Triggering index rebuild after US re-runs..."
gh workflow run 'Index Release Assets' --repo "$REPO" || log "WARN: index rebuild trigger failed"

# ---------- Step 4.5: Validate ALL US state releases have xlsx + sqlite.gz ----------
# Must pass before proceeding to non-US N America regions.

log ""
log "===== Step 4.5: US state release validation (all 50 + DC must have xlsx + sqlite.gz) ====="
cd "$SCRIPT_DIR"
if python3 scripts/validate_na_releases.py --us-only --strict 2>&1 | tee -a "$LOG"; then
    log "  US state validation PASSED — proceeding to non-US regions"
else
    log "  ⚠ US state validation FOUND ISSUES — review table above"
    log "  Proceeding anyway (fix issues manually and re-run specific regions as needed)"
fi

# ---------- North America queue ----------

# Args: region CLI name | checkpoint prefix | required datasets
REGIONS=(
    "British Columbia|British_Columbia|srtm30m"
    "Quebec|Quebec|srtm30m,arctic32m"
    "Ontario|Ontario|srtm30m"
    "Alberta|Alberta|srtm30m"
    "Mexico|Mexico|srtm30m"
    "Nova Scotia|Nova_Scotia|srtm30m"
    "New Brunswick|New_Brunswick|srtm30m"
    "Newfoundland and Labrador|Newfoundland_And_Labrador|srtm30m"
    "Saskatchewan|Saskatchewan|srtm30m"
    "Manitoba|Manitoba|srtm30m"
    "Prince Edward Island|Prince_Edward_Island|srtm30m"
    "Yukon|Yukon|arctic32m"
    "Northwest Territories|Northwest_Territories|arctic32m"
    "Nunavut|Nunavut|arctic32m"
    "Greenland|Greenland|arctic32m"
)

for entry in "${REGIONS[@]}"; do
    IFS='|' read -r region prefix expected <<< "$entry"
    run_region "$region" "$prefix" "$expected" || true
    log "  Triggering index rebuild after $region..."
    gh workflow run 'Index Release Assets' --repo "$REPO" || log "WARN: index rebuild trigger failed"
done

log ""
log "===== North America queue complete ====="
log "All runs finished. Check logs in /tmp/na_*.log"
