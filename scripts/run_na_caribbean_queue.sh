#!/usr/bin/env bash
# run_na_caribbean_queue.sh
# Post-Europe follow-up: finish the remaining North America (non-US) regions.
#
# Status as of 2026-04-24:
#   Done — Canada (all 13 provinces/territories), Greenland, Mexico, Puerto Rico
#   Remaining — Central America + Caribbean (11 PBFs)
#
# Routing: Geofabrik classifies all of these under `central-america/` so the
# releases will land at `central-america/<country>/README.md`, even though the
# GitHub repo currently has empty `north-america/<country>/` placeholder dirs.
# That is the correct behaviour (see utils/cloud_cache.py:get_country_continent).
#
# Caveats:
#   - haiti-and-domrep is a single combined Geofabrik PBF. One release will be
#     produced under central-america/haiti-and-domrep/ rather than separate
#     Haiti and Dominican Republic releases.
#   - Trinidad and Tobago is NOT in geo_definitions.py — omitted here. Add an
#     entry first if it needs to be analyzed.
#   - Small island nations typically complete in under an hour each.

set -uo pipefail
shopt -s nullglob

REPO="stevehollx/global-road-and-trail-climbs"
LOG="/tmp/na_caribbean_queue.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/data/checkpoint_data"
OUTPUT_DIR="$SCRIPT_DIR/output"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

clean_region() {
    local prefix="$1"
    for d in "$CHECKPOINT_DIR"/"${prefix}"_all_region_* \
             "$CHECKPOINT_DIR"/"${prefix}"_all_country_*; do
        [[ -d "$d" ]] && { log "  Deleting checkpoint: $(basename "$d")"; rm -rf "$d"; }
    done
    for pat in "${prefix}_climbs_*" "${prefix}_errors_*"; do
        for f in "$OUTPUT_DIR"/$pat; do
            [[ -f "$f" ]] && { log "  Deleting output: $(basename "$f")"; rm -f "$f"; }
        done
    done
}

slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' _' '--' | sed -E 's/-+/-/g; s/^-//; s/-$//'
}

release_exists() {
    local region_slug="$1"
    gh release list -R "$REPO" -L 300 2>/dev/null \
        | awk -F'\t' '{print $3}' \
        | grep -qE "(^|-)${region_slug}-v[0-9]"
}

cleanup_after_success() {
    local region="$1"
    local prefix="$2"
    local region_slug
    region_slug=$(slugify "$region")
    if ! release_exists "$region_slug"; then
        log "  Cleanup: no release tag for '$region_slug' — keeping artifacts"
        return 1
    fi
    log "  Cleanup: release '$region_slug' published; pruning disk"
    for d in "$CHECKPOINT_DIR"/"${prefix}"_all_country_* \
             "$CHECKPOINT_DIR"/"${prefix}"_all_region_*; do
        [[ -d "$d" ]] && { log "    rm checkpoint $(basename "$d")"; rm -rf "$d"; }
    done
    local pbf="$SCRIPT_DIR/data/planet_osm_data/${region_slug}-latest.osm.pbf"
    [[ -f "$pbf" ]] && { log "    rm $(basename "$pbf")"; rm -f "$pbf"; }
    for pat in "${prefix}_climbs_*" "${prefix}_errors_*"; do
        for f in "$OUTPUT_DIR"/$pat; do
            [[ -f "$f" ]] && { log "    rm output $(basename "$f")"; rm -f "$f"; }
        done
    done
    return 0
}

run_region() {
    local region="$1"
    local prefix="$2"
    local region_log="/tmp/na_$(echo "$prefix" | tr '[:upper:]' '[:lower:]').log"
    log ""
    log "===== Starting: $region ====="
    log "  Log: $region_log"
    cd "$SCRIPT_DIR"
    local rc=0
    ./climb-analyzer -r "$region" >> "$region_log" 2>&1 || rc=$?
    if [[ $rc -eq 0 || $rc -eq 120 ]]; then
        log "  Run finished rc=$rc for $region"
    else
        log "  Run FAILED rc=$rc for $region (check $region_log)"
    fi
    cleanup_after_success "$region" "$prefix" || true
}

QUEUE=(
    # ---------- Small Caribbean first (fast iterations) ----------
    "Bahamas|Bahamas"
    "Jamaica|Jamaica"
    "Cuba|Cuba"
    "Haiti and Domrep|Haiti_And_Domrep"

    # ---------- Central America mainland ----------
    "Belize|Belize"
    "El Salvador|El_Salvador"
    "Guatemala|Guatemala"
    "Honduras|Honduras"
    "Nicaragua|Nicaragua"
    "Costa Rica|Costa_Rica"
    "Panama|Panama"
)

log "NA/Caribbean queue starting (${#QUEUE[@]} entries)"

for entry in "${QUEUE[@]}"; do
    IFS='|' read -r region prefix <<< "$entry"
    clean_region "$prefix"
    run_region "$region" "$prefix"
done

log ""
log "===== NA/Caribbean queue complete ====="
log "Follow-ups:"
log "  - Trinidad & Tobago: add entry to climb_analyzer/data/geo_definitions.py first"
log "  - haiti-and-domrep yields a combined release; may want to split later"
log "  - Review any failed regions in /tmp/na_*.log"
