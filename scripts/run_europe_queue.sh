#!/usr/bin/env bash
# run_europe_queue.sh
# Europe queue ordered by cycling-climbing popularity.
#
# Structure:
#   Tier 1  - small/warm-up countries (Belgium first)
#   Tier 2  - large countries run as SUBREGIONS (Spain, Italy, France, Germany)
#   Tier 3  - UK + Ireland
#   Tier 4  - Nordics (Denmark, Norway, Sweden, Finland)
#   Tier 5  - Central + Eastern Europe
#   Tier 6  - Balkans + Baltics + small islands
#   Tier 7  - uncertain/lower priority (Iceland, Turkey, Ukraine, Belarus, Russia)
#
# Caveats:
#   - Norway/Sweden/Finland: priority dict has latitude split (>60°N -> arctic32m),
#     but download phase only pulls country-level priority (srtm30m+aw3d30). Northern
#     coords will fall back to aw3d30 unless arctic32m is manually extended to cover
#     Fennoscandia. That is acceptable (aw3d30 works, just lower resolution).
#   - Iceland: priority is ["arctic32m"] only; WILL return nulls unless arctic32m
#     VRT is extended to Iceland tiles first. Script will run it but expect failures.
#   - France: stale checkpoint deleted before scheduling.
#   - Subregion names are passed as distinctive last-path components; region_detector
#     does fuzzy matching on lowercase+dehyphenated form.

set -uo pipefail
shopt -s nullglob

REPO="stevehollx/global-road-and-trail-climbs"
LOG="/tmp/europe_queue.log"
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
    local lower="$(echo "$prefix" | tr '[:upper:]' '[:lower:]')"
    for d in "$CHECKPOINT_DIR"/"${lower}"*_all_country_* \
             "$CHECKPOINT_DIR"/europe"${lower}"_all_country_* \
             "$CHECKPOINT_DIR"/europe"${lower}"*_all_country_*; do
        [[ -d "$d" ]] && { log "  Deleting checkpoint: $(basename "$d")"; rm -rf "$d"; }
    done
    for pat in "${prefix}_climbs_*" "${prefix}_errors_*" "${prefix}_State_Analysis_errors_*"; do
        for f in "$OUTPUT_DIR"/$pat; do
            [[ -f "$f" ]] && { log "  Deleting output: $(basename "$f")"; rm -f "$f"; }
        done
    done
}

# Slugify "Pais Vasco" -> "pais-vasco", "Nord-Ovest" -> "nord-ovest".
slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' _' '--' | sed -E 's/-+/-/g; s/^-//; s/-$//'
}

# Returns 0 if a GitHub release tag exists for this region (any continent/country prefix).
release_exists() {
    local region_slug="$1"
    gh release list -R "$REPO" -L 300 2>/dev/null \
        | awk -F'\t' '{print $3}' \
        | grep -qE "(^|-)${region_slug}-v[0-9]"
}

# After a successful run, verify a release exists; if so, free disk for this region.
# Safe to call unconditionally — if no release found, keeps all artifacts for retry.
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

    # Checkpoint (current region's and any prior stale)
    for d in "$CHECKPOINT_DIR"/"${prefix}"_all_country_* \
             "$CHECKPOINT_DIR"/"${prefix}"_all_region_*; do
        [[ -d "$d" ]] && { log "    rm checkpoint $(basename "$d")"; rm -rf "$d"; }
    done

    # Per-region OSM PBF (subregion PBFs are not shared across siblings)
    local pbf="$SCRIPT_DIR/data/planet_osm_data/${region_slug}-latest.osm.pbf"
    [[ -f "$pbf" ]] && { log "    rm $(basename "$pbf")"; rm -f "$pbf"; }

    # Output artifacts — release is authoritative source
    for pat in "${prefix}_climbs_*" "${prefix}_errors_*" "${prefix}_State_Analysis_errors_*"; do
        for f in "$OUTPUT_DIR"/$pat; do
            [[ -f "$f" ]] && { log "    rm output $(basename "$f")"; rm -f "$f"; }
        done
    done
    return 0
}

run_region() {
    local region="$1"
    local prefix="$2"
    local region_log="/tmp/europe_$(echo "$prefix" | tr '[:upper:]' '[:lower:]').log"
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

# Each entry format:  "<CLI region name>|<checkpoint/output prefix>"
# Group separators use comments only.

QUEUE=(
    # ---------- Resuming after reboot (2026-04-24): first 50 entries completed
    # ---------- through Rhone-Alpes. Aquitaine was interrupted (no release),
    # ---------- restart there. See scripts/run_europe_queue.sh history for full
    # ---------- queue.
    "Aquitaine|Aquitaine"
    "Midi-Pyrenees|Midi_Pyrenees"
    "Languedoc-Roussillon|Languedoc_Roussillon"
    "Provence-Alpes-Cote-d-Azur|Provence_Alpes_Cote_D_Azur"
    "Corse|Corse"

    # ---------- Tier 2d: Germany Länder ----------
    "Schleswig-Holstein|Schleswig_Holstein"
    "Hamburg|Hamburg"
    "Bremen|Bremen"
    "Niedersachsen|Niedersachsen"
    "Mecklenburg-Vorpommern|Mecklenburg_Vorpommern"
    "Nordrhein-Westfalen|Nordrhein_Westfalen"
    "Hessen|Hessen"
    "Rheinland-Pfalz|Rheinland_Pfalz"
    "Saarland|Saarland"
    "Baden-Wuerttemberg|Baden_Wuerttemberg"
    "Bayern|Bayern"
    "Berlin|Berlin"
    "Brandenburg|Brandenburg"
    "Sachsen-Anhalt|Sachsen_Anhalt"
    "Sachsen|Sachsen"
    "Thueringen|Thueringen"

    # ---------- Tier 3: UK + Ireland ----------
    "United Kingdom|United_Kingdom"
    "Ireland and Northern Ireland|Ireland"

    # ---------- Tier 4: Nordics (arctic fallback caveat above) ----------
    "Denmark|Denmark"
    "Norway|Norway"
    "Sweden|Sweden"
    "Finland|Finland"

    # ---------- Tier 5: Central + Eastern Europe ----------
    "Poland|Poland"
    "Czech Republic|Czech_Republic"
    "Slovakia|Slovakia"
    "Hungary|Hungary"
    "Romania|Romania"
    "Bulgaria|Bulgaria"

    # ---------- Tier 6: Balkans, Baltics, islands ----------
    "Croatia|Croatia"
    "Serbia|Serbia"
    "Bosnia-Herzegovina|Bosnia_Herzegovina"
    "Montenegro|Montenegro"
    "Albania|Albania"
    "Macedonia|Macedonia"
    "Kosovo|Kosovo"
    "Greece|Greece"
    "Cyprus|Cyprus"
    "Malta|Malta"
    "Estonia|Estonia"
    "Latvia|Latvia"
    "Lithuania|Lithuania"
    "Faroe Islands|Faroe_Islands"
    "Isle of Man|Isle_Of_Man"
    "Guernsey-Jersey|Guernsey_Jersey"
    "Azores|Azores"

    # ---------- Tier 7: uncertain / lower priority ----------
    "Iceland|Iceland"           # WILL return nulls unless arctic32m VRT extended to Iceland
    "Turkey|Turkey"
    "Ukraine|Ukraine"
    "Belarus|Belarus"
    "Moldova|Moldova"
    "Russia|Russia"             # huge; expect long run + latitude fallback for north
)

log "Europe queue starting (${#QUEUE[@]} entries)"

for entry in "${QUEUE[@]}"; do
    IFS='|' read -r region prefix <<< "$entry"
    clean_region "$prefix"
    run_region "$region" "$prefix"
done

# ---------- Final validation ----------
log ""
log "===== Final validation ====="
cd "$SCRIPT_DIR"
# Reuse the NA validator if a Europe-specific one isn't ready
if [[ -f scripts/validate_na_releases.py ]]; then
    python3 scripts/validate_na_releases.py 2>&1 | tee -a "$LOG" || log "  WARN: validation noted issues"
fi

log ""
log "===== Europe queue complete ====="
log "Follow-ups:"
log "  - Iceland: build arctic32m VRT covering Iceland (65-67N, -24 to -13E) and re-run"
log "  - Nordics: consider extending arctic32m VRT to Fennoscandia for >60N coverage"
log "  - Review any failed regions in /tmp/europe_*.log"
