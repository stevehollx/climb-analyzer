#!/bin/bash
# Clear climb analysis checkpoints to force re-run of climb detection
# Keeps elevation data and merged segments intact

CHECKPOINT_DIR="data/checkpoint_data"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Clearing climb analysis checkpoints from all regions..."

# Count files before deletion
count=0

for dir in "$CHECKPOINT_DIR"/*/; do
    if [ -d "$dir" ]; then
        region=$(basename "$dir")
        deleted=0

        # Delete climb analysis progress
        if [ -f "${dir}climb_analysis_progress.pkl" ]; then
            rm "${dir}climb_analysis_progress.pkl"
            ((deleted++))
        fi

        # Delete temp climbs files
        for f in "${dir}"climbs_temp_*.pkl; do
            if [ -f "$f" ]; then
                rm "$f"
                ((deleted++))
            fi
        done

        # Delete connected climbs progress
        if [ -f "${dir}connected_climbs_progress.pkl" ]; then
            rm "${dir}connected_climbs_progress.pkl"
            ((deleted++))
        fi

        # Delete geocoding progress
        if [ -f "${dir}geocoding_progress.pkl" ]; then
            rm "${dir}geocoding_progress.pkl"
            ((deleted++))
        fi

        # Delete tmp files
        if [ -d "${dir}tmp" ]; then
            for f in "${dir}tmp/"tmp*_final_sorted.pkl "${dir}tmp/"tmp*_geocode.pkl; do
                if [ -f "$f" ]; then
                    rm "$f"
                    ((deleted++))
                fi
            done
        fi

        if [ $deleted -gt 0 ]; then
            echo "  $region: deleted $deleted files"
            ((count+=deleted))
        fi
    fi
done

echo "Done. Deleted $count checkpoint files."
echo "Elevation data and merged segments preserved."
