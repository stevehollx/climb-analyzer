# Checkpointing

Climb Analyzer automatically saves progress during long analyses, allowing you to resume if interrupted.

## Overview

Checkpointing protects against:

- Power failures
- System crashes
- Manual interruption (Ctrl+C)
- Out of memory errors
- Network timeouts

## How It Works

### Automatic Checkpoints

Progress is saved:

1. **Time-based**: Every 15 minutes (configurable)
2. **Milestone-based**: At 10%, 25%, 50%, 75%, 90% completion (hardcoded)
3. **On interrupt**: When you press Ctrl+C

### What Gets Saved

```
data/checkpoint_data/{region_name}/
├── elevation_progress.pkl    # Elevation fetching progress
├── analysis_state.pkl        # Analysis state
└── climb_results.pkl         # Partial results
```

Checkpoint data includes:
- Batch index (which batch to resume from)
- All elevations fetched so far
- Coordinates already processed
- Success/failure counts

## Resuming Analysis

When you restart an interrupted analysis:

```bash
./climb-analyzer -r "California"
```

The system detects existing checkpoints:

```
Found existing elevation checkpoint - resuming from previous progress...
  Resuming from batch 19/58
  Already have 234,567 elevations from 234,123 coords

Processing: 32%|████████░░░░░░░░░░░░| (elev_err: 1.2%)
```

## Graceful Shutdown

Pressing `Ctrl+C` triggers graceful shutdown:

```
^C
Graceful shutdown during: region_elevation_fetching

🛑 Graceful shutdown requested - saving checkpoint...
✓ Checkpoint saved at batch 18/58
  Elevations saved: 234,567

⏸️  Analysis paused. Run again to resume from this checkpoint.
```

!!! tip "Wait for Save"
    After pressing Ctrl+C, wait for the "Checkpoint saved" message before closing the terminal.

## Configuration

In `config.yaml`:

```yaml
# Save checkpoint every 15 minutes
CHECKPOINT_INTERVAL_MIN: 15.0

# Save at these progress percentages
CHECKPOINT_MILESTONES_PERC:
  - 10
  - 25
  - 50
  - 75
  - 90
```

### Adjusting Frequency

For unstable systems, increase frequency:

```yaml
CHECKPOINT_INTERVAL_MIN: 5.0  # Every 5 minutes
CHECKPOINT_MILESTONES_PERC:
  - 5
  - 10
  - 20
  - 30
  - 40
  - 50
  - 60
  - 70
  - 80
  - 90
  - 95
```

For stable systems with fast storage:

```yaml
CHECKPOINT_INTERVAL_MIN: 30.0  # Every 30 minutes
CHECKPOINT_MILESTONES_PERC:
  - 25
  - 50
  - 75
```

## Managing Checkpoints

### View Checkpoints

```bash
ls data/checkpoint_data/
# Shows: California/  Vermont/  Colorado/
```

### Clear Specific Checkpoint

```bash
rm -rf data/checkpoint_data/California/
```

### Clear All Checkpoints

```bash
./climb-analyzer -C
```

Or manually:

```bash
rm -rf data/checkpoint_data/
```

### Force Fresh Start

To restart an analysis from scratch, clear its checkpoint first:

```bash
rm -rf data/checkpoint_data/Vermont/
./climb-analyzer -r "Vermont"
```

## Checkpoint Status

During analysis, checkpoint saves are shown:

```
Processing: 45%|█████████░░░░░░░░░░░| (elev_err: 2.3%)
💾 Checkpoint saved at batch 18/40

Processing: 50%|██████████░░░░░░░░░░| (elev_err: 2.3%)
💾 Milestone checkpoint (50%) saved
```

## Storage Requirements

Checkpoint files are typically:

| Region Size | Checkpoint Size |
|-------------|-----------------|
| Small state (VT) | 50-200 MB |
| Medium state (CO) | 200-500 MB |
| Large state (CA) | 500 MB - 2 GB |
| Small country | 100-500 MB |
| Large country | 1-5 GB |

Checkpoints are automatically cleared when analysis completes successfully.

## Troubleshooting

### "Checkpoint data appears corrupted"

```bash
# Clear and restart
rm -rf data/checkpoint_data/{region}/
./climb-analyzer -r "{region}"
```

### "Cannot resume - checkpoint version mismatch"

Occurs when code is updated between runs:

```bash
# Clear old checkpoint
rm -rf data/checkpoint_data/{region}/
./climb-analyzer -r "{region}"
```

### Checkpoint not saving

Check disk space:

```bash
df -h
# Ensure sufficient space in data/ directory
```

### Resume starts from wrong batch

The checkpoint may be from a different analysis configuration. Clear and restart:

```bash
rm -rf data/checkpoint_data/{region}/
./climb-analyzer -r "{region}" -s paved  # With same options
```

## Deleting checkpoints

Use argument `-C, --delete-checkpoints` to delete all checkpoint data

If you want to delete checkpoints in batching to prevent disk growth for latge analysis:
` -X, --delete-data-on-complete`


## Best Practices

1. **Don't force-quit**: Use Ctrl+C and wait for checkpoint save
2. **Monitor progress**: Watch for checkpoint messages
3. **Check disk space**: Ensure room for checkpoint files
4. **Keep checkpoints during long analyses**: Don't clear until complete
5. **Use same options**: Resume with identical CLI options
6. **Delete checkpoints occasionally**: They will take up space.
---

Next: [Cloud Cache](cloud-cache.md)
