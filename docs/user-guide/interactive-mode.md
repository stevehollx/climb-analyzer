# Interactive Mode

The interactive mode provides a menu-driven interface for configuring and running analyses.

## Starting Interactive Mode

```bash
./climb-analyzer
# or
./climb-analyzer -i
```

## Main Menu

```
╔════════════════════════════════════════╗
║         CLIMB ANALYZER v2.1.0          ║
╠════════════════════════════════════════╣
║  1. Run Analysis                       ║
║  2. Data Management                    ║
║  3. Configuration                      ║
║  4. Help & Documentation               ║
║  5. Exit                               ║
╚════════════════════════════════════════╝
```

## Run Analysis

### Scope Selection

Choose the type of analysis:

```
Select analysis scope:
  1. Address (radius-based)
  2. US State
  3. Country
  4. Batch (multiple regions)
```

### Address Analysis

For radius-based analysis around a location:

```
Enter address: Boulder, CO
Enter radius (miles): 25
```

### Region Selection

For state or country analysis:

```
Select region:
  1. Alabama
  2. Alaska
  3. Arizona
  ...

Or type region name: Vermont
```

### Analysis Options

Configure analysis parameters:

```
Surface filter:
  1. All surfaces (default)
  2. Paved only
  3. Gravel only
  4. Dirt only

Score type:
  1. Basic (distance × grade)
  2. FIETS index
  3. PDI (PJAMM difficulty)

Units:
  1. Imperial (miles/feet)
  2. Metric (km/meters)

Cycling filter:
  1. All roads (default)
  2. Cycling accessible only
```

## Data Management

### View Data Status

Shows what data is currently downloaded:

```
Data Status:
╔═══════════════════════════════════════════════╗
║ OSM Data                                      ║
╠═══════════════════════════════════════════════╣
║ vermont-latest.osm.pbf         142 MB   ✓    ║
║ new-hampshire-latest.osm.pbf   186 MB   ✓    ║
╠═══════════════════════════════════════════════╣
║ Elevation Data                                ║
╠═══════════════════════════════════════════════╣
║ NED 10m                        2.1 GB   ✓    ║
║ SRTM 30m                       1.8 GB   ✓    ║
╠═══════════════════════════════════════════════╣
║ Total                          4.2 GB        ║
╚═══════════════════════════════════════════════╝
```

### Download Data

Download OSM or elevation data for a region:

```
Download data for:
  1. US State
  2. Country
  3. Specific elevation dataset
```

### Clean Up Data

Remove cached data:

```
Clean up options:
  1. Delete checkpoints only
  2. Delete OSM data
  3. Delete elevation data
  4. Delete all data
```

## Configuration

### View Current Config

Displays current `config.yaml` settings:

```
Current Configuration:
  Deployment: local
  Elevation API: http://opentopodata-server:5000/v1
  Batch Size: 100
  Max Concurrent: 16
  Cloud Cache: enabled
```

### Edit Configuration

Modify settings interactively:

```
Edit setting:
  1. Deployment type (local/cloud)
  2. Elevation batch size
  3. Max concurrent requests
  4. Checkpoint interval
  5. Cloud cache enabled
```

### NASA Earthdata Credentials

Configure credentials for elevation downloads:

```
Enter NASA Earthdata username: your_username
Enter NASA Earthdata password: ********
✓ Credentials saved
```

## Progress Display

During analysis, progress is shown:

```
Analyzing Vermont...

Step 1/6: Loading OSM data
  [████████████████████] 100% Complete

Step 2/6: Building spatial index
  [████████████████████] 100% Complete

Step 3/6: Extracting road segments
  [████████████░░░░░░░░] 62% 124,532/200,000 segments

Step 4/6: Fetching elevations
  [██████░░░░░░░░░░░░░░] 31% (elev_err: 1.2%)
  💾 Checkpoint saved at batch 18/58

Step 5/6: Analyzing climbs
  [████████████████████] 100% Found 3,421 climbs

Step 6/6: Writing output
  ✓ Vermont_climbs_all_basic_2025-12-17.xlsx
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Select option |
| `q` | Quit / Back |
| `Ctrl+C` | Graceful shutdown (saves checkpoint) |
| `↑/↓` | Navigate menu |
| Numbers | Quick select option |

## Graceful Shutdown

Pressing `Ctrl+C` during analysis:

```
^C
Graceful shutdown during: region_elevation_fetching

🛑 Graceful shutdown requested - saving checkpoint...
✓ Checkpoint saved at batch 18/58
  Elevations saved: 234,567

⏸️  Analysis paused. Run again to resume from this checkpoint.
```

## Cloud Cache Prompts

When cached data is available:

```
✓ Vermont analysis found in cloud cache (2025-10-25)
  Download cached analysis? (y/n): y

Downloading...
✓ Downloaded 1 file, 8.2 MB
```

After completing a "clean" analysis:

```
✓ Analysis complete! Found 3,421 climbs

This is a clean analysis - contribute to cloud cache? (y/n): y
Creating pull request...
✓ PR created: https://github.com/.../pull/42
```

## Tips

1. **Use tab completion** - Many menus support partial matching
2. **Check data status first** - Know what's already downloaded
3. **Pre-download for offline** - Use Data Management to prepare
4. **Let checkpoints work** - Don't force-quit, use Ctrl+C
5. **Review config** - Ensure settings match your hardware

---

Next: [Scoring Methods](scoring-methods.md)
