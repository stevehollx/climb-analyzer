# Analyzing Large Countries

Some countries are too large to analyze in a single run. This guide explains how to handle them.

## The Problem

Large countries like France, Germany, USA, and China have millions of roads that can exhaust memory:

| Country | Roads | Memory Needed |
|---------|-------|---------------|
| France | ~8 million | 12+ GB |
| Germany | ~10 million | 14+ GB |
| USA | ~50 million | 40+ GB |

**Typical failure**: Out of memory at 40-50% completion.

## Solution: Automatic Country Splitting

As of v2.1.0, Climb Analyzer handles large countries automatically:

1. **Auto-Detection**: Countries with predefined splits are automatically divided
2. **Chunked Processing**: Each region is processed separately (e.g., France North + France South)
3. **Boundary Merging**: Split climbs are rejoined post-analysis
4. **Memory Efficient**: ~500MB instead of 16GB

### Countries with Predefined Splits

| Country | Splits | Description |
|---------|--------|-------------|
| **France** | 2 | North/South at 46°N (near Lyon) |
| **Germany** | 2 | North/South at 51°N (near Cologne) |
| **United States** | 7 | Northeast, Southeast, Midwest, Southwest, Northwest, Alaska, Hawaii |
| **Canada** | 5 | West, Prairies, Central, East, North |
| **Russia** | 4 | West, Ural, Siberia, Far East |
| **China** | 2 | North/South at Yangtze River |
| **Brazil** | 2 | North/South at 15°S (near Brasília) |
| **Australia** | 3 | West, Central, East |
| **Argentina** | 2 | North/South at 35°S |
| **India** | 2 | North/South at Tropic of Cancer |

### How It Works

When you analyze a large country:

```bash
./climb-analyzer -r "France"
```

The system automatically:

1. Detects France needs splitting (predefined in `large_country_handler.py`)
2. Processes "France North" and "France South" separately
3. Identifies climbs that span the 46°N boundary
4. Merges boundary climbs in a memory-efficient post-process step

**Example:**
```
Before merge: Route D117 (8.2 km) + Route D117 (3.1 km) at boundary
After merge:  Route D117 (11.3 km) - complete climb!
```

### Configuration

Enabled by default. To disable:

```yaml
# config.yaml
ENABLE_CROSS_CHUNK_POSTPROCESS: false
```

## Manual Splitting

For countries not in the predefined list or when you need custom boundaries, use bounding box mode.

!!! note
    Manual splits require a post-process merge to reconnect boundary climbs. Use `--merge-regions` after analysis.

### France

```bash
# North France (Paris, Normandy, Brittany)
./climb-analyzer -b 46.0,-5.0,51.2,9.6 --name "France-North"

# South France (Provence, Alps, Pyrenees)
./climb-analyzer -b 41.3,-5.0,46.0,9.6 --name "France-South"
```

### Germany

```bash
# North Germany (Hamburg, Berlin)
./climb-analyzer -b 51.0,5.9,55.1,15.0 --name "Germany-North"

# South Germany (Munich, Stuttgart)
./climb-analyzer -b 47.3,5.9,51.0,15.0 --name "Germany-South"
```

### United States

```bash
# Northeast
./climb-analyzer -b 38.0,-83.0,48.0,-67.0 --name "USA-Northeast"

# Southeast
./climb-analyzer -b 24.0,-90.0,38.0,-75.0 --name "USA-Southeast"

# Midwest
./climb-analyzer -b 36.0,-105.0,49.0,-83.0 --name "USA-Midwest"

# Southwest
./climb-analyzer -b 28.0,-125.0,42.0,-102.0 --name "USA-Southwest"

# Northwest
./climb-analyzer -b 42.0,-125.0,49.0,-102.0 --name "USA-Northwest"
```

### Other Large Countries

**Canada:**
```bash
./climb-analyzer -b 48.0,-141.0,70.0,-110.0 --name "Canada-West"
./climb-analyzer -b 41.0,-95.0,55.0,-74.0 --name "Canada-Central"
./climb-analyzer -b 44.0,-80.0,55.0,-52.0 --name "Canada-East"
```

**Russia:**
```bash
./climb-analyzer -b 41.0,19.0,82.0,60.0 --name "Russia-West"
./climb-analyzer -b 50.0,70.0,75.0,110.0 --name "Russia-Siberia"
./climb-analyzer -b 42.0,110.0,75.0,180.0 --name "Russia-FarEast"
```

**China:**
```bash
./climb-analyzer -b 35.0,73.0,54.0,135.0 --name "China-North"
./climb-analyzer -b 18.0,73.0,35.0,135.0 --name "China-South"
```

## Memory Requirements

| Segments | Memory | Examples |
|----------|--------|----------|
| < 1M | 2-3 GB | Small European countries, US states |
| 1-3M | 3-5 GB | Spain, Poland, Sweden |
| 3-5M | 5-8 GB | UK, Italy, Japan |
| 5-10M | 8-12 GB | France, Germany |
| > 10M | 12+ GB | USA, Russia, China |

### Checking Available Memory

```bash
# Linux/Mac
free -h

# Python
python3 -c "import psutil; print(f'Available: {psutil.virtual_memory().available / (1024**3):.1f} GB')"
```

## Merging Existing Files

For files created before v2.1.0 or manually split analyses:

```bash
# Merge multiple region files
./climb-analyzer --merge-regions France_North.xlsx France_South.xlsx

# Merge with glob pattern
./climb-analyzer --merge-regions output/France_*.xlsx

# Custom distance threshold (default: 0.5 km)
./climb-analyzer --merge-regions output/*.xlsx --merge-distance-km 0.3
```

## Performance Tips

### 1. Increase Swap Space (Linux)

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 2. Use SSD Storage

Put data directory on SSD for faster processing.

### 3. Run During Off-Hours

Large analyses take 4-8+ hours per region.

### 4. Use Checkpoints

Analysis auto-saves every 15 minutes. Safe to interrupt.

### 5. Monitor Memory

```bash
watch -n 5 free -h
```

## Countries That Work Whole

These typically complete without splitting:

- UK / United Kingdom
- Spain
- Italy
- Poland
- Sweden, Norway, Finland
- Japan
- South Korea
- Mexico
- Argentina
- South Africa
- Turkey
- Most US states
- Most Canadian provinces

## Troubleshooting

### "Killed" or "Out of Memory"

- Use manual splits (above)
- Reduce batch size in config
- Add swap space

### Analysis Takes Too Long

- Split into smaller regions
- Use cloud cache if available
- Run multiple regions in parallel on different machines

### Spatial Index Too Large

- Download pre-split OSM files from Geofabrik
- Use region-specific .pbf files

---

Next: [Elevation System](../data/elevation-system.md)
