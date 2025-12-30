# Output Format

Climb Analyzer produces Excel (.xlsx) files containing detailed climb data.

## Output Location

Files are saved to the `output/` directory:

```
output/
├── Vermont_climbs_all_basic_2025-12-17_v2.1.0_e6000.xlsx
├── Colorado_climbs_paved_fiets_2025-12-17_v2.1.0_e10000.xlsx
└── Vermont_errors_all_basic_2025-12-17.txt
```

## Filename Format

```
{Region}_climbs_{surface}_{score}_{date}_v{version}_e{threshold}.xlsx
```

| Component | Description | Example |
|-----------|-------------|---------|
| Region | State/country name | Vermont |
| surface | Surface filter | all, paved, gravel |
| score | Scoring method | basic, fiets, pdi |
| date | Analysis date | 2025-12-17 |
| version | Analyzer version | v2.1.0 |
| threshold | Min score filter | e6000 |

## Column Reference

### Location Fields

| Column | Description | Example |
|--------|-------------|---------|
| Street Name | Road/trail name | Mount Mansfield Road |
| City | Nearest city | Stowe |
| State | State/province | Vermont |
| Country | Country | United States |
| From Center (mi) | Distance from region center | 23.4 |

### Classification

| Column | Description | Example |
|--------|-------------|---------|
| Cycling | Cycling allowed? | Yes / No |
| Category | Climb category | Cat 2, HC |
| Highway Type | OSM road classification | primary, secondary, track |
| Surface | Road surface | paved, gravel, dirt |
| Tracktype | Trail type (if applicable) | grade1, grade2 |

### Scores

| Column | Description | Example |
|--------|-------------|---------|
| Basic Score | distance × grade | 45,230 |
| FIETS Score | FIETS index | 6.8 |
| PDI Score | PJAMM difficulty | 342 |

### Physical Characteristics

| Column | Description | Example |
|--------|-------------|---------|
| Elev Gain (ft) | Total elevation gained | 1,234 |
| Height (ft) | Summit elevation | 4,393 |
| Prominence (ft) | Rise from start to summit | 1,180 |
| Length (mi) | Total climb distance | 3.2 |
| Avg Grade (%) | Average gradient | 5.8 |
| Max Grade (%) | Maximum gradient | 12.3 |

### OSM Reference

| Column | Description | Example |
|--------|-------------|---------|
| Way ID | Primary OSM way ID | 123456789 |
| OSM Link | Hyperlink to OSM | [Link] |
| All Way IDs | All way IDs in climb | 123456789,234567890 |
| Connected Climbs | Adjacent climbs | Way 345678901 |

### Coordinates

| Column | Description | Example |
|--------|-------------|---------|
| Start Lat | Starting latitude | 44.5234 |
| Start Lon | Starting longitude | -72.8123 |
| End Lat | Ending latitude | 44.5456 |
| End Lon | Ending longitude | -72.7890 |

## Units

Output units depend on the `-u` flag:

| Field | Imperial | Metric |
|-------|----------|--------|
| Elevation | feet (ft) | meters (m) |
| Distance | miles (mi) | kilometers (km) |
| From Center | miles (mi) | kilometers (km) |

## Error Files

Climbs that couldn't be processed are logged:

```
Vermont_errors_all_basic_2025-12-17.txt
```

Contains:
- Way ID
- Error description
- Coordinates (if available)

Common errors:
- Missing elevation data
- Invalid geometry
- Incomplete OSM data

## Multi-File Output

Large regions may produce multiple files:

```
California_climbs_all_basic_2025-12-17-1.xlsx  (rows 1-100,000)
California_climbs_all_basic_2025-12-17-2.xlsx  (rows 100,001-200,000)
California_climbs_all_basic_2025-12-17-3.xlsx  (rows 200,001-250,000)
```

Excel has a ~1 million row limit; files are split automatically.

## Viewing Results

### Excel / LibreOffice

Open `.xlsx` files directly. Use filters and sorting:

1. Click column header
2. Data → Filter
3. Sort by score, category, etc.

### Web GUI

```bash
./climb-analyzer -g
# Open http://localhost:3000
# Navigate to Visualize
```

Features:
- Interactive map
- Filter by category, score, surface
- Click climb for elevation profile
- Export filtered results

### iOS App

Import `.xlsx` files to the companion iOS app for mobile viewing with GPS integration.

### Python/Pandas

```python
import pandas as pd

df = pd.read_excel('output/Vermont_climbs_all_basic_2025-12-17.xlsx')

# Top 10 climbs by basic score
top_10 = df.nlargest(10, 'Basic Score')

# All HC category climbs
hc_climbs = df[df['Category'] == 'HC']

# Paved climbs over 5% average grade
steep_paved = df[(df['Surface'] == 'paved') & (df['Avg Grade (%)'] > 5)]
```

## Data Quality

### Completeness

- **Street Name**: May be "Unnamed" for trails without names
- **City/State**: Populated via reverse geocoding (automatic)
- **Surface**: From OSM tags; may be "unknown"

### Accuracy

- **Elevation**: ±3m for NED, ±10m for SRTM
- **Distance**: Derived from OSM geometry
- **Grade**: Calculated from elevation profile

### Connected Climbs

When a climb spans multiple roads with different names, the `Connected Climbs` column links related segments. Example:

```
Main climb: "Mountain Road" (Way 12345)
Connected: "Summit Drive" (Way 23456), "Peak Lane" (Way 34567)
```

These form a continuous uphill route despite name changes.

## Best Practices

1. **Sort by score** - Find best climbs quickly
2. **Filter by category** - Focus on difficulty level
3. **Use "From Center"** - Find climbs near a specific area
4. **Check surface** - Ensure road type matches your bike
5. **Verify in OSM** - Click OSM link to see full road details

---

Next: [Checkpointing](../features/checkpointing.md)
