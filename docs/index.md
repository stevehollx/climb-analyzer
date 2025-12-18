# Climb Analyzer

**Analyze road and trail climbs worldwide using OpenStreetMap data**

Climb Analyzer is an open-source tool that calculates and documents significant hill and mountain climbs around the world. Built for cyclists, runners, and endurance athletes seeking elevation challenges.

<div class="grid cards" markdown>

-   :material-run-fast:{ .lg .middle } __Quick Start__

    ---

    Get up and running in minutes with Docker

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-map-marker-path:{ .lg .middle } __Find Climbs__

    ---

    Analyze any region - states, countries, or custom areas

    [:octicons-arrow-right-24: User Guide](user-guide/cli-reference.md)

-   :material-cloud-download:{ .lg .middle } __Cloud Cache__

    ---

    Download pre-computed analyses or contribute your own

    [:octicons-arrow-right-24: Cloud Cache](features/cloud-cache.md)

-   :material-chart-line:{ .lg .middle } __Scoring Methods__

    ---

    Three scoring algorithms: Basic, FIETS, and PDI

    [:octicons-arrow-right-24: Scoring](user-guide/scoring-methods.md)

</div>

## What is Climb Analyzer?

Climb Analyzer processes OpenStreetMap (OSM) data combined with elevation data to identify and score every significant climb in a region. It works with:

- **Roads** - Paved roads, highways, bike paths
- **Trails** - Hiking trails, mountain bike trails, greenways
- **Any surface** - Gravel, dirt, mixed surfaces

No other tool or platform gives you full access to know all climbs in an area, for you to decide what climbs you want to leverage on your route. Strava segments is not a bottoms-up analysis and is human curated and messy with overlapping segments. PJAMM has a list of classic climbs, but it isn't an analysis of all roads and paths, so not relevant for users looking for climbs in flatter or less iconic areas. You need a bottoms-up analysis of roads and climbs to fully be able to plan the right climbs you want to hit for your area, which is what this does.

## My goal
Analyze all global road and trail climbs and stash in a [database](https://github.com/stevehollx/global-road-and-trail-climbs/tree/main) for all to be able to freely leverage for route planning.

### Key Features

- **Complete coverage** - Analyzes ALL roads and trails in OpenStreetMap
- **Multiple scoring methods** - Basic, FIETS, and PDI difficulty indices
- **Intelligent climb detection** - Automatically identifies climb segments
- **Connected climbs** - Links adjacent segments that form continuous climbs
- **Cloud cache** - Share and download pre-computed analyses
- **Checkpointing** - Resume interrupted analyses
- **Web GUI** - Visualize climbs on interactive maps
- **iOS app** - Visualize and find climbs on the go from your mobile device

## Scoring Methods

### Basic Score
Simple formula: `distance × average_grade`

Good for quick comparisons. Higher score = longer or steeper climb.

### FIETS Index
Developed by Dutch cycling magazine *Fiets*:

```
FIETS = (H² / D × 10) + max(0, T - 1000) / 1000
```

Where H = elevation gain, D = distance, T = summit height.

### PDI (PJAMM Difficulty Index)
Most sophisticated scoring from [PJAMM Cycling](https://pjammcycling.com):

- Accounts for total work (not just elevation gain)
- Includes wind/friction resistance on flat sections
- Penalizes descents that offer recovery

## Output Data

Each climb includes:

| Field | Description |
|-------|-------------|
| Street Name | Road or trail name |
| City/State | Location (reverse geocoded) |
| Category | HC, Cat 1-4 (cycling classification) |
| Scores | Basic, FIETS, PDI |
| Elevation Gain | Total climb in feet/meters |
| Length | Distance in miles/km |
| Avg/Max Grade | Steepness percentages |
| Surface | Paved, gravel, dirt, etc. |
| Cycling Allowed | Yes/No |
| OSM Link | Direct link to OpenStreetMap |

## Data Sources

- **Roads & Trails**: [OpenStreetMap](https://www.openstreetmap.org/) (ODbL 1.0)
- **Elevation**: NASA SRTM, USGS NED, ASTER, AW3D30, ArcticDEM

## Community

The [Global Road and Trail Climbs](https://github.com/stevehollx/global-road-and-trail-climbs) repository contains pre-computed analyses contributed by the community. Use the cloud cache feature to download existing data or contribute your own analyses.

## License

- **Code**: MIT License
- **Data**: [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/)

---

Ready to get started? Head to the [Installation Guide](getting-started/installation.md).
