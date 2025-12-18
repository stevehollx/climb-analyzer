# Climb Analyzer

**Analyze road and trail climbs worldwide using OpenStreetMap data**

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://stevehollx.github.io/climb-analyzer/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Have you visited a new area and curious what the best climbs in the area are? Strave segments search too cumbersome to find climbs? Want an ability to find **the best** climbs in an area?

Climb Analyzer calculates and documents significant hill and mountain climbs around the world. Built for cyclists, runners, and endurance athletes seeking elevation challenges. It is a thorough bottoms-up analysis of every road or path in OpenStreetMaps.

**[📖 Full Documentation](https://stevehollx.github.io/climb-analyzer/)**

We're using this app to build a [database of climbs](https://github.com/stevehollx/global-road-and-trail-climbs) for everyone to use. When climb databases have been calculated by someone, they are then available to be visualized in this app's UI, or on the go for iOS devices with **climb analyzer: pocket**.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer
./setup_wizard.sh
./climb-analyzer setup

# Analyze a region interactively
./climb-analyzer

# Analyze regions programatically in bulk
./climb-analyzer -r "North Carolina, Georgia, Bristol, France"

#See command line arguments
./climb-analyzer --help
```

## Features

- **Complete Coverage** - Analyzes ALL roads and trails in OpenStreetMap
- **Multiple Climb Scoring formats** - Basic, FIETS, and PDI difficulty indices
- **Cloud Cache** - Download pre-computed analyses or contribute your own
- **Checkpointing** - Resume interrupted analyses
- **Web GUI** - Visualize climbs on interactive maps
- **iOS pocket app** - Visualize climbs on the go on iOS devices. Sorry, no Android support.

## Scoring Methods

| Method | Formula | Best For |
|--------|---------|----------|
| **Basic** | distance × grade | Quick comparisons |
| **FIETS** | H²/D×10 + altitude bonus | European mountain climbs |
| **PDI** | Physics-based total work | Most accurate difficulty |


## Output

Results are Excel files with detailed climb data:

| Field | Description |
|-------|-------------|
| Street Name | Road/trail name |
| Category | HC, Cat 1-4 |
| Scores | Basic, FIETS, PDI |
| Elevation Gain | Total climb |
| Avg/Max Grade | Steepness |
| Surface | Paved, gravel, dirt |
| OSM Link | View in OpenStreetMap |

## Pocket iOS App

<img src="https://github.com/stevehollx/global-road-and-trail-climbs/blob/main/images/ios_map.png" alt="iOS map" width="30%"> <img src="https://github.com/stevehollx/global-road-and-trail-climbs/blob/main/images/ios_climb_info.png" alt="iOS climb info" width="30%">

Companion iOS app for viewing climbs on the go.

## Data Sources

- **Roads & Trails**: [OpenStreetMap](https://www.openstreetmap.org/) (ODbL 1.0)
- **Elevation**: NASA SRTM, USGS NED, ASTER, AW3D30, ArcticDEM

## Contributing

The easiest way to contribute is by running analyses for regions not yet in the cloud cache:

```bash
./climb-analyzer -r "Region Name"
# Accept the prompt to contribute when analysis completes
```

For helping maintin, enhance, and fix issues, see [Contributing Guide](https://stevehollx.github.io/climb-analyzer/contributing/how-to-contribute/) for more ways to help.

I have put significant energy and money into this project. Say thanks by buying me a beer.

## Documentation

Full documentation available at **[stevehollx.github.io/climb-analyzer](https://stevehollx.github.io/climb-analyzer/)**

- [Installation](https://stevehollx.github.io/climb-analyzer/getting-started/installation/)
- [CLI Reference](https://stevehollx.github.io/climb-analyzer/user-guide/cli-reference/)
- [Scoring Methods](https://stevehollx.github.io/climb-analyzer/user-guide/scoring-methods/)
- [Cloud Cache](https://stevehollx.github.io/climb-analyzer/features/cloud-cache/)
- [Web GUI](https://stevehollx.github.io/climb-analyzer/features/web-gui/)

## License

- **Code**: MIT License
- **Data**: [Open Database License (ODbL) v1.0](LICENSE)

If you intend to use this data for commercial purposes, please contact me for permission first.

**Attribution:**
- Elevation data © NASA SRTM, USGS NED, NASA/METI ASTER, PGC ArcticDEM
- Road data © OpenStreetMap contributors, ODbL 1.0
- Climb analysis © 2025 Steve Holl

See more info at [LICENSE](https://github.com/stevehollx/climb-analyzer/LICENSE)

---

**[Get Started →](https://stevehollx.github.io/climb-analyzer/getting-started/installation/)**
