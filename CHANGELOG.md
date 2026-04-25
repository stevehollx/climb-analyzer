# Changelog

## [RC-May26] - 2026-04-25

Release candidate consolidating RC3 (GitHub Releases integration), RC4 (connected-climbs and surface-aware merging), RC5a (iOS partition engine + GUI overhaul), and the in-progress cross-region merging and data-pipeline fixes on top.

### Added

#### iOS / SQLite partitioning
- iOS-compatible SQLite export (`sqlite_export.py`): camelCase columns, UUID primary keys, geohash columns (p1–p6), R-tree spatial index, `file_stats` aggregates, 19 optimized indexes.
- Geographic partition engine: splits oversized regions into Geofabrik subregions, falling back to a Quadtree when subregions still exceed the iOS WASM limit.
- Streaming gzip compression for SQLite release assets (`utils/file_splitter.py:gzip_file`) — ~3× smaller than raw `.sqlite`. Auto-splits gz output into numbered chunks when it exceeds 1.95 GB.

#### Web GUI
- New pages: `/data` (data management / repo browser) and `/settings` (CLI prefs).
- New API routes: `/api/data-download`, `/api/repo-index`.
- New components: `CommandPreview` for showing the equivalent CLI invocation; client-side SQLite parser for browser DB loading.
- Climb detail drawer + map enhancements (interactive elevation profile, surface-aware styling, bidirectional connected-climb links).
- Settings persistence layer (`gui/lib/settings.ts`) and CLI-command builder (`gui/lib/cli-command.ts`).

#### GitHub Releases pipeline
- `cloud_cache.upload_to_release()` and release-management methods on `utils/github_client.py` — replaces git-LFS for asset distribution.
- Release-based xlsx indexing via `scripts/index_xlsx_files.py` (uses full country paths like `united-states-of-america`, writes database fields to `index.json`).
- New automation scripts:
  - `scripts/upload_sqlite_gz_to_releases.py` — gzip + upload missing sqlite assets to existing releases.
  - `scripts/migrate_releases_to_gzip.py` — convert legacy raw `.sqlite` (and split) assets to `.sqlite.gz`, deleting old assets and refreshing region READMEs.
  - `scripts/regenerate_us_state_readmes.py` — fix US state READMEs that drifted from release contents after backfill.
  - `scripts/validate_na_releases.py` — print xlsx / sqlite.gz / dataset-coverage validation table for North America.
  - `scripts/merge_cross_region_releases.py` — top-level orchestrator for cross-region climb merging (downloads, regenerates profiles via osmium+opentopodata, uploads new release tags, opens PR).
  - Region queue runners: `scripts/run_europe_queue.sh`, `run_north_america_queue.sh`, `run_na_caribbean_queue.sh`, `run_final_queue.sh`.

#### Cross-region merging
- New core modules:
  - `climb_analyzer/core/cross_region_merger.py` — schema-clean merged-row builder writing the same physical climb back to both adjacent region dataframes.
  - `climb_analyzer/core/elevation_recompute.py` — recomputes geometry + elevation profile from OSM way IDs via `osmium getid` + pyosmium + opentopodata batch lookups.

#### Elevation datasets
- ArcticDEM (`arctic32m`) wired into `opentopodata-config.yaml` and into the downloader registry under both `arcticdem` and `arctic32m` keys (priority-list naming finally matches downloader names — see Fixed below).
- `rema32m` alias added for REMA downloader.

### Changed

- **Elevation profile output**: dynamic interval scaling targets ≤1500 segments for trails > 10 km; compact format (~15 chars/segment, integer m for distance/elevation, 1-decimal grade). 4000 km PCT now fits in Excel's 32,767-char cell limit. Short climbs (< 10 km) keep full resolution. Parser is backward-compatible.
- **Surface-aware segment merging**: same-name segments with different surfaces (paved vs gravel) stay separate; connected-climb relationships are now bidirectional.
- **Setup wizard** is self-contained (no external deps); NASA Earthdata credential prompts removed (all datasets use public sources).
- **Docker workflow**: auto-detect `DOCKER_GID` (Linux: `getent`/socket stat; macOS: socket stat or 0); documented in `.env.example`. Better error messages from `opentopodata_manager` when the rebuild path can't write.
- **CLI docs**: removed deprecated `-C/-P/-E/-A` flags in favor of `cleanup --osm/--elevation/--all`; removed `-g yes/no` (geocoding is always on); fixed `--radius` typo to `--distance`.
- **Region lookup**: `DataManager.get_region_bounds()` now preserves the full canonical path (`"us/georgia"` vs `"europe/georgia"`) before falling back to the last segment — fixes cross-continent name collisions.
- **Checkpoint datasets format**: `datasets_used.json` is now a dict `{"priority": [...], "actually_used": [...]}`; legacy list format still loads. Reports the configured cascade even when only one dataset supplied data.
- **OSM file lookup**: `find_osm_file_for_region()` accepts a `canonical_path` hint so ambiguous names resolve correctly without sequence/similarity guessing.

### Fixed

- **arctic32m never downloaded** — three stacked bugs (downloader/priority key mismatch `arcticdem`↔`arctic32m`; canonical paths like `us/colorado` not stripped before title-casing in `get_dataset_priority_for_region`; `datasets_used` recorded the cascade rather than which dataset actually returned values). All three corrected; `get_datasets_used()` now returns datasets in cascade order, not alphabetical. See `bugs.md` 2026-04-19 (FIX #20).
- **DEM downloader rollback** for island regions: only rolls back unavailable-tile cache when *all* tiles fail, not >50% — prevents Hawaii / small-island runs from being incorrectly marked as failed.
- **`update_config()` missing required arguments** after the cleanup refactor.
- **Error log filename mismatch** vs xlsx output (`Region_errors_date.txt`); date now syncs with xlsx when analysis spans midnight; surface filter dropped from filename.
- **Elevation datasets showing "Unknown"** in error log header — `stop_elevation_logging()` rewrites the datasets line after fetching completes.
- **Pickle deserialization** mismatch on `ClimbMetrics` after package restructure.
- **Way boundaries merge** now captures all way IDs.
- **Checkpoint cleanup** matches sanitized region names.

### Documentation

- README.md: added Output Formats section.
- `docs/internal/XLSX-INDEXING-AUTOMATION.md`: updated for the new `index.json` schema.
- iOS app privacy policy added.

### Tests

- Unit tests for elevation fetcher, geocoding, and OSM parsing.
- Validation scripts: `pjamm_scraper`, `validate_climbs`.

### Repo hygiene

- `.gitignore`: ignore `.env` / `.env.*`, `.next/`, `.smbdelete*` SMB stubs, and `tests/.pjamm_cache.json`.

### Removed

- v2.3.0 releases for CO, WA, HI, OR (pending reanalysis under the connected-climbs fix).
- Deprecated CLI flags `-C`, `-P`, `-E`, `-A`, `-g`.

## [2.2.2] - 2025-12-15
- Fix some peak splits not splitting properly causing a high peak to not be at the end of the climb. Also found a character limit in the elevation profile excel cell so added smart normalization to keep that under the limit for very long climbs.

## [2.2.0] - 2025-12-13
- Fixed filtering for motorways and interstates that don't allow pedestrians or cyclists from showing up in results.
- Fixed some climbs that weren't being split at the top of the highest peak of the merged ways. It will now split at the end of the way segment that includes the absolute highest elevation. It isn't splitting at the actual peak to save stashing all node elevations in the merged way data. I'll assess if this is required to split at the actual peak node after further review since it is a substantial data rework to include all of that node data in the way data.

## [2.1.0] - 2025-11-01

### Added
- **Cross-chunk climb merging**: Automatically merges climbs that were split at chunk boundaries during large-region analysis
  - Memory-efficient post-processing: operates on climb results (~500MB) instead of segments (~16GB)
  - Prevents missed climbs in large countries like France, Germany, California
  - Configurable via `ENABLE_CROSS_CHUNK_POSTPROCESS` in config.yaml (enabled by default)
  - New CLI utility: `./climb-analyzer --merge-climbs` or `-M` to reprocess existing output files
- Renumbered processing steps for clarity (removed obsolete "Step 4: Skip cross-chunk merge" message)

### Changed
- Large region analysis now includes "Step 6: Post-processing cross-chunk merges" automatically
- Improved step labeling: Step 4 (Extract elevations), Step 5 (Analyze climbs), Step 6 (Merge splits)

### Technical Details
- New module: `climb_analyzer/core/cross_chunk_merge.py` with DataFrame-based merge algorithm
- Uses same proven logic as cross-region merge (spatial indexing, duplicate filtering)
- Typical overhead: 2-5 minutes for France-sized regions
- No risk of OOM crashes (operates on final results, not intermediate segments)

## [2.0.0] - 2025-10-17

Complete rearchitecture for easier distribution, installation, and performance.

### Added
- Setup wizard to guide user through deployment options cloud or local
- CLI arguments for non-interactive running and independent calls for some tasks
- Batch mode to run multiple areas at once
- Checkpointing system
- Can merge climbs that span across subregions. Intentionally not supporting inter-country climb merging, though, so you know you are traversing a country boundary.

### Changed
- MUCH FASTER! Removed segment chunking/fetching/merging in favor of straight way extraction. Removes need for expensive merging and dedup processes. Now calling coordinate extraction and elevation fetching at the same time in batches. US state analysis can now finish in around 2-4 hours for average sized states, instead of what was about 20 hours before.
- Removed overpass docker container instead using osmium and reading pbf files directly
- Changed state/country selection to be geofabrik subregions. All region detection is dynamic now from geofabrik and can be dynamically updated with a geofabrik screenscraper, run with ```utils/update_geo_definitions.py```
- Restructured elevation data to support the entire globe, specifying datasets that cover those areas. Support up to 3 datasets per region in some areas for better coverage, handling overlaps.
- Moved to pyproject.toml format
- Improved memory management--large area analysis may still hit memory limits especially if you have < 16 GB RAM

### Fixed
- Memory leaks in merge operations
- Region bounds calculation
- Defect where long climbs like Crater Rd in Maui weren't fully merging.

## [1.0.0] - 2025-10-17

Initial release. Somewhat functional but wasn't fit for wide distribution.
Required overpass container, and manual setup.
Had a defect in road merging logic so some long climbs weren't detected properly.
Not suggested for use.