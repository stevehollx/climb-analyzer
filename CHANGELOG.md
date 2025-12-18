# Changelog

## [2.2.1] - 2025-12-15
- Fix some peak splits not splitting, causing the highest point of the climb to be in the middle of the elevation profile. This was rare, but seen on some very long undulating trails.

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