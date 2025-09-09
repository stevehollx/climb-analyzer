# Cycling Climb Finder
 
Author: Steve Holl
v1.0 - Sep 2025 - Initial public release.

## Who is this for?
Have you ever visited or moved to a new area as a cyclist, and wanted to know what the top climbs for the area are? Strava no longer lets you search for segments by category on the map view (this was in the UI about 10 years ago), and their segment API only returns a limited set of segments, so there isn't a good way to find all good climbs in a selected area programmatically anymore.

So I built an application to perform this function. It leverages open topographic data to find and rank road climbs, ideally used for cyclists.

Input a zip code and a distance radius, and it will find, classify, and rank the climbs in that area. It will classify based on grade and duration. You can override the minimum grade or distance of the results if you want to change the floor criteria for climb results.

I am also including some tools that help to instantiate a local opentopodata API, that help with downloading and reformatting the elevation data. Finally, check out my other project `overpass-api-macos-m1` which helps to run a local OpenStreetMap Overpass API on Apple Silicon, which you may also need if running this project.

## Dependencies
* Python (using geopy, overpy, and pandas)
* Docker
* OpenStreet Maps Overpass API (overpass-api.de/) with cloud limits, or run locally with wikitorn/overpass-api
* OpenTopoData server (cloud with limits, or run locally)
* (optional) Topo data and/or OSM planet data, depending if hosting local APIs. You'll need roughly 500GB to 1 TB of free space if hosting OSM and topo data locally, depending on what area you are using data for.

## Installation summary
1. (optional) Download topographic data
2. (optional) Build and host a local OpenTopoData API server
3. (optional) Build a local openstreetmap overpass API server.
4. Run ClimbAnalyzer to analyze road climbs for your location(s) of interest.

#### Quickstart tip: Jump to Step 4 for performing just a small area anlysis, without needing to set up the local APIs mentioned in Steps 1-3.
Steps 1 - 3 are optional, but highly recommended. Without running a local API topo data server, you will hit OpenTopoData's 1000 calls/day limit very quickly. An alternative is to sign up for a paid OpenTopoData account. The overpass-api.de API has higher limits, but may still throttle you after a searches in a day, depending on your locations' road density. So if you have the disk space, I'd recommend running both local APIs.

### Step 1- Download topographic data
I recommend SRTM30m for global data sets, or NED10m if you are in North America. SRTM30m is less precise but 78GB compressed vs. NED10m's 210 GB compressed.
1. Create an API account at: opentopography.org/newUser
2. Run `fetch-srtm30m` or `fetch-ned10m` to download the topo data. It will take roughly 4-6 hours to download SRTM30M, and plan for a day for the NED10m data.
3. Note that you can also use other topo data sources if you want. See the [Open Topo Data](https://www.opentopodata.org/datasets/srtm/) dataset documentation for more info.
5. If using NED10m data, run convert_ned10m to convert to the filenaming format opentopodata expects. This runs quick. Optionally, also compress the files with compress_ned10m`.
6. If using SRTM30m v3, there are some [bad files](https://www.opentopodata.org/notes/invalid-srtm-zips/). Run `fix_bad_files` to correct them. Then, optionally compress them with `compress_srt.sh`. This compression gives a 10x improvement in query performance. It will take a few hours to process.

More documentation on these steps is found at [OpenTopoData Server Documentation](https://www.opentopodata.org/server/).

### Step 2 - Install opentopodata API server locally
0. This is highly recommended, since you will be hammering topo elevation requests in queries, and opentopodata.org's public API has a 1000 call/day limit which you will hit in about a single 30 mile search.
1. Follow the instructions to install and host a local [Open Topo Data](https://www.opentopodata.org/server/) server to host a local API from the downloaded top files. Note that on macs you need to disable Airport Receiver to free up TCP/5000 or remap to a different port in the Docker config.
2. Verify that the server is working with test data with: http://localhost:5000/v1/test-dataset?locations=-43.5,172.5|27.6,1.98&interpolation=cubic
3. Move your files to opentopodata/data and add the data reference to opentopodata's config.yaml as described in their [docs](https://www.opentopodata.org/datasets/srtm/). Add `./data` to .dockerignore so that the docker build process doesn't include the huge topo data directories in the docker image. Use `make build-m1` if on a silicon mac.
4. Rebuild and run opentopodata.

### Step 3 - Install overpass-api server locally
0. I recommend doing this if planning to search multiple areas in one day, as there is about a 10k/day request limit, but you can get a few medium sized queries (~30 miles) within a day without the local API, depending on your locations' road density.
1. See the overpass-api/README for these separate instructions to build the local API. I am using a custom build since Apple Silicon macs don't play nice with the off-the-shelf docker images from drolbr or wiktorn.

### Step 4 - Stage Climb Analyzer
1. Edit `climb-analyzer.py` and change the URLs to your local APIs if you are running local servers. It defaults to the cloud APIs that have usage limits mentioned above.

## Running Climb Analyzer to analyze climbs

0. Install python dependencies: `pip install overpy requests geopy numpy pandas tqdm openpyxl`
1. Run ./climb-analyzer.py
2. Input your details to analyze data.
3. Example input and output:

```
Surface filter options:
1. Paved roads only (highways, primary, secondary roads)
2. Gravel roads (tracks with gravel surface)
3. Dirt trails/tracks (OSM track classification)
4. All road types
Enter your choice (1-4): 4
Selected surface filter: all

Unit system options:
1. Metric (meters, kilometers)
2. Imperial (feet, miles)
Enter your choice (1-2, default imperial): 2
Selected unit system: imperial

Analysis scope options:
1. Address with radius (recommended for most users)
2. Entire state (WARNING: Very large computational task)
3. Entire country (WARNING: Extremely large computational task)
Enter your choice (1-3, default 1): 1

Enter the street address to analyze, *without* the country code: 448 Curtis Creek Rd, Old Fort, NC 28762
Enter country name (optional, helps with geocoding accuracy): us
Enter the search radius (default 15 miles): 10
Enter minimum climb score to display (default 6000): 6000


============================================================================================================================================
CLIMB ANALYSIS RESULTS (343 climbs) - Surface filter: all
============================================================================================================================================
                          Street Name Category  Score  Elev Gain (ft)  Height (ft)  Prominence (ft)  Length (mi)  Distance (mi)  Avg Grade (%)  Max Grade (%)   Surface      Tracktype     Way ID                                                   OSM Link
                         Snook's Nose       HC  88627             198         2907             2907         3.91           2.43          14.09          64.94   unpaved not_applicable  410475879   [410475879](https://www.openstreetmap.org/way/410475879)
                    Curtis Creek Road       HC  80377            2917         2637             2637        10.73           6.55           4.65          38.95   unpaved         grade1  492153905   [492153905](https://www.openstreetmap.org/way/492153905)
                      Graybeard Trail        1  66499            2264         2181             2181         4.10           1.54          10.07          47.65    ground not_applicable  224378846   [224378846](https://www.openstreetmap.org/way/224378846)
                     Heartbreak Ridge        2  62115            2256         2037             2037         3.84           2.78          10.05          59.21   unpaved not_applicable  278087888   [278087888](https://www.openstreetmap.org/way/278087888)
                                track        2  58827            1932         1930             1930         1.70           1.23          21.56          66.02   unpaved    unspecified   16648981     [16648981](https://www.openstreetmap.org/way/16648981)
                    Grey Rock Parkway        2  51586             171         1692             1692         4.28           2.69           7.48          26.70   asphalt not_applicable   16753390     [16753390](https://www.openstreetmap.org/way/16753390)
                      Buck Creek Road        2  50413              77         1653             1653         5.03           2.80           6.22          30.77   asphalt not_applicable   16653777     [16653777](https://www.openstreetmap.org/way/16653777)
              Laurel Log Branch Trail        2  49935            1651         1638             1638         1.55           1.40          19.97          58.84    ground not_applicable  410475880   [410475880](https://www.openstreetmap.org/way/410475880)
                Armstrong Creek Trail        2  49116              89         1611             1611         2.75           2.13          11.08          69.40    ground not_applicable  664032102   [664032102](https://www.openstreetmap.org/way/664032102)
           Kitsuma/Youngs Ridge Trail        2  45809            2486         1502             1502         7.11           0.00           4.00          78.58    ground not_applicable  333032293   [333032293](https://www.openstreetmap.org/way/333032293)
```



## Additional Info

### Analysis and resuming progress
The script supports resuming tasks if they are canceled during the road segment data fetching (chunks), as well as during fetching elevations since these two processes can take hours to complete. Simply enter the same parameters as the canceled search, and it will find the appropriate checkpoint and resume if you canceled during processing.

Analysis is done in chunks of areas in batches to keep memory requirements low instead of loading all road segments at once. Each chunk is processed independently, and then segments are grouped by street name and merged, so that we can assemble roads that may traverse from one chunk into the next.

## Road merging
Merging road segments is not a clear discipline, since roads can change names, have multiple names, take hard turns at intersections and keep the same name, etc. That being said, this is the logic for considering it a continuous road:

### When roads get merged:
The system tries to connect road segments that have the same street name AND whose endpoints are close together (within about 200 meters). For example, if "Mountain View Road" is split across multiple data chunks, it will try to reconnect those pieces into one continuous road. Duplicate segments are dropped (we expect duplicates since there is overlapping areas in the tiled area we analyze).

Different street names won't be merged, even if they connect physically, so those will show up as multiple independent climbs. If endpoints are too far apart (more than 200m gap), they stay separate.

Also if the road surface changes, the roads won't merge, since someone on a road bike may not appreciate a change to gravel. So those segments would be independently represented in the output, attributed by their appropriate surface types.

## Criteria to be considered a climb
The system is actually quite liberal about what it calls a climb:

* Any road segment with elevation data gets analyzed as a potential climb
* It doesn't require the road to be "only uphill" - it will analyze roads that go up and down
* A "climb" is simply the entire length of a named road, regardless of whether it has flat or downhill sections mixed in
* The minimum requirement is just having at least 2 points with elevation data
* By default anything over a score of 8000 is represented, but this threshold can be adjusted at runtime. In my area, I'd consider around 8000 a good start of being considered a memorable climb, taking a few minutes or more to complete.

## Scoring climbs
Climbs are scored using:

  Main Score = Distance (in meters) × Average Grade (%)

They are then approximated to categories, similar to how Le Tour de France does it though they account for where the climb is along the stage, which isn't relevant for our purposes.

* Suppressed from results: score < 4000 (can be adjusted)
* Cat 4: score > 8000
* Cat 3: score > 16000
* Cat 2: score > 32000
* Cat 1: score > 64000
* Cat 0 (HC): score > 80000

This system rewards both length and steepness, so a long moderate climb can score higher than a short steep one. It's designed to identify significant climbs that would challenge cyclists or hikers, not just steep driveways.

## Disk usage info
Using NED10m data and N America, my Docker VM uses about 1.1TB of disk. I'd recommend a 2TB drive for US analysis, but if you use the SRTM30m data and maybe smaller planet file like Europe, it may run on 512GB.

## Static exports

I may start exporting dumps of climbs to another static repo, for those that just want a climb database to filter and interact with. If I do that, I'll export state by state for score >8000, probably starting with a bias for the Southeast US states.

## Caveat

I haven't tested yet:
1. The super large state exports.
2. Exporting a whole country.
3. Analysis of countries outside of the US (but I have tested with SRTM30m data, so it should behave the same).

## Changelog

v1.0 - Sep 2025 - Initial release. Basic sanity checks are passing. There may be optimization efforts or some complex climbs that can't be merged that get missed, but I'm seeing all long climbs I know of in the areas I am familiar with.
