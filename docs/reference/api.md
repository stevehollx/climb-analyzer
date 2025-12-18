# API Reference

The Web GUI communicates with the backend via REST API endpoints.

## Base URL

```
http://localhost:3000/api
```

## Configuration

### GET /api/config

Fetch current configuration.

**Response:**
```json
{
  "DEPLOYMENT_TYPE": "local",
  "TOPO_API_BASE_URL": "http://opentopodata-server:5000/v1",
  "ELEVATION_BATCH_SIZE": 100,
  "ELEVATION_MAX_CONCURRENT": 16,
  "CLOUD_CACHE_ENABLED": true
}
```

### POST /api/config

Update configuration.

**Request:**
```json
{
  "DEPLOYMENT_TYPE": "cloud",
  "ELEVATION_MAX_CONCURRENT": 2
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully"
}
```

## Analysis

### POST /api/analyze

Start a new analysis.

**Request:**
```json
{
  "region": "Vermont",
  "cyclingOnly": false,
  "scoreThreshold": 6000,
  "scoreType": "basic",
  "units": "imperial",
  "geocoding": true
}
```

**Response:**
```json
{
  "jobId": "vermont-1731868945",
  "status": "started",
  "message": "Analysis started successfully"
}
```

### GET /api/progress/{jobId}

Server-Sent Events stream for progress updates.

**Response (SSE):**
```
data: {"percent": 15, "message": "Processing OSM data..."}

data: {"percent": 45, "message": "Fetching elevations..."}

data: {"percent": 100, "message": "Complete!", "complete": true}
```

## Results

### GET /api/results

List all output files.

**Response:**
```json
{
  "files": [
    {
      "filename": "Vermont_climbs_all_basic_2025-12-17.xlsx",
      "path": "/app/output/Vermont_climbs_all_basic_2025-12-17.xlsx",
      "size": "1.2 MB",
      "modified": "2025-12-17T14:23:15Z",
      "region": "Vermont"
    }
  ]
}
```

### GET /api/output-file?path={path}

Parse and return climb data from Excel file.

**Response:**
```json
{
  "climbs": [
    {
      "streetName": "Mount Mansfield Road",
      "city": "Stowe",
      "state": "Vermont",
      "basicScore": 45230,
      "elevGainFt": 1234,
      "lengthMi": 3.2,
      "avgGradePct": 5.8,
      "category": "Cat 2",
      "lat": 44.5234,
      "lon": -72.8123
    }
  ],
  "totalClimbs": 127
}
```

## Data Management

### GET /api/data-info

Get disk usage information.

**Response:**
```json
{
  "osm": {
    "files": [{"name": "vermont-latest.osm.pbf", "size": "152 MB"}],
    "totalSize": "152 MB"
  },
  "elevation": {
    "datasets": {
      "ned10m": {"tiles": 25, "size": "2.1 GB"},
      "srtm30m": {"tiles": 4, "size": "180 MB"}
    },
    "totalSize": "2.28 GB"
  },
  "checkpoints": {
    "totalSize": "234 MB"
  },
  "totalDiskUsage": "2.67 GB"
}
```

### DELETE /api/data/checkpoints

Delete all checkpoint files.

**Response:**
```json
{
  "success": true,
  "deletedFiles": 3,
  "freedSpace": "1.2 GB"
}
```

### DELETE /api/data/osm

Delete OSM data.

**Query Parameters:**
- `region` (optional): Specific region to delete

**Response:**
```json
{
  "success": true,
  "deletedFiles": 1,
  "freedSpace": "152 MB"
}
```

### DELETE /api/data/elevation

Delete elevation data.

**Query Parameters:**
- `dataset` (optional): Specific dataset to delete

**Response:**
```json
{
  "success": true,
  "deletedFiles": 25,
  "freedSpace": "2.1 GB"
}
```

### DELETE /api/data/all

Delete all data.

**Query Parameters:**
- `includeOutputs` (optional): Also delete output files

**Response:**
```json
{
  "success": true,
  "deletedFiles": 62,
  "freedSpace": "4.5 GB"
}
```

## Downloads

### POST /api/download

Start a data download.

**Request (OSM):**
```json
{
  "type": "osm",
  "region": "Vermont"
}
```

**Request (Elevation):**
```json
{
  "type": "elevation",
  "dataset": "ned10m",
  "region": "Vermont"
}
```

**Response:**
```json
{
  "jobId": "download-osm-vermont-1731868945",
  "status": "started",
  "estimatedSize": "152 MB"
}
```

## Regions

### GET /api/regions

List available regions.

**Response:**
```json
{
  "usStates": [
    {"name": "Vermont", "code": "VT"},
    {"name": "California", "code": "CA"}
  ],
  "countries": [
    {"name": "Switzerland", "code": "CH"},
    {"name": "Iceland", "code": "IS"}
  ]
}
```

## Utilities

### GET /api/data-coverage?region={region}

Check data availability for a region.

**Response:**
```json
{
  "region": "Vermont",
  "hasOSM": true,
  "hasIndex": true,
  "hasElevation": true,
  "elevationDatasets": ["ned10m", "srtm30m"],
  "ready": true
}
```

### POST /api/rebuild-opentopodata

Rebuild elevation server (local mode only).

**Response:**
```json
{
  "success": true,
  "message": "OpenTopoData container rebuilt"
}
```

### GET /api/cloud-cache

Get cloud cache status.

**Response:**
```json
{
  "enabled": true,
  "repo": "stevehollx/global-road-and-trail-climbs",
  "authenticated": true,
  "canPush": true
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": true,
  "message": "Description of what went wrong",
  "code": "ERROR_CODE"
}
```

Common error codes:

| Code | Description |
|------|-------------|
| `NOT_FOUND` | Resource doesn't exist |
| `INVALID_REQUEST` | Bad request parameters |
| `SERVER_ERROR` | Internal server error |
| `UNAUTHORIZED` | Authentication required |

---

Back to [Documentation Home](../index.md)
