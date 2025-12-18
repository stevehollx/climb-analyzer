# Cloud Cache

The cloud cache enables sharing and downloading pre-computed climb analyses, saving time and resources.

## Overview

The [Global Road and Trail Climbs](https://github.com/stevehollx/global-road-and-trail-climbs) repository stores community-contributed analyses organized by continent and region.

## Downloading Cached Data

### Automatic Check

When you run an analysis, the system automatically checks the cloud cache:

```bash
./climb-analyzer -r "Iceland"
```

```
✓ Iceland analysis found in cloud cache (2025-10-25)
Download cached analysis? (yes/no): yes

Downloading...
✓ Downloaded 1 file, 12.5 MB
```

### Benefits

- **Instant results**: No processing time
- **Zero setup**: No elevation data needed
- **Community data**: Benefit from others' work

## Contributing Data

After completing a "clean" analysis, you can contribute:

```
✓ Analysis complete! Found 1,234 climbs

This is a clean analysis - contribute to cloud cache? (yes/no): yes
Creating pull request...
✓ Pull request created: https://github.com/.../pull/42

Your contribution will be available after review!
```

### What is a "Clean" Analysis?

Only standardized analyses can be cached:

| Setting | Required Value |
|---------|----------------|
| Surface Filter | `all` |
| Minimum Score | `0` |
| Cycling Filter | `off` |
| Score Type | `basic` |

!!! note "Why Standardize?"
    This ensures everyone gets the same base dataset. Apply your own filters locally after downloading.

## Repository Structure

```
global-road-and-trail-climbs/
├── africa/
│   ├── kenya/
│   └── south-africa/
├── asia/
│   ├── japan/
│   └── nepal/
├── europe/
│   ├── france/
│   ├── iceland/
│   └── switzerland/
├── north-america/
│   ├── canada/
│   ├── mexico/
│   └── united-states-of-america/
│       ├── california/
│       ├── colorado/
│       └── vermont/
├── oceania/
│   └── australia/
└── south-america/
    └── chile/
```

## File Naming

```
{region}_climbs_{surface}_{score}_{date}[-{part}].xlsx
```

Examples:
- `iceland_climbs_all_basic_2025-10-25.xlsx`
- `california_climbs_all_basic_2025-10-25-1.xlsx` (part 1)
- `california_climbs_all_basic_2025-10-25-2.xlsx` (part 2)

## Configuration

### Enable/Disable

In `config.yaml`:

```yaml
CLOUD_CACHE_ENABLED: true  # or false
```

Or via environment variable:

```bash
export CLOUD_CACHE_ENABLED=false
```

### Repository Setting

```yaml
CLOUD_CACHE_REPO: 'stevehollx/global-road-and-trail-climbs'
```

## Authentication

### For Downloads

No authentication needed - data is publicly accessible.

### For Uploads

Uses embedded GitHub App credentials. No personal GitHub account required.

!!! info "Anonymous Contributions"
    The embedded credentials allow PR creation without a GitHub account. Your contribution is attributed to the "Global Climbs Contributor" bot.

## What to Contribute

**Good candidates:**

- Countries without existing analysis
- US states without existing analysis
- Updates to analyses older than 6 months

**Not needed:**

- Small radius analyses (download parent region)
- Custom filtered analyses
- Duplicate recent analyses

## Privacy

### What Gets Uploaded

- Climb coordinates (latitude/longitude)
- Elevation profiles
- Distance and gradient data
- Calculated scores

### What NEVER Gets Uploaded

- Your search location
- IP address
- Personal information
- System details

## Troubleshooting

### "Cloud cache check failed"

```bash
# Check internet connection
ping github.com

# Disable cache temporarily
export CLOUD_CACHE_ENABLED=false
./climb-analyzer -r "Vermont"
```

### "Download failed"

- Check network connectivity
- Large files may timeout - try again
- Files can be 50-100MB per region

### "Upload failed"

```bash
# Test authentication
python3 test_cloud_cache_auth.py
```

Common issues:
- Network timeout
- GitHub API rate limit (rare)
- File too large

## Technical Details

### Performance

| Operation | Time |
|-----------|------|
| Check if cached | ~1 second |
| Download 10MB | ~5-10 seconds |
| Upload 10MB | ~10-20 seconds |
| Create PR | ~2-3 seconds |

### Rate Limits

- GitHub API: 5,000 requests/hour (authenticated)
- Downloads: Unlimited
- Uploads: Unlimited

No limits should be hit during normal usage.

### File Size Limits

- Maximum: 100MB per file
- Large analyses automatically split

---

Next: [Web GUI](web-gui.md)
