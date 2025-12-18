# Cloud Cache Contributions

Contributing climb analyses to the cloud cache helps the entire community.

## What to Contribute

### Good Candidates

- **Countries** without existing analysis
- **US states** without existing analysis
- **Updates** to analyses older than 6 months
- **Regions** with significant OSM improvements

### Not Needed

- Small radius analyses (download parent region instead)
- Custom filtered analyses (only "clean/ufiltered" accepted)
- Duplicate recent analyses, unless it has less elevation errors or run with a newer version of climb analyzer.

## Requirements for Contribution

Analyses must be "clean" (standardized):

| Setting | Required Value |
|---------|----------------|
| Surface Filter | `all` |
| Minimum Score | `0` |
| Cycling Filter | `off` (disabled) |

**Why?** Standardized data ensures everyone gets the same base dataset that they can filter locally.

## How to Contribute

### Step 1: Run Clean Analysis

```bash
# Default settings produce clean analysis
./climb-analyzer -r "Region Name"
```

Ensure you're using default options (no `-s`, `--cycling-filter`, `-m`, or `-t` flags).

### Step 2: Accept Contribution Prompt

After analysis completes:

```
✓ Analysis complete! Found 1,234 climbs

This is a clean analysis - contribute to cloud cache? (yes/no): yes
```

It auto contributed, but can be turned off in settings, if you don't like to help others.

### Step 3: Wait for PR

```
Creating pull request...
✓ Pull request created: https://github.com/.../pull/42

Your contribution will be available after human approval.
```

### Step 4: Maintainer Review

A maintainer will review and merge your PR. The data becomes available to everyone.

## No GitHub Account Needed

The embedded credentials handle authentication:

- You don't need a GitHub account
- No OAuth or login required
- Contributions attributed to "Global Climbs Contributor" bot

## Checking What's Already Contributed

### Via Cloud Cache Check

```bash
./climb-analyzer -r "Region Name"
# Will show if cached data exists
```

### Via Repository

Browse: https://github.com/stevehollx/global-road-and-trail-climbs

```
global-road-and-trail-climbs/
├── africa/
├── asia/
├── europe/
├── north-america/
│   └── united-states-of-america/
│       ├── alabama/
│       ├── alaska/
│       └── ...
└── ...
```

## Priority Regions

### High Priority (Not Yet Analyzed)

Check the repository for gaps. My priority is roughly based on cycling popularity:
1. All US states, with cross-state climbs analyzed (i will eprform cross state analysis when all states done).
2. EU countries, starting with Western Europe
3. Japan
4. Australia
5. South America

### Medium Priority (Need Updates)

Regions analyzed >6 months ago may benefit from:

- OSM improvements
- Better elevation data
- Bug fixes in analyzer

## Large Region Tips

For large countries like France or Germany:

1. Split into sub-regions if needed
2. Ensure sufficient RAM (8GB+)
3. Use local mode for faster processing
4. Allow several hours for completion
5. Run regions in batch mode--it will merge climbs across region boundaries automatically this way.

See [Large Country Analysis](../features/large-countries.md).

## After Contributing

Your analysis:

1. Creates a Pull Request
2. Waits for maintainer review
3. Gets merged to main branch
4. Becomes available to all users

Typical review time: 1-8 days.

## Troubleshooting

### "Not a clean analysis"

Ensure you're using default settings:

```bash
# Correct - clean analysis
./climb-analyzer -r "Vermont"

# Wrong - not clean
./climb-analyzer -r "Vermont" -s paved
./climb-analyzer -r "Vermont" --cycling-filter
./climb-analyzer -r "Vermont" -m 10000
```

### "Upload failed"

- Check internet connection
- Try again (transient network issues)
- Very large files may timeout

### "PR not created"

```bash
# Test authentication
python3 test_cloud_cache_auth.py
```

## Recognition

All contributions are tracked in the repository's PR history. Thank you for helping build the community dataset!

---

Back to [How to Contribute](how-to-contribute.md)
