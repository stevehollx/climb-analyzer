# Parallel Processing

Climb Analyzer uses parallel processing for efficient elevation fetching and analysis.

## Concurrent Elevation Requests

### Configuration

In `config.yaml`:

```yaml
# Local mode (high concurrency)
ELEVATION_MAX_CONCURRENT: 16

# Cloud mode (rate-limited)
ELEVATION_MAX_CONCURRENT: 2

# Batch size per request
ELEVATION_BATCH_SIZE: 100
```

### How It Works

```
Coordinates to fetch: 1,000,000

Batched into: 10,000 batches of 100 coords each
Concurrent: 16 batches at a time

Effective throughput: ~1,600 coords/second (local)
```

### Adjusting for Your System

**More RAM, faster network:**
```yaml
ELEVATION_MAX_CONCURRENT: 32
ELEVATION_BATCH_SIZE: 200
```

**Limited resources:**
```yaml
ELEVATION_MAX_CONCURRENT: 4
ELEVATION_BATCH_SIZE: 50
```

## Batch Region Processing

Process multiple regions sequentially:

```bash
./climb-analyzer -r "VT,NH,ME,MA,CT,RI"
```

Each region processed completely before starting next.

### With Auto-Cleanup

Free disk space between regions:

```bash
./climb-analyzer -r "VT,NH,ME,MA,CT,RI" -X
```

## Parallel Machine Processing

For very large workloads, run on multiple machines:

### Machine 1: Northeast
```bash
./climb-analyzer -r "VT,NH,ME,MA,CT,RI"
```

### Machine 2: Mid-Atlantic
```bash
./climb-analyzer -r "NY,PA,NJ,MD,DE"
```

### Machine 3: Southeast
```bash
./climb-analyzer -r "VA,NC,SC,GA,FL"
```

Results can be combined later if needed.

## Performance Tuning

### Elevation API Bottleneck

If elevation fetching is slow:

1. **Use local mode**: Unlimited queries vs cloud rate limits
2. **Increase concurrency**: `ELEVATION_MAX_CONCURRENT: 32`
3. **Check server health**: `curl http://localhost:5000/health`

### OSM Processing Bottleneck

If OSM extraction is slow:

1. **Use SSD storage**: Put `data/` on SSD
2. **More RAM**: Allows larger in-memory processing
3. **Pre-build indexes**: Run `-D` separately from analysis

### Memory Optimization

For large regions with limited RAM:

```yaml
# Reduce batch sizes
ELEVATION_BATCH_SIZE: 50
OSM_CHUNK_SIZE_KM: 20  # Smaller geographic chunks
```

## Monitoring Performance

### During Analysis

Watch the progress bar:

```
Processing: 45%|████████████░░░░░░░░| (elev_err: 1.2%)
```

- Percentage: Overall progress
- Error rate: Elevation lookup failures

### System Monitoring

```bash
# Memory usage
watch -n 5 free -h

# CPU usage
htop

# Docker stats
docker stats
```

## Benchmarks

Typical performance on modern hardware (SSD, 16GB RAM, local mode):

| Region | Size | Time |
|--------|------|------|
| Rhode Island | Small | 5-10 min |
| Vermont | Small | 15-30 min |
| Colorado | Medium | 2-4 hours |
| California | Large | 8-12 hours |
| France | Very Large | 12-24 hours |

### Factors Affecting Speed

1. **Region size**: More roads = more processing
2. **Disk speed**: SSD vs HDD makes 2-3x difference
3. **Network**: Cloud mode limited by API rate
4. **RAM**: Affects chunk size and caching
5. **Concurrency**: More threads = faster elevation

---

Next: [Configuration](configuration.md)
