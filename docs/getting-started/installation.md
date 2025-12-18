# Installation

Climb Analyzer runs in Docker for consistent cross-platform support. This guide covers installation on macOS, Linux, and Windows (WSL).

## Prerequisites

- **Docker** - [Install Docker Desktop](https://docs.docker.com/get-docker/)
- **Git** - For cloning the repository
- **8GB+ RAM** - 16GB recommended for large regions
- **50GB+ disk space** - For elevation data and OSM files

## Quick Install

```bash
# Clone the repository
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer

# Run the setup wizard
./climb-analyzer setup
```

The setup wizard will:

1. Check Docker installation
2. Build the Docker container
3. Configure data directories
4. Optionally install the Web GUI
5. Set up elevation data sources

## Platform-Specific Notes

### macOS

Works on both Intel and Apple Silicon (M1/M2/M3). The setup automatically detects your architecture.

```bash
# Apple Silicon users may need Rosetta for some operations
softwareupdate --install-rosetta
```

### Linux

Ensure your user is in the `docker` group:

```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Windows (WSL)

1. Install [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install Docker Desktop with WSL 2 backend
3. Run commands from WSL terminal

```bash
# In WSL terminal
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer
./climb-analyzer setup
```

## Verifying Installation

```bash
# Check CLI is working
./climb-analyzer --help

# Check Docker container
docker ps | grep opentopodata-server

# Run a test analysis (small region)
./climb-analyzer -r "Rhode Island"
```

## Elevation Data Setup

For **local mode** (recommended for serious use), you need elevation data:

### NASA Earthdata Account

1. Create free account at [NASA Earthdata](https://urs.earthdata.nasa.gov/users/new)
2. The setup wizard will prompt for credentials
3. Credentials stored in `.credentials/netrc`

### Data Downloads

The setup wizard handles elevation data, or download manually:

```bash
# Via interactive menu
./climb-analyzer
# Select: Data Management → Download Elevation Data

# Or via CLI
./climb-analyzer -D -r "Vermont"
```

## Cloud Mode vs Local Mode

| Mode | Pros | Cons |
|------|------|------|
| **Cloud** | No data download, quick start | Rate limited (100K coords/day) |
| **Local** | Unlimited queries, faster | Requires disk space, setup |

Configure in `config.yaml`:

```yaml
# Cloud mode (uses api.opentopodata.org)
DEPLOYMENT_TYPE: 'cloud'

# Local mode (uses local OpenTopoData server)
DEPLOYMENT_TYPE: 'local'
```

## Updating

```bash
# Pull latest code
git pull

# Rebuild container
docker compose build

# Restart
docker compose down && docker compose up -d
```

## Troubleshooting

### "Docker not found"

Ensure Docker Desktop is running and the CLI is in your PATH:

```bash
# Test Docker
docker --version
docker compose --version
```

### "Permission denied"

```bash
# Fix ownership
sudo chown -R $(id -u):$(id -g) .

# Or run with sudo (not recommended)
sudo ./climb-analyzer setup
```

### "Out of memory"

Increase Docker memory allocation:

1. Docker Desktop → Settings → Resources
2. Set Memory to 8GB or higher
3. Restart Docker

### Container won't start

```bash
# Check logs
docker logs climb-analyzer

# Rebuild from scratch
docker compose down -v
docker compose build --no-cache
docker compose up -d
```
### Rebuilding containers

```bash
./climb-analyzer build
./utils/rebuild-opentopodata.sh

```
---

Next: [Quick Start Guide](quickstart.md)
