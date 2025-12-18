# Climb Analyzer - Installation Guide

Comprehensive guide for installing and configuring Climb Analyzer on macOS and Ubuntu Linux.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [Installation Methods](#-installation-methods)
  - [Docker Installation (Recommended)](#docker-installation-recommended)
  - [Local Python Installation](#local-python-installation-advanced)
- [Configuration](#-configuration)
- [First Run](#-first-run)
- [Web GUI Setup](#-web-gui-setup)
- [Troubleshooting](#-troubleshooting)
- [Upgrading](#-upgrading)
- [Uninstallation](#-uninstallation)

---

## 🚀 Quick Start

**For experienced users:**

```bash
# 1. Clone the repository
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer

# 2. Ensure Docker and Docker Compose are installed
docker --version
docker compose version

# 3. Run the setup wizard (builds Docker image, configures everything)
./climb-analyzer setup

# 4. Launch the web GUI (optional)
./climb-analyzer -g

# 5. Run your first analysis
./climb-analyzer -r "Rhode Island"
```

---

## 💻 System Requirements

### Operating Systems
- **macOS**: 10.15 (Catalina) or later
- **Ubuntu Linux**: 20.04 LTS or later
- **Other Linux**: Debian-based distributions with systemd

### Software Requirements
- **Docker**: 20.10 or later
- **Docker Compose**: 2.0 or later (or `docker-compose` 1.29+)
- **Git**: Any recent version
- **Bash**: 4.0+ (pre-installed on macOS/Linux)

### Hardware Requirements

#### Minimal Configuration (Cloud Mode)
- **CPU**: 2+ cores
- **Memory**: 4 GB RAM
- **Disk Space**: 5-10 GB
- **Network**: Stable internet connection required

#### Recommended Configuration (Local Mode - Small Regions)
- **CPU**: 4+ cores (8+ recommended)
- **Memory**: 8 GB RAM (16 GB recommended)
- **Disk Space**: 50-100 GB
- **Network**: Internet for initial data downloads

#### Large-Scale Configuration (Local Mode - States/Countries)
- **CPU**: 8+ cores
- **Memory**: 16-32 GB RAM
- **Disk Space**: 200-500 GB per large region
  - Example: France requires ~400 GB (OSM: 3.7 GB, DEM: ~350 GB, outputs: ~50 GB)
- **Network**: High-speed internet for data downloads

### Disk Space Breakdown

**Cloud Mode** (online analysis, minimal storage):
- Docker images: ~2 GB
- Application code: ~500 MB
- Temporary files: 1-2 GB
- **Total**: ~5 GB

**Local Mode** (offline analysis, per region):
- Docker images: ~2 GB
- OSM data (.pbf files): 150 MB - 5 GB per region
- Spatial indexes: 2-10x OSM file size
- Elevation data (DEM tiles): 10-350 GB depending on region and tier
- Checkpoint data: 1-10 GB during analysis
- Output files: 5-50 MB per analysis
- **Total**: 20 GB (small) to 400+ GB (large countries)

### External Services

#### Required for Elevation Data Downloads
- **NASA Earthdata Account**: Free account required for downloading SRTM, ASTER datasets
  - Register at: https://urs.earthdata.nasa.gov/users/new
  - Credentials stored locally in `.credentials/netrc`

#### Optional for Cloud Cache
- **GitHub App**: For sharing analysis results with community
  - See [docs/GITHUB_APP_SETUP.md](docs/GITHUB_APP_SETUP.md)
  - Not required for basic usage

---

## 📦 Installation Methods

### Docker Installation (Recommended)

Docker provides a consistent, isolated environment with all dependencies pre-installed. This is the **recommended method** for most users.

#### Step 1: Install Prerequisites

**macOS:**
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Or install via Homebrew
brew install --cask docker

# Start Docker Desktop from Applications folder
# Verify installation
docker --version
docker compose version
```

**Ubuntu Linux:**
```bash
# Update package index
sudo apt-get update

# Install required packages
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Docker Compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to the docker group (avoids sudo for docker commands)
sudo usermod -aG docker $USER

# Log out and log back in for group changes to take effect
# Verify installation
docker --version
docker compose version
```

#### Step 2: Clone the Repository

```bash
cd /path/to/your/projects
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer
```

#### Step 3: Set Up Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# Required fields:
#   - HOST_UID: Your user ID (run: id -u)
#   - HOST_GID: Your group ID (run: id -g)
#   - EARTHDATA_USER: NASA Earthdata username
#   - EARTHDATA_PASS: NASA Earthdata password

# Quick setup (auto-detects UID/GID)
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env

# Add NASA Earthdata credentials (register at https://urs.earthdata.nasa.gov)
echo "EARTHDATA_USER=your_username" >> .env
echo "EARTHDATA_PASS=your_password" >> .env
```

**Important**: macOS users typically have `UID=501` and `GID=20`, Linux users typically have `UID=1000` and `GID=1000`.

#### Step 4: Run the Setup Wizard

The setup wizard will guide you through:
- Docker image building
- Deployment mode selection (LOCAL vs CLOUD)
- OpenTopoData elevation server setup (LOCAL mode only)
- NASA Earthdata credential configuration
- Elevation dataset tier selection
- Configuration file creation
- Optional web GUI installation

```bash
./climb-analyzer setup
```

**Setup Wizard Steps:**

1. **System Checks**: Verifies Docker, Docker Compose, Python, disk space
2. **Docker Image Building**: Builds the `climb-analyzer` container (~5-10 minutes)
3. **Deployment Mode Selection**:
   - **LOCAL**: Offline analysis using downloaded OSM files and local elevation server
     - Best for: Multiple analyses, large regions, offline work
     - Requires: Significant disk space, one-time data downloads
   - **CLOUD**: Online analysis using Overpass API and public elevation APIs
     - Best for: Quick analyses, small regions, minimal storage
     - Requires: Stable internet connection, slower for large regions
4. **OpenTopoData Setup** (LOCAL mode only): Builds elevation data server container
5. **NASA Earthdata Credentials**: Stores credentials for DEM downloads
6. **Elevation Dataset Tiers**: Configures which elevation datasets to use
   - **primary**: Highest quality (SRTM, NED)
   - **secondary**: Medium quality (AW3D30, ASTER)
   - **tertiary**: Lowest quality (ETOPO, global fallback)
7. **Configuration**: Creates `config.yaml` with your settings
8. **Web GUI Installation** (optional): Installs Next.js web interface
9. **Launch Prompt**: Option to start web GUI immediately

**Example Output:**
```
✓ Docker version: 24.0.6
✓ Docker Compose version: 2.21.0
✓ Python version: 3.11.6
✓ Disk space available: 234.5 GB

Building Docker image...
[+] Building 245.3s (23/23) FINISHED

Select deployment mode:
  [1] LOCAL - Offline analysis (recommended for multiple analyses)
  [2] CLOUD - Online analysis (recommended for quick, small regions)
Choice: 1

Building OpenTopoData server...
[+] Building 89.2s (18/18) FINISHED

NASA Earthdata credentials required for elevation data downloads.
Register at: https://urs.earthdata.nasa.gov/users/new
Username: your_username
Password: [hidden]
✓ Credentials saved to .credentials/netrc

Select elevation dataset tiers (primary+secondary+tertiary recommended):
  [1] primary only (fastest, good coverage)
  [2] primary+secondary (better coverage)
  [3] primary+secondary+tertiary (best coverage, slower)
Choice: 3

✓ Configuration saved to config.yaml

Install web GUI? (y/n): y
Installing web GUI dependencies...
✓ Web GUI installed successfully

Setup complete! 🎉

Run your first analysis:
  ./climb-analyzer -r "Rhode Island"

Or launch the web GUI:
  ./climb-analyzer -g
  Then visit: http://localhost:3000
```

#### Step 5: Verify Installation

```bash
# Test Docker container
./climb-analyzer shell
# Inside container:
climb-analyzer --version
exit

# Check configuration
cat config.yaml

# List available commands
./climb-analyzer --help
```

---

### Local Python Installation (Advanced)

**⚠️ Warning**: This method is more complex and requires manual installation of system libraries. **Docker installation is strongly recommended** for most users.

This method is suitable for:
- Python developers who want to modify the code
- Environments where Docker is not available
- Integration with existing Python environments

#### Prerequisites

**System Libraries** (must be installed first):

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system libraries
brew install osmium-tool spatialindex gdal geos proj
```

**Ubuntu Linux:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    libosmium-dev \
    osmium-tool \
    libspatialindex-dev \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    build-essential \
    git
```

#### Installation Steps

**1. Clone Repository:**
```bash
git clone https://github.com/stevehollx/climb-analyzer.git
cd climb-analyzer
```

**2. Create Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Python Package:**

For **LOCAL mode** (offline analysis):
```bash
pip install -e ".[local]"
```

For **CLOUD mode** (online analysis):
```bash
pip install -e ".[cloud]"
```

For **both modes** (development):
```bash
pip install -e ".[all]"
```

**4. Run Installation Script:**
```bash
python install.py
```

This will:
- Verify system libraries are installed
- Install Python dependencies
- Create default `config.yaml`
- Test imports

**5. Configure Application:**
```bash
# Edit config.yaml with your preferences
nano config.yaml

# Add NASA Earthdata credentials
mkdir -p .credentials
echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" > .credentials/netrc
chmod 600 .credentials/netrc
```

**6. Install OpenTopoData (LOCAL mode only):**

For local mode, you need to set up an OpenTopoData server:

```bash
# Clone OpenTopoData repository
git clone https://github.com/ajnisbet/opentopodata.git
cd opentopodata

# Install dependencies
pip install -r requirements.txt

# Configure datasets (edit config.yaml to point to your elevation data)
# Start server
python app.py
```

See [docs/ELEVATION_SYSTEM_SUMMARY.md](docs/ELEVATION_SYSTEM_SUMMARY.md) for detailed elevation server setup.

**7. Verify Installation:**
```bash
climb-analyzer --version
climb-analyzer --help
```

---

## ⚙️ Configuration

### Configuration File: `config.yaml`

The `config.yaml` file controls all aspects of Climb Analyzer behavior. It's created automatically by the setup wizard, but you can edit it manually.

**Key Configuration Sections:**

```yaml
# Deployment type: 'local' or 'cloud'
DEPLOYMENT_TYPE: local

# Elevation API settings
TOPO_API_BASE_URL: http://opentopodata-server:5000/v1  # LOCAL mode
# TOPO_API_BASE_URL: https://api.opentopodata.org/v1  # CLOUD mode
ELEVATION_BATCH_SIZE: 100              # Coords per request
ELEVATION_MAX_CONCURRENT: 16           # Parallel workers (LOCAL: 16, CLOUD: 2)
ELEVATION_DATASET_TIERS: primary+secondary+tertiary  # Dataset priority

# Checkpoint settings (for long-running analyses)
CHECKPOINT_INTERVAL_MIN: 15            # Auto-save every 15 minutes
CHECKPOINT_MILESTONES_PERC: [10, 25, 50, 75, 90]  # Progress milestones

# Processing settings
OSM_CHUNK_SIZE_KM: 30                  # Spatial chunk size

# Cloud cache (optional - for sharing results)
CLOUD_CACHE_ENABLED: true
CLOUD_CACHE_REPO: stevehollx/global-road-and-trail-climbs

# Tracked data (managed automatically by the system)
OSM_COVERAGE: []                       # Analyzed regions
OSM_PLANET_DATA: []                    # Downloaded OSM files
ELEVATION_DATASETS: {}                 # Downloaded elevation data
```

### Environment Variables: `.env`

Environment variables are loaded from `.env` file (never commit this file to version control):

```bash
# NASA Earthdata credentials (required for DEM downloads)
EARTHDATA_USER=your_username
EARTHDATA_PASS=your_password

# Docker user mapping (for file permissions)
HOST_UID=501   # macOS default, Linux usually 1000
HOST_GID=20    # macOS default, Linux usually 1000

# GitHub App credentials (optional - for cloud cache)
GITHUB_APP_ID=123456
GITHUB_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"
```

### Deployment Modes Comparison

| Feature | LOCAL Mode | CLOUD Mode |
|---------|-----------|------------|
| **OSM Data Source** | Downloaded .pbf files | Overpass API |
| **Elevation Source** | Local OpenTopoData server | Public API (opentopodata.org) |
| **Disk Space** | High (20-400 GB per region) | Low (~5 GB) |
| **Internet Required** | Only for initial downloads | Always required |
| **Analysis Speed** | Fast (local processing) | Slower (API rate limits) |
| **Region Size Limit** | None (can process entire countries) | Small-medium regions only |
| **Setup Complexity** | Higher (data downloads) | Lower (minimal setup) |
| **Best For** | Multiple analyses, large regions | Quick analyses, small regions |

### Elevation Dataset Tiers

Climb Analyzer supports multiple elevation datasets with automatic fallback:

**Primary** (highest quality, best coverage):
- **SRTM** (Shuttle Radar Topography Mission): 1-arc-second (~30m) for 60°N-56°S
- **NED** (National Elevation Dataset): 1-arc-second (~10m) for USA
- **EUDEM** (EU Digital Elevation Model): 25m for Europe

**Secondary** (medium quality, gaps in primary):
- **AW3D30** (ALOS World 3D): 1-arc-second (~30m) global
- **ASTER** (Advanced Spaceborne Thermal Emission): 1-arc-second (~30m) global

**Tertiary** (lowest quality, global fallback):
- **ETOPO** (Earth Topography): Low-resolution global dataset
- **ARCTICDEM** (Polar Geospatial Center): Arctic regions (60°N+)

**Recommended Setting**: `primary+secondary+tertiary` for best coverage

---

## 🎬 First Run

### Interactive Mode (Guided Setup)

The easiest way to run your first analysis:

```bash
./climb-analyzer
```

This launches an interactive menu that guides you through:
1. Region selection (US states, countries, custom areas)
2. Climb type filtering (cycling-only vs all climbs)
3. Score threshold (basic climb score minimum)
4. Data download (OSM and elevation data if needed)
5. Analysis execution
6. Output file location

**Example Session:**
```
=== Climb Analyzer v2.0.1 ===

Select region type:
  [1] US State
  [2] Country
  [3] Custom Region
  [4] Exit
Choice: 1

Select US State:
  [1] Rhode Island
  [2] California
  [3] New York
  ...
Choice: 1

Include cycling-only climbs? (y/n): n
Minimum basic score (0 = all climbs): 6000

Checking for required data...
✓ OSM data found: rhode-island-latest.osm.pbf
✓ Elevation data configured

Starting analysis...
[========================================] 100% Complete
Analysis complete! 🎉

Output saved to: output/Rhode_Island_climbs_all_basic_2025-11-17_v2.0.1_e6000.xlsx
```

### Command-Line Mode (Direct Execution)

For advanced users or automated workflows:

```bash
# Analyze a US state
./climb-analyzer -r "Rhode Island"

# Analyze a country
./climb-analyzer -r "Luxembourg"

# Include only cycling-friendly climbs
./climb-analyzer -r "California" -c

# Set minimum climb score threshold
./climb-analyzer -r "Hawaii" -e 10000

# Custom output directory
./climb-analyzer -r "Monaco" -o /path/to/output/

# Resume from checkpoint (if analysis was interrupted)
./climb-analyzer -r "France" --resume

# Show detailed progress
./climb-analyzer -r "Rhode Island" -v
```

**Common CLI Options:**

| Option | Description |
|--------|-------------|
| `-r, --region` | Region name (US state or country) |
| `-c, --cycling-only` | Include only cycling-friendly climbs |
| `-e, --score-threshold` | Minimum basic climb score (default: 6000) |
| `-o, --output-dir` | Output directory for results |
| `--resume` | Resume from checkpoint |
| `-v, --verbose` | Show detailed progress |
| `-g, --gui` | Launch web GUI |
| `--setup` | Run setup wizard |
| `--help` | Show all available options |

See [docs/CLI_ARGUMENTS_SPEC.md](docs/CLI_ARGUMENTS_SPEC.md) for complete CLI reference.

### Understanding Output Files

Analysis results are saved as Excel files (.xlsx) in the `output/` directory:

**Filename Format:**
```
{Region}_climbs_{climb_type}_{score_type}_{date}_v{version}_e{threshold}.xlsx
```

**Example:**
```
Rhode_Island_climbs_all_basic_2025-11-17_v2.0.1_e6000.xlsx
```

**File Contents:**

Each row represents one climb with columns:
- **Street Name**: Road/trail name
- **City**: Nearest city
- **State/Country**: Administrative region
- **From Center (mi)**: Distance from region center
- **Cycling**: Cycling allowed? (Yes/No)
- **Category**: Cycling category (HC, Cat1-4)
- **Basic Score**: Distance × Average Grade
- **FIETS Score**: Dutch climbing score
- **PDI Score**: PJAMM Difficulty Index
- **Elev Gain (ft)**: Total elevation gained
- **Height (ft)**: Highest point
- **Prominence (ft)**: Height from start to summit
- **Length (mi)**: Total climb length
- **Avg Grade (%)**: Average gradient
- **Max Grade (%)**: Maximum gradient
- **Highway Type**: Road classification (primary, secondary, etc.)
- **Surface**: Pavement type (paved, gravel, dirt, etc.)
- **Tracktype**: Trail type (singletrack, doubletrack, etc.)
- **Way ID**: OpenStreetMap identifier
- **OSM Link**: Clickable link to OpenStreetMap
- **Connected Climbs**: Adjacent climbs that continue upward

---

## 🌐 Web GUI Setup

The web GUI provides a modern interface for managing analyses, visualizing climbs on maps, and downloading data.

### Installation

The web GUI is installed automatically during setup wizard, or you can install it manually:

```bash
# Inside Docker container
./climb-analyzer shell
cd gui/
npm install
exit
```

### Launching the GUI

```bash
# Start GUI in background (port 3000)
./climb-analyzer -g

# GUI is now accessible at: http://localhost:3000

# Stop GUI server
./climb-analyzer --stop-gui

# Check GUI server status
./climb-analyzer --gui-status
```

### GUI Features

**Dashboard:**
- System status overview
- Recent analyses
- Data coverage summary
- Quick action buttons

**Analyze:**
- Region selection interface
- Configuration options
- Real-time progress tracking
- Analysis history

**Visualize:**
- Interactive map with all climbs
- Filter by score, grade, length
- Elevation profile viewer
- Export climb data

**Download:**
- OSM data downloader with region search
- Elevation data downloader by dataset
- Download progress tracking

**Config:**
- Edit `config.yaml` in web interface
- Validate configuration
- View current settings

**Manage:**
- View disk usage by data type
- Clean up old checkpoints
- Delete unused OSM/elevation data
- View detailed file listings

See [GUI_SETUP.md](GUI_SETUP.md) for detailed GUI documentation.

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Docker Permission Errors

**Problem**: `Permission denied` errors when accessing files, or files owned by `root` instead of your user.

**Solution**: Ensure `HOST_UID` and `HOST_GID` in `.env` match your user:

```bash
# Check your UID and GID
id -u  # UID (usually 501 on macOS, 1000 on Linux)
id -g  # GID (usually 20 on macOS, 1000 on Linux)

# Update .env file
echo "HOST_UID=$(id -u)" > .env
echo "HOST_GID=$(id -g)" >> .env

# Rebuild containers
./climb-analyzer build
docker compose up -d
```

#### 2. Memory Errors / OOM (Out of Memory)

**Problem**: Analysis fails with `MemoryError` or container is killed by OOM killer.

**Causes**:
- Bloom filter capacity too small for large regions
- Insufficient Docker memory allocation
- Region too large for available RAM

**Solutions**:

```bash
# Check memory configuration in config.yaml
cat config.yaml | grep -A 5 "ELEVATION"

# For large regions (France, California), ensure bloom filter capacity is adequate
# Edit climb_analyzer.py if needed (capacity should be 150M+ for very large regions)

# Increase Docker memory allocation (Docker Desktop settings)
# Recommended: 8 GB minimum, 16 GB for large regions

# Use checkpointing to resume if analysis fails partway
./climb-analyzer -r "Large Region" --resume
```

See [.claude/bugs.md](.claude/bugs.md) for detailed memory troubleshooting.

#### 3. OpenTopoData Connection Errors

**Problem**: `Connection refused` to `opentopodata-server:5000` in LOCAL mode.

**Solution**:

```bash
# Check if OpenTopoData container is running
docker ps | grep opentopodata

# If not running, start it
docker compose up -d opentopodata-server

# Check container logs
docker logs opentopodata-server

# Rebuild OpenTopoData container (if needed)
cd opentopodata/
docker build -t opentopodata:latest .
docker compose up -d opentopodata-server
```

See [docs/DOCKER_IN_DOCKER_FIX.md](docs/DOCKER_IN_DOCKER_FIX.md) for advanced Docker troubleshooting.

#### 4. NASA Earthdata 401 Unauthorized

**Problem**: Elevation data downloads fail with `401 Unauthorized`.

**Solution**:

```bash
# Verify credentials are correct
cat .credentials/netrc

# Expected format:
# machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD

# If incorrect, update:
echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" > .credentials/netrc
chmod 600 .credentials/netrc

# Test credentials
curl -n -L https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/ | head
# Should show directory listing, not 401 error
```

#### 5. OSM Data Download Fails

**Problem**: OSM .pbf file download fails or is corrupt.

**Solution**:

```bash
# Delete incomplete/corrupt file
rm data/planet_osm_data/region-name-latest.osm.pbf

# Re-download using interactive mode
./climb-analyzer
# Follow prompts to download OSM data

# Or download manually from Geofabrik
# https://download.geofabrik.de/
wget https://download.geofabrik.de/north-america/us/rhode-island-latest.osm.pbf -O data/planet_osm_data/rhode-island-latest.osm.pbf
```

#### 6. Web GUI Won't Start

**Problem**: `./climb-analyzer -g` fails or GUI is not accessible.

**Solution**:

```bash
# Check if port 3000 is already in use
lsof -i :3000  # macOS/Linux

# Kill process using port 3000 (if needed)
kill $(lsof -t -i :3000)

# Check GUI server logs
cat gui/gui-server.log

# Reinstall GUI dependencies
./climb-analyzer shell
cd gui/
rm -rf node_modules/ .next/
npm install
npm run build
exit

# Restart GUI
./climb-analyzer -g
```

#### 7. Analysis Stuck at 0% or Very Slow

**Problem**: Analysis appears to hang or progresses extremely slowly.

**Possible Causes & Solutions**:

1. **Cloud Mode with Large Region**: Overpass API is rate-limited. Switch to LOCAL mode:
   ```bash
   # Edit config.yaml
   DEPLOYMENT_TYPE: local

   # Download OSM data first
   ./climb-analyzer
   # Select option to download OSM data
   ```

2. **Elevation Data Not Available**: Missing DEM tiles. Check configuration:
   ```bash
   # Verify elevation datasets are configured
   cat config.yaml | grep ELEVATION_DATASET_TIERS

   # Ensure: primary+secondary+tertiary
   ```

3. **Network Issues**: Check internet connection and firewall settings.

4. **Checkpoint Corruption**: Delete checkpoint and restart:
   ```bash
   rm -rf data/checkpoint_data/Region_Name_*
   ./climb-analyzer -r "Region Name"
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check Existing Documentation**:
   - [docs/README.md](docs/README.md) - Documentation index
   - [docs/ELEVATION_SYSTEM_SUMMARY.md](docs/ELEVATION_SYSTEM_SUMMARY.md)
   - [docs/LOCAL_MODE_REGIONAL_EXTRACTION.md](docs/LOCAL_MODE_REGIONAL_EXTRACTION.md)
   - [docs/CLOUD_MODE_REGIONAL_EXTRACTION.md](docs/CLOUD_MODE_REGIONAL_EXTRACTION.md)

2. **Enable Verbose Logging**:
   ```bash
   ./climb-analyzer -r "Region" -v 2>&1 | tee analysis.log
   ```

3. **Check Docker Logs**:
   ```bash
   docker logs climb-analyzer
   docker logs opentopodata-server
   ```

4. **Report Issues**:
   - GitHub Issues: https://github.com/stevehollx/climb-analyzer/issues
   - Include: OS, Docker version, `config.yaml` (redact credentials), error logs

---

## 🔄 Upgrading

### Docker Installation

```bash
cd /path/to/climb-analyzer

# Pull latest code
git pull origin main

# Rebuild Docker image (if Dockerfile changed)
./climb-analyzer build

# Restart containers
docker compose down
docker compose up -d

# Verify version
./climb-analyzer --version
```

**Note**: Existing data (`data/`, `output/`, `config.yaml`) is preserved during upgrades.

### Local Python Installation

```bash
cd /path/to/climb-analyzer

# Pull latest code
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Reinstall package
pip install -e ".[all]" --upgrade

# Verify version
climb-analyzer --version
```

### Breaking Changes

Check [CHANGELOG.md](CHANGELOG.md) for breaking changes between versions. Common upgrade tasks:

- **v1.x → v2.x**: Configuration format changed. Run `./climb-analyzer setup` to regenerate `config.yaml`
- **Bloom filter sizing**: If upgrading from pre-bloom filter version, delete old checkpoints for large regions

---

## 🗑️ Uninstallation

### Docker Installation

```bash
cd /path/to/climb-analyzer

# Stop and remove containers
docker compose down -v

# Remove Docker images
docker rmi climb-analyzer:latest
docker rmi opentopodata:latest

# Remove application directory (WARNING: deletes all data!)
cd ..
rm -rf climb-analyzer/

# Or, keep data but remove code:
cd climb-analyzer/
rm -rf climb_analyzer/ gui/ utils/ docs/ scripts/
# Keep: data/, output/, config.yaml
```

### Local Python Installation

```bash
cd /path/to/climb-analyzer

# Deactivate virtual environment
deactivate

# Uninstall Python package
pip uninstall climb-analyzer

# Remove virtual environment
rm -rf venv/

# Remove application directory (WARNING: deletes all data!)
cd ..
rm -rf climb-analyzer/
```

### Partial Cleanup (Free Disk Space)

To free disk space without removing everything:

```bash
# Remove elevation data (largest - 10-350 GB per region)
rm -rf data/elevation_data/

# Remove OSM data (medium - 150 MB to 5 GB per region)
rm -rf data/planet_osm_data/

# Remove checkpoints (medium - 1-10 GB)
rm -rf data/checkpoint_data/

# Remove Docker build cache
docker system prune -a

# Keep: output files, configuration, application code
```

---

## 📚 Additional Resources

### Documentation
- [README.md](README.md) - Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [docs/](docs/) - Technical documentation
  - [CLI_ARGUMENTS_SPEC.md](docs/CLI_ARGUMENTS_SPEC.md) - Complete CLI reference
  - [ELEVATION_SYSTEM_SUMMARY.md](docs/ELEVATION_SYSTEM_SUMMARY.md) - Elevation data architecture
  - [CHECKPOINTING_ADDED.md](docs/CHECKPOINTING_ADDED.md) - Checkpoint system documentation
  - [CLOUD_CACHE.md](docs/CLOUD_CACHE.md) - Cloud cache setup

### GUI Documentation
- [GUI_SETUP.md](GUI_SETUP.md) - Web interface guide
- [gui/README.md](gui/README.md) - GUI architecture and development

### Community
- **Repository**: https://github.com/stevehollx/climb-analyzer
- **Issues**: https://github.com/stevehollx/climb-analyzer/issues
- **Discussions**: https://github.com/stevehollx/climb-analyzer/discussions

### Related Projects
- **Global Climbs Database**: https://github.com/stevehollx/global-road-and-trail-climbs
- **OpenTopoData**: https://github.com/ajnisbet/opentopodata
- **OpenStreetMap**: https://www.openstreetmap.org

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Need Help?** Open an issue on GitHub or consult the [docs/](docs/) directory for detailed technical documentation.

Happy climbing! 🚴⛰️
