# Climb Analyzer - Main Application Container
FROM python:3.11-slim

# Install system dependencies for local OSM processing
RUN apt-get update && apt-get install -y \
    # Core build tools
    build-essential \
    cmake \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    # OSM processing libraries
    libosmium-dev \
    osmium-tool \
    libspatialindex-dev \
    # Geographic data processing (for geopandas)
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    # Additional utilities
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install Node.js 18.x for web GUI
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Docker CLI (for controlling host Docker via socket)
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code first (needed for editable install)
COPY . .

# Install Python dependencies from pyproject.toml
# Install with 'all' extra to get both local and cloud mode dependencies
# Note: Must be after COPY so editable install can find the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[all]"

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/planet_osm_data \
    data/osm_indexes \
    data/elevation_data \
    output \
    .credentials && \
    chmod -R 777 output .credentials

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEPLOYMENT_TYPE=local

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["python", "climb_analyzer_main.py"]
