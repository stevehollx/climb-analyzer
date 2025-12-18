import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration loader and manager for YAML config files"""
    
    def __init__(self, config_file: str = 'config.yaml'):
        self.config_file = Path(config_file)
        self._config_data = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        
        try:
            with open(self.config_file, 'r') as f:
                self._config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __getattr__(self, name: str):
        """Allow direct attribute access to config values"""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        # Convert attribute name to config key (handle common patterns)
        if name.upper() in self._config_data:
            return self._config_data[name.upper()]
        elif name in self._config_data:
            return self._config_data[name]
        else:
            raise AttributeError(f"Configuration key '{name}' not found")

# Global config instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config

# Compatibility layer - expose config values as module-level constants
def _load_config_constants():
    """Load config values as module constants for backward compatibility"""
    import os
    config = get_config()

    # Get deployment type first (check env var, then config)
    deployment_type = os.environ.get('DEPLOYMENT_TYPE') or config.get('DEPLOYMENT_TYPE', 'cloud')

    # Set elevation concurrency based on deployment type
    # Cloud mode: 1 thread (avoid overwhelming public elevation API)
    # Local mode: Use config value (default 2, or higher for local elevation servers)
    if deployment_type == 'cloud':
        elevation_max_concurrent = 1
    else:
        elevation_max_concurrent = config.get('ELEVATION_MAX_CONCURRENT', 2)

    globals().update({
        'DEPLOYMENT_TYPE': deployment_type,
        'OVERPASS_API_URL': os.environ.get('OVERPASS_API_URL') or config.get('OVERPASS_API_URL'),
        'OVERPASS_API_DELAY_SEC': config.get('OVERPASS_API_DELAY_SEC', 0.1),
        'CLOUD_MODE_MAX_RADIUS_KM': config.get('CLOUD_MODE_MAX_RADIUS_KM', 40.0),
        'CLOUD_MODE_MAX_RADIUS_MILES': config.get('CLOUD_MODE_MAX_RADIUS_MILES', 25.0),
        'TOPO_API_BASE_URL': os.environ.get('TOPO_API_BASE_URL') or config.get('TOPO_API_BASE_URL'),
        'ELEVATION_MAX_CONCURRENT': elevation_max_concurrent,
        'ELEVATION_BATCH_SIZE': config.get('ELEVATION_BATCH_SIZE', 100),
        'ELEVATION_REQUEST_TIMEOUT_SEC': config.get('ELEVATION_REQUEST_TIMEOUT_SEC', 45),
        'ELEVATION_MAX_RETRIES': config.get('ELEVATION_MAX_RETRIES', 3),
        'ELEVATION_BACKOFF_FACTOR': config.get('ELEVATION_BACKOFF_FACTOR', 2.0),
        'CHECKPOINT_INTERVAL_MIN': config.get('CHECKPOINT_INTERVAL_MIN', 15.0),
        'CHECKPOINT_MILESTONES_PERC': config.get('CHECKPOINT_MILESTONES_PERC', [25, 50, 75, 100]),
        'GEOCODING_MAX_CONCURRENT': config.get('GEOCODING_MAX_CONCURRENT', 8),
        'GEOCODING_RETRY_ATTEMPTS': config.get('GEOCODING_RETRY_ATTEMPTS', 2),
        'CLOUD_CACHE_ENABLED': config.get('CLOUD_CACHE_ENABLED', True),
        'CLOUD_CACHE_REPO': config.get('CLOUD_CACHE_REPO', 'stevehollx/global-road-and-trail-climbs'),
        'ENABLE_CROSS_CHUNK_POSTPROCESS': config.get('ENABLE_CROSS_CHUNK_POSTPROCESS', True),
    })

# Load constants when module is imported
try:
    _load_config_constants()
except FileNotFoundError:
    # Config file doesn't exist - will be handled by main application
    pass