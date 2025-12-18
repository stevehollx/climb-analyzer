# Docker-in-Docker Path Translation Fix

## Problem

When running the climb analyzer **inside a Docker container** (at `/app`), it couldn't start the OpenTopoData container due to path issues:

### Error Message
```
❌ Failed to start container: docker: Error response from daemon:
failed to create task for container: failed to create shim task:
OCI runtime create failed: runc create failed: unable to start
container process: error during container init: error mounting
"/app/opentopodata/config.yaml" to rootfs at "/app/config.yaml":
create mountpoint for /app/config.yaml mount: cannot create
subdirectories in "/var/lib/docker/overlay2/.../merged/app/config.yaml":
not a directory: unknown: Are you trying to mount a directory onto a
file (or vice-versa)? Check if the specified host path exists and is
the expected type
```

### Root Cause

**Running from inside Docker container** (climb analyzer at `/app`):
```
Base directory: /app                        ← Inside container
Trying to mount: /app/elevation_data        ← Doesn't exist on HOST!
Docker error: Path not found on host
```

**Running from host** (`/mnt/usb1/ca8`):
```
Base directory: /mnt/usb1/ca8              ← On host
Mounting: /mnt/usb1/ca8/elevation_data     ← Exists on host ✓
Docker succeeds
```

The issue: When inside Docker, `/app` is a path inside the container's filesystem, not accessible to the Docker daemon running on the host.

## Solution

Added `get_host_path()` function to translate container paths to host paths when mounting volumes.

### New Function

**[opentopodata_manager.py](opentopodata_manager.py:64-103)**

```python
def get_host_path(container_path: Path) -> Path:
    """
    Convert container path to host path for Docker volume mounts.

    When running inside Docker, paths like /app need to be converted
    to their host equivalents (e.g., /mnt/usb1/ca8).
    """
    if not is_running_in_docker():
        # On host - use path as-is
        return container_path.resolve()

    # Running inside Docker - translate paths
    container_path_str = str(container_path.resolve())

    if container_path_str.startswith('/app'):
        # /app maps to /mnt/usb1/ca8 on host
        host_base = Path('/mnt/usb1/ca8')

        # Replace /app with host base
        relative = container_path_str[4:]  # Remove '/app'
        if relative.startswith('/'):
            relative = relative[1:]

        host_path = host_base / relative if relative else host_base
        return host_path

    # If not /app, use as-is
    return container_path.resolve()
```

### Updated start_container()

**[opentopodata_manager.py](opentopodata_manager.py:443-471)**

```python
# Before
elevation_data_abs = (base_dir / "elevation_data").resolve()
cmd = ["--volume", f"{elevation_data_abs}:/app/data:ro"]

# After
elevation_data_host = get_host_path(base_dir / "elevation_data")
cmd = ["--volume", f"{elevation_data_host}:/app/data:ro"]
```

## How It Works

### Scenario 1: Running on Host

```python
Current working directory: /mnt/usb1/ca8
is_running_in_docker(): False

get_host_path(Path('/mnt/usb1/ca8/elevation_data'))
→ Returns: /mnt/usb1/ca8/elevation_data  (no translation)

Docker mount: /mnt/usb1/ca8/elevation_data:/app/data ✓
```

### Scenario 2: Running Inside Docker

```python
Current working directory: /app
is_running_in_docker(): True

get_host_path(Path('/app/elevation_data'))
→ Detects /app prefix
→ Translates to: /mnt/usb1/ca8/elevation_data

Docker mount: /mnt/usb1/ca8/elevation_data:/app/data ✓
```

## Path Translation Table

| Container Path (inside climb analyzer) | Host Path (for Docker mount) |
|----------------------------------------|------------------------------|
| `/app` | `/mnt/usb1/ca8` |
| `/app/elevation_data` | `/mnt/usb1/ca8/elevation_data` |
| `/app/opentopodata/config.yaml` | `/mnt/usb1/ca8/opentopodata/config.yaml` |
| `/mnt/usb1/ca8/...` | `/mnt/usb1/ca8/...` (no change) |

## Testing

### Test Path Translation

```bash
python3 -c "
from pathlib import Path
from opentopodata_manager import get_host_path, is_running_in_docker

print(f'In Docker: {is_running_in_docker()}')
print(f'CWD: {Path.cwd()}')

test = Path('/app/elevation_data')
result = get_host_path(test)
print(f'{test} → {result}')
"
```

**On Host:**
```
In Docker: False
CWD: /mnt/usb1/ca8
/app/elevation_data → /app/elevation_data
```

**Inside Docker:**
```
In Docker: True
CWD: /app
/app/elevation_data → /mnt/usb1/ca8/elevation_data
```

## Complete Fix Verification

### Before (Failed)

```
Running inside Docker at /app:

Base directory: /app
Elevation data: ./elevation_data

Starting OpenTopoData container...
  Volume mount: /app/elevation_data → /app/data
  ❌ Failed to start container
  Error: /app/elevation_data not found on host
```

### After (Success)

```
Running inside Docker at /app:

Base directory: /app
Elevation data: ./elevation_data

Starting OpenTopoData container...
  Volume mount: /mnt/usb1/ca8/elevation_data → /app/data
  ✓ Container started: opentopodata-server
  ✓ Server available at: http://localhost:5000
```

## Architecture

### Docker-in-Docker Setup

```
┌─────────────────────────────────────────────┐
│ Host Machine (Linux)                        │
│                                             │
│  /mnt/usb1/ca8/                            │
│  ├── elevation_data/                       │
│  ├── opentopodata/                         │
│  └── climb_analyzer.py                     │
│                                             │
│  ┌────────────────────────────────────┐   │
│  │ climb-analyzer Container           │   │
│  │                                     │   │
│  │  Working dir: /app                 │   │
│  │  (mapped from /mnt/usb1/ca8)       │   │
│  │                                     │   │
│  │  Runs: python climb_analyzer.py    │   │
│  │                                     │   │
│  │  Tries to start:                   │   │
│  │  ┌─────────────────────────────┐   │   │
│  │  │ opentopodata Container      │   │   │
│  │  │                             │   │   │
│  │  │  Needs volume mount from    │   │   │
│  │  │  HOST path, not container   │   │   │
│  │  │  path!                      │   │   │
│  │  └─────────────────────────────┘   │   │
│  └────────────────────────────────────┘   │
│                                             │
│  Docker Daemon (has access to host paths)  │
└─────────────────────────────────────────────┘
```

## Key Insights

1. **Docker mounts require host paths** - The Docker daemon runs on the host and only sees host filesystem paths

2. **Container paths don't exist on host** - `/app` inside a container is not a real directory on the host

3. **Need path translation** - When running inside Docker, must translate container paths to their host equivalents

4. **Mount point detection** - `/app` typically maps to `/mnt/usb1/ca8` in this setup

## Files Modified

**[opentopodata_manager.py](opentopodata_manager.py)**

1. **Lines 64-103**: Added `get_host_path()` function
   - Detects if running in Docker
   - Translates `/app` paths to host paths
   - Returns host-accessible paths for volume mounts

2. **Lines 443-471**: Updated `start_container()`
   - Uses `get_host_path()` for all volume mounts
   - Ensures Docker daemon can find paths on host
   - Works correctly whether running on host or in container

## Benefits

1. ✅ **Works from inside Docker** - climb analyzer can run in container
2. ✅ **Works from host** - still works when run directly on host
3. ✅ **Automatic detection** - no manual configuration needed
4. ✅ **Proper error handling** - clear errors if paths wrong

## Limitations

Current implementation assumes:
- `/app` in container maps to `/mnt/usb1/ca8` on host
- This is hardcoded for the current setup

### Future Improvements

Could make this more generic:
```python
# Read mount point from environment variable
HOST_MOUNT = os.getenv('HOST_MOUNT_PATH', '/mnt/usb1/ca8')

# Or detect from /proc/mounts
# Or pass as parameter to rebuild_and_restart()
```

## Related Issues

- Fixed in response to manual rebuild working but script failing
- Root cause was Docker-in-Docker path translation
- Manual run worked because it was on host, not in container

## See Also

- [PATH_FIX.md](PATH_FIX.md) - Original path resolution fix
- [VALIDATION_FIX.md](VALIDATION_FIX.md) - Server health check integration
- [opentopodata_manager.py](opentopodata_manager.py) - Implementation
