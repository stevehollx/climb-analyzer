#!/bin/bash
echo "Testing coordinate transformation for Atlanta query..."
docker exec opentopodata-server python3 -c "
from opentopodata.config import load_datasets
from opentopodata import utils
from decimal import Decimal
import numpy as np

datasets = load_datasets()
ned = datasets.get('ned10m')

# Atlanta coordinates
lat, lon = 33.7490, -84.3880
print(f'Query: lat={lat}, lon={lon}')
print()

# What OpenTopoData does internally:
# 1. Convert to filename projection (EPSG:4269)
xs, ys = utils.reproject_latlons(
    np.array([lat]),
    np.array([lon]),
    epsg=ned.filename_epsg
)
print(f'After reproject to EPSG:{ned.filename_epsg}:')
print(f'  x={xs[0]}, y={ys[0]}')
print()

# 2. Floor to tile corner
tile_size = ned.filename_tile_size
northing = utils.decimal_base_floor(ys[0], tile_size)
easting = utils.decimal_base_floor(xs[0], tile_size)
print(f'After flooring with tile_size={tile_size}:')
print(f'  northing={northing}, easting={easting}')
print()

# 3. Look up in tile lookup
key = (northing, easting)
print(f'Lookup key: {key}')
tile_path = ned._tile_lookup.get(key)
print(f'Found tile: {tile_path.split(\"/\")[-1] if tile_path else \"NOT FOUND\"}')
print()

# 4. If found, try to read from the file
if tile_path:
    import rasterio
    from rasterio.warp import transform

    try:
        with rasterio.open(tile_path) as src:
            print(f'Opening: {tile_path.split(\"/\")[-1]}')
            print(f'File CRS: {src.crs}')
            print(f'File bounds: {src.bounds}')

            # Transform query coords to file CRS
            file_xs, file_ys = transform('EPSG:4326', src.crs, [lon], [lat])
            print(f'Query coords in file CRS: x={file_xs[0]:.6f}, y={file_ys[0]:.6f}')

            # Check if in bounds
            in_bounds = (src.bounds.left <= file_xs[0] <= src.bounds.right and
                        src.bounds.bottom <= file_ys[0] <= src.bounds.top)
            print(f'In bounds: {in_bounds}')

            if in_bounds:
                # Try to sample the elevation
                row, col = src.index(file_xs[0], file_ys[0])
                print(f'Pixel position: row={row}, col={col}')
                value = src.read(1, window=((row, row+1), (col, col+1)))
                print(f'Elevation value: {value[0, 0]}')

    except Exception as e:
        print(f'Error reading file: {e}')
        import traceback
        traceback.print_exc()
"
