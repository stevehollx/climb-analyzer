#!/bin/bash
echo "Testing n34w085 tile for Atlanta..."
docker exec opentopodata-server python3 -c "
import rasterio
from rasterio.warp import transform

filepath = '/app/data/ned10m/USGS_13_n34w085.tif'
lat, lon = 33.7490, -84.3880

with rasterio.open(filepath) as src:
    print(f'File: {filepath.split(\"/\")[-1]}')
    print(f'Bounds: {src.bounds}')
    print()

    # Transform Atlanta coords
    file_xs, file_ys = transform('EPSG:4326', src.crs, [lon], [lat])
    print(f'Atlanta coords: {lat}, {lon}')
    print(f'Transformed: x={file_xs[0]:.6f}, y={file_ys[0]:.6f}')

    # Check bounds
    in_bounds = (src.bounds.left <= file_xs[0] <= src.bounds.right and
                src.bounds.bottom <= file_ys[0] <= src.bounds.top)
    print(f'In bounds: {in_bounds}')

    if in_bounds:
        # Read elevation
        row, col = src.index(file_xs[0], file_ys[0])
        value = src.read(1, window=((row, row+1), (col, col+1)))
        print(f'Elevation: {value[0, 0]} meters')
"
