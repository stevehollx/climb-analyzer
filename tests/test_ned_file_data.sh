#!/bin/bash
echo "Testing NED file data quality..."
docker exec opentopodata-server python3 -c "
import rasterio
import numpy as np

# Test the n33w085 file (should contain Atlanta)
filepath = '/app/data/ned10m/USGS_13_n33w085.tif'

with rasterio.open(filepath) as src:
    print(f'File: {filepath.split(\"/\")[-1]}')
    print(f'CRS: {src.crs}')
    print(f'Bounds: {src.bounds}')
    print(f'Size: {src.width} x {src.height}')
    print(f'Data type: {src.dtypes[0]}')
    print(f'NoData value: {src.nodata}')
    print()

    # Read a sample of data
    data = src.read(1)
    print(f'Data shape: {data.shape}')
    print(f'Data min: {np.nanmin(data)}')
    print(f'Data max: {np.nanmax(data)}')
    print(f'Data mean: {np.nanmean(data):.2f}')
    print(f'Has NaN: {np.any(np.isnan(data))}')
    print(f'Has NoData ({src.nodata}): {np.any(data == src.nodata) if src.nodata else \"N/A\"}')
    print()

    # Sample a few random points
    print('Sample elevations from different parts of the file:')
    for i, (row, col) in enumerate([(100, 100), (5000, 5000), (10000, 10000)]):
        if row < src.height and col < src.width:
            val = data[row, col]
            print(f'  Position ({row}, {col}): {val}')
"
