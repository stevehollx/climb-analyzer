#!/bin/bash

for file in *.tif; do
    gdal_translate -co COMPRESS=LZW -co TILED=YES -co PREDICTOR=2 "$file" "compressed_${file}"
done
