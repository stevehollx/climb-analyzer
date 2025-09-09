# Downoads the National Elevation Dataset (NED) 10m topo data for United States

import os
import requests

# Base URL format
base_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"

# Input file with lines like: s14w171.tif
input_file = "ned10m_urls.txt"


# Make sure download folder exists
os.makedirs(download_folder, exist_ok=True)

with open(input_file, "r") as f_in:
    for line in f_in:
        part = line.strip()
        if not part:
            continue

        # remove .tif if present for folder naming
        tile_name = part.replace(".tif", "")
        url = f"{base_url}/{tile_name}/USGS_13_{tile_name}.tif"
        local_path = os.path.join(f"USGS_13_{tile_name}.tif")

        print(f"Downloading {url} ...")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(local_path, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)

            print(f"Saved to {local_path}")

        except requests.HTTPError as e:
            print(f"Failed to download {url}: {e}")
        except requests.RequestException as e:
            print(f"Network error for {url}: {e}")