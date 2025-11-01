import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point
from shapely.ops import unary_union
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import polars as pl
from natsort import natsorted

## Find parquet files
data_dir = Path("flood_data")
assert data_dir.exists(), f"Data directory {data_dir} does not exist."

parquet_files = natsorted(list(data_dir.rglob("*-post-processing.parquet")))
print(f"Found {len(parquet_files)} parquet files.")

## ---- Configuration ----
output_dir = data_dir / "monthly_flood_maps"
output_dir.mkdir(parents = True, exist_ok = True)
years = natsorted(list(range(2015, 2025)))
buffer_meters = 240  # Use 240m to buffer flood maps

def generate_monthly_flood_maps(parquet_path, year, buffer_meters = 240, output_dir = output_dir):
    
    print(f"\n\nProcessing {parquet_path.name} for year {year}...")

    # Load template metadata from corresponding 240m buffer tif
    template_geotiff_path = parquet_path.parent / f"{buffer_meters}m-buffer.tif"
    assert template_geotiff_path.exists(), f"Template GeoTIFF {template_geotiff_path} does not exist. Only 240m or 80m buffer tif files are available."
    print(f"Using template GeoTIFF: {template_geotiff_path.name}")

    with rasterio.open(template_geotiff_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        bounds = src.bounds
        template_crs = src.crs
        exclusion_mask = src.read(1) == 1  # Exclusion mask pixels


    # Load parquet data using polars for efficiency
    df = pl.scan_parquet(parquet_path)
        
    for month in range(1, 13):
        print(f"\nGenerating flood map for {year}-{month:02d}...")
        # Filter data for the specific year and month
        expr = (
            (pl.col('year') == year) &
            (pl.col('month') == month) &
            (pl.col('dem_metric_2') < 10) &
            (pl.col('soil_moisture_sca') > 1) &
            (pl.col('soil_moisture_zscore') > 1) &
            (pl.col('soil_moisture') > 20) &
            (pl.col('temp') > 0) &
            (pl.col('land_cover') != 60) &
            (pl.col('edge_false_positives') == 0)     
        )
        monthly_df = df.filter(expr).collect()
        if monthly_df.shape[0] == 0:
            print(f"No flood points for {year}-{month:02d}. ")
            monthly_flood_map = np.zeros((height, width), dtype = 'uint8')

        else:
            # Convert to GeoDataFrame
            print("Creating GeoDataFrame from filtered rows")
            monthly_df = monthly_df.to_pandas()
            geometry = [Point(xy) for xy in zip(monthly_df['lon'], monthly_df['lat'])]
            gdf = gpd.GeoDataFrame(monthly_df, geometry = geometry, crs = "EPSG:4326")
            print(f"Number of flood points: {len(gdf)}")
            monthly_df = None
            geometry = None

            # Buffer points if required
            if buffer_meters and buffer_meters > 0:
                print(f"Buffering flood points by {buffer_meters} meters...")
                # Choose a CRS for buffering
                if template_crs.is_geographic:
                    print("Template CRS is geographic. Reprojecting gdf to EPSG:3857 for buffering...")
                    buf_crs = "EPSG:3857"
                else:
                    print("Template CRS is projected. Using template CRS for buffering...")
                    buf_crs = template_crs
                
                # Reproject gdf to buffer CRS
                print(f"Reprojecting gdf to {buf_crs} for buffering...")
                gdf = gdf.to_crs(buf_crs) # Units in meters

                # Buffer in meters
                print(f"Buffering points by {buffer_meters} meters...")
                gdf['geom_buf'] = gdf.geometry.buffer(buffer_meters)

                # Merge buffered geometries
                union_geom = unary_union(gdf['geom_buf'])

                # Reproject back to template CRS
                print(f"Reprojecting back to {template_crs}...")
                union_geom = gpd.GeoSeries([union_geom], crs = buf_crs).to_crs(template_crs).iloc[0]

            else:
                print("No buffering: rasterizing single-point blobs")
                gdf = gdf.to_crs(template_crs)
                px = abs(transform.a)
                half_px = px / 2.0
                union_geom = [(pt.buffer(half_px)) for pt in gdf.geometry]

                # print("No buffering applied.")
                # gdf = gdf.to_crs(template_crs)
                # union_geom = unary_union(gdf.geometry)


            # Rasterize the unioned geometry
            print("Rasterizing buffered geometries...")
            shapes = [(union_geom)]  
            monthly_flood_map = rasterize(
                shapes,
                out_shape = (height, width),
                transform = transform,
                fill = 0, # Value 0 for non-flooded areas
                default_value = 2, # Value 2 for flooded areas
                all_touched = True,
                dtype = 'uint8'
            )
        
        # Apply exclusion mask
        if exclusion_mask is not None:
            print("Applying exclusion mask...")
            monthly_flood_map = np.where(exclusion_mask, 1, monthly_flood_map)

        # Save to GeoTIFF
        output_path = output_dir / f"{year}" / f"{year}-{month:02d}_{parquet_path.parent.stem}_flood_map.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving flood map to {output_path}...")
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'uint8',
            'crs': template_crs,
            'transform': transform,
            'compress': 'lzw'
        }
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(monthly_flood_map, 1)

        print(f"Flood pixels percent: {np.sum(monthly_flood_map == 2) / monthly_flood_map.size * 100:.4f}%")
        print(f"Completed {year}-{month:02d}.\n")
        print("--"*40)

        



        
        


           



                

               