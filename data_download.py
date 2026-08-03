#!/usr/bin/env python
# coding: utf-8

# ## Downloading flood conditioning variables

# ## Elevation, slope

# In[2]:


import ee
import numpy as np
import time
from pathlib import Path
import pandas as pd
from natsort import natsorted


# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

def tile_name_from_latlon(lat, lon):

    """
    Example usage:
    Nairobi coordinates: 0.0236° S, 37.9062° E
    print(f"Nairobi tile: {tile_name_from_latlon(lat = 1.29, lon = 36.82)}")
    """

    # Data is in 3 x 3 degree tiles so

    lat_tile = (lat // 3) * 3
    lon_tile = (lon // 3) * 3
    lat_tile, lon_tile = int(lat_tile), int(lon_tile)

    lat_prefix = 'N' if lat_tile >= 0 else 'S'
    lon_prefix = 'E' if lon_tile >= 0 else 'W' 

    return f"{lat_prefix}{abs(lat_tile):02d}{lon_prefix}{abs(lon_tile):03d}"

def get_country_bounds(country_name):
    """
    Returns the bounding box of a country as [min_lon, min_lat, max_lon, max_lat]
    """
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    country = countries.filter(ee.Filter.eq('country_na', country_name))
    bounds = np.array(country.geometry().bounds().getInfo()['coordinates'][0])

    # Get min and max latitudes and longitudes
    min_lon, max_lon = bounds[:, 0].min(), bounds[:, 0].max()
    min_lat, max_lat = bounds[:, 1].min(), bounds[:, 1].max()
    return [min_lon, min_lat, max_lon, max_lat]

def generate_tiles_from_bounds(bounds, tile_size=3):
    """
    Generates a list of tile names that cover the given bounding box.

    Returns:
    A list of tuples, where each tuple contains the bounding box of a tile in the format (min_lon, min_lat, max_lon, max_lat).
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    # Align the bounds to the tile grid
    start_lon = (min_lon // tile_size) * tile_size
    end_lon = (max_lon // tile_size) * tile_size
    start_lat = (min_lat // tile_size) * tile_size
    end_lat = (max_lat // tile_size) * tile_size

    tiles = []
    for lat in range(int(start_lat), int(end_lat) + tile_size, tile_size):
        for lon in range(int(start_lon), int(end_lon) + tile_size, tile_size):
            tile = (lon, lat, lon + tile_size, lat + tile_size)
            tiles.append(tile)

    return tiles

TASKS_TO_COMPARE = 50
def download_data_for_tile(tile, image, dataset_name, export_params, tasks_to_compare = TASKS_TO_COMPARE):
    """
    Exports data for a given tile and dataset to Google Drive.
    """
    min_lon, min_lat, max_lon, max_lat = tile
    tile_name = tile_name_from_latlon(min_lat, min_lon)


    tile_geom = ee.Geometry.Rectangle(tile)
    export_params['region'] = tile_geom

    # Check if a task with the same description already exists
    tasks = ee.batch.Task.list()

    if tasks_to_compare is not None:
        tasks = tasks[:tasks_to_compare]

    task_exists = any(
    ((t.config.get('description') == f"{tile_name}_{dataset_name}") and
    (t.state in ['READY', 'RUNNING'])) or
    ((t.config.get('description') == f"{tile_name}_{dataset_name}") and
    (t.state == 'COMPLETED'))
    for t in tasks
    )

    if task_exists:
        print(f"Task for {tile_name} and dataset {dataset_name} already exists. Skipping export.")
        return True

    print(f"Exporting tile: {tile_name} for dataset: {dataset_name}")
    task = ee.batch.Export.image.toDrive(
        image = image,
        description = f"{tile_name}_{dataset_name}",
        **export_params
    )
    task.start()
    return True

def check_task_status(n = 50):
    tasks = ee.batch.Task.list()
    print(f"{'TASK DESCRIPTION':<30} | {'STATE':<10} | {'ID'}")
    print("-" * 60)
    for task in tasks[:n]:  # Check the first n tasks
        status = task.status()
        description = status.get('description', 'No Description')
        state = status.get('state', 'UNKNOWN')
        task_id = status.get('id')
        print(f"{description:<30} | {state:<10} | {task_id}")



def tile_name_to_coord_bounds(tile_name):
    """Convert a tile name like 'N00E036' back to (min_lon, min_lat, max_lon, max_lat).

    Inverse of tile_name_from_latlon(). Tile names encode the SW corner;
    each tile spans 3° × 3°.
    """
    lat_prefix = tile_name[0]   # 'N' or 'S'
    lat_val = int(tile_name[1:3])  # e.g. '06' → 6
    lon_prefix = tile_name[3] # 'E' or 'W'
    lon_val = int(tile_name[4:7])  # e.g. '033' → 33

    min_lat = lat_val if lat_prefix == 'N' else -lat_val
    min_lon = lon_val if lon_prefix == 'E' else -lon_val

    max_lat = min_lat + 3
    max_lon = min_lon + 3

    return (min_lon, min_lat, max_lon, max_lat)

def load_flood_dates_for_tile(tile_name, base_path = Path("flood_data"), start_date_index = 0, end_date_index  = -1):
    """Load the CSV of extracted flood dates for a tile.

    Returns:
        List of date strings ['2020-04-15', ...] sorted chronologically,
        or empty list if CSV not found.
    """
    root_dir = tile_name[:3]
    csv_path = base_path / root_dir / tile_name / f"{tile_name}-post-processing_flood_dates.csv"

    if not csv_path.exists():
        print(f"⚠️  Flood dates CSV not found for {tile_name}: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    dates = df['date'].tolist()
    dates = natsorted(list(set(dates)))  # Ensure uniqueness and sort
    print(f"  Read {len(dates)} flood dates")

    dates = dates[start_date_index:end_date_index] + dates[end_date_index:]  

    print(f"  Loaded {len(dates)} flood dates for {tile_name} (range: {dates[0]} → {dates[-1]})")
    return dates



# Define export parameters
export_params = {
    # 'scale': 30, # Force 30m resolution for ALL layers
    # 'crs': 'EPSG:4326',
    'maxPixels': 1e13,
    'fileFormat': 'GeoTIFF',
    # 'folder': 'Kenya_Flood_Data_3x3' # Creates this folder in Drive
}





# In[3]:


# Get the bounding box for Kenya
kenya_bounds = get_country_bounds("Kenya")

# Generate 3x3 degree tiles that cover Kenya
kenya_tiles = generate_tiles_from_bounds(kenya_bounds, tile_size=3)

# # Define global datasets
# # DEM: SRTM V3 (30m)
# srtm = ee.Image("USGS/SRTMGL1_003")
# dem_global = srtm.select('elevation')

# # Slope: Derived from SRTM
# slope_global = ee.Terrain.slope(dem_global)


def download_elevation_and_slope(tile, export_params, kwargs = None):
    """
    Exports elevation and slope data for a given tile to Google Drive.
    """
    export_params = dict(export_params)  # Ensure it's a dictionary
    if kwargs is not None:
        export_params.update(kwargs)  # Update with any additional parameters

    # Define global datasets
    # DEM: SRTM V3 (30m)
    srtm = ee.Image("USGS/SRTMGL1_003")
    dem_global = srtm.select('elevation')

    # Slope: Derived from SRTM
    slope_global = ee.Terrain.slope(dem_global)

    # Export DEM
    download_data_for_tile(tile, dem_global, "DEM", export_params)
    # Export Slope
    download_data_for_tile(tile, slope_global, "Slope", export_params)





# # Download DEM and Slope for each tile
# for tile in kenya_tiles:
#     download_elevation_and_slope(tile, export_params, kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.0002777777777777778, 0, -180.0001388888889, 0, -0.0002777777777777778, 60.00013888888889]})
#     time.sleep(10)
#     break

# # Check task status
# check_task_status()


# ## CHIRPS precipitation (daily + rolling aggregation)
# 
# Goal: for each Sentinel-1 acquisition date $t$, export CHIRPS precipitation for that day and/or compute revisit-aware rolling aggregates ending at $t$ (e.g., 1/3/7/14-day sums).

# In[4]:


from datetime import date as date_type, datetime, timedelta

CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"

def _to_ee_date(d):
    if isinstance(d, ee.Date):
        return d

    if isinstance(d, str):
        # Expect YYYY-MM-DD
        return ee.Date(d)

    if isinstance(d, datetime):
        return ee.Date(d.strftime("%Y-%m-%d"))

    if isinstance(d, date_type):
        return ee.Date(d.strftime("%Y-%m-%d"))

    raise TypeError(f"Unsupported date type: {type(d)}")

def _day_to_str(day):
    if isinstance(day, str):
        return day
    if isinstance(day, (datetime, date_type)):
        return day.strftime("%Y-%m-%d")
    # Fallback: client-side fetch. Prefer passing str/datetime to avoid this.
    return _to_ee_date(day).format("YYYY-MM-dd").getInfo()

def chirps_daily_precipitation(day):
    """Return CHIRPS precipitation image for a single UTC day.

    Args:
        day: 'YYYY-MM-DD', datetime/date, or ee.Date.

    Returns:
        ee.Image with band 'precipitation' (mm/day).
    """
    start = _to_ee_date(day)
    end = start.advance(1, 'day')

    img = (
        ee.ImageCollection(CHIRPS_DAILY)
        .filterDate(start, end)
        .select('precipitation')
        .first()
    )

    return ee.Image(img).rename('precipitation').set({"date": start.format("YYYY-MM-dd")})

def chirps_aggregate(end_day, window_days, reducer = "sum"):
    """Aggregate CHIRPS precipitation over multiple days ending at end_day (inclusive).

     Args:
        end_day: 'YYYY-MM-DD', datetime/date, or ee.Date (window ends here, inclusive)
        window_days: int, e.g. 1, 3, 7, 14
        reducer: 'sum' | 'mean' | 'max'

    Returns:
        ee.Image with one band named like 'precip_14d_sum'. Units:
          - sum: mm over window
          - mean: mm/day
          - max: mm/day (max daily value within window)
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    end_og = _to_ee_date(end_day)
    end = _to_ee_date(end_day).advance(1, 'day')  # Advance by 1 day to make end_day inclusive
    start = end.advance(-window_days, 'day')

    ic = (
        ee.ImageCollection(CHIRPS_DAILY)
        .filterDate(start, end)
        .select('precipitation')
    )

    reducer = reducer.lower() # Make reducer lowercase for easier handling

    if reducer == "sum":
        agg = ic.sum()

    elif reducer == "mean":
        agg = ic.mean()

    elif reducer == "max":
        agg = ic.max()

    else:
        raise ValueError(f"Unsupported reducer: {reducer}. Use 'sum', 'mean', or 'max'.")

    band_name = f"precip_{int(window_days)}d_{reducer}"

    return ee.Image(agg).rename(band_name).set({
        "start_date": start.format("YYYY-MM-dd"),
        "end_date": end_og.format("YYYY-MM-dd"),
        "window_days": window_days,
        "reducer": reducer
    })


def download_chirps_precipitation_for_day(tile, day, export_params, dataset_prefix = "CHIRPS_precip", kwargs = None):
    """Export CHIRPS precipitation for a single day for the given 3°×3° tile.

    Notes:
      - We override scale to CHIRPS native (~0.05° ≈ 5.5km). Do NOT export at 30m.
      - Uses the existing download_data_for_tile() helper.

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        day: 'YYYY-MM-DD' or datetime/date
        export_params: dict used by ee.batch.Export.image.toDrive
        scale_m: export scale in meters (default ~5.5km)
        dataset_prefix: name prefix used in Drive description
    """
    # assert scale_m is not None, "Must specify scale_m for CHIRPS export (e.g. 5500 for ~5.5km native resolution)"
    day_str = _day_to_str(day)
    image = chirps_daily_precipitation(day)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)


    dataset_name = f"{dataset_prefix}_{day_str.replace('-', '_')}"

    return download_data_for_tile(tile, image, dataset_name, export_params)


def download_chirps_precipitation_aggregate(tile, end_day, window_days, reducer, export_params, dataset_prefix="CHIRPS", kwargs=None):
    """Export a CHIRPS rolling-window aggregate ending at end_day for the given tile.

    Example dataset_name: CHIRPS_precip_14d_sum_2020_04_15

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        end_day: 'YYYY-MM-DD' or datetime/date
        window_days: e.g. 14
        reducer: 'sum' | 'mean' | 'max'
        export_params: dict used by ee.batch.Export.image.toDrive
        scale_m: export scale in meters (default ~5.5km)
    """

    end_str = _day_to_str(end_day)
    image = chirps_aggregate(end_day, window_days, reducer = reducer)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_precip_{window_days}d_{reducer}_{end_str.replace('-', '_')}"

    return download_data_for_tile(tile, image, dataset_name, export_params)



# Example usage (uncomment):
# What's best reducer?
# tile = kenya_tiles[0]
# download_chirps_precipitation_for_day(tile, "2020-04-15", export_params, kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.05, 0, -180, 0, -0.05, 50]})
# download_chirps_precipitation_aggregate(tile, "2020-04-15", window_days=14, reducer="sum", export_params=export_params, kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.05, 0, -180, 0, -0.05, 50]})


# ## ERA5-Land soil moisture (hourly → daily snapshot + rolling aggregates)
# 
# Goal: for each Sentinel-1 acquisition date $t$, export soil moisture features aligned to the pass date.
# 
# - Default: **daily mean** of hourly ERA5-Land on date $t$ (robust when acquisition time is unknown)
# - Optional: rolling-window means (e.g., 14-day mean ending at $t$) and lag-deltas (e.g., $SM(t)-SM(t-7)$)

# In[5]:


ERA5_LAND_HOURLY = "ECMWF/ERA5_LAND/HOURLY"

# Volumetric soil water at 4 depth layers
ERA5_SM_BANDS = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    # "volumetric_soil_water_layer_4",
]

def era5_sm_daily_mean_image(day):
    """Daily mean soil moisture for the given UTC day.

    Uses hourly ERA5-Land, averages all hours within [day, day+1).

    Returns an ee.Image with bands: sm_l1, sm_l2, sm_l3, sm_l4
    """
    start = _to_ee_date(day)
    end = start.advance(1, 'day')

    img = (
        ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate(start, end)
        .select(ERA5_SM_BANDS)
        .mean()
    )

    return ee.Image(img).rename([f"sm_l{i+1}" for i in range(len(ERA5_SM_BANDS))]).set({
        "date": start.format("YYYY-MM-dd"),
        "aggregation": "daily_mean",
    })

def era5_sm_aggregate(end_day, window_days, reducer = "mean"):
    """Aggregate ERA5-Land hourly soil moisture over a rolling window ending at end_day (inclusive).

    Note: For soil moisture, 'mean' over the window is usually the most sensible.

    Args:
        end_day: window ends here (inclusive)
        window_days: number of days in window (>=1)
        reducer: 'mean' | 'max' | 'min'

    Returns:
        ee.Image with 3 bands named like: sm_l1_14d_mean, ...
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    end_og = _to_ee_date(end_day)
    end = _to_ee_date(end_day).advance(1, 'day')  # Advance by 1 day to make end_day inclusive
    start = end.advance(-window_days, 'day')

    ic = (
        ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate(start, end)
        .select(ERA5_SM_BANDS)

    )
    reducer = reducer.lower()
    if reducer == "sum":
        agg = ic.sum()
    elif reducer == "mean":
        agg = ic.mean()
    elif reducer == "max":
        agg = ic.max()
    else:
        raise ValueError("reducer must be one of: 'sum', 'mean', 'max'")

    band_names = [f"sm_l{i+1}_{window_days}d_{reducer}" for i in range(len(ERA5_SM_BANDS))]
    return ee.Image(agg).rename(band_names).set({
        "start_date": start.format("YYYY-MM-dd"),
        "end_date": end_og.format("YYYY-MM-dd"),
        "window_days": window_days,
        "reducer": reducer
    })

def download_era5_sm_for_day(tile, day, export_params, dataset_prefix = "ERA5_SM", kwargs = None):
    """Export ERA5-Land soil moisture for a single day for the given tile.

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        day: 'YYYY-MM-DD' or datetime/date
        export_params: dict used by ee.batch.Export.image.toDrive
        dataset_prefix: name prefix used in Drive description
    """
    day_str = _day_to_str(day)
    image = era5_sm_daily_mean_image(day)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_daily_mean_{day_str.replace('-', '_')}"

    return download_data_for_tile(tile, image, dataset_name, export_params)

def download_era5_sm_aggregate(tile, end_day, window_days, reducer, export_params, dataset_prefix="ERA5_SM", kwargs=None):
    """Export ERA5-Land soil moisture aggregate for the given tile.

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        end_day: window ends here (inclusive)
        window_days: number of days in window (>=1)
        reducer: 'mean' | 'max' | 'min'
        export_params: dict used by ee.batch.Export.image.toDrive
        dataset_prefix: name prefix used in Drive description
    """

    end_str = _day_to_str(end_day)
    image = era5_sm_aggregate(end_day, window_days, reducer = reducer)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_{window_days}d_{reducer}_{end_str.replace('-', '_')}"

    return download_data_for_tile(tile, image, dataset_name, export_params)

# # Example usage (uncomment):
# tile = kenya_tiles[0]
# download_era5_sm_for_day(tile, "2020-04-15", export_params, dataset_prefix="ERA5_SM", kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.1, 0, -180.05, 0, -0.1, 90.05]})
# download_era5_sm_aggregate(tile, "2020-04-15", window_days=14, reducer="mean", export_params=export_params, dataset_prefix="ERA5_SM", kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.1, 0, -180.05, 0, -0.1, 90.05]})


# ## ERA5-Land temperature (hourly → daily mean + rolling aggregates)
# 
# Goal: export pass-date aligned temperature features for each Sentinel-1 acquisition date $t$.
# 
# - Default: **daily mean** of hourly ERA5-Land on date $t$
# - Optional: rolling-window means/min/max ending at $t$ for short-term thermal context

# In[6]:


ERA5_TEMP_BAND = "temperature_2m"


def era5_temperature_daily_mean_image(day):
    """Daily mean 2m temperature for the given UTC day."""
    start = _to_ee_date(day)
    end = start.advance(1, 'day')

    img = (
        ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate(start, end)
        .select([ERA5_TEMP_BAND])
        .mean()
    )

    return ee.Image(img).rename(["temperature_2m"]).set({
        "date": start.format("YYYY-MM-dd"),
        "aggregation": "daily_mean",
    })


def era5_temperature_aggregate(end_day, window_days, reducer="mean"):
    """Aggregate 2m temperature over a rolling window ending at end_day (inclusive)."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    end_og = _to_ee_date(end_day)
    end = end_og.advance(1, 'day')
    start = end.advance(-window_days, 'day')

    ic = (
        ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate(start, end)
        .select([ERA5_TEMP_BAND])
    )

    reducer = reducer.lower()
    if reducer == "mean":
        agg = ic.mean()
    elif reducer == "min":
        agg = ic.min()
    elif reducer == "max":
        agg = ic.max()
    else:
        raise ValueError("reducer must be one of: 'mean', 'min', 'max'")

    return ee.Image(agg).rename([f"temperature_2m_{window_days}d_{reducer}"]).set({
        "start_date": start.format("YYYY-MM-dd"),
        "end_date": end_og.format("YYYY-MM-dd"),
        "window_days": window_days,
        "reducer": reducer,
    })


def download_era5_temperature_for_day(tile, day, export_params, dataset_prefix="ERA5_TEMP", kwargs=None):
    """Export ERA5-Land daily-mean 2m temperature for the given tile."""
    day_str = _day_to_str(day)
    image = era5_temperature_daily_mean_image(day)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_daily_mean_{day_str.replace('-', '_')}"
    return download_data_for_tile(tile, image, dataset_name, export_params)


def download_era5_temperature_aggregate(tile, end_day, window_days, reducer, export_params, dataset_prefix="ERA5_TEMP", kwargs=None):
    """Export ERA5-Land temperature aggregate for the given tile."""
    end_str = _day_to_str(end_day)
    image = era5_temperature_aggregate(end_day, window_days, reducer=reducer)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_{window_days}d_{reducer}_{end_str.replace('-', '_')}"
    return download_data_for_tile(tile, image, dataset_name, export_params)


def schedule_era5_temperature_exports_for_tile(
    tile,
    s1_dates,
    export_params,
    *,
    export_daily=True,
    aggregates=((3, "mean"), (7, "mean"), (14, "mean")),
    sleep_s=0,
    kwargs=None,
):
    """Schedule ERA5 temperature exports for one tile for a list of Sentinel-1 pass dates."""
    dates = sorted(set(_day_to_str(d) for d in s1_dates))

    for day_str in dates:
        if export_daily:
            download_era5_temperature_for_day(tile, day_str, export_params, kwargs=kwargs)
            if sleep_s:
                time.sleep(sleep_s)

        for window_days, reducer in aggregates:
            download_era5_temperature_aggregate(
                tile,
                day_str,
                window_days=int(window_days),
                reducer=str(reducer),
                export_params=export_params,
                kwargs=kwargs,
            )
            if sleep_s:
                time.sleep(sleep_s)


# Example usage (uncomment):
# tile = kenya_tiles[0]
# download_era5_temperature_for_day(tile, "2020-04-15", export_params, kwargs={"crs": "EPSG:4326", "crsTransform": [0.1, 0, -180.05, 0, -0.1, 90.05]})
# download_era5_temperature_aggregate(tile, "2020-04-15", 7, "mean", export_params, kwargs={"crs": "EPSG:4326", "crsTransform": [0.1, 0, -180.05, 0, -0.1, 90.05]})


# ## ERA5-Land runoff (hourly → daily sums + rolling aggregates)
# 
# Goal: export pass-date aligned hydrologic fluxes for each Sentinel-1 acquisition date $t$.
# 
# - Default: **daily sums** for `surface_runoff`, `runoff`, `total_precipitation`, and `total_evaporation`
# - Optional: rolling-window sums/means/max over 3/7/14 days ending at $t$

# In[7]:


ERA5_LAND_HOURLY = "ECMWF/ERA5_LAND/HOURLY"
ERA5_RUNOFF_BANDS = [
    "surface_runoff",
    "runoff",

]

def era5_runoff_daily_sum_image(day):
    """Daily sum runoff for the given UTC day."""
    start = _to_ee_date(day)
    end = start.advance(1, 'day')

    img = (
        ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate(start, end)
        .select(ERA5_RUNOFF_BANDS)
        .sum()
    )

    return ee.Image(img).rename(["surface_runoff", "runoff"]).set({
        "date": start.format("YYYY-MM-dd"),
        "aggregation": "daily_sum",
    })

def era5_runoff_aggregate(end_day, window_days, reducer = "sum"):
    """Aggregate runoff over a rolling window ending at end_day (inclusive)."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    end_og = _to_ee_date(end_day)
    end = _to_ee_date(end_day).advance(1, 'day')  # Advance by 1 day to make end_day inclusive
    start = end.advance(-window_days, 'day')

    ic = (
        ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate(start, end)
        .select(ERA5_RUNOFF_BANDS)
    )

    reducer = reducer.lower()
    if reducer == "sum":
        agg = ic.sum()
    elif reducer == "mean":
        agg = ic.mean()
    elif reducer == "max":
        agg = ic.max()
    else:
        raise ValueError("reducer must be one of: 'sum', 'mean', 'max'")

    band_names = [f"{band}_{window_days}d_{reducer}" for band in ERA5_RUNOFF_BANDS]
    return ee.Image(agg).rename(band_names).set({
        "start_date": start.format("YYYY-MM-dd"),
        "end_date": end_og.format("YYYY-MM-dd"),
        "window_days": window_days,
        "reducer": reducer
    })

def download_era5_runoff_for_day(tile, day, export_params, dataset_prefix = "ERA5_Runoff", kwargs = None):
    """Export ERA5-Land runoff for a single day for the given tile."""
    day_str = _day_to_str(day)
    image = era5_runoff_daily_sum_image(day)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_daily_sum_{day_str.replace('-', '_')}"
    return download_data_for_tile(tile, image, dataset_name, export_params)


def download_era5_runoff_aggregate(tile, end_day, window_days, reducer, export_params, dataset_prefix="ERA5_Runoff", kwargs=None):
    """Export ERA5-Land runoff aggregate for the given tile."""
    end_str = _day_to_str(end_day)
    image = era5_runoff_aggregate(end_day, window_days, reducer = reducer)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_{window_days}d_{reducer}_{end_str.replace('-', '_')}"
    return download_data_for_tile(tile, image, dataset_name, export_params)

# Example usage (uncomment):
# tile = kenya_tiles[0]
# download_era5_runoff_for_day(tile, "2020-04-15", export_params, dataset_prefix="ERA5_Runoff", kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.1, 0, -180.05, 0, -0.1, 90.05]})
# download_era5_runoff_aggregate(tile, "2020-04-15", window_days=14, reducer="sum", export_params=export_params, dataset_prefix="ERA5_Runoff", kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.1, 0, -180.05, 0, -0.1, 90.05]})


# ## MODIS NDVI (16-day composites aligned to Sentinel-1 dates)
# 
# Goal: export vegetation context for each Sentinel-1 acquisition date $t$.
# 
# Because MODIS NDVI is a **16-day composite** rather than daily, we use a **causal recent-composite strategy**:
# 
# - default: most recent available MODIS NDVI composite on or before date $t$
# - optional: aggregate all MODIS NDVI composites within a lookback window ending at $t$

# In[8]:


# MODIS_NDVI_COLLECTION = "MODIS/061/MOD13A2"
MODIS_NDVI_COLLECTION = "MODIS/061/MOD13Q1"
MODIS_NDVI_SCALE = 0.0001

def modis_ndvi_recent_image(end_day, lookback_days = 32):
    """Get the most recent MODIS NDVI image within [end_day - lookback_days, end_day].

    Note: MODIS NDVI is every 16 days, so we look back ~32 days to have a good chance of getting at least one image.

    Returns:
        ee.Image with band 'NDVI' scaled to [0, 1].
    """
    if lookback_days < 16:
        raise ValueError("lookback_days should usually be >= 16 for MODIS NDVI")

    end_og = _to_ee_date(end_day)
    end = _to_ee_date(end_day).advance(1, 'day')  # Advance by 1 day to make end_day inclusive
    start = end.advance(-lookback_days, 'day')

    img = (
        ee.ImageCollection(MODIS_NDVI_COLLECTION)
        .filterDate(start, end)
        .select('NDVI')
        .sort('system:time_start', False) # Sort by time descending to get the most recent image first
        .first()
    )

    return ee.Image(img).multiply(MODIS_NDVI_SCALE).rename(['ndvi']).set({
        'end_date': end_og.format('YYYY-MM-dd'),
        'lookback_days': lookback_days,
        'selection': 'most_recent_composite',
    })

def modis_ndvi_aggregate(end_day, lookback_days = 32, reducer = "mean"):
    """Aggregate MODIS NDVI over images within [end_day - lookback_days, end_day].

    Args:
        end_day: window ends here (inclusive)
        lookback_days: number of days to look back for MODIS images
        reducer: 'mean' | 'max' | 'min'

    Returns:
        ee.Image with band named like 'ndvi_32d_mean'.
    """
    if lookback_days < 16:
        raise ValueError("lookback_days should usually be >= 16 for MODIS NDVI")

    end_og = _to_ee_date(end_day)
    end = _to_ee_date(end_day).advance(1, 'day')  # Advance by 1 day to make end_day inclusive
    start = end.advance(-lookback_days, 'day')

    ic = (
        ee.ImageCollection(MODIS_NDVI_COLLECTION)
        .filterDate(start, end)
        .select('NDVI')
        .map(lambda img: ee.Image(img).multiply(MODIS_NDVI_SCALE))

    )

    reducer = reducer.lower()
    if reducer == "mean":
        agg = ic.mean()
    elif reducer == "max":
        agg = ic.max()
    elif reducer == "min":
        agg = ic.min()
    else:
        raise ValueError("reducer must be one of: 'mean', 'max', 'min'")

    band_name = f"ndvi_{lookback_days}d_{reducer}"
    return ee.Image(agg).rename(band_name).set({
        'end_date': end_og.format('YYYY-MM-dd'),
        'lookback_days': lookback_days,
        'reducer': reducer,
    })


def download_modis_ndvi_for_day(tile, end_day, export_params, lookback_days = 32, dataset_prefix = "MODIS_NDVI", kwargs = None):
    """Export the most recent MODIS NDVI image for the given tile.

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        end_day: 'YYYY-MM-DD' or datetime/date (window ends here, inclusive)
        export_params: dict used by ee.batch.Export.image.toDrive
        lookback_days: number of days to look back for MODIS images
        dataset_prefix: name prefix used in Drive description
    """
    end_str = _day_to_str(end_day)
    image = modis_ndvi_recent_image(end_day, lookback_days)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_recent_{lookback_days}d_{end_str.replace('-', '_')}"
    return download_data_for_tile(tile, image, dataset_name, export_params)


def download_modis_ndvi_aggregate(tile, end_day, lookback_days, reducer, export_params, dataset_prefix="MODIS_NDVI", kwargs=None):
    """Export aggregated MODIS NDVI for the given tile.

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        end_day: 'YYYY-MM-DD' or datetime/date (window ends here, inclusive)
        lookback_days: number of days to look back for MODIS images
        reducer: 'mean' | 'max' | 'min'
        export_params: dict used by ee.batch.Export.image.toDrive
        dataset_prefix: name prefix used in Drive description
    """
    end_str = _day_to_str(end_day)
    image = modis_ndvi_aggregate(end_day, lookback_days, reducer)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_{lookback_days}d_{reducer}_{end_str.replace('-', '_')}"
    return download_data_for_tile(tile, image, dataset_name, export_params)



# Example usage (uncomment):
# tile = kenya_tiles[0]
# download_modis_ndvi_for_day(tile, "2020-04-15", export_params, lookback_days=32, dataset_prefix="MODIS_NDVI", kwargs = {"crs": 'EPSG:4326', "scale": 926.625433})
# download_modis_ndvi_aggregate(tile, "2020-04-15", lookback_days=32, reducer="mean", export_params=export_params, dataset_prefix="MODIS_NDVI", kwargs = {"crs": 'EPSG:4326', "scale": 926.625433})




# ## Static soil properties (SoilGrids-style layers via Earth Engine OpenLandMap)
# 
# Goal: export static soil conditioning layers once per tile.
# 
# Practical choice: use Earth Engine-accessible **OpenLandMap** soil-property rasters as SoilGrids-style static predictors.
# 
# Included here:
# - **Clay content** at 250 m

# In[9]:


SOIL_STATIC_ASSETS = {
    'clay': {
        'asset': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
        'bands': ['b10', 'b30', 'b60'],
        # 'bands': ['b0', 'b10', 'b30', 'b60'],
        'rename_prefix': 'clay',
    },
}

def soil_static_property_image(property_name = 'clay'):
    spec = SOIL_STATIC_ASSETS[property_name]
    bands = spec['bands']
    rename = [f"{spec['rename_prefix']}_{band}" for band in bands]

    img = ee.Image(spec['asset']).select(bands).rename(rename)
    return img.set({
        'property_name': property_name,
        'static_layer': True,

    })

def download_soil_static_property(tile, export_params, property_name = 'clay', dataset_prefix = "Soil_Static", kwargs = None):
    """Export a soil static property for the given tile.

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        export_params: dict used by ee.batch.Export.image.toDrive
        property_name: e.g. 'clay'
        dataset_prefix: name prefix used in Drive description
    """
    image = soil_static_property_image(property_name)

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_{property_name}"
    return download_data_for_tile(tile, image, dataset_name, export_params)

# Example usage (uncomment):
# tile = kenya_tiles[0]
# download_soil_static_property(tile, export_params, property_name='clay', dataset_prefix="Soil_Static", kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.002083333, 0, -180, 0, -0.002083333, 87.37]})


# ## ESA WorldCover land cover classes (10 m)
# Goal: export static land cover classes once per tile.
# 

# In[10]:


def download_esa_worldcover(tile, export_params, dataset_prefix="ESA_WorldCover", kwargs=None):
    """Export ESA WorldCover 10m land cover for the given tile."""
    image = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('land_cover')

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)
    dataset_name = f"{dataset_prefix}_v200"
    return download_data_for_tile(tile, image, dataset_name, export_params)





# ## Static hydrological conditioning layers
# 
# Goal: export static hydrological conditioning layers per tile, derived from **MERIT Hydro** (`MERIT/Hydro/v1_0_1`, 3 arc-second / ~90 m):
# 
# 1. **Flow accumulation** — upstream drainage area (`upa` band, in km²). Identifies where water concentrates. Log-transformed for better dynamic range.
# 2. **Distance to rivers** — Euclidean distance (in metres) from each pixel to the nearest "river" pixel, where rivers are defined by thresholding upstream area (≥ 10 km²). High values = far from drainage network = less fluvial flood risk.
# 3. **Height Above Nearest Drainage (HAND)** — bonus: vertical distance to nearest drainage. Arguably the single best flood susceptibility predictor after elevation.

# In[11]:


MERIT_HYDRO = "MERIT/Hydro/v1_0_1"

# ── Flow accumulation (upstream drainage area) ──────────────────────────────────
def merit_flow_accumulation_image(perform_log_transform = True):
    """Return MERIT Hydro upstream drainage area (km²), log10-transformed.

    Band `upa` gives the upstream drainage area in km² for every ~90 m pixel.
    We apply log10(upa + 1) so the huge dynamic range (0 → millions km²)
    compresses into a model-friendly range (~0–6).

    Returns:
        ee.Image with band 'flow_acc_log10' (unitless, log10 km²).
    """
    merit = ee.Image(MERIT_HYDRO)
    upa = merit.select('upa') # Upstream drainage area in km²
    if perform_log_transform:
        flow_acc_log10 = upa.add(1).log10().rename('flow_acc_log10')
        return flow_acc_log10.set({"source": "MERIT_Hydro", "band": "upa", "transform": "log10(upa+1)", "static_layer": True})
    else:
        return upa.set({"source": "MERIT_Hydro", "band": "upa", "static_layer": True})

# ── Height Above Nearest Drainage (HAND) ────────────────────────────────────────
def merit_hand_image():
    """Return MERIT Hydro Height Above Nearest Drainage (HAND) in metres.

    HAND is the vertical distance between each pixel and the nearest river
    pixel along the drainage path.  It is one of the strongest flood
    susceptibility predictors: low HAND = near river level = flood-prone.

    Returns:
        ee.Image with band 'hand_m' (metres).
    """
    merit = ee.Image(MERIT_HYDRO)
    hand = merit.select('hnd').rename('hand_m') # HAND in metres
    return hand.set({"source": "MERIT_Hydro", "band": "hand", "static_layer": True})

def download_merit_hydro_layer(tile, export_params, layer_name = 'flow_accumulation', dataset_prefix = "MERIT_Hydro", perform_log_transform = True, kwargs = None):
    """Export a MERIT Hydro layer for the given tile.

    Args:
        tile: (min_lon, min_lat, max_lon, max_lat)
        export_params: dict used by ee.batch.Export.image.toDrive
        layer_name: 'flow_accumulation' or 'hand'
        dataset_prefix: name prefix used in Drive description
    """
    if layer_name == 'flow_accumulation':
        image = merit_flow_accumulation_image(perform_log_transform = perform_log_transform)
    elif layer_name == 'hand':
        image = merit_hand_image()
    else:
        raise ValueError("layer_name must be 'flow_accumulation' or 'hand'")

    export_params = dict(export_params)
    if kwargs is not None:
        export_params.update(kwargs)

    dataset_name = f"{dataset_prefix}_{layer_name}"
    return download_data_for_tile(tile, image, dataset_name, export_params)

# Example usage (uncomment):
# tile = kenya_tiles[0]
# download_merit_hydro_layer(tile, export_params, layer_name='flow_accumulation', dataset_prefix="MERIT_Hydro", kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.0008333333333333334,0,-180.00041666666667,0,-0.0008333333333333334,84.99958333333333]})
# download_merit_hydro_layer(tile, export_params, layer_name='hand', dataset_prefix="MERIT_Hydro", kwargs = {"crs": 'EPSG:4326', "crsTransform": [0.0008333333333333334,0,-180.00041666666667,0,-0.0008333333333333334,84.99958333333333]})


# ## For the tiles covering Kenya, export the dates where flood were detected.

# In[12]:


from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
from natsort import natsorted

def download_tile(tile,
                  repo_id = "ai-for-good-lab/ai4g-flood-dataset",
                  download_240m_buffer_tif = True,
                  download_80m_buffer_tif = True,
                  download_post_processing_parquet = True,
                  download_recurrence_80m_buffer_tif = True,
                  local_dir = None,
                  overwrite = False
                  ):

    print(f"\nDownloading data for tile: {tile}")

    if local_dir is None:
        local_dir = Path(f"flood_data")

    if not local_dir.exists():
        local_dir.mkdir(parents=True, exist_ok=True)

    root_dir = tile[:3]
    files_to_download = []
    if download_240m_buffer_tif:
        files_to_download.append(f"{root_dir}/{tile}/{tile}-240m-buffer.tif")

    if download_80m_buffer_tif:
        files_to_download.append(f"{root_dir}/{tile}/{tile}-80m-buffer.tif")

    if download_post_processing_parquet:
        files_to_download.append(f"{root_dir}/{tile}/{tile}-post-processing.parquet")

    if download_recurrence_80m_buffer_tif:
        files_to_download.append(f"{root_dir}/{tile}/{tile}-recurrence-80m-buffer.tif")

    for file_path in files_to_download:
        local_file_path = local_dir / Path(file_path)
        print(f"Does local file exist? {local_file_path.exists()}")
        if not local_file_path.exists() or overwrite:
            print(f"Downloading {file_path} to {local_file_path}")
            hf_hub_download(repo_id = repo_id,
                            filename = file_path,
                            local_dir = str(local_dir),
                            repo_type = "dataset",
            )
        else:
            print(f"File {local_file_path} already exists. Skipping download.")


    return



def extract_dates_from_parquet(parquet_file_path, filter_params = None):
    """Extract dates when floods were detected from a post-processing parquet file."""
    df = pd.read_parquet(parquet_file_path)

    # Create a 'date' column from 'year', 'month', 'day'
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    if filter_params is not None:
        print(f"\nApplying filters to data...")

        mask = (
            (df.dem_metric_2 < filter_params["dem_metric_2_max"]) &
            (df.soil_moisture_sca > filter_params["soil_moisture_sca_min"]) &
            (df.soil_moisture_zscore > filter_params["soil_moisture_zscore_min"]) &
            (df.soil_moisture > filter_params["soil_moisture_min"]) &
            (df.temp > filter_params["temp_min"]) &
            (df.land_cover != filter_params["exclude_land_cover"]) &
            (df.edge_false_positives == filter_params["edge_fp_eq"])
        )

        filtered_df = df[mask]
        print(f"  Rows before filter: {len(df)}, after filter: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.1f}%)")

    else:
        filtered_df = df

    # Extract unique dates when floods were detected
    flood_dates = filtered_df['date'].unique()
    flood_dates = natsorted(pd.to_datetime(flood_dates).strftime('%Y-%m-%d').tolist())

    if len(flood_dates) == 0:
        print(f"⚠️  WARNING: No flood dates found in {parquet_file_path.name} after filtering!")

    print(f"Extracted {len(flood_dates)} unique flood dates from {parquet_file_path.name}")

    # Save extracted dates to csv file
    output_csv_path = parquet_file_path.with_name(parquet_file_path.stem + '_flood_dates.csv')
    pd.DataFrame({'date': flood_dates}).to_csv(output_csv_path, index=False)
    print(f"Saved extracted flood dates to {output_csv_path}")

    return True


# In[13]:


# Generate list of tile names from kenya_tiles
kenya_bounds = get_country_bounds("Kenya")
kenya_tiles = generate_tiles_from_bounds(kenya_bounds, tile_size = 3)
tile_names = [tile_name_from_latlon(tile[1], tile[0]) for tile in kenya_tiles]

# Define local directory to save downloaded files
local_data_dir = Path("D:/flood_data")
local_data_dir.mkdir(parents=True, exist_ok=True)

for tile in tile_names:
    download_tile(
        tile = tile,
        download_240m_buffer_tif = False,
        download_80m_buffer_tif = False,
        download_post_processing_parquet = True,
        download_recurrence_80m_buffer_tif = False,
        local_dir = local_data_dir,
        overwrite = False
    )
    time.sleep(3)  # Sleep for 3 seconds between downloads to avoid rate limiting




# In[79]:


# Define parquet filters
filter_params = {
    "dem_metric_2_max": 10,
    "soil_moisture_sca_min": 1,
    "soil_moisture_zscore_min": 1,
    "soil_moisture_min": 20,
    "temp_min": 0,
    "exclude_land_cover": 60,
    "edge_fp_eq": 0
}

# Extract path to parquet file
base_path = local_data_dir
parquet_files = list(base_path.glob("**/*-post-processing.parquet"))
print(f"Found {len(parquet_files)} parquet files in {base_path}")

# Apply the date extraction function to each parquet file
for parquet_file in parquet_files:
    extract_dates_from_parquet(parquet_file, filter_params = filter_params)
    # break


# ## SChedule GEE exports for all flood dates per tile
# 
# For each Kenya tile, load the extracted flood dates CSV and schedule exports of all dynamic conditioning variables (CHIRPS, ERA5 soil moisture, temperature, runoff) aligned to those dates. Static layers (DEM, slope, MERIT Hydro) only need one export per tile.
# 
# **Each dataset is exported at its native resolution, we'll deal with resampling later**
# 
# **Export resolution conventions:**
# - SRTM/static layers: 30m (`crsTransform` for 1 arc-sec)
# - MERIT Hydro: ~90m (3 arc-sec native)
# - CHIRPS: ~5.5 km (0.05° native)
# - ERA5-Land: ~11 km (0.1° native)
# - MODIS NDVI: 250 m (native)
# - OpenLandMap soil properties: 250 m (native)
# - ESA WorldCover: 10 m (native)
# 

# In[13]:


# ── Resolution-specific export kwargs ────────────────────────────────────────────
# Each dataset is exported at its native resolution to avoid inflating file sizes
# with meaningless upsampled pixels.
from natsort import natsorted
from pathlib import Path


# SRTM DEM and Slope
# 1 arc-second (approximately 30m)
SRTM_EXPORT_KWARGS = {
    'folder': 'SRTM',
    'crs': 'EPSG:4326',
    'crs_transform': [0.0002777777777777778, 0, -180.0001388888889,
                      0, -0.0002777777777777778, 60.00013888888889]

}

# CHIRPS Precipitation
# 0.05° ≈ 5.5 km
CHIRPS_EXPORT_KWARGS = {
    'folder': 'CHIRPS',
    'crs': 'EPSG:4326',
    'crs_transform': [0.05, 0, -180, 0, -0.05, 50]
}

# ERA5-Land: soil moisture, temperature, runoff, 
# 0.1° ≈ 11 km
ERA5_EXPORT_KWARGS = {
    'folder': 'ERA5',
    'crs': 'EPSG:4326',
    'crs_transform': [0.1, 0, -180.05, 0, -0.1, 90.05]
}

# MODIS NDVI
# ~250 m
MODIS_EXPORT_KWARGS = {
    'folder': 'MODIS',
    'crs': 'EPSG:4326',
    'scale': 231.65635826395825
}

# OpenLandMap clay content
# 250 m
SOIL_CLAY_EXPORT_KWARGS = {
    'folder': 'SOIL_CLAY',
    'crs': 'EPSG:4326',
    'crs_transform': [0.002083333, 0, -180, 0, -0.002083333, 87.37]
}

# MERIT Hydro flow accumulation and HAND
# 90 m
MERIT_HYDRO_EXPORT_KWARGS = {
    'folder': 'MERIT_HYDRO',
    'crs': 'EPSG:4326',
    'crs_transform': [0.0008333333333333334, 0, -180.00041666666667,
                      0, -0.0008333333333333334, 84.99958333333333]
}

# ESA WorldCover
# 10 m
ESA_WORLDCOVER_EXPORT_KWARGS = {
    'folder': 'ESA_WORLDCOVER',
    'crs': 'EPSG:4326',
    'crs_transform': [8.333333333333333e-05, 0, -180,
                      0, -8.333333333333333e-05, 84]
}









# In[14]:


def schedule_exports_for_tile(
    tile_name,
    flood_dates,
    export_params,
    # -- Static layers (exported once, not date-specific) --
    export_dem_slope = True,
    export_merit_hydro = True,
    export_soil_clay = True,
    export_esa_worldcover = True,

    # -- Dynamic layers (exported per flood date) --
    export_chirps_precipitation = True,
    chirps_windows = ((3, "sum"), (7, "sum"), (14, "sum")),  # (window_days, reducer)

    export_era5_sm = True,
    era5_sm_windows = ((7, "mean"), (14, "mean")),

    export_era5_temp = True,
    era5_temp_windows=((7, "mean"),),

    export_era5_runoff=True,
    era5_runoff_windows=((3, "sum"), (7, "sum"), (14, "sum")),

    export_modis_ndvi = True,
    modis_ndvi_lookback = 32,

    # ── Control ──
    sleep_between_dates=3,
    sleep_between_tasks=1,
    max_dates=None,



):
    """Schedule all GEE export tasks for one Kenya tile.

    Static layers are exported once. Dynamic layers are exported for each
    flood date in the provided list.

    Args:
        tile_name: e.g. 'N00E036'
        flood_dates: list of 'YYYY-MM-DD' strings
        export_params: base export params dict (folder, maxPixels, fileFormat)
        max_dates: if set, only process the first N dates (useful for testing)
    """
    tile_bounds = tile_name_to_coord_bounds(tile_name)
    n_dates = len(flood_dates)

    if max_dates is not None:
        flood_dates = flood_dates[:max_dates]
        print(f"⚠️  Limiting to first {max_dates} dates for testing")

    print(f"\n{'='*70}")
    print(f"  Tile: {tile_name}  |  Bounds: {tile_bounds}  |  Flood dates: {len(flood_dates)}/{n_dates}")
    print(f"{'='*70}")

    # ── 1. Static layers (one export per tile) ───────────────────────────────
    if export_dem_slope:
        print(f"\n  [Static] DEM + Slope")
        download_elevation_and_slope(tile_bounds, export_params, kwargs = SRTM_EXPORT_KWARGS)
        if sleep_between_tasks:
            time.sleep(sleep_between_tasks)
    else:
        print(f"\n  [Static] Skipping DEM + Slope export")

    if export_merit_hydro:
        print(f"\n  [Static] MERIT Hydro (flow accumulation + HAND)")
        download_merit_hydro_layer(tile_bounds, export_params, layer_name = 'flow_accumulation', kwargs = MERIT_HYDRO_EXPORT_KWARGS)
        download_merit_hydro_layer(tile_bounds, export_params, layer_name = 'hand', kwargs = MERIT_HYDRO_EXPORT_KWARGS)
        if sleep_between_tasks:
            time.sleep(sleep_between_tasks)

    else:
        print(f"\n  [Static] Skipping MERIT Hydro export")

    if export_soil_clay:
        print(f"\n  [Static] Soil clay content (OpenLandMap)")
        download_soil_static_property(tile_bounds, export_params, property_name = 'clay', kwargs = SOIL_CLAY_EXPORT_KWARGS)
        if sleep_between_tasks:
            time.sleep(sleep_between_tasks)
    else:
        print(f"\n  [Static] Skipping soil clay content export")

    if export_esa_worldcover:
        print(f"\n  [Static] ESA WorldCover")
        download_esa_worldcover(tile_bounds, export_params, kwargs = ESA_WORLDCOVER_EXPORT_KWARGS)
        if sleep_between_tasks:
            time.sleep(sleep_between_tasks)
    else:
        print(f"\n  [Static] Skipping ESA WorldCover export")

    # ── 2. Dynamic layers (per flood date) ───────────────────────────────────
    print(f"\n  [Dynamic] Scheduling exports for {len(flood_dates)} flood dates...")

    for i, day_str in enumerate(flood_dates):
        # if (i + 1) % 10 == 0 or i == 0:
        print(f"\n    Date {i+1}/{len(flood_dates)}: {day_str}")

        # ── CHIRPS precipitation ──
        if export_chirps_precipitation:
            # Daily precipitation
            download_chirps_precipitation_for_day(tile_bounds, day_str, export_params, kwargs = CHIRPS_EXPORT_KWARGS)

            # Rolling window aggregates
            for window_days, reducer in chirps_windows:
                download_chirps_precipitation_aggregate(tile_bounds, day_str, window_days=window_days, reducer=reducer,
                                                        export_params=export_params, kwargs = CHIRPS_EXPORT_KWARGS)

        # ── ERA5 soil moisture ──
        if export_era5_sm:
            download_era5_sm_for_day(tile_bounds, day_str, export_params, kwargs = {**ERA5_EXPORT_KWARGS, 'folder': 'ERA5_LAND_SM'})
            for window_days, reducer in era5_sm_windows:
                download_era5_sm_aggregate(tile_bounds, day_str, window_days = window_days, reducer = reducer,
                                            export_params = export_params, kwargs = {**ERA5_EXPORT_KWARGS, 'folder': 'ERA5_LAND_SM'})



        # ── ERA5 temperature ──
        if export_era5_temp:
            download_era5_temperature_for_day(tile_bounds, day_str, export_params, kwargs = {**ERA5_EXPORT_KWARGS, 'folder': 'ERA5_LAND_TEMP'})
            for window_days, reducer in era5_temp_windows:
                download_era5_temperature_aggregate(tile_bounds, day_str, window_days = window_days, reducer = reducer,
                                            export_params = export_params, kwargs = {**ERA5_EXPORT_KWARGS, 'folder': 'ERA5_LAND_TEMP'})


        # ── ERA5 runoff ──
        if export_era5_runoff:
            download_era5_runoff_for_day(tile_bounds, day_str, export_params, kwargs = {**ERA5_EXPORT_KWARGS, 'folder': 'ERA5_LAND_RUNOFF'})
            for window_days, reducer in era5_runoff_windows:
                download_era5_runoff_aggregate(tile_bounds, day_str, window_days = window_days, reducer = reducer,
                                            export_params = export_params, kwargs = {**ERA5_EXPORT_KWARGS, 'folder': 'ERA5_LAND_RUNOFF'})

        # ── MODIS NDVI ──
        if export_modis_ndvi:
            download_modis_ndvi_for_day(tile_bounds, day_str, export_params, lookback_days = modis_ndvi_lookback, kwargs = MODIS_EXPORT_KWARGS)

        if sleep_between_dates:
            time.sleep(sleep_between_dates)


    print(f"\n  ✅ Done scheduling exports for {tile_name}")

















# In[15]:


def schedule_all_kenya_exports(
    export_params,
    base_path = Path("flood_data"),
    tile_names = None,
    max_dates_per_tile = None,
    sleep_between_tiles = 5,
    start_date_index = 0,
    end_date_index = -1,
    **kwargs
):
    """Schedule GEE export tasks for all Kenya tiles.

    Args:
        export_params: base export params dict
        base_path: path to local flood_data directory with parquet/CSVs
        tile_names: list of tile name strings, or None to auto-detect from CSVs
        max_dates_per_tile: limit dates per tile (for testing)
        start_date_index, end_date_index: slice of dates to process per tile (for testing/debugging)
        **kwargs: forwarded to schedule_exports_for_tile
    """

    if tile_names is None:
        # Auto-detect tiles from existing flood date CSVs
        csv_files = list(base_path.glob("**/*-post-processing_flood_dates.csv"))
        tile_names = natsorted([f.parent.name for f in csv_files])
        print(f"Auto-detected {len(tile_names)} tiles: {tile_names}")

    for tile_name in tile_names:
        flood_dates = load_flood_dates_for_tile(tile_name, base_path=base_path, start_date_index=start_date_index, end_date_index=end_date_index)
        if not flood_dates:
            print(f"⚠️  No flood dates found for {tile_name}. Skipping export scheduling.")
            continue

        schedule_exports_for_tile(
            tile_name = tile_name,
            flood_dates = flood_dates,
            export_params = export_params,
            max_dates = max_dates_per_tile,
            **kwargs
        )

        if sleep_between_tiles:
            time.sleep(sleep_between_tiles)


    print(f"\n{'='*70}")
    print(f"  All tiles scheduled. Use check_task_status() to monitor progress.")
    print(f"{'='*70}")


# In[ ]:


# Schedule exports for tiles
kenya_bounds = get_country_bounds("Kenya")
kenya_tiles = generate_tiles_from_bounds(kenya_bounds, tile_size = 3)
tile_names = [tile_name_from_latlon(tile[1], tile[0]) for tile in kenya_tiles]
tile_names = natsorted(tile_names)
export_params = {
    # 'folder': 'Kenya_Flood_Dataset',
    'maxPixels': 1e13,
    'fileFormat': 'GeoTIFF'
}
schedule_all_kenya_exports(
    export_params = export_params,
    base_path = Path("D:/flood_data"),
    tile_names = tile_names[3:4],
    max_dates_per_tile = None,  # Set to None to process all dates
    sleep_between_tiles = 5,
)


# In[19]:


# Download cut off at some point so continuing
kenya_bounds = get_country_bounds("Kenya")
kenya_tiles = generate_tiles_from_bounds(kenya_bounds, tile_size = 3)
tile_names = [tile_name_from_latlon(tile[1], tile[0]) for tile in kenya_tiles]
tile_names = natsorted(tile_names)
export_params = {
    # 'folder': 'Kenya_Flood_Dataset',
    'maxPixels': 1e13,
    'fileFormat': 'GeoTIFF'
}
schedule_all_kenya_exports(
    export_params = export_params,
    base_path = Path("D:/flood_data"),
    tile_names = tile_names[3:4],
    max_dates_per_tile = None,  # Set to None to process all dates
    start_date_index = 425,
    end_date_index = -1,
    sleep_between_tiles = 5,
    **{
        'export_dem_slope': False,
        'export_merit_hydro': False,
        'export_soil_clay': False,
        'export_esa_worldcover': False,
    } # Skip static layers since they were already exported in the first run
)


# In[ ]:


# ── Test with one tile, first 2 dates ────────────────────────────────────────────
# schedule_all_kenya_exports(
#     export_params=export_params,
#     base_path=Path("flood_data"),
#     tile_names=["N00E036"],   # single tile for testing
#     max_dates_per_tile=2,     # only first 2 dates
# )

# ── Full run (all tiles, all dates) ─────────────────────────────────────────────
# schedule_all_kenya_exports(
#     export_params=export_params,
#     base_path=Path("flood_data"),
#     tile_names=None,           # auto-detect from CSVs
#     max_dates_per_tile=None,   # all dates
#     sleep_between_tiles=5,
# )

# ── Check export status ──
# check_task_status(50)


# In[20]:


# Obtain dates for when floods were detected in tile N00E033
base_path = Path("D:/flood_data")
tile_name = "N03E033"
flood_dates = load_flood_dates_for_tile(tile_name, base_path=base_path, start_date_index = 424)



# In[21]:


flood_dates[:5]


# In[18]:


flood_dates.index("2024-01-14")


# In[20]:


# Visualize sample exported images
tiff_file = Path("N00E033_CHIRPS_precip_3d_sum_2016_04_25.tif")
import rasterio
import matplotlib.pyplot as plt
with rasterio.open(tiff_file) as src:
    img = src.read(1)  # Read the first band
plt.imshow(img, cmap='viridis')
plt.colorbar(label='Precipitation (mm)')
plt.title('Sample Exported CHIRPS Precipitation')
plt.show()
# Check resolution of exported image
with rasterio.open(tiff_file) as src:
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Pixel size (degrees): ({src.transform.a}, {src.transform.e})")



# ## dataset and loader
