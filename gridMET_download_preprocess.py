import os
from tqdm import tqdm
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import requests
import numpy as np

# ----------------------------
# Step 1: Setup Configuration
# ----------------------------

years = list(range(2000, 2021))
variables = ['pr', 'tmmx', 'tmmn', 'etr']
output_dir = r"D:\Rainfall_runoff_Aayush\gridMET_data"
shapefile_path = r"D:\Rainfall_runoff_Aayush\watershed\watershed.shp"
output_csv = "leaf_river_basin_climate_timeseries.csv"

os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Step 2: Load Basin Shapefile
# ----------------------------

basin = gpd.read_file(shapefile_path)
basin = basin.to_crs("EPSG:4326")  # gridMET data is in EPSG:4326

# ----------------------------
# Step 3: Download + Process in Chunks
# ----------------------------

def download_file(url, local_path):
    if not os.path.exists(local_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

def extract_variable(ds, varname):
    """
    Returns the variable DataArray and the raw unrenamed version (for metadata).
    """
    var_map = {
        'pr': 'precipitation_amount',
        'tmmx': 'air_temperature',
        'tmmn': 'air_temperature',
        'etr': 'potential_evapotranspiration',
    }

    var_actual = var_map.get(varname, varname)
    da = ds[var_actual].rename(varname)
    return da, ds[var_actual]  # renamed DataArray, raw DataArray for metadata

def crop_to_basin(xarr, basin_gdf):
    xarr = xarr.rio.write_crs("EPSG:4326", inplace=True)
    return xarr.rio.clip(basin_gdf.geometry.values, basin_gdf.crs, drop=True)

def apply_scale_and_offset(da, raw_da):
    """
    Apply scale_factor and add_offset from NetCDF metadata, and mask _FillValue.
    """
    scale_factor = raw_da.attrs.get('scale_factor', 1.0)
    add_offset = raw_da.attrs.get('add_offset', 0.0)
    fill_value = raw_da.attrs.get('_FillValue', None)

    # Apply scaling and offset
    da = da.astype(np.float32) * scale_factor + add_offset

    # Mask missing values
    if fill_value is not None:
        da = da.where(raw_da != fill_value)

    return da

def process_variable_year(var, year):
    print(f"📦 Processing {var} for {year}...")
    url = f"https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
    local_path = os.path.join(output_dir, f"{var}_{year}.nc")

    # Step 1: Download
    download_file(url, local_path)

    # Step 2: Open and extract variable
    ds = xr.open_dataset(local_path)
    da, raw_da = extract_variable(ds, var)

    # Step 3: Apply scaling + offset
    da = apply_scale_and_offset(da, raw_da)

    # Step 4: Crop to watershed
    da = crop_to_basin(da, basin)

    # Step 5: Basin average
    avg = da.mean(dim=["lat", "lon"], skipna=True)

    # --- Fix: Drop unnecessary coordinates like 'spatial_ref' ---
    drop_coords = [coord for coord in avg.coords if coord not in ['day', 'time']]
    avg = avg.drop_vars(drop_coords, errors="ignore")

    # Step 6: Convert to DataFrame
    df = avg.to_dataframe(name=var)

    return df


# ----------------------------
# Step 4: Process All Variables by Year
# ----------------------------

df_all_years = []

for year in tqdm(years, desc="📆 Processing years"):
    df_year = pd.DataFrame()
    for var in variables:
        try:
            df_var = process_variable_year(var, year)
            if df_year.empty:
                df_year = df_var
            else:
                df_year = df_year.join(df_var, how="outer")
        except Exception as e:
            print(f"⚠️ Error processing {var}_{year}: {e}")
    df_all_years.append(df_year)

# ----------------------------
# Step 5: Final Output
# ----------------------------

df_final = pd.concat(df_all_years)
df_final.index.name = "date"
df_final.sort_index(inplace=True)
df_final.to_csv(output_csv)
print(f"\n✅ Saved optimized basin-averaged climate data to: {output_csv}")
