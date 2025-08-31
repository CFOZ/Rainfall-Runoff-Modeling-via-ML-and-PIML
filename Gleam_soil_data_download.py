# --- gleam_download_and_clip.py ---
import os
import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import paramiko

# --- Configuration ---
years = range(2000, 2014)
variables = ['Ep', 'SMs', 'SMrz']
archive = 'v4.2a'
output_dir = r"G:\GLEAM4_data"
shapefile_path = r"G:\PIML_CODE\watershed.shp"

SFTP_HOST = "hydras.ugent.be"
SFTP_PORT = 2225
SFTP_USER = "gleamuser"
SFTP_PASS = "GLEAM4#h-cel_924"

os.makedirs(output_dir, exist_ok=True)

def create_sftp_session():
    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USER, password=SFTP_PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print(" SFTP connection established.")
        return sftp, transport
    except Exception as e:
        print(" SFTP connection failed:", e)
        return None, None

def download_and_clip(var, year, basin_gdf, sftp):
    fname = f"{var}_{year}_GLEAM_{archive}.nc"
    remote_path = f"data/{archive}/daily/{year}/{fname}"
    local_path = os.path.join(output_dir, fname)
    clipped_fname = f"{var}_{year}_GLEAM_{archive}_clipped.nc"
    clipped_path = os.path.join(output_dir, clipped_fname)

    if not os.path.exists(local_path):
        print(f" Downloading {remote_path} ...")
        try:
            sftp.get(remote_path, local_path)
            print(" Downloaded:", fname)
        except Exception as e:
            print(" SFTP download error:", e)
            return
    else:
        print(f" Cached locally: {fname}")

    try:
        ds = xr.open_dataset(local_path, engine="netcdf4")
        ds = ds.rename({'lat': 'y', 'lon': 'x'})
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

        da = ds[var]
        if '_FillValue' in da.attrs:
            da = da.where(da != da.attrs['_FillValue'], np.nan)

        clipped = da.rio.clip(basin_gdf.geometry.values, basin_gdf.crs, drop=True)
        clipped.rio.write_crs("EPSG:4326", inplace=True)
        clipped.to_netcdf(clipped_path)

        print(f" Clipped data saved to {clipped_fname}")
        # os.remove(local_path)
        # print(f" Deleted original file: {fname}")
    except Exception as e:
        print(f" Error processing {fname}:", e)

# --- Load Watershed ---
try:
    basin = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    print(" Watershed shapefile loaded.")
except Exception as e:
    print(" Failed to load shapefile:", e)
    exit()

# --- Execute Download & Clip ---
sftp, transport = create_sftp_session()
if not sftp:
    exit()

for var in variables:
    print(f"\n Downloading & clipping variable: {var}")
    for year in years:
        download_and_clip(var, year, basin, sftp)

sftp.close()
transport.close()
print("\n All files downloaded and clipped.")

