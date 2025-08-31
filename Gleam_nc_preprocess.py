import os
import xarray as xr
import pandas as pd

# --- Configuration ---
years = range(2000, 2021)
variables = ['E', 'Ep', 'SMs', 'SMrz']
archive = 'v4.2a'
output_dir = r"D:\Rainfall_runoff_Aayush\GLEAM4_data"
output_csv = os.path.join(output_dir, "leaf_river_basin_GLEAM_timeseries.csv")

# --- Process Clipped Files ---
dfs = []

for var in variables:
    print(f"\n📊 Processing clipped variable: {var}")
    yearly_data = []

    for year in years:
        fname = f"{var}_{year}_GLEAM_{archive}_clipped.nc"
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"⚠️ File missing: {fname}")
            continue

        try:
            # Open dataset and select the variable
            ds = xr.open_dataset(path)
            if var not in ds.data_vars:
                print(f"⚠️ Variable '{var}' not found in {fname}")
                continue

            da = ds[var]
            mean_ts = da.mean(dim=["y", "x"], skipna=True).to_dataframe(name=var)
            yearly_data.append(mean_ts)
        except Exception as e:
            print(f"❌ Failed to process {fname}:", e)

    if yearly_data:
        full_var_df = pd.concat(yearly_data)
        dfs.append(full_var_df)
    else:
        print(f"⚠️ No data found for variable: {var}")

# --- Export Time Series CSV ---
if dfs:
    final_df = pd.concat(dfs, axis=1)
    final_df.index.name = "date"
    final_df.to_csv(output_csv)
    print(f"\n✅ GLEAM time series saved to: {output_csv}")
else:
    print("❌ No clipped data processed.")
