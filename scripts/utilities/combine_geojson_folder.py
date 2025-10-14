import os
import geopandas as gpd
import pandas as pd

# === Paths ===
input_folder = r"data/geojson/global_countries_dggs_merge"
output_file = r"data/geojson/global_countries_dggs_merge.geojson"

# === Expected schema ===
expected_cols = {"NAME", "GID", "zoneID", "geometry"}

# === Collect GeoJSON files ===
geojson_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".geojson")]
if not geojson_files:
    raise FileNotFoundError(f"No GeoJSON files found in {input_folder}")

print(f"Found {len(geojson_files)} GeoJSON files to merge\n")

# === Memory-efficient processing: process in batches ===
BATCH_SIZE = 20  # Process 20 files at a time to avoid memory issues
merged_gdf = None
reference_crs = None

for batch_start in range(0, len(geojson_files), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(geojson_files))
    batch_files = geojson_files[batch_start:batch_end]
    
    print(f"Processing batch {batch_start//BATCH_SIZE + 1}: files {batch_start+1}-{batch_end}")
    
    # Read files in current batch
    gdf_list = []
    for i, file in enumerate(batch_files, batch_start + 1):
        file_path = os.path.join(input_folder, file)
        print(f"  [{i}/{len(geojson_files)}] Reading {file_path}")
        
        try:
            gdf = gpd.read_file(file_path)
            
            # Schema check
            if not expected_cols.issubset(gdf.columns):
                raise ValueError(f" {file_path} does not match expected schema {expected_cols}. "
                                 f"Found columns: {list(gdf.columns)}")
            
            # Set reference CRS from first file
            if reference_crs is None:
                reference_crs = gdf.crs
            else:
                # CRS alignment
                gdf = gdf.to_crs(reference_crs)
            
            gdf_list.append(gdf)
            
        except Exception as e:
            print(f"  ERROR reading {file_path}: {e}")
            continue
    
    # Combine current batch
    if gdf_list:
        batch_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=reference_crs)
        
        # Merge with previous batches
        if merged_gdf is None:
            merged_gdf = batch_gdf
        else:
            merged_gdf = gpd.GeoDataFrame(pd.concat([merged_gdf, batch_gdf], ignore_index=True), crs=reference_crs)
        
        print(f"  Batch completed. Current total records: {len(merged_gdf)}")
        # Clear batch data to free memory
        del gdf_list, batch_gdf

# === Drop duplicates on zoneID ===
if merged_gdf is not None:
    before = len(merged_gdf)
    dupes = merged_gdf[merged_gdf.duplicated("zoneID", keep=False)]
    
    if not dupes.empty:
        print(f"\n Found {len(dupes)} duplicate rows based on 'zoneID'")
        print(dupes[["NAME", "GID", "zoneID"]].head())
        
    merged_gdf = merged_gdf.drop_duplicates(subset="zoneID", keep="first")
    after = len(merged_gdf)
    
    print(f"\nRemoved {before - after} duplicate records based on 'zoneID'")
    print(f"Final record count: {after}")
    
    # === Save merged file ===
    merged_gdf.to_file(output_file, driver="GeoJSON")
    print(f"\n Combined GeoJSON saved to {output_file}")
else:
    print("ERROR: No valid GeoJSON files were processed!")
    exit(1)
