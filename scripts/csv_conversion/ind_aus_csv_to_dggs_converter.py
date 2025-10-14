import os
import time
import logging
from datetime import datetime
import multiprocessing
from functools import partial
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize


class IndAusCSVToDGGSConverter:
    """
    Convert India/Australia coal methane emissions CSVs to DGGS grid values.

    Inputs (per country): CSV with three columns (no header): lat, lon, value (ton/year), points at grid centers.
    Output: Country-specific CSV with DGGS-wide format and Year=2018; IPCC2006 code 1B1a.

    Steps:
    - Load CSV and pivot to a regular grid raster (lat×lon) in EPSG:4326.
    - Rasterize DGGS polygons to a label raster aligned with the grid.
    - Aggregate per-pixel values to DGGS cells via numpy.bincount.
    - Apply scaling to ensure total preserved.
    - Log comprehensive progress, scaling, and file-code mapping (1B1a -> source CSV).
    """

    def __init__(self, root_folder, output_folder, num_cores=None):
        self.root_folder = root_folder
        self.output_folder = output_folder
        self.year = 2018
        self.ipcc_code = '1B1a'

        if num_cores is None:
            self.num_cores = int(os.environ.get('NUM_CORES', 8))
        else:
            self.num_cores = num_cores

        self._setup_logging()
        os.makedirs(self.output_folder, exist_ok=True)

    def _setup_logging(self):
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"ind_aus_csv_to_dggs_conversion_{timestamp}.log"
        log_path = os.path.join(log_folder, log_filename)

        self.logger = logging.getLogger(f"ind_aus_csv_converter_{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        self.log_path = log_path
        self.log_message(f"Logging initialized. Log file: {log_path}")

    def log_message(self, message):
        print(message)
        self.logger.info(message)

    def _load_country_config(self):
        return [
            {
                'country': 'Australia',
                'csv_file': 'gridded_aus_emissions.csv',
                'dggs_geojson': 'data/geojson/global_countries_dggs_merge/Australia_AUS_grid.geojson',
                'output_name': f'Australia_Coal_DGGS_methane_emissions_{self.year}.csv',
            },
            {
                'country': 'India',
                'csv_file': 'gridded_ind_emissions.csv',
                'dggs_geojson': 'data/geojson/global_countries_dggs_merge/India_IND_grid.geojson',
                'output_name': f'India_Coal_DGGS_methane_emissions_{self.year}.csv',
            },
        ]

    def _read_csv_points(self, csv_path):
        df = pd.read_csv(csv_path, header=None, names=['lat', 'lon', 'value'])
        return df

    def _build_grid_from_points(self, df):
        lats = np.sort(df['lat'].unique())
        lons = np.sort(df['lon'].unique())
        # Detect resolution for logging
        if len(lats) > 1:
            dlat = float(np.round(np.min(np.diff(lats)), 6))
        else:
            dlat = np.nan
        if len(lons) > 1:
            dlon = float(np.round(np.min(np.diff(lons)), 6))
        else:
            dlon = np.nan
        self.log_message(f"  Grid size: lat={len(lats)}, lon={len(lons)}, dlat≈{dlat}, dlon≈{dlon}")

        # Pivot to 2D array aligned lat×lon
        pivot = df.pivot(index='lat', columns='lon', values='value')
        # Ensure full grid ordering
        pivot = pivot.reindex(index=lats, columns=lons)
        data = pivot.values.astype(np.float32)

        # Build raster transform (lat/lon are centers)
        if np.isnan(dlon) or np.isnan(dlat):
            raise ValueError("Insufficient points to determine grid resolution")
        half_x = dlon / 2.0
        half_y = dlat / 2.0
        left = float(np.min(lons)) - half_x
        top = float(np.max(lats)) + half_y
        transform = from_origin(left, top, dlon, dlat)

        # Reverse latitude to top-down order and adjust for transform accordingly
        data = data[::-1, :]
        lats_desc = lats[::-1]
        return data, lons, lats_desc, transform

    def _rasterize_dggs_labels(self, dggs_gdf, transform, width, height):
        dggs_proj = dggs_gdf  # already EPSG:4326
        shapes = ((geom, idx + 1) for idx, geom in enumerate(dggs_proj.geometry))
        label_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='int32',
            all_touched=True,
        )
        return label_raster

    def _aggregate_to_dggs(self, data, label_raster):
        labels = label_raster.ravel()
        values = np.nan_to_num(data, nan=0.0).ravel()
        sums = np.bincount(labels, weights=values, minlength=self.num_cells + 1)[1:]
        return sums

    def _process_country(self, cfg):
        country = cfg['country']
        csv_path = os.path.join(self.root_folder, cfg['csv_file'])
        dggs_geojson = cfg['dggs_geojson']
        output_name = cfg['output_name']

        self.log_message("\n" + "=" * 60)
        self.log_message(f"PROCESSING {country.upper()} CSV: {os.path.basename(csv_path)}")
        self.log_message("=" * 60)

        if not os.path.exists(csv_path):
            self.log_message(f"Error: CSV not found: {csv_path}")
            return None
        if not os.path.exists(dggs_geojson):
            self.log_message(f"Warning: DGGS GeoJSON not found (expected by workflow): {dggs_geojson}")

        # Load DGGS grid (WGS84)
        dggs_gdf = gpd.read_file(dggs_geojson)
        if 'zoneID' not in dggs_gdf.columns:
            raise ValueError("DGGS GeoJSON must contain 'zoneID' column")
        self.num_cells = len(dggs_gdf)

        # Read CSV and build raster
        df = self._read_csv_points(csv_path)
        data, lons, lats_desc, transform = self._build_grid_from_points(df)
        height, width = data.shape
        self.log_message(f"  Raster shape: {height}×{width}")

        # Rasterize DGGS labels
        self.log_message("  Rasterizing DGGS cells to label raster...")
        label_raster = self._rasterize_dggs_labels(dggs_gdf, transform, width, height)

        # Aggregate to DGGS
        self.log_message("  Aggregating raster to DGGS cells...")
        dggs_values = self._aggregate_to_dggs(data, label_raster)

        # Calculate totals and apply scaling now (single variable)
        total_raster_value = float(np.sum(data[label_raster > 0]))
        total_weighted_value = float(np.sum(dggs_values))
        if total_weighted_value > 0.0 and total_raster_value > 0.0:
            scaling_factor = total_raster_value / total_weighted_value
            dggs_values = dggs_values * scaling_factor
            self.log_message(f"  Applied scaling factor: {scaling_factor:.6f}")
            self.log_message(f"  Raster total (intersecting pixels): {total_raster_value:.6f}")
            self.log_message(f"  DGGS total (after scaling): {float(np.sum(dggs_values)):.6f}")
        else:
            self.log_message("  Skipped scaling (zero total encountered)")

        # Build output dataframe
        result_df = dggs_gdf[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})
        result_df[self.ipcc_code] = dggs_values
        result_df['Year'] = self.year
        
        # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
        value_cols = [c for c in result_df.columns if c not in ['dggsID', 'Year']]
        self.log_message(f"  Step 1: Filtering small values (< 1e-6 Mg = 1g)")
        small_values_before = 0
        for col in value_cols:
            small_mask = result_df[col] < 1e-6
            small_count = small_mask.sum()
            small_values_before += small_count
            result_df[col] = result_df[col].where(result_df[col] >= 1e-6, 0.0)
        self.log_message(f"    Set {small_values_before} small values to zero across all columns")
        
        # Step 2: Remove rows where all values are 0 (except dggsID, Year)
        rows_before = len(result_df)
        result_df = result_df[~(result_df[value_cols] == 0).all(axis=1)]
        rows_after = len(result_df)
        self.log_message(f"  Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
        
        # Steps complete (only Step 1 and Step 2 retained by design)

        out_path = os.path.join(self.output_folder, output_name)
        result_df.to_csv(out_path, index=False)
        self.log_message(f"  Results saved to: {out_path}")
        self.log_message(f"  Output shape: {result_df.shape}")

        # File-IPCC mapping
        self.log_message("\nFile-IPCC Code Mapping:")
        self.log_message(f"  {self.ipcc_code} -> {os.path.basename(csv_path)}")

        return out_path

    def process_all(self):
        start_time = time.time()
        cfgs = self._load_country_config()
        outputs = []
        for cfg in cfgs:
            out = self._process_country(cfg)
            if out:
                outputs.append(out)
        total_time = time.time() - start_time
        self.log_message("\n" + "=" * 60)
        self.log_message("PROCESSING SUMMARY")
        self.log_message("=" * 60)
        self.log_message(f"Total outputs: {len(outputs)}")
        self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        return outputs


def main():
    root_folder = \
        "/home/mingke.li/GridInventory/2018_India_Australia_coal_mine_methane_emissions"
    output_folder = "output"

    if not os.path.exists(root_folder):
        print(f"Error: Root folder not found: {root_folder}")
        return

    converter = IndAusCSVToDGGSConverter(
        root_folder=root_folder,
        output_folder=output_folder,
    )

    try:
        outputs = converter.process_all()
        converter.log_message("\nCSV → DGGS conversion completed successfully!")
        converter.log_message(f"Outputs: {outputs}")
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


