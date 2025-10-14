import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import multiprocessing
from functools import partial
import time
import logging
from datetime import datetime


class SwissNetCDFToDGGSConverterAggregated:
    """
    Convert Swiss (SGHGI) NetCDF to DGGS grid values using a raster-first,
    area-weighted approach with variable aggregation by IPCC2006 codes.

    Swiss specifics:
    - Units for variables: g m^-2 yr^-1 (grams per square meter per year)
    - Projection: EPSG:21781 (CH1903 / LV03), original x/y are projected
    - Resolution: 500 m × 500 m per pixel
    - Year: 2011 (fixed)

    Target output units: Mg/year

    Conversion to per-pixel Mg/year:
      mass_Mg_per_year = (value_g_per_m2_per_year * pixel_area_m2) / 1e6
    where pixel_area_m2 = 500 m × 500 m = 250,000 m^2 and 1 Mg = 1,000,000 g.
    """

    def __init__(self, netcdf_folder, geojson_path, output_folder, num_cores=None):
        self.netcdf_folder = netcdf_folder
        self.geojson_path = geojson_path
        self.output_folder = output_folder

        if num_cores is None:
            self.num_cores = int(os.environ.get('NUM_CORES', 8))
        else:
            self.num_cores = num_cores

        # Constants
        self.SECONDS_PER_YEAR = 365 * 24 * 3600
        self.PIXEL_SIZE_M = 500.0
        self.PIXEL_AREA_M2 = self.PIXEL_SIZE_M * self.PIXEL_SIZE_M  # 250,000 m^2

        # Logging
        self._setup_logging()
        self.log_message("Loading IPCC2006 variable lookup table for Switzerland...")
        self._load_ipcc_lookup()

        # Load DGGS grid in WGS84 then project to EPSG:21781 for correct area-weight intersections
        self.log_message("Loading DGGS grid and projecting to EPSG:21781 (CH1903 / LV03)...")
        self.dggs_grid_wgs84 = gpd.read_file(self.geojson_path)
        if 'zoneID' not in self.dggs_grid_wgs84.columns:
            raise ValueError("GeoJSON file must contain 'zoneID' column")
        self.dggs_grid_proj = self.dggs_grid_wgs84.to_crs('EPSG:21781')
        self.log_message(f"Loaded {len(self.dggs_grid_proj)} DGGS cells")

        os.makedirs(self.output_folder, exist_ok=True)

        # Find NetCDF files (.nc, .nc4)
        self.netcdf_files = [f for f in os.listdir(self.netcdf_folder) if f.endswith('.nc') or f.endswith('.nc4')]
        self.log_message(f"Found {len(self.netcdf_files)} NetCDF file(s)")

        # Spatial index (bounds list) for projected DGGS grid
        self.log_message("Creating spatial index for projected DGGS cells...")
        self._create_spatial_index()

    def _setup_logging(self):
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"swiss_netcdf_to_dggs_conversion_{timestamp}.log"
        log_path = os.path.join(log_folder, log_filename)

        self.logger = logging.getLogger(f"swiss_converter_{timestamp}")
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

    def _load_ipcc_lookup(self):
        lookup_path = "data/lookup/swiss_netcdf_variable_lookup.csv"
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"IPCC lookup file not found: {lookup_path}")
        self.ipcc_lookup = pd.read_csv(lookup_path)
        if 'variable' not in self.ipcc_lookup.columns or 'IPCC2006' not in self.ipcc_lookup.columns:
            raise ValueError("Lookup CSV must contain 'variable' and 'IPCC2006' columns")
        self.variable_to_ipcc = dict(zip(self.ipcc_lookup['variable'], self.ipcc_lookup['IPCC2006']))
        self.log_message(f"Loaded Swiss IPCC lookup with {len(self.variable_to_ipcc)} mappings")

    def _create_spatial_index(self):
        self.dggs_bounds_proj = []
        for idx, row in self.dggs_grid_proj.iterrows():
            bounds = row.geometry.bounds
            self.dggs_bounds_proj.append((bounds, idx))
        self.log_message(f"  Created spatial index for {len(self.dggs_bounds_proj)} DGGS cells")

    def aggregate_variables_by_ipcc_code(self, nc_data):
        self.log_message("Aggregating variables by IPCC2006 codes...")
        exclude_vars = {'lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'time'}
        variables = [var for var in nc_data.variables if var not in exclude_vars]
        self.log_message(f"  Found {len(variables)} variables to process")

        ipcc_groups = {}
        unmapped = []
        for var in variables:
            if var in self.variable_to_ipcc:
                ipcc_code = self.variable_to_ipcc[var]
                ipcc_groups.setdefault(ipcc_code, []).append(var)
            else:
                unmapped.append(var)
                self.log_message(f"  Skipping variable '{var}' - not found in IPCC lookup table")
        if unmapped:
            self.log_message(f"  Skipped {len(unmapped)} variables not found in IPCC lookup table")

        aggregated = {}
        for ipcc_code, var_list in ipcc_groups.items():
            clean_stack = []
            for v in var_list:
                arr = nc_data[v].values
                if arr.ndim == 3:
                    arr = arr[0, :, :]
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, 0, None)
                clean_stack.append(arr)
            if not clean_stack:
                continue
            aggregated[ipcc_code] = np.sum(clean_stack, axis=0)
        return aggregated

    def convert_aggregated_to_raster(self, netcdf_path):
        self.log_message(f"Processing NetCDF: {os.path.basename(netcdf_path)}")
        ds = xr.open_dataset(netcdf_path)

        # Prefer projected x/y in meters
        x_name = 'x' if 'x' in ds.variables else None
        y_name = 'y' if 'y' in ds.variables else None

        # Fallback to lon/lat center coordinates (degrees) if no x/y; we only need centers to place 500 m boxes in projected CRS.
        if x_name is None or y_name is None:
            if 'lon' in ds.variables and 'lat' in ds.variables:
                lon = ds['lon'].values
                lat = ds['lat'].values
                # Expect 1D arrays; if not, raise with guidance
                if lon.ndim == 1 and lat.ndim == 1:
                    # We cannot compute projected x/y without transforming; however, we only need an index-aligned sequence for columns/rows.
                    # We'll store degrees for now and handle transform via pre-known pixel size in meters centered at transformed points.
                    # Note: We won't transform each pixel center to EPSG:21781 to avoid heavy overhead; rely on presence of x/y where possible.
                    pass
                else:
                    ds.close()
                    raise ValueError("Unsupported lon/lat dimensionality; expected 1D lon/lat arrays.")
            else:
                ds.close()
                raise ValueError("Could not locate projected x/y or lon/lat coordinates in NetCDF")

        aggregated = self.aggregate_variables_by_ipcc_code(ds)

        # Build raster_data with centers in projected meters if x/y exist; else keep lon/lat and transform per-pixel lazily (slower path).
        raster_data = {}
        for ipcc_code, data_array in aggregated.items():
            if data_array.ndim == 3:
                data_array = data_array[0, :, :]
            elif data_array.ndim != 2:
                ds.close()
                raise ValueError(f"Unexpected data dims for {ipcc_code}: {data_array.shape}")

            # Attempt to read units for logging
            try:
                units = ds[ipcc_code].attrs.get('units', '') if ipcc_code in ds.variables else ''
            except Exception:
                units = ''
            if units:
                self.log_message(f"  Variable group {ipcc_code} units (first var's units if available): {units}")

            # Convert g m^-2 yr^-1 to Mg/year per pixel
            # mass_Mg_per_year = (value_g_per_m2_per_year * pixel_area_m2) / 1e6
            mass_per_pixel = data_array * self.PIXEL_AREA_M2 / 1e6
            mass_per_pixel = np.nan_to_num(mass_per_pixel, nan=0.0)
            if np.any(mass_per_pixel < 0):
                mass_per_pixel = np.clip(mass_per_pixel, 0, None)

            entry = {'data': mass_per_pixel, 'pixel_size': self.PIXEL_SIZE_M}
            if x_name and y_name:
                entry['x'] = ds[x_name].values
                entry['y'] = ds[y_name].values
                entry['coords_crs'] = 'EPSG:21781'
            else:
                entry['lon'] = ds['lon'].values
                entry['lat'] = ds['lat'].values
                entry['coords_crs'] = 'EPSG:4326'
            raster_data[ipcc_code] = entry

        ds.close()
        return raster_data

    def calculate_weighted_values_raster_first(self, raster_data, ipcc_code):
        self.log_message(f"    Processing {ipcc_code} using raster-first approach...")
        results = np.zeros(len(self.dggs_grid_proj))
        distributed_raster_total = 0.0

        data = raster_data['data']
        non_zero_mask = data > 0
        if not np.any(non_zero_mask):
            self.log_message(f"      No non-zero pixels found in {ipcc_code}")
            return results.tolist()

        non_zero_coords = np.where(non_zero_mask)
        num_non_zero = len(non_zero_coords[0])
        self.log_message(f"      Found {num_non_zero} non-zero pixels to process")

        # Determine how to get center coordinates for pixel boxes
        use_projected = ('x' in raster_data) and ('y' in raster_data)
        half = raster_data['pixel_size'] / 2.0

        if not use_projected:
            # Fallback: lon/lat present, we need to transform to EPSG:21781 per pixel center. To avoid heavy import at module level, import here.
            from pyproj import Transformer
            transformer = Transformer.from_crs('EPSG:4326', 'EPSG:21781', always_xy=True)

        start_time = time.time()
        processed = 0
        for i in range(num_non_zero):
            row, col = non_zero_coords[0][i], non_zero_coords[1][i]
            pixel_value = data[row, col]

            if use_projected:
                pixel_x = raster_data['x'][col]
                pixel_y = raster_data['y'][row]
            else:
                pixel_lon = raster_data['lon'][col]
                pixel_lat = raster_data['lat'][row]
                pixel_x, pixel_y = transformer.transform(pixel_lon, pixel_lat)

            pixel_geom = box(pixel_x - half, pixel_y - half, pixel_x + half, pixel_y + half)

            intersecting = self._find_intersecting_cells(pixel_geom)
            if intersecting:
                total_intersection_area = 0.0
                intersection_areas = []
                for cell_idx in intersecting:
                    try:
                        cell_geom = self.dggs_grid_proj.iloc[cell_idx].geometry
                        inter = pixel_geom.intersection(cell_geom)
                        area = inter.area
                        intersection_areas.append(area)
                        total_intersection_area += area
                    except Exception:
                        intersection_areas.append(0.0)

                if total_intersection_area > 0:
                    for j, cell_idx in enumerate(intersecting):
                        area_ratio = intersection_areas[j] / total_intersection_area if total_intersection_area > 0 else 0.0
                        results[cell_idx] += pixel_value * area_ratio
                    distributed_raster_total += float(pixel_value)
                else:
                    share = pixel_value / len(intersecting)
                    for cell_idx in intersecting:
                        results[cell_idx] += share
                    distributed_raster_total += float(pixel_value)

            processed += 1
            if processed % 2000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (num_non_zero - processed) / rate if rate > 0 else 0
                self.log_message(
                    f"        Processed {processed}/{num_non_zero} ({processed/num_non_zero*100:.1f}%) - ETA: {remaining:.1f}s"
                )

        total_time = time.time() - start_time
        self.log_message(f"      Completed {num_non_zero} pixels in {total_time:.2f}s")

        # Scale to match total of intersecting pixels (per variable)
        total_raster_value = float(distributed_raster_total)
        total_weighted_value = float(np.sum(results))
        if total_weighted_value > 0.0 and total_raster_value > 0.0:
            scaling_factor = total_raster_value / total_weighted_value
            results = results * scaling_factor
            self.log_message(f"      Applied scaling factor: {scaling_factor:.6f}")
            self.log_message(f"      Raster total (intersecting pixels): {total_raster_value:.6f} | DGGS total (after scaling): {float(np.sum(results)):.6f}")
        else:
            self.log_message("      Skipped scaling (zero total encountered)")

        return results.tolist()

    def _find_intersecting_cells(self, pixel_geom):
        intersecting = []
        bounds1 = pixel_geom.bounds
        for bounds2, cell_idx in self.dggs_bounds_proj:
            if not (bounds1[2] < bounds2[0] or bounds1[0] > bounds2[2] or bounds1[3] < bounds2[1] or bounds1[1] > bounds2[3]):
                try:
                    if pixel_geom.intersects(self.dggs_grid_proj.iloc[cell_idx].geometry):
                        intersecting.append(cell_idx)
                except Exception:
                    continue
        return intersecting

    def calculate_weighted_values_parallel_raster_first(self, raster_data_dict, ipcc_codes):
        self.log_message(f"    Processing {len(ipcc_codes)} aggregated IPCC2006 codes in parallel...")
        num_processes = min(len(ipcc_codes), self.num_cores)
        self.log_message(f"    Using {num_processes} parallel processes")

        process_func = partial(self._process_single_ipcc_code_raster_first)
        args_list = [(ipcc_code, raster_data_dict[ipcc_code]) for ipcc_code in ipcc_codes]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_func, args_list)

        out = {}
        for ipcc_code, values in results:
            out[ipcc_code] = values
        return out

    def _process_single_ipcc_code_raster_first(self, args):
        ipcc_code, raster_data = args
        try:
            return ipcc_code, self.calculate_weighted_values_raster_first(raster_data, ipcc_code)
        except Exception as e:
            print(f"      Error processing {ipcc_code}: {e}")
            return ipcc_code, [0.0] * len(self.dggs_grid_proj)

    def process_all_netcdf_files(self):
        start_time = time.time()
        self.log_message(f"Processing {len(self.netcdf_files)} NetCDF file(s) with IPCC aggregation...")

        if len(self.netcdf_files) == 0:
            self.log_message("No NetCDF files found to process.")
            return None

        year = 2011
        all_frames = []
        for idx, filename in enumerate(self.netcdf_files):
            self.log_message("\n" + "=" * 60)
            self.log_message(f"PROCESSING FILE {idx + 1}/{len(self.netcdf_files)}: {filename}")
            self.log_message("=" * 60)

            netcdf_path = os.path.join(self.netcdf_folder, filename)
            if not os.path.exists(netcdf_path):
                self.log_message(f"Warning: NetCDF file not found: {netcdf_path}")
                continue

            try:
                # Base dataframe with DGGS IDs
                result_df = self.dggs_grid_wgs84[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})

                raster_data = self.convert_aggregated_to_raster(netcdf_path)
                self.log_message(f"Processing {len(raster_data)} aggregated IPCC2006 codes...")

                if len(raster_data) > 1:
                    ipcc_codes = list(raster_data.keys())
                    self.log_message(f"  Processing {len(ipcc_codes)} IPCC codes in parallel...")
                    weighted_dict = self.calculate_weighted_values_parallel_raster_first(raster_data, ipcc_codes)
                    for ipcc_code, values in weighted_dict.items():
                        result_df[ipcc_code] = values
                elif len(raster_data) == 1:
                    ipcc_code = list(raster_data.keys())[0]
                    values = self.calculate_weighted_values_raster_first(raster_data[ipcc_code], ipcc_code)
                    result_df[ipcc_code] = values
                else:
                    self.log_message("No aggregated variables found after lookup. Skipping.")
                    continue

                # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
                value_cols = [c for c in result_df.columns if c != 'dggsID']
                self.log_message(f"  Step 1: Filtering small values (< 1e-6 Mg = 1g)")
                small_values_before = 0
                for col in value_cols:
                    small_mask = result_df[col] < 1e-6
                    small_count = small_mask.sum()
                    small_values_before += small_count
                    result_df[col] = result_df[col].where(result_df[col] >= 1e-6, 0.0)
                self.log_message(f"    Set {small_values_before} small values to zero across all columns")
                
                # Step 2: Remove rows with all zeros across IPCC columns
                rows_before = len(result_df)
                result_df = result_df[~(result_df[value_cols] == 0).all(axis=1)]
                rows_after = len(result_df)
                self.log_message(f"  Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
                result_df['Year'] = year

                all_frames.append(result_df)
                self.log_message(f"  Completed {filename} with shape {result_df.shape}")
                # File-IPCC mapping log
                self.log_message("\nFile-IPCC Code Mapping:")
                for ipcc_code in [c for c in result_df.columns if c not in ['dggsID', 'Year']]:
                    self.log_message(f"  {ipcc_code} -> {filename}")
            except Exception as e:
                self.log_message(f"Error processing file {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_frames:
            self.log_message("No dataframes to combine - processing failed for all files")
            return None

        combined = pd.concat(all_frames, ignore_index=True)
        self.log_message(f"Combined shape: {combined.shape}")
        value_cols = [c for c in combined.columns if c not in ['dggsID', 'Year']]
        
        # Get value columns for processing
        
        # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
        self.log_message(f"Final processing - Step 1: Filtering small values (< 1e-6 Mg = 1g)")
        small_values_before = 0
        for col in value_cols:
            small_mask = combined[col] < 1e-6
            small_count = small_mask.sum()
            small_values_before += small_count
            combined[col] = combined[col].where(combined[col] >= 1e-6, 0.0)
        self.log_message(f"  Set {small_values_before} small values to zero across all columns")
        
        # Step 2: Remove rows with all zeros
        rows_before = len(combined)
        combined = combined[~(combined[value_cols] == 0).all(axis=1)]
        rows_after = len(combined)
        self.log_message(f"Final processing - Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
        
        self.log_message(f"After removing zero rows: {len(combined)} rows remain")

        output_filename = "Switzerland_DGGS_methane_emissions_2011.csv"
        output_path = os.path.join(self.output_folder, output_filename)
        combined.to_csv(output_path, index=False)
        total_time = time.time() - start_time
        self.log_message(f"\nResults saved to: {output_path}")
        self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        return output_path


def main():
    """Entry point for Swiss SGHGI NetCDF → DGGS conversion."""
    # Windows path provided by user
    netcdf_folder = "/home/mingke.li/GridInventory/2011_Swiss_Greenhouse_Gas Inventory_SGHGI"
    geojson_path = "data/geojson/regional_grid/switzerland_grid.geojson"
    output_folder = "output"

    if not os.path.exists(netcdf_folder):
        print(f"Error: NetCDF folder not found: {netcdf_folder}")
        return
    if not os.path.exists(geojson_path):
        print(f"Warning: GeoJSON path not found (expected by workflow): {geojson_path}")

    converter = SwissNetCDFToDGGSConverterAggregated(
        netcdf_folder=netcdf_folder,
        geojson_path=geojson_path,
        output_folder=output_folder,
    )

    try:
        output_path = converter.process_all_netcdf_files()
        converter.log_message("\nConversion completed successfully!")
        converter.log_message(f"Output file: {output_path}")
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


