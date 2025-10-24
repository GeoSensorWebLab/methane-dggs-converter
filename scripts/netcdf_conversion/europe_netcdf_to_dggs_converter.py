import os
import re
import time
import logging
from datetime import datetime
from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin


class EuropeNetCDFToDGGSConverterAggregated:
    """
    Convert CAMS-REG-ANT Europe NetCDF to DGGS grid values using aggregated raster-first processing.

    Key details:
    - Input NetCDF has dimensions: time, lat, lon and ~14 data variables.
    - Time values are day counts; treat time index i as year (2005 + i) covering 2005..2022.
    - Resolution: lat 0.05 degree, lon 0.10 degree (set pixel size accordingly).
    - Variables aggregated to IPCC2006 codes via lookup at data/lookup/cams-reg_netcdf_variable_lookup.csv.
    - Units are Tg; convert to Mg (1 Tg = 1e6 Mg). Area not needed.
    - Uses raster-first area-weighted distribution, parallelized across IPCC codes, with scaling.
    - Saves per-year CSVs under test/test_europe_csv and final combined CSV under output.
    """

    def __init__(self, netcdf_path, grid_parquet_path, output_folder, num_cores=None):
        self.netcdf_path = netcdf_path
        self.grid_parquet_path = grid_parquet_path
        self.output_folder = output_folder

        if num_cores is None:
            self.num_cores = int(os.environ.get("NUM_CORES", 8))
        else:
            self.num_cores = num_cores

        # Logging
        self._setup_logging()

        # Lookup table (variable -> IPCC2006)
        self.log_message("Loading IPCC2006 variable lookup table...")
        self._load_ipcc_lookup()

        # DGGS grid (parquet)
        self.log_message("Loading DGGS grid (parquet)...")
        self.dggs_grid = gpd.read_parquet(self.grid_parquet_path)
        self.log_message(f"Loaded {len(self.dggs_grid)} DGGS cells")

        if "zoneID" not in self.dggs_grid.columns:
            raise ValueError("DGGS parquet must contain 'zoneID' column")

        os.makedirs(self.output_folder, exist_ok=True)

        # Spatial index (simple bounds list)
        self.log_message("Creating spatial index for DGGS cells...")
        self._create_spatial_index()

        # Lazy-built DGGS zone index raster (pixel -> DGGS row index+1); built when NetCDF is opened
        self.zone_index_raster = None

    def _setup_logging(self):
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"europe_netcdf_to_dggs_conversion_{timestamp}.log"
        log_path = os.path.join(log_folder, log_filename)

        self.logger = logging.getLogger(f"europe_converter_{timestamp}")
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

    def _create_spatial_index(self):
        self.dggs_bounds = []
        for idx, row in self.dggs_grid.iterrows():
            geom = row.geometry
            bounds = geom.bounds
            self.dggs_bounds.append((bounds, idx))
        self.log_message(f"  Created spatial index for {len(self.dggs_bounds)} DGGS cells")

    def _load_ipcc_lookup(self):
        lookup_path = "data/lookup/cams-reg_netcdf_variable_lookup.csv"
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"IPCC lookup file not found: {lookup_path}")
        self.ipcc_lookup = pd.read_csv(lookup_path)
        if 'variable' not in self.ipcc_lookup.columns or 'IPCC2006' not in self.ipcc_lookup.columns:
            raise ValueError("Lookup CSV must contain 'variable' and 'IPCC2006' columns")
        self.variable_to_ipcc = dict(zip(self.ipcc_lookup['variable'], self.ipcc_lookup['IPCC2006']))
        self.log_message(f"Loaded IPCC lookup table with {len(self.variable_to_ipcc)} variable mappings")

    def _bounds_intersect(self, b1, b2):
        return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

    def _find_intersecting_cells(self, pixel_geom):
        intersecting_cells = []
        pixel_bounds = pixel_geom.bounds
        for bounds, cell_idx in self.dggs_bounds:
            if self._bounds_intersect(pixel_bounds, bounds):
                try:
                    if pixel_geom.intersects(self.dggs_grid.iloc[cell_idx].geometry):
                        intersecting_cells.append(cell_idx)
                except Exception:
                    continue
        return intersecting_cells

    def _extract_years(self, nc_data):
        time_len = int(nc_data.dims.get('time', len(nc_data['time'])))
        # Map index 0..time_len-1 -> 2005..(2005+time_len-1)
        years = [2005 + i for i in range(time_len)]
        return years

    def _aggregate_variables_by_ipcc_for_time(self, nc_data, time_index):
        """
        Aggregate variables for a single time step into IPCC2006 codes.
        Skips variables not in lookup and logs them.
        Returns dict: { ipcc_code: 2D numpy array (lat, lon) in Tg }
        """
        exclude_vars = {'lat', 'lon', 'time'}
        variables = [v for v in nc_data.variables if v not in exclude_vars]

        ipcc_groups = {}
        unmapped = []
        for var in variables:
            if var in self.variable_to_ipcc:
                ipcc = self.variable_to_ipcc[var]
                ipcc_groups.setdefault(ipcc, []).append(var)
            else:
                unmapped.append(var)

        if unmapped:
            self.log_message(f"  Skipping {len(unmapped)} variables not in IPCC lookup: {unmapped}")

        aggregated = {}
        for ipcc_code, var_list in ipcc_groups.items():
            stack = []
            for var in var_list:
                arr = nc_data[var].values
                # Expect arr dims: (time, lat, lon) or (lat, lon)
                if arr.ndim == 3:
                    arr = arr[time_index, :, :]
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, 0, None)
                stack.append(arr)
            if not stack:
                continue
            aggregated_arr = np.sum(stack, axis=0)
            aggregated[ipcc_code] = aggregated_arr  # still in Tg
            self.log_message(f"    Aggregated {ipcc_code} with {len(var_list)} vars -> {aggregated_arr.shape}")
        return aggregated

    def _rasterize_aggregated_for_time(self, nc_data, aggregated_dict):
        """
        Convert aggregated arrays (Tg) to raster meta and Mg arrays for a single time step.
        Returns dict: { ipcc_code: { data, transform, crs, width, height, lat, lon } }
        """
        lat = nc_data['lat'].values
        lon = nc_data['lon'].values

        # Ensure north-to-south order for raster row order
        lat = lat[::-1]

        # Pixel sizes: lon 0.1, lat 0.05 degrees
        lon_res = 0.1
        lat_res = 0.05
        transform = from_origin(lon.min() - lon_res / 2.0, lat.max() + lat_res / 2.0, lon_res, lat_res)

        # Build DGGS label raster once aligned to the data grid for fast aggregation
        try:
            if self.zone_index_raster is None:
                self.log_message("  Building DGGS zone index raster (one-time)...")
                shapes = ((geom, idx + 1) for idx, geom in enumerate(self.dggs_grid.geometry))
                out_shape = (len(lat), len(lon))
                self.zone_index_raster = rasterize(
                    shapes=shapes,
                    out_shape=out_shape,
                    transform=transform,
                    fill=0,
                    dtype='int32',
                    all_touched=True,
                )
                self.log_message("    Zone index raster built")
        except Exception as e:
            self.log_message(f"  Warning: Failed to build zone index raster, will fallback to per-pixel intersections: {e}")

        raster_data = {}
        for ipcc_code, arr_tg in aggregated_dict.items():
            # Ensure 2D and reverse latitude to match raster row order
            if arr_tg.ndim != 2:
                raise ValueError(f"Unexpected array dims for {ipcc_code}: {arr_tg.shape}")
            arr_tg = arr_tg[::-1, :]

            # Unit conversion: Tg -> Mg (1 Tg = 1e6 Mg)
            arr_mg = arr_tg * 1e6
            arr_mg = np.nan_to_num(arr_mg, nan=0.0, posinf=0.0, neginf=0.0)
            arr_mg = np.clip(arr_mg, 0, None)

            raster_data[ipcc_code] = {
                'data': arr_mg,
                'transform': transform,
                'crs': 'EPSG:4326',
                'width': len(lon),
                'height': len(lat),
                'lat': lat,
                'lon': lon,
            }
        return raster_data

    def _calculate_weighted_values_raster_first(self, raster_data, ipcc_code):
        self.log_message(f"    Processing {ipcc_code} using raster-first approach...")
        results = np.zeros(len(self.dggs_grid))
        distributed_raster_total = 0.0  # sum of only intersecting pixel values (for diagnostics)

        data = raster_data['data']
        # Source total over the entire raster (post unit-conversion), like NYS pipeline
        src_total_value = float(np.sum(data))
        non_zero_mask = data > 0
        if not np.any(non_zero_mask):
            self.log_message(f"      No non-zero pixels found in {ipcc_code}")
            return results.tolist()

        # Fast path: label-raster bin counting (assign each pixel to a single DGGS cell)
        try:
            if self.zone_index_raster is not None and self.zone_index_raster.shape == data.shape:
                flat_labels = self.zone_index_raster.ravel()
                flat_values = data.ravel()
                sums = np.bincount(flat_labels, weights=flat_values, minlength=len(self.dggs_grid) + 1)
                results = sums[1:]  # drop background label 0
                assigned_total_before = float(np.sum(results))
                self.log_message(
                    f"      Assigned/src before scaling: {assigned_total_before:.6f}/{src_total_value:.6f} "
                    f"(fast label mode)"
                )
                if assigned_total_before > 0.0 and src_total_value > 0.0:
                    scaling_factor = src_total_value / assigned_total_before
                    results = results * scaling_factor
                    self.log_message(
                        f"      Applied conservation scaling: {scaling_factor:.12f} | "
                        f"DGGS total after scaling: {float(np.sum(results)):.6f}"
                    )
                else:
                    self.log_message("      Skipped scaling (zero total encountered)")
                return results.tolist()
        except Exception as e:
            self.log_message(f"      Warning: Fast label mode failed; falling back to per-pixel intersections: {e}")

        non_zero_coords = np.where(non_zero_mask)
        num_non_zero = len(non_zero_coords[0])
        self.log_message(f"      Found {num_non_zero} non-zero pixels to process")

        start_time = time.time()
        processed_pixels = 0

        # Pixel size from dataset (lon_res, lat_res)
        lon_res = 0.1
        lat_res = 0.05
        half_lon = lon_res / 2.0
        half_lat = lat_res / 2.0

        for i in range(num_non_zero):
            row, col = non_zero_coords[0][i], non_zero_coords[1][i]
            pixel_value = data[row, col]

            pixel_lon = raster_data['lon'][col]
            pixel_lat = raster_data['lat'][row]

            pixel_geom = box(
                pixel_lon - half_lon,
                pixel_lat - half_lat,
                pixel_lon + half_lon,
                pixel_lat + half_lat,
            )

            intersecting_cells = self._find_intersecting_cells(pixel_geom)
            if intersecting_cells:
                total_intersection_area = 0.0
                intersection_areas = []
                for cell_idx in intersecting_cells:
                    try:
                        cell_geom = self.dggs_grid.iloc[cell_idx].geometry
                        inter = pixel_geom.intersection(cell_geom)
                        inter_area = inter.area
                        intersection_areas.append(inter_area)
                        total_intersection_area += inter_area
                    except Exception as e:
                        self.log_message(f"        Warning: Error calculating intersection for cell {cell_idx}: {e}")
                        intersection_areas.append(0.0)

                if total_intersection_area > 0:
                    for j, cell_idx in enumerate(intersecting_cells):
                        area_ratio = intersection_areas[j] / total_intersection_area
                        weighted_value = pixel_value * area_ratio
                        results[cell_idx] += weighted_value
                    distributed_raster_total += float(pixel_value)
                else:
                    value_per_cell = pixel_value / len(intersecting_cells)
                    for cell_idx in intersecting_cells:
                        results[cell_idx] += value_per_cell
                    distributed_raster_total += float(pixel_value)

            processed_pixels += 1
            if processed_pixels % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_pixels / max(elapsed, 1e-9)
                remaining = (num_non_zero - processed_pixels) / max(rate, 1e-9)
                self.log_message(
                    f"        Processed {processed_pixels}/{num_non_zero} pixels "
                    f"({processed_pixels/num_non_zero*100:.1f}%) - ETA: {remaining:.1f}s"
                )

        total_time = time.time() - start_time
        self.log_message(f"      Completed processing {num_non_zero} pixels in {total_time:.2f} seconds")
        self.log_message(f"      Average: {total_time/num_non_zero*1000:.2f} ms per pixel")

        assigned_total_before = float(np.sum(results))
        self.log_message(
            f"      Assigned/src before scaling: {assigned_total_before:.6f}/{src_total_value:.6f} "
            f"(intersecting raster sum: {float(distributed_raster_total):.6f})"
        )
        if assigned_total_before > 0.0 and src_total_value > 0.0:
            scaling_factor = src_total_value / assigned_total_before
            results = results * scaling_factor
            self.log_message(
                f"      Applied conservation scaling: {scaling_factor:.12f} | "
                f"DGGS total after scaling: {float(np.sum(results)):.6f}"
            )
        else:
            self.log_message("      Skipped scaling (zero total encountered)")

        return results.tolist()

    def _process_single_ipcc_code_raster_first(self, args):
        ipcc_code, raster_data = args
        try:
            weighted_values = self._calculate_weighted_values_raster_first(raster_data, ipcc_code)
            return ipcc_code, weighted_values
        except Exception as e:
            print(f"      Error processing {ipcc_code}: {e}")
            return ipcc_code, [0.0] * len(self.dggs_grid)

    def _calculate_weighted_values_parallel_raster_first(self, raster_data_dict, ipcc_codes):
        # If fast label mode is available, single-process is typically faster and avoids MP logging contention
        if self.zone_index_raster is not None:
            self.log_message(f"    Processing {len(ipcc_codes)} aggregated IPCC2006 codes in fast single-process mode (label raster)")
            out = {}
            for ipcc in ipcc_codes:
                out[ipcc] = self._calculate_weighted_values_raster_first(raster_data_dict[ipcc], ipcc)
            return out

        # Fallback: use multiprocessing for intersection-based path
        self.log_message(f"    Processing {len(ipcc_codes)} aggregated IPCC2006 codes in parallel using raster-first approach...")
        num_processes = min(len(ipcc_codes), self.num_cores)
        self.log_message(f"    Using {num_processes} parallel processes")
        process_func = self._process_single_ipcc_code_raster_first
        args_list = [(ipcc, raster_data_dict[ipcc]) for ipcc in ipcc_codes]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_func, args_list)
        return {ipcc: values for ipcc, values in results}

    def _check_existing_csv_files(self, test_output_folder):
        existing_years = set()
        if not os.path.exists(test_output_folder):
            return existing_years
        for filename in os.listdir(test_output_folder):
            if filename.startswith("Europe_DGGS_methane_emissions_") and filename.endswith(".csv"):
                m = re.search(r'Europe_DGGS_methane_emissions_(\d{4})\.csv', filename)
                if m:
                    existing_years.add(int(m.group(1)))
        return existing_years

    def process_netcdf(self):
        start_time = time.time()
        self.log_message("Opening NetCDF dataset...")
        if not os.path.exists(self.netcdf_path):
            raise FileNotFoundError(f"NetCDF file not found: {self.netcdf_path}")

        nc_data = xr.open_dataset(self.netcdf_path)
        years = self._extract_years(nc_data)
        self.log_message(f"NetCDF contains {len(years)} years: {years[0]}..{years[-1]}")

        test_output_folder = os.path.join("test", "test_europe_csv")
        os.makedirs(test_output_folder, exist_ok=True)
        existing_years = self._check_existing_csv_files(test_output_folder)
        self.log_message(f"Found {len(existing_years)} existing year CSV files: {sorted(existing_years)}")

        all_dataframes = []
        all_ipcc_codes = {}
        file_ipcc_mapping = {}

        # Load existing year CSVs first
        for year in sorted(existing_years):
            path = os.path.join(test_output_folder, f"Europe_DGGS_methane_emissions_{year}.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    all_dataframes.append(df)
                    self.log_message(f"  Loaded existing CSV for year {year}: {df.shape}")
                    value_cols = [c for c in df.columns if c not in ['dggsID', 'Year']]
                    for c in value_cols:
                        all_ipcc_codes[c] = df[c].values
                        file_ipcc_mapping[c] = f"existing_{year}.csv"
                except Exception as e:
                    self.log_message(f"  Error loading existing CSV for year {year}: {e}")

        # Process remaining years
        to_process = [(i, y) for i, y in enumerate(years) if y not in existing_years]
        self.log_message(f"Will process {len(to_process)} years, skip {len(years) - len(to_process)} existing years")

        for idx, (time_index, year) in enumerate(to_process, start=1):
            self.log_message("\n" + "=" * 60)
            self.log_message(f"PROCESSING YEAR {idx}/{len(to_process)}: {year}")
            self.log_message("=" * 60)

            try:
                # Start with DGGS IDs
                year_df = self.dggs_grid[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})

                # Aggregate per IPCC for this time step (Tg)
                aggregated = self._aggregate_variables_by_ipcc_for_time(nc_data, time_index)

                # Convert to raster arrays in Mg
                raster_data = self._rasterize_aggregated_for_time(nc_data, aggregated)

                # Compute weighted values
                if len(raster_data) > 1:
                    ipcc_codes = list(raster_data.keys())
                    weighted_dict = self._calculate_weighted_values_parallel_raster_first(raster_data, ipcc_codes)
                    for ipcc_code, values in weighted_dict.items():
                        year_df[ipcc_code] = values
                        all_ipcc_codes[ipcc_code] = values
                        file_ipcc_mapping[ipcc_code] = f"time_index_{time_index}"
                elif len(raster_data) == 1:
                    ipcc_code = list(raster_data.keys())[0]
                    self.log_message(f"  Processing aggregated IPCC2006 code: {ipcc_code}")
                    values = self._calculate_weighted_values_raster_first(raster_data[ipcc_code], ipcc_code)
                    year_df[ipcc_code] = values
                    all_ipcc_codes[ipcc_code] = values
                    file_ipcc_mapping[ipcc_code] = f"time_index_{time_index}"
                else:
                    self.log_message("  No aggregated IPCC codes for this year; skipping year")
                    continue

                # Post-processing: small values to 0
                value_cols = [c for c in year_df.columns if c != 'dggsID']
                self.log_message("  Step 1: Filtering small values (< 1e-6 Mg = 1g)")
                small_values_before = 0
                for c in value_cols:
                    mask = year_df[c] < 1e-6
                    small_values_before += mask.sum()
                    year_df[c] = year_df[c].where(year_df[c] >= 1e-6, 0.0)
                self.log_message(f"    Set {small_values_before} small values to zero across all columns")

                # Remove rows where all value columns are 0
                rows_before = len(year_df)
                year_df = year_df[~(year_df[value_cols] == 0).all(axis=1)]
                rows_after = len(year_df)
                self.log_message(f"  Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")

                # Year column
                year_df['Year'] = year

                # Save per-year CSV
                individual_filename = f"Europe_DGGS_methane_emissions_{year}.csv"
                individual_path = os.path.join(test_output_folder, individual_filename)
                year_df.to_csv(individual_path, index=False)
                self.log_message(f"  Individual year CSV saved to: {individual_path}")
                self.log_message(f"  Year {year} shape: {year_df.shape}")

                all_dataframes.append(year_df)
                self.log_message("  Year completed successfully")

            except Exception as e:
                self.log_message(f"Error processing year {year}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.log_message("\n" + "=" * 60)
        self.log_message("ALL YEARS PROCESSED - COMBINING INTO FINAL OUTPUT")
        self.log_message("=" * 60)

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            self.log_message(f"Combined all {len(all_dataframes)} year dataframes")
            self.log_message(f"Combined shape: {combined_df.shape}")

            value_columns = [c for c in combined_df.columns if c not in ['dggsID', 'Year']]

            # Final small value filtering
            self.log_message("Final processing - Step 1: Filtering small values (< 1e-6 Mg = 1g)")
            small_values_before = 0
            for c in value_columns:
                mask = combined_df[c] < 1e-6
                small_values_before += mask.sum()
                combined_df[c] = combined_df[c].where(combined_df[c] >= 1e-6, 0.0)
            self.log_message(f"  Set {small_values_before} small values to zero across all columns")

            # Remove rows where all sectors 0
            rows_before = len(combined_df)
            combined_df = combined_df[~(combined_df[value_columns] == 0).all(axis=1)]
            rows_after = len(combined_df)
            self.log_message(f"Final processing - Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")

            output_filename = "Europe_DGGS_methane_emissions_ALL_FILES.csv"
            output_path = os.path.join(self.output_folder, output_filename)
            combined_df.to_csv(output_path, index=False)
            self.log_message(f"\nCombined results saved to: {output_path}")
            self.log_message(f"Final output shape: {combined_df.shape}")

            total_time = time.time() - start_time
            self.log_message("\n" + "=" * 60)
            self.log_message("PROCESSING SUMMARY")
            self.log_message("=" * 60)
            self.log_message(f"Years processed this run: {len(to_process)}")
            self.log_message(f"Existing CSV files loaded: {len(existing_years)}")
            self.log_message(f"Total aggregated IPCC2006 codes: {len(all_ipcc_codes)}")
            self.log_message(f"Individual year CSVs saved to: {test_output_folder}")
            self.log_message(f"Combined output columns: {len(combined_df.columns)}")
            self.log_message(f"Combined output rows: {len(combined_df)}")
            self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

            self.log_message("\nFile-IPCC Code Mapping:")
            for ipcc_code, src in file_ipcc_mapping.items():
                self.log_message(f"  {ipcc_code} -> {src}")

            return output_path
        else:
            self.log_message("No dataframes to combine - processing failed for all years")
            return None


def main():
    # Configuration (Windows path provided by user)
    netcdf_path = (
        "/home/mingke.li/GridInventory/2005-2022_CAMS_REG_ANT_European_Antropogenic_Methane_Emissions/"
        "CAMS-REG-ANT_EUR_0.05x0.1_anthro_ch4_v8.1_yearly.nc"
    )
    grid_parquet_path = "data/geojson/regional_grid/europe_grid_res7.parquet"
    output_folder = "output"

    if not os.path.exists(netcdf_path):
        print(f"Error: NetCDF file not found: {netcdf_path}")
        return
    if not os.path.exists(grid_parquet_path):
        print(f"Error: DGGS parquet not found: {grid_parquet_path}")
        return

    converter = EuropeNetCDFToDGGSConverterAggregated(
        netcdf_path=netcdf_path,
        grid_parquet_path=grid_parquet_path,
        output_folder=output_folder,
    )

    try:
        output_path = converter.process_netcdf()
        converter.log_message("\nAll years conversion completed successfully!")
        converter.log_message(f"Combined output file: {output_path}")
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


