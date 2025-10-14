import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import multiprocessing
from functools import partial
import time
import logging
from datetime import datetime


class NYSNetCDFToDGGSConverterAggregated:
    """
    Convert GNYS (New York State) NetCDF to DGGS grid values using a raster-first,
    area-weighted approach with variable aggregation by IPCC2006 codes.

    GNYS specifics:
    - Units: kg m^-2 s^-1 (flux)
    - Projection: EPSG:26918 (UTM Zone 18N)
    - Resolution: 100 m × 100 m per pixel
    - Year: 2020 (fixed)

    Target output units: Mg/year

    Conversion:
    mass_Mg_per_year = flux_kg_per_m2_per_s * pixel_area_m2 (10000) * seconds_per_year / 1000
                     = flux * 10000 * 31,536,000 / 1000
                     = flux * 10 * 31,536,000
    """

    def __init__(self, netcdf_folder, dggs_grid_path, output_folder, num_cores=None):
        self.netcdf_folder = netcdf_folder
        self.dggs_grid_path = dggs_grid_path
        self.output_folder = output_folder

        if num_cores is None:
            self.num_cores = int(os.environ.get('NUM_CORES', 8))
        else:
            self.num_cores = num_cores

        # Constants
        self.SECONDS_PER_YEAR = 365 * 24 * 3600
        self.PIXEL_SIZE_M = 100.0
        self.PIXEL_AREA_M2 = self.PIXEL_SIZE_M * self.PIXEL_SIZE_M  # 10,000 m^2
        # Configurable threshold for small values (Mg). Can override via env var SMALL_VALUE_THRESHOLD
        self.SMALL_VALUE_THRESHOLD = float(os.environ.get('SMALL_VALUE_THRESHOLD', '1e-6'))
        # Default y axis orientation assumption; will be set per dataset
        self.y_descending = True

        # Logging
        self._setup_logging()
        self.log_message("Loading IPCC2006 variable lookup table for GNYS...")
        self._load_ipcc_lookup()

        # Load DGGS grid from file in WGS84 then project to EPSG:26918 for area-correct intersection
        self.log_message("Loading DGGS grid and projecting to EPSG:26918 (UTM Zone 18N)...")
        if self.dggs_grid_path.endswith('.parquet'):
            # Load single Parquet file
            self.dggs_grid_wgs84 = gpd.read_parquet(self.dggs_grid_path)
        else:
            # Fallback to GeoJSON for backward compatibility
            self.dggs_grid_wgs84 = gpd.read_file(self.dggs_grid_path)
        
        if 'zoneID' not in self.dggs_grid_wgs84.columns:
            raise ValueError("DGGS grid file must contain 'zoneID' column")
        self.dggs_grid_proj = self.dggs_grid_wgs84.to_crs('EPSG:26918')
        self.log_message(f"Loaded {len(self.dggs_grid_proj)} DGGS cells")

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Discover NetCDF files (.nc or .nc4)
        self.netcdf_files = [f for f in os.listdir(self.netcdf_folder) if f.endswith('.nc') or f.endswith('.nc4')]
        self.log_message(f"Found {len(self.netcdf_files)} NetCDF file(s)")

        # Spatial index (bounds list) for projected DGGS grid
        self.log_message("Creating spatial index for projected DGGS cells...")
        self._create_spatial_index()

        # Lazy-built DGGS zone index raster (pixel -> DGGS row index+1)
        self.zone_index_raster = None

    def _setup_logging(self):
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"nys_netcdf_to_dggs_conversion_{timestamp}.log"
        log_path = os.path.join(log_folder, log_filename)

        self.logger = logging.getLogger(f"nys_converter_{timestamp}")
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
        lookup_path = "data/lookup/gnys_netcdf_variable_lookup.csv"
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"IPCC lookup file not found: {lookup_path}")
        self.ipcc_lookup = pd.read_csv(lookup_path)
        if 'variable' not in self.ipcc_lookup.columns or 'IPCC2006' not in self.ipcc_lookup.columns:
            raise ValueError("Lookup CSV must contain 'variable' and 'IPCC2006' columns")
        self.variable_to_ipcc = dict(zip(self.ipcc_lookup['variable'], self.ipcc_lookup['IPCC2006']))
        self.log_message(f"Loaded GNYS IPCC lookup with {len(self.variable_to_ipcc)} mappings")

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
            if len(var_list) == 1:
                aggregated[ipcc_code] = nc_data[var_list[0]].values
            else:
                arr = nc_data[var_list[0]].values
                for v in var_list[1:]:
                    arr = arr + nc_data[v].values
                aggregated[ipcc_code] = arr
        return aggregated

    def group_variables_by_ipcc_code(self, nc_data):
        """Return mapping of IPCC2006 code -> list of variable names (no data loaded)."""
        self.log_message("Grouping variables by IPCC2006 codes (streaming mode)...")
        exclude_vars = {'lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'time', 'grid_cell_area'}
        variables = [var for var in nc_data.variables if var not in exclude_vars]
        ipcc_groups = {}
        unmapped = []
        for var in variables:
            if var in self.variable_to_ipcc:
                ipcc_groups.setdefault(self.variable_to_ipcc[var], []).append(var)
            else:
                unmapped.append(var)
        if unmapped:
            self.log_message(f"  Skipped {len(unmapped)} variables not found in IPCC lookup table")
        self.log_message(f"  Found {len(ipcc_groups)} unique IPCC2006 codes")
        return ipcc_groups

    def _build_zone_index_raster(self, x_coords, y_coords, pixel_size_m):
        """Rasterize DGGS polygons into a label raster aligned with the input grid.
        Each pixel gets the DGGS row index + 1; 0 means no polygon.
        """
        if self.zone_index_raster is not None:
            return self.zone_index_raster
        self.log_message("Building DGGS zone index raster (one-time)...")
        half = pixel_size_m / 2.0
        left = float(np.min(x_coords)) - half
        top = float(np.max(y_coords)) + half
        transform = from_origin(left, top, pixel_size_m, pixel_size_m)
        out_shape = (len(y_coords), len(x_coords))
        # shapes: list of (geometry, value)
        shapes = ((geom, idx + 1) for idx, geom in enumerate(self.dggs_grid_proj.geometry))
        label_raster = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='int32',
            all_touched=True
        )
        self.zone_index_raster = label_raster
        self.log_message("  Zone index raster built")
        return self.zone_index_raster

    def _compute_mass_per_pixel_for_ipcc(self, ds, var_list):
        """Stream-sum variables for an IPCC code and convert to Mg/year per pixel.
        Units: kg m^-2 s^-1; area 100x100m; seconds per year.
        NaN/inf treated as 0 and negatives clipped to 0 before summation.
        """
        stack = []
        for v in var_list:
            arr = ds[v].values
            if arr.ndim == 3:
                arr = arr[0, :, :]
            elif arr.ndim != 2:
                raise ValueError(f"Unexpected dims for variable {v}: {arr.shape}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr, 0, None)
            stack.append(arr)
        if not stack:
            return None
        acc = np.sum(stack, axis=0)
        mass_per_pixel = acc * self.PIXEL_AREA_M2 * self.SECONDS_PER_YEAR / 1000.0
        return mass_per_pixel

    def _sum_pixels_to_cells(self, mass_per_pixel):
        """Sum per-pixel masses to DGGS cells using the precomputed label raster.
        Align orientation if the dataset y-axis is ascending (bottom->top).
        """
        labels = self.zone_index_raster
        if not getattr(self, 'y_descending', True):
            labels = np.flipud(labels)
        flat_labels = labels.ravel()
        flat_values = mass_per_pixel.ravel()
        max_label = len(self.dggs_grid_proj)
        sums = np.bincount(flat_labels, weights=flat_values, minlength=max_label + 1)
        # Drop background 0 index
        return sums[1:]

    def convert_aggregated_to_raster(self, netcdf_path):
        # Deprecated in memory-optimized pipeline; kept for compatibility if needed
        self.log_message(f"Processing NetCDF: {os.path.basename(netcdf_path)}")
        ds = xr.open_dataset(netcdf_path)
        x_name = 'x' if 'x' in ds.variables else ('lon' if 'lon' in ds.variables else None)
        y_name = 'y' if 'y' in ds.variables else ('lat' if 'lat' in ds.variables else None)
        if x_name is None or y_name is None:
            ds.close()
            raise ValueError("Could not locate x/y (or lon/lat) coordinates in NetCDF")
        x = ds[x_name].values
        y = ds[y_name].values
        groups = self.group_variables_by_ipcc_code(ds)
        raster_data = {}
        for ipcc_code, var_list in groups.items():
            mass_per_pixel = self._compute_mass_per_pixel_for_ipcc(ds, var_list)
            raster_data[ipcc_code] = {'data': mass_per_pixel, 'x': x, 'y': y, 'pixel_size': self.PIXEL_SIZE_M}
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

        x_coords = raster_data['x']
        y_coords = raster_data['y']
        half = raster_data['pixel_size'] / 2.0

        start_time = time.time()
        processed = 0
        for i in range(num_non_zero):
            row, col = non_zero_coords[0][i], non_zero_coords[1][i]
            pixel_value = data[row, col]
            pixel_x = x_coords[col]
            pixel_y = y_coords[row]

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

        # Scale to match total of intersecting pixels for this variable
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
            # Note: This runs in a separate process; avoid accessing non-picklable state.
            # We re-create minimal state needed via globals captured in self. Use methods that rely on numpy/shapely only.
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

        year = 2020
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

                # Open dataset once and stream per IPCC group to minimize memory
                ds = xr.open_dataset(netcdf_path)
                x_name = 'x' if 'x' in ds.variables else ('lon' if 'lon' in ds.variables else None)
                y_name = 'y' if 'y' in ds.variables else ('lat' if 'lat' in ds.variables else None)
                if x_name is None or y_name is None:
                    ds.close()
                    raise ValueError("Could not locate x/y (or lon/lat) coordinates in NetCDF")
                x = ds[x_name].values
                y = ds[y_name].values
                # Determine y axis orientation: True if y decreases from top->bottom
                try:
                    self.y_descending = bool(y[0] > y[-1])
                except Exception:
                    self.y_descending = True

                # Build zone index raster once
                self._build_zone_index_raster(x, y, self.PIXEL_SIZE_M)

                groups = self.group_variables_by_ipcc_code(ds)
                total_codes = len(groups)
                self.log_message(f"Processing {total_codes} aggregated IPCC2006 codes (streaming)...")

                # Track File-IPCC mapping
                file_ipcc_mapping = {}

                for idx_code, (ipcc_code, var_list) in enumerate(groups.items(), start=1):
                    mass_per_pixel = self._compute_mass_per_pixel_for_ipcc(ds, var_list)
                    values = self._sum_pixels_to_cells(mass_per_pixel)
                    # Conserve totals per IPCC code
                    src_total = float(np.sum(mass_per_pixel)) if mass_per_pixel is not None else 0.0
                    assigned_total = float(np.sum(values)) if values is not None else 0.0
                    if src_total > 0.0:
                        loss_frac = 1.0 - (assigned_total / src_total) if assigned_total > 0.0 else 1.0
                        self.log_message(
                            f"      {ipcc_code}: assigned/src = {assigned_total:.6f}/{src_total:.6f} (loss {loss_frac*100:.3f}%) before scaling"
                        )
                        if assigned_total > 0.0 and abs(loss_frac) > 1e-12:
                            scale = src_total / assigned_total
                            values = values * scale
                            self.log_message(f"      Applied conservation scaling: {scale:.12f}")

                    result_df[ipcc_code] = values

                    # Map IPCC -> source filename
                    file_ipcc_mapping[ipcc_code] = os.path.basename(netcdf_path)

                    # Progress log for codes
                    self.log_message(
                        f"      Processed {idx_code}/{total_codes} IPCC codes ({idx_code/total_codes*100:.1f}%)"
                    )
                ds.close()

                # Step 1: Set small values (< threshold Mg) to zero
                value_cols = [c for c in result_df.columns if c != 'dggsID']
                self.log_message(f"  Step 1: Filtering small values (< {self.SMALL_VALUE_THRESHOLD} Mg)")
                small_values_before = 0
                for col in value_cols:
                    small_mask = result_df[col] < self.SMALL_VALUE_THRESHOLD
                    small_count = small_mask.sum()
                    small_values_before += small_count
                    result_df[col] = result_df[col].where(result_df[col] >= self.SMALL_VALUE_THRESHOLD, 0.0)
                self.log_message(f"    Set {small_values_before} small values to zero across all columns")
                
                # Step 2: Remove rows with all zeros across IPCC columns
                rows_before = len(result_df)
                result_df = result_df[~(result_df[value_cols] == 0).all(axis=1)]
                rows_after = len(result_df)
                self.log_message(f"  Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
                
                # Steps complete (only Step 1 and Step 2 retained by design)
                result_df['Year'] = year

                all_frames.append(result_df)
                self.log_message(f"  Completed {filename} with shape {result_df.shape}")
                # Mapping summary for this file
                if file_ipcc_mapping:
                    self.log_message("\nFile-IPCC Code Mapping:")
                    for ipcc_code, fname in file_ipcc_mapping.items():
                        self.log_message(f"  {ipcc_code} -> {fname}")
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
        
        # Step 1: Set small values (< threshold Mg) to zero
        self.log_message(f"Final processing - Step 1: Filtering small values (< {self.SMALL_VALUE_THRESHOLD} Mg)")
        small_values_before = 0
        for col in value_cols:
            small_mask = combined[col] < self.SMALL_VALUE_THRESHOLD
            small_count = small_mask.sum()
            small_values_before += small_count
            combined[col] = combined[col].where(combined[col] >= self.SMALL_VALUE_THRESHOLD, 0.0)
        self.log_message(f"  Set {small_values_before} small values to zero across all columns")
        
        # Step 2: Remove rows with all zeros
        rows_before = len(combined)
        combined = combined[~(combined[value_cols] == 0).all(axis=1)]
        rows_after = len(combined)
        self.log_message(f"Final processing - Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
        
        # Steps complete (only Step 1 and Step 2 retained by design)
        self.log_message(f"After removing zero rows: {len(combined)} rows remain")

        output_filename = "NYS_DGGS_methane_emissions_2020.csv"
        output_path = os.path.join(self.output_folder, output_filename)
        combined.to_csv(output_path, index=False)
        total_time = time.time() - start_time
        self.log_message(f"\nResults saved to: {output_path}")
        self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        return output_path


def main():
    """Entry point for standalone execution on HPC or local."""
    # Default configuration (update paths as needed)
    netcdf_folder = "/home/mingke.li/GridInventory/2020_Gridded_New_York_State_methane_emissions_inventory_GNYS"
    dggs_grid_path = "data/geojson/regional_grid/newyorkstate_grid_res10.parquet"
    output_folder = "output"

    if not os.path.exists(netcdf_folder):
        print(f"Error: NetCDF folder not found: {netcdf_folder}")
        return
    if not os.path.exists(dggs_grid_path):
        # Allow running even if the file/directory is not present locally per request; just inform.
        print(f"Warning: DGGS grid path not found (expected by workflow): {dggs_grid_path}")

    converter = NYSNetCDFToDGGSConverterAggregated(
        netcdf_folder=netcdf_folder,
        dggs_grid_path=dggs_grid_path,
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


