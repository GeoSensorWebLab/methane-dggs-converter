import os
import re
import math
import time
import logging
from datetime import datetime
from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon


class ChinaTIFFToDGGSConverter:
    """
    Convert China GeoTIFF methane emission rasters to DGGS grid values.

    Inputs:
    - Directory containing subfolders per variable. Each subfolder has 31 GeoTIFFs, 1990..2020.
      Folder names are variable names and are mapped to IPCC2006 codes via lookup CSV.
    - Each raster band unit: Mg km^-2 a^-1 (megagrams per square kilometer per year).

    Output:
    - Per-year CSV files in test/test_china_csv: wide format, columns per IPCC2006 code, plus dggsID and Year.
    - Combined CSV in output/: China_DGGS_methane_emissions_ALL_FILES.csv.

    Method:
    - For projected rasters (e.g., Krasovsky_1940_Albers), distribute each pixel's mass to intersecting DGGS cells by intersection area in the raster CRS.
    - Compute per-pixel area in km^2 from the raster's CRS and transform.
      • Projected CRSs (e.g., Krasovsky_1940_Albers): area per pixel = |a*e - b*d| (affine determinant) in m^2.
      • Geographic CRS (EPSG:4326): spherical cap formula varying with latitude.
    - Convert pixel values to Mg/year by multiplying Mg km^-2 a^-1 by pixel_area_km2.
    - Aggregate to DGGS cells via area-weighted intersection (projected) or via a label raster fallback (geographic).
    - Stream per year over variables to minimize memory.
    """

    def __init__(self, root_folder, dggs_geojson_path, lookup_csv_path, output_folder, num_cores=None):
        self.root_folder = root_folder
        self.dggs_geojson_path = dggs_geojson_path
        self.lookup_csv_path = lookup_csv_path
        self.output_folder = output_folder

        if num_cores is None:
            self.num_cores = int(os.environ.get('NUM_CORES', 8))
        else:
            self.num_cores = num_cores

        self._setup_logging()
        self._load_lookup()

        self.log_message("Loading DGGS grid (WGS84)...")
        self.dggs_grid_wgs84 = gpd.read_file(self.dggs_geojson_path)
        if 'zoneID' not in self.dggs_grid_wgs84.columns:
            raise ValueError("GeoJSON file must contain 'zoneID' column")
        self.num_cells = len(self.dggs_grid_wgs84)
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs("test/test_china_csv", exist_ok=True)

        # Cache for zone index raster and per-pixel area array per grid key
        self._zone_index_cache = {}
        self._pixel_area_cache = {}
        # Cache for projected DGGS grid and bounds per CRS
        self._proj_dggs_cache = {}

    def _ensure_projected_dggs(self, crs):
        """Project DGGS grid to the provided CRS and cache with bounds."""
        crs_key = crs.to_string() if crs is not None else ""
        if crs_key in self._proj_dggs_cache:
            return self._proj_dggs_cache[crs_key]
        self.log_message("Projecting DGGS grid to raster CRS for area-weighted intersections...")
        dggs_proj = self.dggs_grid_wgs84.to_crs(crs)
        bounds_list = []
        for idx, row in dggs_proj.iterrows():
            bounds_list.append((row.geometry.bounds, idx))
        cache_entry = {"gdf": dggs_proj, "bounds": bounds_list}
        self._proj_dggs_cache[crs_key] = cache_entry
        self.log_message(f"  Projected DGGS cached for CRS: {crs_key}")
        return cache_entry

    def _pixel_polygon(self, transform, row, col):
        """Return pixel polygon (parallelogram) at (row, col) using the affine transform."""
        x0, y0 = (transform * (col, row))
        x1, y1 = (transform * (col + 1, row))
        x2, y2 = (transform * (col + 1, row + 1))
        x3, y3 = (transform * (col, row + 1))
        return Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])

    def _find_intersecting_cells_projected(self, pixel_geom, dggs_cache):
        """Return list of DGGS indices that intersect the pixel geometry (bounds prefilter)."""
        intersecting = []
        minx, miny, maxx, maxy = pixel_geom.bounds
        for bounds, idx in dggs_cache["bounds"]:
            if not (maxx < bounds[0] or minx > bounds[2] or maxy < bounds[1] or miny > bounds[3]):
                try:
                    if pixel_geom.intersects(dggs_cache["gdf"].iloc[idx].geometry):
                        intersecting.append(idx)
                except Exception:
                    continue
        return intersecting

    def _setup_logging(self):
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"china_tiff_to_dggs_conversion_{timestamp}.log"
        log_path = os.path.join(log_folder, log_filename)

        self.logger = logging.getLogger(f"china_tiff_converter_{timestamp}")
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

    def _load_lookup(self):
        if not os.path.exists(self.lookup_csv_path):
            raise FileNotFoundError(f"Lookup CSV not found: {self.lookup_csv_path}")
        df = pd.read_csv(self.lookup_csv_path)
        if 'variable' not in df.columns or 'IPCC2006' not in df.columns:
            raise ValueError("Lookup CSV must contain 'variable' and 'IPCC2006' columns")
        self.variable_to_ipcc = dict(zip(df['variable'], df['IPCC2006']))
        self.log_message(f"Loaded China TIFF lookup with {len(self.variable_to_ipcc)} mappings")

    def _grid_cache_key(self, crs, transform, width, height):
        return (
            crs.to_string() if crs is not None else "",
            float(transform.a), float(transform.b), float(transform.c),
            float(transform.d), float(transform.e), float(transform.f),
            int(width), int(height),
        )

    def _build_zone_index_raster(self, crs, transform, width, height):
        key = self._grid_cache_key(crs, transform, width, height)
        if key in self._zone_index_cache:
            return self._zone_index_cache[key]

        self.log_message("Rasterizing DGGS cells to label raster for current grid...")
        dggs_proj = self.dggs_grid_wgs84.to_crs(crs)
        shapes = ((geom, idx + 1) for idx, geom in enumerate(dggs_proj.geometry))
        label_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='int32',
            all_touched=True
        )
        self._zone_index_cache[key] = label_raster
        self.log_message("  Zone index raster created and cached")
        return label_raster

    def _compute_pixel_area_km2(self, crs, transform, width, height):
        key = self._grid_cache_key(crs, transform, width, height)
        if key in self._pixel_area_cache:
            return self._pixel_area_cache[key]

        # Log CRS information once per unique grid
        epsg = None
        crs_str = None
        if crs is not None:
            try:
                epsg = crs.to_epsg()
            except Exception:
                epsg = None
            try:
                crs_str = crs.to_string()
            except Exception:
                crs_str = str(crs)
        if epsg is not None:
            self.log_message(f"  Raster CRS: EPSG:{epsg}")
        elif crs is not None and crs_str is not None:
            self.log_message(f"  Raster CRS: {crs_str}")

        if crs is not None and getattr(crs, "is_geographic", False):
            # Expect ~0.1° × 0.1° pixels in EPSG:4326; handle sign of transform.e
            delta_lon = abs(transform.a)
            delta_lat = abs(transform.e)
            if abs(delta_lon - 0.1) > 1e-6 or abs(delta_lat - 0.1) > 1e-6:
                self.log_message(
                    f"  Warning: Detected geographic pixel size ~{delta_lon:.6f}×{delta_lat:.6f} degrees (expected 0.1×0.1)"
                )

            lon_rad = math.radians(delta_lon)
            lat_half_rad = math.radians(delta_lat) / 2.0
            y0 = transform.f
            dy = transform.e
            row_idx = np.arange(height, dtype=np.float64)
            lat_centers = y0 + (row_idx + 0.5) * dy
            lat_centers_rad = np.radians(lat_centers)
            lat_low = np.minimum(lat_centers_rad - lat_half_rad, lat_centers_rad + lat_half_rad)
            lat_high = np.maximum(lat_centers_rad - lat_half_rad, lat_centers_rad + lat_half_rad)
            R = 6371008.8
            area_m2_row = (R * R) * lon_rad * (np.sin(lat_high) - np.sin(lat_low))
            area_km2_row = (area_m2_row / 1e6).astype(np.float32)
            pixel_area_km2 = np.repeat(area_km2_row[:, np.newaxis], width, axis=1)
        else:
            # Projected CRS (e.g., Krasovsky_1940_Albers) or unknown: use affine determinant
            # 2x2 part of affine transform is [[a, b],[d, e]]; pixel area (m^2) = |a*e - b*d|
            det = abs((float(transform.a) * float(transform.e)) - (float(transform.b) * float(transform.d)))
            if det == 0.0:
                # Fallback to width*height if determinant is zero (should not happen for valid rasters)
                pixel_width = abs(float(transform.a))
                pixel_height = abs(float(transform.e))
                det = pixel_width * pixel_height
            self.log_message(
                f"  Projected pixel transform (a,b,d,e): {float(transform.a):.6f}, {float(transform.b):.6f}, {float(transform.d):.6f}, {float(transform.e):.6f}"
            )
            self.log_message(f"  Using affine determinant for pixel area: {det:.6f} m^2")
            area_km2 = (det / 1e6)
            pixel_area_km2 = np.full((height, width), float(area_km2), dtype=np.float32)

        self._pixel_area_cache[key] = pixel_area_km2
        return pixel_area_km2

    def _extract_year_from_filename(self, filename):
        m = re.search(r'(\d{4})(?=\.[Tt][Ii][Ff]$)', filename)
        if not m:
            raise ValueError(f"Year not found in filename: {filename}")
        return int(m.group(1))

    def _process_single_tiff(self, args):
        tiff_path, ipcc_code = args
        with rasterio.open(tiff_path) as src:
            data = src.read(1, masked=True).astype(np.float32)
            data = np.ma.filled(data, 0.0)
            transform = src.transform
            width = src.width
            height = src.height
            crs = src.crs

        pixel_area_km2 = self._compute_pixel_area_km2(crs, transform, width, height)

        mass_per_pixel = data * pixel_area_km2
        # Units: data is Mg km^-2 a^-1, pixel_area_km2 is km^2 → mass_per_pixel is Mg/year
        self.log_message("  Unit check: values are 'Mg km^-2 a^-1' × pixel_area_km2 (km^2) ⇒ Mg/year per pixel")
        mass_per_pixel = np.nan_to_num(mass_per_pixel, nan=0.0)
        mass_per_pixel = np.clip(mass_per_pixel, 0, None)

        # Projected CRS: area-weighted intersection; Geographic CRS: fallback to label raster
        if crs is not None and not getattr(crs, "is_geographic", False):
            dggs_cache = self._ensure_projected_dggs(crs)
            results = np.zeros(self.num_cells, dtype=np.float64)
            non_zero = np.where(mass_per_pixel > 0)
            num_non_zero = len(non_zero[0])
            if num_non_zero == 0:
                return ipcc_code, results, 0.0, 0.0
            self.log_message(f"      Non-zero pixels to distribute (projected): {num_non_zero}")
            start_time = time.time()
            for i in range(num_non_zero):
                row = int(non_zero[0][i])
                col = int(non_zero[1][i])
                pixel_value = float(mass_per_pixel[row, col])
                pixel_geom = self._pixel_polygon(transform, row, col)
                intersecting = self._find_intersecting_cells_projected(pixel_geom, dggs_cache)
                if intersecting:
                    total_area = 0.0
                    areas = []
                    for idx in intersecting:
                        try:
                            inter = pixel_geom.intersection(dggs_cache["gdf"].iloc[idx].geometry)
                            area = inter.area
                        except Exception:
                            area = 0.0
                        areas.append(area)
                        total_area += area
                    if total_area > 0.0:
                        for j, idx in enumerate(intersecting):
                            results[idx] += pixel_value * (areas[j] / total_area)
                    else:
                        share = pixel_value / len(intersecting)
                        for idx in intersecting:
                            results[idx] += share
                if (i + 1) % 2000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                    remaining = (num_non_zero - (i + 1)) / rate if rate > 0 else 0.0
                    self.log_message(
                        f"        Processed {i + 1}/{num_non_zero} ({(i + 1)/num_non_zero*100:.1f}%) - ETA: {remaining:.1f}s"
                    )
            total_raster_value = float(np.sum(mass_per_pixel))
            total_weighted_value = float(np.sum(results))
            if total_weighted_value > 0.0 and total_raster_value > 0.0:
                scaling_factor = total_raster_value / total_weighted_value
                results = results * scaling_factor
                self.log_message(f"      Applied scaling factor for {ipcc_code}: {scaling_factor:.6f}")
            return ipcc_code, results, total_raster_value, float(np.sum(results))
        else:
            label_raster = self._build_zone_index_raster(crs, transform, width, height)
            labels = label_raster.ravel()
            values = mass_per_pixel.ravel()
            sums = np.bincount(labels, weights=values, minlength=self.num_cells + 1)[1:]
            total_raster_value = float(np.sum(values[labels > 0]))
            total_weighted_value = float(np.sum(sums))
            if total_weighted_value > 0.0 and total_raster_value > 0.0:
                scaling_factor = total_raster_value / total_weighted_value
                sums = sums * scaling_factor
                self.log_message(f"      Applied scaling factor for {ipcc_code}: {scaling_factor:.6f}")
            return ipcc_code, sums, total_raster_value, float(np.sum(sums))

    def _list_variable_dirs(self):
        all_entries = [os.path.join(self.root_folder, d) for d in os.listdir(self.root_folder)]
        dirs = [d for d in all_entries if os.path.isdir(d)]
        return dirs

    def _map_dir_to_ipcc(self, dir_path):
        var_name = os.path.basename(dir_path.rstrip(os.sep))
        return self.variable_to_ipcc.get(var_name)

    def process_all_years(self):
        start_time = time.time()
        self.log_message("Scanning variable directories...")
        var_dirs = self._list_variable_dirs()

        mapped_dirs = []
        skipped = []
        for d in var_dirs:
            ipcc = self._map_dir_to_ipcc(d)
            if ipcc:
                mapped_dirs.append((d, ipcc))
            else:
                skipped.append(os.path.basename(d))
        self.log_message(f"Found {len(mapped_dirs)} mapped variable directories; skipped {len(skipped)}")

        years = list(range(1990, 2021))
        for year in years:
            self.log_message("\n" + "=" * 60)
            self.log_message(f"PROCESSING YEAR: {year}")
            self.log_message("=" * 60)

            result_df = self.dggs_grid_wgs84[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})

            tasks = []
            for dir_path, ipcc_code in mapped_dirs:
                # Find the tiff for this year in the directory
                tiffs = [f for f in os.listdir(dir_path) if f.lower().endswith(('.tif', '.tiff'))]
                year_files = [os.path.join(dir_path, f) for f in tiffs if self._extract_year_from_filename(f) == year]
                if not year_files:
                    continue
                tasks.append((year_files[0], ipcc_code))

            if not tasks:
                self.log_message(f"No rasters found for year {year}, skipping")
                continue

            num_processes = min(self.num_cores, 8)
            self.log_message(f"  Using {num_processes} parallel workers for {len(tasks)} rasters")
            process_func = partial(self._process_single_tiff)
            if num_processes > 1:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    # progress logging in chunks
                    chunk = 8
                    results = []
                    for i in range(0, len(tasks), chunk):
                        part = pool.map(process_func, tasks[i:i+chunk])
                        results.extend(part)
                        self.log_message(
                            f"      Processed {min(i+chunk, len(tasks))}/{len(tasks)} rasters ({min(i+chunk, len(tasks))/len(tasks)*100:.1f}%)"
                        )
            else:
                results = []
                for i, item in enumerate(tasks, start=1):
                    results.append(process_func(item))
                    if i % 4 == 0 or i == len(tasks):
                        self.log_message(
                            f"      Processed {i}/{len(tasks)} rasters ({i/len(tasks)*100:.1f}%)"
                        )

            # Aggregate by IPCC code (some variables may share code)
            aggregates = {}
            for ipcc_code, sums, total_raster_value, total_weighted_after in results:
                # Log per-raster totals (no scaling applied at this stage)
                self.log_message(f"      Raster total (Mg/yr): {total_raster_value:.6f}")
                self.log_message(f"      DGGS weighted sum (Mg/yr): {total_weighted_after:.6f}")
                if ipcc_code not in aggregates:
                    aggregates[ipcc_code] = sums
                else:
                    aggregates[ipcc_code] = aggregates[ipcc_code] + sums

            for ipcc_code, vector in aggregates.items():
                result_df[ipcc_code] = vector

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
            
            # Step 2: Remove rows where all values are 0
            rows_before = len(result_df)
            result_df = result_df[~(result_df[value_cols] == 0).all(axis=1)]
            rows_after = len(result_df)
            self.log_message(f"  Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
            
            # Steps complete (only Step 1 and Step 2 retained by design)
            result_df['Year'] = year

            out_dir = "test/test_china_csv"
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"China_DGGS_methane_emissions_{year}.csv")
            result_df.to_csv(out_file, index=False)
            self.log_message(f"  Year {year} saved to {out_file} with shape {result_df.shape}")

            # File-IPCC mapping for this year
            self.log_message("\nFile-IPCC Code Mapping:")
            for dir_path, ipcc_code in mapped_dirs:
                # Determine if year file existed
                tiffs = [f for f in os.listdir(dir_path) if f.lower().endswith(('.tif', '.tiff'))]
                year_files = [f for f in tiffs if self._extract_year_from_filename(f) == year]
                if year_files:
                    self.log_message(f"  {ipcc_code} -> {os.path.basename(year_files[0])}")

        # Combine all year CSVs
        self.log_message("\n" + "=" * 60)
        self.log_message("Combining yearly CSVs into final output...")
        self.log_message("=" * 60)

        yearly_files = [os.path.join("test/test_china_csv", f) for f in os.listdir("test/test_china_csv") if f.startswith("China_DGGS_methane_emissions_") and f.endswith('.csv')]
        if not yearly_files:
            self.log_message("No yearly CSVs found to combine")
            return None

        combined_iter = (pd.read_csv(p) for p in sorted(yearly_files))
        combined_df = pd.concat(combined_iter, ignore_index=True)
        
        # Get value columns for processing
        value_cols = [c for c in combined_df.columns if c not in ['dggsID', 'Year']]
        # Compute per-variable original totals before filtering for scaling
        original_totals = {}
        for col in value_cols:
            original_totals[col] = combined_df[col].sum()
        
        # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
        self.log_message(f"Final processing - Step 1: Filtering small values (< 1e-6 Mg = 1g)")
        small_values_before = 0
        for col in value_cols:
            small_mask = combined_df[col] < 1e-6
            small_count = small_mask.sum()
            small_values_before += small_count
            combined_df[col] = combined_df[col].where(combined_df[col] >= 1e-6, 0.0)
        self.log_message(f"  Set {small_values_before} small values to zero across all columns")
        
        # Step 2: Remove rows where all values are 0
        rows_before = len(combined_df)
        combined_df = combined_df[~(combined_df[value_cols] == 0).all(axis=1)]
        rows_after = len(combined_df)
        self.log_message(f"Final processing - Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
        
        # Steps complete (only Step 1 and Step 2 retained by design)

        output_path = os.path.join(self.output_folder, "China_DGGS_methane_emissions_ALL_FILES.csv")
        combined_df.to_csv(output_path, index=False)

        total_time = time.time() - start_time
        self.log_message(f"Combined output saved to: {output_path}")
        self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        return output_path


def main():
    root_folder = "/home/mingke.li/GridInventory/1990-2020_CHN-CH4_Anthropogenic_Methane_Emission_Inventory_of_China"
    dggs_geojson_path = "data/geojson/global_countries_dggs_merge/China_CHN_grid.geojson"
    lookup_csv_path = "data/lookup/china_tiff_variable_lookup.csv"
    output_folder = "output"

    if not os.path.exists(root_folder):
        print(f"Error: Root folder not found: {root_folder}")
        return
    if not os.path.exists(lookup_csv_path):
        print(f"Error: Lookup CSV not found: {lookup_csv_path}")
        return
    if not os.path.exists(dggs_geojson_path):
        print(f"Warning: DGGS GeoJSON not found (expected by workflow): {dggs_geojson_path}")

    converter = ChinaTIFFToDGGSConverter(
        root_folder=root_folder,
        dggs_geojson_path=dggs_geojson_path,
        lookup_csv_path=lookup_csv_path,
        output_folder=output_folder,
    )

    try:
        output_path = converter.process_all_years()
        converter.log_message("\nChina TIFF → DGGS conversion completed successfully!")
        converter.log_message(f"Combined output file: {output_path}")
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


