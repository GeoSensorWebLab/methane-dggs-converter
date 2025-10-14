import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import rasterio.windows
from shapely.geometry import box
import multiprocessing
import time
import logging
from datetime import datetime
import re
from collections import defaultdict


# Module-level cache for worker processes. Each worker lazily loads and caches
# country grids and their spatial index keyed by GID to avoid repeated I/O and
# expensive sindex rebuilds for multiple chunks of the same country.
_WORKER_COUNTRY_CACHE = {}


def _setup_logger():
    # Create log folder if it doesn't exist
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"global_gfei_netcdf_to_dggs_conversion_{timestamp}.log"
    log_path = os.path.join(log_folder, log_filename)
    
    # Create a new logger specifically for this script
    logger = logging.getLogger(f"gfei_converter_{timestamp}")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid conflicts
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger, log_path
def _compute_cell_area_m2(lat_centers: np.ndarray, lon_centers: np.ndarray, cell_deg: float = 0.1) -> np.ndarray:
    """Compute approximate cell areas (m^2) on a sphere for a regular lon/lat grid.
    Uses spherical Earth formula consistent with small-area approximation.
    Returns an array of shape (lat, lon).
    """
    R = 6371000.0  # meters
    half = cell_deg / 2.0
    phi1 = np.deg2rad(lat_centers - half)
    phi2 = np.deg2rad(lat_centers + half)
    dlon = np.deg2rad(cell_deg)
    band_area_m2 = (R * R) * dlon * (np.sin(phi2) - np.sin(phi1))  # shape (lat,)
    return np.repeat(band_area_m2[:, None], len(lon_centers), axis=1)



def _bounds_intersect(b1, b2):
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def _create_temp_raster_from_array(array_2d, lat, lon):
    cell_size = 0.1
    half = cell_size / 2.0
    top_left_lon = lon.min() - half
    top_left_lat = lat.max() + half
    transform = from_origin(top_left_lon, top_left_lat, cell_size, cell_size)
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = os.path.join(temp_folder, f"temp_raster_{ts}.tiff")
    with rasterio.open(
        temp_path,
        'w',
        driver='GTiff',
        height=array_2d.shape[0],
        width=array_2d.shape[1],
        count=1,
        dtype=array_2d.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(array_2d, 1)
    return temp_path


def _clip_raster_by_bbox(raster_path, bbox):
    minx, miny, maxx, maxy = bbox
    buffer = 0.05
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    with rasterio.open(raster_path) as src:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
        clipped = src.read(1, window=window)
        clipped_transform = rasterio.windows.transform(window, src.transform)
        bounds = rasterio.transform.array_bounds(clipped.shape[0], clipped.shape[1], clipped_transform)
        return {
            'data': clipped,
            'transform': clipped_transform,
            'bounds': bounds,
            'crs': 'EPSG:4326',
            'width': clipped.shape[1],
            'height': clipped.shape[0]
        }


def _load_country_grid_cached(gid, grid_pickle_path):
    cached = _WORKER_COUNTRY_CACHE.get(gid)
    if cached is not None:
        return cached['grid'], cached['sindex']
    gdf = pd.read_pickle(grid_pickle_path)
    # Ensure type is GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')
    sindex = gdf.sindex
    _WORKER_COUNTRY_CACHE[gid] = {'grid': gdf, 'sindex': sindex}
    return gdf, sindex


def _pixel_chunk_worker(args):
    gid, grid_pickle_path, clipped_npy_path, bounds, transform, rows, cols, ipcc_code, year = args
    try:
        grid, sindex = _load_country_grid_cached(gid, grid_pickle_path)
        data = np.load(clipped_npy_path, mmap_mode='r')
        # Compute pixel geometry using the provided affine transform
        from rasterio.transform import xy
        a = float(transform.a)
        e = float(transform.e)
        px_w = abs(a)
        px_h = abs(e)
        pixel_area = px_w * px_h
        # Sparse accumulation
        contributions = defaultdict(float)
        chunk_target_sum = 0.0
        for r, c in zip(rows, cols):
            value = float(data[r, c])
            if value <= 0.0:
                continue
            cx, cy = xy(transform, int(r), int(c), offset='center')
            half_x = px_w / 2.0
            half_y = px_h / 2.0
            pg = box(cx - half_x, cy - half_y, cx + half_x, cy + half_y)
            # Spatial index candidate query
            try:
                cand_idx = list(sindex.intersection(pg.bounds))
            except Exception:
                # Fallback to linear scan (should be rare)
                cand_idx = list(range(len(grid)))
            if not cand_idx:
                continue
            total_area = 0.0
            areas = []
            # Compute exact intersections only for candidates
            for idx in cand_idx:
                try:
                    inter = pg.intersection(grid.iloc[idx].geometry)
                    a = float(inter.area)
                except Exception:
                    a = 0.0
                areas.append(a)
                total_area += a
            # Retry with tiny buffer to mitigate edge-touching/precision issues
            if total_area == 0.0 and cand_idx:
                try:
                    eps = min(px_w, px_h) * 1e-9
                    pg_eps = pg.buffer(eps)
                    total_area = 0.0
                    areas = []
                    for idx in cand_idx:
                        try:
                            inter = pg_eps.intersection(grid.iloc[idx].geometry)
                            a = float(inter.area)
                        except Exception:
                            a = 0.0
                        areas.append(a)
                        total_area += a
                except Exception:
                    pass
            if total_area > 0.0:
                for j, idx in enumerate(cand_idx):
                    if areas[j] > 0.0:
                        weight = areas[j] / total_area
                        contributions[idx] += value * weight
                chunk_target_sum += value * (total_area / pixel_area)
            else:
                # Even split across candidates with zero intersection area (degenerate)
                share = value / float(len(cand_idx))
                for idx in cand_idx:
                    contributions[idx] += share
        # Convert sparse dict to two arrays for compact transport
        if contributions:
            idx_array = np.fromiter(contributions.keys(), dtype=np.int64)
            val_array = np.fromiter(contributions.values(), dtype=np.float64)
        else:
            idx_array = np.empty(0, dtype=np.int64)
            val_array = np.empty(0, dtype=np.float64)
        return gid, idx_array, val_array, chunk_target_sum, ipcc_code, year
    except Exception as e:
        return gid, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0.0, ipcc_code, year


class GlobalGFEINetCDFToDGGSConverterOptimized:
    """
    Optimized multi-year global GFEI NetCDF -> DGGS converter.

    Changes vs baseline:
    - Single global pool with dynamic pixel-chunk tasks across countries
    - Per-process country grid + spatial index cache to reduce I/O and sindex rebuilds
    - Preserves numerical logic and output format
    
    GFEI-specific features:
    - Handles emission rate units (Mg km⁻² a⁻¹)
    - Uses pixel area from dataset when available (m²)
    - Extracts year from filename or time variable
    - Converts to Mg/year (Megagrams per year)
    - Processes global data for multiple years
    - Uses GFEI variable to IPCC2006 mapping
    
    Unit Conversion:
    Input: emission_rate (Mg km⁻² a⁻¹) × area (m²)
    Output: mass (Mg/year)
    
    Formula:
    1) Convert emission_rate to Mg m⁻² a⁻¹: emission_rate_Mg_km2_a1 × 1e-6
    2) mass_Mg = (emission_rate_Mg_m2_a1) × area_m2
    Where: 1 km² = 1,000,000 m² (1e6)
    """

    def __init__(self, year_to_folder, geojson_folder, output_folder, max_processes=None, chunk_size_pixels=20000):
        self.year_to_folder = year_to_folder
        self.geojson_folder = geojson_folder
        self.output_folder = output_folder
        
        # Set number of processes from environment or parameter
        if max_processes is None:
            self.max_processes = int(os.environ.get('NUM_CORES', 8))
        else:
            self.max_processes = max_processes
            
        self.chunk_size_pixels = chunk_size_pixels

        self.logger, self.log_path = _setup_logger()
        self._log(f"Logging initialized. Log file: {self.log_path}")
        self._log(f"Initialized optimized GFEI converter for years: {sorted(self.year_to_folder.keys())}")

        self._log("Loading GFEI variable -> IPCC2006 lookup table...")
        self._load_gfei_lookup()

        self._log("Loading merged global DGGS grid...")
        self._load_merged_country_grids()

        os.makedirs(self.output_folder, exist_ok=True)
        test_csv_folder = os.path.join(os.getcwd(), "test", "test_GFEI_csv")
        os.makedirs(test_csv_folder, exist_ok=True)
        self._log(f"Created/verified test_GFEI_csv folder: {test_csv_folder}")

        self._log("Creating spatial index metadata for merged grid...")
        self._create_merged_spatial_index()

        self.area_cache = None
        self._init_area_cache()

        # Prepare per-country cached grid files for worker loading
        self._prepare_country_grid_cache()

    def _log(self, message):
        print(message)
        self.logger.info(message)

    def _load_gfei_lookup(self):
        lookup_path = os.path.join("data", "lookup", "gfei_netcdf_variable_lookup.csv")
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"GFEI lookup file not found: {lookup_path}")
        df = pd.read_csv(lookup_path)
        if 'variable' not in df.columns or 'IPCC2006' not in df.columns:
            raise ValueError("Lookup CSV must contain 'variable' and 'IPCC2006' columns")
        self.variable_to_ipcc = dict(zip(df['variable'], df['IPCC2006']))
        self._log(f"Loaded {len(self.variable_to_ipcc)} variable mappings")

    def _load_merged_country_grids(self):
        merged_geojson_path = os.path.join(self.geojson_folder, "global_countries_dggs_merge.geojson")
        if not os.path.exists(merged_geojson_path):
            raise FileNotFoundError(f"Merged GeoJSON file not found: {merged_geojson_path}")
        self._log(f"Loading merged GeoJSON: {merged_geojson_path}")
        self.merged_grid = gpd.read_file(merged_geojson_path)
        required_columns = ['zoneID', 'GID']
        missing = [c for c in required_columns if c not in self.merged_grid.columns]
        if missing:
            raise ValueError(f"Missing required columns in merged GeoJSON: {missing}")
        self._log(f"Loaded merged grid with {len(self.merged_grid)} DGGS cells across countries")
        self.country_grids = {}
        self.country_geometries = {}
        for gid in self.merged_grid['GID'].unique():
            sub = self.merged_grid[self.merged_grid['GID'] == gid].copy().reset_index(drop=True)
            self.country_grids[gid] = sub
            self.country_geometries[gid] = sub.geometry.union_all()
            self._log(f"  Prepared {gid}: {len(sub)} DGGS cells")
        self._log(f"Prepared {len(self.country_grids)} countries for processing")

    def _create_merged_spatial_index(self):
        self.merged_spatial_index = self.merged_grid.sindex
        self.country_bounds = {gid: geom.bounds for gid, geom in self.country_geometries.items()}
        self._log(f"Created spatial index metadata for {len(self.country_bounds)} countries")

    def _init_area_cache(self):
        area_folder = os.path.join("data", "area_npy")
        os.makedirs(area_folder, exist_ok=True)
        area_path = os.path.join(area_folder, "gfei_global_area.npy")
        if os.path.exists(area_path):
            try:
                self.area_cache = np.load(area_path)
                self._log(f"Loaded cached global area array from {area_path} with shape {self.area_cache.shape}")
                return
            except Exception as e:
                self._log(f"Warning: Failed to load cached area array: {e}")
        if 2020 in self.year_to_folder and os.path.exists(self.year_to_folder[2020]):
            v3_folder = self.year_to_folder[2020]
            v3_files = [f for f in os.listdir(v3_folder) if f.lower().endswith('.nc')]
            if v3_files:
                v3_path = os.path.join(v3_folder, v3_files[0])
                try:
                    ds = xr.open_dataset(v3_path)
                    if 'area' in ds.variables:
                        area = ds['area'].values
                        if area.ndim == 3:
                            area = area[0, :, :]
                        elif area.ndim != 2:
                            raise ValueError(f"Unexpected area dims: {area.shape}")
                        self.area_cache = area
                        np.save(area_path, self.area_cache)
                        self._log(f"Saved global area array to {area_path} with shape {self.area_cache.shape}")
                    else:
                        self._log("Warning: v3 dataset missing 'area'; cannot build area cache")
                    ds.close()
                except Exception as e:
                    self._log(f"Warning: Failed to build area cache from v3: {e}")
        else:
            self._log("Warning: v3 folder not available; area cache not initialized")

    def _prepare_country_grid_cache(self):
        cache_dir = os.path.join("temp", "country_grids_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.country_grid_pickle = {}
        for gid, gdf in self.country_grids.items():
            p = os.path.join(cache_dir, f"{gid}.pkl")
            # Ensure consistent ordering and minimal payload
            gdf[['zoneID', 'geometry']].to_pickle(p)
            self.country_grid_pickle[gid] = p
        self._log(f"Prepared cached country grid files for {len(self.country_grid_pickle)} countries")

    @staticmethod
    def extract_variable_from_filename(filename):
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        
        # Handle different filename patterns across years
        # 2016: Global_Fuel_Exploitation_Inventory_Gas_All.nc -> Gas_All
        # 2019: Global_Fuel_Exploitation_Inventory_v2_2019_Gas_All.nc -> Gas_All  
        # 2020: Global_Fuel_Exploitation_Inventory_v3_2020_Gas_All.nc -> Gas_All
        
        try:
            # Look for the pattern: after "Inventory" or after version number
            for i, p in enumerate(parts):
                if p == "Inventory":
                    # Skip version if present (v2, v3)
                    if i + 1 < len(parts) and parts[i + 1].startswith('v') and parts[i + 1][1:].isdigit():
                        i += 1
                    # Skip year if present
                    if i + 1 < len(parts) and re.fullmatch(r"\d{4}", parts[i + 1]):
                        i += 1
                    # Return everything after
                    if i + 1 < len(parts):
                        variable = '_'.join(parts[i + 1:])
                        if variable:
                            return variable
                    break
        except Exception:
            pass
        
        # Fallback: return the last part if no pattern found
        return parts[-1] if parts else filename

    def _get_area_array(self, lat, lon, ds):
        # Prefer dataset-provided area
        if 'area' in ds.variables:
            area = ds['area'].values
            if area.ndim == 3:
                area = area[0, :, :]
            elif area.ndim != 2:
                raise ValueError(f"Unexpected area dimensions: {area.shape}")
            return area  # assumed in m^2
        # Fallback to cached area if dimensions match; align latitude orientation
        if self.area_cache is not None:
            if self.area_cache.shape == (lat.shape[0], lon.shape[0]):
                area = self.area_cache
                # Assume cached area stored with ascending latitude; flip if target lat is descending
                try:
                    if len(lat) > 1 and float(lat[0]) > float(lat[1]):
                        area = area[::-1, :]
                except Exception:
                    pass
                return area
            else:
                raise ValueError(
                    f"Cached area shape {self.area_cache.shape} does not match data shape {(lat.shape[0], lon.shape[0])}"
                )
        # As last resort, compute area from lat/lon spacing (m^2)
        return _compute_cell_area_m2(lat_centers=lat, lon_centers=lon, cell_deg=(abs(lon[1]-lon[0]) if len(lon)>1 else 0.1))

    def convert_single_to_raster(self, netcdf_path, variable_name):
        self._log(f"Processing NetCDF: {os.path.basename(netcdf_path)} for variable {variable_name}")
        ds = xr.open_dataset(netcdf_path)
        exclude = {'lat', 'lon', 'area', 'time'}
        if variable_name in ds.variables:
            var_name = variable_name
        else:
            candidates = [v for v in ds.variables if v not in exclude]
            if len(candidates) == 0:
                raise ValueError("No data variable found in NetCDF file")
            var_name = None
            for v in candidates:
                if v.lower() == variable_name.lower():
                    var_name = v
                    break
            if var_name is None:
                var_name = candidates[0]
        data = ds[var_name].values
        lat = ds['lat'].values
        lon = ds['lon'].values
        if data.ndim == 3:
            data = data[0, :, :]
        elif data.ndim != 2:
            raise ValueError(f"Unexpected variable dimensions: {data.shape}")
        area = self._get_area_array(lat, lon, ds)
        mg_per_m2 = data * 1e-6
        total_per_pixel = mg_per_m2 * area
        lat = lat[::-1]
        total_per_pixel = total_per_pixel[::-1, :]
        # Derive resolution from lon/lat arrays to avoid hard-coded 0.1°
        if len(lon) > 1:
            dx = float(abs(lon[1] - lon[0]))
        else:
            dx = 0.1
        if len(lat) > 1:
            dy = float(abs(lat[0] - lat[1]))  # lat already reversed
        else:
            dy = 0.1
        half_x = dx / 2.0
        half_y = dy / 2.0
        top_left_lon = lon.min() - half_x
        top_left_lat = lat.max() + half_y
        transform = from_origin(top_left_lon, top_left_lat, dx, dy)
        total_per_pixel = np.nan_to_num(total_per_pixel, nan=0.0, posinf=0.0, neginf=0.0)
        total_per_pixel = np.clip(total_per_pixel, 0, None)
        ds.close()
        return {
            'data': total_per_pixel,
            'transform': transform,
            'crs': 'EPSG:4326',
            'width': len(lon),
            'height': len(lat),
            'lat': lat,
            'lon': lon
        }

    def _check_variable_year_file_exists(self, variable, year):
        folder = os.path.join(os.getcwd(), "test", "test_GFEI_csv")
        fname = f"GFEI_DGGS_methane_emissions_{variable}_{year}.csv"
        return os.path.exists(os.path.join(folder, fname))

    def _check_country_year_file_exists(self, variable, gid, year):
        countries_folder = os.path.join(os.getcwd(), "test", "test_GFEI_csv", variable, "countries")
        country_fname = f"GFEI_DGGS_methane_emissions_{variable}_{gid}_{year}.csv"
        return os.path.exists(os.path.join(countries_folder, country_fname))

    def _save_country_df(self, df, variable, gid, year):
        countries_folder = os.path.join(os.getcwd(), "test", "test_GFEI_csv", variable, "countries")
        os.makedirs(countries_folder, exist_ok=True)
        country_fname = f"GFEI_DGGS_methane_emissions_{variable}_{gid}_{year}.csv"
        country_path = os.path.join(countries_folder, country_fname)
        df.to_csv(country_path, index=False)
        return country_path

    def _generate_tasks_for_country(self, gid, grid_pickle_path, clipped, ipcc_code, year):
        data = clipped['data']
        mask = data > 0
        if not np.any(mask):
            return [], None
        rows, cols = np.where(mask)
        # Save clipped data to memmap-able npy file to avoid sending large arrays
        temp_dir = os.path.join("temp", "clipped_npy")
        os.makedirs(temp_dir, exist_ok=True)
        npy_path = os.path.join(temp_dir, f"clipped_{gid}_{ipcc_code}_{year}.npy")
        # Overwrite if exists from previous run
        np.save(npy_path, data)
        n = len(rows)
        tasks = []
        chunk = self.chunk_size_pixels
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            tasks.append((gid, grid_pickle_path, npy_path, clipped['bounds'], clipped['transform'], rows[start:end], cols[start:end], ipcc_code, year))
        return tasks, npy_path

    def process_single_file(self, year, folder, netcdf_filename):
        self._log(f"\nProcessing year {year} file: {netcdf_filename}")
        variable = self.extract_variable_from_filename(netcdf_filename)
        netcdf_path = os.path.join(folder, netcdf_filename)
        if variable not in self.variable_to_ipcc:
            self._log(f"  Skipping variable '{variable}' - not found in IPCC lookup table")
            return None
        ipcc_code = self.variable_to_ipcc[variable]
        self._log(f"  Variable {variable} mapped to IPCC2006 code: {ipcc_code}")

        # Quick resume for entire variable-year
        if self._check_variable_year_file_exists(variable, year):
            self._log("  *** EXISTING VARIABLE-YEAR FILE FOUND *** Skipping processing")
            try:
                folder_out = os.path.join(os.getcwd(), "test", "test_GFEI_csv")
                fname = f"GFEI_DGGS_methane_emissions_{variable}_{year}.csv"
                path = os.path.join(folder_out, fname)
                df = pd.read_csv(path)
                self._log(f"  Loaded existing file: {path} with {len(df)} records")
                return df
            except Exception as e:
                self._log(f"  Error loading existing file: {e}; will reprocess")

        # Convert to raster once and create a single temp GeoTIFF for clipping
        try:
            raster = self.convert_single_to_raster(netcdf_path, variable)
        except Exception as e:
            self._log(f"  Error converting NetCDF to raster: {e}")
            return None

        temp_tiff_path = _create_temp_raster_from_array(raster['data'], raster['lat'], raster['lon'])

        # Prepare tasks across countries; skip countries with existing per-country CSVs
        aggregate_maps = {}
        aggregate_targets = {}
        country_dataframes = []
        tasks = []
        temp_npy_paths = []

        to_process_gids = []
        for gid in self.country_grids.keys():
            if self._check_country_year_file_exists(variable, gid, year):
                try:
                    countries_folder = os.path.join(os.getcwd(), "test", "test_GFEI_csv", variable, "countries")
                    country_fname = f"GFEI_DGGS_methane_emissions_{variable}_{gid}_{year}.csv"
                    country_path = os.path.join(countries_folder, country_fname)
                    df = pd.read_csv(country_path)
                    country_dataframes.append(df)
                    self._log(f"  Loaded existing country file for {gid}: {len(df)} records")
                except Exception as e:
                    self._log(f"  Error loading existing country file for {gid}: {e}; will reprocess")
                    to_process_gids.append(gid)
            else:
                to_process_gids.append(gid)

        # Build tasks for the remaining countries
        for gid in to_process_gids:
            bounds = self.country_geometries[gid].bounds
            clipped = _clip_raster_by_bbox(temp_tiff_path, bounds)
            t, npy_path = self._generate_tasks_for_country(gid, self.country_grid_pickle[gid], clipped, ipcc_code, year)
            if t:
                tasks.extend(t)
                temp_npy_paths.append(npy_path)
                aggregate_maps[gid] = defaultdict(float)
                aggregate_targets[gid] = 0.0
            else:
                # No non-zero pixels; produce empty df
                df = self.country_grids[gid][['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})
                df['GID'] = gid
                df[ipcc_code] = 0.0
                df['Year'] = year
                df = df[df[ipcc_code] > 0]
                if len(df) > 0:
                    path = self._save_country_df(df, variable, gid, year)
                    self._log(f"  Saved empty country file (no non-zero pixels) for {gid}: {path}")
                # Else skip saving

        # Run dynamic pool on all tasks
        if tasks:
            num_proc = min(len(self.country_grids), self.max_processes)
            self._log(f"  Using {num_proc} parallel processes for {len(tasks)} pixel-chunk tasks across {len(to_process_gids)} countries")
            start = time.time()
            with multiprocessing.Pool(processes=num_proc) as pool:
                for result in pool.imap_unordered(_pixel_chunk_worker, tasks, chunksize=1):
                    gid, idx_array, val_array, chunk_target_sum, ipcc_code_ret, year_ret = result
                    if gid in aggregate_maps:
                        for idx, val in zip(idx_array, val_array):
                            aggregate_maps[gid][int(idx)] += float(val)
                        aggregate_targets[gid] += float(chunk_target_sum)
            self._log(f"  Completed chunk processing in {time.time() - start:.2f}s")

        # Clean up temp files
        try:
            if os.path.exists(temp_tiff_path):
                os.remove(temp_tiff_path)
            for p in temp_npy_paths:
                if p and os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass

        # Build final per-country dataframes from aggregated maps
        for gid in to_process_gids:
            if gid not in aggregate_maps:
                continue
            contrib_map = aggregate_maps[gid]
            if not contrib_map:
                continue
            gdf = self.country_grids[gid]
            results = np.zeros(len(gdf))
            for idx, val in contrib_map.items():
                if 0 <= idx < len(results):
                    results[idx] = results[idx] + val
            total = float(np.sum(results))
            target_sum = float(aggregate_targets[gid])
            # Per-country scaling: scale results so DGGS sum matches intersected raster target
            if total > 0.0 and target_sum > 0.0:
                scale = target_sum / total
                results = results * scale
                self._log(f"  {gid}: Applied scaling factor: {scale:.6f}")
            self._log(f"  {gid}: Total DGGS values: {float(np.sum(results)):.6f}")
            self._log(f"  {gid}: Total raster values (intersected): {target_sum:.6f}")
            df = gdf[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})
            df['GID'] = gid
            df[ipcc_code] = results
            df['Year'] = year
            # Single small-value pass at end: threshold < 1e-6 to zero, then drop zeros
            df.loc[(df[ipcc_code] > 0) & (df[ipcc_code] < 1e-6), ipcc_code] = 0.0
            df = df[df[ipcc_code] > 0]
            if len(df) > 0:
                path = self._save_country_df(df, variable, gid, year)
                self._log(f"  Saved country CSV for {gid}: {path} with {len(df)} records")
                country_dataframes.append(df)

        if not country_dataframes:
            self._log("  No country results produced")
            return None

        combined = pd.concat(country_dataframes, ignore_index=True)
        self._log(f"  Combined {len(combined)} records for variable {variable} in {year}")
        folder_out = os.path.join(os.getcwd(), "test", "test_GFEI_csv")
        os.makedirs(folder_out, exist_ok=True)
        fname = f"GFEI_DGGS_methane_emissions_{variable}_{year}.csv"
        path = os.path.join(folder_out, fname)
        try:
            combined.to_csv(path, index=False)
            self._log(f"  *** SAVED VARIABLE-YEAR CSV *** {path}")
            self._log(f"  File size: {os.path.getsize(path)} bytes")
            self._log(f"  Records saved: {len(combined)}")
        except Exception as e:
            self._log(f"  Error saving per-variable-year CSV: {e}")
        return combined

    def process_all_years(self):
        start_time = time.time()
        self._log(f"Processing years: {sorted(self.year_to_folder.keys())}")
        all_results = []
        for year in sorted(self.year_to_folder.keys()):
            folder = self.year_to_folder[year]
            if not os.path.exists(folder):
                self._log(f"Warning: folder for year {year} not found: {folder}")
                continue
            files = [f for f in os.listdir(folder) if f.lower().endswith('.nc')]
            files.sort()
            self._log(f"Year {year}: Found {len(files)} NetCDF files")
            for nc_file in files:
                self._log(f"  Year {year}: Processing {nc_file}")
                try:
                    df = self.process_single_file(year, folder, nc_file)
                    if df is not None and len(df) > 0:
                        all_results.append(df)
                        self._log(f"  Year {year}: {len(df)} records")
                    else:
                        self._log(f"  Year {year}: No results for {nc_file}")
                except Exception as e:
                    self._log(f"  Error processing {nc_file} ({year}): {e}")

        if not all_results:
            self._log("No results found for any year")
            return None

        self._log("Combining all years into final CSV (wide by IPCC, long by Year)...")
        combined = pd.concat(all_results, ignore_index=True)
        id_cols = ['dggsID', 'GID', 'Year']
        value_columns = [c for c in combined.columns if c not in id_cols]
        # Long then group to remove any duplicates per (dggsID, GID, Year, IPCC)
        long_df = combined.melt(id_vars=id_cols, value_vars=value_columns, var_name='IPCC', value_name='value')
        long_df['value'] = long_df['value'].fillna(0.0)
        long_df = long_df.groupby(id_cols + ['IPCC'], as_index=False)['value'].sum()
        
        # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
        small_values_before = (long_df['value'] < 1e-6).sum()
        long_df['value'] = long_df['value'].where(long_df['value'] >= 1e-6, 0.0)
        self._log(f"  Step 1: Set {small_values_before} small values (< 1e-6 Mg = 1g) to zero")
        
        # Step 2: Remove rows with zero values
        rows_before = len(long_df)
        long_df = long_df[long_df['value'] > 0]
        rows_after = len(long_df)
        self._log(f"  Step 2: Removed {rows_before - rows_after} rows with zero values ({rows_after} rows remaining)")
        
        # Step 3: Create wide format
        self._log(f"  Step 3: Create wide format (pivot to IPCC columns)")
        wide_df = long_df.pivot_table(index=id_cols, columns='IPCC', values='value', aggfunc='sum', fill_value=0.0)
        wide_df = wide_df.reset_index()
        ipcc_cols = sorted([c for c in wide_df.columns if c not in id_cols])
        wide_df = wide_df[id_cols + ipcc_cols]

        output_filename = "GFEI_DGGS_methane_emissions_ALL_FILES.csv"
        output_path = os.path.join(self.output_folder, output_filename)
        wide_df.to_csv(output_path, index=False)
        self._log(f"\nFinal results saved to: {output_path}")
        self._log(f"Final output shape: {wide_df.shape}")

        total_time = time.time() - start_time
        self._log(f"\n{'='*60}")
        self._log("PROCESSING SUMMARY")
        self._log(f"{'='*60}")
        self._log(f"Years processed: {sorted(self.year_to_folder.keys())}")
        self._log(f"Total output records: {len(combined)}")
        self._log(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self._log(f"\nResuming capabilities:")
        self._log(f"  - Intermediate files saved to: test/test_GFEI_csv/[variable]/countries/")
        self._log(f"  - Script will automatically skip existing files on restart")
        # Clean up temporary files after successful completion
        try:
            self._cleanup_temp_files()
        except Exception:
            pass
        return output_path

    def _cleanup_temp_files(self):
        self._log("Cleaning up temporary files...")
        temp_folder = "temp"
        # Remove temp raster GeoTIFFs
        if os.path.exists(temp_folder):
            temp_files = [f for f in os.listdir(temp_folder) if f.startswith("temp_raster_") and f.endswith(".tiff")]
            for temp_file in temp_files:
                try:
                    os.remove(os.path.join(temp_folder, temp_file))
                except Exception:
                    pass
        # Remove clipped numpy folder
        clipped_npy_folder = os.path.join("temp", "clipped_npy")
        if os.path.exists(clipped_npy_folder):
            try:
                import shutil
                shutil.rmtree(clipped_npy_folder)
            except Exception:
                pass
        # Remove country grids cache folder
        country_cache_folder = os.path.join("temp", "country_grids_cache")
        if os.path.exists(country_cache_folder):
            try:
                import shutil
                shutil.rmtree(country_cache_folder)
            except Exception:
                pass
        # Remove temp folder if empty
        if os.path.exists(temp_folder):
            try:
                remaining = os.listdir(temp_folder)
                if not remaining:
                    os.rmdir(temp_folder)
            except Exception:
                pass
        self._log("Temporary file cleanup completed")


def main():
    # Configuration - Updated paths for HPC
    year_to_folder = {
        2016: "/home/mingke.li/GridInventory/2016-2020_Global_Fuel_Exploitation_Inventory_GFEI/2016_v1",
        2019: "/home/mingke.li/GridInventory/2016-2020_Global_Fuel_Exploitation_Inventory_GFEI/2019_v2",
        2020: "/home/mingke.li/GridInventory/2016-2020_Global_Fuel_Exploitation_Inventory_GFEI/2020_v3",
    }
    geojson_folder = "data/geojson"
    output_folder = "output"

    valid = True
    for y, p in year_to_folder.items():
        if not os.path.exists(p):
            print(f"Warning: NetCDF folder for {y} not found: {p}")
    if not os.path.exists(geojson_folder):
        print(f"Error: GeoJSON folder not found: {geojson_folder}")
        valid = False
    if not valid:
        return

    converter = GlobalGFEINetCDFToDGGSConverterOptimized(year_to_folder, geojson_folder, output_folder)
    try:
        out = converter.process_all_years()
        converter._log("\nAll years conversion completed successfully!")
        converter._log(f"Combined output file: {out}")
    except Exception as e:
        converter._log(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()