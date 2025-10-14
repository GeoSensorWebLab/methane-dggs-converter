import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import warnings
import multiprocessing
from functools import partial
import time
import logging
from datetime import datetime
import re

class USNetCDFToDGGSConverterAggregated:
    """
    Convert US NetCDF files to DGGS grid values using aggregated raster-first processing.
    This version aggregates variables by IPCC2006 codes using a lookup table,
    making the process more standardized and efficient.
    
    US-specific features:
    - Handles flux units (molecules CH₄ cm⁻² s⁻¹)
    - Uses grid_cell_area variable for pixel areas (cm²)
    - Extracts year from filename or time variable
    - Converts to Mg/year (Megagrams per year)
    
    Unit Conversion:
    Input: flux (molecules CH₄ cm⁻² s⁻¹) × area (cm²) × time (s/year)
    Output: mass (Mg/year)
    
    Formula: mass_Mg = (flux × area × seconds_per_year / AVOGADRO) × M_CH4 × (1e-6)
    Where: 1 Mg = 1,000,000 g (1e6 g)
    """
    
    def __init__(self, netcdf_folder, geojson_path, output_folder, num_cores=None):
        """
        Initialize the converter.
        
        Args:
            netcdf_folder (str): Path to folder containing NetCDF files
            geojson_path (str): Path to the DGGS GeoJSON file
            output_folder (str): Path to output folder for CSV files
            num_cores (int): Number of CPU cores to use for parallel processing
        """
        self.netcdf_folder = netcdf_folder
        self.geojson_path = geojson_path
        self.output_folder = output_folder
        
        # Set number of cores from environment or parameter
        if num_cores is None:
            self.num_cores = int(os.environ.get('NUM_CORES', 8))
        else:
            self.num_cores = num_cores
        
        # Constants for unit conversion
        self.AVOGADRO = 6.022e23  # molecules/mol
        self.M_CH4 = 16.04        # g/mol
        self.G_TO_MG = 1e-6       # grams to megagrams (Mg): 1 Mg = 1,000,000 g
        
        # Setup logging
        self._setup_logging()
        
        # Load the IPCC2006 variable lookup table
        self.log_message("Loading IPCC2006 variable lookup table...")
        self._load_ipcc_lookup()
        
        # Load the DGGS grid
        self.log_message("Loading DGGS grid...")
        self.dggs_grid = gpd.read_file(geojson_path)
        self.log_message(f"Loaded {len(self.dggs_grid)} DGGS cells")
        
        # Check if zoneID column exists
        if 'zoneID' not in self.dggs_grid.columns:
            raise ValueError("GeoJSON file must contain 'zoneID' column")
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Get list of NetCDF files
        self.netcdf_files = [f for f in os.listdir(netcdf_folder) if f.endswith('.nc')]
        self.log_message(f"Found {len(self.netcdf_files)} NetCDF files")
        
        # Create spatial index for DGGS cells for fast intersection queries
        self.log_message("Creating spatial index for DGGS cells...")
        self._create_spatial_index()
    
    def _setup_logging(self):
        """Setup logging to both console and file."""
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"us_netcdf_to_dggs_conversion_{timestamp}.log"
        log_path = os.path.join(log_folder, log_filename)
        
        # Create a new logger specifically for this script
        self.logger = logging.getLogger(f"us_converter_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid conflicts
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # Add file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
        
        self.log_path = log_path
        self.log_message(f"Logging initialized. Log file: {log_path}")
    
    def log_message(self, message):
        """Log message to both console and file."""
        print(message)
        self.logger.info(message)
    
    def _create_spatial_index(self):
        """Create spatial index for fast DGGS cell intersection queries."""
        # Create bounding boxes for all DGGS cells
        self.dggs_bounds = []
        for idx, row in self.dggs_grid.iterrows():
            geom = row.geometry
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            self.dggs_bounds.append((bounds, idx))
        
        self.log_message(f"  Created spatial index for {len(self.dggs_bounds)} DGGS cells")
    
    def extract_year_from_filename(self, filename):
        """
        Extract year from NetCDF filename.
        
        Args:
            filename (str): NetCDF filename
            
        Returns:
            int: Year extracted from filename
        """
        # Extract year from filename like 'Gridded_GHGI_Methane_v2_2012.nc'
        year_match = re.search(r'(\d{4})\.nc$', filename)
        if year_match:
            return int(year_match.group(1))
        else:
            # Fallback: try to extract from the filename
            year_match = re.search(r'(\d{4})', filename)
            if year_match:
                return int(year_match.group(1))
            else:
                raise ValueError(f"Could not extract year from filename: {filename}")
    
    def _load_ipcc_lookup(self):
        """Load the IPCC2006 variable lookup table."""
        lookup_path = "data/lookup/us_netcdf_variable_lookup.csv"
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"IPCC lookup file not found: {lookup_path}")
        
        self.ipcc_lookup = pd.read_csv(lookup_path)
        self.log_message(f"Loaded IPCC lookup table with {len(self.ipcc_lookup)} variable mappings")
        
        # Create a dictionary for fast lookup
        self.variable_to_ipcc = dict(zip(self.ipcc_lookup['variable'], self.ipcc_lookup['IPCC2006']))
        
        # Log some example mappings
        self.log_message("Example variable mappings:")
        for i, (var, ipcc) in enumerate(self.variable_to_ipcc.items()):
            if i < 5:  # Show first 5 mappings
                self.log_message(f"  {var} -> {ipcc}")
            else:
                break
        self.log_message(f"  ... and {len(self.variable_to_ipcc) - 5} more mappings")

    def aggregate_variables_by_ipcc_code(self, nc_data):
        """
        Aggregate variables by IPCC2006 codes using the lookup table.
        New policy: skip variables not in lookup table and report them.
        
        Args:
            nc_data (xarray.Dataset): NetCDF dataset
            
        Returns:
            dict: Dictionary with aggregated data for each IPCC2006 code
        """
        self.log_message("Aggregating variables by IPCC2006 codes...")
        
        # Get variables (excluding coordinates, time, and area)
        exclude_vars = ['lat', 'lon', 'time', 'grid_cell_area']
        variables = [var for var in nc_data.variables if var not in exclude_vars]
        
        self.log_message(f"  Found {len(variables)} variables to process")
        
        # Group variables by IPCC2006 code
        ipcc_groups = {}
        unmapped_variables = []
        
        for var in variables:
            if var in self.variable_to_ipcc:
                ipcc_code = self.variable_to_ipcc[var]
                if ipcc_code not in ipcc_groups:
                    ipcc_groups[ipcc_code] = []
                ipcc_groups[ipcc_code].append(var)
            else:
                unmapped_variables.append(var)
                self.log_message(f"  Skipping variable '{var}' - not found in IPCC lookup table")
        
        if unmapped_variables:
            self.log_message(f"  Skipped {len(unmapped_variables)} variables not found in IPCC lookup table:")
            for var in unmapped_variables:
                self.log_message(f"    {var}")
        
        self.log_message(f"  Found {len(ipcc_groups)} unique IPCC2006 codes:")
        for ipcc_code, vars_list in ipcc_groups.items():
            self.log_message(f"    {ipcc_code}: {vars_list}")
        
        # Aggregate variables for each IPCC2006 code
        aggregated_data = {}
        for ipcc_code, var_list in ipcc_groups.items():
            self.log_message(f"  Aggregating {ipcc_code}: {var_list}")
            clean_stack = []
            for var in var_list:
                arr = nc_data[var].values
                # Handle optional time dimension
                if arr.ndim == 3:
                    arr = arr[0, :, :]
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, 0, None)
                clean_stack.append(arr)
            if not clean_stack:
                continue
            aggregated_array = np.sum(clean_stack, axis=0)
            aggregated_data[ipcc_code] = aggregated_array
            self.log_message(f"    Aggregated into single array: {aggregated_array.shape}")
        
        return aggregated_data
    
    def convert_aggregated_to_raster(self, netcdf_path):
        """
        Convert a NetCDF file to raster format with pre-aggregated variables by IPCC2006 codes.
        
        Args:
            netcdf_path (str): Path to NetCDF file
            
        Returns:
            dict: Dictionary containing raster data for each aggregated IPCC2006 code
        """
        self.log_message(f"Processing NetCDF: {os.path.basename(netcdf_path)}")
        
        # Load NetCDF data
        nc_data = xr.open_dataset(netcdf_path)
        
        # Aggregate variables by IPCC2006 codes
        aggregated_data = self.aggregate_variables_by_ipcc_code(nc_data)
        
        # Extract coordinates and area
        lat = nc_data['lat'].values
        lon = nc_data['lon'].values
        area = nc_data['grid_cell_area'].values if 'grid_cell_area' in nc_data.variables else None
        
        if area is None:
            raise ValueError("Required variable 'grid_cell_area' not found in NetCDF file")
        
        # Handle area variable dimensions if it has time dimension
        if area.ndim == 4:  # (time, lat, lon, other)
            area = area[0, :, :, :]  # Take first time step
        elif area.ndim == 3:  # (time, lat, lon)
            area = area[0, :, :]  # Take first time step
        elif area.ndim == 2:  # (lat, lon)
            area = area  # Already 2D
        else:
            raise ValueError(f"Unexpected area dimensions: {area.shape}")
        
        # Ensure latitude is in correct order (top to bottom)
        lat = lat[::-1]
        
        # Create transform for raster
        # Since lat/lon are pixel centers, we need to shift by half a pixel
        cell_size = 0.1  # 0.1 degree resolution
        half_cell = cell_size / 2.0
        
        # Calculate the top-left corner of the raster
        # lon.min() and lat.max() are pixel centers, so subtract half a pixel
        top_left_lon = lon.min() - half_cell
        top_left_lat = lat.max() + half_cell
        
        transform = from_origin(top_left_lon, top_left_lat, cell_size, cell_size)
        
        raster_data = {}
        
        for ipcc_code, aggregated_array in aggregated_data.items():
            self.log_message(f"  Processing aggregated IPCC2006 code: {ipcc_code}")
            
            # Handle multi-dimensional data - take first time step if time dimension exists
            if aggregated_array.ndim == 4:  # (time, lat, lon, other)
                aggregated_array = aggregated_array[0, :, :, :]  # Take first time step
            elif aggregated_array.ndim == 3:  # (time, lat, lon)
                aggregated_array = aggregated_array[0, :, :]  # Take first time step
            elif aggregated_array.ndim == 2:  # (lat, lon)
                aggregated_array = aggregated_array  # Already 2D
            else:
                raise ValueError(f"Unexpected aggregated array dimensions: {aggregated_array.shape}")
            
            # Reverse latitude order to match raster format
            aggregated_array = aggregated_array[::-1, :]
            
            # Convert flux units: molecules CH₄ cm⁻² s⁻¹ to Mg/year
            # Using the conversion logic:
            # mass_Mg = (flux * area * seconds_per_year / AVOGADRO) * M_CH4 * G_TO_MG
            
            # Calculate seconds per year (assuming annual data)
            seconds_per_year = 365 * 24 * 3600
            
            # Convert to mass per pixel in Mg/year
            # flux: molecules CH₄ cm⁻² s⁻¹
            # area: cm² (grid_cell_area)
            # result: Mg/year per pixel
            # 
            # Unit conversion breakdown:
            # 1. flux * area * seconds_per_year → molecules CH₄ per pixel per year
            # 2. / AVOGADRO → moles CH₄ per pixel per year  
            # 3. * M_CH4 → grams CH₄ per pixel per year
            # 4. * G_TO_MG (1e-6) → megagrams (Mg) CH₄ per pixel per year
            mass_per_pixel = (aggregated_array * area * seconds_per_year / self.AVOGADRO) * self.M_CH4 * self.G_TO_MG
            
            # Handle missing data
            mass_per_pixel = np.nan_to_num(mass_per_pixel, nan=0.0)
            
            # Validate conversion results (sanity check)
            if np.any(mass_per_pixel > 1e6):  # If any value > 1 million Mg/year
                self.log_message(f"      Warning: Large values detected in {ipcc_code}, max: {np.max(mass_per_pixel):.2e} Mg/year")
            if np.any(mass_per_pixel < 0):
                self.log_message(f"      Warning: Negative values detected in {ipcc_code}, min: {np.min(mass_per_pixel):.2e} Mg/year")
                # Clip negative values to 0
                mass_per_pixel = np.clip(mass_per_pixel, 0, None)
            
            raster_data[ipcc_code] = {
                'data': mass_per_pixel,
                'transform': transform,
                'crs': 'EPSG:4326',
                'width': len(lon),
                'height': len(lat),
                'lat': lat,
                'lon': lon
            }
        
        nc_data.close()
        return raster_data
    
    def calculate_weighted_values_raster_first(self, raster_data, ipcc_code):
        """
        Calculate weighted values using raster-first approach.
        Process non-zero pixels first, then find intersecting DGGS cells.
        Uses area-weighted distribution for accurate results.
        
        Args:
            raster_data (dict): Raster data dictionary
            ipcc_code (str): Name of the aggregated IPCC2006 code being processed
            
        Returns:
            list: Weighted values for each DGGS cell
        """
        self.log_message(f"    Processing {ipcc_code} using raster-first approach...")
        
        # Initialize results array
        results = np.zeros(len(self.dggs_grid))
        distributed_raster_total = 0.0
        
        # Get non-zero pixels
        data = raster_data['data']
        non_zero_mask = data > 0
        
        if not np.any(non_zero_mask):
            self.log_message(f"      No non-zero pixels found in {ipcc_code}")
            return results.tolist()
        
        # Get coordinates of non-zero pixels
        non_zero_coords = np.where(non_zero_mask)
        num_non_zero = len(non_zero_coords[0])
        self.log_message(f"      Found {num_non_zero} non-zero pixels to process")
        
        # Process each non-zero pixel
        start_time = time.time()
        processed_pixels = 0
        
        for i in range(num_non_zero):
            row, col = non_zero_coords[0][i], non_zero_coords[1][i]
            pixel_value = data[row, col]
            
            # Convert pixel coordinates to geographic coordinates
            pixel_lon = raster_data['lon'][col]
            pixel_lat = raster_data['lat'][row]
            
            # Create pixel geometry (small box around the pixel center)
            pixel_size = 0.1  # 0.1 degree resolution
            half_pixel = pixel_size / 2.0
            
            pixel_geom = box(
                pixel_lon - half_pixel, 
                pixel_lat - half_pixel,
                pixel_lon + half_pixel, 
                pixel_lat + half_pixel
            )
            
            # Find intersecting DGGS cells
            intersecting_cells = self._find_intersecting_cells(pixel_geom)
            
            # Distribute pixel value to intersecting cells using area-weighted distribution
            if intersecting_cells:
                # Calculate total intersection area for this pixel
                total_intersection_area = 0
                intersection_areas = []
                
                for cell_idx in intersecting_cells:
                    try:
                        # Calculate actual intersection area between pixel and DGGS cell
                        cell_geometry = self.dggs_grid.iloc[cell_idx].geometry
                        intersection = pixel_geom.intersection(cell_geometry)
                        intersection_area = intersection.area
                        intersection_areas.append(intersection_area)
                        total_intersection_area += intersection_area
                    except Exception as e:
                        # Handle geometry errors gracefully
                        self.log_message(f"        Warning: Error calculating intersection for cell {cell_idx}: {e}")
                        intersection_areas.append(0.0)
                
                # Distribute pixel value proportionally to intersection areas
                if total_intersection_area > 0:
                    for j, cell_idx in enumerate(intersecting_cells):
                        # Weight by intersection area ratio
                        area_ratio = intersection_areas[j] / total_intersection_area
                        weighted_value = pixel_value * area_ratio
                        results[cell_idx] += weighted_value
                    distributed_raster_total += float(pixel_value)
                else:
                    # Fallback: if no valid intersection area, distribute equally
                    value_per_cell = pixel_value / len(intersecting_cells)
                    for cell_idx in intersecting_cells:
                        results[cell_idx] += value_per_cell
                    distributed_raster_total += float(pixel_value)
            
            processed_pixels += 1
            
            # Progress update every 1000 pixels
            if processed_pixels % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_pixels / elapsed
                remaining = (num_non_zero - processed_pixels) / rate
                self.log_message(f"        Processed {processed_pixels}/{num_non_zero} pixels "
                      f"({processed_pixels/num_non_zero*100:.1f}%) - "
                      f"ETA: {remaining:.1f}s")
        
        total_time = time.time() - start_time
        self.log_message(f"      Completed processing {num_non_zero} pixels in {total_time:.2f} seconds")
        self.log_message(f"      Average: {total_time/num_non_zero*1000:.2f} ms per pixel")
        
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
        """
        Find DGGS cells that intersect with a pixel geometry.
        Uses spatial indexing for fast queries.
        
        Args:
            pixel_geom: Shapely geometry of the pixel
            
        Returns:
            list: Indices of intersecting DGGS cells
        """
        intersecting_cells = []
        
        # Get pixel bounds
        pixel_bounds = pixel_geom.bounds  # (minx, miny, maxx, maxy)
        
        # Check each DGGS cell for intersection
        for bounds, cell_idx in self.dggs_bounds:
            # Quick bounding box check first
            if self._bounds_intersect(pixel_bounds, bounds):
                # Detailed geometry intersection check
                try:
                    if pixel_geom.intersects(self.dggs_grid.iloc[cell_idx].geometry):
                        intersecting_cells.append(cell_idx)
                except Exception as e:
                    # Skip cells with invalid geometry
                    continue
        
        return intersecting_cells
    
    def _bounds_intersect(self, bounds1, bounds2):
        """
        Quick check if two bounding boxes intersect.
        
        Args:
            bounds1: (minx, miny, maxx, maxy) for first geometry
            bounds2: (minx, miny, maxx, maxy) for second geometry
            
        Returns:
            bool: True if bounding boxes intersect
        """
        return not (bounds1[2] < bounds2[0] or  # right < left
                   bounds1[0] > bounds2[2] or  # left > right
                   bounds1[3] < bounds2[1] or  # top < bottom
                   bounds1[1] > bounds2[3])    # bottom > top
    
    def calculate_weighted_values_parallel_raster_first(self, raster_data_dict, ipcc_codes):
        """
        Calculate weighted values for multiple aggregated IPCC2006 codes in parallel using raster-first approach.
        
        Args:
            raster_data_dict (dict): Dictionary containing raster data for multiple aggregated IPCC2006 codes
            ipcc_codes (list): List of IPCC2006 codes to process
            
        Returns:
            dict: Dictionary with IPCC2006 codes as keys and weighted values as values
        """
        self.log_message(f"    Processing {len(ipcc_codes)} aggregated IPCC2006 codes in parallel using raster-first approach...")
        
        # Determine number of processes to use - use self.num_cores
        num_processes = min(len(ipcc_codes), self.num_cores)
        self.log_message(f"    Using {num_processes} parallel processes")
        
        # Create partial function with fixed arguments
        process_func = partial(self._process_single_ipcc_code_raster_first, dggs_grid=self.dggs_grid)
        
        # Prepare arguments for multiprocessing
        args_list = [(ipcc_code, raster_data_dict[ipcc_code]) for ipcc_code in ipcc_codes]
        
        # Process in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_func, args_list)
        
        # Convert results to dictionary
        result_dict = {}
        for ipcc_code, weighted_values in results:
            result_dict[ipcc_code] = weighted_values
        
        return result_dict
    
    def _check_existing_csv_files(self, test_output_folder):
        """
        Check which year CSV files already exist in the test output folder.
        
        Args:
            test_output_folder (str): Path to the test output folder
            
        Returns:
            set: Set of years that already have CSV files
        """
        existing_years = set()
        
        if not os.path.exists(test_output_folder):
            return existing_years
        
        # Look for files matching the pattern: US_DGGS_methane_emissions_YYYY.csv
        for filename in os.listdir(test_output_folder):
            if filename.startswith("US_DGGS_methane_emissions_") and filename.endswith(".csv"):
                # Extract year from filename
                year_match = re.search(r'US_DGGS_methane_emissions_(\d{4})\.csv', filename)
                if year_match:
                    year = int(year_match.group(1))
                    existing_years.add(year)
        
        return existing_years
    
    def _process_single_ipcc_code_raster_first(self, args, dggs_grid):
        """
        Process a single aggregated IPCC2006 code for multiprocessing using raster-first approach.
        Uses area-weighted distribution for accurate results.
        
        Args:
            args (tuple): (ipcc_code, raster_data)
            dggs_grid: DGGS grid data
            
        Returns:
            tuple: (ipcc_code, weighted_values)
        """
        ipcc_code, raster_data = args
        
        try:
            # Use the existing instance methods instead of creating a new converter
            # This prevents multiple logging configurations
            weighted_values = self.calculate_weighted_values_raster_first(
                raster_data, ipcc_code
            )
            
            return ipcc_code, weighted_values
            
        except Exception as e:
            # Use print for errors in multiprocessing to avoid logging issues
            print(f"      Error processing {ipcc_code}: {e}")
            # Return zeros if processing fails
            return ipcc_code, [0.0] * len(dggs_grid)
    
    def process_all_netcdf_files(self):
        """
        Process all NetCDF files and combine them into a single CSV output.
        Variables are aggregated by IPCC2006 codes before raster conversion.
        Each file is processed separately and saved individually, then combined.
        Skips processing for years that already have CSV files.
        """
        start_time = time.time()
        self.log_message(f"Processing all {len(self.netcdf_files)} NetCDF files with IPCC2006 code aggregation...")
        
        # Check which year CSV files already exist
        test_output_folder = "test/test_us_csv"
        os.makedirs(test_output_folder, exist_ok=True)
        
        existing_years = self._check_existing_csv_files(test_output_folder)
        self.log_message(f"Found {len(existing_years)} existing year CSV files: {sorted(existing_years)}")
        
        # List to store all individual dataframes
        all_dataframes = []
        
        # Dictionary to store all aggregated IPCC2006 codes from all files
        all_ipcc_codes = {}
        file_ipcc_mapping = {}
        
        # Process each NetCDF file separately
        files_to_process = []
        for netcdf_filename in self.netcdf_files:
            try:
                year = self.extract_year_from_filename(netcdf_filename)
                if year not in existing_years:
                    files_to_process.append((netcdf_filename, year))
                else:
                    self.log_message(f"Skipping {netcdf_filename} (year {year} already processed)")
            except ValueError as e:
                self.log_message(f"Warning: Could not extract year from {netcdf_filename}: {e}")
                files_to_process.append((netcdf_filename, None))
        
        self.log_message(f"Will process {len(files_to_process)} files, skip {len(self.netcdf_files) - len(files_to_process)} existing files")
        
        # Load existing CSV files first
        if existing_years:
            self.log_message(f"Loading {len(existing_years)} existing CSV files...")
            for year in existing_years:
                existing_csv_path = os.path.join(test_output_folder, f"US_DGGS_methane_emissions_{year}.csv")
                if os.path.exists(existing_csv_path):
                    try:
                        existing_df = pd.read_csv(existing_csv_path)
                        all_dataframes.append(existing_df)
                        self.log_message(f"  Loaded existing CSV for year {year}: {existing_df.shape}")
                        
                        # Track IPCC codes from existing files
                        value_cols = [col for col in existing_df.columns if col not in ['dggsID', 'Year']]
                        for col in value_cols:
                            all_ipcc_codes[col] = existing_df[col].values
                            file_ipcc_mapping[col] = f"existing_{year}.csv"
                    except Exception as e:
                        self.log_message(f"  Error loading existing CSV for year {year}: {e}")
        
        # Process new files
        for file_index, (netcdf_filename, year) in enumerate(files_to_process):
            self.log_message(f"\n{'='*60}")
            self.log_message(f"PROCESSING FILE {file_index + 1}/{len(files_to_process)}: {netcdf_filename}")
            self.log_message(f"{'='*60}")
            
            netcdf_path = os.path.join(self.netcdf_folder, netcdf_filename)
            
            if not os.path.exists(netcdf_path):
                self.log_message(f"Warning: NetCDF file not found: {netcdf_path}")
                continue
            
            try:
                # Extract year from filename if not already extracted
                if year is None:
                    year = self.extract_year_from_filename(netcdf_filename)
                self.log_message(f"  Processing year: {year}")
                
                # Initialize result dataframe for this year with DGGS cell IDs (zoneID)
                year_result_df = self.dggs_grid[['zoneID']].copy()
                # Rename zoneID to dggsID
                year_result_df = year_result_df.rename(columns={'zoneID': 'dggsID'})
                
                # Convert NetCDF to aggregated raster data by IPCC2006 codes
                raster_data = self.convert_aggregated_to_raster(netcdf_path)
                
                # Process all aggregated IPCC2006 codes using raster-first approach
                self.log_message(f"Processing {len(raster_data)} aggregated IPCC2006 codes...")
                
                if len(raster_data) > 1:
                    # Use parallel processing for multiple IPCC2006 codes
                    ipcc_codes = list(raster_data.keys())
                    weighted_values_dict = self.calculate_weighted_values_parallel_raster_first(raster_data, ipcc_codes)
                    
                    # Add all IPCC2006 codes to result dataframe for this year
                    for ipcc_code, weighted_values in weighted_values_dict.items():
                        # Use IPCC2006 code as column name
                        year_result_df[ipcc_code] = weighted_values
                        all_ipcc_codes[ipcc_code] = weighted_values
                        file_ipcc_mapping[ipcc_code] = netcdf_filename
                else:
                    # Single IPCC2006 code - use sequential processing
                    ipcc_code = list(raster_data.keys())[0]
                    self.log_message(f"  Processing aggregated IPCC2006 code: {ipcc_code}")
                    weighted_values = self.calculate_weighted_values_raster_first(raster_data[ipcc_code], ipcc_code)
                    
                    # Use IPCC2006 code as column name
                    year_result_df[ipcc_code] = weighted_values
                    all_ipcc_codes[ipcc_code] = weighted_values
                    file_ipcc_mapping[ipcc_code] = netcdf_filename
                
                # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
                value_columns = [col for col in year_result_df.columns if col != 'dggsID']
                self.log_message(f"  Step 1: Filtering small values (< 1e-6 Mg = 1g)")
                small_values_before = 0
                for col in value_columns:
                    small_mask = year_result_df[col] < 1e-6
                    small_count = small_mask.sum()
                    small_values_before += small_count
                    year_result_df[col] = year_result_df[col].where(year_result_df[col] >= 1e-6, 0.0)
                self.log_message(f"    Set {small_values_before} small values to zero across all columns")
                
                # Step 2: Remove rows where all values are 0 (except dggsID) for this year
                rows_before = len(year_result_df)
                year_result_df = year_result_df[~(year_result_df[value_columns] == 0).all(axis=1)]
                rows_after = len(year_result_df)
                self.log_message(f"  Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
                
                # Add Year column with specific year for this file
                year_result_df['Year'] = year
                
                # Save individual year CSV to test_us_csv folder
                test_output_folder = "test/test_us_csv"
                os.makedirs(test_output_folder, exist_ok=True)
                
                individual_filename = f"US_DGGS_methane_emissions_{year}.csv"
                individual_path = os.path.join(test_output_folder, individual_filename)
                year_result_df.to_csv(individual_path, index=False)
                
                self.log_message(f"  Individual year CSV saved to: {individual_path}")
                self.log_message(f"  Year {year} shape: {year_result_df.shape}")
                
                # Add to list for final combination
                all_dataframes.append(year_result_df)
                
                self.log_message(f"  File {file_index + 1} completed successfully")
                
            except Exception as e:
                self.log_message(f"Error processing file {netcdf_filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.log_message(f"\n{'='*60}")
        self.log_message(f"ALL FILES PROCESSED - COMBINING INTO FINAL OUTPUT")
        self.log_message(f"{'='*60}")
        
        # Combine all individual dataframes
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            self.log_message(f"Combined all {len(all_dataframes)} year dataframes")
            self.log_message(f"Combined shape: {combined_df.shape}")
            
            # Get value columns for processing
            value_columns = [col for col in combined_df.columns if col not in ['dggsID', 'Year']]
            
            # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
            self.log_message(f"Final processing - Step 1: Filtering small values (< 1e-6 Mg = 1g)")
            small_values_before = 0
            for col in value_columns:
                small_mask = combined_df[col] < 1e-6
                small_count = small_mask.sum()
                small_values_before += small_count
                combined_df[col] = combined_df[col].where(combined_df[col] >= 1e-6, 0.0)
            self.log_message(f"  Set {small_values_before} small values to zero across all columns")
            
            # Step 2: Remove rows where all values are 0 (except dggsID and Year) from combined data
            rows_before = len(combined_df)
            combined_df = combined_df[~(combined_df[value_columns] == 0).all(axis=1)]
            rows_after = len(combined_df)
            self.log_message(f"Final processing - Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
            
            # Steps complete (only Step 1 and Step 2 retained by design)
            self.log_message(f"After removing zero-value rows: {len(combined_df)} rows with data")
            
            # Save combined CSV to output folder
            output_filename = "US_DGGS_methane_emissions_ALL_FILES.csv"
            output_path = os.path.join(self.output_folder, output_filename)
            
            combined_df.to_csv(output_path, index=False)
            self.log_message(f"\nCombined results saved to: {output_path}")
            self.log_message(f"Final output shape: {combined_df.shape}")
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Show summary of what was processed
            self.log_message(f"\n{'='*60}")
            self.log_message(f"PROCESSING SUMMARY")
            self.log_message(f"{'='*60}")
            self.log_message(f"Total NetCDF files available: {len(self.netcdf_files)}")
            self.log_message(f"Files processed this run: {len(files_to_process)}")
            self.log_message(f"Files skipped (already exist): {len(self.netcdf_files) - len(files_to_process)}")
            self.log_message(f"Existing CSV files loaded: {len(existing_years)}")
            self.log_message(f"Total aggregated IPCC2006 codes: {len(all_ipcc_codes)}")
            self.log_message(f"Individual year CSVs saved to: {test_output_folder}")
            self.log_message(f"Combined output columns: {len(combined_df.columns)}")
            self.log_message(f"Combined output rows: {len(combined_df)}")
            self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            
            # Show file-IPCC code mapping
            self.log_message(f"\nFile-IPCC Code Mapping:")
            for ipcc_code, filename in file_ipcc_mapping.items():
                self.log_message(f"  {ipcc_code} -> {filename}")
            
            return output_path
        else:
            self.log_message("No dataframes to combine - processing failed for all files")
            return None


def main():
    """
    Main function to run the US aggregated NetCDF to DGGS conversion.
    """
    # Configuration - Updated paths for HPC
    netcdf_folder = "/home/mingke.li/GridInventory/2012-2018_U.S._Anthropogenic_Methane_Emissions"
    geojson_path = "data/geojson/global_countries_dggs_merge/United_States_USA_grid.geojson"
    output_folder = "output"
    
    # Check if paths exist
    if not os.path.exists(netcdf_folder):
        print(f"Error: NetCDF folder not found: {netcdf_folder}")
        return
    
    if not os.path.exists(geojson_path):
        print(f"Error: GeoJSON file not found: {geojson_path}")
        return
    
    # Create converter with HPC configuration
    converter = USNetCDFToDGGSConverterAggregated(netcdf_folder, geojson_path, output_folder)
    
    # Process all NetCDF files and combine into single CSV
    try:
        output_path = converter.process_all_netcdf_files()
        converter.log_message(f"\nAll files conversion completed successfully!")
        converter.log_message(f"Combined output file: {output_path}")
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()