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

class USOGNetCDFToDGGSConverterAggregated:
    """
    Convert US Oil and Gas NetCDF files to DGGS grid values using aggregated raster-first processing.
    This version aggregates variables by IPCC2006 codes using a lookup table,
    making the process more standardized and efficient.
    
    US OG-specific features:
    - Handles flux units (kg/h)
    - No area calculation needed (already total emissions per pixel)
    - Extracts year from filename
    - Converts to Mg/year (Megagrams per year)
    - Processes single file for 2021
    
    Unit Conversion:
    Input: flux (kg/h) × time (h/year)
    Output: mass (Mg/year)
    
    Formula: mass_Mg = flux_kg_h × hours_per_year × (1e-3)
    Where: 1 Mg = 1,000 kg (1e3 kg)
    """
    
    def __init__(self, netcdf_folder, geojson_path, output_folder, num_cores=None):
        """
        Initialize the converter.
        
        Args:
            netcdf_folder (str): Path to folder containing US OG NetCDF files
            geojson_path (str): Path to the US DGGS GeoJSON file
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
        self.KG_TO_MG = 1e-3  # kilograms to megagrams (Mg): 1 Mg = 1,000 kg
        
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
        log_filename = f"us_og_netcdf_to_dggs_conversion_{timestamp}.log"
        log_path = os.path.join(log_folder, log_filename)
        
        # Create a new logger specifically for this script
        self.logger = logging.getLogger(f"us_og_converter_{timestamp}")
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
    
    def _load_ipcc_lookup(self):
        """Load the IPCC2006 variable lookup table."""
        lookup_path = "data/lookup/us_OG_netcdf_variable_lookup.csv"
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
        # Extract year from filename like 'EI_ME_2021_inventory_CONUS_point1_degrees_v1.nc'
        year_match = re.search(r'(\d{4})', filename)
        if year_match:
            return int(year_match.group(1))
        else:
            raise ValueError(f"Could not extract year from filename: {filename}")

    def aggregate_variables_by_ipcc_code(self, nc_data):
        """
        Aggregate variables by IPCC2006 codes using the lookup table.
        
        Args:
            nc_data (xarray.Dataset): NetCDF dataset
            
        Returns:
            dict: Dictionary with aggregated data for each IPCC2006 code
        """
        self.log_message("Aggregating variables by IPCC2006 codes...")
        
        # Get variables (excluding coordinates)
        exclude_vars = ['lat', 'lon', 'time']
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
        nc_data = xr.open_dataset(netcdf_path, decode_times=False)
        
        # Aggregate variables by IPCC2006 codes
        aggregated_data = self.aggregate_variables_by_ipcc_code(nc_data)
        
        # Extract coordinates
        lat = nc_data['lat'].values
        lon = nc_data['lon'].values
        
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
            
            # Convert flux units: kg/h to Mg/year
            # Using the conversion logic:
            # mass_Mg = flux_kg_h × hours_per_year × KG_TO_MG
            
            # Calculate hours per year (assuming annual data)
            hours_per_year = 365 * 24
            
            # Convert to mass per pixel in Mg/year
            # flux: kg/h
            # result: Mg/year per pixel
            # 
            # Unit conversion breakdown:
            # 1. flux × hours_per_year → kg per pixel per year
            # 2. × KG_TO_MG (1e-3) → megagrams (Mg) per pixel per year
            mass_per_pixel = aggregated_array * hours_per_year * self.KG_TO_MG
            
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
    
    def process_single_netcdf_file(self, netcdf_filename):
        """
        Process a single NetCDF file and save the result.
        
        Args:
            netcdf_filename (str): Name of the NetCDF file to process
            
        Returns:
            str: Path to the output CSV file
        """
        self.log_message(f"\n{'='*60}")
        self.log_message(f"PROCESSING FILE: {netcdf_filename}")
        self.log_message(f"{'='*60}")
        
        netcdf_path = os.path.join(self.netcdf_folder, netcdf_filename)
        
        if not os.path.exists(netcdf_path):
            self.log_message(f"Error: NetCDF file not found: {netcdf_path}")
            return None
        
        try:
            # Extract year from filename
            year = self.extract_year_from_filename(netcdf_filename)
            self.log_message(f"  Year: {year}")
            
            # Initialize result dataframe with DGGS cell IDs (zoneID)
            result_df = self.dggs_grid[['zoneID']].copy()
            # Rename zoneID to dggsID
            result_df = result_df.rename(columns={'zoneID': 'dggsID'})
            
            # Convert NetCDF to aggregated raster data by IPCC2006 codes
            raster_data = self.convert_aggregated_to_raster(netcdf_path)
            
            # Process all aggregated IPCC2006 codes using raster-first approach
            total_codes = len(raster_data)
            self.log_message(f"Processing {total_codes} aggregated IPCC2006 codes...")
            file_ipcc_mapping = {}
            processed_idx = 0
            for ipcc_code, raster_info in raster_data.items():
                self.log_message(f"  Processing aggregated IPCC2006 code: {ipcc_code}")
                weighted_values = self.calculate_weighted_values_raster_first(
                    raster_info, ipcc_code
                )
                
                # Use IPCC2006 code as column name
                result_df[ipcc_code] = weighted_values
                file_ipcc_mapping[ipcc_code] = netcdf_filename
                processed_idx += 1
                self.log_message(
                    f"      Processed {processed_idx}/{total_codes} IPCC codes ({processed_idx/total_codes*100:.1f}%)"
                )
            
            # Step 1: Set small values (< 1e-6 Mg = 1g) to zero
            value_columns = [col for col in result_df.columns if col != 'dggsID']
            self.log_message(f"  Step 1: Filtering small values (< 1e-6 Mg = 1g)")
            small_values_before = 0
            for col in value_columns:
                small_mask = result_df[col] < 1e-6
                small_count = small_mask.sum()
                small_values_before += small_count
                result_df[col] = result_df[col].where(result_df[col] >= 1e-6, 0.0)
            self.log_message(f"    Set {small_values_before} small values to zero across all columns")
            
            # Step 2: Remove rows where all values are 0 (except dggsID)
            rows_before = len(result_df)
            result_df = result_df[~(result_df[value_columns] == 0).all(axis=1)]
            rows_after = len(result_df)
            self.log_message(f"  Step 2: Removed {rows_before - rows_after} rows with all zero values ({rows_after} rows remaining)")
            
            # Add Year column
            result_df['Year'] = year
            
            # Save to CSV
            output_filename = f"US_OG_DGGS_methane_emissions_{year}.csv"
            output_path = os.path.join(self.output_folder, output_filename)
            
            result_df.to_csv(output_path, index=False)
            self.log_message(f"  Results saved to: {output_path}")
            self.log_message(f"  Output shape: {result_df.shape}")
            if file_ipcc_mapping:
                self.log_message("\nFile-IPCC Code Mapping:")
                for ipcc_code, fname in file_ipcc_mapping.items():
                    self.log_message(f"  {ipcc_code} -> {fname}")
            
            return output_path
            
        except Exception as e:
            self.log_message(f"Error processing file {netcdf_filename}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_netcdf_files(self):
        """
        Process all NetCDF files and save them as separate CSV outputs.
        """
        start_time = time.time()
        self.log_message(f"Processing all {len(self.netcdf_files)} NetCDF files...")
        
        output_files = []
        
        # Process each NetCDF file separately
        for file_index, netcdf_filename in enumerate(self.netcdf_files):
            self.log_message(f"\nProcessing file {file_index + 1}/{len(self.netcdf_files)}")
            
            output_path = self.process_single_netcdf_file(netcdf_filename)
            if output_path:
                output_files.append(output_path)
                self.log_message(f"  File {file_index + 1} completed successfully")
            else:
                self.log_message(f"  File {file_index + 1} failed to process")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Show summary
        self.log_message(f"\n{'='*60}")
        self.log_message(f"PROCESSING SUMMARY")
        self.log_message(f"{'='*60}")
        self.log_message(f"Total NetCDF files processed: {len(self.netcdf_files)}")
        self.log_message(f"Successful outputs: {len(output_files)}")
        self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        if output_files:
            self.log_message(f"\nOutput files created:")
            for output_file in output_files:
                self.log_message(f"  {output_file}")
        
        return output_files


def main():
    """
    Main function to run the US OG NetCDF to DGGS conversion.
    """
    # Configuration - Updated paths for HPC
    netcdf_folder = "/home/mingke.li/GridInventory/2021_US_oil_and_gas_production"
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
    converter = USOGNetCDFToDGGSConverterAggregated(
        netcdf_folder, geojson_path, output_folder
    )
    
    # Process all NetCDF files
    try:
        output_files = converter.process_all_netcdf_files()
        converter.log_message(f"\nAll files conversion completed successfully!")
        converter.log_message(f"Output files: {output_files}")
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
