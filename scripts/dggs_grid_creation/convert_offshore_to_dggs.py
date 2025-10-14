"""
Convert offshore facility GeoJSON file to DGGS geometry.
This script processes offshore facility polygons for each country and converts them to DGGS grid cells.
No filtering by exact geometry is needed - just bounding box based conversion.
"""

import os
import json
import subprocess
import geopandas as gpd
import pandas as pd
from typing import List, Optional
import multiprocessing as mp
from functools import partial

# Configuration variables
INPUT_FILE = "data/geojson/global_offshore_cleaned.geojson"
OUTPUT_FOLDER = "data/geojson/global_offshore_grid"
GRID_TYPE = "rhealpix"
RESOLUTION = 6
FEATURE_LIMIT = None  # Set to a number to limit features for testing
START_FROM_FEATURE = 0  # Start processing from this feature index (0-based)
NUM_CORES = int(os.environ.get('NUM_CORES', 4))  # Number of CPU cores to use for parallel processing


class OffshoreToDGGSConverter:
    """Convert offshore facility GeoJSON features to DGGS geometry."""
    
    def __init__(self, dggs_grid_type: str = "rhealpix", resolution: int = 6):
        self.dggs_grid_type = dggs_grid_type
        self.resolution = resolution
        
    def generate_dggs_grid(self, grid: str, level: int, bbox: Optional[str] = None, 
                           compact: bool = False) -> Optional[dict]:
        """Generate grid geometry for a given bounding box."""
        cmd = ["dgg", grid, "grid", str(level)]
        if compact:
            cmd.append("-compact")
        if bbox:
            cmd.extend(["-bbox", bbox])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"DGGS command failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return None
                
            output = result.stdout.strip()
            if not output:
                print("DGGS command returned empty output")
                return None
                
            geojson_data = json.loads(output)
            return geojson_data
        except Exception as e:
            print(f"Error creating DGGS grid: {e}")
            return None

    def create_dggs_geodataframe(self, grid: str, level: int, bbox: Optional[str] = None, 
                                 compact: bool = False) -> Optional[gpd.GeoDataFrame]:
        """Create a GeoDataFrame from DGGS grid zones."""
        geojson_data = self.generate_dggs_grid(grid, level, bbox=bbox, compact=compact)
        if geojson_data is None:
            return None
            
        try:
            features = geojson_data["features"]
            gdf = gpd.GeoDataFrame.from_features(features)
            gdf = gdf.set_crs("EPSG:4326")
            
            return gdf
        except Exception as e:
            print(f"Error creating GeoDataFrame: {e}")
            return None

    def convert_offshore_facility_to_dggs_grid(self, facility_geom, country_name: str, country_gid: str) -> List[dict]:
        """Convert an offshore facility geometry to DGGS grid features with country info."""
        if facility_geom is None:
            return []
            
        try:
            # Handle different geometry types
            if facility_geom.geom_type == 'MultiPolygon':
                print(f"Processing MultiPolygon with {len(facility_geom.geoms)} polygons for {country_name}")
                all_dggs_features = []
                
                for poly_idx, polygon in enumerate(facility_geom.geoms):
                    polygon_features = self._process_single_polygon(polygon, country_name, country_gid)
                    all_dggs_features.extend(polygon_features)
                
                print(f"Generated {len(all_dggs_features)} total DGGS grid cells for '{country_name}' (GID: {country_gid}) at resolution {self.resolution}")
                return all_dggs_features
                
            elif facility_geom.geom_type == 'Polygon':
                print(f"Processing single Polygon for {country_name}")
                return self._process_single_polygon(facility_geom, country_name, country_gid)
                
            else:
                print(f"Unsupported geometry type: {facility_geom.geom_type} for {country_name}")
                return []
            
        except Exception as e:
            print(f"Error converting offshore facility '{country_name}' to DGGS grid: {e}")
            return []

    def _process_single_polygon(self, polygon, country_name: str, country_gid: str) -> List[dict]:
        """Process a single polygon to generate DGGS grid cells."""
        try:
            extent = polygon.bounds
            bbox_str = f"{extent[1]},{extent[0]},{extent[3]},{extent[2]}"
            
            gdf = self.create_dggs_geodataframe(self.dggs_grid_type, self.resolution, bbox=bbox_str, compact=False)
            if gdf is None:
                return []
            
            # No filtering needed - use all grid cells within bounding box
            # Convert all DGGS cells to features with country information
            dggs_features = []
            for idx, row in gdf.iterrows():
                # Get the zoneID from the DGGS grid
                zone_id = row.get('zoneID', f'cell_{idx}')
                
                # Create feature with country info and zoneID
                feature = {
                    "type": "Feature",
                    "properties": {
                        "NAME": country_name,
                        "GID": country_gid,
                        "zoneID": zone_id
                    },
                    "geometry": row.geometry.__geo_interface__
                }
                dggs_features.append(feature)
            
            return dggs_features
            
        except Exception as e:
            print(f"Error processing polygon for '{country_name}': {e}")
            return []

    def process_single_offshore_facility(self, facility_data: tuple) -> tuple:
        """Process a single offshore facility for parallel execution."""
        idx, row, output_folder = facility_data
        country_name = row.get('NAME', f'Country_{idx}')
        country_gid = row.get('GID', f'GID_{idx}')
        
        print(f"Processing offshore facility {idx + 1}: {country_name} (GID: {country_gid})")
        
        # Convert offshore facility to DGGS grid
        dggs_features = self.convert_offshore_facility_to_dggs_grid(row.geometry, country_name, country_gid)
        
        if dggs_features:
            # Create output filename for this country
            safe_country_name = country_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            output_filename = f"{safe_country_name}_{country_gid}_offshore_grid.geojson"
            output_filepath = os.path.join(output_folder, output_filename)
            
            # Check if file already exists
            if os.path.exists(output_filepath):
                print(f"File already exists for {country_name}, skipping...")
                return (idx, country_name, 0, True)
            
            # Save individual country grid
            country_geojson = {
                "type": "FeatureCollection",
                "features": dggs_features,
                "properties": {
                    "country_name": country_name,
                    "country_gid": country_gid,
                    "grid_type": self.dggs_grid_type,
                    "resolution": self.resolution,
                    "total_grid_cells": len(dggs_features),
                    "conversion_timestamp": str(pd.Timestamp.now())
                }
            }
            
            with open(output_filepath, 'w') as f:
                json.dump(country_geojson, f, indent=2)
            
            print(f"Saved {len(dggs_features)} grid cells for {country_name} to {output_filename}")
            return (idx, country_name, len(dggs_features), True)
        else:
            print(f"Warning: No DGGS grid cells generated for {country_name}")
            return (idx, country_name, 0, False)

    def convert_offshore_geojson_file(self, input_file: str, output_folder: str, 
                                     feature_limit: Optional[int] = None,
                                     num_cores: int = None) -> bool:
        """Convert offshore facility GeoJSON file to DGGS geometry and save each country separately."""
        try:
            print(f"Loading offshore facility GeoJSON file: {input_file}")
            gdf = gpd.read_file(input_file)
            
            if gdf.empty:
                print("Input GeoJSON file is empty")
                return False
                
            print(f"Loaded {len(gdf)} features from {input_file}")
            print(f"CRS: {gdf.crs}")
            print(f"Columns: {list(gdf.columns)}")
            
            if feature_limit and len(gdf) > feature_limit:
                print(f"Limiting to first {feature_limit} features for testing")
                gdf = gdf.head(feature_limit)
            
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)
            
            # Start processing from the specified feature index
            start_idx = START_FROM_FEATURE
            if start_idx >= len(gdf):
                print(f"Start index {start_idx} is beyond the number of features ({len(gdf)})")
                return False
            
            # Prepare data for parallel processing
            facilities_to_process = []
            for idx in range(start_idx, len(gdf)):
                row = gdf.iloc[idx]
                facilities_to_process.append((idx, row, output_folder))
            
            print(f"Starting processing from feature index {start_idx}")
            print(f"Total offshore facilities to process: {len(facilities_to_process)}")
            
            # Set number of cores
            if num_cores is None:
                num_cores = NUM_CORES
            
            print(f"Using {num_cores} CPU cores for parallel processing")
            
            # Process facilities in parallel
            if num_cores > 1 and len(facilities_to_process) > 1:
                with mp.Pool(processes=num_cores) as pool:
                    results = pool.map(self.process_single_offshore_facility, facilities_to_process)
            else:
                # Sequential processing if only 1 core or 1 facility
                results = [self.process_single_offshore_facility(facility_data) for facility_data in facilities_to_process]
            
            # Collect results
            total_cells = 0
            processed_facilities = 0
            successful_facilities = 0
            
            for idx, country_name, cell_count, success in results:
                total_cells += cell_count
                if success:
                    successful_facilities += 1
                processed_facilities += 1
            
            print(f"\nConversion completed successfully!")
            print(f"Output folder: {output_folder}")
            print(f"Total facilities processed: {processed_facilities}")
            print(f"Successful conversions: {successful_facilities}")
            print(f"Total DGGS grid cells generated: {total_cells}")
            
            return True
            
        except Exception as e:
            print(f"Error converting offshore GeoJSON file: {e}")
            return False


def main():
    """Main function to run the conversion."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return
    
    converter = OffshoreToDGGSConverter(dggs_grid_type=GRID_TYPE, resolution=RESOLUTION)
    
    success = converter.convert_offshore_geojson_file(
        input_file=INPUT_FILE,
        output_folder=OUTPUT_FOLDER,
        feature_limit=FEATURE_LIMIT,
        num_cores=NUM_CORES
    )
    
    if success:
        print("\nOffshore facility conversion completed successfully!")
    else:
        print("\nOffshore facility conversion failed!")
        exit(1)


if __name__ == "__main__":
    main()
