import geopandas as gpd
import json
import os

def simplify_global_countries(input_file="data/geojson/global_countries.geojson", 
                             tolerance=0.01, 
                             output_file="data/geojson/global_countries_simplified.geojson"):
    """
    Read the global countries GeoJSON file and simplify the polygons.
    
    Args:
        input_file (str): Path to the input GeoJSON file
        tolerance (float): Simplification tolerance (higher = more simplified)
        output_file (str): Path for the output simplified GeoJSON file
        
    Returns:
        geopandas.GeoDataFrame: The simplified GeoDataFrame or None if error
    """
    try:
        print(f"Reading GeoJSON file: {input_file}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist")
            return None
        
        # Read the GeoJSON file
        gdf = gpd.read_file(input_file)
        
        print(f"Successfully loaded GeoJSON as GeoDataFrame")
        print(f"Original shape: {gdf.shape}")
        print(f"Columns: {gdf.columns.tolist()}")
        print(f"CRS: {gdf.crs}")
        
        # Show sample data before simplification
        print(f"\nSample data before simplification:")
        print("=" * 60)
        print(gdf[['GID', 'NAME', 'CONTINENT']].head(5).to_string(index=False))
        
        # Analyze original geometries
        print(f"\nAnalyzing original geometries...")
        original_vertices = 0
        for geom in gdf.geometry:
            if geom.geom_type == 'Polygon':
                original_vertices += len(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    original_vertices += len(poly.exterior.coords)
        
        print(f"Total vertices before simplification: {original_vertices}")
        
        # Simplify geometries
        print(f"\nSimplifying geometries with tolerance: {tolerance}")
        simplified_gdf = gdf.copy()
        
        # Apply simplification to each geometry
        simplified_geometries = []
        for idx, geom in enumerate(gdf.geometry):
            try:
                simplified_geom = geom.simplify(tolerance, preserve_topology=True)
                simplified_geometries.append(simplified_geom)
                
                if idx < 5:  # Show progress for first 5 countries
                    print(f"  Simplified {gdf.iloc[idx]['NAME']}: {geom.geom_type} -> {simplified_geom.geom_type}")
                    
            except Exception as e:
                print(f"  Error simplifying geometry {idx}: {str(e)}")
                simplified_geometries.append(geom)  # Keep original if simplification fails
        
        simplified_gdf.geometry = simplified_geometries
        
        # Analyze simplified geometries
        print(f"\nAnalyzing simplified geometries...")
        simplified_vertices = 0
        for geom in simplified_gdf.geometry:
            if geom.geom_type == 'Polygon':
                simplified_vertices += len(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    simplified_vertices += len(poly.exterior.coords)
        
        print(f"Total vertices after simplification: {simplified_vertices}")
        reduction = ((original_vertices - simplified_vertices) / original_vertices) * 100
        print(f"Vertex reduction: {reduction:.1f}%")
        
        # Show sample data after simplification
        print(f"\nSample data after simplification:")
        print("=" * 60)
        print(simplified_gdf[['GID', 'NAME', 'CONTINENT']].head(5).to_string(index=False))
        
        # Save simplified GeoJSON
        print(f"\nSaving simplified GeoJSON to: {output_file}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        simplified_gdf.to_file(output_file, driver='GeoJSON')
        
        print(f"Successfully saved simplified GeoJSON")
        print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return simplified_gdf
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def compare_file_sizes(input_file="data/geojson/global_countries.geojson", 
                       output_file="data/geojson/global_countries_simplified.geojson"):
    """
    Compare file sizes between original and simplified GeoJSON files.
    """
    try:
        if os.path.exists(input_file) and os.path.exists(output_file):
            original_size = os.path.getsize(input_file) / (1024*1024)  # MB
            simplified_size = os.path.getsize(output_file) / (1024*1024)  # MB
            
            print(f"\nFile size comparison:")
            print("=" * 40)
            print(f"Original file: {original_size:.2f} MB")
            print(f"Simplified file: {simplified_size:.2f} MB")
            print(f"Size reduction: {((original_size - simplified_size) / original_size) * 100:.1f}%")
        else:
            print("Cannot compare file sizes - one or both files don't exist")
            
    except Exception as e:
        print(f"Error comparing file sizes: {str(e)}")

if __name__ == "__main__":
    print("Simplifying global countries GeoJSON polygons...")
    print("=" * 60)
    
    # Simplify with different tolerance values
    tolerance_values = [0.01, 0.05, 0.1]
    
    for tolerance in tolerance_values:
        print(f"\n{'='*20} Tolerance: {tolerance} {'='*20}")
        
        output_file = f"data/geojson/global_countries_simplified_t{tolerance}.geojson"
        
        result = simplify_global_countries(
            input_file="data/geojson/global_countries.geojson",
            tolerance=tolerance,
            output_file=output_file
        )
        
        if result is not None:
            print(f"Simplification with tolerance {tolerance} completed successfully!")
        else:
            print(f"Simplification with tolerance {tolerance} failed!")
    
    # Compare file sizes
    print(f"\n{'='*60}")
    print("Final file size comparison:")
    compare_file_sizes()
