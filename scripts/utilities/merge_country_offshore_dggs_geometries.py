"""
Script to merge offshore and onshore geospatial data by combining features from 
offshore geojson files into corresponding country geojson files.

This script:
1. Reads offshore geojson files from global_offshore_grid/
2. Reads country geojson files from global_countries_grid/
3. Matches files by country GID (country abbreviation) using consistent naming pattern
4. Merges features from offshore files into country files
5. Prevents duplicate features
6. Maintains data consistency and naming conventions
7. Outputs merged files to global_countries_dggs_merge/
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeojsonMerger:
    """Class to handle merging of offshore and onshore geojson files."""
    
    def __init__(self, offshore_dir: str, countries_dir: str, output_dir: str):
        """
        Initialize the merger with directory paths.
        
        Args:
            offshore_dir: Path to offshore geojson files
            countries_dir: Path to country geojson files
            output_dir: Path to output merged files
        """
        self.offshore_dir = Path(offshore_dir)
        self.countries_dir = Path(countries_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for country files to avoid repeated searches
        self.country_files_cache = {}
        
    def extract_gid_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract country GID from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Country GID if found, None otherwise
        """
        # Remove file extension
        name_without_ext = filename.replace('.geojson', '')
        
        # Pattern: CountryName_GID_grid.geojson or CountryName_GID_offshore_grid.geojson
        # GID is always before '_grid' or '_offshore_grid'
        # Also handles disputed territories: TerritoryName_Z##_grid.geojson
        
        # Check for offshore pattern first: CountryName_GID_offshore_grid
        if name_without_ext.endswith('_offshore_grid'):
            # Extract everything before '_offshore_grid'
            prefix = name_without_ext.replace('_offshore_grid', '')
            # GID should be the last part after splitting by '_'
            parts = prefix.split('_')
            if len(parts) >= 2:
                gid = parts[-1]  # Last part should be the GID
                # Check for standard 3-letter country codes
                if len(gid) == 3 and gid.isupper() and gid.isalpha():
                    return gid
                # Check for zone codes like Z01, Z02, etc.
                elif gid.startswith('Z') and len(gid) == 3 and gid[1:].isdigit():
                    return gid
        
        # Check for regular grid pattern: CountryName_GID_grid
        elif name_without_ext.endswith('_grid'):
            # Extract everything before '_grid'
            prefix = name_without_ext.replace('_grid', '')
            # GID should be the last part after splitting by '_'
            parts = prefix.split('_')
            if len(parts) >= 2:
                gid = parts[-1]  # Last part should be the GID
                # Check for standard 3-letter country codes
                if len(gid) == 3 and gid.isupper() and gid.isalpha():
                    return gid
                # Check for zone codes like Z01, Z02, etc.
                elif gid.startswith('Z') and len(gid) == 3 and gid[1:].isdigit():
                    return gid
        
        # Fallback: try to find any 3-character code in the filename
        # This handles cases where the pattern might be slightly different
        import re
        # Look for 3-letter uppercase codes (standard countries)
        gid_pattern = r'[A-Z]{3}'
        matches = re.findall(gid_pattern, name_without_ext)
        
        if matches:
            # Look for a GID that appears to be in the right position
            for match in matches:
                # Check if it's followed by 'grid' or 'offshore'
                if f'_{match}_grid' in name_without_ext or f'_{match}_offshore' in name_without_ext:
                    return match
                # Check if it's at the end before 'grid'
                if name_without_ext.endswith(f'_{match}_grid'):
                    return match
                # Check if it's at the end before 'offshore_grid'
                if name_without_ext.endswith(f'_{match}_offshore_grid'):
                    return match
            
            # If no pattern match, return the last 3-letter code found
            return matches[-1]
        
        # Additional fallback: look for zone codes (Z##)
        zone_pattern = r'Z\d{2}'
        zone_matches = re.findall(zone_pattern, name_without_ext)
        if zone_matches:
            # Look for zone code in the right position
            for match in zone_matches:
                if f'_{match}_grid' in name_without_ext:
                    return match
                if name_without_ext.endswith(f'_{match}_grid'):
                    return match
            # Return the last zone code found
            return zone_matches[-1]
        
        return None
    
    def build_country_files_index(self) -> Dict[str, str]:
        """
        Build an index of country files by GID for efficient lookup.
        
        Returns:
            Dictionary mapping GID to country filename
        """
        logger.info("Building country files index...")
        index = {}
        skipped_files = []
        
        all_files = list(self.countries_dir.glob('*.geojson'))
        logger.info(f"Found {len(all_files)} total country files")
        
        for file_path in all_files:
            gid = self.extract_gid_from_filename(file_path.name)
            if gid:
                index[gid] = file_path.name
                logger.debug(f"Mapped GID {gid} to file {file_path.name}")
            else:
                skipped_files.append(file_path.name)
        
        logger.info(f"Indexed {len(index)} country files")
        if skipped_files:
            logger.warning(f"Skipped {len(skipped_files)} files (could not extract GID):")
            for filename in skipped_files[:10]:  # Show first 10 skipped files
                logger.warning(f"  {filename}")
            if len(skipped_files) > 10:
                logger.warning(f"  ... and {len(skipped_files) - 10} more")
        
        return index
    
    def load_geojson_file(self, file_path: Path) -> Dict:
        """
        Load and parse a geojson file.
        
        Args:
            file_path: Path to the geojson file
            
        Returns:
            Parsed geojson data as dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def save_geojson_file(self, data: Dict, file_path: Path) -> bool:
        """
        Save geojson data to file.
        
        Args:
            data: Geojson data to save
            file_path: Path where to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
            return False
    
    def merge_features(self, country_data: Dict, offshore_data: Dict) -> Dict:
        """
        Merge features from offshore data into country data.
        
        Args:
            country_data: Country geojson data
            offshore_data: Offshore geojson data
            
        Returns:
            Merged geojson data
        """
        if not offshore_data or 'features' not in offshore_data:
            return country_data
        
        # Get existing features
        merged_features = country_data.get('features', [])
        
        # Add offshore features, avoiding duplicates
        offshore_features = offshore_data.get('features', [])
        
        # Create a set of existing zoneIDs to avoid duplicates
        existing_zone_ids = set()
        for feature in merged_features:
            zone_id = feature.get('properties', {}).get('zoneID')
            if zone_id:
                existing_zone_ids.add(zone_id)
        
        # Add offshore features that don't have duplicate zoneIDs
        added_count = 0
        duplicate_count = 0
        no_zoneid_count = 0
        
        for feature in offshore_features:
            zone_id = feature.get('properties', {}).get('zoneID')
            if not zone_id:
                no_zoneid_count += 1
            elif zone_id in existing_zone_ids:
                duplicate_count += 1
            else:
                merged_features.append(feature)
                existing_zone_ids.add(zone_id)
                added_count += 1
        
        logger.info(f"Offshore features: {len(offshore_features)} total, {added_count} added, {duplicate_count} duplicates, {no_zoneid_count} no zoneID")
        
        # Final safety: deduplicate by zoneID across all merged features (keep first occurrence)
        seen_zone_ids = set()
        deduped_features = []
        removed_dups = 0
        for feat in merged_features:
            zid = feat.get('properties', {}).get('zoneID')
            if zid:
                if zid in seen_zone_ids:
                    removed_dups += 1
                    continue
                seen_zone_ids.add(zid)
            deduped_features.append(feat)
        if removed_dups > 0:
            logger.info(f"Removed {removed_dups} duplicate features by zoneID after merge")
        
        # Update the merged data
        merged_data = country_data.copy()
        merged_data['features'] = deduped_features
        
        # Update total grid cells count if it exists
        if 'properties' in merged_data and 'total_grid_cells' in merged_data['properties']:
            merged_data['properties']['total_grid_cells'] = len(deduped_features)
        
        return merged_data
    
    def process_offshore_file(self, offshore_file: Path, country_files_index: Dict[str, str]) -> bool:
        """
        Process a single offshore file and merge it with corresponding country file.
        
        Args:
            offshore_file: Path to offshore geojson file
            country_files_index: Index of country files by GID
            
        Returns:
            True if successful, False otherwise
        """
        # Extract GID from offshore filename
        gid = self.extract_gid_from_filename(offshore_file.name)
        if not gid:
            logger.warning(f"Could not extract GID from {offshore_file.name}")
            return False
        
        # Find corresponding country file
        country_filename = country_files_index.get(gid)
        if not country_filename:
            logger.warning(f"No country file found for GID {gid} from {offshore_file.name}")
            return False
        
        logger.info(f"Processing {offshore_file.name} -> {country_filename}")
        
        # Load both files
        offshore_data = self.load_geojson_file(offshore_file)
        country_file_path = self.countries_dir / country_filename
        country_data = self.load_geojson_file(country_file_path)
        
        if not offshore_data or not country_data:
            logger.error(f"Failed to load data for {offshore_file.name} or {country_filename}")
            return False
        
        # Merge the data
        merged_data = self.merge_features(country_data, offshore_data)
        
        # Save merged file
        output_file = self.output_dir / country_filename
        if self.save_geojson_file(merged_data, output_file):
            logger.info(f"Successfully merged and saved {output_file}")
            return True
        else:
            logger.error(f"Failed to save merged file {output_file}")
            return False
    
    def run_merge(self) -> None:
        """
        Run the complete merge process for all offshore files.
        """
        logger.info("Starting geojson merge process...")
        
        # Build country files index
        country_files_index = self.build_country_files_index()
        
        if not country_files_index:
            logger.error("No country files found. Cannot proceed.")
            return
        
        # Process each offshore file
        offshore_files = list(self.offshore_dir.glob('*.geojson'))
        logger.info(f"Found {len(offshore_files)} offshore files to process")
        
        successful_merges = 0
        failed_merges = 0
        
        for offshore_file in offshore_files:
            if self.process_offshore_file(offshore_file, country_files_index):
                successful_merges += 1
            else:
                failed_merges += 1
        
        logger.info(f"Merge process completed. Successful: {successful_merges}, Failed: {failed_merges}")
        
        # Copy remaining country files that don't have offshore data
        self.copy_remaining_country_files(country_files_index)

        # Ensure no duplicate zoneID across ALL output geojson files
        self.deduplicate_all_outputs()
    
    def copy_remaining_country_files(self, country_files_index: Dict[str, str]) -> None:
        """
        Copy country files that don't have corresponding offshore data.
        
        Args:
            country_files_index: Index of country files by GID
        """
        logger.info("Copying remaining country files without offshore data...")
        
        # Get list of already processed country files
        processed_files = set()
        for output_file in self.output_dir.glob('*.geojson'):
            processed_files.add(output_file.name)
        
        # Copy unprocessed country files
        copied_count = 0
        for gid, filename in country_files_index.items():
            if filename not in processed_files:
                source_file = self.countries_dir / filename
                target_file = self.output_dir / filename
                
                if source_file.exists():
                    try:
                        import shutil
                        shutil.copy2(source_file, target_file)
                        copied_count += 1
                        logger.debug(f"Copied {filename}")
                    except Exception as e:
                        logger.error(f"Failed to copy {filename}: {e}")
        
        logger.info(f"Copied {copied_count} remaining country files")

    def _deduplicate_output_file(self, file_path: Path) -> int:
        """
        Ensure the output GeoJSON file contains unique zoneID features.
        Returns number of removed duplicates.
        """
        data = self.load_geojson_file(file_path)
        if not data or 'features' not in data:
            return 0
        features = data.get('features', [])
        seen = set()
        dedup = []
        removed = 0
        for feat in features:
            zid = feat.get('properties', {}).get('zoneID')
            if zid and zid in seen:
                removed += 1
                continue
            if zid:
                seen.add(zid)
            dedup.append(feat)
        if removed > 0:
            data['features'] = dedup
            # Update total grid cells count if present
            if 'properties' in data and 'total_grid_cells' in data['properties']:
                data['properties']['total_grid_cells'] = len(dedup)
            self.save_geojson_file(data, file_path)
        return removed

    def deduplicate_all_outputs(self) -> None:
        """Run de-duplication across all output GeoJSON files and log summary."""
        logger.info("Ensuring unique zoneID across all merged output files...")
        total_removed = 0
        files = list(self.output_dir.glob('*.geojson'))
        for fp in files:
            removed = self._deduplicate_output_file(fp)
            if removed > 0:
                logger.info(f"  {fp.name}: removed {removed} duplicate features by zoneID")
            total_removed += removed
        logger.info(f"De-duplication complete. Total duplicates removed across outputs: {total_removed}")


def main():
    """Main function to run the geojson merger."""
    # Set paths - modify these as needed
    offshore_dir = 'data/geojson/global_offshore_grid'
    countries_dir = 'data/geojson/global_countries_grid'
    output_dir = 'data/geojson/global_countries_dggs_merge'
    
    # Enable verbose logging if needed
    # logging.getLogger().setLevel(logging.DEBUG)
    
    # Create merger and run
    merger = GeojsonMerger(offshore_dir, countries_dir, output_dir)
    merger.run_merge()


if __name__ == '__main__':
    main()
