# Harmonized Global to Regional Gridded Methane Inventories in DGGS

A comprehensive data processing pipeline that standardizes global, national, and regional methane emission inventories into the rHEALPix DGGS. This project addresses the challenge of harmonizing diverse methane emission gridded inventories from multiple sources, temporal periods, spatial resolutions, and reporting units into a unified, spatially consistent framework for climate research and policy analysis.

- **Spatial consistency**: Equal-area grid cells for accurate spatial analysis
- **Unit standardization**: All outputs in Mg a⁻¹ (megagrams per year)
- **IPCC2006 sector code**: Standardized emission categorization
- **Scalable processing**: Optimized for large-scale datasets with HPC support
- **Quality assurance**: Area-weighted distribution preserving total emissions


## Project Structure

The project is organized into several script categories:

### 📁 `scripts/dggs_grid_creation/`
Scripts for creating and preparing DGGS grids for all countries:

1. **`create_global_country_geojson.py`**
   - Creates global country boundaries from pygadm

2. **`simplify_global_countries.py`**
   - Simplifies country geometries to reduce file size

3. **`convert_country_geojson_to_dggs.py`**
   - Converts country boundaries to DGGS grid cells
   - Uses rhealpix grid type at resolution 6
   - Outputs: Individual country DGGS files

4. **`convert_offshore_to_dggs.py`**
   - Converts offshore areas to DGGS grid cells
   - Handles marine emission zones

5. **`convert_single_geojson_to_dggs.py`**
   - Converts individual GeoJSON files to DGGS format
   - Utility for single-country processing

### 📁 `scripts/netcdf_conversion/`
Scripts for converting NetCDF data to DGGS using pre-calculated grids:

6. **`canada_netcdf_to_dggs_converter.py`**
   - Converts Canada NetCDF methane data to DGGS
   - Uses IPCC2006 code aggregation
   - Handles 2018 Canada anthropogenic methane emissions
   - Units: molecules CH₄ cm⁻² s⁻¹ → Mg a⁻¹

7. **`china_SACMS_netcdf_to_dggs_converter.py`**
   - Converts China SACMS NetCDF data to DGGS
   - Handles emission rate units (Mg km⁻² a⁻¹)
   - Processes 2011 China anthropogenic methane emissions
   - Uses 0.25° × 0.25° resolution

8. **`cms_netcdf_to_dggs_converter.py`**
   - Converts CMS NetCDF files to DGGS
   - Handles flux units (molecules CH₄ cm⁻² s⁻¹)
   - Processes Canada and Mexico files separately
   - Uses external area data for calculations

9. **`global_edgar_netcdf_to_dggs_optimize.py`**
   - Converts EDGAR global NetCDF data to DGGS
   - Optimized for large-scale processing
   - Handles 1970-2022 EDGAR v8.0 greenhouse gas CH4 emissions
   - Multi-year processing capability

10. **`global_gfei_netcdf_to_dggs_optimize.py`**
    - Converts GFEI global NetCDF data to DGGS
    - Handles 2016-2020 Global Fuel Exploitation Inventory
    - Multi-year processing capability

11. **`mexico_netcdf_to_dggs_converter.py`**
    - Converts Mexico NetCDF methane data to DGGS
    - Uses area data from Canada files
    - Handles 2015 Mexico anthropogenic methane emissions

12. **`nys_netcdf_to_dggs_converter.py`**
    - Converts New York State (GNYS) NetCDF to DGGS
    - Handles flux units (kg m⁻² s⁻¹)
    - Uses EPSG:26918 projection (UTM Zone 18N)
    - Processes 2020 data with 100m resolution

13. **`swiss_netcdf_to_dggs_converter.py`**
    - Converts Swiss (SGHGI) NetCDF to DGGS
    - Handles units (g m⁻² yr⁻¹)
    - Uses EPSG:21781 projection (CH1903/LV03)
    - Processes 2011 data with 500m resolution

14. **`us_netcdf_to_dggs_converter.py`**
    - Converts US NetCDF methane data to DGGS
    - Handles flux units (molecules CH₄ cm⁻² s⁻¹)
    - Processes 2012-2018 US anthropogenic methane emissions

15. **`us_OG_netcdf_to_dggs_converter.py`**
    - Converts US Oil and Gas NetCDF to DGGS
    - Handles flux units (kg/h)
    - Processes 2021 US oil and gas emissions
    - No area calculation needed (total emissions per pixel)

### 📁 `scripts/geotiff_conversion/`
Scripts for converting GeoTIFF raster data to DGGS:

16. **`china_tiff_to_dggs_converter.py`**
    - Converts China GeoTIFF methane emission rasters to DGGS
    - Handles units (Mg km⁻² a⁻¹)
    - Processes 1990-2020 time series data
    - Uses variable-to-IPCC2006 code mapping

### 📁 `scripts/csv_conversion/`
Scripts for converting CSV point data to DGGS:

17. **`ind_aus_csv_to_dggs_converter.py`**
    - Converts India/Australia coal methane emissions CSVs to DGGS
    - Handles point data (lat, lon, value in ton/year)
    - Processes 2018 coal mining emissions
    - Uses IPCC2006 code 1B1a (coal mining)

### 📁 `scripts/utilities/`
Utility scripts for data processing, combining, and cleanup:

18. **`combine_geojson_folder.py`**
    - Combines individual country DGGS files into a single file
    - Generates: `data/geojson/global_countries_dggs_merge.geojson`

19. **`merge_country_offshore_dggs_geometries.py`**
    - Merges country and offshore DGGS grids
    - Handles duplicate zoneID removal

### 📁 `scripts/analysis/`
Analysis scripts for post-processing and summarizing DGGS-converted datasets:

20. **`create_dggs_coverage_geojsons.py`**
    - Creates dissolved and simplified DGGS coverage geometries for each inventory CSV
    - Extracts unique DGGS cell IDs from CSV files and filters corresponding geometries
    - Supports parallel processing for large datasets with automatic CPU detection

21. **`compute_sectoral_breakdown.py`**
    - Computes sectoral methane emission totals for selected national datasets
    - Aggregates emissions by four broad IPCC sectors (Energy, Industrial Processes, Agriculture/Forestry, Waste)
    - Processes US, Canada, Mexico, China, and Switzerland datasets
    - Generates temporal breakdowns for China (1990-2020) and US (2012-2018)

22. **`generate_dggs_dataset_summary.py`**
    - Generates descriptive summary table for DGGS-converted methane inventory CSVs
    - Computes dataset metadata: number of DGGS cells, resolution, year range, IPCC categories
    - Uses configuration mappings for resolution and year range information
    - Optimized for large files with chunked reading

### 📁 `SLURM_job_scripts/`
HPC job scripts for running conversions on cluster systems:

- **`run_*_conversion.sh`**: Individual conversion job scripts
- **`combine_*.sh`**: Data combination job scripts
- **`create_global_dggs_geom.sh`**: DGGS grid creation job script

## Workflow

### Phase 1: DGGS Grid Creation (Pre-calculated Grids)
1. Create global country boundaries (`create_global_country_geojson.py`)
2. Simplify country geometries (`simplify_global_countries.py`)
3. Convert to DGGS grid cells (`convert_country_geojson_to_dggs.py`)
4. Convert offshore areas to DGGS (`convert_offshore_to_dggs.py`)
5. Merge offshore grids and country grids (`merge_country_offshore_dggs_geometries.py`)
6. Combine all grids to one single GeoJSON (`combine_geojson_folder.py`)
7. Create local grids as needed (`convert_single_geojson_to_dggs.py`)

### Phase 2: Data Conversion to DGGS
All conversion processes output standardized CSV files with DGGS cell values in **Mg a⁻¹** units:

#### NetCDF Conversion
**Input Units**: Various (molecules CH₄ cm⁻² s⁻¹, Mg km⁻² a⁻¹, kg m⁻² s⁻¹, kg/h, g m⁻² yr⁻¹)
**Output Units**: Mg a⁻¹ (Megagrams per year)

1. Load NetCDF data and extract variables
2. Apply IPCC2006 code aggregation using lookup tables
3. Convert NetCDF data to raster format
4. Calculate pixel areas from coordinate reference system
5. Convert input units to Mg a⁻¹ using appropriate formulas:
   - **Flux units** (molecules CH₄ cm⁻² s⁻¹): `mass_Mg = (flux × area × seconds_per_year / AVOGADRO) × M_CH4 × (1e-6)`
   - **Emission rate units** (Mg km⁻² a⁻¹): `mass_Mg = emission_rate × area_km2`
   - **Mass flux units** (kg m⁻² s⁻¹): `mass_Mg = flux × pixel_area_m2 × seconds_per_year / 1000`
   - **Mass rate units** (kg/h): `mass_Mg = flux_kg_h × hours_per_year × (1e-3)`
   - **Mass per area per year** (g m⁻² yr⁻¹): `mass_Mg = (value × pixel_area_m2) / 1e6`
6. Apply area-weighted distribution to DGGS cells
7. Apply scaling to preserve total emissions
8. Output CSV files with DGGS cell values

#### GeoTIFF Conversion
**Input Units**: Mg km⁻² a⁻¹ (megagrams per square kilometer per year)
**Output Units**: Mg a⁻¹ (Megagrams per year)

1. Load GeoTIFF raster data
2. Map variable names to IPCC2006 codes using lookup tables
3. Rasterize DGGS cells to zone index raster aligned with GeoTIFF grid
4. Calculate pixel areas in km² from raster CRS and transform
5. Convert pixel values to total emissions: `mass_Mg = pixel_value × pixel_area_km2`
6. Aggregate to DGGS cells using numpy.bincount
7. Apply scaling to preserve total emissions
8. Output CSV files with DGGS cell values

#### CSV Point Data Conversion
**Input Units**: ton/year (tons per year)
**Output Units**: Mg a⁻¹ (Megagrams per year)

1. Load CSV point data (lat, lon, value)
2. Create regular grid raster from point data in EPSG:4326
3. Rasterize DGGS polygons to label raster aligned with the grid
4. Aggregate per-pixel values to DGGS cells via numpy.bincount
5. Convert units: `mass_Mg = value_ton × 1.0` (1 ton = 1 Mg)
6. Apply scaling to preserve total emissions
7. Output CSV files with DGGS cell values


## Data Sources

The project processes various gridded methane emission inventories from multiple sources. Below is a comprehensive table of all datasets used:

| Dataset Name | Spatial Coverage | CRS | Resolution | Temporal Coverage | Files | CRT/IPCC | Sector | Unit | Area Unit | Note | URL |
|--------------|-----------------|-----|------------|-------------------|-------|----------|--------|------|-----------|------|-----|
| EDGAR v8.0 Greenhouse Gas CH4 Emissions | Global | EPSG 4326 | 0.1° × 0.1° | 1970-2022 | 1272 | IPCC 2006 code | Agriculture, chemical, fuel, energy, natural gas, petroleum, waste | ton/year | - | - | [EDGAR](https://data.jrc.ec.europa.eu/dataset/b54d8149-2864-4fb9-96b9-5fd3a020c224) |
| U.S. Anthropogenic Methane Emissions | U.S. | EPSG 4326 | 0.1° × 0.1° | 2012-2018 | 7 | IPCC 2006 code | Coal mines, oil and gas, residential combustion, solid waste, wastewater | molecules CH₄ cm⁻² s⁻¹ | cm² | - | [US EPA](https://zenodo.org/records/8367082) |
| Mexico Anthropogenic Methane Emissions | Mexico | EPSG 4326 | 0.1° × 0.1° | 2015 | 1 | IPCC 2006 code | Coal mines, oil and gas, residential combustion, solid waste, wastewater | Mg a⁻¹ km⁻² | - | - | [Mexico Inventory](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5FUTWM) |
| Global Fuel Exploitation Inventory GFEI | Global | EPSG 4326 | 0.1° × 0.1° | 2016-2020 | 21(v1) 20(v2) 20(v3) | CRT | Coal mines, oil and gas | Mg a⁻¹ km⁻² | - | Multiple versions available | [GFEI](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FHH4EUM&version=&q=&fileTypeGroupFacet=&fileTag=&fileSortField=&fileSortOrder=&tagPresort=true&folderPresort=true) |
| Canada Anthropogenic Methane Emissions | Canada | EPSG 4326 | 0.1° × 0.1° | 2018 | 1 | IPCC 2006 code | Coal mines, oil and gas, residential combustion, solid waste, wastewater | Mg a⁻¹ km⁻² | - | - | [Canada Inventory](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CC3KLO) |
| Gridded New York State methane emissions inventory (GNYS) | New York State | UTM Zone 18N projection EPSG:26918 | 100m × 100m | 2020 | 1 | IPCC 2006 code | Coal mines, oil and gas, residential combustion, solid waste, wastewater | kg m⁻² s⁻¹ | m² | - | [NYS Inventory](https://zenodo.org/records/16761163) |
| US oil and gas methane emissions | U.S. | EPSG 4326 | 0.1° × 0.1° | 2021 | 1 | IPCC 2006 code | Oil and gas | kg/h | - | - | [US OG](https://zenodo.org/records/10909191) |
| Carbon Monitoring System (CMS) data sets on Methane (CH₄) Flux | Canada and Mexico | EPSG 4326 | 0.1° × 0.1° | 2010 - Mexico 2013 - Canada | 2 | IPCC 2006 code | Oil and gas | molecules CH₄ cm⁻² s⁻¹ | cm² | - | [CMS Mexico](https://disc.gsfc.nasa.gov/datasets/CMS_CH4_FLX_MX_1/summary?keywords=methane%20emissions%20from%20Canadian%20and%20Mexican%20oil%20and%20gas%20systems), [CMS Canada](https://disc.gsfc.nasa.gov/datasets/CMS_CH4_FLX_CA_1/summary?keywords=methane%20emissions%20from%20Canadian%20and%20Mexican%20oil%20and%20gas%20systems) |
| Swiss Greenhouse Gas Inventory (SGHGI) | Switzerland | CH1903/LV03 EPSG:21781 | 500 m × 500 m | 2011 | 1 | IPCC 2006 code | Coal mines, oil and gas, residential combustion, solid waste, wastewater | g m⁻² yr⁻¹ | m² | - | [SGHGI](https://doi.pangaea.de/10.1594/PANGAEA.828262) |
| China coal mine methane emissions | China | EPSG 4326 | 0.25° × 0.25° | 2011 | 1 | IPCC 2006 code | Coal mines | Mg km⁻² a⁻¹ | - | - | [China SACMS](https://forms.gle/NGMXUTfMumMFkMZPA) |
| India and Australia coal mine methane emissions | India and Australia | EPSG 4326 | 0.1° × 0.1° | 2018 | 2 (csv) | IPCC 2006 code | Coal mines | ton/year | - | Provided as CSV | [India/Australia](https://zenodo.org/records/6222441) |
| CHN-CH₄ Anthropogenic Methane Emission Inventory of China | China | Krasovsky 1940 Albers projection | 0.1° × 0.1° | 1990-2020 | 8×31 (tiff) | IPCC 2006 code | Coal mines, oil and gas, residential combustion, solid waste, wastewater | Mg km⁻² a⁻¹ | - | Provided as TIFF | [China CHN-CH₄](https://zenodo.org/records/15107383) |


### Dataset Characteristics
- **Spatial Coverage**: Ranges from country-specific (Switzerland, New York State) to global coverage
- **Resolution**: Varies from high-resolution (100m × 100m) to coarser resolution (0.25° × 0.25°)
- **Temporal Coverage**: Spans from 1970 to 2024, with most datasets covering recent years
- **Units**: Diverse units including flux (molecules CH₄ cm⁻² s⁻¹), mass per area per time (Mg km⁻² a⁻¹), and total emissions (ton/year)
- **File Formats**: Primarily NetCDF files, with some CSV and GeoTIFF formats
- **Source Categories**: Mixed category systems including IPCC 2006 codes, CRT codes, and some datasets without standardized codes

### Input Data
- **NetCDF files**: Various methane emission datasets (EDGAR, GFEI, country-specific, local inventories, etc.)
- **GeoTIFF files**: Raster methane emission data (China time series)
- **CSV files**: Point-based emission data (India/Australia coal mining)
- **Lookup tables**: IPCC2006 code mappings in `data/lookup/`
- **Area data**: Grid cell area information in `data/area_npy/`
- **GeoJSON files**: Pre-calculated rHEALPix DGGS grid geometries

### Output Data
- **CSV files**: DGGS cell values with emission data

### Data Formats Supported
- **NetCDF**: Multi-dimensional scientific data format
- **GeoTIFF**: Georeferenced raster images
- **CSV**: Point data with latitude, longitude, and values

## Key Features
- **Pre-calculated grids**: Efficient processing using pre-computed DGGS grids
- **Area-weighted distribution**: Accurate spatial allocation of emission values
- **IPCC2006 aggregation**: Standardized emission categorization
- **Multi-source support**: Handles various data formats and units (NetCDF, GeoTIFF, CSV)
- **Parallel processing**: Optimized for large-scale data processing with multiprocessing
- **Resume capability**: Can restart from intermediate results
- **Unit conversion**: Automatic conversion between different emission units and final output unit as Mg a⁻¹
- **HPC support**: SLURM job scripts for cluster computing
- **Comprehensive logging**: Detailed processing logs for debugging and monitoring


## Usage

### 1. Create DGGS Grids (Run Once)
```bash
# Create global country boundaries and simplify the geometries
python scripts/dggs_grid_creation/create_global_country_geojson.py
python scripts/dggs_grid_creation/simplify_global_countries.py

# Convert to DGGS format
python scripts/dggs_grid_creation/convert_country_geojson_to_dggs.py

# Convert offshore areas
python scripts/dggs_grid_creation/convert_offshore_to_dggs.py

# Merge offshore grids and country grids
python scripts/utilities/merge_country_offshore_dggs_geometries.py

# Combine all grids to one single geojson
python scripts/utilities/combine_geojson_folder.py

# Create local grids
python scripts/dggs_grid_creation/convert_single_geojson_to_dggs
```

### 2. Convert Data to DGGS (Run as Needed)

#### NetCDF Conversions
```bash
# Global datasets
python scripts/netcdf_conversion/global_edgar_netcdf_to_dggs_optimize.py
python scripts/netcdf_conversion/global_gfei_netcdf_to_dggs_optimize.py

# Country-specific datasets
python scripts/netcdf_conversion/canada_netcdf_to_dggs_converter.py
python scripts/netcdf_conversion/us_netcdf_to_dggs_converter.py
python scripts/netcdf_conversion/mexico_netcdf_to_dggs_converter.py
python scripts/netcdf_conversion/swiss_netcdf_to_dggs_converter.py
python scripts/netcdf_conversion/china_SACMS_netcdf_to_dggs_converter.py

# Specialized datasets
python scripts/netcdf_conversion/cms_netcdf_to_dggs_converter.py
python scripts/netcdf_conversion/us_OG_netcdf_to_dggs_converter.py
python scripts/netcdf_conversion/nys_netcdf_to_dggs_converter.py
```

#### GeoTIFF Conversions
```bash
# China time series data
python scripts/geotiff_conversion/china_tiff_to_dggs_converter.py
```

#### CSV Conversions
```bash
# India/Australia coal mining data
python scripts/csv_conversion/ind_aus_csv_to_dggs_converter.py
```

### 3. HPC Cluster Usage
```bash
# Submit individual conversion jobs
sbatch SLURM_job_scripts/run_canada_netcdf_conversion.sh
sbatch SLURM_job_scripts/run_edgar_netcdf_conversion.sh
sbatch SLURM_job_scripts/run_china_tiff_conversion.sh
```
