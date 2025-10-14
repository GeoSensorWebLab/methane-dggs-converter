"""
Create dissolved and simplified DGGS coverage geometries for each inventory CSV in output/.

For each dataset, this script:
 1) Reads the CSV and extracts the unique set of `dggsID` values (column name is exactly 'dggsID')
 2) Loads the appropriate DGGS geometry source (global, country-specific, or regional parquet). Geometry DGGS id column is exactly 'zoneID'.
 3) Filters geometries to those DGGS cells in the dataset
 4) Dissolves them (union), then simplifies the dissolved geometry (no separate script needed)
 5) Writes a single GeoJSON under analysis_results/coverage_geojson/ and a Shapefile under analysis_results/coverage_shapefiles/

Run:
  python scripts/analysis/create_dggs_coverage_geojsons.py

This script auto-detects CPU count from SLURM env vars when run on HPC.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import geopandas as gpd
from shapely import ops
from shapely import wkb as shapely_wkb


# I/O locations
OUTPUT_CSV_DIR = Path("output").resolve()
GEOMETRY_BASE_DIR = Path("data/geojson").resolve()
OUTPUT_GEOJSON_DIR = Path("analysis_results/coverage_geojson").resolve()
OUTPUT_SHP_DIR = Path("analysis_results/coverage_shapefiles").resolve()

# Simplification configuration
SIMPLIFY_TOLERANCE = 0.01  # degrees; adjust if needed

# Files to process (dataset base names without .csv)
DATASET_FILES = [
    "EDGAR_DGGS_methane_emissions_ALL_SECTORS_1970_2022",
    "GFEI_DGGS_methane_emissions_ALL_FILES",
    "US_DGGS_methane_emissions_ALL_FILES",
    "Canada_DGGS_methane_emissions_ALL_FILES",
    "Mexico_DGGS_methane_emissions_ALL_FILES",
    "China_DGGS_methane_emissions_ALL_FILES",
    "Switzerland_DGGS_methane_emissions_2011",
    "US_OG_DGGS_methane_emissions_2021",
    "CMS_Canada_DGGS_methane_emissions_2013",
    "CMS_Mexico_DGGS_methane_emissions_2010",
    "China_SACMS_DGGS_methane_emissions_2011",
    "India_Coal_DGGS_methane_emissions_2018",
    "Australia_Coal_DGGS_methane_emissions_2018",
    "NYS_DGGS_methane_emissions_2020",
]


def read_unique_dggs_ids(csv_path: Path, chunksize: int = 1_000_000) -> Set[str]:
    # Column is guaranteed to be named exactly 'dggsID'
    dggs_col = "dggsID"
    unique_ids: Set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=[dggs_col], dtype={dggs_col: str}, chunksize=chunksize):
        series = chunk[dggs_col].dropna().astype(str)
        unique_ids.update(series.unique().tolist())
    return unique_ids


def geometry_path_for_dataset(dataset_name: str) -> Tuple[Path, str]:
    """Return (path, kind) where kind in {"global","country","regional_parquet"}."""
    # Global sources
    if dataset_name.startswith("EDGAR_") or dataset_name.startswith("GFEI_"):
        return (GEOMETRY_BASE_DIR / "global_countries_dggs_merge.geojson", "global")

    # NYS special regional parquet
    if dataset_name.startswith("NYS_"):
        return (GEOMETRY_BASE_DIR / "regional_grid" / "newyorkstate_grid_res10.parquet", "regional_parquet")

    # Country-specific mapping (explicit paths). Fallbacks use common <Country>_<ISO>_grid.geojson names
    country_paths: Dict[str, str] = {
        "US": "global_countries_dggs_merge/United_States_USA_grid.geojson",
        "US_OG": "global_countries_dggs_merge/United_States_USA_grid.geojson",
        "Canada": "global_countries_dggs_merge/Canada_CAN_grid.geojson",
        "CMS_Canada": "global_countries_dggs_merge/Canada_CAN_grid.geojson",
        "Mexico": "global_countries_dggs_merge/México_MEX_grid.geojson",
        "CMS_Mexico": "global_countries_dggs_merge/México_MEX_grid.geojson",
        "China": "global_countries_dggs_merge/China_CHN_grid.geojson",
        "China_SACMS": "global_countries_dggs_merge/China_CHN_grid.geojson",
        "Switzerland": "regional_grid/switzerland_grid.geojson",
        "India_Coal": "global_countries_dggs_merge/India_IND_grid.geojson",
        "Australia_Coal": "global_countries_dggs_merge/Australia_AUS_grid.geojson",
    }

    for key, rel_path in country_paths.items():
        if dataset_name.startswith(key):
            return (GEOMETRY_BASE_DIR / rel_path, "country")

    # Fallback to global if no match
    return (GEOMETRY_BASE_DIR / "global_countries_dggs_merge.geojson", "global")


def load_geometry(path: Path, kind: str) -> gpd.GeoDataFrame:
    if kind == "regional_parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        # Assume geographic
        gdf.set_crs(epsg=4326, inplace=True)
    return gdf


def filter_by_dggs_ids(gdf: gpd.GeoDataFrame, dggs_ids: Set[str]) -> gpd.GeoDataFrame:
    # Geometry index column is guaranteed to be named exactly 'zoneID'
    id_col = "zoneID"
    if id_col not in gdf.columns:
        raise ValueError("Expected geometry DGGS id column 'zoneID' not found.")
    return gdf[gdf[id_col].astype(str).isin(dggs_ids)].copy()


def dissolve_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[["geometry"]].copy()
    gdf["_grp"] = 1
    dissolved = gdf.dissolve(by="_grp")
    dissolved = dissolved.drop(columns=["_grp"], errors="ignore")
    dissolved.reset_index(drop=True, inplace=True)
    return dissolved


def _chunk_list(items: List, num_chunks: int) -> List[List]:
    if num_chunks <= 1 or len(items) <= 1:
        return [items]
    n = max(1, min(num_chunks, len(items)))
    avg = (len(items) + n - 1) // n
    return [items[i:i + avg] for i in range(0, len(items), avg)]


def _union_wkb_chunk(wkb_chunk: List[bytes]) -> bytes:
    geoms = [shapely_wkb.loads(b) for b in wkb_chunk]
    if not geoms:
        return b""
    u = ops.unary_union(geoms)
    return shapely_wkb.dumps(u)


def dissolve_geometries_parallel(gdf: gpd.GeoDataFrame, processes: int) -> gpd.GeoDataFrame:
    """Parallel dissolve using multi-process unary_union on geometry chunks."""
    if gdf.empty:
        return gdf
    # Prepare WKB payload to avoid heavy pickling of GeoDataFrame objects
    wkb_list: List[bytes] = gdf.geometry.to_wkb().tolist()
    chunks = _chunk_list(wkb_list, max(1, processes))
    if len(chunks) == 1:
        # Nothing to parallelize
        geom = ops.unary_union([shapely_wkb.loads(b) for b in wkb_list])
        return gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs=gdf.crs)

    partial_unions: List[bytes] = []
    with ProcessPoolExecutor(max_workers=processes) as executor:
        for wkb_u in executor.map(_union_wkb_chunk, chunks):
            if wkb_u:
                partial_unions.append(wkb_u)

    # Final union of partial results
    final_geom = ops.unary_union([shapely_wkb.loads(b) for b in partial_unions])
    return gpd.GeoDataFrame({"geometry": [final_geom]}, geometry="geometry", crs=gdf.crs)


def simplify_dissolved(gdf: gpd.GeoDataFrame, tolerance: float = SIMPLIFY_TOLERANCE) -> gpd.GeoDataFrame:
    """Simplify the dissolved geometry while preserving topology.

    The input is expected to be a single-row GeoDataFrame (dissolved output).
    """
    if gdf.empty or gdf.geometry.isna().all():
        return gdf
    # Use shapely simplify with topology preservation
    simplified_geom = gdf.geometry.simplify(tolerance=tolerance, preserve_topology=True)
    out = gdf.copy()
    out.geometry = simplified_geom
    return out


def process_dataset(dataset_name: str, processes: int) -> Optional[Path]:
    csv_path = OUTPUT_CSV_DIR / f"{dataset_name}.csv"
    if not csv_path.exists():
        print(f"Warning: CSV not found, skipping: {csv_path}")
        return None

    # Ensure output dirs exist
    OUTPUT_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SHP_DIR.mkdir(parents=True, exist_ok=True)
    out_geojson_path = OUTPUT_GEOJSON_DIR / f"{dataset_name}.geojson"
    out_shp_path = OUTPUT_SHP_DIR / f"{dataset_name}.shp"
    # If outputs already exist, skip heavy processing
    if out_geojson_path.exists() and out_shp_path.exists():
        print(f"Exists, skipping: {out_geojson_path} and {out_shp_path}")
        return out_geojson_path

    try:
        dggs_ids = read_unique_dggs_ids(csv_path)
        if not dggs_ids:
            print(f"Warning: No dggsID values found in {csv_path.name}")
            return None

        geom_path, kind = geometry_path_for_dataset(dataset_name)
        if not geom_path.exists():
            print(f"Warning: Geometry source not found for {dataset_name}: {geom_path}")
            return None

        geom_gdf = load_geometry(geom_path, kind)
        filtered = filter_by_dggs_ids(geom_gdf, dggs_ids)
        if filtered.empty:
            print(f"Warning: No matching geometries for {dataset_name}")
            return None

        # Choose dissolve strategy: parallel for large inputs, else normal
        if processes > 1 and len(filtered) > 10000:
            dissolved = dissolve_geometries_parallel(filtered[["geometry"]], processes)
        else:
            dissolved = dissolve_geometries(filtered)
        simplified = simplify_dissolved(dissolved, tolerance=SIMPLIFY_TOLERANCE)

        # Write GeoJSON
        simplified.to_file(out_geojson_path, driver="GeoJSON")
        # Write ESRI Shapefile
        simplified.to_file(out_shp_path, driver="ESRI Shapefile")

        print(f"Wrote: {out_geojson_path}")
        print(f"Wrote: {out_shp_path}")
        return out_geojson_path
    except Exception as ex:
        print(f"Error processing {dataset_name}: {ex}")
        return None


def main() -> None:
    datasets: List[str] = DATASET_FILES

    written: List[Path] = []

    processes = int(os.environ.get('NUM_CORES', 8))
    processes = max(1, processes)

    # Process datasets sequentially; use all CPUs inside each dataset for dissolve
    print(f"Using {processes} CPU processes for within-dataset dissolve; processing {len(datasets)} datasets sequentially...")
    for dataset in datasets:
        result = process_dataset(dataset, processes)
        if result is not None:
            written.append(result)

    if not written:
        print("No GeoJSON files were written.")


if __name__ == "__main__":
    main()


