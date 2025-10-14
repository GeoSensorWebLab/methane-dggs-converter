"""
Convert a single region GeoJSON to DGGS cells efficiently by tiling and streaming.

This refactor avoids loading the entire DGGS grid or full output into memory.
It splits the region's bbox into tiles, calls the DGGS CLI per-tile, filters
by intersection with the region geometry, and writes out incrementally.

Recommended output formats for very large results:
- Parquet dataset (fast, compact) — default when format is "parquet"
- Newline-delimited GeoJSON (geojsonl or geojsonl.gz) — streamable
"""

import os
import sys
import json
import math
import gzip
import argparse
import subprocess
from typing import Generator, Iterable, List, Tuple, Optional, Dict, Any

import geopandas as gpd
from shapely.geometry import shape as shapely_shape
from shapely.prepared import prep as shapely_prep


GRID_TYPE_DEFAULT = "rhealpix"
LEVEL_DEFAULT = 10
INPUT_FILE_DEFAULT = os.path.join("data", "geojson", "newyorkstate.geojson")
OUTPUT_DIR_DEFAULT = os.path.join("data", "geojson", "regional_grid")


def run_dggs_grid(grid: str, level: int, bbox: str) -> Dict[str, Any]:
    """Run the DGGS CLI to generate a grid for the given bbox and return GeoJSON as dict.

    Note: We still parse tile output as a whole JSON document. Keep tiles small
    to limit memory use. If an NDJSON mode becomes available in the CLI, switch
    to streaming parse for even lower memory usage.
    """
    cmd = ["dgg", grid, "grid", str(level), "-bbox", bbox]
    print("Running DGGS CLI:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"DGGS command failed (exit {result.returncode}): {result.stderr}")

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("DGGS command returned empty output")

    return json.loads(output)


def generate_tiles(minx: float, miny: float, maxx: float, maxy: float, tile_deg: float) -> List[Tuple[float, float, float, float]]:
    """Generate non-overlapping lon/lat tiles covering the bbox.

    Returns tiles as (minx, miny, maxx, maxy) in lon/lat.
    """
    if tile_deg <= 0:
        return [(minx, miny, maxx, maxy)]

    tiles: List[Tuple[float, float, float, float]] = []
    # Clamp to bounds
    minx_c, miny_c = max(-180.0, minx), max(-90.0, miny)
    maxx_c, maxy_c = min(180.0, maxx), min(90.0, maxy)
    # Steps
    x_steps = max(1, math.ceil((maxx_c - minx_c) / tile_deg))
    y_steps = max(1, math.ceil((maxy_c - miny_c) / tile_deg))
    for yi in range(y_steps):
        ty_min = miny_c + yi * tile_deg
        ty_max = min(ty_min + tile_deg, maxy_c)
        for xi in range(x_steps):
            tx_min = minx_c + xi * tile_deg
            tx_max = min(tx_min + tile_deg, maxx_c)
            tiles.append((tx_min, ty_min, tx_max, ty_max))
    return tiles


class GeoJSONLWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.gz = output_path.endswith(".gz")
        self.fh = gzip.open(output_path, "wt", encoding="utf-8") if self.gz else open(output_path, "w", encoding="utf-8")

    def write_features(self, features: Iterable[Dict[str, Any]]) -> None:
        for feat in features:
            self.fh.write(json.dumps(feat, separators=(",", ":")))
            self.fh.write("\n")

    def close(self) -> None:
        self.fh.close()


class GeoJSONStreamWriter:
    """Minimal streaming FeatureCollection writer to valid .geojson."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.gz = output_path.endswith(".gz")
        self.fh = gzip.open(output_path, "wt", encoding="utf-8") if self.gz else open(output_path, "w", encoding="utf-8")
        self.started = False
        self.count = 0
        # Write header
        self.fh.write('{"type":"FeatureCollection","features":[')

    def write_features(self, features: Iterable[Dict[str, Any]]) -> None:
        for feat in features:
            if self.started:
                self.fh.write(",")
            self.fh.write(json.dumps(feat, separators=(",", ":")))
            self.started = True
            self.count += 1

    def close(self) -> None:
        self.fh.write("]}")
        self.fh.close()


class ParquetPartitionWriter:
    def __init__(self, output_path: str, region: str, level: int):
        self.output_path = output_path
        self.region = region
        self.level = level
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.part_idx = 0
        self.all_rows = []

    def write_batch(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        self.all_rows.extend(rows)

    def close(self) -> None:
        if self.all_rows:
            gdf = gpd.GeoDataFrame(self.all_rows, geometry="geometry", crs="EPSG:4326")
            gdf.to_parquet(self.output_path, index=False)


def iter_intersecting_features(grid_geojson: Dict[str, Any], prepared_region, region_name: str, seen_zone_ids: Optional[set]) -> Generator[Dict[str, Any], None, None]:
    """Yield DGGS features that intersect the region, adding metadata and de-duplicating by zoneID if provided."""
    features = grid_geojson.get("features", [])
    for idx, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shapely_shape(geom)
        if not prepared_region.intersects(shp):
            continue
        props = feat.get("properties", {}) or {}
        zone_id = props.get("zoneID")
        if zone_id is None:
            # Fallback if upstream does not provide zoneID
            zone_id = props.get("id") or f"cell_{idx}"
            props["zoneID"] = zone_id
        if seen_zone_ids is not None:
            if zone_id in seen_zone_ids:
                continue
            seen_zone_ids.add(zone_id)
        props["region"] = region_name
        feat["properties"] = props
        yield feat


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert region GeoJSON to DGGS cells by tiling and streaming.")
    parser.add_argument("--grid", default=GRID_TYPE_DEFAULT, help="DGGS grid type (default: rhealpix)")
    parser.add_argument("--level", type=int, default=LEVEL_DEFAULT, help="DGGS resolution level")
    parser.add_argument("--input", dest="input_path", default=INPUT_FILE_DEFAULT, help="Input region GeoJSON path")
    parser.add_argument("--output-dir", dest="output_dir", default=OUTPUT_DIR_DEFAULT, help="Output directory")
    parser.add_argument("--format", dest="out_format", default="parquet", choices=["parquet", "geojsonl", "geojsonl.gz", "geojson", "geojson.gz"], help="Output format")
    parser.add_argument("--tile-deg", type=float, default=2.0, help="Tile size in degrees (lon/lat)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Number of features per write batch")
    parser.add_argument("--dedup", action="store_true", help="De-duplicate cells across tiles by zoneID (recommended)")
    parser.add_argument("--max-tiles", type=int, default=None, help="Process at most N tiles (for testing)")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Input not found: {args.input_path}")
        return 1

    region_name = os.path.splitext(os.path.basename(args.input_path))[0]

    # Read region geometry and ensure WGS84
    region_gdf = gpd.read_file(args.input_path)
    if region_gdf.empty:
        print("Input GeoJSON is empty")
        return 1
    if region_gdf.crs is None:
        region_gdf = region_gdf.set_crs("EPSG:4326")
    else:
        region_gdf = region_gdf.to_crs("EPSG:4326")

    # Union and prepare geometry for fast spatial predicates
    region_geom = region_gdf.geometry.union_all()
    prepared_region = shapely_prep(region_geom)

    # Determine tiles
    minx, miny, maxx, maxy = region_geom.bounds  # lon/lat
    tiles = generate_tiles(minx, miny, maxx, maxy, args.tile_deg)
    if args.max_tiles is not None:
        tiles = tiles[: args.max_tiles]
    num_tiles = len(tiles)
    print(f"Region bbox (lon/lat): {minx:.6f},{miny:.6f} to {maxx:.6f},{maxy:.6f}; tiles: {num_tiles} @ {args.tile_deg}°")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    total_kept = 0
    seen_zone_ids: Optional[set] = set() if args.dedup else None

    # Writer setup
    writer_geojsonl: Optional[GeoJSONLWriter] = None
    writer_geojson: Optional[GeoJSONStreamWriter] = None
    writer_parquet: Optional[ParquetPartitionWriter] = None

    if args.out_format == "parquet":
        out_path = os.path.join(args.output_dir, f"{region_name}_grid_res{args.level}.parquet")
        writer_parquet = ParquetPartitionWriter(out_path, region_name, args.level)
        print(f"Writing Parquet file to: {out_path}")
    elif args.out_format in ("geojsonl", "geojsonl.gz"):
        ext = ".geojsonl.gz" if args.out_format.endswith(".gz") else ".geojsonl"
        out_path = os.path.join(args.output_dir, f"{region_name}_res{args.level}{ext}")
        writer_geojsonl = GeoJSONLWriter(out_path)
        print(f"Writing GeoJSONL to: {out_path}")
    else:
        # geojson or geojson.gz
        ext = ".geojson.gz" if args.out_format.endswith(".gz") else ".geojson"
        out_path = os.path.join(args.output_dir, f"{region_name}_res{args.level}{ext}")
        writer_geojson = GeoJSONStreamWriter(out_path)
        print(f"Writing GeoJSON FeatureCollection to: {out_path}")

    # Process tiles
    batch_rows: List[Dict[str, Any]] = []  # for parquet
    batch_feats: List[Dict[str, Any]] = []  # for geojson/geojsonl

    for tile_idx, (tx_min, ty_min, tx_max, ty_max) in enumerate(tiles):
        # DGGS bbox expects south,west,north,east (lat_min, lon_min, lat_max, lon_max)
        bbox_str = f"{ty_min},{tx_min},{ty_max},{tx_max}"
        print(f"Tile {tile_idx+1}/{num_tiles}: bbox (lon/lat) {tx_min:.4f},{ty_min:.4f} to {tx_max:.4f},{ty_max:.4f}")

        grid_geojson = run_dggs_grid(args.grid, args.level, bbox_str)

        # Iterate intersecting features and write by batches
        for feat in iter_intersecting_features(grid_geojson, prepared_region, region_name, seen_zone_ids):
            if writer_parquet is not None:
                props = feat.get("properties", {}) or {}
                zone_id = props.get("zoneID")
                batch_rows.append({
                    "zoneID": zone_id,
                    "region": props.get("region", region_name),
                    "geometry": shapely_shape(feat["geometry"]),
                })
                if len(batch_rows) >= args.batch_size:
                    writer_parquet.write_batch(batch_rows)
                    total_kept += len(batch_rows)
                    print(f"  wrote parquet batch; total kept: {total_kept}")
                    batch_rows.clear()
            else:
                batch_feats.append(feat)
                if len(batch_feats) >= args.batch_size:
                    if writer_geojsonl is not None:
                        writer_geojsonl.write_features(batch_feats)
                    elif writer_geojson is not None:
                        writer_geojson.write_features(batch_feats)
                    total_kept += len(batch_feats)
                    print(f"  wrote json batch; total kept: {total_kept}")
                    batch_feats.clear()

        # Free per-tile JSON once processed
        grid_geojson = None  # hint for GC

    # Flush remaining
    if writer_parquet is not None and batch_rows:
        writer_parquet.write_batch(batch_rows)
        total_kept += len(batch_rows)
        batch_rows.clear()
    if (writer_geojsonl is not None or writer_geojson is not None) and batch_feats:
        if writer_geojsonl is not None:
            writer_geojsonl.write_features(batch_feats)
        elif writer_geojson is not None:
            writer_geojson.write_features(batch_feats)
        total_kept += len(batch_feats)
        batch_feats.clear()

    # Close writers
    if writer_parquet is not None:
        writer_parquet.close()
    if writer_geojsonl is not None:
        writer_geojsonl.close()
    if writer_geojson is not None:
        writer_geojson.close()

    print(f"Done. Total DGGS cells kept: {total_kept}")
    return 0 if total_kept > 0 else 1


if __name__ == "__main__":
    sys.exit(main())


