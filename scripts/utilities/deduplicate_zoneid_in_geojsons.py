#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from datetime import datetime

import geopandas as gpd


def deduplicate_geojson(path: str) -> int:
    """
    Remove duplicate rows by zoneID in the given GeoJSON file (in-place).
    Returns the number of rows removed.
    """
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        return 0

    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return 0

    if 'zoneID' not in gdf.columns:
        print(f"[SKIP] No 'zoneID' column in: {path}")
        return 0

    total_rows = len(gdf)
    dup_mask = gdf.duplicated(subset=['zoneID'], keep='first')
    dup_count = int(dup_mask.sum())

    if dup_count == 0:
        print(f"[OK] {os.path.basename(path)}: 0 duplicates found ({total_rows} rows)")
        return 0

    # Drop duplicate rows and overwrite in-place
    gdf_dedup = gdf[~dup_mask].copy()

    try:
        gdf_dedup.to_file(path, driver='GeoJSON')
        print(
            f"[FIXED] {os.path.basename(path)}: removed {dup_count} duplicates by zoneID "
            f"({total_rows} -> {len(gdf_dedup)} rows)"
        )
    except Exception as e:
        print(f"[ERROR] Failed to write {path}: {e}")
        return 0

    return dup_count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate zoneID rows from all GeoJSON files in a directory "
            "and report how many duplicates were removed."
        )
    )
    parser.add_argument(
        "--dir",
        default="/home/mingke.li/methane_grid_calculation_ARC/data/geojson/global_countries_dggs_merge/",
        help="Directory containing GeoJSON files (default: path to global_countries_dggs_merge)",
    )
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.isdir(target_dir):
        print(f"[ERROR] Not a directory: {target_dir}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(target_dir, "*.geojson")))
    if not files:
        print(f"[WARN] No GeoJSON files found in: {target_dir}")
        sys.exit(0)

    print(f"Scanning {len(files)} GeoJSON files in: {target_dir}")
    total_removed = 0
    for fp in files:
        removed = deduplicate_geojson(fp)
        total_removed += removed

    print("\nSummary:")
    print(f"  Total duplicates removed across all files: {total_removed}")


if __name__ == "__main__":
    main()
