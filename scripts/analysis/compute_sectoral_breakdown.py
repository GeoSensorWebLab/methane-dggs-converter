"""
Compute sectoral methane emission totals for selected national datasets.

This script reads five CSVs from the `output/` directory:
  - US_DGGS_methane_emissions_ALL_FILES.csv
  - Canada_DGGS_methane_emissions_ALL_FILES.csv
  - Mexico_DGGS_methane_emissions_ALL_FILES.csv
  - China_DGGS_methane_emissions_ALL_FILES.csv
  - Switzerland_DGGS_methane_emissions_2011.csv

It aggregates total emissions by four broad IPCC sectors based on first-digit column name prefixes:
  - 1* -> Energy
  - 2* -> Industrial Processes and Product Use
  - 3* -> Agriculture, Forestry, and Other Land Use
  - 4* -> Waste

Notes:
  - Identifier columns {'dggsID','Year','GID'} are excluded.
  - A column is mapped to a sector if, after normalization, its first alphanumeric
    character is one of {1,2,3,4}. The script is robust to names like 'IPCC_1A1', '1.A.1', etc.
  - Sums are computed across all rows (all years and cells) for each dataset.

Run:
  python scripts/analysis/compute_sectoral_breakdown.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd


INPUT_DIR = "output"
OUTPUT_DIR = "analysis_results"
OUTPUT_FILENAME = "sectoral_breakdown_summary.csv"
CHINA_TEMPORAL_FILENAME = "china_sectoral_breakdown_1990_2020.csv"
US_TEMPORAL_FILENAME = "us_sectoral_breakdown_2012_2018.csv"
CHUNK_SIZE = 1_000_000

TARGET_FILES = [
    "US_DGGS_methane_emissions_ALL_FILES.csv",
    "Canada_DGGS_methane_emissions_ALL_FILES.csv",
    "Mexico_DGGS_methane_emissions_ALL_FILES.csv",
    "China_DGGS_methane_emissions_ALL_FILES.csv",
    "Switzerland_DGGS_methane_emissions_2011.csv",
]

IDENTIFIER_COLUMNS_CANONICAL: Set[str] = {"dggsid", "year", "gid"}

SECTOR_NAMES = [
    "Energy",
    "Industrial Processes and Product Use",
    "Agriculture, Forestry, and Other Land Use",
    "Waste",
]

# Optional year filters per dataset (basename without .csv)
YEAR_FILTERS = {
    "US_DGGS_methane_emissions_ALL_FILES": 2018,
    "China_DGGS_methane_emissions_ALL_FILES": 2020,
}


def normalize_column_name(name: str) -> str:
    return name.strip().lower()


def read_header_columns(csv_path: Path) -> List[str]:
    df_header = pd.read_csv(csv_path, nrows=0)
    return list(df_header.columns)


def map_column_to_sector(col: str) -> Optional[str]:
    s = col.strip()
    if not s:
        return None
    first = s[0]
    if first == "1":
        return "Energy"
    if first == "2":
        return "Industrial Processes and Product Use"
    if first == "3":
        return "Agriculture, Forestry, and Other Land Use"
    if first == "4":
        return "Waste"
    return None


def select_ipcc_columns(columns: List[str]) -> List[str]:
    normalized_id = {normalize_column_name(c) for c in COLUMNS_ID_LIST}
    return [c for c in columns if normalize_column_name(c) not in normalized_id]


# Keep identifier list separately for select function
COLUMNS_ID_LIST = ["dggsID", "Year", "GID"]


def aggregate_file_by_sector_temporal(csv_path: Path, year_range: Optional[tuple]) -> List[Dict[str, object]]:
    """Aggregate emissions by sector and year for temporal analysis."""
    columns = read_header_columns(csv_path)
    ipcc_columns = select_ipcc_columns(columns)
    if not ipcc_columns:
        return []

    # Build sector -> column list mapping
    sector_to_columns: Dict[str, List[str]] = {s: [] for s in SECTOR_NAMES}
    for col in ipcc_columns:
        sector = map_column_to_sector(col)
        if sector is not None:
            sector_to_columns[sector].append(col)

    # Find Year column
    year_col_actual: Optional[str] = None
    for c in columns:
        if c.lower() == "year":
            year_col_actual = c
            break
    
    if year_col_actual is None:
        print(f"Warning: No Year column found in {csv_path.name}")
        return []

    # Read only the columns we need
    needed_cols: List[str] = sorted(list({c for cols in sector_to_columns.values() for c in cols}) + [year_col_actual])
    
    # Group by year and aggregate
    year_totals: Dict[int, Dict[str, float]] = {}
    
    for chunk in pd.read_csv(
        csv_path,
        usecols=needed_cols,
        chunksize=CHUNK_SIZE,
    ):
        # Apply year range filter if specified
        if year_range is not None:
            chunk = chunk[(chunk[year_col_actual] >= year_range[0]) & 
                         (chunk[year_col_actual] <= year_range[1])]
        
        if chunk.empty:
            continue

        # Ensure numeric
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        
        # Group by year
        for year in chunk[year_col_actual].dropna().unique():
            year_int = int(year)
            year_data = chunk[chunk[year_col_actual] == year_int]
            
            if year_int not in year_totals:
                year_totals[year_int] = {s: 0.0 for s in SECTOR_NAMES}
            
            # Sum by sector for this year
            for sector, cols in sector_to_columns.items():
                if not cols:
                    continue
                sector_year_sum = year_data[cols].sum(numeric_only=True).sum()
                year_totals[year_int][sector] += float(sector_year_sum)

    # Convert to list of dictionaries
    rows = []
    for year in sorted(year_totals.keys()):
        row = {"Year": year}
        total_sum = 0.0
        for sector in SECTOR_NAMES:
            val = year_totals[year].get(sector, 0.0)
            row[sector] = val
            total_sum += val
        row["Total"] = total_sum
        rows.append(row)
    
    return rows


def aggregate_file_by_sector(csv_path: Path, year_filter: Optional[int]) -> Dict[str, float]:
    columns = read_header_columns(csv_path)
    ipcc_columns = select_ipcc_columns(columns)
    if not ipcc_columns:
        return {sector: 0.0 for sector in SECTOR_NAMES}

    # Build sector -> column list mapping
    sector_to_columns: Dict[str, List[str]] = {s: [] for s in SECTOR_NAMES}
    for col in ipcc_columns:
        sector = map_column_to_sector(col)
        if sector is not None:
            sector_to_columns[sector].append(col)

    # If no columns map to a given sector, it will remain empty and sum to 0
    sector_totals: Dict[str, float] = {s: 0.0 for s in SECTOR_NAMES}

    # Read only the columns we need
    needed_cols: List[str] = sorted({c for cols in sector_to_columns.values() for c in cols})
    # Add Year for filtering if required
    year_col_actual: Optional[str] = None
    if year_filter is not None:
        for c in columns:
            if c.lower() == "year":
                year_col_actual = c
                needed_cols = sorted(set(needed_cols + [c]))
                break
    if not needed_cols:
        return sector_totals

    for chunk in pd.read_csv(
        csv_path,
        usecols=needed_cols,
        chunksize=CHUNK_SIZE,
    ):
        # Optional year filter
        if year_filter is not None and year_col_actual is not None and year_col_actual in chunk.columns:
            chunk = chunk[chunk[year_col_actual] == year_filter]
        if chunk.empty:
            continue

        # Ensure numeric and sum per sector
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        for sector, cols in sector_to_columns.items():
            if not cols:
                continue
            # Sum all values across selected columns in this chunk
            sector_chunk_sum = chunk[cols].sum(numeric_only=True).sum()
            sector_totals[sector] += float(sector_chunk_sum)

    return sector_totals


def main() -> None:
    input_dir = Path(INPUT_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / OUTPUT_FILENAME

    rows: List[Dict[str, object]] = []

    for filename in TARGET_FILES:
        csv_path = input_dir / filename
        if not csv_path.exists():
            print(f"Warning: file not found, skipping: {csv_path}")
            continue
        year_filter = YEAR_FILTERS.get(csv_path.stem)
        sector_totals = aggregate_file_by_sector(csv_path, year_filter)
        total_sum = sum(sector_totals.values())
        row: Dict[str, object] = {"data_source": csv_path.stem}
        for sector in SECTOR_NAMES:
            row[sector] = sector_totals.get(sector, 0.0)
        row["Total"] = total_sum
        rows.append(row)

    if not rows:
        print("No rows computed; nothing to write.")
        return

    df = pd.DataFrame(rows)
    # Ensure stable column order
    ordered_cols = ["data_source"] + SECTOR_NAMES + ["Total"]
    df = df[ordered_cols]
    df.sort_values(by=["data_source"], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote sectoral breakdown summary to: {output_csv}")

    # === Generate temporal breakdowns ===
    
    # China temporal breakdown (1990-2020)
    china_path = input_dir / "China_DGGS_methane_emissions_ALL_FILES.csv"
    if china_path.exists():
        china_rows = aggregate_file_by_sector_temporal(china_path, (1990, 2020))
        if china_rows:
            china_df = pd.DataFrame(china_rows)
            china_ordered_cols = ["Year"] + SECTOR_NAMES + ["Total"]
            china_df = china_df[china_ordered_cols]
            china_output = output_dir / CHINA_TEMPORAL_FILENAME
            china_df.to_csv(china_output, index=False)
            print(f"Wrote China temporal breakdown to: {china_output}")
        else:
            print("Warning: No data found for China temporal breakdown")
    else:
        print(f"Warning: China file not found: {china_path}")

    # US temporal breakdown (2012-2018)
    us_path = input_dir / "US_DGGS_methane_emissions_ALL_FILES.csv"
    if us_path.exists():
        us_rows = aggregate_file_by_sector_temporal(us_path, (2012, 2018))
        if us_rows:
            us_df = pd.DataFrame(us_rows)
            us_ordered_cols = ["Year"] + SECTOR_NAMES + ["Total"]
            us_df = us_df[us_ordered_cols]
            us_output = output_dir / US_TEMPORAL_FILENAME
            us_df.to_csv(us_output, index=False)
            print(f"Wrote US temporal breakdown to: {us_output}")
        else:
            print("Warning: No data found for US temporal breakdown")
    else:
        print(f"Warning: US file not found: {us_path}")


if __name__ == "__main__":
    main()


