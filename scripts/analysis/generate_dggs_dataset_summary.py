"""
Generate a descriptive summary table for DGGS-converted methane inventory CSVs.

For each CSV in the input directory, this script computes:
 - data_source: base filename without extension
 - num_dggs_cells: count of unique DGGS cells (unique values in column 'dggsID')
 - dggs_resolution: DGGS resolution level (from mapping JSON or CLI-supplied path)
 - year_range: temporal coverage from mapping JSON only (no CSV parsing)
 - num_ipcc_categories: number of columns excluding identifier columns {'dggsID','Year','GID'}

Notes:
 - This script is optimized to read only required columns and uses chunking for large files.
 - Resolution and year_range are provided via mappings; the script does not derive years from the CSVs.
 - Identifier column name matching is case-insensitive.

Run with defaults:
  python scripts/analysis/generate_dggs_dataset_summary.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd


# Identifier columns to exclude when counting IPCC category columns.
IDENTIFIER_COLUMNS_CANONICAL = {"dggsid", "year", "gid"}

# Default configuration (no CLI args required)
INPUT_DIR = "output"
OUTPUT_DIR = "analysis_results"
OUTPUT_FILENAME = "dggs_dataset_summary.csv"
RESOLUTION_MAP_JSON = os.path.join("analysis_results", "configs", "dggs_resolution_map.json")
YEAR_RANGE_MAP_JSON = os.path.join("analysis_results", "configs", "year_range_map.json")
CHUNK_SIZE = 1_000_000


@dataclass
class DatasetSummary:
    data_source: str
    num_dggs_cells: Optional[int]
    dggs_resolution: Optional[str]
    year_range: Optional[str]
    num_ipcc_categories: Optional[int]

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "data_source": self.data_source,
            "num_dggs_cells": None if self.num_dggs_cells is None else str(self.num_dggs_cells),
            "dggs_resolution": self.dggs_resolution,
            "year_range": self.year_range,
            "num_ipcc_categories": None if self.num_ipcc_categories is None else str(self.num_ipcc_categories),
        }


def load_optional_json_mapping(json_path: Optional[str]) -> Dict[str, str]:
    if not json_path:
        return {}
    path_obj = Path(json_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Mapping file not found: {json_path}")
    with path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Mapping JSON must be a dictionary of {dataset_base_name: value}.")
        # Normalize keys to base names
        return {Path(k).stem: str(v) for k, v in data.items()}


def list_csv_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.csv") if p.is_file()])


def read_header_columns(csv_path: Path) -> List[str]:
    # Read zero rows to obtain columns quickly
    df_header = pd.read_csv(csv_path, nrows=0)
    return list(df_header.columns)


def normalize_column_name(name: str) -> str:
    return name.strip().lower()


def count_ipcc_columns(columns: List[str]) -> int:
    normalized = [normalize_column_name(c) for c in columns]
    excluded = IDENTIFIER_COLUMNS_CANONICAL
    return sum(1 for c in normalized if c not in excluded)


def compute_unique_dggs_ids(
    csv_path: Path, chunksize: int, columns: List[str]
) -> Optional[int]:
    normalized = [normalize_column_name(c) for c in columns]
    try:
        dggs_idx = normalized.index("dggsid")
    except ValueError:
        return None

    # Use chunked reading for scalability
    unique_ids: Set[str] = set()
    # Use original column name for usecols to avoid pandas confusion on duplicate names
    dggs_colname = columns[dggs_idx]

    for chunk in pd.read_csv(
        csv_path,
        usecols=[dggs_colname],
        dtype={dggs_colname: str},
        chunksize=chunksize,
    ):
        # Drop NA then update the set
        series = chunk[dggs_colname].dropna()
        unique_ids.update(series.astype(str).unique().tolist())

    return len(unique_ids)


def infer_data_source_name(csv_path: Path) -> str:
    # Use base filename without extension as data source identifier
    return csv_path.stem


def summarize_csv(
    csv_path: Path,
    resolution_map: Dict[str, str],
    year_range_map: Dict[str, str],
    chunksize: int,
) -> DatasetSummary:
    dataset_name = infer_data_source_name(csv_path)
    columns = read_header_columns(csv_path)

    # Count IPCC columns using only header
    num_ipcc = count_ipcc_columns(columns)

    # Compute unique DGGS cells
    num_cells = compute_unique_dggs_ids(csv_path, chunksize, columns)

    # Year range comes exclusively from mapping
    year_range = year_range_map.get(dataset_name)

    # Resolution is mapping-only unless provided at runtime via JSON
    dggs_resolution = resolution_map.get(dataset_name)

    return DatasetSummary(
        data_source=dataset_name,
        num_dggs_cells=num_cells,
        dggs_resolution=dggs_resolution,
        year_range=year_range,
        num_ipcc_categories=num_ipcc,
    )


def write_summary_csv(summaries: List[DatasetSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([s.to_dict() for s in summaries])
    # Stable, readable column order
    desired_cols = [
        "data_source",
        "num_dggs_cells",
        "dggs_resolution",
        "year_range",
        "num_ipcc_categories",
    ]
    existing_cols = [c for c in desired_cols if c in df.columns]
    df = df[existing_cols]
    df.sort_values(by=["data_source"], inplace=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    input_dir = Path(INPUT_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    output_csv = output_dir / OUTPUT_FILENAME

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

    resolution_map = load_optional_json_mapping(RESOLUTION_MAP_JSON)
    year_range_map = load_optional_json_mapping(YEAR_RANGE_MAP_JSON)

    csv_files = list_csv_files(input_dir)
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    summaries: List[DatasetSummary] = []
    for csv_path in csv_files:
        try:
            summary = summarize_csv(
                csv_path=csv_path,
                resolution_map=resolution_map,
                year_range_map=year_range_map,
                chunksize=CHUNK_SIZE,
            )
            summaries.append(summary)
        except Exception as ex:
            # Robustness: if a file fails, record minimal info for traceability
            summaries.append(
                DatasetSummary(
                    data_source=csv_path.stem,
                    num_dggs_cells=None,
                    dggs_resolution=resolution_map.get(csv_path.stem),
                    year_range=year_range_map.get(csv_path.stem),
                    num_ipcc_categories=None,
                )
            )
            print(f"Warning: failed to summarize {csv_path.name}: {ex}")

    write_summary_csv(summaries, output_csv)
    print(f"Wrote summary to: {output_csv}")


if __name__ == "__main__":
    main()


