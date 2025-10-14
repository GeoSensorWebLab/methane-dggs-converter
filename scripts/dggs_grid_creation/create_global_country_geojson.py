import os
import pandas as pd
import geopandas as gpd
import pygadm

# Continent names accepted by pygadm.Items(name=...)
# CONTINENTS = ["North America", "South America", "Antarctica", "Europe", "Asia", "Oceania", "Africa"]
CONTINENTS = ["North America", "South America", "Antarctica", "Europe", "Oceania", "Africa"]
# Some docs use the misspelling "Antartica" — try it as a fallback
ALIASES = {"Antarctica": ["Antartica"]}

# Desired output schema
FINAL_COLS = ["GID", "NAME", "CONTINENT", "geometry"]

def fetch_adm0_for_continent(continent: str) -> gpd.GeoDataFrame:
    """Fetch ADM0 countries for a continent and return standardized columns."""
    variants = [continent] + ALIASES.get(continent, [])
    # Be forgiving with case
    variants += [continent.title(), continent.lower(), continent.upper()]

    last_err = None
    for name in variants:
        # 1) Try the Items(...) interface
        try:
            obj = pygadm.Items(name=name, content_level=0)  # ADM0
            gdf = getattr(obj, "geodataframe", None) or getattr(obj, "gdf", None) or obj
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                out = gdf.copy()
                out["CONTINENT"] = continent
                out = out.rename(columns={"GID_0": "GID", "NAME_0": "NAME"})
                return out[FINAL_COLS]
        except Exception as e:
            last_err = e

        # 2) Fallback to functional API
        try:
            gdf = pygadm.get_items(name=name, content_level=0)
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                out = gdf.copy()
                out["CONTINENT"] = continent
                out = out.rename(columns={"GID_0": "GID", "NAME_0": "NAME"})
                return out[FINAL_COLS]
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to fetch ADM0 for '{continent}'. Last error: {last_err}")

# Loop all continents and combine
parts = [fetch_adm0_for_continent(cont) for cont in CONTINENTS]

# Ensure all parts have the same CRS before concatenating
base_crs = parts[0].crs if parts else "EPSG:4326"
for part in parts:
    if part.crs != base_crs:
        part = part.to_crs(base_crs)

world_adm0 = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=base_crs)[FINAL_COLS]

# Optional: fix invalid geometries if any
try:
    world_adm0["geometry"] = world_adm0["geometry"].buffer(0)
except Exception:
    pass

# Save to GeoJSON with explicit CRS
output_file = "data/geojson/global_countries.geojson"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Ensure CRS is set before saving
if world_adm0.crs is None:
    world_adm0 = world_adm0.set_crs("EPSG:4326")

world_adm0.to_file(output_file, driver="GeoJSON")

print(f"Saved {len(world_adm0)} features to: {output_file}")
print(f"CRS: {world_adm0.crs}")
print(f"Columns: {world_adm0.columns.tolist()}")
print(f"Sample countries: {world_adm0['NAME'].head(5).tolist()}")