from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask

ROOT = Path(r"D:\fyp")

raster_path = ROOT / "data/raw/population/mys_ppp_2020_constrained.tif"
boundary_path = ROOT / "data/raw/boundary/penang_mukim_boundaries.geojson"
csv_path = ROOT / "data/raw/population/mukim_stat.csv"
output_path = ROOT / "data/processed/mukim_population_raster_comparison.csv"

district_prefix = {
    "01": "SPT",
    "02": "SPU",
    "03": "SPS",
    "04": "DTL",
    "05": "DBD",
}

boundary = gpd.read_file(boundary_path)
population = pd.read_csv(csv_path)

def boundary_label(row):
    district = district_prefix[str(row["KOD_DAERAH"]).zfill(2)]
    name = str(row["NAM"]).strip()

    if name.isdigit() or (len(name) == 1 and name.isalpha()):
        return f"{district} MUKIM {name}".upper()

    name = name.removeprefix("Bandar ").strip().upper()

    aliases = {
        "SUNGEI BAKAP": "SUNGAI BAKAP",
        "BATU FERINGGI": "BATU FERRINGGI",
        "BUTTERWORTH": "BUTTERWORT",
    }
    name = aliases.get(name, name)

    return f"{district} {name}"

boundary["match_name"] = boundary.apply(boundary_label, axis=1)
population["match_name"] = (
    population["NAMA PENTADBIRAN"].astype(str).str.strip().str.upper()
)

# Confirm that every CSV unit matches one boundary feature.
matches = population.merge(
    boundary[["match_name", "geometry"]],
    on="match_name",
    how="outer",
    indicator=True,
)

unmatched = matches[matches["_merge"] != "both"]

print(f"CSV rows: {len(population)}")
print(f"Boundary features: {len(boundary)}")
print(f"Matched rows: {(matches['_merge'] == 'both').sum()}")

if not unmatched.empty:
    print("\nUnmatched names:")
    print(unmatched[["match_name", "_merge"]].to_string(index=False))
    raise SystemExit("\nFix the unmatched names before continuing.")

# Read and clean the population figures.
population["csv_population"] = pd.to_numeric(
    population["Sum of JUMLAH PENDUDUK"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce",
)

results = []

with rasterio.open(raster_path) as src:
    if boundary.crs != src.crs:
        boundary = boundary.to_crs(src.crs)

    for _, row in boundary.iterrows():
        clipped, _ = mask(
            src,
            [row.geometry],
            crop=True,
            filled=False,
        )

        values = clipped[0].compressed()
        values = values[values >= 0]
        raster_population = float(values.sum())

        results.append({
            "match_name": row["match_name"],
            "raster_population": raster_population,
        })

raster_totals = pd.DataFrame(results)

comparison = population.merge(
    raster_totals,
    on="match_name",
    validate="one_to_one",
)

comparison["difference_raster_minus_csv"] = (
    comparison["raster_population"] - comparison["csv_population"]
)

comparison["raster_to_csv_percent"] = (
    comparison["raster_population"]
    / comparison["csv_population"]
    * 100
)

comparison.to_csv(output_path, index=False)

print(f"\nCSV population total: {comparison['csv_population'].sum():,.0f}")
print(f"Raster total by mukim: {comparison['raster_population'].sum():,.0f}")
print(
    "Difference: "
    f"{comparison['raster_population'].sum() - comparison['csv_population'].sum():,.0f}"
)
print(f"\nSaved comparison: {output_path}")

print("\nLargest absolute differences:")
comparison["absolute_difference"] = (
    comparison["difference_raster_minus_csv"].abs()
)
print(
    comparison.nlargest(10, "absolute_difference")[
        [
            "NAMA PENTADBIRAN",
            "csv_population",
            "raster_population",
            "difference_raster_minus_csv",
            "raster_to_csv_percent",
        ]
    ].to_string(index=False)
)