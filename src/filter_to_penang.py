import json
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, shape
from shapely import wkt
from shapely.prepared import prep

ROOT = Path(r"D:\fyp")
BOUNDARY_FILE = ROOT / "data" / "raw" / "boundary" / "penang_boundary.geojson"
GTFS_DIR = ROOT / "data" / "raw" / "gtfs"
PROCESSED_DIR = ROOT / "data" / "processed"

DESTINATION_FILES = [
    "schools.csv",
    "hospitals.csv",
    "clinics.csv",
    "malls.csv",
]


def load_penang_boundary():
    with BOUNDARY_FILE.open(encoding="utf-8") as file:
        geojson = json.load(file)

    # The boundary file contains one feature.
    feature = geojson["features"][0]
    boundary = shape(feature["geometry"])

    return prep(boundary)


def filter_stops(boundary):
    source = GTFS_DIR / "stops.txt"
    output = PROCESSED_DIR / "stops_penang.csv"

    data = pd.read_csv(source)
    inside = data.apply(
        lambda row: boundary.covers(
            Point(float(row["stop_lon"]), float(row["stop_lat"]))
        ),
        axis=1,
    )

    filtered = data.loc[inside].copy()
    filtered.to_csv(output, index=False)

    print(f"Stops: {len(data):,} total; {len(filtered):,} inside boundary")
    print(f"Saved: {output}")


def filter_destinations(boundary, filename):
    source = PROCESSED_DIR / filename
    output = PROCESSED_DIR / f"{source.stem}_penang.csv"

    data = pd.read_csv(source)
    geometries = data["geometry"].fillna("").astype(str)

    inside = geometries.apply(
        lambda value: boundary.covers(wkt.loads(value))
        if value.strip()
        else False
    )

    filtered = data.loc[inside].copy()
    filtered.to_csv(output, index=False)

    print(
        f"{filename}: {len(data):,} total; "
        f"{len(filtered):,} inside boundary"
    )
    print(f"Saved: {output}")


def main():
    boundary = load_penang_boundary()

    filter_stops(boundary)

    for filename in DESTINATION_FILES:
        filter_destinations(boundary, filename)


if __name__ == "__main__":
    main()