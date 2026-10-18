import json
from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import Point, shape
from shapely.prepared import prep

ROOT = Path(r"D:\fyp")
PROCESSED = ROOT / "data" / "processed"
BOUNDARY_FILE = ROOT / "data" / "raw" / "boundary" / "penang_boundary.geojson"

with BOUNDARY_FILE.open(encoding="utf-8") as file:
    geojson = json.load(file)

boundary = prep(shape(geojson["features"][0]["geometry"]))


def check_stops():
    file = PROCESSED / "stops_penang.csv"
    data = pd.read_csv(file)

    points = [
        Point(float(lon), float(lat))
        for lon, lat in zip(data["stop_lon"], data["stop_lat"])
    ]

    print(
        f"stops_penang.csv: {len(data)} rows; "
        f"{data['stop_id'].nunique()} unique stop IDs; "
        f"{sum(boundary.covers(point) for point in points)} inside boundary"
    )


def check_destination(filename):
    file = PROCESSED / filename
    data = pd.read_csv(file)

    missing = data["geometry"].isna().sum()
    points_inside = 0
    invalid = 0

    for value in data["geometry"].dropna():
        try:
            geom = wkt.loads(value)
            if boundary.covers(geom):
                points_inside += 1
            else:
                invalid += 1
        except Exception:
            invalid += 1

    print(
        f"{filename}: {len(data)} rows; "
        f"{points_inside} inside boundary; "
        f"{missing} missing geometries; {invalid} outside or invalid"
    )


check_stops()

for name in [
    "schools_penang.csv",
    "hospitals_penang.csv",
    "clinics_penang.csv",
    "malls_penang.csv",
]:
    check_destination(name)