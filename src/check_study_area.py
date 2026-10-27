import json
import pandas as pd
from shapely.geometry import shape, Point
from shapely import wkt


# ============================================================
# 1. File paths
# ============================================================

boundary_path = r"D:\fyp\data\raw\boundary\penang_boundary.geojson"

gtfs_path = r"D:\fyp\data\raw\gtfs\stops.txt"

schools_path = r"D:\fyp\data\processed\schools.csv"
hospitals_path = r"D:\fyp\data\processed\hospitals.csv"
clinics_path = r"D:\fyp\data\processed\clinics.csv"
malls_path = r"D:\fyp\data\processed\malls.csv"


# ============================================================
# 2. Load Penang boundary
# ============================================================

print("Loading Penang boundary...")

with open(boundary_path, "r", encoding="utf-8") as file:
    boundary_data = json.load(file)

# Convert GeoJSON features into Shapely geometries
boundary_geometries = [
    shape(feature["geometry"])
    for feature in boundary_data["features"]
]

# Combine all boundary features
penang_boundary = boundary_geometries[0]

for geometry in boundary_geometries[1:]:
    penang_boundary = penang_boundary.union(geometry)

print("Penang boundary loaded successfully.")


# ============================================================
# 3. Load bus stops
# ============================================================

print("\nLoading bus stops...")

stops = pd.read_csv(gtfs_path)

print("Bus stops:", len(stops))


# ============================================================
# 4. Load OSM destinations
# ============================================================

print("Loading destination data...")

schools = pd.read_csv(schools_path)
hospitals = pd.read_csv(hospitals_path)
clinics = pd.read_csv(clinics_path)
malls = pd.read_csv(malls_path)

print("Schools:", len(schools))
print("Hospitals:", len(hospitals))
print("Clinics:", len(clinics))
print("Malls:", len(malls))


# ============================================================
# 5. Convert destination geometry
# ============================================================

def get_point(geometry):

    geometry = wkt.loads(geometry)

    # For polygons, use a representative point.
    # For points, this returns the same location.
    return geometry.representative_point()


schools["point"] = schools["geometry"].apply(get_point)
hospitals["point"] = hospitals["geometry"].apply(get_point)
clinics["point"] = clinics["geometry"].apply(get_point)
malls["point"] = malls["geometry"].apply(get_point)


# ============================================================
# 6. Check bus stops
# ============================================================

def check_bus_stops(data):

    inside = []

    for _, row in data.iterrows():

        point = Point(
            row["stop_lon"],
            row["stop_lat"]
        )

        inside.append(
            penang_boundary.covers(point)
        )

    return inside


# ============================================================
# 7. Check destination points
# ============================================================

def check_destinations(data):

    inside = []

    for point in data["point"]:

        inside.append(
            penang_boundary.covers(point)
        )

    return inside


# ============================================================
# 8. Run validation
# ============================================================

print("\n===== STUDY AREA VALIDATION =====")


datasets = {
    "Bus Stops": check_bus_stops(stops),
    "Schools": check_destinations(schools),
    "Hospitals": check_destinations(hospitals),
    "Clinics": check_destinations(clinics),
    "Malls": check_destinations(malls)
}


for name, results in datasets.items():

    total = len(results)
    inside = sum(results)
    outside = total - inside

    print(f"\n{name}:")
    print("Total:", total)
    print("Inside Penang:", inside)
    print("Outside Penang:", outside)

# ============================================================
# 9. Show points outside Penang
# ============================================================

print("\n===== POINTS OUTSIDE PENANG =====")


# Bus stops outside Penang
bus_inside = datasets["Bus Stops"]

outside_bus_stops = stops.loc[
    [not value for value in bus_inside]
]

print("\nBus stops outside Penang:")
print(
    outside_bus_stops[
        ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    ].to_string(index=False)
)


# Schools outside Penang
school_inside = datasets["Schools"]

outside_schools = schools.loc[
    [not value for value in school_inside]
]

print("\nSchools outside Penang:")

if len(outside_schools) > 0:

    print(
        outside_schools[
            ["name", "geometry"]
        ].to_string(index=False)
    )

else:

    print("None")