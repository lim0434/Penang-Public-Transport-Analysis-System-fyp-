import osmnx as ox
import pandas as pd
import networkx as nx

place = "Penang, Malaysia"


# Load GTFS bus stops
print("\nLoading bus stops...")

stops = pd.read_csv(
    r"D:\fyp\data\raw\gtfs\stops.txt"
)

print("Bus stops:", len(stops))
print(stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].head())


# Download walking network for the full study area
print("\nDownloading walking network...")

north = stops["stop_lat"].max() + 0.01
south = stops["stop_lat"].min() - 0.01
east = stops["stop_lon"].max() + 0.01
west = stops["stop_lon"].min() - 0.01

print("North:", north)
print("South:", south)
print("East:", east)
print("West:", west)

# Divide the study area into smaller areas
mid_lat = (north + south) / 2
mid_lon = (east + west) / 2

bboxes = [
    (west, south, mid_lon, mid_lat),   # Southwest
    (mid_lon, south, east, mid_lat),   # Southeast
    (west, mid_lat, mid_lon, north),   # Northwest
    (mid_lon, mid_lat, east, north)    # Northeast
]

graphs = []

for i, bbox in enumerate(bboxes, start=1):
    print(f"Downloading OSM walking network part {i}/4...")

    try:
        graph = ox.graph_from_bbox(
            bbox=bbox,
            network_type="walk"
        )

        graphs.append(graph)
        print(f"Part {i} loaded successfully.")

    except Exception as e:
        print(f"Part {i} failed: {e}")

# Combine all parts
if graphs:
    network = nx.compose_all(graphs)
    print("All OSM walking network parts combined.")
else:
    raise RuntimeError("No OSM walking network could be downloaded.")

print("OSM walking network loaded")
print("Nodes:", len(network.nodes))
print("Edges:", len(network.edges))


# Save walking network
network_path = r"D:\fyp\data\processed\walking_network.graphml"

ox.save_graphml(
    network,
    filepath=network_path
)

print("Walking network saved.")
print(f"Saved to: {network_path}")

# Find the nearest walking network node for each bus stop
bus_stop_nodes = ox.distance.nearest_nodes(
    network,
    X=stops["stop_lon"],
    Y=stops["stop_lat"]
)

stops["nearest_node"] = bus_stop_nodes

print("\nNearest walking nodes assigned.")
print("Bus stops:", len(stops))
print("Unique nearest nodes:", stops["nearest_node"].nunique())

# Download schools
print("\nDownloading schools...")

schools = ox.features_from_place(
    place,
    tags={"amenity": "school"}
)

print("Schools found:", len(schools))

print("\nSchool columns:")
print(schools.columns.tolist())

print("\nMissing values:")
print(schools[["name", "geometry"]].isnull().sum())

print("\nSchool types:")
print(schools["amenity"].value_counts(dropna=False))


# Download hospitals
print("\nDownloading hospitals...")

hospitals = ox.features_from_place(
    place,
    tags={"amenity": "hospital"}
)

print("Hospitals found:", len(hospitals))

print("\nHospital columns:")
print(hospitals.columns.tolist())

print("\nMissing values:")
print(hospitals[["name", "geometry"]].isnull().sum())

print("\nHospital types:")
print(hospitals["amenity"].value_counts(dropna=False))


# Download clinics
print("\nDownloading clinics...")

clinics = ox.features_from_place(
    place,
    tags={"amenity": "clinic"}
)

print("Clinics found:", len(clinics))

print("\nClinic columns:")
print(clinics.columns.tolist())

print("\nMissing values:")
print(clinics[["name", "geometry"]].isnull().sum())

print("\nClinic types:")
print(clinics["amenity"].value_counts(dropna=False))


# Download malls
print("\nDownloading malls...")

malls = ox.features_from_place(
    place,
    tags={"shop": "mall"}
)

print("Malls found:", len(malls))

print("\nMall columns:")
print(malls.columns.tolist())

print("\nMissing values:")
print(malls[["name", "geometry"]].isnull().sum())

print("\nMall types:")
print(malls["shop"].value_counts(dropna=False))

print(
    stops[
        ["stop_id", "stop_name", "stop_lat", "stop_lon", "nearest_node"]
    ].head()
)


# Check nearest node coordinates
print("\nChecking nearest node coordinates...")

for node in stops["nearest_node"].head(5):
    print(
        node,
        network.nodes[node]["y"],
        network.nodes[node]["x"]
    )


# Check network CRS
print("\nWalking network CRS:")
print(network.graph["crs"])


# Check walking network coordinate range
print("\nWalking network coordinate range:")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(network)

print("Latitude range:")
print(nodes_gdf["y"].min(), "to", nodes_gdf["y"].max())

print("Longitude range:")
print(nodes_gdf["x"].min(), "to", nodes_gdf["x"].max())


# Check bus stop coordinate range
print("\nBus stop coordinate range:")

print("Latitude:")
print(stops["stop_lat"].min(), "to", stops["stop_lat"].max())

print("Longitude:")
print(stops["stop_lon"].min(), "to", stops["stop_lon"].max())


# Save destination data
print("\nSaving destination data...")

schools.to_csv(
    r"D:\fyp\data\processed\schools.csv",
    index=False
)

hospitals.to_csv(
    r"D:\fyp\data\processed\hospitals.csv",
    index=False
)

clinics.to_csv(
    r"D:\fyp\data\processed\clinics.csv",
    index=False
)

malls.to_csv(
    r"D:\fyp\data\processed\malls.csv",
    index=False
)

print("Destination data saved.")