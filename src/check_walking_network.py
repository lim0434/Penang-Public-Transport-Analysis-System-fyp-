
import os
import pandas as pd
import osmnx as ox

# ==========================================
# 1. DEFINE FILE PATHS
# ==========================================

graph_path = r"D:\fyp\data\processed\walking_network.graphml"
stops_path = r"D:\fyp\data\processed\stops_with_nearest_node.csv"


# ==========================================
# 2. CHECK FILES
# ==========================================

print("===== FILE CHECK =====")

print("Network file exists:", os.path.exists(graph_path))
print("Stops file exists:", os.path.exists(stops_path))

if not os.path.exists(graph_path) or not os.path.exists(stops_path):
    raise FileNotFoundError(
        "A required file is missing. Please check the file paths."
    )


# ==========================================
# 3. CHECK BUS STOP MAPPING
# ==========================================

stops = pd.read_csv(stops_path)

print("\n===== BUS STOP MAPPING =====")

print("Total bus stops:", len(stops))
print("Missing nearest nodes:", stops["nearest_node"].isna().sum())
print("Unique nearest nodes:", stops["nearest_node"].nunique())

print("\nSample mapping:")

print(
    stops[
        ["stop_id", "stop_name", "nearest_node"]
    ].head()
)


# ==========================================
# 4. LOAD WALKING NETWORK
# ==========================================

network = ox.io.load_graphml(graph_path)

print("\n===== WALKING NETWORK =====")

print("Network nodes:", len(network.nodes))
print("Network edges:", len(network.edges))
print("Network CRS:", network.graph["crs"])


# ==========================================
# 5. CHECK EDGE LENGTHS
# ==========================================

edge_lengths = [
    data.get("length")
    for _, _, _, data in network.edges(keys=True, data=True)
]

edge_lengths = pd.Series(edge_lengths, dtype="float64")

print("\n===== EDGE LENGTH VALIDATION =====")

print("Total edge records:", len(edge_lengths))
print("Missing edge lengths:", edge_lengths.isna().sum())
print("Zero or negative lengths:", (edge_lengths <= 0).sum())

if edge_lengths.notna().any():
    print("Shortest edge (metres):", edge_lengths.min())
    print("Longest edge (metres):", edge_lengths.max())


# ==========================================
# 6. FINAL CHECK
# ==========================================

print("\n===== VALIDATION SUMMARY =====")

if (
    len(stops) == 1930
    and stops["nearest_node"].notna().all()
    and edge_lengths.notna().all()
    and (edge_lengths > 0).all()
):
    print("Basic validation passed.")
else:
    print("Please review the validation results above.")
