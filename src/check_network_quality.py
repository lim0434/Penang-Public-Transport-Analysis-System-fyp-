
import osmnx as ox
import networkx as nx
import pandas as pd

# ==========================================
# 1. LOAD WALKING NETWORK
# ==========================================

graph_path = r"D:\fyp\data\processed\walking_network.graphml"

print("Loading walking network...")

network = ox.io.load_graphml(graph_path)

print("Nodes:", len(network.nodes))
print("Edges:", len(network.edges))


# ==========================================
# 2. CHECK LONG EDGES
# ==========================================

print("\n===== LONG EDGE CHECK =====")

edge_records = []

for u, v, key, data in network.edges(keys=True, data=True):
    edge_records.append({
        "from_node": u,
        "to_node": v,
        "length_m": float(data["length"])
    })

edges_df = pd.DataFrame(edge_records)

print("Total edges:", len(edges_df))
print("Edges longer than 400 m:", (edges_df["length_m"] > 400).sum())
print("Edges longer than 1,000 m:", (edges_df["length_m"] > 1000).sum())

print("\nLongest 10 edges:")

print(
    edges_df.nlargest(10, "length_m").to_string(index=False)
)


# ==========================================
# 3. CHECK NETWORK CONNECTIVITY
# ==========================================

print("\n===== NETWORK CONNECTIVITY CHECK =====")

# Treat the walking network as undirected for this diagnostic
undirected_network = nx.Graph(network)

components = list(nx.connected_components(undirected_network))

print("Connected components:", len(components))

component_sizes = sorted(
    [len(component) for component in components],
    reverse=True
)

print("\nLargest component sizes:")

for i, size in enumerate(component_sizes[:10], start=1):
    print(f"Component {i}: {size:,} nodes")

largest_size = component_sizes[0]

print("\nLargest component percentage:",
      round(largest_size / len(network.nodes) * 100, 2), "%")


# ==========================================
# 4. FINAL SUMMARY
# ==========================================

print("\n===== QUALITY CHECK SUMMARY =====")

print("Long edges need review before accessibility analysis.")
print("Connectivity results help identify separate network areas.")
print("No files have been changed.")

print("\n===== LONG EDGE LOCATION CHECK =====")

for _, row in edges_df.nlargest(10, "length_m").iterrows():
    u = row["from_node"]
    v = row["to_node"]

    u_data = network.nodes[int(u)]
    v_data = network.nodes[int(v)]

    print(f"\nLength: {row['length_m']:.2f} metres")
    print(f"Start: ({u_data['y']}, {u_data['x']})")
    print(f"End:   ({v_data['y']}, {v_data['x']})")

print("\n===== LONG EDGE DETAILS =====")

for u, v, key, data in sorted(
    network.edges(keys=True, data=True),
    key=lambda item: float(item[3].get("length", 0)),
    reverse=True
)[:10]:

    print("\n------------------------------")
    print("From node:", u)
    print("To node:", v)
    print("Length (m):", data.get("length"))
    print("Highway type:", data.get("highway"))
    print("Name:", data.get("name"))
    print("Geometry available:", "geometry" in data)

    if "geometry" in data:
        print("Geometry type:", data["geometry"].geom_type)

print("\n===== CONNECTED COMPONENT LOCATIONS =====")

components = sorted(
    nx.connected_components(nx.Graph(network)),
    key=len,
    reverse=True
)

for i, component in enumerate(components, start=1):
    component_nodes = [network.nodes[node] for node in component]

    latitudes = [node["y"] for node in component_nodes]
    longitudes = [node["x"] for node in component_nodes]

    print(f"\nComponent {i}")
    print("Nodes:", len(component))
    print(
        "Latitude range:",
        round(min(latitudes), 6),
        "to",
        round(max(latitudes), 6)
    )
    print(
        "Longitude range:",
        round(min(longitudes), 6),
        "to",
        round(max(longitudes), 6)
    )

print("\n===== DISTANCE BETWEEN COMPONENTS =====")

import numpy as np
from scipy.spatial import cKDTree

# Sort components from largest to smallest
components = sorted(
    nx.connected_components(nx.Graph(network)),
    key=len,
    reverse=True
)

# Approximate coordinates in metres.
# Longitude is scaled for Penang's latitude.
mean_lat = 5.4
metres_per_degree_lat = 111320
metres_per_degree_lon = 111320 * np.cos(np.radians(mean_lat))

component_coordinates = []

for component in components:
    nodes = list(component)

    lon = np.array([network.nodes[n]["x"] for n in nodes])
    lat = np.array([network.nodes[n]["y"] for n in nodes])

    x = lon * metres_per_degree_lon
    y = lat * metres_per_degree_lat

    component_coordinates.append(
        np.column_stack((x, y))
    )

# Find the nearest points between each pair of components
for i in range(len(components)):
    for j in range(i + 1, len(components)):
        tree = cKDTree(component_coordinates[j])
        distances, _ = tree.query(component_coordinates[i], k=1)

        print(
            f"Component {i + 1} to {j + 1}: "
            f"{distances.min():.2f} metres (approx.)"
        )
