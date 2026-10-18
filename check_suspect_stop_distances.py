from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

processed = Path(r"D:\fyp\data\processed")
graph = nx.read_graphml(
    processed / "walking_network_candidate.graphml",
    force_multigraph=True,
)

components = sorted(
    nx.connected_components(nx.Graph(graph)),
    key=len,
    reverse=True,
)
main_nodes = list(set().union(*components[:2]))

# Store the large-section nodes' coordinates.
main_coords = np.array([
    [float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"])]
    for node in main_nodes
])
main_lats = np.radians(main_coords[:, 0])
main_lons = np.radians(main_coords[:, 1])

stops = pd.read_csv(
    processed / "stops_with_nearest_node_candidate.csv",
    dtype={"stop_id": str, "nearest_node": str},
)

suspect_ids = {
    "12001979",  # All Season Place
    "12002322",  # Penang Sentral
    "12002108",  # Jambatan Pulau Pinang
    "12001667",  # Penang Swimming Club
    "12000868",  # SMK Convent Green Lane
    "12001984",  # Jabatan Laut Wilayah Utara
}

print("Straight-line distance to nearest node in the two largest sections:")
print()

for stop in stops[stops["stop_id"].isin(suspect_ids)].itertuples():
    lat = np.radians(float(stop.stop_lat))
    lon = np.radians(float(stop.stop_lon))

    dlat = main_lats - lat
    dlon = main_lons - lon

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(main_lats) * np.sin(dlon / 2) ** 2
    )
    distances = 6_371_000 * 2 * np.arcsin(np.sqrt(a))
    nearest_index = int(np.argmin(distances))

    nearest_lat, nearest_lon = main_coords[nearest_index]

    print(
        f"{stop.stop_name} ({stop.stop_id}): "
        f"{distances[nearest_index]:.1f} m; "
        f"nearest large-section node at "
        f"{nearest_lat:.6f}, {nearest_lon:.6f}"
    )