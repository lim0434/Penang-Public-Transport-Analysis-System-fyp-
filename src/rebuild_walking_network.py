"""Build a candidate Penang walking network from overlapping OSM sections.

This script does not replace the current network or stop mapping. It writes
separate candidate files for validation.
"""

from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd


PROJECT_ROOT = Path(r"D:\fyp")
STOPS_PATH = PROJECT_ROOT / "data" / "raw" / "gtfs" / "stops.txt"
OUTPUT_NETWORK = PROJECT_ROOT / "data" / "processed" / "walking_network_candidate.graphml"
OUTPUT_STOPS = PROJECT_ROOT / "data" / "processed" / "stops_with_nearest_node_candidate.csv"

# About 1.1 km at Penang's latitude. Each download extends this far past
# its section edge so roads crossing a section boundary appear in both graphs.
OVERLAP_DEGREES = 0.01


def main():
    if not STOPS_PATH.exists():
        raise FileNotFoundError(f"GTFS stops file not found: {STOPS_PATH}")

    stops = pd.read_csv(STOPS_PATH, low_memory=False)
    required_columns = {"stop_id", "stop_lat", "stop_lon"}
    missing_columns = required_columns.difference(stops.columns)
    if missing_columns:
        raise ValueError(f"GTFS stops file is missing columns: {sorted(missing_columns)}")

    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
    if stops.empty:
        raise ValueError("No GTFS stops have valid latitude and longitude values.")

    north = float(stops["stop_lat"].max()) + OVERLAP_DEGREES
    south = float(stops["stop_lat"].min()) - OVERLAP_DEGREES
    east = float(stops["stop_lon"].max()) + OVERLAP_DEGREES
    west = float(stops["stop_lon"].min()) - OVERLAP_DEGREES
    mid_lat = (north + south) / 2
    mid_lon = (east + west) / 2

    # Expand each quadrant beyond its shared centre line. This creates a
    # deliberate overlap while keeping the outer extent unchanged.
    bboxes = [
        (west, south, mid_lon + OVERLAP_DEGREES, mid_lat + OVERLAP_DEGREES),
        (mid_lon - OVERLAP_DEGREES, south, east, mid_lat + OVERLAP_DEGREES),
        (west, mid_lat - OVERLAP_DEGREES, mid_lon + OVERLAP_DEGREES, north),
        (mid_lon - OVERLAP_DEGREES, mid_lat - OVERLAP_DEGREES, east, north),
    ]

    print("GTFS stops with coordinates:", len(stops))
    print("Downloading four overlapping walking-network sections...")
    graphs = []
    for part_number, bbox in enumerate(bboxes, start=1):
        print(f"\nDownloading section {part_number}/4: {bbox}")
        graph = ox.graph_from_bbox(
            bbox=bbox,
            network_type="walk",
            retain_all=True,
            truncate_by_edge=True,
        )
        if graph.number_of_nodes() == 0:
            raise RuntimeError(f"Downloaded section {part_number} contains no nodes.")
        print(f"Section {part_number}: {graph.number_of_nodes():,} nodes, "
              f"{graph.number_of_edges():,} edges")
        graphs.append(graph)

    # OSM node IDs are shared across overlapping requests, so composition
    # joins matching nodes and combines the downloaded graph sections.
    network = nx.compose_all(graphs)
    if network.number_of_nodes() == 0 or network.number_of_edges() == 0:
        raise RuntimeError("The merged candidate network is empty.")

    print("\nCandidate network merged.")
    print("Nodes:", f"{network.number_of_nodes():,}")
    print("Edges:", f"{network.number_of_edges():,}")

    node_ids = ox.distance.nearest_nodes(
        network,
        X=stops["stop_lon"].astype(float).to_numpy(),
        Y=stops["stop_lat"].astype(float).to_numpy(),
    )
    stops["nearest_node"] = node_ids
    missing_matches = int(stops["nearest_node"].isna().sum())
    print("Stops without a nearest node:", missing_matches)
    print("Unique matched nodes:", stops["nearest_node"].nunique())

    if missing_matches:
        raise RuntimeError("Some stops were not matched. Candidate files were not saved.")

    OUTPUT_NETWORK.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(network, filepath=OUTPUT_NETWORK)
    stops.to_csv(OUTPUT_STOPS, index=False)

    print("\nCandidate files saved:")
    print(OUTPUT_NETWORK)
    print(OUTPUT_STOPS)
    print("\nThe current walking_network.graphml and stops_with_nearest_node.csv were not changed.")
    print("Run the network quality checks on these candidate files before adopting them.")


if __name__ == "__main__":
    main()
