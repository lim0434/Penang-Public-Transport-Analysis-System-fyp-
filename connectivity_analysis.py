from heapq import heappop, heappush
from itertools import count
from pathlib import Path

import osmnx as ox
import pandas as pd
from pyproj import Geod
from shapely import wkt


ROOT = Path(r"D:\fyp")
DATA = ROOT / "data" / "processed"

NETWORK_FILE = DATA / "walking_network_candidate.graphml"
STOPS_FILE = DATA / "stops_penang.csv"
OUTPUT_FILE = DATA / "connectivity_by_destination.csv"

THRESHOLD_M = 400
SNAP_REVIEW_M = 100
DESTINATION_FILES = {
    "School": "schools_penang.csv",
    "Hospital": "hospitals_penang.csv",
    "Clinic": "clinics_penang.csv",
    "Mall": "malls_penang.csv",
}


def snap_gaps(graph, nodes, longitudes, latitudes, geod):
    node_x = [graph.nodes[int(node)]["x"] for node in nodes]
    node_y = [graph.nodes[int(node)]["y"] for node in nodes]
    _, _, gaps = geod.inv(longitudes, latitudes, node_x, node_y)
    return [float(gap) for gap in gaps]


def distances_to_nearest_stop(reverse_graph, stop_nodes):
    """Run one multi-source Dijkstra search, keeping the nearest stop node."""
    distances = {}
    source_for_node = {}
    queue = []
    sequence = count()

    for node in stop_nodes:
        node = int(node)
        if node not in distances:
            distances[node] = 0.0
            source_for_node[node] = node
            heappush(queue, (0.0, next(sequence), node))

    while queue:
        current_distance, _, current_node = heappop(queue)
        if current_distance != distances[current_node]:
            continue

        for _, neighbour, edge_data in reverse_graph.out_edges(
            current_node, data=True
        ):
            edge_length = float(edge_data["length"])
            new_distance = current_distance + edge_length

            if new_distance < distances.get(neighbour, float("inf")):
                distances[neighbour] = new_distance
                source_for_node[neighbour] = source_for_node[current_node]
                heappush(
                    queue,
                    (new_distance, next(sequence), neighbour),
                )

    return distances, source_for_node


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. Move or rename it before running "
            "this script so the existing results are not overwritten."
        )

    for path in [NETWORK_FILE, STOPS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    graph = ox.load_graphml(NETWORK_FILE)
    if not graph.is_directed():
        reverse_graph = graph
    else:
        reverse_graph = graph.reverse(copy=False)

    stops = pd.read_csv(STOPS_FILE)
    stop_nodes = ox.distance.nearest_nodes(
        graph,
        X=stops["stop_lon"].astype(float).tolist(),
        Y=stops["stop_lat"].astype(float).tolist(),
    )
    geod = Geod(ellps="WGS84")
    stop_gaps = snap_gaps(
        graph,
        stop_nodes,
        stops["stop_lon"].astype(float).tolist(),
        stops["stop_lat"].astype(float).tolist(),
        geod,
    )

    # If several stops share a graph node, keep the closest stop point.
    stop_at_node = {}
    for i, node in enumerate(stop_nodes):
        node = int(node)
        gap = stop_gaps[i]
        if node not in stop_at_node or gap < stop_at_node[node]["snap_gap_m"]:
            stop_at_node[node] = {
                "stop_id": stops.iloc[i]["stop_id"],
                "stop_name": stops.iloc[i]["stop_name"],
                "snap_gap_m": gap,
            }

    distance_by_node, source_by_node = distances_to_nearest_stop(
        reverse_graph, stop_at_node.keys()
    )

    results = []
    for destination_type, filename in DESTINATION_FILES.items():
        destination_file = DATA / filename
        if not destination_file.exists():
            raise FileNotFoundError(f"Required file not found: {destination_file}")

        destinations = pd.read_csv(destination_file)
        points = [
            wkt.loads(value).representative_point()
            for value in destinations["geometry"]
        ]
        longitudes = [point.x for point in points]
        latitudes = [point.y for point in points]
        destination_nodes = ox.distance.nearest_nodes(
            graph, X=longitudes, Y=latitudes
        )
        destination_gaps = snap_gaps(
            graph, destination_nodes, longitudes, latitudes, geod
        )

        for i, (_, row) in enumerate(destinations.iterrows()):
            node = int(destination_nodes[i])
            route_distance = distance_by_node.get(node)
            source_node = source_by_node.get(node)
            destination_name = row.get("name")

            if pd.isna(destination_name) or not str(destination_name).strip():
                destination_name = f"{destination_type} {i + 1:03d}"

            if route_distance is None:
                nearest_stop_id = None
                nearest_stop_name = None
                stop_gap = None
                status = "No route in candidate network - review"
            else:
                stop = stop_at_node[source_node]
                nearest_stop_id = stop["stop_id"]
                nearest_stop_name = stop["stop_name"]
                stop_gap = round(stop["snap_gap_m"], 1)
                status = (
                    "Within 400 m"
                    if route_distance <= THRESHOLD_M
                    else "Beyond 400 m"
                )

            results.append(
                {
                    "destination_id": (
                        f"{destination_type.lower()}_{i + 1:04d}"
                    ),
                    "destination_type": destination_type,
                    "destination_name": str(destination_name),
                    "destination_geometry": points[i].geom_type,
                    "nearest_stop_id": nearest_stop_id,
                    "nearest_stop_name": nearest_stop_name,
                    "network_distance_m": (
                        round(float(route_distance), 1)
                        if route_distance is not None
                        else None
                    ),
                    "destination_snap_gap_m": round(destination_gaps[i], 1),
                    "stop_snap_gap_m": stop_gap,
                    "status": status,
                    "snap_needs_review": (
                        destination_gaps[i] > SNAP_REVIEW_M
                        or (stop_gap is not None and stop_gap > SNAP_REVIEW_M)
                    ),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(results_df)} destinations to: {OUTPUT_FILE}")
    print("\nStatus counts:")
    print(results_df["status"].value_counts().to_string())
    print(
        "\nDestinations with a snap gap over "
        f"{SNAP_REVIEW_M} m: "
        f"{int(results_df['snap_needs_review'].sum())}"
    )
    print(
        "Network distance is measured between snapped road-network nodes. "
        "Snap gaps are listed separately."
    )


if __name__ == "__main__":
    main()