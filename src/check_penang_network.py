from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(r"D:\fyp")
PROCESSED = ROOT / "data" / "processed"

stops = pd.read_csv(PROCESSED / "stops_penang.csv", dtype={"stop_id": str})
mapping = pd.read_csv(
    PROCESSED / "stops_with_nearest_node_candidate.csv",
    dtype={"stop_id": str, "nearest_node": str},
)

# Keep the candidate mapping for Penang stops only.
penang_mapping = stops[["stop_id"]].merge(
    mapping[["stop_id", "nearest_node"]],
    on="stop_id",
    how="left",
)

print(f"Penang stops: {len(stops):,}")
print(f"Stops with a mapped node: {penang_mapping['nearest_node'].notna().sum():,}")
print(f"Stops without a mapped node: {penang_mapping['nearest_node'].isna().sum():,}")

print("Loading the candidate walking network...")
graph = nx.read_graphml(PROCESSED / "walking_network_candidate.graphml")
graph_nodes = set(graph.nodes)

penang_mapping["node_exists"] = penang_mapping["nearest_node"].isin(graph_nodes)
print(f"Mapped nodes found in network: {penang_mapping['node_exists'].sum():,}")
print(f"Mapped nodes missing from network: {(~penang_mapping['node_exists']).sum():,}")

if graph.is_directed():
    components = list(nx.weakly_connected_components(graph))
else:
    components = list(nx.connected_components(graph))

node_to_component = {
    node: component_id
    for component_id, component in enumerate(components)
    for node in component
}

penang_mapping["component"] = penang_mapping["nearest_node"].map(node_to_component)
component_sizes = {i: len(c) for i, c in enumerate(components)}
penang_mapping["component_size"] = penang_mapping["component"].map(component_sizes)

stop_counts = (
    penang_mapping.dropna(subset=["component"])
    .groupby(["component", "component_size"])
    .size()
    .reset_index(name="penang_stops")
    .sort_values("component_size", ascending=False)
)

print(f"Network components: {len(components):,}")
print("\nComponents containing Penang stops:")
print(stop_counts.to_string(index=False))

small = penang_mapping[
    penang_mapping["component_size"].fillna(0) < 100
]
print(f"\nPenang stops in components smaller than 100 nodes: {len(small):,}")
if len(small):
    print(small[["stop_id", "nearest_node", "component_size"]].to_string(index=False))