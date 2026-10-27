from pathlib import Path
import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.mask import mask
from scipy.spatial import cKDTree


ROOT = Path(r"D:\fyp")
GHS_RASTER = Path(
    r"C:\Users\User\Downloads\GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R9_C29"
    r"\GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R9_C29.tif"
)
BOUNDARY = ROOT / "data/raw/boundary/penang_mukim_boundaries.geojson"
NETWORK = ROOT / "data/processed/walking_network_candidate.graphml"
STOPS = ROOT / "data/processed/stops_penang.csv"
STOP_NODES = ROOT / "data/processed/stops_with_nearest_node_candidate.csv"
OUTPUT = ROOT / "data/processed/ghs_coverage_dtl_mukim15.csv"

CENSUS_TOTAL = 13
WALK_LIMIT_M = 400


def haversine_m(lon1, lat1, lon2, lat2):
    """Calculate straight-line distance between two coordinates in metres."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 6_371_000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# Select DTL MUKIM 15 from the Penang boundary file.
areas = gpd.read_file(BOUNDARY)
area = areas[
    (areas["KOD_DAERAH"].astype(str).str.zfill(2) == "04")
    & (areas["NAM"].astype(str).str.strip() == "15")
]
if len(area) != 1:
    raise ValueError(f"Expected one DTL MUKIM 15 boundary; found {len(area)}")

# Clip the GHS raster to the mukim and get positive-population cell centres.
with rasterio.open(GHS_RASTER) as raster:
    clipped, transform = mask(
        raster,
        [area.to_crs(raster.crs).geometry.iloc[0]],
        crop=True,
        filled=False,
    )

    band = clipped[0]
    valid = (
        ~np.ma.getmaskarray(band)
        & np.isfinite(band.data)
        & (band.data > 0)
    )
    rows, cols = np.where(valid)
    raw_population = band.data[valid].astype(float)

    x, y = rasterio.transform.xy(transform, rows, cols, offset="center")
    to_wgs84 = Transformer.from_crs(
        raster.crs, "EPSG:4326", always_xy=True
    )
    cell_lon, cell_lat = to_wgs84.transform(
        np.asarray(x), np.asarray(y)
    )

if len(raw_population) == 0 or raw_population.sum() <= 0:
    raise ValueError("No positive GHS population cells found in the mukim.")

# Use GHS values as weights, scaled to the DOSM census total.
estimated_population = raw_population * (
    CENSUS_TOTAL / raw_population.sum()
)

# Load the existing walking network and make walking links two-way.
graph = nx.read_graphml(NETWORK).to_undirected()
for _, _, data in graph.edges(data=True):
    data["length"] = float(data["length"])

node_ids = list(graph.nodes)
node_xy = np.array(
    [
        [float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])]
        for node in node_ids
    ]
)

mean_latitude = float(node_xy[:, 1].mean())
node_tree = cKDTree(
    np.column_stack(
        (
            node_xy[:, 0] * math.cos(math.radians(mean_latitude)),
            node_xy[:, 1],
        )
    )
)

# Connect a temporary source to every bus stop. One network search then
# finds the walking distance from every node to its nearest stop.
stops = pd.read_csv(STOPS, dtype={"stop_id": str})
stop_nodes = pd.read_csv(
    STOP_NODES, dtype={"stop_id": str, "nearest_node": str}
)
stops = stops.merge(
    stop_nodes[["stop_id", "nearest_node"]],
    on="stop_id",
    how="left",
    validate="one_to_one",
)

if stops["nearest_node"].isna().any():
    raise ValueError("Some bus stops have no mapped walking-network node.")

source = "__temporary_all_bus_stops__"
graph.add_node(source)

for _, stop in stops.iterrows():
    node = stop["nearest_node"]
    stop_offset = haversine_m(
        float(stop["stop_lon"]),
        float(stop["stop_lat"]),
        float(graph.nodes[node]["x"]),
        float(graph.nodes[node]["y"]),
    )
    graph.add_edge(source, node, length=float(stop_offset))

distance_to_stop = nx.single_source_dijkstra_path_length(
    graph, source, weight="length"
)

# Snap each population cell centre to the nearest network node.
query_xy = np.column_stack(
    (
        cell_lon * math.cos(math.radians(mean_latitude)),
        cell_lat,
    )
)
_, nearest_indices = node_tree.query(query_xy)
nearest_nodes = [node_ids[i] for i in nearest_indices]
nearest_xy = node_xy[nearest_indices]

cell_to_node = haversine_m(
    cell_lon,
    cell_lat,
    nearest_xy[:, 0],
    nearest_xy[:, 1],
)
walk_distance = np.array(
    [distance_to_stop.get(node, np.inf) for node in nearest_nodes]
) + cell_to_node

# Save cell-level details so the estimate can be reviewed.
results = pd.DataFrame(
    {
        "cell_longitude": cell_lon,
        "cell_latitude": cell_lat,
        "ghs_population_weight": raw_population,
        "estimated_population_scaled_to_census_13": estimated_population,
        "walking_distance_to_nearest_stop_m": walk_distance,
        "within_400m": walk_distance <= WALK_LIMIT_M,
    }
)
results.to_csv(OUTPUT, index=False)

covered = float(estimated_population[walk_distance <= WALK_LIMIT_M].sum())
outside = CENSUS_TOTAL - covered

print(f"Positive GHS cells: {len(raw_population)}")
print(f"Raw GHS population sum: {raw_population.sum():.3f}")
print(f"Walking cells with a route: {np.isfinite(walk_distance).sum()}")
print(f"Estimated population within 400 m: {covered:.2f}")
print(f"Estimated population beyond 400 m: {outside:.2f}")
print(f"Saved cell details to: {OUTPUT}")