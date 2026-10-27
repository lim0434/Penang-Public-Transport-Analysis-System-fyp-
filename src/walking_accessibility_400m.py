import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkt
import geopandas as gpd
import rasterio
from rasterio.mask import mask as raster_mask

ROOT = Path(r"D:\fyp")
DATA = ROOT / "data" / "processed"
CUTOFF_M = 400

DESTINATION_FILES = {
    "schools": "schools_penang.csv",
    "hospitals": "hospitals_penang.csv",
    "clinics": "clinics_penang.csv",
    "malls": "malls_penang.csv",
}


def distance_metres(lon1, lat1, lon2, lat2):
    """Return straight-line distance between two coordinates in metres."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return 6_371_000 * 2 * math.asin(math.sqrt(a))


print("Loading candidate walking network...")
graph = nx.read_graphml(DATA / "walking_network_candidate.graphml")

# Treat walking routes as available in both directions.
walk_graph = graph.to_undirected()
# Convert GraphML edge lengths from text to numbers.
if walk_graph.is_multigraph():
    edges = walk_graph.edges(keys=True, data=True)
else:
    edges = walk_graph.edges(data=True)

for edge in edges:
    data = edge[-1]
    if "length" not in data:
        raise ValueError("A walking-network edge has no length value.")
    data["length"] = float(data["length"])

node_ids = list(walk_graph.nodes)
node_xy = np.array([
    [float(walk_graph.nodes[node]["x"]), float(walk_graph.nodes[node]["y"])]
    for node in node_ids
])

mean_latitude = float(node_xy[:, 1].mean())
scaled_nodes = node_xy.copy()
scaled_nodes[:, 0] *= math.cos(math.radians(mean_latitude))
node_tree = cKDTree(scaled_nodes)


def nearest_node(lon, lat):
    query = [
        lon * math.cos(math.radians(mean_latitude)),
        lat,
    ]
    _, index = node_tree.query(query)

    node = node_ids[index]
    node_lon, node_lat = node_xy[index]
    offset = distance_metres(lon, lat, node_lon, node_lat)

    return node, offset


# Snap each destination to its nearest network node.
destinations = {category: [] for category in DESTINATION_FILES}
snap_records = []

for category, filename in DESTINATION_FILES.items():
    data = pd.read_csv(DATA / filename)

    for _, row in data.iterrows():
        geometry = wkt.loads(row["geometry"])

        # Use the point itself, or a representative point inside a polygon.
        point = (
            geometry
            if geometry.geom_type == "Point"
            else geometry.representative_point()
        )

        node, offset = nearest_node(point.x, point.y)

        destinations[category].append((node, offset))
        snap_records.append({
            "category": category,
            "name": row.get("name", ""),
            "snap_distance_m": offset,
        })

snap_data = pd.DataFrame(snap_records)

print("\nDestination snap-distance check:")
for category in DESTINATION_FILES:
    distances = snap_data.loc[
        snap_data["category"] == category, "snap_distance_m"
    ]
    print(
        f"{category}: {len(distances)} destinations; "
        f"median {distances.median():.1f} m; "
        f"maximum {distances.max():.1f} m; "
        f"over 100 m: {(distances > 100).sum()}"
    )

far_destinations = snap_data[snap_data["snap_distance_m"] > 100]
if not far_destinations.empty:
    print("\nDestinations more than 100 m from the network:")
    print(
        far_destinations[
            ["category", "name", "snap_distance_m"]
        ].to_string(index=False)
    )

# Match filtered Penang stops to their candidate-network nodes.
stops = pd.read_csv(DATA / "stops_penang.csv", dtype={"stop_id": str})
mapping = pd.read_csv(
    DATA / "stops_with_nearest_node_candidate.csv",
    dtype={"stop_id": str, "nearest_node": str},
)

stops = stops.merge(
    mapping[["stop_id", "nearest_node"]],
    on="stop_id",
    how="left",
    validate="one_to_one",
)

if stops["nearest_node"].isna().any():
    raise ValueError("Some Penang stops have no mapped network node.")

missing_nodes = set(stops["nearest_node"]) - set(walk_graph.nodes)
if missing_nodes:
    raise ValueError(
        f"{len(missing_nodes)} mapped stop nodes are missing from the network."
    )

# Run a 400 m walking-distance search from each bus stop.
results = []

for _, stop in stops.iterrows():
    source = stop["nearest_node"]
    stop_offset = distance_metres(
        float(stop["stop_lon"]),
        float(stop["stop_lat"]),
        float(walk_graph.nodes[source]["x"]),
        float(walk_graph.nodes[source]["y"]),
    )

    remaining_distance = CUTOFF_M - stop_offset

    if remaining_distance >= 0:
        reachable_nodes = nx.single_source_dijkstra_path_length(
            walk_graph,
            source,
            cutoff=remaining_distance,
            weight="length",
        )
    else:
        reachable_nodes = {}

    result = {
        "stop_id": stop["stop_id"],
        "stop_name": stop["stop_name"],
        "stop_lat": stop["stop_lat"],
        "stop_lon": stop["stop_lon"],
        "stop_snap_distance_m": round(stop_offset, 1),
    }

    for category, places in destinations.items():
        count = 0

        for node, destination_offset in places:
            network_distance = reachable_nodes.get(node)

            if (
                network_distance is not None
                and network_distance + destination_offset <= remaining_distance
            ):
                count += 1

        result[f"{category}_within_400m"] = count

    result["all_destinations_within_400m"] = sum(
        result[f"{category}_within_400m"]
        for category in DESTINATION_FILES
    )
    results.append(result)

output = DATA / "accessibility_400m_by_stop.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output, index=False)

print(f"\nSaved: {output}")
print(f"Stops analysed: {len(results_df):,}")
print(
    "Stops with no listed destinations within 400 m: "
    f"{(results_df['all_destinations_within_400m'] == 0).sum():,}"
)
print("\nAverage destinations within 400 m per stop:")
print(
    results_df[
        [f"{category}_within_400m" for category in DESTINATION_FILES]
    ].mean().round(1).to_string()
)
# Estimate population access to bus stops by mukim.
print("\nCalculating population access by mukim...")

mukim_file = ROOT / "data" / "raw" / "boundary" / "penang_mukim_boundaries.geojson"
population_file = ROOT / "data" / "raw" / "population" / "mukim_stat.csv"
raster_file = ROOT / "data" / "raw" / "population" / "mys_ppp_2020_constrained.tif"
population_output = DATA / "population_accessibility_400m_by_mukim.csv"

district_prefix = {
    "01": "SPT",
    "02": "SPU",
    "03": "SPS",
    "04": "DTL",
    "05": "DBD",
}

def make_mukim_label(row):
    district = district_prefix[str(row["KOD_DAERAH"]).zfill(2)]
    name = str(row["NAM"]).strip()

    if name.isdigit() or (len(name) == 1 and name.isalpha()):
        return f"{district} MUKIM {name}".upper()

    name = name.removeprefix("Bandar ").strip().upper()
    aliases = {
        "SUNGEI BAKAP": "SUNGAI BAKAP",
        "BATU FERINGGI": "BATU FERRINGGI",
        "BUTTERWORTH": "BUTTERWORT",
    }
    return f"{district} {aliases.get(name, name)}"

mukims = gpd.read_file(mukim_file)
population_table = pd.read_csv(population_file)

mukims["match_name"] = mukims.apply(make_mukim_label, axis=1)
population_table["match_name"] = (
    population_table["NAMA PENTADBIRAN"].astype(str).str.strip().str.upper()
)
population_table["csv_population"] = pd.to_numeric(
    population_table["Sum of JUMLAH PENDUDUK"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce",
)

mukims = mukims.merge(
    population_table[["match_name", "csv_population"]],
    on="match_name",
    how="left",
    validate="one_to_one",
)

if mukims["csv_population"].isna().any() or len(mukims) != 97:
    raise ValueError("Mukim names did not all match the population CSV.")

# Add a temporary source connected to every bus stop.
# This lets one shortest-path run find the nearest stop from each network node.
virtual_source = "__all_bus_stops__"
if virtual_source in walk_graph:
    raise ValueError("Temporary bus-stop node already exists in the network.")

walk_graph.add_node(virtual_source)

for _, stop in stops.iterrows():
    node = stop["nearest_node"]
    offset = distance_metres(
        float(stop["stop_lon"]),
        float(stop["stop_lat"]),
        float(walk_graph.nodes[node]["x"]),
        float(walk_graph.nodes[node]["y"]),
    )
    walk_graph.add_edge(virtual_source, node, length=offset)

print("Finding walking distance from network nodes to the nearest bus stop...")
distance_to_stop = nx.single_source_dijkstra_path_length(
    walk_graph,
    virtual_source,
    weight="length",
)

node_ids_array = np.asarray(node_ids, dtype=object)
population_results = []

with rasterio.open(raster_file) as src:
    if mukims.crs is None:
        raise ValueError("Mukim boundary file has no CRS.")

    if mukims.crs != src.crs:
        mukims = mukims.to_crs(src.crs)

    for _, mukim in mukims.iterrows():
        clipped, transform = raster_mask(
            src,
            [mukim.geometry],
            crop=True,
            filled=False,
        )

        raster_values = clipped[0]
        valid = (
            ~np.ma.getmaskarray(raster_values)
            & np.isfinite(raster_values.data)
            & (raster_values.data > 0)
        )

        if not valid.any():
            if mukim["match_name"] == "DTL MUKIM 15":
                ghs_file = DATA / "ghs_coverage_dtl_mukim15.csv"
                ghs = pd.read_csv(ghs_file)

                cell_population = ghs[
                    "estimated_population_scaled_to_census_13"
                ].to_numpy(dtype=float)
                route_distances = ghs[
                    "walking_distance_to_nearest_stop_m"
                ].to_numpy(dtype=float)

                total_population = cell_population.sum()
                routable = np.isfinite(route_distances)
                covered = route_distances <= CUTOFF_M

                covered_population = cell_population[covered].sum()
                coverage_percent = (
                    100 * covered_population / total_population
                )

                if cell_population[routable].sum() > 0:
                    mean_distance = np.average(
                        route_distances[routable],
                        weights=cell_population[routable],
                    )
                else:
                    mean_distance = np.nan

                no_route_percent = (
                    100
                    * cell_population[~routable].sum()
                    / total_population
                )

                population_results.append({
                    "match_name": mukim["match_name"],
                    "raster_population": float(
                        ghs["ghs_population_weight"].sum()
                    ),
                    "population_within_400m": covered_population,
                    "population_coverage_percent": coverage_percent,
                    "estimated_csv_population_covered": (
                        mukim["csv_population"]
                        * coverage_percent / 100
                    ),
                    "mean_walk_distance_m_routable_population": mean_distance,
                    "population_without_network_route_percent": (
                        no_route_percent
                    ),
                    "population_source": (
                        "GHS-POP 2020; grid values scaled to census total"
                    ),
                })
                continue

            population_results.append({
                "match_name": mukim["match_name"],
                "raster_population": 0.0,
                "population_within_400m": 0.0,
                "population_coverage_percent": np.nan,
                "estimated_csv_population_covered": np.nan,
                "mean_walk_distance_m_routable_population": np.nan,
                "population_without_network_route_percent": np.nan,
            })
            continue

        rows, cols = np.where(valid)
        cell_population = raster_values.data[valid].astype(float)

        cell_lon, cell_lat = rasterio.transform.xy(
            transform,
            rows,
            cols,
            offset="center",
        )
        cell_lon = np.asarray(cell_lon)
        cell_lat = np.asarray(cell_lat)

        # Snap each populated raster cell to its nearest walking-network node.
        query_xy = np.column_stack((
            cell_lon * math.cos(math.radians(mean_latitude)),
            cell_lat,
        ))
        snap_offsets, nearest_indices = node_tree.query(query_xy)
        snapped_nodes = node_ids_array[nearest_indices]

        # Use straight-line distance from each cell centre to its snapped node.
        nearest_node_xy = node_xy[nearest_indices]
        lat1 = np.radians(cell_lat)
        lat2 = np.radians(nearest_node_xy[:, 1])
        dlat = lat2 - lat1
        dlon = np.radians(nearest_node_xy[:, 0] - cell_lon)

        haversine_a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        cell_offsets = (
            6_371_000
            * 2
            * np.arcsin(np.sqrt(np.clip(haversine_a, 0, 1)))
        )

        route_distances = np.array([
            distance_to_stop.get(node, np.inf)
            for node in snapped_nodes
        ]) + cell_offsets

        total_population = cell_population.sum()
        covered = route_distances <= CUTOFF_M
        routable = np.isfinite(route_distances)

        covered_population = cell_population[covered].sum()
        coverage_percent = 100 * covered_population / total_population

        if cell_population[routable].sum() > 0:
            mean_distance = np.average(
                route_distances[routable],
                weights=cell_population[routable],
            )
        else:
            mean_distance = np.nan

        no_route_percent = (
            100 * cell_population[~routable].sum() / total_population
        )

        population_results.append({
            "match_name": mukim["match_name"],
            "raster_population": total_population,
            "population_within_400m": covered_population,
            "population_coverage_percent": coverage_percent,
            "estimated_csv_population_covered": (
                mukim["csv_population"] * coverage_percent / 100
            ),
            "mean_walk_distance_m_routable_population": mean_distance,
            "population_without_network_route_percent": no_route_percent,
        })

population_results = pd.DataFrame(population_results)

population_results = population_results.merge(
    mukims[["match_name", "csv_population"]],
    on="match_name",
    validate="one_to_one",
)

population_results.to_csv(population_output, index=False)

print(f"Saved: {population_output}")
print(f"Mukims analysed: {len(population_results)}")
print(
    "Average population coverage across mukims: "
    f"{population_results['population_coverage_percent'].mean():.1f}%"
)
print(
    "Average raster population without a route to a bus stop: "
    f"{population_results['population_without_network_route_percent'].mean():.1f}%"
)