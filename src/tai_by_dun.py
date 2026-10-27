import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from shapely import contains_xy
from shapely.geometry import Point
from shapely.ops import nearest_points


ROOT = Path(r"D:\fyp")
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

DUN_BOUNDARY_FILE = RAW / "boundary" / "electoral_1_dun.geojson"
POPULATION_RASTER_FILE = (
    RAW / "population" / "mys_ppp_2020_constrained.tif"
)
DUN_POPULATION_FILE = RAW / "dosm" / "population_dun.csv"
DUN_INCOME_FILE = RAW / "dosm" / "hh_income_dun.csv"

STOPS_FILE = PROCESSED / "stops_penang.csv"
STOP_MAPPING_FILE = PROCESSED / "stops_with_nearest_node_candidate.csv"
STOP_ACCESS_FILE = PROCESSED / "accessibility_400m_by_stop.csv"
NETWORK_FILE = PROCESSED / "walking_network_candidate.graphml"

OUTPUT_FILE = PROCESSED / "tai_by_dun_socioeconomic.csv"

CUTOFF_M = 400
TARGET_STATE = "Pulau Pinang"
TARGET_YEAR = "2024-01-01"
MAX_DUN_GAP_M = 75

DESTINATION_COLUMNS = {
    "schools": "schools_within_400m",
    "hospitals": "hospitals_within_400m",
    "clinics": "clinics_within_400m",
    "malls": "malls_within_400m",
}


def distance_metres(lon1, lat1, lon2, lat2):
    """Calculate straight-line distance between two coordinates."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    return 6_371_000 * 2 * math.asin(math.sqrt(a))


def area_km2(geometry):
    """Estimate area using a local latitude-longitude conversion."""
    latitude = math.radians(geometry.centroid.y)

    def local_coordinates(x, y, z=None):
        return (
            np.asarray(x) * 111_320 * math.cos(latitude),
            np.asarray(y) * 110_574,
        )

    from shapely.ops import transform

    projected = transform(local_coordinates, geometry)
    return projected.area / 1_000_000


def minmax_score(values, reverse=False):
    """Scale values to 0–1. Reverse the score when lower is better."""
    scores = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()

    if not valid.any():
        return scores

    minimum = values[valid].min()
    maximum = values[valid].max()

    if maximum == minimum:
        scores.loc[valid] = 0.5
    elif reverse:
        scores.loc[valid] = (
            (maximum - values[valid]) / (maximum - minimum)
        )
    else:
        scores.loc[valid] = (
            (values[valid] - minimum) / (maximum - minimum)
        )

    return scores


print("Loading DUN boundaries...")
duns = gpd.read_file(DUN_BOUNDARY_FILE)

if duns.crs is None:
    raise ValueError("The DUN boundary file has no CRS.")

if str(duns.crs).upper() != "EPSG:4326":
    raise ValueError(
        f"Expected DUN boundaries in EPSG:4326, found {duns.crs}."
    )

duns = duns[
    duns["state"].str.casefold() == TARGET_STATE.casefold()
].copy()
duns["dun"] = duns["dun"].astype(str).str.strip()

if len(duns) != 40 or duns["dun"].duplicated().any():
    raise ValueError(
        f"Expected 40 unique Penang DUNs; found {len(duns)}."
    )

duns["area_km2"] = duns.geometry.apply(area_km2)
print(f"Penang DUNs loaded: {len(duns)}")


print("Loading stops and saved destination-access results...")
stops = pd.read_csv(STOPS_FILE, dtype={"stop_id": str})
mapping = pd.read_csv(
    STOP_MAPPING_FILE,
    dtype={"stop_id": str, "nearest_node": str},
)
stop_access = pd.read_csv(STOP_ACCESS_FILE, dtype={"stop_id": str})

stops = stops.merge(
    mapping[["stop_id", "nearest_node"]],
    on="stop_id",
    how="left",
    validate="one_to_one",
)

if stops["nearest_node"].isna().any():
    raise ValueError("Some bus stops have no mapped network node.")

stop_access_columns = [
    "stop_id",
    *DESTINATION_COLUMNS.values(),
]

stops = stops.merge(
    stop_access[stop_access_columns],
    on="stop_id",
    how="left",
    validate="one_to_one",
)

if stops[list(DESTINATION_COLUMNS.values())].isna().any().any():
    raise ValueError("Some stops have no destination-access results.")


print("Assigning bus stops to DUNs...")
dun_geometries = list(zip(duns["dun"], duns.geometry))
stop_dun_records = []

for _, stop in stops.iterrows():
    point = Point(float(stop["stop_lon"]), float(stop["stop_lat"]))

    matched_dun = next(
        (
            dun_name
            for dun_name, geometry in dun_geometries
            if geometry.covers(point)
        ),
        None,
    )

    if matched_dun is None:
        nearest_dun, nearest_geometry = min(
            dun_geometries,
            key=lambda item: item[1].distance(point),
        )

        nearest_point = nearest_points(point, nearest_geometry)[1]
        gap_m = distance_metres(
            point.x,
            point.y,
            nearest_point.x,
            nearest_point.y,
        )

        if gap_m > MAX_DUN_GAP_M:
            raise ValueError(
                f"Stop {stop['stop_id']} is {gap_m:.1f} m "
                "from the nearest DUN boundary."
            )

        matched_dun = nearest_dun
        print(
            f"Boundary fallback: stop {stop['stop_id']} "
            f"({stop.get('stop_name', '')}) assigned to {matched_dun}; "
            f"gap {gap_m:.1f} m"
        )

    stop_dun_records.append({
        "stop_id": stop["stop_id"],
        "dun": matched_dun,
    })

stop_duns = pd.DataFrame(stop_dun_records)

stops = stops.merge(
    stop_duns,
    on="stop_id",
    how="left",
    validate="one_to_one",
)

stop_summary = stops.groupby("dun").agg(
    bus_stops=("stop_id", "count"),
    schools_per_stop=("schools_within_400m", "mean"),
    hospitals_per_stop=("hospitals_within_400m", "mean"),
    clinics_per_stop=("clinics_within_400m", "mean"),
    malls_per_stop=("malls_within_400m", "mean"),
).reset_index()

tai = duns[["dun", "code_dun", "area_km2", "geometry"]].merge(
    stop_summary,
    on="dun",
    how="left",
    validate="one_to_one",
)

tai["bus_stops"] = tai["bus_stops"].fillna(0).astype(int)

destination_mean_columns = [
    f"{category}_per_stop"
    for category in DESTINATION_COLUMNS
]

for column in destination_mean_columns:
    tai[column] = tai[column].fillna(0)

tai["bus_stops_per_km2"] = tai["bus_stops"] / tai["area_km2"]


print("Loading walking network...")
graph = nx.read_graphml(NETWORK_FILE)
walk_graph = graph.to_undirected()

if walk_graph.is_multigraph():
    edge_iterator = walk_graph.edges(keys=True, data=True)
else:
    edge_iterator = walk_graph.edges(data=True)

for edge in edge_iterator:
    data = edge[-1]

    if "length" not in data:
        raise ValueError("A walking-network edge has no length.")

    data["length"] = float(data["length"])

node_ids = list(walk_graph.nodes)

node_xy = np.array([
    [
        float(walk_graph.nodes[node]["x"]),
        float(walk_graph.nodes[node]["y"]),
    ]
    for node in node_ids
])

mean_latitude = float(node_xy[:, 1].mean())
scaled_nodes = node_xy.copy()
scaled_nodes[:, 0] *= math.cos(math.radians(mean_latitude))

node_tree = cKDTree(scaled_nodes)
node_ids_array = np.asarray(node_ids, dtype=object)


# Add a temporary source linked to every bus stop.
virtual_source = "__all_bus_stops_for_dun_analysis__"

if virtual_source in walk_graph:
    raise ValueError("Temporary source node already exists.")

walk_graph.add_node(virtual_source)

for _, stop in stops.iterrows():
    node = stop["nearest_node"]

    if node not in walk_graph:
        raise ValueError(
            f"Mapped node for stop {stop['stop_id']} is missing."
        )

    offset = distance_metres(
        float(stop["stop_lon"]),
        float(stop["stop_lat"]),
        float(walk_graph.nodes[node]["x"]),
        float(walk_graph.nodes[node]["y"]),
    )

    walk_graph.add_edge(virtual_source, node, length=offset)


print("Finding walking distance from network nodes to bus stops...")
distance_to_stop = nx.single_source_dijkstra_path_length(
    walk_graph,
    virtual_source,
    weight="length",
)


print("Calculating 400 m population access for each DUN...")

# This trusted local TIFF has about 184.5 million pixels.
# Raise Pillow's default limit just above its known size.
Image.MAX_IMAGE_PIXELS = 190_000_000

population_records = []

with Image.open(POPULATION_RASTER_FILE) as raster:
    pixel_scale = raster.tag_v2.get(33550)
    tiepoint = raster.tag_v2.get(33922)

    if pixel_scale is None or tiepoint is None:
        raise ValueError(
            "The population TIFF has no georeferencing tags."
        )

    scale_x, scale_y = pixel_scale[:2]
    raster_col, raster_row = tiepoint[:2]
    origin_lon, origin_lat = tiepoint[3:5]

    for _, dun in tai.iterrows():
        min_lon, min_lat, max_lon, max_lat = dun.geometry.bounds

        col_start = max(
            0,
            int(math.floor(
                raster_col + (min_lon - origin_lon) / scale_x
            )),
        )
        col_stop = min(
            raster.width,
            int(math.ceil(
                raster_col + (max_lon - origin_lon) / scale_x
            )),
        )
        row_start = max(
            0,
            int(math.floor(
                raster_row + (origin_lat - max_lat) / scale_y
            )),
        )
        row_stop = min(
            raster.height,
            int(math.ceil(
                raster_row + (origin_lat - min_lat) / scale_y
            )),
        )

        if col_start >= col_stop or row_start >= row_stop:
            population_records.append({
                "dun": dun["dun"],
                "raster_population_2020": 0.0,
                "population_coverage_percent": np.nan,
                "mean_walk_distance_m_routable_population": np.nan,
                "population_without_network_route_percent": np.nan,
            })
            continue

        values = np.asarray(
            raster.crop((col_start, row_start, col_stop, row_stop)),
            dtype=float,
        )

        cols = np.arange(col_start, col_stop)
        rows = np.arange(row_start, row_stop)

        cell_lons = (
            origin_lon
            + (cols + 0.5 - raster_col) * scale_x
        )
        cell_lats = (
            origin_lat
            - (rows + 0.5 - raster_row) * scale_y
        )

        lon_grid, lat_grid = np.meshgrid(cell_lons, cell_lats)
        inside_dun = contains_xy(dun.geometry, lon_grid, lat_grid)

        valid = inside_dun & np.isfinite(values) & (values > 0)

        if not valid.any():
            population_records.append({
                "dun": dun["dun"],
                "raster_population_2020": 0.0,
                "population_coverage_percent": np.nan,
                "mean_walk_distance_m_routable_population": np.nan,
                "population_without_network_route_percent": np.nan,
            })
            continue

        cell_population = values[valid]
        cell_lon = lon_grid[valid]
        cell_lat = lat_grid[valid]

        query_xy = np.column_stack((
            cell_lon * math.cos(math.radians(mean_latitude)),
            cell_lat,
        ))

        _, nearest_indices = node_tree.query(query_xy)
        snapped_nodes = node_ids_array[nearest_indices]
        nearest_node_xy = node_xy[nearest_indices]

        lat1 = np.radians(cell_lat)
        lat2 = np.radians(nearest_node_xy[:, 1])
        dlat = lat2 - lat1
        dlon = np.radians(nearest_node_xy[:, 0] - cell_lon)

        haversine_a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
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
            100 * cell_population[~routable].sum() / total_population
        )

        population_records.append({
            "dun": dun["dun"],
            "raster_population_2020": total_population,
            "population_coverage_percent": coverage_percent,
            "mean_walk_distance_m_routable_population": mean_distance,
            "population_without_network_route_percent": no_route_percent,
        })


population_access = pd.DataFrame(population_records)

tai = tai.merge(
    population_access,
    on="dun",
    how="left",
    validate="one_to_one",
)


# Give areas with no routable population the worst observed walking score.
walk_column = "mean_walk_distance_m_routable_population"

no_route_areas = (
    tai[walk_column].isna()
    & tai["raster_population_2020"].gt(0)
    & tai["population_coverage_percent"].eq(0)
)

if tai[walk_column].notna().any():
    worst_walk_distance = tai[walk_column].max()
    tai.loc[no_route_areas, walk_column] = worst_walk_distance


print("Adding 2024 income and population...")
income = pd.read_csv(DUN_INCOME_FILE)
income["date"] = income["date"].astype(str)

income_2024 = income[
    (income["state"] == TARGET_STATE)
    & (income["date"] == TARGET_YEAR)
].copy()

income_2024["dun"] = income_2024["dun"].astype(str).str.strip()

population = pd.read_csv(DUN_POPULATION_FILE)
population["date"] = population["date"].astype(str)

population_2024 = population[
    (population["state"] == TARGET_STATE)
    & (population["date"] == TARGET_YEAR)
    & (population["sex"] == "both")
    & (population["age"] == "overall")
    & (population["ethnicity"] == "overall")
].copy()

population_2024["dun"] = (
    population_2024["dun"].astype(str).str.strip()
)

if len(income_2024) != 40 or income_2024["dun"].duplicated().any():
    raise ValueError("Expected 40 unique 2024 Penang income records.")

if (
    len(population_2024) != 40
    or population_2024["dun"].duplicated().any()
):
    raise ValueError("Expected 40 unique 2024 Penang population records.")

tai["dun"] = tai["dun"].astype(str).str.strip()

tai = tai.merge(
    income_2024[["dun", "income_median", "income_mean"]],
    on="dun",
    how="left",
    validate="one_to_one",
)

tai = tai.merge(
    population_2024[["dun", "population"]],
    on="dun",
    how="left",
    validate="one_to_one",
)

if tai["income_median"].isna().any() or tai["population"].isna().any():
    raise ValueError(
        "Some DUNs did not match the income or population data."
    )

# OpenDOSM population values are in thousands of people.
tai["population_2024_persons"] = tai["population"] * 1_000


print("Calculating DUN TAI...")
tai["stop_density_score"] = minmax_score(tai["bus_stops_per_km2"])
tai["population_coverage_score"] = minmax_score(
    tai["population_coverage_percent"]
)
tai["walking_distance_score"] = minmax_score(
    tai[walk_column],
    reverse=True,
)

destination_score_columns = []

for category in DESTINATION_COLUMNS:
    score_column = f"{category}_access_score"
    tai[score_column] = minmax_score(tai[f"{category}_per_stop"])
    destination_score_columns.append(score_column)

tai["destination_access_score"] = tai[
    destination_score_columns
].mean(axis=1)

tai_score_columns = [
    "stop_density_score",
    "population_coverage_score",
    "walking_distance_score",
    "destination_access_score",
]

tai["TAI"] = tai[tai_score_columns].sum(axis=1, min_count=4) / 4
tai["TAI_rank"] = tai["TAI"].rank(
    ascending=False,
    method="min",
).astype("Int64")


complete = tai[
    ["TAI", "income_median", "population_2024_persons"]
].notna().all(axis=1)

if complete.sum() < 3:
    raise ValueError("Too few complete DUNs for correlation analysis.")

income_result = spearmanr(
    tai.loc[complete, "income_median"],
    tai.loc[complete, "TAI"],
)

population_result = spearmanr(
    tai.loc[complete, "population_2024_persons"],
    tai.loc[complete, "TAI"],
)


output_columns = [
    "dun",
    "code_dun",
    "area_km2",
    "bus_stops",
    "bus_stops_per_km2",
    "raster_population_2020",
    "population_coverage_percent",
    "mean_walk_distance_m_routable_population",
    "population_without_network_route_percent",
    "schools_per_stop",
    "hospitals_per_stop",
    "clinics_per_stop",
    "malls_per_stop",
    "income_median",
    "income_mean",
    "population_2024_persons",
    *tai_score_columns,
    "TAI",
    "TAI_rank",
]

tai[output_columns].to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")
print(f"DUNs analysed: {len(tai)}")
print(f"DUNs with a TAI score: {tai['TAI'].notna().sum()}")

print("\nSpearman correlation: 2024 median household income vs DUN TAI")
print(
    f"rho={income_result.statistic:.3f}; "
    f"p-value={income_result.pvalue:.4f}"
)

print("\nSpearman correlation: 2024 population vs DUN TAI")
print(
    f"rho={population_result.statistic:.3f}; "
    f"p-value={population_result.pvalue:.4f}"
)

print("\nFive lowest DUN TAI scores:")
print(
    tai.nsmallest(5, "TAI")[["dun", "TAI", "TAI_rank"]]
    .to_string(index=False)
)

print("\nFive highest DUN TAI scores:")
print(
    tai.nlargest(5, "TAI")[["dun", "TAI", "TAI_rank"]]
    .to_string(index=False)
)