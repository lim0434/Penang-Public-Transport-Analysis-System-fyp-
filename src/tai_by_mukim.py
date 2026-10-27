import math
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from shapely.geometry import Point
from shapely.ops import transform as geometry_transform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ROOT = Path(r"D:\fyp")
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

BOUNDARY_FILE = RAW / "boundary" / "penang_mukim_boundaries.geojson"
STOPS_FILE = PROCESSED / "stops_penang.csv"
STOP_ACCESS_FILE = PROCESSED / "accessibility_400m_by_stop.csv"
POPULATION_ACCESS_FILE = (
    PROCESSED / "population_accessibility_400m_by_mukim.csv"
)
OUTPUT_FILE = PROCESSED / "tai_by_mukim.csv"

DISTRICT_PREFIX = {
    "01": "SPT",
    "02": "SPU",
    "03": "SPS",
    "04": "DTL",
    "05": "DBD",
}

DESTINATION_COLUMNS = {
    "schools": "schools_within_400m",
    "hospitals": "hospitals_within_400m",
    "clinics": "clinics_within_400m",
    "malls": "malls_within_400m",
}


def make_mukim_label(row):
    district_code = str(row["KOD_DAERAH"]).zfill(2)
    district = DISTRICT_PREFIX[district_code]
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


def area_km2(geometry):
    """Estimate area using a local longitude-latitude conversion."""
    latitude = math.radians(geometry.centroid.y)

    projected = geometry_transform(
        lambda x, y, z=None: (
            np.asarray(x) * 111_320 * math.cos(latitude),
            np.asarray(y) * 110_574,
        ),
        geometry,
    )

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


print("Loading mukim boundaries and saved accessibility results...")

# Load and prepare the mukim boundaries.
mukims = gpd.read_file(BOUNDARY_FILE)

if mukims.crs is None:
    raise ValueError("Mukim boundary file has no CRS.")

if str(mukims.crs).upper() != "EPSG:4326":
    mukims = mukims.to_crs("EPSG:4326")

mukims["match_name"] = mukims.apply(make_mukim_label, axis=1)

if mukims["match_name"].duplicated().any():
    raise ValueError("Some boundary names are duplicated.")

mukims["area_km2"] = mukims.geometry.apply(area_km2)

# Load the saved stop and population accessibility results.
stops = pd.read_csv(STOPS_FILE, dtype={"stop_id": str})
stop_access = pd.read_csv(STOP_ACCESS_FILE, dtype={"stop_id": str})
population_access = pd.read_csv(POPULATION_ACCESS_FILE)

# Assign each bus stop to the boundary area containing it.
mukim_geometries = list(zip(mukims["match_name"], mukims.geometry))
stop_mukim_rows = []

for _, stop in stops.iterrows():
    point = Point(float(stop["stop_lon"]), float(stop["stop_lat"]))

    matched_name = next(
        (
            name
            for name, geometry in mukim_geometries
            if geometry.covers(point)
        ),
        None,
    )

    if matched_name is None:
        raise ValueError(
            f"Stop {stop['stop_id']} could not be assigned to a mukim."
        )

    stop_mukim_rows.append({
        "stop_id": stop["stop_id"],
        "match_name": matched_name,
    })

stop_mukims = pd.DataFrame(stop_mukim_rows)

# Attach destination counts to each bus stop.
stop_data = stops[["stop_id"]].merge(
    stop_access[
        ["stop_id", *DESTINATION_COLUMNS.values()]
    ],
    on="stop_id",
    how="left",
    validate="one_to_one",
).merge(
    stop_mukims,
    on="stop_id",
    how="left",
    validate="one_to_one",
)

if stop_data[list(DESTINATION_COLUMNS.values())].isna().any().any():
    raise ValueError(
        "Some stops are missing destination-access results."
    )

# Summarise bus stops and nearby destinations for each area.
stop_summary = stop_data.groupby("match_name").agg(
    bus_stops=("stop_id", "count"),
    schools_per_stop=("schools_within_400m", "mean"),
    hospitals_per_stop=("hospitals_within_400m", "mean"),
    clinics_per_stop=("clinics_within_400m", "mean"),
    malls_per_stop=("malls_within_400m", "mean"),
).reset_index()

tai = mukims[["match_name", "area_km2"]].merge(
    stop_summary,
    on="match_name",
    how="left",
    validate="one_to_one",
)

tai["bus_stops"] = tai["bus_stops"].fillna(0).astype(int)

for category in DESTINATION_COLUMNS:
    column = f"{category}_per_stop"
    tai[column] = tai[column].fillna(0)

tai["bus_stops_per_km2"] = tai["bus_stops"] / tai["area_km2"]

# Add population coverage, population totals and walking-distance results.
population_columns = [
    "match_name",
    "csv_population",
    "population_coverage_percent",
    "estimated_csv_population_covered",
    "mean_walk_distance_m_routable_population",
    "population_without_network_route_percent",
]

tai = tai.merge(
    population_access[population_columns],
    on="match_name",
    how="left",
    validate="one_to_one",
)

if len(tai) != len(mukims):
    raise ValueError(
        "Some boundary areas did not match the population results."
    )

# Give populated areas with no network route the worst observed
# walking-distance value. Areas with missing population data stay missing.
walk_column = "mean_walk_distance_m_routable_population"

no_route_with_population = (
    tai[walk_column].isna()
    & tai["population_coverage_percent"].eq(0)
    & tai["csv_population"].gt(0)
)

worst_observed_distance = tai[walk_column].max()

tai.loc[no_route_with_population, walk_column] = worst_observed_distance

# Normalise bus-stop density, population coverage and walking distance.
tai["stop_density_score"] = minmax_score(tai["bus_stops_per_km2"])

tai["population_coverage_score"] = minmax_score(
    tai["population_coverage_percent"]
)

tai["walking_distance_score"] = minmax_score(
    tai[walk_column],
    reverse=True,
)

# Normalise each destination type, then average the four scores.
destination_score_columns = []

for category in DESTINATION_COLUMNS:
    score_column = f"{category}_access_score"

    tai[score_column] = minmax_score(
        tai[f"{category}_per_stop"]
    )

    destination_score_columns.append(score_column)

tai["destination_access_score"] = tai[
    destination_score_columns
].mean(axis=1)

# Combine the four main indicators with equal weights.
score_columns = [
    "stop_density_score",
    "population_coverage_score",
    "walking_distance_score",
    "destination_access_score",
]

tai["TAI"] = tai[score_columns].sum(axis=1, min_count=4) / 4

tai["TAI_rank"] = tai["TAI"].rank(
    ascending=False,
    method="min",
).astype("Int64")

# Prepare complete records for clustering.
complete = tai[score_columns].notna().all(axis=1)
cluster_input = tai.loc[complete, score_columns]

# Compare possible cluster counts.
print("\nChecking possible cluster counts:")

for test_k in range(2, 7):
    test_model = KMeans(
        n_clusters=test_k,
        random_state=42,
        n_init=20,
    )

    test_labels = test_model.fit_predict(cluster_input)
    silhouette = silhouette_score(cluster_input, test_labels)

    print(
        f"k={test_k}: "
        f"inertia={test_model.inertia_:.3f}; "
        f"silhouette={silhouette:.3f}"
    )

# Use three clusters for the Low, Medium and High groups.
cluster_count = 3

model = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=20,
)

tai["cluster_id"] = pd.Series(
    pd.NA,
    index=tai.index,
    dtype="Int64",
)

tai.loc[complete, "cluster_id"] = model.fit_predict(
    tai.loc[complete, score_columns]
)

# Name the clusters according to their mean TAI.
cluster_means = (
    tai.loc[complete]
    .groupby("cluster_id")["TAI"]
    .mean()
    .sort_values()
)

cluster_names = ["Low", "Medium", "High"]

cluster_labels = {
    cluster_id: name
    for cluster_id, name in zip(
        cluster_means.index,
        cluster_names,
    )
}

tai["TAI_category"] = "No data"

for cluster_id, label in cluster_labels.items():
    tai.loc[tai["cluster_id"] == cluster_id, "TAI_category"] = label

# Prepare the cluster and correlation summaries.
correlation_indicators = [
    "bus_stops_per_km2",
    "population_coverage_percent",
    walk_column,
    "destination_access_score",
]

correlation_data = tai.loc[complete, correlation_indicators]

profile_columns = [
    "stop_density_score",
    "population_coverage_score",
    "walking_distance_score",
    "destination_access_score",
]

cluster_profile = (
    tai.loc[complete]
    .groupby("TAI_category")[profile_columns]
    .mean()
    .reindex(["Low", "Medium", "High"])
    .round(3)
)

mean_tai_by_cluster = (
    tai.loc[complete]
    .groupby("TAI_category")["TAI"]
    .mean()
    .reindex(["Low", "Medium", "High"])
    .round(3)
)

# Print the results in order.
print("\nK-means cluster counts:")
print(
    tai.loc[complete, "TAI_category"]
    .value_counts()
    .reindex(["Low", "Medium", "High"])
    .to_string()
)

print("\nMean TAI by cluster:")
print(mean_tai_by_cluster.to_string())

print("\nMean indicator scores by cluster (0–1; higher is better):")
print(cluster_profile.to_string())

print("\nSpearman correlations between indicators:")

for first, second in combinations(correlation_indicators, 2):
    result = spearmanr(
        correlation_data[first],
        correlation_data[second],
    )

    print(
        f"{first} vs {second}: "
        f"rho={result.statistic:.3f}, "
        f"p-value={result.pvalue:.4f}"
    )

# Save the TAI results.
tai.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Areas with a TAI score: {tai['TAI'].notna().sum()} of {len(tai)}")
print(f"Areas without a score: {tai['TAI'].isna().sum()}")

print("\nLowest TAI scores:")
print(
    tai.nsmallest(5, "TAI")[
        ["match_name", "TAI", "TAI_rank", "TAI_category"]
    ].to_string(index=False)
)

print("\nHighest TAI scores:")
print(
    tai.nlargest(5, "TAI")[
        ["match_name", "TAI", "TAI_rank", "TAI_category"]
    ].to_string(index=False)
)