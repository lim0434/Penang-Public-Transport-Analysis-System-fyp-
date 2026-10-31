from __future__ import annotations

import heapq
import math
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial import cKDTree
from PIL import Image
from shapely import contains_xy
from shapely.geometry import Point

from ui_helpers import clean_web_text, make_area_label, read_geojson


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
POPULATION_RASTER = ROOT / "data" / "raw" / "population" / "mys_ppp_2020_constrained.tif"
BOUNDARY_FILE = ROOT / "data" / "raw" / "boundary" / "penang_mukim_boundaries.geojson"
STOPS_FILE = DATA / "stops_penang.csv"
TAI_FILE = DATA / "tai_by_mukim.csv"
COVERAGE_FILE = DATA / "population_accessibility_400m_by_mukim.csv"
DESTINATIONS_FILE = DATA / "connectivity_by_destination.csv"
GHS_FALLBACK_FILE = DATA / "ghs_coverage_dtl_mukim15.csv"
STOP_ACCESS_FILE = DATA / "accessibility_400m_by_stop.csv"
WALK_LIMIT_M = 400.0
WALKING_SPEED_M_PER_MIN = 80.0


def _distance_m(lon1, lat1, lon2, lat2):
    lat1, lat2 = math.radians(float(lat1)), math.radians(float(lat2))
    dlat = lat2 - lat1
    dlon = math.radians(float(lon2) - float(lon1))
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


@st.cache_resource(show_spinner="Preparing the walking network and current bus stops...")
def _load_network_context():
    graph = nx.read_graphml(DATA / "walking_network_candidate.graphml")
    graph = graph.to_undirected()
    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
        for _, _, _, data in edge_iter:
            data["length"] = float(data["length"])
    else:
        for _, _, data in graph.edges(data=True):
            data["length"] = float(data["length"])

    node_ids = list(graph.nodes)
    node_xy = np.asarray([
        [float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])]
        for node in node_ids
    ])
    mean_lat = float(node_xy[:, 1].mean())
    tree = cKDTree(np.column_stack((node_xy[:, 0] * math.cos(math.radians(mean_lat)), node_xy[:, 1])))

    stops = pd.read_csv(STOPS_FILE)
    stop_sources = []
    for stop in stops.itertuples(index=False):
        node, offset = _snap_to_network(stop.stop_lon, stop.stop_lat, node_ids, node_xy, tree, mean_lat)
        stop_sources.append((node, offset))
    baseline_distances = _distances_from_sources(graph, stop_sources)
    return graph, node_ids, node_xy, tree, mean_lat, stops, stop_sources, baseline_distances


def _snap_to_network(lon, lat, node_ids, node_xy, tree, mean_lat):
    query = [float(lon) * math.cos(math.radians(mean_lat)), float(lat)]
    _, index = tree.query(query)
    node = node_ids[int(index)]
    node_lon, node_lat = node_xy[int(index)]
    return node, _distance_m(lon, lat, node_lon, node_lat)


def _edge_length(graph, edge_data):
    if graph.is_multigraph():
        return min(float(attrs["length"]) for attrs in edge_data.values())
    return float(edge_data["length"])


def _distances_from_sources(graph, sources):
    """Dijkstra search with each stop's snap distance as its start cost."""
    distances = {}
    queue = []
    for node, initial_distance in sources:
        initial_distance = float(initial_distance)
        if initial_distance < distances.get(node, float("inf")):
            distances[node] = initial_distance
            heapq.heappush(queue, (initial_distance, str(node), node))

    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        for neighbour, edge_data in graph.adj[node].items():
            candidate = distance + _edge_length(graph, edge_data)
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, str(neighbour), neighbour))
    return distances


@st.cache_data(show_spinner="Loading population cells for this area...")
def _load_population_cells(area_name):
    census = pd.read_csv(COVERAGE_FILE).set_index("match_name")
    census_population = float(census.loc[area_name, "csv_population"])

    if area_name == "DTL MUKIM 15":
        ghs = pd.read_csv(GHS_FALLBACK_FILE)
        return (
            ghs["cell_longitude"].to_numpy(float),
            ghs["cell_latitude"].to_numpy(float),
            ghs["estimated_population_scaled_to_census_13"].to_numpy(float),
            census_population,
            "GHS-POP 2020 scaled to census population",
        )

    areas = read_geojson(BOUNDARY_FILE)
    areas["match_name"] = areas.apply(make_area_label, axis=1)
    area = areas.loc[areas["match_name"] == area_name].to_crs("EPSG:4326")
    if area.empty:
        raise ValueError(f"Could not find the boundary for {area_name}.")
    geometry = area.geometry.iloc[0]

    # Pillow reads this trusted local GeoTIFF without loading Rasterio's blocked DLLs.
    # Temporarily disable Pillow's default size check for this known project raster.
    previous_pixel_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(POPULATION_RASTER) as raster:
            pixel_width, pixel_height, _ = raster.tag_v2[33550]
            _, _, _, origin_x, origin_y, _ = raster.tag_v2[33922]
            min_x, min_y, max_x, max_y = geometry.bounds
            col_start = max(0, math.floor((min_x - origin_x) / pixel_width))
            col_end = min(raster.width, math.ceil((max_x - origin_x) / pixel_width))
            row_start = max(0, math.floor((origin_y - max_y) / pixel_height))
            row_end = min(raster.height, math.ceil((origin_y - min_y) / pixel_height))
            if col_start >= col_end or row_start >= row_end:
                return np.array([]), np.array([]), np.array([]), census_population, "WorldPop 2020"

            values = np.asarray(
                raster.crop((col_start, row_start, col_end, row_end)), dtype=float
            )
    finally:
        Image.MAX_IMAGE_PIXELS = previous_pixel_limit

    longitudes = origin_x + (np.arange(col_start, col_end) + 0.5) * pixel_width
    latitudes = origin_y - (np.arange(row_start, row_end) + 0.5) * pixel_height
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    valid = (
        contains_xy(geometry, longitude_grid, latitude_grid)
        & np.isfinite(values)
        & (values > 0)
    )
    if not valid.any():
        return np.array([]), np.array([]), np.array([]), census_population, "WorldPop 2020"

    longitudes = longitude_grid[valid]
    latitudes = latitude_grid[valid]
    raw_weights = values[valid]
    scale = census_population / float(raw_weights.sum()) if raw_weights.sum() else 0.0
    return (
        longitudes.astype(float),
        latitudes.astype(float),
        raw_weights * scale,
        census_population,
        "WorldPop 2020 scaled to the area census total",
    )

@st.cache_data(show_spinner="Finding residential areas beyond current walking access...")
def _residential_candidate_sites(area_name):
    """Create distinct candidate locations in high-population uncovered clusters."""
    graph, node_ids, node_xy, tree, mean_lat, _, _, base_distances = _load_network_context()
    areas = read_geojson(BOUNDARY_FILE)
    areas["match_name"] = areas.apply(make_area_label, axis=1)
    area_shape = areas.loc[areas["match_name"] == area_name].to_crs("EPSG:4326")
    if area_shape.empty:
        return pd.DataFrame()
    area_geometry = area_shape.geometry.iloc[0]
    pop_lon, pop_lat, population, _, _ = _load_population_cells(area_name)
    if len(population) == 0:
        return pd.DataFrame()

    query = np.column_stack((pop_lon * math.cos(math.radians(mean_lat)), pop_lat))
    _, pop_indices = tree.query(query)
    pop_nodes = [node_ids[int(index)] for index in pop_indices]
    node_index_by_id = {node: index for index, node in enumerate(node_ids)}
    pop_offsets = np.asarray([
        _distance_m(lon, lat, node_xy[int(index), 0], node_xy[int(index), 1])
        for lon, lat, index in zip(pop_lon, pop_lat, pop_indices)
    ])
    current_walk = np.asarray([
        base_distances.get(node, float("inf")) for node in pop_nodes
    ]) + pop_offsets
    uncovered = (current_walk > WALK_LIMIT_M) & (population > 0)
    if not uncovered.any():
        return pd.DataFrame()

    # Group nearby population cells into 400 m squares, then place each
    # candidate at the nearest walking-network node to the group's centre.
    mean_lat_radians = math.radians(mean_lat)
    x_m = pop_lon * 111_320 * math.cos(mean_lat_radians)
    y_m = pop_lat * 110_540
    cell_frame = pd.DataFrame({
        "x_m": x_m[uncovered],
        "y_m": y_m[uncovered],
        "longitude": pop_lon[uncovered],
        "latitude": pop_lat[uncovered],
        "population": population[uncovered],
    })
    cell_frame["grid_x"] = np.floor(cell_frame["x_m"] / 400).astype(int)
    cell_frame["grid_y"] = np.floor(cell_frame["y_m"] / 400).astype(int)
    clusters = []
    for _, group in cell_frame.groupby(["grid_x", "grid_y"], sort=False):
        weight = group["population"].to_numpy(float)
        total = float(weight.sum())
        if total <= 0:
            continue
        centre_lon = float(np.average(group["longitude"], weights=weight))
        centre_lat = float(np.average(group["latitude"], weights=weight))
        node, _ = _snap_to_network(
            centre_lon, centre_lat, node_ids, node_xy, tree, mean_lat
        )
        node_index = node_index_by_id[node]
        node_lon, node_lat = node_xy[node_index]
        if not area_geometry.covers(Point(float(node_lon), float(node_lat))):
            continue
        current_distance = float(base_distances.get(node, float("inf")))
        if current_distance <= WALK_LIMIT_M:
            continue
        clusters.append({
            "population": total,
            "node": node,
            "longitude": float(node_lon),
            "latitude": float(node_lat),
            "current_distance": current_distance,
        })

    clusters.sort(key=lambda item: item["population"], reverse=True)
    selected = []
    for cluster in clusters:
        if any(
            _distance_m(
                cluster["longitude"], cluster["latitude"],
                other["longitude"], other["latitude"],
            ) < 400
            for other in selected
        ):
            continue
        selected.append(cluster)

    rows = []
    for index, cluster in enumerate(selected, start=1):
        rows.append({
            "destination_id": f"residential_{area_name.lower().replace(' ', '_')}_{index}",
            "destination_name": f"Residential area {index}",
            "destination_type": "Residential area",
            "destination_lon": cluster["longitude"],
            "destination_lat": cluster["latitude"],
            "network_distance_m": cluster["current_distance"],
            "destination_snap_gap_m": 0.0,
            "stop_snap_gap_m": 0.0,
            "mukim": area_name,
        })
    return pd.DataFrame(rows)



@st.cache_data(show_spinner="Ranking possible stop locations...")
def _rank_candidate_sites(area_name, candidate_ids, candidate_rows=()):
    """Rank candidate sites by estimated extra walking access."""
    graph, node_ids, node_xy, tree, mean_lat, _, _, base_distances = _load_network_context()
    if candidate_rows:
        candidates = pd.DataFrame(candidate_rows)
    else:
        candidates = pd.read_csv(DESTINATIONS_FILE)
    candidates = candidates[
        (candidates["mukim"] == area_name)
        & candidates["destination_id"].isin(candidate_ids)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    pop_lon, pop_lat, population, _, _ = _load_population_cells(area_name)
    _, pop_indices = tree.query(np.column_stack((pop_lon * math.cos(math.radians(mean_lat)), pop_lat)))
    pop_nodes = [node_ids[int(i)] for i in pop_indices]
    pop_offsets = np.asarray([
        _distance_m(lon, lat, node_xy[int(i), 0], node_xy[int(i), 1])
        for lon, lat, i in zip(pop_lon, pop_lat, pop_indices)
    ])
    pop_before = np.asarray([
        base_distances.get(node, float("inf")) + offset <= WALK_LIMIT_M
        for node, offset in zip(pop_nodes, pop_offsets)
    ])

    dest = pd.read_csv(DESTINATIONS_FILE)
    dest = dest[dest["mukim"] == area_name]
    dlon = pd.to_numeric(dest["destination_lon"], errors="coerce").to_numpy(float)
    dlat = pd.to_numeric(dest["destination_lat"], errors="coerce").to_numpy(float)
    _, di = tree.query(np.column_stack((dlon * math.cos(math.radians(mean_lat)), dlat)))
    dnodes = [node_ids[int(i)] for i in di]
    doffsets = np.asarray([
        _distance_m(lon, lat, node_xy[int(i), 0], node_xy[int(i), 1])
        for lon, lat, i in zip(dlon, dlat, di)
    ])
    dest_before = np.asarray([
        base_distances.get(node, float("inf")) + offset <= WALK_LIMIT_M
        for node, offset in zip(dnodes, doffsets)
    ], dtype=bool)

    rows=[]
    for c in candidates.itertuples(index=False):
        node, gap = _snap_to_network(c.destination_lon, c.destination_lat, node_ids, node_xy, tree, mean_lat)
        dist = nx.single_source_dijkstra_path_length(
            graph, node, cutoff=max(0.0, WALK_LIMIT_M-gap), weight="length"
        )
        pop_after = np.asarray([
            gap + dist.get(n, float("inf")) + off for n, off in zip(pop_nodes, pop_offsets)
        ])
        dest_after = np.asarray([
            gap + dist.get(n, float("inf")) + off for n, off in zip(dnodes, doffsets)
        ])
        other_destinations = dest["destination_id"].to_numpy() != c.destination_id
        rows.append({
            "destination_id": c.destination_id,
            "Place": c.destination_name,
            "Type": c.destination_type,
            "Estimated residents gaining access": float(population[(~pop_before) & (pop_after <= WALK_LIMIT_M)].sum()),
            "Key places gaining access": int(
                ((~dest_before) & (dest_after <= WALK_LIMIT_M) & other_destinations).sum()
            ),
            "Walking network gap (m)": float(gap),
            "Current estimated walking distance to a bus stop (m)": float(c.network_distance_m)+float(c.destination_snap_gap_m)+float(c.stop_snap_gap_m),
        })
    ranked=pd.DataFrame(rows).sort_values(
        ["Estimated residents gaining access","Key places gaining access"],
        ascending=False, kind="stable"
    ).reset_index(drop=True)
    ranked.insert(0,"Priority",np.arange(1,len(ranked)+1))
    return ranked

def _candidate_destination_counts(graph, candidates, places, node_ids, node_xy, tree, mean_lat):
    """Count listed places within the walking limit of each added stop."""
    valid = places.dropna(subset=["destination_lon", "destination_lat"]).copy()
    place_nodes = []
    place_gaps = []
    for row in valid.itertuples(index=False):
        node, gap = _snap_to_network(
            row.destination_lon, row.destination_lat,
            node_ids, node_xy, tree, mean_lat,
        )
        place_nodes.append(node)
        place_gaps.append(gap)

    counts = []
    for _, candidate in candidates.iterrows():
        node, stop_gap = _snap_to_network(
            candidate["destination_lon"], candidate["destination_lat"],
            node_ids, node_xy, tree, mean_lat,
        )
        remaining = max(0.0, WALK_LIMIT_M - stop_gap)
        distances = nx.single_source_dijkstra_path_length(
            graph, node, cutoff=remaining, weight="length"
        )
        row_counts = {"School": 0, "Hospital": 0, "Clinic": 0, "Mall": 0}
        for place, place_node, place_gap in zip(
            valid.itertuples(index=False), place_nodes, place_gaps
        ):
            if stop_gap + distances.get(place_node, float("inf")) + place_gap <= WALK_LIMIT_M:
                row_counts[place.destination_type] += 1
        counts.append(row_counts)
    return counts


def _scenario_tai_score(area_name, added_stop_count, coverage_percent, mean_walk_m, added_place_counts):
    """Recalculate the selected mukim's TAI on the original Penang score scale."""
    tai = pd.read_csv(TAI_FILE)
    row = tai.loc[tai["match_name"] == area_name]
    if row.empty or added_stop_count < 1:
        return float("nan")
    row = row.iloc[0]
    existing_stops = int(row["bus_stops"])
    new_stop_total = existing_stops + int(added_stop_count)
    if new_stop_total <= 0:
        return float("nan")

    def fixed_minmax(value, column, reverse=False):
        values = pd.to_numeric(tai[column], errors="coerce").dropna()
        if values.empty or not np.isfinite(value):
            return float("nan")
        minimum, maximum = float(values.min()), float(values.max())
        if maximum == minimum:
            return 0.5
        score = ((maximum - value) if reverse else (value - minimum)) / (maximum - minimum)
        return float(np.clip(score, 0.0, 1.0))

    stop_density = new_stop_total / float(row["area_km2"])
    density_score = fixed_minmax(stop_density, "bus_stops_per_km2")
    coverage_score = fixed_minmax(coverage_percent, "population_coverage_percent")
    walking_score = fixed_minmax(
        mean_walk_m, "mean_walk_distance_m_routable_population", reverse=True
    )

    category_columns = {
        "School": "schools_per_stop",
        "Hospital": "hospitals_per_stop",
        "Clinic": "clinics_per_stop",
        "Mall": "malls_per_stop",
    }
    destination_scores = []
    for place_type, column in category_columns.items():
        existing_per_stop = float(row[column]) if pd.notna(row[column]) else 0.0
        existing_total = existing_per_stop * existing_stops
        added_total = sum(counts.get(place_type, 0) for counts in added_place_counts)
        after_per_stop = (existing_total + added_total) / new_stop_total
        destination_scores.append(fixed_minmax(after_per_stop, column))

    destination_score = float(np.nanmean(destination_scores))
    components = [density_score, coverage_score, walking_score, destination_score]
    if not np.isfinite(components).all():
        return float("nan")
    return float(np.mean(components))


def _map_for_area(area_name, all_areas, stops, candidates=None, snapped_locations=None, gained_cells=None):
    map_view = folium.Map(
        location=[5.38, 100.35],
        zoom_start=9,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Street map",
        show=True,
    ).add_to(map_view)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        show=False,
    ).add_to(map_view)
    coverage_scale = folium.LinearColormap(
        colors=["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"],
        vmin=0,
        vmax=100,
        caption="Residents within a short walk (%)",
    )

    def style_area(feature):
        value = feature["properties"].get("population_coverage_percent")
        if value is None:
            fill = "#bdbdbd"
        else:
            fill = coverage_scale(float(value))
        return {
            "fillColor": fill,
            "color": "#555555",
            "weight": 0.8,
            "fillOpacity": 0.72,
        }

    areas_geo = all_areas.to_crs("EPSG:4326").copy()
    if "display_name" in areas_geo.columns:
        areas_geo["display_name"] = areas_geo["display_name"].map(clean_web_text)
    folium.GeoJson(
        areas_geo.to_json(),
        name="Existing population coverage",
        style_function=style_area,
        tooltip=folium.GeoJsonTooltip(
            fields=["display_name", "population_coverage_percent"],
            aliases=["Area:", "Residents within a short walk (%):"],
            localize=True,
            sticky=False,
        ),
    ).add_to(map_view)

    area_shape = areas_geo.loc[areas_geo["match_name"] == area_name]
    selected_geometry = area_shape.geometry.iloc[0]
    folium.GeoJson(
        area_shape.to_json(),
        name="Selected area",
        style_function=lambda _: {"color": "#ef7a3a", "weight": 3, "fillOpacity": 0},
        tooltip=f"Selected area: {clean_web_text(all_areas.loc[all_areas['match_name'] == area_name, 'display_name'].iloc[0])}",
    ).add_to(map_view)

    for stop in stops.itertuples(index=False):
        stop_point = Point(float(stop.stop_lon), float(stop.stop_lat))
        if selected_geometry.covers(stop_point):
            folium.Marker(
                [stop_point.y, stop_point.x],
                tooltip=f"Existing bus stop: {clean_web_text(str(stop.stop_name))}",
                icon=folium.Icon(
                    color="blue",
                    icon="bus",
                    prefix="fa",
                    icon_color="white",
                ),
            ).add_to(map_view)

    if gained_cells is not None and not gained_cells.empty:
        residents_layer = folium.FeatureGroup(
            name="Residential areas that may gain access",
            show=False,
        )
        cells_to_draw = gained_cells
        if len(cells_to_draw) > 350:
            sample_indices = np.linspace(0, len(cells_to_draw) - 1, 350, dtype=int)
            cells_to_draw = cells_to_draw.iloc[sample_indices]
        for cell in cells_to_draw.itertuples(index=False):
            folium.Marker(
                [cell.latitude, cell.longitude],
                tooltip=f"Estimated population cell gaining access: {cell.population:,.1f}",
                icon=folium.Icon(color="green", icon="home", prefix="fa", icon_color="white"),
            ).add_to(residents_layer)
        residents_layer.add_to(map_view)

    place_icons = {
        "School": "graduation-cap",
        "Hospital": "hospital-o",
        "Clinic": "medkit",
        "Mall": "shopping-cart",
        "Residential area": "home",
    }
    for _, proposed in (candidates.iterrows() if candidates is not None else []):
        destination_id = proposed["destination_id"]
        place_lat = float(proposed["destination_lat"])
        place_lon = float(proposed["destination_lon"])
        place_type = str(proposed["destination_type"])
        place_name = clean_web_text(str(proposed["destination_name"]))
        is_residential_candidate = place_type == "Residential area"
        folium.Marker(
            [place_lat, place_lon],
            tooltip=(
                f"Residential area suggestion: {place_name}"
                if is_residential_candidate else f"{place_type}: {place_name}"
            ),
            popup=(
                f"Possible stop near {place_name}"
                if is_residential_candidate else f"Candidate key place: {place_name}"
            ),
            icon=folium.Icon(
                color="darkgreen" if is_residential_candidate else "purple",
                icon=place_icons.get(place_type, "info-sign"),
                prefix="fa",
            ),
        ).add_to(map_view)

        snapped_location = (snapped_locations or {}).get(destination_id)
        if snapped_location is None:
            marker_lat, marker_lon = place_lat, place_lon
            tooltip = "Proposed bus stop (preview location)"
        else:
            marker_lat, marker_lon = snapped_location
            tooltip = "Proposed bus stop, snapped to the walking network"
        folium.Marker(
            [marker_lat, marker_lon],
            tooltip=tooltip,
            popup=f"Proposed bus stop near {place_type.lower()}: {place_name}",
            icon=folium.Icon(color="orange", icon="bus", prefix="fa"),
        ).add_to(map_view)

    folium.LayerControl(collapsed=True).add_to(map_view)
    map_view.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;z-index:9999;bottom:18px;left:14px;"
        "width:210px;background:rgba(255,255,255,.96);padding:11px 13px;"
        "border:1px solid #d8dce3;border-radius:10px;box-shadow:0 2px 10px #0002;"
        "font:12px/1.45 Arial,sans-serif;color:#252a31'>"
        "<strong>Map key</strong>"
        "<div style='height:9px;border-radius:8px;margin:9px 0 4px;"
        "background:linear-gradient(90deg,#b2182b,#ef8a62,#f7f7f7,#67a9cf,#2166ac)'></div>"
        "<div style='display:flex;justify-content:space-between;color:#5f6368'>"
        "<span>Lower share</span><span>Higher share</span></div>"
        "<div style='border-top:1px solid #e6e8eb;margin-top:8px;padding-top:7px;display:grid;gap:5px'>"
        "<div><i style='display:inline-block;width:10px;height:10px;background:#3478c8;border-radius:50%;margin-right:7px'></i>Existing bus stop</div>"
        "<div><i style='display:inline-block;width:10px;height:10px;background:#ef7a3a;border-radius:50%;margin-right:7px'></i>Proposed bus stop</div>"
        "<div><i style='display:inline-block;width:10px;height:10px;background:#7b4ab5;border-radius:50%;margin-right:7px'></i>Key place</div>"
        "<div><i style='display:inline-block;width:10px;height:10px;background:#246b3b;border-radius:50%;margin-right:7px'></i>Residential suggestion</div>"
        "</div></div>"
    ))
    bounds = areas_geo.total_bounds
    map_view.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]], padding=(18, 18))
    return map_view


def render(tai, all_areas, all_stops, scored, socioeconomic):
    st.caption("Test how proposed bus stops could improve access within about a 5-minute walk.")
    with st.expander("How suggested stop places are chosen"):
        st.write(
            "The system looks for listed schools, hospitals, clinics and malls beyond a 400 m walk from a current bus stop. "
            "It also finds residential clusters with people outside that range. Candidate places are ranked by the estimated "
            "number of residents and other key places that could come within 400 m if a stop were added. These are examples "
            "for testing, not confirmed stop sites."
        )

    coverage = pd.read_csv(COVERAGE_FILE)
    area_options = coverage.merge(
        tai[["match_name", "TAI_category", "TAI", "TAI_rank"]],
        on="match_name", how="left", validate="one_to_one",
    ).dropna(subset=["population_coverage_percent"])
    area_options = area_options[
        area_options["TAI_category"].isin(["Low", "Medium"])
    ].sort_values("population_coverage_percent")
    if area_options.empty:
        st.warning("No Low or Medium TAI areas have usable population-coverage data.")
        return

    display_names = all_areas.set_index("match_name")["display_name"].to_dict()
    labels = {}
    for row in area_options.itertuples(index=False):
        official_name = row.match_name
        labels[official_name] = display_names.get(official_name, official_name)
    area_names = area_options["match_name"].tolist()
    area_name = st.selectbox(
        "Choose a Low or Medium access area",
        area_names,
        format_func=lambda name: clean_web_text(labels.get(name, name)),
        help="Numbered mukims include a nearby bus-stop name to help locate them. The official mukim number remains in the label.",
    )

    destinations = pd.read_csv(DESTINATIONS_FILE)
    distance_parts = [
        "network_distance_m",
        "destination_snap_gap_m",
        "stop_snap_gap_m",
    ]
    for column in distance_parts:
        destinations[column] = pd.to_numeric(destinations[column], errors="coerce")
    # The saved network distance covers snapped nodes only. Include both endpoint
    # snap gaps so the 400 m check uses the same total as the simulation.
    destinations["estimated_walking_distance_m"] = destinations[
        distance_parts
    ].sum(axis=1, min_count=len(distance_parts))
    area_destinations = destinations[
        (destinations["mukim"] == area_name)
        & (destinations["estimated_walking_distance_m"] > WALK_LIMIT_M)
    ].copy()
    page_export = pd.DataFrame()
    page_export_name = "suggested_bus_stop_locations.csv"
    try:
        residential_sites = _residential_candidate_sites(area_name)
    except ImportError as error:
        st.error(
            "Windows blocked Rasterio, which is needed for population-based "
            "simulation. Ask your administrator to allow the Rasterio files. "
            f"Details: {error}"
        )
        return
    all_candidate_sites = pd.concat(
        [area_destinations, residential_sites], ignore_index=True, sort=False
    )
    if all_candidate_sites.empty:
        candidates = all_candidate_sites.copy()
        st.warning(
            "No uncovered key places or residential areas with population data "
            "were found for suggested stop locations in this area."
        )
    else:
        ranked_sites = _rank_candidate_sites(
            area_name,
            tuple(all_candidate_sites["destination_id"].tolist()),
            tuple(all_candidate_sites.to_dict("records")),
        )
        if ranked_sites.empty:
            candidates = all_candidate_sites.copy()
            st.warning("No candidate stop locations could be ranked for this area.")
        else:
            st.subheader("Suggested places to check")
            st.caption(
                "Key places means the schools, hospitals, clinics and malls in the data. Rows are ranked by estimated new access for residents and key places. "
                "These places have not been checked on site for road safety, land use "
                "or space for a bus stop."
            )
            shown = ranked_sites.head(5).copy()
            shown["Estimated residents gaining access"] = shown["Estimated residents gaining access"].round().astype(int)
            shown["Walking network gap (m)"] = shown["Walking network gap (m)"].round().astype(int)
            shown["Current walk to bus stop (min)"] = (
                shown["Current estimated walking distance to a bus stop (m)"]
                / WALKING_SPEED_M_PER_MIN
            ).round(1)
            shown = shown.drop(columns=["Current estimated walking distance to a bus stop (m)"])
            page_export = shown.drop(
                columns=["destination_id", "Walking network gap (m)"],
                errors="ignore",
            ).copy()
            st.caption(
                "Estimated residents and key places gaining access are model estimates for each site on its own. "
                "Current walk is the estimated time from the place to the nearest existing bus stop."
            )
            st.dataframe(shown.drop(columns=["destination_id"]), hide_index=True, use_container_width=True)
            top = ranked_sites.iloc[0]
            st.info(
                f"Suggested starting point: **{clean_web_text(str(top['Place']))}** "
                f"({clean_web_text(str(top['Type']))}). "
                f"An estimated {top['Estimated residents gaining access']:,.0f} residents "
                f"and {int(top['Key places gaining access'])} other listed key places "
                "may gain access within a short walk."
            )
            all_candidate_sites["choice_label"] = all_candidate_sites["destination_type"].astype(str) + " — " + all_candidate_sites["destination_name"].astype(str)
            choice_lookup = all_candidate_sites.set_index("destination_id")["choice_label"].to_dict()
            selected_ids = st.multiselect(
                "Choose proposed stop sites",
                ranked_sites["destination_id"].tolist(),
                default=[top["destination_id"]],
                format_func=lambda key: clean_web_text(choice_lookup.get(key, key)),
                help="The highest-ranked suggestion is selected as a starting point. Add or remove any number of sites.",
            )
            candidates = all_candidate_sites[
                all_candidate_sites["destination_id"].isin(selected_ids)
            ].copy()

    map_placeholder = st.empty()
    preview_snapped_locations = {}
    if not candidates.empty:
        _, node_ids, node_xy, tree, mean_lat, _, _, _ = _load_network_context()
        for _, proposed in candidates.iterrows():
            node, _ = _snap_to_network(
                proposed["destination_lon"], proposed["destination_lat"],
                node_ids, node_xy, tree, mean_lat,
            )
            node_index = node_ids.index(node)
            preview_snapped_locations[proposed["destination_id"]] = (
                float(node_xy[node_index, 1]), float(node_xy[node_index, 0])
            )
    map_view = _map_for_area(
        area_name, all_areas, all_stops, candidates,
        snapped_locations=preview_snapped_locations,
    )
    with map_placeholder.container():
        components.html(map_view.get_root().render(), height=580, scrolling=False)

    selected_ids = set(candidates["destination_id"].tolist())
    area_destinations_all = destinations[destinations["mukim"] == area_name].copy()
    baseline_unserved_destinations = area_destinations_all[
        (area_destinations_all["estimated_walking_distance_m"] > WALK_LIMIT_M)
        & (~area_destinations_all["destination_id"].isin(selected_ids))
    ]
    current_area = area_options.loc[area_options["match_name"] == area_name].iloc[0]

    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown("**BEFORE — current bus stops**")
        st.write(display_names.get(area_name, area_name))
        st.metric("Current TAI score", f"{float(current_area['TAI']):.3f}")
        st.caption(
            f"{current_area['TAI_category']} baseline access group. The scenario score "
            "appears in the AFTER panel when the simulation runs."
        )
        st.metric("Key places outside a short walk", f"{len(baseline_unserved_destinations):,}")
    with after_col:
        st.markdown("**AFTER — selected proposed stops**")
        st.caption(f"{len(candidates)} proposed stop(s) selected")
        after_tai = st.empty()
        after_destination_gap = st.empty()
        destination_types_placeholder = st.empty()

    run_scenario = st.button(
        "Run simulation",
        type="primary",
        disabled=candidates.empty,
        use_container_width=True,
    )
    if not run_scenario or candidates.empty:
        return (page_export, page_export_name) if not page_export.empty else None

    try:
        with st.spinner("Calculating walking distances and population estimates..."):
            graph, node_ids, node_xy, tree, mean_lat, _, base_sources, base_distances = _load_network_context()
            candidate_sources = []
            snapped_locations = {}
            for _, proposed in candidates.iterrows():
                candidate_node, candidate_gap = _snap_to_network(
                    proposed["destination_lon"], proposed["destination_lat"],
                    node_ids, node_xy, tree, mean_lat,
                )
                candidate_index = node_ids.index(candidate_node)
                snapped_lon, snapped_lat = node_xy[candidate_index]
                candidate_sources.append((candidate_node, candidate_gap))
                snapped_locations[proposed["destination_id"]] = (
                    float(snapped_lat), float(snapped_lon)
                )
            scenario_distances = _distances_from_sources(
                graph, base_sources + candidate_sources
            )

            pop_lon, pop_lat, population, total_population, population_source = _load_population_cells(area_name)
            if len(population) == 0:
                st.warning("No usable population-grid cells were found for this area.")
                return
            query = np.column_stack((pop_lon * math.cos(math.radians(mean_lat)), pop_lat))
            _, pop_indices = tree.query(query)
            pop_nodes = [node_ids[int(index)] for index in pop_indices]
            pop_offsets = np.asarray([
                _distance_m(lon, lat, node_xy[int(index), 0], node_xy[int(index), 1])
                for lon, lat, index in zip(pop_lon, pop_lat, pop_indices)
            ])
            base_pop_dist = np.asarray([
                base_distances.get(node, float("inf")) for node in pop_nodes
            ]) + pop_offsets
            new_pop_dist = np.asarray([
                scenario_distances.get(node, float("inf")) for node in pop_nodes
            ]) + pop_offsets
            base_covered = base_pop_dist <= WALK_LIMIT_M
            new_covered = new_pop_dist <= WALK_LIMIT_M
            base_population = float(population[base_covered].sum())
            scenario_population = float(population[new_covered].sum())
            gained_population = float(population[(~base_covered) & new_covered].sum())
            routable_after = np.isfinite(new_pop_dist)
            scenario_mean_walk = (
                float(np.average(
                    new_pop_dist[routable_after],
                    weights=population[routable_after],
                ))
                if routable_after.any() and population[routable_after].sum() > 0
                else float(pd.to_numeric(tai["mean_walk_distance_m_routable_population"], errors="coerce").max())
            )
            scenario_coverage_percent = (
                100 * scenario_population / total_population if total_population else 0.0
            )
            area_dest = destinations[
                (destinations["mukim"] == area_name)
                & (~destinations["destination_id"].isin(selected_ids))
            ].copy()
            d_lons = pd.to_numeric(area_dest["destination_lon"], errors="coerce").to_numpy(float)
            d_lats = pd.to_numeric(area_dest["destination_lat"], errors="coerce").to_numpy(float)
            _, d_indices = tree.query(np.column_stack((d_lons * math.cos(math.radians(mean_lat)), d_lats)))
            d_nodes = [node_ids[int(index)] for index in d_indices]
            d_offsets = np.asarray([
                _distance_m(lon, lat, node_xy[int(index), 0], node_xy[int(index), 1])
                for lon, lat, index in zip(d_lons, d_lats, d_indices)
            ])
            d_base = np.asarray([
                base_distances.get(node, float("inf")) for node in d_nodes
            ]) + d_offsets
            d_new = np.asarray([
                scenario_distances.get(node, float("inf")) for node in d_nodes
            ]) + d_offsets
            d_base_covered = d_base <= WALK_LIMIT_M
            d_new_covered = d_new <= WALK_LIMIT_M
            candidate_place_counts = _candidate_destination_counts(
                graph, candidates, destinations, node_ids, node_xy, tree, mean_lat
            )
            scenario_tai = _scenario_tai_score(
                area_name,
                len(candidates),
                scenario_coverage_percent,
                scenario_mean_walk,
                candidate_place_counts,
            )
            newly_reached = (~d_base_covered) & d_new_covered
            new_destinations = int(newly_reached.sum())
            remaining_destinations = int((~d_new_covered).sum())
            # Measure each stop's unique contribution by removing it from the
            # combined scenario. This avoids counting overlapping benefits twice.
            site_impacts = []
            for candidate_index, (_, proposed) in enumerate(candidates.iterrows()):
                other_sources = [
                    source for source_index, source in enumerate(candidate_sources)
                    if source_index != candidate_index
                ]
                without_distances = _distances_from_sources(
                    graph, base_sources + other_sources
                )
                without_pop_dist = np.asarray([
                    without_distances.get(node, float("inf")) for node in pop_nodes
                ]) + pop_offsets
                without_pop_covered = without_pop_dist <= WALK_LIMIT_M
                unique_resident_cells = new_covered & ~without_pop_covered
                unique_residents = float(population[unique_resident_cells].sum())
                if unique_residents > 0:
                    before_walk_min = float(np.average(
                        base_pop_dist[unique_resident_cells],
                        weights=population[unique_resident_cells],
                    )) / WALKING_SPEED_M_PER_MIN
                    after_walk_min = float(np.average(
                        new_pop_dist[unique_resident_cells],
                        weights=population[unique_resident_cells],
                    )) / WALKING_SPEED_M_PER_MIN
                else:
                    before_walk_min = after_walk_min = float("nan")
                without_dest_dist = np.asarray([
                    without_distances.get(node, float("inf")) for node in d_nodes
                ]) + d_offsets
                without_dest_covered = without_dest_dist <= WALK_LIMIT_M
                unique_places = int((d_new_covered & ~without_dest_covered).sum())
                site_impacts.append({
                    "Suggested stop near": str(proposed["destination_name"]),
                    "Residents gained by this stop": round(unique_residents),
                    "Other key places gained by this stop": unique_places,
                    "Average walk before (min)": round(before_walk_min, 1) if np.isfinite(before_walk_min) else "—",
                    "Average walk after (min)": round(after_walk_min, 1) if np.isfinite(after_walk_min) else "—",
                })
            destination_gains_by_type = (
                area_dest.loc[newly_reached, "destination_type"]
                .value_counts()
                .reindex(["School", "Clinic", "Hospital", "Mall"], fill_value=0)
            )
            st.session_state["recommendation_simulation_result"] = {
                "area": str(area_name),
                "sites": [str(name) for name in candidates["destination_name"].tolist()],
                "tai_before": float(current_area["TAI"]),
                "tai_after": float(scenario_tai) if np.isfinite(scenario_tai) else None,
                "population_before": int(round(base_population)),
                "population_after": int(round(scenario_population)),
                "residents_gained": int(round(gained_population)),
                "key_places_gained": int(new_destinations),
                "place_gains_by_type": {
                    str(place_type): int(destination_gains_by_type.get(place_type, 0))
                    for place_type in ["School", "Clinic", "Hospital", "Mall"]
                },
            }
            after_tai.metric(
                "Scenario TAI score",
                f"{scenario_tai:.3f}" if np.isfinite(scenario_tai) else "Not available",
                delta=(
                    f"{scenario_tai - float(current_area['TAI']):+.3f} from current"
                    if np.isfinite(scenario_tai) else None
                ),
                help=(
                    "Recalculated with the same four equally weighted TAI indicators. "
                    "The original Penang normalisation ranges are held fixed; the TAI "
                    "rank and Low/Medium/High group are not recalculated."
                ),
            )
            after_destination_gap.metric(
                "Key places still outside a short walk",
                f"{remaining_destinations:,}",
                delta=f"−{new_destinations} newly within range",
            )
            st.subheader("Residents within a short walk")
            st.caption(
                "Estimated residents living within a 400 m walking route of a bus stop. Before uses current stops. "
                "After includes the selected proposed stops. The difference is the estimated number newly within range."
            )
            st.bar_chart(
                pd.DataFrame(
                    {"Estimated residents": [base_population, scenario_population]},
                    index=["Before", "After"],
                ),
                y_label="Estimated residents",
                color="#3678c8",
            )
            with destination_types_placeholder.container():
                st.subheader("Key places newly within a short walk")
                place_columns = st.columns(4)
                for column, place_type in zip(
                    place_columns, ["School", "Clinic", "Hospital", "Mall"]
                ):
                    with column:
                        st.metric(
                            f"{place_type}s",
                            int(destination_gains_by_type.get(place_type, 0)),
                        )
            st.subheader("Contribution of each proposed stop")
            st.caption(
                "These are people and other listed places that become reachable only "
                "when this stop is included, while the other selected stops stay. "
                "Overlapping reach is not counted twice. Before and after times are "
                "average walking times for the residents gained by that stop."
            )
            st.dataframe(
                pd.DataFrame(site_impacts),
                hide_index=True,
                use_container_width=True,
            )
            gained_mask = (~base_covered) & new_covered
            gained_cells = pd.DataFrame({
                "longitude": pop_lon[gained_mask],
                "latitude": pop_lat[gained_mask],
                "population": population[gained_mask],
            })
            updated_map = _map_for_area(
                area_name,
                all_areas,
                all_stops,
                candidates,
                snapped_locations=snapped_locations,
                gained_cells=gained_cells,
            )
            with map_placeholder.container():
                components.html(updated_map.get_root().render(), height=580, scrolling=False)
            page_export = pd.DataFrame(site_impacts)
            page_export.insert(0, "Mukim", labels.get(area_name, area_name))
            page_export["Proposed stops together"] = ", ".join(candidates["destination_name"].astype(str))
            page_export["Combined TAI before"] = float(current_area["TAI"])
            page_export["Combined TAI after"] = float(scenario_tai) if np.isfinite(scenario_tai) else None
            page_export["Residents gained total"] = int(round(gained_population))
            page_export["Key places gained total"] = int(new_destinations)
            page_export_name = "scenario_stop_contributions.csv"
    except ImportError as error:
        st.error(
            "Windows blocked Rasterio, which is needed for population-based "
            "simulation. Ask your administrator to allow the Rasterio files. "
            f"Details: {error}"
        )
        return
    except (OSError, ValueError, KeyError, nx.NetworkXException) as error:
        st.error(f"Could not calculate this scenario: {clean_web_text(str(error))}")
        return

    return page_export, page_export_name
