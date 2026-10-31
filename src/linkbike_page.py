import math
from html import escape
from pathlib import Path

import folium
import geopandas as gpd
import json
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import Fullscreen, MarkerCluster
from scipy.spatial import cKDTree
from folium.plugins import GroupedLayerControl
from ui_helpers import clean_web_text
from scenario_simulation import _load_population_cells


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
LINKBIKE_RESULTS_FILE = DATA / "linkbike_gap_coverage.csv"
POIS_FILE = DATA / "connectivity_by_destination.csv"
STOPS_FILE = DATA / "stops_penang.csv"
STOP_MAPPING_FILE = DATA / "stops_with_nearest_node_candidate.csv"
NETWORK_FILE = DATA / "walking_network_candidate.graphml"
STATIONS_FILE = ROOT / "data" / "raw" / "linkbike_stations.csv"
STATION_COLUMNS = ("station_name", "latitude", "longitude")


def _remember_map_row(widget_key, kind):
    event = st.session_state.get(widget_key)
    selection = getattr(event, "selection", None)
    rows = getattr(selection, "rows", [])
    if rows:
        st.session_state["linkbike_map_selection"] = {"kind": kind, "row": int(rows[0]), "table_key": widget_key}
    else:
        current = st.session_state.get("linkbike_map_selection")
        if current and current.get("kind") == kind:
            st.session_state["linkbike_map_selection"] = None


@st.cache_data
def _load_pois():
    pois = pd.read_csv(POIS_FILE)
    required = ["destination_type", "destination_name", "destination_lat", "destination_lon"]
    missing = [column for column in required if column not in pois.columns]
    if missing:
        raise ValueError(f"POI data is missing required columns: {', '.join(missing)}")
    pois = pois[required].copy()
    pois["destination_type"] = pois["destination_type"].fillna("Other").astype(str)
    pois["destination_name"] = pois["destination_name"].fillna("Unnamed destination").astype(str)
    for column in ("destination_lat", "destination_lon"):
        pois[column] = pd.to_numeric(pois[column], errors="coerce")
    pois = pois.dropna(subset=["destination_lat", "destination_lon"])
    return pois.reset_index(drop=True)


@st.cache_data
def _load_bus_stops():
    stops = pd.read_csv(STOPS_FILE, dtype={"stop_id": str})
    mapping = pd.read_csv(
        STOP_MAPPING_FILE,
        dtype={"stop_id": str, "nearest_node": str},
    )
    return stops.merge(
        mapping[["stop_id", "nearest_node"]],
        on="stop_id",
        how="left",
        validate="one_to_one",
    )


@st.cache_resource
def _load_walking_network():
    graph = nx.read_graphml(NETWORK_FILE).to_undirected()
    for _, _, edge_data in graph.edges(data=True):
        edge_data["length"] = float(edge_data["length"])

    nodes = list(graph.nodes)
    coordinates = np.array(
        [[float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])] for node in nodes]
    )
    mean_latitude = float(coordinates[:, 1].mean())
    scaled_coordinates = coordinates.copy()
    scaled_coordinates[:, 0] *= math.cos(math.radians(mean_latitude))
    return graph, nodes, coordinates, cKDTree(scaled_coordinates), mean_latitude


def _distance_metres(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return 6_371_000 * 2 * math.asin(math.sqrt(value))


def _normalise_stations(data):
    aliases = {
        "station_name": ("station_name", "name", "station"),
        "latitude": ("latitude", "lat"),
        "longitude": ("longitude", "lon", "lng"),
    }
    lower_columns = {str(column).strip().lower(): column for column in data.columns}
    selected = {}
    for target, choices in aliases.items():
        source = next((lower_columns[name] for name in choices if name in lower_columns), None)
        if source is None:
            raise ValueError(
                "CSV needs station_name, latitude and longitude columns. "
                "Name, lat and lon are also accepted."
            )
        selected[target] = source

    include_column = next(
        (
            column
            for column in data.columns
            if str(column).strip().lower() in {"analysis_include", "include_in_analysis"}
        ),
        None,
    )
    if include_column is not None:
        include = data[include_column].astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "include"}
        )
        data = data.loc[include].copy()
        if data.empty:
            raise ValueError("No stations are marked for analysis in the CSV.")

    stations = data[[selected[column] for column in STATION_COLUMNS]].copy()
    stations.columns = STATION_COLUMNS
    stations["station_name"] = stations["station_name"].fillna("Unnamed station").astype(str).str.strip()
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude"])
    valid = stations["latitude"].between(-90, 90) & stations["longitude"].between(-180, 180)
    stations = stations.loc[valid].reset_index(drop=True)
    if stations.empty:
        raise ValueError("No valid station coordinates were found in the CSV.")
    return stations


def _snap_coordinates(data, lat_column, lon_column, nodes, coordinates, tree, mean_latitude):
    snapped_nodes, snap_distances = [], []
    for row in data.itertuples(index=False):
        latitude = float(getattr(row, lat_column))
        longitude = float(getattr(row, lon_column))
        query = [longitude * math.cos(math.radians(mean_latitude)), latitude]
        _, index = tree.query(query)
        node = nodes[int(index)]
        node_lon, node_lat = coordinates[int(index)]
        snapped_nodes.append(node)
        snap_distances.append(_distance_metres(latitude, longitude, node_lat, node_lon))
    return snapped_nodes, snap_distances


def _analyse(stations, all_stops, threshold_m, study_radius_m=None):
    graph, nodes, coordinates, tree, mean_latitude = _load_walking_network()
    stations = stations.copy()
    stations["station_node"], stations["station_snap_m"] = _snap_coordinates(
        stations, "latitude", "longitude", nodes, coordinates, tree, mean_latitude
    )

    stops = all_stops.dropna(subset=["nearest_node", "stop_lat", "stop_lon"]).copy()
    stops["nearest_node"] = stops["nearest_node"].astype(str)
    stops = stops[stops["nearest_node"].isin(graph)].copy()
    if stops.empty:
        raise ValueError("No bus stops have a matching walking-network node.")

    stops = stops.reset_index(drop=True)
    stops["stop_snap_m"] = [
        _distance_metres(
            float(stop.stop_lat), float(stop.stop_lon),
            float(graph.nodes[str(stop.nearest_node)]["y"]),
            float(graph.nodes[str(stop.nearest_node)]["x"]),
        )
        for stop in stops.itertuples(index=False)
    ]

    if study_radius_m is None:
        local_stops = stops.reset_index(drop=True)
    else:
        nearby_stop_indexes = set()
        for station in stations.itertuples(index=False):
            lengths = nx.single_source_dijkstra_path_length(
                graph, station.station_node, cutoff=study_radius_m, weight="length"
            )
            for stop_index, stop in enumerate(stops.itertuples(index=False)):
                network_distance = lengths.get(str(stop.nearest_node))
                if network_distance is None:
                    continue
                total_distance = (
                    float(station.station_snap_m)
                    + float(network_distance)
                    + float(stop.stop_snap_m)
                )
                if total_distance <= study_radius_m:
                    nearby_stop_indexes.add(stop_index)
        local_stops = stops.loc[sorted(nearby_stop_indexes)].reset_index(drop=True)
        if local_stops.empty:
            raise ValueError("No bus stops fall within the selected walking-distance study area around the stations.")

    pairs = []
    max_stop_snap = float(local_stops["stop_snap_m"].max())
    for station in stations.itertuples():
        cutoff = threshold_m + float(station.station_snap_m) + max_stop_snap
        lengths = nx.single_source_dijkstra_path_length(
            graph, station.station_node, cutoff=cutoff, weight="length"
        )
        for stop in local_stops.itertuples(index=False):
            network_distance = lengths.get(str(stop.nearest_node))
            if network_distance is None:
                continue
            total_distance = (
                float(station.station_snap_m)
                + float(network_distance)
                + float(stop.stop_snap_m)
            )
            if total_distance <= threshold_m:
                pairs.append({
                    "station_index": station.Index,
                    "station_name": station.station_name,
                    "station_node": station.station_node,
                    "stop_id": str(stop.stop_id),
                    "stop_name": str(stop.stop_name),
                    "stop_node": str(stop.nearest_node),
                    "walking_distance_m": total_distance,
                })

    pairs = pd.DataFrame(pairs)
    best_for_station = {}
    best_for_stop = {}
    if not pairs.empty:
        pairs = pairs.sort_values("walking_distance_m")
        best_for_station = pairs.drop_duplicates("station_index").set_index("station_index").to_dict("index")
        best_for_stop = pairs.drop_duplicates("stop_id").set_index("stop_id").to_dict("index")

    station_status = stations.copy()
    station_status["nearby_stop"] = station_status.index.map(
        lambda index: best_for_station.get(index, {}).get("stop_name")
    )
    station_status["walking_distance_m"] = station_status.index.map(
        lambda index: best_for_station.get(index, {}).get("walking_distance_m")
    )

    stop_status = local_stops.copy()
    stop_status["nearest_linkbike_station"] = stop_status["stop_id"].astype(str).map(
        lambda stop_id: best_for_stop.get(stop_id, {}).get("station_name")
    )
    stop_status["walking_distance_m"] = stop_status["stop_id"].astype(str).map(
        lambda stop_id: best_for_stop.get(stop_id, {}).get("walking_distance_m")
    )
    stop_status["status"] = np.where(
        stop_status["walking_distance_m"].notna(),
        f"Within {threshold_m} m",
        f"No LinkBike station within {threshold_m} m",
    )

    return graph, stations, station_status, stop_status, best_for_station


def _make_map(graph, stations, station_status, stops, best_for_station, show_all_stops=False, pois=None, selected_point=None):
    centre = [float(stations["latitude"].mean()), float(stations["longitude"].mean())]
    map_view = folium.Map(location=centre, zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Street map", show=True,
    ).add_to(map_view)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite", show=False,
    ).add_to(map_view)

    station_lookup = stations.reset_index(drop=True)
    stop_lookup = stops.assign(stop_id=stops["stop_id"].astype(str)).set_index("stop_id")
    for index, connection in best_for_station.items():
        try:
            route = nx.shortest_path(
                graph, connection["station_node"], connection["stop_node"], weight="length"
            )
            station = station_lookup.iloc[int(index)]
            stop = stop_lookup.loc[str(connection["stop_id"])]
            points = [
                [float(station.latitude), float(station.longitude)],
                *[
                    [float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"])]
                    for node in route
                ],
                [float(stop.stop_lat), float(stop.stop_lon)],
            ]
            if len(points) > 1:
                folium.PolyLine(
                    points, color="#ffffff", weight=9, opacity=0.95,
                ).add_to(map_view)
                folium.PolyLine(
                    points, color="#7c3aed", weight=6, opacity=0.95,
                    tooltip=f"Walking network · {connection['walking_distance_m']:.0f} m",
                ).add_to(map_view)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    for stop in stops.itertuples(index=False):
        is_connected = pd.notna(stop.walking_distance_m)
        is_selected = (
            selected_point is not None
            and selected_point.get("kind") == "stop"
            and str(stop.stop_id) == str(selected_point.get("id"))
        )
        if not is_connected and not show_all_stops and not is_selected:
            continue
        folium.Marker(
            [float(stop.stop_lat), float(stop.stop_lon)],
            tooltip=escape(clean_web_text(str(stop.stop_name))) + (
                f" · {stop.walking_distance_m:.0f} m walking distance"
                if is_connected else " · no LinkBike station within threshold"
            ),
            icon=folium.Icon(
                color="orange" if is_selected else ("blue" if is_connected else "gray"), icon="bus", prefix="fa"
            ),
        ).add_to(map_view)

    if pois is not None and not pois.empty:
        poi_styles = {
            "Mall": ("darkpurple", "shopping-cart"),
            "Hospital": ("red", "plus-square"),
            "School": ("blue", "graduation-cap"),
            "Clinic": ("cadetblue", "medkit"),
        }
        map_view.get_root().header.add_child(
            folium.Element("<style>.poi-cluster-hidden{display:none!important;}</style>")
        )
        poi_group = MarkerCluster(
            name="Common destinations",
            options={"disableClusteringAtZoom": 14},
            icon_create_function=(
                "function(cluster) { return L.divIcon({"
                "html: '', className: 'poi-cluster-hidden', iconSize: [0, 0]}); }"
            ),
        ).add_to(map_view)
        for poi in pois.itertuples(index=False):
            colour, icon = poi_styles.get(str(poi.destination_type), ("gray", "map-marker"))
            popup = (
                f"<div style='width:250px;font-size:14px;line-height:1.5;padding:6px;'>"
                f"<b>{escape(clean_web_text(str(poi.destination_name)))}</b><br>"
                f"{escape(clean_web_text(str(poi.destination_type)))}</div>"
            )
            folium.Marker(
                [float(poi.destination_lat), float(poi.destination_lon)],
                tooltip=f"{escape(clean_web_text(str(poi.destination_type)))}: {escape(clean_web_text(str(poi.destination_name)))}",
                popup=folium.Popup(popup, min_width=280, max_width=360),
                icon=folium.Icon(color=colour, icon=icon, prefix="fa"),
            ).add_to(poi_group)

    for station in station_status.itertuples(index=False):
        connected = pd.notna(station.walking_distance_m)
        is_selected = (
            selected_point is not None
            and selected_point.get("kind") == "station"
            and str(station.station_name) == str(selected_point.get("name"))
        )
        folium.Marker(
            [float(station.latitude), float(station.longitude)],
            tooltip=escape(clean_web_text(str(station.station_name))),
            popup=folium.Popup(
                f"<div style='width:260px;font-size:14px;line-height:1.5;padding:6px;'>"
                f"<b>{escape(clean_web_text(str(station.station_name)))}</b><br>"
                + (f"Nearest bus stop: {escape(clean_web_text(str(station.nearby_stop)))}<br>"
                   f"{station.walking_distance_m:.0f} m walking distance"
                   if connected else "No bus stop within the selected walking distance")
                + "</div>",
                min_width=280,
                max_width=360,
            ),
            icon=folium.Icon(color="orange" if is_selected else ("darkgreen" if connected else "red"), icon="bicycle", prefix="fa"),
        ).add_to(map_view)

    if selected_point is not None:
        folium.CircleMarker(
            [selected_point["lat"], selected_point["lon"]],
            radius=18,
            color="#f59e0b",
            weight=4,
            fill=False,
            tooltip=f"Selected: {escape(clean_web_text(str(selected_point['label'])))}",
        ).add_to(map_view)

    Fullscreen(position="topright").add_to(map_view)
    folium.LayerControl(collapsed=True).add_to(map_view)
    visible_stops = stops if show_all_stops else stops[stops["walking_distance_m"].notna()]
    fit_points = stations[["latitude", "longitude"]].rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )
    if not visible_stops.empty:
        stop_points = visible_stops[["stop_lat", "stop_lon"]].rename(
            columns={"stop_lat": "lat", "stop_lon": "lon"}
        )
        fit_points = pd.concat([fit_points, stop_points], ignore_index=True)
    if pois is not None and not pois.empty:
        poi_points = pois[["destination_lat", "destination_lon"]].rename(
            columns={"destination_lat": "lat", "destination_lon": "lon"}
        )
        fit_points = pd.concat([fit_points, poi_points], ignore_index=True)
    bounds = [
        [float(fit_points["lat"].min()), float(fit_points["lon"].min())],
        [float(fit_points["lat"].max()), float(fit_points["lon"].max())],
    ]
    if selected_point is not None:
        map_view.location = [selected_point["lat"], selected_point["lon"]]
        map_view.zoom_start = 17
    else:
        map_view.fit_bounds(bounds, padding=(24, 24))
    return map_view


ZONE_COLOURS = {
    "Bus only": "#6f72d8",
    "LinkBike only": "#16a779",
    "Both": "#40368f",
    "Neither": "#e7e7f2",
}


@st.cache_data(show_spinner="Building the Penang coverage map...")
def _four_zone_coverage(stations_json, stops_json, population_json, threshold_m, network_mtime):
    """Estimate resident access zones from the walking network and population grid."""
    _ = network_mtime
    stations = pd.DataFrame(json.loads(stations_json))
    stops = pd.DataFrame(json.loads(stops_json))
    graph, node_ids, node_xy, tree, mean_latitude = _load_walking_network()
    station_nodes, station_snap_gaps = _snap_coordinates(
        stations, "latitude", "longitude", node_ids, node_xy, tree, mean_latitude
    )
    source_stop_nodes = []
    source_stop_gaps = []
    valid_stops = stops.dropna(subset=["nearest_node", "stop_lat", "stop_lon"]).copy()
    for stop in valid_stops.itertuples(index=False):
        node = str(stop.nearest_node)
        if node not in graph:
            continue
        source_stop_nodes.append(node)
        source_stop_gaps.append(
            _distance_metres(
                float(stop.stop_lat), float(stop.stop_lon),
                float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"]),
            )
        )

    def distances_from_sources(name, source_nodes, source_gaps):
        source_costs = {}
        for node, gap in zip(source_nodes, source_gaps):
            source_costs[node] = min(source_costs.get(node, math.inf), float(gap))
        if not source_costs:
            return {}
        virtual_node = f"__four_zone_{name}__"
        graph.add_node(virtual_node)
        for node, gap in source_costs.items():
            graph.add_edge(virtual_node, node, length=gap)
        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph, virtual_node, cutoff=float(threshold_m), weight="length"
            )
        finally:
            graph.remove_node(virtual_node)
        lengths.pop(virtual_node, None)
        return lengths

    bus_lengths = distances_from_sources("bus", source_stop_nodes, source_stop_gaps)
    bike_lengths = distances_from_sources("bike", station_nodes, station_snap_gaps)

    population = pd.DataFrame(json.loads(population_json))
    if population.empty:
        raise ValueError("No population cells are available for the coverage comparison.")
    pop_lon = population["longitude"].to_numpy(float)
    pop_lat = population["latitude"].to_numpy(float)
    pop_points = np.column_stack(
        (pop_lon * math.cos(math.radians(mean_latitude)), pop_lat)
    )
    _, pop_nearest = tree.query(pop_points)
    pop_nodes = [node_ids[int(index)] for index in pop_nearest]
    pop_node_lon = node_xy[pop_nearest, 0]
    pop_node_lat = node_xy[pop_nearest, 1]
    pop_lat1 = np.radians(pop_lat)
    pop_lat2 = np.radians(pop_node_lat)
    pop_delta_lat = pop_lat2 - pop_lat1
    pop_delta_lon = np.radians(pop_node_lon - pop_lon)
    pop_haversine = (
        np.sin(pop_delta_lat / 2) ** 2
        + np.cos(pop_lat1) * np.cos(pop_lat2) * np.sin(pop_delta_lon / 2) ** 2
    )
    pop_snap_m = 6_371_000 * 2 * np.arcsin(np.sqrt(np.clip(pop_haversine, 0, 1)))
    pop_bus_m = np.array([bus_lengths.get(node, math.inf) for node in pop_nodes]) + pop_snap_m
    pop_bike_m = np.array([bike_lengths.get(node, math.inf) for node in pop_nodes]) + pop_snap_m
    population["bus_access"] = pop_bus_m <= float(threshold_m)
    population["bike_access"] = pop_bike_m <= float(threshold_m)
    population["zone"] = np.select(
        [population["bus_access"] & population["bike_access"], population["bus_access"], population["bike_access"]],
        ["Both", "Bus only", "LinkBike only"],
        default="Neither",
    )
    zone_stats = population.groupby("zone")["residents"].sum().to_dict()
    by_area_zone = population.groupby(["area", "zone"])["residents"].sum().unstack(fill_value=0)
    area_stats = {}
    for area_name, row in by_area_zone.iterrows():
        bus_only = float(row.get("Bus only", 0.0))
        linkbike_only = float(row.get("LinkBike only", 0.0))
        both = float(row.get("Both", 0.0))
        neither = float(row.get("Neither", 0.0))
        area_stats[str(area_name)] = {
            "Bus only": bus_only,
            "LinkBike only": linkbike_only,
            "Both": both,
            "Neither": neither,
            "Mukim population": bus_only + linkbike_only + both + neither,
        }
    for zone in ("Bus only", "LinkBike only", "Both", "Neither"):
        zone_stats.setdefault(zone, 0.0)
    return zone_stats, area_stats


def _make_four_zone_map(all_areas, area_stats):
    map_view = folium.Map(location=[5.35, 100.35], zoom_start=8, tiles=None, control_scale=True)
    street = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Street map", show=True,
    ).add_to(map_view)
    satellite = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite", show=False,
    ).add_to(map_view)
    coverage_layers = []
    layer_details = [
        ("Bus only", "Bus only", True),
        ("LinkBike only", "LinkBike only", False),
        ("Both", "Both", False),
        ("Neither", "Neither", False),
    ]
    colours = ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3", "#3f007d"]
    simplified_areas = all_areas.to_crs("EPSG:32647").copy()
    simplified_areas.geometry = simplified_areas.geometry.simplify(50, preserve_topology=True)
    simplified_areas = simplified_areas.to_crs("EPSG:4326")
    for title, value_key, show in layer_details:
        areas = simplified_areas.copy()
        areas["display_name"] = areas.get("display_name", areas["match_name"])
        areas["estimated_residents"] = areas["match_name"].map(
            lambda name: area_stats.get(str(name), {}).get(value_key, 0.0)
        )
        areas["resident_share_percent"] = areas.apply(
            lambda row: (
                100 * row["estimated_residents"]
                / area_stats.get(str(row["match_name"]), {}).get("Mukim population", 0.0)
                if area_stats.get(str(row["match_name"]), {}).get("Mukim population", 0.0) > 0
                else 0.0
            ),
            axis=1,
        )
        max_value = max(float(areas["estimated_residents"].max()), 1.0)
        geojson = json.loads(
            areas[["display_name", "estimated_residents", "resident_share_percent", "geometry"]]
            .to_json(drop_id=True)
        )
        layer = folium.FeatureGroup(name=title, overlay=True, control=False, show=show)

        def style_feature(feature, max_value=max_value):
            value = float(feature["properties"].get("estimated_residents", 0.0))
            if value <= 0:
                return {"fillColor": "#d9d9d9", "color": "#ffffff", "weight": 1, "fillOpacity": 0.8}
            band = max(1, min(int((value / max_value) * (len(colours) - 1)), len(colours) - 1))
            return {"fillColor": colours[band], "color": "#ffffff", "weight": 1, "fillOpacity": 0.8}

        folium.GeoJson(
            geojson,
            name=title,
            style_function=style_feature,
            tooltip=folium.GeoJsonTooltip(
                fields=["display_name", "estimated_residents", "resident_share_percent"],
                aliases=["Mukim", "Estimated residents", "Share of mukim residents (%)"],
                localize=True,
                sticky=False,
            ),
            popup=folium.GeoJsonPopup(
                fields=["display_name", "estimated_residents", "resident_share_percent"],
                aliases=["Mukim", "Estimated residents", "Share of mukim residents (%)"],
                localize=True,
            ),
        ).add_to(layer)
        layer.add_to(map_view)
        coverage_layers.append(layer)
    legend = (
        "<div style='position:fixed;bottom:22px;left:14px;z-index:9999;"
        "background:white;padding:9px 11px;border:1px solid #ccc;border-radius:7px;"
        "font:12px Arial;box-shadow:0 1px 5px #999'>Residents in selected layer<br>"
        "<span style='color:#d9d9d9'>■</span> none&nbsp;&nbsp;"
        "<span style='color:#cbc9e2'>■</span> fewer&nbsp;&nbsp;"
        "<span style='color:#3f007d'>■</span> more</div>"
    )
    map_view.get_root().html.add_child(folium.Element(legend))
    GroupedLayerControl(
        groups={"Map style": [street, satellite], "Residents by mukim": coverage_layers},
        exclusive_groups=True,
        collapsed=False,
    ).add_to(map_view)
    bounds = all_areas.to_crs("EPSG:4326").total_bounds
    map_view.fit_bounds([[float(bounds[1]), float(bounds[0])], [float(bounds[3]), float(bounds[2])]], padding=(14, 14))
    return map_view


def _render_four_zone_analysis(stations, bus_stops, all_areas, threshold_m):
    st.divider()
    st.subheader("Walking access by mukim")
    st.caption(
        f"Choose a layer to see estimated residents within {threshold_m} m on foot: "
        "bus only, LinkBike only, both, or neither. Darker mukims have more residents in that group."
    )
    st.caption("This shows walking access to stations. It does not show bike trips or confirm bus-bike transfers.")
    stations_json = stations[["station_name", "latitude", "longitude"]].to_json(orient="records")
    stop_columns = ["stop_id", "stop_lat", "stop_lon", "nearest_node"]
    stops_json = bus_stops[stop_columns].to_json(orient="records")
    network_mtime = NETWORK_FILE.stat().st_mtime_ns
    with st.spinner("Estimating residents near bus stops and LinkBike stations..."):
        population_rows = []
        try:
            for area_name in all_areas["match_name"].dropna().astype(str).unique():
                pop_lon, pop_lat, residents, _, _ = _load_population_cells(area_name)
                if len(residents):
                    population_rows.append(
                        pd.DataFrame({
                            "area": area_name,
                            "longitude": pop_lon,
                            "latitude": pop_lat,
                            "residents": residents,
                        })
                    )
        except (OSError, ValueError, KeyError, ImportError) as error:
            st.warning(f"Population comparison could not be calculated: {clean_web_text(str(error))}")
            return
        if not population_rows:
            st.warning("Population estimates are not available, so the comparison cannot be shown.")
            return
        population_json = pd.concat(population_rows, ignore_index=True).to_json(
            orient="records", double_precision=6
        )
    zone_stats, area_stats = _four_zone_coverage(
        stations_json, stops_json, population_json,
        int(threshold_m), network_mtime
    )
    components.html(
        _make_four_zone_map(all_areas, area_stats).get_root().render(),
        height=560, scrolling=False,
    )
    bus_residents = (
        zone_stats["Bus only"] + zone_stats["Both"]
    )
    combined_residents = bus_residents + zone_stats["LinkBike only"]
    added_residents = zone_stats["LinkBike only"]
    total_residents = sum(zone_stats.values())
    before_col, after_col, added_col = st.columns(3)
    before_col.metric(
        "Bus access",
        f"{bus_residents:,.0f}",
        f"{(100 * bus_residents / total_residents) if total_residents else 0:.1f}% of residents",
    )
    after_col.metric(
        "Residents near bus or LinkBike",
        f"{combined_residents:,.0f}",
        f"{(100 * combined_residents / total_residents) if total_residents else 0:.1f}% of residents",
    )
    added_col.metric(
        "Added by LinkBike",
        f"{added_residents:,.0f}",
        "estimated",
    )
    st.caption(
        "This counts residents near a bus stop, a LinkBike station, or both. "
        "Added by LinkBike counts residents near a LinkBike station but not a bus stop. "
        "These are estimates, not actual trips."
    )


def render(tai, all_areas, all_stops, scored, socioeconomic):
    st.caption(
        "LinkBike is Penang’s shared-bike service. A station is a place where a shared bicycle can be picked up or returned. "
        "This page uses the recorded station locations to check how close they are to RapidPenang bus stops by foot."
    )
    try:
        if not STATIONS_FILE.exists():
            st.error(f"LinkBike station data was not found: {STATIONS_FILE}")
            return
        station_data = pd.read_csv(STATIONS_FILE)
        stations = _normalise_stations(station_data)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(clean_web_text(str(error)))
        return

    threshold_m = st.selectbox(
        "Walking connection distance",
        [300, 400, 500],
        index=1,
        format_func=lambda value: f"{value} m",
    )
    st.caption("The selected distance defines a potential connection.")
    try:
        bus_stops = _load_bus_stops()
        graph, stations, station_status, stop_status, best_for_station = _analyse(
            stations, bus_stops, int(threshold_m), None
        )
    except (OSError, ValueError, KeyError, nx.NetworkXException) as error:
        st.error(f"Could not calculate LinkBike connections: {clean_web_text(str(error))}")
        return

    try:
        poi_data = _load_pois()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load common destinations: {clean_web_text(str(error))}")
        return

    show_all_stops = st.checkbox(
        "Show all bus stops across Penang",
        value=False,
        help="By default, the map shows only bus stops within the selected walking distance of a LinkBike station.",
    )

    stop_gap_rows = stop_status[stop_status["walking_distance_m"].isna()].reset_index(drop=True)
    station_gap_rows = station_status[station_status["walking_distance_m"].isna()].reset_index(drop=True)
    stop_table_key = f"linkbike_gap_stops_{threshold_m}"
    station_table_key = f"linkbike_gap_stations_{threshold_m}"
    selection = st.session_state.get("linkbike_map_selection")
    selected_point = None
    if selection and selection.get("table_key") == stop_table_key and selection.get("kind") == "stop":
        row = selection.get("row", -1)
        if 0 <= row < len(stop_gap_rows):
            stop = stop_gap_rows.iloc[row]
            selected_point = {
                "kind": "stop", "id": str(stop["stop_id"]),
                "lat": float(stop["stop_lat"]), "lon": float(stop["stop_lon"]),
                "label": str(stop["stop_name"]),
            }
    elif selection and selection.get("table_key") == station_table_key and selection.get("kind") == "station":
        row = selection.get("row", -1)
        if 0 <= row < len(station_gap_rows):
            station = station_gap_rows.iloc[row]
            selected_point = {
                "kind": "station", "name": str(station["station_name"]),
                "lat": float(station["latitude"]), "lon": float(station["longitude"]),
                "label": str(station["station_name"]),
            }

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("LinkBike stations", f"{len(stations):,}")
    metric2.metric("Stations near a bus stop", f"{station_status['walking_distance_m'].notna().sum():,}")
    metric3.metric("Bus stops beyond connection distance", f"{stop_status['walking_distance_m'].isna().sum():,}")

    map_col, detail_col = st.columns([1.25, 0.85], gap="medium")
    with map_col:
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:7px;margin:0 0 8px'>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#18864b;border-radius:50%;margin-right:5px'></i>LinkBike with nearby bus</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#d64545;border-radius:50%;margin-right:5px'></i>LinkBike without nearby bus</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#3478c8;border-radius:50%;margin-right:5px'></i>Bus stop with nearby LinkBike</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#999;border-radius:50%;margin-right:5px'></i>Bus stop beyond range</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:13px;height:3px;background:#7c3aed;vertical-align:middle;margin-right:5px'></i>Walking route</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#6b2f91;border-radius:50%;margin-right:5px'></i>Mall</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#d64545;border-radius:50%;margin-right:5px'></i>Hospital</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#3478c8;border-radius:50%;margin-right:5px'></i>School</span>"
            "<span style='background:#f4f6f8;border:1px solid #e0e4e8;border-radius:16px;padding:4px 9px;font-size:12px'>"
            "<i style='display:inline-block;width:9px;height:9px;background:#4f9ca0;border-radius:50%;margin-right:5px'></i>Clinic</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        components.html(_make_map(graph, stations, station_status, stop_status, best_for_station, show_all_stops, poi_data, selected_point).get_root().render(), height=560, scrolling=False)

    with detail_col:
        st.subheader("Bus stops without a nearby LinkBike connection")
        st.caption(
            "These bus stops have no LinkBike station within the selected walking distance. "
            "Select a row to show it on the map. This can help identify places to review for a connection."
        )
        gaps = stop_gap_rows[["stop_name"]].rename(columns={"stop_name": "Bus stop"})
        if gaps.empty:
            st.success("Every RapidPenang bus stop has a LinkBike station within the selected distance.")
        else:
            st.dataframe(
                gaps,
                hide_index=True,
                use_container_width=True,
                height=280,
                on_select=lambda: _remember_map_row(stop_table_key, "stop"),
                selection_mode="single-row",
                key=stop_table_key,
            )

    station_gaps = station_gap_rows[["station_name"]].rename(columns={"station_name": "LinkBike station"})
    st.subheader(f"LinkBike stations without a nearby bus stop ({len(station_gaps)})")
    if station_gaps.empty:
        st.success("Every LinkBike station has a bus stop within the selected walking distance.")
    else:
        st.caption(
            "These LinkBike stations have no bus stop within the selected walking distance. "
            "Select a row to show it on the map. This can help identify places to review for a connection."
        )
        st.dataframe(
            station_gaps,
            hide_index=True,
            use_container_width=True,
            on_select=lambda: _remember_map_row(station_table_key, "station"),
            selection_mode="single-row",
            key=station_table_key,
        )

    _render_four_zone_analysis(stations, bus_stops, all_areas, int(threshold_m))

    export_output = stop_status[[
        "stop_id", "stop_name", "stop_lat", "stop_lon",
        "nearest_linkbike_station", "walking_distance_m", "status",
    ]].copy()
    export_output["walking_distance_m"] = export_output["walking_distance_m"].round(2)
    LINKBIKE_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    export_output.to_csv(LINKBIKE_RESULTS_FILE, index=False)
    return export_output, f"linkbike_connections_{threshold_m}m.csv"































