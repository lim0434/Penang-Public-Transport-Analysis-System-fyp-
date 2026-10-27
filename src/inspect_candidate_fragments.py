from pathlib import Path

import folium
import networkx as nx
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, box

# Input files
processed = Path(r"D:\fyp\data\processed")
graph_file = processed / "walking_network_candidate.graphml"
mapping_file = processed / "stops_with_nearest_node_candidate.csv"

# Output folder and map
output_folder = processed / "candidate_stop_checks"
output_folder.mkdir(exist_ok=True)
output_map = output_folder / "all_12_stops.html"

# Load the candidate network and stop mapping
graph = nx.read_graphml(graph_file, force_multigraph=True)
simple_graph = nx.Graph(graph)

# Rank connected components from largest to smallest
components = sorted(
    nx.connected_components(simple_graph),
    key=len,
    reverse=True,
)

component_rank = {
    node: rank
    for rank, component in enumerate(components, start=1)
    for node in component
}

# Find stops outside the two largest components
stops = pd.read_csv(mapping_file, dtype={"nearest_node": str})
stops["component"] = stops["nearest_node"].map(component_rank)
stops_to_check = stops[~stops["component"].isin([1, 2])]

print(f"Stops to inspect: {len(stops_to_check)}")

# Create one map centred on Penang
map_view = folium.Map(
    location=[5.4, 100.32],
    zoom_start=9,
    tiles=None,
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="Esri Street Map",
    overlay=False,
    control=True,
    show=True,
).add_to(map_view)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="Esri Satellite Imagery",
    overlay=False,
    control=True,
    show=False,
).add_to(map_view)

# Prepare one selectable map layer for each stop
stop_areas = []

for stop in stops_to_check.itertuples():
    rank = int(stop.component)
    component_size = len(components[rank - 1])
    stop_name = str(stop.stop_name)
    stop_lat = float(stop.stop_lat)
    stop_lon = float(stop.stop_lon)

    layer = folium.FeatureGroup(
        name=f"{stop_name} — stop {stop.stop_id} — component {rank}"
    )

    # Mark the bus stop
    folium.Marker(
        [stop_lat, stop_lon],
        tooltip=stop_name,
        popup=folium.Popup(
            f'<b>{stop_name}</b><br><a href="https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={stop_lat}%2C{stop_lon}" target="_blank">Open Google Street View</a>',
            max_width=280,
        ),
        icon=folium.Icon(color="blue"),
    ).add_to(layer)

    # Mark the walking-network node matched to the stop
    node = stop.nearest_node
    node_lat = float(graph.nodes[node]["y"])
    node_lon = float(graph.nodes[node]["x"])

    folium.CircleMarker(
        [node_lat, node_lon],
        radius=5,
        color="red",
        fill=True,
        fill_opacity=1,
        tooltip="Matched walking-network node",
    ).add_to(layer)

    # Show paths in a small area around the stop
    area = box(
        stop_lon - 0.008,
        stop_lat - 0.006,
        stop_lon + 0.008,
        stop_lat + 0.006,
    )

    stop_areas.append(
        {
            "layer": layer,
            "area": area,
            "component": rank,
        }
    )

# Draw nearby network lines into the relevant stop layers
for start, end, data in simple_graph.edges(data=True):
    try:
        line = wkt.loads(data["geometry"])
    except (KeyError, TypeError, ValueError):
        line = LineString(
            [
                (
                    float(graph.nodes[start]["x"]),
                    float(graph.nodes[start]["y"]),
                ),
                (
                    float(graph.nodes[end]["x"]),
                    float(graph.nodes[end]["y"]),
                ),
            ]
        )

    line_bounds = line.bounds

    for stop_area in stop_areas:
        area = stop_area["area"]
        area_bounds = area.bounds

        # Skip lines that do not overlap this stop's map area
        if (
            line_bounds[2] < area_bounds[0]
            or line_bounds[0] > area_bounds[2]
            or line_bounds[3] < area_bounds[1]
            or line_bounds[1] > area_bounds[3]
        ):
            continue

        clipped_line = line.intersection(area)
        if clipped_line.is_empty:
            continue

        # Green marks either of the two largest components.
        # Orange marks smaller components.
        colour = (
            "green"
            if component_rank[start] in (1, 2)
            else "orange"
        )

        if clipped_line.geom_type == "LineString":
            parts = [clipped_line]
        else:
            parts = getattr(clipped_line, "geoms", [])

        for part in parts:
            if part.geom_type == "LineString":
                points = [(lat, lon) for lon, lat in part.coords]
                folium.PolyLine(
                    points,
                    color=colour,
                    weight=3,
                ).add_to(stop_area["layer"])

# Add all stop layers and the layer menu
for stop_area in stop_areas:
    stop_area["layer"].add_to(map_view)

folium.LayerControl(collapsed=False).add_to(map_view)
map_view.save(output_map)

print(f"Map saved to: {output_map}")