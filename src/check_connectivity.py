from pathlib import Path
import re
import unicodedata

import folium
import networkx as nx
import osmnx as ox
import pandas as pd
from pyproj import Geod
from shapely import wkt


DATA = Path(r"D:\fyp\data\processed")
GRAPH_FILE = DATA / "walking_network_candidate.graphml"
STOPS_FILE = DATA / "stops_penang.csv"
RESULTS_FILE = DATA / "connectivity_by_destination.csv"
MAP_FILE = DATA / "connectivity_flags_map.html"

DESTINATION_FILES = {
    "School": "schools_penang.csv",
    "Hospital": "hospitals_penang.csv",
    "Clinic": "clinics_penang.csv",
    "Mall": "malls_penang.csv",
}

DUPLICATE_DISTANCE_M = 100
GEOD = Geod(ellps="WGS84")


def clean_name(value):
    """Standardise a name before checking for duplicate records."""
    if pd.isna(value):
        return ""

    name = unicodedata.normalize("NFKC", str(value))
    name = re.sub(r"\s+", " ", name)
    return name.strip().casefold()


def read_point_from_source(destination_type, destination_id):
    """Read a destination geometry using its ID and source CSV row."""
    row_number = int(destination_id.rsplit("_", 1)[1]) - 1
    source_file = DATA / DESTINATION_FILES[destination_type]
    source_data = pd.read_csv(source_file)

    if row_number < 0 or row_number >= len(source_data):
        raise IndexError(
            f"Destination ID {destination_id} does not match "
            f"a row in {source_file}"
        )

    return wkt.loads(source_data.iloc[row_number]["geometry"]).representative_point()


def check_flagged_locations():
    """Print flagged locations and save an HTML map for visual review."""
    for file_path in [GRAPH_FILE, STOPS_FILE, RESULTS_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    graph = ox.load_graphml(GRAPH_FILE)
    stops = pd.read_csv(STOPS_FILE)
    results = pd.read_csv(RESULTS_FILE)

    stop_nodes = ox.distance.nearest_nodes(
        graph,
        X=stops["stop_lon"].astype(float).tolist(),
        Y=stops["stop_lat"].astype(float).tolist(),
    )

    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    stop_components = [
        component
        for component in components
        if any(node in component for node in stop_nodes)
    ]

    if not stop_components:
        raise ValueError("No network component contains a bus stop.")

    stop_connected_nodes = set().union(*stop_components)
    connected_graph = graph.subgraph(stop_connected_nodes).copy()

    no_route = results[
        results["status"].str.contains("No route", case=False, na=False)
    ]
    snap_review = results[
        results["snap_needs_review"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    ]

    review_records = pd.concat([no_route, snap_review]).drop_duplicates(
        subset=["destination_id"]
    )

    print(f"Destinations with no route: {len(no_route)}")
    print(f"Destinations needing snap review: {len(snap_review)}")
    print(f"Flagged destination records: {len(review_records)}")
    print(
        "Network nodes connected to at least one bus stop: "
        f"{len(stop_connected_nodes)}"
    )

    map_records = []

    for _, result in review_records.iterrows():
        destination_type = result["destination_type"]
        point = read_point_from_source(
            destination_type,
            result["destination_id"],
        )

        nearest_node = ox.distance.nearest_nodes(
            connected_graph,
            X=point.x,
            Y=point.y,
        )

        node_x = float(connected_graph.nodes[int(nearest_node)]["x"])
        node_y = float(connected_graph.nodes[int(nearest_node)]["y"])

        _, _, distance_m = GEOD.inv(
            point.x,
            point.y,
            node_x,
            node_y,
        )

        is_no_route = "No route" in str(result["status"])
        colour = "red" if is_no_route else "orange"

        print(f"\n{destination_type}: {result['destination_name']}")
        print(f"  Status: {result['status']}")
        print(f"  Destination coordinates: {point.y:.6f}, {point.x:.6f}")
        print(
            "  Distance to nearest bus-stop-connected network node: "
            f"{distance_m:.1f} m"
        )

        map_records.append(
            {
                "name": str(result["destination_name"]),
                "destination_type": destination_type,
                "status": str(result["status"]),
                "colour": colour,
                "destination_lat": point.y,
                "destination_lon": point.x,
                "network_lat": node_y,
                "network_lon": node_x,
                "distance_m": distance_m,
            }
        )

    if map_records:
        centre_lat = sum(item["destination_lat"] for item in map_records) / len(map_records)
        centre_lon = sum(item["destination_lon"] for item in map_records) / len(map_records)

        map_view = folium.Map(
            location=[centre_lat, centre_lon],
            zoom_start=10,
            tiles=None,
        )

        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Esri, DeLorme, HERE, USGS, Intermap, iPC, NRCAN, Esri Japan, "
            "METI, Esri China (Hong Kong), Esri (Thailand), MapmyIndia, TomTom",
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
            attr="Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            name="Esri Satellite",
            overlay=False,
            control=True,
            show=False,
        ).add_to(map_view)

        for item in map_records:
            popup_text = (
                f"<b>{item['name']}</b><br>"
                f"Type: {item['destination_type']}<br>"
                f"Status: {item['status']}<br>"
                "Distance to nearest bus-stop-connected network node: "
                f"{item['distance_m']:.1f} m"
            )

            folium.PolyLine(
                locations=[
                    [item["destination_lat"], item["destination_lon"]],
                    [item["network_lat"], item["network_lon"]],
                ],
                color=item["colour"],
                weight=2,
                opacity=0.8,
            ).add_to(map_view)

            folium.CircleMarker(
                location=[item["network_lat"], item["network_lon"]],
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.9,
                tooltip="Nearest bus-stop-connected network node",
            ).add_to(map_view)

            folium.CircleMarker(
                location=[item["destination_lat"], item["destination_lon"]],
                radius=7,
                color=item["colour"],
                fill=True,
                fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=350),
                tooltip=f"{item['destination_type']}: {item['name']}",
            ).add_to(map_view)

        legend = """
        <div style="
            position: fixed; bottom: 28px; left: 28px; z-index: 9999;
            background: white; padding: 10px 12px; border: 1px solid #888;
            border-radius: 4px; font-size: 13px;">
            <b>Connectivity review</b><br>
            <span style="color: red;">●</span> No route in candidate network<br>
            <span style="color: orange;">●</span> Snap location needs review<br>
            <span style="color: blue;">●</span> Nearest bus-stop-connected network node<br>
            Lines show straight-line distance to that network node.
        </div>
        """
        map_view.get_root().html.add_child(folium.Element(legend))
        folium.LayerControl(collapsed=False).add_to(map_view)
        map_view.save(MAP_FILE)
        print(f"\nSaved review map to: {MAP_FILE}")
    else:
        print("\nNo flagged records were found, so no map was created.")


def check_same_name_duplicates():
    """Print same-category, same-name records within 100 metres."""
    print(
        "\nSame-category, same-name records within "
        f"{DUPLICATE_DISTANCE_M} metres:"
    )

    found_duplicate = False

    for destination_type, filename in DESTINATION_FILES.items():
        source_file = DATA / filename
        if not source_file.exists():
            raise FileNotFoundError(f"Required file not found: {source_file}")

        destinations = pd.read_csv(source_file)
        records = []

        for index, row in destinations.iterrows():
            name = clean_name(row.get("name"))
            if not name:
                continue

            point = wkt.loads(row["geometry"]).representative_point()
            records.append(
                {
                    "row_number": index + 1,
                    "name": str(row["name"]).strip(),
                    "normalised_name": name,
                    "longitude": point.x,
                    "latitude": point.y,
                }
            )

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                first = records[i]
                second = records[j]

                if first["normalised_name"] != second["normalised_name"]:
                    continue

                _, _, distance_m = GEOD.inv(
                    first["longitude"],
                    first["latitude"],
                    second["longitude"],
                    second["latitude"],
                )

                if distance_m <= DUPLICATE_DISTANCE_M:
                    found_duplicate = True
                    print(
                        f"{destination_type}: rows "
                        f"{first['row_number']} and {second['row_number']}, "
                        f"name '{first['name']}', "
                        f"distance {distance_m:.1f} m"
                    )

    if not found_duplicate:
        print("No matching records were found.")


def main():
    check_flagged_locations()
    check_same_name_duplicates()


if __name__ == "__main__":
    main()



