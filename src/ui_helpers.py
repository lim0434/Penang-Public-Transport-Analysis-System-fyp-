from pathlib import Path
import json

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
BOUNDARY_FILE = ROOT / "data" / "raw" / "boundary" / "penang_mukim_boundaries.geojson"
TAI_FILE = DATA / "tai_by_mukim.csv"
SOCIOECONOMIC_FILE = DATA / "tai_by_dun_socioeconomic.csv"
STOPS_FILE = DATA / "stops_penang.csv"

DISTRICT_PREFIX = {
    "01": "SPT",
    "02": "SPU",
    "03": "SPS",
    "04": "DTL",
    "05": "DBD",
}
DISTRICT_LABELS = {
    "SPT": "Seberang Perai Tengah",
    "SPU": "Seberang Perai Utara",
    "SPS": "Seberang Perai Selatan",
    "DTL": "Timur Laut",
    "DBD": "Barat Daya",
}

TAI_COLOURS = {
    "Low": "#d7191c",
    "Medium": "#fdae61",
    "High": "#1a9641",
    "No score": "#bdbdbd",
}


def clean_web_text(value):
    """Remove underscores and semicolons from text shown in the web app."""
    if isinstance(value, str):
        return value.replace("_", " ").replace(";", "")
    return value


def clean_web_table(frame):
    """Return a cleaned display copy without changing source or export data."""
    display = frame.copy()
    display.columns = [clean_web_text(column) for column in display.columns]
    for column in display.columns:
        dtype = display[column].dtype
        if (
            dtype == object
            or isinstance(dtype, pd.StringDtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            display[column] = display[column].astype(object).map(clean_web_text)
    if isinstance(display.index, pd.MultiIndex):
        arrays = [
            [clean_web_text(value) for value in display.index.get_level_values(level)]
            for level in range(display.index.nlevels)
        ]
        display.index = pd.MultiIndex.from_arrays(arrays, names=display.index.names)
    else:
        display.index = display.index.map(clean_web_text)
    return display


def read_geojson(path):
    """Read a GeoJSON FeatureCollection without a GDAL-backed file engine."""
    with Path(path).open("r", encoding="utf-8") as source:
        document = json.load(source)

    crs = document.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    return gpd.GeoDataFrame.from_features(document["features"], crs=crs)


def make_area_label(row):
    code = str(row["KOD_DAERAH"]).replace(".0", "").zfill(2)
    district = DISTRICT_PREFIX.get(code)
    if district is None:
        raise ValueError(f"Unknown district code: {code}")

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


@st.cache_data
def load_data():
    tai = pd.read_csv(TAI_FILE)
    tai["TAI_category"] = tai["TAI_category"].fillna("No score")

    areas = read_geojson(BOUNDARY_FILE)
    areas["match_name"] = areas.apply(make_area_label, axis=1)
    areas = areas.merge(
        tai,
        on="match_name",
        how="left",
        validate="one_to_one",
    )
    areas["TAI_category"] = areas["TAI_category"].fillna("No score")

    stops = pd.read_csv(STOPS_FILE)
    return tai, areas, stops


def format_count(value):
    return f"{value:,.0f}"


def select_areas(district_choice, area_choice, all_areas):
    selected = all_areas.copy()

    if district_choice != "All districts":
        prefix = district_choice.split(" — ")[0]
        selected = selected[
            selected["match_name"].str.startswith(prefix + " ")
        ]

    if area_choice != "All areas":
        selected = selected[selected["match_name"] == area_choice]

    return selected


def stops_inside_areas(stops, areas):
    geometries = list(areas.geometry)
    keep = []

    for row in stops.itertuples(index=False):
        point = Point(float(row.stop_lon), float(row.stop_lat))
        keep.append(any(geometry.covers(point) for geometry in geometries))

    return stops.loc[keep].copy()


def build_map(areas, stops, show_tai, show_stops=True):
    map_view = folium.Map(
        location=[5.38, 100.35],
        zoom_start=9,
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
    )

    coverage_scale = folium.LinearColormap(
        colors=["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"],
        vmin=0,
        vmax=100,
        caption="Population within 400 m (%)",
    )

    def style_area(feature):
        properties = feature["properties"]
        if show_tai:
            category = properties.get("TAI_category", "No score")
            fill = TAI_COLOURS.get(category, "#bdbdbd")
        else:
            value = properties.get("population_coverage_percent")
            fill = (
                coverage_scale(float(value))
                if value is not None
                else "#bdbdbd"
            )

        return {
            "fillColor": fill,
            "color": "#555555",
            "weight": 0.8,
            "fillOpacity": 0.72,
        }

    if show_tai:
        tooltip_fields = [
            "match_name",
            "TAI",
            "TAI_category",
            "population_coverage_percent",
            "bus_stops",
        ]
        tooltip_aliases = [
            "Area:",
            "TAI score:",
            "TAI category:",
            "Population within 400 m (%):",
            "Bus stops:",
        ]
    else:
        tooltip_fields = [
            "match_name",
            "population_coverage_percent",
            "bus_stops",
        ]
        tooltip_aliases = [
            "Area:",
            "Population within 400 m (%):",
            "Bus stops:",
        ]

    folium.GeoJson(
        areas.to_json(),
        name="Area accessibility",
        style_function=style_area,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
        ),
    ).add_to(map_view)

    if not show_tai:
        coverage_scale.add_to(map_view)

    stop_layer = folium.FeatureGroup(name="Bus stops", show=show_stops)
    for row in stops.itertuples(index=False):
        label = str(row.stop_name) if pd.notna(row.stop_name) else "Unnamed stop"
        folium.CircleMarker(
            location=[float(row.stop_lat), float(row.stop_lon)],
            radius=3,
            color="#54228a",
            fill=True,
            fill_color="#54228a",
            fill_opacity=0.9,
            tooltip=label,
        ).add_to(stop_layer)
    stop_layer.add_to(map_view)

    folium.LayerControl(collapsed=False).add_to(map_view)

    bounds = areas.total_bounds
    map_view.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return map_view

