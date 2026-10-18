import html
import json

import folium
import geopandas as gpd
import pandas as pd
from shapely import wkt


# ============================================================
# 1. File paths
# ============================================================

gtfs_path = r"D:\fyp\data\raw\gtfs\stops.txt"

schools_path = r"D:\fyp\data\processed\schools.csv"
hospitals_path = r"D:\fyp\data\processed\hospitals.csv"
clinics_path = r"D:\fyp\data\processed\clinics.csv"
malls_path = r"D:\fyp\data\processed\malls.csv"

boundary_path = r"D:\fyp\data\raw\boundary\penang_boundary.geojson"
mukim_boundary_path = (
    r"D:\fyp\data\raw\boundary\penang_mukim_boundaries.geojson"
)
tai_path = r"D:\fyp\data\processed\tai_by_mukim.csv"

output_path = r"D:\fyp\data\processed\penang_map.html"


# ============================================================
# 2. Load bus stops and destination data
# ============================================================

print("Loading bus stops and destination data...")

stops = pd.read_csv(gtfs_path)
schools = pd.read_csv(schools_path)
hospitals = pd.read_csv(hospitals_path)
clinics = pd.read_csv(clinics_path)
malls = pd.read_csv(malls_path)

print(f"Bus stops: {len(stops):,}")
print(f"Schools: {len(schools):,}")
print(f"Hospitals: {len(hospitals):,}")
print(f"Clinics: {len(clinics):,}")
print(f"Malls: {len(malls):,}")


# ============================================================
# 3. Convert destination geometries to map coordinates
# ============================================================

def add_coordinates(data):
    """Create latitude and longitude columns from WKT geometry."""
    data = data.copy()
    data["geometry"] = data["geometry"].apply(wkt.loads)

    points = data["geometry"].apply(
        lambda geometry: (
            geometry
            if geometry.geom_type == "Point"
            else geometry.representative_point()
        )
    )

    data["latitude"] = points.apply(lambda point: point.y)
    data["longitude"] = points.apply(lambda point: point.x)

    return data


schools = add_coordinates(schools)
hospitals = add_coordinates(hospitals)
clinics = add_coordinates(clinics)
malls = add_coordinates(malls)


# ============================================================
# 4. Load Penang and mukim boundaries
# ============================================================

print("Loading boundaries and TAI results...")

with open(boundary_path, "r", encoding="utf-8") as file:
    boundary_data = json.load(file)

mukims = gpd.read_file(mukim_boundary_path)
tai = pd.read_csv(tai_path)

district_prefix = {
    "01": "SPT",
    "02": "SPU",
    "03": "SPS",
    "04": "DTL",
    "05": "DBD",
}


def make_mukim_label(row):
    """Create the same area name used in the TAI results."""
    district_code = str(row["KOD_DAERAH"]).replace(".0", "").zfill(2)

    if district_code not in district_prefix:
        raise ValueError(
            f"Unknown district code in mukim boundary: {district_code}"
        )

    district = district_prefix[district_code]
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


mukims["match_name"] = mukims.apply(make_mukim_label, axis=1)

if mukims["match_name"].duplicated().any():
    raise ValueError("Mukim boundary area names are duplicated.")

if tai["match_name"].duplicated().any():
    raise ValueError("TAI area names are duplicated.")

matched_names = mukims["match_name"].isin(tai["match_name"])

print(
    f"Mukim areas matched to TAI data: "
    f"{matched_names.sum()} of {len(mukims)}"
)

if not matched_names.all():
    unmatched = mukims.loc[~matched_names, "match_name"].tolist()
    raise ValueError(f"Some boundary areas did not match TAI data: {unmatched}")

mukims = mukims.merge(
    tai[["match_name", "TAI", "TAI_category"]],
    on="match_name",
    how="left",
    validate="one_to_one",
)

mukims["TAI_category"] = mukims["TAI_category"].fillna("No score")

print(
    "Areas with a TAI score:",
    mukims["TAI"].notna().sum(),
)
print(
    "Areas without a TAI score:",
    mukims["TAI"].isna().sum(),
)


# ============================================================
# 5. Create the base map
# ============================================================

map_center = [
    stops["stop_lat"].mean(),
    stops["stop_lon"].mean(),
]

m = folium.Map(
    location=map_center,
    zoom_start=10,
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
)


# ============================================================
# 6. Create map layers
# ============================================================

boundary_layer = folium.FeatureGroup(name="Penang Boundary", show=False)
tai_layer = folium.FeatureGroup(name="TAI by Mukim", show=False)
bus_layer = folium.FeatureGroup(name="Bus Stops", show=False)
school_layer = folium.FeatureGroup(name="Schools", show=False)
hospital_layer = folium.FeatureGroup(name="Hospitals", show=False)
clinic_layer = folium.FeatureGroup(name="Clinics", show=False)
mall_layer = folium.FeatureGroup(name="Malls", show=False)

# ============================================================
# 7. Add the Penang state boundary
# ============================================================

print("Adding Penang boundary...")

folium.GeoJson(
    boundary_data,
    name="Penang Boundary",
    style_function=lambda feature: {
        "fillColor": "#ffffff",
        "color": "black",
        "weight": 3,
        "fillOpacity": 0,
    },
    tooltip="Penang State Boundary",
).add_to(boundary_layer)


# ============================================================
# 8. Add the TAI mukim layer
# ============================================================

print("Adding TAI mukim layer...")

tai_colours = {
    "Low": "#d7191c",
    "Medium": "#fdae61",
    "High": "#1a9641",
    "No score": "#bdbdbd",
}


def tai_style(feature):
    category = feature["properties"].get("TAI_category", "No score")
    colour = tai_colours.get(category, "#bdbdbd")

    return {
        "fillColor": colour,
        "color": "#444444",
        "weight": 1,
        "fillOpacity": 0.65,
    }


folium.GeoJson(
    mukims.to_json(),
    name="TAI by Mukim",
    style_function=tai_style,
    tooltip=folium.GeoJsonTooltip(
        fields=["match_name", "TAI", "TAI_category"],
        aliases=["Area:", "TAI score:", "Category:"],
        localize=True,
        sticky=False,
    ),
).add_to(tai_layer)


# ============================================================
# 9. Add bus stops
# ============================================================

print("Adding bus stops...")

for _, row in stops.iterrows():
    stop_name = (
        "Unnamed bus stop"
        if pd.isna(row["stop_name"])
        else str(row["stop_name"])
    )

    folium.CircleMarker(
        location=[row["stop_lat"], row["stop_lon"]],
        radius=2,
        popup=html.escape(stop_name),
        tooltip=html.escape(stop_name),
        color="#2455a4",
        fill=True,
        fill_opacity=0.8,
    ).add_to(bus_layer)


# ============================================================
# 10. Add destination markers
# ============================================================

def add_destination_markers(data, layer, default_name):
    for _, row in data.iterrows():
        name = row["name"]

        if pd.isna(name) or not str(name).strip():
            name = f"Unnamed {default_name}"
        else:
            name = str(name)

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=html.escape(name),
            tooltip=html.escape(name),
        ).add_to(layer)


print("Adding destination markers...")

add_destination_markers(schools, school_layer, "school")
add_destination_markers(hospitals, hospital_layer, "hospital")
add_destination_markers(clinics, clinic_layer, "clinic")
add_destination_markers(malls, mall_layer, "mall")


# ============================================================
# 11. Add layers to map
# ============================================================

boundary_layer.add_to(m)
tai_layer.add_to(m)
bus_layer.add_to(m)
school_layer.add_to(m)
hospital_layer.add_to(m)
clinic_layer.add_to(m)
mall_layer.add_to(m)


# ============================================================
# 12. Add a TAI legend
# ============================================================

legend_html = """
<div style="
    position: fixed;
    bottom: 35px;
    left: 35px;
    z-index: 9999;
    background: white;
    padding: 10px 12px;
    border: 1px solid #777;
    border-radius: 4px;
    font-size: 13px;
">
    <b>TAI category</b><br>
    <span style="color:#d7191c;">&#9632;</span> Low<br>
    <span style="color:#fdae61;">&#9632;</span> Medium<br>
    <span style="color:#1a9641;">&#9632;</span> High<br>
    <span style="color:#bdbdbd;">&#9632;</span> No score
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))


# ============================================================
# 13. Add layer control and fit the map to Penang
# ============================================================

folium.LayerControl(collapsed=False).add_to(m)

bounds = [
    [stops["stop_lat"].min(), stops["stop_lon"].min()],
    [stops["stop_lat"].max(), stops["stop_lon"].max()],
]

m.fit_bounds(bounds)


# ============================================================
# 14. Save the map
# ============================================================

m.save(output_path)

print("\nMap created successfully.")
print(f"Saved to: {output_path}")