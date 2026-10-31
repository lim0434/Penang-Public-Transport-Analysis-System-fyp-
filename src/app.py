from copy import copy
import inspect
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import Point
from linkbike_page import render as render_linkbike_page
from recommendations import render as render_recommendations_page
from scenario_simulation import render as render_scenario_page
from ui_helpers import clean_web_table, clean_web_text, read_geojson

_original_dataframe = st._main.dataframe


def _display_dataframe(data=None, *args, **kwargs):
    is_styled = isinstance(data, Styler)
    if isinstance(data, pd.DataFrame):
        data = clean_web_table(data)
    elif is_styled:
        data = copy(data)
        data.data = clean_web_table(data.data)

    table = data.data if is_styled else data
    if isinstance(table, pd.DataFrame):
        caller = inspect.currentframe().f_back
        source_name = Path(caller.f_code.co_filename).stem
        widget_key = f"table_filter_{source_name}_{caller.f_lineno}"
        column_options = ["No filter", *[str(column) for column in table.columns]]
        selected_column = st.selectbox(
            "Filter by column",
            column_options,
            key=f"{widget_key}_column",
        )

        total_rows = len(table)
        filtered = table
        if selected_column != "No filter":
            selected_series = table[selected_column]
            is_numeric = pd.api.types.is_numeric_dtype(selected_series)
            if is_numeric and selected_series.nunique(dropna=True) > 10:
                numbers = pd.to_numeric(selected_series, errors="coerce")
                range_count = min(5, int(numbers.nunique(dropna=True)))
                binned_numbers = pd.qcut(numbers, q=range_count, duplicates="drop")
                intervals = list(binned_numbers.cat.categories)
                range_options = [("all", None), *[("range", interval) for interval in intervals]]

                def _format_range(interval):
                    def _format_bound(value):
                        if pd.api.types.is_integer_dtype(selected_series):
                            return f"{int(round(value)):,}"
                        return f"{value:,.2f}".rstrip("0").rstrip(".")

                    return f"{_format_bound(interval.left)}–{_format_bound(interval.right)}"

                selected_range = st.selectbox(
                    "Select a number range",
                    range_options,
                    format_func=lambda option: (
                        "All ranges" if option[0] == "all" else _format_range(option[1])
                    ),
                    key=f"{widget_key}_range",
                )
                if selected_range[0] == "range":
                    filtered = table.loc[
                        binned_numbers.eq(selected_range[1]).fillna(False)
                    ]
            else:
                values = selected_series.dropna().drop_duplicates().tolist()
                values.sort(key=lambda value: str(value).casefold())
                value_options = [("all", None)]
                value_options.extend(("value", value) for value in values)
                if selected_series.isna().any():
                    value_options.append(("blank", None))

                selected_value = st.selectbox(
                    "Select a value",
                    value_options,
                    format_func=lambda option: (
                        "All values"
                        if option[0] == "all"
                        else "Blank"
                        if option[0] == "blank"
                        else str(option[1])
                    ),
                    key=f"{widget_key}_value",
                )
                if selected_value[0] == "value":
                    filtered = table.loc[
                        selected_series.eq(selected_value[1]).fillna(False)
                    ]
                elif selected_value[0] == "blank":
                    filtered = table.loc[selected_series.isna()]

            if len(filtered) != total_rows:
                if is_styled:
                    data.data = filtered
                else:
                    data = filtered
                st.caption(f"Showing {len(filtered)} of {total_rows} rows")

    return _original_dataframe(data, *args, **kwargs)


st.dataframe = _display_dataframe


def _spearman_rho(first, second):
    """Calculate Spearman rho from ranked values without importing SciPy."""
    pair = pd.concat(
        [
            pd.to_numeric(first, errors="coerce"),
            pd.to_numeric(second, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if len(pair) < 2:
        return float("nan")

    ranks = pair.rank(method="average")
    rank_a = ranks.iloc[:, 0]
    rank_b = ranks.iloc[:, 1]
    centred_a = rank_a - rank_a.mean()
    centred_b = rank_b - rank_b.mean()
    denominator = float(
        centred_a.pow(2).sum() * centred_b.pow(2).sum()
    ) ** 0.5
    if denominator == 0:
        return float("nan")
    return float((centred_a * centred_b).sum() / denominator)


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
PAGES = [
    "Coverage Analysis",
    "TAI Rankings",
    "Clustering",
    "Socioeconomic Correlation",
    "LinkBike Connections",
    "Scenario Simulation",
    "Recommendations",
]
TAI_COLOURS = {
    "Low": "#d7191c",
    "Medium": "#fdae61",
    "High": "#1a9641",
    "No score": "#bdbdbd",
}


st.set_page_config(
    page_title="Penang Transit DSS",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
    section[data-testid="stSidebar"] {
        background: #f7f6f9;
        border-right: 1px solid #e8e4ed;
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        min-height: 46px;
        text-align: left;
        border-radius: 10px;
        border: 1px solid transparent;
        background: transparent;
        color: #25212c;
        font-size: 15px;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: #f0eafa;
        border-color: #8b69b1;
        color: #4e267e;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7e2ed;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(30, 20, 45, 0.04);
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #54228a !important;
        font-size: 2.1rem !important;
        font-weight: 500 !important;
        line-height: 1.05 !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {color: #5f6368 !important;}
    div[data-testid="stDownloadButton"] {margin-top: 100px;}
    div[data-testid="stDownloadButton"] button {
        border: 1px solid #76519d !important;
        color: #54228a !important;
        border-radius: 9px;
        font-weight: 600 !important;
        background: #ffffff !important;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {font-size: 1rem; line-height: 1.5;}
    [data-testid="stCaptionContainer"] p {
        font-size: 1.02rem !important;
        line-height: 1.45;
    }
    [data-testid="stMetricLabel"] {font-size: 1rem;}
    .block-container h2 {font-size: 1.55rem;}
    .block-container h3 {font-size: 1.25rem;}
    .eyebrow {
        color: #725b87;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    areas["display_name"] = areas["match_name"].map(
        lambda name: str(name).split(" ", 1)[-1].title()
    )
    stop_lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stop_lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    valid_stops = stop_lon.notna() & stop_lat.notna()
    stop_points = gpd.GeoSeries(
        gpd.points_from_xy(stop_lon[valid_stops], stop_lat[valid_stops]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32647").reset_index(drop=True)
    stop_names = stops.loc[valid_stops, "stop_name"].fillna("bus stop").astype(str).tolist()
    projected_areas = areas.to_crs("EPSG:32647")
    for index, area in projected_areas.iterrows():
        label = str(area["match_name"])
        if " MUKIM " in label and len(stop_points):
            ordered_stops = stop_points.distance(
                area.geometry.representative_point()
            ).sort_values().index.tolist()
            nearest_index = next(
                (
                    int(index)
                    for index in ordered_stops
                    if not (
                        len("".join(ch for ch in stop_names[index].upper() if ch.isalnum())) <= 4
                        and any(ch.isdigit() for ch in stop_names[index])
                    )
                ),
                int(ordered_stops[0]),
            )
            district = DISTRICT_LABELS.get(label.split(" ", 1)[0], "Penang")
            mukim_number = label.split(" MUKIM ", 1)[1]
            areas.at[index, "display_name"] = (
                f"Mukim {mukim_number}, {district} — near {stop_names[nearest_index]}"
            )
    return tai, areas, stops


for required_file in (TAI_FILE, BOUNDARY_FILE, STOPS_FILE, SOCIOECONOMIC_FILE):
    if not required_file.exists():
        st.error(f"Required project file not found: {clean_web_text(str(required_file))}")
        st.stop()

tai, all_areas, all_stops = load_data()
socioeconomic = pd.read_csv(SOCIOECONOMIC_FILE)
scored = tai[tai["TAI"].notna()].copy()
area_display_names = all_areas.set_index("match_name")["display_name"].to_dict()

if st.session_state.get("page") == "LinkBike Gap Coverage":
    st.session_state.page = "LinkBike Connections"
if st.session_state.get("page") not in PAGES:
    st.session_state.page = "Coverage Analysis"


def change_page(page_name):
    st.session_state.page = page_name


# Sidebar navigation
st.sidebar.markdown(
    "<div style='font-size:22px;font-weight:700;padding:8px 2px 20px;'>"
    "Penang Transit DSS</div>",
    unsafe_allow_html=True,
)

for page_name in PAGES:
    st.sidebar.button(
        page_name,
        key=f"nav_{page_name}",
        type="primary" if st.session_state.page == page_name else "secondary",
        use_container_width=True,
        on_click=change_page,
        args=(page_name,),
    )

st.sidebar.markdown("---")
st.sidebar.caption("Penang Island and Seberang Perai")


def format_count(value):
    return f"{value:,.0f}"


def select_areas(district_choice, area_choice):
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
        caption=(
            "TAI coverage value"
            if show_tai
            else "Residents within about a 5-minute walk (%)"
        ),
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
            "display_name",
            "TAI",
            "TAI_category",
            "population_coverage_percent",
            "bus_stops",
        ]
        tooltip_aliases = [
            "Area:",
            "TAI score:",
            "TAI category:",
            "Residents within about 5 minutes (%):",
            "Bus stops:",
        ]
    else:
        tooltip_fields = [
            "display_name",
            "population_coverage_percent",
            "bus_stops",
        ]
        tooltip_aliases = [
            "Area:",
            "Residents within about 5 minutes (%):",
            "Bus stops:",
        ]

    coverage_layer = folium.FeatureGroup(
        name=(
            "TAI by area"
            if show_tai
            else "People within about 5 minutes (400 m)"
        ),
        show=True,
    )
    folium.GeoJson(
        areas.assign(display_name=areas["display_name"].map(clean_web_text)).to_json()
        if "display_name" in areas.columns else areas.to_json(),
        name="Area coverage",
        style_function=style_area,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
        ),
    ).add_to(coverage_layer)
    coverage_layer.add_to(map_view)

    if show_tai:
        legend_content = """
          <div style='display:grid;gap:6px;margin-top:8px'>
            <div><i style='display:inline-block;width:12px;height:12px;background:#d7191c;border-radius:3px;margin-right:7px'></i>Low access</div>
            <div><i style='display:inline-block;width:12px;height:12px;background:#fdae61;border-radius:3px;margin-right:7px'></i>Medium access</div>
            <div><i style='display:inline-block;width:12px;height:12px;background:#1a9641;border-radius:3px;margin-right:7px'></i>High access</div>
            <div><i style='display:inline-block;width:12px;height:12px;background:#bdbdbd;border-radius:3px;margin-right:7px'></i>No score</div>
          </div>
        """
    else:
        legend_content = """
          <div style='height:10px;border-radius:8px;margin:10px 0 5px;background:linear-gradient(90deg,#b2182b,#ef8a62,#f7f7f7,#67a9cf,#2166ac)'></div>
          <div style='display:flex;justify-content:space-between;color:#5f6368'><span>Fewer</span><span>More</span></div>
          <div style='margin-top:3px;color:#5f6368'>residents within about 5 minutes</div>
        """
    legend_html = (
        "<div style='position:fixed;z-index:9999;bottom:24px;right:14px;"
        "min-width:165px;background:rgba(255,255,255,.96);padding:11px 13px;"
        "border:1px solid #d8dce3;border-radius:10px;box-shadow:0 2px 10px #0002;"
        "font:12px/1.4 Arial,sans-serif;color:#252a31'>"
        + ("<strong>TAI access</strong>" if show_tai else "<strong>Walking access</strong>")
        + legend_content
        + "</div>"
    )
    map_view.get_root().html.add_child(folium.Element(legend_html))

    stop_layer = folium.FeatureGroup(name="Bus stops", show=show_stops)
    for row in stops.itertuples(index=False):
        label = str(row.stop_name) if pd.notna(row.stop_name) else "Unnamed stop"
        folium.CircleMarker(
            location=[float(row.stop_lat), float(row.stop_lon)],
            radius=3.5,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color="#1976d2",
            fill_opacity=0.9,
            tooltip=clean_web_text(label),
        ).add_to(stop_layer)
    stop_layer.add_to(map_view)

    folium.LayerControl(collapsed=True).add_to(map_view)

    bounds = areas.total_bounds
    map_view.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return map_view


def population_summary_card(column, title, value, detail):
    column.markdown(
        "<div style='min-height:112px;background:#fff;border:1px solid #e7e2ed;"
        "border-radius:12px;padding:12px 16px;box-shadow:0 1px 3px rgba(30,20,45,.04)'>"
        f"<div style='font-size:.98rem;color:#5f6368'>{title}</div>"
        f"<div style='font-size:1.8rem;font-weight:500;line-height:1.2;color:#54228a'>{value}</div>"
        f"<div style='font-size:.9rem;color:#68636d;margin-top:4px'>{detail}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def set_page_export(slot, export_result):
    if export_result is None:
        return
    frame, file_name = export_result
    if frame is None or frame.empty:
        return
    slot.download_button(
        "Export",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
        key=f"page_export_{file_name}",
    )


# Main page heading and page-specific export control
page = st.session_state.page
head_col, export_col = st.columns([5, 1])

with head_col:
    st.markdown(
        f"<div class='eyebrow'>PENANG TRANSIT DSS</div>"
        f"<h1 style='margin-top:0'>{page}</h1>"
        "<div style='color:#68636d;margin-top:-10px'>"
        "Public transport accessibility across Penang</div>",
        unsafe_allow_html=True,
    )

export_slot = export_col.empty()
st.write("")

page_guides = {
    "Coverage Analysis": (
        "This page estimates how many residents live within a short walk of a bus stop. "
        "The study uses a 400 m walking route, roughly five minutes on foot. "
        "The table helps compare areas and spot places that may need closer review. "
        "Population figures are estimates, not a count of individual people."
    ),
    "TAI Rankings": (
        "TAI means Transport Accessibility Index. It combines four access scores with equal "
        "weight. Each score runs from 0 to 1, and a higher score means better access in this "
        "comparison. The ranking helps planners see which areas may need attention and why."
    ),
    "Clustering": (
        "Clustering groups mukims with similar scores across the four TAI measures. The bar "
        "chart and score table show the average profile of each group. Higher scores mean "
        "better relative access. For walking distance, a shorter walk gets a higher score. The area table "
        "shows which group each mukim belongs to. This helps reveal shared access patterns "
        "and supports area-wide planning. The groups are a summary of this dataset, not fixed "
        "official ratings."
    ),
    "Socioeconomic Correlation": (
        "DUN means a state constituency. Each dot represents one DUN. ‘Median income’ is the "
        "middle household income value, and rho shows whether income or population tends to "
        "rise or fall with TAI. Rho near 1 means they tend to rise together, near -1 means one "
        "tends to rise as the other falls, and near 0 means little pattern. This page helps "
        "check whether access differences may align with social or population differences. "
        "A pattern does not prove that one factor causes another."
    ),
    "LinkBike Connections": (
        "This page checks whether LinkBike stations and RapidPenang bus stops are near each "
        "other by a walking route. The selected distance is the maximum route length used to "
        "call them ‘near’. The tables list bus stops or bike stations without a nearby match. "
        "The four-zone map groups residents as Bus only, LinkBike only, Both, or Neither within "
        "the selected distance. ‘Added by LinkBike’ counts estimated residents near a bike "
        "station but not a bus stop. This helps show where bike-sharing may add access. It does "
        "not count trips or prove that people transfer between services."
    ),
    "Scenario Simulation": (
        "This page tests ‘what if a bus stop were added here?’ Suggested places are ranked by "
        "estimated residents and key places that may gain access. ‘Before’ uses current stops. "
        "‘After’ recalculates access with the selected example stops added. The TAI score "
        "summarises the same four access measures. The contribution table shows each selected "
        "stop’s extra benefit after allowing for overlap with the other selected stops. This "
        "helps compare options before further site checks. Suggested sites are examples, not "
        "confirmed suitable locations."
    ),
    "Recommendations": (
        "This page brings earlier results together. Priority mukims are ordered from lower TAI "
        "scores. The bus-stop suggestions show places the model estimates could help residents "
        "or key places gain walking access. The LinkBike check estimates where bike stations "
        "may add access beyond bus stops. The tables show the evidence behind each suggestion, "
        "so planners can decide what to investigate first. Suggestions need local checks before "
        "they are treated as actual projects."
    ),
}
with st.expander("What this page shows and why it is useful"):
    st.write(page_guides.get(page, "This page presents transport access results for planning review."))


if page == "Coverage Analysis":
    district_options = ["All districts"] + [
        f"{prefix} — {label}"
        for prefix, label in DISTRICT_LABELS.items()
    ]

    filter_col1, filter_col2 = st.columns([1, 1.25])

    with filter_col1:
        district_choice = st.selectbox("District", district_options)

    prefix = (
        None
        if district_choice == "All districts"
        else district_choice.split(" — ")[0]
    )

    if prefix is None:
            area_options = ["All areas"] + sorted(all_areas["match_name"].tolist())
    else:
        district_areas = all_areas[
            all_areas["match_name"].str.startswith(prefix + " ")
        ]
        area_options = ["All areas"] + sorted(
            district_areas["match_name"].tolist()
        )

    with filter_col2:
        area_names = all_areas.set_index("match_name")["display_name"].to_dict()
        area_choice = st.selectbox(
            "Area",
            area_options,
            format_func=lambda name: clean_web_text("All areas" if name == "All areas" else area_names.get(name, name)),
        )

    selected_areas = select_areas(district_choice, area_choice)
    selected_stops = stops_inside_areas(all_stops, selected_areas)
    selected_names = set(selected_areas["match_name"])
    selected_tai = tai[tai["match_name"].isin(selected_names)]

    assessed_tai = selected_tai.dropna(
        subset=["estimated_csv_population_covered", "csv_population"]
    )
    covered_population = assessed_tai["estimated_csv_population_covered"].sum()
    population_total = assessed_tai["csv_population"].sum()
    population_farther = max(population_total - covered_population, 0)
    weighted_coverage = (
        100 * covered_population / population_total
        if population_total > 0
        else 0
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Bus stops", format_count(len(selected_stops)))
    metric2.metric("Areas shown", format_count(len(selected_areas)))
    metric3.metric("People within 5 min (%)", f"{weighted_coverage:.1f}%")

    with st.container(border=True):
        st.markdown("**People by distance to a bus stop**")
        near_col, farther_col = st.columns(2)
        near_col.metric("People near a bus stop", format_count(covered_population))
        farther_col.metric(
            "People farther from a bus stop", format_count(population_farther)
        )

    st.subheader("Bus stops and nearby residents")
    map_col, key_col = st.columns([4, 1])

    with key_col:
        st.markdown("**Map key**")
        st.markdown(
            "Use the map’s Layers control to show or hide the area colours and "
            "blue bus-stop dots. The colour shows the share of residents with a "
            "walking route of about five minutes or less to a stop."
        )

    with map_col:
        map_view = build_map(
            selected_areas,
            selected_stops,
            show_tai=False,
            show_stops=True,
        )
        components.html(map_view.get_root().render(), height=620, scrolling=False)

    with st.expander("View selected area coverage data"):
        st.caption(
            "Stops per km² means the number of bus stops in each square kilometre. Within 5 min (%) "
            "is the share of the area’s estimated residents with a bus stop within a 400 m walking route. "
            "People within 5 min is the estimated number of those residents."
        )
        columns = [
            "display_name",
            "bus_stops",
            "bus_stops_per_km2",
            "csv_population",
            "population_coverage_percent",
            "estimated_csv_population_covered",
        ]
        columns = [column for column in columns if column in selected_tai.columns or column == "display_name"]
        table = selected_tai.merge(
            selected_areas[["match_name", "display_name"]],
            on="match_name", how="left", validate="one_to_one",
        )
        table = table[columns].rename(columns={
            "display_name": "Area",
            "bus_stops": "Bus stops",
            "bus_stops_per_km2": "Stops per km²",
            "csv_population": "Area population",
            "population_coverage_percent": "Within 5 min (%)",
            "estimated_csv_population_covered": "People within 5 min (est.)",
        })
        st.dataframe(
            table.round(2),
            hide_index=True,
            use_container_width=True,
        )
    set_page_export(export_slot, (table.round(2), "coverage_selected_areas.csv"))



elif page == "TAI Rankings":
    st.write(
        "TAI combines four equally weighted public transport access measures."
    )

    no_score_count = int(tai["TAI"].isna().sum())
    mean_tai = scored["TAI"].mean() if not scored.empty else 0

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Areas with a score", f"{len(scored)} of {len(tai)}")
    metric2.metric(
        "Average TAI",
        f"{mean_tai:.2f}",
        help="The mean TAI across mukims with a score. It is an index out of 1, not a percentage of people.",
    )
    metric3.metric("Areas without a score", no_score_count)
    st.caption(
        f"Average TAI is {mean_tai:.2f} out of 1. Higher scores mean stronger relative access. "
        "For this data, 0.33 is towards the lower end of the scale. The system groups mukims with similar access "
        "results and labels each group Low, Medium or High. These labels do not come from fixed TAI score ranges."
    )

    population = pd.to_numeric(
        tai.get("csv_population", pd.Series(index=tai.index, dtype=float)),
        errors="coerce",
    )
    total_population = population.sum(min_count=1)
    assessed_population = tai.dropna(
        subset=["csv_population", "estimated_csv_population_covered"]
    )
    total_assessed_population = pd.to_numeric(
        assessed_population["csv_population"], errors="coerce"
    ).sum()
    covered_population = pd.to_numeric(
        assessed_population["estimated_csv_population_covered"], errors="coerce"
    ).sum()
    outside_population = max(total_assessed_population - covered_population, 0)
    outside_share = (
        100 * outside_population / total_assessed_population
        if total_assessed_population > 0
        else float("nan")
    )

    st.subheader("Population by TAI group")
    st.caption(
        "Each group counts everyone living in its mukims. The last card shows "
        "estimated residents outside the modelled 400 m walking range."
    )
    group_cols = st.columns(4)
    for column, category in zip(group_cols[:3], ["Low", "Medium", "High"]):
        in_group = tai["TAI_category"].eq(category)
        group_population = population.where(in_group).sum(min_count=1)
        group_share = (
            100 * group_population / total_population
            if pd.notna(group_population)
            and pd.notna(total_population)
            and total_population > 0
            else float("nan")
        )
        area_count = int(in_group.sum())
        population_summary_card(
            column,
            f"{category} TAI mukims",
            f"{group_population:,.0f}" if pd.notna(group_population) else "Not available",
            f"{area_count} mukims · {group_share:.1f}% of population"
            if pd.notna(group_share)
            else f"{area_count} mukims",
        )

    population_summary_card(
        group_cols[3],
        "Residents beyond 400 m",
        f"{outside_population:,.0f}" if total_assessed_population > 0 else "Not available",
        f"{outside_share:.1f}% of population"
        if pd.notna(outside_share)
        else "Population estimate unavailable",
    )

    st.subheader("TAI map")
    show_tai_stops = st.checkbox("Show bus stop markers", value=True)
    tai_map = build_map(
        all_areas,
        all_stops,
        show_tai=True,
        show_stops=show_tai_stops,
    )
    components.html(tai_map.get_root().render(), height=600, scrolling=False)

    st.subheader("Area rankings")
    st.caption(
        "TAI rank is the area’s position, with 1 being the highest score. TAI is the combined score. "
        "TAI category is its Low, Medium or High group. Stop density score rates stops for the area’s "
        "size. Population coverage score rates the share of residents within 400 m. Walking distance "
        "score rates how short residents’ walks are. Destination access score rates access to schools, "
        "hospitals, clinics and malls. The raw counts and percentages show the values behind those scores. "
        "All four scores run from 0 to 1, and higher is better."
    )
    ranked = scored.sort_values("TAI", ascending=False)
    columns = [
        "TAI_rank",
        "Area",
        "TAI",
        "TAI_category",
        "stop_density_score",
        "population_coverage_score",
        "walking_distance_score",
        "destination_access_score",
        "bus_stops_per_km2",
        "population_coverage_percent",
        "Average walk to a bus stop (metres)",
    ]
    ranked = ranked.rename(
        columns={"mean_walk_distance_m_routable_population": "Average walk to a bus stop (metres)"}
    )
    columns = [column for column in columns if column in ranked.columns]
    ranked["Area"] = ranked["match_name"].map(area_display_names).fillna(ranked["match_name"])
    columns = ["TAI_rank", "Area"] + [
        column for column in columns if column not in {"TAI_rank", "match_name"}
    ]
    display_columns = [
        column for column in columns
        if column not in {"TAI_rank", "Area", "TAI_category"}
    ]
    tai_display_format = {column: "{:.2f}" for column in display_columns}
    tai_display_format["TAI_rank"] = "{:.0f}"
    st.dataframe(
        ranked[columns].style.format(tai_display_format),
        hide_index=True,
        use_container_width=True,
    )
    set_page_export(export_slot, (ranked[columns].copy(), "tai_rankings.csv"))

    with st.expander("How the TAI indicators relate"):
        st.write(
            "Calculation: TAI = (stop density score + population coverage score + walking distance score "
            "+ destination access score) ÷ 4. Each part is put on a 0–1 scale first. The four parts "
            "have equal weight. For walking distance, shorter walks receive higher scores."
        )
        indicator_columns = {
            "Bus stop density": "bus_stops_per_km2",
            "Population coverage": "population_coverage_percent",
            "Average walk to a bus stop (metres)": "mean_walk_distance_m_routable_population",
            "Destination access": "destination_access_score",
        }
        st.write(
            "Average walking distance from residents to the nearest bus stop, in metres. "
            "It is weighted by population and includes only places connected to the walking network. "
            "The TAI reverses this measure, so shorter distances give a better score."
        )
        st.write(
            "Spearman rho shows whether two indicators tend to rise or fall together. "
            "It does not show that one causes another."
        )
        st.caption(
            "Indicator 1 and Indicator 2 name the pair being compared. Spearman rho ranges "
            "from -1 to 1. Areas compared is the number of mukims with data for both indicators."
        )
        available = {
            label: column
            for label, column in indicator_columns.items()
            if column in scored.columns
        }
        rows = []
        items = list(available.items())
        for index, (label_a, column_a) in enumerate(items):
            for label_b, column_b in items[index + 1:]:
                pair = scored[[column_a, column_b]].dropna()
                rows.append({
                    "Indicator 1": label_a,
                    "Indicator 2": label_b,
                    "Spearman rho": _spearman_rho(
                        pair[column_a], pair[column_b]
                    ),
                    "Areas compared": len(pair),
                })
        correlation_table = pd.DataFrame(rows)
        st.dataframe(
            correlation_table.style.format({"Spearman rho": "{:.3f}"}),
            hide_index=True,
            use_container_width=True,
        )


elif page == "Clustering":
    profile_columns = {
        "Bus stop density": "stop_density_score",
        "Population coverage": "population_coverage_score",
        "Walking distance": "walking_distance_score",
        "Destination access": "destination_access_score",
    }
    counts = scored["TAI_category"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    available_profile = {
        label: column
        for label, column in profile_columns.items()
        if column in scored.columns
    }
    profile = (
        scored.groupby("TAI_category")[list(available_profile.values())]
        .mean()
        .reindex(["Low", "Medium", "High"])
        .rename(columns={
            column: label
            for label, column in available_profile.items()
        })
    )

    low_col, medium_col, high_col = st.columns(3)
    low_col.metric("Low accessibility", int(counts["Low"]))
    medium_col.metric("Medium accessibility", int(counts["Medium"]))
    high_col.metric("High accessibility", int(counts["High"]))

    st.subheader("Average indicator profile")
    st.caption(
        "Each group is compared across four average scores, from 0 to 1. Higher scores mean more bus stops per "
        "square kilometre, more residents within a 400 m walk of a stop, shorter walks to bus stops, and better "
        "access to schools, hospitals, clinics and malls per bus stop. The scores compare mukims in this study."
    )

    st.bar_chart(profile, y=list(available_profile.keys()))

    st.subheader("Average indicator scores")
    st.caption(
        "Each row is a group and each column is its average score. Bus stop density means stops per square "
        "kilometre. Resident coverage means the percentage of residents estimated to live within a 400 m walk of a bus stop. Walking distance "
        "gives a higher score for shorter walks. Destination access reflects access to schools, hospitals, clinics "
        "and malls per bus stop. Higher scores mean better relative access."
    )
    st.dataframe(
        profile.style.format("{:.2f}"),
        hide_index=False,
        use_container_width=True,
    )

    st.subheader("Cluster map")
    show_cluster_stops = st.checkbox(
        "Show bus stop markers on cluster map",
        value=False,
        key="show_cluster_stops",
    )
    cluster_map = build_map(
        all_areas,
        all_stops,
        show_tai=True,
        show_stops=show_cluster_stops,
    )
    components.html(
        cluster_map.get_root().render(),
        height=600,
        scrolling=False,
    )

    st.subheader("Area results")
    st.caption("TAI is the area’s combined access score. The category shows its relative group: Low, Medium or High.")
    cluster_results = scored.sort_values(["TAI_category", "TAI"]).copy()
    cluster_results["Area"] = cluster_results["match_name"].map(area_display_names).fillna(
        cluster_results["match_name"]
    )
    cluster_export = cluster_results[["Area", "TAI", "TAI_rank", "TAI_category", "cluster_id"]].copy()
    cluster_display = cluster_export.drop(columns=["cluster_id"])
    st.dataframe(
        cluster_display.style.format({"TAI": "{:.2f}", "TAI_rank": "{:.0f}"}),
        hide_index=True,
        use_container_width=True,
    )
    set_page_export(export_slot, (cluster_export, "accessibility_clusters.csv"))

    with st.expander("Why use three clusters?"):
        st.write(
            "The silhouette score was highest for two clusters (0.469). "
            "Three clusters scored 0.425, while four scored 0.445. "
            "Three clusters were kept to provide clear Low, Medium and High "
            "groups for the DSS. The categories are relative and exploratory."
        )


elif page == "Socioeconomic Correlation":
    st.write(
        "This page compares 2024 median household income and population "
        "with public transport accessibility across Penang's 40 DUNs. "
        "These results show a relationship. They do not prove that one factor causes another."
    )

    required_columns = {
        "dun", "income_median", "population_2024_persons", "TAI",
        "TAI_rank", "population_coverage_percent", "bus_stops",
    }
    missing_columns = required_columns - set(socioeconomic.columns)
    if missing_columns:
        st.error(
            "The DUN results file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    else:
        socio_complete = socioeconomic.dropna(
            subset=["income_median", "population_2024_persons", "TAI"]
        ).copy()
        income_rho = _spearman_rho(
            socio_complete["income_median"], socio_complete["TAI"]
        )
        population_rho = _spearman_rho(
            socio_complete["population_2024_persons"], socio_complete["TAI"]
        )

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("DUNs compared", len(socio_complete))
        metric2.metric(
            "Income and TAI (rho)",
            f"{income_rho:.3f}",
        )
        metric3.metric(
            "Population and TAI (rho)",
            f"{population_rho:.3f}",
        )
        st.caption(
            f"Income and TAI (rho {income_rho:.3f}): DUNs with higher income tend to have higher TAI. "
            f"Population and TAI (rho {population_rho:.3f}): DUNs with more people tend to have lower TAI. "
            "The income pattern is moderate and positive. The population pattern is weak and negative. "
            "Neither result is automatically good or bad. The patterns do not show cause and effect, "
            "but they can help planners check where access may differ between communities."
        )
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Median income and TAI")
            income_chart_data = socio_complete.rename(
                columns={"income_median": "Median income (RM)"}
            )
            st.scatter_chart(
                income_chart_data, x="Median income (RM)", y="TAI",
                x_label="2024 median household income (RM)",
                y_label="DUN TAI", use_container_width=True,
            )
        with chart_col2:
            st.subheader("Population and TAI")
            population_chart_data = socio_complete.rename(
                columns={"population_2024_persons": "Population (2024)"}
            )
            st.scatter_chart(
                population_chart_data, x="Population (2024)", y="TAI",
                x_label="2024 population (people)",
                y_label="DUN TAI", use_container_width=True,
            )
        st.caption(
            "Each dot is one DUN. Dots rising from left to right mean higher values tend to go with higher TAI. "
            "Dots falling from left to right mean higher values tend to go with lower TAI."
        )

        st.subheader("DUN results")
        st.caption(
            "Median income is the middle household income in RM. Population is the 2024 estimate. TAI is the combined access score, "
            "TAI rank orders the DUNs, Bus stops is the stop count, and Population within 400 m "
            "is the estimated share living within the set walking range."
        )
        table_columns = [
            "dun", "income_median", "population_2024_persons", "TAI",
            "TAI_rank", "bus_stops", "population_coverage_percent",
        ]
        socio_table = (
            socio_complete[table_columns]
            .sort_values("TAI", ascending=False)
            .rename(columns={
                "dun": "DUN",
                "income_median": "Median income (RM)",
                "population_2024_persons": "Population (2024)",
                "TAI_rank": "TAI rank",
                "bus_stops": "Bus stops",
                "population_coverage_percent": "Population within 400 m (%)",
            })
        )
        st.dataframe(
            socio_table.style.format({
                "Median income (RM)": "{:,.0f}",
                "Population (2024)": "{:,.0f}",
                "TAI": "{:.2f}",
                "TAI rank": "{:.0f}",
                "Population within 400 m (%)": "{:.2f}",
            }),
            hide_index=True,
            use_container_width=True,
        )
        set_page_export(export_slot, (socio_table, "socioeconomic_results.csv"))


elif page == "LinkBike Connections":
    export_result = render_linkbike_page(tai, all_areas, all_stops, scored, socioeconomic)
    set_page_export(export_slot, export_result)

elif page == "Scenario Simulation":
    export_result = render_scenario_page(tai, all_areas, all_stops, scored, socioeconomic)
    set_page_export(export_slot, export_result)

elif page == "Recommendations":
    export_result = render_recommendations_page(tai, all_areas, all_stops, scored, socioeconomic)
    set_page_export(export_slot, export_result)






