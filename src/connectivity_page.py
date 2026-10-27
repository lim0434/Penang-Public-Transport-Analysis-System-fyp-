from html import escape
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import Fullscreen

CONNECTIVITY_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "connectivity_by_destination.csv"
)

STATUS_COLOURS = {
    "Within 400 m": "#3346a8",
    "Beyond 400 m": "#e85d2a",
    "No route in candidate network - review": "#9b59b6",
}


def _make_map(destinations):
    map_view = folium.Map(
        location=[5.38, 100.35],
        zoom_start=9,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Street Map",
        control=True,
        show=True,
    ).add_to(map_view)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        control=True,
        show=False,
    ).add_to(map_view)

    marker_groups = {
        status: folium.FeatureGroup(name=status, show=True)
        for status in STATUS_COLOURS
    }
    for row in destinations.itertuples(index=False):
        status = row.status
        if status not in marker_groups:
            continue
        name = escape(str(row.destination_name))
        category = escape(str(row.destination_type))
        district = escape(str(row.district))
        distance = (
            "No route in candidate network"
            if pd.isna(row.estimated_walking_distance_m)
            else f"{row.estimated_walking_distance_m / 1000:.2f} km estimated walking distance"
        )
        snap_note = "<br><b>Snap location needs review</b>" if row.snap_needs_review else ""
        popup = f"<b>{name}</b><br>{category}<br>{district}<br>{distance}{snap_note}"
        folium.CircleMarker(
            location=[row.destination_lat, row.destination_lon],
            radius=6 if row.snap_needs_review else 4,
            color="#ffffff" if row.snap_needs_review else STATUS_COLOURS[status],
            weight=2 if row.snap_needs_review else 1,
            fill=True,
            fill_color=STATUS_COLOURS[status],
            fill_opacity=0.9,
            tooltip=f"{row.destination_type}: {row.destination_name}",
            popup=folium.Popup(popup, max_width=280),
        ).add_to(marker_groups[status])

    for group in marker_groups.values():
        group.add_to(map_view)
    Fullscreen(position="topright").add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    return map_view


def _destination_card(row):
    distance = f"{row.estimated_walking_distance_m / 1000:.1f} km"
    name = escape(str(row.destination_name))
    category = escape(str(row.destination_type))
    district = escape(str(row.district))
    return f"""
    <div style="display:flex;gap:10px;align-items:flex-start;padding:11px 4px;
                border-bottom:1px solid #e9e7e3;">
      <div style="min-width:28px;height:28px;border-radius:50%;background:#f5f4ef;
                  display:flex;align-items:center;justify-content:center;color:#e85d2a;">●</div>
      <div style="flex:1;min-width:0;">
        <div style="font-weight:650;color:#252525;line-height:1.25">{name}</div>
        <div style="font-size:12px;color:#686868;margin-top:3px">{category} · {district}</div>
      </div>
      <div style="font-weight:700;color:#e85d2a;white-space:nowrap">{distance}</div>
    </div>
    """


def render(tai, all_areas, all_stops, scored, socioeconomic):
    if not CONNECTIVITY_FILE.exists():
        st.error(f"Connectivity results not found: {CONNECTIVITY_FILE}")
        st.stop()

    data = pd.read_csv(CONNECTIVITY_FILE)
    data["destination_name"] = data["destination_name"].fillna("Unnamed destination")
    data["snap_needs_review"] = data["snap_needs_review"].fillna(False).astype(bool)
    distance_parts = [
        "network_distance_m",
        "destination_snap_gap_m",
        "stop_snap_gap_m",
    ]
    for column in distance_parts:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["estimated_walking_distance_m"] = data[
        distance_parts
    ].sum(axis=1, min_count=len(distance_parts))
    has_route = data["network_distance_m"].notna() & data["estimated_walking_distance_m"].notna()
    data["status"] = "Beyond 400 m"
    data.loc[data["estimated_walking_distance_m"] <= 400, "status"] = "Within 400 m"
    data.loc[~has_route, "status"] = "No route in candidate network - review"

    districts = sorted(data["district"].dropna().unique())
    district_options = ["All districts"] + districts
    type_options = ["All types"] + sorted(data["destination_type"].dropna().unique())
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        district_choice = st.selectbox("District", district_options, key="connectivity_district")
    with filter_col2:
        type_choice = st.selectbox("Destination type", type_options, key="connectivity_type")

    selected = data.copy()
    if district_choice != "All districts":
        selected = selected[selected["district"] == district_choice]
    if type_choice != "All types":
        selected = selected[selected["destination_type"] == type_choice]

    within = selected[selected["status"] == "Within 400 m"]
    beyond = selected[selected["status"] == "Beyond 400 m"]
    no_route = selected[selected["status"] == "No route in candidate network - review"]

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total destinations analysed", f"{len(selected):,}")
    metric2.metric("Connected within 400 m", f"{len(within):,}")
    metric3.metric("Beyond 400 m", f"{len(beyond):,}")
    if len(no_route):
        st.info(
            f"{len(no_route):,} destination record(s) have no route in the candidate network and need review. "
            "They are kept separate from destinations beyond 400 m."
        )
    snap_count = int(selected["snap_needs_review"].sum())
    if snap_count:
        st.caption(f"{snap_count:,} destination record(s) also have a snap location flagged for review.")

    map_col, list_col = st.columns([1.08, 0.92], gap="medium")
    with map_col:
        st.markdown(
            "<div style='background:#f7f6f1;border:1px solid #e7e4dc;border-radius:12px;padding:8px'>"
            "<div style='font-size:13px;font-weight:600;padding:2px 5px 7px'>Connectivity status</div>"
            "<span style='color:#3346a8'>●</span> Within 400 m &nbsp; "
            "<span style='color:#e85d2a'>●</span> Beyond 400 m &nbsp; "
            "<span style='color:#9b59b6'>●</span> No route (review)</div>",
            unsafe_allow_html=True,
        )
        components.html(_make_map(selected).get_root().render(), height=520, scrolling=False)
    with list_col:
        st.markdown(
            "<div style='border:1px solid #dedbd5;border-radius:12px;padding:12px 14px 4px;'>"
            "<div style='font-size:16px;font-weight:700'>Critically disconnected destinations</div>"
            "<div style='font-size:12px;color:#686868;margin:3px 0 8px'>Sorted by walking distance to the nearest bus stop</div>",
            unsafe_allow_html=True,
        )
        critical = beyond.sort_values("estimated_walking_distance_m", ascending=False).head(8)
        if critical.empty:
            st.success("No destinations beyond 400 m match these filters.")
        else:
            for row in critical.itertuples(index=False):
                st.markdown(_destination_card(row), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander(f"Review {len(no_route):,} destinations with no route"):
            if no_route.empty:
                st.write("No no-route records match these filters.")
            else:
                st.dataframe(
                    no_route[["destination_type", "destination_name", "district", "mukim", "destination_snap_gap_m"]],
                    hide_index=True,
                    use_container_width=True,
                )

    with st.expander("Compare results by mukim"):
        summary = (
            selected.groupby(["district", "mukim"], dropna=False)
            .agg(
                destinations=("destination_id", "count"),
                within_400m=("status", lambda values: int((values == "Within 400 m").sum())),
                beyond_400m=("status", lambda values: int((values == "Beyond 400 m").sum())),
                no_route=("status", lambda values: int((values == "No route in candidate network - review").sum())),
            )
            .reset_index()
            .sort_values(["district", "mukim"])
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption(
            "Estimated distance includes the network route and short snap gaps from both ends. "
            "No-route records are unclassified and are not counted as beyond 400 m."
        )

