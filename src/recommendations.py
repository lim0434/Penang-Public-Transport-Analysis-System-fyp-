from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from scenario_simulation import (
    WALKING_SPEED_M_PER_MIN,
    WALK_LIMIT_M,
    _load_population_cells,
    _rank_candidate_sites,
    _residential_candidate_sites,
)
from linkbike_page import (
    NETWORK_FILE as LINKBIKE_NETWORK_FILE,
    STATIONS_FILE as LINKBIKE_STATIONS_FILE,
    _four_zone_coverage,
    _load_bus_stops,
    _normalise_stations,
)
from ui_helpers import DISTRICT_LABELS, clean_web_text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
DESTINATIONS_FILE = DATA / "connectivity_by_destination.csv"
LINKBIKE_GAP_FILE = DATA / "linkbike_gap_coverage.csv"

def _area_label(match_name):
    parts = str(match_name).split(" ", 1)
    if len(parts) != 2:
        return str(match_name).title()
    prefix, name = parts
    district = DISTRICT_LABELS.get(prefix, prefix)
    if name.startswith("MUKIM "):
        name = "Mukim " + name.removeprefix("MUKIM ")
    else:
        name = name.title()
    return f"{name} · {district}"


@st.cache_data(show_spinner=False)
def _load_destinations():
    destinations = pd.read_csv(DESTINATIONS_FILE)
    distance_columns = [
        "network_distance_m",
        "destination_snap_gap_m",
        "stop_snap_gap_m",
    ]
    for column in distance_columns:
        if column not in destinations:
            destinations[column] = np.nan
        destinations[column] = pd.to_numeric(destinations[column], errors="coerce")
    calculated_distance = destinations[distance_columns].sum(
        axis=1, min_count=len(distance_columns)
    )
    if "estimated_walking_distance_m" in destinations:
        saved_distance = pd.to_numeric(
            destinations["estimated_walking_distance_m"], errors="coerce"
        )
        destinations["estimated_walking_distance_m"] = saved_distance.fillna(
            calculated_distance
        )
    else:
        destinations["estimated_walking_distance_m"] = calculated_distance
    if "status" not in destinations:
        destinations["status"] = np.where(
            destinations["estimated_walking_distance_m"] <= WALK_LIMIT_M,
            "Within 400 m",
            "Beyond 400 m",
        )
    return destinations


def _priority_table(scored, destinations, display_names):
    low = scored[scored["TAI_category"] == "Low"].copy()
    low = low.sort_values(["TAI", "TAI_rank"], ascending=[True, False])
    low["Residents outside a short walk"] = (
        pd.to_numeric(low["csv_population"], errors="coerce").fillna(0)
        - pd.to_numeric(
            low["estimated_csv_population_covered"], errors="coerce"
        ).fillna(0)
    ).clip(lower=0)

    gaps = destinations[
        (destinations["status"] == "Beyond 400 m")
        & destinations["mukim"].notna()
    ].groupby("mukim").size()
    low["Key places beyond 400 m"] = low["match_name"].map(gaps).fillna(0).astype(int)
    low.insert(0, "Priority", np.arange(1, len(low) + 1))
    low.insert(
        2,
        "Area",
        low["match_name"].map(display_names).fillna(low["match_name"].map(_area_label)),
    )
    return low


def _priority_style(value, maximum):
    value = float(value)
    maximum = max(float(maximum), 1.0)
    if maximum <= 1 or value <= maximum / 3:
        colour = "#f8d7da"
    elif value <= 2 * maximum / 3:
        colour = "#fff3cd"
    else:
        colour = "#d1e7dd"
    return f"background-color: {colour}; color: #222"


def _ranked_candidates(area_name, destinations):
    key_sites = destinations[
        (destinations["mukim"] == area_name)
        & (destinations["status"] == "Beyond 400 m")
    ].copy()
    home_sites = _residential_candidate_sites(area_name)
    candidates = pd.concat([key_sites, home_sites], ignore_index=True, sort=False)
    if candidates.empty:
        return pd.DataFrame(), candidates
    ranked = _rank_candidate_sites(
        area_name,
        tuple(candidates["destination_id"].tolist()),
        tuple(candidates.to_dict("records")),
    )
    return ranked, candidates


def _linkbike_summary(all_stops=None):
    if not LINKBIKE_GAP_FILE.exists():
        return None
    gaps = pd.read_csv(LINKBIKE_GAP_FILE)
    if {"stop_lon", "stop_lat"}.issubset(gaps.columns):
        local = gaps.copy()
        status_column = "status" if "status" in local.columns else "Connection status"
    elif all_stops is not None and {"Bus stop ID", "Connection status"}.issubset(gaps.columns):
        if not {"stop_id", "stop_lat", "stop_lon"}.issubset(all_stops.columns):
            return None
        stops = all_stops[["stop_id", "stop_lat", "stop_lon"]].copy()
        stops["stop_id"] = stops["stop_id"].astype(str)
        gaps["Bus stop ID"] = gaps["Bus stop ID"].astype(str)
        local = gaps.merge(stops, left_on="Bus stop ID", right_on="stop_id", how="inner")
        status_column = "Connection status"
    else:
        return None
    local = local.dropna(subset=["stop_lon", "stop_lat"])
    if local.empty:
        return None
    no_bike = local[local[status_column].astype(str).str.startswith("No LinkBike station")]
    return len(local), len(no_bike)


@st.cache_data(show_spinner="Checking LinkBike coverage by mukim...")
def _linkbike_resident_coverage(area_names, threshold_m, station_mtime, stop_mtime, network_mtime):
    """Return estimated LinkBike-only residents by mukim using the shared walking analysis."""
    _ = (station_mtime, stop_mtime, network_mtime)
    stations = _normalise_stations(pd.read_csv(LINKBIKE_STATIONS_FILE))
    bus_stops = _load_bus_stops()
    population_rows = []
    for area_name in area_names:
        longitude, latitude, residents, _, _ = _load_population_cells(area_name)
        if len(residents):
            population_rows.append(pd.DataFrame({
                "area": area_name,
                "longitude": longitude,
                "latitude": latitude,
                "residents": residents,
            }))
    if not population_rows:
        return {}
    population_json = pd.concat(population_rows, ignore_index=True).to_json(
        orient="records", double_precision=6
    )
    _, area_stats = _four_zone_coverage(
        stations[["station_name", "latitude", "longitude"]].to_json(orient="records"),
        bus_stops[["stop_id", "stop_lat", "stop_lon", "nearest_node"]].to_json(orient="records"),
        population_json,
        int(threshold_m),
        network_mtime,
    )
    return {
        area_name: {
            "residents": float(values.get("LinkBike only", 0.0)),
            "share_percent": (
                100 * float(values.get("LinkBike only", 0.0))
                / float(values.get("Mukim population", 0.0))
                if float(values.get("Mukim population", 0.0)) > 0
                else 0.0
            ),
        }
        for area_name, values in area_stats.items()
    }

def render(tai, all_areas, all_stops, scored, socioeconomic):
    destinations = _load_destinations()
    display_names = all_areas.set_index("match_name")["display_name"].to_dict()
    priority = _priority_table(scored, destinations, display_names)

    st.caption(
        "Priority mukims are areas suggested for closer review, ordered from the lowest TAI upwards. "
        "This is a planning order, not a confirmed decision to build."
    )
    st.caption("A short walk means a modelled 400 m walking route, about five minutes on foot.")

    with st.container(border=True):
        st.subheader("Priority mukims")
        st.caption(
            "A mukim is a local area. Priority is its order for review, starting with the lowest TAI. "
            "TAI score is the combined access result. Residents within short walk (%) is the estimated "
            "share near a bus stop. Residents outside a short walk and Key places beyond 400 m count "
            "estimated residents and listed places with a longer walk."
        )
        priority_columns = [
            "Priority",
            "Area",
            "TAI",
            "population_coverage_percent",
            "Residents outside a short walk",
            "Key places beyond 400 m",
        ]
        shown_priority = priority[priority_columns].head(6).rename(columns={
            "Area": "Mukim",
            "TAI": "TAI score",
            "population_coverage_percent": "Residents within short walk (%)",
        })
        maximum_area_priority = max(int(priority["Priority"].max()), 1)
        page_export = shown_priority.copy()
        page_export_name = "priority_mukims.csv"
        st.dataframe(
            shown_priority.style.format({
                "TAI score": "{:.2f}",
                "Residents within short walk (%)": "{:.2f}",
                "Residents outside a short walk": "{:,.0f}",
            }).apply(
                lambda row: [_priority_style(row["Priority"], maximum_area_priority)] * len(row),
                axis=1,
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Show all low-TAI mukims"):
            all_priority = priority[priority_columns].rename(columns={
                "Area": "Mukim",
                "TAI": "TAI score",
                "population_coverage_percent": "Residents within short walk (%)",
            })
            st.dataframe(
                all_priority.style.format({
                    "TAI score": "{:.2f}",
                    "Residents within short walk (%)": "{:.2f}",
                    "Residents outside a short walk": "{:,.0f}",
                }).apply(
                    lambda row: [_priority_style(row["Priority"], maximum_area_priority)] * len(row),
                    axis=1,
                ),
                hide_index=True,
                use_container_width=True,
            )

    eligible_areas = scored[scored["TAI_category"] == "Low"].sort_values("TAI")
    if eligible_areas.empty:
        st.warning("No low-access mukims have a TAI score for recommendations.")
        return
    area_names = eligible_areas["match_name"].tolist()
    default_area = next(
        (name for name in priority["match_name"].tolist() if name in area_names),
        area_names[0],
    )
    with st.container(border=True):
        st.subheader("Suggested bus-stop locations")
        selected_area = st.selectbox(
            "Choose a low-access mukim",
            area_names,
            index=area_names.index(default_area),
            format_func=lambda name: clean_web_text(display_names.get(name, _area_label(name))),
            key="recommendations_area",
        )
        st.caption(
            "Select an area to see model-ranked places for a possible new stop. The current-walk column is the "
            "estimated walking time from that suggested place to the nearest existing bus stop. The model uses "
            "a 400 m walking route, about five minutes, as the short-walk range. Sites still need local checks "
            "for road safety, land use and space."
        )
        if st.button("Find possible locations", type="primary"):
            with st.spinner("Finding and ranking possible locations..."):
                try:
                    ranked, candidates = _ranked_candidates(selected_area, destinations)
                except ImportError as error:
                    st.error(
                        "Windows blocked Rasterio, which is needed to estimate "
                        "population gains. Ask your administrator to allow the "
                        f"Rasterio files. Details: {clean_web_text(str(error))}"
                    )
                    return
            st.session_state["recommendation_candidate_area"] = selected_area
            st.session_state["recommendation_ranked_sites"] = ranked
            st.session_state["recommendation_candidate_points"] = candidates

        result_area = st.session_state.get("recommendation_candidate_area")
        ranked = st.session_state.get("recommendation_ranked_sites", pd.DataFrame())
        if result_area == selected_area and not ranked.empty:
            shown = ranked.head(5).copy()
            shown["Current walk to bus stop (min)"] = (
                shown["Current estimated walking distance to a bus stop (m)"]
                / WALKING_SPEED_M_PER_MIN
            ).round(1)
            page_export = shown[[
                "Priority", "Place", "Type", "Current walk to bus stop (min)",
            ]].copy()
            page_export_name = "suggested_bus_stop_locations.csv"
            display_columns = [
                "Priority",
                "Place",
                "Type",
                "Estimated residents gaining access",
                "Key places gaining access",
                "Current walk to bus stop (min)",
            ]
            st.caption(
                "Each row is one possible site. Estimated residents gaining access and Key places gaining access "
                "show what may newly be within a 400 m walk. Type says whether the candidate is a residential area "
                "or a listed key place."
            )
            st.dataframe(
                shown[display_columns].style.format({
                    "Estimated residents gaining access": "{:,.0f}",
                    "Key places gaining access": "{:,.0f}",
                    "Current walk to bus stop (min)": "{:.1f}",
                }),
                hide_index=True,
                use_container_width=True,
            )
            top_site = shown.iloc[0]
            st.markdown(
                f"**Bus stop idea:** Check a stop near {clean_web_text(str(top_site['Place']))} "
                f"in {clean_web_text(display_names.get(selected_area, _area_label(selected_area)))}. "
                f"It may help {float(top_site['Estimated residents gaining access']):,.0f} residents "
                f"and {int(top_site['Key places gaining access'])} key places."
            )
            st.caption(
                "The figures estimate residents and other key places that may gain "
                "walking access from each site on its own. Sites are snapped to the "
                "mapped walking network for simulation. Check road safety, land use "
                "and operator needs before treating a site as suitable."
            )
        else:
            if result_area == selected_area and ranked.empty:
                st.info("No candidate locations could be generated for this mukim.")
            else:
                st.info("Choose a mukim and select ‘Find possible locations’ to rank suggestions.")

    simulation = st.session_state.get("recommendation_simulation_result")
    if simulation:
        page_export = pd.DataFrame([{
            "Mukim": display_names.get(simulation["area"], _area_label(simulation["area"])),
            "Proposed stops": ", ".join(simulation["sites"]),
            "TAI before": simulation["tai_before"],
            "TAI after": simulation["tai_after"],
            "Residents newly within a short walk": simulation["residents_gained"],
            "Key places newly within a short walk": simulation["key_places_gained"],
        }])
        page_export_name = "latest_scenario_summary.csv"
        with st.container(border=True):
            st.subheader("Latest tested scenario")
            st.caption(
                f"{clean_web_text(display_names.get(simulation['area'], _area_label(simulation['area'])))}: "
                f"{', '.join(clean_web_text(site) for site in simulation['sites'])}."
            )
            metric_cols = st.columns(3)
            metric_cols[0].metric(
                "TAI before → after",
                f"{simulation['tai_before']:.3f} → "
                + (f"{simulation['tai_after']:.3f}" if simulation['tai_after'] is not None else "Not available"),
            )
            metric_cols[1].metric(
                "Residents newly within a short walk",
                f"{simulation['residents_gained']:,}",
            )
            metric_cols[2].metric(
                "Key places newly within a short walk",
                f"{simulation['key_places_gained']:,}",
            )
            st.caption(
                "The TAI score is recalculated for the selected mukim. Its category "
                "and rank are not recalculated. A place beyond the 400 m walking "
                "range means a longer walk to a stop, not that bus service is absent."
            )
            gains = pd.DataFrame({
                "Key-place type": list(simulation["place_gains_by_type"].keys()),
                "Newly within a short walk": list(simulation["place_gains_by_type"].values()),
            })
            st.caption("This table counts newly reachable places by type after the selected stops are added.")
            st.dataframe(gains, hide_index=True, use_container_width=True)

    with st.container(border=True):
        st.subheader("Key places far from a bus stop")
        st.caption(
            "Key places are schools, hospitals, clinics and malls in the data. Current walk is the estimated walking "
            "time from each place to its nearest existing bus stop. The short-walk range is 400 m by the walking network, "
            "about five minutes. Review status says whether the place is beyond that range or has no route in the map data. "
            "The table shows the "
            "8 places with the longest walks and places with no route in the map data. "
            "Open the full list to see all places beyond 400 m."
        )
        beyond = destinations[destinations["status"] == "Beyond 400 m"].copy()
        beyond = beyond.sort_values("estimated_walking_distance_m", ascending=False)
        no_route = destinations[
            destinations["status"].astype(str).str.contains("No route", case=False, na=False)
        ].copy()
        disconnected = pd.concat([no_route, beyond.head(8)], ignore_index=True)
        disconnected["mukim"] = disconnected["mukim"].map(display_names).fillna(
            disconnected["mukim"]
        )
        disconnected["Current walk (min)"] = (
            disconnected["estimated_walking_distance_m"] / WALKING_SPEED_M_PER_MIN
        ).round(1)
        disconnected["Review status"] = np.where(
            disconnected["status"].astype(str).str.contains("No route", case=False, na=False),
            "No route in map data",
            "Beyond short walk",
        )
        st.dataframe(
            disconnected[[
                "destination_type", "destination_name", "mukim", "Current walk (min)",
                "Review status",
            ]].rename(columns={
                "destination_type": "Type",
                "destination_name": "Key place",
                "mukim": "Mukim",
            }).style.format({"Current walk (min)": "{:.1f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"{len(beyond)} key places are beyond the 400 m walking range. "
            f"{len(no_route)} more have no walking route in the map data."
        )
        with st.expander(f"Show all {len(beyond)} places beyond 400 m"):
            all_beyond = beyond.copy()
            all_beyond["mukim"] = all_beyond["mukim"].map(display_names).fillna(
                all_beyond["mukim"]
            )
            all_beyond["Current walk (min)"] = (
                all_beyond["estimated_walking_distance_m"] / WALKING_SPEED_M_PER_MIN
            ).round(1)
            all_beyond["Review status"] = "Beyond short walk"
            st.dataframe(
                all_beyond[[
                    "destination_type", "destination_name", "mukim", "Current walk (min)",
                    "Review status",
                ]].rename(columns={
                    "destination_type": "Type",
                    "destination_name": "Key place",
                    "mukim": "Mukim",
                }).style.format({"Current walk (min)": "{:.1f}"}),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

    linkbike_summary = _linkbike_summary(all_stops)
    if linkbike_summary:
        total_stops, no_nearby_bike = linkbike_summary
        st.subheader("LinkBike check")
        st.caption(
            f"{no_nearby_bike:,} of {total_stops:,} bus stops have no LinkBike station "
            "within a 400 m walk."
        )
        try:
            area_names = tuple(all_areas["match_name"].dropna().astype(str).unique())
            resident_coverage = _linkbike_resident_coverage(
                area_names,
                400,
                LINKBIKE_STATIONS_FILE.stat().st_mtime_ns,
                (DATA / "stops_penang.csv").stat().st_mtime_ns,
                LINKBIKE_NETWORK_FILE.stat().st_mtime_ns,
            )
            if resident_coverage:
                area_col, rank_col = st.columns(2)
                with area_col:
                    chosen_area = st.selectbox(
                        "Area",
                        ["All Penang", *sorted(resident_coverage)],
                        format_func=lambda name: (
                            name if name == "All Penang"
                            else clean_web_text(display_names.get(name, _area_label(name)))
                        ),
                        key="recommendations_linkbike_area",
                    )
                with rank_col:
                    rank_by = st.selectbox(
                        "Rank by",
                        ["Most residents", "Highest share"],
                        key="recommendations_linkbike_rank",
                    )
                st.caption(
                    "Most residents ranks by the estimated number. Highest share ranks by the "
                    "percentage of that mukim’s residents. Both show residents near LinkBike "
                    "but not a bus stop within 400 m."
                )
                metric_key = "residents" if rank_by == "Most residents" else "share_percent"
                candidates = [
                    (name, values)
                    for name, values in resident_coverage.items()
                    if values["residents"] > 0
                    and (chosen_area == "All Penang" or name == chosen_area)
                ]
                candidates.sort(
                    key=lambda item: (item[1][metric_key], item[1]["residents"]),
                    reverse=True,
                )
                candidates = candidates[:3] if chosen_area == "All Penang" else candidates[:1]
                if candidates:
                    for name, values in candidates:
                        area_name = clean_web_text(display_names.get(name, _area_label(name)))
                        st.markdown(
                            f"**{area_name}:** LinkBike may add access for about "
                            f"{values['residents']:,.0f} residents "
                            f"({values['share_percent']:.1f}% of the mukim)."
                        )
                else:
                    st.info("No extra residents are estimated near LinkBike only in this area at 400 m.")
        except (OSError, ValueError, KeyError, ImportError) as error:
            st.caption(
                f"LinkBike resident estimate is unavailable: {clean_web_text(str(error))}"
            )
        st.caption("This is an estimate of nearby access, not confirmed bus-bike trips.")

    return page_export, page_export_name




