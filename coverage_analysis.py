from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui_helpers import (
    DISTRICT_LABELS,
    build_map,
    format_count,
    select_areas,
    stops_inside_areas,
)

COVERAGE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "population_accessibility_400m_by_mukim.csv"
)


def render(tai, all_areas, all_stops, scored, socioeconomic):
    coverage = pd.read_csv(COVERAGE_FILE)
    district_options = ["All districts"] + [
        f"{prefix} — {label}"
        for prefix, label in DISTRICT_LABELS.items()
    ]

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1.25, 1.5])

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
        area_choice = st.selectbox("Mukim / Bandar area", area_options)

    with filter_col3:
        st.write("")
        show_stops = st.checkbox("Show bus stop markers", value=True)

    selected_areas = select_areas(district_choice, area_choice, all_areas)
    selected_stops = stops_inside_areas(all_stops, selected_areas)
    selected_names = set(selected_areas["match_name"])

    # Keep TAI attributes separate. Coverage values come from the
    # population-accessibility output and do not change TAI scores.
    selected_tai = tai[tai["match_name"].isin(selected_names)].copy()
    selected_coverage = coverage[
        coverage["match_name"].isin(selected_names)
    ].copy()

    selected_coverage = selected_coverage.merge(
        selected_tai[["match_name", "bus_stops", "bus_stops_per_km2"]],
        on="match_name",
        how="left",
        validate="one_to_one",
    )

    if "population_source" not in selected_coverage.columns:
        selected_coverage["population_source"] = (
            "WorldPop 2020 population grid"
        )
    else:
        selected_coverage["population_source"] = (
            selected_coverage["population_source"].fillna(
                "WorldPop 2020 population grid"
            )
        )

    missing_coverage_mask = (
        selected_coverage["population_coverage_percent"].isna()
        | selected_coverage["estimated_csv_population_covered"].isna()
    )
    selected_coverage["coverage_status"] = "Available"
    selected_coverage.loc[
        missing_coverage_mask, "coverage_status"
    ] = "Not assessed: no population-grid value"
    selected_coverage["estimated_population_outside_400m"] = (
        selected_coverage["csv_population"]
        - selected_coverage["estimated_csv_population_covered"]
    ).clip(lower=0)

    coverage_data = selected_coverage.loc[
        ~missing_coverage_mask
    ].copy()
    missing_coverage = selected_coverage.loc[
        missing_coverage_mask
    ].copy()

    covered_population = coverage_data[
        "estimated_csv_population_covered"
    ].sum()
    population_total = coverage_data["csv_population"].fillna(0).sum()
    outside_population = max(population_total - covered_population, 0)
    population_without_coverage = (
        missing_coverage["csv_population"].fillna(0).sum()
    )
    weighted_coverage = (
        100 * covered_population / population_total
        if population_total > 0
        else 0
    )
    areas_with_coverage = coverage_data["match_name"].nunique()
    areas_without_coverage = missing_coverage["match_name"].nunique()

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("🚌  Total bus stops", format_count(len(selected_stops)))
    metric2.metric(
        "📍  Areas with data",
        f"{areas_with_coverage:,} / {len(selected_areas):,}",
    )
    metric3.metric(
        "👥  Estimated population assessed",
        format_count(population_total),
    )

    metric4, metric5, metric6 = st.columns(3)
    metric4.metric(
        "🚶  Estimated population within 400 m",
        format_count(covered_population),
    )
    metric5.metric(
        "🚶  Estimated population beyond 400 m",
        format_count(outside_population),
    )
    metric6.metric("◉  Population coverage", f"{weighted_coverage:.1f}%")

    coverage_note = (
        "Population coverage uses raster-based population estimates and "
        "400 m walking-network distances."
    )
    if "DTL MUKIM 15" in selected_names:
        coverage_note += (
            " DTL MUKIM 15 uses GHS-POP 2020 grid values scaled to its "
            "census total of 13; this is an estimate."
        )
    if areas_without_coverage:
        coverage_note += (
            f" {areas_without_coverage} selected area(s) have no usable "
            "population-grid values. Their "
            f"{format_count(population_without_coverage)} residents are "
            "not assessed."
        )
    coverage_note += " TAI results are on the TAI Rankings page."
    st.caption(coverage_note)

    st.subheader("400 m population coverage")
    map_col, key_col = st.columns([4, 1])

    with key_col:
        st.markdown("**Map key**")
        st.markdown(
            "Area colour shows the share of estimated population within "
            "400 m walking distance of a bus stop. Purple dots show bus stops. "
            "Grey areas have no usable population-grid values."
        )

    # Use coverage values for this map only. The TAI page keeps its own values.
    map_areas = selected_areas.drop(
        columns=["population_coverage_percent"],
        errors="ignore",
    ).merge(
        selected_coverage[["match_name", "population_coverage_percent"]],
        on="match_name",
        how="left",
        validate="one_to_one",
    )

    with map_col:
        map_view = build_map(
            map_areas,
            selected_stops,
            show_tai=False,
            show_stops=show_stops,
        )
        components.html(
            map_view.get_root().render(),
            height=620,
            scrolling=False,
        )

    with st.expander("View selected area coverage data"):
        columns = [
            "match_name",
            "bus_stops",
            "bus_stops_per_km2",
            "csv_population",
            "population_coverage_percent",
            "estimated_csv_population_covered",
            "estimated_population_outside_400m",
            "population_source",
            "coverage_status",
        ]
        columns = [
            column for column in columns if column in selected_coverage.columns
        ]
        st.dataframe(
            selected_coverage[columns],
            hide_index=True,
            use_container_width=True,
        )
