import streamlit as st


def render(tai, all_areas, all_stops, scored, socioeconomic):
    low_areas = scored[scored["TAI_category"] == "Low"].sort_values("TAI")
    no_stop_areas = tai[tai["bus_stops"].fillna(0) == 0]

    st.write(
        "These are areas for further planning review. Check routes, service "
        "frequency and local conditions before proposing changes."
    )

    st.subheader("Low TAI areas")
    columns = [
        "match_name",
        "TAI",
        "bus_stops",
        "population_coverage_percent",
        "mean_walk_distance_m_routable_population",
    ]
    columns = [column for column in columns if column in low_areas.columns]
    st.dataframe(
        low_areas[columns],
        hide_index=True,
        use_container_width=True,
    )

    if not no_stop_areas.empty:
        st.warning(
            f"{len(no_stop_areas)} areas have no bus stops recorded inside "
            "their boundary. Check nearby stops and boundary placement."
        )

    st.info(
        "This analysis covers bus stop access and nearby destinations. "
        "It does not include bus frequency, operating hours, fares or "
        "complete journey times."
    )
