import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import spearmanr

from ui_helpers import build_map


def render(tai, all_areas, all_stops, scored, socioeconomic):
    st.write(
        "The TAI combines four equally weighted indicators: bus stop density, "
        "population coverage, walking distance and destination access."
    )

    no_score_count = int(tai["TAI"].isna().sum())
    low_count = int((scored["TAI_category"] == "Low").sum())
    high_count = int((scored["TAI_category"] == "High").sum())
    mean_tai = scored["TAI"].mean() if not scored.empty else 0

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Areas with a score", f"{len(scored)} of {len(tai)}")
    metric2.metric("Average TAI", f"{mean_tai:.3f}")
    metric3.metric("Low TAI areas", low_count)
    metric4.metric("Areas without a score", no_score_count)

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
    ranked = scored.sort_values("TAI", ascending=False)
    columns = [
        "TAI_rank",
        "match_name",
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
    st.dataframe(
        ranked[columns],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("How the TAI indicators relate"):
        indicator_columns = {
            "Bus stop density": "bus_stops_per_km2",
            "Population coverage": "population_coverage_percent",
            "Average walk to a bus stop (metres)": "mean_walk_distance_m_routable_population",
            "Destination access": "destination_access_score",
        }
        available = {
            label: column
            for label, column in indicator_columns.items()
            if column in scored.columns
        }
        st.write(
            "Average walking distance from residents to the nearest bus stop, in metres. "
            "It is weighted by population and includes only places connected to the walking network. "
            "The TAI reverses this measure, so shorter distances give a better score."
        )
        st.write(
            "Spearman correlation shows how the indicators vary together. It does not show that one causes another. "
            "P-values below 0.001 are shown as < 0.001. Smaller values give stronger evidence of a relationship, "
            "but do not prove cause and effect."
        )
        rows = []
        items = list(available.items())
        for i, (label_a, column_a) in enumerate(items):
            for label_b, column_b in items[i + 1:]:
                pair = scored[[column_a, column_b]].dropna()
                rho, p_value = spearmanr(pair[column_a], pair[column_b])
                rows.append({
                    "Indicator 1": label_a,
                    "Indicator 2": label_b,
                    "Spearman rho": rho,
                    "p-value": p_value,
                    "Areas compared": len(pair),
                })
        correlation_table = pd.DataFrame(rows)
        st.dataframe(
            correlation_table.style.format(
                {
                    "Spearman rho": "{:.3f}",
                    "p-value": lambda value: "< 0.001" if abs(value) < 0.001 else f"{value:.3f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
