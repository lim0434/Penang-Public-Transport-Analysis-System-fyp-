import streamlit as st
import streamlit.components.v1 as components

from ui_helpers import build_map
from clustering_metrics import calculate_silhouette_scores


def render(tai, all_areas, all_stops, scored, socioeconomic):
    st.write(
        "K-means groups the 97 areas with complete TAI data using four "
        "normalised accessibility indicators. The groups show relative "
        "patterns across this study area."
    )

    counts = (
        scored["TAI_category"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    low_col, medium_col, high_col = st.columns(3)
    low_col.metric("Low accessibility", int(counts["Low"]))
    medium_col.metric("Medium accessibility", int(counts["Medium"]))
    high_col.metric("High accessibility", int(counts["High"]))

    st.subheader("Average indicator profile")
    st.caption(
        "Scores range from 0 to 1. Higher scores indicate better relative "
        "accessibility. The walking-distance score is reversed, so shorter "
        "distances receive higher scores."
    )

    profile_columns = {
        "Bus stop density": "stop_density_score",
        "Population coverage": "population_coverage_score",
        "Walking distance": "walking_distance_score",
        "Destination access": "destination_access_score",
    }
    available_profile = {
        label: column
        for label, column in profile_columns.items()
        if column in scored.columns
    }

    profile = (
        scored.groupby("TAI_category")[
            list(available_profile.values())
        ]
        .mean()
        .reindex(["Low", "Medium", "High"])
        .rename(columns={
            column: label
            for label, column in available_profile.items()
        })
    )

    st.bar_chart(profile, y=list(available_profile.keys()))

    with st.expander("View average indicator scores"):
        st.dataframe(
            profile.round(3),
            hide_index=False,
            use_container_width=True,
            show_filter=False,
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
    st.dataframe(
        scored.sort_values(["TAI_category", "TAI"])[
            ["match_name", "TAI", "TAI_rank", "TAI_category", "cluster_id"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    scores = calculate_silhouette_scores(scored)
    with st.expander("Why use three clusters?"):
        if not scores:
            st.write("There is not enough complete area data to calculate these scores.")
        else:
            score_text = ", ".join(
                f"{count} groups: {score:.3f}"
                for count, score in sorted(scores.items())
            )
            best_count = max(scores, key=scores.get)
            st.write(f"Calculated from the current mukim scores: {score_text}.")
            st.write(
                f"{best_count} groups had the highest score, meaning clearer separation "
                "for these data. The page uses three groups to show Low, Medium and High. "
                "If another count scored higher, three groups remain a reporting choice."
            )

def show_silhouette_results(scored):
    """Show freshly calculated K-means silhouette scores on the clustering page."""
    from clustering_metrics import calculate_silhouette_scores

    scores = calculate_silhouette_scores(scored)
    with st.expander("Why use three clusters?"):
        if not scores:
            st.write("There is not enough complete area data to calculate these scores.")
            return

        score_text = ", ".join(
            f"{count} groups: {score:.3f}"
            for count, score in sorted(scores.items())
        )
        best_count = max(scores, key=scores.get)
        st.write(f"Calculated from the current mukim scores: {score_text}.")
        st.write(
            f"{best_count} groups had the highest score, meaning clearer separation "
            "for these data. The page uses three groups to show Low, Medium and High. "
            "If another count scored higher, three groups remain a reporting choice."
        )

