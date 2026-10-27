import pandas as pd
import streamlit as st
from scipy.stats import spearmanr


def render(tai, all_areas, all_stops, scored, socioeconomic):
    st.write(
        "This page compares 2024 median household income and population "
        "with public transport accessibility across Penang's 40 DUNs. "
        "Spearman correlation measures association, not cause and effect."
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
        income_result = spearmanr(
            socio_complete["income_median"], socio_complete["TAI"]
        )
        population_result = spearmanr(
            socio_complete["population_2024_persons"], socio_complete["TAI"]
        )

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("DUNs compared", len(socio_complete))
        metric2.metric(
            "Income and TAI (rho)",
            f"{income_result.statistic:.3f}",
            help=f"p-value: {income_result.pvalue:.4g}",
        )
        metric3.metric(
            "Population and TAI (rho)",
            f"{population_result.statistic:.3f}",
            help=f"p-value: {population_result.pvalue:.4g}",
        )
        st.caption(
            f"Income association p-value: {income_result.pvalue:.4g}. "
            f"Population association p-value: {population_result.pvalue:.4g}. "
            "A p-value below 0.05 is commonly treated as statistically significant."
        )

        st.subheader("What the results suggest")
        if income_result.pvalue < 0.05:
            income_direction = (
                "Higher median household income tended to occur alongside "
                "higher TAI scores across the DUNs."
            )
            income_strength = (
                "The positive correlation is moderate in size."
                if abs(income_result.statistic) < 0.7
                else "The positive correlation is strong in size."
            )
            st.write(
                f"**Income:** {income_direction} {income_strength} "
                "The association is statistically significant in this analysis."
            )
        else:
            st.write(
                "**Income:** The analysis does not show a statistically "
                "significant association between income and TAI."
            )

        if population_result.pvalue < 0.05:
            st.write(
                f"**Population:** The correlation is "
                f"{population_result.statistic:.3f}. It is statistically "
                "significant, but this does not show that population causes "
                "higher or lower accessibility."
            )
        else:
            st.write(
                f"**Population:** The correlation is "
                f"{population_result.statistic:.3f}, but it is not "
                "statistically significant. These results do not provide "
                "strong evidence of an association between population size "
                "and TAI."
            )

        st.caption(
            "These are area-level associations. They do not show cause and "
            "effect, and they should be interpreted with the difference in "
            "data years in mind."
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Median income and TAI")
            st.scatter_chart(
                socio_complete, x="income_median", y="TAI",
                x_label="2024 median household income (RM)",
                y_label="DUN TAI", use_container_width=True,
            )
        with chart_col2:
            st.subheader("Population and TAI")
            st.scatter_chart(
                socio_complete, x="population_2024_persons", y="TAI",
                x_label="2024 population (people)",
                y_label="DUN TAI", use_container_width=True,
            )

        st.caption(
            "The population grid used to estimate 400 m access is from 2020. "
            "Income and DUN population are from 2024. Consider this time "
            "difference when interpreting the results."
        )
        st.subheader("DUN results")
        table_columns = [
            "dun", "income_median", "population_2024_persons", "TAI",
            "TAI_rank", "bus_stops", "population_coverage_percent",
        ]
        st.dataframe(
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
            .style.format({
                "Median income (RM)": "{:,.0f}",
                "Population (2024)": "{:,.0f}",
                "TAI": "{:.3f}",
                "Population within 400 m (%)": "{:.1f}",
            }),
            hide_index=True,
            use_container_width=True,
        )
