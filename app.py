import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui_helpers import (
    BOUNDARY_FILE,
    SOCIOECONOMIC_FILE,
    STOPS_FILE,
    TAI_FILE,
    load_data,
)
from coverage_analysis import COVERAGE_FILE, render as render_coverage_analysis
from tai_rankings import render as render_tai_rankings
from clustering import render as render_clustering
from socioeconomic_correlation import render as render_socioeconomic_correlation
from recommendations import render as render_recommendations


PAGES = [
    ("Coverage Analysis", "▧"),
    ("TAI Rankings", "▥"),
    ("Clustering", "◌"),
    ("Socioeconomic Correlation", "♧"),
    ("Recommendations", "♡"),
]


st.set_page_config(
    page_title="Penang Transit DSS",
    page_icon="🚌",
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
    [data-testid="stMetricValue"] {color: #54228a;}
    div[data-testid="stDownloadButton"] button {
        border: 1px solid #76519d;
        color: #54228a;
        border-radius: 9px;
        background: white;
    }
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


for required_file in (TAI_FILE, BOUNDARY_FILE, STOPS_FILE, SOCIOECONOMIC_FILE, COVERAGE_FILE):
    if not required_file.exists():
        st.error(f"Required project file not found: {required_file}")
        st.stop()

tai, all_areas, all_stops = load_data()
socioeconomic = pd.read_csv(SOCIOECONOMIC_FILE)
scored = tai[tai["TAI"].notna()].copy()

if st.session_state.get("page") not in dict(PAGES):
    st.session_state.page = "Coverage Analysis"


def change_page(page_name):
    st.session_state.page = page_name


st.sidebar.markdown(
    "<div style='font-size:22px;font-weight:700;padding:8px 2px 20px;'>"
    "Penang Transit DSS</div>",
    unsafe_allow_html=True,
)

for page_name, icon in PAGES:
    st.sidebar.button(
        f"{icon}   {page_name}",
        key=f"nav_{page_name}",
        type="primary" if st.session_state.page == page_name else "secondary",
        use_container_width=True,
        on_click=change_page,
        args=(page_name,),
    )

st.sidebar.markdown("---")
st.sidebar.caption("Penang Island and Seberang Perai")


# Main page heading and export control
page = st.session_state.page
heading_icon = dict((name, icon) for name, icon in PAGES)[page]
head_col, export_col = st.columns([5, 1])

with head_col:
    st.markdown(
        f"<div class='eyebrow'>PENANG TRANSIT DSS</div>"
        f"<h1 style='margin-top:0'>{heading_icon} {page}</h1>"
        "<div style='color:#68636d;margin-top:-10px'>"
        "Public transport accessibility across Penang</div>",
        unsafe_allow_html=True,
    )

with export_col:
    st.write("")
    if page == "Coverage Analysis":
        coverage_export = pd.read_csv(COVERAGE_FILE)
        export_data = coverage_export.to_csv(index=False).encode("utf-8")
        export_name = "population_accessibility_400m_by_mukim.csv"
    elif page == "Socioeconomic Correlation":
        export_data = socioeconomic.to_csv(index=False).encode("utf-8")
        export_name = "tai_by_dun_socioeconomic.csv"
    else:
        export_data = tai.to_csv(index=False).encode("utf-8")
        export_name = "tai_by_mukim.csv"

    st.download_button(
        "↓  Export",
        data=export_data,
        file_name=export_name,
        mime="text/csv",
        use_container_width=True,
    )

st.write("")


renderers = {
    "Coverage Analysis": render_coverage_analysis,
    "TAI Rankings": render_tai_rankings,
    "Clustering": render_clustering,
    "Socioeconomic Correlation": render_socioeconomic_correlation,
    "Recommendations": render_recommendations,
}
renderers[page](tai, all_areas, all_stops, scored, socioeconomic)
