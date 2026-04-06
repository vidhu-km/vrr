import os
import numpy as np
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium
import branca.colormap as cm

import geopandas as gpd
import streamlit.components.v1 as components  # (kept if you still need it)

# ------------------------
# Streamlit Setup
# ------------------------
st.set_page_config(page_title="VRR Shapefile Map (Navigable)", layout="wide")
st.title("VRR Shapefile Map (Navigable)")

# ------------------------
# Configuration
# ------------------------
EXCEL_FILE = "vrr_data.xlsx"
SHP_FILE = "ooipsectiongrid.shp"

ZOOM_START = 7

MAX_VRR = 3.0
VRR_MIN = 0.0

# Use a green-only gradient (smooth).
# We'll build a custom LinearColormap from Branca with green shades.
# You can tweak these stops if you want lighter/darker extremes.
GREEN_STOPS = [
    "#eaffea",  # near 0
    "#b7f7b7",
    "#4fe24f",
    "#0dbf0d",  # higher
    "#006800",  # max
]

TILE_LAYER = "CartoDB positron"

# ------------------------
# Validate input
# ------------------------
if not os.path.exists(EXCEL_FILE):
    st.error(f"Missing input file: `{EXCEL_FILE}`. Place it in the app folder.")
    st.stop()

if not os.path.exists(SHP_FILE):
    st.error(f"Missing shapefile: `{SHP_FILE}`. Place it in the app folder.")
    st.stop()

# ------------------------
# Caching: load Excel + shapefile
# ------------------------
@st.cache_data(show_spinner=False)
def load_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    required_cols = [
        "performance_oil_cumtodate",
        "performance_gas_cumtodate",
        "performance_water_cumtodate",
        "performance_waterinj_cumtodate",
        "header_sectionname",
        "header_resourceplay",
        "header_midpointlatitude",
        "header_midpointlongitude",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XLSX: {missing}")

    # numeric conversions
    num_cols = [
        "performance_oil_cumtodate",
        "performance_gas_cumtodate",
        "performance_water_cumtodate",
        "performance_waterinj_cumtodate",
    ]
    df[num_cols] = (
        df[num_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    # clean section name
    df["header_sectionname"] = df["header_sectionname"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_shp(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "Section" not in gdf.columns:
        raise ValueError("Shapefile missing required attribute `Section`.")
    # normalize for join
    gdf["Section"] = gdf["Section"].astype(str).str.strip()

    # ensure CRS exists; if missing, assume WGS84 (leaflet expects lat/lon)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


@st.cache_data(show_spinner=False)
def compute_vrr_by_section(df_xlsx: pd.DataFrame) -> pd.DataFrame:
    # If XLSX contains multiple rows per header_sectionname,
    # we aggregate to ensure one VRR per section.
    # Adjust aggregation logic if your data is strictly 1 row per section.
    agg = (
        df_xlsx.groupby("header_sectionname", as_index=False)
        .agg(
            performance_oil_cumtodate=("performance_oil_cumtodate", "sum"),
            performance_gas_cumtodate=("performance_gas_cumtodate", "sum"),
            performance_water_cumtodate=("performance_water_cumtodate", "sum"),
            performance_waterinj_cumtodate=("performance_waterinj_cumtodate", "sum"),
        )
    )

    oil = agg["performance_oil_cumtodate"].to_numpy()
    gas = agg["performance_gas_cumtodate"].to_numpy()
    water = agg["performance_water_cumtodate"].to_numpy()
    inj = agg["performance_waterinj_cumtodate"].to_numpy()

    gor = np.where(oil > 0, gas / oil, 0.0)

    gas_term = np.maximum(0, oil * 0.01033 * (gor - 120))
    denom = oil * 1.36 + water * 1.01 + gas_term

    vrr = np.where(denom > 0, (inj * 1.01) / denom, np.nan)
    vrr = np.nan_to_num(vrr, nan=0.0, posinf=0.0, neginf=0.0)

    vrr_disp = np.clip(vrr, VRR_MIN, MAX_VRR)

    agg["vrr"] = vrr_disp
    return agg[["header_sectionname", "vrr"]]


df_xlsx = load_xlsx(EXCEL_FILE)
gdf = load_shp(SHP_FILE)
vrr_df = compute_vrr_by_section(df_xlsx)

# ------------------------
# Join VRR to shapefile polygons
# ------------------------
gdf2 = gdf.merge(
    vrr_df,
    left_on="Section",
    right_on="header_sectionname",
    how="left"
)

# If some sections don't match, set VRR to 0 (or choose NaN and handle separately)
gdf2["vrr"] = gdf2["vrr"].fillna(0.0)

# ------------------------
# Create folium map
# ------------------------
center = [gdf2.geometry.centroid.y.mean(), gdf2.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=ZOOM_START, tiles=TILE_LAYER)

# ------------------------
# Branca smooth green gradient colormap
# ------------------------
vrr_colormap = cm.LinearColormap(
    GREEN_STOPS,
    vmin=VRR_MIN,
    vmax=MAX_VRR
)
vrr_colormap.caption = "VRR (green gradient)"

def style_fn(feature):
    v = feature["properties"].get("vrr", 0.0)
    if v is None or np.isnan(v):
        v = 0.0
    color = vrr_colormap(v)
    return {
        "fillColor": color,
        "color": "rgba(0,0,0,0.25)",
        "weight": 0.7,
        "fillOpacity": 0.65,
    }

# ------------------------
# Add GeoJSON polygons colored by VRR
# ------------------------
geojson = folium.GeoJson(
    data=gdf2.to_json(),
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["Section", "vrr"],
        aliases=["Section", "VRR"],
        localize=True,
        sticky=False,
        labels=True,
        fmt={ "vrr": "{:.3f}" }
    )
)
geojson.add_to(m)

# Add legend
vrr_colormap.add_to(m)

st.subheader("VRR Map (Shapefile polygons, smooth green colormap)")
st_folium(m, use_container_width=True, height=650)