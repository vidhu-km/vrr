import os
import numpy as np
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium
import branca.colormap as cm

import geopandas as gpd

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

GREEN_STOPS = [
    "#eaffea",
    "#b7f7b7",
    "#4fe24f",
    "#0dbf0d",
    "#006800",
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
# Cache: load Excel + shapefile
# ------------------------
@st.cache_data(show_spinner=False)
def load_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required_cols = [
        "Section",
        "section prod oil",
        "section prod gas",
        "section prod water",
        "section inj water",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XLSX: {missing}")

    num_cols = [
        "section prod oil",
        "section prod gas",
        "section prod water",
        "section inj water",
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["Section"] = df["Section"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False, allow_output_mutation=True)
def load_shp(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "Section" not in gdf.columns:
        raise ValueError("Shapefile missing required attribute `Section`.")

    gdf["Section"] = gdf["Section"].astype(str).str.strip()

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf

@st.cache_data(show_spinner=False)
def compute_vrr_by_section(df_xlsx: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_xlsx.groupby("Section", as_index=False)
        .agg(
            section_prod_oil=("section prod oil", "sum"),
            section_prod_gas=("section prod gas", "sum"),
            section_prod_water=("section prod water", "sum"),
            section_inj_water=("section inj water", "sum"),
        )
    )

    oil = agg["section_prod_oil"].to_numpy()
    gas = agg["section_prod_gas"].to_numpy()
    water = agg["section_prod_water"].to_numpy()
    inj = agg["section_inj_water"].to_numpy()

    gor = np.where(oil > 0, gas / oil, 0.0)
    gas_term = np.maximum(0, oil * 0.01033 * (gor - 120))
    denom = oil * 1.36 + water * 1.01 + gas_term

    vrr = np.where(denom > 0, (inj * 1.01) / denom, 0.0)
    vrr = np.nan_to_num(vrr, nan=0.0, posinf=0.0, neginf=0.0)
    vrr_disp = np.clip(vrr, VRR_MIN, MAX_VRR)
    agg["vrr"] = vrr_disp.round(3)  # round for tooltip display
    return agg[["Section", "vrr"]]

@st.cache_data(show_spinner=False)
def make_joined_geojson(excel_path: str, shp_path: str) -> tuple[str, list[float]]:
    df_xlsx = load_xlsx(excel_path)
    gdf = load_shp(shp_path)
    vrr_df = compute_vrr_by_section(df_xlsx)

    gdf2 = gdf.merge(vrr_df, on="Section", how="left")
    gdf2["vrr"] = gdf2["vrr"].fillna(0.0)

    # Compute centroid safely using projected CRS
    gdf_proj = gdf2.to_crs(epsg=3857)
    centroid = gdf_proj.geometry.centroid
    center = [float(centroid.y.mean()), float(centroid.x.mean())]

    return gdf2.to_json(), center

@st.cache_resource(show_spinner=False, allow_output_mutation=True)
def build_folium_map(geojson_str: str, center: list[float]) -> folium.Map:
    m = folium.Map(location=center, zoom_start=ZOOM_START, tiles=TILE_LAYER)

    vrr_colormap = cm.LinearColormap(GREEN_STOPS, vmin=VRR_MIN, vmax=MAX_VRR)
    vrr_colormap.caption = "VRR (green gradient)"

    def style_fn(feature):
        v = feature["properties"].get("vrr", 0.0) or 0.0
        color = vrr_colormap(v)
        return {
            "fillColor": color,
            "color": "rgba(0,0,0,0.25)",
            "weight": 0.7,
            "fillOpacity": 0.65,
        }

    geojson = folium.GeoJson(
        data=geojson_str,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["Section", "vrr"],
            aliases=["Section", "VRR"],
            localize=True,
            sticky=False,
            labels=True
        ),
    )
    geojson.add_to(m)
    vrr_colormap.add_to(m)
    return m

# ------------------------
# Run
# ------------------------
geojson_str, center = make_joined_geojson(EXCEL_FILE, SHP_FILE)
m = build_folium_map(geojson_str, center)

st.subheader("VRR Map (Shapefile polygons, smooth green colormap)")
st_folium(m, use_container_width=True, height=650)