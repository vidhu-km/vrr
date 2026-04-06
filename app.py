import os
import xlsxwriter
import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium import LayerControl
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
BAKKEN_SHP_FILE = "Bakken Units.shp"

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
# Validate input files
# ------------------------
if not os.path.exists(EXCEL_FILE):
    st.error(f"Missing input file: `{EXCEL_FILE}`. Place it in the app folder.")
    st.stop()

if not os.path.exists(SHP_FILE):
    st.error(f"Missing shapefile: `{SHP_FILE}`. Place it in the app folder.")
    st.stop()

if not os.path.exists(BAKKEN_SHP_FILE):
    st.warning(
        f"Optional shapefile `{BAKKEN_SHP_FILE}` not found — "
        "the Bakken Units layer will be skipped."
    )
    BAKKEN_AVAILABLE = False
else:
    BAKKEN_AVAILABLE = True

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
    df[num_cols] = df[num_cols] * 1000
    df["Section"] = df["Section"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_shp(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
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

    oil = agg["section_prod_oil"].to_numpy(dtype=float)
    gas = agg["section_prod_gas"].to_numpy(dtype=float)
    water = agg["section_prod_water"].to_numpy(dtype=float)
    inj = agg["section_inj_water"].to_numpy(dtype=float)

    gor = np.where(oil > 0, gas / oil, 0.0)
    gas_term = np.maximum(0, oil * 0.01033 * (gor - 120))
    denom = oil * 1.36 + water * 1.01 + gas_term

    vrr = np.where(denom > 0, (inj * 1.01) / denom, 0.0)
    vrr = np.nan_to_num(vrr, nan=0.0, posinf=0.0, neginf=0.0)
    vrr_disp = np.clip(vrr, VRR_MIN, MAX_VRR)
    agg["vrr"] = vrr_disp.round(3)
    return agg[["Section", "vrr"]]


@st.cache_data(show_spinner=False)
def make_joined_geojson(excel_path: str, shp_path: str) -> tuple:
    df_xlsx = load_xlsx(excel_path)
    gdf = load_shp(shp_path)

    if "Section" not in gdf.columns:
        raise ValueError("Shapefile missing required attribute `Section`.")
    gdf["Section"] = gdf["Section"].astype(str).str.strip()

    vrr_df = compute_vrr_by_section(df_xlsx)
    gdf2 = gdf.merge(vrr_df, on="Section", how="left")
    gdf2["vrr"] = gdf2["vrr"].fillna(0.0)

    # Compute centroid in projected CRS, then convert back to lat/lon
    gdf_proj = gdf2.to_crs(epsg=3857)
    centroid_proj = gdf_proj.geometry.centroid
    centroid_gdf = gpd.GeoDataFrame(geometry=centroid_proj, crs="EPSG:3857")
    centroid_gdf = centroid_gdf.to_crs(epsg=4326)
    center_lat = float(centroid_gdf.geometry.y.mean())
    center_lon = float(centroid_gdf.geometry.x.mean())

    return gdf2.to_json(), [center_lat, center_lon]


@st.cache_data(show_spinner=False)
def load_bakken_geojson(path: str) -> str:
    gdf = load_shp(path)
    return gdf.to_json()


# -------------------------------------------------------
# Build map — NOT cached (folium Maps are mutable objects
# with closures that break Streamlit's cache).
# The *data* feeding it is already cached above, so this
# is cheap: it only serialises the already-prepared JSON.
# -------------------------------------------------------
def build_folium_map(
    geojson_str: str,
    center: list,
    bakken_geojson_str=None,
) -> folium.Map:
    m = folium.Map(location=center, zoom_start=ZOOM_START, tiles=TILE_LAYER)

    # ---- Bakken Units layer ----
    if bakken_geojson_str is not None:
        bakken_layer = folium.GeoJson(
            data=bakken_geojson_str,
            name="Bakken Units",
            style_function=lambda _: {
                "fillColor": "transparent",
                "color": "#e6550d",
                "weight": 2.0,
                "fillOpacity": 0.0,
                "dashArray": "5 3",
            },
        )
        bakken_layer.add_to(m)

    # ---- VRR layer ----
    vrr_colormap = cm.LinearColormap(GREEN_STOPS, vmin=VRR_MIN, vmax=MAX_VRR)
    vrr_colormap.caption = "VRR (green gradient)"

    def style_fn(feature, _cmap=vrr_colormap):
        v = feature["properties"].get("vrr", 0.0)
        if v is None or v == 0:
            fill_color = "transparent"
            fill_opacity = 0.0
        else:
            v = float(v)
            fill_color = _cmap(v)
            fill_opacity = 0.65
        return {
            "fillColor": fill_color,
            "color": "rgba(0,0,0,0.25)",
            "weight": 0.7,
            "fillOpacity": fill_opacity,
        }

    vrr_layer = folium.GeoJson(
        data=geojson_str,
        name="VRR Sections",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["Section", "vrr"],
            aliases=["Section", "VRR"],
            localize=True,
            sticky=False,
            labels=True,
        ),
    )
    vrr_layer.add_to(m)
    vrr_colormap.add_to(m)

    LayerControl(collapsed=False).add_to(m)

    return m


# ------------------------
# Run
# ------------------------
geojson_str, center = make_joined_geojson(EXCEL_FILE, SHP_FILE)

bakken_geojson_str = None
if BAKKEN_AVAILABLE:
    bakken_geojson_str = load_bakken_geojson(BAKKEN_SHP_FILE)

m = build_folium_map(geojson_str, center, bakken_geojson_str)

st.subheader("VRR Map (Shapefile polygons, smooth green colormap)")
st_folium(m, use_container_width=True, height=650, returned_objects=[])

# ------------------------
# Display VRR table
# ------------------------
st.subheader("VRR by Section")

# Load the processed VRR data
df_xlsx = load_xlsx(EXCEL_FILE)
vrr_df = compute_vrr_by_section(df_xlsx)

# Display table in Streamlit
st.dataframe(vrr_df.style.format({"vrr": "{:.3f}"}), use_container_width=True)

# ------------------------
# Download button
# ------------------------
from io import BytesIO

# Save the VRR DataFrame to an in-memory Excel file
output = BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    vrr_df.to_excel(writer, index=False, sheet_name="VRR")
    writer.save()
output.seek(0)

st.download_button(
    label="Download VRR Excel File",
    data=output,
    file_name="vrr_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)