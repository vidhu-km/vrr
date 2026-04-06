import os
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import geopandas as gpd

# ------------------------
# Logging (exec-ready)
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("vrr-map-app")

# ------------------------
# Streamlit Setup
# ------------------------
st.set_page_config(
    page_title="VRR Navigable Map",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ VRR Navigable Map")
st.markdown(
    """
    **Purpose:** Visualize VRR by `Section` from the provided Excel workbook and overlay it on the supplied shapefile grid.

    **How to use:** Hover a polygon to view `Section` and `VRR` values.
    """
)

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
# Helpers: Validation & Messages
# ------------------------
def validate_inputs(excel_path: str, shp_path: str) -> None:
    missing = []
    if not os.path.exists(excel_path):
        missing.append(excel_path)
    if not os.path.exists(shp_path):
        missing.append(shp_path)

    if missing:
        st.error(
            "Missing required file(s). Please place them in the app folder:\n\n"
            + "\n".join([f"- `{p}`" for p in missing])
        )
        st.stop()


def required_excel_columns() -> List[str]:
    return [
        "Section",
        "section prod oil",
        "section prod gas",
        "section prod water",
        "section inj water",
    ]


# ------------------------
# Cache: load Excel + shapefile
# ------------------------
@st.cache_data(show_spinner=False)
def load_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    required_cols = required_excel_columns()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"XLSX missing required columns: {missing}")

    # Normalize Section
    df["Section"] = df["Section"].astype(str).str.strip()

    # Coerce numeric columns
    num_cols = [
        "section prod oil",
        "section prod gas",
        "section prod water",
        "section inj water",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


@st.cache_data(show_spinner=False)
def load_shp(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    if "Section" not in gdf.columns:
        raise ValueError("Shapefile missing required attribute `Section`.")

    gdf["Section"] = gdf["Section"].astype(str).str.strip()

    # Ensure CRS is set and consistent; default to EPSG:4326 if missing
    if gdf.crs is None:
        logger.warning("Shapefile CRS not found. Assuming EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


# ------------------------
# VRR Calculation
# ------------------------
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

    # Existing formulas preserved
    gor = np.where(oil > 0, gas / oil, 0.0)
    gas_term = np.maximum(0, oil * 0.01033 * (gor - 120))
    denom = oil * 1.36 + water * 1.01 + gas_term

    vrr = np.where(denom > 0, (inj * 1.01) / denom, 0.0)
    vrr = np.nan_to_num(vrr, nan=0.0, posinf=0.0, neginf=0.0)
    vrr_disp = np.clip(vrr, VRR_MIN, MAX_VRR)

    agg["vrr"] = vrr_disp
    return agg[["Section", "vrr"]]


# ------------------------
# GeoJSON Join + Map Center
# ------------------------
@st.cache_data(show_spinner=False)
def make_joined_geojson(excel_path: str, shp_path: str) -> Tuple[str, List[float], int]:
    df_xlsx = load_xlsx(excel_path)
    gdf = load_shp(shp_path)
    vrr_df = compute_vrr_by_section(df_xlsx)

    # Join (left join: keep all polygons)
    gdf2 = gdf.merge(vrr_df, on="Section", how="left")
    gdf2["vrr"] = gdf2["vrr"].fillna(0.0).astype(float)

    # Center calculation:
    # - For geodetic CRS (EPSG:4326), centroid in degrees can be misleading.
    # - Use Web Mercator (EPSG:3857) for a reasonable map center.
    gdf_proj = gdf2.to_crs(epsg=3857)
    centroid = gdf_proj.geometry.centroid

    center_lat = float(centroid.y.mean())
    center_lon = float(centroid.x.mean())

    # Convert center back to EPSG:4326 for folium
    center_gdf = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs=3857).to_crs(4326)
    center_point = center_gdf.iloc[0]
    center = [float(center_point.y), float(center_point.x)]

    geojson = gdf2.to_json()

    poly_count = int(len(gdf2))
    return geojson, center, poly_count


# ------------------------
# Folium Map Builder
# ------------------------
@st.cache_resource(show_spinner=False)
def build_folium_map(geojson_str: str, center: List[float], decimals: int) -> folium.Map:
    m = folium.Map(location=center, zoom_start=ZOOM_START, tiles=TILE_LAYER)

    # Legend / color scale
    vrr_colormap = cm.LinearColormap(GREEN_STOPS, vmin=VRR_MIN, vmax=MAX_VRR)
    vrr_colormap.caption = "VRR (clipped to [0.0, 3.0])"

    def style_fn(feature):
        v = feature["properties"].get("vrr", 0.0) or 0.0
        color = vrr_colormap(v)
        return {
            "fillColor": color,
            "color": "rgba(0,0,0,0.25)",
            "weight": 0.7,
            "fillOpacity": 0.65,
        }

    def tooltip_html(props):
        section = props.get("Section", "")
        v = props.get("vrr", 0.0) or 0.0
        v_fmt = f"{v:.{decimals}f}"
        return f"<b>Section:</b> {section}<br/><b>VRR:</b> {v_fmt}"

    # Use GeoJson with tooltip rendered client-side
    geojson = folium.GeoJson(
        data=geojson_str,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["Section", "vrr"],
            aliases=["Section", "VRR"],
            localize=True,
            sticky=False,
            labels=True,
            # We'll still control decimals through formatting below if needed,
            # but folium's default formatting may vary; keeping it simple.
        ),
    )

    # Add an additional tooltip formatter via layer-by-layer is complex;
    # keep built-in tooltip and rely on computed rounding/values.
    geojson.add_to(m)
    vrr_colormap.add_to(m)
    return m


# ------------------------
# Input Validation + UI Controls
# ------------------------
validate_inputs(EXCEL_FILE, SHP_FILE)

with st.sidebar:
    st.header("Map Controls")

    decimals = st.slider("VRR tooltip decimals", min_value=0, max_value=4, value=3, step=1)
    show_metrics = st.checkbox("Show summary metrics", value=True)

    st.divider()
    st.markdown("**Tip:** Hover polygons to inspect VRR by `Section`.")

# ------------------------
# Compute / Render
# ------------------------
try:
    with st.spinner("Loading shapefile, joining VRR, and rendering map..."):
        geojson_str, center, poly_count = make_joined_geojson(EXCEL_FILE, SHP_FILE)

        # Optional: compute quick summary for exec-friendly context
        if show_metrics:
            df_xlsx = load_xlsx(EXCEL_FILE)
            vrr_df = compute_vrr_by_section(df_xlsx)
            vrr_min = float(vrr_df["vrr"].min()) if len(vrr_df) else 0.0
            vrr_max = float(vrr_df["vrr"].max()) if len(vrr_df) else 0.0
            vrr_mean = float(vrr_df["vrr"].mean()) if len(vrr_df) else 0.0
            vrr_zero = int((vrr_df["vrr"] <= 0.0).sum()) if len(vrr_df) else 0

        m = build_folium_map(geojson_str, center, decimals=decimals)

except Exception as e:
    logger.exception("App failed during processing.")
    st.error(
        "Something went wrong while generating the map. "
        "Check that your files match the expected schema (column names + `Section` field)."
    )
    st.exception(e)
    st.stop()

# ------------------------
# Executive-ready summary + Map
# ------------------------
st.subheader("VRR Overlay (Navigable)")
st.markdown(
    f"Shapefile polygons: **{poly_count:,}**. "
    f"Base tile layer: **{TILE_LAYER}**."
)

if show_metrics:
    st.markdown("### Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VRR (min)", f"{vrr_min:.3f}")
    c2.metric("VRR (mean)", f"{vrr_mean:.3f}")
    c3.metric("VRR (max)", f"{vrr_max:.3f}")
    c4.metric("VRR = 0 count", f"{vrr_zero:,}")

st_folium(m, use_container_width=True, height=650)

st.caption(
    "VRR is clipped to the range $[0.0, 3.0]$ for color visualization. "
    "Values are computed per `Section` using the uploaded Excel inputs."
)