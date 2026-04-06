import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import folium
from streamlit_folium import st_folium
import matplotlib

import streamlit.components.v1 as components


# ------------------------
# Streamlit Setup
# ------------------------
st.set_page_config(page_title="VRR Map (Navigable)", layout="wide")
st.title("VRR Map (Navigable)")

# ------------------------
# Configuration
# ------------------------
EXCEL_FILE = "vrr_data.xlsx"

HEAT_RES = 300
MAX_VRR = 3.0
VRR_MIN = 0.0
COLORMAP = "viridis"
OPACITY = 0.65

TILE_LAYER = "CartoDB positron"
ZOOM_START = 7

GRADIENT_PNG = "vrr_gradient_overlay.png"

# Grid arrays kept for hover
# (We will export them to JS as JSON.)
# ------------------------
# Validate input
# ------------------------
if not os.path.exists(EXCEL_FILE):
    st.error(f"Missing input file: `{EXCEL_FILE}`. Place it in the app folder.")
    st.stop()

# ------------------------
# Load & validate data
# ------------------------
df = pd.read_excel(EXCEL_FILE)

required_cols = [
    "performance_oil_cumtodate",
    "performance_gas_cumtodate",
    "performance_water_cumtodate",
    "performance_waterinj_cumtodate",
    "header_midpointlatitude",
    "header_midpointlongitude",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in XLSX: {missing}")
    st.stop()

num_cols = [
    "performance_oil_cumtodate",
    "performance_gas_cumtodate",
    "performance_water_cumtodate",
    "performance_waterinj_cumtodate",
]

df[num_cols] = (
    df[num_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)

lat_col = "header_midpointlatitude"
lon_col = "header_midpointlongitude"

coord_mask = (
    df[lat_col].notna() & df[lon_col].notna() &
    (df[lat_col] != 0) & (df[lon_col] != 0)
)
df = df.loc[coord_mask].copy()

if df.empty:
    st.error("No valid latitude/longitude coordinates found in the XLSX.")
    st.stop()

# ------------------------
# Compute VRR
# ------------------------
oil = df["performance_oil_cumtodate"].to_numpy()
gas = df["performance_gas_cumtodate"].to_numpy()
water = df["performance_water_cumtodate"].to_numpy()
inj = df["performance_waterinj_cumtodate"].to_numpy()

gor = np.where(oil > 0, gas / oil, 0.0)

gas_term = np.maximum(0, oil * 0.01033 * (gor - 120))
denom = oil * 1.36 + water * 1.01 + gas_term

vrr = np.where(denom > 0, (inj * 1.01) / denom, np.nan)
vrr = np.nan_to_num(vrr, nan=0.0, posinf=0.0, neginf=0.0)

vrr_disp = np.clip(vrr, VRR_MIN, MAX_VRR)

# ------------------------
# Build smooth gradient surface
# ------------------------
lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
lat_min, lat_max = df[lat_col].min(), df[lat_col].max()

lon_grid = np.linspace(lon_min, lon_max, HEAT_RES)
lat_grid = np.linspace(lat_min, lat_max, HEAT_RES)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

points = df[[lon_col, lat_col]].to_numpy()
values = vrr_disp

grid_linear = griddata(points, values, (lon_mesh, lat_mesh), method="linear")
grid_nearest = griddata(points, values, (lon_mesh, lat_mesh), method="nearest")

grid_vrr = np.where(np.isnan(grid_linear), grid_nearest, grid_linear)

# ------------------------
# Render gradient overlay PNG (internal only)
# ------------------------
cmap = plt.get_cmap(COLORMAP)

fig = plt.figure(figsize=(8, 8), dpi=150)
ax = fig.add_subplot(111)

ax.imshow(
    grid_vrr,
    extent=[lon_min, lon_max, lat_min, lat_max],
    origin="lower",
    cmap=cmap,
    vmin=VRR_MIN,
    vmax=MAX_VRR,
    interpolation="bicubic",
)

ax.axis("off")
fig.tight_layout(pad=0)
fig.savefig(GRADIENT_PNG, transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)

# ------------------------
# Create interactive map
# ------------------------
center = [df[lat_col].mean(), df[lon_col].mean()]
m = folium.Map(location=center, zoom_start=ZOOM_START, tiles=TILE_LAYER)

bounds = [
    [lat_min, lon_min],
    [lat_max, lon_max],
]

folium.raster_layers.ImageOverlay(
    image=GRADIENT_PNG,
    bounds=bounds,
    opacity=OPACITY,
    interactive=False,
    cross_origin=False,
    zindex=1,
).add_to(m)

st.subheader("VRR Map (Interactive Pan/Zoom + Hover Value)")

# Render map; we capture the HTML so we can hook hover behavior.
map_container = st_folium(m, use_container_width=True, height=650)

# ------------------------
# Hover tooltip via JS sampling from the VRR grid
# ------------------------
# Convert grid + axes to JSON-friendly lists.
# Note: Keep it light; HEAT_RES=300 => 90k numbers. This is ok but can be heavy.
grid_list = grid_vrr.astype(float).tolist()
lon_list = lon_grid.astype(float).tolist()
lat_list = lat_grid.astype(float).tolist()

# To find nearest indices quickly, we’ll map lat/lon -> grid index
# using linear scaling (since grids are evenly spaced).
# grid index:
#   i = floor((lat - lat_min)/(lat_max - lat_min) * (HEAT_RES-1))
#   j = floor((lon - lon_min)/(lon_max - lon_min) * (HEAT_RES-1))

html = f"""
<div id="vrr-hover-holder" style="position:relative;">
  <div id="vrr-tooltip"
       style="
         position:absolute;
         z-index:9999;
         background:rgba(0,0,0,0.7);
         color:white;
         padding:6px 10px;
         border-radius:4px;
         font-family: Arial, sans-serif;
         font-size:13px;
         pointer-events:none;
         display:none;
       ">
    VRR: <span id="vrr-value">--</span>
  </div>
</div>

<script>
(function() {{
  // Folium injects Leaflet. We attach to the first map found in this component.
  // streamlit-folium renders map in its own iframe-less DOM.
  const maps = window.L ? null : null;
  // Locate Leaflet map instance
  // (We try to find map container by scanning for leaflet panes.)
  function findLeafletMap() {{
    if (!window.L) return null;
    for (const id in window.L._maps) {{
      return window.L._maps[id];
    }}
    return null;
  }}

  const map = findLeafletMap();
  if (!map) {{
    // Retry shortly
    setTimeout(() => {{
      const map2 = findLeafletMap();
      if (!map2) return;
      attach(map2);
    }}, 500);
    return;
  }}
  attach(map);

  function attach(mapRef) {{
    const latMin = {lat_min};
    const latMax = {lat_max};
    const lonMin = {lon_min};
    const lonMax = {lon_max};
    const n = {HEAT_RES};

    const grid = {grid_list};

    const tooltip = document.getElementById('vrr-tooltip');
    const valueEl = document.getElementById('vrr-value');

    function clamp(x, a, b) {{
      return Math.max(a, Math.min(b, x));
    }}

    function latLonToIndex(lat, lon) {{
      // Map into [0, n-1]
      const fi = (lat - latMin) / (latMax - latMin) * (n - 1);
      const fj = (lon - lonMin) / (lonMax - lonMin) * (n - 1);
      const i = clamp(Math.round(fi), 0, n-1);
      const j = clamp(Math.round(fj), 0, n-1);
      return [i, j];
    }}

    mapRef.on('mousemove', function(e) {{
      const lat = e.latlng.lat;
      const lon = e.latlng.lng;

      // Only show if within overlay bounds
      if (lat < latMin || lat > latMax || lon < lonMin || lon > lonMax) {{
        tooltip.style.display = 'none';
        return;
      }}

      const [i, j] = latLonToIndex(lat, lon);
      const v = grid[i][j];

      valueEl.textContent = (v ?? 0).toFixed(3) ;
      tooltip.style.display = 'block';

      // Position tooltip near cursor
      const point = mapRef.latLngToContainerPoint(e.latlng);
      tooltip.style.left = (point.x + 12) + 'px';
      tooltip.style.top  = (point.y + 12) + 'px';
    }});

    mapRef.on('mouseout', function(e) {{
      tooltip.style.display = 'none';
    }});
  }}
}})();
</script>
"""

components.html(html, height=0)