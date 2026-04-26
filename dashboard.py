"""
Streamlit dashboard: Sleeper LADs and the AI labour shed across 2017-2025.
Version 2.3: Red Sleeper flows (Dark), Yellow Global flows, and Blue Choropleth.
"""

from __future__ import annotations
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sleeper LADs: AI exposure via the labour shed",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent
PANEL_CSV   = DATA_DIR / "ons_panel_lad_2014_2025.csv"
COMMUTE_CSV = DATA_DIR / "commute_matrix_2011.csv"
SHAPE_ZIP   = next(DATA_DIR.glob("Local_Authority_Districts_December_2024*.zip"), None)

SECTOR_EXPOSURE = {
    "01-03": 0.16, "05-39": 0.32, "41-43": 0.20, "45": 0.34, "46": 0.45,
    "47":    0.38, "49-53": 0.30, "55-56": 0.26, "58-63": 0.62, "64-66": 0.60,
    "68":    0.48, "69-75": 0.58, "77-82": 0.56, "84": 0.50, "85": 0.42,
    "86-88": 0.34, "90-99": 0.38,
}

SLEEPER_PERCENTILE = 75   
OWN_PERCENTILE = 50       

# ---------------------------------------------------------------------------
# Data loading and computation
# ---------------------------------------------------------------------------

@st.cache_data
def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_CSV)
    panel = panel[panel["area_type"] == "LAD"].copy()
    return panel

@st.cache_data
def load_commute() -> pd.DataFrame:
    return pd.read_csv(COMMUTE_CSV, index_col=0)

@st.cache_data
def load_lads_shape():
    if SHAPE_ZIP is None or not SHAPE_ZIP.exists():
        return None
    return gpd.read_file(f"zip://{SHAPE_ZIP.as_posix()}").to_crs(4326)

@st.cache_data
def compute_for_year(year: int) -> pd.DataFrame:
    panel = load_panel()
    yr = panel[panel["year"] == year].copy()
    yr["share"] = yr.groupby("region_code")["enterprises"].transform(
        lambda s: s / s.sum()
    )
    yr["sector_exposure"] = yr["sic_codes"].map(SECTOR_EXPOSURE)
    yr["w_exposure"] = yr["share"] * yr["sector_exposure"]

    own = (yr.groupby(["region_code", "region_name"], as_index=False)
             .agg(own_exposure=("w_exposure", "sum"),
                  total_enterprises=("enterprises", "sum")))

    commute = load_commute()
    common = sorted(set(own["region_name"]) & set(commute.index) & set(commute.columns))
    own_idx = own.set_index("region_name").loc[common]

    W = commute.loc[common, common].astype(float).values
    Wd = W.copy()
    np.fill_diagonal(Wd, 0.0)
    rs = Wd.sum(axis=1, keepdims=True)
    rs[rs == 0] = np.nan
    W_spill = np.nan_to_num(Wd / rs, nan=0.0)

    own_idx["shed_exposure"] = W_spill @ own_idx["own_exposure"].values
    return own_idx.reset_index()

@st.cache_data
def get_sector_composition(year: int, region_code: str) -> pd.DataFrame:
    panel = load_panel()
    sub = panel[(panel["year"] == year) & (panel["region_code"] == region_code)].copy()
    sub = sub.sort_values("enterprises", ascending=False)
    sub["share"] = sub["enterprises"] / sub["enterprises"].sum()
    return sub[["industry_group", "enterprises", "share"]]

def top_destinations(origin_name: str, top_n: int = 3) -> pd.DataFrame:
    commute = load_commute()
    if origin_name not in commute.index:
        return pd.DataFrame(columns=["destination", "share"])
    row = commute.loc[origin_name].copy()
    if origin_name in row.index:
        row[origin_name] = 0
    total = row.sum()
    if total == 0:
        return pd.DataFrame(columns=["destination", "share"])
    top = row.nlargest(top_n).reset_index()
    top.columns = ["destination", "flow"]
    top["share"] = top["flow"] / total
    return top[["destination", "share"]]

# ---------------------------------------------------------------------------
# UI Header
# ---------------------------------------------------------------------------

st.title("Sleeper LADs: AI exposure via the labour shed")
st.markdown(
    "**Sleeper LADs** have low own-business AI exposure but high **labour-shed** exposure. "
    "These regions are structurally vulnerable due to the markets their residents commute into."
)
st.info("💡 **Interactive Map:** Hover over any region to see the top industry sectors and exposure metrics.")

with st.sidebar:
    st.header("Navigation")
    year = st.slider("Select Year", min_value=2017, max_value=2025, value=2024, step=1)
    
    st.divider()
    st.subheader("Commuting Visuals")
    
    show_sleeper_lines = st.checkbox(
        "Show Sleeper LAD flows", value=True,
        help="Highlight flows from sleeper LADs to their primary work hubs"
    )
    
    show_global_lines = st.checkbox(
        "Show Global Commute lines", value=True,
        help="Visualize the major commuting corridors across the UK"
    )
    
    n_global = st.slider("Max global lines", min_value=10, max_value=500, value=150, step=10)
    
    st.markdown("---")
    st.markdown("**Flow Legend:**")
    st.markdown("🔴 **Dark Red:** Sleeper LAD flows")
    st.markdown("🟡 **Yellow:** Global commuting flows")

# ---------------------------------------------------------------------------
# Map Engine
# ---------------------------------------------------------------------------

with st.spinner("Compiling regional data and generating UK map..."):
    data = compute_for_year(year).copy()
    own_cutoff = float(data["own_exposure"].quantile(OWN_PERCENTILE / 100))
    shed_cutoff = float(data["shed_exposure"].quantile(SLEEPER_PERCENTILE / 100))
    data["is_sleeper"] = (
        (data["own_exposure"] < own_cutoff) & (data["shed_exposure"] >= shed_cutoff)
    )

    n_sleeper = int(data["is_sleeper"].sum())
    median_shed = data["shed_exposure"].median()

    # Layout Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Sleeper LADs", f"{n_sleeper}")
    m2.metric("Median Shed Exposure", f"{median_shed:.3f}")
    m3.metric("Current Year", f"{year}")

    shape = load_lads_shape()
    if shape is None:
        st.error("Error: Local Authority District shapefile not found.")
        st.stop()

    shape_data = shape.merge(data, left_on="LAD24CD", right_on="region_code", how="left")

    # Blue Color Gradient
    shed_min, shed_max = float(data["shed_exposure"].min()), float(data["shed_exposure"].max())
    colormap = cm.linear.Blues_09.scale(shed_min, shed_max)
    colormap.caption = f"Labour-shed AI exposure ({year})"

    def style_function(feature):
        val = feature["properties"].get("shed_exposure")
        is_sleeper = feature["properties"].get("is_sleeper", False)
        if val is None or pd.isna(val):
            return {"fillColor": "#f5f5f5", "color": "white", "weight": 0.1, "fillOpacity": 0.4}
        return {
            "fillColor": colormap(val),
            "color": "#d62728" if is_sleeper else "white", 
            "weight": 2.2 if is_sleeper else 0.4,
            "fillOpacity": 0.8,
        }

    def tooltip_html(row, comp_df: pd.DataFrame) -> str:
        name = row.get("LAD24NM", "Unknown")
        own = row.get("own_exposure")
        shed = row.get("shed_exposure")
        sleeper_note = "<span style='color:#d62728; font-weight:bold'>• SLEEPER LAD</span><br>" if row.get("is_sleeper") else ""
        
        industry_rows = "".join(
            f"<tr><td style='font-size:10px; padding-right:10px'>{r['industry_group'][:28]}</td>"
            f"<td style='text-align:right; font-size:10px; font-weight:bold'>{r['share']*100:.1f}%</td></tr>"
            for _, r in comp_df.head(5).iterrows()
        )
        return f"""
        <div style='font-family:sans-serif; width:240px; padding:5px'>
          <strong style='font-size:14px'>{name}</strong><br>{sleeper_note}
          <div style='margin-top:5px'>Own Exposure: <b>{own:.3f}</b></div>
          <div>Shed Exposure: <b>{shed:.3f}</b></div>
          <hr style='margin:8px 0'>
          <strong style='font-size:11px'>Top Industries:</strong>
          <table style='width:100%; border-spacing:0'>{industry_rows}</table>
        </div>
        """

    shape_data["_tooltip"] = ""
    for idx, row in shape_data.iterrows():
        if not pd.isna(row.get("region_code")):
            comp = get_sector_composition(year, row["region_code"])
            shape_data.at[idx, "_tooltip"] = tooltip_html(row, comp)
        else:
            shape_data.at[idx, "_tooltip"] = f"<b>{row.get('LAD24NM')}</b><br>Data Unavailable"

    # Map setup
    m = folium.Map(location=[54.2, -2.5], zoom_start=6, tiles="cartodbpositron")
    colormap.add_to(m)

    # Add Regions
    folium.GeoJson(
        shape_data.to_json(),
        style_function=style_function,
        highlight_function=lambda x: {"weight": 3, "color": "#000", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(fields=["_tooltip"], aliases=[""], labels=False, sticky=True)
    ).add_to(m)

    # Centroids for line drawing
    centroids = shape_data[["LAD24NM", "geometry"]].copy()
    centroids["centroid"] = centroids.geometry.representative_point()
    name_to_centroid = {r.LAD24NM: (r.centroid.y, r.centroid.x) for r in centroids.itertuples()}

    # Global flows (Yellow)
    if show_global_lines:
        commute = load_commute()
        long_commute = commute.stack().reset_index()
        long_commute.columns = ['origin', 'dest', 'flow']
        long_commute = long_commute[long_commute['origin'] != long_commute['dest']]
        top_flows = long_commute.nlargest(n_global, 'flow')
        
        global_layer = folium.FeatureGroup(name="Global Flows")
        for _, row in top_flows.iterrows():
            if row['origin'] in name_to_centroid and row['dest'] in name_to_centroid:
                folium.PolyLine(
                    locations=[name_to_centroid[row['origin']], name_to_centroid[row['dest']]],
                    weight=1.2, color="#FFD700", opacity=0.4 # Yellow
                ).add_to(global_layer)
        global_layer.add_to(m)

    # Sleeper flows (Dark Red)
    if show_sleeper_lines and n_sleeper > 0:
        sleeper_layer = folium.FeatureGroup(name="Sleeper Flows")
        sleepers = data[data["is_sleeper"]]
        for _, sr in sleepers.iterrows():
            origin = sr["region_name"]
            if origin in name_to_centroid:
                for _, td in top_destinations(origin, top_n=3).iterrows():
                    if td["destination"] in name_to_centroid:
                        folium.PolyLine(
                            locations=[name_to_centroid[origin], name_to_centroid[td["destination"]]],
                            weight=2.5, color="#d62728", opacity=0.6, # Dark Red
                            tooltip=f"Flow: {origin} ➔ {td['destination']}"
                        ).add_to(sleeper_layer)
        sleeper_layer.add_to(m)

    st_folium(m, width=None, height=750, returned_objects=[])