"""Build the briefing-note figure: rank scatter (left) + commute destinations
(right), for the year set in YEAR.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


HERE = "C:/Users/vishe/Downloads/Assignment/Datasets"
YEAR = 2024
OUT_PNG = HERE / "fig_sleepers_brief.png"
OUT_PDF = HERE / "fig_sleepers_brief.pdf"

PANEL_CSV   = HERE / "ons_panel_lad_2014_2025.csv"
COMMUTE_CSV = HERE / "commute_matrix_2011.csv"

# Drop these from the destinations panel - they're not LADs
NON_LAD_DEST_PATTERNS = ("Mainly work at", "No fixed", "work mainly",
                         "Offshore", "Outside UK", "home or off")

# Destination colour groupings
LONDON_CORE = {"Westminster", "Camden", "City of London", "Tower Hamlets",
               "Islington", "Southwark", "Hackney", "Lambeth"}
INNER_LDN   = {"Newham", "Greenwich", "Lewisham", "Wandsworth",
               "Kensington and Chelsea", "Hammersmith and Fulham",
               "Bromley", "Redbridge", "Waltham Forest", "Enfield",
               "Haringey", "Barnet", "Havering", "Barking and Dagenham"}
EDINBURGH = {"Edinburgh", "City of Edinburgh"}
ABERDEEN  = {"Aberdeen", "Aberdeen City"}

LABEL_POS = {  # (data x, data y, alignment) - hand-tuned for 2024 sleepers
    "Bexley":               (66, 97, "left"),
    "Aberdeenshire":        (3,  92, "left"),
    "Barking and Dagenham": (3,  85, "left"),
    "Havering":             (3,  78, "left"),
    "Broxbourne":           (3,  71, "left"),
    "East Lothian":         (3,  64, "left"),
    "Midlothian":           (66, 88, "left"),
    "Luton":                (66, 67, "left"),
}


def top_destinations(commute: pd.DataFrame, origin: str, n: int = 3):
    if origin not in commute.index:
        return []
    row = commute.loc[origin].copy()
    if origin in row.index:
        row[origin] = 0
    row = row[[d for d in row.index
               if not any(p.lower() in d.lower() for p in NON_LAD_DEST_PATTERNS)]]
    if row.sum() == 0:
        return []
    return [(d, v / row.sum()) for d, v in row.nlargest(n).items()]


def categorise(name: str) -> tuple[str, str, int]:
    """(label, hex colour, sort order)"""
    if name in LONDON_CORE: return ("London core", "#d62728", 0)
    if name in INNER_LDN:   return ("Other inner London", "#fdae61", 1)
    if name in EDINBURGH:   return ("Edinburgh", "#2c7bb6", 2)
    if name in ABERDEEN:    return ("Aberdeen", "#1a9641", 3)
    return ("Other destinations", "#999999", 4)


def draw_scatter(ax, df: pd.DataFrame) -> None:
    others = df[~df["sleeper"]]
    sleepers = df[df["sleeper"]]

    ax.scatter(others["own_pct"], others["shed_pct"],
               s=18, alpha=0.4, color="#888888", zorder=2)

    ax.plot([0, 100], [0, 100], "--", color="#555555",
            linewidth=0.7, alpha=0.6, zorder=1)
    ax.text(53, 49, "Equal own and shed exposure", fontsize=8,
            color="#555555", ha="left", style="italic",
            rotation=42, rotation_mode="anchor", alpha=0.8)

    ax.scatter(sleepers["own_pct"], sleepers["shed_pct"],
               s=80, alpha=0.95, color="#d62728",
               edgecolor="black", linewidth=0.6, zorder=5)

    for _, r in sleepers.iterrows():
        if r["region_name"] not in LABEL_POS:
            continue
        lx, ly, ha = LABEL_POS[r["region_name"]]
        ax.plot([r["own_pct"], lx], [r["shed_pct"], ly],
                color="#999999", linewidth=0.6, alpha=0.7, zorder=3)
        ax.text(lx, ly, r["region_name"], fontsize=9, fontweight="bold",
                color="#a01010", ha=ha, va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))

    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Own AI exposure percentile (100 = most exposed)", fontsize=9)
    ax.set_ylabel("Labour-shed AI exposure percentile (100 = most exposed)", fontsize=9)
    ax.set_title("Sleeper LADs sit far above the diagonal: their workforces\n"
                 "are more exposed than their local businesses suggest",
                 fontsize=10.5, fontweight="bold", loc="left")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=6, alpha=0.6, label=f"Other LADs (n={len(others)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=9,
               label=f"Sleeper LADs (n={len(sleepers)})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_destinations(ax, df: pd.DataFrame, commute: pd.DataFrame) -> None:
    sleepers = df[df["sleeper"]].sort_values("shed_exposure", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(sleepers))[::-1]
    seen: set[str] = set()

    for i, lad in enumerate(sleepers["region_name"]):
        parts = [(d, s, *categorise(d)) for d, s in top_destinations(commute, lad, 3)]
        parts.sort(key=lambda x: x[4])
        left = 0.0
        for d, share, cat, col, _ in parts:
            label = cat if cat not in seen else None
            seen.add(cat)
            ax.barh(y_pos[i], share, left=left, color=col,
                    edgecolor="white", linewidth=0.5, label=label, height=0.65)
            if share >= 0.10:
                ax.text(left + share / 2, y_pos[i], d[:14],
                        va="center", ha="center", fontsize=7,
                        color="white", fontweight="bold")
            left += share

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sleepers["region_name"], fontsize=9)
    ax.set_xlabel("Share of LAD's outward commuters", fontsize=9)
    ax.set_title("Where their workers commute (top 3 destinations)",
                 fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

def shed_exposure(own: pd.DataFrame, commute: pd.DataFrame) -> pd.DataFrame:
    """Add a shed_exposure column - commuting-weighted mean of own exposure
    over the LADs each LAD's residents commute to. Diagonal zeroed first
    so it measures outward spillover only.

    Restricts the result to LADs present in both `own` and `commute`."""
    common = sorted(set(own["region_name"]) & set(commute.index) & set(commute.columns))
    own_aligned = own.set_index("region_name").loc[common].copy()

    W = commute.loc[common, common].astype(float).values
    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)
    rs = W_off.sum(axis=1, keepdims=True)
    rs[rs == 0] = np.nan
    W_spill = np.nan_to_num(W_off / rs, nan=0.0)

    own_aligned["shed_exposure"] = W_spill @ own_aligned["own_exposure"].values
    return own_aligned.reset_index()


def add_sleeper_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Within-year ranks: below-median own AND top-quartile shed."""
    df = df.copy()
    df["own_pct"]  = df["own_exposure"].rank(ascending=True, pct=True) * 100
    df["shed_pct"] = df["shed_exposure"].rank(ascending=True, pct=True) * 100
    own_med = df["own_exposure"].median()
    shed_75 = df["shed_exposure"].quantile(0.75)
    df["sleeper"] = (df["own_exposure"] < own_med) & (df["shed_exposure"] >= shed_75)
    return df

def own_exposure(panel: pd.DataFrame, year: int,
                 scores: dict[str, float] | None = None) -> pd.DataFrame:
    """Enterprise-weighted average sector exposure per LAD."""
    scores = scores 
    df = panel.loc[panel["year"] == year, ["region_code", "region_name",
                                            "sic_codes", "enterprises"]].copy()
    df["share"] = df.groupby("region_code")["enterprises"].transform(
        lambda s: s / s.sum()
    )
    df["weighted"] = df["share"] * df["sic_codes"].map(scores)
    return (df.groupby(["region_code", "region_name"], as_index=False)
              .agg(own_exposure=("weighted", "sum"),
                   total_enterprises=("enterprises", "sum")))


def compute(year: int,
            scores: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: full pipeline. Returns (df_with_metrics, commute_matrix)."""
    panel = pd.read_csv(PANEL_CSV)
    panel = panel[panel["area_type"] == "LAD"]
    commute = pd.read_csv(COMMUTE_CSV, index_col=0)

    own = own_exposure(panel, year, scores)
    df  = shed_exposure(own, commute)
    df  = add_sleeper_flag(df)
    return df, commute


def main() -> None:
    df, commute = compute(YEAR)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    draw_scatter(ax_left, df)
    draw_destinations(ax_right, df, commute)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"-> {OUT_PNG.name} and {OUT_PDF.name}")


if __name__ == "__main__":
    main()
