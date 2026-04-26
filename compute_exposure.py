import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURATION & PATHS ---
# Converted to Path object so the / operator works correctly
HERE = Path("C:/Users/vishe/Downloads/Assignment/Datasets")
PANEL_CSV   = HERE / "ons_panel_lad_2014_2025.csv"
COMMUTE_CSV = HERE / "commute_matrix_2011.csv"

# --- SECTOR EXPOSURE DATA ---
sector_scores = pd.DataFrame([
    ("01-03", 0.16, "Agriculture/forestry/fishing: largely physical, weather-dependent work."),
    ("05-39", 0.32, "Production: mix of physical work and engineering/design tasks."),
    ("41-43", 0.20, "Construction: physical site work limits direct AI exposure."),
    ("45",     0.34, "Motor trades: mix of manual repair work and customer/admin tasks."),
    ("46",     0.45, "Wholesale: heavy admin, logistics, and B2B sales tasks."),
    ("47",     0.38, "Retail: customer service highly exposed, in-store tasks less so."),
    ("49-53", 0.30, "Transport & storage: physical movement, but admin and routing exposed."),
    ("55-56", 0.26, "Accommodation & food services: customer-facing physical service."),
    ("58-63", 0.62, "Information & communication: highest exposure - software, media, telecoms."),
    ("64-66", 0.60, "Finance & insurance: high exposure but mixed substitution/augmentation."),
    ("68",     0.48, "Property: admin-heavy, valuation and customer-facing tasks exposed."),
    ("69-75", 0.58, "Professional, scientific & technical: legal, accounting, consulting all high."),
    ("77-82", 0.56, "Business admin & support: clerical/admin tasks highly substitutable."),
    ("84",     0.50, "Public administration: heavy text/admin work, judgement-dependent."),
    ("85",     0.42, "Education: teaching judgement remains central."),
    ("86-88", 0.34, "Health: heavy human-contact work; admin tasks exposed but care less so."),
    ("90-99", 0.38, "Arts, entertainment & other services: mixed creative/personal-service."),
], columns=["sic_codes", "exposure", "rationale"])

SECTOR_MAP = dict(zip(sector_scores['sic_codes'], sector_scores['exposure']))

# --- CORE LOGIC FUNCTIONS ---

def own_exposure(panel: pd.DataFrame, year: int, scores_map: dict) -> pd.DataFrame:
    """Enterprise-weighted average sector exposure per LAD."""
    df = panel.loc[panel["year"] == year, ["region_code", "region_name", 
                                            "sic_codes", "enterprises"]].copy()
    
    df["share"] = df.groupby("region_code")["enterprises"].transform(
        lambda s: s / s.sum()
    )
    
    df["weighted"] = df["share"] * df["sic_codes"].map(scores_map)
    
    return (df.groupby(["region_code", "region_name"], as_index=False)
              .agg(own_exposure=("weighted", "sum"),
                   total_enterprises=("enterprises", "sum")))

def shed_exposure(own: pd.DataFrame, commute: pd.DataFrame) -> pd.DataFrame:
    """Commuting-weighted mean of own exposure (outward spillover only)."""
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
    """Identify LADs with low internal risk but high external risk exposure."""
    df = df.copy()
    own_med = df["own_exposure"].median()
    shed_75 = df["shed_exposure"].quantile(0.75)
    
    df["sleeper"] = (df["own_exposure"] < own_med) & (df["shed_exposure"] >= shed_75)
    df["own_pct"]  = df["own_exposure"].rank(pct=True) * 100
    df["shed_pct"] = df["shed_exposure"].rank(pct=True) * 100
    
    return df

def compute_pipeline(year: int, scores_map: dict):
    """Full processing pipeline."""
    try:
        panel = pd.read_csv(PANEL_CSV)
        panel_lad = panel[panel["area_type"] == "LAD"]
        commute = pd.read_csv(COMMUTE_CSV, index_col=0)
    except FileNotFoundError:
        print(f"Error: Data files not found in {HERE}")
        return None, None, None

    own = own_exposure(panel_lad, year, scores_map)
    df  = shed_exposure(own, commute)
    df  = add_sleeper_flag(df)
    
    return df, commute, panel

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"Mean exposure across sectors: {sector_scores['exposure'].mean():.2f}")
    
    # 1. Run Pipeline
    results_df, commute_matrix, raw_panel = compute_pipeline(2024, SECTOR_MAP)
    
    if results_df is not None:
        # 2. Display Results
        sleepers = results_df[results_df["sleeper"]].sort_values("shed_exposure", ascending=False)
        
        print(f"\n2024 Analysis Summary:")
        print(f"- Total LADs analyzed: {len(results_df)}")
        print(f"- Sleeper LADs identified: {len(sleepers)}")
        print("-" * 50)
        print(sleepers[["region_name", "own_exposure", "shed_exposure", 
                        "total_enterprises", "shed_pct"]].round(3).to_string(index=False))

        # 3. Verification Checks
        print("\n--- Running Verification Checks ---")
        
        # Check for unmapped sectors
        missing_sectors = set(raw_panel['sic_codes']) - set(SECTOR_MAP.keys())
        print(f"Unmapped sectors: {missing_sectors}")
        
        # Check LAD alignment
        print(f"LADs in Panel: {raw_panel['region_name'].nunique()}")
        print(f"LADs in Results: {results_df['region_name'].nunique()}")
        
        # Assert range
        try:
            assert results_df['own_exposure'].between(0.1, 0.7).all()
            print("Check passed: own_exposure is within expected bounds (0.1 - 0.7).")
        except AssertionError:
            print("Check FAILED: own_exposure is outside expected bounds!")