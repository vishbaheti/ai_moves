"""Build the LAD x LAD commuting matrix from the NOMIS WU01UK origin-destination
2011 Census file.

The NOMIS export is wide-format with quirks: 8 preamble rows, a header row of
codes plus a row of names (the very first destination column has the name in
the codes row, code missing - we back-fill by name), each origin appears as
two duplicate rows, and the trailing ~20 columns are aggregates (UK, regions,
'mainly works at home', etc.) we drop.

We map 2011 'merged LAD' geography to current (2024) names. Most LADs match
on name; the exceptions are post-2011 mergers (Cumbria, North Yorkshire,
Northants, Suffolk, Bucks, Dorset, Somerset, NI 2014 reorganisation,
Westminster + City of London merged in 2011 geography).

Inputs (place in same folder, or pass as argv):
    752526794144824.csv        - the NOMIS commute matrix (rename if needed)
    ons_panel_lad_2014_2025.csv - just for verifying the name mapping

Output:
    commute_matrix_2011.csv    - square matrix indexed by current LAD names
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = "C:/Users/vishe/Downloads/Assignment/Datasets"

LAD_CODE_RE = re.compile(r"^(E41|S12|W40|95[A-Z]{2})")

# 2011 (or pre-merger) name -> 2024 name.
NAME_MAP = {
    # Cumbria 2023
    "Allerdale": "Cumberland", "Carlisle": "Cumberland", "Copeland": "Cumberland",
    "Barrow-in-Furness": "Westmorland and Furness",
    "Eden": "Westmorland and Furness",
    "South Lakeland": "Westmorland and Furness",
    # North Yorkshire 2023
    "Craven": "North Yorkshire", "Hambleton": "North Yorkshire",
    "Harrogate": "North Yorkshire", "Richmondshire": "North Yorkshire",
    "Ryedale": "North Yorkshire", "Scarborough": "North Yorkshire",
    "Selby": "North Yorkshire",
    # Northants 2021
    "Corby": "North Northamptonshire",
    "East Northamptonshire": "North Northamptonshire",
    "Kettering": "North Northamptonshire",
    "Wellingborough": "North Northamptonshire",
    "Daventry": "West Northamptonshire",
    "Northampton": "West Northamptonshire",
    "South Northamptonshire": "West Northamptonshire",
    # Suffolk 2019
    "Forest Heath": "West Suffolk", "St Edmundsbury": "West Suffolk",
    "Suffolk Coastal": "East Suffolk", "Waveney": "East Suffolk",
    # Bucks 2020
    "Aylesbury Vale": "Buckinghamshire", "Chiltern": "Buckinghamshire",
    "South Bucks": "Buckinghamshire", "Wycombe": "Buckinghamshire",
    # Dorset / BCP 2019
    "Christchurch": "Bournemouth Christchurch and Poole",
    "Bournemouth": "Bournemouth Christchurch and Poole",
    "Poole": "Bournemouth Christchurch and Poole",
    "East Dorset": "Dorset", "North Dorset": "Dorset",
    "Purbeck": "Dorset", "West Dorset": "Dorset",
    "Weymouth and Portland": "Dorset",
    # Somerset 2023
    "Mendip": "Somerset", "Sedgemoor": "Somerset",
    "South Somerset": "Somerset", "Taunton Deane": "Somerset",
    "West Somerset": "Somerset",
    # 2011 Census quirks
    "Westminster,City of London": "Westminster",
    "Cornwall,Isles of Scilly": "Cornwall",
    "Kingston upon Hull, City of": "Kingston upon Hull City of",
    "Herefordshire, County of": "Herefordshire County of",
    # Renamed
    "Shepway": "Folkestone and Hythe",
    # NI 2014 reorganisation (26 -> 11)
    "Magherafelt": "Mid Ulster", "Cookstown": "Mid Ulster", "Dungannon": "Mid Ulster",
    "Ballymena": "Mid and East Antrim", "Larne": "Mid and East Antrim",
    "Carrickfergus": "Mid and East Antrim",
    "Strabane": "Derry City and Strabane", "Derry": "Derry City and Strabane",
    "Limvady": "Causeway Coast and Glens",
    "Coleraine": "Causeway Coast and Glens",
    "Ballymoney": "Causeway Coast and Glens", "Moyle": "Causeway Coast and Glens",
    "Newry and Mourne": "Newry Mourne and Down", "Down": "Newry Mourne and Down",
    "Omagh": "Fermanagh and Omagh", "Fermanagh": "Fermanagh and Omagh",
    "Armagh": "Armagh City Banbridge and Craigavon",
    "Banbridge": "Armagh City Banbridge and Craigavon",
    "Craigavon": "Armagh City Banbridge and Craigavon",
    "Castlereagh": "Lisburn and Castlereagh",
    "Lisburn": "Lisburn and Castlereagh",
    "Ards": "Ards and North Down", "North Down": "Ards and North Down",
    "Antrim": "Antrim and Newtownabbey",
    "Newtownabbey": "Antrim and Newtownabbey",
}


def parse_nomis(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, skiprows=8, low_memory=False, on_bad_lines="skip")

    # Data rows are those whose first cell is an integer (the row number)
    data_idx = []
    for i in range(len(raw)):
        try:
            int(float(str(raw.iat[i, 0])))
            data_idx.append(i)
        except (ValueError, TypeError):
            continue

    # Each origin is duplicated; keep one of each by origin code (last column)
    data = raw.iloc[data_idx].drop_duplicates(subset=[raw.shape[1] - 1]).reset_index(drop=True)

    origin_codes = data.iloc[:, -1].astype(str).tolist()
    origin_names = data.iloc[:, -2].astype(str).tolist()
    origin_name_to_code = dict(zip(origin_names, origin_codes))

    # Destination codes: row 0 generally has codes, row 1 names. The first
    # destination column has the name in row 0 instead of the code - back-fill
    # via the origin lookup.
    ncols = raw.shape[1]
    dest_names: list[str] = []
    dest_codes: list[str | None] = []
    for c in range(1, ncols - 2):
        cell0 = str(raw.iat[0, c]) if pd.notna(raw.iat[0, c]) else ""
        cell1 = str(raw.iat[1, c]) if pd.notna(raw.iat[1, c]) else ""
        if cell0.startswith("{place of work}"):
            dest_names.append(cell0.replace("{place of work}", "").strip())
            dest_codes.append(None)
        elif cell1.startswith("{place of work}"):
            dest_names.append(cell1.replace("{place of work}", "").strip())
            dest_codes.append(cell0.strip())
        else:  # Trailing aggregate columns
            dest_names.append(cell1.strip())
            dest_codes.append(cell0.strip())

    dest_codes = [c if c else origin_name_to_code.get(n)
                  for c, n in zip(dest_codes, dest_names)]

    flow = data.iloc[:, 1:-2].copy()
    flow.columns = dest_names
    flow.index = origin_names
    flow = flow.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Drop non-LAD destination columns (UK total, regions, "Mainly work at home", etc)
    is_lad = [bool(c and LAD_CODE_RE.match(c)) for c in dest_codes]
    return flow.loc[:, is_lad]


def to_modern_geography(flow: pd.DataFrame) -> pd.DataFrame:
    """Apply NAME_MAP to both axes and aggregate flows that share a target name."""
    flow.index = [NAME_MAP.get(n, n) for n in flow.index]
    flow.columns = [NAME_MAP.get(n, n) for n in flow.columns]
    return flow.groupby(level=0).sum().T.groupby(level=0).sum().T


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "752526794144824.csv"
    if not src.exists():
        sys.exit(f"NOMIS file not found at {src}")

    flow = parse_nomis(src)
    print(f"Parsed {flow.shape[0]} origins x {flow.shape[1]} LAD destinations")

    flow = to_modern_geography(flow)
    print(f"After name mapping: {flow.shape[0]} x {flow.shape[1]}")

    # Restrict to LADs in our panel (drops Bristol/Eilean Siar/Vale of Glamorgan
    # which have minor name differences not worth handling explicitly)
    panel_path = HERE / "ons_panel_lad_2014_2025.csv"
    if panel_path.exists():
        panel_names = set(pd.read_csv(panel_path, usecols=["region_name"])["region_name"])
        common = sorted(set(flow.index) & set(flow.columns) & panel_names)
        flow = flow.loc[common, common]
        print(f"After restricting to ONS panel LADs: {flow.shape}")

    out = HERE / "commute_matrix_2011.csv"
    flow.to_csv(out)
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
