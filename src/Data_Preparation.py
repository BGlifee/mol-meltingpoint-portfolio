# src/Build_features.py

from pathlib import Path
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_EXT     = PROJECT_ROOT / "data" / "external"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"

# Canonicalize SMILES strings
def canonicalize(smiles: str) -> str | None:
    try:
        if not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles.strip())
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None

# Ensure canonical SMILES column in DataFrame
def ensure_canonical(df: pd.DataFrame,
                     smiles_col: str = "SMILES",
                     out_col: str = "canonical_smiles") -> pd.DataFrame:
    df = df.copy()
    df[smiles_col] = df[smiles_col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if out_col not in df.columns or df[out_col].isna().all():
        df[out_col] = df[smiles_col].apply(canonicalize)
    return df

# Load external Bradley melting point dataset
def load_external_bradley() -> pd.DataFrame:
    file_path = DATA_EXT / "BradleyMeltingPointDataset.xlsx"
    dfs = pd.read_excel(file_path, sheet_name=["Sheet1", "Sheet2", "Sheet3"])
    df = pd.concat(dfs.values(), ignore_index=True)

    df.columns = df.columns.str.strip()

    selected_cols = ["smiles", "name", "mpC"]
    df = df[selected_cols].drop_duplicates(subset=["smiles"])

    # Known bad row (corrupt / invalid entry)
    if 248849 in df.index:
        df = df.drop(index=248849)

    return df.reset_index(drop=True)

# Align external dataset columns with training dataset and merge
def align_and_merge_with_train(train: pd.DataFrame,
                               ext_raw: pd.DataFrame) -> pd.DataFrame:
    # normalize external columns
    ext = ext_raw.copy()
    ext.columns = ext.columns.str.strip()
    lower_map = {c.lower(): c for c in ext.columns}

    if "SMILES" not in ext.columns and "smiles" in lower_map:
        ext = ext.rename(columns={lower_map["smiles"]: "SMILES"})

    if "Tm" not in ext.columns and "mpc" in lower_map:
        ext["Tm"] = pd.to_numeric(ext[lower_map["mpc"]], errors="coerce") + 273.15

    if "name" not in ext.columns and "NAME" in ext.columns:
        ext = ext.rename(columns={"NAME": "name"})

    target_cols = ["id", "SMILES", "Tm", "name"]
    for col in target_cols:
        if col not in ext.columns:
            ext[col] = pd.NA

    ext = ext[target_cols]

    # basic schema check
    required = {"id", "SMILES", "Tm"}
    missing = required - set(train.columns)
    assert not missing, f"Train is missing columns: {missing}"

    ext_aligned = ext.reindex(columns=train.columns)

    merged = pd.concat([train, ext_aligned], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["SMILES"], keep="first")

    return merged

# Remove test set leakage based on canonical SMILES
def remove_test_leakage(merged: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    merged_c  = ensure_canonical(merged,  smiles_col="SMILES", out_col="canonical_smiles")
    df_test_c = ensure_canonical(df_test, smiles_col="SMILES", out_col="canonical_smiles")

    merged_c  = merged_c.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    df_test_c = df_test_c.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

    test_set = set(df_test_c["canonical_smiles"].dropna())
    overlap_ct = merged_c["canonical_smiles"].isin(test_set).sum()
    print(f"[leakage] overlaps (canonical): {overlap_ct}")

    merged_dedup = merged_c[~merged_c["canonical_smiles"].isin(test_set)].reset_index(drop=True)

    # use canonical as main SMILES
    merged_dedup["SMILES"] = merged_dedup["canonical_smiles"]
    merged_dedup = merged_dedup.drop(columns=["canonical_smiles"])

    return merged_dedup

# Build the base dataset by merging train and external data, removing test leakage
def build_base_dataset() -> pd.DataFrame:
    df_train = pd.read_csv(DATA_RAW / "train.csv")
    df_test  = pd.read_csv(DATA_RAW / "test.csv")

    ext = load_external_bradley()
    merged = align_and_merge_with_train(df_train, ext)
    merged_dedup = remove_test_leakage(merged, df_test)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    merged_dedup.to_csv(DATA_PROC / "merged_dedup.csv", index=False)
    merged_dedup.to_feather(DATA_PROC / "merged_dedup.feather")

    print(f"[final] rows: {len(merged_dedup)}")
    return merged_dedup

