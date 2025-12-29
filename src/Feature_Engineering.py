# 코어 기능: X_all 과 (optional) y를 만드는 내부 함수
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
from scipy.sparse import csr_matrix, hstack, save_npz
from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import save_npz

RDLogger.DisableLog('rdApp.*')  # silence RDKit logs


# ---------- safe mol ----------
def _safe_mol(s):
    """Safely convert a SMILES string to an RDKit Mol object."""
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


# ---------- RDKit descriptors (10) ----------
DESC = [
    "RotatableBonds",
    "TPSA",
    "LogP",
    "MolWt",
    "NumHDonors",
    "NumHAcceptors",
    "RingCount",
    "FractionCSP3",
    "BalabanJ",
    "Kappa1",
]


def smiles_to_desc(smiles: pd.Series) -> np.ndarray:
    """
    Series of SMILES -> (n_samples, n_descriptors) array of RDKit descriptors.

    Any failures / invalid molecules are left as NaN and handled later.
    """
    n = len(smiles)
    n_desc = len(DESC)
    rows = np.empty((n, n_desc), dtype=float)
    rows[:] = np.nan

    for i, s in enumerate(smiles):
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            rows[i, 0] = float(Lipinski.NumRotatableBonds(m))
            rows[i, 1] = float(Descriptors.TPSA(m))
            rows[i, 2] = float(Descriptors.MolLogP(m))
            rows[i, 3] = float(Descriptors.MolWt(m))
            rows[i, 4] = float(Lipinski.NumHDonors(m))
            rows[i, 5] = float(Lipinski.NumHAcceptors(m))
            rows[i, 6] = float(rdMolDescriptors.CalcNumRings(m))
            rows[i, 7] = float(rdMolDescriptors.CalcFractionCSP3(m))
            rows[i, 8] = float(Descriptors.BalabanJ(m))
            rows[i, 9] = float(Descriptors.Kappa1(m))
        except Exception:
            # leave failed positions as NaN
            pass

    return rows


# ---------- Morgan fingerprint (optional) ----------
def smiles_to_morgan_bits(
    smiles: pd.Series,
    n_bits: int = 512,
    radius: int = 2,
) -> csr_matrix:
    """
    Series of SMILES -> sparse matrix of binary Morgan fingerprints.
    """
    indptr = [0]
    indices = []
    data = []

    for s in smiles:
        m = _safe_mol(s)
        if m is None:
            indptr.append(len(indices))
            continue

        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
        onbits = list(bv.GetOnBits())
        indices.extend(onbits)
        data.extend([1] * len(onbits))
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), shape=(len(smiles), n_bits), dtype=np.uint8)

def _build_features_core(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    group_prefix: str = "Group",
    use_morgan: bool = True,
    n_bits: int = 512,
    radius: int = 2,
    drop_invalid: bool = True,
):
    assert smiles_col in df.columns, f"{smiles_col} not in DataFrame columns"

    group_cols = [c for c in df.columns if c.startswith(group_prefix)]
    if not group_cols:
        raise ValueError(f"No group columns found with prefix '{group_prefix}'")

    # 1) validate SMILES
    valid_mask = df[smiles_col].apply(lambda s: _safe_mol(s) is not None)
    if drop_invalid:
        df_out = df.loc[valid_mask].reset_index(drop=True)
    else:
        df_out = df.copy()

    work = df_out[[smiles_col] + group_cols].copy()

    # GroupN features
    X_group = work[group_cols].to_numpy(dtype=float)
    X_group = np.nan_to_num(X_group, nan=0.0, posinf=0.0, neginf=0.0)
    X_group_sparse = csr_matrix(X_group)

    # RDKit descriptors
    X_desc = smiles_to_desc(work[smiles_col])
    X_desc_ok = np.nan_to_num(X_desc, nan=0.0, posinf=0.0, neginf=0.0)
    X_desc_sparse = csr_matrix(X_desc_ok)

    blocks = [X_group_sparse, X_desc_sparse]

    if use_morgan:
        X_fp = smiles_to_morgan_bits(work[smiles_col], n_bits=n_bits, radius=radius)
        blocks.append(X_fp)

    X_all = hstack(blocks, format="csr")

    return X_all, df_out


PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../mol-desc-predictor
ART_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR.mkdir(parents=True, exist_ok=True)



def build_features_train(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "Tm",
    save: bool = True,
    tag: str = "default",
    **kwargs,
):
    X_all, df_out = _build_features_core(df, smiles_col=smiles_col, **kwargs)
    y = pd.to_numeric(df_out[target_col], errors="coerce").values

    if save:
        save_npz(ART_DIR / f"X_train_{tag}.npz", X_all)
        np.save(ART_DIR / f"y_train_{tag}.npy", y)
        df_out.to_parquet(ART_DIR / f"meta_train_{tag}.parquet", index=False)

    return X_all, y, df_out


def build_features_test(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    save: bool = True,
    tag: str = "default",
    **kwargs,
):
    X_all, df_out = _build_features_core(df, smiles_col=smiles_col, **kwargs)

    if save:
        save_npz(ART_DIR / f"X_test_{tag}.npz", X_all)
        df_out.to_parquet(ART_DIR / f"meta_test_{tag}.parquet", index=False)

    return X_all, df_out
