"""Optional patient-target comparison helpers.

Patient data is never bundled with this repository. Callers must provide
explicit external paths to patient signal NIfTI files or a patient root folder.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


TARGET_VOXELS = {
    "Deep CC": (35, 47, 9),
    "Partial CSF/WM": (50, 54, 10),
    "Pure CSF": (42, 52, 8),
}


def load_patient_signal(signal_path: str | Path) -> np.ndarray:
    """Load an external patient diffusion signal image.

    Parameters
    ----------
    signal_path
        External NIfTI path. The image must be 4D with measurements in the
        final dimension.

    Returns
    -------
    numpy.ndarray
        Signal image with shape ``(x, y, z, n_measurements)``.
    """

    path = Path(signal_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Patient signal file not found: {path}")
    arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=float)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D patient signal image, got shape {arr.shape}: {path}")
    return arr


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize a signal vector by its first nonzero b0-like value.

    Parameters
    ----------
    signal
        Signal vector or array with measurements in the final dimension.

    Returns
    -------
    numpy.ndarray
        Normalized signal with the same shape as ``signal``.
    """

    arr = np.asarray(signal, dtype=float)
    denom = np.nanmean(arr[..., :1], axis=-1, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return arr / denom


def extract_target_signal(patient_signal: np.ndarray, target_name: str = "Brain mean", mask: np.ndarray | None = None) -> np.ndarray:
    """Extract one target signal from a patient image.

    Parameters
    ----------
    patient_signal
        4D patient signal image with measurements in the final dimension.
    target_name
        ``Brain mean`` or one of the named voxel targets in ``TARGET_VOXELS``.
    mask
        Optional boolean brain mask for ``Brain mean``.

    Returns
    -------
    numpy.ndarray
        Normalized target signal vector with shape ``(n_measurements,)``.
    """

    arr = np.asarray(patient_signal, dtype=float)
    if target_name == "Brain mean":
        vox = arr[mask.astype(bool)] if mask is not None else arr.reshape(-1, arr.shape[-1])
        return normalize_signal(np.nanmean(vox, axis=0))
    if target_name not in TARGET_VOXELS:
        raise KeyError(f"Unknown patient target {target_name!r}. Available: Brain mean, {', '.join(TARGET_VOXELS)}")
    x, y, z = TARGET_VOXELS[target_name]
    return normalize_signal(arr[x, y, z, :])


def rank_simulated_cases(patient_target: np.ndarray, simulated_signals: dict[str, np.ndarray]) -> pd.DataFrame:
    """Rank simulated cases by RMSE to a patient target signal.

    Parameters
    ----------
    patient_target
        Normalized patient target signal with shape ``(n_measurements,)``.
    simulated_signals
        Mapping from case label to simulated signal vector.

    Returns
    -------
    pandas.DataFrame
        Sorted table with columns ``case`` and ``rmse``.
    """

    import pandas as pd  # type: ignore

    target = np.asarray(patient_target, dtype=float).reshape(-1)
    rows = []
    for label, signal in simulated_signals.items():
        sig = normalize_signal(np.asarray(signal, dtype=float).reshape(-1))
        n = min(target.size, sig.size)
        rows.append({"case": label, "rmse": float(np.sqrt(np.nanmean((sig[:n] - target[:n]) ** 2)))})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
