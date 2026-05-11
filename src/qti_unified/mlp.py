"""Prediction adapter around the original ``QTI_ML/QTI_MLP.py`` module.

The dataset preprocessing, MLP architecture, and reverse z-score operation are
provided by the vendored ``qti_unified._legacy_qti_mlp`` copy. This module only
keeps the unified workflow API around that original implementation.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from ._legacy_qti_mlp import QTI_Dataset, QTI_MLP, rev_zscore


INVAR_KEYS = ["MD", "FA", "uFA", "C_c", "C_MD"]
QtiMlp = QTI_MLP


def collect_benchmark_model_paths(model_root: str | Path) -> list[Path]:
    """Select canonical external benchmark checkpoints from the old naming scheme."""

    root = Path(model_root).expanduser().resolve()
    pattern = "QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_*_ep100_LR1e-03_bs512_outnorm_sched_92"
    raw = sorted(p for p in root.glob(pattern) if p.is_file() and not p.name.endswith("_zscore.csv"))
    if not raw:
        raise FileNotFoundError(f"No benchmark checkpoints found in {root} with pattern {pattern!r}.")

    by_test_patient: dict[int, list[Path]] = {}
    for path in raw:
        match = re.search(r"_ts_P(\d+)_vs_P(\d+)_", path.name)
        if match:
            by_test_patient.setdefault(int(match.group(2)), []).append(path)

    selected = []
    for test_id in sorted(by_test_patient):
        candidates = sorted(by_test_patient[test_id])
        preferred = [p for p in candidates if "_20231029_" in p.name]
        selected.append(sorted(preferred or candidates)[-1])
    if not selected:
        raise FileNotFoundError(f"No canonical fold checkpoints found in {root}.")
    return selected


def load_zscore_array(zscore_path: str | Path, n_invars: int) -> np.ndarray:
    """Load the checkpoint output z-score table used by ``rev_zscore``."""

    path = Path(zscore_path).expanduser().resolve()
    try:
        import pandas as pd  # type: ignore

        arr = pd.read_csv(path).values.astype(np.float32)
        if arr.shape == (2, int(n_invars)):
            return arr
    except Exception:
        pass
    text = path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    vals = np.asarray([float(x) for x in nums], dtype=np.float32)
    need = 2 * int(n_invars)
    if vals.size < need:
        raise ValueError(f"Could not parse {need} z-score values from {path}.")
    return vals[:need].reshape(2, int(n_invars))


def prepare_signal_inputs(signal_paths: list[str | Path]) -> tuple[torch.Tensor, list[int], QTI_Dataset]:
    """Run the old ``QTI_Dataset`` prediction-time preprocessing sequence."""

    ds_pred = QTI_Dataset(
        [str(Path(p).expanduser().resolve()) for p in signal_paths],
        scalar_invars_path=None,
        mask_path=None,
        slice_ind=None,
        invar_keys=INVAR_KEYS,
        zscore_output=True,
    )
    ds_pred.thresh_neg_vals()
    ds_pred.apply_masked_tensor()
    ds_pred.zscore_norm_input()
    output_shape = [ds_pred.X.size(i) for i in range(len(ds_pred.X.size()) - 1)]
    ds_pred.flatten_slice_dim()
    return ds_pred.X, output_shape, ds_pred


def ensemble_predict(
    signal_paths: list[str | Path],
    model_paths: list[str | Path],
    invar_keys: list[str] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Run the old benchmark MLP ensemble on generated SNR signal files."""

    keys = INVAR_KEYS if invar_keys is None else list(invar_keys)
    x, output_shape, _ = prepare_signal_inputs(signal_paths)
    x = x.to(device)
    n_scans = int(x.size(-1))
    preds = []

    for model_path in model_paths:
        path = Path(model_path).expanduser().resolve()
        model = QTI_MLP(n_scans, len(keys), False, False, True)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            pred = model(x)

        zscore = torch.tensor(load_zscore_array(str(path) + "_zscore.csv", len(keys)), dtype=torch.float32, device=device)
        pred = rev_zscore(pred, zscore)
        pred = torch.nan_to_num(pred, nan=0.0)
        pred = torch.permute(torch.reshape(pred, output_shape + [pred.size(-1)]), (0, 2, 3, 1, 4)).detach().cpu().numpy()
        preds.append(pred)

    return np.mean(preds, axis=0)
