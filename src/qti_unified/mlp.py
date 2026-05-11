"""Minimal MLP inference adapter for external benchmark checkpoints.

The module intentionally contains only prediction-time code. Checkpoints and
z-score CSV files are external inputs, normally supplied with ``--model-root``.
"""

from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn


INVAR_KEYS = ["MD", "FA", "uFA", "C_c", "C_MD"]


class QtiMlp(nn.Module):
    """Prediction-time MLP architecture matching the benchmark checkpoints.

    Parameters
    ----------
    n_scans
        Number of diffusion measurements in each input signal.
    n_invars
        Number of scalar outputs. The patho workflow uses five outputs.
    layer_norm
        Whether to prepend a non-affine layer normalization layer.
    final_sigmoid
        Whether to append a final sigmoid activation.
    bias
        Whether linear layers include bias terms.
    """

    def __init__(self, n_scans: int, n_invars: int, layer_norm: bool = False, final_sigmoid: bool = False, bias: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(int(n_scans), elementwise_affine=False))
        layers.extend(
            [
                nn.Linear(int(n_scans), 128, bias=bias),
                nn.ReLU(),
                nn.Linear(128, 256, bias=bias),
                nn.ReLU(),
                nn.Linear(256, 128, bias=bias),
                nn.ReLU(),
                nn.Linear(128, 32, bias=bias),
                nn.ReLU(),
                nn.Linear(32, int(n_invars), bias=bias),
            ]
        )
        if final_sigmoid:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on flattened signal vectors.

        Parameters
        ----------
        x
            Tensor with trailing dimension ``n_scans``.

        Returns
        -------
        torch.Tensor
            Predicted normalized scalar outputs.
        """

        return self.layers(x)


def collect_benchmark_model_paths(model_root: str | Path) -> list[Path]:
    """Select canonical external benchmark checkpoints.

    Parameters
    ----------
    model_root
        Directory containing checkpoint files and matching ``*_zscore.csv``.

    Returns
    -------
    list[pathlib.Path]
        Selected checkpoint files, one per benchmark test patient.
    """

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
    """Load model-output z-score statistics.

    Parameters
    ----------
    zscore_path
        CSV or text file next to a checkpoint.
    n_invars
        Number of scalar outputs expected by the model.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(2, n_invars)`` where row 0 is mean and row 1 is std.
    """

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


def reverse_zscore(pred: torch.Tensor, zscore: torch.Tensor) -> torch.Tensor:
    """Reverse global output z-score normalization.

    Parameters
    ----------
    pred
        Model output tensor with shape ``(n_voxels, n_invars)``.
    zscore
        Tensor with shape ``(2, n_invars)``.

    Returns
    -------
    torch.Tensor
        Physical-scale predictions.
    """

    return pred * zscore[1, :].unsqueeze(0) + zscore[0, :].unsqueeze(0)


def read_nifti_signals(signal_paths: list[str | Path]) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    """Read 4D signal NIfTI files into a tensor for prediction.

    Parameters
    ----------
    signal_paths
        Paths to NIfTI files with last dimension equal to scan count.

    Returns
    -------
    tuple[torch.Tensor, list[tuple[int, ...]]]
        Tensor with shape ``(n_files, z, x, y, n_scans)`` and original image
        shapes for reporting.
    """

    arrays = []
    shapes = []
    for path in signal_paths:
        img = nib.load(str(path))
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"Expected a 4D NIfTI signal, got shape {arr.shape}: {path}")
        shapes.append(tuple(arr.shape))
        arrays.append(arr)
    data = np.asarray(arrays, dtype=np.float32)
    return torch.permute(torch.from_numpy(data), (0, 3, 1, 2, 4)), shapes


def prepare_signal_inputs(signal_paths: list[str | Path]) -> tuple[torch.Tensor, list[int], list[tuple[int, ...]]]:
    """Load, clamp, z-score normalize, and flatten signal inputs.

    Parameters
    ----------
    signal_paths
        NIfTI signal paths.

    Returns
    -------
    tuple[torch.Tensor, list[int], list[tuple[int, ...]]]
        Flattened input tensor, output shape prefix used for reshaping, and
        original NIfTI shapes.
    """

    x, shapes = read_nifti_signals(signal_paths)
    x[x < 0.0] = 0.0
    mean = torch.mean(x, dim=tuple(range(1, len(x.size()))), keepdim=True, dtype=torch.float32)
    std = torch.std(x, dim=tuple(range(1, len(x.size()))), keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    x = (x - mean) / std
    output_shape = [x.size(i) for i in range(len(x.size()) - 1)]
    flat = torch.flatten(x.contiguous(), end_dim=-2)
    return flat, output_shape, shapes


def ensemble_predict(
    signal_paths: list[str | Path],
    model_paths: list[str | Path],
    invar_keys: list[str] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Run ensemble-averaged MLP prediction on signal files.

    Parameters
    ----------
    signal_paths
        NIfTI signal paths.
    model_paths
        External checkpoint paths.
    invar_keys
        Output invariant names. Defaults to ``MD, FA, uFA, C_c, C_MD``.
    device
        Torch device string.

    Returns
    -------
    numpy.ndarray
        Prediction array with shape ``(n_files, z, x, y, n_invars)``.
    """

    keys = INVAR_KEYS if invar_keys is None else list(invar_keys)
    x, output_shape, _ = prepare_signal_inputs(signal_paths)
    x = x.to(device)
    n_scans = int(x.size(-1))
    preds = []
    for model_path in model_paths:
        path = Path(model_path).expanduser().resolve()
        model = QtiMlp(n_scans, len(keys), layer_norm=False, final_sigmoid=False, bias=True)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x)
        zscore = torch.tensor(load_zscore_array(str(path) + "_zscore.csv", len(keys)), dtype=torch.float32, device=device)
        pred = reverse_zscore(pred, zscore)
        pred = torch.nan_to_num(pred, nan=0.0)
        pred = torch.reshape(pred, output_shape + [pred.size(-1)]).detach().cpu().numpy()
        preds.append(pred)
    return np.mean(preds, axis=0)
