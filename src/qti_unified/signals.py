"""Signal synthesis, GT JSON writing, and notebook-compatible output layout.

The functions here write patho outputs under a caller-provided data root. They
do not assume patient data, model checkpoints, or covariance-fit folders live
inside the repository.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np

from .config import GeneratedPathoCase, PathoCase
from .qti_math import default_btens, dti_signal, gt_scalars_from_params, params_to_dtens, qti_cumulant_signal, qti_params_from_dtd, read_xps_mat, rician_noise, xps_to_bt


def load_btens(xps_path: str | Path | None = None) -> tuple[np.ndarray, str | None]:
    """Load external b-tensors or return the built-in smoke-test protocol.

    Parameters
    ----------
    xps_path
        Optional external XPS ``.mat`` path. Real analyses should pass the
        acquisition protocol used by the MLP.

    Returns
    -------
    tuple[numpy.ndarray, str | None]
        B-tensor array with shape ``(n_measurements, 6)`` and the resolved XPS
        path string, or ``None`` when the built-in protocol is used.
    """

    if xps_path is None:
        return default_btens(), None
    path = Path(xps_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"XPS file not found: {path}")
    return xps_to_bt(read_xps_mat(path)), str(path)


def save_nifti_signal(path_base: Path, signal: np.ndarray) -> str:
    """Save a signal vector as a 4D NIfTI file.

    Parameters
    ----------
    path_base
        Output path without suffix.
    signal
        Signal vector with shape ``(n_measurements,)``.

    Returns
    -------
    str
        Written ``.nii`` path.
    """

    arr4d = np.asarray(signal, dtype=np.float32).reshape(1, 1, 1, -1)
    path = path_base.with_suffix(".nii")
    try:
        import nibabel as nib  # type: ignore

        nib.save(nib.Nifti1Image(arr4d, np.eye(4)), str(path))
    except Exception:
        _write_minimal_nifti_float32(path, arr4d)
    return str(path)


def _write_minimal_nifti_float32(path: Path, data: np.ndarray) -> None:
    data = np.asarray(data, dtype="<f4")
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 4, *data.shape, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<4s", header, 344, b"n+1\0")
    with path.open("wb") as f:
        f.write(header)
        f.write(b"\0\0\0\0")
        f.write(data.tobytes(order="C"))


def patho_output_paths(data_root: str | Path, scenario: str, case: str, snr: float | None = 30) -> dict[str, Path]:
    """Build all notebook-compatible output paths for one case.

    Parameters
    ----------
    data_root
        Generated-data root.
    scenario
        Scenario folder name.
    case
        Case folder/file stem.
    snr
        SNR value used for noisy output. ``None`` still returns an ``SNR`` path
        for consistency but callers may ignore it.

    Returns
    -------
    dict[str, pathlib.Path]
        DTD, result, GT, exact, cumexp, and noisy-output paths.
    """

    root = Path(data_root).expanduser().resolve()
    phantom_id = f"{scenario}__{case}"
    dtd_dir = root / "DTDs_cov_suite_2_patho" / scenario
    result_dir = root / "Results_2_MLP_patho" / scenario / case
    snr_label = "SNR" + str(int(snr or 0))
    snr_dir = root / "Results_SNR_fit_2_MLP_patho" / scenario / case / snr_label
    return {
        "dtd_dir": dtd_dir,
        "result_dir": result_dir,
        "snr_dir": snr_dir,
        "dtd_json": dtd_dir / f"{case}.json",
        "gt_json": result_dir / f"{phantom_id}__GT_params.json",
        "metadata_json": result_dir / "metadata.json",
        "exact_signal": result_dir / f"{phantom_id}__exact.nii",
        "cumexp_signal": result_dir / f"{phantom_id}__cumexp.nii",
    }


def write_case_outputs(
    case: PathoCase,
    data_root: str | Path,
    xps_path: str | Path | None = None,
    snr: float | None = 30,
    n_realizations: int = 100,
    s0: float = 1.0,
) -> GeneratedPathoCase:
    """Write DTD, GT, exact signal, cumexp signal, and noisy realizations.

    Parameters
    ----------
    case
        Patho DTD case to write.
    data_root
        Generated-data root. The path is created if needed.
    xps_path
        Optional external XPS ``.mat`` path. If omitted, a deterministic
        smoke-test protocol is used.
    snr
        Bulk SNR. ``None`` disables noisy signal writing.
    n_realizations
        Number of noisy realizations to write when ``snr`` is not ``None``.
    s0
        Baseline signal amplitude.

    Returns
    -------
    GeneratedPathoCase
        Object containing GT, signals, and written paths.
    """

    btens, resolved_xps = load_btens(xps_path)
    paths = patho_output_paths(data_root, case.scenario, case.case, snr=snr)
    paths["dtd_dir"].mkdir(parents=True, exist_ok=True)
    paths["result_dir"].mkdir(parents=True, exist_ok=True)

    gt = gt_scalars_from_params(case.params)
    dtens = params_to_dtens(case.params)
    exact = float(s0) * dti_signal(dtens, btens)
    cumexp = float(s0) * qti_cumulant_signal(qti_params_from_dtd(case.params), btens)

    paths["dtd_json"].write_text(json.dumps(case.params, indent=2), encoding="utf-8")
    exact_path = save_nifti_signal(paths["result_dir"] / f"{case.scenario}__{case.case}__exact", exact)
    cumexp_path = save_nifti_signal(paths["result_dir"] / f"{case.scenario}__{case.case}__cumexp", cumexp)

    payload = {
        "scenario": case.scenario,
        "case": case.case,
        "dtd_file": str(paths["dtd_json"]),
        "xps_mat": resolved_xps,
        "gt": gt,
        "metadata": {**case.metadata, "s0": float(s0), "n_measurements": int(btens.shape[0]), "btens_units": "s/m^2"},
    }
    paths["gt_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["metadata_json"].write_text(json.dumps(payload["metadata"], indent=2), encoding="utf-8")

    if resolved_xps is not None:
        try:
            shutil.copy2(resolved_xps, paths["result_dir"] / f"{case.scenario}__{case.case}_xps.mat")
        except OSError:
            pass

    noisy = None
    written_paths = {
        "dtd_json": str(paths["dtd_json"]),
        "gt_json": str(paths["gt_json"]),
        "metadata_json": str(paths["metadata_json"]),
        "exact_signal": exact_path,
        "cumexp_signal": cumexp_path,
    }
    if snr is not None and int(n_realizations) > 0:
        paths["snr_dir"].mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(case.metadata.get("seed", 0)) + 1000003)
        noisy = np.stack([rician_noise(exact, float(snr), float(s0), rng) for _ in range(int(n_realizations))], axis=0)
        signal_paths = []
        for i, signal in enumerate(noisy):
            signal_paths.append(save_nifti_signal(paths["snr_dir"] / f"signal_real_{i:03d}", signal))
        noise_meta = {
            "scenario": case.scenario,
            "case": case.case,
            "SNR": float(snr),
            "n_realizations": int(n_realizations),
            "seed": int(case.metadata.get("seed", 0)),
            "dtd_file": str(paths["dtd_json"]),
            "btens_file": resolved_xps,
            "verified_shape": [1, 1, 1, int(btens.shape[0])],
        }
        (paths["snr_dir"] / "metadata.json").write_text(json.dumps(noise_meta, indent=2), encoding="utf-8")
        written_paths["noisy_dir"] = str(paths["snr_dir"])
        written_paths["noisy_signals"] = json.dumps(signal_paths)

    return GeneratedPathoCase(case=case, gt=gt, exact_signal=exact.astype(np.float32), cumexp_signal=cumexp.astype(np.float32), noisy_signals=None if noisy is None else noisy.astype(np.float32), paths=written_paths)
