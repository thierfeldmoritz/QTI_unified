"""Compatibility layer around the original SynQTI-IR QTI math utilities.

The numerical implementation lives in the vendored ``utils.dtd_math`` module
copied from ``C:/SynQTI-IR/utils/dtd_math.py``. This file keeps the
``qti_unified`` library API stable while delegating tensor math, Watson helpers,
QTI scalar formulas, signal models, and SNR noise to the original code.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.io as sio

from utils import dtd_math as legacy


DTI_signal = legacy.DTI_signal
QTI_signal = legacy.QTI_signal
axisym_eigvals_to_projection_metrics = legacy.axisym_eigvals_to_projection_metrics
c_c = legacy.c_c
c_m = legacy.c_m
c_md = legacy.c_md
c_mu = legacy.c_mu
compute_cumulant_tensors = legacy.compute_cumulant_tensors
convert_1x6_to_1x21 = legacy.convert_1x6_to_1x21
convert_3x3_to_1x6 = legacy.convert_3x3_to_1x6
fa = legacy.fa
from_6x6_to_21x1 = legacy.from_6x6_to_21x1
generate_dtd_gauss_random_orientations = legacy.generate_dtd_gauss_random_orientations
generate_fibers_WM_watson = legacy.generate_fibers_WM_watson
get_rot_matrix = legacy.get_rot_matrix
kappa_from_odi = legacy.kappa_from_odi
md = legacy.md
projection_metrics_to_eigvals = legacy.projection_metrics_to_eigvals
random_unit_vectors = legacy.random_unit_vectors
rician_complex = legacy.rician_complex
scaled_noise_from_snr = legacy.scaled_noise_from_snr
spherical_fibonacci_points = legacy.spherical_fibonacci_points
trace_shape_to_eigvals = legacy.trace_shape_to_eigvals
truncated_normal = legacy.truncated_normal
ufa = legacy.ufa
v_md = legacy.v_md
watson_sample_orientations = legacy.watson_sample_orientations


def _tensor_from_eigvals(lambda_1: float, lambda_2: float, lambda_3: float, direction: object) -> np.ndarray:
    u = np.asarray(direction, dtype=float)
    u /= np.linalg.norm(u) + 1e-15
    rot = legacy.get_rot_matrix(u)
    return rot @ np.diag([float(lambda_1), float(lambda_2), float(lambda_3)]) @ rot.T


def _finite_or_none(value: object) -> float | None:
    val = float(np.asarray(value, dtype=float).reshape(-1)[0])
    return val if math.isfinite(val) else None


def _finite_or_zero_if_numerical_zero(value: object, zero_reference: object) -> float | None:
    ref = float(np.asarray(zero_reference, dtype=float).reshape(-1)[0])
    if abs(ref) < 1e-12:
        return 0.0
    return _finite_or_none(value)


def params_to_dtens(params: list[dict[str, float]]) -> np.ndarray:
    """Convert DTD parameter dictionaries to 3x3 tensors using old rotation code."""

    return np.asarray(
        [
            _tensor_from_eigvals(
                p["lambda_1"],
                p["lambda_2"],
                p["lambda_3"],
                (p["u1"], p["u2"], p["u3"]),
            )
            for p in params
        ],
        dtype=float,
    )


def qti_params_from_dtd(params: list[dict[str, float]]) -> np.ndarray:
    """Build the old ``[S0, mean D, covariance C]`` QTI parameter vector."""

    dtens_voigt = legacy.convert_3x3_to_1x6(params_to_dtens(params))
    avg_d, cov = legacy.compute_cumulant_tensors(dtens_voigt)
    return np.concatenate([np.array([1.0]), avg_d.ravel(), cov.ravel()])[None, :]


def gt_scalars_from_params(params: list[dict[str, float]]) -> dict[str, float | None]:
    """Compute stored GT scalars through ``utils.dtd_math`` formulas."""

    qti_params = qti_params_from_dtd(params)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_m_value = legacy.c_m(qti_params)
        c_mu_value = legacy.c_mu(qti_params)
        md_si = _finite_or_none(legacy.md(qti_params))
        v_md_si = _finite_or_none(legacy.v_md(qti_params))
        c_c_value = None if abs(float(np.asarray(c_mu_value).item())) < 1e-12 else _finite_or_none(legacy.c_c(qti_params))
        md_um = None if md_si is None else md_si * 1e9
        return {
            "MD_SI_m2_per_s": md_si,
            "E_SI_m2_per_s": md_si,
            "MD": md_um,
            "MD_um2_per_ms": md_um,
            "E_um2_per_ms": md_um,
            "V_SI_m4_per_s2": v_md_si,
            "V_um4_per_ms2": None if v_md_si is None else v_md_si * 1e18,
            "FA": _finite_or_zero_if_numerical_zero(legacy.fa(qti_params), c_m_value),
            "uFA": _finite_or_zero_if_numerical_zero(legacy.ufa(qti_params), c_mu_value),
            "C_MD": _finite_or_none(legacy.c_md(qti_params)),
            "C_c": c_c_value,
        }


def dti_signal(dtens: np.ndarray, btens: np.ndarray) -> np.ndarray:
    """Compute the exact DTD mixture signal with ``utils.dtd_math.DTI_signal``."""

    d6 = legacy.convert_3x3_to_1x6(dtens) if np.asarray(dtens).shape[-2:] == (3, 3) else np.asarray(dtens, dtype=float)
    b6 = np.asarray(btens, dtype=float).reshape(-1, 6)
    return legacy.DTI_signal(d6[None, :, :], b6[:, None, :]).mean(axis=-1)


def qti_cumulant_signal(qti_params: np.ndarray, btens: np.ndarray) -> np.ndarray:
    """Compute the second-order cumulant signal with ``utils.dtd_math.QTI_signal``."""

    return np.asarray(legacy.QTI_signal(np.asarray(qti_params, dtype=float).reshape(1, 28), np.asarray(btens, dtype=float).reshape(-1, 6)), dtype=float)


def xps_to_bt(xps: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
    """Return b-tensor Voigt vectors from an XPS dictionary or array."""

    if isinstance(xps, np.ndarray):
        return np.asarray(xps, dtype=float).reshape(-1, 6)
    if "bt" in xps:
        return np.asarray(xps["bt"], dtype=float).reshape(-1, 6)
    b = np.asarray(xps["b"], dtype=float).reshape(-1)
    b_delta = np.asarray(xps["b_delta"], dtype=float).reshape(-1)
    u = np.asarray(xps["u"], dtype=float)
    if u.shape[0] == 3 and u.shape[-1] != 3:
        u = u.T
    mats = []
    for bi, bd, ui in zip(b, b_delta, u.reshape(-1, 3)):
        lam = legacy.projection_metrics_to_eigvals(float(bi), float(bd))
        mats.append(_tensor_from_eigvals(lam[0], lam[1], lam[2], ui))
    return legacy.convert_3x3_to_1x6(np.asarray(mats))


def read_xps_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Read XPS fields from a ``.mat`` file using the old notebook schema."""

    data = sio.loadmat(str(path))
    xps = data["xps"][0, 0]
    field_keys = ["n", "b", "b_delta", "b_eta", "bt", "u", "s_ind"]
    processed: dict[str, np.ndarray] = {}
    for key in field_keys:
        if hasattr(xps, "dtype") and xps.dtype.names and key in xps.dtype.names:
            value = xps[key]
        else:
            value = xps[field_keys.index(key)]
        arr = np.asarray(value)
        if arr.shape == (1, 1):
            processed[key] = np.asarray(arr.item())
        elif arr.ndim == 2 and arr.shape[1] == 1:
            processed[key] = arr.flatten()
        else:
            processed[key] = arr
    processed["bt"] = xps_to_bt(processed)
    return processed


def default_btens(n_measurements: int = 54) -> np.ndarray:
    """Build the small smoke-test protocol through old tensor utilities."""

    rng = np.random.default_rng(12345)
    dirs = legacy.random_unit_vectors(int(n_measurements), rng=rng)
    dirs[0] = np.asarray([1.0, 0.0, 0.0])
    bvals = np.linspace(0.0, 2.0e9, int(n_measurements))
    tensors = []
    for bval, direction in zip(bvals, dirs):
        tensors.append(_tensor_from_eigvals(float(bval), 0.0, 0.0, direction))
    return legacy.convert_3x3_to_1x6(np.asarray(tensors))


def rician_noise(clean: np.ndarray, snr: float, s0: float, rng: np.random.Generator) -> np.ndarray:
    """Add the old ``scaled_noise_from_snr`` Rician-style noise."""

    return legacy.scaled_noise_from_snr(np.asarray(clean, dtype=float), snr, s0, rng)
