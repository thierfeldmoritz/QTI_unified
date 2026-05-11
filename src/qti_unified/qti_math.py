"""Minimal QTI tensor math used by the standalone patho workflow.

This module vendors the small subset of the original SynQTI-IR ``utils.dtd_math``
behavior that is required for patho DTD generation, exact signal synthesis, and
the five stored ground-truth metrics. All diffusion eigenvalues are SI units
(``m^2/s``), all b-tensors are SI units (``s/m^2``), and Voigt vectors use the
Westin/QTI convention ``[xx, yy, zz, sqrt(2)xy, sqrt(2)xz, sqrt(2)yz]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import scipy.io as sio


SQRT2 = float(np.sqrt(2.0))
E_ISO_6 = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0], dtype=float)
E_BULK_6X6 = np.outer(E_ISO_6, E_ISO_6)
E_ISO_6X6 = np.eye(6, dtype=float) / 3.0
E_SHEAR_6X6 = E_ISO_6X6 - E_BULK_6X6


def voigt_inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute an inner product along the last Voigt dimension.

    Parameters
    ----------
    a, b
        Broadcast-compatible arrays with last dimension length 6 or 21.

    Returns
    -------
    numpy.ndarray
        Elementwise inner product with the final dimension removed.
    """

    return np.sum(np.asarray(a, dtype=float) * np.asarray(b, dtype=float), axis=-1)


def convert_3x3_to_1x6(tensors: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 tensors to 6-element Voigt vectors.

    Parameters
    ----------
    tensors
        Array with shape ``(..., 3, 3)``.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(..., 6)`` in QTI Voigt order.
    """

    d = np.asarray(tensors, dtype=float)
    if d.shape[-2:] != (3, 3):
        raise ValueError("Expected tensor array with trailing shape (3, 3).")
    return np.stack(
        [d[..., 0, 0], d[..., 1, 1], d[..., 2, 2], SQRT2 * d[..., 0, 1], SQRT2 * d[..., 0, 2], SQRT2 * d[..., 1, 2]],
        axis=-1,
    )


def tensor6x6_to_1x21(tensors: np.ndarray) -> np.ndarray:
    """Convert 6x6 covariance tensors to 21-element QTI Voigt vectors.

    Parameters
    ----------
    tensors
        Array with shape ``(..., 6, 6)``.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(..., 21)`` using the same ordering as the source
        notebook's ``tm_6x6_to_1x21`` helper.
    """

    t = np.asarray(tensors, dtype=float)
    if t.shape[-2:] != (6, 6):
        raise ValueError("Expected tensor array with trailing shape (6, 6).")
    t1 = t[..., [0, 1, 2], [0, 1, 2]]
    t2 = SQRT2 * t[..., [0, 0, 1], [1, 2, 2]]
    t3 = SQRT2 * t[..., [0, 1, 2], [5, 4, 3]]
    t4 = SQRT2 * t[..., [0, 0, 1], [3, 4, 3]]
    t5 = SQRT2 * t[..., [1, 2, 2], [5, 4, 5]]
    t6 = t[..., [3, 4, 5], [3, 4, 5]]
    t7 = SQRT2 * t[..., [3, 3, 4], [4, 5, 5]]
    return np.concatenate((t1, t2, t3, t4, t5, t6, t7), axis=-1)


def self_outer_1x6_to_1x21(vectors: np.ndarray) -> np.ndarray:
    """Convert self outer products of 6-vectors to 21-vector form.

    Parameters
    ----------
    vectors
        Array with shape ``(..., 6)``.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(..., 21)``.
    """

    v = np.asarray(vectors, dtype=float)
    if v.shape[-1] != 6:
        raise ValueError("Expected Voigt vectors with trailing length 6.")
    outer = v[..., :, None] * v[..., None, :]
    return tensor6x6_to_1x21(outer)


E_BULK_21 = tensor6x6_to_1x21(E_BULK_6X6)
E_ISO_21 = tensor6x6_to_1x21(E_ISO_6X6)
E_SHEAR_21 = tensor6x6_to_1x21(E_SHEAR_6X6)


def normalize_vector(vector: Iterable[float]) -> np.ndarray:
    """Return a unit vector with a small zero-norm guard.

    Parameters
    ----------
    vector
        Three-vector orientation.

    Returns
    -------
    numpy.ndarray
        Unit vector with shape ``(3,)``.
    """

    arr = np.asarray(vector, dtype=float)
    return arr / (np.linalg.norm(arr) + 1e-15)


def get_rot_y(angle: float) -> np.ndarray:
    """Return a rotation matrix around the y-axis.

    Parameters
    ----------
    angle
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(3, 3)``.
    """

    return np.asarray([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])


def get_rot_z(angle: float) -> np.ndarray:
    """Return a rotation matrix around the z-axis.

    Parameters
    ----------
    angle
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(3, 3)``.
    """

    return np.asarray([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])


def get_rot_matrix(direction: Iterable[float]) -> np.ndarray:
    """Return the notebook-compatible rotation matrix for a tensor direction.

    Parameters
    ----------
    direction
        Principal eigenvector of a diffusion tensor.

    Returns
    -------
    numpy.ndarray
        Rotation matrix with shape ``(3, 3)``.
    """

    d = normalize_vector(direction)
    if d[0] == 1.0 and d[1] == 0.0 and d[2] == 0.0:
        return np.eye(3)
    if d[1] == 0 and d[0] != 0 and d[2] != 0:
        return get_rot_y(-np.arctan2(d[2], d[0]))
    if d[2] == 0 and d[0] != 0 and d[1] != 0:
        return get_rot_z(np.arctan2(d[1], d[0]))
    phi = np.arctan2(abs(d[1]), d[0])
    if d[1] > 0.0:
        phi *= -1.0
    rot_z = get_rot_z(phi)
    d_temp = rot_z @ d
    eta = np.arctan2(d_temp[2], d_temp[0])
    rot_y = get_rot_y(eta)
    return rot_z.T @ rot_y.T


def make_tensor(lambda_1: float, lambda_2: float, lambda_3: float, direction: Iterable[float]) -> np.ndarray:
    """Build a 3x3 diffusion tensor from eigenvalues and orientation.

    Parameters
    ----------
    lambda_1, lambda_2, lambda_3
        Eigenvalues in ``m^2/s``.
    direction
        Principal eigenvector.

    Returns
    -------
    numpy.ndarray
        Diffusion tensor with shape ``(3, 3)``.
    """

    rot = get_rot_matrix(direction)
    return rot @ np.diag([float(lambda_1), float(lambda_2), float(lambda_3)]) @ rot.T


def params_to_dtens(params: list[dict[str, float]]) -> np.ndarray:
    """Convert DTD parameter dictionaries to 3x3 tensors.

    Parameters
    ----------
    params
        DTD entries containing ``lambda_1``...``lambda_3`` and ``u1``...``u3``.

    Returns
    -------
    numpy.ndarray
        Tensor array with shape ``(n_tensors, 3, 3)``.
    """

    return np.asarray([make_tensor(p["lambda_1"], p["lambda_2"], p["lambda_3"], (p["u1"], p["u2"], p["u3"])) for p in params])


def compute_cumulant_tensors(dtens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean diffusion and covariance tensors from a DTD.

    Parameters
    ----------
    dtens
        Tensor array with shape ``(n, 3, 3)`` or Voigt array with shape
        ``(n, 6)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Mean diffusion tensor ``(1, 6)`` and covariance tensor ``(1, 21)``.
    """

    arr = np.asarray(dtens, dtype=float)
    if arr.shape[-2:] == (3, 3):
        arr = convert_3x3_to_1x6(arr)
    if arr.ndim != 2 or arr.shape[-1] != 6:
        raise ValueError("Expected DTD tensors with shape (n, 3, 3) or (n, 6).")
    avg = np.mean(arr, axis=0, keepdims=True)
    mean_outer = np.mean(arr[:, :, None] * arr[:, None, :], axis=0, keepdims=True)
    cov = tensor6x6_to_1x21(mean_outer) - self_outer_1x6_to_1x21(avg)
    return avg, cov


def qti_params_from_dtd(params: list[dict[str, float]]) -> np.ndarray:
    """Build the ``[S0, mean D, covariance C]`` parameter vector.

    Parameters
    ----------
    params
        DTD parameter dictionaries.

    Returns
    -------
    numpy.ndarray
        QTI parameter array with shape ``(1, 28)``. ``S0`` is fixed to one.
    """

    avg_d, cov = compute_cumulant_tensors(params_to_dtens(params))
    return np.concatenate([np.asarray([[1.0]]), avg_d, cov], axis=1)


def md(qti_params: np.ndarray) -> np.ndarray:
    """Mean diffusivity in ``m^2/s`` from QTI params.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        Mean diffusivity values.
    """

    return voigt_inner_product(np.asarray(qti_params)[..., 1:7], E_ISO_6)


def v_md(qti_params: np.ndarray) -> np.ndarray:
    """Microscopic mean-diffusivity variance in ``m^4/s^2``.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        Variance values.
    """

    return voigt_inner_product(np.asarray(qti_params)[..., 7:28], E_BULK_21)


def mean_d_sq(qti_params: np.ndarray) -> np.ndarray:
    """Return ``C + <D> outer <D>`` in 21-vector form.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        Mean tensor-square vector with trailing length 21.
    """

    params = np.asarray(qti_params, dtype=float)
    return params[..., 7:28] + self_outer_1x6_to_1x21(params[..., 1:7])


def c_md(qti_params: np.ndarray) -> np.ndarray:
    """Normalized variance of microscopic mean diffusivities.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        ``C_MD`` values.
    """

    denom = voigt_inner_product(mean_d_sq(qti_params), E_BULK_21)
    return v_md(qti_params) / denom


def c_mu(qti_params: np.ndarray) -> np.ndarray:
    """Normalized microscopic anisotropy ``C_mu``.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        ``C_mu`` values.
    """

    mds = mean_d_sq(qti_params)
    return 1.5 * voigt_inner_product(mds, E_SHEAR_21) / voigt_inner_product(mds, E_ISO_21)


def c_m(qti_params: np.ndarray) -> np.ndarray:
    """Normalized macroscopic anisotropy ``C_M``.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        ``C_M`` values.
    """

    dsq = self_outer_1x6_to_1x21(np.asarray(qti_params, dtype=float)[..., 1:7])
    return 1.5 * voigt_inner_product(dsq, E_SHEAR_21) / voigt_inner_product(dsq, E_ISO_21)


def ufa(qti_params: np.ndarray) -> np.ndarray:
    """Microscopic fractional anisotropy ``uFA``.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        ``uFA`` values.
    """

    return np.sqrt(c_mu(qti_params))


def fa(qti_params: np.ndarray) -> np.ndarray:
    """Macroscopic fractional anisotropy ``FA``.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        ``FA`` values.
    """

    return np.sqrt(c_m(qti_params))


def c_c(qti_params: np.ndarray) -> np.ndarray:
    """Microscopic orientation coherence ``C_c``.

    Parameters
    ----------
    qti_params
        Array with trailing length 28.

    Returns
    -------
    numpy.ndarray
        ``C_c`` values.
    """

    return c_m(qti_params) / c_mu(qti_params)


def _finite_or_none(value: object) -> float | None:
    arr = np.asarray(value, dtype=float).reshape(-1)
    val = float(arr[0])
    return val if np.isfinite(val) else None


def _zero_when_reference_zero(value: object, reference: object) -> float | None:
    ref = float(np.asarray(reference, dtype=float).reshape(-1)[0])
    if abs(ref) < 1e-12:
        return 0.0
    return _finite_or_none(value)


def gt_scalars_from_params(params: list[dict[str, float]]) -> dict[str, float | None]:
    """Compute and format the stored five GT metrics for one DTD.

    Parameters
    ----------
    params
        DTD parameter dictionaries.

    Returns
    -------
    dict[str, float | None]
        JSON-ready GT dictionary containing SI and display-unit MD plus
        ``FA``, ``uFA``, ``C_MD``, and ``C_c``.
    """

    q = qti_params_from_dtd(params)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_val = c_m(q)
        cmu_val = c_mu(q)
        md_si = _finite_or_none(md(q))
        md_display = None if md_si is None else md_si * 1e9
        vmd_si = _finite_or_none(v_md(q))
        c_c_val = None if abs(float(np.asarray(cmu_val).reshape(-1)[0])) < 1e-12 else _finite_or_none(c_c(q))
        return {
            "MD_SI_m2_per_s": md_si,
            "E_SI_m2_per_s": md_si,
            "MD": md_display,
            "MD_um2_per_ms": md_display,
            "E_um2_per_ms": md_display,
            "V_SI_m4_per_s2": vmd_si,
            "V_um4_per_ms2": None if vmd_si is None else vmd_si * 1e18,
            "FA": _zero_when_reference_zero(fa(q), cm_val),
            "uFA": _zero_when_reference_zero(ufa(q), cmu_val),
            "C_MD": _finite_or_none(c_md(q)),
            "C_c": c_c_val,
        }


def dti_signal(dtens: np.ndarray, btens: np.ndarray) -> np.ndarray:
    """Compute the exact DTD mixture signal for all b-tensors.

    Parameters
    ----------
    dtens
        Tensor array with shape ``(n_tensors, 3, 3)`` or ``(n_tensors, 6)``.
    btens
        B-tensor Voigt array with shape ``(n_measurements, 6)`` in ``s/m^2``.

    Returns
    -------
    numpy.ndarray
        Exact signal vector with shape ``(n_measurements,)`` and ``S0=1``.
    """

    d6 = convert_3x3_to_1x6(dtens) if np.asarray(dtens).shape[-2:] == (3, 3) else np.asarray(dtens, dtype=float)
    b6 = np.asarray(btens, dtype=float).reshape(-1, 6)
    return np.exp(-voigt_inner_product(b6[:, None, :], d6[None, :, :])).mean(axis=1)


def qti_cumulant_signal(qti_params: np.ndarray, btens: np.ndarray) -> np.ndarray:
    """Compute the second-order cumulant-expansion signal.

    Parameters
    ----------
    qti_params
        QTI params with trailing length 28.
    btens
        B-tensor Voigt array with shape ``(n_measurements, 6)`` in ``s/m^2``.

    Returns
    -------
    numpy.ndarray
        Cumulant-expansion signal vector with shape ``(n_measurements,)``.
    """

    q = np.asarray(qti_params, dtype=float).reshape(-1, 28)[0]
    b6 = np.asarray(btens, dtype=float).reshape(-1, 6)
    b21 = self_outer_1x6_to_1x21(b6)
    log_signal = -voigt_inner_product(b6, q[1:7]) + 0.5 * voigt_inner_product(b21, q[7:28])
    return np.exp(log_signal)


def projection_metrics_to_eigvals(d_iso: float, d_delta: float) -> tuple[float, float, float]:
    """Convert projection metrics to axisymmetric eigenvalues.

    Parameters
    ----------
    d_iso
        Isotropic diffusivity in ``m^2/s``.
    d_delta
        Shape parameter.

    Returns
    -------
    tuple[float, float, float]
        Eigenvalues in descending axisymmetric order.
    """

    return (float(d_iso) * (1.0 + 2.0 * float(d_delta)), float(d_iso) * (1.0 - float(d_delta)), float(d_iso) * (1.0 - float(d_delta)))


def trace_shape_to_eigvals(d_trace: float, d_shape: float) -> np.ndarray:
    """Convert trace and tensor shape to eigenvalues.

    Parameters
    ----------
    d_trace
        Tensor trace in ``m^2/s``.
    d_shape
        Axisymmetric shape parameter.

    Returns
    -------
    numpy.ndarray
        Eigenvalue array with shape ``(3,)``.
    """

    d_iso = float(d_trace) / 3.0
    return np.asarray([d_iso * (1.0 + 2.0 * d_shape), d_iso * (1.0 - d_shape), d_iso * (1.0 - d_shape)], dtype=float)


def axisym_eigvals_to_projection_metrics(d_par: float, d_perp: float) -> tuple[float, float]:
    """Convert axisymmetric eigenvalues to ``d_iso`` and ``d_delta``.

    Parameters
    ----------
    d_par, d_perp
        Parallel and perpendicular diffusivities in ``m^2/s``.

    Returns
    -------
    tuple[float, float]
        Isotropic diffusivity and shape parameter.
    """

    d_iso = (float(d_par) + 2.0 * float(d_perp)) / 3.0
    d_delta = (float(d_par) - float(d_perp)) / (float(d_par) + 2.0 * float(d_perp))
    return d_iso, d_delta


def xps_to_bt(xps: dict[str, np.ndarray]) -> np.ndarray:
    """Convert an XPS dictionary to b-tensor Voigt vectors.

    Parameters
    ----------
    xps
        Dictionary containing either ``bt``/``B`` tensors or ``b``,
        ``b_delta``, and ``u`` fields.

    Returns
    -------
    numpy.ndarray
        B-tensors with shape ``(n_measurements, 6)``.
    """

    if "bt" in xps:
        bt = np.asarray(xps["bt"], dtype=float)
        if bt.ndim >= 2 and bt.shape[-1] == 6:
            return bt.reshape(-1, 6)
        if bt.shape[-2:] == (3, 3):
            return convert_3x3_to_1x6(bt)
    if "B" in xps:
        b = np.asarray(xps["B"], dtype=float)
        if b.ndim >= 2 and b.shape[-1] == 6:
            return b.reshape(-1, 6)
        if b.shape[-2:] == (3, 3):
            return convert_3x3_to_1x6(b)
    if not {"b", "b_delta", "u"}.issubset(xps):
        raise ValueError("XPS requires bt/B or b, b_delta, and u fields.")
    bvals = np.asarray(xps["b"], dtype=float).reshape(-1)
    b_delta = np.asarray(xps["b_delta"], dtype=float).reshape(-1)
    u = np.asarray(xps["u"], dtype=float)
    if u.shape[0] == 3 and u.shape[-1] != 3:
        u = u.T
    u = u.reshape(-1, 3)
    mats = []
    for bi, bd, ui in zip(bvals, b_delta, u):
        lam = projection_metrics_to_eigvals(float(bi), float(bd))
        mats.append(make_tensor(lam[0], lam[1], lam[2], ui))
    return convert_3x3_to_1x6(np.asarray(mats))


def _unwrap_mat_field(obj: object) -> np.ndarray:
    arr = np.asarray(obj)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return np.squeeze(arr)


def read_xps_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Read an XPS ``.mat`` file into a small Python dictionary.

    Parameters
    ----------
    path
        External XPS ``.mat`` file.

    Returns
    -------
    dict[str, numpy.ndarray]
        Extracted XPS fields. The dictionary always includes ``bt``.
    """

    data = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
    out: dict[str, np.ndarray] = {}
    xps = data.get("xps")
    if xps is not None:
        cur = xps
        while isinstance(cur, np.ndarray) and cur.dtype == object and cur.size == 1:
            cur = cur.item()
        for name in ("b", "b_delta", "b_eta", "u", "bt", "B"):
            if hasattr(cur, name):
                out[name] = _unwrap_mat_field(getattr(cur, name))
            elif isinstance(cur, np.ndarray) and cur.dtype.names and name in cur.dtype.names:
                out[name] = _unwrap_mat_field(cur[name])
    for name in ("b", "b_delta", "b_eta", "u", "bt", "B"):
        if name in data and name not in out:
            out[name] = _unwrap_mat_field(data[name])
    out["bt"] = xps_to_bt(out)
    return out


def default_btens(n_measurements: int = 54) -> np.ndarray:
    """Return a deterministic lightweight b-tensor protocol for smoke tests.

    Parameters
    ----------
    n_measurements
        Number of measurements to generate. The default 54 matches the MLP
        benchmark input size.

    Returns
    -------
    numpy.ndarray
        B-tensor Voigt array with shape ``(n_measurements, 6)`` in ``s/m^2``.

    Notes
    -----
    Real analyses should pass the external acquisition XPS file with
    ``--xps-path``. This fallback exists so the package and tests are usable
    without shipping large or private data.
    """

    rng = np.random.default_rng(12345)
    dirs = rng.normal(size=(int(n_measurements), 3))
    dirs[0] = np.asarray([1.0, 0.0, 0.0])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-15
    bvals = np.linspace(0.0, 2.0e9, int(n_measurements))
    btens = []
    for b, u in zip(bvals, dirs):
        btens.append(make_tensor(float(b), 0.0, 0.0, u))
    return convert_3x3_to_1x6(np.asarray(btens))


def rician_noise(clean: np.ndarray, snr: float, s0: float, rng: np.random.Generator) -> np.ndarray:
    """Add Rician noise using the same normalized form as the notebooks.

    Parameters
    ----------
    clean
        Clean signal vector.
    snr
        Bulk signal-to-noise ratio.
    s0
        Baseline signal amplitude.
    rng
        NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Noisy signal vector with the same shape as ``clean``.
    """

    clean = np.asarray(clean, dtype=float)
    z1 = rng.standard_normal(clean.shape)
    z2 = rng.standard_normal(clean.shape)
    return float(s0) * np.sqrt((clean / float(s0) + z1 / float(snr)) ** 2 + (z2 / float(snr)) ** 2)
