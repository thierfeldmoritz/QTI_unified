"""Patho DTD scenario and case generation.

The public entry point is :func:`generate_patho_suite`, which yields the same
scenario/case naming layout as the current patho notebooks while keeping the
implementation compact and deterministic. Each generated case is a list of
diffusion tensor parameter dictionaries using SI units (``m^2/s``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable

import numpy as np

from .config import PathoCase
from .qti_math import axisym_eigvals_to_projection_metrics, projection_metrics_to_eigvals, trace_shape_to_eigvals


def _stable_seed(base_seed: int, scenario: str, case: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{scenario}:{case}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _unit(v: Iterable[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    return arr / (np.linalg.norm(arr) + 1e-15)


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(int(n), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-15
    return v


def _spherical_fibonacci_points(count: int) -> np.ndarray:
    i = np.arange(int(count), dtype=float) + 0.5
    phi = 2.0 * np.pi * i / ((1.0 + np.sqrt(5.0)) / 2.0)
    z = 1.0 - 2.0 * i / float(count)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)


def _kappa_from_odi(odi: float, cap: float = 10000.0, eps: float = 1e-12) -> float:
    odi = float(np.clip(odi, eps, 1.0 - eps))
    return float(min(1.0 / np.tan(0.5 * np.pi * odi), cap))


def _watson_sample(mu: Iterable[float], odi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    mu_arr = _unit(mu)
    udirs = _spherical_fibonacci_points(2000)
    dots = udirs @ mu_arr
    logits = _kappa_from_odi(odi) * (dots**2)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    out = udirs[rng.choice(len(udirs), size=int(n), p=weights)]
    out[np.sum(out * mu_arr[None, :], axis=1) < 0] *= -1.0
    return out


def _truncated_normal(rng: np.random.Generator, mean: float, sd: float, lo: float, hi: float, size: int) -> np.ndarray:
    if sd <= 0:
        return np.full(int(size), float(np.clip(mean, lo, hi)))
    out = np.empty(int(size), dtype=float)
    i = 0
    while i < int(size):
        cand = rng.normal(mean, sd, int(size) - i)
        cand = cand[(cand >= lo) & (cand <= hi)]
        if cand.size:
            take = min(cand.size, int(size) - i)
            out[i : i + take] = cand[:take]
            i += take
    return out


def _param(lambda_1: float, lambda_2: float, lambda_3: float, direction: Iterable[float]) -> dict[str, float]:
    lam = np.maximum(np.sort(np.asarray([lambda_1, lambda_2, lambda_3], dtype=float))[::-1], 1e-12)
    if lam[0] <= lam[1]:
        lam[0] = lam[1] + 1e-12
    u = _unit(direction)
    d_iso = float(lam.mean())
    d_delta = float((lam[0] - 0.5 * (lam[1] + lam[2])) / lam.sum())
    return {
        "lambda_1": float(lam[0]),
        "lambda_2": float(lam[1]),
        "lambda_3": float(lam[2]),
        "u1": float(u[0]),
        "u2": float(u[1]),
        "u3": float(u[2]),
        "d_iso": d_iso,
        "d_delta": d_delta,
    }


def _fibers_watson(
    n: int,
    rng: np.random.Generator,
    mu: Iterable[float] = (1.0, 0.0, 0.0),
    odi: float = 0.2,
    base_d_par: float = 1.7e-9,
    base_d_perp: float = 0.3e-9,
    sigma_iso: float = 0.10,
    sigma_delta: float = 0.10,
) -> list[dict[str, float]]:
    n = int(max(0, n))
    dirs = _watson_sample(mu, odi, n, rng)
    base_iso, base_delta = axisym_eigvals_to_projection_metrics(base_d_par, base_d_perp)
    d_iso = _truncated_normal(rng, base_iso, sigma_iso * base_iso, 1e-12, 3.05e-9, n)
    d_delta = _truncated_normal(rng, base_delta, sigma_delta, -0.5, 1.0, n)
    params = []
    for iso, delta, direction in zip(d_iso, d_delta, dirs):
        lam = trace_shape_to_eigvals(3.0 * iso, delta)
        params.append(_param(lam[0], lam[1], lam[2], direction))
    return params


def _inject_aniso_contamination(params: list[dict[str, float]], rng: np.random.Generator, fraction: float = 0.03) -> None:
    n_contam = int(round(float(fraction) * len(params)))
    if n_contam <= 0:
        return
    for idx in rng.choice(len(params), size=n_contam, replace=False):
        p = params[int(idx)]
        d_iso = max(float(p["d_iso"]), 1e-12)
        d_delta = float(rng.uniform(0.55, 0.80))
        lam = projection_metrics_to_eigvals(d_iso, d_delta)
        params[int(idx)] = _param(lam[0], lam[1], lam[2], _random_unit_vectors(1, rng)[0])


def _finish(scenario: str, case: str, params: list[dict[str, float]], seed: int, descriptor: object) -> PathoCase:
    return PathoCase(
        scenario=scenario,
        case=case,
        params=params,
        metadata={"scenario": scenario, "case": case, "n_tensors": len(params), "seed": int(seed), "descriptor": descriptor},
    )


def _gen_o_prog(case: str, n: int, seed: int) -> PathoCase:
    odi = float(case.split("_")[-1])
    rng = np.random.default_rng(seed)
    target_e_diso = 0.8e-9
    ddelta = math.sqrt(0.47)
    params = _fibers_watson(
        n,
        rng,
        mu=(1, 0, 0),
        odi=odi,
        base_d_par=target_e_diso * (1.0 + 2.0 * ddelta),
        base_d_perp=target_e_diso * (1.0 - ddelta),
        sigma_iso=0.0,
        sigma_delta=0.1 * ddelta,
    )
    return _finish("o_prog_gauss_2", case, params, seed, odi)


def _gen_iso_bimod(case: str, n: int, seed: int) -> PathoCase:
    mean_um = 0.80
    var_um = float(case.rsplit("_V", 1)[1])
    rng = np.random.default_rng(seed)
    sigma_um = 0.05
    delta = math.sqrt(max(var_um - sigma_um**2, 0.0))
    n1 = n // 2
    vals = np.concatenate([rng.normal(mean_um - delta, sigma_um, n1), rng.normal(mean_um + delta, sigma_um, n - n1)])
    params = [_param(v, v, v, (1, 0, 0)) for v in np.clip(vals * 1e-9, 1e-12, None)]
    _inject_aniso_contamination(params, rng)
    return _finish("iso_bimod_gauss_2", case, params, seed, var_um)


def _gen_iso_uni(case: str, n: int, seed: int) -> PathoCase:
    mean_um = float(case.replace("iso_uni", ""))
    rng = np.random.default_rng(seed)
    vals = np.clip(rng.normal(mean_um, 0.10 * mean_um, n) * 1e-9, 1e-12, None)
    params = [_param(v, v, v, (1, 0, 0)) for v in vals]
    _inject_aniso_contamination(params, rng)
    return _finish("iso_uni_gauss_2", case, params, seed, mean_um)


def _gen_frac(case: str, n: int, seed: int) -> PathoCase:
    f_iso = float(case.replace("Frac_", ""))
    rng = np.random.default_rng(seed)
    n_iso = int(round(f_iso * n))
    n_wm = n - n_iso
    wm = _fibers_watson(n_wm, rng, mu=(1, 0, 0), odi=0.219, base_d_par=1.77e-9, base_d_perp=0.31e-9, sigma_iso=0.05, sigma_delta=0.06)
    csf = [_param(v, v, v, (1, 0, 0)) for v in np.clip(rng.normal(3.0, 0.1, n_iso) * 1e-9, 1e-12, None)]
    params = wm + csf
    rng.shuffle(params)
    return _finish("frac_gauss_2", case, params, seed, f_iso)


def _gen_sticks(case: str, n: int, seed: int) -> PathoCase:
    d_perp_um = float(case.rsplit("_", 1)[1])
    rng = np.random.default_rng(seed)
    params = [_param(1.7e-9, d_perp_um * 1e-9, d_perp_um * 1e-9, u) for u in _random_unit_vectors(n, rng)]
    return _finish("sticks_2", case, params, seed, d_perp_um)


def _gen_needle_sphere(case: str, n: int, seed: int) -> PathoCase:
    n_needle = max(1, n // 2)
    n_sphere = max(1, n - n_needle)
    params = [_param(1.7e-9, 0.1e-9, 0.1e-9, (0, 0, 1)) for _ in range(n_needle)]
    params += [_param(2.0e-9, 2.0e-9, 2.0e-9, (1, 0, 0)) for _ in range(n_sphere)]
    return _finish("needle_sphere_2", case, params, seed, "needle+sphere")


def _gen_spheres(case: str, n: int, seed: int) -> PathoCase:
    rng = np.random.default_rng(seed)
    vals = np.clip(rng.normal(1.40, 0.08, n) * 1e-9, 1e-12, None)
    params = [_param(v, v, v, (1, 0, 0)) for v in vals]
    return _finish("spheres_2", case, params, seed, 1.40)


def _gen_spheres_sizevar(case: str, n: int, seed: int) -> PathoCase:
    width_um = float(case.rsplit("_w", 1)[1])
    rng = np.random.default_rng(seed)
    if width_um == 0.0:
        vals = np.full(n, 1.40)
    else:
        vals = rng.uniform(max(1e-6, 1.40 - width_um), 1.40 + width_um, n)
    params = [_param(v * 1e-9, v * 1e-9, v * 1e-9, (1, 0, 0)) for v in vals]
    return _finish("spheres_sizevar_2", case, params, seed, width_um)


def _gen_fanning(case: str, n: int, seed: int) -> PathoCase:
    span = float(case.rsplit("span", 1)[1])
    rng = np.random.default_rng(seed)
    params = []
    for angle in rng.uniform(-0.5 * span, 0.5 * span, n):
        params.append(_param(1.7e-9, 0.1e-9, 0.1e-9, (math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0)))
    return _finish("fanning_sticks_1500", case, params, seed, span)


def _gen_demyelination(case: str, n: int, seed: int) -> PathoCase:
    end_um = float(case.rsplit("end", 1)[1])
    rng = np.random.default_rng(seed)
    dperp = np.linspace(0.03e-9, end_um * 1e-9, n)
    dirs = _watson_sample((1, 0, 0), 0.18, n, rng)
    params = [_param(1.7e-9, dp, dp, u) for dp, u in zip(dperp, dirs)]
    return _finish("demyelination_progressive_sticks", case, params, seed, end_um)


def _gen_partial(case: str, n: int, seed: int) -> PathoCase:
    f_iso = float(case.rsplit("fiso", 1)[1])
    rng = np.random.default_rng(seed)
    n_iso = int(round(f_iso * n))
    n_tissue = n - n_iso
    n_stick = n_tissue // 2
    n_zep = n_tissue - n_stick
    stick = _fibers_watson(n_stick, rng, mu=(1, 0, 0), odi=0.12, base_d_par=1.7e-9, base_d_perp=0.10e-9, sigma_iso=0.04, sigma_delta=0.05)
    zep = _fibers_watson(n_zep, rng, mu=(0, 1, 0), odi=0.36, base_d_par=1.2e-9, base_d_perp=0.45e-9, sigma_iso=0.06, sigma_delta=0.07)
    params = stick + zep + [_param(3.0e-9, 3.0e-9, 3.0e-9, (1, 0, 0)) for _ in range(n_iso)]
    rng.shuffle(params)
    return _finish("partial_volume_triplet", case, params, seed, f_iso)


def _gen_packed_crossing(case: str, n: int, seed: int) -> PathoCase:
    angle = float(case.split("_", 2)[1].replace("angle", ""))
    rng = np.random.default_rng(seed)
    theta = math.radians(angle)
    mu2 = (math.cos(theta), math.sin(theta), 0.0)
    n_b1 = n // 2
    n_b2 = n - n_b1
    params = _fibers_watson(n_b1, rng, mu=(1, 0, 0), odi=0.18, base_d_par=1.7e-9, base_d_perp=0.10e-9, sigma_iso=0.05, sigma_delta=0.06)
    params += _fibers_watson(n_b2, rng, mu=mu2, odi=0.20, base_d_par=1.7e-9, base_d_perp=0.11e-9, sigma_iso=0.05, sigma_delta=0.06)
    rng.shuffle(params)
    return _finish("packed_crossings_conserved_2", case, params, seed, angle)


def _gen_crossing_needles(case: str, n: int, seed: int) -> PathoCase:
    angle = float(case.rsplit("angle", 1)[1])
    theta = math.radians(angle)
    n1 = max(1, n // 2)
    n2 = max(1, n - n1)
    params = [_param(1.7e-9, 0.1e-9, 0.1e-9, (1, 0, 0)) for _ in range(n1)]
    params += [_param(1.7e-9, 0.1e-9, 0.1e-9, (math.cos(theta), math.sin(theta), 0.0)) for _ in range(n2)]
    return _finish("crossing_needles_2", case, params, seed, angle)


def _gen_crossing_needles_3(case: str, n: int, seed: int) -> PathoCase:
    counts = [n // 3, n // 3, n - 2 * (n // 3)]
    dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    params: list[dict[str, float]] = []
    for count, direction in zip(counts, dirs):
        params += [_param(1.7e-9, 0.1e-9, 0.1e-9, direction) for _ in range(max(1, count))]
    return _finish("crossing_needles_3_xyz", case, params, seed, "orthogonal_xyz")


CASE_NAMES: dict[str, list[str]] = {
    "crossing_needles_2": [f"crossing_needles_2_angle{x:02d}" for x in [20, 28, 36, 44, 52, 60, 68, 76, 84, 90]],
    "crossing_needles_3_xyz": ["crossing_needles_3_xyz"],
    "demyelination_progressive_sticks": ["demyelination_progressive_sticks_end0.03", "demyelination_progressive_sticks_end0.10", "demyelination_progressive_sticks_end0.15", "demyelination_progressive_sticks_end0.20", "demyelination_progressive_sticks_end0.25", "demyelination_progressive_sticks_end0.30"],
    "fanning_sticks_1500": [f"fanning_sticks_1500_span{x:02d}" for x in range(0, 91, 10)],
    "frac_gauss_2": [f"Frac_{x:.2f}" for x in np.arange(0.0, 1.0, 0.1)],
    "iso_bimod_gauss_2": [f"iso_bimod0.80_V{x:.2f}" for x in [0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.27, 0.33, 0.41, 0.49]],
    "iso_uni_gauss_2": [f"iso_uni{x:.2f}" for x in [0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.85, 2.10, 2.35, 2.60]],
    "needle_sphere_2": ["needle_sphere_2tensors"],
    "o_prog_gauss_2": [f"ODI_{x:.2f}" for x in [0.02, 0.08, 0.14, 0.20, 0.28, 0.36, 0.46, 0.58, 0.72, 0.90]],
    "packed_crossings_conserved_2": [f"crossing_angle{x:02d}_frac50_50" for x in [20, 28, 36, 44, 52, 60, 68, 76, 84, 90]],
    "partial_volume_triplet": [f"partial_volume_triplet_fiso{x:.2f}" for x in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]],
    "spheres_2": ["spheres_csf_1500"],
    "spheres_sizevar_2": [f"spheres_sizevar_1500_w{x:.2f}" for x in [0.00, 0.20, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00]],
    "sticks_2": [f"stick_1500_dperp_{x:.2f}" for x in [0.05, 0.08, 0.11, 0.14, 0.18, 0.22, 0.27, 0.33, 0.40, 0.48]],
}


GENERATORS: dict[str, Callable[[str, int, int], PathoCase]] = {
    "crossing_needles_2": _gen_crossing_needles,
    "crossing_needles_3_xyz": _gen_crossing_needles_3,
    "demyelination_progressive_sticks": _gen_demyelination,
    "fanning_sticks_1500": _gen_fanning,
    "frac_gauss_2": _gen_frac,
    "iso_bimod_gauss_2": _gen_iso_bimod,
    "iso_uni_gauss_2": _gen_iso_uni,
    "needle_sphere_2": _gen_needle_sphere,
    "o_prog_gauss_2": _gen_o_prog,
    "packed_crossings_conserved_2": _gen_packed_crossing,
    "partial_volume_triplet": _gen_partial,
    "spheres_2": _gen_spheres,
    "spheres_sizevar_2": _gen_spheres_sizevar,
    "sticks_2": _gen_sticks,
}


def available_scenarios() -> list[str]:
    """List patho scenario names supported by the generator.

    Returns
    -------
    list[str]
        Sorted scenario names.
    """

    return sorted(CASE_NAMES)


def generate_patho_case(scenario: str, case: str, n_tensors: int = 1500, seed: int = 42) -> PathoCase:
    """Generate one patho DTD case.

    Parameters
    ----------
    scenario
        Scenario name from :func:`available_scenarios`.
    case
        Case tag listed in ``CASE_NAMES[scenario]``.
    n_tensors
        Number of microscopic tensors to generate.
    seed
        Base seed. The final case seed is made deterministic from scenario and
        case names.

    Returns
    -------
    PathoCase
        Generated DTD case.
    """

    if scenario not in GENERATORS:
        raise KeyError(f"Unknown scenario {scenario!r}. Available: {', '.join(available_scenarios())}")
    if case not in CASE_NAMES[scenario]:
        raise KeyError(f"Unknown case {case!r} for {scenario!r}.")
    case_seed = _stable_seed(seed, scenario, case)
    return GENERATORS[scenario](case, int(n_tensors), case_seed)


def generate_patho_suite(scenarios: list[str] | None = None, n_tensors: int = 1500, seed: int = 42) -> list[PathoCase]:
    """Generate all requested patho DTD cases.

    Parameters
    ----------
    scenarios
        Optional subset of scenario names. ``None`` generates all scenarios.
    n_tensors
        Number of microscopic tensors per case.
    seed
        Base seed for deterministic case-specific seeding.

    Returns
    -------
    list[PathoCase]
        Generated cases in stable scenario/case order.
    """

    selected = available_scenarios() if scenarios is None else list(scenarios)
    cases: list[PathoCase] = []
    for scenario in selected:
        if scenario not in CASE_NAMES:
            raise KeyError(f"Unknown scenario {scenario!r}.")
        for case in CASE_NAMES[scenario]:
            cases.append(generate_patho_case(scenario, case, n_tensors=n_tensors, seed=seed))
    return cases
