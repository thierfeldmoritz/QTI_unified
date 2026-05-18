"""Patho DTD scenario generation from the original SynQTI-IR notebook logic.

The reusable sampling and tensor utilities come from the vendored
``utils.dtd_math`` module copied from ``C:/SynQTI-IR``. This module only wraps
the notebook scenarios as library-style functions and preserves the patho
scenario/case names used by the old workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from utils.dtd_math import generate_fibers_WM_watson, random_unit_vectors, truncated_normal

from .config import PathoCase


def _stable_seed(base_seed: int, scenario: str, case: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{scenario}:{case}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _axisym_param(lambda_1: float, lambda_23: float, direction: object) -> dict[str, float]:
    u = np.asarray(direction, dtype=float)
    u /= np.linalg.norm(u) + 1e-15
    d_iso = (float(lambda_1) + 2.0 * float(lambda_23)) / 3.0
    d_delta = (float(lambda_1) - float(lambda_23)) / (float(lambda_1) + 2.0 * float(lambda_23))
    return {
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_23),
        "lambda_3": float(lambda_23),
        "u1": float(u[0]),
        "u2": float(u[1]),
        "u3": float(u[2]),
        "d_iso": float(d_iso),
        "d_delta": float(d_delta),
    }


def _sphere_param(d_iso: float) -> dict[str, float]:
    return {
        "lambda_1": float(d_iso),
        "lambda_2": float(d_iso),
        "lambda_3": float(d_iso),
        "u1": 1.0,
        "u2": 0.0,
        "u3": 0.0,
        "d_iso": float(d_iso),
        "d_delta": 0.0,
    }


def _positive_normal(rng: np.random.Generator, mean: float, sigma: float, n: int) -> np.ndarray:
    out = np.empty(int(n), dtype=float)
    i = 0
    while i < int(n):
        cand = rng.normal(mean, sigma, int(n) - i)
        cand = cand[cand > 0]
        if cand.size:
            take = min(cand.size, int(n) - i)
            out[i : i + take] = cand[:take]
            i += take
    return out


def _finish(scenario: str, case: str, params: list[dict[str, float]], seed: int, descriptor: object) -> PathoCase:
    return PathoCase(
        scenario=scenario,
        case=case,
        params=params,
        metadata={"scenario": scenario, "case": case, "n_tensors": len(params), "seed": int(seed), "descriptor": descriptor},
    )


def _gen_o_prog(case: str, n: int, seed: int) -> PathoCase:
    odi = float(case.split("_")[-1])
    rng = np.random.default_rng(11)
    params = generate_fibers_WM_watson(
        n=int(n),
        mu=(1, 0, 0),
        odi=odi,
        base_d_par=1.70e-9,
        base_d_perp=0.03e-9,
        sigma_iso=0.0,
        sigma_delta=0.0,
        udirs=None,
        grid_size=2000,
        rng=rng,
    )
    return _finish("o_prog_gauss_2", case, params, seed, odi)


def _gen_iso_bimod(case: str, n: int, seed: int) -> PathoCase:
    mean_um = 0.80
    var_um = float(case.rsplit("_V", 1)[1])
    sigma_um = 0.05
    mean = mean_um * 1e-9
    variance = var_um * 1e-18
    sigma = sigma_um * 1e-9
    delta = math.sqrt(max(variance - sigma**2, 0.0))
    if mean - delta <= 0:
        raise ValueError(f"Mean too small for requested variance: {mean - delta:.2e}")
    rng = np.random.default_rng(42)
    n_a = int(n) // 2
    n_b = int(n) - n_a
    d_iso = np.concatenate([_positive_normal(rng, mean - delta, sigma, n_a), _positive_normal(rng, mean + delta, sigma, n_b)])
    return _finish("iso_bimod_gauss_2", case, [_sphere_param(d) for d in d_iso], seed, var_um)


def _gen_iso_uni(case: str, n: int, seed: int) -> PathoCase:
    mean_um = float(case.replace("iso_uni", ""))
    rng = np.random.default_rng(7)
    d_iso = _positive_normal(rng, mean_um * 1e-9, 0.10 * mean_um * 1e-9, int(n))
    return _finish("iso_uni_gauss_2", case, [_sphere_param(d) for d in d_iso], seed, mean_um)


def _gen_frac(case: str, n: int, seed: int) -> PathoCase:
    f_iso = float(case.replace("Frac_", ""))
    rng = np.random.default_rng(11)
    n_iso = int(round(f_iso * int(n)))
    n_wm = int(n) - n_iso
    wm = (
        generate_fibers_WM_watson(
            n=n_wm,
            mu=(1, 0, 0),
            odi=0.2,
            base_d_par=1.70e-9,
            base_d_perp=0.03e-9,
            sigma_iso=0.0,
            sigma_delta=0.0,
            udirs=None,
            grid_size=2000,
            rng=rng,
        )
        if n_wm > 0
        else []
    )
    params = wm + [_sphere_param(3.0e-9) for _ in range(n_iso)]
    rng.shuffle(params)
    return _finish("frac_gauss_2", case, params, seed, f_iso)


def _gen_sticks(case: str, n: int, seed: int) -> PathoCase:
    d_perp = float(case.rsplit("_", 1)[1]) * 1e-9
    dirs = random_unit_vectors(int(n), rng=np.random.default_rng(42))
    params = [_axisym_param(1.7e-9, d_perp, u) for u in dirs]
    return _finish("sticks_2", case, params, seed, d_perp * 1e9)


def _gen_needle_sphere(case: str, n: int, seed: int) -> PathoCase:
    params = [_axisym_param(1.7e-9, 0.03e-9, (0, 0, 1)), _sphere_param(3.0e-9)]
    return _finish("needle_sphere_2", case, params, seed, "needle+sphere")


def _gen_spheres(case: str, n: int, seed: int) -> PathoCase:
    rng = np.random.default_rng(42)
    d_iso = truncated_normal(rng, mean=3.00e-9, sd=0.10e-9, lo=2.80e-9, hi=3.20e-9, size=int(n))
    return _finish("spheres_2", case, [_sphere_param(d) for d in d_iso], seed, 3.00)


def _gen_spheres_sizevar(case: str, n: int, seed: int) -> PathoCase:
    width_um = float(case.rsplit("_w", 1)[1])
    rng = np.random.default_rng(42)
    unit_samples = rng.random(int(n))
    if int(n) >= 2:
        unit_samples[0] = 0.0
        unit_samples[1] = 1.0
    rng.shuffle(unit_samples)
    d_iso_um = 3.00 - width_um * unit_samples
    return _finish("spheres_sizevar_2", case, [_sphere_param(d * 1e-9) for d in d_iso_um], seed, width_um)


def _gen_fanning(case: str, n: int, seed: int) -> PathoCase:
    span = float(case.rsplit("span", 1)[1])
    theta = np.zeros(int(n), dtype=float) if int(n) < 2 else np.linspace(0.0, span, int(n))
    dirs = np.column_stack((np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)), np.zeros(int(n), dtype=float)))
    params = [_axisym_param(1.7e-9, 0.03e-9, u) for u in dirs]
    return _finish("fanning_sticks_1500", case, params, seed, span)


def _gen_demyelination(case: str, n: int, seed: int) -> PathoCase:
    end_um = float(case.rsplit("end", 1)[1])
    rng = np.random.default_rng(2026)
    mu = np.array([1.0, 0.0, 0.0], dtype=float)
    params = generate_fibers_WM_watson(
        n=int(n),
        mu=tuple(mu),
        odi=0.05,
        base_d_par=1.7e-9,
        base_d_perp=0.03e-9,
        sigma_iso=0.03,
        sigma_delta=0.04,
        udirs=None,
        grid_size=2000,
        rng=rng,
    )
    progress = np.linspace(0.0, 1.0, int(n))
    d_perp_curve = 0.03e-9 + (end_um * 1e-9 - 0.03e-9) * progress**1.35
    for i, p in enumerate(params):
        u = np.array([p["u1"], p["u2"], p["u3"]], dtype=float)
        if np.dot(u, mu) < 0.0:
            u = -u
        lam = np.array([1.7e-9, float(d_perp_curve[i]), float(d_perp_curve[i])])
        p.update(_axisym_param(lam[0], lam[1], u))
    return _finish("demyelination_progressive_sticks", case, params, seed, end_um)


def _gen_partial(case: str, n: int, seed: int) -> PathoCase:
    f_iso = float(case.rsplit("fiso", 1)[1])
    rng = np.random.default_rng(32026)
    rem = 1.0 - f_iso
    f_stick = rem * 0.70
    f_zep = rem * 0.30
    fractions = np.array([f_stick, f_zep, f_iso], dtype=float)
    fractions /= fractions.sum()
    n_stick = int(round(fractions[0] * int(n)))
    n_zep = int(round(fractions[1] * int(n)))
    n_csf = int(n) - n_stick - n_zep
    stick = generate_fibers_WM_watson(n=n_stick, mu=(1, 0, 0), odi=0.30, base_d_par=1.7e-9, base_d_perp=0.03e-9, sigma_iso=0.04, sigma_delta=0.05, udirs=None, grid_size=2000, rng=rng)
    mu_zep = (math.cos(math.radians(32.0)), math.sin(math.radians(32.0)), 0.0)
    zep = generate_fibers_WM_watson(n=n_zep, mu=mu_zep, odi=0.30, base_d_par=1.45e-9, base_d_perp=0.45e-9, sigma_iso=0.05, sigma_delta=0.06, udirs=None, grid_size=2000, rng=rng)
    params = stick + zep + [_sphere_param(3.0e-9) for _ in range(n_csf)]
    params = [params[i] for i in rng.permutation(int(n))]
    return _finish("partial_volume_triplet", case, params, seed, f_iso)


def _gen_packed_crossing(case: str, n: int, seed: int) -> PathoCase:
    angle = float(case.split("_", 2)[1].replace("angle", ""))
    idx = CASE_NAMES["packed_crossings_conserved_2"].index(case)
    rng = np.random.default_rng(1000 + idx)
    theta = math.radians(angle)
    n_b1 = int(round(0.50 * int(n)))
    n_b2 = int(n) - n_b1
    b1 = generate_fibers_WM_watson(n=n_b1, mu=(1, 0, 0), odi=0.20, base_d_par=1.7e-9, base_d_perp=0.03e-9, sigma_iso=0.05, sigma_delta=0.06, udirs=None, grid_size=2000, rng=rng)
    b2 = generate_fibers_WM_watson(n=n_b2, mu=(math.cos(theta), math.sin(theta), 0.0), odi=0.20, base_d_par=1.7e-9, base_d_perp=0.03e-9, sigma_iso=0.05, sigma_delta=0.06, udirs=None, grid_size=2000, rng=rng)
    params = b1 + b2
    params = [params[j] for j in rng.permutation(int(n))]
    return _finish("packed_crossings_conserved_2", case, params, seed, angle)


def _gen_crossing_needles(case: str, n: int, seed: int) -> PathoCase:
    angle = float(case.rsplit("angle", 1)[1])
    theta = math.radians(angle)
    params = [_axisym_param(1.7e-9, 0.03e-9, (1, 0, 0)), _axisym_param(1.7e-9, 0.03e-9, (math.cos(theta), math.sin(theta), 0.0))]
    return _finish("crossing_needles_2", case, params, seed, angle)


def _gen_crossing_needles_3(case: str, n: int, seed: int) -> PathoCase:
    params = [_axisym_param(1.7e-9, 0.03e-9, (1, 0, 0)), _axisym_param(1.7e-9, 0.03e-9, (0, 1, 0)), _axisym_param(1.7e-9, 0.03e-9, (0, 0, 1))]
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
    """List patho scenario names supported by the old notebook generator."""

    return sorted(CASE_NAMES)


def reference_winner_root() -> Path:
    """Return the checked-in patho winner DTD reference-data directory."""

    return Path(__file__).resolve().parents[2] / "reference_data" / "DTDs_cov_suite_2_patho_winners"


def winner_case_names(scenarios: list[str] | None = None) -> dict[str, str]:
    """Map scenarios to the single checked-in winner case available for each."""

    selected = available_scenarios() if scenarios is None else list(scenarios)
    root = reference_winner_root()
    winners: dict[str, str] = {}
    for scenario in selected:
        if scenario not in CASE_NAMES:
            raise KeyError(f"Unknown scenario {scenario!r}.")
        files = sorted((root / scenario).glob("*.json"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected exactly one winner JSON for {scenario!r} under {root}. Found {len(files)}.")
        winners[scenario] = files[0].stem
    return winners


def generate_patho_case(scenario: str, case: str, n_tensors: int = 1500, seed: int = 42) -> PathoCase:
    """Generate one patho DTD case with the vendored old sampling utilities."""

    if scenario not in GENERATORS:
        raise KeyError(f"Unknown scenario {scenario!r}. Available: {', '.join(available_scenarios())}")
    if case not in CASE_NAMES[scenario]:
        raise KeyError(f"Unknown case {case!r} for {scenario!r}.")
    case_seed = _stable_seed(seed, scenario, case)
    return GENERATORS[scenario](case, int(n_tensors), case_seed)


def generate_patho_suite(scenarios: list[str] | None = None, n_tensors: int = 1500, seed: int = 42) -> list[PathoCase]:
    """Generate all requested patho DTD cases in stable scenario/case order."""

    selected = available_scenarios() if scenarios is None else list(scenarios)
    cases: list[PathoCase] = []
    for scenario in selected:
        if scenario not in CASE_NAMES:
            raise KeyError(f"Unknown scenario {scenario!r}.")
        for case in CASE_NAMES[scenario]:
            cases.append(generate_patho_case(scenario, case, n_tensors=n_tensors, seed=seed))
    return cases


def load_winner_patho_case(scenario: str) -> PathoCase:
    """Load one checked-in winner DTD case for a patho scenario."""

    winners = winner_case_names([scenario])
    case = winners[scenario]
    path = reference_winner_root() / scenario / f"{case}.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(params, list) or not params:
        raise ValueError(f"Winner DTD JSON must contain a non-empty parameter list: {path}")
    return PathoCase(
        scenario=scenario,
        case=case,
        params=params,
        metadata={
            "scenario": scenario,
            "case": case,
            "n_tensors": len(params),
            "source": "reference_winner",
            "reference_json": str(path),
        },
    )


def load_winner_patho_suite(scenarios: list[str] | None = None) -> list[PathoCase]:
    """Load the checked-in winner DTD cases for all requested scenarios."""

    return [load_winner_patho_case(scenario) for scenario in winner_case_names(scenarios)]
