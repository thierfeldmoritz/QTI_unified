import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def _orthonormal_basis_from_u(u):
    u = _unit(u)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(u, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    v = np.cross(ref, u)
    v = _unit(v)
    w = _unit(np.cross(u, v))
    return np.column_stack([u, v, w])


def _parse_params(params):
    lam = np.array(
        [[p["lambda_1"], p["lambda_2"], p["lambda_3"]] for p in params], dtype=float
    )
    u = np.array([[p["u1"], p["u2"], p["u3"]] for p in params], dtype=float)
    u = np.array([_unit(ui) for ui in u], dtype=float)
    diso = lam.mean(axis=1)
    denom = np.maximum(lam.sum(axis=1), 1e-15)
    ddelta = (lam[:, 0] - 0.5 * (lam[:, 1] + lam[:, 2])) / denom
    return lam, u, diso, ddelta


def _infer_single_population_kind(diso, ddelta):
    """Infer whether a set is sphere-like or stick-like from anisotropy stats."""
    abs_dd = np.abs(np.asarray(ddelta, dtype=float))
    frac_low_aniso = float(np.mean(abs_dd < 0.15))
    frac_high_aniso = float(np.mean(abs_dd > 0.45))

    if frac_high_aniso >= 0.75:
        return "sticks_2", frac_low_aniso, frac_high_aniso
    if frac_low_aniso >= 0.90 and frac_high_aniso <= 0.10:
        return "spheres_2", frac_low_aniso, frac_high_aniso
    return None, frac_low_aniso, frac_high_aniso


def infer_single_population_kind_from_json(json_path):
    """Public helper for quick QA in notebook cells and scripts."""
    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    lam, _, diso, ddelta = _parse_params(params)
    kind, frac_low_aniso, frac_high_aniso = _infer_single_population_kind(diso, ddelta)
    return {
        "kind": kind,
        "fraction_low_anisotropy": frac_low_aniso,
        "fraction_high_anisotropy": frac_high_aniso,
        "n_tensors": int(lam.shape[0]),
    }


def _resolve_scenario_name_from_data(scenario_name, diso, ddelta, auto_fix=True, json_path=None):
    """Auto-fix obvious sticks/spheres swaps when requested scenario contradicts data."""
    if not auto_fix:
        return scenario_name, False

    name = str(scenario_name).lower().strip()
    expected = None
    if "sticks" in name and "fanning" not in name and "demyelination" not in name:
        expected = "sticks_2"
    elif "spheres" in name or "iso_uni" in name:
        expected = "spheres_2"

    if expected is None:
        return scenario_name, False

    inferred, frac_low_aniso, frac_high_aniso = _infer_single_population_kind(diso, ddelta)
    if inferred is None or inferred == expected:
        return scenario_name, False

    source = ""
    if json_path is not None:
        source = f" from {Path(json_path).name}"
    print(
        f"[dtd_scenario_plotting] Auto-corrected scenario '{scenario_name}' to '{inferred}'"
        f"{source} (low-aniso={frac_low_aniso:.2f}, high-aniso={frac_high_aniso:.2f})."
    )
    return inferred, True


def _subsample_idx(n, max_tensors=280, seed=0):
    if n <= max_tensors:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_tensors, replace=False))


def _classify_triplet(diso, ddelta, u, mu_stick=None, mu_zep=None):
    mu_stick = _unit([1.0, 0.0, 0.0] if mu_stick is None else mu_stick)
    if mu_zep is None:
        mu_zep = [np.cos(np.deg2rad(32.0)), np.sin(np.deg2rad(32.0)), 0.0]
    mu_zep = _unit(mu_zep)

    is_csf = (diso > np.quantile(diso, 0.8)) & (np.abs(ddelta) < 0.2)
    sim_stick = np.abs(u @ mu_stick)
    sim_zep = np.abs(u @ mu_zep)
    is_stick = (~is_csf) & (sim_stick >= sim_zep)
    is_zep = (~is_csf) & (~is_stick)
    labels = np.empty(diso.size, dtype=object)
    labels[is_stick] = "stick"
    labels[is_zep] = "zep"
    labels[is_csf] = "sphere"
    return labels


def _classify_crossing(u, angle_deg=None):
    mu1 = np.array([1.0, 0.0, 0.0], dtype=float)
    if angle_deg is None:
        # Data-driven split in xy-plane as fallback.
        phi = np.arctan2(u[:, 1], u[:, 0])
        th = np.median(phi)
        return np.where(phi <= th, "bundle1", "bundle2")
    th = np.deg2rad(float(angle_deg))
    mu2 = _unit([np.cos(th), np.sin(th), 0.0])
    s1 = np.abs(u @ mu1)
    s2 = np.abs(u @ mu2)
    return np.where(s1 >= s2, "bundle1", "bundle2")


def _layout_generic_cloud(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _layout_from_scenario(scenario_name, lam, u, diso, ddelta, meta=None, seed=0):
    meta = {} if meta is None else dict(meta)
    n = lam.shape[0]
    rng = np.random.default_rng(seed)
    name = str(scenario_name).lower().strip()

    # Default grouping.
    groups = np.array(["all"] * n, dtype=object)
    xyz = _layout_generic_cloud(n, seed=seed)

    if "o_prog" in name:
        mu = np.array(meta.get("mu", [1.0, 0.0, 0.0]), dtype=float)
        mu = _unit(mu)
        angle = np.arccos(np.clip(np.abs(u @ mu), 0.0, 1.0)) / (0.5 * np.pi)
        az = np.arctan2(u[:, 2], u[:, 1]) / np.pi
        xyz[:, 0] = -0.45 + 0.9 * angle
        xyz[:, 1] = 0.30 * az
        xyz[:, 2] = 0.08 * rng.normal(size=n)
        groups = np.where(np.abs(ddelta) < 0.2, "defect-ish", "wm")

    elif "iso_bimod" in name:
        thr = np.median(diso)
        low = diso <= thr
        groups = np.where(low, "mode_low", "mode_high")
        xyz[low] = np.column_stack(
            [
                rng.normal(-0.25, 0.06, low.sum()),
                rng.normal(0.0, 0.10, low.sum()),
                rng.normal(0.0, 0.10, low.sum()),
            ]
        )
        xyz[~low] = np.column_stack(
            [
                rng.normal(0.25, 0.06, (~low).sum()),
                rng.normal(0.0, 0.10, (~low).sum()),
                rng.normal(0.0, 0.10, (~low).sum()),
            ]
        )
        contam = np.abs(ddelta) > 0.3
        groups[contam] = "contam"

    elif "iso_uni" in name or "spheres" in name:
        mu_d = float(np.mean(diso))
        sd_d = float(np.std(diso) + 1e-15)
        r = np.clip((diso - mu_d) / (3.0 * sd_d), -0.4, 0.4)
        phi = rng.uniform(0, 2 * np.pi, n)
        ct = rng.uniform(-1, 1, n)
        st = np.sqrt(np.maximum(0.0, 1.0 - ct**2))
        xyz[:, 0] = (0.22 + r) * st * np.cos(phi)
        xyz[:, 1] = (0.22 + r) * st * np.sin(phi)
        xyz[:, 2] = (0.22 + r) * ct
        groups = np.where(np.abs(ddelta) > 0.3, "contam", "iso")

    elif "frac" in name:
        csf = (diso > np.quantile(diso, 0.8)) & (np.abs(ddelta) < 0.2)
        groups = np.where(csf, "csf", "wm")
        xyz[~csf] = np.column_stack(
            [
                rng.normal(-0.2, 0.10, (~csf).sum()),
                rng.normal(0.0, 0.18, (~csf).sum()),
                rng.normal(0.0, 0.18, (~csf).sum()),
            ]
        )
        xyz[csf] = np.column_stack(
            [
                rng.normal(0.24, 0.08, csf.sum()),
                rng.normal(0.0, 0.14, csf.sum()),
                rng.normal(0.0, 0.14, csf.sum()),
            ]
        )

    elif "sticks" in name and "fanning" not in name and "demyelination" not in name:
        x = np.linspace(-0.45, 0.45, n)
        rng.shuffle(x)
        xyz[:, 0] = x
        xyz[:, 1] = 0.15 * rng.normal(size=n)
        xyz[:, 2] = 0.15 * rng.normal(size=n)
        groups = np.where(np.abs(ddelta) < 0.25, "perturbed", "stick")

    elif "fanning" in name:
        phi = np.arctan2(u[:, 1], u[:, 0])
        phi = (phi - phi.min()) / (phi.max() - phi.min() + 1e-15)
        ang = (-0.7 + 1.4 * phi) * (np.pi / 2)
        r = 0.35 + 0.08 * rng.normal(size=n)
        xyz[:, 0] = r * np.cos(ang)
        xyz[:, 1] = r * np.sin(ang)
        xyz[:, 2] = 0.10 * rng.normal(size=n)
        groups = np.where(np.abs(ddelta) < 0.25, "defect-ish", "fan")

    elif "demyelination" in name:
        order = np.argsort(diso)
        x = np.linspace(-0.45, 0.45, n)
        xyz[order, 0] = x
        xyz[:, 1] = 0.14 * rng.normal(size=n)
        xyz[:, 2] = 0.14 * rng.normal(size=n)
        groups = np.where(np.abs(ddelta) < np.median(np.abs(ddelta)), "late", "early")

    elif "partial_volume_triplet" in name:
        labels = _classify_triplet(
            diso,
            ddelta,
            u,
            mu_stick=meta.get("mu_stick", [1.0, 0.0, 0.0]),
            mu_zep=meta.get("mu_zep", [np.cos(np.deg2rad(32.0)), np.sin(np.deg2rad(32.0)), 0.0]),
        )
        groups = labels
        mask = labels == "stick"
        xyz[mask] = np.column_stack(
            [
                rng.normal(-0.25, 0.09, mask.sum()),
                rng.normal(0.18, 0.08, mask.sum()),
                rng.normal(0.0, 0.09, mask.sum()),
            ]
        )
        mask = labels == "zep"
        xyz[mask] = np.column_stack(
            [
                rng.normal(-0.18, 0.10, mask.sum()),
                rng.normal(-0.18, 0.08, mask.sum()),
                rng.normal(0.0, 0.09, mask.sum()),
            ]
        )
        mask = labels == "sphere"
        xyz[mask] = np.column_stack(
            [
                rng.normal(0.28, 0.08, mask.sum()),
                rng.normal(0.0, 0.12, mask.sum()),
                rng.normal(0.0, 0.12, mask.sum()),
            ]
        )

    elif "packed_crossings_conserved_2" in name or "packed_crossing" in name:
        labels = _classify_crossing(u, angle_deg=meta.get("angle_deg", None))
        groups = labels.astype(object)
        mask = labels == "bundle1"
        xyz[mask] = np.column_stack(
            [
                rng.normal(-0.26, 0.11, mask.sum()),
                rng.normal(0.15, 0.08, mask.sum()),
                rng.normal(0.0, 0.10, mask.sum()),
            ]
        )
        mask = labels == "bundle2"
        xyz[mask] = np.column_stack(
            [
                rng.normal(0.26, 0.11, mask.sum()),
                rng.normal(-0.15, 0.08, mask.sum()),
                rng.normal(0.0, 0.10, mask.sum()),
            ]
        )
        defect_like = (np.abs(ddelta) < 0.35) | (diso > np.quantile(diso, 0.90))
        groups[defect_like] = "defect-ish"

    elif "crossing" in name:
        labels = _classify_crossing(u, angle_deg=meta.get("angle_deg", None))
        groups = labels
        mask = labels == "bundle1"
        xyz[mask] = np.column_stack(
            [
                rng.normal(-0.20, 0.12, mask.sum()),
                rng.normal(0.12, 0.08, mask.sum()),
                rng.normal(0.0, 0.10, mask.sum()),
            ]
        )
        mask = labels == "bundle2"
        xyz[mask] = np.column_stack(
            [
                rng.normal(0.20, 0.12, mask.sum()),
                rng.normal(-0.12, 0.08, mask.sum()),
                rng.normal(0.0, 0.10, mask.sum()),
            ]
        )

    return xyz, groups


def _group_color_map(group_names):
    base = {
        "all": "#4E79A7",
        "wm": "#2E86AB",
        "defect-ish": "#F28E2B",
        "mode_low": "#59A14F",
        "mode_high": "#9C755F",
        "contam": "#E15759",
        "iso": "#76B7B2",
        "csf": "#17BECF",
        "stick": "#1F77B4",
        "zep": "#FF7F0E",
        "sphere": "#2CA02C",
        "fan": "#2E86AB",
        "early": "#4E79A7",
        "late": "#E15759",
        "bundle1": "#1F77B4",
        "bundle2": "#D62728",
        "perturbed": "#F28E2B",
    }
    cmap = {}
    for g in group_names:
        if g in base:
            cmap[g] = base[g]
        else:
            cmap[g] = "#999999"
    return cmap


def _make_sphere_template(nu=14, nv=10):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    pts = np.array([x.ravel(), y.ravel(), z.ravel()])
    return x, y, z, pts


def plot_scenario_voxel(
    params,
    scenario_name,
    scenario_meta=None,
    max_tensors=280,
    ellipsoid_scale=0.09,
    seed=0,
    save_path=None,
    ax=None,
    json_path=None,
):
    """
    Scenario-aware voxel plotting that preserves semantic structure.

    Parameters
    ----------
    params : list[dict]
        List of tensor parameter dicts with lambda_1..3 and u1..3.
    scenario_name : str
        Scenario key (e.g., 'o_prog_gauss_2', 'partial_volume_triplet').
    scenario_meta : dict | None
        Optional metadata such as angle_deg, mu vectors, or known fractions.
    max_tensors : int
        Maximum number of tensors to draw for readability.
    ellipsoid_scale : float
        Global size scale of rendered ellipsoids.
    seed : int
        Random seed for downsampling and layout jitter.
    save_path : str | Path | None
        Optional path to save static figure.
    ax : matplotlib 3d axis | None
        Optional existing axis.
    """
    if not params:
        raise ValueError("params is empty")

    scenario_meta = {} if scenario_meta is None else dict(scenario_meta)
    lam, u, diso, ddelta = _parse_params(params)
    auto_fix_name = bool(scenario_meta.get("auto_fix_single_population_name", True))
    scenario_name_for_layout, scenario_was_corrected = _resolve_scenario_name_from_data(
        scenario_name,
        diso,
        ddelta,
        auto_fix=auto_fix_name,
        json_path=json_path,
    )

    idx = _subsample_idx(lam.shape[0], max_tensors=max_tensors, seed=seed)
    lam = lam[idx]
    u = u[idx]
    diso = diso[idx]
    ddelta = ddelta[idx]

    xyz, groups = _layout_from_scenario(
        scenario_name_for_layout,
        lam,
        u,
        diso,
        ddelta,
        meta=scenario_meta,
        seed=seed,
    )
    unique_groups = sorted(set(groups.tolist()))
    color_map = _group_color_map(unique_groups)

    x0, y0, z0, sphere_pts = _make_sphere_template()
    lam_ref = max(float(np.percentile(lam[:, 0], 95)), 1e-18)

    own_fig = False
    if ax is None:
        own_fig = True
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    for i in range(lam.shape[0]):
        basis = _orthonormal_basis_from_u(u[i])
        axes = ellipsoid_scale * np.sqrt(np.maximum(lam[i], 0.0) / lam_ref)
        transformed = basis @ (axes[:, None] * sphere_pts)

        xe = transformed[0].reshape(x0.shape) + xyz[i, 0]
        ye = transformed[1].reshape(y0.shape) + xyz[i, 1]
        ze = transformed[2].reshape(z0.shape) + xyz[i, 2]

        ax.plot_surface(
            xe,
            ye,
            ze,
            color=color_map[groups[i]],
            linewidth=0.0,
            antialiased=False,
            alpha=0.75,
            shade=True,
        )

    lim = 0.62
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_name = str(scenario_name)
    if scenario_was_corrected:
        title_name = f"{scenario_name} [auto:{scenario_name_for_layout}]"
    ax.set_title(f"{title_name} (shown {lam.shape[0]}/{len(params)})")

    # Legend with counts per semantic group.
    handles = []
    labels = []
    for g in unique_groups:
        count = int(np.sum(groups == g))
        h = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[g], markersize=8)
        handles.append(h)
        labels.append(f"{g} ({count})")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=180, bbox_inches="tight")

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax


def _infer_crossing_angle_from_json_path(json_path):
    match = re.search(r"crossing_angle([0-9.]+)", Path(json_path).stem.lower())
    if match:
        return float(match.group(1))
    match = re.search(r"angle([0-9.]+)", Path(json_path).stem.lower())
    if match:
        return float(match.group(1))
    return None


def plot_scenario_voxel_from_json(
    json_path,
    scenario_name,
    scenario_meta=None,
    max_tensors=280,
    ellipsoid_scale=0.09,
    seed=0,
    save_path=None,
    ax=None,
):
    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    scenario_meta = {} if scenario_meta is None else dict(scenario_meta)
    name = str(scenario_name).lower().strip()
    if ("packed_crossings_conserved_2" in name or "packed_crossing" in name) and "angle_deg" not in scenario_meta:
        inferred_angle = _infer_crossing_angle_from_json_path(json_path)
        if inferred_angle is not None:
            scenario_meta["angle_deg"] = inferred_angle

    return plot_scenario_voxel(
        params=params,
        scenario_name=scenario_name,
        scenario_meta=scenario_meta,
        max_tensors=max_tensors,
        ellipsoid_scale=ellipsoid_scale,
        seed=seed,
        save_path=save_path,
        ax=ax,
        json_path=json_path,
    )
