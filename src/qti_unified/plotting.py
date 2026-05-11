"""Patho comparison tables and figures.

Plotting reads stored GT values from ``*_GT_params.json`` and never recomputes
ground truth from DTDs. Optional covariance-fit paths are read only when
available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

INVAR_KEYS = ["MD", "FA", "uFA", "C_c", "C_MD"]


def load_gt_values(gt_json: str | Path, invar_keys: list[str] | None = None) -> dict[str, float]:
    """Read stored GT values from a GT JSON file.

    Parameters
    ----------
    gt_json
        Path to ``*_GT_params.json``.
    invar_keys
        Invariants to return. Defaults to ``MD, FA, uFA, C_c, C_MD``.

    Returns
    -------
    dict[str, float]
        Stored GT values in plotting units.
    """

    keys = INVAR_KEYS if invar_keys is None else list(invar_keys)
    payload = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    gt = payload["gt"]
    out: dict[str, float] = {}
    for key in keys:
        if key == "MD":
            value = gt.get("MD_um2_per_ms", gt.get("E_um2_per_ms"))
        else:
            value = gt.get(key)
        out[key] = float(value) if value is not None and np.isfinite(value) else np.nan
    return out


def covariance_fit_values(dps_path: str | Path | None, invar_keys: list[str] | None = None) -> dict[str, float]:
    """Read optional covariance-fit scalar values from a ``dps.mat`` file.

    Parameters
    ----------
    dps_path
        Path to ``dtd_covariance_dps.mat``. Missing paths return NaNs.
    invar_keys
        Invariants to read.

    Returns
    -------
    dict[str, float]
        Covariance-fit values keyed by invariant.
    """

    keys = INVAR_KEYS if invar_keys is None else list(invar_keys)
    if dps_path is None:
        return {key: np.nan for key in keys}
    path = Path(dps_path)
    if not path.exists():
        return {key: np.nan for key in keys}
    try:
        import scipy.io as sio  # type: ignore

        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        dps = mat.get("dps")
        out = {}
        for key in keys:
            if dps is not None and hasattr(dps, key):
                arr = np.asarray(getattr(dps, key), dtype=float).reshape(-1)
                out[key] = float(arr[0]) if arr.size else np.nan
            else:
                out[key] = np.nan
        return out
    except Exception:
        return {key: np.nan for key in keys}


def parse_case_for_plotting(scenario: str, case: str) -> tuple[float, str, str]:
    """Parse a scenario/case tag into x-value, figure key, and x-label.

    Parameters
    ----------
    scenario
        Scenario folder name.
    case
        Case tag.

    Returns
    -------
    tuple[float, str, str]
        X value, figure grouping key, and x-axis label. Single-case scenarios
        return ``0.0``.
    """

    scen = scenario.strip().lower().replace("-", "_").replace(" ", "_")
    tag = case.strip().lower()
    if scen in {"crossing_needles_3_xyz", "needle_sphere_2", "spheres_2"}:
        return 0.0, scen, "Case"
    if scen == "crossing_needles_2":
        match = re.search(r"angle([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "crossing_angle", "Crossing angle [deg]"
    if scen == "packed_crossings_conserved_2":
        match = re.search(r"crossing_angle([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "crossing_angle", "Crossing angle [deg]"
    if scen == "fanning_sticks_1500":
        match = re.search(r"span([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "fanning_span", "Fan span [deg]"
    if scen == "sticks_2":
        match = re.search(r"dperp[_-]?([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "sticks_dperp", "D_perp [um2/ms]"
    if scen == "spheres_sizevar_2":
        match = re.search(r"_w([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "spheres_sizevar", "Sphere size variance [w]"
    if scen.startswith("frac_gauss"):
        match = re.search(r"frac_([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "frac", "Isotropic fraction"
    if scen.startswith("iso_bimod"):
        match = re.search(r"iso_bimod([0-9.]+)_v([0-9.]+)", tag)
        if match:
            return float(match.group(2)), f"md_{float(match.group(1))}", "V[D_iso] [um4/ms2]"
    if scen.startswith("iso_uni"):
        match = re.search(r"iso_uni([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "iso_uni", "E[D_iso] [um2/ms]"
    if scen.startswith("o_prog"):
        match = re.search(r"odi_([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "o_prog", "ODI"
    if scen.startswith("demyelination_progressive_sticks"):
        match = re.search(r"end([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "demyelination", "End D_perp [um2/ms]"
    if scen.startswith("partial_volume_triplet"):
        match = re.search(r"fiso([0-9.]+)", tag)
        if match:
            return float(match.group(1)), "partial_volume", "Isotropic fraction"
    return 0.0, "case_index", "Case"


def build_comparison_table(case_rows: list[dict[str, object]], predictions: np.ndarray, invar_keys: list[str] | None = None) -> pd.DataFrame:
    """Build the per-realization comparison table used for CSVs and plots.

    Parameters
    ----------
    case_rows
        Case metadata rows containing ``scenario``, ``case``, ``gt_json``, and
        optional ``cov_dps_path``.
    predictions
        Prediction array with shape ``(n_rows, z, x, y, n_invars)``.
    invar_keys
        Invariant names.

    Returns
    -------
    pandas.DataFrame
        One row per signal realization with GT, MLP, and optional cov-fit
        columns.
    """

    import pandas as pd  # type: ignore

    keys = INVAR_KEYS if invar_keys is None else list(invar_keys)
    records = []
    gt_cache: dict[str, dict[str, float]] = {}
    for i, row in enumerate(case_rows):
        gt_path = str(row["gt_json"])
        if gt_path not in gt_cache:
            gt_cache[gt_path] = load_gt_values(gt_path, keys)
        cov = covariance_fit_values(row.get("cov_dps_path"), keys)
        pred_vec = predictions[i, 0, 0, 0, :]
        rec = dict(row)
        for j, key in enumerate(keys):
            rec[f"pred_{key}"] = float(pred_vec[j])
            rec[f"gt_{key}"] = gt_cache[gt_path][key]
            rec[f"cov_{key}"] = cov.get(key, np.nan)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def scenario_plot_name(scenario: str) -> str:
    """Return the output PNG name for a scenario.

    Parameters
    ----------
    scenario
        Scenario folder name.

    Returns
    -------
    str
        File name for the saved plot.
    """

    alias = {
        "crossing_needles_2": "crossing",
        "crossing_needles_3_xyz": "crossing_needles_3_xyz",
        "packed_crossings_conserved_2": "packed_crossing",
        "demyelination_progressive_sticks": "demyelination",
        "fanning_sticks_1500": "fanning",
        "frac_gauss_2": "frac",
        "iso_bimod_gauss_2": "iso_bimod",
        "iso_uni_gauss_2": "iso_uni",
        "needle_sphere_2": "needle_sphere_2",
        "o_prog_gauss_2": "o_prog",
        "partial_volume_triplet": "partial_volume",
        "spheres_2": "spheres",
        "spheres_sizevar_2": "spheres_sizevar",
        "sticks_2": "sticks",
    }
    return f"{alias.get(scenario, scenario)}_patho_cov.png"


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def plot_scenario_comparisons(df: pd.DataFrame, out_dir: str | Path, invar_keys: list[str] | None = None) -> list[Path]:
    """Create GT/MLP/cov-fit mean-std figures for every scenario.

    Parameters
    ----------
    df
        Comparison table from :func:`build_comparison_table`.
    out_dir
        Output directory for PNG files.
    invar_keys
        Invariant names to plot.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """

    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore

    keys = INVAR_KEYS if invar_keys is None else list(invar_keys)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    plot_specs = [
        ("MD", "E[D_iso] [um2/ms]"),
        ("FA", "FA"),
        ("uFA", "uFA"),
        ("C_c", "C_c"),
        ("C_MD", "C_MD"),
    ]
    plot_specs = [spec for spec in plot_specs if spec[0] in keys]

    for scenario, df_s in df.groupby("scenario", sort=True):
        parsed = [parse_case_for_plotting(str(r["scenario"]), str(r["case"])) for _, r in df_s.iterrows()]
        df_s = df_s.copy()
        df_s["plot_x"] = [p[0] for p in parsed]
        df_s["plot_key"] = [p[1] for p in parsed]
        xlabel = parsed[0][2] if parsed else "Case"

        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        axes = axes.ravel()
        for ax_i, (key, ylabel) in enumerate(plot_specs):
            ax = axes[ax_i]
            x_sorted = np.asarray(sorted(df_s["plot_x"].unique()), dtype=float)
            gt_vals = []
            pred_mu = []
            pred_sd = []
            cov_mu = []
            cov_sd = []
            for x in x_sorted:
                grp = df_s[df_s["plot_x"] == x]
                gt_vals.append(_mean_std(grp[f"gt_{key}"].tolist())[0])
                mu, sd = _mean_std(grp[f"pred_{key}"].tolist())
                pred_mu.append(mu)
                pred_sd.append(sd)
                c_mu_val, c_sd_val = _mean_std(grp[f"cov_{key}"].tolist())
                cov_mu.append(c_mu_val)
                cov_sd.append(c_sd_val)

            if x_sorted.size == 1:
                ax.plot(x_sorted, gt_vals, color="k", marker="*", linestyle="None", markersize=8)
            else:
                ax.plot(x_sorted, gt_vals, color="k", linestyle="--", linewidth=1.5)
            ax.errorbar(x_sorted, pred_mu, yerr=pred_sd, fmt="o", color="tab:blue", ecolor="tab:blue", markersize=5, capsize=3)
            if np.any(np.isfinite(cov_mu)):
                ax.errorbar(x_sorted, cov_mu, yerr=cov_sd, fmt="^", color="tab:orange", ecolor="tab:orange", markersize=5, capsize=3)
            ax.grid(True, ls=":", alpha=0.4)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(key)
        for ax in axes[len(plot_specs) :]:
            ax.axis("off")
        legend_handles = [
            Line2D([0], [0], color="k", linestyle="--", linewidth=1.0, marker="*", markersize=3.0, label="Ground truth"),
            Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=0.9, marker="o", markersize=2.4, label="MLP mean +/- std"),
            Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=0.9, marker="^", markersize=2.4, label="Cov fit mean +/- std"),
        ]
        axes[0].legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            fontsize=6,
            handlelength=1.2,
            handletextpad=0.3,
            labelspacing=0.2,
            borderpad=0.15,
            borderaxespad=0.2,
            markerscale=0.55,
            numpoints=1,
        )
        fig.suptitle(f"{str(scenario).replace('_', ' ').title()} | patho", fontsize=14, y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = out / scenario_plot_name(str(scenario))
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
