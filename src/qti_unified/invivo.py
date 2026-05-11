"""In-vivo versus patho synthetic winner plots.

This module ports the ``invivo_compare_patho`` winner-plot workflow from
``C:/SynQTI-IR/SynDTDs_MLP_patho.ipynb`` into reusable package code. Patient
images and XPS files remain external inputs; only derived PNGs or CSV summaries
should be copied into this repository.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .qti_math import read_xps_mat


TARGET_OPTIONS = ["Deep CC", "Partial CSF/WM", "Pure CSF", "Brain mean"]


@dataclass(frozen=True)
class HighlightVoxel:
    """A named patient voxel highlighted in the all-four comparison plot.

    Parameters
    ----------
    name
        Human-readable target name.
    xyz
        Voxel coordinate in data-array order.
    color
        Matplotlib-compatible color.
    """

    name: str
    xyz: tuple[int, int, int]
    color: str


DEFAULT_HIGHLIGHT_VOXELS = (
    HighlightVoxel("Deep CC", (35, 47, 9), "#ff00ff"),
    HighlightVoxel("Partial CSF/WM", (50, 54, 10), "#00ff66"),
    HighlightVoxel("Pure CSF", (42, 52, 8), "#00e5ff"),
)


def short_scenario_name(scenario_name: str) -> str:
    """Return the compact filename stem used by the reference notebook.

    Parameters
    ----------
    scenario_name
        Full scenario name.

    Returns
    -------
    str
        Short, filesystem-safe scenario name.
    """

    s = scenario_name.lower()
    clean = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if "demy" in s:
        return "demyelination"
    if "crossing_needles_3_xyz" in s:
        return "crossing_xyz"
    if "crossing_needles_2" in s:
        return "crossing_needles"
    if "packed_crossings" in s or "packed_crossing" in s:
        return "packed_crossings"
    if "cross" in s:
        return clean[:24] if clean else "crossing"
    if "orient" in s or "progress" in s or "o_prog" in s:
        return "o_prog"
    if "partial" in s:
        return "partial"
    if "iso" in s and ("bimod" in s or "bi_mod" in s or "bi-mod" in s):
        return "iso_bimod"
    if "iso" in s and ("uni" in s or "uniform" in s):
        return "iso_uni"
    return clean[:24] if clean else "scenario"


def short_target_name(target_name: str) -> str:
    """Return a compact filename-safe target name.

    Parameters
    ----------
    target_name
        Target label.

    Returns
    -------
    str
        Short target label.
    """

    mapping = {
        "Deep CC": "deepcc",
        "Partial CSF/WM": "partial",
        "Pure CSF": "purecsf",
        "Brain mean": "brainmean",
    }
    return mapping.get(target_name, re.sub(r"[^a-z0-9]+", "", target_name.lower()))


def bt_to_bvals(btens: np.ndarray) -> np.ndarray:
    """Compute b-values from Voigt b-tensor rows.

    Parameters
    ----------
    btens
        B-tensors with shape ``(n, 6)``.

    Returns
    -------
    numpy.ndarray
        B-value vector with shape ``(n,)``.
    """

    bt = np.asarray(btens, dtype=float)
    return bt[:, 0] + bt[:, 1] + bt[:, 2]


def _xps_u_rows(xps: dict[str, np.ndarray]) -> np.ndarray:
    u = np.asarray(xps["u"], dtype=float)
    if u.shape[0] == 3 and u.shape[-1] != 3:
        u = u.T
    return u.reshape(-1, 3)


def normalize_voxels_by_b0(bvals: np.ndarray, signal: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Normalize voxel signals by their b0 signal.

    Parameters
    ----------
    bvals
        B-values aligned to the signal columns.
    signal
        Array with shape ``(n_voxels, n_measurements)``.
    eps
        Small lower bound for the denominator.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Normalized signal and per-voxel S0 values.
    """

    b = np.asarray(bvals, dtype=float).reshape(-1)
    sig = np.asarray(signal, dtype=float)
    b0_idx = np.where(b == 0)[0]
    if b0_idx.size > 0:
        s0 = np.mean(sig[:, b0_idx], axis=1)
    else:
        s0 = np.max(sig, axis=1)
    denom = np.maximum(s0, eps)
    return sig / denom[:, None], s0


def protocol_match_indices(xps_subset: dict[str, np.ndarray], xps_full: dict[str, np.ndarray], label: str = "reduced brain XPS") -> np.ndarray:
    """Match reduced XPS rows to a full acquisition protocol.

    Parameters
    ----------
    xps_subset
        Reduced XPS dictionary.
    xps_full
        Full XPS dictionary.
    label
        Label used in error messages.

    Returns
    -------
    numpy.ndarray
        Integer indices into the full protocol.
    """

    sub_rows = np.column_stack(
        [
            np.asarray(xps_subset["b"], dtype=float).reshape(-1) / 1e9,
            np.asarray(xps_subset["b_delta"], dtype=float).reshape(-1),
            np.asarray(xps_subset.get("b_eta", np.zeros_like(xps_subset["b"])), dtype=float).reshape(-1),
            _xps_u_rows(xps_subset),
        ]
    )
    full_rows = np.column_stack(
        [
            np.asarray(xps_full["b"], dtype=float).reshape(-1) / 1e9,
            np.asarray(xps_full["b_delta"], dtype=float).reshape(-1),
            np.asarray(xps_full.get("b_eta", np.zeros_like(xps_full["b"])), dtype=float).reshape(-1),
            _xps_u_rows(xps_full),
        ]
    )
    dist = np.linalg.norm(sub_rows[:, None, :] - full_rows[None, :, :], axis=2)
    idx = np.argmin(dist, axis=1)
    best = dist[np.arange(len(idx)), idx]
    if np.any(best > 1e-8):
        bad = np.where(best > 1e-8)[0][:5]
        raise ValueError(f"{label}: could not match reduced channels to full XPS rows; bad rows={bad.tolist()}")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{label}: reduced channels did not match unique full-XPS rows")
    return idx.astype(int)


def _find_case_files(case_root: Path) -> tuple[list[Path], list[Path]]:
    nii = list(case_root.glob("*__exact.nii")) + list(case_root.glob("*_exact.nii")) + list(case_root.glob("*exact.nii.gz"))
    xps = list(case_root.glob("*_xps.mat"))
    return nii, xps


def _load_nifti(path: str | Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=float)


@dataclass
class InvivoPathoPlotter:
    """Create in-vivo winner plots for generated patho scenarios.

    Parameters
    ----------
    results_root
        Generated patho result root containing ``scenario/case/*__exact.nii``.
    brain_signal_path
        External 4D patient signal NIfTI path.
    brain_xps_path
        Reduced patient XPS path.
    brain_full_xps_path
        Full patient XPS path used to map reduced channels into the image.
    brain_mask_path
        External brain mask NIfTI path.
    output_dir
        Directory where PNGs and summary CSV are written.
    z_idx
        Slice index used for brain mean and simulated 4D signals.
    highlights
        Named target voxels to overlay and consider for winning targets.
    max_sim_voxels
        Maximum simulated voxels plotted per scenario.
    """

    results_root: Path
    brain_signal_path: Path
    brain_xps_path: Path
    brain_full_xps_path: Path
    brain_mask_path: Path
    output_dir: Path
    z_idx: int = 10
    highlights: tuple[HighlightVoxel, ...] = DEFAULT_HIGHLIGHT_VOXELS
    max_sim_voxels: int = 200

    def __post_init__(self) -> None:
        self.results_root = Path(self.results_root).expanduser().resolve()
        self.brain_signal_path = Path(self.brain_signal_path).expanduser().resolve()
        self.brain_xps_path = Path(self.brain_xps_path).expanduser().resolve()
        self.brain_full_xps_path = Path(self.brain_full_xps_path).expanduser().resolve()
        self.brain_mask_path = Path(self.brain_mask_path).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()

        self.xps_brain = read_xps_mat(self.brain_xps_path)
        self.xps_brain_full = read_xps_mat(self.brain_full_xps_path)
        self.bt_brain = np.asarray(self.xps_brain["bt"], dtype=float)
        self.b_brain = np.asarray(self.xps_brain.get("b", bt_to_bvals(self.bt_brain)), dtype=float).reshape(-1)
        self.bshape_brain = np.asarray(self.xps_brain.get("b_delta", np.zeros_like(self.b_brain)), dtype=float).reshape(-1)
        self.brain_channel_idx = protocol_match_indices(self.xps_brain, self.xps_brain_full)

        self.brain_img = nib.load(str(self.brain_signal_path))
        self.brain_dataobj = self.brain_img.dataobj
        self.x_size, self.y_size, self.z_size, self.n_full = self.brain_img.shape
        if self.n_full <= int(np.max(self.brain_channel_idx)):
            raise ValueError("Brain image does not contain all matched reduced-XPS volume indices.")
        self.brain_mask = np.asarray(nib.load(str(self.brain_mask_path)).get_fdata(), dtype=bool)
        self._slice_cache: dict[int, np.ndarray] = {}
        self._voxel_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self.scenario_cases = self.build_scenario_cases()

    def get_reduced_brain_slice(self, z_idx: int) -> np.ndarray:
        """Return a reduced-XPS patient slice.

        Parameters
        ----------
        z_idx
            Slice index.

        Returns
        -------
        numpy.ndarray
            Slice array with shape ``(x, y, n_reduced_measurements)``.
        """

        z = int(z_idx)
        if z not in self._slice_cache:
            try:
                data = np.asarray(self.brain_dataobj[:, :, z, self.brain_channel_idx], dtype=float)
            except Exception:
                data = np.asarray(self.brain_dataobj[:, :, z, :], dtype=float)[..., self.brain_channel_idx]
            self._slice_cache[z] = data
        return self._slice_cache[z]

    def get_reduced_brain_voxel(self, x: int, y: int, z: int) -> np.ndarray:
        """Return one reduced-XPS patient voxel signal.

        Parameters
        ----------
        x, y, z
            Voxel coordinate in data-array order.

        Returns
        -------
        numpy.ndarray
            Signal vector with shape ``(n_reduced_measurements,)``.
        """

        key = (int(x), int(y), int(z))
        if key not in self._voxel_cache:
            self._voxel_cache[key] = np.asarray(self.brain_dataobj[key[0], key[1], key[2], :], dtype=float)[self.brain_channel_idx]
        return self._voxel_cache[key]

    def build_scenario_cases(self) -> dict[str, list[tuple[str, Path, Path]]]:
        """Discover exact simulated signals and copied XPS files by scenario.

        Returns
        -------
        dict[str, list[tuple[str, pathlib.Path, pathlib.Path]]]
            Mapping scenario to ``(case, signal_path, xps_path)`` rows.
        """

        scenario_cases: dict[str, list[tuple[str, Path, Path]]] = {}
        for scen_dir in sorted(p for p in self.results_root.iterdir() if p.is_dir()):
            scenario = scen_dir.name
            cases = []
            for case_root in sorted(p for p in scen_dir.iterdir() if p.is_dir() and not p.name.startswith("plots_")):
                nii_list, xps_list = _find_case_files(case_root)
                if not nii_list or not xps_list:
                    continue
                for sig_path in sorted(nii_list):
                    case = re.sub(rf"^{re.escape(scenario)}__", "", sig_path.stem)
                    case = re.sub(r"__?exact$", "", case)
                    cases.append((case, sig_path, xps_list[0]))
            if cases:
                scenario_cases[scenario] = cases
        return scenario_cases

    def target_signal(self, target_name: str, z_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Extract and normalize one in-vivo target signal.

        Parameters
        ----------
        target_name
            ``Brain mean`` or one of the configured highlight voxel names.
        z_idx
            Optional slice override.

        Returns
        -------
        tuple[numpy.ndarray, int]
            Normalized target signal and measurement count.
        """

        z = self.z_idx if z_idx is None else int(z_idx)
        n_meas = len(self.brain_channel_idx)
        if target_name == "Brain mean":
            s_brain = self.get_reduced_brain_slice(z)
            mask_slice = self.brain_mask[:, :, z]
            s_brain = s_brain[mask_slice, :]
            s_brain, s0 = normalize_voxels_by_b0(self.b_brain, s_brain)
            s_brain = s_brain[s0 > 1e-6, :]
            if s_brain.shape[0] == 0:
                raise ValueError(f"No valid brain voxels found on slice z={z}.")
            return np.mean(s_brain, axis=0), n_meas

        matches = [voxel for voxel in self.highlights if voxel.name == target_name]
        if not matches:
            raise ValueError(f"Unknown target: {target_name}")
        voxel = matches[0]
        x, y, vz = voxel.xyz
        if not (0 <= x < self.x_size and 0 <= y < self.y_size and 0 <= vz < self.z_size):
            raise ValueError(f"Target voxel {target_name} is outside image bounds: {voxel.xyz}")
        s_vox = self.get_reduced_brain_voxel(x, y, vz)[None, :]
        s_vox_norm, s0_vox = normalize_voxels_by_b0(self.b_brain, s_vox)
        if s0_vox[0] <= 1e-6:
            raise ValueError(f"Target voxel {target_name} has near-zero b0 signal.")
        return s_vox_norm[0], n_meas

    def load_sim_signals_for_fit(self, sig_path: str | Path, xps_path: str | Path, n_meas: int, z_idx: int | None = None) -> np.ndarray:
        """Load and normalize candidate simulated signals.

        Parameters
        ----------
        sig_path
            Exact simulated signal NIfTI path.
        xps_path
            Scenario XPS path copied next to the generated case.
        n_meas
            Number of measurements to compare.
        z_idx
            Optional slice override for 4D simulated images.

        Returns
        -------
        numpy.ndarray
            Normalized simulated signals with shape ``(n_voxels, n_meas)``.
        """

        z = self.z_idx if z_idx is None else int(z_idx)
        xps_scen = read_xps_mat(xps_path)
        bt_scen = np.asarray(xps_scen["bt"], dtype=float)
        b_scen = np.asarray(xps_scen.get("b", bt_to_bvals(bt_scen)), dtype=float).reshape(-1)
        s_sim = _load_nifti(sig_path)
        if s_sim.ndim == 1:
            s_sim = s_sim[:n_meas][None, :]
        elif s_sim.ndim == 4:
            z_sim = z if s_sim.shape[2] > z else (s_sim.shape[2] // 2)
            s_sim = s_sim[:, :, z_sim, :n_meas].reshape(-1, n_meas)
        else:
            s_sim = np.squeeze(s_sim)
            s_sim = s_sim.reshape(-1, s_sim.shape[-1])[:, :n_meas]
        s_sim, s0_sim = normalize_voxels_by_b0(b_scen[: s_sim.shape[1]], s_sim)
        return s_sim[s0_sim > 1e-6, :]

    def compute_results_for_target(self, scenario: str, target_name: str, z_idx: int | None = None) -> list[dict[str, object]]:
        """Rank all cases in one scenario by RMSE to an in-vivo target.

        Parameters
        ----------
        scenario
            Scenario name discovered under ``results_root``.
        target_name
            Target signal name.
        z_idx
            Optional slice override.

        Returns
        -------
        list[dict[str, object]]
            Sorted result dictionaries; the best match is first.
        """

        z = self.z_idx if z_idx is None else int(z_idx)
        target_signal, n_meas = self.target_signal(target_name, z)
        sort_idx = np.lexsort((self.bshape_brain, self.b_brain))
        target_sorted = target_signal[sort_idx]
        results = []
        for case, sig_path, xps_path in self.scenario_cases[scenario]:
            s_sim = self.load_sim_signals_for_fit(sig_path, xps_path, n_meas, z)
            if s_sim.shape[0] == 0:
                continue
            s_sim_sorted = s_sim[:, sort_idx]
            rmse_per_voxel = np.sqrt(np.mean((s_sim_sorted - target_sorted[None, :]) ** 2, axis=1))
            best_voxel_idx = int(np.argmin(rmse_per_voxel))
            results.append(
                {
                    "case": case,
                    "rmse": float(rmse_per_voxel[best_voxel_idx]),
                    "best_voxel_idx": best_voxel_idx,
                    "best_signal": s_sim_sorted[best_voxel_idx],
                    "n_sim_voxels": int(s_sim_sorted.shape[0]),
                    "sig_path": Path(sig_path),
                    "xps_path": Path(xps_path),
                    "target_sorted": target_sorted,
                    "n_meas": int(n_meas),
                }
            )
        return sorted(results, key=lambda row: float(row["rmse"]))

    def save_winner_all4_plot(self, scenario: str, winner_target: str, winner_best: dict[str, object], out_dir: str | Path | None = None, z_idx: int | None = None) -> Path:
        """Save the all-four in-vivo winner plot for one scenario.

        Parameters
        ----------
        scenario
            Scenario name.
        winner_target
            Winning target name.
        winner_best
            Best-match dictionary from :meth:`compute_results_for_target`.
        out_dir
            Optional output directory. Defaults to ``self.output_dir``.
        z_idx
            Optional slice override.

        Returns
        -------
        pathlib.Path
            Written PNG path.
        """

        import matplotlib.pyplot as plt  # type: ignore

        z = self.z_idx if z_idx is None else int(z_idx)
        out = self.output_dir if out_dir is None else Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        n_meas = int(winner_best["n_meas"])

        s_brain = self.get_reduced_brain_slice(z)
        mask_slice = self.brain_mask[:, :, z]
        highlight_signals = []
        for voxel in self.highlights:
            x, y, vz = voxel.xyz
            if not (0 <= x < self.x_size and 0 <= y < self.y_size and 0 <= vz < self.z_size):
                continue
            s_vox = self.get_reduced_brain_voxel(x, y, vz)[None, :]
            s_vox_norm, s0_vox = normalize_voxels_by_b0(self.b_brain, s_vox)
            if s0_vox[0] <= 1e-6:
                continue
            highlight_signals.append({"name": voxel.name, "xyz": voxel.xyz, "color": voxel.color, "signal": s_vox_norm[0]})

        s_brain = s_brain[mask_slice, :]
        s_brain, s0 = normalize_voxels_by_b0(self.b_brain, s_brain)
        s_brain = s_brain[s0 > 1e-6, :]

        sort_idx = np.lexsort((self.bshape_brain, self.b_brain))
        bshape_sorted = self.bshape_brain[sort_idx]
        s_brain_sorted = s_brain[:, sort_idx]

        s_sim = self.load_sim_signals_for_fit(winner_best["sig_path"], winner_best["xps_path"], n_meas, z)
        if s_sim.shape[0] > self.max_sim_voxels:
            rng = np.random.default_rng(1)
            sel = rng.choice(s_sim.shape[0], size=self.max_sim_voxels, replace=False)
            s_sim = s_sim[sel, :]
        s_sim_sorted = s_sim[:, sort_idx]

        nvox = s_brain_sorted.shape[0]
        x_real = np.tile(np.arange(n_meas), nvox)
        y_real = s_brain_sorted.reshape(-1)
        c_real = np.tile(bshape_sorted, nvox)
        nsim = s_sim_sorted.shape[0]
        x_sim = np.tile(np.arange(s_sim_sorted.shape[1]), nsim)
        y_sim = s_sim_sorted.reshape(-1)
        mean_brain = np.mean(s_brain_sorted, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        cmap = plt.cm.bwr
        c_norm = (c_real - c_real.min()) / max(c_real.max() - c_real.min(), 1e-12)
        rgba = cmap(c_norm)
        rgba[:, 3] = 0.05
        ax.scatter(x_real, y_real, s=12, linewidths=0, facecolors=rgba, zorder=1)
        ax.plot(np.arange(n_meas), mean_brain, color="yellow", linewidth=2.0, label="Brain mean", zorder=2000)

        for hv in highlight_signals:
            y_h = hv["signal"][sort_idx]
            ax.plot(
                np.arange(n_meas),
                y_h,
                color=hv["color"],
                linewidth=3.0,
                label=f"{hv['name']} [{hv['xyz'][0]},{hv['xyz'][1]},{hv['xyz'][2]}]",
                zorder=4,
            )
            ax.scatter(np.arange(n_meas), y_h, s=16, color=hv["color"], edgecolors="none", zorder=5)

        ax.scatter(
            x_sim,
            y_sim,
            marker="x",
            s=40,
            linewidths=1.8,
            color="black",
            alpha=1.0,
            label=f"Winning simulated voxels ({winner_best['case']})",
            zorder=1000,
        )
        ax.set_title(
            f"Scenario winner | {scenario} | {winner_best['case']} | "
            f"target={winner_target} | RMSE={float(winner_best['rmse']):.6f}"
        )
        ax.set_xlabel("Channel index (sorted by b-value -> b-shape)")
        ax.set_ylabel("Signal intensity (normalized)")
        ax.set_xlim(-1, n_meas)
        ax.set_ylim(-0.02, 1.12)
        sm = plt.cm.ScalarMappable(cmap="bwr")
        sm.set_array(c_real)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("b-shape")
        ax.grid(False)
        ax.legend(loc="upper center")

        out_path = out / f"{short_scenario_name(scenario)}_winner_all4.png"
        plt.tight_layout()
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def export_winner_plots(self, scenarios: list[str] | None = None, target_options: list[str] | None = None, summary_csv: str | Path | None = None) -> object:
        """Export one winner all-four PNG per patho scenario.

        Parameters
        ----------
        scenarios
            Optional scenario subset. Defaults to all discovered scenarios.
        target_options
            Candidate targets. Defaults to the notebook's four target options.
        summary_csv
            Optional summary CSV path. Defaults to ``best_fit_summary_patho.csv``
            in ``output_dir``.

        Returns
        -------
        pandas.DataFrame | list[dict[str, object]]
            Summary rows as a DataFrame when pandas is installed, otherwise a
            list of dictionaries.
        """

        selected = sorted(self.scenario_cases) if scenarios is None else list(scenarios)
        targets = TARGET_OPTIONS if target_options is None else list(target_options)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for scenario in selected:
            target_best_rows = []
            for target_name in targets:
                results = self.compute_results_for_target(scenario, target_name, self.z_idx)
                if not results:
                    continue
                best = results[0]
                target_best_rows.append(
                    {
                        "scenario": scenario,
                        "scenario_short": short_scenario_name(scenario),
                        "target": target_name,
                        "target_short": short_target_name(target_name),
                        "best_dtd": best["case"],
                        "rmse": best["rmse"],
                        "best_voxel_idx": best["best_voxel_idx"],
                        "best_obj": best,
                    }
                )
            if not target_best_rows:
                continue
            winner = min(target_best_rows, key=lambda row: float(row["rmse"]))
            out_png = self.save_winner_all4_plot(scenario, str(winner["target"]), winner["best_obj"], self.output_dir, self.z_idx)
            summary_rows.append(
                {
                    "scenario": winner["scenario"],
                    "scenario_short": winner["scenario_short"],
                    "winning_target": winner["target"],
                    "winning_target_short": winner["target_short"],
                    "best_dtd": winner["best_dtd"],
                    "rmse": winner["rmse"],
                    "best_voxel_idx": winner["best_voxel_idx"],
                    "winner_all4_png": out_png.name,
                    "tag": "patho",
                }
            )

        csv_path = Path(summary_csv).expanduser().resolve() if summary_csv else self.output_dir / "best_fit_summary_patho.csv"
        if summary_rows:
            try:
                import pandas as pd  # type: ignore

                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(csv_path, index=False)
                return summary_df
            except ModuleNotFoundError:
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
                    writer.writeheader()
                    writer.writerows(summary_rows)
        return summary_rows


def export_invivo_patho_winner_plots(
    results_root: str | Path,
    brain_signal_path: str | Path,
    brain_xps_path: str | Path,
    brain_full_xps_path: str | Path,
    brain_mask_path: str | Path,
    output_dir: str | Path,
    z_idx: int = 10,
    scenarios: list[str] | None = None,
) -> object:
    """Convenience wrapper for exporting in-vivo patho winner plots.

    Parameters
    ----------
    results_root
        Generated patho result root containing exact signals.
    brain_signal_path
        External 4D patient signal NIfTI path.
    brain_xps_path
        Reduced patient XPS path.
    brain_full_xps_path
        Full patient XPS path.
    brain_mask_path
        External brain mask NIfTI path.
    output_dir
        Directory where PNGs and summary CSV are written.
    z_idx
        Slice index.
    scenarios
        Optional scenario subset.

    Returns
    -------
    pandas.DataFrame | list[dict[str, object]]
        Export summary.
    """

    plotter = InvivoPathoPlotter(
        results_root=Path(results_root),
        brain_signal_path=Path(brain_signal_path),
        brain_xps_path=Path(brain_xps_path),
        brain_full_xps_path=Path(brain_full_xps_path),
        brain_mask_path=Path(brain_mask_path),
        output_dir=Path(output_dir),
        z_idx=z_idx,
    )
    return plotter.export_winner_plots(scenarios=scenarios)
