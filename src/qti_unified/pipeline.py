"""End-to-end patho workflow orchestration.

This module ties the small domain modules together into the operations exposed
by the CLI. It keeps repository code separate from large data by accepting all
patient, model, covariance-fit, and generated-data locations as explicit paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import GeneratedPathoCase, PathoRunConfig, resolve_data_root, resolve_optional_path
from .patho import generate_patho_suite
from .qti_math import gt_scalars_from_params
from .signals import write_case_outputs


def _require_pandas():
    """Import pandas with a clear setup message.

    Returns
    -------
    module
        Imported pandas module.
    """

    try:
        import pandas as pd  # type: ignore

        return pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for tables. Install the package with `python -m pip install -e .[dev]`.") from exc


def generate_patho_data(config: PathoRunConfig, scenarios: list[str] | None = None) -> list[GeneratedPathoCase]:
    """Generate patho DTDs, stored GT, exact/cumexp signals, and noisy signals.

    Parameters
    ----------
    config
        Runtime generation options. ``data_root`` is created when needed;
        ``xps_path`` may point to an external acquisition protocol.
    scenarios
        Optional subset of patho scenario names. ``None`` generates all known
        patho scenarios.

    Returns
    -------
    list[GeneratedPathoCase]
        Generated case objects including paths and in-memory smoke-test arrays.

    Side Effects
    ------------
    Writes JSON and NIfTI outputs under ``config.data_root`` only.
    """

    data_root = resolve_data_root(config.data_root)
    cases = generate_patho_suite(scenarios=scenarios, n_tensors=config.n_tensors, seed=config.seed)
    generated: list[GeneratedPathoCase] = []
    for case in cases:
        generated.append(
            write_case_outputs(
                case=case,
                data_root=data_root,
                xps_path=config.xps_path,
                snr=config.snr,
                n_realizations=config.n_realizations,
                s0=config.s0,
            )
        )
    return generated


def discover_prediction_cases(
    data_root: str | Path | None = None,
    cov_fit_root: str | Path | None = None,
    snr_folder: str = "SNR30",
) -> list[dict[str, object]]:
    """Find generated noisy patho signals and their stored GT files.

    Parameters
    ----------
    data_root
        Generated-data root. Defaults to ``QTI_DATA_ROOT`` or ``<repo>/data``.
    cov_fit_root
        Optional external covariance-fit root. Missing paths are recorded but
        never required.
    snr_folder
        SNR subfolder name, for example ``SNR30``.

    Returns
    -------
    list[dict[str, object]]
        One row per noisy signal realization, with scenario, case, signal path,
        GT JSON path, and optional covariance-fit path.
    """

    root = resolve_data_root(data_root)
    results_root = root / "Results_2_MLP_patho"
    snr_root = root / "Results_SNR_fit_2_MLP_patho"
    cov_root = resolve_optional_path(cov_fit_root, "QTI_COV_FIT_ROOT")
    rows: list[dict[str, object]] = []
    if not snr_root.exists():
        return rows

    for noisy_dir in sorted(p for p in snr_root.glob(f"*/*/{snr_folder}") if p.is_dir()):
        scenario = noisy_dir.parents[1].name
        case = noisy_dir.parents[0].name
        gt_json = results_root / scenario / case / f"{scenario}__{case}__GT_params.json"
        if not gt_json.exists():
            raise FileNotFoundError(f"Stored GT JSON is missing for {scenario}/{case}: {gt_json}")
        for signal_path in sorted(noisy_dir.glob("*.nii*")):
            cov_dps_path = None
            if cov_root is not None:
                cov_dps_path = cov_root / scenario / case / snr_folder / signal_path.stem / "wlls" / "dtd_covariance_dps.mat"
            rows.append(
                {
                    "scenario": scenario,
                    "case": case,
                    "signal_path": str(signal_path),
                    "gt_json": str(gt_json),
                    "cov_dps_path": str(cov_dps_path) if cov_dps_path is not None else None,
                    "cov_dps_exists": bool(cov_dps_path and cov_dps_path.exists()),
                    "snr_folder": snr_folder,
                    "realization": signal_path.stem,
                }
            )
    return rows


def compare_patho_predictions(
    data_root: str | Path | None = None,
    model_root: str | Path | None = None,
    cov_fit_root: str | Path | None = None,
    snr_folder: str = "SNR30",
    output_root: str | Path | None = None,
    device: str = "cpu",
) -> object:
    """Run MLP inference on generated noisy patho signals and create figures.

    Parameters
    ----------
    data_root
        Generated-data root containing ``Results_SNR_fit_2_MLP_patho`` and
        ``Results_2_MLP_patho``.
    model_root
        External model checkpoint directory. When ``None``, ``QTI_MODEL_ROOT``
        is used.
    cov_fit_root
        Optional external covariance-fit directory. Missing covariance fits are
        plotted as absent rather than failing the workflow.
    snr_folder
        SNR subfolder to compare.
    output_root
        Output directory for the comparison CSV and PNG figures. Defaults to
        ``<data_root>/runs/QTI_MLP_synthetic_compare_patho``.
    device
        Torch device string, usually ``cpu`` or ``cuda``.

    Returns
    -------
    pandas.DataFrame
        Per-realization comparison table with GT, MLP, and optional cov-fit
        scalar columns.

    Side Effects
    ------------
    Writes ``patho_comparison.csv`` and scenario PNG files under ``output_root``.
    """

    from .mlp import INVAR_KEYS, collect_benchmark_model_paths, ensemble_predict
    from .plotting import build_comparison_table, plot_scenario_comparisons

    root = resolve_data_root(data_root)
    resolved_model_root = resolve_optional_path(model_root, "QTI_MODEL_ROOT")
    if resolved_model_root is None:
        raise ValueError("model_root is required, either as --model-root or QTI_MODEL_ROOT.")
    rows = discover_prediction_cases(root, cov_fit_root=cov_fit_root, snr_folder=snr_folder)
    if not rows:
        raise FileNotFoundError(f"No generated patho noisy signals found under {root / 'Results_SNR_fit_2_MLP_patho'}")

    model_paths = collect_benchmark_model_paths(resolved_model_root)
    signal_paths = [str(row["signal_path"]) for row in rows]
    predictions = ensemble_predict(signal_paths, model_paths=model_paths, invar_keys=INVAR_KEYS, device=device)
    df = build_comparison_table(rows, predictions, invar_keys=INVAR_KEYS)

    out = Path(output_root).expanduser().resolve() if output_root else root / "runs" / "QTI_MLP_synthetic_compare_patho"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "patho_comparison.csv", index=False)
    plot_scenario_comparisons(df, out, invar_keys=INVAR_KEYS)
    return df


def validate_gt(data_root: str | Path | None = None, tolerance: float = 1e-10) -> pd.DataFrame:
    """Recompute GT metrics from DTD JSONs and compare them to stored GT JSONs.

    Parameters
    ----------
    data_root
        Generated-data root containing ``Results_2_MLP_patho``.
    tolerance
        Absolute tolerance used to mark scalar values as passing.

    Returns
    -------
    pandas.DataFrame
        Validation table with scenario, case, key, stored value, recomputed
        value, absolute difference, and pass flag.

    Side Effects
    ------------
    This is the only public workflow function that recomputes GT metrics.
    """

    pd = _require_pandas()
    root = resolve_data_root(data_root)
    result_root = root / "Results_2_MLP_patho"
    records: list[dict[str, object]] = []
    for gt_json in sorted(result_root.glob("*/*/*_GT_params.json")):
        payload = json.loads(gt_json.read_text(encoding="utf-8"))
        dtd_path = Path(payload["dtd_file"])
        if not dtd_path.exists():
            raise FileNotFoundError(f"DTD JSON referenced by {gt_json} does not exist: {dtd_path}")
        params = json.loads(dtd_path.read_text(encoding="utf-8"))
        recomputed = gt_scalars_from_params(params)
        stored = payload["gt"]
        for key in ["MD", "MD_um2_per_ms", "FA", "uFA", "C_MD", "C_c"]:
            lhs = stored.get(key)
            rhs = recomputed.get(key)
            if lhs is None or rhs is None:
                diff = np.nan
                ok = lhs is rhs
            else:
                diff = abs(float(lhs) - float(rhs))
                ok = bool(diff <= float(tolerance))
            records.append(
                {
                    "scenario": payload.get("scenario"),
                    "case": payload.get("case"),
                    "key": key,
                    "stored": lhs,
                    "recomputed": rhs,
                    "abs_diff": diff,
                    "ok": ok,
                    "gt_json": str(gt_json),
                }
            )
    return pd.DataFrame.from_records(records)


def load_generated_exact_signals(data_root: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load generated exact patho signals for patient-target ranking.

    Parameters
    ----------
    data_root
        Generated-data root containing ``Results_2_MLP_patho``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping ``scenario/case`` to exact signal vector.
    """

    import nibabel as nib

    root = resolve_data_root(data_root)
    out: dict[str, np.ndarray] = {}
    for signal_path in sorted((root / "Results_2_MLP_patho").glob("*/*/*__exact.nii*")):
        scenario = signal_path.parents[1].name
        case = signal_path.parents[0].name
        arr = np.asarray(nib.load(str(signal_path)).get_fdata(), dtype=float).reshape(-1)
        out[f"{scenario}/{case}"] = arr
    return out


def compare_patient_target(
    patient_signal_path: str | Path,
    data_root: str | Path | None = None,
    target: str = "Brain mean",
    output_csv: str | Path | None = None,
) -> object:
    """Rank generated patho exact signals against an external patient target.

    Parameters
    ----------
    patient_signal_path
        External 4D patient signal NIfTI path. It is read but never copied into
        the repository.
    data_root
        Generated-data root containing exact synthetic patho signals.
    target
        Patient target name. Use ``Brain mean`` or one of the named target
        voxels from :mod:`qti_unified.patient`.
    output_csv
        Optional CSV path for the ranking table.

    Returns
    -------
    pandas.DataFrame
        Sorted table with simulated case label and RMSE to the patient target.
    """

    from .patient import extract_target_signal, load_patient_signal, rank_simulated_cases

    patient_signal = load_patient_signal(patient_signal_path)
    target_signal = extract_target_signal(patient_signal, target_name=target)
    simulated = load_generated_exact_signals(data_root)
    if not simulated:
        raise FileNotFoundError("No exact generated patho signals found. Run patho-generate first.")
    df = rank_simulated_cases(target_signal, simulated)
    if output_csv is not None:
        path = Path(output_csv).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def write_generation_manifest(generated: list[GeneratedPathoCase], data_root: str | Path) -> Path:
    """Write a compact manifest for a generated patho run.

    Parameters
    ----------
    generated
        Generated case objects returned by :func:`generate_patho_data`.
    data_root
        Generated-data root.

    Returns
    -------
    pathlib.Path
        Manifest JSON path.
    """

    root = resolve_data_root(data_root)
    run_root = root / "runs" / "QTI_MLP_synthetic_compare_patho"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "n_cases": len(generated),
        "cases": [
            {
                "scenario": item.case.scenario,
                "case": item.case.case,
                "gt_json": item.paths.get("gt_json"),
                "dtd_json": item.paths.get("dtd_json"),
                "exact_signal": item.paths.get("exact_signal"),
                "cumexp_signal": item.paths.get("cumexp_signal"),
                "noisy_dir": item.paths.get("noisy_dir"),
            }
            for item in generated
        ],
    }
    path = run_root / "generation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
