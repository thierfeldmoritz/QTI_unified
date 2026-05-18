"""Command-line interface for the patho-only QTI workflow.

The CLI mirrors the notebook workflow but makes every large input explicit:
generated data, model checkpoints, covariance fits, and patient data are paths
outside git.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PathoRunConfig, resolve_data_root, resolve_optional_path
from .patho import available_scenarios
from .pipeline import compare_patho_predictions, compare_patient_target, generate_patho_data, validate_gt, write_generation_manifest


def _float_or_none(value: str) -> float | None:
    if value.lower() in {"none", "off", "false", "no"}:
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all patho workflow subcommands registered.
    """

    parser = argparse.ArgumentParser(prog="qti-unified", description="Patho-focused QTI synthetic generation, MLP comparison, and plotting.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("patho-generate", help="Generate patho DTDs, stored GT, exact/cumexp signals, and SNR signals.")
    gen.add_argument("--data-root", type=Path, default=None, help="Generated-data root. Defaults to QTI_DATA_ROOT or <repo>/data.")
    gen.add_argument("--xps-path", type=Path, default=None, help="External XPS .mat protocol. Defaults to QTI_XPS_PATH or built-in smoke-test protocol.")
    gen.add_argument("--scenario", action="append", choices=available_scenarios(), help="Scenario to generate. Repeat to select multiple; omit for all.")
    gen.add_argument("--winners-only", action="store_true", help="Use the checked-in per-scenario winner DTD JSONs instead of generating every case.")
    gen.add_argument("--n-tensors", type=int, default=1500, help="Tensor count per generated DTD case; ignored with --winners-only.")
    gen.add_argument("--seed", type=int, default=42, help="Base seed; ignored with --winners-only.")
    gen.add_argument("--snr", type=_float_or_none, default=30.0, help="SNR value, or 'none' to skip noisy signals.")
    gen.add_argument("--n-realizations", type=int, default=100, help="Noisy signal realizations per case.")
    gen.add_argument("--s0", type=float, default=1.0, help="Baseline signal amplitude.")

    cmp_parser = sub.add_parser("patho-compare", help="Run MLP predictions and create GT/MLP/cov-fit plots.")
    cmp_parser.add_argument("--data-root", type=Path, default=None, help="Generated-data root.")
    cmp_parser.add_argument("--model-root", type=Path, default=None, help="External benchmark checkpoint root. Defaults to QTI_MODEL_ROOT.")
    cmp_parser.add_argument("--cov-fit-root", type=Path, default=None, help="Optional external covariance-fit root. Defaults to QTI_COV_FIT_ROOT.")
    cmp_parser.add_argument("--snr-folder", default="SNR30", help="SNR folder to compare, for example SNR30.")
    cmp_parser.add_argument("--output-root", type=Path, default=None, help="Comparison CSV/figure output directory.")
    cmp_parser.add_argument("--device", default="cpu", help="Torch device for inference.")

    run = sub.add_parser("patho-run", help="Generate patho data and then run MLP comparison.")
    run.add_argument("--data-root", type=Path, default=None, help="Generated-data root.")
    run.add_argument("--xps-path", type=Path, default=None, help="External XPS .mat protocol.")
    run.add_argument("--model-root", type=Path, default=None, help="External benchmark checkpoint root.")
    run.add_argument("--cov-fit-root", type=Path, default=None, help="Optional external covariance-fit root.")
    run.add_argument("--scenario", action="append", choices=available_scenarios(), help="Scenario to generate. Repeat to select multiple; omit for all.")
    run.add_argument("--winners-only", action="store_true", help="Use the checked-in per-scenario winner DTD JSONs instead of generating every case.")
    run.add_argument("--n-tensors", type=int, default=1500, help="Tensor count per generated DTD case; ignored with --winners-only.")
    run.add_argument("--seed", type=int, default=42, help="Base seed; ignored with --winners-only.")
    run.add_argument("--snr", type=_float_or_none, default=30.0, help="SNR value, or 'none' to skip noisy signals.")
    run.add_argument("--n-realizations", type=int, default=100, help="Noisy signal realizations per case.")
    run.add_argument("--s0", type=float, default=1.0, help="Baseline signal amplitude.")
    run.add_argument("--snr-folder", default="SNR30", help="SNR folder to compare, for example SNR30.")
    run.add_argument("--output-root", type=Path, default=None, help="Comparison CSV/figure output directory.")
    run.add_argument("--device", default="cpu", help="Torch device for inference.")

    val = sub.add_parser("validate-gt", help="Recompute GT from DTD JSONs and compare to stored GT JSONs.")
    val.add_argument("--data-root", type=Path, default=None, help="Generated-data root.")
    val.add_argument("--tolerance", type=float, default=1e-10, help="Absolute tolerance for scalar differences.")
    val.add_argument("--output-csv", type=Path, default=None, help="Optional validation CSV path.")

    pat = sub.add_parser("patient-compare", help="Rank generated exact patho signals against an external patient signal.")
    pat.add_argument("--patient-signal", type=Path, required=True, help="External 4D patient signal NIfTI path.")
    pat.add_argument("--data-root", type=Path, default=None, help="Generated-data root.")
    pat.add_argument("--target", default="Brain mean", help="Patient target: Brain mean, Deep CC, Partial CSF/WM, or Pure CSF.")
    pat.add_argument("--output-csv", type=Path, default=None, help="Optional patient ranking CSV path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv
        Optional argument list for tests. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "patho-generate":
        config = PathoRunConfig(
            data_root=resolve_data_root(args.data_root),
            xps_path=resolve_optional_path(args.xps_path, "QTI_XPS_PATH"),
            snr=args.snr,
            n_realizations=args.n_realizations,
            n_tensors=args.n_tensors,
            seed=args.seed,
            s0=args.s0,
        )
        generated = generate_patho_data(config, scenarios=args.scenario, winners_only=args.winners_only)
        manifest = write_generation_manifest(generated, config.data_root)
        print(f"Generated {len(generated)} patho cases under {config.data_root}")
        print(f"Manifest: {manifest}")
        return 0

    if args.command == "patho-compare":
        df = compare_patho_predictions(
            data_root=args.data_root,
            model_root=args.model_root,
            cov_fit_root=args.cov_fit_root,
            snr_folder=args.snr_folder,
            output_root=args.output_root,
            device=args.device,
        )
        print(f"Wrote comparison for {len(df)} signal realizations.")
        return 0

    if args.command == "patho-run":
        config = PathoRunConfig(
            data_root=resolve_data_root(args.data_root),
            xps_path=resolve_optional_path(args.xps_path, "QTI_XPS_PATH"),
            snr=args.snr,
            n_realizations=args.n_realizations,
            n_tensors=args.n_tensors,
            seed=args.seed,
            s0=args.s0,
        )
        generated = generate_patho_data(config, scenarios=args.scenario, winners_only=args.winners_only)
        manifest = write_generation_manifest(generated, config.data_root)
        print(f"Generated {len(generated)} patho cases under {config.data_root}")
        print(f"Manifest: {manifest}")
        df = compare_patho_predictions(
            data_root=config.data_root,
            model_root=args.model_root,
            cov_fit_root=args.cov_fit_root,
            snr_folder=args.snr_folder,
            output_root=args.output_root,
            device=args.device,
        )
        print(f"Wrote comparison for {len(df)} signal realizations.")
        return 0

    if args.command == "validate-gt":
        df = validate_gt(data_root=args.data_root, tolerance=args.tolerance)
        n_bad = int((~df["ok"].astype(bool)).sum()) if not df.empty else 0
        if args.output_csv is not None:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False)
            print(f"Validation CSV: {args.output_csv}")
        print(f"Validated {len(df)} scalar values; failures: {n_bad}")
        return 1 if n_bad else 0

    if args.command == "patient-compare":
        df = compare_patient_target(
            patient_signal_path=args.patient_signal,
            data_root=args.data_root,
            target=args.target,
            output_csv=args.output_csv,
        )
        if args.output_csv is not None:
            print(f"Patient ranking CSV: {args.output_csv}")
        print(df.head(10).to_string(index=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
