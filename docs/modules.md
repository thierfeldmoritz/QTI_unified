# Modules

This page is a handoff map for new contributors. Public functions and classes
also have docstrings with parameter, return, shape, unit, and side-effect
details.

## `qti_unified.config`

Configuration and small dataclasses.

- `resolve_data_root`: resolves `--data-root`, `QTI_DATA_ROOT`, or local
  gitignored `data/`.
- `resolve_optional_path`: resolves optional external paths from CLI args or
  environment variables.
- `PathoCase`: one generated DTD scenario/case before signal synthesis.
- `GeneratedPathoCase`: generated case plus GT, signals, and written paths.
- `PathoRunConfig`: runtime options shared by generation commands.

Use this module when adding new CLI options or path policy logic.

## `qti_unified.patho`

Patho scenario definitions and DTD parameter generation.

- `available_scenarios`: lists supported patho scenario names.
- `generate_patho_case`: creates one `PathoCase`.
- `generate_patho_suite`: creates all cases for selected scenarios.

The module owns case naming. If a filename should remain notebook-compatible,
change it here first and add a test.

## `qti_unified.qti_math`

Minimal vendored QTI tensor, signal, and GT math.

- Tensor conversion helpers use Voigt notation with SI units.
- `params_to_dtens` converts DTD parameter dictionaries to 3x3 tensors.
- `qti_params_from_dtd` computes first and second cumulant tensors.
- `gt_scalars_from_params` computes the stored GT scalar payload.
- `dti_signal` and `qti_cumulant_signal` synthesize exact and cumulant signals.
- `read_xps_mat` and `xps_to_bt` load external acquisition protocols.

This is the only place GT formulas should live. Prediction and plotting code
must read GT JSON files instead of recomputing metrics.

## `qti_unified.signals`

Signal synthesis and notebook-compatible writing.

- `load_btens`: loads external XPS b-tensors or the built-in smoke-test
  protocol.
- `patho_output_paths`: constructs all output folders for one scenario/case.
- `write_case_outputs`: writes DTD JSON, GT JSON, metadata, exact/cumexp NIfTI,
  and optional noisy SNR realizations.

Use this module for generated-data layout changes. Do not add patient,
checkpoint, or covariance-fit assumptions here.

## `qti_unified.mlp`

Prediction-time adapter for external benchmark MLP checkpoints.

- `QtiMlp`: MLP architecture matching the benchmark checkpoints.
- `collect_benchmark_model_paths`: selects canonical fold checkpoints from an
  external model root.
- `prepare_signal_inputs`: loads NIfTI signals, clamps negatives, z-scores, and
  flattens inputs.
- `ensemble_predict`: averages physical-scale predictions over checkpoints.

This module never trains and never stores checkpoints.

## `qti_unified.plotting`

Comparison table and figure creation.

- `load_gt_values`: reads stored GT JSON values.
- `covariance_fit_values`: reads optional external covariance-fit `dps.mat`
  values.
- `parse_case_for_plotting`: converts scenario/case names to x-axis values.
- `build_comparison_table`: combines case rows, MLP predictions, GT, and cov
  values into a DataFrame.
- `plot_scenario_comparisons`: saves scenario PNGs with GT, MLP mean/std, and
  optional covariance-fit overlays.

Single-case scenarios use star GT markers. Missing covariance fits are allowed.

## `qti_unified.patient`

Optional patient-target comparison helpers.

- `load_patient_signal`: loads an external 4D patient signal NIfTI.
- `normalize_signal`: normalizes signals by the first measurement.
- `extract_target_signal`: returns a named voxel target or a brain mean target.
- `rank_simulated_cases`: ranks generated simulations by RMSE to the target.

No patient data is stored here. Commands must receive an external patient path.

## `qti_unified.invivo`

In-vivo patho winner plot export, ported from
`C:\SynQTI-IR\SynDTDs_MLP_patho.ipynb` cell 52.

- `InvivoPathoPlotter`: loads external patient signal, reduced/full XPS, mask,
  and generated exact patho signals, then computes scenario winners.
- `export_invivo_patho_winner_plots`: convenience wrapper that writes one
  `*_winner_all4.png` per scenario plus `best_fit_summary_patho.csv`.
- `short_scenario_name`: keeps output names compatible with the original
  notebook export.

The module keeps patient NIfTI, mask, and XPS files external. Reference derived
PNGs are documented in [In-Vivo Patho Comparison PNGs](invivo_compare_patho.md).

## `qti_unified.pipeline`

End-to-end orchestration used by the CLI.

- `generate_patho_data`: generates patho cases and writes all synthetic outputs.
- `discover_prediction_cases`: discovers generated noisy signal files and stored
  GT JSONs.
- `compare_patho_predictions`: runs external MLP checkpoints and writes tables
  and plots.
- `validate_gt`: recomputes GT for audit only.
- `compare_patient_target`: ranks generated exact signals against an external
  patient target.
- `write_generation_manifest`: writes a small manifest for generated cases.

Use this module when adding a workflow step that combines multiple lower-level
modules.

## `qti_unified.cli`

Argparse entry point exposed as `qti-unified`.

Subcommands:

- `patho-generate`
- `patho-compare`
- `patho-run`
- `patient-compare`
- `validate-gt`

Keep CLI behavior thin. Heavy behavior belongs in `pipeline.py` so tests can
call it directly.
