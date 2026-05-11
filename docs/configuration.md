# Configuration

This repo treats all large files as external. The package can run with command
line paths, environment variables, or the example TOML file as human
documentation. The current CLI reads command line flags first, then environment
variables, then a small local default for generated data.

## Generated Data Root

`QTI_DATA_ROOT` is the only path with a default. If unset, generated data goes
to:

```text
C:\QTI_Unified\data
```

That folder is gitignored. You can also choose a faster scratch disk:

```powershell
$env:QTI_DATA_ROOT = "D:\qti_patho_generated"
qti-unified patho-generate --scenario crossing_needles_2
```

## Acquisition Protocol

`QTI_XPS_PATH` points to the external XPS `.mat` protocol. Real MLP comparisons
should use the same protocol used during model training.

```powershell
$env:QTI_XPS_PATH = "C:\SynQTI-IR\data\xps\xps_sub_min_pp.mat"
qti-unified patho-generate --xps-path $env:QTI_XPS_PATH
```

When no XPS path is supplied, the code uses a deterministic built-in protocol
for smoke tests only.

## Model Checkpoints

`QTI_MODEL_ROOT` points to external benchmark MLP checkpoints and matching
`*_zscore.csv` files.

```powershell
$env:QTI_MODEL_ROOT = "C:\QTI_ML\BENCHMARK_ABSTRACT"
qti-unified patho-compare --model-root $env:QTI_MODEL_ROOT
```

The repo never copies checkpoints into git. Checkpoints are selected by the
benchmark naming pattern used by the original MLP code.

## Covariance-Fit Outputs

`QTI_COV_FIT_ROOT` is optional. It should contain covariance-fit `dps.mat`
files in a scenario/case/SNR layout.

```powershell
$env:QTI_COV_FIT_ROOT = "C:\SynQTI-IR\data\Fit_Results\dtd_covariance_snr30_batch_sub_min_pp_patho"
qti-unified patho-compare --cov-fit-root $env:QTI_COV_FIT_ROOT
```

If a covariance-fit file is missing, plotting simply omits the orange series
for that case.

## Patient Data

Patient data is never bundled. For patient comparison, pass the exact external
signal path:

```powershell
qti-unified patient-compare --patient-signal D:\patients\P01\signal.nii.gz --target "Deep CC"
```

`QTI_PATIENT_ROOT` is included in the examples for local organization, but the
CLI requires concrete patient input paths so nobody accidentally assumes patient
files live in this repo.

## Config Examples

`.env.example` is useful for shell setup. `qti_unified.toml.example` documents
the same paths and default generation values for a lab handoff. The package is
intentionally simple and does not require a TOML parser at runtime.

## Gitignore Policy

These locations and file types are ignored:

- `data/`, `runs/`, `figures/`, `patient_data/`, `external/`
- NIfTI and MAT files
- Torch checkpoints
- z-score CSV files
- covariance-fit output folders

Only code, tests, docs, and small examples belong in git.
