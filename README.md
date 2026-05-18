# QTI Unified Patho Workflow

`qti-unified` is a small, standalone repo for the current patho-only QTI
synthetic workflow. It replaces the old split between DTD generation notebooks
and plotting notebooks with one documented package and one CLI.

The numerical/sampling code is vendored from the original projects:
`utils.dtd_math`, `utils.dtd_utils`, and `qti_phantom.py` come from
`C:\SynQTI-IR`, and the prediction adapter uses the original `QTI_MLP.py`
implementation from `C:\QTI_ML`.

The repo does not store large data. Generated signals, patient data,
covariance-fit outputs, and model checkpoints are all external paths and are
ignored by git.

A small curated set of winning patho DTD JSON files is included under
`reference_data/DTDs_cov_suite_2_patho_winners`. These are the per-scenario
winners copied from `C:\SynQTI-IR\data\DTDs_cov_suite_2_patho_winners`; generated
signals and patient-derived inputs remain external. Use `--winners-only` with
`patho-generate` or `patho-run` to run the normal pipeline on only those
checked-in winner DTDs.

## What This Repo Does

- Generates patho DTD cases and notebook-compatible output folders.
- Computes GT scalar metrics once during generation and stores them in
  `*_GT_params.json`.
- Writes exact DTD signals, cumulant-expansion signals, and optional noisy SNR
  realizations.
- Runs external benchmark MLP checkpoints on generated noisy signals.
- Plots GT, MLP mean/std, and optional covariance-fit overlays.
- Can run the same generation and comparison pipeline on only the checked-in
  winner DTDs with `--winners-only`.
- Optionally compares generated exact signals with an external patient target.

## Install

From `C:\QTI_Unified`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

If your environment already has the scientific stack and Torch installed, this
editable install is enough to expose the `qti-unified` command.

## External Paths

Copy the examples and edit them for your machine:

```powershell
Copy-Item .env.example .env
Copy-Item qti_unified.toml.example qti_unified.toml
```

The CLI reads environment variables directly:

- `QTI_DATA_ROOT`: local generated-data root, for example `C:\QTI_Unified\data`.
- `QTI_XPS_PATH`: optional external XPS `.mat` protocol.
- `QTI_MODEL_ROOT`: external benchmark MLP checkpoint directory.
- `QTI_COV_FIT_ROOT`: optional external covariance-fit result directory.
- `QTI_PATIENT_ROOT`: optional external patient root for local bookkeeping.

Patient scans must be supplied explicitly to `patient-compare`; no patient data
is bundled here.

## Quickstart

Generate a tiny smoke-test subset using the built-in protocol:

```powershell
qti-unified patho-generate --scenario needle_sphere_2 --n-tensors 16 --n-realizations 2
qti-unified validate-gt
```

Run a real generation with your acquisition protocol:

```powershell
qti-unified patho-generate --xps-path $env:QTI_XPS_PATH --n-tensors 1500 --snr 30 --n-realizations 100
```

Generate only the checked-in per-scenario winners:

```powershell
qti-unified patho-generate --winners-only --xps-path $env:QTI_XPS_PATH --snr 30 --n-realizations 100
```

You can combine `--winners-only` with `--scenario` to generate only selected
winner cases:

```powershell
qti-unified patho-generate --winners-only --scenario crossing_needles_2 --scenario sticks_2
```

Run MLP comparison and plots:

```powershell
qti-unified patho-compare --model-root $env:QTI_MODEL_ROOT --cov-fit-root $env:QTI_COV_FIT_ROOT
```

Run everything in one command:

```powershell
qti-unified patho-run --xps-path $env:QTI_XPS_PATH --model-root $env:QTI_MODEL_ROOT --cov-fit-root $env:QTI_COV_FIT_ROOT
```

Add `--winners-only` to `patho-run` to generate and compare only the checked-in
winner DTDs.

Compare to an external patient signal:

```powershell
qti-unified patient-compare --patient-signal D:\patients\P01\signal.nii.gz --target "Brain mean"
```

## Output Layout

Generated files keep the old notebook-compatible names, but only for patho
cases:

```text
<QTI_DATA_ROOT>/
  DTDs_cov_suite_2_patho/
  Results_2_MLP_patho/
  Results_SNR_fit_2_MLP_patho/
  runs/QTI_MLP_synthetic_compare_patho/
```

Every generated case has a `*_GT_params.json` file with:

- `MD_um2_per_ms`
- `MD`
- `FA`
- `uFA`
- `C_MD`
- `C_c`

Prediction and plotting commands read this stored GT JSON. They do not
recompute GT. Use `qti-unified validate-gt` when you intentionally want to
recompute and compare.

## Winner DTDs

The checked-in winner DTDs live in:

```text
reference_data/DTDs_cov_suite_2_patho_winners/
```

There is one JSON file per supported patho scenario. These files are small DTD
parameter lists, not NIfTI signals. When you pass `--winners-only`,
`patho-generate` loads these JSONs, computes GT, synthesizes exact/cumexp
signals, and writes the normal output folders under `QTI_DATA_ROOT`.

Important details:

- `--scenario` still filters scenarios; with `--winners-only`, it selects the
  winner JSON for each requested scenario.
- `--n-tensors` and `--seed` apply to generated scenario suites, but they do not
  change checked-in winner JSON files.
- `patho-compare`, `validate-gt`, and `patient-compare` do not need a special
  winner mode. They read the outputs produced by `patho-generate`.
- External XPS protocols, MLP checkpoints, covariance fits, patient data, and
  generated NIfTI signals remain outside git.

## Documentation

- [Configuration](docs/configuration.md) explains path resolution and examples.
- [Workflow](docs/workflow.md) explains generation, GT, SNR, MLP, plots, and
  patient comparison.
- [Modules](docs/modules.md) explains each code module, main functions, inputs,
  outputs, and side effects.
- [In-vivo patho comparison PNGs](docs/invivo_compare_patho.md) shows the
  included derived winner plots and how to recreate them from external patient
  data.

## Troubleshooting

- `No benchmark checkpoints found`: set `QTI_MODEL_ROOT` or pass
  `--model-root` to the folder containing checkpoint files and matching
  `*_zscore.csv` files.
- `Stored GT JSON is missing`: run `patho-generate` before `patho-compare`.
- Missing covariance fits are allowed. Plots are still produced without the
  orange covariance-fit series.
- Use `--snr none` for generation tests that should skip noisy signal files.
- The built-in acquisition protocol exists only for smoke tests. For real
  comparisons, pass the XPS protocol used by the MLP.
