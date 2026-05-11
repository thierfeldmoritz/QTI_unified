# Workflow

The workflow is patho-only. Old sweep logic is intentionally absent.

## 1. Generate DTDs

`patho-generate` creates scenario/case DTD JSON files under:

```text
<QTI_DATA_ROOT>\DTDs_cov_suite_2_patho
```

Each DTD parameter dictionary stores eigenvalues in SI units and a unit
orientation vector. Scenarios live in `qti_unified.patho`.

```powershell
qti-unified patho-generate --scenario crossing_needles_2 --n-tensors 1500
```

## 2. Store GT Once

During generation, GT scalars are computed through `qti_unified.qti_math`,
which delegates the formulas to the vendored SynQTI-IR `utils.dtd_math`
functions. The output is written to:

```text
<QTI_DATA_ROOT>\Results_2_MLP_patho\<scenario>\<case>\*_GT_params.json
```

The stored GT payload includes `MD`, `MD_um2_per_ms`, `FA`, `uFA`, `C_MD`, and
`C_c`. Later prediction and plotting code reads this file and does not
recompute GT.

Use validation only when you intentionally want to recompute:

```powershell
qti-unified validate-gt --tolerance 1e-10
```

## 3. Write Signals

Generation writes exact and cumulant-expansion NIfTI signals next to the GT
JSON:

```text
<QTI_DATA_ROOT>\Results_2_MLP_patho\<scenario>\<case>\*__exact.nii
<QTI_DATA_ROOT>\Results_2_MLP_patho\<scenario>\<case>\*__cumexp.nii
```

If `--snr` is enabled, noisy realizations are written to:

```text
<QTI_DATA_ROOT>\Results_SNR_fit_2_MLP_patho\<scenario>\<case>\SNR30
```

Use `--snr none` to skip noisy files for a quick GT-only run.

## 4. Run MLP Prediction

`patho-compare` discovers generated noisy signals and applies external
benchmark checkpoints:

```powershell
qti-unified patho-compare --model-root C:\QTI_ML\BENCHMARK_ABSTRACT
```

The output table is:

```text
<QTI_DATA_ROOT>\runs\QTI_MLP_synthetic_compare_patho\patho_comparison.csv
```

Each row represents one noisy signal realization and contains stored GT values,
MLP predictions, and optional covariance-fit values.

## 5. Plot GT, MLP Mean/Std, And Cov Fit

Plots are written by scenario into:

```text
<QTI_DATA_ROOT>\runs\QTI_MLP_synthetic_compare_patho
```

For normal scenario sweeps, GT is drawn as a black dashed line and MLP as blue
mean/std markers. For single-case scenarios such as `needle_sphere_2`,
`spheres_2`, and `crossing_needles_3_xyz`, GT is drawn as a black star. If
covariance-fit data exists, it is shown as an orange mean/std series.

## 6. Optional Patient Comparison

Patient comparison ranks generated exact patho signals by RMSE to one external
patient target:

```powershell
qti-unified patient-compare --patient-signal D:\patients\P01\signal.nii.gz --target "Brain mean"
```

The patient image is read from its external path and is never copied into this
repo. This step is optional and is not required for synthetic generation or MLP
plots.

## Full Run

```powershell
qti-unified patho-run `
  --xps-path C:\SynQTI-IR\data\xps\xps_sub_min_pp.mat `
  --model-root C:\QTI_ML\BENCHMARK_ABSTRACT `
  --cov-fit-root C:\SynQTI-IR\data\Fit_Results\dtd_covariance_snr30_batch_sub_min_pp_patho `
  --snr 30 `
  --n-realizations 100
```
