# Reference Data

This folder contains small, checked-in reference artifacts that are safe to keep
in git. It is separate from `data/`, which is gitignored and used for generated
outputs.

## `DTDs_cov_suite_2_patho_winners`

This directory contains one winning DTD JSON file per supported patho scenario,
copied from:

```text
C:\SynQTI-IR\data\DTDs_cov_suite_2_patho_winners
```

Use these winner DTDs in the normal pipeline with:

```powershell
qti-unified patho-generate --winners-only
qti-unified patho-run --winners-only
```

`--scenario` can still be used to choose a subset. For example:

```powershell
qti-unified patho-generate --winners-only --scenario crossing_needles_2
```

The winner JSON files are inputs only. Generated signals, GT JSONs, noisy SNR
realizations, comparison CSVs, and figures are still written under
`QTI_DATA_ROOT`.
