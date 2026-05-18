# Patho Winner DTDs

This folder mirrors the curated winner-only DTD set from:

```text
C:\SynQTI-IR\data\DTDs_cov_suite_2_patho_winners
```

Each scenario subfolder contains exactly one JSON file. The JSON file is a DTD
parameter list with eigenvalues in SI units and unit orientation vectors. These
files can be used by the full synthetic workflow through `--winners-only`.

Current winners:

```text
crossing_needles_2/crossing_needles_2_angle84.json
crossing_needles_3_xyz/crossing_needles_3_xyz.json
demyelination_progressive_sticks/demyelination_progressive_sticks_end0.30.json
fanning_sticks_1500/fanning_sticks_1500_span90.json
frac_gauss_2/Frac_0.50.json
iso_bimod_gauss_2/iso_bimod0.80_V0.13.json
iso_uni_gauss_2/iso_uni1.00.json
needle_sphere_2/needle_sphere_2tensors.json
o_prog_gauss_2/ODI_0.72.json
packed_crossings_conserved_2/crossing_angle90_frac50_50.json
partial_volume_triplet/partial_volume_triplet_fiso0.40.json
spheres_2/spheres_csf_1500.json
spheres_sizevar_2/spheres_sizevar_1500_w0.00.json
sticks_2/stick_1500_dperp_0.33.json
```

Example:

```powershell
qti-unified patho-generate --winners-only --xps-path $env:QTI_XPS_PATH --snr 30
```

With `--winners-only`, `--n-tensors` and `--seed` do not alter these checked-in
DTD files. They only affect newly generated scenario suites.
