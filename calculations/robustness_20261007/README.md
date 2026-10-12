# Finite-auxiliary robustness, error tolerances and resource calibration

This calculation accompanies *The noise cost of reservoir locality in broadband quantum amplifier chains*.
All supplied data are theoretical calculations and experimental design predictions. No experimental measurement is reported.

## Files and units

The entry point is `calculations/robustness_20261007/`. The `code/` directory contains unchanged calculation sources; `inputs/` contains the frozen seed reservoirs and prior mathematical certificates. `config.json` supplies the scan and interval settings. `requirements.txt` lists the numerical dependencies.

Rate matrices, angular frequencies and couplings in the calculations are dimensionless, normalized by the angular-frequency scale `J0`. A saved dimensionless value `x` corresponds to the physical angular quantity `J0*x`; its laboratory frequency value is `J0*x/(2*pi)`. The chosen conversion is `J0/(2*pi) = 1 MHz`. Thus `kappa_b = 80` means `J0*kappa_b/(2*pi) = 80 MHz`, and `Omega = 0.05` means `J0*Omega/(2*pi) = 50 kHz`. Both dimensionless passive port rates are fixed at `0.2`, corresponding to `0.2 MHz` after this conversion. Complete internal loss includes intrinsic loss; passive port rates are fixed separately. Noise values and measurement errors are in quanta, and bath occupation is dimensionless.

Three archives contain **all final scientific results**, including the unsuccessful frozen-reference comparison:

| Archive | Contents after extraction |
| --- | --- |
| `results_data.zip` | Complete scan shards, continuous-box cells, summaries, exact Markov constants, independent checks, baseline hardware matrices and all trace-budget matrices |
| `results_adaptive_witnesses.zip` | 216 calibrated range-two references and 648 nearest-neighbor lower-bound matrix witnesses |
| `results_optimality_witnesses.zip` | All 648 matrix witnesses for the frozen nominal-reference comparison |

Extract all three archives **into this directory**. Their entries already start with `results/`; do not add a second `results/` directory. No shard, certificate cell or retained matrix file is omitted. Duplicate transport tarballs, original job logs, Python caches, administrative synchronization receipts and the machine-specific SSH synchronization utility are excluded.

`SOURCE_SHA256.json` gives the original SHA256 and byte size of every scientific source file included in this release. `FILE_MANIFEST_SHA256.json` records the staged files and every original member in each archive. It excludes itself to avoid a circular checksum. The historical provenance in `inputs/provenance.json` refers to the original calculation run; it does not claim that later manuscript files have the same hashes.

## What each result tests

### 58,320 feasible-device comparisons

`results/shards/scan_*.csv` contains the full grid: two drift models, independent relative changes of `-1%, 0, +1%` in three drift parameters, six auxiliary linewidths `20, 35, 50, 65, 80, 100`, and three intrinsic-loss settings `0, 0.01, 0.02`. These give 972 physical settings. For each setting the calculation uses the finite family at `N=64` and the extended family at `N=64,96,128`, five band halfwidths `0.03,0.04,0.05,0.06,0.07`, and three bath occupations `0,0.001,0.02`.

The count is `972 * 4 * 5 * 3 = 58,320`. The 864 samples whose minimum sampled power gain is below one remain in the saved CSVs, in the feasibility tests, and in every noise-difference sign check. They are not treated as successful high-gain amplifiers. The positive feasible-device differences across this grid are numerical comparisons of two specified devices; they are neither continuous certificates nor optimality claims for every possible reservoir. The minimum saved feasible-device gap is `0.30891347395360924` quanta.

### 648 calibrated optimality comparisons

`results/physical_adaptive_optimality_scan.csv` contains `216 * 3 = 648` comparisons at `N=64`: two models, linewidths `80,100`, intrinsic loss `0,0.02`, and independent drift variations `-0.5%,0,+0.5%`. Each of the 216 calibrated devices uses one fixed range-two reference designed at halfwidth `0.05`, tested at halfwidths `0.045,0.05,0.055`. Retuning after physical parameter calibration is allowed; retuning separately at each frequency is not.

Each saved difference is a nearest-neighbor lower bound minus the noise of a verified feasible range-two reference. The minimum is `0.49473896307271037` quanta. The references need not be globally optimal. These are parameter-grid results, not a proof over a continuous region or over all chain lengths. Matrix filenames and SHA256 values in the CSV identify every reference and lower-bound witness.

### Frozen-reference limitation

`results/physical_optimality_scan.csv` tests the same number of physical settings and bands with the nominal reference frozen during calibration. Only 274 of the 648 saved differences are positive; the minimum is `-1.7408382822078616`. This limitation is retained. It must not be combined with the adaptive-reference result or described as uniform robustness to independent pump errors.

### 20 matched-resource comparisons

`results/matched_trace_budget.csv` contains two nominal models, dimensionless linewidths `kappa_b=80,100`, and common complete trace caps `tau=175,190,210,250,325`: `2 * 2 * 5 = 20` cases at `N=64`, with spectral cap `R=8`, intrinsic loss `0.01`, occupation `0.001` and halfwidth `0.05`. The laboratory linewidths `J0*kappa_b/(2*pi)` are `80,100 MHz`, the trace caps `J0*tau/(2*pi)` are `175,190,210,250,325 MHz`, and the spectral cap `J0*R/(2*pi)` is `8 MHz`. Every certified difference is positive; the minimum is `0.5719147532405344` quanta.

The comparison matches both complete spectral norms and the trace of gain plus complete loss. With dimensionless rate matrices, coupling matrices and auxiliary linewidth, the coupling-strength relation is

```text
Gamma_g = 4 U_g U_g^dagger / kappa_b
Gamma_l = D + 4 U_l U_l^dagger / kappa_b
sum |U_g|^2 + sum |U_l|^2
    = kappa_b * [Tr(Gamma_g) + Tr(Gamma_l) - N*eta] / 4.
```

This is a coupling-strength proxy. It does not equate a spectral cap with an individual coupling cap, and it does not predict pump power in watts. The range-two device is used through its verified feasible upper bound; no claim requires the range-two optimization to have converged to its global optimum.

### 262,144 continuous parameter cells

`results/box_shards/box_*.json` contains all final fine-grid cells for the uniform model at `N=64`. Three independently varied drift parameters have relative halfspan `0.005` and 32 subdivisions each; linewidth `80..100` has four subdivisions; intrinsic loss `0..0.02` has two. The count is `32^3 * 4 * 2 = 262,144`. Bath occupation ranges over `0..0.02` and band halfwidth over `0.03..0.07`; continuous frequency enclosure uses 280 cells up to `0.07`, each of width `0.00025`.

The comparison is between two specified feasible finite-auxiliary devices after common calibration. The broad box certifies a difference above `0.4196154451004667` quanta and whole-band power gain above `3.3357788722674995` (`5.23197 dB`). The nested high-gain subregion has relative drift halfspan `0.00125`, contains exactly 4,096 of these cells, and certifies a difference above `0.48477120728176426` quanta and whole-band gain above `16.74507788177962` (`12.23887 dB`). These are continuous finite-device separation bounds. They do not certify continuous optimality against every nearest-neighbor reservoir, and they do not establish finite-linewidth separation for every `N`.

For matched responses and ports, the resolvable difference satisfies `Delta_res >= Delta_0 - C_Q*delta - 2u`, conditional on retained positivity and complete rate feasibility. Here dimensionless `delta` bounds the sum of the force-noise matrix norm errors and `u` is the measurement error per device. At `delta=0.001`, the laboratory error scale is `J0*delta/(2*pi) = 1 kHz`. With `u=0.05` quanta, the broad and high-gain regions retain bounds of `0.20435722457209501` and `0.2927104583242429` quanta. This is not an independent pump-percentage error bound.

### Nominal Markov results for every finite N >= 64

`results/markov_error_certificate.json` preserves exact rational constants for the two specified nominal Markov drift models at halfwidth `0.05` and complete cap `R=8`. Compatible boundary responses and a controlled extension make the bounds apply to every finite `N>=64`. These are distinct from the finite-linewidth `N=64` results.

The conservative difference bounds before force-noise and measurement errors are `0.824774505221434` for the uniform model and `0.1751128021821854` for the dimerized model. With measurement error `u=0.05`, the combined force-noise error ceilings `J0*delta/(2*pi)` from these witnesses are approximately `8.693612 kHz` and `0.998181 kHz`. A negative perturbed lower bound means that this witness no longer certifies a positive resolvable difference; it does not show a reversal of the physical noise advantage.

## Restore and verify the released files

From the repository root:

```bash
cd calculations/robustness_20261007
python verify_release_files.py
python -m zipfile -e results_data.zip .
python -m zipfile -e results_adaptive_witnesses.zip .
python -m zipfile -e results_optimality_witnesses.zip .
python verify_release_files.py --extracted-results
```

The file verifier uses only the Python standard library, reads files without changing them, and validates both archive membership and every SHA256 value. Successful full verification prints `all_checks_passed: true`. `--available` explicitly permits missing result archives for the small supplementary attachment and reports `full_inventory_verified: false` whenever an archive is absent. A partial attachment must not be described as the complete matrix inventory.

The small journal attachment `SupplementaryData.zip` includes the code, frozen inputs, configuration, manifests and `results_data.zip`. Retrieve the two full matrix-witness archives from the [public calculation directory, release v1.3.0](https://github.com/phycxli/Local-Noise-Optima-and-Size-Uniform-Broadband-Reservoir-Costs/tree/v1.3.0/calculations/robustness_20261007). The same manifests verify both delivery routes. Obtain those archives to reproduce the 648-case summaries and frozen-reference comparison in full.

## Recheck and recompute in a working copy

Use a separate working copy after checksum verification. The scientific scripts write their outputs: `verify.py` rewrites `results/independent_verification.json`, and `summarize.py` also resolves the two baseline hardware problems and rewrites their NPZ files. Solver-dependent regenerated bytes should not be used to overwrite the checksum reference release.

Python 3.13 was used for the original local checks, with NumPy `2.2.6`, SciPy `1.16.2`, CVXPY `1.9.3`, Clarabel `0.11.1`, mpmath `1.3.0` and Matplotlib `3.10.7`. The long scans used Python `3.13.11`, NumPy `2.1.3` and SciPy `1.16.0`. The exact positive-semidefinite factor checks, direct auxiliary model, 70-digit response calculations and independent quadrature checks test consistency across those paths.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python code/verify.py
python code/summarize.py
```

On Windows PowerShell use `.\.venv\Scripts\Activate.ps1` and set the thread variables with `$env:OPENBLAS_NUM_THREADS='1'`, `$env:OMP_NUM_THREADS='1'` and `$env:MKL_NUM_THREADS='1'`.

To recompute the full campaign in that working copy:

```bash
python code/prepare.py
for i in $(seq 0 11); do python code/run_scan.py --shard "$i" --shards 12; done
for i in $(seq 0 7); do python code/certify_box.py --shard "$i" --shards 8; done
python code/certify_markov_errors.py
python code/optimality_scan.py
python code/adaptive_optimality_scan.py
python code/trace_budget_scan.py
python code/verify.py
python code/summarize.py
```

Each shard is independent and can be scheduled on one CPU with one BLAS thread. The continuous-cell and semidefinite scans are the expensive stages; the saved complete results allow file verification before running them. Keep the frozen inputs for the prior boundary witnesses and the prior size-uniform certificate: `prepare.py` can regenerate the new seed reservoirs, but the supplied earlier certificates are also inputs to the all-length checks. No external machine account is required by the calculation commands above.
