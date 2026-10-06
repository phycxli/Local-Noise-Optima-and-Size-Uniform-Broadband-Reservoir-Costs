# Reproducibility

Run commands from the repository root in an environment installed from
`requirements.txt`. No API key, external data service, TeX installation, or
machine-specific directory is required for these calculations.

## Published Inputs and Working Outputs

The source bytes, CSV records, and matrix archives retain their original paths
because certificates refer to their SHA256 hashes. `scripts/reproduce.py`
checks the release manifest and makes a working copy of `src/`, `scripts/`,
`results/`, and `figures/` under the requested output directory. Calculations
and generated reports operate on that copy.

```bash
python scripts/reproduce.py manifest
python scripts/reproduce.py verify
python scripts/reproduce.py figures
python scripts/reproduce.py experiment
```

Each executed script has its own UTF-8 log in `reproduction/logs/`. The workflow
report records the command, return code, elapsed time, Python/dependency
versions, and generated-file hashes. A failed command stops the workflow with
a nonzero exit status and identifies its log.

## Verification Stages

| Script | What is checked |
|---|---|
| `audit_final_prl_proofs.py` | Independent monomial/Legendre Gram equality, S16 dual and prefix identities, S5 complex local kernels, and S17 exact bridge algebra on finite examples. |
| `check_rate_capped_band_optimization.py` | All 27 archived finite-chain primal/dual bounds, drift constraints, range masks, and complete rate caps. |
| `check_size_uniform_resources.py` | All 18 prefix certificates and six all-length constructions; rational tail bounds and necessary resource relations. |
| `audit_size_uniform_resources.py` | Continuous bandwidth bounds, direct extension checks, and both stored range-two constructions. |
| `check_fig4_resources.py` | Exact coverage and bound algebra on 160 bandwidth cells per model, fixed reference and range-two devices, and 24 independent direct-resolvent noise integrals. |
| `check_boundary_argument_certificate.py` | Source/data hashes and exact coverage of saved supplementary interval certificates, plus inherited finite-prefix metadata. |

The independent finite examples check algebra and conventions. The all-length
performance statements use the prefix witnesses and analytic tail bounds.
The supplementary integrity check reuses saved interval boxes; it does not
rerun every interval operation or reconstruct the inherited CRT prefix.

The base size-uniform workflow covers two models, three bandwidths, and three
ranges (18 prefix problems at length 64). The construction audit checks direct
resolvents and complete matrix spectra at lengths 64, 65, 96, and 128; its
24 response/noise checks and six range-two matrix checks are floating-point
cross-checks. `strengthen_fig4_resources.py` stores 81 nearest-neighbor
witnesses per model on halfwidths 0.03 through 0.07, uses unchanged reference
and range-two prefixes designed at 0.05, and adds 48 floating-point extension
checks. The exact interval certificates and their separate audit are in
`results/fig4_shared_device_certificate.json` and
`results/reports/fig4_strengthening_scientific_audit.json`.

## Regenerate the Figures

```bash
python scripts/reproduce.py figures
```

Outputs are four main PDF/PNG figure pairs and Fig. S1 under
`reproduction/figures/`. Fig. S1 uses saved CSVs through `--plot-only`.
The figure script cross-checks spectral averages against certified means and
requires separated lower/upper bounds for the range-two targets. Numerical
figure checks are recorded in
`reproduction/results/reports/physical_narrative_figure_audit.json`.

## Finite-Auxiliary Numerical Comparison

```bash
python scripts/reproduce.py experiment
```

`audit_experimental_realization.py` uses the archived `N=64`, `R=8`,
`Omega=0.05` reservoirs with baseline matrix admixture `epsilon=0.005`,
intrinsic energy loss `eta=0.01`, auxiliary energy linewidth `kappa_b=50`,
and internal bath occupation `n_b=0.001`. It checks both complete rate
matrices, augmented drift stability, equality of the effective responses,
and full-versus-eliminated response/noise at three frequencies. It integrates
1601 frequencies and compares the 801-point subset. The output is
`reproduction/results/reports/experimental_realization_audit_2026_10_05.json`.
The power-gain minimum is sampled. This workflow reports finite-device
numerical predictions, without an all-length finite-memory certificate.

## Rerun the Optimizer

```bash
python scripts/reproduce.py optimize --output-dir reproduction-optimization
```

This runs the finite-chain optimizer and checker, the prefix optimizer and
checker, the audit with `--solve-local-construction`, and main plotting.
It also runs `strengthen_fig4_resources.py` and `check_fig4_resources.py`.
It covers 27 finite-chain problems, 18 prefix problems, two extra range-two
constructions, and the continuous bandwidth witnesses. Runtime is substantially
longer than verification.
Each script writes new metadata and certificate hashes inside the working copy.

The finite-chain and dense nearest-neighbor programs use CVXPY and Clarabel
with absolute, relative, and feasibility tolerances `2e-10`, at most 300
iterations, and one solver thread. Candidate primal matrices and dual factors
are subsequently checked using exact rational products and residual corrections.
Solver tolerances and a successful status do not certify a bound. The current
script sources and archived hashes record the settings used for each workflow.

Binary optimizer matrices need not match between systems. Accept a recomputed
result only after its rational checker succeeds and the strict comparisons
remain positive. Published digits refer to the original archived matrices.

## Supplementary Calculations

After any workflow has created the working copy, run a supplementary script
from that copy, for example:

```bash
python reproduction/scripts/run_local_bosonic_reservoir.py
python reproduction/scripts/run_finite_frequency_local_optimality.py
python reproduction/scripts/run_uniform_rate_bound.py
python reproduction/scripts/run_disordered_transfer_cocycle.py
```

The corresponding data and plots are written to `reproduction/results/` and
`reproduction/figures/`. The complete supplementary script list is classified
in `docs/file_catalog.json`; script source and `--help`, where supported,
specify the scanned parameters.

To rerun the outward-rounded interval tail and response prefactor:

```bash
python reproduction/scripts/certify_boundary_evans_all_N.py --part both
python reproduction/scripts/certify_boundary_response_prefactor.py
python reproduction/scripts/check_boundary_argument_certificate.py
```

The exact finite prefix can be reconstructed using
`analyze_H_sign_variation.py` and its modular helpers. Use its explicit size
arguments and output path to avoid conflating a new reconstruction with the
saved prefix. Larger exact reconstructions can be expensive.

## Troubleshooting

| Observation | Action |
|---|---|
| Release hash mismatch | Start from an unchanged `v1.2.0` checkout; run new calculations in a working output directory. |
| Missing Python package | Check the active interpreter and install `requirements.txt` into that environment. |
| Optimizer cannot produce positive certificate margins | Inspect the solver log and certify the new output; a solver status alone is insufficient. |
| Missing/stale base certificate | Run the relevant checker before the extension audit. |
| A supplementary reconstruction is slow | Verify the saved certificate first; use a small explicit size range when checking the reconstruction code. |
