# Data Dictionary and Scientific Scope

## Models and Conventions

The shared-band calculation holds the drift, endpoint ports, detection band, and
complete gain and loss operator-norm budgets fixed. The budget is
`0 <= G,L <= R I`, with `G-L=M`; it is not a bound on one jump coefficient or a
measured pump power. Matrix range `r` counts physical sites.

| Model | Parameters |
|---|---|
| Uniform scalar chain | `t_R=1`, `t_L=0.25`, `gamma=1.2`, `kappa_in=kappa_out=0.2`. |
| Dimerized scalar chain | Alternating `gamma=(6/5,23/20)`, `t_R=(23/20,17/20)`, `t_L=(3/20,7/20)`; `kappa_in=kappa_out=1/5`. The first site/bond uses index zero. |
| Pointwise two-orbital validation | `8,9,10` cells at detunings `-0.15,0,0.15`; onsite block `[[-1,-0.12i],[-0.12i,-0.9]]`, right/left blocks `diag(1.2,1.0)` / `diag(0.10,0.08)`. Ports of rate `0.2` couple to the first orbital at each end. This is separate from both the dimerized scalar chain and the boundary-certificate model. |
| Supplementary two-band chain | Fixed `J=0.01` boundary-certificate instance; parameters and port conventions are specified by the boundary scripts. |

`src/shared_band_resources.py` stores model parameters as exact fractions.
The noise observable is normally ordered input-referred excess photon noise.
Its conversion to symmetrized added noise follows the manuscript convention;
the fixed-drift differences are unchanged by that conversion.

## Core Datasets

| Dataset | Meaning |
|---|---|
| `results/local_noise_optimization_explicit.csv` | Explicit pointwise local construction and baseline comparison used in main Fig. 2. |
| `results/finite_frequency_local_optimality_scalar.csv` and `results/finite_frequency_local_optimality_multiband.csv` | Fig. S1 samples, ordered by size then detuning; SDP discrepancies and explicit-kernel errors are numerical residuals, not a physical locality penalty. |
| `results/rate_capped_band_optimization.csv` | 27 shared-band optimizations: `N=40,48,64`, `R=8,32,256`, and `r=1,2,all`, at halfwidth `0.05`. |
| `results/rate_capped_band_optimization_metadata.json` | Optimizer arguments, environment, and source provenance. |
| `results/rate_capped_band_optimization_certificate.json` | Exact checks and conservative optimum intervals for the saved finite-chain matrices. |
| `results/size_uniform_resources.csv` | 18 optimized prefixes: two models, halfwidths `0.03,0.05,0.07`, and three ranges. |
| `results/size_uniform_resources_metadata.json` | Exact model descriptions, arguments, and generator/helper hashes. |
| `results/size_uniform_resources_certificate.json` | Rational prefix witnesses, full-budget extension bounds, and all-length resource constants. |
| `results/size_uniform_resources_summary.csv` | Tabular `alpha`, `beta`, lower and upper bounds, locality gaps, and necessary rates. |
| `results/size_uniform_resources_audit.json` | Continuous halfwidth interval, stronger gain checks, and two explicit range-two constructions. |
| `results/final_prl_proof_audit.json` | Independent finite-example checks of conventions, local kernels, prefix identities, and bridge extensions. |
| `results/reports/experimental_realization_audit_2026_10_05.json` | Finite-auxiliary numerical predictions for the specified feasible devices, including intrinsic loss and thermal baths; no measured data or whole-band gain certificate. |

In CSV files, `matrix_file` is relative to the repository root, and
`matrix_sha256` is the hash of the NPZ archive where present. `halfwidth` defines
the symmetric band `[-halfwidth,halfwidth]`. `radius=all` removes the range
restriction. `lower` and `upper` bound an optimum; they are not error bars from
random measurement samples.

The all-length reference uses a reserved bridge rate. Its prefix gain/loss caps
are `6.75` (uniform) and `6.8` (dimerized), while the completed device cap is `8`.
The reference prefix has range at most `63`, independent of total chain length.
Do not read that reference as a nearest-neighbor device.

## Matrix Archives

`results/rate_capped_band_optimization_matrices/` contains the 27 finite-chain
archives listed by the CSV. `results/size_uniform_resources_matrices/` contains
18 prefix archives. `results/size_uniform_range2_matrices/` contains the two
extra prefix constructions used to establish the minimum-range targets.

Use `numpy.load(path, allow_pickle=False)` to inspect an archive. The exact list
of arrays is available as `archive.files`.

| Array family | Meaning |
|---|---|
| `gain`, `loss` | Saved real symmetric gain and loss rate matrices. The checker enforces the exact drift difference. |
| `mask` | Allowed matrix entries for the declared range. |
| `cap` | Complete rate cap of the optimization problem. |
| `primal_*_factor`, `primal_*_shift` | Factors and shifts certifying positivity of gain, loss, and both budget slacks. |
| `dual_factor_0` through `dual_factor_3` | Gram factors for the four PSD dual constraints. |

Every stored binary float is interpreted as its exact dyadic rational during
certification. Positive margins and residual penalties establish conservative
bounds without treating a floating-point solver status as a proof.

## Supplementary Data

`docs/file_catalog.json` lists the supplementary response and boundary datasets
with their calculation entry points. These include the physical local
realization, covariance and endpoint responses, spectral and port scans,
disorder samples, and the specified two-band boundary certificates.

The files `H_sign_variation_N24_80.csv` and `H_sign_variation_N81_120.csv` record
the inherited exact finite prefix. The interval-box CSV files cover the real
frequency interval and the complex contour; their JSON records contain source
and data hashes. The integrity verifier checks coverage and provenance. It
does not reconstruct the inherited CRT polynomials.

## Scope

S5 is a conditional local realization of a known pointwise optimum. Its rates
can depend on the selected frequency. The shared-band device uses one fixed
reservoir pair. The size-uniform numerical constants and range-two targets are
certified for the two specified Jacobi families and every finite `N>=64`.
The continuous halfwidth guarantee is `0.0495 <= Omega <= 0.0505`.
Other certified bandwidths are separate cases.

These data do not establish a universal sharp scaling law for arbitrary range,
arbitrary disorder, general multiorbital models, or interacting systems. The
specified two-band sign crossings have gain at most one and serve as
supplementary model evidence.
