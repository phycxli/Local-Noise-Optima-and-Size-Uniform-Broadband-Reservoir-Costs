# Data and code availability

The numerical data, optimized reservoir matrices, primal and dual factors,
continuous-band and parameter-region certificates, and Python calculation
code supporting this study are available in the project repository,
version 1.3.0:

https://github.com/phycxli/Local-Noise-Optima-and-Size-Uniform-Broadband-Reservoir-Costs/tree/v1.3.0

The baseline results are in `results/`. The additional finite-auxiliary
calculations, calibrated-device comparisons, trace-budget tests and
noise-tolerance code are in `calculations/robustness_20261007/`, with an
English guide and three result archives. Together those archives contain
every retained scientific result, including comparisons that failed to
show a noise advantage for the frozen reference. Their file verifier
checks the complete archive inventory and SHA256 values without
recomputing the science.

The balanced single-frequency construction used in the current Fig. 2 is
in `results/balanced_pointwise/`. Commands for the current result figures
are in `docs/CURRENT_RESULTS.md`. Dependency versions and commands for
verification and full recomputation are supplied separately, since new
solver runs can produce different matrix bytes within numerical tolerance.

These data are theoretical calculations and device-design predictions.
No experimental measurements were made. The all-length bounds concern
the two specified nominal Markov chains; the calibrated finite-auxiliary
noise comparisons concern N=64. A parameter scan, a continuous two-device
certificate and a nearest-neighbor optimality bound have distinct scopes,
as documented in the additional guide.

`release_manifest.json` gives file-level SHA256 values. The additional
calculation's `FILE_MANIFEST_SHA256.json` also records every archive member.
