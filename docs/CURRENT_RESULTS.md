# Current manuscript figures and additional data

Release v1.3.0 contains the revised figures and the complete scientific
results for the finite-auxiliary calculations. All results are model
predictions; no experimental measurements were made.

## Balanced single-frequency reservoirs

The files in `results/balanced_pointwise/` contain the construction used
in Fig. 2: `pointwise.csv`, `frequency.csv`, `reservoir.npz` and `audit.json`.
The kernel uses geometric balancing, as in the manuscript. The strength is
selected numerically from a generalized eigenvalue of the Schur complement,
with a small positive feasibility margin. This is a numerical evaluation
of the uncapped construction, not a certified global minimum rate.

At N=60 and zero detuning, the input-referred normal-ordered noise is
0.9946863 quanta, compared with 197.6579531 for the original bond reservoirs.
The full gain and loss matrix norms are 134.6238 and 139.5204 in J0 units.
These rates are not matched to the R=8 shared-band comparison. At detuning
0.05, the same frozen local reservoir gives 15.2603 quanta and departs from
the single-frequency floor.

To recompute these results, use a separate working copy of the checkout:

```bash
python scripts/build_balanced_fig2.py
python scripts/plot_current_results.py
```

The second command regenerates Figs. 2-4 from the saved data. Its rendering
functions are copied verbatim from the manuscript figure generator.
Fig. 4(c) shows nearest-neighbor lower bounds and range-two upper bounds
relative to the model-specific noise targets. A negative range-two bound
and a positive nearest-neighbor bound certify minimum range two. The
marked endpoints are the extent certified by these bounds, not measured
or exact physical transition points.

Fig. 1 is a proposed planar circuit schematic. The PDF and SVG are vector
assets; the diagram does not represent an assembled or measured device.

## Finite auxiliary modes and robustness

Read `calculations/robustness_20261007/README.md`, extract the result
archives and verify the scientific file manifest. That guide separates
the 20 nominal comparisons with both spectral and trace budgets, the
648 calibrated grid comparisons with spectral budgets, the continuous
two-device region, and the nominal all-length Markov error bounds.

After extraction, the supplementary figure generator reads those saved
results directly:

```bash
python scripts/plot_supplement_figures.py
```

Supplementary Fig. 2(b) includes both auxiliary linewidths. Its curves
almost coincide because their differences are below 5e-5 quanta, not
because one linewidth has been omitted. Panel (c) includes both the broad
continuous region and its high-gain subregion. The nominal dimerized
all-length tolerance witness ceases to certify a resolvable gap above
about 0.998 kHz; this does not imply that the actual noise ordering reverses.

## Earlier verification records

The existing `docs/VERIFICATION.json` records actual runs of v1.2.0.
Those records remain historical evidence for calculations that are
unchanged. They are not labelled as new v1.3.0 runs. The new archive
manifest verifies the added scientific payload. Full recomputation
commands and dependency versions are provided in the additional guide.
