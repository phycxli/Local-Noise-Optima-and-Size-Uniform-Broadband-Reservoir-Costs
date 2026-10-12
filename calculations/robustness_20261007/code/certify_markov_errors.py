"""Exact all-length force-noise tolerance at each of the two nominal drifts."""

from fractions import Fraction as F
import json
import sys

import numpy as np

from physics import ROOT
from shared_band_resources import (MODELS, exact_gram, floquet_envelope, gram_entry,
                                   legendre_coefficients, source_matrix, tail_bound)


def exact_array(a):
    return np.array([F.from_float(float(x)) for x in a.flat], dtype=object).reshape(a.shape)


def downward(x):
    return float(np.nextafter(float(x), -np.inf))


def upward(x):
    return float(np.nextafter(float(x), np.inf))


def main():
    sys.set_int_max_str_digits(0)
    base = json.loads((ROOT / "inputs/previous_size_uniform_certificate.json").read_text(encoding="utf-8"))
    results = []
    for name, model in MODELS.items():
        coefficients = legendre_coefficients(model, 160, F(1, 20))
        gram = exact_gram(model, coefficients, 64)
        with np.load(ROOT / f"inputs/{name}_extended_r2.npz", allow_pickle=False) as archive:
            loss = exact_array(archive["loss"])
        gain = loss + source_matrix(model, 64)
        prefix_upper = sum((gram[i, j] * gain[j, i] for i in range(64) for j in range(64)), F(0))
        rates = [model.right[j] + model.left[j] for j in range(2)]
        upper = prefix_upper + sum((rates[j % 2] / 2 * (
            gram_entry(model, coefficients, j, j) + 2 * gram_entry(model, coefficients, j, j + 1)
            + gram_entry(model, coefficients, j + 1, j + 1)) for j in range(63, 158)), F(0))
        envelope = floquet_envelope(model, F(1, 20))
        upper += sum(rates) * tail_bound(model, envelope, 158)
        energy = sum((gram_entry(model, coefficients, j, j) for j in range(158)), F(0))
        energy += tail_bound(model, envelope, 158)
        group = next(g for g in base["groups"] if g["model"] == name and g["halfwidth"] == .05)
        witness = next(w for w in group["locality_bounds"] if w["radius"] == 1)
        lower = F(witness["alpha_exact"]) - 8 * F(witness["beta_with_boundary_exact"])
        gap = lower - upper
        noise_error, measurement_each = F(1, 2000), F(1, 20)
        residual = gap - noise_error * energy - 2 * measurement_each
        cap = F(13, 2) + rates[1]
        assert cap < 8 and gap > 0
        row = {"model": name, "N_minimum": 64, "halfwidth": .05, "complete_cap": 8,
               "reference_range": 2, "conservative_reference_complete_cap": float(cap),
               "local_optimum_lower": downward(lower), "reference_upper": upward(upper),
               "optimality_gap_lower": downward(gap), "all_length_profile_energy_upper": upward(energy),
               "force_noise_norm_error_sum": float(noise_error), "measurement_error_each": float(measurement_each),
               "remaining_gap_lower": downward(residual),
               "maximum_combined_force_error_after_measurement": downward((gap - 2 * measurement_each) / energy),
               "reference_upper_exact": str(upper), "energy_upper_exact": str(energy),
               "gap_lower_exact": str(gap), "remaining_gap_exact": str(residual)}
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("_exact")}), flush=True)
    report = {"scope": "Nominal Markov drifts only, every finite N>=64, Omega=.05, fixed ports and cap 8",
              "proof": "For common measured response, |Tr(W_N Delta Q)| <= ||Delta Q|| Tr(W_N) <= delta*C; subtract the sum of two device noise errors and measurement errors",
              "assumptions": ["Matched drift/ports and fixed shared reservoirs", "PSD and complete rate feasibility independently retained",
                              "The error changes the force-noise matrix; no arbitrary drift or gain error is included",
                              "Normally ordered input-referred band noise; errors are spectral norms, not total pump power"],
              "rows": results, "all_remaining_gaps_positive": all(r["remaining_gap_lower"] > 0 for r in results)}
    (ROOT / "results/markov_error_certificate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
