"""Check a proposed microwave implementation using the saved N=64 reservoirs.

These are finite-device numerical predictions, not new optimality certificates
or experimental data. All frequencies and rates below are in units of J0.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import eigh, eigvals, solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import MODELS, drift_matrix, normalized_profile, source_matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def psd_factor(matrix):
    values = eigh(matrix, eigvals_only=True)
    assert values[0] > -1e-9, values[0]
    factor = np.linalg.cholesky(matrix)
    assert np.max(np.abs(factor @ factor.conj().T - matrix)) < 1e-10
    return factor


def main():
    n, width, cap = 64, .05, 8.
    linewidth, intrinsic, mixing, occupation = 50., .01, .005, .001
    model = MODELS["uniform"]
    drift = drift_matrix(model, n)
    source = np.asarray(source_matrix(model, n), dtype=float)
    coherent = (drift - drift.T) / 2
    ports = np.zeros((n, n))
    ports[0, 0] = float(model.kappa_in)
    ports[-1, -1] = float(model.kappa_out)
    fixed_loss = intrinsic * np.eye(n)
    baseline_gain = np.zeros((n, n))
    baseline_loss = 2.4 * np.eye(n)
    for j in range(n - 1):
        baseline_gain[j:j+2, j:j+2] += .625 * np.array([[1, 1], [1, 1]])
        baseline_loss[j:j+2, j:j+2] += .625 * np.array([[1, -1], [-1, 1]])
    assert np.max(np.abs(baseline_gain - baseline_loss - source)) < 1e-14

    frequencies = np.linspace(-width, width, 1601)
    alpha = 1 / (1 + 2j * frequencies / linewidth)
    effective = (coherent - (fixed_loss + ports) / 2)[None, :, :] + (
        alpha[:, None, None] * (source + fixed_loss)[None, :, :] / 2)
    resolvent = 1j * frequencies[:, None, None] * np.eye(n) - effective
    output = np.zeros((len(frequencies), n, 1), dtype=complex)
    output[:, -1, 0] = 1
    weights = np.linalg.solve(resolvent.conj().transpose(0, 2, 1), output)[:, :, 0]
    profiles = weights / (weights[:, :1] * np.sqrt(float(model.kappa_in)))
    power_gain = float(model.kappa_in * model.kappa_out) * abs(weights[:, 0])**2
    ideal_profiles = np.array([normalized_profile(model, n, w) for w in frequencies])
    certificate_path = ROOT / "results/rate_capped_band_optimization_certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    results, source_hashes = [], {str(certificate_path.relative_to(ROOT)): sha(certificate_path)}
    zero = np.zeros_like(drift, dtype=complex)

    for radius in ("1", "2", "all"):
        record = next(row for row in certificate["rows"] if row["N"] == n
                      and row["rate_cap"] == cap and row["radius"] == radius)
        path = ROOT / record["matrix_file"]
        assert sha(path) == record["matrix_sha256"]
        source_hashes[record["matrix_file"]] = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            original_loss = archive["loss"]
        original_gain = original_loss + source
        loss = (1 - mixing) * original_loss + mixing * baseline_loss
        gain = loss + source
        programmed_loss = loss - fixed_loss
        rates = {"gain": [float(x) for x in eigh(gain, eigvals_only=True)[[0, -1]]],
                 "complete_loss": [float(x) for x in eigh(loss, eigvals_only=True)[[0, -1]]],
                 "programmed_loss": [float(x) for x in eigh(programmed_loss, eigvals_only=True)[[0, -1]]]}
        assert min(values[0] for values in rates.values()) > 0
        assert max(rates["gain"][-1], rates["complete_loss"][-1]) < cap
        assert np.max(np.abs(gain - loss - source)) < 1e-14
        if radius != "all":
            distance = abs(np.arange(n)[:, None] - np.arange(n))
            assert np.max(np.abs(gain[distance > int(radius)])) < 1e-14
            assert np.max(np.abs(loss[distance > int(radius)])) < 1e-14

        loss_coupling = np.sqrt(linewidth) / 2 * psd_factor(programmed_loss)
        gain_coupling = np.sqrt(linewidth) / 2 * psd_factor(gain)
        if radius != "all":
            assert max(np.max(abs(matrix[distance > int(radius)]))
                       for matrix in (loss_coupling, gain_coupling)) < 1e-12
        auxiliary = -linewidth / 2 * np.eye(n)
        augmented = np.block([
            [coherent - (fixed_loss + ports) / 2, -1j * loss_coupling, -1j * gain_coupling],
            [-1j * loss_coupling.conj().T, auxiliary, zero],
            [1j * gain_coupling.conj().T, zero, auxiliary]])
        stability = float(np.max(eigvals(augmented).real))
        assert stability < -.1
        reduction_residual, noise_residual = 0., 0.
        creation_input = np.vstack((zero, zero, np.sqrt(linewidth) * np.eye(n)))
        for index in (0, len(frequencies) // 2, len(frequencies) - 1):
            w = frequencies[index]
            block = 1j * w * np.eye(3 * n) - augmented
            reduced_output = solve(block.conj().T, np.eye(3 * n)[:, n - 1])
            reduction_residual = max(reduction_residual, float(
                np.max(abs(reduced_output[:n] - weights[index])) / np.max(abs(weights[index]))))
            full_noise = float(np.linalg.norm(creation_input.conj().T @ reduced_output)**2)
            reduced_noise = float((abs(alpha[index])**2 *
                (weights[index].conj() @ gain @ weights[index])).real)
            noise_residual = max(noise_residual, abs(full_noise - reduced_noise) / full_noise)
        assert max(reduction_residual, noise_residual) < 1e-10

        quadratic = lambda matrix, p: np.einsum("fi,ij,fj->f", p.conj(), matrix, p).real
        ideal_noise = quadratic(original_gain, ideal_profiles)
        cold_noise = abs(alpha)**2 * quadratic(gain, profiles)
        warm_noise = (cold_noise + occupation * (abs(alpha)**2 *
                     quadratic(gain + programmed_loss, profiles) + quadratic(fixed_loss, profiles)))
        means = {name: float(simpson(values, x=frequencies) / (2 * width))
                 for name, values in (("ideal_saved", ideal_noise),
                                      ("finite_auxiliary_cold", cold_noise),
                                      ("finite_auxiliary_thermal", warm_noise))}
        errors = {name: abs(means[name] - float(simpson(values[::2], x=frequencies[::2]) / (2 * width)))
                  for name, values in (("ideal_saved", ideal_noise),
                                       ("finite_auxiliary_cold", cold_noise),
                                       ("finite_auxiliary_thermal", warm_noise))}
        assert max(errors.values()) < 1e-8
        assert record["reported_lower"] - 1e-8 <= means["ideal_saved"] <= record["reported_upper"] + 1e-8
        results.append({"radius": radius, "matrix_file": record["matrix_file"],
                        "complete_rate_extrema": rates, "band_average_noise": means,
                        "maximum_single_endpoint_coupling_MHz": float(max(
                            np.max(abs(loss_coupling)), np.max(abs(gain_coupling)))),
                        "factorization": "Lower Cholesky columns; local columns have support diameter at most the specified range",
                        "quadrature_step_halving_difference": errors,
                        "augmented_spectral_abscissa": stability,
                        "full_vs_eliminated_response_relative_residual": reduction_residual,
                        "full_vs_eliminated_cold_noise_relative_residual": noise_residual})

    one, two = results[:2]
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Proposed N=64 finite-linewidth implementation; numerical predictions, no measured data or optimality/all-length certificate",
              "source_sha256": sha(Path(__file__)), "input_sha256": source_hashes,
              "units": "J0/(2*pi)=1 MHz; all listed rates are angular rates divided by J0",
              "settings": {"N": n, "band_halfwidth": width, "complete_rate_cap": cap,
                           "auxiliary_energy_linewidth": linewidth, "intrinsic_energy_loss": intrinsic,
                           "baseline_mixing_fraction": mixing, "all_internal_bath_occupations": occupation,
                           "passive_port_occupations": 0., "frequency_grid_points": len(frequencies)},
              "baseline_bond_single_endpoint_coupling_MHz": float(np.sqrt(.625 * linewidth / 4)),
              "finite_auxiliary_min_band_power_gain": float(min(power_gain)),
              "finite_auxiliary_min_band_gain_dB": float(10 * np.log10(min(power_gain))),
              "same_effective_frequency_response_for_all_configurations": True,
              "rows": results,
              "range_one_minus_two": {name: one["band_average_noise"][name] - two["band_average_noise"][name]
                                      for name in one["band_average_noise"]},
              "all_checks_passed": True}
    target = ROOT / "results/reports/experimental_realization_audit_2026_10_05.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
