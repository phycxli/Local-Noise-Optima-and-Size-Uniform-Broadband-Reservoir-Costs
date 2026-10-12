"""Accept scan coverage, quantify tolerance and calibrate hardware resources."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from physics import CONFIG, ROOT, parameters, matrices, grams
from prepare import solve


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    rows = [r for p in sorted((ROOT / "results/shards").glob("scan_*.csv")) for r in read_csv(p)]
    assert len(rows) == 58320
    keys = {(r["case_id"], r["family"], r["N"], r["halfwidth"], r["occupation"]) for r in rows}
    assert len(keys) == len(rows)
    expected = {(str(i), family, str(n), str(width), str(nb)) for i in range(972)
                for family, n in [("finite", 64), ("extended", 64), ("extended", 96), ("extended", 128)]
                for width in CONFIG["band_grid"] for nb in CONFIG["occupation_grid"]}
    assert keys == expected
    assert all(r["same_response"] == "True" for r in rows)
    assert max(float(r["maximum_complete_rate"]) for r in rows) < 8
    assert max(float(r["augmented_spectral_abscissa"]) for r in rows) < 0
    scan = {"rows": len(rows), "minimum_feasible_gap": min(float(r["feasible_device_gap"]) for r in rows),
            "maximum_complete_rate": max(float(r["maximum_complete_rate"]) for r in rows),
            "maximum_augmented_spectral_abscissa": max(float(r["augmented_spectral_abscissa"]) for r in rows),
            "minimum_sampled_gain": min(float(r["minimum_sampled_power_gain"]) for r in rows),
            "rows_below_unity_gain": sum(float(r["minimum_sampled_power_gain"]) < 1 for r in rows),
            "all_feasible_gaps_positive": all(float(r["feasible_device_gap"]) > 0 for r in rows)}
    cfg = CONFIG["certificate_box"]
    cells = []
    for path in sorted((ROOT / "results/box_shards").glob("box_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["configuration"] == cfg
        assert d["all_gap_bounds_positive"] and d["stability_sufficient_bound"]["sufficient_for_all_finite_N"]
        cells.extend(d["cells"])
    count = cfg["parameter_subdivisions"]**3 * cfg["linewidth_subdivisions"] * cfg["intrinsic_subdivisions"]
    assert len(cells) == count and {c["cell_id"] for c in cells} == set(range(count))
    gap, energy, gain = min(c["band_gap_lower"] for c in cells), max(c["band_profile_energy_upper"] for c in cells), min(c["whole_band_power_gain_lower"] for c in cells)
    noise_error, measurement_each = .001, .05
    remaining = gap - energy * noise_error - 2 * measurement_each
    assert gap > 0 and gain > 1 and remaining > 0
    box = {"cells": len(cells), "frequency_cells_per_box": round(.07 / cfg["frequency_step"]),
           "gap_lower": gap, "profile_energy_upper": energy, "whole_band_gain_lower": gain,
           "whole_band_gain_lower_dB": float(10 * np.log10(gain)),
           "force_noise_norm_error_sum": noise_error, "measurement_error_each": measurement_each,
           "remaining_measured_gap_lower": remaining,
           "maximum_combined_force_error_after_measurement": (gap - 2 * measurement_each) / energy,
           "analytical_complete_rate_upper": 7.8 + .02 + 2 * 1.2 * .005 + 2 * 1.25 * .005,
           "configuration": cfg,
           "claim": "Continuous finite N=64 separation of two feasible fixed devices, with common calibration",
           "not_claimed": "Continuous optimality separation or finite-linewidth all-N noise separation"}
    nominal = (1.2, 1., .25)
    high_gain_cells = [c for c in cells if all(
        c["box"][j][0] >= v * (1 - .00125) - 1e-12 and c["box"][j][1] <= v * (1 + .00125) + 1e-12
        for j, v in enumerate(nominal))]
    assert len(high_gain_cells) == 4096
    high_gap = min(c["band_gap_lower"] for c in high_gain_cells)
    high_energy = max(c["band_profile_energy_upper"] for c in high_gain_cells)
    high_gain = min(c["whole_band_power_gain_lower"] for c in high_gain_cells)
    box["high_gain_subregion"] = {"relative_parameter_halfspan": .00125, "cells": len(high_gain_cells),
                                  "same_other_parameter_intervals": True, "gap_lower": high_gap,
                                  "profile_energy_upper": high_energy, "whole_band_gain_lower": high_gain,
                                  "whole_band_gain_lower_dB": float(10 * np.log10(high_gain)),
                                  "remaining_measured_gap_lower": high_gap - .001 * high_energy - .1}
    adaptive = read_csv(ROOT / "results/physical_adaptive_optimality_scan.csv")
    assert len(adaptive) == 648 and all(float(r["optimality_gap_lower_float"]) > 0 for r in adaptive)
    for r in adaptive:
        for file_key, sha_key in (("reference_file", "reference_sha256"), ("local_witness_file", "local_witness_sha256")):
            assert hashlib.sha256((ROOT / r[file_key]).read_bytes()).hexdigest() == r[sha_key]
    optimality = {"rows": len(adaptive), "minimum_gap_lower_float": min(float(r["optimality_gap_lower_float"]) for r in adaptive),
                  "maximum_primal_dual_gap": max(float(r["primal_dual_gap"]) for r in adaptive),
                  "per_model_minima": {name: min(float(r["optimality_gap_lower_float"]) for r in adaptive if r["model"] == name) for name in ("uniform", "dimerized")},
                  "scope": "N=64 parameter grid, kappa_b=80/100, eta=0/.02, independent gamma/right/left factors -0.5%/0/+0.5%, Omega=.045/.05/.055",
                  "reference_design_halfwidth": .05, "shared_at_all_frequencies_and_three_bandwidths": True,
                  "retuned_only_after_physical_parameter_calibration": True}
    p, linewidth, eta, occupation = parameters("uniform"), 80., .01, .001
    w, z, *_ = grams(p, 64, .05, linewidth, eta, 193)
    source = matrices(p, 64)[1]
    resources = []
    for radius, cap in ((1, 8.), (2, 7.8)):
        arrays, record = solve(z, source, cap, radius, eta)
        path = ROOT / f"results/baseline_hardware_r{radius}.npz"
        np.savez_compressed(path, **arrays)
        gain, loss = arrays["gain"], arrays["loss"]
        ug = np.sqrt(linewidth) / 2 * np.linalg.cholesky(gain)
        ul = np.sqrt(linewidth) / 2 * np.linalg.cholesky(loss - eta * np.eye(64))
        distance = abs(np.arange(64)[:, None] - np.arange(64))
        assert max(np.max(abs(ug[distance > radius])), np.max(abs(ul[distance > radius]))) < 1e-10
        for u in (ug, ul):
            for column in u.T:
                support = np.flatnonzero(abs(column) > 1e-10)
                assert not len(support) or support[-1] - support[0] <= radius
        thermal_noise = ((1 + 2 * occupation) * np.sum(z * gain.T)
                         - occupation * np.sum(z * (source + eta * np.eye(64)).T) + occupation * eta * np.trace(w))
        resources.append({"radius": radius, "noise": float(thermal_noise),
                          "complete_gain_max_eigenvalue_MHz": float(np.linalg.eigvalsh(gain).max()),
                          "complete_loss_max_eigenvalue_MHz": float(np.linalg.eigvalsh(loss).max()),
                          "programmed_loss_min_eigenvalue_MHz": float(np.linalg.eigvalsh(loss - eta * np.eye(64)).min()),
                          "maximum_coupling_per_addressed_mode_MHz": float(max(abs(ug).max(), abs(ul).max())),
                          "coupling_matrix_operator_norm_MHz": float(max(np.linalg.norm(ug, 2), np.linalg.norm(ul, 2))),
                          "gain_auxiliaries": 64, "loss_auxiliaries": 64,
                          "total_addressed_mode_couplings": int(np.count_nonzero(abs(ug) > 1e-10) + np.count_nonzero(abs(ul) > 1e-10)),
                          "maximum_modes_per_auxiliary": radius + 1,
                          "trace_gain_MHz": float(np.trace(gain)), "trace_complete_loss_MHz": float(np.trace(loss)),
                          "squared_coupling_sum_MHz2": float(np.sum(abs(ug)**2) + np.sum(abs(ul)**2)),
                          "matrix_file": path.relative_to(ROOT).as_posix()})
    markov = json.loads((ROOT / "results/markov_error_certificate.json").read_text(encoding="utf-8"))
    independent = json.loads((ROOT / "results/independent_verification.json").read_text(encoding="utf-8"))
    assert independent["all_checks_passed"] and markov["all_remaining_gaps_positive"]
    report = {"parameter_scan": scan, "continuous_box_certificate": box, "adaptive_physical_optimality_grid": optimality,
              "frozen_reference_limitations": json.loads((ROOT / "results/physical_optimality_summary.json").read_text(encoding="utf-8")),
              "baseline_hardware_settings": {"N": 64, "linewidth_MHz": linewidth, "intrinsic_MHz": eta, "halfwidth_MHz": .05,
                                             "bath_occupation": occupation, "J0_over_2pi_MHz": 1, "ports_MHz": [.2, .2]},
              "baseline_hardware_resources": resources,
              "matched_trace_budget_audit": json.loads((ROOT / "results/matched_trace_budget_summary.json").read_text(encoding="utf-8")),
              "markov_all_length_error_certificates": [{k: v for k, v in r.items() if not k.endswith("_exact")} for r in markov["rows"]],
              "all_acceptance_checks_passed": True}
    (ROOT / "results/summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
