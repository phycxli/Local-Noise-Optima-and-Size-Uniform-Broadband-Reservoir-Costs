"""Calibrate at each physical setting, then freeze a range-two device across the band."""

from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import time

import numpy as np

from physics import ROOT, grams, matrices, noises, parameters
from prepare import solve


def evaluate(item):
    index, name, changes, linewidth, eta = item
    p = parameters(name, changes)
    source = matrices(p, 64)[1]
    w0, z0, *_ = grams(p, 64, .05, linewidth, eta, 193)
    reference, reference_record = solve(z0, source, 7.8, 2, eta)
    path = ROOT / "results/adaptive_witnesses" / f"reference_{index:03d}.npz"
    np.savez_compressed(path, **reference)
    reference_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for width_index, width in enumerate((.045, .05, .055)):
        w, z, *_ = grams(p, 64, width, linewidth, eta, 193)
        local, _ = solve(z, source, 8., 1, eta)
        a, b, c, d = [local[f"dual_factor_{j}"] @ local[f"dual_factor_{j}"].T for j in range(4)]
        e = z - a - b + c + d
        penalty = np.maximum(-np.diag(e), 0).sum() + abs(np.diag(e, 1)).sum()
        shifted = source + eta * np.eye(64)
        f_lower = np.sum((d - b) * shifted.T) - (8 - eta) * np.trace(c) - 8 * np.trace(d) - (8 - eta) * penalty
        lower_cold = float(f_lower + np.sum(z * shifted.T) - 1e-7)
        upper_cold = float(np.sum(z * local["gain"].T))
        assert upper_cold - lower_cold < 1e-4
        witness_path = ROOT / "results/adaptive_witnesses" / f"local_{index:03d}_{width_index}.npz"
        np.savez_compressed(witness_path, **local)
        occupation = .001
        offset = -occupation * np.sum(z * shifted.T) + occupation * eta * np.trace(w)
        lower = (1 + 2 * occupation) * lower_cold + offset
        value = noises(reference["gain"], reference["loss"], w, z, eta, occupation)
        row = {"index": index, "model": name, "delta_gamma": changes[0], "delta_right": changes[1],
               "delta_left": changes[2], "linewidth": linewidth, "intrinsic": eta, "halfwidth": width,
               "occupation": occupation, "N": 64, "local_cold_lower": lower_cold, "local_cold_upper": upper_cold,
               "local_thermal_lower": float(lower), "reference_r2_noise": value,
               "optimality_gap_lower_float": float(lower - value), "primal_dual_gap": upper_cold - lower_cold,
               "reference_complete_rate": reference_record["complete_rate"],
               "reference_design_halfwidth": .05, "same_reference_across_three_bandwidths": True,
               "reference_file": path.relative_to(ROOT).as_posix(), "reference_sha256": reference_hash,
               "local_witness_file": witness_path.relative_to(ROOT).as_posix(),
               "local_witness_sha256": hashlib.sha256(witness_path.read_bytes()).hexdigest()}
        rows.append(row)
    return rows


def main():
    (ROOT / "results/adaptive_witnesses").mkdir(parents=True, exist_ok=True)
    axes = itertools.product(("uniform", "dimerized"), itertools.product((-.005, 0., .005), repeat=3),
                             (80., 100.), (0., .02))
    cases = [(i, *case) for i, case in enumerate(axes)]
    start, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=6) as executor:
        for group in executor.map(evaluate, cases, chunksize=2):
            rows.extend(group)
    with (ROOT / "results/physical_adaptive_optimality_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"scope": "N=64 grid of calibrated physical parameters; one range-two device per setting, fixed throughout and across the three tested bands",
              "not_claimed": "Continuous optimality box or all-length finite-auxiliary optimality theorem",
              "rows": len(rows), "distinct_fixed_reference_devices": len(cases),
              "minimum_gap_lower_float": min(r["optimality_gap_lower_float"] for r in rows),
              "positive_gap_cases": sum(r["optimality_gap_lower_float"] > 0 for r in rows),
              "maximum_primal_dual_gap": max(r["primal_dual_gap"] for r in rows),
              "seconds": time.perf_counter() - start}
    (ROOT / "results/physical_adaptive_optimality_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
