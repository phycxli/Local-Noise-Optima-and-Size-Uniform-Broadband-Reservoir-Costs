"""Independent physical nearest-neighbor optima inside the proposed hardware window."""

from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

from physics import ROOT, device, grams, matrices, noises, parameters
from prepare import solve


def evaluate(item):
    index, name, changes, linewidth, eta, width = item
    p = parameters(name, changes)
    w, z, *_ = grams(p, 65, width, linewidth, eta, 193)
    source = matrices(p, 64)[1]
    archive, solver_record = solve(z[:64, :64], source, 8., 1, eta)
    factors = [archive[f"dual_factor_{j}"] for j in range(4)]
    a, b, c, d = [f @ f.T for f in factors]
    e = (z[:64, :64] - a - b + c + d)
    penalty = np.maximum(-np.diag(e), 0).sum() + abs(np.diag(e, 1)).sum()
    shifted = source + eta * np.eye(64)
    f_lower = np.sum((d - b) * shifted.T) - (8 - eta) * np.trace(c) - 8 * np.trace(d) - (8 - eta) * penalty
    cold_lower = f_lower + np.sum(z[:64, :64] * shifted.T) - 1e-7
    cold_upper = np.sum(z[:64, :64] * archive["gain"].T)
    assert cold_upper - cold_lower < 1e-4, (index, cold_lower, cold_upper)
    path = ROOT / "results/optimality_witnesses" / f"witness_{index:04d}.npz"
    np.savez_compressed(path, **archive)
    occupation = .001
    offset = -occupation * np.sum(z[:64, :64] * shifted.T) + occupation * eta * np.trace(w[:64, :64])
    lower = (1 + 2 * occupation) * cold_lower + offset
    g, loss = device(p, name, "finite", 2, 64, eta)
    reference = noises(g, loss, w[:64, :64], z[:64, :64], eta, occupation)
    return {"index": index, "model": name, "delta_gamma": changes[0], "delta_right": changes[1],
            "delta_left": changes[2], "linewidth": linewidth, "intrinsic": eta, "halfwidth": width,
            "occupation": occupation, "N": 64, "local_cold_lower": cold_lower, "local_cold_upper": float(cold_upper),
            "local_thermal_lower": float(lower), "reference_r2_noise": reference,
            "optimality_gap_lower_float": float(lower - reference), "primal_dual_gap": float(cold_upper - cold_lower),
            "witness_file": path.relative_to(ROOT).as_posix(), "witness_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    (ROOT / "results/optimality_witnesses").mkdir(parents=True, exist_ok=True)
    axes = itertools.product(("uniform", "dimerized"), itertools.product((-.005, 0., .005), repeat=3),
                             (80., 100.), (0., .02), (.045, .05, .055))
    cases = [(i, *case) for i, case in enumerate(axes)]
    start, results = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=6) as executor:
        for row in executor.map(evaluate, cases, chunksize=4):
            results.append(row)
            if len(results) % 54 == 0:
                print(json.dumps({"completed": len(results), "of": len(cases), "seconds": time.perf_counter() - start}), flush=True)
    with (ROOT / "results/physical_optimality_scan.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = {"scope": "Finite N=64 physical nearest-neighbor optimization; parameter grid, not a continuous-box or all-length proof",
              "rows": len(results), "minimum_gap_lower_float": min(r["optimality_gap_lower_float"] for r in results),
              "maximum_primal_dual_gap": max(r["primal_dual_gap"] for r in results),
              "positive_gap_cases": sum(r["optimality_gap_lower_float"] > 0 for r in results),
              "seconds": time.perf_counter() - start,
              "bank": "Saved primal matrices and four PSD dual factors for every case"}
    (ROOT / "results/physical_optimality_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
