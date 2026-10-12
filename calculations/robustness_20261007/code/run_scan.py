"""Slurm array scan. Reservoirs are fixed over each band and calibrated identically."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import time

import numpy as np

from physics import (CONFIG, MODELS, ROOT, device, grams, local_lower, noises,
                     parameters, response, stability)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    (ROOT / "results/shards").mkdir(parents=True, exist_ok=True)
    cases = list(itertools.product(MODELS, itertools.product(CONFIG["relative_parameter_grid"], repeat=3),
                                   CONFIG["linewidth_grid"], CONFIG["intrinsic_grid"]))
    selected = [(i, item) for i, item in enumerate(cases) if i % args.shards == args.shard]
    if args.limit:
        selected = selected[:args.limit]
    start, count = time.perf_counter(), 0
    filename = ROOT / f"results/shards/scan_{args.shard:02d}.csv"
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for case_id, (name, changes, linewidth, intrinsic) in selected:
            p = parameters(name, changes)
            for family, n in [("finite", 64)] + [("extended", n) for n in CONFIG["sizes"]]:
                pairs = [device(p, name, family, r, n, intrinsic) for r in (1, 2)]
                extrema = [[np.linalg.eigvalsh(x)[[0, -1]] for x in pair] for pair in pairs]
                rate = max(v[-1] for pair in extrema for v in pair)
                minimum = min(v[0] - (intrinsic if j == 1 else 0)
                              for pair in extrema for j, v in enumerate(pair))
                assert rate <= 8. + 1e-10 and minimum > -1e-10, (rate, minimum)
                abscissa = stability(p, n, linewidth, intrinsic)
                for width in CONFIG["band_grid"]:
                    # One extra site is needed to evaluate the all-length prefix boundary.
                    w, z, _, profiles, alpha, weights = grams(p, max(n, 65), width, linewidth, intrinsic)
                    wg, zg = w[:n, :n], z[:n, :n]
                    check_frequencies = np.r_[np.linspace(-width, width, 41), width * np.polynomial.legendre.leggauss(97)[0]]
                    gain_min = float(min(response(p, n, check_frequencies, linewidth, intrinsic)[2]))
                    d = pairs[0][0] - pairs[1][0]
                    gap_cold = float(np.sum(zg * d.T))
                    expected_d = device(parameters(name), name, family, 1, n, 0.)[0] - device(parameters(name), name, family, 2, n, 0.)[0]
                    difference_residual = float(np.max(abs(d - expected_d)))
                    assert difference_residual < 2e-14
                    for occupation in CONFIG["occupation_grid"]:
                        values = [noises(g, loss, wg, zg, intrinsic, occupation) for g, loss in pairs]
                        lower = local_lower(p, z, w, intrinsic, occupation, name, n > 64)
                        gap = values[0] - values[1]
                        assert abs(gap - (1 + 2 * occupation) * gap_cold) < 2e-10 * (1 + abs(gap))
                        row = {"case_id": case_id, "model": name, "family": family, "N": n,
                               "delta_gamma": changes[0], "delta_right": changes[1], "delta_left": changes[2],
                               "linewidth": linewidth, "intrinsic": intrinsic, "occupation": occupation,
                               "halfwidth": width, "noise_r1": values[0], "noise_r2": values[1],
                               "feasible_device_gap": gap, "local_optimum_lower_float": lower,
                               "lower_minus_r2": lower - values[1], "minimum_sampled_power_gain": gain_min,
                               "maximum_complete_rate": float(rate), "minimum_programmed_psd_eigenvalue": float(minimum),
                               "augmented_spectral_abscissa": abscissa,
                               "same_response": True, "gap_identity_residual": abs(gap - (1 + 2 * occupation) * gap_cold)}
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        count += 1
            handle.flush()
            print(json.dumps({"shard": args.shard, "case": case_id, "rows": count,
                              "seconds": round(time.perf_counter() - start, 2)}), flush=True)
    report = {"shard": args.shard, "shards": args.shards, "selected_cases": len(selected), "rows": count,
              "seconds": time.perf_counter() - start, "python": platform.python_version(),
              "numpy": np.__version__, "slurm_job_id": os.getenv("SLURM_JOB_ID"), "all_checks_passed": True}
    (ROOT / f"results/shards/scan_{args.shard:02d}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
