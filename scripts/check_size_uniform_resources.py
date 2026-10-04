"""Rational certificates for prefix lower bounds and all-length constructions."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import (MODELS, exact_gram, floquet_envelope, gram_entry,
                                   legendre_coefficients, source_matrix, tail_bound,
                                   uniform_gain_certificate)
from check_rate_capped_band_optimization import exact_factor_product, rational_array, row_norm, trace_product


def downward(value: F) -> float:
    return float(np.nextafter(float(value), -np.inf))


def upward(value: F) -> float:
    return float(np.nextafter(float(value), np.inf))


def residual_penalty(residual: np.ndarray) -> F:
    n = len(residual)
    return (sum((max(-residual[j, j], F(0)) for j in range(n)), F(0))
            + sum((abs(residual[i, j]) for i in range(n) for j in range(i + 1, n)), F(0)))


def check_prefix(row: dict, gram: np.ndarray, source: np.ndarray) -> dict:
    path = ROOT / row["matrix_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["matrix_sha256"]:
        raise ValueError("The matrix archive hash changed.")
    n, cap = int(row["prefix"]), F(row["rate_cap"])
    radius = None if row["radius"] == "all" else int(row["radius"])
    mask = (np.ones((n, n), dtype=bool) if radius is None else
            np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= radius)
    with np.load(path, allow_pickle=False) as saved:
        loss = rational_array(saved["loss"])
        if loss.shape != (n, n) or not np.array_equal(loss, loss.T):
            raise ValueError("The reservoir is not real symmetric with the declared size.")
        if not np.array_equal(saved["mask"], mask) or any(loss[~mask]):
            raise ValueError("The declared reservoir range is not respected.")
        gain, eye = loss + source, np.eye(n, dtype=object) * F(1)
        targets = {"loss": loss, "gain": gain, "loss_slack": cap * eye - loss,
                   "gain_slack": cap * eye - gain}
        margins = []
        for name, target in targets.items():
            factor_product = exact_factor_product(saved[f"primal_{name}_factor"])
            shift = F.from_float(float(saved[f"primal_{name}_shift"]))
            margin = shift - row_norm(target - shift * eye - factor_product)
            if margin <= 0:
                raise ValueError(f"Exact positivity check failed: {path.name}/{name}")
            margins.append(margin)
        a, b, c, d = [exact_factor_product(saved[f"dual_factor_{j}"]) for j in range(4)]
        residual = (gram - a - b + c + d) * mask
        alpha = trace_product(gram, source) + trace_product(d - b, source)
        beta = sum((c[j, j] + d[j, j] for j in range(n)), F(0)) + residual_penalty(residual)
        lower, upper = alpha - cap * beta, trace_product(gram, gain)
        if lower > upper:
            raise ValueError("The rational lower bound exceeds the feasible objective.")
    return {"model": row["model"], "prefix": n, "halfwidth": float(F(row["halfwidth"])),
            "radius": row["radius"], "rate_cap": float(cap), "minimum_psd_margin": downward(min(margins)),
            "finite_lower": downward(lower), "finite_upper": upward(upper),
            "alpha_exact": str(alpha), "beta_exact": str(beta),
            "pointwise_floor_at_prefix": float(trace_product(gram, source)),
            "matrix_file": row["matrix_file"], "matrix_sha256": row["matrix_sha256"],
            "quadrature_error": float(row["quadrature_error"]),
            "response_normalization_error": float(row["response_normalization_error"]),
            "exact_checks_passed": True, "_alpha": alpha, "_beta": beta, "_upper": upper}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="size_uniform_resources")
    parser.add_argument("--tail-start", type=int, default=160)
    args = parser.parse_args()
    if args.tail_start % 2:
        raise ValueError("The tail cutoff must be even.")
    csv_path = ROOT / "results" / f"{args.stem}.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        inputs = list(csv.DictReader(handle))
    metadata = json.loads((ROOT / "results" / f"{args.stem}_metadata.json").read_text(encoding="utf-8"))
    for filename, key in (("scripts/run_size_uniform_resources.py", "source_sha256"),
                          ("src/shared_band_resources.py", "model_helper_sha256"),
                          ("scripts/run_rate_capped_band_optimization.py", "solver_helper_sha256")):
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError(f"A source hash changed: {filename}")
    arguments = metadata["arguments"]
    expected = {(name, str(F(width)), str(arguments["prefix"]), radius)
                for name in arguments["models"] for width in arguments["bands"]
                for radius in ("1", "2", "all")}
    declared = {(row["model"], str(F(row["halfwidth"])), row["prefix"], row["radius"])
                for row in inputs}
    if declared != expected or len(inputs) != len(expected):
        raise ValueError("The saved CSV does not contain exactly the declared problems.")
    if metadata["models"] != [MODELS[name].description() for name in arguments["models"]]:
        raise ValueError("The declared drift parameters differ from the exact model.")
    cases, groups = [], []
    keys = sorted({(row["model"], int(row["prefix"]), F(row["halfwidth"])) for row in inputs})
    for name, prefix, width in keys:
        if prefix % 2 or args.tail_start <= prefix + 4:
            raise ValueError("Use an even prefix below the tail cutoff.")
        model = MODELS[name]
        coefficients = legendre_coefficients(model, args.tail_start, width)
        gram = exact_gram(model, coefficients, prefix)
        source = source_matrix(model, prefix)
        rows = [row for row in inputs if row["model"] == name and int(row["prefix"]) == prefix
                and F(row["halfwidth"]) == width]
        checked = [check_prefix(row, gram, source) for row in rows]
        reference = next(row for row in checked if row["radius"] == "all")
        envelope = floquet_envelope(model, width)
        gain = uniform_gain_certificate(model, envelope, width, prefix)
        gain["whole_band_gain_lower_all_N"] = float(np.nextafter(
            gain["whole_band_gain_lower_all_N"], -np.inf))
        bond_rates = [model.right[j] + model.left[j] for j in range(2)]
        bridge = bond_rates[(prefix - 1) % 2]
        gain_tail_cap = sum(bond_rates)
        loss_tail_cap = gain_tail_cap + 2 * max(model.gamma)
        reference_cap = F(str(reference["rate_cap"]))
        complete_cap = max(reference_cap, loss_tail_cap) + bridge
        if complete_cap > 8:
            raise ValueError("The extended reference exceeds the comparison budget.")
        bridge_noise = bridge / 2 * (gram_entry(model, coefficients, prefix - 1, prefix - 1)
                                     + 2 * gram_entry(model, coefficients, prefix - 1, prefix)
                                     + gram_entry(model, coefficients, prefix, prefix))
        finite_tail = sum((bond_rates[j % 2] / 2
                          * (gram_entry(model, coefficients, j, j)
                             + 2 * gram_entry(model, coefficients, j, j + 1)
                             + gram_entry(model, coefficients, j + 1, j + 1))
                          for j in range(prefix, args.tail_start - 2)), F(0))
        remainder = gain_tail_cap * tail_bound(model, envelope, args.tail_start - 2)
        reference_upper = reference["_upper"] + bridge_noise + finite_tail + remainder
        group = {"model": name, "prefix": prefix, "halfwidth": float(width),
                 "all_N_from": prefix, "complete_reference_rate_cap": float(complete_cap),
                 "nonlocal_construction_upper": upward(reference_upper),
                 "reference_upper_exact": str(reference_upper),
                 "bridge_noise": upward(bridge_noise), "finite_tail_noise": upward(finite_tail),
                 "tail_noise_remainder": upward(remainder),
                 "floquet_q": float(envelope["q"]), "floquet_small_root_bound": float(envelope["q_small"]),
                 "root_separation_lower": downward(envelope["separation"]),
                 "schur_ellipse_margin": downward(envelope["ellipse_margin"]),
                 "stability_margin": downward(envelope["stability_margin"]),
                 **gain, "locality_bounds": []}
        for case in checked:
            if case["radius"] == "all":
                continue
            radius = int(case["radius"])
            boundary = sum((abs(gram_entry(model, coefficients, i, j))
                            for i in range(max(0, prefix - radius), prefix)
                            for j in range(prefix, prefix + radius) if j - i <= radius), F(0))
            slope = case["_beta"] + boundary
            lower = case["_alpha"] - 8 * slope
            gap = lower - reference_upper
            bound = {"radius": radius, "alpha": downward(case["_alpha"]),
                     "beta_with_boundary": upward(slope), "alpha_exact": str(case["_alpha"]),
                     "beta_with_boundary_exact": str(slope), "boundary_coefficient": upward(boundary),
                     "all_N_local_noise_lower_at_R8": downward(lower),
                     "all_N_locality_gap_lower_at_R8": downward(gap),
                     "positive_gap": gap > 0,
                     "necessary_R_for_reference_plus_0p1": downward(max((case["_alpha"] - reference_upper - F(1, 10)) / slope, F(0))),
                     "necessary_R_for_reference_plus_0p05": downward(max((case["_alpha"] - reference_upper - F(1, 20)) / slope, F(0)))}
            group["locality_bounds"].append(bound)
        groups.append(group)
        for case in checked:
            cases.append({key: value for key, value in case.items() if not key.startswith("_")})
        printable = {key: value for key, value in group.items() if key not in {"reference_upper_exact", "locality_bounds"}}
        printable["locality_bounds"] = [{key: value for key, value in bound.items() if not key.endswith("_exact")}
                                        for bound in group["locality_bounds"]]
        print(json.dumps(printable), flush=True)
    certificate = {"arithmetic": "Exact rational Gram entries, matrix residuals, Schur ellipses and geometric tail sums.",
                   "theorem_scope": "Period-two positive Jacobi drifts; finite prefix plus explicit reservoir extension; every finite N >= prefix.",
                   "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                   "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "rational_helper_sha256": hashlib.sha256(
                       (ROOT / "scripts/check_rate_capped_band_optimization.py").read_bytes()).hexdigest(),
                   "model_helper_sha256": metadata["model_helper_sha256"],
                   "tail_cutoff": args.tail_start, "cases": cases, "groups": groups,
                   "all_checks_passed": all(case["exact_checks_passed"] for case in cases),
                   "all_nearest_neighbor_uniform_gaps_positive": all(
                       bound["positive_gap"] for group in groups for bound in group["locality_bounds"] if bound["radius"] == 1)}
    output = ROOT / "results" / f"{args.stem}_certificate.json"
    output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    summary = [{"model": group["model"], "halfwidth": group["halfwidth"], "all_N_from": group["all_N_from"],
                "reference_upper": group["nonlocal_construction_upper"], "radius": bound["radius"],
                "local_lower_R8": bound["all_N_local_noise_lower_at_R8"],
                "locality_gap_lower_R8": bound["all_N_locality_gap_lower_at_R8"],
                "alpha": bound["alpha"], "beta": bound["beta_with_boundary"],
                "necessary_R_for_ref_plus_0p1": bound["necessary_R_for_reference_plus_0p1"]}
               for group in groups for bound in group["locality_bounds"]]
    with (ROOT / "results" / f"{args.stem}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Certified {len(cases)} finite problems and {len(groups)} all-length constructions.", flush=True)


if __name__ == "__main__":
    main()
