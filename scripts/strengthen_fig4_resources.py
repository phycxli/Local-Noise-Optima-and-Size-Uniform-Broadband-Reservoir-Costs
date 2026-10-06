"""Certify continuous bandwidth bounds for two fixed shared reservoir devices."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time

import cvxpy as cp
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import (MODELS, drift_matrix, exact_gram, floquet_envelope,
                                   legendre_coefficients, normalized_profile, source_matrix)
from audit_size_uniform_resources import boundary_coefficient, extension, extension_bound
from check_rate_capped_band_optimization import rational_array, trace_product
from check_size_uniform_resources import check_prefix, downward, upward
from run_rate_capped_band_optimization import allowed_mask, dual_bound, repair_primal

STEM = "fig4_shared_device"
TARGETS = {"uniform": F(13, 5), "dimerized": F(29, 10)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid(start: F, stop: F, step: F) -> list[F]:
    count = (stop - start) / step
    if count.denominator != 1 or step <= 0:
        raise ValueError("The exact grid must divide the declared interval.")
    return [start + j * step for j in range(int(count) + 1)]


def inputs() -> tuple[list[dict], dict, dict]:
    path = ROOT / "results/size_uniform_resources_certificate.json"
    certificate = json.loads(path.read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "results/size_uniform_resources_audit.json").read_text(encoding="utf-8"))
    hashes = {"csv_sha256": "results/size_uniform_resources.csv",
              "checker_sha256": "scripts/check_size_uniform_resources.py",
              "rational_helper_sha256": "scripts/check_rate_capped_band_optimization.py",
              "model_helper_sha256": "src/shared_band_resources.py"}
    if not certificate["all_checks_passed"] or any(
            certificate[key] != sha(ROOT / filename) for key, filename in hashes.items()):
        raise ValueError("The base rational certificate is stale.")
    if audit["base_certificate_sha256"] != sha(path):
        raise ValueError("The saved construction audit is stale.")
    with (ROOT / "results/size_uniform_resources.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)), certificate, audit


def solve_local(gram: np.ndarray, error: np.ndarray, source: np.ndarray, path: Path) -> float:
    started = time.perf_counter()
    n, cap = len(gram), 8.
    diagonal, bond = cp.Variable(n), cp.Variable(n - 1)
    # Parameterize only allowed entries; the four PSD constraints are unchanged.
    h = cp.diag(diagonal) + cp.diag(bond, 1) + cp.diag(bond, -1)
    q, eye = source / cap, np.eye(n)
    constraints = [h >> 0, h + q >> 0, eye - h >> 0, eye - h - q >> 0]
    objective = cp.sum(cp.multiply(np.diag(gram), diagonal)) + 2 * cp.sum(
        cp.multiply(np.diag(gram, 1), bond))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=2e-10, tol_gap_rel=2e-10,
                  tol_feas=2e-10, max_iter=300, max_threads=1)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or h.value is None:
        raise RuntimeError(f"The local SDP failed: {problem.status}")
    mask = allowed_mask(n, 1)
    loss, _, checks = repair_primal(h.value, source, cap, mask)
    gain = loss + source
    _, dual = dual_bound(gram, error, source, cap, [c.dual_value for c in constraints], mask)
    arrays = {"gain": gain, "loss": loss, "gram": gram, "source": source,
              "gram_entry_error": error, "mask": mask, "cap": np.array(cap)}
    for key, check in checks.items():
        arrays[f"primal_{key}_factor"] = check["factor"]
        arrays[f"primal_{key}_shift"] = np.array(check["shift"])
    for j, factor in enumerate(dual["factors"]):
        arrays[f"dual_factor_{j}"] = factor
    np.savez_compressed(path, **arrays)
    return time.perf_counter() - started


def local_witness(name: str, width: F, rows: list[dict]) -> dict:
    model, m = MODELS[name], 64
    coefficients = legendre_coefficients(model, 66, width)
    gram, source = exact_gram(model, coefficients, m), source_matrix(model, m)
    row = next((r for r in rows if r["model"] == name and F(r["halfwidth"]) == width
                and r["radius"] == "1"), None)
    if row is None:
        directory = ROOT / "results" / f"{STEM}_matrices"
        directory.mkdir(exist_ok=True)
        path = directory / f"{name}_m64_B{float(width):g}_R8_r1.npz"
        numeric = np.asarray(gram, dtype=float)
        error = np.array([float(abs(v - F.from_float(float(v))))
                          for v in gram.flat]).reshape(gram.shape)
        if not path.exists():
            print(f"Solving local witness: {name}, Omega={width}", flush=True)
            elapsed = solve_local(numeric, error, np.asarray(source, dtype=float), path)
            print(f"  elapsed={elapsed:.2f}s", flush=True)
        nodes, weights = leggauss(m + 8)
        profile = np.column_stack([normalized_profile(model, m, float(width) * x)
                                   for x in nodes])
        quadrature_error = float(np.max(np.abs(numeric - (profile * (weights / 2)) @ profile.conj().T)))
        probe = drift_matrix(model, m)
        probe[0, 0] -= float(model.kappa_in / 2)
        probe[-1, -1] -= float(model.kappa_out / 2)
        response = np.linalg.solve(-1j * float(width) * np.eye(m) - probe.T, np.eye(m)[:, -1])
        response /= response[0] * np.sqrt(float(model.kappa_in))
        normalization_error = float(np.max(np.abs(response - normalized_profile(model, m, float(width)))))
        row = {"model": name, "prefix": "64", "halfwidth": str(width), "radius": "1",
               "rate_cap": "8", "matrix_file": str(path.relative_to(ROOT)).replace("\\", "/"),
               "matrix_sha256": sha(path), "quadrature_error": str(quadrature_error),
               "response_normalization_error": str(normalization_error)}
    checked = check_prefix(row, gram, source)
    beta = checked["_beta"] + boundary_coefficient(model, coefficients, m, 1)
    return {"width": width, "alpha": checked["_alpha"], "beta": beta,
            "lower": max(F(0), checked["_alpha"] - 8 * beta),
            "row": row, "minimum_psd_margin": checked["minimum_psd_margin"]}


def fixed_devices(name: str, rows: list[dict], audit: dict) -> dict:
    model, width, m = MODELS[name], F(1, 20), 64
    reference = next(r for r in rows if r["model"] == name and F(r["halfwidth"]) == width
                     and r["radius"] == "all")
    range_two = next(r for r in audit["range_two_constructions"] if r["model"] == name)
    coefficients = legendre_coefficients(model, m, width)
    gram, source = exact_gram(model, coefficients, m), source_matrix(model, m)
    result = {}
    for key, row in (("reference", reference), ("range2", range_two)):
        check_prefix(row, gram, source)
        with np.load(ROOT / row["matrix_file"], allow_pickle=False) as saved:
            loss = rational_array(saved["loss"])
            numeric_loss = saved["loss"].copy()
        bridge = model.right[(m - 1) % 2] + model.left[(m - 1) % 2]
        complete_cap = max(F(row["rate_cap"]), sum(model.right) + sum(model.left)
                           + 2 * max(model.gamma)) + bridge
        if complete_cap > 8:
            raise ValueError("The fixed extension exceeds the complete budget.")
        result[key] = {"gain": loss + source, "loss": numeric_loss,
                       "matrix_file": row["matrix_file"], "matrix_sha256": row["matrix_sha256"],
                       "complete_cap": str(complete_cap)}
    return result


def endpoint(name: str, width: F, devices: dict) -> dict:
    model = MODELS[name]
    coefficients = legendre_coefficients(model, 160, width)
    gram = exact_gram(model, coefficients, 64)
    envelope = floquet_envelope(model, width)
    return {key: extension_bound(model, coefficients, 64, trace_product(gram, device["gain"]),
                                 envelope) for key, device in devices.items()}


def contiguous_target_intervals(cells: list[dict]) -> list[list[float]]:
    result = []
    for cell in cells:
        if not cell["minimum_range_two_certified"]:
            continue
        a, b = cell["halfwidth_interval"]
        if result and result[-1][1] == a:
            result[-1][1] = b
        else:
            result.append([a, b])
    return result


def direct_checks(name: str, devices: dict, cells: list[dict]) -> list[dict]:
    model = MODELS[name]
    result = []
    for key, device in devices.items():
        for n in (64, 65, 96, 128):
            gain, loss = extension(model, device["loss"], n)
            source = np.asarray(source_matrix(model, n), dtype=float)
            values = [np.linalg.eigvalsh(v) for v in (gain, loss)]
            mismatch = float(np.max(np.abs(gain - loss - source)))
            rate, minimum = float(max(v.max() for v in values)), float(min(v.min() for v in values))
            if mismatch > 1e-13 or rate > 8 + 1e-10 or minimum < -1e-12:
                raise ValueError("A fixed device failed a direct matrix check.")
            if key == "range2":
                distance = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
                if np.any(gain[distance > 2]) or np.any(loss[distance > 2]):
                    raise ValueError("A range-two extension contains forbidden entries.")
            nodes, weights = leggauss(n + 8)
            for width in (.03, .05, .07):
                profile = np.column_stack([normalized_profile(model, n, width * x) for x in nodes])
                noise = float(np.dot(weights / 2, np.einsum("iw,ij,jw->w", profile.conj(), gain, profile).real))
                cell = next(c for c in cells if c["halfwidth_interval"][0] <= width
                            <= c["halfwidth_interval"][1])
                ceiling = cell[f"{key}_upper"]
                if noise > ceiling + 1e-8:
                    raise ValueError("Direct band integration exceeds the certified ceiling.")
                result.append({"device": key, "N": n, "halfwidth": width,
                               "complete_rate": rate, "psd_minimum": minimum,
                               "drift_mismatch": mismatch, "quadrature_noise": noise,
                               "cell_upper": ceiling})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-step", default="0.0005")
    parser.add_argument("--cell-step", default="0.00025")
    args = parser.parse_args()
    rows, base, audit = inputs()
    anchors = grid(F(3, 100), F(7, 100), F(args.anchor_step))
    edges = grid(F(3, 100), F(7, 100), F(args.cell_step))
    groups, all_rows = [], []
    for name in MODELS:
        devices = fixed_devices(name, rows, audit)
        witnesses = {width: local_witness(name, width, rows) for width in anchors}
        cells = []
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            upper_b = endpoint(name, b, devices)
            reference_upper, range2_upper = (b / a * upper_b[k] for k in ("reference", "range2"))
            choices = [w for s, w in witnesses.items() if s <= a]
            witness = max(choices, key=lambda w: w["width"] / b * w["lower"])
            lower = witness["width"] / b * witness["lower"]
            target = TARGETS[name]
            if range2_upper < target and lower <= target:
                if a not in witnesses:
                    witnesses[a] = local_witness(name, a, rows)
                candidate = witnesses[a]
                candidate_lower = a / b * candidate["lower"]
                if candidate_lower > lower:
                    lower, witness = candidate_lower, candidate
            gap = lower - reference_upper
            if gap <= 0:
                raise ValueError(f"No positive continuous locality gap: {name}/{a}/{b}")
            cells.append({"halfwidth_interval": [float(a), float(b)],
                          "halfwidth_interval_exact": [str(a), str(b)],
                          "witness_halfwidth_exact": str(witness["width"]),
                          "local_lower": downward(lower), "local_lower_exact": str(lower),
                          "reference_upper": upward(reference_upper), "reference_upper_exact": str(reference_upper),
                          "range2_upper": upward(range2_upper), "range2_upper_exact": str(range2_upper),
                          "gap_lower": downward(gap), "gap_lower_exact": str(gap),
                          "reference_endpoint_upper_exact": str(upper_b["reference"]),
                          "range2_endpoint_upper_exact": str(upper_b["range2"]),
                          "minimum_range_two_certified": range2_upper < target < lower})
            if j % 20 == 0 or j + 1 == len(edges) - 1:
                print(f"{name}: certified {j+1}/{len(edges)-1} bandwidth cells; gap>{float(gap):.6f}", flush=True)
        coverage = contiguous_target_intervals(cells)
        summary = {"model": name, "all_N_from": 64, "complete_rate_cap": 8,
                   "halfwidth_interval": [.03, .07], "certified_cells": len(cells),
                   "local_witnesses": len(witnesses),
                   "minimum_gap_lower": min(c["gap_lower"] for c in cells),
                   "target_noise": float(TARGETS[name]), "minimum_range_two_intervals": coverage,
                   "fixed_devices": {k: {p: v for p, v in d.items() if p not in {"gain", "loss"}}
                                     for k, d in devices.items()},
                   "witnesses": [{"halfwidth_exact": str(w["width"]), "alpha_exact": str(w["alpha"]),
                                  "beta_exact": str(w["beta"]), "local_lower_R8": downward(w["lower"]),
                                  "minimum_psd_margin": w["minimum_psd_margin"], **w["row"]}
                                 for _, w in sorted(witnesses.items())],
                   "cells": cells, "direct_checks": direct_checks(name, devices, cells)}
        gains = [g for g in audit["stronger_gain_certificates"] if g["model"] == name
                 and F(str(g["halfwidth"])) == F(7, 100)]
        summary["whole_band_gain_lower_all_N"] = gains[0]["whole_band_gain_lower_all_N"]
        groups.append(summary)
        all_rows.extend(w["row"] for _, w in sorted(witnesses.items()))
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in {"cells", "witnesses", "direct_checks", "fixed_devices"}}), flush=True)
    record = {"arguments": vars(args), "arithmetic": "Exact rational prefix, dual, bridge, tail and interval bounds.",
              "continuous_proof": "For s<=a<=Omega<=b: J*(Omega)>=s/b L(s); fixed J_G(Omega)<=b/a U_G(b).",
              "same_device_across_all_bandwidths": True, "bandwidth_curves_are_certified_cell_bounds": True,
              "direct_checks_are_floating_cross_checks": True,
              "base_certificate_sha256": sha(ROOT / "results/size_uniform_resources_certificate.json"),
              "base_audit_sha256": sha(ROOT / "results/size_uniform_resources_audit.json"),
              "source_sha256": sha(Path(__file__)),
              "helper_sha256": {p: sha(ROOT / p) for p in
                                ("src/shared_band_resources.py", "scripts/audit_size_uniform_resources.py",
                                 "scripts/check_size_uniform_resources.py", "scripts/run_rate_capped_band_optimization.py")},
              "groups": groups, "all_checks_passed": True}
    output = ROOT / "results" / f"{STEM}_certificate.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "results" / f"{STEM}_witnesses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
