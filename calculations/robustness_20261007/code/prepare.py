"""Freeze input provenance and compute reservoirs with a hardware rate reserve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cvxpy as cp
import numpy as np

from physics import CONFIG, MODELS, ROOT, grams, matrices, parameters

PROJECT = ROOT.parents[1]
if (PROJECT / "src").is_dir():
    sys.path.insert(0, str(PROJECT / "src"))
from shared_band_resources import MODELS as EXACT_MODELS, exact_gram, legendre_coefficients
from fractions import Fraction as F


def solve(gram, source, cap, radius, intrinsic=0.):
    n, eye = len(gram), np.eye(len(gram))
    entries = [cp.Variable(n - j) for j in range(radius + 1)]
    f = cp.diag(entries[0])
    for j in range(1, radius + 1):
        f += cp.diag(entries[j], j) + cp.diag(entries[j], -j)
    q = (source + intrinsic * eye) / cap
    constraints = [f >> 0, f + q >> 0, (1 - intrinsic / cap) * eye - f >> 0, eye - f - q >> 0]
    problem = cp.Problem(cp.Minimize(cp.trace(gram @ f)), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=2e-10, tol_gap_rel=2e-10,
                  tol_feas=2e-10, max_iter=300, max_threads=1)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(problem.status)
    raw = cap * (f.value + f.value.T) / 2
    anchor = (cap * eye - source) / 2 - intrinsic * eye
    violation = max(0., *(-np.linalg.eigvalsh(x).min() for x in
                          (raw, raw + source + intrinsic * eye,
                           (cap - intrinsic) * eye - raw, cap * eye - raw - source - intrinsic * eye)))
    anchor_margin = min(np.linalg.eigvalsh(x).min() for x in
                       (anchor, anchor + source + intrinsic * eye,
                        (cap - intrinsic) * eye - anchor, cap * eye - anchor - source - intrinsic * eye))
    mixing = (violation + 2e-7) / (anchor_margin + violation + 2e-7)
    programmed_loss = (1 - mixing) * raw + mixing * anchor
    loss = programmed_loss + intrinsic * eye
    gain = loss + source
    arrays = {"loss": loss, "gain": gain, "cap": np.array(cap), "gram": gram, "source": source}
    margins = {}
    for name, target in (("programmed_loss", programmed_loss), ("gain", gain),
                         ("complete_loss_slack", cap * eye - loss), ("gain_slack", cap * eye - gain)):
        shift = np.linalg.eigvalsh(target).min() / 2
        factor = np.linalg.cholesky(target - shift * eye)
        residual = np.max(np.sum(abs(target - shift * eye - factor @ factor.T), axis=1))
        assert shift > residual + 1e-10
        arrays[f"primal_{name}_factor"] = factor
        arrays[f"primal_{name}_shift"] = np.array(shift)
        margins[name] = float(shift - residual - 1e-10)
    for j, constraint in enumerate(constraints):
        raw_dual = (constraint.dual_value + constraint.dual_value.T) / 2
        values, vectors = np.linalg.eigh(raw_dual)
        positive = (vectors * np.maximum(values, 0)) @ vectors.T
        positive += 1e-11 * max(1., np.linalg.norm(positive, 2)) * eye
        arrays[f"dual_factor_{j}"] = np.linalg.cholesky(positive)
    return arrays, {"solver_status": problem.status, "noise": float(np.sum(gram * gain.T)),
                    "complete_rate": float(max(np.linalg.eigvalsh(loss).max(), np.linalg.eigvalsh(gain).max())),
                    "positive_factor_margins": margins, "repair_mixing": mixing}


def main():
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    records = []
    for name in MODELS:
        p = parameters(name)
        source = matrices(p, 64)[1]
        model = EXACT_MODELS[name]
        gram = np.asarray(exact_gram(model, legendre_coefficients(model, 64, F(1, 20)), 64), dtype=float)
        for family, cap in (("finite", CONFIG["finite_seed_cap"]), ("extended", CONFIG["extended_seed_cap"])):
            for radius in (1, 2):
                arrays, record = solve(gram, source, cap, radius)
                target = ROOT / "inputs" / f"{name}_{family}_r{radius}.npz"
                np.savez_compressed(target, **arrays)
                records.append({"model": name, "family": family, "radius": radius,
                                "file": target.relative_to(ROOT).as_posix(), **record})
                print(json.dumps(records[-1]), flush=True)
        _, physical_gram, _, _, _, _ = grams(p, 64, .05, 50., .01, order=193)
        arrays, record = solve(physical_gram, source, 8., 1, intrinsic=.01)
        target = ROOT / "inputs" / f"{name}_physical_dual.npz"
        np.savez_compressed(target, **arrays)
        records.append({"model": name, "file": target.relative_to(ROOT).as_posix(), "purpose": "physical local lower witness", **record})
        print(json.dumps(records[-1]), flush=True)
    previous_path = ROOT / "inputs/provenance.json"
    provenance = (json.loads(previous_path.read_text(encoding="utf-8"))["project_inputs_sha256"]
                  if previous_path.exists() else {})
    for filename in ("main/main.tex", "supplement/supplement.tex", "src/shared_band_resources.py",
                     "results/rate_capped_band_optimization_certificate.json",
                     "results/size_uniform_resources_certificate.json"):
        path = PROJECT / filename
        if path.exists():
            provenance[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in MODELS:
        path = PROJECT / f"results/size_uniform_resources_matrices/{name}_m64_B0.05_R8_r1.npz"
        target = ROOT / "inputs" / f"{name}_previous_local_witness.npz"
        if path.exists():
            target.write_bytes(path.read_bytes())
            provenance[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "inputs/provenance.json").write_text(json.dumps({"project_inputs_sha256": provenance, "new_seeds": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
