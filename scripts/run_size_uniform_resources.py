"""Build finite-prefix witnesses and shared reservoirs for all longer chains."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import MODELS, drift_matrix, exact_gram, legendre_coefficients, normalized_profile, source_matrix
from run_rate_capped_band_optimization import EPS, psd_check, solve_shared


def reuse_saved(model_name: str, width: F, prefix: int, radius: str, cap: float, path: Path) -> bool:
    if model_name != "uniform" or width != F(1, 20) or prefix != 64:
        return False
    original = ROOT / "results/rate_capped_band_optimization_matrices" / f"N64_B0.05_R8_r{radius}.npz"
    with np.load(original, allow_pickle=False) as old:
        arrays = {key: old[key] for key in old.files}
    eye, gain, loss = np.eye(prefix), arrays["gain"], arrays["loss"]
    for name, matrix in (("loss", loss), ("gain", gain),
                         ("loss_slack", cap * eye - loss), ("gain_slack", cap * eye - gain)):
        error = 16 * EPS * (1 + np.abs(matrix) + np.abs(gain) + np.abs(loss))
        certificate = psd_check(matrix, error)
        if not certificate["valid"]:
            return False
        arrays[f"primal_{name}_factor"] = certificate["factor"]
        arrays[f"primal_{name}_shift"] = np.array(certificate["shift"])
    arrays["cap"] = np.array(cap)
    np.savez_compressed(path, **arrays)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["uniform", "dimerized"], choices=list(MODELS))
    parser.add_argument("--bands", nargs="+", default=["0.03", "0.05", "0.07"])
    parser.add_argument("--prefix", type=int, default=64)
    parser.add_argument("--stem", default="size_uniform_resources")
    args = parser.parse_args()
    directory = ROOT / "results" / f"{args.stem}_matrices"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        model = MODELS[name]
        source = np.asarray(source_matrix(model, args.prefix), dtype=float)
        bridge = float(model.right[(args.prefix - 1) % 2] + model.left[(args.prefix - 1) % 2])
        reference_cap = 8 - bridge
        for band in args.bands:
            width = F(band)
            coefficients = legendre_coefficients(model, args.prefix, width)
            rational = exact_gram(model, coefficients, args.prefix)
            gram = np.asarray(rational, dtype=float)
            error = np.array([float(abs(value - F.from_float(float(value)))) for value in rational.flat]).reshape(gram.shape)
            nodes, weights = leggauss(args.prefix + 8)
            sample = np.column_stack([np.sqrt(w / 2) * normalized_profile(model, args.prefix, float(width) * x)
                                      for x, w in zip(nodes, weights)])
            discrepancy = np.max(np.abs(gram - (sample @ sample.conj().T).real))
            drift = drift_matrix(model, args.prefix)
            probe = drift.copy()
            probe[0, 0] -= float(model.kappa_in / 2)
            probe[-1, -1] -= float(model.kappa_out / 2)
            target = np.eye(args.prefix)[:, -1]
            y = np.linalg.solve((-1j * float(width) * np.eye(args.prefix) - probe.T), target)
            normalized_error = np.max(np.abs(y / (y[0] * np.sqrt(float(model.kappa_in)))
                                             - normalized_profile(model, args.prefix, float(width))))
            for radius in ("1", "2", "all"):
                cap = reference_cap if radius == "all" else 8.
                path = directory / f"{name}_m{args.prefix}_B{float(width):g}_R{cap:g}_r{radius}.npz"
                reused = reuse_saved(name, width, args.prefix, radius, cap, path)
                if not reused:
                    print(f"Solving {name}, Omega={width}, range={radius}, cap={cap}", flush=True)
                    solve_shared(args.prefix, cap, gram, error, source,
                                 None if radius == "all" else int(radius), 2e-10, path)
                row = {"model": name, "prefix": args.prefix, "halfwidth": str(width),
                       "radius": radius, "rate_cap": str(F(str(cap))),
                       "matrix_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                       "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "reused_previous_solution": reused,
                       "quadrature_error": float(discrepancy), "response_normalization_error": float(normalized_error)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                with (ROOT / "results" / f"{args.stem}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    metadata = {"arguments": vars(args), "models": [MODELS[name].description() for name in args.models],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_helper_sha256": hashlib.sha256((ROOT / "src/shared_band_resources.py").read_bytes()).hexdigest(),
                "solver_helper_sha256": hashlib.sha256((ROOT / "scripts/run_rate_capped_band_optimization.py").read_bytes()).hexdigest(),
                "purpose": "Finite-prefix witnesses plus a frequency-independent extension; all-N claims require the exact checker."}
    (ROOT / "results" / f"{args.stem}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
