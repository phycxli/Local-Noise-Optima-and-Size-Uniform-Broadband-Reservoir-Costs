"""Evaluate the manuscript's geometrically balanced pointwise reservoirs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigvalsh, null_space

from run_local_noise_optimization import EXPLICIT_SIZES, KAPPA, PARAMS, response_data
from run_finite_frequency_local_optimality import scalar_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from liouvillian_geometry import local_bosonic_hatano_lindblad_matrices
from shared_band_resources import MODELS, normalized_profile

OUT = ROOT / "results/balanced_pointwise"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def construction(n):
    data = response_data(n)
    y = np.asarray(data["y"])
    q = y / np.linalg.norm(y)
    source = np.asarray(data["hermitian_part"])
    kernel = scalar_kernel(y).real
    basis = null_space(q.reshape(1, -1))
    m = float(q @ source @ q)
    assert m > 0
    cross = basis.T @ source @ q
    complement = np.outer(cross, cross) / m - basis.T @ source @ basis
    alpha_min = max(0., float(eigvalsh(complement, basis.T @ kernel @ basis)[-1]))
    alpha = alpha_min * (1 + 1e-10) + 1e-9
    loss, gain = alpha * kernel, source + alpha * kernel
    h = y / (np.sqrt(KAPPA) * y[0])
    noise = float(h @ gain @ h)
    record = {"N": n, "alpha": alpha, "alpha_family_infimum_numeric": alpha_min,
              "noise": noise, "current_noise": data["current_noise"],
              "fixed_drift_value": data["fixed_drift_value"], "caves_floor": data["caves_floor"],
              "power_gain": data["power_gain"], "construction_valid": 1,
              "complete_gain_rate": float(eigvalsh(gain)[-1]),
              "complete_loss_rate": float(eigvalsh(loss)[-1]),
              "minimum_gain_eigenvalue": float(eigvalsh(gain)[0]),
              "minimum_loss_eigenvalue": float(eigvalsh(loss)[0]),
              "normalized_kernel_error": float(np.linalg.norm(kernel @ q))}
    assert min(record["minimum_gain_eigenvalue"], record["minimum_loss_eigenvalue"]) > -1e-8
    assert abs(noise - data["fixed_drift_value"]) < 1e-9
    assert record["normalized_kernel_error"] < 1e-11
    assert np.max(np.abs(gain - loss - source)) < 1e-12
    return record, data, gain, loss, kernel


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    points = []
    for n in EXPLICIT_SIZES:
        if response_data(n)["power_gain"] > 1:
            points.append(construction(n)[0])
    point_file = OUT / "pointwise.csv"
    save_csv(point_file, points)
    reference, data, gain, loss, kernel = construction(60)
    _, original, original_loss = local_bosonic_hatano_lindblad_matrices(60, PARAMS)
    source, probed = np.asarray(data["hermitian_part"]), np.asarray(data["x"]).copy()
    probed[0, 0] -= KAPPA / 2
    probed[-1, -1] -= KAPPA / 2
    output = np.eye(60)[:, -1]
    rows, recurrence_errors, identities = [], [], []
    for omega in np.linspace(-.05, .05, 401):
        y = np.linalg.solve(-1j * omega * np.eye(60) - probed.T, output)
        h = y / (np.sqrt(KAPPA) * y[0])
        recurrence_errors.append(float(np.max(np.abs(h - normalized_profile(MODELS["uniform"], 60, float(omega))))))
        floor = float(np.vdot(h, source @ h).real)
        excess = float(np.vdot(h, loss @ h).real)
        noise = float(np.vdot(h, gain @ h).real)
        identities.append(abs(noise - floor - excess))
        rows.append({"N": 60, "omega": float(omega), "original_noise": float(np.vdot(h, original @ h).real),
                     "fixed_local_noise": noise, "pointwise_floor": floor,
                     "projected_loss_noise": excess, "power_gain": float(KAPPA**2 * abs(y[0])**2)})
    assert max(recurrence_errors) < 1e-10 and max(identities) < 1e-9
    assert abs(rows[200]["fixed_local_noise"] - reference["noise"]) < 1e-9
    frequency_file, matrix_file = OUT / "frequency.csv", OUT / "reservoir.npz"
    save_csv(frequency_file, rows)
    np.savez_compressed(matrix_file, gain=gain, loss=loss, kernel=kernel, original_gain=original,
                        original_loss=original_loss, drift=data["x"], probed_drift=probed,
                        alpha=reference["alpha"], kappa=KAPPA)
    record = {"all_checks_passed": True, "kernel_normalization": "geometric mean, as in the manuscript",
              "alpha_selection": "Largest generalized eigenvalue of the Schur complement; small positive numerical margin",
              "scope": "Numerical evaluation of the existing uncapped construction, not a new exact-rate certificate",
              "reference": reference, "center": rows[200], "positive_endpoint": rows[-1],
              "N64": construction(64)[0], "samples": len(rows),
              "max_recurrence_error": max(recurrence_errors), "max_noise_identity_error": max(identities),
              "rate_cap": None, "fixed_couplings_across_frequency": True,
              "pointwise_csv": point_file.relative_to(ROOT).as_posix(),
              "frequency_csv": frequency_file.relative_to(ROOT).as_posix(),
              "output_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (point_file, frequency_file, matrix_file)},
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}}
    (OUT / "audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("all_checks_passed", "reference", "N64", "positive_endpoint")}, indent=2))


if __name__ == "__main__":
    build()
