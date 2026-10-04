"""Consistency checks for the CP/rate bound; no SDP optimization is performed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from audit_multiband_crossing_boundary import drift
from boundary_evans import real_log_mismatch
from boundary_evans_interval import (complex_box, float_down, float_up, interval,
                                     iv, lower, real_band_bounds, upper)

ROOT = Path(__file__).resolve().parents[1]


def rate_lower_bound(q, my_norm2, rate):
    k = my_norm2 / rate
    return 0.0 if k == 0 else max(k - abs(q), 0.0) ** 2 / (4 * k)


def random_checks():
    rng = np.random.default_rng(271828)
    count, maximum_violation = 0, -np.inf
    for _ in range(250):
        raw = [rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)) for _ in range(2)]
        matrices = [a.conj().T @ a for a in raw]
        gain, loss = [a / np.linalg.eigvalsh(a)[-1] for a in matrices]
        m = gain - loss
        vectors = [rng.normal(size=8) + 1j * rng.normal(size=8)]
        eigenvalues, eigenvectors = np.linalg.eigh(m)
        if eigenvalues[0] < 0 < eigenvalues[-1]:
            positive = -eigenvalues[0] / (eigenvalues[-1] - eigenvalues[0])
            vectors.append(np.sqrt(positive) * eigenvectors[:, -1]
                           + np.sqrt(1 - positive) * eigenvectors[:, 0])
        for y in vectors:
            y = y / np.linalg.norm(y)
            q = float(np.vdot(y, m @ y).real)
            noise = float(np.vdot(y, gain @ y).real)
            norm2 = float(np.vdot(m @ y, m @ y).real)
            bound = rate_lower_bound(q, norm2, 1.0)
            violation = bound - (noise - max(q, 0))
            maximum_violation = max(maximum_violation, violation)
            if violation > 1e-11:
                raise ArithmeticError("The general CP/rate inequality audit failed")
            count += 1
    return count, maximum_violation


def limit_certificate():
    lo, hi = "0.2098976657", "0.2098976659"
    left = real_band_bounds(complex_box(lo, lo), 119)
    right = real_band_bounds(complex_box(hi, hi), 119)
    if not upper(left["tau"]) < 1 or not lower(right["tau"]) > 1:
        raise ArithmeticError("The proposed limiting-frequency bracket is not certified")
    band = real_band_bounds(complex_box(lo, hi), 119)
    coefficient = iv.log(band["C_infinity"] / interval("0.2")) / band["log_tau_prime"]
    return {"limiting_frequency_bracket_verified": True, "omega_infinity_bracket": [lo, hi],
            "leading_1_over_m_shift_interval": [float_down(coefficient), float_up(coefficient)],
            "ell_infinity_interval": [float_down(band["log_tau_prime"]), float_up(band["log_tau_prime"])]}


def model_checks():
    rows = []
    kappa, rate, sigma2 = 0.2, 5.1, 0.40005625
    br = sigma2 / (kappa * rate)
    for n in (24, 80, 121, 200):
        root = brentq(lambda w: real_log_mismatch(w, n), 0, 0.25, xtol=2e-14)
        xk = drift(n, True)
        m = drift(n, False) + drift(n, False).conj().T
        source = np.zeros(2 * n, complex)
        source[-2] = 1
        for v in (-0.2, 0, 0.2):
            omega = root + v / (n - 2)
            a = csc_matrix(-1j * omega * np.eye(2 * n) - xk.conj().T)
            y = spsolve(a, source)
            q = float(np.vdot(y, m @ y).real)
            denominator = kappa * abs(y[0]) ** 2
            gain_noise = 0.2 * float(np.vdot(y, y).real)
            power_gain = kappa * kappa * abs(y[0]) ** 2
            reflection = abs(1 - kappa * np.conjugate(y[-2])) ** 2
            balance_error = abs(power_gain + reflection - 1 - kappa * q)
            if balance_error > 1e-10 or (v == 0 and power_gain > 1 + 1e-9):
                raise ArithmeticError("The scattering-row balance audit failed")
            actual_excess = (gain_noise - max(q, 0)) / denominator
            r = abs(q) / denominator
            lower_bound = max(br - r, 0) ** 2 / (4 * br)
            if actual_excess + 1e-10 < lower_bound:
                raise ArithmeticError("The input-referred physical bound audit failed")
            rows.append({"N": n, "omega_star_numerical": root, "scaled_detuning": v,
                         "omega": omega, "q": q, "r_N": r,
                         "rate_budget": rate, "analytic_excess_lower_bound": lower_bound,
                         "local_feasible_example_excess": actual_excess,
                         "power_gain": power_gain, "output_reflection_power": reflection,
                         "scattering_row_balance_error": balance_error})
    return rows


def main():
    count, violation = random_checks()
    limiting = limit_certificate()
    rows = model_checks()
    csv_path = ROOT / "results" / "boundary_rate_noise_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"general_PSD_checks_passed": count, "max_random_bound_violation": violation,
              "model_consistency_checks_passed": len(rows),
              "noise_bound_at_crossing": "0.5000703125/R",
              "feasible_local_rate_budget": 5.1,
              "bound_at_rate_5p1": 0.5000703125 / 5.1,
              "power_gain_at_crossings": {str(row["N"]): row["power_gain"] for row in rows
                                           if row["scaled_detuning"] == 0},
              "no_SDP_optimality_claim": True, **limiting}
    (ROOT / "results" / "boundary_rate_noise_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
