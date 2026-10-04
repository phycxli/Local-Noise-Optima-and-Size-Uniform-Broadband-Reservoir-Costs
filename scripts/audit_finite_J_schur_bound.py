"""Audit the finite-J Schur-complement perturbation of the scalar active band.

The script separates the exact block formula from a norm-based perturbation
bound.  It reports whether the simple Neumann bound is uniform in N; it does
not assume that it is.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import svdvals
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
T_RIGHT = np.array([1.0, 0.8])
T_LEFT = np.array([0.25, 0.35])
GAMMA = np.array([1.2, 1.15])
KAPPA = 0.2
J_VALUES = (0.001, 0.005, 0.01, 0.02)
OMEGAS = np.linspace(0.0, 2.0, 101)


def band_matrix(n: int, band: int, omega: float, probed: bool = True) -> np.ndarray:
    size = n
    matrix = np.zeros((size, size), dtype=complex)
    gamma = GAMMA[band]
    right = T_RIGHT[band]
    left = T_LEFT[band]
    for cell in range(n):
        matrix[cell, cell] = gamma - 1j * omega
        if probed and band == 0 and cell in (0, n - 1):
            matrix[cell, cell] += KAPPA / 2.0
        if cell + 1 < n:
            # A=-i omega-X^dagger: its upper/lower entries are
            # -t_R and -t_L, respectively.
            matrix[cell, cell + 1] = -right
            matrix[cell + 1, cell] = -left
    return matrix


def full_response(n: int, omega: float, mixing: float) -> tuple[float, float]:
    size = 2 * n
    a = np.zeros((size, size), dtype=complex)
    for band in (0, 1):
        block = band_matrix(n, band, omega)
        sl = slice(band * n, (band + 1) * n)
        a[sl, sl] = block
    coupling = -1j * mixing * np.eye(n)
    a[:n, n:] = coupling
    a[n:, :n] = coupling
    source = np.zeros(size, dtype=complex)
    source[n - 1] = 1.0
    y = np.linalg.solve(a, source)
    m = np.zeros((size, size), dtype=complex)
    for band in (0, 1):
        block = np.zeros((n, n), dtype=complex)
        gamma = GAMMA[band]
        right = T_RIGHT[band]
        left = T_LEFT[band]
        for cell in range(n):
            block[cell, cell] = -2.0 * gamma
            if cell + 1 < n:
                block[cell, cell + 1] = right + left
                block[cell + 1, cell] = right + left
        sl = slice(band * n, (band + 1) * n)
        m[sl, sl] = block
    q = float(np.vdot(y, m @ y).real)
    return q, float(np.linalg.norm(y))


def scalar_response(n: int, omega: float) -> float:
    return full_response(n, omega, 0.0)[0]


def roots_on_grid(fun) -> list[float]:
    values = [fun(omega) for omega in OMEGAS]
    roots = []
    for left, right, f_left, f_right in zip(
        OMEGAS[:-1], OMEGAS[1:], values[:-1], values[1:]
    ):
        if f_left * f_right < 0:
            root = brentq(fun, left, right, xtol=1e-11)
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(float(root))
    return roots


def main() -> None:
    rows = []
    for n in (24, 32, 40, 80, 120):
        norms_1 = []
        norms_2 = []
        products = []
        q0_values = []
        for omega in OMEGAS:
            a1 = band_matrix(n, 0, float(omega))
            a2 = band_matrix(n, 1, float(omega))
            inv1 = np.linalg.inv(a1)
            inv2 = np.linalg.inv(a2)
            n1 = float(svdvals(inv1)[0])
            n2 = float(svdvals(inv2)[0])
            norms_1.append(n1)
            norms_2.append(n2)
            products.append(n1 * n2)
            q0_values.append(scalar_response(n, float(omega)))

        roots0 = roots_on_grid(lambda omega: scalar_response(n, omega))
        base = {
            "N": n,
            "max_norm_A1_inv": max(norms_1),
            "max_norm_A2_inv": max(norms_2),
            "max_norm_product": max(products),
            "q0_at_omega0": q0_values[0],
            "q0_at_omega2": q0_values[-1],
            "scalar_roots": ";".join(f"{root:.12g}" for root in roots0),
        }
        print(base, flush=True)
        for mixing in J_VALUES:
            deltas = []
            q_values = []
            for omega, q0 in zip(OMEGAS, q0_values):
                qj, _ = full_response(n, float(omega), mixing)
                deltas.append(qj - q0)
                q_values.append(qj)
            rootsj = roots_on_grid(lambda omega: full_response(n, omega, mixing)[0])
            rows.append({
                **base,
                "J": mixing,
                "max_qJ_minus_q0": max(deltas),
                "min_qJ_minus_q0": min(deltas),
                "qJ_at_omega0": q_values[0],
                "qJ_at_omega2": q_values[-1],
                "finite_J_roots": ";".join(f"{root:.12g}" for root in rootsj),
                "max_neumann_parameter": mixing * mixing * max(products),
            })
            print(rows[-1], flush=True)

    output = ROOT / "results" / "finite_J_schur_bound_audit.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
