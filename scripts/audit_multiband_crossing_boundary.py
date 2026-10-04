"""Find finite-chain sign crossings and audit the two-band boundary bound."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
T_RIGHT = np.array([1.0, 0.8])
T_LEFT = np.array([0.25, 0.35])
GAMMA = np.array([1.2, 1.15])
J_MIX = 0.01
KAPPA = 0.2
PORT = np.array([1.0, 0.0], dtype=complex)
SIZES = (8, 12, 16, 18, 20, 22, 24, 32, 40, 60, 80)
OMEGA_GRID = np.linspace(-2.0, 2.0, 2001)


def blocks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    onsite = -np.diag(GAMMA).astype(complex) - 1j * J_MIX * sigma_x
    upper = np.diag(T_LEFT).astype(complex)
    lower = np.diag(T_RIGHT).astype(complex)
    return onsite, upper, lower


def drift(n: int, probed: bool) -> np.ndarray:
    onsite, upper, lower = blocks()
    x = np.zeros((2 * n, 2 * n), dtype=complex)
    for cell in range(n):
        sl = slice(2 * cell, 2 * cell + 2)
        x[sl, sl] = onsite
        if cell + 1 < n:
            sr = slice(2 * cell + 2, 2 * cell + 4)
            x[sl, sr] = upper
            x[sr, sl] = lower
    if probed:
        x[0, 0] -= KAPPA / 2.0
        x[-2, -2] -= KAPPA / 2.0
    return x


def boundary_matrix(omega: float) -> np.ndarray:
    onsite, upper, lower = blocks()
    input_damping = KAPPA / 2.0 * np.outer(PORT, PORT.conj())
    rhs_map = -1j * omega * np.eye(2) - onsite.conj().T + input_damping
    return (
        onsite + onsite.conj().T
        + (upper + lower.conj().T) @ np.linalg.solve(lower.conj().T, rhs_map)
    )


def response_data(n: int, omega: float, dense: bool = False) -> dict[str, float]:
    xk = drift(n, probed=True)
    m = drift(n, probed=False) + drift(n, probed=False).conj().T
    a_dag = csc_matrix((1j * omega * np.eye(2 * n) - xk).conj().T)
    source = np.zeros(2 * n, dtype=complex)
    source[-2:] = PORT
    y = np.linalg.solve(a_dag.toarray(), source) if dense else spsolve(a_dag, source)
    my = m @ y
    norm2 = float(np.vdot(y, y).real)
    signature = float(np.vdot(y, my).real / norm2)
    input_amp = complex(np.vdot(PORT, y[:2]))
    denominator = KAPPA * abs(input_amp) ** 2
    pi_direct = float(np.vdot(my, my).real / denominator)
    projected = my - y * (np.vdot(y, my) / norm2)
    pi_projected = float(np.vdot(projected, projected).real / denominator)
    singular_values = np.linalg.svd(boundary_matrix(omega), compute_uv=False)
    sigma_min = float(singular_values[-1])
    return {
        "signature": signature,
        "Pi_direct": pi_direct,
        "Pi_projected": pi_projected,
        "sigma_min_C": sigma_min,
        "Pi_boundary_bound": sigma_min**2 / KAPPA,
    }


def crossings(n: int) -> list[float]:
    values = [response_data(n, float(w))["signature"] for w in OMEGA_GRID]
    roots: list[float] = []
    for j in range(len(OMEGA_GRID) - 1):
        if values[j] * values[j + 1] < 0.0:
            root = brentq(
                lambda w: response_data(n, w)["signature"],
                OMEGA_GRID[j], OMEGA_GRID[j + 1], xtol=1e-13,
            )
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(float(root))
    return roots


def main() -> None:
    rows: list[dict[str, float]] = []
    for n in SIZES:
        roots = crossings(n)
        if not roots:
            rows.append({"N": n, "crossing_index": -1, "omega_star": np.nan})
            continue
        for index, omega in enumerate(roots):
            data = response_data(n, omega)
            rows.append({"N": n, "crossing_index": index, "omega_star": omega, **data})

    out = RES_DIR / "multiband_crossing_boundary_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "N", "crossing_index", "omega_star", "signature", "Pi_direct",
                "Pi_projected", "sigma_min_C", "Pi_boundary_bound",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    positive_roots = [row for row in rows if row.get("crossing_index", -1) >= 0 and row["omega_star"] > 0]
    dense_errors = []
    for n in (24, 40, 80):
        row = next(row for row in positive_roots if row["N"] == n)
        dense = response_data(n, row["omega_star"], dense=True)
        dense_errors.append(abs(dense["Pi_direct"] - row["Pi_direct"]))
    if max(dense_errors) > 1e-9:
        raise RuntimeError(f"dense/sparse validation failed: {max(dense_errors):.3e}")

    roots_only = [row for row in rows if row.get("crossing_index", -1) >= 0]
    print(f"crossing records: {len(roots_only)}; positive roots: {len(positive_roots)}")
    print(f"max |s_N(omega*)|: {max(abs(row['signature']) for row in roots_only):.3e}")
    print(f"min direct Pi_N: {min(row['Pi_direct'] for row in roots_only):.9f}")
    print(f"min sigma_min(C): {min(row['sigma_min_C'] for row in roots_only):.9f}")
    print(f"min pointwise boundary bound: {min(row['Pi_boundary_bound'] for row in roots_only):.9f}")
    print(f"max dense/sparse Pi mismatch: {max(dense_errors):.3e}")
    print(f"analytic all-frequency bound: sigma_min(C)>=0.6325, Pi_N>=2.00028125")
    print(out)


if __name__ == "__main__":
    main()
