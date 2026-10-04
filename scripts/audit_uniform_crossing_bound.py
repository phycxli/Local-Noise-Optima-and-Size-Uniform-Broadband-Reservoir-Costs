"""Audit the scalar-chain uniform crossing bound used in the supplement."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"


def scalar_drift(n: int, t_r: float = 1.0, t_l: float = 0.25, gamma: float = 1.2):
    x = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(x, -gamma)
    for j in range(n - 1):
        x[j + 1, j] = t_r
        x[j, j + 1] = t_l
    return x


def crossing_data(n: int, omega: float, kappa_in: float = 0.2):
    x = scalar_drift(n)
    x_kappa = x.copy()
    x_kappa[0, 0] -= kappa_in / 2.0
    a = 1j * omega * np.eye(n) - x_kappa
    y = np.linalg.solve(a.conj().T, np.eye(n)[:, -1])
    m = x + x.conj().T
    unit = y / np.linalg.norm(y)
    signature = float(np.real(np.vdot(unit, m @ unit)))
    transverse = m @ unit - signature * unit
    prefactor = float(np.vdot(y, y).real / (kappa_in * abs(y[0]) ** 2))
    pi_value = prefactor * float(np.vdot(transverse, transverse).real)
    return signature, pi_value


def main() -> None:
    roots_path = RES_DIR / "rate_noise_tradeoff_roots.csv"
    with roots_path.open(newline="", encoding="utf-8") as handle:
        roots = list(csv.DictReader(handle))

    rows = []
    for row in roots:
        n = int(float(row["N"]))
        omega = float(row["omega_star"])
        signature, pi_value = crossing_data(n, omega)
        rows.append({"N": n, "omega_star": omega, "signature": signature, "Pi_N": pi_value})

    # From the first boundary row of the adjoint resolvent,
    # Pi_N >= | -2 gamma + (t_R+t_L)(gamma+kappa_in/2-i omega)/t_R |^2/kappa_in.
    t_r, t_l, gamma, kappa_in = 1.0, 0.25, 1.2, 0.2
    uniform_bound = (
        abs(-2.0 * gamma + (t_r + t_l) / t_r * (gamma + kappa_in / 2.0)) ** 2
        / kappa_in
    )
    for row in rows:
        if row["Pi_N"] + 1e-10 < uniform_bound:
            raise AssertionError(f"bound failed at N={row['N']}")

    out = RES_DIR / "finite_rate_crossing_uniform_bound.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["N", "omega_star", "signature", "Pi_N"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"uniform Pi_N lower bound = {uniform_bound:.9f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
