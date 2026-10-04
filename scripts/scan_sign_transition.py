"""Locate the finite-frequency sign transition of the fixed-drift optimum.

The normalized dissipative signature is

    s(omega) = u(omega)^dagger M u(omega),
    u = y/||y||,
    y = (i omega I - X_kappa)^(-dagger) e_out.

Zeros of s separate the loss-kernel and gain-kernel realizations.  The
script records the transverse coupling b=P M u and the universal rate lower
bound ||b||^2/|s| for any finite-rate CP realization that exactly attains the
pointwise optimum.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
KAPPA_IN = 0.2
KAPPA_OUT = 0.2
GAMMA = 1.2
T_RIGHT = 1.0
T_LEFT = 0.25


def scalar_drift(n: int) -> np.ndarray:
    x = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(x, -GAMMA)
    for j in range(n - 1):
        x[j + 1, j] = T_RIGHT
        x[j, j + 1] = T_LEFT
    return x


def signature(x: np.ndarray, omega: float) -> tuple[float, float, float]:
    n = len(x)
    output = n - 1
    probe = x.copy()
    probe[0, 0] -= KAPPA_IN / 2.0
    probe[output, output] -= KAPPA_OUT / 2.0
    e = np.zeros(n, dtype=complex)
    e[output] = 1.0
    y = np.linalg.solve((1j * omega * np.eye(n) - probe).conj().T, e)
    u = y / np.linalg.norm(y)
    m = x + x.conj().T
    s = float(np.real(np.vdot(u, m @ u)))
    transverse = m @ u - s * u
    return s, float(np.linalg.norm(transverse)), float(np.linalg.norm(y))


def root_in_interval(x: np.ndarray, lo: float, hi: float) -> float | None:
    flo = signature(x, lo)[0]
    fhi = signature(x, hi)[0]
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return None
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        fmid = signature(x, mid)[0]
        if flo * fmid <= 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def main() -> None:
    rows: list[dict[str, float | str]] = []
    sizes = (8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160)
    grid = np.linspace(0.0, 0.5, 501)
    for n in sizes:
        x = scalar_drift(n)
        values = [signature(x, float(w))[0] for w in grid]
        for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if fa == 0.0 or fa * fb < 0.0:
                root = root_in_interval(x, float(a), float(b))
                if root is not None:
                    s, transverse, norm_y = signature(x, root)
                    rows.append(
                        {
                            "N": n,
                            "omega_star": root,
                            "s_at_root": s,
                            "transverse_norm": transverse,
                            "response_norm": norm_y,
                        }
                    )
        for omega in (0.0, 0.05, 0.10, 0.15, 0.20):
            s, transverse, norm_y = signature(x, omega)
            rows.append(
                {
                    "N": n,
                    "omega_star": omega,
                    "s_at_root": s,
                    "transverse_norm": transverse,
                    "response_norm": norm_y,
                }
            )
    RES_DIR.mkdir(exist_ok=True)
    path = RES_DIR / "sign_transition_scan.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")
    print("sign-transition roots:")
    for row in rows:
        if abs(float(row["s_at_root"])) < 1.0e-8:
            lower = float(row["transverse_norm"]) ** 2 / max(abs(float(row["s_at_root"])), 1.0e-300)
            print(row["N"], row["omega_star"], row["transverse_norm"], lower)


if __name__ == "__main__":
    main()
