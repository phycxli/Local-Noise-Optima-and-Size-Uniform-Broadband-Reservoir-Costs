"""Audit a transfer-root/Rouche certificate for the finite-J two-band chain.

The calculation is deliberately a certificate audit, not an all-N theorem:
it tests a compact frequency interval and a finite J interval.  The contour is
the geometric-mean circle between the second and third transfer roots at J=0.
If the sampled Rouche margin stays positive, the root count is stable on that
grid and provides the numerical constants needed for an analytic continuation
argument.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"

T_RIGHT = np.diag([1.0, 0.8]).astype(complex)
T_LEFT = np.diag([0.25, 0.35]).astype(complex)
GAMMA = np.array([1.2, 1.15])
OMEGA_GRID = np.linspace(-2.0, 2.0, 401)
J_GRID = np.linspace(0.0, 0.05, 21)
THETA_GRID = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
KAPPA = 0.2
PORT = np.array([1.0, 0.0], dtype=complex)


def blocks(j_mix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    onsite = -np.diag(GAMMA).astype(complex) - 1j * j_mix * sigma_x
    # These are the upper and lower blocks of the drift convention used by
    # run_multiband_transfer_response.py.
    return onsite, np.diag([0.25, 0.35]).astype(complex), T_RIGHT.copy()


def transfer(j_mix: float, omega: float) -> np.ndarray:
    onsite, upper, lower = blocks(j_mix)
    a0 = 1j * omega * np.eye(2) - onsite
    ap = -upper
    am = -lower
    return np.block(
        [
            [-np.linalg.solve(ap, a0), -np.linalg.solve(ap, am)],
            [np.eye(2), np.zeros((2, 2), dtype=complex)],
        ]
    )


def sorted_roots(j_mix: float, omega: float) -> np.ndarray:
    roots = np.linalg.eigvals(transfer(j_mix, omega))
    return roots[np.argsort(np.abs(roots))]


def char_value(tmat: np.ndarray, z: complex) -> complex:
    return np.linalg.det(z * np.eye(4, dtype=complex) - tmat)


def boundary_matrix(j_mix: float, omega: float) -> np.ndarray:
    onsite, upper, lower = blocks(j_mix)
    probe = KAPPA / 2.0 * np.outer(PORT, PORT.conj())
    rhs_map = -1j * omega * np.eye(2) - onsite.conj().T + probe
    return onsite + onsite.conj().T + (upper + lower.conj().T) @ np.linalg.solve(
        lower.conj().T, rhs_map
    )


def main() -> None:
    rows: list[dict[str, float]] = []
    for omega in OMEGA_GRID:
        roots0 = sorted_roots(0.0, float(omega))
        r_inner = float(abs(roots0[1]))
        r_outer = float(abs(roots0[2]))
        contour_radius = float(np.sqrt(r_inner * r_outer))
        root_radii = np.abs(roots0)
        factor_bound = float(np.prod(np.abs(contour_radius - root_radii)))
        contour = contour_radius * np.exp(1j * THETA_GRID)
        t0 = transfer(0.0, float(omega))
        p0_abs = np.array([abs(char_value(t0, z)) for z in contour])
        m0 = float(np.min(p0_abs))
        for j_mix in J_GRID:
            roots = sorted_roots(float(j_mix), float(omega))
            gap = float(np.log(abs(roots[2]) / abs(roots[1])))
            response_mu = float(np.log(abs(roots[1])))
            t_j = transfer(float(j_mix), float(omega))
            if j_mix == 0.0:
                c = 0.0
                rouche_margin = m0
            else:
                delta = np.array(
                    [abs(char_value(t_j, z) - char_value(t0, z)) for z in contour]
                )
                c = float(np.max(delta) / (j_mix * j_mix))
                rouche_margin = float(m0 - np.max(delta))
            sigma_min = float(np.min(np.linalg.svd(boundary_matrix(float(j_mix), float(omega)), compute_uv=False)))
            rows.append(
                {
                    "omega": float(omega),
                    "J": float(j_mix),
                    "contour_radius": contour_radius,
                    "root_gap_log_abs_z3_over_z2": gap,
                    "response_exponent_log_abs_z2": response_mu,
                    "rouche_m0": m0,
                    "rouche_factor_bound": factor_bound,
                    "rouche_C": c,
                    "rouche_margin": rouche_margin,
                    "boundary_sigma_min": sigma_min,
                }
            )

    out = RES / "finite_J_transfer_uniform_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positive = [row for row in rows if row["rouche_margin"] > 0.0]
    summary = {
        "grid_points": len(rows),
        "min_root_gap": min(row["root_gap_log_abs_z3_over_z2"] for row in rows),
        "min_rouche_margin": min(row["rouche_margin"] for row in rows),
        "min_rouche_factor_bound": min(row["rouche_factor_bound"] for row in rows),
        "max_contour_C": max(row["contour_radius"] ** 2 / (0.25 * 0.35) for row in rows),
        "min_boundary_sigma": min(row["boundary_sigma_min"] for row in rows),
        "max_abs_response_exponent": max(abs(row["response_exponent_log_abs_z2"]) for row in rows),
        "positive_rouche_fraction": len(positive) / len(rows),
    }
    summary["conservative_J_threshold"] = float(
        np.sqrt(summary["min_rouche_factor_bound"] / summary["max_contour_C"])
    )
    summary_path = RES / "reports/finite_J_transfer_uniform_audit.txt"
    summary_path.write_text(
        "# Finite-J transfer-root uniformity audit\n\n"
        + "The audit uses a 401-point frequency grid on [-2,2], 21 values of J on [0,0.05], and a 721-point contour.\n\n"
        + "| quantity | value |\n|---|---:|\n"
        + "\n".join(f"| `{key}` | {value:.10g} |" if isinstance(value, float) else f"| `{key}` | {value} |" for key, value in summary.items())
        + "\n\nThe factor bound uses |z-r| >= ||z|-|r|| and is conservative. These are sampled numerical margins. They support, but do not replace, a proof of the corresponding uniform contour and boundary inequalities.\n",
        encoding="utf-8",
    )
    print(summary_path)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
