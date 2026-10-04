"""Audit exact finite-port response and noise bounds for asymmetric probes."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liouvillian_geometry import (  # noqa: E402
    LocalBosonicHatanoParams,
    local_bosonic_hatano_lindblad_matrices,
)


TR, TL, GAMMA = 1.0, 0.25, 1.2
PARAMS = LocalBosonicHatanoParams(t_right=TR, t_left=TL, gamma=GAMMA)
SIZES = (10, 20, 30, 40, 60, 80, 100)
PORT_PAIRS = ((0.03, 0.4), (0.2, 0.2), (0.7, 0.05))


def psd_factor(matrix: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(matrix)
    if values.min() < -tol:
        raise ValueError(f"non-PSD matrix, minimum eigenvalue={values.min()}")
    return vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None)))


def analytic_chi(n: int, kappa_i: float, kappa_o: float) -> float:
    s = np.sqrt(TR * TL)
    rho = np.sqrt(TR / TL)
    x = GAMMA / (2.0 * s)
    q = x + np.sqrt(x * x - 1.0)
    # U_n(x) = (q^(n+1)-q^(-n-1))/(q-q^-1), stable at the chosen x.
    u = lambda m: (q ** (m + 1) - q ** (-(m + 1))) / (q - q**-1) if m >= 0 else 0.0
    denominator = (
        s * u(n)
        + 0.5 * (kappa_i + kappa_o) * u(n - 1)
        + (kappa_i * kappa_o / (4.0 * s)) * u(n - 2)
    )
    return rho ** (n - 1) / denominator


def one_sample(n: int, kappa_i: float, kappa_o: float) -> dict[str, float]:
    h, gg, gl = local_bosonic_hatano_lindblad_matrices(n, PARAMS)
    eye = np.eye(n, dtype=complex)
    ei = np.zeros(n, dtype=complex)
    eo = np.zeros(n, dtype=complex)
    ei[0], eo[-1] = 1.0, 1.0
    x = -1j * h + 0.5 * (gg - gl)
    xk = x - 0.5 * kappa_i * np.outer(ei, ei) - 0.5 * kappa_o * np.outer(eo, eo)
    chi = np.linalg.solve(-xk, eye)
    chi_n1 = chi[-1, 0]
    chi_formula = analytic_chi(n, kappa_i, kappa_o)

    ci = np.column_stack((psd_factor(gl), np.sqrt(kappa_i) * ei, np.sqrt(kappa_o) * eo))
    cg = psd_factor(gg)
    smat = np.eye(ci.shape[1], dtype=complex) - ci.conj().T @ chi @ ci
    tmat = ci.conj().T @ chi @ cg
    idx_i, idx_o = ci.shape[1] - 2, ci.shape[1] - 1
    coherent_gain = smat[idx_o, idx_i]
    out_noise = float(np.real((tmat @ tmat.conj().T)[idx_o, idx_o]))
    cp_bound = abs(coherent_gain) ** 2 - 1.0
    comm_resid = float(np.linalg.norm(smat @ smat.conj().T - tmat @ tmat.conj().T - np.eye(ci.shape[1])))

    c = 0.5 * (TR + TL)
    bond = np.zeros(n, dtype=complex)
    bond[:2] = np.sqrt(c)
    local_channel_direct = kappa_o * abs(eo.conj() @ chi @ bond) ** 2
    local_channel_formula = (
        kappa_o * c * ((TR + GAMMA + kappa_i / 2.0) / TR) ** 2 * abs(chi_n1) ** 2
    )
    return {
        "N": n,
        "kappa_in": kappa_i,
        "kappa_out": kappa_o,
        "chi_direct": float(abs(chi_n1)),
        "chi_formula": float(abs(chi_formula)),
        "chi_relerr": float(abs(chi_n1 - chi_formula) / max(abs(chi_n1), 1e-300)),
        "cp_noise": out_noise,
        "cp_bound": cp_bound,
        "cp_margin": out_noise - cp_bound,
        "commutator_residual": comm_resid,
        "local_channel_direct": local_channel_direct,
        "local_channel_formula": local_channel_formula,
        "local_channel_relerr": float(abs(local_channel_direct - local_channel_formula) / max(local_channel_direct, 1e-300)),
    }


def main() -> None:
    rows = [one_sample(n, ki, ko) for ki, ko in PORT_PAIRS for n in SIZES]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / "asymmetric_finite_probe_transfer_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    max_chi = max(r["chi_relerr"] for r in rows)
    max_local = max(r["local_channel_relerr"] for r in rows)
    min_cp = min(r["cp_margin"] for r in rows)
    max_comm = max(r["commutator_residual"] for r in rows)
    print(f"samples={len(rows)}")
    print(f"max analytic/direct chi relative error={max_chi:.3e}")
    print(f"max local-channel formula relative error={max_local:.3e}")
    print(f"minimum CP noise-bound margin={min_cp:.6g}")
    print(f"maximum commutator identity residual={max_comm:.3e}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

