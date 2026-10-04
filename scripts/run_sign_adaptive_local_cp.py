"""Scan the sign-adaptive fixed-frequency local CP construction.

For a fixed drift M=X+X^dagger and response covector y, every compatible
factorization obeys

    y^dagger Gamma_g y >= max(y^dagger M y, 0).

The two local constructions tested here are

    s>0: Gamma_l=alpha K,     Gamma_g=M+alpha K,
    s<0: Gamma_g=alpha K,     Gamma_l=-M+alpha K,

where K is a nearest-neighbor PSD kernel with K y=0.  The script reports the
sign, the exact port-corrected resolvent identity, the required rate, and the
realized excess noise.
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


def two_band_drift(n_cells: int) -> np.ndarray:
    d = 2
    x = np.zeros((d * n_cells, d * n_cells), dtype=complex)
    onsite = np.array([[-1.0, -0.12j], [-0.12j, -0.9]], dtype=complex)
    right = np.diag([1.2, 1.0]).astype(complex)
    left = np.diag([0.10, 0.08]).astype(complex)
    for cell in range(n_cells):
        sl = slice(d * cell, d * (cell + 1))
        x[sl, sl] = onsite
    for cell in range(n_cells - 1):
        left_sl = slice(d * cell, d * (cell + 1))
        right_sl = slice(d * (cell + 1), d * (cell + 2))
        x[right_sl, left_sl] = right
        x[left_sl, right_sl] = left
    return x


def response_data(
    x: np.ndarray, omega: float, input_index: int, output_index: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    probe = x.copy()
    probe[input_index, input_index] -= KAPPA_IN / 2.0
    probe[output_index, output_index] -= KAPPA_OUT / 2.0
    eye = np.eye(len(x), dtype=complex)
    source = eye[:, output_index]
    y = np.linalg.solve((1j * omega * eye - probe).conj().T, source)
    m = x + x.conj().T
    denominator = KAPPA_IN * abs(y[input_index]) ** 2
    return y, m, float(np.real(np.vdot(y, m @ y)) / denominator), denominator


def scalar_kernel(y: np.ndarray) -> np.ndarray:
    kernel = np.zeros((len(y), len(y)), dtype=complex)
    for j in range(len(y) - 1):
        scale = np.sqrt(abs(y[j]) * abs(y[j + 1]))
        vector = np.zeros(len(y), dtype=complex)
        vector[j] = np.conj(y[j + 1]) / scale
        vector[j + 1] = -np.conj(y[j]) / scale
        kernel += np.outer(vector, vector.conj())
    return kernel


def block_kernel(y: np.ndarray, n_cells: int, d: int = 2) -> np.ndarray:
    blocks = [y[d * j : d * (j + 1)] for j in range(n_cells)]
    kernel = np.zeros((d * n_cells, d * n_cells), dtype=complex)
    for j, block in enumerate(blocks):
        sl = slice(d * j, d * (j + 1))
        kernel[sl, sl] += np.eye(d) - np.outer(block, block.conj()) / np.vdot(block, block)
    for j in range(n_cells - 1):
        left = blocks[j] / np.linalg.norm(blocks[j])
        right = blocks[j + 1] / np.linalg.norm(blocks[j + 1])
        vector = np.zeros(d * n_cells, dtype=complex)
        vector[d * j : d * (j + 1)] = np.sqrt(
            np.linalg.norm(blocks[j + 1]) / np.linalg.norm(blocks[j])
        ) * left
        vector[d * (j + 1) : d * (j + 2)] = -np.sqrt(
            np.linalg.norm(blocks[j]) / np.linalg.norm(blocks[j + 1])
        ) * right
        kernel += np.outer(vector, vector.conj())
    return kernel


def adaptive_pair(m: np.ndarray, kernel: np.ndarray, sign: float) -> tuple[float, np.ndarray, np.ndarray]:
    active_source = sign * m

    def minimum(alpha: float) -> float:
        return float(np.min(np.linalg.eigvalsh(active_source + alpha * kernel)))

    upper = 1.0
    while minimum(upper) < -1.0e-10 and upper < 1.0e14:
        upper *= 10.0
    if minimum(upper) < -1.0e-8:
        raise RuntimeError("No finite rate found for the local kernel")
    lower = 0.0
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if minimum(middle) >= 0.0:
            upper = middle
        else:
            lower = middle
    alpha = upper
    passive = alpha * kernel
    active = active_source + passive
    if sign > 0.0:
        gain, loss = active, passive
    else:
        gain, loss = passive, active
    return alpha, gain, loss


def one_row(model: str, size: int, omega: float) -> dict[str, float | str]:
    if model == "scalar":
        x = scalar_drift(size)
        input_index, output_index = 0, size - 1
        kernel_fn = lambda y: scalar_kernel(y)
    else:
        x = two_band_drift(size)
        input_index, output_index = 0, 2 * size - 2
        kernel_fn = lambda y: block_kernel(y, size)
    y, m, fixed, denominator = response_data(x, omega, input_index, output_index)
    sign = 1.0 if fixed > 1.0e-12 else -1.0 if fixed < -1.0e-12 else 0.0
    kernel = kernel_fn(y)
    if sign == 0.0:
        return {"model": model, "N": size, "omega": omega, "s": fixed, "branch": "boundary"}
    alpha, gain, loss = adaptive_pair(m, kernel, sign)
    unit = y / np.linalg.norm(y)
    s_unit = float(np.real(np.vdot(unit, m @ unit)))
    transverse = m @ unit - s_unit * unit
    transverse_norm = float(np.linalg.norm(transverse))
    resource_lower_bound = transverse_norm**2 / max(abs(s_unit), 1.0e-300)
    direct = float(np.real(np.vdot(y, m @ y)))
    predicted = float(
        -2.0 * np.real(y[output_index])
        + KAPPA_IN * abs(y[input_index]) ** 2
        + KAPPA_OUT * abs(y[output_index]) ** 2
    ) / denominator
    realized = float(np.real(np.vdot(y, gain @ y)) / denominator)
    return {
        "model": model,
        "N": size,
        "omega": omega,
        "s": fixed,
        "s_unit": s_unit,
        "transverse_norm": transverse_norm,
        "resource_lower_bound": resource_lower_bound,
        "branch": "loss-kernel" if sign > 0 else "gain-kernel",
        "alpha": alpha,
        "noise": realized,
        "lower_bound": max(fixed, 0.0),
        "identity_error": abs(fixed - predicted),
        "gain_min": float(np.min(np.linalg.eigvalsh(gain))),
        "loss_min": float(np.min(np.linalg.eigvalsh(loss))),
        "kernel_residual": float(np.linalg.norm(kernel @ y) / np.linalg.norm(y)),
        "rate": float(max(np.max(np.linalg.eigvalsh(gain)), np.max(np.linalg.eigvalsh(loss)))),
    }


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for model, sizes, omegas in (
        ("scalar", (4, 8, 12, 16, 20, 24, 32, 40, 64, 96, 128),
         (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)),
        ("two_band", (2, 3, 4, 6, 8, 10, 16), (0.0, 0.10, 0.15)),
    ):
        for size in sizes:
            for omega in omegas:
                rows.append(one_row(model, size, omega))
    RES_DIR.mkdir(exist_ok=True)
    path = RES_DIR / "sign_adaptive_local_cp.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if "alpha" in r]
    branches = {b: sum(r["branch"] == b for r in rows) for b in {r["branch"] for r in rows}}
    print(f"rows={len(rows)} valid={len(valid)} branches={branches}")
    print(f"max_identity_error={max(float(r['identity_error']) for r in valid):.3e}")
    print(f"max_psd_violation={max(max(0.0, -float(r['gain_min']), -float(r['loss_min'])) for r in valid):.3e}")
    print(f"max_kernel_residual={max(float(r['kernel_residual']) for r in valid):.3e}")
    print(f"min_realized_minus_bound={min(float(r['noise'])-float(r['lower_bound']) for r in valid):.3e}")
    print(f"min_rate_over_resource_lower_bound={min(float(r['rate'])/float(r['resource_lower_bound']) for r in valid):.6f}")
    for model in ("scalar", "two_band"):
        subset=[r for r in rows if r["model"]==model]
        print(model, "s_range", min(float(r["s"]) for r in subset), max(float(r["s"]) for r in subset))


if __name__ == "__main__":
    main()
