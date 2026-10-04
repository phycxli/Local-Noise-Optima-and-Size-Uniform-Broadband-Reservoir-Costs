"""Two-band check of the transfer-root response/noise theorem.

Each unit cell contains two bosonic modes.  The two orbitals have different
nonreciprocal bond reservoirs and are coupled coherently within each cell.
The model is therefore a local CP Gaussian Lindblad chain with genuinely
matrix-valued, noncommuting onsite and hopping blocks.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

T_RIGHT = np.array([1.0, 0.8])
T_LEFT = np.array([0.25, 0.35])
GAMMA = np.array([1.2, 1.15])
J_WORK = 0.01
KAPPA = 0.2
J_VALUES = np.linspace(0.0, 0.18, 37)
FIT_SIZES = np.array([30, 40, 50, 60, 70, 80], dtype=int)
NOISE_SIZES = np.arange(20, 101, 10, dtype=int)


def bulk_blocks(j_mix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return onsite, superdiagonal, and subdiagonal drift blocks."""

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    x_zero = -np.diag(GAMMA).astype(complex) - 1j * j_mix * sigma_x
    x_plus = np.diag(T_LEFT).astype(complex)
    x_minus = np.diag(T_RIGHT).astype(complex)
    return x_zero, x_plus, x_minus


def transfer_roots(j_mix: float, omega: float = 0.0) -> np.ndarray:
    """Return roots z of det[i omega - X(z)] sorted by modulus."""

    x_zero, x_plus, x_minus = bulk_blocks(j_mix)
    a_zero = 1j * omega * np.eye(2) - x_zero
    a_plus = -x_plus
    a_minus = -x_minus
    transfer = np.block(
        [
            [-np.linalg.solve(a_plus, a_zero), -np.linalg.solve(a_plus, a_minus)],
            [np.eye(2), np.zeros((2, 2))],
        ]
    )
    roots = np.linalg.eigvals(transfer)
    return roots[np.argsort(np.abs(roots))]


def endpoint_green_block(n_cells: int, j_mix: float, omega: float = 0.0) -> np.ndarray:
    """Compute the right-to-left endpoint Green block by block LDU recursion."""

    x_zero, x_plus, x_minus = bulk_blocks(j_mix)
    a_zero = 1j * omega * np.eye(2) - x_zero
    a_plus = -x_plus
    a_minus = -x_minus

    schur = a_zero.copy()
    propagated_rhs = np.eye(2, dtype=complex)
    for _ in range(1, n_cells):
        schur_inverse = np.linalg.inv(schur)
        propagated_rhs = -a_minus @ schur_inverse @ propagated_rhs
        schur = a_zero - a_minus @ schur_inverse @ a_plus
    return np.linalg.solve(schur, propagated_rhs)


def finite_drift(n_cells: int, j_mix: float) -> np.ndarray:
    x_zero, x_plus, x_minus = bulk_blocks(j_mix)
    drift = np.zeros((2 * n_cells, 2 * n_cells), dtype=complex)
    for cell in range(n_cells):
        sl = slice(2 * cell, 2 * cell + 2)
        drift[sl, sl] = x_zero
        if cell + 1 < n_cells:
            sr = slice(2 * cell + 2, 2 * cell + 4)
            drift[sl, sr] = x_plus
            drift[sr, sl] = x_minus
    return drift


def local_cp_matrices(
    n_cells: int, j_mix: float, kappa: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the probed drift and gain matrix of the local CP realization."""

    dim = 2 * n_cells
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    gamma_gain = np.zeros((dim, dim), dtype=complex)
    gamma_loss = np.zeros((dim, dim), dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    for cell in range(n_cells):
        sl = slice(2 * cell, 2 * cell + 2)
        hamiltonian[sl, sl] += j_mix * sigma_x
        for orbital in range(2):
            site = 2 * cell + orbital
            gamma_loss[site, site] += 2.0 * GAMMA[orbital]

    for cell in range(n_cells - 1):
        for orbital in range(2):
            left = 2 * cell + orbital
            right = 2 * (cell + 1) + orbital
            c_rate = 0.5 * (T_RIGHT[orbital] + T_LEFT[orbital])
            velocity = 0.5 * (T_RIGHT[orbital] - T_LEFT[orbital])
            hamiltonian[right, left] += 1j * velocity
            hamiltonian[left, right] += -1j * velocity

            gain = np.zeros(dim, dtype=complex)
            gain[left] = np.sqrt(c_rate)
            gain[right] = np.sqrt(c_rate)
            gamma_gain += np.outer(gain, gain.conj())

            loss = np.zeros(dim, dtype=complex)
            loss[left] = np.sqrt(c_rate)
            loss[right] = -np.sqrt(c_rate)
            gamma_loss += np.outer(loss, loss.conj())

    drift = -1j * hamiltonian + 0.5 * (gamma_gain - gamma_loss)
    if kappa:
        drift[0, 0] -= 0.5 * kappa
        drift[-2, -2] -= 0.5 * kappa
    return drift, gamma_gain


def fit_exponent(sizes: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(sizes.astype(float), np.log(values), 1)[0])


def scan_transfer() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for j_mix in J_VALUES:
        roots = transfer_roots(float(j_mix))
        transfer_exponent = float(np.log(abs(roots[1])))
        root_separation = float(np.log(abs(roots[2]) / abs(roots[1])))
        responses = np.array(
            [np.linalg.norm(endpoint_green_block(int(n), float(j_mix)), 2) for n in FIT_SIZES]
        )
        direct_exponent = fit_exponent(FIT_SIZES, responses)
        drift = finite_drift(60, float(j_mix))
        gap = float(-np.max(np.linalg.eigvals(drift).real))
        rows.append(
            {
                "J": float(j_mix),
                "transfer_exponent_log_abs_z_d": transfer_exponent,
                "direct_endpoint_response_exponent": direct_exponent,
                "root_separation_log_abs_z_dplus1_over_z_d": root_separation,
                "obc_rapidity_gap_N60": gap,
            }
        )
    return rows


def scan_noise() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for n_cells in NOISE_SIZES:
        drift, gamma_gain = local_cp_matrices(int(n_cells), J_WORK, KAPPA)
        chi = np.linalg.inv(-drift)
        transfer_amplitude = float(KAPPA * abs(chi[-2, 0]))
        output_noise = float(
            KAPPA * np.real(chi[-2, :] @ gamma_gain @ chi[-2, :].conj())
        )
        quantum_bound = transfer_amplitude**2 - 1.0
        rows.append(
            {
                "N": float(n_cells),
                "transfer_amplitude": transfer_amplitude,
                "output_gain_noise": output_noise,
                "universal_cp_bound": quantum_bound,
                "bound_residual": output_noise - quantum_bound,
                "obc_rapidity_gap": float(-np.max(np.linalg.eigvals(drift).real)),
            }
        )
    return rows


def write_results(
    transfer_rows: list[dict[str, float]], noise_rows: list[dict[str, float]]
) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "multiband_transfer_scan.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transfer_rows[0]))
        writer.writeheader()
        writer.writerows(transfer_rows)
    with (RES_DIR / "multiband_response_noise.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(noise_rows[0]))
        writer.writeheader()
        writer.writerows(noise_rows)

    roots = transfer_roots(J_WORK)
    mu = float(np.log(abs(roots[1])))
    direct = fit_exponent(
        FIT_SIZES,
        np.array([np.linalg.norm(endpoint_green_block(int(n), J_WORK), 2) for n in FIT_SIZES]),
    )
    noise_slope = fit_exponent(
        NOISE_SIZES,
        np.array([row["output_gain_noise"] for row in noise_rows]),
    )
    max_difference = max(
        abs(row["transfer_exponent_log_abs_z_d"] - row["direct_endpoint_response_exponent"])
        for row in transfer_rows
    )
    summary = (
        "# Multiband transfer-response verification\n\n"
        f"- Working interband coupling: J = {J_WORK}\n"
        f"- Transfer-root exponent log|z_2|: {mu:.8f}\n"
        f"- Direct endpoint-block exponent: {direct:.8f}\n"
        f"- Output-noise exponent: {noise_slope:.8f}\n"
        f"- Twice the transfer exponent: {2.0 * mu:.8f}\n"
        f"- Maximum transfer/direct exponent mismatch in scan: {max_difference:.3e}\n"
        f"- Minimum universal-bound residual: {min(row['bound_residual'] for row in noise_rows):.6g}\n"
        f"- Minimum OBC rapidity gap: {min(row['obc_rapidity_gap'] for row in noise_rows):.6g}\n"
    )
    (RES_DIR / "reports/multiband_transfer_response.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


def make_figure(
    transfer_rows: list[dict[str, float]], noise_rows: list[dict[str, float]]
) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.25), constrained_layout=True)

    j_values = np.array([row["J"] for row in transfer_rows])
    transfer_mu = np.array(
        [row["transfer_exponent_log_abs_z_d"] for row in transfer_rows]
    )
    direct_mu = np.array(
        [row["direct_endpoint_response_exponent"] for row in transfer_rows]
    )
    gaps = np.array([row["obc_rapidity_gap_N60"] for row in transfer_rows])
    separations = np.array(
        [row["root_separation_log_abs_z_dplus1_over_z_d"] for row in transfer_rows]
    )

    ax = axes[0]
    ax.plot(j_values, transfer_mu, color="#1769aa", lw=2, label=r"$\log|z_2|$")
    ax.plot(j_values, direct_mu, "o", color="#d1495b", ms=3.2, label="direct fit")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.axvline(J_WORK, color="#444444", ls="--", lw=0.9)
    ax.fill_between(j_values, 0.0, transfer_mu, where=transfer_mu > 0, color="#e9c46a", alpha=0.35)
    ax.set_xlabel(r"interband coupling $J$")
    ax.set_ylabel("response exponent")
    ax.set_title("(a) transfer root fixes response")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(j_values, gaps, color="#2a9d8f", lw=2, label="OBC rapidity gap")
    ax.plot(j_values, separations, color="#6a4c93", lw=1.8, label=r"$\log|z_3/z_2|$")
    ax.axvline(J_WORK, color="#444444", ls="--", lw=0.9)
    ax.set_xlabel(r"interband coupling $J$")
    ax.set_ylabel("gap / root separation")
    ax.set_title("(b) response transition stays gapped")
    ax.legend(frameon=False)

    ax = axes[2]
    sizes = np.array([row["N"] for row in noise_rows])
    response = np.array([row["transfer_amplitude"] for row in noise_rows])
    noise = np.array([row["output_gain_noise"] for row in noise_rows])
    quantum_bound = np.array([row["universal_cp_bound"] for row in noise_rows])
    ax.semilogy(sizes, response, "o-", color="#1769aa", label=r"$|G_{o\leftarrow i}|$")
    ax.semilogy(sizes, noise, "s-", color="#d1495b", label=r"$S_{\rm out}^{\rm exc}$")
    positive = quantum_bound > 0.0
    ax.semilogy(
        sizes[positive],
        quantum_bound[positive],
        "^-",
        color="#2a9d8f",
        label=r"$|G|^2-1$",
    )
    ax.set_xlabel("number of cells $N$")
    ax.set_ylabel("measured value")
    ax.set_title("(c) CP noise inherits twice the exponent")
    ax.legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.25)
    for extension in ("png", "pdf"):
        destination = FIG_DIR / f"multiband_transfer_response.{extension}"
        fig.savefig(destination, dpi=300 if extension == "png" else None)
        arxiv_destination = ARXIV_FIG_DIR / destination.name
        fig.savefig(arxiv_destination, dpi=300 if extension == "png" else None)
    plt.close(fig)


def main() -> None:
    transfer_rows = scan_transfer()
    noise_rows = scan_noise()
    write_results(transfer_rows, noise_rows)
    make_figure(transfer_rows, noise_rows)


if __name__ == "__main__":
    main()
