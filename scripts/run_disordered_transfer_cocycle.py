"""Disordered transfer-cocycle test of response and CP output-noise exponents.

The two-band local CP chain from ``run_multiband_transfer_response.py`` is
given independent onsite damping disorder in every cell and orbital.  A QR
cocycle computes the Oseledets spectrum, block-LDU recursion computes the
endpoint Green block, and the full local CP realization gives finite-probe
response and output gain noise.
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
J_MIX = 0.01
KAPPA = 0.2
W_VALUES = np.array([0.0, 0.04, 0.08, 0.12, 0.16])
RESPONSE_SIZES = np.array([30, 50, 70, 90, 110, 130], dtype=int)
LYAPUNOV_LENGTH = 8000
LYAPUNOV_SAMPLES = 8
RESPONSE_SAMPLES = 80
CP_SAMPLES = 16
GAP_SIZE = 80
GAP_SAMPLES = 12

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
A_PLUS = -np.diag(T_LEFT).astype(complex)
A_MINUS = -np.diag(T_RIGHT).astype(complex)


def onsite_a_zero(disorder: np.ndarray, omega: float = 0.0) -> np.ndarray:
    return np.diag(GAMMA + disorder).astype(complex) + 1j * (
        omega * np.eye(2) + J_MIX * SIGMA_X
    )


def transfer_matrix(disorder: np.ndarray, omega: float = 0.0) -> np.ndarray:
    a_zero = onsite_a_zero(disorder, omega)
    return np.block(
        [
            [-np.linalg.solve(A_PLUS, a_zero), -np.linalg.solve(A_PLUS, A_MINUS)],
            [np.eye(2), np.zeros((2, 2))],
        ]
    )


def lyapunov_spectrum(disorder_strength: float, seed: int) -> np.ndarray:
    """Compute the four transfer-cocycle exponents by stabilized QR iteration."""

    rng = np.random.default_rng(seed)
    q_matrix = np.eye(4, dtype=complex)
    accumulated = np.zeros(4)
    for _ in range(LYAPUNOV_LENGTH):
        disorder = rng.uniform(-disorder_strength, disorder_strength, size=2)
        q_matrix, r_matrix = np.linalg.qr(
            transfer_matrix(disorder) @ q_matrix
        )
        accumulated += np.log(np.maximum(np.abs(np.diag(r_matrix)), 1.0e-300))
    return np.sort(accumulated / LYAPUNOV_LENGTH)


def endpoint_green_block(disorder: np.ndarray) -> np.ndarray:
    """Return the endpoint Green block for one onsite-disorder realization."""

    schur = onsite_a_zero(disorder[0])
    propagated_rhs = np.eye(2, dtype=complex)
    for onsite_disorder in disorder[1:]:
        schur_inverse = np.linalg.inv(schur)
        propagated_rhs = -A_MINUS @ schur_inverse @ propagated_rhs
        schur = onsite_a_zero(onsite_disorder) - A_MINUS @ schur_inverse @ A_PLUS
    return np.linalg.solve(schur, propagated_rhs)


def finite_drift(disorder: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    n_cells = disorder.shape[0]
    drift = np.zeros((2 * n_cells, 2 * n_cells), dtype=complex)
    for cell in range(n_cells):
        sl = slice(2 * cell, 2 * cell + 2)
        drift[sl, sl] = -np.diag(GAMMA + disorder[cell]) - 1j * J_MIX * SIGMA_X
        if cell + 1 < n_cells:
            sr = slice(2 * cell + 2, 2 * cell + 4)
            drift[sl, sr] = np.diag(T_LEFT)
            drift[sr, sl] = np.diag(T_RIGHT)
    if kappa:
        drift[0, 0] -= 0.5 * kappa
        drift[-2, -2] -= 0.5 * kappa
    return drift


def gain_noise_from_output_row(output_row: np.ndarray) -> float:
    """Evaluate y Gamma_g y^dagger from the local bond-gain channels."""

    n_cells = output_row.size // 2
    total = 0.0
    for cell in range(n_cells - 1):
        for orbital in range(2):
            c_rate = 0.5 * (T_RIGHT[orbital] + T_LEFT[orbital])
            left = 2 * cell + orbital
            right = 2 * (cell + 1) + orbital
            total += c_rate * abs(output_row[left] + output_row[right]) ** 2
    return float(total)


def cp_observables(disorder: np.ndarray) -> tuple[float, float, float]:
    drift = finite_drift(disorder, KAPPA)
    output = np.zeros(drift.shape[0], dtype=complex)
    output[-2] = 1.0
    output_row = np.linalg.solve((-drift).T, output)
    transfer_amplitude = float(KAPPA * abs(output_row[0]))
    output_noise = float(KAPPA * gain_noise_from_output_row(output_row))
    bound_residual = output_noise - (transfer_amplitude**2 - 1.0)
    return transfer_amplitude, output_noise, bound_residual


def fit_slope(sizes: np.ndarray, mean_logs: np.ndarray) -> float:
    return float(np.polyfit(sizes.astype(float), mean_logs, 1)[0])


def realization_seed(w_index: int, n_cells: int, sample: int, offset: int) -> int:
    return offset + 100000 * w_index + 1000 * n_cells + sample


def run_scan() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    summary_rows: list[dict[str, float]] = []
    scaling_rows: list[dict[str, float]] = []

    for w_index, disorder_strength in enumerate(W_VALUES):
        lyapunov_samples = np.array(
            [
                lyapunov_spectrum(
                    float(disorder_strength),
                    realization_seed(w_index, 0, sample, 100),
                )
                for sample in range(LYAPUNOV_SAMPLES)
            ]
        )
        lyapunov_mean = np.mean(lyapunov_samples, axis=0)
        lyapunov_std = np.std(lyapunov_samples, axis=0, ddof=1)

        block_mean_logs: list[float] = []
        cp_response_mean_logs: list[float] = []
        cp_noise_mean_logs: list[float] = []
        minimum_bound_residual = np.inf

        for n_cells in RESPONSE_SIZES:
            block_logs = []
            for sample in range(RESPONSE_SAMPLES):
                rng = np.random.default_rng(
                    realization_seed(w_index, int(n_cells), sample, 200)
                )
                disorder = rng.uniform(
                    -disorder_strength, disorder_strength, size=(n_cells, 2)
                )
                block_logs.append(np.log(np.linalg.norm(endpoint_green_block(disorder), 2)))

            response_logs = []
            noise_logs = []
            bound_residuals = []
            for sample in range(CP_SAMPLES):
                rng = np.random.default_rng(
                    realization_seed(w_index, int(n_cells), sample, 300)
                )
                disorder = rng.uniform(
                    -disorder_strength, disorder_strength, size=(n_cells, 2)
                )
                response, noise, residual = cp_observables(disorder)
                response_logs.append(np.log(response))
                noise_logs.append(np.log(noise))
                bound_residuals.append(residual)

            block_mean = float(np.mean(block_logs))
            cp_response_mean = float(np.mean(response_logs))
            cp_noise_mean = float(np.mean(noise_logs))
            block_mean_logs.append(block_mean)
            cp_response_mean_logs.append(cp_response_mean)
            cp_noise_mean_logs.append(cp_noise_mean)
            minimum_bound_residual = min(minimum_bound_residual, min(bound_residuals))
            scaling_rows.append(
                {
                    "W": float(disorder_strength),
                    "N": float(n_cells),
                    "mean_log_endpoint_block_norm": block_mean,
                    "stderr_log_endpoint_block_norm": float(
                        np.std(block_logs, ddof=1) / np.sqrt(RESPONSE_SAMPLES)
                    ),
                    "mean_log_port_response": cp_response_mean,
                    "stderr_log_port_response": float(
                        np.std(response_logs, ddof=1) / np.sqrt(CP_SAMPLES)
                    ),
                    "mean_log_output_noise": cp_noise_mean,
                    "stderr_log_output_noise": float(
                        np.std(noise_logs, ddof=1) / np.sqrt(CP_SAMPLES)
                    ),
                    "minimum_cp_bound_residual": float(min(bound_residuals)),
                }
            )

        gap_values = []
        for sample in range(GAP_SAMPLES):
            rng = np.random.default_rng(
                realization_seed(w_index, GAP_SIZE, sample, 400)
            )
            disorder = rng.uniform(
                -disorder_strength, disorder_strength, size=(GAP_SIZE, 2)
            )
            drift = finite_drift(disorder, KAPPA)
            gap_values.append(float(-np.max(np.linalg.eigvals(drift).real)))

        summary_rows.append(
            {
                "W": float(disorder_strength),
                **{
                    f"lyapunov_{index + 1}": float(lyapunov_mean[index])
                    for index in range(4)
                },
                **{
                    f"lyapunov_{index + 1}_std": float(lyapunov_std[index])
                    for index in range(4)
                },
                "endpoint_block_response_exponent": fit_slope(
                    RESPONSE_SIZES, np.array(block_mean_logs)
                ),
                "fixed_port_response_exponent": fit_slope(
                    RESPONSE_SIZES, np.array(cp_response_mean_logs)
                ),
                "output_noise_exponent": fit_slope(
                    RESPONSE_SIZES, np.array(cp_noise_mean_logs)
                ),
                "root_separation_lyapunov_3_minus_2": float(
                    lyapunov_mean[2] - lyapunov_mean[1]
                ),
                "minimum_obc_rapidity_gap": float(np.min(gap_values)),
                "median_obc_rapidity_gap": float(np.median(gap_values)),
                "minimum_cp_bound_residual": float(minimum_bound_residual),
            }
        )
    return summary_rows, scaling_rows


def write_results(
    summary_rows: list[dict[str, float]], scaling_rows: list[dict[str, float]]
) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "disordered_transfer_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (RES_DIR / "disordered_transfer_scaling.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scaling_rows[0]))
        writer.writeheader()
        writer.writerows(scaling_rows)

    max_block_mismatch = max(
        abs(row["endpoint_block_response_exponent"] - row["lyapunov_2"])
        for row in summary_rows
    )
    max_noise_mismatch = max(
        abs(0.5 * row["output_noise_exponent"] - row["lyapunov_2"])
        for row in summary_rows
    )
    summary = (
        "# Disordered transfer-cocycle verification\n\n"
        f"- Disorder values: {', '.join(f'{w:.2f}' for w in W_VALUES)}\n"
        f"- Maximum `|endpoint exponent - Lambda_2|`: {max_block_mismatch:.3e}\n"
        f"- Maximum `|noise exponent / 2 - Lambda_2|`: {max_noise_mismatch:.3e}\n"
        f"- Minimum sampled OBC rapidity gap: {min(row['minimum_obc_rapidity_gap'] for row in summary_rows):.6g}\n"
        f"- Minimum CP noise-bound residual: {min(row['minimum_cp_bound_residual'] for row in summary_rows):.6g}\n"
    )
    (RES_DIR / "reports/disordered_transfer_cocycle.txt").write_text(
        summary, encoding="utf-8"
    )
    print(summary, end="")


def make_figure(
    summary_rows: list[dict[str, float]], scaling_rows: list[dict[str, float]]
) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    w_values = np.array([row["W"] for row in summary_rows])
    lyapunov = np.array(
        [[row[f"lyapunov_{index}"] for row in summary_rows] for index in range(1, 5)]
    )

    ax = axes[0, 0]
    colors = ["#8d99ae", "#1769aa", "#6a4c93", "#d1495b"]
    for index in range(4):
        ax.plot(
            w_values,
            lyapunov[index],
            "o-",
            color=colors[index],
            lw=1.8 if index == 1 else 1.2,
            ms=3.5,
            label=rf"$\Lambda_{index + 1}$",
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("onsite damping disorder $W$")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title("(a) transfer-cocycle spectrum")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    block_exp = np.array(
        [row["endpoint_block_response_exponent"] for row in summary_rows]
    )
    port_exp = np.array(
        [row["fixed_port_response_exponent"] for row in summary_rows]
    )
    noise_half = 0.5 * np.array(
        [row["output_noise_exponent"] for row in summary_rows]
    )
    ax.plot(w_values, lyapunov[1], "o-", color="#1769aa", label=r"$\Lambda_2$")
    ax.plot(w_values, block_exp, "s--", color="#d1495b", label="endpoint block")
    ax.plot(w_values, port_exp, "^--", color="#e76f51", label="fixed port")
    ax.plot(w_values, noise_half, "D-.", color="#2a9d8f", label="noise exponent / 2")
    ax.set_xlabel("onsite damping disorder $W$")
    ax.set_ylabel("response-rate estimate")
    ax.set_title(r"(b) response and noise follow $\Lambda_2$")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    minimum_gap = np.array(
        [row["minimum_obc_rapidity_gap"] for row in summary_rows]
    )
    median_gap = np.array(
        [row["median_obc_rapidity_gap"] for row in summary_rows]
    )
    separation = np.array(
        [row["root_separation_lyapunov_3_minus_2"] for row in summary_rows]
    )
    ax.plot(w_values, minimum_gap, "o-", color="#d1495b", label="minimum OBC gap")
    ax.plot(w_values, median_gap, "s-", color="#2a9d8f", label="median OBC gap")
    ax.set_xlabel("onsite damping disorder $W$")
    ax.set_ylabel("OBC rapidity gap")
    ax2 = ax.twinx()
    ax2.plot(w_values, separation, "^-", color="#6a4c93", label=r"$\Lambda_3-\Lambda_2$")
    ax2.set_ylabel("cocycle separation")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax.set_title("(c) sampled chains remain gapped")

    ax = axes[1, 1]
    target_w = float(W_VALUES[-1])
    target_rows = [row for row in scaling_rows if abs(row["W"] - target_w) < 1.0e-12]
    sizes = np.array([row["N"] for row in target_rows])
    block_rates = np.array(
        [row["mean_log_endpoint_block_norm"] / row["N"] for row in target_rows]
    )
    noise_rates = np.array(
        [0.5 * row["mean_log_output_noise"] / row["N"] for row in target_rows]
    )
    lambda_target = summary_rows[-1]["lyapunov_2"]
    ax.plot(1.0 / sizes, block_rates, "s-", color="#d1495b", label=r"$\langle\log\|G\|\rangle/N$")
    ax.plot(1.0 / sizes, noise_rates, "D-", color="#2a9d8f", label=r"$\langle\log S\rangle/(2N)$")
    ax.axhline(lambda_target, color="#1769aa", lw=1.8, label=r"$\Lambda_2$")
    ax.set_xlabel(r"inverse size $1/N$")
    ax.set_ylabel("finite-size rate")
    ax.set_title(rf"(d) convergence at $W={target_w:.2f}$")
    ax.legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.23)
    for extension in ("png", "pdf"):
        destination = FIG_DIR / f"disordered_transfer_cocycle.{extension}"
        fig.savefig(destination, dpi=300 if extension == "png" else None)
        fig.savefig(
            ARXIV_FIG_DIR / destination.name,
            dpi=300 if extension == "png" else None,
        )
    plt.close(fig)


def main() -> None:
    summary_rows, scaling_rows = run_scan()
    write_results(summary_rows, scaling_rows)
    make_figure(summary_rows, scaling_rows)


if __name__ == "__main__":
    main()
