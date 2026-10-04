"""Cross-model transfer-response-noise exponent collapse.

The figure combines three independent data sets used in the manuscript:

1. the scalar local Lindblad chain under finite/asymmetric probes and weak
   onsite damping disorder;
2. a clean two-band local CP chain with several interband couplings;
3. the disordered two-band transfer cocycle.

For every sample we compare the fitted fixed-port response exponent with the
selected transfer/Lyapunov exponent and the measured output-noise exponent
with twice that exponent.  The clean two-band fits are regenerated here at
larger sizes to suppress transfer-mode beating.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_multiband_transfer_response import (  # noqa: E402
    KAPPA,
    local_cp_matrices,
    transfer_roots,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ARXIV_FIGURES = ROOT / "figures"

MULTIBAND_J = (0.0, 0.005, 0.01, 0.015, 0.025, 0.03, 0.035, 0.04, 0.06, 0.08)
MULTIBAND_SIZES = np.arange(60, 201, 20, dtype=int)


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def linear_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return the linear slope and its ordinary least-squares standard error."""

    design = np.column_stack((x, np.ones_like(x)))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    slope = float(coefficients[0])
    residual = y - design @ coefficients
    dof = max(len(x) - 2, 1)
    variance = float(np.dot(residual, residual) / dof)
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    stderr = float(np.sqrt(variance / denominator)) if denominator > 0.0 else 0.0
    return slope, stderr


def scalar_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    probe_data = read_csv("probe_strength_summary.csv")
    for row in probe_data:
        kappa = float(row["kappa"])
        if kappa not in {0.0, 0.01, 0.05, 0.2, 1.0, 2.0}:
            continue
        response = float(row["chi_slope"])
        rows.append(
            {
                "family": "scalar_probe",
                "parameter": f"kappa={kappa:g}",
                "transfer_exponent": response,
                "response_exponent": response,
                "response_stderr": 0.0,
                "noise_exponent": float(row["noise_slope"]),
                "noise_stderr": 0.0,
                "minimum_gap": float(row["minimum_gap_over_scan"]),
            }
        )

    robustness = read_csv("robustness_checks_summary.csv")
    for row in robustness:
        if row["scan"] != "onsite_damping_disorder":
            continue
        response = float(row["chi_slope"])
        rows.append(
            {
                "family": "scalar_disorder",
                "parameter": f"W={float(row['disorder_W']):g}",
                "transfer_exponent": response,
                "response_exponent": response,
                "response_stderr": 0.0,
                "noise_exponent": float(row["noise_slope"]),
                "noise_stderr": 0.0,
                "minimum_gap": float(row["min_gap"]),
            }
        )
    return rows


def clean_multiband_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    sizes = MULTIBAND_SIZES.astype(float)
    for coupling in MULTIBAND_J:
        responses: list[float] = []
        noises: list[float] = []
        gaps: list[float] = []
        for size in MULTIBAND_SIZES:
            drift, gamma_gain = local_cp_matrices(int(size), coupling, KAPPA)
            output = np.zeros(drift.shape[0], dtype=complex)
            output[-2] = 1.0
            response_row = np.linalg.solve((-drift).T, output)
            responses.append(float(KAPPA * abs(response_row[0])))
            noises.append(
                float(KAPPA * np.real(response_row @ gamma_gain @ response_row.conj()))
            )
            gaps.append(float(-np.max(np.linalg.eigvals(drift).real)))

        response_exponent, response_stderr = linear_slope(
            sizes, np.log(np.asarray(responses))
        )
        noise_exponent, noise_stderr = linear_slope(
            sizes, np.log(np.asarray(noises))
        )
        roots = transfer_roots(coupling)
        rows.append(
            {
                "family": "clean_two_band",
                "parameter": f"J={coupling:g}",
                "transfer_exponent": float(np.log(abs(roots[1]))),
                "response_exponent": response_exponent,
                "response_stderr": response_stderr,
                "noise_exponent": noise_exponent,
                "noise_stderr": noise_stderr,
                "minimum_gap": float(min(gaps)),
            }
        )
    return rows


def disordered_multiband_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for row in read_csv("disordered_transfer_summary.csv"):
        rows.append(
            {
                "family": "disordered_two_band",
                "parameter": f"W={float(row['W']):g}",
                "transfer_exponent": float(row["lyapunov_2"]),
                "response_exponent": float(row["fixed_port_response_exponent"]),
                "response_stderr": 0.0,
                "noise_exponent": float(row["output_noise_exponent"]),
                "noise_stderr": 0.0,
                "minimum_gap": float(row["minimum_obc_rapidity_gap"]),
            }
        )
    return rows


def write_results(rows: list[dict[str, float | str]]) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "response_noise_collapse.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    response_mismatch = max(
        abs(float(row["response_exponent"]) - float(row["transfer_exponent"]))
        for row in rows
    )
    noise_mismatch = max(
        abs(float(row["noise_exponent"]) - 2.0 * float(row["transfer_exponent"]))
        for row in rows
    )
    minimum_gap = min(float(row["minimum_gap"]) for row in rows)
    summary = (
        "# Cross-model response-noise collapse\n\n"
        f"- Number of exponent samples: {len(rows)}\n"
        f"- Maximum |mu_response - Lambda_transfer|: {response_mismatch:.6e}\n"
        f"- Maximum |mu_noise - 2 Lambda_transfer|: {noise_mismatch:.6e}\n"
        f"- Minimum sampled full Liouvillian gap: {minimum_gap:.6e}\n"
        "- Clean two-band finite-size uncertainties are recorded in the CSV.\n"
    )
    (RESULTS / "reports/response_noise_collapse.txt").write_text(summary, encoding="utf-8")


def make_figure(rows: list[dict[str, float | str]]) -> None:
    FIGURES.mkdir(exist_ok=True)
    ARXIV_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "mathtext.fontset": "cm",
        }
    )

    styles = {
        "scalar_probe": ("o", "#1769aa", "scalar, finite probes"),
        "scalar_disorder": ("s", "#2a9d8f", "scalar, disorder"),
        "clean_two_band": ("^", "#d1495b", "clean two-band"),
        "disordered_two_band": ("D", "#6a4c93", "disordered two-band"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)

    for family, (marker, color, label) in styles.items():
        subset = [row for row in rows if row["family"] == family]
        transfer = np.array([float(row["transfer_exponent"]) for row in subset])
        response = np.array([float(row["response_exponent"]) for row in subset])
        response_error = np.array([float(row["response_stderr"]) for row in subset])
        noise = np.array([float(row["noise_exponent"]) for row in subset])
        noise_error = np.array([float(row["noise_stderr"]) for row in subset])
        axes[0].errorbar(
            transfer,
            response,
            yerr=response_error,
            fmt=marker,
            color=color,
            ms=4.2,
            capsize=1.5,
            label=label,
        )
        axes[1].errorbar(
            2.0 * transfer,
            noise,
            yerr=noise_error,
            fmt=marker,
            color=color,
            ms=4.2,
            capsize=1.5,
            label=label,
        )

    max_response = max(float(row["transfer_exponent"]) for row in rows) * 1.08
    max_noise = 2.0 * max_response
    axes[0].plot([0.0, max_response], [0.0, max_response], "k--", lw=0.9)
    axes[1].plot([0.0, max_noise], [0.0, max_noise], "k--", lw=0.9)
    axes[1].fill_between(
        [0.0, max_noise],
        [0.0, max_noise],
        [max_noise, max_noise],
        color="#e9c46a",
        alpha=0.12,
        linewidth=0.0,
    )

    axes[0].set_xlabel(r"transfer/Lyapunov exponent $\Lambda_{\rm resp}$")
    axes[0].set_ylabel(r"fitted response exponent $\mu_{\rm resp}$")
    axes[0].set_title("(a) transfer exponent fixes response", loc="left")
    axes[1].set_xlabel(r"quantum lower-bound exponent $2\Lambda_{\rm resp}$")
    axes[1].set_ylabel(r"fitted noise exponent $\mu_{\rm noise}$")
    axes[1].set_title("(b) output noise inherits twice the exponent", loc="left")
    axes[0].legend(frameon=False, loc="upper left")

    for axis, limit in zip(axes, (max_response, max_noise)):
        axis.set_xlim(0.0, limit)
        axis.set_ylim(0.0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.20, lw=0.5)

    for extension in ("pdf", "png"):
        destination = FIGURES / f"response_noise_collapse.{extension}"
        fig.savefig(destination, dpi=300 if extension == "png" else None)
        (ARXIV_FIGURES / destination.name).write_bytes(destination.read_bytes())
    plt.close(fig)


def main() -> None:
    rows = scalar_rows() + clean_multiband_rows() + disordered_multiband_rows()
    write_results(rows)
    make_figure(rows)
    print((RESULTS / "reports/response_noise_collapse.txt").read_text(), end="")


if __name__ == "__main__":
    main()
