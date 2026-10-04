"""Robustness checks beyond the symmetric clean Toeplitz chain.

Two small tests address the most likely "fine tuning" objection:

1. asymmetric input-output probes, kappa_in != kappa_out;
2. weak onsite damping disorder, gamma_j = gamma + delta_j, which preserves a
   strictly local completely positive Lindblad realization as long as
   gamma_j >= 0.

The asymmetric-probe part is backed by the exact endpoint determinant formula.
The disorder part is numerical and reports medians over disorder realizations.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from liouvillian_geometry import (  # noqa: E402
    LocalBosonicHatanoParams,
    local_bosonic_hatano_lindblad_matrices,
)


FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)
SIZES = np.array([20, 30, 40, 50, 60, 80, 100, 120], dtype=int)
ASYM_PROBES = [
    (0.01, 1.0),
    (0.05, 0.5),
    (0.2, 1.0),
    (1.0, 0.2),
    (1.0, 1.0),
    (2.0, 0.05),
]
DISORDER_STRENGTHS = [0.0, 0.02, 0.05, 0.08]
DISORDER_SAMPLES = 80
DISORDER_PROBE = (0.2, 0.2)
RNG_SEED = 20260704


def gain_matrix(n_sites: int) -> np.ndarray:
    _, gamma_gain, _ = local_bosonic_hatano_lindblad_matrices(n_sites, PARAMS)
    return gamma_gain.real


def drift(
    n_sites: int,
    gamma_diag: np.ndarray | None = None,
    kappa_in: float = 0.0,
    kappa_out: float = 0.0,
) -> np.ndarray:
    if gamma_diag is None:
        gamma_diag = PARAMS.gamma * np.ones(n_sites)
    x = np.zeros((n_sites, n_sites), dtype=complex)
    np.fill_diagonal(x, -gamma_diag)
    for j in range(n_sites - 1):
        x[j + 1, j] = PARAMS.t_right
        x[j, j + 1] = PARAMS.t_left
    x[0, 0] -= 0.5 * kappa_in
    x[-1, -1] -= 0.5 * kappa_out
    return x


def response_noise_values(
    n_sites: int,
    kappa_in: float,
    kappa_out: float,
    gamma_diag: np.ndarray | None = None,
) -> dict[str, float]:
    x = drift(n_sites, gamma_diag=gamma_diag, kappa_in=kappa_in, kappa_out=kappa_out)
    gamma_gain = gain_matrix(n_sites)
    edge = np.zeros(n_sites, dtype=complex)
    edge[-1] = 1.0
    y = np.linalg.solve((-x).T, edge)
    chi = float(abs(y[0]))
    noise = float(np.real(np.vdot(y, gamma_gain @ y)))
    local_gamma_1 = PARAMS.gamma if gamma_diag is None else float(gamma_diag[0])
    lower = PARAMS.bond_rate * (
        (PARAMS.t_right + local_gamma_1 + 0.5 * kappa_in) / PARAMS.t_right
    ) ** 2 * chi * chi
    gap = float(-np.max(np.linalg.eigvals(x).real))
    return {
        "chi": chi,
        "noise": noise,
        "noise_lower": lower,
        "gap": gap,
    }


def chebyshev_u(n_order: int, x_value: float) -> float:
    if n_order == -1:
        return 0.0
    if n_order == 0:
        return 1.0
    if n_order == 1:
        return 2.0 * x_value
    previous = 1.0
    current = 2.0 * x_value
    for _ in range(2, n_order + 1):
        previous, current = current, 2.0 * x_value * current - previous
    return current


def asymmetric_probe_chi_formula(n_sites: int, kappa_in: float, kappa_out: float) -> float:
    s = math.sqrt(PARAMS.t_right * PARAMS.t_left)
    rho = math.sqrt(PARAMS.t_right / PARAMS.t_left)
    x_value = PARAMS.gamma / (2.0 * s)
    alpha_l = 0.5 * kappa_in
    alpha_r = 0.5 * kappa_out
    denom = (
        s * chebyshev_u(n_sites, x_value)
        + (alpha_l + alpha_r) * chebyshev_u(n_sites - 1, x_value)
        + (alpha_l * alpha_r / s) * chebyshev_u(n_sites - 2, x_value)
    )
    return rho ** (n_sites - 1) / denom


def asymptotic_response_exponent() -> float:
    s = math.sqrt(PARAMS.t_right * PARAMS.t_left)
    rho = math.sqrt(PARAMS.t_right / PARAMS.t_left)
    x_value = PARAMS.gamma / (2.0 * s)
    lam = x_value + math.sqrt(x_value * x_value - 1.0)
    return math.log(rho / lam)


def fit_slope(sizes: np.ndarray, values: np.ndarray) -> float:
    mask = values > 0.0
    return float(np.polyfit(sizes[mask].astype(float), np.log(values[mask]), deg=1)[0])


def scan_asymmetric_probes() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    summary: list[dict[str, float]] = []
    for kappa_in, kappa_out in ASYM_PROBES:
        per_pair: list[dict[str, float]] = []
        for n_sites in SIZES:
            values = response_noise_values(n_sites, kappa_in, kappa_out)
            formula = asymmetric_probe_chi_formula(n_sites, kappa_in, kappa_out)
            row = {
                "scan": "asymmetric_probe",
                "N": float(n_sites),
                "kappa_in": kappa_in,
                "kappa_out": kappa_out,
                "disorder_W": 0.0,
                "sample": -1.0,
                "chi": values["chi"],
                "chi_formula": formula,
                "chi_formula_relative_error": abs(values["chi"] - formula)
                / max(abs(formula), 1.0e-300),
                "noise": values["noise"],
                "noise_lower": values["noise_lower"],
                "gap": values["gap"],
            }
            rows.append(row)
            per_pair.append(row)
        sizes = np.array([row["N"] for row in per_pair])
        chi = np.array([row["chi"] for row in per_pair])
        noise = np.array([row["noise"] for row in per_pair])
        lower = np.array([row["noise_lower"] for row in per_pair])
        summary.append(
            {
                "scan": "asymmetric_probe",
                "kappa_in": kappa_in,
                "kappa_out": kappa_out,
                "disorder_W": 0.0,
                "chi_slope": fit_slope(sizes, chi),
                "noise_slope": fit_slope(sizes, noise),
                "noise_lower_slope": fit_slope(sizes, lower),
                "max_formula_error": max(row["chi_formula_relative_error"] for row in per_pair),
                "min_noise_over_lower": float(np.min(noise / lower)),
                "min_gap": min(row["gap"] for row in per_pair),
            }
        )
    return rows, summary


def scan_disorder() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rng = np.random.default_rng(RNG_SEED)
    kappa_in, kappa_out = DISORDER_PROBE
    rows: list[dict[str, float]] = []
    summary: list[dict[str, float]] = []
    for disorder_w in DISORDER_STRENGTHS:
        med_rows: list[dict[str, float]] = []
        for n_sites in SIZES:
            samples: list[dict[str, float]] = []
            for sample in range(DISORDER_SAMPLES):
                gamma_diag = PARAMS.gamma + rng.uniform(-disorder_w, disorder_w, size=n_sites)
                values = response_noise_values(
                    n_sites,
                    kappa_in,
                    kappa_out,
                    gamma_diag=gamma_diag,
                )
                row = {
                    "scan": "onsite_damping_disorder",
                    "N": float(n_sites),
                    "kappa_in": kappa_in,
                    "kappa_out": kappa_out,
                    "disorder_W": disorder_w,
                    "sample": float(sample),
                    "chi": values["chi"],
                    "chi_formula": float("nan"),
                    "chi_formula_relative_error": float("nan"),
                    "noise": values["noise"],
                    "noise_lower": values["noise_lower"],
                    "gap": values["gap"],
                }
                rows.append(row)
                samples.append(row)
            med_rows.append(
                {
                    "N": float(n_sites),
                    "chi": float(np.median([row["chi"] for row in samples])),
                    "noise": float(np.median([row["noise"] for row in samples])),
                    "noise_lower": float(np.median([row["noise_lower"] for row in samples])),
                    "gap": float(np.median([row["gap"] for row in samples])),
                    "min_gap": float(np.min([row["gap"] for row in samples])),
                    "noise_over_lower": float(
                        np.min([row["noise"] / row["noise_lower"] for row in samples])
                    ),
                }
            )
        sizes = np.array([row["N"] for row in med_rows])
        chi = np.array([row["chi"] for row in med_rows])
        noise = np.array([row["noise"] for row in med_rows])
        lower = np.array([row["noise_lower"] for row in med_rows])
        summary.append(
            {
                "scan": "onsite_damping_disorder",
                "kappa_in": kappa_in,
                "kappa_out": kappa_out,
                "disorder_W": disorder_w,
                "chi_slope": fit_slope(sizes, chi),
                "noise_slope": fit_slope(sizes, noise),
                "noise_lower_slope": fit_slope(sizes, lower),
                "max_formula_error": float("nan"),
                "min_noise_over_lower": min(row["noise_over_lower"] for row in med_rows),
                "min_gap": min(row["min_gap"] for row in med_rows),
            }
        )
    return rows, summary


def write_outputs(rows: list[dict[str, float]], summary: list[dict[str, float]]) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "robustness_checks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (RES_DIR / "robustness_checks_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def make_figure(rows: list[dict[str, float]], summary: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), constrained_layout=True)
    analytic = asymptotic_response_exponent()

    asym_summary = [row for row in summary if row["scan"] == "asymmetric_probe"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.88, len(asym_summary)))
    ax = axes[0, 0]
    for color, item in zip(colors, asym_summary):
        selected = [
            row
            for row in rows
            if row["scan"] == "asymmetric_probe"
            and row["kappa_in"] == item["kappa_in"]
            and row["kappa_out"] == item["kappa_out"]
        ]
        selected.sort(key=lambda row: row["N"])
        sizes = np.array([row["N"] for row in selected])
        chi = np.array([row["chi"] for row in selected])
        label = rf"$({item['kappa_in']:g},{item['kappa_out']:g})$"
        ax.semilogy(sizes, chi / chi[0], "o-", color=color, ms=3.0, lw=1.15, label=label)
    ax.set_xlabel(r"chain length $N$")
    ax.set_ylabel(r"normalized $|\chi_{\kappa,N1}(0)|$")
    ax.set_title(r"asymmetric probes keep the response exponent")
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.legend(frameon=False, ncol=2, title=r"$(\kappa_{\rm in},\kappa_{\rm out})$")
    ax.text(-0.12, 1.04, "(a)", transform=ax.transAxes, weight="bold")

    ax = axes[0, 1]
    labels = [rf"$({row['kappa_in']:g},{row['kappa_out']:g})$" for row in asym_summary]
    xpos = np.arange(len(asym_summary))
    ax.plot(xpos, [row["chi_slope"] for row in asym_summary], "o-", label=r"$|\chi|$")
    ax.plot(xpos, [row["noise_slope"] for row in asym_summary], "s-", label=r"$S$")
    ax.plot(xpos, [row["noise_lower_slope"] for row in asym_summary], "d-", label="bound")
    ax.axhline(analytic, color="0.25", ls="--", lw=0.9, label=r"$\log(\rho/\lambda)$")
    ax.axhline(2.0 * analytic, color="0.25", ls=":", lw=0.9, label=r"$2\log(\rho/\lambda)$")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("fitted exponent")
    ax.set_title("asymmetric probe slopes")
    ax.grid(True, axis="y", color="0.88", lw=0.6)
    ax.legend(frameon=False, ncol=2)
    ax.text(-0.12, 1.04, "(b)", transform=ax.transAxes, weight="bold")

    disorder_summary = [row for row in summary if row["scan"] == "onsite_damping_disorder"]
    ax = axes[1, 0]
    for row in disorder_summary:
        selected = [
            item
            for item in rows
            if item["scan"] == "onsite_damping_disorder"
            and item["disorder_W"] == row["disorder_W"]
        ]
        medians = []
        for n_sites in SIZES:
            vals = [item["chi"] for item in selected if item["N"] == float(n_sites)]
            medians.append(float(np.median(vals)))
        medians = np.array(medians)
        ax.semilogy(SIZES, medians / medians[0], "o-", ms=3.0, lw=1.15, label=rf"$W={row['disorder_W']:g}$")
    ax.set_xlabel(r"chain length $N$")
    ax.set_ylabel(r"median normalized $|\chi|$")
    ax.set_title(r"weak CP damping disorder preserves growth")
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.legend(frameon=False)
    ax.text(-0.12, 1.04, "(c)", transform=ax.transAxes, weight="bold")

    ax = axes[1, 1]
    w_values = np.array([row["disorder_W"] for row in disorder_summary])
    ax.plot(w_values, [row["chi_slope"] for row in disorder_summary], "o-", label=r"median $|\chi|$")
    ax.plot(w_values, [row["noise_slope"] for row in disorder_summary], "s-", label=r"median $S$")
    ax.plot(w_values, [row["noise_lower_slope"] for row in disorder_summary], "d-", label="median bound")
    ax.axhline(analytic, color="0.25", ls="--", lw=0.9, label=r"$\log(\rho/\lambda)$")
    ax.axhline(2.0 * analytic, color="0.25", ls=":", lw=0.9, label=r"$2\log(\rho/\lambda)$")
    ax.set_xlabel(r"onsite damping disorder strength $W$")
    ax.set_ylabel("fitted exponent")
    ax.set_title("disorder-averaged slopes")
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.legend(frameon=False, ncol=2)
    ax.text(-0.12, 1.04, "(d)", transform=ax.transAxes, weight="bold")

    for out_dir in (FIG_DIR, ARXIV_FIG_DIR):
        fig.savefig(out_dir / "robustness_checks.png", dpi=220)
        fig.savefig(out_dir / "robustness_checks.pdf")
    plt.close(fig)


def write_note(summary: list[dict[str, float]]) -> None:
    asym = [row for row in summary if row["scan"] == "asymmetric_probe"]
    disorder = [row for row in summary if row["scan"] == "onsite_damping_disorder"]
    max_asym_error = max(row["max_formula_error"] for row in asym)
    max_asym_chi_drift = max(abs(row["chi_slope"] - asymptotic_response_exponent()) for row in asym)
    max_disorder_w = max(row["disorder_W"] for row in disorder)
    min_disorder_gap = min(row["min_gap"] for row in disorder)
    note = f"""# Robustness Checks

Robustness checks beyond the clean symmetric-probe Toeplitz chain.

- maximum asymmetric-probe formula error: `{max_asym_error:.3e}`;
- maximum asymmetric-probe susceptibility slope drift from `log(rho/lambda)`:
  `{max_asym_chi_drift:.3e}`;
- largest onsite damping disorder strength: `{max_disorder_w:.3g}`;
- minimum rapidity gap over all disorder samples: `{min_disorder_gap:.6g}`.

Generated files:

- `figures/robustness_checks.png`
- `figures/robustness_checks.pdf`
- `results/robustness_checks.csv`
- `results/robustness_checks_summary.csv`
"""
    (RES_DIR / "reports/robustness_checks.txt").write_text(note, encoding="utf-8")


def main() -> None:
    asym_rows, asym_summary = scan_asymmetric_probes()
    disorder_rows, disorder_summary = scan_disorder()
    rows = asym_rows + disorder_rows
    summary = asym_summary + disorder_summary
    write_outputs(rows, summary)
    make_figure(rows, summary)
    write_note(summary)
    print("Wrote figures/robustness_checks.png")
    print("Wrote results/robustness_checks_summary.csv")


if __name__ == "__main__":
    main()
