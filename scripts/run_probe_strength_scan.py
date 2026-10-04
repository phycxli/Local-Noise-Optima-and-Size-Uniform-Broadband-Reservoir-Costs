"""Finite input-output probe strength scan.

The input-output derivation is formally written in the weak-probe limit.  This
script checks that the measurable zero-frequency response is not singular in
that limit.  We attach symmetric passive probes to the two ends,

    X_kappa = X - kappa/2 |1><1| - kappa/2 |N><N|,

and compute the de-embedded observables

    |T(0)| / kappa = |e_N^T (-X_kappa)^(-1) e_1|,
    S_out^exc(0) / kappa = e_N^T chi_kappa Gamma_g chi_kappa^dagger e_N.

The omitted factors are known port-coupling constants and do not affect the
system-size exponent.
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
    local_bosonic_hatano_drift,
    local_bosonic_hatano_lindblad_matrices,
)


FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)
SIZES = np.array([20, 30, 40, 50, 60, 80, 100, 120, 140], dtype=int)
CURVE_KAPPAS = [0.0, 0.01, 0.05, 0.2, 1.0]
FIT_KAPPAS = np.geomspace(1.0e-4, 2.0, 22)


def gain_matrix(n_sites: int) -> np.ndarray:
    _, gamma_gain, _ = local_bosonic_hatano_lindblad_matrices(n_sites, PARAMS)
    return gamma_gain.real


def probe_drift(n_sites: int, kappa: float) -> np.ndarray:
    x = local_bosonic_hatano_drift(n_sites, PARAMS).copy()
    if kappa:
        x[0, 0] -= 0.5 * kappa
        x[-1, -1] -= 0.5 * kappa
    return x


def zero_frequency_values(n_sites: int, kappa: float) -> dict[str, float]:
    x = probe_drift(n_sites, kappa)
    gamma_gain = gain_matrix(n_sites)
    edge = np.zeros(n_sites, dtype=complex)
    edge[-1] = 1.0

    # y_i = [(-X_kappa)^(-1)]_{N i}.
    y = np.linalg.solve((-x).T, edge)
    chi_deembedded = float(abs(y[0]))
    chi_formula = finite_probe_chi_formula(n_sites, kappa)
    noise_deembedded = float(np.real(np.vdot(y, gamma_gain @ y)))
    first_channel_direct = PARAMS.bond_rate * abs(y[0] + y[1]) ** 2
    first_channel_formula = finite_probe_noise_lower_bound(n_sites, kappa)
    gap = float(-np.max(np.linalg.eigvals(x).real))

    return {
        "deembedded_transmission_zero": chi_deembedded,
        "finite_probe_chebyshev_chi": chi_formula,
        "finite_probe_chi_relative_error": abs(chi_deembedded - chi_formula)
        / max(abs(chi_formula), 1.0e-300),
        "deembedded_output_noise_zero": noise_deembedded,
        "first_channel_noise_lower_bound": float(first_channel_formula),
        "first_channel_noise_direct": float(first_channel_direct),
        "first_channel_noise_relative_error": abs(first_channel_direct - first_channel_formula)
        / max(abs(first_channel_formula), 1.0e-300),
        "noise_over_first_channel_bound": noise_deembedded
        / max(first_channel_formula, 1.0e-300),
        "transmission_zero": float(kappa * chi_deembedded),
        "output_noise_zero": float(kappa * noise_deembedded),
        "output_noise_lower_bound_zero": float(kappa * first_channel_formula),
        "rapidity_gap": gap,
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


def finite_probe_chi_formula(n_sites: int, kappa: float) -> float:
    """Exact endpoint response with symmetric finite probes.

    The determinant of A_kappa=-X_kappa is

        s^N U_N(x) + kappa s^(N-1) U_(N-1)(x)
        + kappa^2 s^(N-2) U_(N-2)(x) / 4,

    where s=sqrt(t_R t_L) and x=gamma/(2s).  Since
    (A_kappa^-1)_{N1}=t_R^(N-1)/det(A_kappa), division by s^(N-1) gives the
    numerically convenient expression below.
    """

    s = math.sqrt(PARAMS.t_right * PARAMS.t_left)
    rho = math.sqrt(PARAMS.t_right / PARAMS.t_left)
    x_value = PARAMS.gamma / (2.0 * s)
    denom = (
        s * chebyshev_u(n_sites, x_value)
        + kappa * chebyshev_u(n_sites - 1, x_value)
        + (kappa * kappa / (4.0 * s)) * chebyshev_u(n_sites - 2, x_value)
    )
    return rho ** (n_sites - 1) / denom


def finite_probe_noise_lower_bound(n_sites: int, kappa: float) -> float:
    """Strict first-gain-channel lower bound on de-embedded output noise."""

    alpha = 0.5 * kappa
    prefactor = PARAMS.bond_rate * (
        (PARAMS.t_right + PARAMS.gamma + alpha) / PARAMS.t_right
    ) ** 2
    chi = finite_probe_chi_formula(n_sites, kappa)
    return prefactor * chi * chi


def asymptotic_response_exponent() -> float:
    s = math.sqrt(PARAMS.t_right * PARAMS.t_left)
    rho = math.sqrt(PARAMS.t_right / PARAMS.t_left)
    x = PARAMS.gamma / (2.0 * s)
    lam = x + math.sqrt(x * x - 1.0)
    return math.log(rho / lam)


def fit_slope(sizes: np.ndarray, values: np.ndarray) -> float:
    mask = values > 0.0
    return float(np.polyfit(sizes[mask].astype(float), np.log(values[mask]), deg=1)[0])


def all_kappas() -> list[float]:
    values = set(float(k) for k in CURVE_KAPPAS)
    values.update(float(k) for k in FIT_KAPPAS)
    return sorted(values)


def scan() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    by_kappa: dict[float, list[dict[str, float]]] = {}

    for kappa in all_kappas():
        by_kappa[kappa] = []
        for n_sites in SIZES:
            values = zero_frequency_values(int(n_sites), kappa)
            row = {"kappa": kappa, "N": float(n_sites), **values}
            rows.append(row)
            by_kappa[kappa].append(row)

    no_probe_100 = next(
        row for row in by_kappa[0.0] if int(row["N"]) == 100
    )
    summary: list[dict[str, float]] = []
    for kappa, k_rows in by_kappa.items():
        sizes = np.array([row["N"] for row in k_rows], dtype=float)
        chi = np.array([row["deembedded_transmission_zero"] for row in k_rows])
        noise = np.array([row["deembedded_output_noise_zero"] for row in k_rows])
        lower = np.array([row["first_channel_noise_lower_bound"] for row in k_rows])
        gaps = np.array([row["rapidity_gap"] for row in k_rows])
        formula_errors = np.array([row["finite_probe_chi_relative_error"] for row in k_rows])
        lower_errors = np.array([row["first_channel_noise_relative_error"] for row in k_rows])
        noise_over_lower = np.array([row["noise_over_first_channel_bound"] for row in k_rows])
        at_100 = next(row for row in k_rows if int(row["N"]) == 100)
        summary.append(
            {
                "kappa": kappa,
                "chi_slope": fit_slope(sizes, chi),
                "noise_slope": fit_slope(sizes, noise),
                "noise_lower_bound_slope": fit_slope(sizes, lower),
                "chi_at_N100": at_100["deembedded_transmission_zero"],
                "noise_at_N100": at_100["deembedded_output_noise_zero"],
                "noise_lower_bound_at_N100": at_100["first_channel_noise_lower_bound"],
                "chi_ratio_to_no_probe_N100": (
                    at_100["deembedded_transmission_zero"]
                    / no_probe_100["deembedded_transmission_zero"]
                ),
                "noise_ratio_to_no_probe_N100": (
                    at_100["deembedded_output_noise_zero"]
                    / no_probe_100["deembedded_output_noise_zero"]
                ),
                "minimum_gap_over_scan": float(np.min(gaps)),
                "max_chebyshev_formula_relative_error": float(np.max(formula_errors)),
                "max_first_channel_formula_relative_error": float(np.max(lower_errors)),
                "min_noise_over_first_channel_bound": float(np.min(noise_over_lower)),
            }
        )

    return rows, summary


def write_csv(rows: list[dict[str, float]], summary: list[dict[str, float]]) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "probe_strength_scan.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (RES_DIR / "probe_strength_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def rows_for_kappa(rows: list[dict[str, float]], kappa: float) -> list[dict[str, float]]:
    return [row for row in rows if abs(row["kappa"] - kappa) < 1.0e-14]


def make_figure(rows: list[dict[str, float]], summary: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    colors = plt.cm.plasma(np.linspace(0.12, 0.86, len(CURVE_KAPPAS)))
    markers = ["o", "v", "D", "s", "^"]
    linestyles = ["-", "--", "-.", ":", "-"]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    for color, marker, linestyle, kappa in zip(colors, markers, linestyles, CURVE_KAPPAS):
        k_rows = rows_for_kappa(rows, kappa)
        sizes = np.array([row["N"] for row in k_rows])
        chi = np.array([row["deembedded_transmission_zero"] for row in k_rows])
        label = "no probe" if kappa == 0.0 else rf"$\kappa={kappa:g}$"
        ax.semilogy(
            sizes,
            chi / chi[0],
            marker=marker,
            linestyle=linestyle,
            ms=3.3,
            lw=1.25,
            color=color,
            label=label,
        )
    ax.set_xlabel(r"chain length $N$")
    ax.set_ylabel(r"normalized $|T(0)|/\kappa$")
    ax.set_title("de-embedded transmission keeps its exponent")
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.legend(frameon=False)
    ax.text(-0.13, 1.04, "(a)", transform=ax.transAxes, weight="bold")

    ax = axes[0, 1]
    for color, marker, linestyle, kappa in zip(colors, markers, linestyles, CURVE_KAPPAS):
        k_rows = rows_for_kappa(rows, kappa)
        sizes = np.array([row["N"] for row in k_rows])
        noise = np.array([row["deembedded_output_noise_zero"] for row in k_rows])
        label = "no probe" if kappa == 0.0 else rf"$\kappa={kappa:g}$"
        ax.semilogy(
            sizes,
            noise / noise[0],
            marker=marker,
            linestyle=linestyle,
            ms=3.2,
            lw=1.2,
            color=color,
            label=label,
        )
    ax.set_xlabel(r"chain length $N$")
    ax.set_ylabel(r"normalized $S_{\rm out}^{\rm exc}(0)/\kappa$")
    ax.set_title("de-embedded output noise keeps its exponent")
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.text(-0.13, 1.04, "(b)", transform=ax.transAxes, weight="bold")

    finite = [row for row in summary if row["kappa"] > 0.0]
    kappas = np.array([row["kappa"] for row in finite])
    chi_slopes = np.array([row["chi_slope"] for row in finite])
    noise_slopes = np.array([row["noise_slope"] for row in finite])
    no_probe = next(row for row in summary if row["kappa"] == 0.0)
    analytic = asymptotic_response_exponent()

    ax = axes[1, 0]
    ax.semilogx(kappas, chi_slopes, "o-", color="#1f77b4", lw=1.25, ms=3.4, label=r"$|T|/\kappa$")
    ax.semilogx(kappas, noise_slopes, "s-", color="#d62728", lw=1.25, ms=3.2, label=r"$S_{\rm out}^{\rm exc}/\kappa$")
    lower_slopes = np.array([row["noise_lower_bound_slope"] for row in finite])
    ax.semilogx(
        kappas,
        lower_slopes,
        "d-",
        color="#2ca02c",
        lw=1.05,
        ms=3.0,
        label=r"lower bound",
    )
    ax.axhline(no_probe["chi_slope"], color="#1f77b4", ls=":", lw=1.1)
    ax.axhline(no_probe["noise_slope"], color="#d62728", ls=":", lw=1.1)
    ax.axhline(no_probe["noise_lower_bound_slope"], color="#2ca02c", ls=":", lw=1.0)
    ax.axhline(analytic, color="0.25", ls="--", lw=0.9, label=r"$\log(\rho/\lambda)$")
    ax.set_xlabel(r"probe rate $\kappa$")
    ax.set_ylabel(r"fitted exponent")
    ax.set_title("growth exponents are insensitive to finite probes")
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.legend(frameon=False, loc="center right")
    ax.text(-0.13, 1.04, "(c)", transform=ax.transAxes, weight="bold")

    ax = axes[1, 1]
    chi_ratios = np.array([row["chi_ratio_to_no_probe_N100"] for row in finite])
    noise_ratios = np.array([row["noise_ratio_to_no_probe_N100"] for row in finite])
    gaps = np.array([row["minimum_gap_over_scan"] for row in finite])
    ax.semilogx(kappas, chi_ratios, "o-", color="#1f77b4", lw=1.25, ms=3.4, label=r"$|T|$ prefactor")
    ax.semilogx(kappas, noise_ratios, "s-", color="#d62728", lw=1.25, ms=3.2, label=r"noise prefactor")
    ax.set_xlabel(r"probe rate $\kappa$")
    ax.set_ylabel(r"ratio to no-probe value at $N=100$")
    ax.set_title("finite probes mainly change prefactors")
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.legend(frameon=False, loc="lower left")
    ax.text(
        0.05,
        0.13,
        rf"stable scan: $\min\Delta_\kappa={np.min(gaps):.3f}$",
        transform=ax.transAxes,
        fontsize=8,
        color="0.25",
    )
    ax.text(-0.13, 1.04, "(d)", transform=ax.transAxes, weight="bold")

    for out_dir in (FIG_DIR, ARXIV_FIG_DIR):
        fig.savefig(out_dir / "probe_strength_scan.png", dpi=220)
        fig.savefig(out_dir / "probe_strength_scan.pdf")
    plt.close(fig)


def write_note(summary: list[dict[str, float]]) -> None:
    no_probe = next(row for row in summary if row["kappa"] == 0.0)
    kappa_one = min(summary, key=lambda row: abs(row["kappa"] - 1.0))
    kappa_two = min(summary, key=lambda row: abs(row["kappa"] - 2.0))
    finite = [row for row in summary if row["kappa"] > 0.0]
    chi_spread = max(abs(row["chi_slope"] - no_probe["chi_slope"]) for row in finite)
    noise_spread = max(abs(row["noise_slope"] - no_probe["noise_slope"]) for row in finite)
    min_gap = min(row["minimum_gap_over_scan"] for row in finite)
    max_formula_error = max(row["max_chebyshev_formula_relative_error"] for row in summary)
    max_lower_error = max(row["max_first_channel_formula_relative_error"] for row in summary)
    min_noise_over_lower = min(row["min_noise_over_first_channel_bound"] for row in summary)
    lower_slope_spread = max(
        abs(row["noise_lower_bound_slope"] - no_probe["noise_lower_bound_slope"])
        for row in finite
    )
    note = f"""# Probe Strength Scan

Symmetric passive probes are attached with `kappa_in = kappa_out = kappa`.
The scan uses the exact finite-probe rapidity matrix
`X_kappa = X - kappa/2 |1><1| - kappa/2 |N><N|`.

Main results:

- no-probe susceptibility slope: `{no_probe['chi_slope']:.8g}`;
- no-probe noise slope: `{no_probe['noise_slope']:.8g}`;
- maximum finite-probe susceptibility-slope drift over the scan:
  `{chi_spread:.3e}`;
- maximum finite-probe noise-slope drift over the scan: `{noise_spread:.3e}`;
- minimum rapidity gap over finite probes and sizes: `{min_gap:.6g}`.
- maximum relative error between direct inversion and the finite-probe
  Chebyshev formula: `{max_formula_error:.3e}`.
- maximum relative error between the direct first-channel contribution and the
  analytic noise lower bound: `{max_lower_error:.3e}`;
- maximum finite-probe lower-bound slope drift: `{lower_slope_spread:.3e}`;
- minimum total-noise / first-channel-bound ratio over the scan:
  `{min_noise_over_lower:.6g}`.

At `N=100` and `kappa=1`, the de-embedded transmission and noise peaks are
reduced only by prefactors `{kappa_one['chi_ratio_to_no_probe_N100']:.4g}` and
`{kappa_one['noise_ratio_to_no_probe_N100']:.4g}` relative to the no-probe
values.  At `kappa=2`, the corresponding ratios are
`{kappa_two['chi_ratio_to_no_probe_N100']:.4g}` and
`{kappa_two['noise_ratio_to_no_probe_N100']:.4g}`.

Generated files:

- `figures/probe_strength_scan.png`
- `figures/probe_strength_scan.pdf`
- `results/probe_strength_scan.csv`
- `results/probe_strength_summary.csv`
"""
    (RES_DIR / "reports/probe_strength_scan.txt").write_text(note, encoding="utf-8")


def main() -> None:
    rows, summary = scan()
    write_csv(rows, summary)
    make_figure(rows, summary)
    write_note(summary)
    print("Wrote figures/probe_strength_scan.png")
    print("Wrote results/probe_strength_summary.csv")


if __name__ == "__main__":
    main()
