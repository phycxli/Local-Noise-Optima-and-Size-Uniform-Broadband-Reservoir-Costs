"""Frequency-resolved observable spectra for the local reservoir.

This script adds a directly measurable frequency-domain diagnostic:

    chi_N1(omega) = e_N^T (i omega I - X)^(-1) e_1

and the right-edge normal-noise spectrum

    S_N(omega) = e_N^T chi(omega) Gamma_g chi(omega)^dagger e_N.

Up to the usual vacuum floor and output-coupling factors, S_N is the
phase-insensitive part of a homodyne-accessible edge noise spectrum.  It is
also the integrand of the H2/Gramian representation of the steady occupation.
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


def exact_gap(n_sites: int, p: LocalBosonicHatanoParams = PARAMS) -> float:
    return p.gamma - 2.0 * math.sqrt(p.t_right * p.t_left) * math.cos(
        math.pi / (n_sites + 1)
    )


def gain_matrix(n_sites: int, p: LocalBosonicHatanoParams = PARAMS) -> np.ndarray:
    _, gamma_gain, _ = local_bosonic_hatano_lindblad_matrices(n_sites, p)
    return gamma_gain.real


def spectral_curves(
    n_sites: int,
    omegas: np.ndarray,
    p: LocalBosonicHatanoParams = PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return |chi_N1(omega)| and right-edge noise spectrum."""

    x = local_bosonic_hatano_drift(n_sites, p)
    gamma_gain = gain_matrix(n_sites, p)
    eye = np.eye(n_sites, dtype=complex)
    edge = np.zeros(n_sites, dtype=complex)
    edge[-1] = 1.0

    chi_end = np.empty_like(omegas, dtype=float)
    noise = np.empty_like(omegas, dtype=float)
    for idx, omega in enumerate(omegas):
        matrix = 1j * omega * eye - x
        # y_i = [(i omega I - X)^(-1)]_{N i}
        y = np.linalg.solve(matrix.T, edge)
        chi_end[idx] = abs(y[0])
        noise[idx] = float(np.real(np.vdot(y, gamma_gain @ y)))
    return chi_end, noise


def positive_half_width(omegas: np.ndarray, values: np.ndarray) -> float:
    """Return positive half-width at half maximum around omega=0."""

    zero_idx = int(np.argmin(np.abs(omegas)))
    peak = values[zero_idx]
    target = 0.5 * peak
    right_values = values[zero_idx:]
    right_omegas = omegas[zero_idx:]
    below = np.flatnonzero(right_values <= target)
    if len(below) == 0:
        return float("nan")
    idx = int(below[0])
    if idx == 0:
        return 0.0
    x0, x1 = right_omegas[idx - 1], right_omegas[idx]
    y0, y1 = right_values[idx - 1], right_values[idx]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def scan() -> tuple[np.ndarray, dict[int, dict[str, np.ndarray | float]]]:
    sizes = [20, 40, 60, 80, 100]
    omegas = np.linspace(-0.8, 0.8, 801)
    out: dict[int, dict[str, np.ndarray | float]] = {}
    for n_sites in sizes:
        chi_end, noise = spectral_curves(n_sites, omegas)
        out[n_sites] = {
            "chi_end": chi_end,
            "noise": noise,
            "gap": exact_gap(n_sites),
            "inverse_gap": 1.0 / exact_gap(n_sites),
            "chi_zero": float(chi_end[int(np.argmin(np.abs(omegas)))]),
            "noise_zero": float(noise[int(np.argmin(np.abs(omegas)))]),
            "chi_half_width": positive_half_width(omegas, chi_end * chi_end),
            "noise_half_width": positive_half_width(omegas, noise),
            "integrated_noise_window": float(
                np.trapezoid(noise, omegas) / (2.0 * math.pi)
            ),
        }
    return omegas, out


def write_outputs(
    omegas: np.ndarray,
    data: dict[int, dict[str, np.ndarray | float]],
) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    curves_path = RES_DIR / "spectral_observables_curves.csv"
    with curves_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["omega"]
        for n_sites in data:
            fieldnames.extend([f"chi_N{n_sites}", f"noise_N{n_sites}"])
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, omega in enumerate(omegas):
            row: dict[str, float] = {"omega": float(omega)}
            for n_sites, values in data.items():
                row[f"chi_N{n_sites}"] = float(values["chi_end"][idx])  # type: ignore[index]
                row[f"noise_N{n_sites}"] = float(values["noise"][idx])  # type: ignore[index]
            writer.writerow(row)

    summary_path = RES_DIR / "spectral_observables_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "N",
            "gap",
            "inverse_gap",
            "chi_zero",
            "noise_zero",
            "chi_half_width",
            "noise_half_width",
            "integrated_noise_window",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for n_sites, values in data.items():
            writer.writerow(
                {
                    "N": float(n_sites),
                    "gap": float(values["gap"]),
                    "inverse_gap": float(values["inverse_gap"]),
                    "chi_zero": float(values["chi_zero"]),
                    "noise_zero": float(values["noise_zero"]),
                    "chi_half_width": float(values["chi_half_width"]),
                    "noise_half_width": float(values["noise_half_width"]),
                    "integrated_noise_window": float(values["integrated_noise_window"]),
                }
            )

    sizes = np.array(list(data.keys()), dtype=float)
    chi_zero = np.array([float(data[n]["chi_zero"]) for n in data])
    noise_zero = np.array([float(data[n]["noise_zero"]) for n in data])
    gaps = np.array([float(data[n]["gap"]) for n in data])
    chi_width = np.array([float(data[n]["chi_half_width"]) for n in data])
    noise_width = np.array([float(data[n]["noise_half_width"]) for n in data])

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(data)))
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    for color, (n_sites, values) in zip(colors, data.items()):
        ax.semilogy(omegas, values["chi_end"], color=color, lw=1.4, label=f"N={n_sites}")  # type: ignore[arg-type]
    ax.axvline(0.0, color="0.25", lw=0.7)
    ax.set_xlabel(r"drive frequency $\omega$")
    ax.set_ylabel(r"$|\chi_{N1}(\omega)|$")
    ax.set_title("End-to-end susceptibility spectrum")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for color, (n_sites, values) in zip(colors, data.items()):
        ax.semilogy(omegas, values["noise"], color=color, lw=1.4, label=f"N={n_sites}")  # type: ignore[arg-type]
    ax.axvline(0.0, color="0.25", lw=0.7)
    ax.set_xlabel(r"analysis frequency $\omega$")
    ax.set_ylabel(r"$S_N(\omega)$")
    ax.set_title("Right-edge noise spectrum")

    ax = axes[1, 0]
    ax.semilogy(sizes, chi_zero / chi_zero[0], "o-", label=r"$|\chi_{N1}(0)|$")
    ax.semilogy(sizes, noise_zero / noise_zero[0], "s-", label=r"$S_N(0)$")
    ax.semilogy(sizes, (1.0 / gaps) / (1.0 / gaps[0]), "^-", color="0.25", label=r"$1/\Delta_N$")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("normalized peak")
    ax.set_title("Zero-frequency peaks grow")
    ax.legend(frameon=False)
    ax.grid(True, which="major", color="0.88", lw=0.6)

    ax = axes[1, 1]
    ax.plot(sizes, chi_width, "o-", label=r"$|\chi|^2$ half-width")
    ax.plot(sizes, noise_width, "s-", label=r"$S_N$ half-width")
    ax.plot(sizes, gaps, "^-", color="0.25", label=r"$\Delta_N$")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("frequency scale")
    ax.set_title("Spectral clocks narrow while gap stays finite")
    ax.legend(frameon=False)
    ax.grid(True, which="major", color="0.88", lw=0.6)

    for out_dir in (FIG_DIR, ARXIV_FIG_DIR):
        fig.savefig(out_dir / "spectral_observables.png", dpi=220)
        fig.savefig(out_dir / "spectral_observables.pdf")
    plt.close(fig)

    chi_slope = float(np.polyfit(sizes, np.log(chi_zero), deg=1)[0])
    noise_slope = float(np.polyfit(sizes, np.log(noise_zero), deg=1)[0])
    note = f"""# Spectral Observables

Frequency-resolved drive and noise spectra for the local reservoir at
`t_R=1`, `t_L=0.25`, `gamma=1.2`.

Definitions:

```text
chi_N1(omega) = e_N^T (i omega I - X)^(-1) e_1
S_N(omega) = e_N^T chi(omega) Gamma_g chi(omega)^dagger e_N
```

The noise spectrum is the positive frequency-domain integrand of the
right-edge Gaussian occupation.  Up to a vacuum floor and output-coupling
constants, it is accessible through phase-insensitive edge noise or homodyne
measurements.

Fitted peak growth slopes:

- `log |chi_N1(0)| / N`: `{chi_slope:.6g}`;
- `log S_N(0) / N`: `{noise_slope:.6g}`.

At `N=100`:

- `|chi_N1(0)| = {chi_zero[-1]:.6g}`;
- `S_N(0) = {noise_zero[-1]:.6g}`;
- `chi half-width = {chi_width[-1]:.6g}`;
- `noise half-width = {noise_width[-1]:.6g}`;
- `gap Delta_N = {gaps[-1]:.6g}`.

Generated files:

- `figures/spectral_observables.png`
- `figures/spectral_observables.pdf`
- `results/spectral_observables_curves.csv`
- `results/spectral_observables_summary.csv`
"""
    (RES_DIR / "reports/spectral_observables.txt").write_text(note, encoding="utf-8")


def main() -> None:
    omegas, data = scan()
    write_outputs(omegas, data)
    print("Wrote figures/spectral_observables.png")
    print("Wrote results/spectral_observables_summary.csv")


if __name__ == "__main__":
    main()
