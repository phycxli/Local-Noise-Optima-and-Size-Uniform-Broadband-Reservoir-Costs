"""Input-referred noise and saturation-limited dynamic-range diagnostics.

The Gaussian Lindblad model is linear and therefore has no intrinsic
saturation.  To expose the physical cost of its exponential response, this
script introduces an assumed local occupation ceiling n_sat and computes the
largest coherent input flux that keeps every site below that ceiling after the
vacuum covariance is included.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from liouvillian_geometry import (  # noqa: E402
    LocalBosonicHatanoParams,
    local_bosonic_hatano_drift,
    local_bosonic_hatano_lindblad_matrices,
)


PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)
KAPPA = 0.2
SIZES = np.arange(10, 141, 5, dtype=int)
SATURATION_LEVELS = (1.0e2, 1.0e4, 1.0e6, 1.0e8)

FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def probe_drift(n_sites: int) -> np.ndarray:
    x = local_bosonic_hatano_drift(n_sites, PARAMS).real.copy()
    x[0, 0] -= 0.5 * KAPPA
    x[-1, -1] -= 0.5 * KAPPA
    return x


def steady_covariance(n_sites: int, x: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Solve X N + N X^T + source = 0."""

    covariance = solve_continuous_lyapunov(x, -source)
    return 0.5 * (covariance + covariance.T)


def data_at_size(n_sites: int) -> dict[str, float]:
    x = probe_drift(n_sites)
    _, gamma_gain, _ = local_bosonic_hatano_lindblad_matrices(n_sites, PARAMS)
    gamma_gain = gamma_gain.real

    left = np.zeros(n_sites)
    left[0] = 1.0
    right = np.zeros(n_sites)
    right[-1] = 1.0
    response_column = np.linalg.solve(-x, left)
    response_row = np.linalg.solve((-x).T, right)

    transfer = KAPPA * response_column[-1]
    power_gain = float(abs(transfer) ** 2)
    output_excess_noise = float(
        KAPPA * np.real(response_row @ gamma_gain @ response_row)
    )
    input_referred_excess = output_excess_noise / max(power_gain, 1.0e-300)
    cp_output_floor = max(power_gain - 1.0, 0.0)
    cp_input_floor = max(1.0 - 1.0 / max(power_gain, 1.0e-300), 0.0)

    covariance = steady_covariance(n_sites, x, gamma_gain)
    occupations = np.real(np.diag(covariance))
    coherent_occupation_per_flux = KAPPA * np.abs(response_column) ** 2
    residual = np.linalg.norm(x @ covariance + covariance @ x.T + gamma_gain)
    residual /= max(np.linalg.norm(gamma_gain), 1.0e-300)

    row: dict[str, float] = {
        "N": float(n_sites),
        "kappa": KAPPA,
        "power_gain": power_gain,
        "output_excess_noise": output_excess_noise,
        "cp_output_floor": cp_output_floor,
        "input_referred_excess_noise": input_referred_excess,
        "cp_input_referred_floor": cp_input_floor,
        "right_edge_vacuum_occupation": float(occupations[-1]),
        "maximum_vacuum_occupation": float(np.max(occupations)),
        "covariance_residual": float(residual),
        "stability_margin": float(-np.max(np.linalg.eigvals(x).real)),
    }

    for n_sat in SATURATION_LEVELS:
        label = f"{n_sat:.0e}".replace("+", "")
        if np.max(occupations) >= n_sat:
            max_input_flux = 0.0
        else:
            active = coherent_occupation_per_flux > 1.0e-300
            max_input_flux = float(
                np.min(
                    (n_sat - occupations[active])
                    / coherent_occupation_per_flux[active]
                )
            )
        max_output_signal = power_gain * max_input_flux
        output_dynamic_range = max_output_signal / (1.0 + output_excess_noise)
        row[f"max_input_flux_nsat_{label}"] = max_input_flux
        row[f"output_dynamic_range_nsat_{label}"] = output_dynamic_range

    return row


def positive(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[result <= 0.0] = np.nan
    return result


def make_figure(rows: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.2,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
        }
    )

    sizes = np.array([row["N"] for row in rows])
    gain = np.array([row["power_gain"] for row in rows])
    output_noise = np.array([row["output_excess_noise"] for row in rows])
    output_floor = np.array([row["cp_output_floor"] for row in rows])
    input_noise = np.array([row["input_referred_excess_noise"] for row in rows])
    input_floor = np.array([row["cp_input_referred_floor"] for row in rows])
    edge_vacuum = np.array([row["right_edge_vacuum_occupation"] for row in rows])
    max_vacuum = np.array([row["maximum_vacuum_occupation"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.semilogy(sizes, gain, "o-", ms=3, lw=1.0, label=r"$|\mathcal{G}|^2$")
    ax.semilogy(sizes, output_noise, "s-", ms=3, lw=1.0,
                label=r"$S_{\rm out}^{\rm exc}$")
    ax.semilogy(sizes, positive(output_floor), "k--", lw=1.0,
                label=r"$|\mathcal{G}|^2-1$")
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel("output units")
    ax.set_title("(a) Gain and output noise", loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.semilogy(sizes, input_noise, "o-", ms=3, lw=1.0,
                label=r"$S_{\rm out}^{\rm exc}/|\mathcal{G}|^2$")
    ax.semilogy(sizes, positive(input_floor), "k--", lw=1.0,
                label="CP floor")
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel("input-referred noise")
    ax.set_title("(b) Noise cost of gain", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.semilogy(sizes, edge_vacuum, "o-", ms=3, lw=1.0,
                label="right edge")
    ax.semilogy(sizes, max_vacuum, "s-", ms=3, lw=1.0,
                label="largest site")
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(SATURATION_LEVELS)))
    for color, n_sat in zip(colors, SATURATION_LEVELS):
        ax.axhline(n_sat, color=color, ls=":", lw=0.8)
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel("vacuum occupation")
    ax.set_title("(c) Saturation budget", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for color, n_sat in zip(colors, SATURATION_LEVELS):
        label = f"{n_sat:.0e}".replace("+", "")
        flux = np.array([row[f"max_input_flux_nsat_{label}"] for row in rows])
        exponent = int(np.log10(n_sat))
        ax.semilogy(sizes, positive(flux), "o-", color=color, ms=2.7, lw=1.0,
                    label=rf"$n_{{\rm sat}}=10^{{{exponent}}}$")
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel(r"maximum input flux $\Phi_{\rm in}^{\max}$")
    ax.set_title("(d) Available coherent-input range", loc="left")
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.18, lw=0.5)

    for extension in ("pdf", "png"):
        path = FIG_DIR / f"dynamic_range_noise.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        target = ARXIV_FIG_DIR / path.name
        target.write_bytes(path.read_bytes())
    plt.close(fig)


def write_outputs(rows: list[dict[str, float]]) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "dynamic_range_noise.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    high_gain = rows[-5:]
    asymptotic_input_noise = float(
        np.mean([row["input_referred_excess_noise"] for row in high_gain])
    )
    lines = [
        "# Dynamic range and input-referred noise",
        "",
        f"- Probe rate: `{KAPPA}`",
        f"- Large-size input-referred excess noise: `{asymptotic_input_noise:.6g}`",
        f"- Maximum covariance residual: `{max(row['covariance_residual'] for row in rows):.3e}`",
        "- The linear model has no intrinsic saturation; all dynamic-range values use an assumed local occupation ceiling.",
    ]
    for n_sat in SATURATION_LEVELS:
        usable = [
            int(row["N"])
            for row in rows
            if row["maximum_vacuum_occupation"] < n_sat
        ]
        max_size = max(usable) if usable else 0
        lines.append(f"- Largest scanned size below `n_sat={n_sat:.0e}`: `N={max_size}`")
    (RES_DIR / "reports/dynamic_range_noise.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    rows = [data_at_size(int(n_sites)) for n_sites in SIZES]
    write_outputs(rows)
    make_figure(rows)
    print((RES_DIR / "reports/dynamic_range_noise.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
