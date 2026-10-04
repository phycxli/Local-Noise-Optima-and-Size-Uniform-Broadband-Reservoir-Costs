"""Uniform local-reservoir rate bound and multiband noninterference checks."""

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

from run_finite_frequency_local_optimality import (  # noqa: E402
    KAPPA,
    block_kernel,
    response_data,
    scalar_drift,
    scalar_kernel,
    two_band_drift,
)
from run_rate_constrained_local_cp import explicit_metrics  # noqa: E402

RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"

SCALAR_SIZES = (24, 28, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256)
SCALAR_OMEGAS = (0.0, 0.03, 0.10)
MULTIBAND_SIZES = (8, 10, 12, 14, 16, 20, 24, 30, 40, 50)
MULTIBAND_OMEGAS = (0.0, 0.10, 0.15)

GAMMA = 1.2
T_RIGHT = 1.0
T_LEFT = 0.25


def scalar_root(omega: float) -> complex:
    roots = np.roots([T_RIGHT, -(GAMMA - 1j * omega), T_LEFT])
    return complex(roots[np.argmax(np.abs(roots))])


def jacobi_margin(y: np.ndarray) -> float:
    ratios = np.abs(y[1:] / y[:-1])
    margins = []
    for bond, ratio in enumerate(ratios):
        margin = ratio + 1.0 / ratio
        if bond > 0:
            margin -= np.sqrt(ratio / ratios[bond - 1])
        if bond + 1 < len(ratios):
            margin -= np.sqrt(ratios[bond + 1] / ratio)
        margins.append(margin)
    return float(min(margins))


def kernel_constants(data: dict[str, object], kernel: np.ndarray) -> dict[str, float]:
    y = np.asarray(data["y"])
    source = np.asarray(data["source"])
    unit = y / np.linalg.norm(y)
    eigenvalues = np.linalg.eigvalsh(kernel)
    positive = eigenvalues[eigenvalues > 1.0e-9]
    drift_norm = float(np.max(np.abs(np.linalg.eigvalsh(source))))
    response_margin = float(np.real(np.vdot(unit, source @ unit)))
    kernel_gap = float(np.min(positive))
    kernel_norm = float(np.max(eigenvalues))
    schur_rate_bound = drift_norm + kernel_norm * (
        drift_norm + drift_norm**2 / response_margin
    ) / kernel_gap
    return {
        "kernel_gap": kernel_gap,
        "kernel_norm": kernel_norm,
        "response_margin": response_margin,
        "drift_norm": drift_norm,
        "schur_rate_bound": schur_rate_bound,
    }


def scalar_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for omega in SCALAR_OMEGAS:
        root = scalar_root(omega)
        root_margin = abs(root) + 1.0 / abs(root) - 2.0
        for size in SCALAR_SIZES:
            data = response_data(scalar_drift(size), omega)
            if float(data["fixed"]) <= 0.0:
                continue
            y = np.asarray(data["y"])
            kernel = scalar_kernel(y)
            metrics = explicit_metrics(data, kernel)
            rows.append(
                {
                    "N": float(size),
                    "omega": omega,
                    "fixed_noise": float(data["fixed"]),
                    "gain": float(data["gain"]),
                    "jacobi_margin": jacobi_margin(y),
                    "root_margin": float(root_margin),
                    "rate_required": metrics["rate_required"],
                    **kernel_constants(data, kernel),
                }
            )
    return rows


def adjoint_response_roots(omega: float) -> np.ndarray:
    drift = two_band_drift(3)
    onsite = drift[2:4, 2:4]
    subdiagonal = drift[2:4, 0:2]
    superdiagonal = drift[2:4, 4:6]
    a_zero = -1j * omega * np.eye(2) - onsite.conj().T
    a_plus = -subdiagonal.conj().T
    a_minus = -superdiagonal.conj().T
    transfer = np.block(
        [
            [-np.linalg.solve(a_plus, a_zero), -np.linalg.solve(a_plus, a_minus)],
            [np.eye(2), np.zeros((2, 2))],
        ]
    )
    roots = np.linalg.eigvals(transfer)
    return roots[np.argsort(np.abs(roots))]


def multiband_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for omega in MULTIBAND_OMEGAS:
        roots = adjoint_response_roots(omega)
        internal_separation = float(np.log(abs(roots[-1]) / abs(roots[-2])))
        for size in MULTIBAND_SIZES:
            data = response_data(
                two_band_drift(size),
                omega,
                input_index=0,
                output_index=2 * size - 2,
            )
            if float(data["fixed"]) <= 0.0:
                continue
            y = np.asarray(data["y"])
            kernel = block_kernel(y, size)
            metrics = explicit_metrics(data, kernel)
            rows.append(
                {
                    "N": float(size),
                    "omega": omega,
                    "fixed_noise": float(data["fixed"]),
                    "gain": float(data["gain"]),
                    "internal_root_separation": internal_separation,
                    "rate_required": metrics["rate_required"],
                    **kernel_constants(data, kernel),
                }
            )
    return rows


def analytic_working_bound() -> dict[str, float]:
    root = float(abs(scalar_root(0.0)))
    ratio_1 = (GAMMA + KAPPA / 2.0) / T_RIGHT
    ratio_2 = GAMMA / T_RIGHT - T_LEFT / (T_RIGHT * ratio_1)
    ratio_3 = GAMMA / T_RIGHT - T_LEFT / (T_RIGHT * ratio_2)
    profile_norm_bound = (
        1.0
        + ratio_1**2
        + (ratio_1 * ratio_2) ** 2 / (1.0 - ratio_3**2)
    )
    gain_40 = float(response_data(scalar_drift(40), 0.0)["gain"])
    response_margin = KAPPA * (1.0 - 1.0 / gain_40) / profile_norm_bound
    kernel_gap = root + 1.0 / root - 2.0
    kernel_norm = ratio_1 + 1.0 / root + 2.0 * np.sqrt(ratio_1 / root)
    drift_norm = 2.0 * GAMMA + 2.0 * (T_RIGHT + T_LEFT)
    alpha = (drift_norm + drift_norm**2 / response_margin) / kernel_gap
    return {
        "root": root,
        "kernel_gap": kernel_gap,
        "profile_norm_bound": profile_norm_bound,
        "response_margin": response_margin,
        "kernel_norm": kernel_norm,
        "drift_norm": drift_norm,
        "alpha_bound": alpha,
        "rate_bound": drift_norm + kernel_norm * alpha,
    }


def write_csv(name: str, rows: list[dict[str, float]]) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_rate_rows() -> list[dict[str, float | str]]:
    path = RES_DIR / "rate_constrained_local_cp_exact_rate.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_figure(
    scalar: list[dict[str, float]], multiband: list[dict[str, float]]
) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)

    zero = sorted(
        (row for row in scalar if row["omega"] == 0.0), key=lambda row: row["N"]
    )
    axes[0].semilogy(
        [row["N"] for row in zero],
        [row["kernel_gap"] for row in zero],
        "o-",
        ms=2.5,
        lw=0.9,
        label=r"$\lambda_2(K_N)$",
    )
    axes[0].semilogy(
        [row["N"] for row in zero],
        [row["jacobi_margin"] for row in zero],
        "s--",
        ms=2.5,
        lw=0.9,
        label="Gershgorin margin",
    )
    axes[0].axhline(zero[-1]["root_margin"], color="0.25", ls=":", lw=0.9)
    axes[0].set_title("(a) Uniform kernel coercivity", loc="left", fontsize=8.6)
    axes[0].set_xlabel(r"size $N$")
    axes[0].set_ylabel("coercivity scale")
    axes[0].legend(frameon=False, fontsize=6.0)

    for omega, marker in ((0.0, "o-"), (0.03, "s--"), (0.10, "^-")):
        subset = sorted(
            (row for row in scalar if row["omega"] == omega),
            key=lambda row: row["N"],
        )
        axes[1].semilogy(
            [row["N"] for row in subset],
            [row["rate_required"] for row in subset],
            marker,
            ms=2.5,
            lw=0.9,
            label=rf"$\omega={omega:g}$",
        )
    exact = exact_rate_rows()
    scalar_exact = [row for row in exact if row["model"] == "scalar"]
    if scalar_exact:
        axes[1].semilogy(
            [float(row["N"]) for row in scalar_exact],
            [float(row["rate"]) for row in scalar_exact],
            "kx",
            ms=4.5,
            label="exact SDP",
        )
    axes[1].set_title("(b) Scalar rate saturation", loc="left", fontsize=8.6)
    axes[1].set_xlabel(r"size $N$")
    axes[1].set_ylabel(r"$R_{\rm req}$")
    axes[1].legend(frameon=False, fontsize=5.8)

    for omega, marker in ((0.0, "o-"), (0.10, "s--"), (0.15, "^-")):
        subset = sorted(
            (row for row in multiband if row["omega"] == omega),
            key=lambda row: row["N"],
        )
        separation = subset[0]["internal_root_separation"]
        axes[2].semilogy(
            [row["N"] for row in subset],
            [row["rate_required"] for row in subset],
            marker,
            ms=2.5,
            lw=0.9,
            label=rf"$\omega={omega:g}$, $\delta_z={separation:.2f}$",
        )
    multiband_exact = [row for row in exact if row["model"] == "two_band"]
    if multiband_exact:
        axes[2].semilogy(
            [float(row["N"]) for row in multiband_exact],
            [float(row["rate"]) for row in multiband_exact],
            "kx",
            ms=4.5,
            label="exact SDP",
        )
    axes[2].set_title("(c) Root interference", loc="left", fontsize=8.6)
    axes[2].set_xlabel(r"cells $N$")
    axes[2].set_ylabel(r"$R_{\rm req}$")
    axes[2].legend(frameon=False, fontsize=5.3)

    for axis in axes:
        axis.grid(alpha=0.18, lw=0.5)
    for extension in ("pdf", "png"):
        path = FIG_DIR / f"uniform_rate_bound.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        (ARXIV_FIG_DIR / path.name).write_bytes(path.read_bytes())
    plt.close(fig)


def write_summary(
    scalar: list[dict[str, float]], multiband: list[dict[str, float]]
) -> None:
    bound = analytic_working_bound()
    zero_tail = [
        row for row in scalar if row["omega"] == 0.0 and row["N"] >= 40.0
    ]
    separations = {
        omega: next(
            row["internal_root_separation"]
            for row in multiband
            if row["omega"] == omega
        )
        for omega in MULTIBAND_OMEGAS
    }
    lines = [
        "# Uniform local-reservoir rate bound",
        "",
        "The scalar bond Gram is a Jacobi matrix.  Its Gershgorin margin",
        "provides an N-independent lower bound on the nonzero kernel gap.",
        "",
        f"- Working-point transfer root |q|: {bound['root']:.9f}",
        f"- Analytic kernel margin q + q^-1 - 2: {bound['kernel_gap']:.9e}",
        (
            "- Minimum sampled Jacobi margin for N >= 40: "
            f"{min(row['jacobi_margin'] for row in zero_tail):.9e}"
        ),
        (
            "- Minimum sampled kernel gap for N >= 40: "
            f"{min(row['kernel_gap'] for row in zero_tail):.9e}"
        ),
        f"- Analytic profile norm bound: {bound['profile_norm_bound']:.9f}",
        f"- Analytic response-margin lower bound for N >= 40: {bound['response_margin']:.9e}",
        f"- Conservative uniform rate upper bound: {bound['rate_bound']:.9e}",
        (
            "- Observed explicit rate range for N >= 40: "
            f"{min(row['rate_required'] for row in zero_tail):.6f} to "
            f"{max(row['rate_required'] for row in zero_tail):.6f}"
        ),
        "",
        "The multiband diagnostic is the modulus splitting inside the selected",
        "adjoint-response transfer pair.  Equal moduli permit persistent beating",
        "and size-dependent rate peaks; a split pair supports single-mode",
        "dominance once the corresponding boundary coefficient is nonzero.",
        f"- Two-band internal root separation at omega=0: {separations[0.0]:.9e}",
        f"- Two-band internal root separation at omega=0.10: {separations[0.10]:.9e}",
        f"- Two-band internal root separation at omega=0.15: {separations[0.15]:.9e}",
    ]
    (RES_DIR / "reports/uniform_rate_bound_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    scalar = scalar_rows()
    multiband = multiband_rows()
    write_csv("uniform_rate_bound_scalar.csv", scalar)
    write_csv("uniform_rate_bound_multiband.csv", multiband)
    write_summary(scalar, multiband)
    make_figure(scalar, multiband)
    print((RES_DIR / "reports/uniform_rate_bound_summary.txt").read_text(), end="")


if __name__ == "__main__":
    main()
