"""Audit a finite-band quantum-noise bound inside the high-gain window.

For a fixed passive input-output probe, complete positivity gives the
pointwise inequality S_exc(omega) >= |G(omega)|^2 - 1.  This script integrates
that inequality over a frequency band on which the transfer exponent is
positive, and compares it with the explicit local bond-reservoir noise.

The output is deliberately an audit, not a theorem for arbitrary reservoirs:
the asymptotic statement needs uniform transfer-root separation and nonzero
port projections over the chosen band.  Those assumptions are recorded in the
accompanying markdown result.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_banded

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"

T_RIGHT = 1.0
T_LEFT = 0.25
GAMMA = 1.2
KAPPA_IN = 0.2
KAPPA_OUT = 0.2
SIZES = tuple(range(24, 321, 8))
BAND_HALFWIDTHS = (0.02, 0.05, 0.10, 0.15)
QUADRATURE_ORDER = 321


def transfer_exponent(omega: float) -> float:
    scale = np.sqrt(T_RIGHT * T_LEFT)
    rho = np.sqrt(T_RIGHT / T_LEFT)
    z = (GAMMA + 1j * omega) / (2.0 * scale)
    q = z + np.sqrt(z * z - 1.0)
    if abs(q) < 1.0:
        q = 1.0 / q
    return float(np.log(rho) - np.log(abs(q)))


def response_vector(n: int, omega: float) -> np.ndarray:
    diagonal = GAMMA * np.ones(n, dtype=complex)
    diagonal[0] += KAPPA_IN / 2.0
    diagonal[-1] += KAPPA_OUT / 2.0
    diagonal -= 1j * omega
    banded = np.zeros((3, n), dtype=complex)
    # (i omega I-X_probe)^dagger has upper -t_R and lower -t_L.
    banded[0, 1:] = -T_RIGHT
    banded[1, :] = diagonal
    banded[2, :-1] = -T_LEFT
    rhs = np.zeros(n, dtype=complex)
    rhs[-1] = 1.0
    return solve_banded((1, 1), banded, rhs, check_finite=False)


def gain_quadratic(y: np.ndarray) -> float:
    coefficient = 0.5 * (T_RIGHT + T_LEFT)
    return float(coefficient * np.sum(np.abs(y[:-1] + y[1:]) ** 2))


def band_nodes(halfwidth: float) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(QUADRATURE_ORDER)
    return halfwidth * nodes, 0.5 * weights


def row(n: int, halfwidth: float) -> dict[str, float]:
    frequencies, weights = band_nodes(halfwidth)
    powers: list[float] = []
    noises: list[float] = []
    cp_floor: list[float] = []
    exponents: list[float] = []
    prefactors: list[float] = []
    for omega in frequencies:
        y = response_vector(n, float(omega))
        power = KAPPA_IN * KAPPA_OUT * abs(y[0]) ** 2
        noise = KAPPA_OUT * gain_quadratic(y)
        lam = transfer_exponent(float(omega))
        powers.append(power)
        noises.append(noise)
        cp_floor.append(max(power - 1.0, 0.0))
        exponents.append(lam)
        prefactors.append(np.sqrt(max(power, 1.0e-300)) * np.exp(-n * lam))
    average = lambda values: float(np.dot(weights, np.asarray(values)))
    return {
        "N": float(n),
        "halfwidth": float(halfwidth),
        "lambda_min": min(exponents),
        "lambda_center": transfer_exponent(0.0),
        "min_gain_power": min(powers),
        "band_gain_power": average(powers),
        "band_noise": average(noises),
        "band_cp_floor": average(cp_floor),
        "prefactor_min": min(prefactors),
        "prefactor_max": max(prefactors),
    }


def first_crossing(rows: list[dict[str, float]], key: str) -> float | None:
    for item in rows:
        if item[key] > 1.0:
            return item["N"]
    return None


def fit(rows: list[dict[str, float]], key: str, halfwidth: float, tail: int = 96) -> float:
    subset = [r for r in rows if r["halfwidth"] == halfwidth and r["N"] >= tail]
    x = np.asarray([r["N"] for r in subset])
    y = np.log(np.maximum(np.asarray([r[key] for r in subset]), 1.0e-300))
    return float(np.polyfit(x, y, 1)[0])


def write_summary(all_rows: list[dict[str, float]]) -> None:
    lines = [
        "# High-gain finite-band CP noise bound",
        "",
        "For fixed endpoint probes, the canonical input-output relation implies",
        "S_out^exc(omega) >= |G_N(omega)|^2 - 1.  The band average therefore",
        "obeys the same inequality.  This audit tests the bound on finite bands",
        "where the transfer exponent is positive throughout the band.",
        "",
        f"- sizes = {SIZES[0]}..{SIZES[-1]} in steps of {SIZES[1]-SIZES[0]}",
        f"- quadrature order = {QUADRATURE_ORDER}",
        "",
    ]
    for halfwidth in BAND_HALFWIDTHS:
        subset = [r for r in all_rows if r["halfwidth"] == halfwidth]
        min_gain_n = first_crossing(subset, "min_gain_power")
        lines.extend(
            [
                f"## Halfwidth {halfwidth:g}",
                "",
                f"- transfer exponent minimum: {subset[0]['lambda_min']:.10f}",
                f"- first scanned size with minimum band gain above one: {min_gain_n}",
                f"- raw band-gain slope (tail): {fit(subset, 'band_gain_power', halfwidth):.10f}",
                f"- raw band-noise slope (tail): {fit(subset, 'band_noise', halfwidth):.10f}",
                f"- CP band-floor slope (tail): {fit(subset, 'band_cp_floor', halfwidth):.10f}",
                f"- twice center exponent: {2*subset[0]['lambda_center']:.10f}",
                f"- prefactor range at largest size: {subset[-1]['prefactor_min']:.6e} .. {subset[-1]['prefactor_max']:.6e}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The integrated CP floor is already exponentially large in a fixed",
            "O(1) frequency band once the finite chain is long enough for the",
            "whole band to have power gain above one.  The explicit local bond",
            "reservoir lies above this floor.  The fitted slopes test the expected",
            "Laplace prefactor N^(-1/2), but they are numerical validation of the",
            "uniform-band transfer assumptions, not a new locality no-go theorem.",
        ]
    )
    (RES_DIR / "reports/high_gain_band_bound.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(all_rows: list[dict[str, float]]) -> None:
    if plt is None:
        return
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), constrained_layout=True)
    for halfwidth in BAND_HALFWIDTHS:
        subset = [r for r in all_rows if r["halfwidth"] == halfwidth]
        ns = [r["N"] for r in subset]
        axes[0].semilogy(
            ns,
            [r["min_gain_power"] for r in subset],
            "o-",
            ms=2.0,
            lw=0.8,
            label=rf"$\Omega={halfwidth:g}$",
        )
    axes[0].axhline(1.0, color="0.25", lw=0.7, ls="--")
    axes[0].set_xlabel("size $N$")
    axes[0].set_ylabel(r"$\min_{|\omega|\leq\Omega}|G_N(\omega)|^2$")
    axes[0].set_title("(a) Fixed-width high-gain band", loc="left", fontsize=8.8)
    axes[0].legend(frameon=False, fontsize=5.8, ncol=2)
    axes[0].grid(alpha=0.18, lw=0.5)

    subset = [r for r in all_rows if r["halfwidth"] == 0.15]
    ns = [r["N"] for r in subset]
    axes[1].semilogy(ns, [r["band_cp_floor"] for r in subset], "o-", ms=2.0,
                     lw=0.8, label="CP lower bound")
    axes[1].semilogy(ns, [r["band_noise"] for r in subset], "s--", ms=2.0,
                     lw=0.8, label="local bond reservoir")
    axes[1].set_xlabel("size $N$")
    axes[1].set_ylabel(r"band-averaged excess noise")
    axes[1].set_title(r"(b) $\Omega=0.15$ noise bound", loc="left", fontsize=8.8)
    axes[1].legend(frameon=False, fontsize=5.8)
    axes[1].grid(alpha=0.18, lw=0.5)
    for extension in ("pdf", "png"):
        path = FIG_DIR / f"high_gain_band_bound.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        (ARXIV_FIG_DIR / path.name).write_bytes(path.read_bytes())
    plt.close(fig)


def main() -> None:
    rows = [row(n, width) for width in BAND_HALFWIDTHS for n in SIZES]
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "high_gain_band_bound.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows)
    make_figure(rows)
    print((RES_DIR / "reports/high_gain_band_bound.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
