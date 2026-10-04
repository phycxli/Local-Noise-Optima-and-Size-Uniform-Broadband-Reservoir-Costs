"""Finite-frequency and multiband locality-constrained CP noise tests.

For a fixed drift X and passive endpoint probes, define

    y(omega) = (i omega I - X_kappa)^(-dagger) e_o,
    M = X + X^dagger.

The input-referred excess noise is y^dagger Gamma_g y /
(kappa_i |y_i|^2), with Gamma_g - Gamma_l = M.  This script compares
unrestricted and nearest-neighbor SDPs with an explicit local kernel
construction at finite frequency for a scalar chain and a two-band chain.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"

KAPPA = 0.2
SCALAR_SIZES = (27, 30, 33)
SCALAR_OMEGAS = (-0.03, 0.0, 0.03)
MULTIBAND_SIZES = (8, 9, 10)
MULTIBAND_OMEGAS = (-0.15, 0.0, 0.15)


def scalar_drift(n: int) -> np.ndarray:
    x = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(x, -1.2)
    for j in range(n - 1):
        x[j + 1, j] = 1.0
        x[j, j + 1] = 0.25
    return x


def two_band_drift(n_cells: int) -> np.ndarray:
    d = 2
    dim = d * n_cells
    x = np.zeros((dim, dim), dtype=complex)
    onsite = np.array([[-1.0, -0.12j], [-0.12j, -0.9]], dtype=complex)
    right = np.diag([1.2, 1.0]).astype(complex)
    left = np.diag([0.10, 0.08]).astype(complex)
    for j in range(n_cells):
        x[d * j : d * (j + 1), d * j : d * (j + 1)] = onsite
    for j in range(n_cells - 1):
        x[d * (j + 1) : d * (j + 2), d * j : d * (j + 1)] = right
        x[d * j : d * (j + 1), d * (j + 1) : d * (j + 2)] = left
    return x


def response_data(
    x: np.ndarray,
    omega: float,
    input_index: int = 0,
    output_index: int | None = None,
) -> dict[str, object]:
    n = x.shape[0]
    if output_index is None:
        output_index = n - 1
    x_probe = x.copy()
    x_probe[input_index, input_index] -= KAPPA / 2.0
    x_probe[output_index, output_index] -= KAPPA / 2.0
    eye = np.eye(n, dtype=complex)
    output = eye[:, output_index]
    y = np.linalg.solve((1j * omega * eye - x_probe).conj().T, output)
    source = x + x.conj().T
    denominator = KAPPA * max(float(abs(y[input_index]) ** 2), 1.0e-300)
    fixed = float(np.real(np.vdot(y, source @ y)) / denominator)
    gain = float(KAPPA**2 * abs(y[input_index]) ** 2)
    caves = 1.0 - 1.0 / max(gain, 1.0e-300)
    reflection = abs(1.0 - KAPPA * y[output_index]) ** 2 / max(
        KAPPA**2 * abs(y[input_index]) ** 2, 1.0e-300
    )
    return {
        "y": y,
        "source": source,
        "input_index": input_index,
        "output_index": output_index,
        "fixed": fixed,
        "gain": gain,
        "caves": caves,
        "reflection": reflection,
        "caves_residual": fixed - caves - reflection,
    }


def solve_sdp(
    data: dict[str, object], range_value: int | None, block_size: int = 1
) -> float:
    y = np.asarray(data["y"])
    y_norm_sq = float(np.vdot(y, y).real)
    y_objective = y / np.sqrt(y_norm_sq)
    source = np.asarray(data["source"])
    n = len(y)
    gain = cp.Variable((n, n), hermitian=True)
    constraints = [gain >> 0, gain - source >> 0]
    if range_value is not None:
        constraints.extend(
            gain[i, j] == 0
            for i in range(n)
            for j in range(i + 1, n)
            if j // block_size > i // block_size + range_value
        )
    problem = cp.Problem(
        cp.Minimize(cp.real(cp.quad_form(y_objective, gain))), constraints
    )
    try:
        value = problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1.0e-6,
            tol_feas=1.0e-6,
            tol_gap_rel=1.0e-6,
            max_iter=300,
        )
    except cp.error.SolverError:
        value = problem.solve(
            solver="SCS",
            eps=2.0e-6,
            max_iters=50000,
            normalize=True,
        )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SDP failed: {problem.status}")
    input_index = int(data["input_index"])
    return float(
        max(float(value), 0.0)
        * y_norm_sq
        / (KAPPA * abs(y[input_index]) ** 2)
    )


def scalar_kernel(y: np.ndarray) -> np.ndarray:
    """Return a balanced nearest-neighbor PSD kernel with kernel span{y}.

    The unweighted bond vectors are algebraically valid but can have an
    exponentially bad condition number when the response profile is skin
    localized.  The geometric-mean normalization keeps neighboring bond
    amplitudes on the scale of the local response ratio.
    """
    n = len(y)
    kernel = np.zeros((n, n), dtype=complex)
    for j in range(n - 1):
        scale = np.sqrt(abs(y[j]) * abs(y[j + 1]))
        if scale < 1.0e-300:
            raise ValueError("A scalar response entry is too small.")
        vector = np.zeros(n, dtype=complex)
        vector[j] = np.conj(y[j + 1]) / scale
        vector[j + 1] = -np.conj(y[j]) / scale
        kernel += np.outer(vector, vector.conj())
    return kernel


def block_kernel(y: np.ndarray, n_cells: int, d: int = 2) -> np.ndarray:
    """Block-tridiagonal PSD kernel with kernel span{y}.

    Onsite projectors remove components transverse to the local response vector.
    Geometrically balanced bond terms then equalize the scalar amplitudes
    multiplying those response vectors from cell to cell without introducing
    an artificial rate scale from the exponentially varying response norm.
    """

    blocks = [y[d * j : d * (j + 1)] for j in range(n_cells)]
    dim = d * n_cells
    kernel = np.zeros((dim, dim), dtype=complex)
    for j, block in enumerate(blocks):
        norm2 = float(np.vdot(block, block).real)
        if norm2 < 1.0e-14:
            raise ValueError("A local response block is too small.")
        projector = np.eye(d, dtype=complex) - np.outer(block, block.conj()) / norm2
        sl = slice(d * j, d * (j + 1))
        kernel[sl, sl] += projector
    for j in range(n_cells - 1):
        left_norm = float(np.linalg.norm(blocks[j]))
        right_norm = float(np.linalg.norm(blocks[j + 1]))
        left = blocks[j] / left_norm
        right = blocks[j + 1] / right_norm
        vector = np.zeros((dim,), dtype=complex)
        vector[d * j : d * (j + 1)] = np.sqrt(right_norm / left_norm) * left
        vector[d * (j + 1) : d * (j + 2)] = -np.sqrt(left_norm / right_norm) * right
        kernel += np.outer(vector, vector.conj())
    return kernel


def explicit_noise(data: dict[str, object], kernel: np.ndarray) -> dict[str, float]:
    y = np.asarray(data["y"])
    source = np.asarray(data["source"])
    if float(np.real(np.vdot(y, source @ y))) <= 0.0:
        return {"noise": np.nan, "formula_error": np.nan, "kernel_residual": np.nan}

    def minimum_eigenvalue(alpha: float) -> float:
        return float(np.min(np.linalg.eigvalsh(source + alpha * kernel)))

    upper = 1.0e-10
    while minimum_eigenvalue(upper) < -1.0e-9 and upper < 1.0e15:
        upper *= 10.0
    if minimum_eigenvalue(upper) < -1.0e-8:
        raise RuntimeError("finite-frequency kernel construction failed")
    lower = upper / 10.0
    for _ in range(90):
        middle = 0.5 * (lower + upper)
        if minimum_eigenvalue(middle) >= 0.0:
            upper = middle
        else:
            lower = middle
    gain_matrix = source + upper * kernel
    denominator = KAPPA * abs(y[int(data["input_index"])]) ** 2
    noise = float(np.real(np.vdot(y, gain_matrix @ y)) / denominator)
    return {
        "noise": noise,
        "formula_error": noise - float(data["fixed"]),
        "kernel_residual": float(np.linalg.norm(kernel @ y) / max(np.linalg.norm(y), 1.0e-300)),
    }


def run_scalar() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for n in SCALAR_SIZES:
        for omega in SCALAR_OMEGAS:
            data = response_data(scalar_drift(n), omega)
            if float(data["fixed"]) <= 0.0:
                raise RuntimeError(
                    f"scalar sample outside theorem regime: N={n}, omega={omega}"
                )
            nonlocal_noise = solve_sdp(data, None)
            local_noise = solve_sdp(data, 1)
            explicit = explicit_noise(data, scalar_kernel(np.asarray(data["y"])))
            rows.append(
                {
                    "model": "scalar",
                    "N": float(n),
                    "omega": omega,
                    "gain": float(data["gain"]),
                    "fixed": float(data["fixed"]),
                    "caves": float(data["caves"]),
                    "reflection": float(data["reflection"]),
                    "caves_residual": float(data["caves_residual"]),
                    "sdp_Rinf": nonlocal_noise,
                    "sdp_R1": local_noise,
                    "explicit": explicit["noise"],
                    "explicit_error": explicit["formula_error"],
                    "kernel_residual": explicit["kernel_residual"],
                }
            )
    return rows


def run_multiband() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for n_cells in MULTIBAND_SIZES:
        for omega in MULTIBAND_OMEGAS:
            data = response_data(
                two_band_drift(n_cells),
                omega,
                input_index=0,
                output_index=2 * n_cells - 2,
            )
            if float(data["fixed"]) <= 0.0:
                raise RuntimeError(
                    f"two-band sample outside theorem regime: "
                    f"N={n_cells}, omega={omega}"
                )
            nonlocal_noise = solve_sdp(data, None)
            local_noise = solve_sdp(data, 1, block_size=2)
            explicit = explicit_noise(
                data, block_kernel(np.asarray(data["y"]), n_cells)
            )
            rows.append(
                {
                    "model": "two_band",
                    "N": float(n_cells),
                    "omega": omega,
                    "gain": float(data["gain"]),
                    "fixed": float(data["fixed"]),
                    "caves": float(data["caves"]),
                    "reflection": float(data["reflection"]),
                    "caves_residual": float(data["caves_residual"]),
                    "sdp_Rinf": nonlocal_noise,
                    "sdp_R1": local_noise,
                    "explicit": explicit["noise"],
                    "explicit_error": explicit["formula_error"],
                    "kernel_residual": explicit["kernel_residual"],
                }
            )
    return rows


def write_csv(name: str, rows: list[dict[str, float]]) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    scalar: list[dict[str, float]], multiband: list[dict[str, float]]
) -> None:
    def metrics(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            "sdp_gap": max(abs(row["sdp_R1"] - row["sdp_Rinf"]) for row in rows),
            "explicit_error": max(abs(row["explicit_error"]) for row in rows),
            "kernel_residual": max(row["kernel_residual"] for row in rows),
        }

    scalar_metrics = metrics(scalar)
    multiband_metrics = metrics(multiband)
    lines = [
        "# Finite-frequency local optimality",
        "",
        "The complex-Hermitian SDP compares unrestricted and block-nearest-neighbor",
        "CP factorizations at fixed drift. The explicit kernel construction is",
        "compared with the analytic fixed-drift optimum.",
        "",
        f"- Scalar samples: {len(scalar)}",
        f"- Two-band samples: {len(multiband)}",
        f"- Scalar maximum SDP locality gap: {scalar_metrics['sdp_gap']:.6e}",
        f"- Two-band maximum SDP locality gap: {multiband_metrics['sdp_gap']:.6e}",
        f"- Scalar maximum explicit formula error: {scalar_metrics['explicit_error']:.6e}",
        f"- Two-band maximum explicit formula error: {multiband_metrics['explicit_error']:.6e}",
        f"- Scalar maximum kernel residual: {scalar_metrics['kernel_residual']:.6e}",
        f"- Two-band maximum kernel residual: {multiband_metrics['kernel_residual']:.6e}",
        f"- Maximum Caves identity residual: {max(abs(row['caves_residual']) for row in scalar + multiband):.6e}",
        "",
        "The SDP gaps are numerical solver deviations; the explicit PSD kernels",
        "saturate the analytic optimum to the reported formula and kernel errors.",
    ]
    (RES_DIR / "reports/finite_frequency_local_optimality_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def make_figure(scalar: list[dict[str, float]], multiband: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)
    for omega in sorted({row["omega"] for row in scalar}):
        subset = sorted((row for row in scalar if row["omega"] == omega), key=lambda r: r["N"])
        axes[0].semilogy(
            [row["N"] for row in subset],
            [row["sdp_R1"] for row in subset],
            "o-",
            ms=2.5,
            lw=0.9,
            label=rf"$\omega={omega:g}$",
        )
    axes[0].set_title("(a) Scalar optimum", loc="left", fontsize=9.0)
    axes[0].set_xlabel(r"size $N$")
    axes[0].set_ylabel(r"$\mathcal{N}_{R=1}^{\rm exc}$")
    axes[0].legend(frameon=False, ncol=2, fontsize=6.0)

    gaps = np.array([abs(row["sdp_R1"] - row["sdp_Rinf"]) for row in scalar])
    axes[1].semilogy(np.arange(len(gaps)), np.maximum(gaps, 1.0e-10), "o", ms=2.5)
    axes[1].axhline(1.0e-6, color="0.4", ls=":", lw=0.8)
    axes[1].set_title("(b) Scalar locality gap", loc="left", fontsize=9.0)
    axes[1].set_xlabel("finite-frequency sample")
    axes[1].set_ylabel(r"$|\mathcal{N}_{R=1}-\mathcal{N}_{\infty}|$")

    mb_gaps = np.array([abs(row["sdp_R1"] - row["sdp_Rinf"]) for row in multiband])
    mb_errors = np.array([abs(row["explicit_error"]) for row in multiband])
    axes[2].semilogy(mb_gaps, "o-", ms=2.5, lw=0.9, label="two-band SDP gap")
    axes[2].semilogy(mb_errors, "s--", ms=2.5, lw=0.9, label="explicit error")
    axes[2].axhline(1.0e-6, color="0.4", ls=":", lw=0.8)
    axes[2].set_title("(c) Two-band validation", loc="left", fontsize=9.0)
    axes[2].set_xlabel("finite-frequency sample")
    axes[2].set_ylabel("absolute deviation")
    axes[2].legend(frameon=False, fontsize=6.0)
    for ax in axes:
        ax.grid(alpha=0.18, lw=0.5)
    for extension in ("pdf", "png"):
        path = FIG_DIR / f"finite_frequency_local_optimality.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        (ARXIV_FIG_DIR / path.name).write_bytes(path.read_bytes())
    plt.close(fig)


def main() -> None:
    scalar = run_scalar()
    multiband = run_multiband()
    write_csv("finite_frequency_local_optimality_scalar.csv", scalar)
    write_csv("finite_frequency_local_optimality_multiband.csv", multiband)
    write_summary(scalar, multiband)
    make_figure(scalar, multiband)
    print(f"scalar samples: {len(scalar)}")
    print(f"multiband samples: {len(multiband)}")
    print(
        "max scalar locality gap:",
        max(abs(row["sdp_R1"] - row["sdp_Rinf"]) for row in scalar),
    )
    print(
        "max scalar formula error:",
        max(abs(row["explicit_error"]) for row in scalar),
    )
    print(
        "max two-band locality gap:",
        max(abs(row["sdp_R1"] - row["sdp_Rinf"]) for row in multiband),
    )
    print(
        "max two-band formula error:",
        max(abs(row["explicit_error"]) for row in multiband),
    )
    print(
        "max Caves identity residual:",
        max(abs(row["caves_residual"]) for row in scalar + multiband),
    )
    print(
        "max scalar kernel residual:",
        max(row["kernel_residual"] for row in scalar),
    )
    print(
        "max two-band kernel residual:",
        max(row["kernel_residual"] for row in multiband),
    )


if __name__ == "__main__":
    main()
