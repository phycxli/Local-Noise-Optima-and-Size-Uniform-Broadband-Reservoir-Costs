"""Rate-constrained local CP optimization at fixed drift.

The explicit block-local kernel is normalized to unit spectral norm, so
alpha_star has the units of a dissipative rate.  The script measures the
smallest alpha for which M + alpha K is positive semidefinite and compares
that construction with locality-constrained CP SDPs with a spectral rate cap
R_max.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

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

RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"

SCALAR_SIZES = (20, 24, 28, 32, 36, 40, 48, 56, 64)
SCALAR_OMEGAS = (0.0, 0.03)
MULTIBAND_SIZES = (6, 8, 10, 12)
MULTIBAND_OMEGAS = (0.0, 0.10, 0.15)
EXACT_RATE_CASES = (
    ("scalar", 28, 0.0, 1),
    ("scalar", 40, 0.0, 1),
    ("scalar", 64, 0.0, 1),
    ("two_band", 10, 0.15, 2),
    ("two_band", 16, 0.15, 2),
)
RUN_FULL_RATE_SDP = os.environ.get("RUN_FULL_RATE_SDP", "0") == "1"


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    norm = float(np.max(np.linalg.eigvalsh(kernel)))
    if norm <= 0.0:
        raise ValueError("Kernel must have a positive eigenvalue.")
    return kernel / norm


def alpha_star(source: np.ndarray, kernel: np.ndarray) -> float:
    """Return the smallest alpha found by PSD bisection."""

    def minimum_eigenvalue(alpha: float) -> float:
        return float(np.min(np.linalg.eigvalsh(source + alpha * kernel)))

    if minimum_eigenvalue(0.0) >= -1.0e-10:
        return 0.0
    upper = 1.0
    while minimum_eigenvalue(upper) < -1.0e-9 and upper < 1.0e16:
        upper *= 10.0
    if minimum_eigenvalue(upper) < -1.0e-8:
        raise RuntimeError("Could not find a finite alpha_star.")
    lower = 0.0
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if minimum_eigenvalue(middle) >= 0.0:
            upper = middle
        else:
            lower = middle
    return float(upper)


def explicit_metrics(
    data: dict[str, object], raw_kernel: np.ndarray
) -> dict[str, float]:
    source = np.asarray(data["source"])
    y = np.asarray(data["y"])
    kernel = normalize_kernel(raw_kernel)
    alpha = alpha_star(source, kernel)
    safety_alpha = max(alpha * (1.0 + 1.0e-7), alpha + 1.0e-10)
    gamma_l = safety_alpha * kernel
    gamma_g = source + gamma_l
    denominator = KAPPA * abs(y[int(data["input_index"])]) ** 2
    return {
        "alpha_star": alpha,
        "rate_required": float(
            max(
                np.max(np.linalg.eigvalsh(gamma_g)),
                np.max(np.linalg.eigvalsh(gamma_l)),
            )
        ),
        "gamma_g_min": float(np.min(np.linalg.eigvalsh(gamma_g))),
        "gamma_l_min": float(np.min(np.linalg.eigvalsh(gamma_l))),
        "kernel_norm": float(np.max(np.linalg.eigvalsh(kernel))),
        "kernel_residual": float(
            np.linalg.norm(kernel @ y) / max(np.linalg.norm(y), 1.0e-300)
        ),
        "noise": float(
            np.real(np.vdot(y, gamma_g @ y)) / max(denominator, 1.0e-300)
        ),
        "fixed_noise": float(data["fixed"]),
    }


def local_constraints(
    matrix: cp.Variable, n: int, block_size: int
) -> list[cp.Constraint]:
    return [
        matrix[i, j] == 0
        for i in range(n)
        for j in range(i + 1, n)
        if j // block_size > i // block_size + 1
    ]


def solve_rate_sdp(
    data: dict[str, object], rate_cap: float, block_size: int
) -> dict[str, object]:
    y = np.asarray(data["y"])
    y_norm_sq = float(np.vdot(y, y).real)
    y_objective = y / np.sqrt(y_norm_sq)
    source = np.asarray(data["source"])
    n = len(y)
    eye = np.eye(n, dtype=complex)
    gain = cp.Variable((n, n), hermitian=True)
    loss = cp.Variable((n, n), hermitian=True)
    constraints: list[cp.Constraint] = [
        gain >> 0,
        loss >> 0,
        rate_cap * eye - gain >> 0,
        rate_cap * eye - loss >> 0,
        gain - loss == source,
    ]
    constraints += local_constraints(gain, n, block_size)
    constraints += local_constraints(loss, n, block_size)
    # Optimize only the nonnegative excess term.  Subtracting the fixed
    # source contribution avoids catastrophic cancellation for skin-localized
    # y and prevents a numerical SDP solution from falling below the analytic
    # fixed-drift lower bound.
    denominator = KAPPA * abs(y[int(data["input_index"])]) ** 2
    extra_scale = y_norm_sq / max(denominator, 1.0e-300)
    objective = cp.Minimize(
        extra_scale * cp.real(cp.quad_form(y_objective, gain - source))
    )
    problem = cp.Problem(objective, constraints)
    try:
        value = problem.solve(
            solver="CLARABEL",
            tol_gap_abs=2.0e-7,
            tol_feas=2.0e-7,
            tol_gap_rel=2.0e-7,
            max_iter=500,
        )
    except cp.error.SolverError:
        value = problem.solve(
            solver="SCS",
            eps=2.0e-6,
            max_iters=60000,
            normalize=True,
        )
    return {
        "status": problem.status,
        "noise": (
            float(data["fixed"]) + float(value)
            if value is not None
            else np.nan
        ),
        "extra_noise": float(value) if value is not None else np.nan,
        "fixed_noise": float(data["fixed"]),
        "rate_cap": rate_cap,
    }


def solve_exact_optimum_rate(
    data: dict[str, object], n_cells: int, block_size: int
) -> dict[str, object]:
    """Minimize the local rate while exactly saturating the fixed-drift bound.

    A PSD nearest-neighbor loss Gramian that annihilates y can be decomposed
    into PSD terms on adjacent cell pairs.  Null-space bases impose that
    condition exactly, which is substantially better conditioned than asking
    a generic SDP to resolve Gamma_l y=0 when y is exponentially localized.
    """
    source = np.asarray(data["source"], dtype=complex)
    y = np.asarray(data["y"], dtype=complex)
    dimension = len(y)
    loss = cp.Constant(np.zeros((dimension, dimension), dtype=complex))
    terms: list[tuple[np.ndarray, cp.Variable]] = []
    constraints: list[cp.Constraint] = []
    for cell in range(n_cells - 1):
        indices = np.concatenate(
            [
                np.arange(block_size * cell, block_size * (cell + 1)),
                np.arange(block_size * (cell + 1), block_size * (cell + 2)),
            ]
        )
        pair = y[indices]
        basis = null_space(pair.conj().reshape(1, -1))
        gram = cp.Variable((basis.shape[1], basis.shape[1]), hermitian=True)
        constraints.append(gram >> 0)
        embedding = np.zeros((dimension, basis.shape[1]), dtype=complex)
        embedding[indices, :] = basis
        terms.append((embedding, gram))
        loss = loss + embedding @ gram @ embedding.conj().T

    gain = cp.Constant(source) + loss
    rate = cp.Variable(nonneg=True)
    eye = np.eye(dimension, dtype=complex)
    constraints += [gain >> 0, rate * eye - loss >> 0, rate * eye - gain >> 0]
    problem = cp.Problem(cp.Minimize(rate), constraints)
    try:
        value = problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1.0e-8,
            tol_feas=1.0e-8,
            max_iter=1000,
        )
    except cp.error.SolverError:
        value = problem.solve(
            solver="SCS",
            eps=1.0e-6,
            max_iters=200000,
            normalize=True,
        )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or value is None:
        return {"status": problem.status, "rate": np.nan}
    loss_value = np.zeros((dimension, dimension), dtype=complex)
    for embedding, gram in terms:
        loss_value += embedding @ gram.value @ embedding.conj().T
    gain_value = source + loss_value
    return {
        "status": problem.status,
        "rate": float(value),
        "loss_norm": float(np.max(np.linalg.eigvalsh(loss_value))),
        "gain_norm": float(np.max(np.linalg.eigvalsh(gain_value))),
        "gain_min": float(np.min(np.linalg.eigvalsh(gain_value))),
        "kernel_residual": float(
            np.linalg.norm(loss_value @ y)
            / max(np.linalg.norm(loss_value) * np.linalg.norm(y), 1.0e-300)
        ),
        "noise": float(data["fixed"]),
    }


def scaling_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for n in SCALAR_SIZES:
        for omega in SCALAR_OMEGAS:
            data = response_data(scalar_drift(n), omega)
            if float(data["fixed"]) <= 0.0:
                continue
            metrics = explicit_metrics(data, scalar_kernel(np.asarray(data["y"])))
            rows.append(
                {
                    "model": "scalar",
                    "N": float(n),
                    "omega": omega,
                    "fixed_noise": float(data["fixed"]),
                    **metrics,
                }
            )
    for n_cells in MULTIBAND_SIZES:
        for omega in MULTIBAND_OMEGAS:
            data = response_data(
                two_band_drift(n_cells),
                omega,
                input_index=0,
                output_index=2 * n_cells - 2,
            )
            if float(data["fixed"]) <= 0.0:
                continue
            metrics = explicit_metrics(
                data, block_kernel(np.asarray(data["y"]), n_cells)
            )
            rows.append(
                {
                    "model": "two_band",
                    "N": float(n_cells),
                    "omega": omega,
                    "fixed_noise": float(data["fixed"]),
                    **metrics,
                }
            )
    return rows


def rate_sdp_rows() -> list[dict[str, object]]:
    if not RUN_FULL_RATE_SDP:
        return []
    cases = [
        ("scalar", 28, 0.0, 1),
        ("two_band", 10, 0.15, 2),
    ]
    rows: list[dict[str, object]] = []
    for model, size, omega, block_size in cases:
        if model == "scalar":
            data = response_data(scalar_drift(size), omega)
            metrics = explicit_metrics(data, scalar_kernel(np.asarray(data["y"])))
        else:
            data = response_data(
                two_band_drift(size),
                omega,
                input_index=0,
                output_index=2 * size - 2,
            )
            metrics = explicit_metrics(
                data, block_kernel(np.asarray(data["y"]), size)
            )
        for ratio in (0.5, 0.9, 1.0, 1.1, 2.0):
            result = solve_rate_sdp(
                data, ratio * metrics["rate_required"], block_size
            )
            rows.append(
                {
                    "model": model,
                    "N": float(size),
                    "omega": omega,
                    "ratio": ratio,
                    "explicit_rate_required": metrics["rate_required"],
                    **result,
                }
            )
    return rows


def exact_rate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, size, omega, block_size in EXACT_RATE_CASES:
        if model == "scalar":
            data = response_data(scalar_drift(size), omega)
        else:
            data = response_data(
                two_band_drift(size),
                omega,
                input_index=0,
                output_index=2 * size - 2,
            )
        if float(data["fixed"]) <= 0.0:
            continue
        result = solve_exact_optimum_rate(data, size, block_size)
        rows.append(
            {
                "model": model,
                "N": float(size),
                "omega": omega,
                "fixed_noise": float(data["fixed"]),
                **result,
            }
        )
    return rows


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    rows: list[dict[str, float]], exact_rows: list[dict[str, object]]
) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for omega, marker in ((0.0, "o-"), (0.03, "s--")):
        subset = [
            row
            for row in rows
            if row["model"] == "scalar" and row["omega"] == omega
        ]
        subset.sort(key=lambda row: row["N"])
        if subset:
            axes[0].semilogy(
                [row["N"] for row in subset],
                [row["rate_required"] for row in subset],
                marker,
                ms=3,
                lw=1.0,
                label=rf"explicit, $\omega={omega:g}$",
            )
    exact_scalar = [row for row in exact_rows if row["model"] == "scalar"]
    if exact_scalar:
        axes[0].semilogy(
            [row["N"] for row in exact_scalar],
            [row["rate"] for row in exact_scalar],
            "kx",
            ms=5,
            mew=1.0,
            label="exact-rate SDP",
        )
    axes[0].set_title("(a) Scalar finite-rate resource", loc="left", fontsize=9.0)
    axes[0].set_xlabel(r"size $N$")
    axes[0].set_ylabel(r"$R_{\rm req}$")
    axes[0].legend(frameon=False, fontsize=7.0)

    for omega, marker in ((0.0, "o-"), (0.10, "s--"), (0.15, "^-")):
        subset = [
            row
            for row in rows
            if row["model"] == "two_band" and row["omega"] == omega
        ]
        subset.sort(key=lambda row: row["N"])
        if subset:
            axes[1].semilogy(
                [row["N"] for row in subset],
                [row["rate_required"] for row in subset],
                marker,
                ms=3,
                lw=1.0,
                label=rf"explicit, $\omega={omega:g}$",
            )
    exact_multiband = [row for row in exact_rows if row["model"] == "two_band"]
    if exact_multiband:
        axes[1].semilogy(
            [row["N"] for row in exact_multiband],
            [row["rate"] for row in exact_multiband],
            "kx",
            ms=5,
            mew=1.0,
            label="exact-rate SDP",
        )
    axes[1].set_title("(b) Multiband interference", loc="left", fontsize=9.0)
    axes[1].set_xlabel(r"cells $N$")
    axes[1].set_ylabel(r"$R_{\rm req}$")
    axes[1].legend(frameon=False, fontsize=7.0)
    for axis in axes:
        axis.grid(alpha=0.18, lw=0.5)
    for extension in ("pdf", "png"):
        path = FIG_DIR / f"rate_constrained_local_cp.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        (ARXIV_FIG_DIR / path.name).write_bytes(path.read_bytes())
    plt.close(fig)


def write_summary(
    rows: list[dict[str, float]],
    sdp_rows: list[dict[str, object]],
    exact_rows: list[dict[str, object]],
) -> None:
    if sdp_rows:
        capped_note = [
            "The rate-capped SDP rows test feasibility and noise at caps relative to",
            "the explicit construction. Infeasible rows lie below the tested rate",
            "threshold; feasible rows approach the fixed-drift optimum.",
        ]
    else:
        capped_note = [
            "The slower generic rate-capped scan is disabled by default.",
            "Set RUN_FULL_RATE_SDP=1 to regenerate those rows.",
        ]
    lines = [
        "# Rate-constrained local CP optimization",
        "",
        "The explicit block-local kernel is normalized to unit spectral norm.",
        "The reported alpha_star is therefore a rate-scale quantity.",
        "",
        f"- Scaling samples: {len(rows)}",
        f"- Maximum required rate: {max(row['rate_required'] for row in rows):.6e}",
        f"- Minimum required rate: {min(row['rate_required'] for row in rows):.6e}",
        f"- Maximum alpha_star: {max(row['alpha_star'] for row in rows):.6e}",
        f"- Minimum alpha_star: {min(row['alpha_star'] for row in rows):.6e}",
        (
            "- Maximum explicit noise/formula error: "
            f"{max(abs(row['noise'] - row['fixed_noise']) for row in rows):.6e}"
        ),
        "",
        *capped_note,
        "",
        "The exact-rate SDP imposes Gamma_l y = 0 through local pair null-space",
        "bases and minimizes max(||Gamma_l||, ||Gamma_g||).  It is the minimum",
        "rate needed to attain the fixed-drift noise floor, not merely a bound",
        "from one chosen kernel normalization.",
        f"- Exact-rate samples: {len(exact_rows)}",
        (
            "- Exact-rate range: "
            f"{min(row['rate'] for row in exact_rows):.6e} to "
            f"{max(row['rate'] for row in exact_rows):.6e}"
            if exact_rows
            else "- Exact-rate range: unavailable"
        ),
        (
            "- Largest exact-rate PSD residual: "
            f"{max(max(-float(row['gain_min']), 0.0) for row in exact_rows):.6e}"
            if exact_rows
            else "- Largest exact-rate PSD residual: unavailable"
        ),
    ]
    (RES_DIR / "reports/rate_constrained_local_cp_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    rows = scaling_rows()
    sdp_rows = rate_sdp_rows()
    exact_rows = exact_rate_rows()
    write_csv("rate_constrained_local_cp_scaling.csv", rows)
    if sdp_rows:
        write_csv("rate_constrained_local_cp_sdp.csv", sdp_rows)
    else:
        stale_sdp = RES_DIR / "rate_constrained_local_cp_sdp.csv"
        if stale_sdp.exists():
            stale_sdp.unlink()
    write_csv("rate_constrained_local_cp_exact_rate.csv", exact_rows)
    write_summary(rows, sdp_rows, exact_rows)
    make_figure(rows, exact_rows)
    print((RES_DIR / "reports/rate_constrained_local_cp_summary.txt").read_text(), end="")
    for row in sdp_rows:
        print(row)
    for row in exact_rows:
        print(row)


if __name__ == "__main__":
    main()
