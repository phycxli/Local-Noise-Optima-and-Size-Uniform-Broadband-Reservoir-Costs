"""Locality-constrained CP noise optimization for a fixed drift matrix.

For a fixed first-moment drift X and fixed passive probes, the intrinsic gain
and loss Gram matrices obey

    Gamma_gain - Gamma_loss = X + X.T.

At zero frequency the input-referred output excess noise is linear in
Gamma_gain.  This script solves the resulting SDP for unrestricted, nearest-
neighbor, range-2, and range-3 Gram matrices, and compares the SDP with an
explicit nearest-neighbor kernel construction.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import cvxpy as cp
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


PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)
KAPPA = 0.2
SDP_SIZES = (20, 28, 32, 35, 40, 50, 60)
EXPLICIT_SIZES = tuple(range(20, 101, 5))
RANGES = (None, 1, 2, 3)
PARAMETER_CASES = (
    (0.10, 0.90, 0.20, 12),
    (0.20, 1.10, 0.20, 20),
    (0.25, 1.20, 0.20, 40),
    (0.35, 1.30, 0.20, 40),
    (0.40, 1.34, 0.20, 30),
    (0.50, 1.46, 0.20, 40),
    (0.25, 1.20, 0.05, 60),
    (0.25, 1.20, 1.00, 30),
    (0.25, 1.20, 2.00, 30),
)

FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def response_data(n_sites: int) -> dict[str, object]:
    x = local_bosonic_hatano_drift(n_sites, PARAMS).real
    x_probe = x.copy()
    x_probe[0, 0] -= 0.5 * KAPPA
    x_probe[-1, -1] -= 0.5 * KAPPA

    output = np.zeros(n_sites)
    output[-1] = 1.0
    y = np.linalg.solve((-x_probe).T, output)
    hermitian_part = x + x.T
    power_gain = float(KAPPA**2 * y[0] ** 2)
    caves_floor = max(0.0, 1.0 - 1.0 / max(power_gain, 1.0e-300))
    fixed_drift_value = float(
        y @ hermitian_part @ y / (KAPPA * max(y[0] ** 2, 1.0e-300))
    )
    reflection_penalty = float((y[-1] - 1.0 / KAPPA) ** 2 / max(y[0] ** 2, 1.0e-300))
    _, gamma_gain, _ = local_bosonic_hatano_lindblad_matrices(n_sites, PARAMS)
    current_noise = float(
        y @ gamma_gain.real @ y / (KAPPA * max(y[0] ** 2, 1.0e-300))
    )
    return {
        "x": x,
        "y": y,
        "hermitian_part": hermitian_part,
        "power_gain": power_gain,
        "caves_floor": caves_floor,
        "fixed_drift_value": fixed_drift_value,
        "reflection_penalty": reflection_penalty,
        "caves_relation_residual": fixed_drift_value
        - caves_floor
        - reflection_penalty,
        "current_noise": current_noise,
    }


def solve_sdp(n_sites: int, range_value: int | None) -> dict[str, float]:
    data = response_data(n_sites)
    y = data["y"]
    source = data["hermitian_part"]
    objective_matrix = np.outer(y, y)

    gain = cp.Variable((n_sites, n_sites), symmetric=True)
    constraints = [gain >> 0, gain - source >> 0]
    if range_value is not None:
        constraints.extend(
            gain[i, j] == 0
            for i in range(n_sites)
            for j in range(i + range_value + 1, n_sites)
        )

    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(objective_matrix, gain))), constraints
    )
    value = problem.solve(
        solver="CLARABEL",
        tol_gap_abs=2.0e-8,
        tol_feas=2.0e-8,
        tol_gap_rel=2.0e-8,
        max_iter=600,
    )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"SDP failed for N={n_sites}, R={range_value}: {problem.status}"
        )

    gain_value = np.asarray(gain.value)
    loss_value = gain_value - source
    denominator = KAPPA * max(float(y[0] ** 2), 1.0e-300)
    return {
        "noise": float(max(float(value), 0.0) / denominator),
        "gain_psd_min": float(np.min(np.linalg.eigvalsh(gain_value))),
        "loss_psd_min": float(np.min(np.linalg.eigvalsh(loss_value))),
        "constraint_residual": float(
            np.linalg.norm(gain_value - loss_value - source)
        ),
        "status": problem.status,
    }


def explicit_kernel_construction(n_sites: int) -> dict[str, float]:
    """Construct a local reservoir with the optimal fixed-drift noise value.

    For y^T S y > 0, the edge vectors v_j = y_{j+1} e_j - y_j e_{j+1}
    span the subspace orthogonal to y.  L_alpha = alpha sum_j v_j v_j^T is
    PSD, nearest-neighbor, and satisfies L_alpha y = 0.  For sufficiently large
    alpha, G_alpha = S + L_alpha is PSD as well, giving a local CP factorization
    with y^T G_alpha y = y^T S y.
    """

    data = response_data(n_sites)
    y = np.asarray(data["y"], dtype=float)
    source = np.asarray(data["hermitian_part"], dtype=float)
    scale = max(float(np.max(np.abs(y))), 1.0e-300)
    normalized = y / scale

    kernel = np.zeros((n_sites, n_sites))
    for j in range(n_sites - 1):
        vector = np.zeros(n_sites)
        vector[j] = normalized[j + 1]
        vector[j + 1] = -normalized[j]
        kernel += np.outer(vector, vector)

    def minimum_eigenvalue(alpha: float) -> float:
        return float(np.min(np.linalg.eigvalsh(source + alpha * kernel)))

    if float(y @ source @ y) <= 0.0:
        return {
            "noise": 0.0,
            "alpha": np.nan,
            "gain_psd_min": np.nan,
            "loss_psd_min": np.nan,
            "kernel_residual": float(np.linalg.norm(kernel @ normalized)),
            "construction_valid": 0.0,
        }

    upper = 1.0e-12
    while minimum_eigenvalue(upper) < -1.0e-9 and upper < 1.0e14:
        upper *= 10.0
    if minimum_eigenvalue(upper) < -1.0e-8:
        raise RuntimeError(f"kernel construction failed at N={n_sites}")
    lower = upper / 10.0
    for _ in range(90):
        middle = 0.5 * (lower + upper)
        if minimum_eigenvalue(middle) >= 0.0:
            upper = middle
        else:
            lower = middle

    loss = upper * kernel
    gain = source + loss
    denominator = KAPPA * max(float(y[0] ** 2), 1.0e-300)
    return {
        "noise": float(max(float(y @ gain @ y / denominator), 0.0)),
        "alpha": float(upper),
        "gain_psd_min": float(np.min(np.linalg.eigvalsh(gain))),
        "loss_psd_min": float(np.min(np.linalg.eigvalsh(loss))),
        "kernel_residual": float(np.linalg.norm(loss @ normalized)),
        "construction_valid": 1.0,
    }


def collect_rows() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    sdp_rows: list[dict[str, float]] = []
    for n_sites in SDP_SIZES:
        data = response_data(n_sites)
        row = {
            "N": float(n_sites),
            "power_gain": float(data["power_gain"]),
            "caves_floor": float(data["caves_floor"]),
            "fixed_drift_value": float(data["fixed_drift_value"]),
            "reflection_penalty": float(data["reflection_penalty"]),
            "caves_relation_residual": float(data["caves_relation_residual"]),
            "current_noise": float(data["current_noise"]),
        }
        for range_value in RANGES:
            label = "inf" if range_value is None else str(range_value)
            result = solve_sdp(n_sites, range_value)
            row[f"sdp_noise_R{label}"] = result["noise"]
            row[f"sdp_gain_psd_min_R{label}"] = result["gain_psd_min"]
            row[f"sdp_loss_psd_min_R{label}"] = result["loss_psd_min"]
        row["locality_gap_R1"] = (
            row["sdp_noise_R1"] - row["sdp_noise_Rinf"]
        )
        sdp_rows.append(row)

    explicit_rows: list[dict[str, float]] = []
    for n_sites in EXPLICIT_SIZES:
        data = response_data(n_sites)
        result = explicit_kernel_construction(n_sites)
        explicit_rows.append(
            {
                "N": float(n_sites),
                "power_gain": float(data["power_gain"]),
                "caves_floor": float(data["caves_floor"]),
                "fixed_drift_value": float(data["fixed_drift_value"]),
                "reflection_penalty": float(data["reflection_penalty"]),
                "caves_relation_residual": float(data["caves_relation_residual"]),
                "current_noise": float(data["current_noise"]),
                **result,
            }
        )
    return sdp_rows, explicit_rows


def collect_parameter_rows() -> list[dict[str, float]]:
    global PARAMS, KAPPA

    default_params = PARAMS
    default_kappa = KAPPA
    rows: list[dict[str, float]] = []
    try:
        for t_left, gamma, kappa, n_sites in PARAMETER_CASES:
            PARAMS = LocalBosonicHatanoParams(
                t_right=1.0, t_left=t_left, gamma=gamma
            )
            KAPPA = kappa
            data = response_data(n_sites)
            nonlocal_result = solve_sdp(n_sites, None)
            local_result = solve_sdp(n_sites, 1)
            explicit_result = explicit_kernel_construction(n_sites)
            rows.append(
                {
                    "t_left": t_left,
                    "gamma": gamma,
                    "kappa": kappa,
                    "N": float(n_sites),
                    "power_gain": float(data["power_gain"]),
                    "caves_floor": float(data["caves_floor"]),
                    "fixed_drift_value": float(data["fixed_drift_value"]),
                    "sdp_noise_Rinf": nonlocal_result["noise"],
                    "sdp_noise_R1": local_result["noise"],
                    "explicit_noise_R1": explicit_result["noise"],
                    "locality_gap_R1": local_result["noise"]
                    - nonlocal_result["noise"],
                    "explicit_formula_error": explicit_result["noise"]
                    - float(data["fixed_drift_value"]),
                    "caves_relation_residual": float(
                        data["caves_relation_residual"]
                    ),
                }
            )
    finally:
        PARAMS = default_params
        KAPPA = default_kappa
    return rows


def positive(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[result <= 0.0] = np.nan
    return result


def make_figure(sdp_rows: list[dict[str, float]], explicit_rows: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
        }
    )

    sdp_n = np.array([row["N"] for row in sdp_rows])
    exp_n = np.array([row["N"] for row in explicit_rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.semilogy(
        exp_n,
        positive(np.array([row["current_noise"] for row in explicit_rows])),
        "o-",
        ms=2.8,
        lw=1.0,
        label="original bonds",
    )
    ax.semilogy(
        sdp_n,
        positive(np.array([row["sdp_noise_Rinf"] for row in sdp_rows])),
        "s-",
        ms=3,
        lw=1.0,
        label="SDP, nonlocal",
    )
    ax.semilogy(
        sdp_n,
        positive(np.array([row["sdp_noise_R1"] for row in sdp_rows])),
        "^-",
        ms=3,
        lw=1.0,
        label="SDP, nearest neighbor",
    )
    ax.semilogy(
        exp_n,
        positive(np.array([row["fixed_drift_value"] for row in explicit_rows])),
        "k--",
        lw=1.0,
        label="fixed-drift optimum",
    )
    ax.semilogy(
        exp_n,
        positive(np.array([row["caves_floor"] for row in explicit_rows])),
        color="0.45",
        ls=":",
        lw=1.0,
        label="Caves floor",
    )
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel(r"input-referred excess noise")
    ax.set_title("(a) Optimized noise", loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for range_value, marker in zip((1, 2, 3), ("o", "s", "^")):
        label = str(range_value)
        values = np.array([row[f"sdp_noise_R{label}"] for row in sdp_rows])
        ax.plot(sdp_n, values, marker + "-", ms=3, lw=0.9, label=rf"$R={range_value}$")
    ax.plot(
        sdp_n,
        [row["sdp_noise_Rinf"] for row in sdp_rows],
        "k--",
        lw=1.0,
        label=r"$R=\infty$",
    )
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel(r"$\mathcal{N}_{\rm in}^{\rm exc}$")
    ax.set_title("(b) Locality dependence", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    gaps = np.array([row["locality_gap_R1"] for row in sdp_rows])
    ax.semilogy(sdp_n, positive(np.abs(gaps)), "o-", ms=3, lw=1.0)
    ax.axhline(1.0e-6, color="0.4", ls=":", lw=0.8, label="solver scale")
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel(r"$|\mathcal{N}_{R=1}-\mathcal{N}_{\infty}|$")
    ax.set_title("(c) No resolved locality penalty", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    penalty = np.array([row["reflection_penalty"] for row in explicit_rows])
    ax.semilogy(exp_n, positive(penalty), "o-", ms=2.8, lw=1.0,
                label=r"$\mathcal{N}_{\rm opt}-\mathcal{N}_{\rm Caves}$")
    fit_mask = (exp_n >= 50) & np.isfinite(penalty) & (penalty > 0.0)
    fit = np.polyfit(exp_n[fit_mask], np.log(penalty[fit_mask]), deg=1)
    fit_curve = np.exp(fit[1] + fit[0] * exp_n)
    ax.semilogy(exp_n, fit_curve, "k--", lw=1.0,
                label=rf"fit slope ${fit[0]:.3f}$")
    ax.set_xlabel(r"size $N$")
    ax.set_ylabel("reflection penalty")
    ax.set_title("(d) Approach to the Caves limit", loc="left")
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.18, lw=0.5)

    for extension in ("pdf", "png"):
        path = FIG_DIR / f"local_noise_optimization.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        (ARXIV_FIG_DIR / path.name).write_bytes(path.read_bytes())
    plt.close(fig)


def write_outputs(
    sdp_rows: list[dict[str, float]],
    explicit_rows: list[dict[str, float]],
    parameter_rows: list[dict[str, float]],
) -> None:
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "local_noise_optimization_sdp.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sdp_rows[0]))
        writer.writeheader()
        writer.writerows(sdp_rows)
    with (RES_DIR / "local_noise_optimization_explicit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(explicit_rows[0]))
        writer.writeheader()
        writer.writerows(explicit_rows)
    with (RES_DIR / "local_noise_optimization_parameters.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0]))
        writer.writeheader()
        writer.writerows(parameter_rows)

    high_gain = [row for row in sdp_rows if row["power_gain"] > 5.0]
    max_gap = max(abs(row["locality_gap_R1"]) for row in high_gain)
    explicit_valid = [row for row in explicit_rows if row["construction_valid"] > 0.5]
    max_formula_error = max(
        abs(row["noise"] - row["fixed_drift_value"]) for row in explicit_valid
    )
    max_alpha_residual = max(row["kernel_residual"] for row in explicit_valid)
    max_caves_residual = max(
        abs(row["caves_relation_residual"]) for row in explicit_rows
        if row["power_gain"] > 1.0
    )
    max_parameter_gap = max(abs(row["locality_gap_R1"]) for row in parameter_rows)
    max_parameter_formula_error = max(
        abs(row["explicit_formula_error"]) for row in parameter_rows
    )
    lines = [
        "# Locality-constrained CP noise optimization",
        "",
        f"- Probe rate: `{KAPPA}`",
        f"- Largest high-gain SDP locality gap |R=1 - R=infinity|: `{max_gap:.3e}`",
        f"- Largest explicit-construction noise/formula discrepancy: `{max_formula_error:.3e}`",
        f"- Largest explicit kernel residual ||L y|| in normalized units: `{max_alpha_residual:.3e}`",
        f"- Largest fixed-drift/Caves identity residual in the gain regime: `{max_caves_residual:.3e}`",
        f"- Largest R=1/nonlocal gap over the parameter scan: `{max_parameter_gap:.3e}`",
        f"- Largest explicit/formula error over the parameter scan: `{max_parameter_formula_error:.3e}`",
        "- The optimized nearest-neighbor reservoir reaches the fixed-drift optimum; the original bond factorization is highly nonoptimal.",
    ]
    (RES_DIR / "reports/local_noise_optimization.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    sdp_rows, explicit_rows = collect_rows()
    parameter_rows = collect_parameter_rows()
    write_outputs(sdp_rows, explicit_rows, parameter_rows)
    make_figure(sdp_rows, explicit_rows)
    print((RES_DIR / "reports/local_noise_optimization.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
