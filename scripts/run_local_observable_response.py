"""Observable consequences of the local pseudospectral Krylov reservoir.

This script turns the rapidity anomaly into directly measurable Gaussian
open-system quantities:

1. steady normal covariance / occupation profile,
2. covariance buildup times,
3. zero-frequency linear susceptibility,
4. two-time retarded Green function from a local impulse.

For the local bosonic reservoir, the normal covariance obeys

    dN/dt = X N + N X^dagger + Gamma_gain,

where N_ij = <a_i^dagger a_j>.  We integrate this Lyapunov equation using the
tridiagonal structure of X, avoiding the ill-conditioned eigenbasis of the
non-normal rapidity block.
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
RES_DIR = ROOT / "results"
PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)


def exact_obc_gap(n_sites: int, p: LocalBosonicHatanoParams = PARAMS) -> float:
    return p.gamma - 2.0 * math.sqrt(p.t_right * p.t_left) * math.cos(
        math.pi / (n_sites + 1)
    )


def gain_matrix(n_sites: int, p: LocalBosonicHatanoParams = PARAMS) -> np.ndarray:
    _, gamma_gain, _ = local_bosonic_hatano_lindblad_matrices(n_sites, p)
    return gamma_gain.real


def covariance_rhs(
    covariance: np.ndarray,
    gamma_gain: np.ndarray,
    p: LocalBosonicHatanoParams = PARAMS,
) -> np.ndarray:
    """Fast RHS for dN/dt = XN + NX^T + Gamma_gain."""

    out = -2.0 * p.gamma * covariance.copy()
    out[1:, :] += p.t_right * covariance[:-1, :]
    out[:-1, :] += p.t_left * covariance[1:, :]
    out[:, 1:] += p.t_right * covariance[:, :-1]
    out[:, :-1] += p.t_left * covariance[:, 1:]
    return out + gamma_gain


def integrate_covariance_observables(
    n_sites: int,
    t_max: float,
    dt: float = 0.04,
    store_every: int = 25,
    p: LocalBosonicHatanoParams = PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate covariance from vacuum and return compact observables.

    The last stored covariance is used as the steady-state approximation.  The
    returned relative error curve is computed against that final covariance.
    """

    gamma_gain = gain_matrix(n_sites, p)
    covariance = np.zeros((n_sites, n_sites), dtype=float)
    times: list[float] = []
    totals: list[float] = []
    right_edge: list[float] = []
    max_occupation: list[float] = []
    snapshots: list[np.ndarray] = []

    steps = int(round(t_max / dt))
    for step in range(steps + 1):
        if step % store_every == 0:
            diagonal = np.diag(covariance)
            times.append(step * dt)
            totals.append(float(np.trace(covariance)))
            right_edge.append(float(diagonal[-1]))
            max_occupation.append(float(np.max(diagonal)))
            snapshots.append(covariance.copy())
        if step == steps:
            break
        k1 = covariance_rhs(covariance, gamma_gain, p)
        k2 = covariance_rhs(covariance + 0.5 * dt * k1, gamma_gain, p)
        k3 = covariance_rhs(covariance + 0.5 * dt * k2, gamma_gain, p)
        k4 = covariance_rhs(covariance + dt * k3, gamma_gain, p)
        covariance = covariance + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if step % 2000 == 0:
            covariance = 0.5 * (covariance + covariance.T)

    steady = 0.5 * (covariance + covariance.T)
    steady_norm = np.linalg.norm(steady, "fro")
    errors = np.array(
        [
            np.linalg.norm(snapshot - steady, "fro") / max(steady_norm, 1e-300)
            for snapshot in snapshots
        ],
        dtype=float,
    )
    residual = np.linalg.norm(covariance_rhs(steady, gamma_gain, p), "fro") / max(
        np.linalg.norm(gamma_gain, "fro"), 1e-300
    )
    return (
        np.array(times),
        np.array(totals),
        np.array(right_edge),
        np.array(max_occupation),
        errors,
        float(residual),
    )


def first_crossing_time(times: np.ndarray, values: np.ndarray, fraction: float) -> float:
    target = fraction * values[-1]
    hits = np.flatnonzero(values >= target)
    return float(times[hits[0]]) if len(hits) else float("nan")


def settling_time(times: np.ndarray, errors: np.ndarray, threshold: float) -> float:
    above = np.flatnonzero(errors > threshold)
    if len(above) == 0:
        return 0.0
    idx = int(above[-1]) + 1
    return float(times[idx]) if idx < len(times) else float("nan")


def apply_drift(state: np.ndarray, p: LocalBosonicHatanoParams = PARAMS) -> np.ndarray:
    out = -p.gamma * state.copy()
    out[1:] += p.t_right * state[:-1]
    out[:-1] += p.t_left * state[1:]
    return out


def impulse_trace(
    n_sites: int,
    t_max: float,
    dt: float = 0.02,
    store_every: int = 25,
    p: LocalBosonicHatanoParams = PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return norm and end-to-end retarded Green amplitude after a left impulse."""

    state = np.zeros(n_sites, dtype=float)
    state[0] = 1.0
    times: list[float] = []
    norms: list[float] = []
    end_to_end: list[float] = []
    steps = int(round(t_max / dt))
    for step in range(steps + 1):
        if step % store_every == 0:
            times.append(step * dt)
            norms.append(float(np.linalg.norm(state)))
            end_to_end.append(float(abs(state[-1])))
        if step == steps:
            break
        k1 = apply_drift(state, p)
        k2 = apply_drift(state + 0.5 * dt * k1, p)
        k3 = apply_drift(state + 0.5 * dt * k2, p)
        k4 = apply_drift(state + dt * k3, p)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.array(times), np.array(norms), np.array(end_to_end)


def susceptibility_observables(
    n_sites: int,
    p: LocalBosonicHatanoParams = PARAMS,
) -> tuple[float, float, float, float]:
    """Return zero-frequency response observables."""

    x = local_bosonic_hatano_drift(n_sites, p)
    chi0 = np.linalg.inv(-x)
    singular = np.linalg.svd(chi0, compute_uv=False)[0]
    return (
        float(singular),
        float(abs(chi0[-1, 0])),
        float(abs(chi0[0, 0])),
        float(abs(chi0[-1, -1])),
    )


def scan() -> list[dict[str, float]]:
    sizes = [20, 40, 60, 80, 100]
    rows: list[dict[str, float]] = []
    velocity = PARAMS.t_right - PARAMS.t_left

    for n_sites in sizes:
        cov_tmax = 4.0 * n_sites / velocity
        times, total, right, max_occ, errors, residual = integrate_covariance_observables(
            n_sites, t_max=cov_tmax
        )
        impulse_times, impulse_norm, impulse_end = impulse_trace(
            n_sites, t_max=3.5 * n_sites / velocity
        )
        peak_end_idx = int(np.argmax(impulse_end))
        peak_norm_idx = int(np.argmax(impulse_norm))
        chi_norm, chi_end, chi_left, chi_right = susceptibility_observables(n_sites)
        gap = exact_obc_gap(n_sites)
        rows.append(
            {
                "N": float(n_sites),
                "gap_exact": gap,
                "inverse_gap": 1.0 / gap,
                "covariance_total_ss": float(total[-1]),
                "covariance_right_edge_ss": float(right[-1]),
                "covariance_max_occupation_ss": float(max_occ[-1]),
                "covariance_t50_total": first_crossing_time(times, total, 0.5),
                "covariance_t90_total": first_crossing_time(times, total, 0.9),
                "covariance_t90_right_edge": first_crossing_time(times, right, 0.9),
                "covariance_error_settling_einv": settling_time(times, errors, math.exp(-1.0)),
                "covariance_residual": residual,
                "susceptibility_norm_omega0": chi_norm,
                "susceptibility_end_to_end_omega0": chi_end,
                "susceptibility_left_local_omega0": chi_left,
                "susceptibility_right_local_omega0": chi_right,
                "retarded_green_end_peak": float(impulse_end[peak_end_idx]),
                "retarded_green_end_peak_time": float(impulse_times[peak_end_idx]),
                "retarded_green_norm_peak": float(impulse_norm[peak_norm_idx]),
                "retarded_green_norm_peak_time": float(impulse_times[peak_norm_idx]),
            }
        )

        if n_sites == 80:
            with (RES_DIR / "local_observable_covariance_trace_N80.csv").open(
                "w", newline="", encoding="utf-8"
            ) as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "time",
                        "total_occupation",
                        "right_edge_occupation",
                        "max_occupation",
                        "relative_covariance_error",
                    ]
                )
                writer.writerows(zip(times, total, right, max_occ, errors))
            with (RES_DIR / "local_observable_impulse_trace_N80.csv").open(
                "w", newline="", encoding="utf-8"
            ) as fh:
                writer = csv.writer(fh)
                writer.writerow(["time", "norm", "end_to_end_green"])
                writer.writerows(zip(impulse_times, impulse_norm, impulse_end))

    return rows


def write_outputs(rows: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    csv_path = RES_DIR / "local_observable_response_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cov_trace = np.loadtxt(
        RES_DIR / "local_observable_covariance_trace_N80.csv",
        delimiter=",",
        skiprows=1,
    )
    impulse_trace_data = np.loadtxt(
        RES_DIR / "local_observable_impulse_trace_N80.csv",
        delimiter=",",
        skiprows=1,
    )

    sizes = np.array([row["N"] for row in rows])
    inv_gap = np.array([row["inverse_gap"] for row in rows])
    cov_t90 = np.array([row["covariance_t90_right_edge"] for row in rows])
    cov_settle = np.array([row["covariance_error_settling_einv"] for row in rows])
    chi_norm = np.array([row["susceptibility_norm_omega0"] for row in rows])
    chi_end = np.array([row["susceptibility_end_to_end_omega0"] for row in rows])
    total_ss = np.array([row["covariance_total_ss"] for row in rows])
    right_ss = np.array([row["covariance_right_edge_ss"] for row in rows])
    green_peak_time = np.array([row["retarded_green_end_peak_time"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))

    ax = axes[0, 0]
    ax.plot(sizes, inv_gap, "o-", label="1 / rapidity gap")
    ax.plot(sizes, cov_t90, "s-", label="90% right-edge noise")
    ax.plot(sizes, cov_settle, "^-", label="covariance error settling")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("time")
    ax.set_title("Covariance times outgrow the gap time")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.semilogy(sizes, total_ss, "o-", label="total occupation")
    ax.semilogy(sizes, right_ss, "s--", label="right-edge occupation")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("steady occupation")
    ax.set_title("Steady Gaussian noise is skin-amplified")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.semilogy(sizes, chi_norm, "o-", label="||chi(0)||")
    ax.semilogy(sizes, chi_end, "s--", label="|chi_N1(0)|")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("zero-frequency response")
    ax.set_title("Susceptibility sees the pseudospectrum")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(cov_trace[:, 0], cov_trace[:, 2] / cov_trace[-1, 2], label="right-edge noise")
    ax.plot(cov_trace[:, 0], cov_trace[:, 4], label="relative covariance error")
    ax2 = ax.twinx()
    ax2.plot(
        impulse_trace_data[:, 0],
        impulse_trace_data[:, 2],
        color="tab:red",
        label="|G_N1(t)|",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("normalized covariance")
    ax2.set_ylabel("end-to-end Green function")
    ax.set_title("Observable traces for N=80")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="center right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "local_observable_response.png", dpi=220)
    plt.close(fig)

    note = f"""# Local Observable Response

This scan converts the local reservoir rapidity anomaly into measurable
Gaussian observables.

Model parameters:

- `t_R = {PARAMS.t_right}`
- `t_L = {PARAMS.t_left}`
- `gamma = {PARAMS.gamma}`

Observables:

- normal covariance `N_ij = <a_i^dagger a_j>`;
- covariance buildup and settling times;
- zero-frequency susceptibility `chi(0) = (-X)^-1`;
- end-to-end retarded Green function after a left-edge impulse.

At `N = {int(sizes[-1])}`:

- `1/gap = {inv_gap[-1]:.6g}`;
- right-edge covariance 90% buildup time = `{cov_t90[-1]:.6g}`;
- covariance error settling time = `{cov_settle[-1]:.6g}`;
- total steady occupation = `{total_ss[-1]:.6g}`;
- right-edge steady occupation = `{right_ss[-1]:.6g}`;
- `||chi(0)|| = {chi_norm[-1]:.6g}`;
- `|chi_N1(0)| = {chi_end[-1]:.6g}`;
- end-to-end Green peak time = `{green_peak_time[-1]:.6g}`.

Generated files:

- `figures/local_observable_response.png`
- `results/local_observable_response_scaling.csv`
- `results/local_observable_covariance_trace_N80.csv`
- `results/local_observable_impulse_trace_N80.csv`
"""
    (RES_DIR / "reports/local_observable_response.txt").write_text(note, encoding="utf-8")


def main() -> None:
    rows = scan()
    write_outputs(rows)
    print("Wrote results/local_observable_response_scaling.csv")
    print("Wrote figures/local_observable_response.png")


if __name__ == "__main__":
    main()
