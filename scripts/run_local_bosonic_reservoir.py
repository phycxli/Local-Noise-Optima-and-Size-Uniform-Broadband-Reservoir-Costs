"""Local active-reservoir realization of the pseudospectral Krylov phase.

This script removes the main weakness of the exact spectral embedding: the
jump operators are now strictly local.  For

    X_{j,j} = -gamma,  X_{j+1,j} = t_right,  X_{j,j+1} = t_left,

we use the local bosonic Lindblad construction

    H_{j+1,j} = i (t_right - t_left) / 2,
    L^g_j = sqrt(c) (a_j^dagger + a_{j+1}^dagger),
    L^l_j = sqrt(c) (a_j - a_{j+1}),
    L^0_j = sqrt(2 gamma) a_j,

where c = (t_right + t_left) / 2.  The first moments obey

    d<a>/dt = [-iH + (Gamma_gain - Gamma_loss)/2] <a> = X <a>.

The same non-normal rapidity block is therefore realized by nearest-neighbor
gain/loss jumps and local onsite loss.
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
    arnoldi_basis,
    biorthogonal_krylov_basis,
    biorthogonal_krylov_spread,
    local_bosonic_hatano_drift,
    local_bosonic_hatano_jump_summary,
    local_bosonic_hatano_lindblad_matrices,
    local_bosonic_hatano_reconstructed_drift,
    orthogonal_krylov_spread,
)


FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)


def exact_obc_gap(n_sites: int, p: LocalBosonicHatanoParams = PARAMS) -> float:
    return p.gamma - 2.0 * math.sqrt(p.t_right * p.t_left) * math.cos(
        math.pi / (n_sites + 1)
    )


def apply_drift(state: np.ndarray, p: LocalBosonicHatanoParams = PARAMS) -> np.ndarray:
    out = -p.gamma * state.copy()
    out[1:] += p.t_right * state[:-1]
    out[:-1] += p.t_left * state[1:]
    return out


def rk4_trajectory(
    n_sites: int,
    t_max: float,
    dt: float = 0.02,
    store_every: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros(n_sites, dtype=complex)
    state[0] = 1.0
    times: list[float] = []
    states: list[np.ndarray] = []
    time = 0.0
    steps = int(round(t_max / dt))
    for step in range(steps + 1):
        if step % store_every == 0:
            times.append(time)
            states.append(state.copy())
        k1 = apply_drift(state)
        k2 = apply_drift(state + 0.5 * dt * k1)
        k3 = apply_drift(state + 0.5 * dt * k2)
        k4 = apply_drift(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += dt
    return np.array(times), np.array(states)


def settling_time(times: np.ndarray, signal: np.ndarray, threshold: float) -> float:
    above = np.flatnonzero(signal > threshold)
    if len(above) == 0:
        return 0.0
    idx = int(above[-1]) + 1
    return float(times[idx]) if idx < len(times) else float("nan")


def midgap_singular_value(n_sites: int) -> float:
    gap = exact_obc_gap(n_sites)
    z = -0.5 * gap
    x = local_bosonic_hatano_drift(n_sites, PARAMS)
    return float(np.linalg.svd(z * np.eye(n_sites) - x, compute_uv=False)[-1])


def real_axis_pseudo_abscissa(
    n_sites: int,
    epsilon: float = 1e-8,
    n_grid: int = 160,
) -> float:
    gap = exact_obc_gap(n_sites)
    x = local_bosonic_hatano_drift(n_sites, PARAMS)
    xs = np.linspace(-1.2 * gap, 0.08, n_grid)
    ident = np.eye(n_sites)
    hits = []
    for z in xs:
        s_min = np.linalg.svd(z * ident - x, compute_uv=False)[-1]
        if s_min <= epsilon:
            hits.append(z)
    return float(max(hits)) if hits else float("nan")


def local_cp_checks(n_sites: int) -> dict[str, float]:
    target = local_bosonic_hatano_drift(n_sites, PARAMS)
    reconstructed = local_bosonic_hatano_reconstructed_drift(n_sites, PARAMS)
    h, gamma_gain, gamma_loss = local_bosonic_hatano_lindblad_matrices(n_sites, PARAMS)
    summary = local_bosonic_hatano_jump_summary(n_sites, PARAMS)
    h_locality_error = 0.0
    for i in range(n_sites):
        for j in range(n_sites):
            if abs(i - j) > 1:
                h_locality_error += abs(h[i, j]) ** 2
                h_locality_error += abs(gamma_gain[i, j]) ** 2
                h_locality_error += abs(gamma_loss[i, j]) ** 2
    checks = {
        "local_reconstruction_error": float(
            np.linalg.norm(reconstructed - target) / np.linalg.norm(target)
        ),
        "min_gain_eigenvalue": float(np.linalg.eigvalsh(gamma_gain).min()),
        "min_loss_eigenvalue": float(np.linalg.eigvalsh(gamma_loss).min()),
        "nonlocal_matrix_weight": float(math.sqrt(h_locality_error)),
        **summary,
    }
    return checks


def scan_scaling() -> list[dict[str, float]]:
    sizes = [20, 30, 40, 60, 80, 100]
    threshold = math.exp(-1.0)
    velocity = PARAMS.t_right - PARAMS.t_left
    rows: list[dict[str, float]] = []

    for n_sites in sizes:
        x = local_bosonic_hatano_drift(n_sites, PARAMS)
        times, states = rk4_trajectory(n_sites, t_max=3.5 * n_sites / velocity)
        norms = np.linalg.norm(states, axis=1)
        seed = np.zeros(n_sites, dtype=complex)
        seed[0] = 1.0
        arnoldi_vectors, _ = arnoldi_basis(x, seed, max_dim=n_sites)
        bi_v, bi_w, _ = biorthogonal_krylov_basis(x, seed, max_dim=n_sites)
        visible = norms > threshold
        if np.any(visible):
            visible_states = states[visible]
            orth_peak = float(
                max(orthogonal_krylov_spread(arnoldi_vectors, state) for state in visible_states)
            )
            bi_peak = float(
                max(biorthogonal_krylov_spread(bi_w, state) for state in visible_states)
            )
        else:
            orth_peak = 0.0
            bi_peak = 0.0
        gap = exact_obc_gap(n_sites)
        s_mid = midgap_singular_value(n_sites)
        rows.append(
            {
                "N": float(n_sites),
                "gap_exact": gap,
                "inverse_gap": 1.0 / gap,
                "pbc_abscissa": -PARAMS.gamma + PARAMS.t_right + PARAMS.t_left,
                "midgap_smin": s_mid,
                "log_midgap_smin": math.log(s_mid),
                "settling_time_einv": settling_time(times, norms, threshold),
                "norm_peak": float(np.max(norms)),
                "orthogonal_krylov_visible_peak": orth_peak,
                "biorthogonal_krylov_visible_peak": bi_peak,
                "biorthogonal_dimension": float(bi_v.shape[1]),
                "biorthogonality_error": float(
                    np.linalg.norm(bi_w.conj().T @ bi_v - np.eye(bi_v.shape[1]))
                ),
                "real_axis_alpha_eps_1e-8": real_axis_pseudo_abscissa(n_sites),
                **local_cp_checks(n_sites),
            }
        )
    return rows


def write_trace(n_sites: int = 80) -> None:
    x = local_bosonic_hatano_drift(n_sites, PARAMS)
    seed = np.zeros(n_sites, dtype=complex)
    seed[0] = 1.0
    arnoldi_vectors, _ = arnoldi_basis(x, seed, max_dim=n_sites)
    _, bi_w, _ = biorthogonal_krylov_basis(x, seed, max_dim=n_sites)
    times, states = rk4_trajectory(n_sites, t_max=3.5 * n_sites / (PARAMS.t_right - PARAMS.t_left))
    rows = []
    for time, state in zip(times, states):
        rows.append(
            [
                float(time),
                float(np.linalg.norm(state)),
                orthogonal_krylov_spread(arnoldi_vectors, state),
                biorthogonal_krylov_spread(bi_w, state),
            ]
        )
    with (RES_DIR / f"local_bosonic_reservoir_trace_N{n_sites}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "norm", "orthogonal_krylov_spread", "biorthogonal_krylov_spread"])
        writer.writerows(rows)


def write_outputs(rows: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    csv_path = RES_DIR / "local_bosonic_reservoir_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_trace(n_sites=80)
    trace = np.loadtxt(
        RES_DIR / "local_bosonic_reservoir_trace_N80.csv",
        delimiter=",",
        skiprows=1,
    )

    sizes = np.array([row["N"] for row in rows])
    inv_gap = np.array([row["inverse_gap"] for row in rows])
    settle = np.array([row["settling_time_einv"] for row in rows])
    log_smin = np.array([row["log_midgap_smin"] for row in rows])
    orth_peak = np.array([row["orthogonal_krylov_visible_peak"] for row in rows])
    bi_peak = np.array([row["biorthogonal_krylov_visible_peak"] for row in rows])
    reconstruction = np.array([row["local_reconstruction_error"] for row in rows])
    min_gain = np.array([row["min_gain_eigenvalue"] for row in rows])
    min_loss = np.array([row["min_loss_eigenvalue"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))

    ax = axes[0, 0]
    ax.plot(sizes, inv_gap, "o-", label="1 / rapidity gap")
    ax.plot(sizes, settle, "s-", label="settling time")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("time")
    ax.set_title("Local Lindblad reservoir: gap misses settling")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(sizes, log_smin, "o-", color="tab:red")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("log sigma_min(-gap/2 - X)")
    ax.set_title("Pseudospectrum intrudes into the gap")

    ax = axes[1, 0]
    ax.plot(trace[:, 0], trace[:, 1], color="0.2", label="norm")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("norm")
    ax2 = ax.twinx()
    ax2.plot(trace[:, 0], trace[:, 2], color="tab:blue", label="Arnoldi spread")
    ax2.plot(trace[:, 0], trace[:, 3], color="tab:orange", label="two-sided spread")
    ax2.set_ylabel("Krylov spread")
    ax.set_title("Krylov front for N=80")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")

    ax = axes[1, 1]
    ax.semilogy(sizes, reconstruction, "o-", label="reconstruction error")
    ax.plot(sizes, np.maximum(min_gain, 1e-18), "s--", label="min eig Gamma_gain")
    ax.plot(sizes, min_loss, "^:", label="min eig Gamma_loss")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("CP/local check")
    ax.set_title("Exact local CP realization")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "local_bosonic_reservoir.png", dpi=220)
    plt.close(fig)

    p = PARAMS
    note = f"""# Local Bosonic Reservoir

This is the local-jump upgrade of the pseudospectral Krylov construction.

Target rapidity block:

```text
X_jj = -gamma
X_(j+1),j = t_right
X_j,(j+1) = t_left
```

with `t_right={p.t_right}`, `t_left={p.t_left}`, `gamma={p.gamma}`.

Local Lindblad realization:

```text
H_(j+1),j = i (t_right - t_left) / 2
L^g_j = sqrt(c) (a_j^dagger + a_(j+1)^dagger)
L^l_j = sqrt(c) (a_j - a_(j+1))
L^0_j = sqrt(2 gamma) a_j
c = (t_right + t_left) / 2
```

The construction gives

```text
X = -i H + (Gamma_gain - Gamma_loss) / 2
```

using only nearest-neighbor bond jumps and onsite loss.

Main numerical facts at `N={int(sizes[-1])}`:

- reconstruction error: `{reconstruction[-1]:.3e}`;
- min eigenvalue of `Gamma_gain`: `{min_gain[-1]:.3e}`;
- min eigenvalue of `Gamma_loss`: `{min_loss[-1]:.3e}`;
- `1/gap`: `{inv_gap[-1]:.6g}`;
- last `e^-1` settling time: `{settle[-1]:.6g}`;
- `log sigma_min(-gap/2 - X)`: `{log_smin[-1]:.6g}`;
- visible Arnoldi Krylov spread: `{orth_peak[-1]:.6g}`;
- visible two-sided Krylov spread: `{bi_peak[-1]:.6g}`.

Generated files:

- `figures/local_bosonic_reservoir.png`
- `results/local_bosonic_reservoir_scaling.csv`
- `results/local_bosonic_reservoir_trace_N80.csv`
"""
    (RES_DIR / "reports/local_bosonic_reservoir.txt").write_text(note, encoding="utf-8")


def main() -> None:
    rows = scan_scaling()
    write_outputs(rows)
    print("Wrote results/local_bosonic_reservoir_scaling.csv")
    print("Wrote figures/local_bosonic_reservoir.png")


if __name__ == "__main__":
    main()
