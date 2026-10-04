"""Analytic bounds and formula checks for the local reservoir paper.

This script validates the formulas that should become the theory spine of the
Letter:

1. OBC rapidity gap:
       Delta_N = gamma - 2 s cos(pi/(N+1)),  s = sqrt(t_R t_L).
2. Similarity transform:
       X = D B D^{-1},  D_j = rho^j,  rho = sqrt(t_R/t_L).
3. End-to-end zero-frequency susceptibility:
       chi_N1(0) = rho^(N-1) / [s U_N(gamma/(2s))],
   where U_N is the Chebyshev polynomial of the second kind.
4. Exponential response window:
       2s < gamma < t_R + t_L.

The script cross-checks the exact formula against direct matrix inversion and
compares the analytic response growth with the observable covariance data.
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
)


FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)


def chebyshev_u(n: int, x: float) -> float:
    """Return U_n(x) by stable forward recurrence for moderate n."""

    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    previous = 1.0
    current = 2.0 * x
    for _ in range(2, n + 1):
        previous, current = current, 2.0 * x * current - previous
    return current


def exact_gap(n_sites: int, p: LocalBosonicHatanoParams = PARAMS) -> float:
    s = math.sqrt(p.t_right * p.t_left)
    return p.gamma - 2.0 * s * math.cos(math.pi / (n_sites + 1))


def endpoint_susceptibility_formula(
    n_sites: int,
    p: LocalBosonicHatanoParams = PARAMS,
) -> float:
    """Return |(-X)^-1_{N,1}| from the Chebyshev formula."""

    s = math.sqrt(p.t_right * p.t_left)
    rho = math.sqrt(p.t_right / p.t_left)
    x = p.gamma / (2.0 * s)
    return rho ** (n_sites - 1) / (s * chebyshev_u(n_sites, x))


def endpoint_susceptibility_numeric(
    n_sites: int,
    p: LocalBosonicHatanoParams = PARAMS,
) -> float:
    x = local_bosonic_hatano_drift(n_sites, p)
    chi = np.linalg.inv(-x)
    return float(abs(chi[-1, 0]))


def asymptotic_response_factor(p: LocalBosonicHatanoParams = PARAMS) -> float:
    s = math.sqrt(p.t_right * p.t_left)
    rho = math.sqrt(p.t_right / p.t_left)
    x = p.gamma / (2.0 * s)
    lam = x + math.sqrt(x * x - 1.0)
    return rho / lam


def read_observable_csv() -> dict[int, dict[str, float]]:
    path = RES_DIR / "local_observable_response_scaling.csv"
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_sites = int(float(row["N"]))
            out[n_sites] = {key: float(value) for key, value in row.items()}
    return out


def read_local_reservoir_csv() -> dict[int, dict[str, float]]:
    path = RES_DIR / "local_bosonic_reservoir_scaling.csv"
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_sites = int(float(row["N"]))
            out[n_sites] = {key: float(value) for key, value in row.items()}
    return out


def scan() -> list[dict[str, float]]:
    observable = read_observable_csv()
    local = read_local_reservoir_csv()
    sizes = [20, 30, 40, 60, 80, 100, 120]
    rows: list[dict[str, float]] = []
    factor = asymptotic_response_factor(PARAMS)
    for n_sites in sizes:
        formula = endpoint_susceptibility_formula(n_sites)
        numeric = endpoint_susceptibility_numeric(n_sites)
        row = {
            "N": float(n_sites),
            "gap_exact": exact_gap(n_sites),
            "inverse_gap": 1.0 / exact_gap(n_sites),
            "endpoint_chi_formula": formula,
            "endpoint_chi_numeric": numeric,
            "relative_chi_error": abs(formula - numeric) / max(abs(numeric), 1e-300),
            "asymptotic_factor": factor,
            "predicted_log_growth": (n_sites - 1) * math.log(factor),
            "similarity_condition_number": math.sqrt(PARAMS.t_right / PARAMS.t_left)
            ** (n_sites - 1),
        }
        if n_sites in observable:
            row.update(
                {
                    "covariance_t90_right_edge": observable[n_sites][
                        "covariance_t90_right_edge"
                    ],
                    "covariance_error_settling_einv": observable[n_sites][
                        "covariance_error_settling_einv"
                    ],
                    "susceptibility_norm_omega0": observable[n_sites][
                        "susceptibility_norm_omega0"
                    ],
                }
            )
        else:
            row.update(
                {
                    "covariance_t90_right_edge": float("nan"),
                    "covariance_error_settling_einv": float("nan"),
                    "susceptibility_norm_omega0": float("nan"),
                }
            )
        if n_sites in local:
            row["midgap_resolvent_norm"] = 1.0 / local[n_sites]["midgap_smin"]
        else:
            row["midgap_resolvent_norm"] = float("nan")
        rows.append(row)
    return rows


def fit_log_slope(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    coeffs = np.polyfit(xs, np.log(ys), deg=1)
    return float(coeffs[0]), float(coeffs[1])


def write_outputs(rows: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    csv_path = RES_DIR / "theory_bound_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sizes = np.array([row["N"] for row in rows])
    gaps = np.array([row["gap_exact"] for row in rows])
    inv_gaps = np.array([row["inverse_gap"] for row in rows])
    chi_formula = np.array([row["endpoint_chi_formula"] for row in rows])
    chi_numeric = np.array([row["endpoint_chi_numeric"] for row in rows])
    cond = np.array([row["similarity_condition_number"] for row in rows])
    midgap_resolvent = np.array([row["midgap_resolvent_norm"] for row in rows])
    cov_t90 = np.array([row["covariance_t90_right_edge"] for row in rows])
    cov_settle = np.array([row["covariance_error_settling_einv"] for row in rows])
    chi_norm = np.array([row["susceptibility_norm_omega0"] for row in rows])

    p = PARAMS
    s = math.sqrt(p.t_right * p.t_left)
    gammas = np.linspace(0.85 * 2.0 * s, 1.08 * (p.t_right + p.t_left), 300)
    delta_inf = gammas - 2.0 * s
    alpha_pbc = -gammas + p.t_right + p.t_left

    valid_chi = sizes <= 100
    slope, intercept = fit_log_slope(sizes[valid_chi], chi_formula[valid_chi])
    predicted_slope = math.log(asymptotic_response_factor(p))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))

    ax = axes[0, 0]
    ax.plot(gammas, delta_inf, label="OBC thermodynamic gap")
    ax.plot(gammas, alpha_pbc, label="PBC spectral abscissa")
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.axvspan(2.0 * s, p.t_right + p.t_left, color="tab:orange", alpha=0.18)
    ax.axvline(p.gamma, color="tab:red", ls="--", label="chosen gamma")
    ax.set_xlabel("gamma")
    ax.set_ylabel("rate")
    ax.set_title("Anomaly window: stable OBC, unstable bulk tendency")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.semilogy(sizes, chi_numeric, "o", label="direct inverse")
    ax.semilogy(sizes, chi_formula, "-", label="Chebyshev formula")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("|chi_N1(0)|")
    ax.set_title("Exact end-to-end susceptibility")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.semilogy(sizes, chi_formula, "o-", label="|chi_N1(0)|")
    ax.semilogy(sizes, cond / cond[0] * chi_formula[0], ":", label="D condition number trend")
    finite = np.isfinite(midgap_resolvent)
    ax.semilogy(
        sizes[finite],
        midgap_resolvent[finite] / midgap_resolvent[finite][0] * chi_formula[0],
        "s--",
        label="midgap resolvent trend",
    )
    ax.set_xlabel("chain length N")
    ax.set_ylabel("scaled response")
    ax.set_title("Non-normality controls response growth")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    finite_cov = np.isfinite(cov_t90)
    ax.plot(sizes[finite_cov], inv_gaps[finite_cov], "o-", label="1 / gap")
    ax.plot(sizes[finite_cov], cov_t90[finite_cov], "s-", label="90% right-edge covariance")
    ax.plot(sizes[finite_cov], cov_settle[finite_cov], "^-", label="covariance settling")
    ax2 = ax.twinx()
    ax2.semilogy(
        sizes[finite_cov],
        chi_norm[finite_cov],
        "d--",
        color="tab:purple",
        label="||chi(0)||",
    )
    ax.set_xlabel("chain length N")
    ax.set_ylabel("time")
    ax2.set_ylabel("response norm")
    ax.set_title("Observable clocks follow the pseudospectrum")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "theory_bound_validation.png", dpi=220)
    plt.close(fig)

    note = f"""# Theory Bound Validation

This file records the analytic checks that can anchor the PRL.

Parameters:

- `t_R = {p.t_right}`
- `t_L = {p.t_left}`
- `gamma = {p.gamma}`
- `s = sqrt(t_R t_L) = {s:.6g}`
- `rho = sqrt(t_R/t_L) = {math.sqrt(p.t_right / p.t_left):.6g}`

Anomaly window:

```text
2 sqrt(t_R t_L) < gamma < t_R + t_L
```

Numerically this is:

```text
{2.0 * s:.6g} < gamma < {p.t_right + p.t_left:.6g}
```

The chosen `gamma = {p.gamma}` lies inside the window.

Exact end-to-end susceptibility:

```text
chi_N1(0) = rho^(N-1) / [s U_N(gamma / 2s)]
```

where `U_N` is the Chebyshev polynomial of the second kind.

For the chosen parameters:

- fitted `log |chi_N1(0)| / N` slope up to `N=100`: `{slope:.6g}`;
- asymptotic prediction `log(rho/lambda)`: `{predicted_slope:.6g}`;
- maximum direct-vs-formula relative error: `{max(row['relative_chi_error'] for row in rows):.3e}`.

Generated files:

- `figures/theory_bound_validation.png`
- `results/theory_bound_validation.csv`
"""
    (RES_DIR / "reports/theory_bound_validation.txt").write_text(note, encoding="utf-8")


def main() -> None:
    rows = scan()
    write_outputs(rows)
    print("Wrote results/theory_bound_validation.csv")
    print("Wrote figures/theory_bound_validation.png")


if __name__ == "__main__":
    main()
