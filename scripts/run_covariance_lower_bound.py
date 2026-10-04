"""Validate the analytic covariance/noise lower bound.

The bound keeps only the first bond-gain noise channel and measures the
right-edge occupation.  For the local reservoir,

    N_ss,NN >= [c Delta/(pi e sqrt(N))]
              [(t_R+gamma)/t_R]^2 |chi_N1(0)|^2,

where Delta = gamma - 2 sqrt(t_R t_L) and chi_N1(0) is the end-to-end
zero-frequency susceptibility.  In the anomaly window this proves

    N_ss,NN >= const * N^(-1/2) * (rho/lambda)^(2N).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

T_R = 1.0
T_L = 0.25
GAMMA = 1.2


def chebyshev_u(n: int, x: float) -> float:
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    previous = 1.0
    current = 2.0 * x
    for _ in range(2, n + 1):
        previous, current = current, 2.0 * x * current - previous
    return current


def endpoint_chi(n_sites: int) -> float:
    s = math.sqrt(T_R * T_L)
    rho = math.sqrt(T_R / T_L)
    x = GAMMA / (2.0 * s)
    return rho ** (n_sites - 1) / (s * chebyshev_u(n_sites, x))


def response_factor() -> float:
    s = math.sqrt(T_R * T_L)
    rho = math.sqrt(T_R / T_L)
    x = GAMMA / (2.0 * s)
    lam = x + math.sqrt(x * x - 1.0)
    return rho / lam


def covariance_lower_bound(n_sites: int) -> float:
    s = math.sqrt(T_R * T_L)
    delta = GAMMA - 2.0 * s
    c = 0.5 * (T_R + T_L)
    chi = endpoint_chi(n_sites)
    prefactor = c * delta / (math.pi * math.e * math.sqrt(n_sites))
    prefactor *= ((T_R + GAMMA) / T_R) ** 2
    return prefactor * chi * chi


def read_actual_covariance() -> dict[int, dict[str, float]]:
    path = RES_DIR / "local_observable_response_scaling.csv"
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_sites = int(float(row["N"]))
            out[n_sites] = {
                "right_edge": float(row["covariance_right_edge_ss"]),
                "total": float(row["covariance_total_ss"]),
                "t90_right": float(row["covariance_t90_right_edge"]),
            }
    return out


def scan() -> list[dict[str, float]]:
    actual = read_actual_covariance()
    sizes = [20, 30, 40, 60, 80, 100, 120, 140]
    rows: list[dict[str, float]] = []
    factor = response_factor()
    for n_sites in sizes:
        bound = covariance_lower_bound(n_sites)
        chi = endpoint_chi(n_sites)
        row = {
            "N": float(n_sites),
            "endpoint_chi": chi,
            "covariance_lower_bound": bound,
            "asymptotic_squared_factor": factor * factor,
            "log_lower_bound": math.log(bound),
            "right_edge_covariance_numeric": float("nan"),
            "total_covariance_numeric": float("nan"),
            "numeric_over_bound": float("nan"),
            "t90_right_edge_numeric": float("nan"),
        }
        if n_sites in actual:
            row["right_edge_covariance_numeric"] = actual[n_sites]["right_edge"]
            row["total_covariance_numeric"] = actual[n_sites]["total"]
            row["numeric_over_bound"] = actual[n_sites]["right_edge"] / bound
            row["t90_right_edge_numeric"] = actual[n_sites]["t90_right"]
        rows.append(row)
    return rows


def fit_slope(rows: list[dict[str, float]], key: str, max_n: int = 140) -> float:
    xs = np.array([row["N"] for row in rows if row["N"] <= max_n and row[key] > 0])
    ys = np.array([row[key] for row in rows if row["N"] <= max_n and row[key] > 0])
    return float(np.polyfit(xs, np.log(ys), deg=1)[0])


def write_outputs(rows: list[dict[str, float]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    csv_path = RES_DIR / "covariance_lower_bound.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sizes = np.array([row["N"] for row in rows])
    bound = np.array([row["covariance_lower_bound"] for row in rows])
    chi_sq = np.array([row["endpoint_chi"] ** 2 for row in rows])
    numeric = np.array([row["right_edge_covariance_numeric"] for row in rows])
    ratio = np.array([row["numeric_over_bound"] for row in rows])
    t90 = np.array([row["t90_right_edge_numeric"] for row in rows])
    finite_numeric = np.isfinite(numeric)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(sizes, bound, "o-", label="analytic lower bound")
    ax.semilogy(sizes, chi_sq / chi_sq[0] * bound[0], ":", label="|chi_N1(0)|^2 trend")
    if np.any(finite_numeric):
        ax.semilogy(sizes[finite_numeric], numeric[finite_numeric], "s--", label="numeric N_ss,NN")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("right-edge occupation")
    ax.set_title("Noise lower bound grows exponentially")
    ax.legend(frameon=False)

    ax = axes[1]
    if np.any(finite_numeric):
        ax.plot(sizes[finite_numeric], ratio[finite_numeric], "o-")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("numeric / lower bound")
    ax.set_title("Bound is conservative but nonzero")

    ax = axes[2]
    if np.any(finite_numeric):
        ax.plot(sizes[finite_numeric], t90[finite_numeric], "o-", label="90% right-edge noise")
    ax.plot(sizes, 1.0 / (GAMMA - 2.0 * math.sqrt(T_R * T_L)) * np.ones_like(sizes), "--", label="1 / Delta_infty")
    ax.set_xlabel("chain length N")
    ax.set_ylabel("time")
    ax.set_title("Observable buildup still outgrows gap time")
    ax.legend(frameon=False)

    fig.savefig(FIG_DIR / "covariance_lower_bound.png", dpi=220)
    plt.close(fig)

    bound_slope = fit_slope(rows, "covariance_lower_bound")
    predicted = 2.0 * math.log(response_factor())
    finite_rows = [row for row in rows if np.isfinite(row["right_edge_covariance_numeric"])]
    numeric_slope = fit_slope(finite_rows, "right_edge_covariance_numeric", max_n=100) if finite_rows else float("nan")

    note = f"""# Covariance Lower Bound

The right-edge steady occupation obeys the conservative analytic lower bound

```text
N_ss,NN >= [c Delta/(pi e sqrt(N))]
          [(t_R+gamma)/t_R]^2 |chi_N1(0)|^2.
```

Parameters:

- `t_R = {T_R}`;
- `t_L = {T_L}`;
- `gamma = {GAMMA}`;
- `Delta = gamma - 2 sqrt(t_R t_L) = {GAMMA - 2.0 * math.sqrt(T_R * T_L):.6g}`.

Growth slopes:

- fitted lower-bound slope: `{bound_slope:.6g}`;
- predicted asymptotic slope `2 log(rho/lambda)`: `{predicted:.6g}`;
- fitted numeric right-edge covariance slope through `N=100`: `{numeric_slope:.6g}`.

Generated files:

- `figures/covariance_lower_bound.png`
- `results/covariance_lower_bound.csv`
"""
    (RES_DIR / "reports/covariance_lower_bound.txt").write_text(note, encoding="utf-8")


def main() -> None:
    rows = scan()
    write_outputs(rows)
    print("Wrote results/covariance_lower_bound.csv")
    print("Wrote figures/covariance_lower_bound.png")


if __name__ == "__main__":
    main()
