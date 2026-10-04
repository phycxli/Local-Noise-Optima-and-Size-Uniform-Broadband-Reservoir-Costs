"""Locate the first sampled violation of q_J <= q_0 near the transfer bound."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from audit_finite_J_schur_bound import full_response


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SIZES = (24, 32, 40, 80)
J_VALUES = np.arange(0.030, 0.061, 0.001)


def maximize_delta(n: int, j_mix: float) -> tuple[float, float]:
    def negative_delta(omega: float) -> float:
        return -(
            full_response(n, omega, j_mix)[0]
            - full_response(n, omega, 0.0)[0]
        )

    # The coarse audit locates the only positive bump near 0.2-0.25. Also
    # check the endpoints so this remains a bounded search, not a grid claim.
    result = minimize_scalar(negative_delta, bounds=(0.05, 0.45), method="bounded", options={"xatol": 1e-10})
    candidates = [(float(result.x), -float(result.fun)), (0.0, -negative_delta(0.0)), (2.0, -negative_delta(2.0))]
    return max(candidates, key=lambda item: item[1])


def main() -> None:
    rows: list[dict[str, float]] = []
    for n in SIZES:
        for j_mix in J_VALUES:
            omega, delta = maximize_delta(n, float(j_mix))
            rows.append({"N": float(n), "J": float(j_mix), "max_delta": delta, "omega_at_max": omega})
            print(rows[-1], flush=True)

    out = RES / "finite_J_sign_threshold.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for n in SIZES:
        subset = [row for row in rows if row["N"] == n and row["max_delta"] > 0.0]
        if subset:
            first = min(subset, key=lambda row: row["J"])
            print(f"N={n}: first sampled positive J={first['J']:.3f}, delta={first['max_delta']:.6g}, omega={first['omega_at_max']:.6g}")
        else:
            print(f"N={n}: no positive delta on scanned J interval")


if __name__ == "__main__":
    main()
