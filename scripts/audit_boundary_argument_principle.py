"""Count crossing-polynomial zeros in a complex-frequency strip.

The polynomial F_N(z) is the analytic continuation of the real-frequency
quadratic-form numerator after the coefficient conjugation has been carried
out.  This script computes the winding number on a rectangle and compares it
with the exact real-axis root count.  It is a finite-size argument-principle
audit, not an all-N theorem.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mpmath as mp
import sympy as sp

from transfer_sturm_root_isolation import exact_transfer_data


ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
mp.mp.dps = 80


def rectangle_points(left: mp.mpf, right: mp.mpf, height: mp.mpf, samples: int):
    points = []
    for j in range(samples):
        t = mp.mpf(j) / samples
        points.append(left + (right - left) * t - 1j * height)
    for j in range(samples):
        t = mp.mpf(j) / samples
        points.append(right + 1j * (-height + 2 * height * t))
    for j in range(samples):
        t = mp.mpf(j) / samples
        points.append(right - (right - left) * t + 1j * height)
    for j in range(samples):
        t = mp.mpf(j) / samples
        points.append(left + 1j * (height - 2 * height * t))
    return points


def mp_coefficients(poly: sp.Poly):
    coeffs = []
    for coeff in poly.all_coeffs():
        coeffs.append(mp.mpf(str(coeff.p)) / mp.mpf(str(coeff.q)))
    scale = max(abs(c) for c in coeffs)
    return [c / scale for c in coeffs]


def horner(coeffs, z):
    value = mp.mpc(0)
    for coeff in coeffs:
        value = value * z + coeff
    return value


def winding_number(values):
    total = mp.mpf(0)
    for first, second in zip(values, values[1:] + values[:1]):
        total += mp.arg(second / first)
    return int(mp.nint(total / (2 * mp.pi))), total / (2 * mp.pi)


def audit(n: int, height: float, samples: int) -> dict[str, object]:
    _, poly, _, pole_count = exact_transfer_data(n)
    coeffs = mp_coefficients(poly)
    left, right, h = mp.mpf(-2), mp.mpf(2), mp.mpf(str(height))
    points = rectangle_points(left, right, h, samples)
    values = [horner(coeffs, z) for z in points]
    winding, winding_float = winding_number(values)
    min_boundary = min(abs(v) for v in values)
    real_count = int(poly.count_roots(-2, 2))
    return {
        "N": n,
        "degree_F": poly.degree(),
        "height": height,
        "samples_per_side": samples,
        "winding_number": winding,
        "winding_float": float(winding_float),
        "real_root_count": real_count,
        "pole_count": pole_count,
        "min_normalized_boundary_abs": float(min_boundary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[24, 32, 40])
    parser.add_argument("--heights", nargs="+", type=float, default=[0.02, 0.05, 0.07])
    parser.add_argument("--samples", type=int, default=1200)
    args = parser.parse_args()

    rows = []
    for n in args.sizes:
        for height in args.heights:
            row = audit(n, height, args.samples)
            rows.append(row)
            print(row, flush=True)

    out = RES / "boundary_argument_principle_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = (
        "# Boundary argument-principle audit\n\n"
        "The contour is the positively oriented rectangle `[-2,2] x [-h,h]` in "
        "the complex frequency plane. The polynomial is normalized by its largest "
        "coefficient before evaluation.\n\n"
        + "\n".join(
            f"- N={row['N']}, h={row['height']}: winding={row['winding_number']}, "
            f"real roots={row['real_root_count']}, min boundary={row['min_normalized_boundary_abs']:.3e}"
            for row in rows
        )
        + "\n\nThe result is a finite-size contour certificate; a uniform-in-N theorem requires a uniform boundary lower bound.\n"
    )
    (RES / "reports/boundary_argument_principle_audit.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
