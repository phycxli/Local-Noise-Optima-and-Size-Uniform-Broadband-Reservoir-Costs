"""Exact Sturm isolation of finite-chain sign crossings for the two-band model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sympy as sp
from sympy.polys.matrices import DomainMatrix

from audit_multiband_crossing_boundary import response_data

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"


def exact_model(n: int):
    """Return A(omega) and M with exact rational model parameters."""
    w = sp.symbols("omega", real=True)
    gamma = (sp.Rational("1.2"), sp.Rational("1.15"))
    t_right = (sp.Rational("1.0"), sp.Rational("0.8"))
    t_left = (sp.Rational("0.25"), sp.Rational("0.35"))
    j_mix = sp.Rational("0.01")
    kappa = sp.Rational("0.2")
    dim = 2 * n

    # A=(i*omega I-X_kappa)^dagger. Its block hoppings are -T_R above
    # and -T_L below the diagonal; probe damping enters the two end blocks.
    a = sp.zeros(dim)
    m = sp.zeros(dim)
    for cell in range(n):
        for orbital in range(2):
            idx = 2 * cell + orbital
            probe = kappa / 2 if orbital == 0 and cell in (0, n - 1) else 0
            a[idx, idx] = gamma[orbital] + probe - sp.I * w
            m[idx, idx] = -2 * gamma[orbital]
        a[2 * cell, 2 * cell + 1] = -sp.I * j_mix
        a[2 * cell + 1, 2 * cell] = -sp.I * j_mix

        if cell + 1 < n:
            for orbital in range(2):
                left = 2 * cell + orbital
                right = left + 2
                a[left, right] = -t_right[orbital]
                a[right, left] = -t_left[orbital]
                m[left, right] = t_right[orbital] + t_left[orbital]
                m[right, left] = t_right[orbital] + t_left[orbital]

    return w, a, m


def numerator_polynomial(n: int):
    w, a, m = exact_model(n)
    out_index = 2 * n - 2
    rhs = sp.zeros(2 * n, 1)
    rhs[out_index, 0] = 1
    a_dm = DomainMatrix.from_Matrix(a).to_sparse()
    rhs_dm = DomainMatrix.from_Matrix(rhs).to_sparse()
    response_num, determinant = a_dm.solve_den(rhs_dm)
    determinant_expr = determinant.as_expr()
    determinant_real = sp.Poly(sp.re(determinant_expr), w, domain=sp.QQ)
    determinant_imag = sp.Poly(sp.im(determinant_expr), w, domain=sp.QQ)
    common_factor = determinant_real.gcd(determinant_imag)
    real_pole_count = int(common_factor.count_roots(-2, 2))
    if real_pole_count:
        raise ArithmeticError(f"A_N has a real-frequency pole in [-2,2] at N={n}")
    adj_col = response_num.to_Matrix()

    # The common |det A|^2 denominator is positive on the real axis in the
    # stable model. Hence zeros of s_N are exactly the real zeros of this F_N.
    f = sp.S.Zero
    for row in range(2 * n):
        for col in range(2 * n):
            if m[row, col] != 0:
                f += sp.conjugate(adj_col[row]) * m[row, col] * adj_col[col]
    poly = sp.Poly(sp.expand(f), w, domain=sp.QQ)
    if poly.is_zero:
        raise ArithmeticError(f"identically zero numerator at N={n}")
    return w, poly, determinant, real_pole_count


def isolate(n: int, eps: sp.Rational) -> list[dict[str, object]]:
    w, poly, determinant, real_pole_count = numerator_polynomial(n)
    intervals = poly.intervals(eps=eps, inf=-2, sup=2)
    count = int(poly.count_roots(-2, 2))
    if sum(multiplicity for _, multiplicity in intervals) != count:
        raise ArithmeticError(f"Sturm count and isolating intervals disagree at N={n}")

    rows: list[dict[str, object]] = []
    for (lo, hi), multiplicity in intervals:
        root = float(sp.N((lo + hi) / 2, 30))
        data = response_data(n, root)
        rows.append({
            "N": n,
            "degree_F": poly.degree(),
            "real_root_count_in_scan": count,
            "real_pole_count_in_scan": real_pole_count,
            "multiplicity": multiplicity,
            "omega_lo_rational": str(lo),
            "omega_hi_rational": str(hi),
            "omega_mid": root,
            "signature_at_mid": data["signature"],
            "Pi_direct_at_mid": data["Pi_direct"],
            "sigma_min_C_at_mid": data["sigma_min_C"],
            "det_degree": determinant.degree(),
        })
    if not rows:
        rows.append({
            "N": n,
            "degree_F": poly.degree(),
            "real_root_count_in_scan": 0,
            "real_pole_count_in_scan": real_pole_count,
            "multiplicity": 0,
            "omega_lo_rational": "",
            "omega_hi_rational": "",
            "omega_mid": "",
            "signature_at_mid": "",
            "Pi_direct_at_mid": "",
            "sigma_min_C_at_mid": "",
            "det_degree": determinant.degree(),
        })
    print(
        f"N={n}: deg(F)={poly.degree()}, Sturm roots in [-2,2]={count}, "
        f"common determinant roots={real_pole_count}"
    )
    for row in rows:
        if row["multiplicity"]:
            print(
                f"  [{row['omega_lo_rational']}, {row['omega_hi_rational']}], "
                f"mult={row['multiplicity']}, omega~{row['omega_mid']:.12g}, "
                f"s(mid)={row['signature_at_mid']:.3e}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[24])
    parser.add_argument("--digits", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("multiband_sturm_root_isolation.csv"))
    args = parser.parse_args()
    eps = sp.Rational(1, 10**args.digits)
    rows = [row for n in args.sizes for row in isolate(n, eps)]

    out = args.output if args.output.is_absolute() else RES_DIR / args.output
    fields = list(rows[0])
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
