"""Transfer-recursion construction of the exact crossing polynomial."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import sympy as sp

from audit_multiband_crossing_boundary import response_data

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"


def exact_transfer_data(n: int):
    """Build F_N and the boundary denominator using a 4x4 transfer recurrence."""
    w = sp.symbols("omega", real=True)
    gamma = (sp.Rational("1.2"), sp.Rational("1.15"))
    t_right = (sp.Rational("1.0"), sp.Rational("0.8"))
    t_left = (sp.Rational("0.25"), sp.Rational("0.35"))
    j_mix = sp.Rational("0.01")
    kappa = sp.Rational("0.2")
    q_domain = sp.QQ_I

    def p(expr=0):
        return sp.Poly(expr, w, domain=q_domain)

    zero, one = p(0), p(1)
    eye = [[one, zero], [zero, one]]
    zero2 = [[zero, zero], [zero, zero]]

    def add(a, b):
        return [[a[i][j] + b[i][j] for j in range(2)] for i in range(2)]

    def mm(a, b):
        return [[sum((a[i][k] * b[k][j] for k in range(2)), zero)
                 for j in range(2)] for i in range(2)]

    def diagonal(vals):
        return [[p(vals[0]), zero], [zero, p(vals[1])]]

    def onsite(cell: int):
        probe = kappa / 2 if cell in (0, n - 1) else 0
        return [
            [p(gamma[0] + probe - sp.I * w), p(-sp.I * j_mix)],
            [p(-sp.I * j_mix), p(gamma[1] - sp.I * w)],
        ]

    # A has upper block U=-T_R, lower block L=-T_L.  Thus
    # E_{j+1}=(-U^{-1}B_j)E_j+(-U^{-1}L)E_{j-1}.
    c = diagonal((1 / t_right[0], 1 / t_right[1]))
    d = diagonal((-t_left[0] / t_right[0], -t_left[1] / t_right[1]))
    e_prev, e_curr = zero2, eye
    transfer_blocks = [e_curr]
    for cell in range(n - 1):
        e_next = add(mm(mm(c, onsite(cell)), e_curr), mm(d, e_prev))
        transfer_blocks.append(e_next)
        e_prev, e_curr = e_curr, e_next

    lower = diagonal((-t_left[0], -t_left[1]))
    boundary = add(mm(lower, transfer_blocks[-2]), mm(onsite(n - 1), transfer_blocks[-1]))
    q11, q12 = boundary[0]
    q21, q22 = boundary[1]
    denominator = q11 * q22 - q12 * q21
    # For output port p=(1,0)^T, adj(Q_N)p=(Q_22,-Q_21)^T.
    r = [q22, -q21]
    profiles = []
    for e in transfer_blocks:
        profiles.append([
            e[0][0] * r[0] + e[0][1] * r[1],
            e[1][0] * r[0] + e[1][1] * r[1],
        ])

    f = zero
    h = (t_right[0] + t_left[0], t_right[1] + t_left[1])
    for cell, profile in enumerate(profiles):
        for orbital in range(2):
            f += sp.Poly(
                sp.conjugate(profile[orbital].as_expr())
                * (-2 * gamma[orbital]) * profile[orbital].as_expr(),
                w, domain=q_domain,
            )
            if cell + 1 < n:
                next_profile = profiles[cell + 1][orbital]
                cross = (
                    sp.conjugate(profile[orbital].as_expr())
                    * h[orbital] * next_profile.as_expr()
                    + sp.conjugate(next_profile.as_expr())
                    * h[orbital] * profile[orbital].as_expr()
                )
                f += sp.Poly(cross, w, domain=q_domain)

    f_expr = sp.expand(f.as_expr())
    poly = sp.Poly(f_expr, w, domain=sp.QQ)
    den_expr = sp.expand(denominator.as_expr())
    den_real = sp.Poly(sp.re(den_expr), w, domain=sp.QQ)
    den_imag = sp.Poly(sp.im(den_expr), w, domain=sp.QQ)
    common = den_real.gcd(den_imag)
    pole_count = int(common.count_roots(-2, 2))
    return w, poly, denominator, pole_count


def isolate(n: int, eps: sp.Rational) -> list[dict[str, object]]:
    started = perf_counter()
    print(f"N={n}: building transfer polynomial", flush=True)
    w, poly, denominator, pole_count = exact_transfer_data(n)
    print(f"N={n}: polynomial built in {perf_counter()-started:.2f}s; Sturm counting", flush=True)
    intervals = poly.intervals(eps=eps, inf=-2, sup=2)
    count = int(poly.count_roots(-2, 2))
    if sum(mult for _, mult in intervals) != count:
        raise ArithmeticError(f"Sturm count and isolating intervals disagree at N={n}")
    if pole_count:
        raise ArithmeticError(f"response denominator has a real pole in [-2,2] at N={n}")

    rows = []
    for (lo, hi), mult in intervals:
        root = float(sp.N((lo + hi) / 2, 30))
        data = response_data(n, root)
        rows.append({
            "N": n,
            "degree_F": poly.degree(),
            "real_root_count_in_scan": count,
            "real_pole_count_in_scan": pole_count,
            "multiplicity": mult,
            "omega_lo_rational": str(lo),
            "omega_hi_rational": str(hi),
            "omega_mid": root,
            "signature_at_mid": data["signature"],
            "Pi_direct_at_mid": data["Pi_direct"],
            "sigma_min_C_at_mid": data["sigma_min_C"],
            "denominator_degree": denominator.degree(),
        })
    if not rows:
        rows.append({
            "N": n,
            "degree_F": poly.degree(),
            "real_root_count_in_scan": count,
            "real_pole_count_in_scan": pole_count,
            "multiplicity": 0,
            "omega_lo_rational": "",
            "omega_hi_rational": "",
            "omega_mid": "",
            "signature_at_mid": "",
            "Pi_direct_at_mid": "",
            "sigma_min_C_at_mid": "",
            "denominator_degree": denominator.degree(),
        })
    print(
        f"N={n}: deg(F)={poly.degree()}, roots={count}, poles={pole_count}, "
        f"total={perf_counter()-started:.2f}s",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[24])
    parser.add_argument("--digits", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("multiband_transfer_sturm.csv"))
    args = parser.parse_args()
    eps = sp.Rational(1, 10**args.digits)
    rows = [row for n in args.sizes for row in isolate(n, eps)]
    out = args.output if args.output.is_absolute() else RES_DIR / args.output
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
