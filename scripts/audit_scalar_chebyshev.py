"""Exact scalar (J=0) Chebyshev-route audit for the active band."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]


def sign_variations(values: list[sp.Rational]) -> int:
    signs = [1 if value > 0 else -1 for value in values if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def scalar_polynomials(n: int):
    """Return the exact scalar quadratic-form polynomial G_N(x)."""
    omega = sp.symbols("omega", real=True)
    z = sp.Rational(6, 5) - sp.I * omega
    profiles = []
    previous = sp.Poly(0, omega, domain=sp.QQ_I)
    current = sp.Poly(1, omega, domain=sp.QQ_I)
    profiles.append(current)
    for cell in range(n - 1):
        onsite = z + (sp.Rational(1, 10) if cell == 0 else 0)
        next_profile = onsite * current - sp.Rational(1, 4) * previous
        profiles.append(next_profile)
        previous, current = current, next_profile

    quadratic = 0
    for profile in profiles:
        expression = profile.as_expr()
        quadratic += -sp.Rational(12, 5) * sp.conjugate(expression) * expression
    for left, right in zip(profiles, profiles[1:]):
        quadratic += sp.Rational(5, 2) * sp.re(
            sp.conjugate(left.as_expr()) * right.as_expr()
        )

    g_omega = sp.Poly(sp.expand(quadratic), omega, domain=sp.QQ)
    if any(g_omega.nth(k) for k in range(1, g_omega.degree() + 1, 2)):
        raise ArithmeticError(f"G_{n} is not even")
    x = sp.symbols("x", real=True)
    g_x = sp.Poly(
        list(reversed([
            g_omega.nth(2 * k) for k in range(g_omega.degree() // 2 + 1)
        ])),
        x,
        domain=sp.QQ,
    )
    return g_x, sp.Poly(-g_x + sp.Rational(1, 5), x, domain=sp.QQ)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=8)
    parser.add_argument("--stop", type=int, default=120)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "scalar_chebyshev_audit.csv")
    args = parser.parse_args()

    rows = []
    for n in range(args.start, args.stop + 1):
        g, r = scalar_polynomials(n)
        g_coefficients = [sp.Rational(g.nth(k)) for k in range(g.degree() + 1)]
        r_coefficients = [sp.Rational(r.nth(k)) for k in range(r.degree() + 1)]
        nonconstant_r = r_coefficients[1:]
        row = {
            "N": n,
            "degree_G": g.degree(),
            "G0_sign": 1 if g.nth(0) > 0 else -1 if g.nth(0) < 0 else 0,
            "G4_sign": 1 if g.eval(4) > 0 else -1 if g.eval(4) < 0 else 0,
            "G_sign_variations": sign_variations(g_coefficients),
            "R_nonconstant_all_positive": all(
                bool(value > 0) for value in r_coefficients[1:]
            ),
            "R_sign_variations": sign_variations(r_coefficients),
        }
        rows.append(row)
        if n in (args.start, 23, 24, 32, 40, args.stop):
            print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
