"""Finite-field transfer recursion for the multiband crossing polynomial.

The two-band working point has rational coefficients.  This module evaluates
the same continuant recurrence over a prime field, representing a complex
coefficient by a pair (real, imaginary) modulo p.  It is a fast modular
front-end for coefficient reconstruction and Sturm-Habicht audits; it is not
by itself a real-root certificate.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rat(num: int, den: int, p: int) -> int:
    return (num % p) * pow(den % p, p - 2, p) % p


def zpoly() -> list[tuple[int, int]]:
    return [(0, 0)]


def cpoly(a: int, p: int) -> list[tuple[int, int]]:
    return [(a % p, 0)]


def trim(a: list[tuple[int, int]]) -> list[tuple[int, int]]:
    while len(a) > 1 and a[-1] == (0, 0):
        a.pop()
    return a


def padd(a: list[tuple[int, int]], b: list[tuple[int, int]], p: int):
    out = [(0, 0)] * max(len(a), len(b))
    for j in range(len(out)):
        ar, ai = a[j] if j < len(a) else (0, 0)
        br, bi = b[j] if j < len(b) else (0, 0)
        out[j] = ((ar + br) % p, (ai + bi) % p)
    return trim(out)


def pneg(a: list[tuple[int, int]], p: int):
    return [((-ar) % p, (-ai) % p) for ar, ai in a]


def pmul(a: list[tuple[int, int]], b: list[tuple[int, int]], p: int):
    out = [(0, 0)] * (len(a) + len(b) - 1)
    for i, (ar, ai) in enumerate(a):
        for j, (br, bi) in enumerate(b):
            cr, ci = out[i + j]
            out[i + j] = (
                (cr + ar * br - ai * bi) % p,
                (ci + ar * bi + ai * br) % p,
            )
    return trim(out)


def pconj(a: list[tuple[int, int]], p: int):
    return [(ar, (-ai) % p) for ar, ai in a]


def mzero(p: int):
    return [[zpoly(), zpoly()], [zpoly(), zpoly()]]


def meye(p: int):
    return [[cpoly(1, p), zpoly()], [zpoly(), cpoly(1, p)]]


def madd(a, b, p: int):
    return [[padd(a[i][j], b[i][j], p) for j in range(2)] for i in range(2)]


def mmul(a, b, p: int):
    out = mzero(p)
    for i in range(2):
        for j in range(2):
            value = zpoly()
            for k in range(2):
                value = padd(value, pmul(a[i][k], b[k][j], p), p)
            out[i][j] = value
    return out


def mvec(a, v, p: int):
    return [
        padd(pmul(a[i][0], v[0], p), pmul(a[i][1], v[1], p), p)
        for i in range(2)
    ]


def mdagger(a, p: int):
    return [
        [pconj(a[j][i], p) for j in range(2)]
        for i in range(2)
    ]


def quadratic_form(vector, matrix, p: int):
    image = mvec(matrix, vector, p)
    value = zpoly()
    for j in range(2):
        value = padd(value, pmul(pconj(vector[j], p), image[j], p), p)
    return value


def onsite(cell: int, n: int, p: int):
    probe = rat(1, 10, p) if cell in (0, n - 1) else 0
    return [
        [
            padd(cpoly(rat(6, 5, p) + probe, p), [(0, -1 % p)], p),
            cpoly(0, p),
        ],
        [cpoly(0, p), cpoly(rat(23, 20, p), p)],
    ]


def add_imag_constant(poly, value: int, p: int):
    out = [x for x in poly]
    out[0] = (out[0][0], (out[0][1] + value) % p)
    return out


def onsite(cell: int, n: int, p: int):
    probe = rat(1, 10, p) if cell in (0, n - 1) else 0
    first = cpoly(rat(6, 5, p) + probe, p)
    if len(first) == 1:
        first.append((0, (-1) % p))
    else:
        first[1] = (0, (-1) % p)
    mix = cpoly(0, p)
    mix[0] = (0, (-rat(1, 100, p)) % p)
    second = cpoly(rat(23, 20, p), p)
    second.append((0, (-1) % p))
    return [[first, mix], [mix, second]]


def real_diag(a: int, b: int, p: int):
    return [[cpoly(a, p), zpoly()], [zpoly(), cpoly(b, p)]]


def exact_modular_polynomial(n: int, p: int):
    """Return the modular F_N and complex response denominator."""
    c = real_diag(rat(1, 1, p), rat(5, 4, p), p)
    d = real_diag(-rat(1, 4, p), -rat(7, 16, p), p)
    lower = real_diag(-rat(1, 1, p), -rat(7, 20, p), p)
    e_prev = mzero(p)
    e_curr = meye(p)
    blocks = [e_curr]
    for cell in range(n - 1):
        e_next = madd(mmul(mmul(c, onsite(cell, n, p), p), e_curr, p),
                       mmul(d, e_prev, p), p)
        blocks.append(e_next)
        e_prev, e_curr = e_curr, e_next

    boundary = madd(mmul(lower, blocks[-2], p),
                    mmul(onsite(n - 1, n, p), blocks[-1], p), p)
    q11, q12 = boundary[0]
    q21, q22 = boundary[1]
    denominator = padd(pmul(q11, q22, p), pneg(pmul(q12, q21, p), p), p)
    r = [q22, pneg(q21, p)]
    profiles = [mvec(block, r, p) for block in blocks]

    f = zpoly()
    gammas = [rat(6, 5, p), rat(23, 20, p)]
    h = [rat(5, 4, p), rat(23, 20, p)]
    for cell, profile in enumerate(profiles):
        for orbital in range(2):
            term = pmul(pconj(profile[orbital], p),
                        pmul(cpoly((-2 * gammas[orbital]) % p, p),
                             profile[orbital], p), p)
            f = padd(f, term, p)
            if cell + 1 < n:
                cross = pmul(
                    pconj(profile[orbital], p),
                    cpoly(h[orbital], p),
                    p,
                )
                cross = pmul(cross, profiles[cell + 1][orbital], p)
                reverse = pmul(
                    pconj(profiles[cell + 1][orbital], p),
                    cpoly(h[orbital], p),
                    p,
                )
                reverse = pmul(reverse, profile[orbital], p)
                f = padd(f, padd(cross, reverse, p), p)
    if any(imag for _, imag in f):
        raise ArithmeticError("F_N is not real modulo the selected prime")
    return [real % p for real, _ in f], denominator


def batch_modular_polynomials(sizes: list[int], p: int):
    """Build F_N and det(Q_N) for many N while sharing prefix continuants."""
    sizes = sorted(set(sizes))
    if not sizes:
        return {}
    maximum = max(sizes)
    c = real_diag(rat(1, 1, p), rat(5, 4, p), p)
    d = real_diag(-rat(1, 4, p), -rat(7, 16, p), p)
    lower = real_diag(-rat(1, 1, p), -rat(7, 20, p), p)
    onsite_m = real_diag(-rat(12, 5, p), -rat(23, 10, p), p)
    bond_m = real_diag(rat(5, 4, p), rat(23, 20, p), p)
    e_prev = mzero(p)
    e_curr = meye(p)
    blocks = [e_curr]
    accumulated = mmul(mmul(mdagger(e_curr, p), onsite_m, p), e_curr, p)
    output_block = onsite(0, maximum, p)
    result = {}

    for n in range(1, maximum + 1):
        if n >= 2:
            transfer_cell = n - 2
            next_block = madd(
                mmul(mmul(c, onsite(transfer_cell, maximum, p), p),
                     e_curr, p),
                mmul(d, e_prev, p),
                p,
            )
            accumulated = madd(
                accumulated,
                mmul(mmul(mdagger(next_block, p), onsite_m, p),
                     next_block, p),
                p,
            )
            accumulated = madd(
                accumulated,
                madd(
                    mmul(mmul(mdagger(e_curr, p), bond_m, p),
                         next_block, p),
                    mmul(mmul(mdagger(next_block, p), bond_m, p),
                         e_curr, p),
                    p,
                ),
                p,
            )
            blocks.append(next_block)
            e_prev, e_curr = e_curr, next_block

        if n not in sizes:
            continue
        boundary = madd(
            mmul(lower, blocks[n - 2], p),
            mmul(output_block, blocks[n - 1], p),
            p,
        ) if n > 1 else output_block
        q11, q12 = boundary[0]
        q21, q22 = boundary[1]
        denominator = padd(
            pmul(q11, q22, p),
            pneg(pmul(q12, q21, p), p),
            p,
        )
        r = [q22, pneg(q21, p)]
        f = quadratic_form(r, accumulated, p)
        if any(imag for _, imag in f):
            raise ArithmeticError("batched F_N is not real modulo the prime")
        result[n] = ([real % p for real, _ in f], denominator)
    return result


def compare_with_exact(n: int, p: int):
    """Cross-check every coefficient against the rational transfer code."""
    import sympy as sp
    from transfer_sturm_root_isolation import exact_transfer_data

    _, poly, _, _ = exact_transfer_data(n)
    expected = []
    for coeff in reversed(poly.all_coeffs()):
        num, den = sp.Rational(coeff).as_numer_denom()
        expected.append(int(num) * pow(int(den) % p, p - 2, p) % p)
    actual, _ = exact_modular_polynomial(n, p)
    if actual != expected:
        mismatch = next(
            j for j, (left, right) in enumerate(zip(actual, expected))
            if left != right
        )
        raise AssertionError(
            f"modular mismatch at coefficient {mismatch}: "
            f"{actual[mismatch]} != {expected[mismatch]}"
        )
    return len(actual) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--prime", type=int, default=1000003)
    parser.add_argument("--compare-exact", action="store_true")
    args = parser.parse_args()
    if args.prime in (2, 5):
        raise SystemExit("choose a prime not dividing the rational denominators")
    if args.compare_exact:
        degree = compare_with_exact(args.size, args.prime)
        print(f"N={args.size}: exact modular coefficient check passed, degree={degree}")
    else:
        coeffs, denominator = exact_modular_polynomial(args.size, args.prime)
        print(
            f"N={args.size}: modular degree={len(coeffs)-1}, "
            f"denominator degree={len(denominator)-1}, prime={args.prime}"
        )


if __name__ == "__main__":
    main()
