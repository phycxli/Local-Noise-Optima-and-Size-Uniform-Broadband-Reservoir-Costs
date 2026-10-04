"""Modular reconstruction and exact real-root audit for the multiband chain.

The transfer polynomial is generated over several finite fields, scaled by a
known common denominator, and reconstructed with CRT.  The reconstructed
integer polynomial is then passed to SymPy's exact Sturm/subresultant root
counter.  Independent unused primes are checked after reconstruction.

The modular stage is the acceleration.  The final root count uses an exact
even-polynomial Descartes subdivision; the explicit coefficient-height bound
and independent-prime checks make the reconstructed integer polynomial exact.
Representative sizes are also cross-checked against the standard Sturm count.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import sympy as sp
from sympy.ntheory import nextprime

from modular_transfer_audit import (
    batch_modular_polynomials,
    exact_modular_polynomial,
)


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
BASE = 400


def sign_variations(coefficients: list[int]) -> int:
    signs = [1 if value > 0 else -1 for value in coefficients if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def integer_convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, value in enumerate(left):
        for j, other in enumerate(right):
            result[i + j] += value * other
    return result


def positive_mobius_transform(
    coefficients: list[int], a_num: int, b_num: int, denominator: int
) -> list[int]:
    """Map x=(a+b*t)/(1+t) and return (1+t)^d H(x) over Z."""
    degree = len(coefficients) - 1
    result = [0] * (degree + 1)
    for k, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        affine = [
            math.comb(k, j) * a_num ** (k - j) * b_num ** j
            for j in range(k + 1)
        ]
        one_plus_t = [
            math.comb(degree - k, j) * denominator ** (degree - k)
            for j in range(degree - k + 1)
        ]
        term = integer_convolution(affine, one_plus_t)
        for j, value in enumerate(term):
            result[j] += coefficient * value
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def rational_endpoint_value(
    coefficients: list[int], numerator: int, denominator: int
) -> int:
    degree = len(coefficients) - 1
    return sum(
        coefficient * numerator ** k * denominator ** (degree - k)
        for k, coefficient in enumerate(coefficients)
    )


def descartes_positive_root_count(
    polynomial: sp.Poly, max_depth: int = 80
) -> int:
    """Count distinct roots in (0, 4) with exact Descartes subdivision."""
    squarefree = polynomial.sqf_part()
    coefficients = [int(value) for value in reversed(squarefree.all_coeffs())]
    if rational_endpoint_value(coefficients, 0, 1) == 0:
        raise ArithmeticError("Descartes interval has a root at x=0")
    if rational_endpoint_value(coefficients, 4, 1) == 0:
        raise ArithmeticError("Descartes interval has a root at x=4")

    count = 0
    pending = [(0, 4, 1, 0)]
    while pending:
        a_num, b_num, denominator, depth = pending.pop()
        transformed = positive_mobius_transform(
            coefficients, a_num, b_num, denominator
        )
        variations = sign_variations(transformed)
        if variations == 0:
            continue
        if variations == 1:
            count += 1
            continue
        if depth >= max_depth:
            raise ArithmeticError("Descartes subdivision exceeded max_depth")
        midpoint = a_num + b_num
        pending.append((midpoint, 2 * b_num, 2 * denominator, depth + 1))
        pending.append((2 * a_num, midpoint, 2 * denominator, depth + 1))
    return count


def norm_bounds(n: int) -> tuple[float, float]:
    """Return safe l1 coefficient-norm bounds for F_N and det(Q_N)."""
    a = 1.25 * 2.31
    b = 0.4375
    e = [1.0]
    if n > 1:
        e.append(a)
        for _ in range(2, n):
            e.append(a * e[-1] + b * e[-2])
    q = 2.31 * e[-1] + (0.35 * e[-2] if n > 1 else 0.0)
    profile_sum = 2.0 * q * sum(e)
    f_bound = 4.9 * profile_sum * profile_sum
    denominator_bound = 2.0 * q * q
    return f_bound, denominator_bound


def height_bits(n: int, exponent: int, norm: float) -> int:
    if norm <= 0:
        return 8
    return math.ceil(exponent * math.log2(BASE) + math.log2(norm)) + 2


def crt_update(values: list[int], residues: list[int], modulus: int, prime: int):
    inv = pow(modulus % prime, prime - 2, prime)
    for j, residue in enumerate(residues):
        t = ((residue - values[j]) % prime) * inv % prime
        values[j] += modulus * t


def centered(values: list[int], modulus: int):
    half = modulus // 2
    return [value if value <= half else value - modulus for value in values]


def reconstruct(n: int, prime_start: int = 1_000_003):
    f_bound, denominator_bound = norm_bounds(n)
    f_bits = height_bits(n, 12 * n - 7, f_bound)
    d_bits = height_bits(n, 4 * n - 2, denominator_bound)
    target_bits = max(f_bits, d_bits) + 1

    f_values = None
    dr_values = None
    di_values = None
    degrees = None
    modulus = 1
    prime = prime_start
    used_primes: list[int] = []
    skipped_primes: list[int] = []

    while modulus.bit_length() <= target_bits:
        prime = int(nextprime(prime))
        f_mod, den_mod = exact_modular_polynomial(n, prime)
        candidate_degrees = (
            len(f_mod) - 1,
            len(den_mod) - 1,
        )
        if degrees is None:
            degrees = candidate_degrees
            f_values = [0] * len(f_mod)
            dr_values = [0] * len(den_mod)
            di_values = [0] * len(den_mod)
        if candidate_degrees != degrees:
            skipped_primes.append(prime)
            continue

        f_scale = pow(BASE, 12 * n - 7, prime)
        d_scale = pow(BASE, 4 * n - 2, prime)
        f_residues = [(value * f_scale) % prime for value in f_mod]
        dr_residues = [(value[0] * d_scale) % prime for value in den_mod]
        di_residues = [(value[1] * d_scale) % prime for value in den_mod]
        crt_update(f_values, f_residues, modulus, prime)
        crt_update(dr_values, dr_residues, modulus, prime)
        crt_update(di_values, di_residues, modulus, prime)
        modulus *= prime
        used_primes.append(prime)
        prime += 1

    f_int = centered(f_values, modulus)
    dr_int = centered(dr_values, modulus)
    di_int = centered(di_values, modulus)
    if max(map(abs, f_int), default=0).bit_length() > f_bits:
        raise ArithmeticError("F_N exceeded the explicit coefficient-height bound")
    if max(map(abs, dr_int), default=0).bit_length() > d_bits:
        raise ArithmeticError("det(Q_N) exceeded the explicit height bound")

    # An independent prime is not used in the CRT reconstruction.
    verification_prime = int(nextprime(prime))
    f_check, den_check = exact_modular_polynomial(n, verification_prime)
    f_scale = pow(BASE, 12 * n - 7, verification_prime)
    d_scale = pow(BASE, 4 * n - 2, verification_prime)
    if any(
        value % verification_prime !=
        (residue * f_scale) % verification_prime
        for value, residue in zip(f_int, f_check)
    ):
        raise ArithmeticError("independent-prime check failed for F_N")
    if any(
        value % verification_prime !=
        (residue[0] * d_scale) % verification_prime
        for value, residue in zip(dr_int, den_check)
    ):
        raise ArithmeticError("independent-prime check failed for Re det(Q_N)")
    if any(
        value % verification_prime !=
        (residue[1] * d_scale) % verification_prime
        for value, residue in zip(di_int, den_check)
    ):
        raise ArithmeticError("independent-prime check failed for Im det(Q_N)")

    return {
        "F": f_int,
        "den_real": dr_int,
        "den_imag": di_int,
        "modulus": modulus,
        "used_primes": used_primes,
        "skipped_primes": skipped_primes,
        "verification_prime": verification_prime,
        "f_bound_bits": f_bits,
        "d_bound_bits": d_bits,
        "degree_F": degrees[0],
        "degree_denominator": degrees[1],
    }


def reconstruct_many(sizes: list[int], prime_start: int = 1_000_003):
    """CRT-reconstruct several sizes while sharing each finite-field prefix."""
    sizes = sorted(set(sizes))
    states = {}
    for n in sizes:
        f_bound, denominator_bound = norm_bounds(n)
        states[n] = {
            "f_bits": height_bits(n, 12 * n - 7, f_bound),
            "d_bits": height_bits(n, 4 * n - 2, denominator_bound),
            "target_bits": max(
                height_bits(n, 12 * n - 7, f_bound),
                height_bits(n, 4 * n - 2, denominator_bound),
            ) + 1,
            "modulus": 1,
            "degrees": None,
            "F": None,
            "den_real": None,
            "den_imag": None,
            "used_primes": [],
            "skipped_primes": [],
        }

    active = set(sizes)
    prime = prime_start
    while active:
        prime = int(nextprime(prime))
        batch = batch_modular_polynomials(sorted(active), prime)
        for n in list(active):
            state = states[n]
            f_mod, den_mod = batch[n]
            candidate_degrees = (len(f_mod) - 1, len(den_mod) - 1)
            if state["degrees"] is None:
                state["degrees"] = candidate_degrees
                state["F"] = [0] * len(f_mod)
                state["den_real"] = [0] * len(den_mod)
                state["den_imag"] = [0] * len(den_mod)
            if candidate_degrees != state["degrees"]:
                state["skipped_primes"].append(prime)
                continue
            f_scale = pow(BASE, 12 * n - 7, prime)
            d_scale = pow(BASE, 4 * n - 2, prime)
            crt_update(
                state["F"],
                [(value * f_scale) % prime for value in f_mod],
                state["modulus"],
                prime,
            )
            crt_update(
                state["den_real"],
                [(value[0] * d_scale) % prime for value in den_mod],
                state["modulus"],
                prime,
            )
            crt_update(
                state["den_imag"],
                [(value[1] * d_scale) % prime for value in den_mod],
                state["modulus"],
                prime,
            )
            state["modulus"] *= prime
            state["used_primes"].append(prime)
            if state["modulus"].bit_length() > state["target_bits"]:
                active.remove(n)
        prime += 1

    verification_prime = int(nextprime(prime))
    verification = batch_modular_polynomials(sizes, verification_prime)
    output = {}
    for n in sizes:
        state = states[n]
        modulus = state["modulus"]
        f_int = centered(state["F"], modulus)
        dr_int = centered(state["den_real"], modulus)
        di_int = centered(state["den_imag"], modulus)
        if max(map(abs, f_int), default=0).bit_length() > state["f_bits"]:
            raise ArithmeticError(f"F_{n} exceeded the coefficient-height bound")
        if max(map(abs, dr_int), default=0).bit_length() > state["d_bits"]:
            raise ArithmeticError(
                f"det(Q_{n}) exceeded the coefficient-height bound"
            )
        f_check, den_check = verification[n]
        f_scale = pow(BASE, 12 * n - 7, verification_prime)
        d_scale = pow(BASE, 4 * n - 2, verification_prime)
        if any(
            value % verification_prime != (residue * f_scale) % verification_prime
            for value, residue in zip(f_int, f_check)
        ):
            raise ArithmeticError(f"independent-prime check failed for F_{n}")
        if any(
            value % verification_prime !=
            (residue[0] * d_scale) % verification_prime
            for value, residue in zip(dr_int, den_check)
        ):
            raise ArithmeticError(f"independent-prime check failed for Re det(Q_{n})")
        if any(
            value % verification_prime !=
            (residue[1] * d_scale) % verification_prime
            for value, residue in zip(di_int, den_check)
        ):
            raise ArithmeticError(f"independent-prime check failed for Im det(Q_{n})")
        output[n] = {
            "F": f_int,
            "den_real": dr_int,
            "den_imag": di_int,
            "modulus": modulus,
            "used_primes": state["used_primes"],
            "skipped_primes": state["skipped_primes"],
            "verification_prime": verification_prime,
            "f_bound_bits": state["f_bits"],
            "d_bound_bits": state["d_bits"],
            "degree_F": state["degrees"][0],
            "degree_denominator": state["degrees"][1],
        }
    return output


def exact_root_audit(n: int, isolate: bool = False,
                     prime_start: int = 1_000_003, data=None):
    if data is None:
        data = reconstruct(n, prime_start=prime_start)
    w = sp.symbols("omega", real=True)
    f_poly = sp.Poly(list(reversed(data["F"])), w, domain=sp.ZZ)
    # F_N is exactly even in omega for this real parameter set.  Counting the
    # roots of H_N(x)=F_N(sqrt(x)) on x in (0,4) halves the Sturm degree.
    odd_coefficients = data["F"][1::2]
    if any(value != 0 for value in odd_coefficients):
        raise ArithmeticError("F_N is not even; the omega^2 reduction is invalid")
    x = sp.symbols("x", real=True)
    h_poly = sp.Poly(
        list(reversed(data["F"][::2])), x, domain=sp.ZZ
    )
    _, h_primitive = h_poly.primitive()
    if h_primitive.eval(0) == 0 or h_primitive.eval(4) == 0:
        raise ArithmeticError("an endpoint root needs separate interval handling")
    positive_x_count = descartes_positive_root_count(h_primitive)
    root_count = 2 * positive_x_count
    _, primitive = f_poly.primitive()
    intervals = primitive.intervals(
        eps=sp.Rational(1, 10**10), inf=-2, sup=2
    ) if isolate else []

    dr_poly = sp.Poly(list(reversed(data["den_real"])), w, domain=sp.ZZ)
    di_poly = sp.Poly(list(reversed(data["den_imag"])), w, domain=sp.ZZ)
    denominator_gcd = dr_poly.gcd(di_poly)
    pole_count = (
        0 if denominator_gcd.degree() == 0
        else int(denominator_gcd.count_roots(-2, 2))
    )

    rows: list[dict[str, object]] = []
    for interval, multiplicity in intervals:
        midpoint = sp.N((interval[0] + interval[1]) / 2, 18)
        rows.append({
            "N": n,
            "degree_F": data["degree_F"],
            "root_count": root_count,
            "pole_count": pole_count,
            "multiplicity": multiplicity,
            "interval_lo": str(interval[0]),
            "interval_hi": str(interval[1]),
            "midpoint": str(midpoint),
            "crt_modulus_bits": data["modulus"].bit_length(),
            "prime_count": len(data["used_primes"]),
            "verification_prime": data["verification_prime"],
            "root_method": "even-reduced-exact-descartes",
        })
    if not rows:
        rows.append({
            "N": n,
            "degree_F": data["degree_F"],
            "root_count": root_count,
            "pole_count": pole_count,
            "multiplicity": 0,
            "interval_lo": "",
            "interval_hi": "",
            "midpoint": "",
            "crt_modulus_bits": data["modulus"].bit_length(),
            "prime_count": len(data["used_primes"]),
            "verification_prime": data["verification_prime"],
            "root_method": "even-reduced-exact-descartes",
        })
    return rows, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[24])
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--prime-start", type=int, default=1_000_003)
    parser.add_argument("--output", type=Path, default=Path(
        "modular_sturm_habicht_audit.csv"
    ))
    args = parser.parse_args()

    rows = []
    if len(args.sizes) > 1:
        print("shared-prefix modular reconstruction", flush=True)
        data_map = reconstruct_many(args.sizes, prime_start=args.prime_start)
        for n in args.sizes:
            result, data = exact_root_audit(
                n, isolate=args.isolate, data=data_map[n]
            )
            rows.extend(result)
            print(
                f"N={n}: roots={result[0]['root_count']}, "
                f"poles={result[0]['pole_count']}, degree={data['degree_F']}, "
                f"primes={len(data['used_primes'])}, "
                f"CRT bits={data['modulus'].bit_length()}",
                flush=True,
            )
    else:
        for n in args.sizes:
            print(f"N={n}: modular reconstruction", flush=True)
            result, data = exact_root_audit(
                n, isolate=args.isolate, prime_start=args.prime_start
            )
            rows.extend(result)
            print(
                f"N={n}: roots={result[0]['root_count']}, "
                f"poles={result[0]['pole_count']}, degree={data['degree_F']}, "
                f"primes={len(data['used_primes'])}, "
                f"CRT bits={data['modulus'].bit_length()}",
                flush=True,
            )

    output = args.output if args.output.is_absolute() else RES_DIR / args.output
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
