"""Audit coefficient sign variation and endpoint signs of H_N(x)=F_N(sqrt(x)).

This is a diagnostic only: it does not promote finite-size evidence to an
all-N theorem.  Exact coefficients come from the existing CRT reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modular_sturm_habicht_audit import reconstruct_many  # noqa: E402


def variations(coefficients: list[int]) -> int:
    signs = [1 if value > 0 else -1 for value in coefficients if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def transition_index(coefficients: list[int]) -> int:
    """Index of the unique sign transition in ascending-power order."""
    signs = [(idx, 1 if value > 0 else -1)
             for idx, value in enumerate(coefficients) if value]
    transitions = [idx for (idx, left), (_, right) in zip(signs, signs[1:])
                   if left != right]
    return transitions[0] if len(transitions) == 1 else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[24, 25, 32, 40, 64, 80])
    parser.add_argument("--prime-start", type=int, default=1_000_003)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "H_sign_variation_audit.csv")
    args = parser.parse_args()

    data_map = reconstruct_many(args.sizes, prime_start=args.prime_start)
    rows = []
    for n in args.sizes:
        data = data_map[n]
        odd = data["F"][1::2]
        if any(value != 0 for value in odd):
            raise ArithmeticError(f"F_{n} is not even")
        primitive = sp.Poly(
            list(reversed(data["F"][::2])), sp.Symbol("x"), domain=sp.ZZ
        ).primitive()[1]
        coefficients = [int(value) for value in reversed(primitive.all_coeffs())]
        h0 = int(primitive.eval(0))
        h4 = int(primitive.eval(4))
        rows.append({
            "N": n,
            "degree_H": primitive.degree(),
            "sign_variations": variations(coefficients),
            "transition_index_ascending_power": transition_index(coefficients),
            "endpoint_sign_0": 0 if h0 == 0 else (1 if h0 > 0 else -1),
            "endpoint_sign_4": 0 if h4 == 0 else (1 if h4 > 0 else -1),
            "endpoint_product_sign": 0 if h0 * h4 == 0 else (1 if h0 * h4 > 0 else -1),
            "crt_modulus_bits": data["modulus"].bit_length(),
            "prime_count": len(data["used_primes"]),
        })
        print(
            f"N={n}: degree={primitive.degree()}, "
            f"variations={rows[-1]['sign_variations']}, "
            f"transition={rows[-1]['transition_index_ascending_power']}, "
            f"sign(H(0))={rows[-1]['endpoint_sign_0']}, "
            f"sign(H(4))={rows[-1]['endpoint_sign_4']}, "
            f"product={rows[-1]['endpoint_product_sign']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
