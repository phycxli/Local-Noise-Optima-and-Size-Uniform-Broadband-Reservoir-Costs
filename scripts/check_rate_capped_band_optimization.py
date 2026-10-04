"""Check saved finite-band reservoir bounds using exact rational arithmetic.

Every stored binary float is interpreted as its exact dyadic rational.  PSD
checks use saved Cholesky factors and rational residual row norms.  The dual
uses Gram factors, which are PSD by construction, and a stationarity penalty.
No optimizer or floating-point eigensolver is used by this checker.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import numpy as np

from run_rate_capped_band_optimization import nonlocal_pointwise_certificate

ROOT = Path(__file__).resolve().parents[1]


def rational_array(array: np.ndarray) -> np.ndarray:
    return np.array([F.from_float(float(value)) for value in array.flat], dtype=object).reshape(array.shape)


def exact_factor_product(array: np.ndarray) -> np.ndarray:
    ratios = [float(value).as_integer_ratio() for value in array.flat]
    denominator = max(value[1] for value in ratios)
    integers = np.array([numerator * (denominator // divisor) for numerator, divisor in ratios],
                        dtype=object).reshape(array.shape)
    product = integers @ integers.T
    divisor = denominator**2
    return np.array([F(int(value), divisor) for value in product.flat], dtype=object).reshape(product.shape)


def exact_gram(n: int, width: F) -> np.ndarray:
    c = np.full((n, n), F(0), dtype=object)
    c[0, 0] = F(1)
    c[1, 0], c[1, 1] = F(13, 10), width
    for j in range(1, n - 1):
        for ell in range(j + 2):
            value = F(6, 5) * c[j, ell] - c[j - 1, ell] / 4
            if ell:
                value += width * F(ell, 2 * ell - 1) * c[j, ell - 1]
            if ell + 1 < n:
                value -= width * F(ell + 1, 2 * ell + 3) * c[j, ell + 1]
            c[j + 1, ell] = value
    gram = np.full((n, n), F(0), dtype=object)
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = 5 * sum(
                (c[i, ell] * c[j, ell] / (2 * ell + 1) for ell in range(min(i, j) + 1)), F(0))
    return gram


def source_matrix(n: int) -> np.ndarray:
    source = np.full((n, n), F(0), dtype=object)
    for j in range(n):
        source[j, j] = F(-12, 5)
        if j + 1 < n:
            source[j, j + 1] = source[j + 1, j] = F(5, 4)
    return source


def row_norm(matrix: np.ndarray) -> F:
    return max(sum((abs(value) for value in row), F(0)) for row in matrix)


def trace_product(a: np.ndarray, b: np.ndarray) -> F:
    return sum((a[i, j] * b[j, i] for i in range(len(a)) for j in range(len(a))), F(0))


def endpoint_gain(n: int, width: F) -> F:
    real, imaginary = F(13, 10), width
    previous = (F(1), F(0))
    current = (real, imaginary)
    for j in range(1, n):
        diagonal = F(13, 10) if j == n - 1 else F(6, 5)
        following = (diagonal * current[0] - width * current[1] - previous[0] / 4,
                     diagonal * current[1] + width * current[0] - previous[1] / 4)
        previous, current = current, following
    return F(1, 25) / (current[0]**2 + current[1]**2)


def check_row(row: dict, gram: np.ndarray) -> dict:
    path = ROOT / row["matrix_file"]
    with np.load(path, allow_pickle=False) as saved:
        n = int(row["N"])
        cap = F(row["rate_cap"])
        source = source_matrix(n)
        loss = rational_array(saved["loss"])
        if loss.shape != (n, n) or not np.array_equal(loss, loss.T):
            raise ValueError("The saved loss matrix must be real symmetric with the declared size.")
        gain = loss + source
        eye = np.eye(n, dtype=object) * F(1)
        mask = saved["mask"]
        radius = None if row["radius"] == "all" else int(row["radius"])
        expected_mask = (np.ones((n, n), dtype=bool) if radius is None else
                         np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= radius)
        if not np.array_equal(mask, expected_mask):
            raise ValueError("The saved locality mask disagrees with the declared range.")
        if any(value for value in loss[~mask]):
            raise ValueError("A forbidden reservoir entry is nonzero.")
        targets = {"loss": loss, "gain": gain, "loss_slack": cap * eye - loss,
                   "gain_slack": cap * eye - gain}
        margins = {}
        for name, target in targets.items():
            factor_product = exact_factor_product(saved[f"primal_{name}_factor"])
            shift = F.from_float(float(saved[f"primal_{name}_shift"]))
            remainder = row_norm(target - shift * eye - factor_product)
            margins[name] = shift - remainder
            if margins[name] <= 0:
                raise ValueError(f"Exact PSD check failed: {path.name}/{name}")
        a, b, c, d = [exact_factor_product(saved[f"dual_factor_{index}"]) for index in range(4)]
        residual = (gram - a - b + c + d) * mask
        residual_norm = row_norm(residual)
        objective = trace_product(d - b, source / cap) - sum(c[i, i] + d[i, i] for i in range(n))
        exact_lower = trace_product(gram, source) + cap * (objective - n * residual_norm)
        exact_upper = trace_product(gram, gain)
        if F(row["lower"]) > exact_lower or F(row["upper"]) < exact_upper:
            raise ValueError(f"The reported bounds are insufficiently conservative: {path.name}")
        width = F(row["halfwidth"])
        min_gain = endpoint_gain(n, width)
        return {"N": n, "halfwidth": float(width), "rate_cap": float(cap), "radius": row["radius"],
                "exact_matrix_checks_passed": True, "minimum_psd_margin": float(min(margins.values())),
                "rational_lower": float(exact_lower), "rational_upper": float(exact_upper),
                "reported_lower": float(row["lower"]), "reported_upper": float(row["upper"]),
                "bound_gap": float(exact_upper - exact_lower), "min_band_gain": float(min_gain),
                "whole_band_high_gain": min_gain > F(11, 10),
                "matrix_file": row["matrix_file"], "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="rate_capped_band_optimization")
    args = parser.parse_args()
    csv_path = ROOT / "results" / f"{args.stem}.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        inputs = list(csv.DictReader(handle))
    rows, cache, pointwise = [], {}, []
    for row in inputs:
        key = (int(row["N"]), F(row["halfwidth"]))
        if key not in cache:
            cache[key] = exact_gram(*key)
            point_path = ROOT / "results" / f"{args.stem}_matrices" / f"N{key[0]}_B{float(key[1]):g}_pointwise.json"
            if point_path.exists():
                data = json.loads(point_path.read_text(encoding="utf-8"))
                regenerated = nonlocal_pointwise_certificate(key[0], float(key[1]))
                if data["polynomial_coefficients"] != regenerated["polynomial_coefficients"]:
                    raise ValueError("Pointwise polynomial does not match the model and projector construction.")
                coefficients = [F(value) for value in data["polynomial_coefficients"]]
                margin = coefficients[0] - sum(abs(value) for value in coefficients[1:])
                if margin != F(int(data["margin_numerator"]), int(data["margin_denominator"])):
                    raise ValueError("Pointwise rational certificate mismatch.")
                if margin <= 0 or data["alpha"] != 7:
                    raise ValueError("The frequency-adaptive pointwise family is not certified.")
                pointwise.append({"N": key[0], "halfwidth": float(key[1]), "margin": float(margin),
                                  "positive_polynomial": margin > 0, "rate_upper": 7.1})
        checked = check_row(row, cache[key])
        rows.append(checked)
        print(json.dumps(checked), flush=True)
    locality = []
    for row in rows:
        if row["radius"] != "1":
            continue
        reference = next(v for v in rows if v["N"] == row["N"] and v["halfwidth"] == row["halfwidth"]
                         and v["rate_cap"] == row["rate_cap"] and v["radius"] == "all")
        locality.append({"N": row["N"], "halfwidth": row["halfwidth"], "rate_cap": row["rate_cap"],
                         "lower": row["reported_lower"] - reference["reported_upper"],
                         "upper": row["reported_upper"] - reference["reported_lower"]})
    output = {"arithmetic": "Exact rational matrix products and residual row bounds; binary floats interpreted exactly.",
              "scope": "Specified finite scalar chains and symmetric bands, not an all-size or multiband theorem.",
              "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": rows, "pointwise_certificates": pointwise, "nearest_neighbor_locality_gaps": locality,
              "all_checks_passed": (all(v["exact_matrix_checks_passed"] for v in rows)
                                    and all(v["positive_polynomial"] for v in pointwise))}
    path = ROOT / "results" / f"{args.stem}_certificate.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
