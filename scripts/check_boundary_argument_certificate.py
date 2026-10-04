"""Check saved certificates, exact box coverage, hashes and prefix metadata.

This is an artifact-integrity verifier. Interval operations are re-run by the
certification scripts; the inherited CRT polynomials are not reconstructed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from certify_boundary_evans_all_N import analytic_gate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ArithmeticError(f"Hash mismatch: {path}")


def check_sources(report):
    for name, expected in report["source_sha256"].items():
        check_hash(ROOT / name, expected)


def check_interval_cover(intervals, lo, hi):
    current = Decimal(lo)
    for a, b in sorted(intervals):
        if a != current or b <= a:
            raise ArithmeticError(f"Frequency cover gap/overlap at {a}, current={current}")
        current = b
    if current != Decimal(hi):
        raise ArithmeticError("Frequency cover has the wrong endpoint")


def check_rectangle_cover(rows):
    boxes = [tuple(Decimal(row[key]) for key in ("re_lo", "re_hi", "im_lo", "im_hi"))
             for row in rows]
    endpoints = sorted({x for box in boxes for x in box[:2]})
    if endpoints[0] != Decimal("0.149") or endpoints[-1] != Decimal("0.251"):
        raise ArithmeticError("Wrong complex rectangle")
    for a, b in zip(endpoints, endpoints[1:]):
        middle = (a + b) / 2
        vertical = [(c, d) for x, y, c, d in boxes if x < middle < y]
        check_interval_cover(vertical, "-0.001", "0.001")


def exact_height(n):
    a, b = Fraction(5, 4) * Fraction(231, 100), Fraction(7, 16)
    e = [Fraction(1), a]
    for _ in range(2, n):
        e.append(a * e[-1] + b * e[-2])
    q = Fraction(231, 100) * e[-1] + Fraction(7, 20) * e[-2]
    bound = Fraction(49, 10) * (2 * q * sum(e)) ** 2 * 400 ** (12 * n - 7)
    return (bound.numerator + bound.denominator - 1) // bound.denominator


def check_prefix():
    paths = [RESULTS / "H_sign_variation_N24_80.csv", RESULTS / "H_sign_variation_N81_120.csv"]
    rows = [row for path in paths for row in read_csv(path)]
    if sorted(int(row["N"]) for row in rows) != list(range(24, 121)):
        raise ArithmeticError("Inherited finite-size certificate has a missing/duplicate N")
    for row in rows:
        n = int(row["N"])
        if not (int(row["degree_H"]) == 2 * n - 1
                and int(row["sign_variations"]) == 1
                and int(row["endpoint_sign_0"]) == 1
                and int(row["endpoint_sign_4"]) == -1
                and int(row["endpoint_product_sign"]) == -1):
            raise ArithmeticError(f"Incorrect inherited Descartes data at N={n}")
        if int(row["crt_modulus_bits"]) < (2 * exact_height(n)).bit_length() + 1:
            raise ArithmeticError(f"CRT modulus not above the exact rational height bound at N={n}")
    return {"count": len(rows), "range": [24, 120],
            "coverage_and_exact_rational_height_checked": True,
            "polynomials_reconstructed_this_run": False,
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def main():
    report = json.loads((RESULTS / "boundary_evans_all_N_certificate.json").read_text(encoding="utf-8"))
    check_sources(report)
    if sorted(report["completed_parts"]) != ["complex", "real"]:
        raise ArithmeticError("Incomplete all-N certificate")
    real_path = RESULTS / "boundary_evans_interval_real_boxes.csv"
    complex_path = RESULTS / "boundary_evans_interval_complex_boxes.csv"
    check_hash(real_path, report["real_certificate_sha256"])
    check_hash(complex_path, report["complex_certificate_sha256"])
    real, complex_rows = read_csv(real_path), read_csv(complex_path)
    for name, lo, hi in (("low", "0", "0.15"), ("middle", "0.15", "0.25"),
                         ("high", "0.25", "2")):
        selected = [row for row in real if row["region"] == name]
        check_interval_cover([(Decimal(row["re_lo"]), Decimal(row["re_hi"])) for row in selected], lo, hi)
    check_rectangle_cover(complex_rows)
    gate = analytic_gate()
    if gate != report["analytic_gate"]:
        raise ArithmeticError("Saved analytic gate does not match the current calculation")
    response = json.loads((RESULTS / "boundary_response_prefactor_certificate.json").read_text(encoding="utf-8"))
    check_sources(response)
    response_path = RESULTS / "boundary_response_prefactor_boxes.csv"
    check_hash(response_path, response["certificate_sha256"])
    response_rows = read_csv(response_path)
    check_interval_cover([(Decimal(row["re_lo"]), Decimal(row["re_hi"])) for row in response_rows], "0", "2")
    prefix = check_prefix()
    result = {"integrity_check_passed": True, "real_box_count": len(real),
              "complex_box_count": len(complex_rows), "response_box_count": len(response_rows),
              "inherited_exact_prefix": prefix, "fresh_all_N_tail": [121, "infinity"],
              "fresh_response_prefactor_scope": [24, "infinity"],
              "analytic_gate": gate,
              "proof_level": "analytic derivation plus outward-rounded interval certificates; not a proof-assistant formalisation"}
    path = RESULTS / "boundary_argument_certificate_integrity.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
