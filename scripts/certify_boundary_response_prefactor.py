"""Certify nonzero input-output boundary factors uniformly for all N>=24.

The exact Green function is chi_oi^sharp=(a_N/d_N)*lambda_a^(-(N-2)).
This audits a_N and d_N, not the distinct noise-signature numerator F_N.
"""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from boundary_evans_interval import (PAIRS, certified_modal_data, complex_box,
                                     float_down, float_up, interval, lower,
                                     remainder_bound, upper)
from certify_boundary_evans_all_N import Box, write_csv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
M0 = 22


def bounds_for(box):
    data = certified_modal_data(complex_box(box.re_lo, box.re_hi))
    passive, active = data.roots[3], data.roots[2]
    ratios_a = [root / passive for root in data.roots[:3]]
    ratios_d = [(data.roots[k] * data.roots[ell]) / (active * passive)
                for k, ell in PAIRS[:-1]]
    ea, _, qa = remainder_bound(data.a[:3], ratios_a, M0)
    ed, _, qd = remainder_bound(data.d[:-1], ratios_d, M0)
    amin = lower(abs(data.a[3].value)) - ea
    amax = upper(abs(data.a[3].value)) + ea
    dmin = lower(abs(data.d[-1].value)) - ed
    dmax = upper(abs(data.d[-1].value)) + ed
    if not amin > 0 or not dmin > 0:
        raise ArithmeticError("Boundary coefficient may vanish")
    ratio_low, ratio_high = amin / dmax, amax / dmin
    if not lower(ratio_low) > interval("0.15") or not upper(ratio_high) < interval("1.32"):
        raise ArithmeticError("Requested response bounds not certified")
    return {"a_modulus_lower": float_down(amin), "d_modulus_lower": float_down(dmin),
            "prefactor_lower": float_down(ratio_low), "prefactor_upper": float_up(ratio_high),
            "q_a_upper": float_up(qa), "q_d_upper": float_up(qd),
            "root_disk_margin_lower": float_down(data.disk_margin)}


def main():
    with (RESULTS / "boundary_evans_interval_real_boxes.csv").open(encoding="utf-8") as handle:
        initial = [Box(Decimal(row["re_lo"]), Decimal(row["re_hi"])) for row in csv.DictReader(handle)]
    pending, rows = list(reversed(initial)), []
    start, last = perf_counter(), perf_counter()
    while pending:
        box = pending.pop()
        try:
            bounds = bounds_for(box)
        except ArithmeticError:
            if box.depth >= 20:
                raise
            first, second = box.split()
            pending.extend((second, first))
            continue
        rows.append({"re_lo": str(box.re_lo), "re_hi": str(box.re_hi),
                     "depth": box.depth, **bounds})
        if perf_counter() - last > 10:
            print(f"response: accepted={len(rows)}, pending={len(pending)}, "
                  f"elapsed={perf_counter()-start:.1f}s", flush=True)
            last = perf_counter()
    out = RESULTS / "boundary_response_prefactor_boxes.csv"
    write_csv(out, rows)
    sources = (Path(__file__), Path(__file__).with_name("boundary_evans_interval.py"),
               Path(__file__).with_name("boundary_evans.py"))
    report = {"certified": True, "scope": "fixed J=0.01; all integer N>=24; real |omega|<=2",
              "formula": "chi_oi^sharp=(a_N/d_N)*lambda_a^(-(N-2))",
              "conservative_prefactor_bounds": [0.15, 1.32], "box_count": len(rows),
              "certified_global_lower": min(row["prefactor_lower"] for row in rows),
              "certified_global_upper": max(row["prefactor_upper"] for row in rows),
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources},
              "certificate_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (RESULTS / "boundary_response_prefactor_certificate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
