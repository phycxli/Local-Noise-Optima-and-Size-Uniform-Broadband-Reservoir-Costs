"""Computer-assisted, all-N boundary crossing and local Rouche certificate.

Closed frequency boxes cover whole intervals, not sampled points. Every
accepted box has four disjoint Taylor/Rouche root disks and outward-rounded
modal bounds. A fixed m>=119 remainder estimate covers all N>=121 at once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from time import perf_counter

from boundary_evans_interval import (complex_band_bounds, complex_box, float_down,
                                     float_up, interval, iv, lower,
                                     real_band_bounds, upper)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
M0 = 119
getcontext().prec = 60


@dataclass(frozen=True)
class Box:
    re_lo: Decimal
    re_hi: Decimal
    im_lo: Decimal = Decimal(0)
    im_hi: Decimal = Decimal(0)
    depth: int = 0

    def split(self):
        if self.re_hi - self.re_lo >= self.im_hi - self.im_lo:
            mid = (self.re_lo + self.re_hi) / 2
            return (Box(self.re_lo, mid, self.im_lo, self.im_hi, self.depth + 1),
                    Box(mid, self.re_hi, self.im_lo, self.im_hi, self.depth + 1))
        mid = (self.im_lo + self.im_hi) / 2
        return (Box(self.re_lo, self.re_hi, self.im_lo, mid, self.depth + 1),
                Box(self.re_lo, self.re_hi, mid, self.im_hi, self.depth + 1))


def tiled(lo, hi, step):
    current = Decimal(lo)
    end = Decimal(hi)
    width = Decimal(step)
    while current < end:
        following = min(end, current + width)
        yield current, following
        current = following


def check_real(bounds, region):
    if region == "low":
        return (upper(bounds["tau"]) < interval("0.95")
                and upper(bounds["C_N"]) < interval(6)
                and upper(bounds["C_infinity"]) < interval(6))
    if region == "middle":
        return (lower(bounds["log_tau_prime"]) > interval("0.85")
                and upper(abs(bounds["log_C_N_prime"])) < interval(10)
                and upper(abs(bounds["log_C_infinity_prime"])) < interval(10)
                and lower(bounds["C_N"]) > 1
                and lower(bounds["C_infinity"]) > 1)
    if region == "high":
        return (lower(bounds["tau"]) > interval("1.03")
                and lower(bounds["C_N"]) > 1
                and lower(bounds["C_infinity"]) > 1)
    raise ValueError(region)


def check_complex(bounds):
    return (upper(bounds["abs_log_tau_prime"]) < 2
            and upper(bounds["abs_log_C_infinity_prime"]) < 100
            and bounds["relative_C_error"] < interval("0.001")
            and lower(bounds["abs_C_infinity"]) > 1
            and bounds["q"] < interval("0.93"))


def row_for(box, region, bounds):
    row = {"region": region, "re_lo": str(box.re_lo), "re_hi": str(box.re_hi),
           "im_lo": str(box.im_lo), "im_hi": str(box.im_hi), "depth": box.depth}
    for name, value in bounds.items():
        row[f"{name}_lo"] = float_down(value)
        row[f"{name}_hi"] = float_up(value)
    return row


def certify_boxes(initial, region, complex_mode=False, max_depth=20):
    pending = list(reversed(initial))
    rows, attempts = [], 0
    start, last_report = perf_counter(), perf_counter()
    while pending:
        box = pending.pop()
        attempts += 1
        z = complex_box(box.re_lo, box.re_hi, box.im_lo, box.im_hi)
        reason = "bound not met"
        try:
            bounds = complex_band_bounds(z, M0) if complex_mode else real_band_bounds(z, M0)
            accepted = check_complex(bounds) if complex_mode else check_real(bounds, region)
        except (ArithmeticError, ZeroDivisionError, ValueError) as error:
            reason, accepted = str(error), False
        if accepted:
            rows.append(row_for(box, region, bounds))
        else:
            if box.depth >= max_depth:
                raise ArithmeticError(f"Uncertified box: {box}; {reason}")
            first, second = box.split()
            pending.extend((second, first))
        if perf_counter() - last_report > 10:
            print(f"{region}: accepted={len(rows)}, pending={len(pending)}, "
                  f"attempts={attempts}, elapsed={perf_counter()-start:.1f}s", flush=True)
            last_report = perf_counter()
    print(f"{region}: CERTIFIED {len(rows)} boxes in {perf_counter()-start:.1f}s", flush=True)
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analytic_gate():
    m = interval(M0)
    radius_coefficient, cauchy_distance = interval("0.01"), interval("0.0005")
    radius = radius_coefficient / m
    derivative_lower = interval("0.85") * m - 10
    derivative_upper = 2 * m + 100
    taylor_error = derivative_upper * radius * radius / (2 * cauchy_distance)
    phase_upper = derivative_upper * radius + taylor_error
    exponential_error = iv.exp(phase_upper) - 1 - phase_upper
    leading_margin = derivative_lower * radius - taylor_error - exponential_error
    perturbation = interval("0.001") * iv.exp(phase_upper)
    low = interval(6) * interval("0.95") ** M0
    high = interval("1.03") ** M0
    checks = {"low_sign": bool(upper(low) < interval("0.2")),
              "high_sign": bool(lower(high) > interval("0.2")),
              "real_monotonicity": bool(lower(derivative_lower) > 0),
              "local_rouche": bool(lower(leading_margin) > upper(perturbation)),
              "disk_inside_extension": bool(upper(radius + cauchy_distance) < interval("0.001"))}
    if not all(checks.values()):
        raise ArithmeticError(f"Analytic final gate failed: {checks}")
    return {"checks": checks, "low_C_tau_power_upper": float_up(low),
            "high_C_tau_power_lower": float_down(high),
            "rouche_leading_margin_over_kappa_lower": float_down(leading_margin),
            "rouche_perturbation_over_kappa_upper": float_up(perturbation),
            "phase_upper_at_cutoff": float_up(phase_upper)}


def source_hashes():
    paths = (Path(__file__), Path(__file__).with_name("boundary_evans_interval.py"),
             Path(__file__).with_name("boundary_evans.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("real", "complex", "both"), default="both")
    parser.add_argument("--real-step", default="0.002")
    parser.add_argument("--complex-step", default="0.001")
    args = parser.parse_args()
    report = {"scope": "fixed J=0.01 model; all integer N>=121; real |omega|<=2",
              "interval_digits": iv.dps, "m0": M0, "source_sha256": source_hashes(),
              "analytic_gate": analytic_gate(), "completed_parts": []}
    if args.part in ("real", "both"):
        rows = []
        for region, lo, hi in (("low", "0", "0.15"),
                               ("middle", "0.15", "0.25"),
                               ("high", "0.25", "2")):
            boxes = [Box(a, b) for a, b in tiled(lo, hi, args.real_step)]
            rows.extend(certify_boxes(boxes, region))
        path = RESULTS / "boundary_evans_interval_real_boxes.csv"
        write_csv(path, rows)
        report["real_box_count"] = len(rows)
        report["real_certificate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["completed_parts"].append("real")
    if args.part in ("complex", "both"):
        boxes = [Box(a, b, c, d)
                 for a, b in tiled("0.149", "0.251", args.complex_step)
                 for c, d in tiled("-0.001", "0.001", args.complex_step)]
        rows = certify_boxes(boxes, "complex", complex_mode=True)
        path = RESULTS / "boundary_evans_interval_complex_boxes.csv"
        write_csv(path, rows)
        report["complex_box_count"] = len(rows)
        report["complex_certificate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["completed_parts"].append("complex")
    name = "boundary_evans_all_N_certificate" if args.part == "both" else f"boundary_evans_{args.part}_certificate"
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
