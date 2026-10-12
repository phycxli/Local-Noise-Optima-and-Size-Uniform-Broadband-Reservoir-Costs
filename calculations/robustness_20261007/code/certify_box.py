"""Enclose a continuous finite-device parameter box, including every frequency.

Interval coefficients use outward rounding. A causal Green convolution bounds
response perturbations without the wrapping explosion of naive interval recurrence.
This certifies two feasible shared devices, not the optimized locality gap.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from physics import CONFIG, ROOT, device, parameters, stability_sufficient_bounds

EPS = np.finfo(float).eps


class Interval:
    def __init__(self, lower, upper=None):
        self.lo = np.asarray(lower, dtype=np.longdouble)
        self.hi = np.asarray(lower if upper is None else upper, dtype=np.longdouble)

    @staticmethod
    def enclose(lower, upper):
        return Interval(np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf))

    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return self.enclose(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Interval) else -Interval(other))

    def __rsub__(self, other):
        return Interval(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        candidates = np.stack(np.broadcast_arrays(self.lo * other.lo, self.lo * other.hi,
                                                  self.hi * other.lo, self.hi * other.hi))
        return self.enclose(candidates.min(axis=0), candidates.max(axis=0))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        if np.any((other.lo <= 0) & (other.hi >= 0)):
            raise ValueError("Interval division through zero")
        reciprocal = self.enclose(1 / other.hi, 1 / other.lo)
        return self * reciprocal

    def square(self):
        low = np.minimum(self.lo**2, self.hi**2)
        high = np.maximum(self.lo**2, self.hi**2)
        low = np.where((self.lo <= 0) & (self.hi >= 0), 0, low)
        return self.enclose(low, high)


def cadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def csub(a, b):
    return a[0] - b[0], a[1] - b[1]


def cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cdiv(a, b):
    denominator = b[0].square() + b[1].square()
    numerator = cmul(a, (b[0], -b[1]))
    return numerator[0] / denominator, numerator[1] / denominator


def disk(rect):
    real, imaginary = rect
    centre = np.asarray((real.lo + real.hi) / 2, dtype=float) + 1j * np.asarray((imaginary.lo + imaginary.hi) / 2, dtype=float)
    re = np.maximum(abs(real.lo - centre.real), abs(real.hi - centre.real))
    im = np.maximum(abs(imaginary.lo - centre.imag), abs(imaginary.hi - centre.imag))
    radius = np.asarray(np.sqrt(re**2 + im**2), dtype=float)
    return centre, np.nextafter(radius * (1 + 32 * EPS) + 1e-18, np.inf)


def coefficients(box, wlo, whi):
    gamma, right, left, linewidth, intrinsic = [Interval(*pair) for pair in box]
    w = Interval(wlo, whi)
    t = 2 * w / linewidth
    denominator = Interval(1) + t.square()
    ac = (Interval(1) / denominator, t / denominator)
    defect = (Interval(1) - ac[0], -ac[1])
    vr = csub((right, Interval(0)), cmul(defect, ((right + left) / 2, Interval(0))))
    vl = csub((left, Interval(0)), cmul(defect, ((right + left) / 2, Interval(0))))
    diag = cadd(cadd(cmul(ac, (gamma, Interval(0))), cmul(defect, (intrinsic / 2, Interval(0)))),
                (Interval(0), -w))
    a, b = cdiv(diag, vr), cdiv(vl, vr)
    boundary = cdiv((Interval(.1), Interval(0)), vr)
    last = cadd(diag, (Interval(.1), Interval(0)))
    return disk(a), disk(b), disk(boundary), disk(last), disk(vl), ac[0]


def response_enclosure(a_disk, b_disk, boundary_disk, n):
    a, da = a_disk
    b, db = b_disk
    c, dc = boundary_disk
    count = len(a)
    h, impulse = np.zeros((count, n), dtype=complex), np.zeros((count, n), dtype=complex)
    h[:, 0], h[:, 1] = 1, a + c
    impulse[:, 0], impulse[:, 1] = 1, a
    rh, rg = np.zeros((count, n)), np.zeros((count, n))
    rh[:, 1] = 16 * EPS * (abs(a) + abs(c))
    for j in range(1, n - 1):
        h[:, j + 1] = a * h[:, j] - b * h[:, j - 1]
        impulse[:, j + 1] = a * impulse[:, j] - b * impulse[:, j - 1]
        rh[:, j + 1] = ((abs(a) * rh[:, j] + abs(b) * rh[:, j - 1])
                        + 32 * EPS * (abs(a * h[:, j]) + abs(b * h[:, j - 1]))) * (1 + 1e-12)
        rg[:, j + 1] = ((abs(a) * rg[:, j] + abs(b) * rg[:, j - 1])
                        + 32 * EPS * (abs(a * impulse[:, j]) + abs(b * impulse[:, j - 1]))) * (1 + 1e-12)
    impulse_bound = (abs(impulse) + rg) * (1 + 1e-12)
    h_bound = abs(h) + rh
    error = np.zeros((count, n))
    error[:, 1] = da + dc
    initial = da + dc
    for j in range(2, n):
        # The forcing at k only involves already-bounded response sites k,k-1.
        forcing = (da[:, None] * (h_bound[:, 1:j] + error[:, 1:j])
                   + db[:, None] * (h_bound[:, :j-1] + error[:, :j-1]))
        error[:, j] = (impulse_bound[:, j - 1] * initial
                       + np.sum(impulse_bound[:, j-2::-1] * forcing, axis=1)) * (1 + 1e-12)
    return h, (error + rh) * (1 + 1e-12)


def cells(interval, count):
    edges = np.linspace(*interval, count + 1)
    return list(zip(edges[:-1], edges[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = CONFIG["certificate_box"]
    span, subdivisions = cfg["relative_parameter_halfspan"], cfg["parameter_subdivisions"]
    nominal = [1.2, 1., .25]
    axes = [cells((v * (1 - span), v * (1 + span)), subdivisions) for v in nominal]
    axes += [cells(cfg["linewidth_interval"], cfg["linewidth_subdivisions"]),
             cells(cfg["intrinsic_interval"], cfg["intrinsic_subdivisions"])]
    boxes = list(itertools.product(*axes))
    selected = [(i, box) for i, box in enumerate(boxes) if i % args.shards == args.shard]
    if args.limit:
        selected = selected[:args.limit]
    maxw, dw = cfg["band_halfwidth_interval"][1], cfg["frequency_step"]
    edges = np.arange(round(maxw / dw) + 1) * dw
    wlo, whi = edges[:-1], edges[1:]
    # Enlarge binary grid endpoints to include the exact decimal frequency cells.
    ilo, ihi = np.nextafter(wlo, -np.inf), np.nextafter(whi, np.inf)
    p = parameters("uniform")
    g1 = device(p, "uniform", "finite", 1, 64, 0)[0]
    g2 = device(p, "uniform", "finite", 2, 64, 0)[0]
    difference = g1 - g2
    difference_error = 4 * EPS * (abs(g1) + abs(g2))
    start, records = time.perf_counter(), []
    for cell_id, box in selected:
        a, b, boundary, last, vl, alpha_power = coefficients(box, ilo, ihi)
        h, error = response_enclosure(a, b, boundary, 64)
        bound = abs(h) + error
        centre = np.einsum("fi,ij,fj->f", h.conj(), difference, h).real
        dh = h @ difference.T
        uncertainty = (2 * np.sum(abs(dh) * error, axis=1)
                       + np.einsum("fi,ij,fj->f", error, abs(difference), error)
                       + np.einsum("fi,ij,fj->f", bound, difference_error, bound))
        uncertainty += 128 * EPS * 64 * np.einsum("fi,ij,fj->f", bound, abs(difference), bound) + 1e-8
        q = Interval.enclose(np.asarray(centre - uncertainty, dtype=np.longdouble),
                             np.asarray(centre + uncertainty, dtype=np.longdouble))
        f = q * alpha_power / Interval(.2)
        flower = np.asarray(f.lo, dtype=float)
        prefix_integral = np.r_[0., np.cumsum(flower * dw)]
        trace = np.sum(bound**2, axis=1) / .2 * (1 + 1e-12)
        prefix_trace = np.r_[0., np.cumsum(trace * dw)]
        first = round(cfg["band_halfwidth_interval"][0] / dw)
        lower_curves, trace_curves = [], []
        for j in range(first, len(flower)):
            integral_lower = prefix_integral[j] + min(0., flower[j]) * dw - 1e-8
            lower_curves.append(integral_lower / (whi[j] if integral_lower >= 0 else wlo[j]))
            trace_curves.append((prefix_trace[j] + trace[j] * dw) / wlo[j])
        d0, dr = last
        l0, lr = vl
        denominator = abs(d0 * h[:, -1] - l0 * h[:, -2])
        denominator += (abs(d0) * error[:, -1] + dr * bound[:, -1]
                        + abs(l0) * error[:, -2] + lr * bound[:, -2] + 1e-12)
        gain_lower = .04 / denominator**2
        record = {"cell_id": cell_id, "box": [[float(v) for v in pair] for pair in box],
                  "band_gap_lower": float(np.nextafter(min(lower_curves), -np.inf)),
                  "band_profile_energy_upper": float(np.nextafter(max(trace_curves), np.inf)),
                  "whole_band_power_gain_lower": float(np.nextafter(min(gain_lower), -np.inf)),
                  "maximum_response_radius": float(error.max()),
                  "minimum_pointwise_gap_bound": float(min(flower))}
        records.append(record)
        if len(records) % 8 == 0:
            print(json.dumps({"shard": args.shard, "cells": len(records),
                              "gap_lower_so_far": min(r["band_gap_lower"] for r in records),
                              "seconds": round(time.perf_counter() - start, 2)}), flush=True)
    stability_bound = stability_sufficient_bounds(span, cfg["linewidth_interval"][0], cfg["intrinsic_interval"][1])
    result = {"configuration": cfg, "shard": args.shard, "shards": args.shards, "cells": records,
              "arithmetic": "Outward-rounded long-double coefficient rectangles; double causal response disks with explicit roundoff allowances",
              "scope": "N=64; two fixed feasible devices, common drift calibration; continuous parameter/frequency/bandwidth box",
              "thermal_extension": "The band gap is multiplied by 1+2*n_b at uniform internal occupation",
              "stability_sufficient_bound": stability_bound, "seconds": time.perf_counter() - start,
              "all_gap_bounds_positive": all(r["band_gap_lower"] > 0 for r in records)}
    directory = ROOT / "results/box_shards"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"box_{args.shard:02d}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cells", "configuration"}}), flush=True)


if __name__ == "__main__":
    main()
