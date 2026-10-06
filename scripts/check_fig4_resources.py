"""Audit interval coverage and independently integrate the fixed devices."""

from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from shared_band_resources import MODELS, drift_matrix
from audit_size_uniform_resources import extension
from strengthen_fig4_resources import contiguous_target_intervals, downward, upward


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / "results/fig4_shared_device_certificate.json"
    certificate = json.loads(path.read_text(encoding="utf-8"))
    expected = {"scripts/strengthen_fig4_resources.py": certificate["source_sha256"],
                "results/size_uniform_resources_certificate.json": certificate["base_certificate_sha256"],
                "results/size_uniform_resources_audit.json": certificate["base_audit_sha256"],
                **certificate["helper_sha256"]}
    assert certificate["all_checks_passed"]
    assert all(sha(ROOT / filename) == digest for filename, digest in expected.items())
    records, physical = [], []
    for group in certificate["groups"]:
        model = MODELS[group["model"]]
        witnesses = group["witnesses"]
        target = F(str(group["target_noise"]))
        assert group["all_N_from"] == 64 and group["complete_rate_cap"] == 8
        assert len(witnesses) >= 81 and len(group["cells"]) == 160
        assert all(sha(ROOT / w["matrix_file"]) == w["matrix_sha256"] for w in witnesses)
        assert all(sha(ROOT / d["matrix_file"]) == d["matrix_sha256"]
                   for d in group["fixed_devices"].values())
        previous = F(3, 100)
        for cell in group["cells"]:
            a, b = map(F, cell["halfwidth_interval_exact"])
            assert a == previous and a < b and b - a == F(1, 4000)
            lower = max(F(w["halfwidth_exact"]) / b * max(F(0),
                        F(w["alpha_exact"]) - 8 * F(w["beta_exact"]))
                        for w in witnesses if F(w["halfwidth_exact"]) <= a)
            assert lower == F(cell["local_lower_exact"])
            reference = b / a * F(cell["reference_endpoint_upper_exact"])
            range2 = b / a * F(cell["range2_endpoint_upper_exact"])
            assert reference == F(cell["reference_upper_exact"])
            assert range2 == F(cell["range2_upper_exact"])
            assert lower - reference == F(cell["gap_lower_exact"]) > 0
            assert cell["minimum_range_two_certified"] == (range2 < target < lower)
            assert cell["local_lower"] == downward(lower)
            assert cell["reference_upper"] == upward(reference)
            assert cell["range2_upper"] == upward(range2)
            previous = b
        assert previous == F(7, 100)
        assert group["minimum_range_two_intervals"] == contiguous_target_intervals(group["cells"])
        assert group["minimum_gap_lower"] == min(c["gap_lower"] for c in group["cells"])
        # Direct resolvents supply a separate integration path from the polynomial recurrence.
        for n in (64, 65):
            probe = drift_matrix(model, n)
            probe[0, 0] -= float(model.kappa_in / 2)
            probe[-1, -1] -= float(model.kappa_out / 2)
            nodes, weights = leggauss(n + 8)
            for width in (.030125, .049875, .069875):
                profiles = np.column_stack([
                    np.linalg.solve(-1j * width * x * np.eye(n) - probe.T, np.eye(n)[:, -1])
                    for x in nodes])
                profiles /= profiles[0, :] * np.sqrt(float(model.kappa_in))
                cell = next(c for c in group["cells"] if c["halfwidth_interval"][0] < width
                            < c["halfwidth_interval"][1])
                for key, device in group["fixed_devices"].items():
                    with np.load(ROOT / device["matrix_file"], allow_pickle=False) as saved:
                        gain, loss = extension(model, saved["loss"], n)
                    noise = float(np.dot(weights / 2,
                        np.einsum("iw,ij,jw->w", profiles.conj(), gain, profiles).real))
                    assert noise <= cell[f"{key}_upper"] + 1e-8
                    physical.append({"model": model.name, "device": key, "N": n,
                                     "halfwidth": width, "direct_resolvent_noise": noise,
                                     "certified_upper": cell[f"{key}_upper"]})
        records.append({"model": model.name, "local_witnesses": len(witnesses),
                        "certified_cells": len(group["cells"]),
                        "minimum_gap_lower": group["minimum_gap_lower"],
                        "minimum_range_two_intervals": group["minimum_range_two_intervals"]})
    output = {"certificate_sha256": sha(path), "checker_sha256": sha(Path(__file__)),
              "all_checks_passed": True, "exact_interval_algebra_and_coverage": True,
              "independent_direct_resolvent_checks_are_floating": True,
              "models": records, "direct_resolvent_checks": physical}
    (ROOT / "results/reports/fig4_strengthening_scientific_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": True, "models": records,
                      "direct_resolvent_checks": len(physical)}, indent=2))


if __name__ == "__main__":
    main()
