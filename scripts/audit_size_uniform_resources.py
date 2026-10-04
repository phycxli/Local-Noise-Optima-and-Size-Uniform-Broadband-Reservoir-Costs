"""Check explicit extensions and certify a continuous bandwidth interval."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import (MODELS, drift_matrix, exact_gram, floquet_envelope,
                                   endpoint_gain, gram_entry, legendre_coefficients, normalized_profile,
                                   profile_bound, source_matrix, tail_bound,
                                   uniform_gain_certificate)
from check_size_uniform_resources import check_prefix, downward, upward
from run_rate_capped_band_optimization import solve_shared


def extension(model, prefix_loss: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    m = len(prefix_loss)
    loss = np.zeros((n, n))
    gain = np.zeros((n, n))
    loss[:m, :m] = prefix_loss
    gain[:m, :m] = prefix_loss + np.asarray(source_matrix(model, m), dtype=float)
    for j in range(m, n):
        loss[j, j] = 2 * float(model.gamma[j % 2])
    for j in range(m - 1, n - 1):
        rate = float(model.right[j % 2] + model.left[j % 2]) / 2
        gain[j:j + 2, j:j + 2] += rate * np.array([[1., 1.], [1., 1.]])
        loss[j:j + 2, j:j + 2] += rate * np.array([[1., -1.], [-1., 1.]])
    return gain, loss


def extension_bound(model, coefficients, prefix: int, finite_upper: F,
                    envelope: dict) -> F:
    rates = [model.right[j] + model.left[j] for j in range(2)]
    last = len(coefficients) - 2
    bridge = rates[(prefix - 1) % 2] / 2 * (
        gram_entry(model, coefficients, prefix - 1, prefix - 1)
        + 2 * gram_entry(model, coefficients, prefix - 1, prefix)
        + gram_entry(model, coefficients, prefix, prefix))
    tail = sum((rates[j % 2] / 2 * (
        gram_entry(model, coefficients, j, j)
        + 2 * gram_entry(model, coefficients, j, j + 1)
        + gram_entry(model, coefficients, j + 1, j + 1))
        for j in range(prefix, last)), F(0))
    return finite_upper + bridge + tail + sum(rates) * tail_bound(model, envelope, last)


def boundary_coefficient(model, coefficients, prefix: int, radius: int) -> F:
    return sum((abs(gram_entry(model, coefficients, i, j))
                for i in range(max(0, prefix - radius), prefix)
                for j in range(prefix, prefix + radius) if j - i <= radius), F(0))


def direct_audit(rows: list[dict], groups: list[dict]) -> list[dict]:
    checks = []
    for group in groups:
        model = MODELS[group["model"]]
        width, m = F(str(group["halfwidth"])), group["prefix"]
        reference = next(row for row in rows if row["model"] == model.name
                         and F(row["halfwidth"]) == width and row["radius"] == "all")
        with np.load(ROOT / reference["matrix_file"], allow_pickle=False) as saved:
            prefix_loss = saved["loss"].copy()
        envelope = floquet_envelope(model, width)
        for n in (m, m + 1, 96, 128):
            gain, loss = extension(model, prefix_loss, n)
            source = np.asarray(source_matrix(model, n), dtype=float)
            mismatch = float(np.max(np.abs(gain - loss - source)))
            spectra = [np.linalg.eigvalsh(matrix) for matrix in (gain, loss)]
            psd_minimum, rate = min(v.min() for v in spectra), max(v.max() for v in spectra)
            frequencies = (0., float(width) / 3, float(width), -float(width))
            response_error, bound_ratio = 0., 0.
            probe = drift_matrix(model, n)
            probe[0, 0] -= float(model.kappa_in / 2)
            probe[-1, -1] -= float(model.kappa_out / 2)
            target = np.eye(n)[:, -1]
            for omega in frequencies:
                physical = np.linalg.solve(-1j * omega * np.eye(n) - probe.T, target)
                physical /= physical[0] * np.sqrt(float(model.kappa_in))
                polynomial = normalized_profile(model, n, omega)
                response_error = max(response_error, float(np.max(np.abs(physical - polynomial))))
                bounds = np.array([float(profile_bound(envelope, j)) for j in range(n)])
                bound_ratio = max(bound_ratio, float(np.max(np.abs(polynomial)
                                  * np.sqrt(float(model.kappa_in)) / bounds)))
            nodes, weights = leggauss(n + 8)
            profiles = np.column_stack([normalized_profile(model, n, float(width) * x) for x in nodes])
            values = np.einsum("iw,ij,jw->w", profiles.conj(), gain, profiles).real
            noise = float(np.dot(weights / 2, values))
            if (mismatch > 1e-13 or psd_minimum < -1e-12 or rate > 8 + 1e-10
                    or response_error > 1e-10 or bound_ratio > 1 + 1e-10
                    or noise > group["nonlocal_construction_upper"] + 1e-8):
                raise ValueError(f"Direct extension audit failed: {model.name}/{width}/{n}")
            checks.append({"model": model.name, "halfwidth": float(width), "N": n,
                           "drift_mismatch": mismatch, "psd_minimum": float(psd_minimum),
                           "complete_rate": float(rate), "response_error": response_error,
                           "profile_bound_ratio": bound_ratio, "quadrature_noise": noise})
    return checks


def stronger_gain_certificates(groups: list[dict]) -> list[dict]:
    result = []
    for group in groups:
        model, m = MODELS[group["model"]], group["prefix"]
        width = F(str(group["halfwidth"]))
        envelope = floquet_envelope(model, width)
        prefix_gain, cutoff = endpoint_gain(model, m, width), m
        while True:
            profile = max(profile_bound(envelope, cutoff - 2), profile_bound(envelope, cutoff - 1))
            denominator = (max(model.gamma) + model.kappa_out / 2 + width + max(model.left)) * profile
            tail_lower = model.kappa_in * model.kappa_out / denominator**2
            if tail_lower >= prefix_gain:
                break
            cutoff += 2
            if cutoff > 1000:
                raise ValueError("The stronger gain tail bound failed to close.")
        gains = [endpoint_gain(model, n, width) for n in range(m, cutoff)]
        lower = min(gains + [tail_lower])
        result.append({"model": model.name, "halfwidth": float(width), "all_N_from": m,
                       "whole_band_gain_lower_all_N": downward(lower), "gain_lower_exact": str(lower),
                       "tail_from_N": cutoff, "finite_lengths_exactly_checked": len(gains)})
    return result


def continuous_certificate(rows: list[dict], halfspan: F) -> list[dict]:
    result = []
    a, b = F(1, 20) - halfspan, F(1, 20) + halfspan
    if a <= 0:
        raise ValueError("The continuous bandwidth interval must be positive.")
    for name, model in MODELS.items():
        base = [row for row in rows if row["model"] == name and F(row["halfwidth"]) == F(1, 20)]
        m = int(base[0]["prefix"])
        coefficients_a, coefficients_b = [legendre_coefficients(model, 160, width) for width in (a, b)]
        gram_a, gram_b = [exact_gram(model, c, m) for c in (coefficients_a, coefficients_b)]
        source = source_matrix(model, m)
        local = check_prefix(next(row for row in base if row["radius"] == "1"), gram_a, source)
        reference = check_prefix(next(row for row in base if row["radius"] == "all"), gram_b, source)
        envelope = floquet_envelope(model, b)
        gain_certificate = uniform_gain_certificate(model, envelope, b, m)
        beta = local["_beta"] + boundary_coefficient(model, coefficients_a, m, 1)
        alpha = local["_alpha"]
        reference_b = extension_bound(model, coefficients_b, m, reference["_upper"], envelope)
        lower = max(F(0), a / b * (alpha - 8 * beta))
        upper = b / a * reference_b
        gap = lower - upper
        result.append({"model": name, "all_N_from": m, "halfwidth_interval": [float(a), float(b)],
                       "halfwidth_interval_exact": [str(a), str(b)],
                       "same_prefix_matrix_for_all_widths": reference["matrix_file"],
                       "local_lower_R8": downward(lower), "reference_upper_R8": upward(upper),
                       "locality_gap_lower_R8": downward(gap), "positive_gap": gap > 0,
                       "alpha_at_a_exact": str(alpha), "beta_at_a_exact": str(beta),
                       "reference_upper_at_b_exact": str(reference_b),
                       "necessary_R_for_ref_plus_0p1": downward(max(F(0),
                           (alpha - (b / a)**2 * reference_b - b / a * F(1, 10)) / beta)),
                       "whole_band_gain_lower": float(np.nextafter(
                           gain_certificate["whole_band_gain_lower_all_N"], -np.inf))})
    return result


def range_two_constructions(rows: list[dict], solve: bool) -> list[dict]:
    directory = ROOT / "results/size_uniform_range2_matrices"
    directory.mkdir(exist_ok=True)
    result = []
    width = F(1, 20)
    for name, model in MODELS.items():
        base = next(row for row in rows if row["model"] == name and F(row["halfwidth"]) == width)
        m = int(base["prefix"])
        cap = F(8) - model.right[(m - 1) % 2] - model.left[(m - 1) % 2]
        path = directory / f"{name}_m{m}_B0.05_R{float(cap):g}_r2.npz"
        coefficients = legendre_coefficients(model, 160, width)
        gram = exact_gram(model, coefficients, m)
        source = source_matrix(model, m)
        if solve:
            numerical_gram = np.asarray(gram, dtype=float)
            error = np.array([float(abs(value - F.from_float(float(value))))
                              for value in gram.flat]).reshape(gram.shape)
            print(f"Solving explicit range-two prefix: {name}, cap={cap}", flush=True)
            solve_shared(m, float(cap), numerical_gram, error, np.asarray(source, dtype=float),
                         2, 2e-10, path)
        if not path.exists():
            continue
        row = {"model": name, "prefix": str(m), "halfwidth": str(width), "radius": "2",
               "rate_cap": str(cap), "matrix_file": str(path.relative_to(ROOT)).replace("\\", "/"),
               "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "quadrature_error": base["quadrature_error"],
               "response_normalization_error": base["response_normalization_error"]}
        checked = check_prefix(row, gram, source)
        upper = extension_bound(model, coefficients, m, checked["_upper"], floquet_envelope(model, width))
        with np.load(path, allow_pickle=False) as saved:
            loss = saved["loss"].copy()
        extended_checks = []
        for n in (m + 1, 96, 128):
            gain_n, loss_n = extension(model, loss, n)
            spectra = [np.linalg.eigvalsh(matrix) for matrix in (gain_n, loss_n)]
            rate = max(values.max() for values in spectra)
            minimum = min(values.min() for values in spectra)
            distance = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            if minimum < -1e-12 or rate > 8 + 1e-10 or np.any(gain_n[distance > 2]) or np.any(loss_n[distance > 2]):
                raise ValueError("The explicit range-two extension failed its direct audit.")
            extended_checks.append({"N": n, "complete_rate": float(rate), "psd_minimum": float(minimum)})
        result.append({**row, "all_N_from": m, "complete_rate_cap": 8,
                       "all_N_noise_upper": upward(upper), "noise_upper_exact": str(upper),
                       "minimum_prefix_psd_margin": checked["minimum_psd_margin"],
                       "direct_extended_checks": extended_checks,
                       "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halfspan", default="0.0005")
    parser.add_argument("--solve-local-construction", action="store_true")
    args = parser.parse_args()
    with (ROOT / "results/size_uniform_resources.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path = ROOT / "results/size_uniform_resources_certificate.json"
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if (not certificate["all_checks_passed"]
            or certificate["csv_sha256"] != hashlib.sha256(
                (ROOT / "results/size_uniform_resources.csv").read_bytes()).hexdigest()
            or certificate["checker_sha256"] != hashlib.sha256(
                (ROOT / "scripts/check_size_uniform_resources.py").read_bytes()).hexdigest()
            or certificate["model_helper_sha256"] != hashlib.sha256(
                (ROOT / "src/shared_band_resources.py").read_bytes()).hexdigest()
            or certificate["rational_helper_sha256"] != hashlib.sha256(
                (ROOT / "scripts/check_rate_capped_band_optimization.py").read_bytes()).hexdigest()):
        raise ValueError("The base rational certificate is absent or stale.")
    direct = direct_audit(rows, certificate["groups"])
    print(f"Direct extended-matrix audits passed: {len(direct)}", flush=True)
    gain_certificates = stronger_gain_certificates(certificate["groups"])
    continuous = continuous_certificate(rows, F(args.halfspan))
    print(json.dumps([{k: v for k, v in group.items() if not k.endswith("_exact")}
                      for group in continuous]), flush=True)
    local = range_two_constructions(rows, args.solve_local_construction)
    output = {"base_certificate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "solver_helper_sha256": hashlib.sha256(
                  (ROOT / "scripts/run_rate_capped_band_optimization.py").read_bytes()).hexdigest(),
              "direct_audits_are_floating_cross_checks": True,
              "continuous_bounds_use_exact_rational_endpoints_and_positive_integrals": True,
              "direct_audits": direct, "continuous_bandwidth": continuous,
              "stronger_gain_certificates": gain_certificates,
              "range_two_constructions": local,
              "all_continuous_nearest_neighbor_gaps_positive": all(v["positive_gap"] for v in continuous)}
    (ROOT / "results/size_uniform_resources_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in group.items() if not k.endswith("_exact")}
                      for group in local]), flush=True)


if __name__ == "__main__":
    main()
