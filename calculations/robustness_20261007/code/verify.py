"""Independent dyadic positivity, full auxiliary model and numerical cross-checks."""

from fractions import Fraction as F
import hashlib
import itertools
import json
import sys

import mpmath as mp
import numpy as np
from scipy.linalg import eigvals, solve

from physics import (CONFIG, MODELS, ROOT, device, grams, matrices, noises, parameters,
                     response, stability)
from certify_box import coefficients, response_enclosure


def rational(a):
    return np.array([F.from_float(float(x)) for x in a.flat], dtype=object).reshape(a.shape)


def factor_product(a):
    ratios = [float(x).as_integer_ratio() for x in a.flat]
    denominator = max(x[1] for x in ratios)
    ints = np.array([x * (denominator // y) for x, y in ratios], dtype=object).reshape(a.shape)
    product = ints @ ints.T
    return np.array([F(int(v), denominator**2) for v in product.flat], dtype=object).reshape(product.shape)


def positivity():
    records = []
    paths = [ROOT / f"inputs/{name}_{family}_r{radius}.npz"
             for name, family, radius in itertools.product(MODELS, ("finite", "extended"), (1, 2))]
    paths += sorted((ROOT / "results/trace_budget_witnesses").glob("*.npz"))
    for path in paths:
        radius = int(path.stem[-1])
        with np.load(path, allow_pickle=False) as d:
            loss, source, cap = rational(d["loss"]), rational(d["source"]), F.from_float(float(d["cap"]))
            eye = np.eye(len(loss), dtype=object) * F(1)
            eta = F.from_float(float(d["intrinsic"])) if "intrinsic" in d else F(0)
            gain = loss + source
            targets = {"programmed_loss": loss - eta * eye, "gain": gain,
                       "complete_loss_slack": cap * eye - loss, "gain_slack": cap * eye - gain}
            for key, target in targets.items():
                shift = F.from_float(float(d[f"primal_{key}_shift"]))
                residual = target - shift * eye - factor_product(d[f"primal_{key}_factor"])
                bound = max(sum((abs(v) for v in row), F(0)) for row in residual)
                margin = shift - bound - F(1, 10**11)
                assert margin > 0
                records.append({"file": path.relative_to(ROOT).as_posix(), "constraint": key, "exact_margin_lower": float(margin)})
            distance = abs(np.arange(len(loss))[:, None] - np.arange(len(loss)))
            assert all(v == 0 for v in loss[distance > radius])
            if "trace_cap" in d:
                assert sum((gain[j, j] + loss[j, j] for j in range(len(gain))), F(0)) < F.from_float(float(d["trace_cap"]))
    return records


def full_auxiliary():
    records = []
    for name, changes, linewidth, eta in (("uniform", (0., 0., 0.), 50., .01),
                                         ("uniform", (.005, -.005, .005), 80., .02),
                                         ("uniform", (-.005, .005, -.005), 100., 0.),
                                         ("dimerized", (0., 0., 0.), 80., .01)):
        p, n, occupation = parameters(name, changes), 64, .001
        x, _ = matrices(p, n)
        coherent = (x - x.T) / 2
        ports = np.zeros(n)
        ports[0], ports[-1] = CONFIG["port_rates"]
        zero, eye = np.zeros((n, n), dtype=complex), np.eye(n)
        for radius in (1, 2):
            gain, loss = device(p, name, "finite", radius, n, eta)
            ug = np.sqrt(linewidth) / 2 * np.linalg.cholesky(gain)
            ul = np.sqrt(linewidth) / 2 * np.linalg.cholesky(loss - eta * eye)
            aux = -linewidth / 2 * eye
            augmented = np.block([[coherent - np.diag((eta + ports) / 2), -1j * ul, -1j * ug],
                                  [-1j * ul.T, aux, zero], [1j * ug.T, zero, aux]])
            max_error, max_gain_error, noise_error = 0., 0., 0.
            for omega in (-.07, -.05, 0., .05, .07):
                block = 1j * omega * np.eye(3 * n) - augmented
                y = solve(block.conj().T, np.eye(3 * n)[:, n - 1])
                h, alpha, power_gain = response(p, n, np.array([omega]), linewidth, eta)
                actual_profile = y[:n] / (y[0] * np.sqrt(.2))
                max_error = max(max_error, float(np.max(abs(actual_profile - h[0]))))
                max_gain_error = max(max_gain_error, abs(.04 * abs(y[0])**2 / power_gain[0] - 1))
                measured = (((occupation + 1) * linewidth * np.vdot(y[2*n:], y[2*n:]).real
                             + occupation * linewidth * np.vdot(y[n:2*n], y[n:2*n]).real
                             + occupation * eta * np.vdot(y[:n], y[:n]).real) / (.2 * abs(y[0])**2))
                theoretical = ((occupation + 1) * abs(alpha[0])**2 * np.vdot(h[0], gain @ h[0]).real
                               + occupation * abs(alpha[0])**2 * np.vdot(h[0], (loss - eta * eye) @ h[0]).real
                               + occupation * eta * np.vdot(h[0], h[0]).real)
                noise_error = max(noise_error, abs(measured - theoretical) / measured)
            site_scale = np.r_[1., np.cumprod(np.sqrt(p["right"][np.arange(n - 1) % 2]
                                                    / p["left"][np.arange(n - 1) % 2]))]
            scale = np.tile(site_scale, 3)
            balanced_augmented = augmented * scale[None, :] / scale[:, None]
            eigenvalue_difference = abs(np.max(eigvals(balanced_augmented).real) - stability(p, n, linewidth, eta))
            assert max(max_error, max_gain_error, noise_error, eigenvalue_difference) < 2e-8, (
                name, changes, linewidth, eta, radius, max_error, max_gain_error, noise_error, eigenvalue_difference)
            records.append({"model": name, "changes": changes, "linewidth": linewidth, "intrinsic": eta, "radius": radius,
                            "normalized_response_error": max_error, "relative_power_gain_error": max_gain_error,
                            "relative_warm_noise_error": noise_error, "spectral_abscissa_difference": eigenvalue_difference})
    return records


def high_precision_enclosures():
    mp.mp.dps = 70
    rng = np.random.default_rng(CONFIG["seed"])
    worst_ratio, records = 0., []
    cfg = CONFIG["certificate_box"]
    for j in range(12):
        # Include narrow endpoint cells and interior cells independently of the solver grid.
        centres = np.array([1.2, 1., .25]) * (1 + rng.uniform(-.0048, .0048, 3))
        radii = np.array([1.2, 1., .25]) * .005 / cfg["parameter_subdivisions"]
        box = [(float(c - r), float(c + r)) for c, r in zip(centres, radii)]
        kb, eta = rng.uniform(82.5, 97.5), rng.uniform(.005, .015)
        box += [(kb - 2.5, kb + 2.5), (eta - .005, eta + .005)]
        omega = rng.uniform(0.0002, .0698)
        lo, hi = np.array([omega - .000125]), np.array([omega + .000125])
        a, b, boundary, _, _, _ = coefficients(box, lo, hi)
        centre, radius = response_enclosure(a, b, boundary, 64)
        for sample in range(4):
            values = [rng.uniform(*pair) for pair in box]
            w = mp.mpf(str(rng.uniform(lo[0], hi[0])))
            gamma, right, left, linewidth, intrinsic = [mp.mpf(str(v)) for v in values]
            ac = 1 / (1 - 2j * w / linewidth)
            vr = (right - left) / 2 + ac * (right + left) / 2
            vl = -(right - left) / 2 + ac * (right + left) / 2
            diag = intrinsic / 2 + ac * (gamma - intrinsic / 2) - 1j * w
            h = [mp.mpc(1), (diag + mp.mpf("0.1")) / vr]
            for k in range(1, 63):
                h.append((diag * h[k] - vl * h[k - 1]) / vr)
            differences = np.array([float(abs(value - mp.mpc(complex(centre[0, k])))) for k, value in enumerate(h)])
            ratio = max(differences[1:] / radius[0, 1:])
            worst_ratio = max(worst_ratio, ratio)
            assert ratio < 1
        records.append({"box": box, "maximum_error_over_radius": float(ratio)})
    return {"precision_digits": 70, "random_samples": 48, "maximum_error_over_radius": float(worst_ratio), "records": records}


def quadratures():
    records = []
    for name, changes, width, linewidth, eta, n in (
        ("uniform", (0., 0., 0.), .05, 50., .01, 64),
        ("uniform", (.01, -.01, .01), .07, 20., .02, 128),
        ("uniform", (-.01, .01, .01), .07, 20., .02, 64),
        ("dimerized", (.01, -.01, .01), .07, 100., .02, 128)):
        p = parameters(name, changes)
        values = []
        for order in (97, 193):
            w, z, *_ = grams(p, n, width, linewidth, eta, order)
            pair = device(p, name, "extended", 2, n, eta)
            values.append(noises(*pair, w, z, eta, .02))
        error = abs(values[0] - values[1])
        assert error < 1e-8
        records.append({"model": name, "N": n, "changes": changes, "noise_97": values[0], "noise_193": values[1], "difference": error})
    return records


def main():
    report = {"exact_psd_factor_checks": positivity(), "full_auxiliary_cross_checks": full_auxiliary(),
              "high_precision_disk_checks": high_precision_enclosures(), "quadrature_cross_checks": quadratures(),
              "scope": "Automated independent mathematical/numerical checks, no experimental measurements or human author approval",
              "all_checks_passed": True}
    (ROOT / "results/independent_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact_psd_checks": len(report["exact_psd_factor_checks"]),
                      "full_auxiliary_checks": len(report["full_auxiliary_cross_checks"]),
                      "high_precision_max_disk_ratio": report["high_precision_disk_checks"]["maximum_error_over_radius"],
                      "all_checks_passed": True}), flush=True)


if __name__ == "__main__":
    main()
