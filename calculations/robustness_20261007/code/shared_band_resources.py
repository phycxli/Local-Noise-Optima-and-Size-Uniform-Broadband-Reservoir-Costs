"""Exact normalized responses and tail bounds for period-two Jacobi drifts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt

import numpy as np


@dataclass(frozen=True)
class JacobiModel:
    name: str
    gamma: tuple[F, F]
    right: tuple[F, F]
    left: tuple[F, F]
    kappa_in: F = F(1, 5)
    kappa_out: F = F(1, 5)

    def description(self) -> dict:
        return {"name": self.name, "gamma": list(map(str, self.gamma)),
                "right": list(map(str, self.right)), "left": list(map(str, self.left)),
                "kappa_in": str(self.kappa_in), "kappa_out": str(self.kappa_out)}


MODELS = {
    "uniform": JacobiModel("uniform", (F(6, 5),) * 2, (F(1),) * 2, (F(1, 4),) * 2),
    "dimerized": JacobiModel("dimerized", (F(6, 5), F(23, 20)),
                            (F(23, 20), F(17, 20)), (F(3, 20), F(7, 20))),
}


def source_matrix(model: JacobiModel, n: int) -> np.ndarray:
    source = np.full((n, n), F(0), dtype=object)
    for j in range(n):
        source[j, j] = -2 * model.gamma[j % 2]
        if j + 1 < n:
            source[j, j + 1] = source[j + 1, j] = model.right[j % 2] + model.left[j % 2]
    return source


def drift_matrix(model: JacobiModel, n: int) -> np.ndarray:
    drift = np.diag([-float(model.gamma[j % 2]) for j in range(n)])
    for j in range(n - 1):
        drift[j + 1, j], drift[j, j + 1] = float(model.right[j % 2]), float(model.left[j % 2])
    return drift


def legendre_coefficients(model: JacobiModel, n: int, width: F) -> list[list[F]]:
    c = [[F(0) for _ in range(n)] for _ in range(n)]
    c[0][0] = F(1)
    c[1][0] = (model.gamma[0] + model.kappa_in / 2) / model.right[0]
    c[1][1] = width / model.right[0]
    for j in range(1, n - 1):
        a = model.gamma[j % 2] / model.right[j % 2]
        b = model.left[(j - 1) % 2] / model.right[j % 2]
        w = width / model.right[j % 2]
        for ell in range(j + 2):
            value = a * c[j][ell] - b * c[j - 1][ell]
            if ell:
                value += w * F(ell, 2 * ell - 1) * c[j][ell - 1]
            if ell + 1 < n:
                value -= w * F(ell + 1, 2 * ell + 3) * c[j][ell + 1]
            c[j + 1][ell] = value
    return c


def gram_entry(model: JacobiModel, c: list[list[F]], i: int, j: int) -> F:
    return sum((c[i][ell] * c[j][ell] / (2 * ell + 1)
                for ell in range(min(i, j) + 1)), F(0)) / model.kappa_in


def exact_gram(model: JacobiModel, c: list[list[F]], n: int) -> np.ndarray:
    gram = np.full((n, n), F(0), dtype=object)
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = gram_entry(model, c, i, j)
    return gram


def normalized_profile(model: JacobiModel, n: int, omega: float) -> np.ndarray:
    h = np.zeros(n, dtype=complex)
    h[0] = 1
    h[1] = (float(model.gamma[0] + model.kappa_in / 2) - 1j * omega) / float(model.right[0])
    for j in range(1, n - 1):
        h[j + 1] = ((float(model.gamma[j % 2]) - 1j * omega) * h[j]
                    - float(model.left[(j - 1) % 2]) * h[j - 1]) / float(model.right[j % 2])
    return h / np.sqrt(float(model.kappa_in))


def endpoint_gain(model: JacobiModel, n: int, width: F) -> F:
    previous = (F(1), F(0))
    current = ((model.gamma[0] + model.kappa_in / 2) / model.right[0],
               -width / model.right[0])
    for j in range(1, n - 1):
        gamma, right = model.gamma[j % 2], model.right[j % 2]
        left = model.left[(j - 1) % 2]
        following = ((gamma * current[0] + width * current[1] - left * previous[0]) / right,
                     (gamma * current[1] - width * current[0] - left * previous[1]) / right)
        previous, current = current, following
    gamma, left = model.gamma[(n - 1) % 2] + model.kappa_out / 2, model.left[(n - 2) % 2]
    real = gamma * current[0] + width * current[1] - left * previous[0]
    imaginary = gamma * current[1] - width * current[0] - left * previous[1]
    return model.kappa_in * model.kappa_out / (real**2 + imaginary**2)


def sqrt_lower(value: F, scale: int = 10**6) -> F:
    return F(isqrt(value.numerator * scale**2 // value.denominator), scale)


def floquet_envelope(model: JacobiModel, width: F) -> dict:
    product = model.right[0] * model.right[1]
    tau = (model.gamma[0] * model.gamma[1] / product
           - model.left[0] / model.right[1] - model.left[1] / model.right[0])
    determinant = model.left[0] * model.left[1] / product
    real_min, real_max = tau - width**2 / product, max(abs(tau), abs(tau - width**2 / product))
    imag_max = sum(model.gamma) * width / product
    q = None
    for k in range(1, 1000):
        trial = F(k, 1000)
        if trial**2 <= determinant:
            continue
        ellipse = (real_max / (trial + determinant / trial))**2 + (imag_max / (trial - determinant / trial))**2
        if ellipse < 1:
            q = trial
            break
    if q is None or real_min <= 0:
        raise ValueError("This band has no certified decaying, separated Floquet roots.")
    discriminant_min = real_min**2 - imag_max**2 - 4 * determinant
    if discriminant_min <= 0:
        raise ValueError("A separated-root tail bound is unavailable.")
    separation = sqrt_lower(discriminant_min)
    q_small = 2 * determinant / (real_min + separation)
    h1 = (model.gamma[0] + model.kappa_in / 2 + width) / model.right[0]
    h2 = ((model.gamma[1] + width) * h1 + model.left[0]) / model.right[1]
    h3 = ((model.gamma[0] + width) * h2 + model.left[1] * h1) / model.right[0]
    plus = ((h2 + q_small) / separation, (h3 + q_small * h1) / separation)
    minus = ((h2 + q) / separation, (h3 + q * h1) / separation)
    offdiag_bounds = [sqrt_lower(model.right[j] * model.left[j]) + F(1, 10**6) for j in range(2)]
    stability_margin = min(model.gamma) - sum(offdiag_bounds)
    if stability_margin <= 0 or q_small >= 1:
        raise ValueError("The model does not meet the certified stability conditions.")
    return {"q": q, "q_small": q_small, "separation": separation,
            "ellipse_margin": 1 - ellipse, "plus": plus, "minus": minus,
            "stability_margin": stability_margin}


def profile_bound(envelope: dict, j: int) -> F:
    k, parity = divmod(j, 2)
    return envelope["plus"][parity] * envelope["q"]**k + envelope["minus"][parity] * envelope["q_small"]**k


def tail_bound(model: JacobiModel, envelope: dict, start: int) -> F:
    if start % 2:
        raise ValueError("The tail starts at a complete two-site cell.")
    k = start // 2
    q, small = envelope["q"], envelope["q_small"]
    plus, minus = envelope["plus"], envelope["minus"]
    return (sum(v**2 for v in plus) * q**(2 * k) / (1 - q**2)
            + 2 * sum(plus[j] * minus[j] for j in range(2)) * (q * small)**k / (1 - q * small)
            + sum(v**2 for v in minus) * small**(2 * k) / (1 - small**2)) / model.kappa_in


def uniform_gain_certificate(model: JacobiModel, envelope: dict, width: F, prefix: int) -> dict:
    cutoff = prefix
    while True:
        k = (cutoff - 2) // 2
        maximum_profile = max(envelope["plus"][j] * envelope["q"]**k
                              + envelope["minus"][j] * envelope["q_small"]**k for j in range(2))
        denominator = (max(model.gamma) + model.kappa_out / 2 + width + max(model.left)) * maximum_profile
        lower = model.kappa_in * model.kappa_out / denominator**2
        if lower > F(11, 10):
            break
        cutoff += 2
        if cutoff > 1000:
            raise ValueError("The high-gain tail could not be certified.")
    gains = [endpoint_gain(model, n, width) for n in range(prefix, cutoff)]
    minimum = min([lower] + gains)
    if minimum <= F(11, 10):
        raise ValueError("A finite chain fails the whole-band high-gain condition.")
    return {"tail_start_N": cutoff, "whole_band_gain_lower_all_N": float(minimum),
            "gain_at_prefix": float(endpoint_gain(model, prefix, width)), "finite_chains_checked": len(gains)}
