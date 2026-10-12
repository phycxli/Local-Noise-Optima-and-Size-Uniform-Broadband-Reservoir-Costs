"""Finite-linewidth reservoirs, shared calibration, and resource accounting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigvals

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
MODELS = {
    "uniform": {"gamma": [1.2, 1.2], "right": [1., 1.], "left": [.25, .25]},
    "dimerized": {"gamma": [1.2, 1.15], "right": [1.15, .85], "left": [.15, .35]},
}


def parameters(name, changes=(0., 0., 0.)):
    return {key: np.asarray(MODELS[name][key]) * (1 + changes[j])
            for j, key in enumerate(("gamma", "right", "left"))}


def matrices(p, n):
    parity = np.arange(n) % 2
    x = np.diag(-p["gamma"][parity])
    j = np.arange(n - 1)
    x[j + 1, j], x[j, j + 1] = p["right"][j % 2], p["left"][j % 2]
    return x, x + x.T


def calibration(p, name, n):
    """Positive onsite/bond additions implement Delta M without changing range."""
    _, source = matrices(p, n)
    _, original = matrices(parameters(name), n)
    delta = source - original
    g, loss = np.zeros_like(delta), np.zeros_like(delta)
    diag = np.diag(delta)
    np.fill_diagonal(g, np.maximum(diag, 0))
    np.fill_diagonal(loss, np.maximum(-diag, 0))
    for j in range(n - 1):
        value = delta[j, j + 1]
        c, sign = abs(value) / 2, np.sign(value)
        g[j:j+2, j:j+2] += c * np.array([[1, sign], [sign, 1]])
        loss[j:j+2, j:j+2] += c * np.array([[1, -sign], [-sign, 1]])
    assert np.max(abs(g - loss - delta)) < 2e-14
    return g, loss


def device(p, name, family, radius, n, intrinsic):
    path = ROOT / "inputs" / f"{name}_{family}_r{radius}.npz"
    with np.load(path, allow_pickle=False) as saved:
        prefix_loss = saved["loss"].copy()
    m = len(prefix_loss)
    if family == "finite" and n != m:
        raise ValueError("The finite device cannot be extended without its bridge reserve.")
    g, loss = np.zeros((n, n)), np.zeros((n, n))
    loss[:m, :m] = prefix_loss
    g[:m, :m] = prefix_loss + matrices(parameters(name), m)[1]
    cg, cl = calibration(p, name, m)
    g[:m, :m] += cg
    loss[:m, :m] += cl
    for j in range(m, n):
        loss[j, j] = 2 * p["gamma"][j % 2]
    for j in range(m - 1, n - 1):
        c = (p["right"][j % 2] + p["left"][j % 2]) / 2
        g[j:j+2, j:j+2] += c * np.array([[1, 1], [1, 1]])
        loss[j:j+2, j:j+2] += c * np.array([[1, -1], [-1, 1]])
    g += intrinsic * np.eye(n)
    loss += intrinsic * np.eye(n)
    assert np.max(abs(g - loss - matrices(p, n)[1])) < 3e-14
    return g, loss


def response(p, n, frequencies, linewidth, intrinsic):
    """Adjoint response normalized by input amplitude; no ill-conditioned solve."""
    w = np.asarray(frequencies)
    alpha = 1 / (1 + 2j * w / linewidth)
    ac = alpha.conj()
    h = np.zeros((len(w), n), dtype=complex)
    h[:, 0] = 1.
    for j in range(n - 1):
        parity = j % 2
        r, ell, gamma = p["right"][parity], p["left"][parity], p["gamma"][parity]
        right = (r - ell) / 2 + ac * (r + ell) / 2
        diagonal = intrinsic / 2 + ac * (gamma - intrinsic / 2) - 1j * w
        if j == 0:
            diagonal += CONFIG["port_rates"][0] / 2
            h[:, 1] = diagonal / right
        else:
            pr, pl = p["right"][(j - 1) % 2], p["left"][(j - 1) % 2]
            left = -(pr - pl) / 2 + ac * (pr + pl) / 2
            h[:, j + 1] = (diagonal * h[:, j] - left * h[:, j - 1]) / right
    pr, pl = p["right"][(n - 2) % 2], p["left"][(n - 2) % 2]
    left = -(pr - pl) / 2 + ac * (pr + pl) / 2
    last = intrinsic / 2 + ac * (p["gamma"][(n - 1) % 2] - intrinsic / 2)
    denominator = (last + CONFIG["port_rates"][1] / 2 - 1j * w) * h[:, -1] - left * h[:, -2]
    gain = np.prod(CONFIG["port_rates"]) / abs(denominator)**2
    return h / np.sqrt(CONFIG["port_rates"][0]), alpha, gain


def grams(p, n, width, linewidth, intrinsic, order=None):
    nodes, weights = leggauss(order or CONFIG["quadrature_order"])
    h, alpha, power_gain = response(p, n, width * nodes, linewidth, intrinsic)
    weights = weights / 2
    w = (h.T * weights) @ h.conj()
    z = (h.T * (weights * abs(alpha)**2)) @ h.conj()
    return w.real, z.real, float(min(power_gain)), h, alpha, weights


def noises(g, loss, w, z, intrinsic, occupation):
    return float((occupation + 1) * np.sum(z * g.T)
                 + occupation * np.sum(z * (loss - intrinsic * np.eye(len(g))).T)
                 + occupation * intrinsic * np.trace(w))


def stability(p, n, linewidth, intrinsic):
    """Spectrum of the exact Schur reduction; other auxiliary roots are -kappa/2."""
    x, source = matrices(p, n)
    c = (x - x.T) / 2
    # Diagonal similarity avoids the exponentially ill-conditioned raw chain.
    ratio = np.sqrt(p["right"][np.arange(n - 1) % 2] / p["left"][np.arange(n - 1) % 2])
    for j in range(n - 1):
        c[j + 1, j] /= ratio[j]
        c[j, j + 1] *= ratio[j]
        source[j + 1, j] /= ratio[j]
        source[j, j + 1] *= ratio[j]
    ports = np.zeros(n)
    ports[0], ports[-1] = CONFIG["port_rates"]
    a0 = c - np.diag((intrinsic + ports) / 2)
    k = (source + intrinsic * np.eye(n)) / 2
    reduced = np.block([[a0, linewidth / 2 * np.eye(n)], [k, -linewidth / 2 * np.eye(n)]])
    return float(max(np.max(eigvals(reduced).real), -linewidth / 2))


def stability_sufficient_bounds(relative_halfspan, minimum_linewidth, maximum_intrinsic, name="uniform"):
    p = parameters(name)
    lo, hi = 1 - relative_halfspan, 1 + relative_halfspan
    gamma_min, gamma_max = min(p["gamma"]) * lo, max(p["gamma"]) * hi
    coupling_sum = sum(np.sqrt(p["right"] * p["left"])) * hi
    mu = gamma_min - coupling_sum
    a = gamma_max + max(CONFIG["port_rates"]) / 2 + coupling_sum
    rho_hi = np.sqrt(p["right"] * hi / (p["left"] * lo))
    rho_lo = np.sqrt(p["right"] * lo / (p["left"] * hi))
    bond_hi = (p["right"] + p["left"]) * hi / 2
    # Both absolute row and column sums are bounded by this common value.
    b = gamma_max + maximum_intrinsic / 2 + max(bond_hi * rho_hi) + max(bond_hi / rho_lo)
    margin = mu * (minimum_linewidth / 2 - b) - b * a
    return {"slow_decay_lower": float(mu), "slow_norm_upper": float(a), "B_norm_upper": float(b),
            "energy_inequality_margin": float(margin), "sufficient_for_all_finite_N": bool(mu > 0 and margin > 0)}


def local_lower(p, z, w, intrinsic, occupation, n, include_boundary):
    m, cap = CONFIG["prefix"], CONFIG["complete_rate_cap"]
    with np.load(ROOT / "inputs" / f"{n}_physical_dual.npz", allow_pickle=False) as archive:
        factors = [archive[f"dual_factor_{j}"] for j in range(4)]
    a, b, c, d = [f @ f.T for f in factors]
    source = matrices(p, m)[1] + intrinsic * np.eye(m)
    zp, wp = z[:m, :m], w[:m, :m]
    mask = abs(np.arange(m)[:, None] - np.arange(m)) <= 1
    e = (zp - a - b + c + d) * mask
    penalty = np.maximum(-np.diag(e), 0).sum() + np.abs(np.diag(e, 1)).sum()
    f_lower = np.sum((d - b) * source.T) - (cap - intrinsic) * np.trace(c) - cap * np.trace(d)
    f_lower -= (cap - intrinsic) * penalty
    value = ((1 + 2 * occupation) * f_lower + (occupation + 1) * np.sum(zp * source.T)
             + occupation * intrinsic * np.trace(wp))
    if include_boundary:
        value -= ((occupation + 1) * cap + occupation * (cap - intrinsic)) * abs(z[m - 1, m])
    # Floating quadrature and factor products are independently cross-checked.
    return float(max(0., value - 2e-7 * (1 + np.linalg.norm(zp, 1))))
