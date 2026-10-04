"""Independent convention and short-chain algebra checks for the final review.

The monomial Gram uses a different basis from the certificate generator.
Finite Fock matrices check operator identities on an untruncated low sector;
they are not a performance or all-length proof.
"""

from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import MODELS, exact_gram, legendre_coefficients, source_matrix


def monomial_gram(model, n, width):
    coefficients = [[F(0) for _ in range(n)] for _ in range(n)]
    coefficients[0][0] = F(1)
    coefficients[1][0] = (model.gamma[0] + model.kappa_in / 2) / model.right[0]
    coefficients[1][1] = 1 / model.right[0]
    for j in range(1, n - 1):
        for k in range(j + 2):
            coefficients[j + 1][k] = (
                model.gamma[j % 2] * coefficients[j][k]
                - model.left[(j - 1) % 2] * coefficients[j - 1][k]
                + (coefficients[j][k - 1] if k else F(0))) / model.right[j % 2]
    result = np.full((n, n), F(0), dtype=object)
    for i in range(n):
        for j in range(n):
            result[i, j] = sum((
                (-1 if ((ell - k) // 2) % 2 else 1) * coefficients[i][k]
                * coefficients[j][ell] * width**(k + ell) / (k + ell + 1)
                for k in range(i + 1) for ell in range(j + 1) if (k + ell) % 2 == 0
            ), F(0)) / model.kappa_in
    return result


def trace(a, b):
    return sum((a[i, j] * b[j, i] for i in range(len(a)) for j in range(len(a))), F(0))


def gram_and_lifting_checks():
    cases = []
    for name, model in MODELS.items():
        for width in (F(1, 20), F(7, 100)):
            n, m, rate = 9, 4, F(8)
            w = monomial_gram(model, n, width)
            assert np.array_equal(w, exact_gram(model, legendre_coefficients(model, n, width), n))
            assert np.array_equal(w[:m, :m], monomial_gram(model, m, width))
            source = source_matrix(model, n)
            loss = (rate * np.eye(n, dtype=object) - source) / 2
            gain = loss + source
            for radius in (1, 2):
                mask = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :]) <= radius
                factors = [np.array([[F((i + 2 * j + k) % 5 - 2, 17)
                                      for j in range(2)] for i in range(m)], dtype=object)
                           for k in range(4)]
                a, b, c, d = [factor @ factor.T for factor in factors]
                residual = (w[:m, :m] - a - b + c + d) * mask
                alpha = trace(w[:m, :m], source[:m, :m]) + trace(d - b, source[:m, :m])
                penalty = sum((max(-residual[i, i], F(0)) for i in range(m)), F(0))
                penalty += sum((abs(residual[i, j]) for i in range(m) for j in range(i + 1, m)), F(0))
                dual_rate = sum((c[i, i] + d[i, i] for i in range(m)), F(0))
                lp, gp, eye = loss[:m, :m], gain[:m, :m], rate * np.eye(m, dtype=object)
                slack_terms = trace(a, lp) + trace(b, gp) + trace(c, eye - lp) + trace(d, eye - gp)
                assert trace(w[:m, :m], gp) == alpha - rate * dual_rate + trace(residual, lp) + slack_terms
                boundary = sum((abs(w[i, j]) for i in range(m) for j in range(m, n)
                                if j - i <= radius), F(0))
                assert trace(w, gain) >= trace(w[:m, :m], gp) - rate * boundary
                assert trace(w, gain) >= alpha - rate * (dual_rate + penalty + boundary)
                cases.append({"model": name, "halfwidth": str(width), "N": n, "prefix": m,
                              "range": radius, "exact_monomial_gram_agreement": True,
                              "exact_dual_identity": True, "crossing_bound_passed": True})
    return cases


def adjacent_cell_factors(matrix, block_size):
    remainder = matrix.astype(complex).copy()
    factors = []
    n = len(matrix)
    tolerance = 2e-10 * max(1., np.linalg.norm(matrix, 2))
    for start in range(0, n, block_size):
        stop = start + block_size
        leading = remainder[start:stop, start:stop]
        values, vectors = np.linalg.eigh((leading + leading.conj().T) / 2)
        assert values.min() >= -tolerance
        live = values > 1e-11 * max(1., values.max())
        basis, roots = vectors[:, live], np.sqrt(values[live])
        factor = np.zeros((n, len(roots)), dtype=complex)
        factor[start:stop] = basis * roots
        if stop < n:
            coupling = remainder[start:stop, stop:stop + block_size]
            assert np.linalg.norm(coupling - basis @ (basis.conj().T @ coupling)) < tolerance
            factor[stop:stop + block_size] = (coupling.conj().T @ basis) / roots
            assert np.linalg.norm(remainder[start:stop, stop + block_size:]) < tolerance
        remainder -= factor @ factor.conj().T
        factors.append(factor)
    error = float(np.linalg.norm(remainder, 2))
    assert error < tolerance
    return factors, error


def local_kernel_checks():
    # Complex blocks exercise phases, both signs, and singular local factorizations.
    rng = np.random.default_rng(20261004)
    cases = []
    for d in (1, 2, 3):
        cells, n = 4, 4 * d
        blocks = rng.normal(size=(cells, d)) + 1j * rng.normal(size=(cells, d))
        blocks *= np.array([0.7, 1.2, 0.9, 1.1])[:, None]
        norms = np.linalg.norm(blocks, axis=1)
        directions = blocks / norms[:, None]
        y = blocks.ravel()
        u = y / np.linalg.norm(y)
        complement = np.linalg.svd(u.conj()[None, :], full_matrices=True)[2].conj().T[:, 1:]
        kernel = np.zeros((n, n), dtype=complex)
        for j in range(cells):
            sl = slice(j * d, (j + 1) * d)
            kernel[sl, sl] = np.eye(d) - np.outer(directions[j], directions[j].conj())
        for j in range(cells - 1):
            bond = np.zeros(n, dtype=complex)
            bond[j * d:(j + 1) * d] = np.sqrt(norms[j + 1] / norms[j]) * directions[j]
            bond[(j + 1) * d:(j + 2) * d] = -np.sqrt(norms[j] / norms[j + 1]) * directions[j + 1]
            kernel += np.outer(bond, bond.conj())
        kernel_error = float(np.linalg.norm(kernel @ y))
        assert kernel_error < 1e-12
        k0 = np.linalg.eigvalsh(complement.conj().T @ kernel @ complement).min()
        assert k0 > 0
        base = np.zeros((n, n), dtype=complex)
        for j in range(cells):
            sl = slice(j * d, (j + 1) * d)
            diagonal = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            base[sl, sl] = (diagonal + diagonal.conj().T) / 2
            if j + 1 < cells:
                next_sl = slice((j + 1) * d, (j + 2) * d)
                bond = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
                base[sl, next_sl], base[next_sl, sl] = bond, bond.conj().T
        for sign in (1, -1):
            source = base + (sign * 0.3 - np.vdot(u, base @ u).real) * np.eye(n)
            signed_source = sign * source
            transverse = complement.conj().T @ signed_source @ complement
            coupling = complement.conj().T @ signed_source @ u
            s = np.vdot(u, signed_source @ u).real
            alpha = (np.linalg.norm(transverse, 2) + np.linalg.norm(coupling)**2 / s + 1) / k0
            if sign > 0:
                loss, gain = alpha * kernel, source + alpha * kernel
            else:
                gain, loss = alpha * kernel, -source + alpha * kernel
            spectrum_minimum = min(np.linalg.eigvalsh(gain).min(), np.linalg.eigvalsh(loss).min())
            assert spectrum_minimum > -1e-10
            drift_error = float(np.linalg.norm(gain - loss - source, 2))
            floor = max(float(np.vdot(y, source @ y).real), 0.)
            floor_error = float(abs(np.vdot(y, gain @ y).real - floor))
            factors = [adjacent_cell_factors(matrix, d) for matrix in (gain, loss)]
            assert max(drift_error, floor_error) < 1e-9
            cases.append({"cell_modes": d, "cells": cells, "sign": sign,
                          "kernel_residual": kernel_error, "transverse_kernel_minimum": float(k0),
                          "finite_alpha": float(alpha), "minimum_gram_eigenvalue": float(spectrum_minimum),
                          "drift_residual": drift_error, "pointwise_floor_residual": floor_error,
                          "adjacent_factorization_residual": max(item[1] for item in factors),
                          "scope": "complex finite matrix construction; not a shared-band or all-length proof"})
    return cases


def exact_extension_checks():
    cases = []
    m, rate = 4, F(8)
    for name, model in MODELS.items():
        bonds = [model.right[j] + model.left[j] for j in range(2)]
        bridge = bonds[(m - 1) % 2]
        cap = rate - bridge
        assert 2 * max(model.gamma) + sum(bonds) <= cap
        prefix_source = source_matrix(model, m)
        prefix_loss = (cap * np.eye(m, dtype=object) - prefix_source) / 2
        prefix_gain = prefix_loss + prefix_source
        assert all(sum(abs(v) for v in row) <= cap for row in prefix_source)
        for n in (m, m + 1, 9, 12):
            gain, loss = [np.full((n, n), F(0), dtype=object) for _ in range(2)]
            gain[:m, :m], loss[:m, :m] = prefix_gain, prefix_loss
            for j in range(m, n):
                loss[j, j] = 2 * model.gamma[j % 2]
            for j in range(m - 1, n - 1):
                half = bonds[j % 2] / 2
                gain[j:j + 2, j:j + 2] += half * np.array([[1, 1], [1, 1]], dtype=object)
                loss[j:j + 2, j:j + 2] += half * np.array([[1, -1], [-1, 1]], dtype=object)
            assert np.array_equal(gain - loss, source_matrix(model, n))
            for matrix in (gain, loss):
                assert all(matrix[j, j] >= sum(abs(matrix[j, k]) for k in range(n) if k != j)
                           for j in range(n))
                assert all(sum(abs(v) for v in row) <= rate for row in matrix)
                assert all(matrix[j, k] == 0 for j in range(n) for k in range(n) if abs(j - k) > 1)
            cases.append({"model": name, "prefix": m, "N": n, "bridge_rate": str(bridge),
                          "reserved_prefix_cap": str(cap), "exact_drift_and_cap_checks": True,
                          "scope": "rational feasible prefix; optimized saved prefixes use the resource checker"})
    return cases


def fock_conventions():
    cutoff = 5
    single = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    annihilators = [np.kron(single, np.eye(cutoff)), np.kron(np.eye(cutoff), single)]
    h = np.array([[0.2, 0.1j], [-0.1j, -0.1]])
    gains = [np.array([0.25, 0.2j])]
    losses = [np.array([1.0, 0.1 + 0.2j]), np.array([0.3j, 1.2])]
    gg = sum(np.outer(v, v.conj()) for v in gains)
    gl = sum(np.outer(v, v.conj()) for v in losses)
    drift = -1j * h + (gg - gl) / 2
    assert np.linalg.eigvals(drift).real.max() < 0
    ham = sum(h[i, j] * annihilators[i].conj().T @ annihilators[j]
              for i in range(2) for j in range(2))
    jumps = [sum(v[i] * annihilators[i].conj().T for i in range(2)) for v in gains]
    jumps += [sum(v[i].conjugate() * annihilators[i] for i in range(2)) for v in losses]

    def adjoint(operator):
        result = 1j * (ham @ operator - operator @ ham)
        for jump in jumps:
            jj = jump.conj().T @ jump
            result += jump.conj().T @ operator @ jump - (jj @ operator + operator @ jj) / 2
        return result

    low = [a * cutoff + b for a in range(cutoff - 2) for b in range(cutoff - 2)]

    def error(operator):
        return float(np.max(np.abs(operator[np.ix_(low, low)])))

    first = max(error(adjoint(annihilators[i]) - sum(drift[i, j] * annihilators[j]
                                                  for j in range(2))) for i in range(2))
    cov = [[annihilators[j].conj().T @ annihilators[i] for j in range(2)] for i in range(2)]
    second = max(error(adjoint(cov[i][j])
                       - sum(drift[i, k] * cov[k][j] + drift[j, k].conjugate() * cov[i][k]
                             for k in range(2)) - gg[i, j] * np.eye(cutoff**2))
                 for i in range(2) for j in range(2))
    assert max(first, second) < 1e-12
    return {"complex_jump_first_moment_residual": first, "complex_covariance_residual": second,
            "cutoff": cutoff, "low_sector_dimensions": len(low),
            "scope": "operator identities in the exact low occupation sector, not all-length dynamics"}


def quadrature_convention():
    # One mode gives an independent vacuum normalization check.
    eta, internal_loss, port = 0.2, 1.0, 0.3
    chi = 2 / (internal_loss + port - eta)
    s = np.array([1 - port * chi, -np.sqrt(port * internal_loss) * chi])
    t = np.sqrt(port * eta) * chi
    photon_noise = t**2
    quadrature_noise = float(s @ s + t**2)
    assert abs(float(s @ s - t**2) - 1) < 1e-12
    assert abs(quadrature_noise - (1 + 2 * photon_noise)) < 1e-12
    return {"output_photon_noise": photon_noise, "vacuum_normalized_quadrature_noise": quadrature_noise,
            "commutator_residual": abs(float(s @ s - t**2) - 1),
            "correct_relation": "S_quad=1+2*S_photon at a symmetric spectrum",
            "legacy_half_factor_error": photon_noise}


def main():
    result = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "independent_exact_checks": gram_and_lifting_checks(),
              "proposition_S5_complex_kernel_checks": local_kernel_checks(),
              "proposition_S17_exact_extension_checks": exact_extension_checks(),
              "fock_operator_checks": fock_conventions(),
              "quadrature_check": quadrature_convention(),
              "all_checks_passed": True,
              "reviewer": "Automated independent algebra and numerical checks",
              "is_human_author_verification": False,
              "scope": "S5 complex kernels; S16 exact Gram/dual/lifting identities; S17 exact extensions; full results use saved rational certificates"}
    path = ROOT / "results/final_prl_proof_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
