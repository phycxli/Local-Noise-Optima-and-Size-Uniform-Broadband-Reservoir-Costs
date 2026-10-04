"""Boundary-only analytic crossing numerator and its exact modal expansion.

A(z)=-i*z-X_kappa^dagger. The sharp operation is coefficient conjugation:
f^sharp(z)=conj(f(conj(z))). The largest two transfer roots are labelled
active/passive only where their moduli are separated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

TR = np.diag([1.0, 0.8]).astype(complex)
TL = np.diag([0.25, 0.35]).astype(complex)
GAMMA = np.diag([1.2, 1.15]).astype(complex)
KAPPA = 0.2
PORT = np.array([1.0, 0.0], dtype=complex)
PAIRS = tuple(combinations(range(4), 2))


@dataclass
class ModalData:
    roots: np.ndarray
    a: np.ndarray
    b: np.ndarray
    d: np.ndarray
    condition: float


def onsite(z: complex, mixing: float = 0.01, sharp: bool = False):
    sign = 1.0 if sharp else -1.0
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    bulk = GAMMA + sign * 1j * (z * np.eye(2) + mixing * sx)
    return bulk, bulk + KAPPA / 2.0 * np.outer(PORT, PORT)


def boundary_values(z: complex, n: int, mixing: float = 0.01, sharp=False):
    if n < 2:
        raise ValueError("N must be at least 2")
    bulk, edge = onsite(z, mixing, sharp)
    previous, current = np.zeros((2, 2), dtype=complex), np.eye(2, dtype=complex)
    for cell in range(n - 1):
        nxt = np.linalg.solve(TR, (edge if cell == 0 else bulk) @ current - TL @ previous)
        previous, current = current, nxt
    q = edge @ current - TL @ previous
    r = np.array([q[1, 1], -q[1, 0]], dtype=complex)
    return q[1, 1], (current @ r)[0], np.linalg.det(q)


def crossing_numerator(z: complex, n: int, mixing: float = 0.01):
    a, b, d = boundary_values(z, n, mixing)
    ash, bsh, dsh = boundary_values(z, n, mixing, sharp=True)
    return KAPPA * (ash * a + bsh * b) - dsh * b - bsh * d


def modal_data(z: complex, mixing: float = 0.01, sharp=False) -> ModalData:
    bulk, edge = onsite(z, mixing, sharp)
    transfer = np.block([[np.linalg.solve(TR, bulk), -np.linalg.solve(TR, TL)],
                         [np.eye(2), np.zeros((2, 2), dtype=complex)]])
    roots, right = np.linalg.eig(transfer)
    order = np.argsort(np.abs(roots))
    roots, right = roots[order], right[:, order]
    left = np.linalg.inv(right)
    initial = np.vstack([np.linalg.solve(TR, edge), np.eye(2)])
    terminal = np.hstack([edge, -TL])
    top = np.hstack([np.eye(2), np.zeros((2, 2), dtype=complex)])
    qm, em = [], []
    for k in range(4):
        pk = np.outer(right[:, k], left[k])
        qm.append(terminal @ pk @ initial)
        em.append(top @ pk @ initial)
    a = np.array([q[1, 1] for q in qm])
    b, d = [], []
    for k, ell in PAIRS:
        qk, ql = qm[k], qm[ell]
        d.append(qk[0, 0] * ql[1, 1] + ql[0, 0] * qk[1, 1]
                 - qk[0, 1] * ql[1, 0] - ql[0, 1] * qk[1, 0])
        rk, rl = np.array([qk[1, 1], -qk[1, 0]]), np.array([ql[1, 1], -ql[1, 0]])
        b.append((em[k] @ rl + em[ell] @ rk)[0])
    return ModalData(roots, a, np.array(b), np.array(d), float(np.linalg.cond(right)))


def modal_boundary_values(data: ModalData, n: int):
    m = n - 2
    powers = data.roots ** m
    products = np.array([(data.roots[k] * data.roots[ell]) ** m for k, ell in PAIRS])
    return data.a @ powers, data.b @ products, data.d @ products


def scaled_modes(data: ModalData, n: int):
    m = n - 2
    active, passive = data.roots[2:]
    a = data.a @ (data.roots / passive) ** m
    ratios = np.array([data.roots[k] * data.roots[ell] / (active * passive)
                       for k, ell in PAIRS])
    products = ratios ** m
    return a, data.b @ products, data.d @ products


def coefficient_and_phase(z: complex, n: int, mixing: float = 0.01):
    data, sh0 = modal_data(z, mixing), modal_data(np.conjugate(z), mixing)
    sh = ModalData(np.conjugate(sh0.roots), np.conjugate(sh0.a),
                   np.conjugate(sh0.b), np.conjugate(sh0.d), sh0.condition)
    a, b, d = scaled_modes(data, n)
    ash, bsh, dsh = scaled_modes(sh, n)
    c = (dsh * b + bsh * d - KAPPA * bsh * b) / (ash * a)
    tau = data.roots[2] * sh.roots[2]
    return c, tau, data, sh


def limiting_coefficient(z: complex, mixing: float = 0.01):
    data, sh0 = modal_data(z, mixing), modal_data(np.conjugate(z), mixing)
    a, b, d = data.a[-1], data.b[-1], data.d[-1]
    ash, bsh, dsh = np.conjugate(sh0.a[-1]), np.conjugate(sh0.b[-1]), np.conjugate(sh0.d[-1])
    c = (dsh * b + bsh * d - KAPPA * bsh * b) / (ash * a)
    tau = data.roots[2] * np.conjugate(sh0.roots[2])
    return c, tau


def real_log_mismatch(omega: float, n: int, mixing: float = 0.01):
    c, tau, _, _ = coefficient_and_phase(complex(omega), n, mixing)
    if c.real <= 0.0 or abs(c.imag) > 1e-7 * c.real:
        raise ValueError(f"C_N is not positive at N={n}, omega={omega}: {c}")
    return float((n - 2) * np.log(tau.real) + np.log(c.real / KAPPA))
