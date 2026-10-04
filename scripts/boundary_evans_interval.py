"""Outward-rounded interval enclosures for the two-band boundary Evans data.

Root centres from NumPy only propose disks. A strict Taylor/Rouche inequality
certifies one root in each disk for every frequency in the supplied box.
The boundary residues and their derivatives are then interval evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mpmath as mp
import numpy as np

from boundary_evans import modal_data

iv = mp.iv
iv.dps = 45
PAIRS = tuple(combinations(range(4), 2))
KAPPA = iv.mpf("0.2")
J = iv.mpf("0.01")
TR = (iv.mpf(1), iv.mpf("0.8"))
TL = (iv.mpf("0.25"), iv.mpf("0.35"))
GAMMA = (iv.mpf("1.2"), iv.mpf("1.15"))
I = iv.mpc(0, 1)


def interval(lo, hi=None):
    return iv.mpf([str(lo), str(lo if hi is None else hi)])


def complex_box(re_lo, re_hi, im_lo=0, im_hi=0):
    return iv.mpc(interval(re_lo, re_hi), interval(im_lo, im_hi))


def upper(value):
    return value.b


def lower(value):
    return value.a


def float_up(value):
    return float(np.nextafter(float(upper(value)), np.inf))


def float_down(value):
    return float(np.nextafter(float(lower(value)), -np.inf))


def conjugate(value):
    return iv.mpc(value.real, -value.imag)


@dataclass
class Jet:
    value: object
    derivative: object = 0

    @staticmethod
    def coerce(other):
        return other if isinstance(other, Jet) else Jet(other, 0)

    def __add__(self, other):
        other = self.coerce(other)
        return Jet(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        return Jet(self.value * other.value,
                   self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        return Jet(self.value / other.value,
                   (self.derivative * other.value - self.value * other.derivative)
                   / (other.value * other.value))

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def conj(self):
        return Jet(conjugate(self.value), conjugate(self.derivative))

    def real(self):
        return Jet(self.value.real, self.derivative.real)


def mm(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(2)), Jet(0))
             for j in range(2)] for i in range(2)]


def polynomial_taylor(c, z, sharp=False):
    sign = -1 if sharp else 1
    polynomials = []
    for a in range(2):
        onsite = GAMMA[a] - sign * I * z
        polynomials.append([TR[a] * c * c - onsite * c + TL[a],
                            2 * TR[a] * c - onsite, TR[a]])
    coefficients = [iv.mpc(0) for _ in range(5)]
    for i, pi in enumerate(polynomials[0]):
        for j, pj in enumerate(polynomials[1]):
            coefficients[i + j] += pi * pj
    coefficients[0] += J * J * c * c
    coefficients[1] += 2 * J * J * c
    coefficients[2] += J * J
    return coefficients


def certified_root_boxes(z, sharp=False):
    midpoint = complex((float(z.real.a) + float(z.real.b)) / 2,
                       (float(z.imag.a) + float(z.imag.b)) / 2)
    centres = modal_data(midpoint, sharp=sharp).roots
    boxes, radii = [], []
    min_margin = iv.mpf(1)
    for centre in centres:
        cr, ci = repr(float(centre.real)), repr(float(centre.imag))
        c = iv.mpc(interval(cr), interval(ci))
        coefficients = polynomial_taylor(c, z, sharp)
        slope = lower(abs(coefficients[1]))
        if not slope > 0:
            raise ArithmeticError("Taylor linear coefficient includes zero")
        estimate = float_up(2 * upper(abs(coefficients[0])) / slope)
        radius = max(estimate, 1e-12)
        for _ in range(12):
            r = interval(repr(radius))
            remainder = upper(abs(coefficients[0])) + sum(
                (upper(abs(coefficients[k])) * r ** k for k in range(2, 5)),
                iv.mpf(0))
            linear = slope * r
            if remainder < linear:
                min_margin = min(min_margin, lower((linear - remainder) / linear))
                break
            radius *= 2
        else:
            raise ArithmeticError("No certified root disk; refine the frequency box")
        cr_iv, ci_iv = interval(cr), interval(ci)
        box = iv.mpc(cr_iv + interval(-radius, radius),
                     ci_iv + interval(-radius, radius))
        boxes.append(box)
        radii.append(radius)
    for k, ell in PAIRS:
        ck = iv.mpc(interval(repr(float(centres[k].real))),
                    interval(repr(float(centres[k].imag))))
        cl = iv.mpc(interval(repr(float(centres[ell].real))),
                    interval(repr(float(centres[ell].imag))))
        if not lower(abs(ck - cl)) > interval(radii[k]) + interval(radii[ell]):
            raise ArithmeticError("Root disks overlap")
    if not max(upper(abs(x)) for x in boxes[:2]) < lower(abs(boxes[2])):
        raise ArithmeticError("Inner/active labels are not certified")
    if not upper(abs(boxes[2])) < lower(abs(boxes[3])):
        raise ArithmeticError("Active/passive labels are not certified")
    return boxes, min_margin


@dataclass
class ModalIntervals:
    roots: list[Jet]
    a: list[Jet]
    b: list[Jet]
    d: list[Jet]
    disk_margin: object
    factors: list


def certified_modal_data(z, sharp=False):
    root_boxes, margin = certified_root_boxes(z, sharp)
    sign = -1 if sharp else 1
    zj = Jet(z, 1)
    roots, factors = [], []
    for root in root_boxes:
        p = [TR[a] * root * root - (GAMMA[a] - sign * I * z) * root + TL[a]
             for a in range(2)]
        big = int(float_up(abs(p[1])) > float_up(abs(p[0])))
        target = 1 - big
        if not lower(abs(p[big])) > 0:
            raise ArithmeticError("Null-vector pivot includes zero")
        onsite_big = GAMMA[big] - sign * I * z
        denominator = (2 * TR[target] * root - GAMMA[target] + sign * I * z
                       + J * J * root * (2 * TL[big] - onsite_big * root)
                       / (p[big] * p[big]))
        if not lower(abs(denominator)) > 0:
            raise ArithmeticError("Reduced P_lambda includes zero")
        rootj = Jet(root, -sign * I * root * (1 - J * J * root * root / (p[big] * p[big]))
                    / denominator)
        roots.append(rootj)
        pj = [rootj * TR[a] * rootj - (Jet(GAMMA[a]) - zj * (sign * I)) * rootj + TL[a]
              for a in range(2)]
        # At a certified root adj(L) is rank one. Enforcing P=0 avoids
        # dependency inflation from separately evaluating its tiny diagonal.
        if not lower(abs(pj[big].value)) > 0:
            raise ArithmeticError("Null-vector pivot includes zero")
        pivot = pj[big]
        pj[1 - big] = -(rootj * rootj * (J * J)) / pivot
        off_ratio = rootj * (-sign * I * J) / pivot
        u = [off_ratio, Jet(1)] if big == 0 else [Jet(1), off_ratio]
        h = [(rootj * TR[0] + KAPPA / 2) * u[0], rootj * TR[1] * u[1]]
        denj = (rootj * (2 * TR[target]) - GAMMA[target] + zj * (sign * I)
                + rootj * (J * J) * (Jet(2 * TL[big])
                  - (Jet(GAMMA[big]) - zj * (sign * I)) * rootj) / (pivot * pivot))
        factor = rootj / denj
        factors.append((factor, u, h))
    a = [factor * h[1] * h[1] for factor, _, h in factors]
    b, d = [], []
    for k, ell in PAIRS:
        fk, uk, hk = factors[k]
        fl, ul, hl = factors[ell]
        wedge = hk[0] * hl[1] - hl[0] * hk[1]
        d.append(fk * fl * wedge * wedge)
        b.append(fk * fl * wedge * (uk[0] * hl[1] - ul[0] * hk[1]))
    return ModalIntervals(roots, a, b, d, margin, factors)


def remainder_bound(coefficients, ratios, m0):
    error, derivative_error, max_ratio = iv.mpf(0), iv.mpf(0), iv.mpf(0)
    for coefficient, ratio in zip(coefficients, ratios):
        q = upper(abs(ratio.value))
        if not q * interval(m0 + 1) / interval(m0) < 1:
            raise ArithmeticError("m*q^(m-1) is not uniformly decreasing at this cutoff")
        error += upper(abs(coefficient.value)) * q ** m0
        derivative_error += (upper(abs(coefficient.derivative)) * q ** m0
                             + upper(abs(coefficient.value)) * m0
                             * upper(abs(ratio.derivative)) * q ** (m0 - 1))
        max_ratio = max(max_ratio, q)
    return upper(error), upper(derivative_error), max_ratio


def expand_jet(jet, error, derivative_error):
    def ball(radius):
        r = float_up(radius)
        return complex_box(-r, r, -r, r)
    return Jet(jet.value + ball(error), jet.derivative + ball(derivative_error))


def scaled_enclosures(data, m0):
    passive, active = data.roots[3], data.roots[2]
    aratios = [root / passive for root in data.roots[:3]]
    pratios = [(data.roots[k] * data.roots[ell]) / (active * passive)
               for k, ell in PAIRS[:-1]]
    ea, eap, qa = remainder_bound(data.a[:3], aratios, m0)
    eb, ebp, qb = remainder_bound(data.b[:-1], pratios, m0)
    ed, edp, _ = remainder_bound(data.d[:-1], pratios, m0)
    a = expand_jet(data.a[3], ea, eap)
    b = expand_jet(data.b[-1], eb, ebp)
    d = expand_jet(data.d[-1], ed, edp)
    if not lower(abs(data.a[3].value)) > ea:
        raise ArithmeticError("Scaled a_N may vanish")
    return a, b, d, (ea, eb, ed), max(qa, qb), (eap, ebp, edp)


def absolute_square(jet):
    return Jet(abs(jet.value) ** 2,
               2 * (conjugate(jet.value) * jet.derivative).real)


def real_coefficient(a, b, d):
    return ((d.conj() * b).real() * 2 - absolute_square(b) * KAPPA) / absolute_square(a)


def analytic_coefficient(a, b, d, ash, bsh, dsh):
    return (dsh * b + bsh * d - bsh * b * KAPPA) / (ash * a)


def leading_components(data):
    fa, ua, ha = data.factors[2]
    _, up, hp = data.factors[3]
    wedge = ha[0] * hp[1] - hp[0] * ha[1]
    v = ua[0] * hp[1] - up[0] * ha[1]
    return fa, wedge, v, hp[1]


def real_leading_parts(data):
    fa, w, v, hp = leading_components(data)
    hp2 = absolute_square(hp)
    scale = absolute_square(fa) * absolute_square(w) / (hp2 * hp2)
    return scale * w.conj() * v, scale * absolute_square(v)


def complex_leading_coefficient(data, sh):
    fa, w, v, hp = leading_components(data)
    fas, ws, vs, hps = leading_components(sh)
    denominator = hps * hp
    return (fas * fa * ws * w * (ws * v + vs * w - vs * v * KAPPA)
            / (denominator * denominator))


def real_band_bounds(z, m0):
    data = certified_modal_data(z)
    _, _, _, errors, q, derivative_errors = scaled_enclosures(data, m0)
    normalised = [Jet(1) + expand_jet(Jet(0), e, ep) / leading
                  for leading, e, ep in zip((data.a[3], data.b[-1], data.d[-1]),
                                            errors, derivative_errors)]
    an, bn, dn = normalised
    zlead, hlead = real_leading_parts(data)
    cinf = zlead.real() * 2 - hlead * KAPPA
    cn = ((zlead * dn.conj() * bn).real() * 2
          - hlead * absolute_square(bn) * KAPPA) / absolute_square(an)
    tau = abs(data.roots[2].value) ** 2
    log_tau_prime = 2 * (data.roots[2].derivative / data.roots[2].value).real
    if not lower(cn.value) > 0 or not lower(cinf.value) > 0:
        raise ArithmeticError("C_N or C_infinity positivity is not certified")
    return {"tau": tau, "log_tau_prime": log_tau_prime,
            "C_N": cn.value, "C_infinity": cinf.value,
            "log_C_N_prime": cn.derivative / cn.value,
            "log_C_infinity_prime": cinf.derivative / cinf.value,
            "q": q, "a_error": errors[0], "disk_margin": data.disk_margin}


def coefficient_error(data, sharp_data, errors, sharp_errors, cinf):
    ea, eb, ed = errors
    eas, ebs, eds = sharp_errors
    a, b, d = [upper(abs(x.value)) for x in (data.a[3], data.b[-1], data.d[-1])]
    ash, bsh, dsh = [upper(abs(x.value)) for x in
                     (sharp_data.a[3], sharp_data.b[-1], sharp_data.d[-1])]
    en = (dsh * eb + b * eds + eds * eb + bsh * ed + d * ebs + ebs * ed
          + KAPPA * (bsh * eb + b * ebs + ebs * eb))
    eden = ash * ea + a * eas + eas * ea
    denlow = ((lower(abs(data.a[3].value)) - ea)
              * (lower(abs(sharp_data.a[3].value)) - eas))
    if not denlow > 0 or not lower(abs(cinf.value)) > 0:
        raise ArithmeticError("Analytic boundary factors may vanish")
    error = (en + upper(abs(cinf.value)) * eden) / denlow
    return upper(error / lower(abs(cinf.value)))


def complex_band_bounds(z, m0):
    data, sh = certified_modal_data(z), certified_modal_data(z, sharp=True)
    _, _, _, errors, q, _ = scaled_enclosures(data, m0)
    _, _, _, sharp_errors, qs, _ = scaled_enclosures(sh, m0)
    cinf = complex_leading_coefficient(data, sh)
    lp = (data.roots[2].derivative / data.roots[2].value
          + sh.roots[2].derivative / sh.roots[2].value)
    if not lower(abs(cinf.value)) > 0:
        raise ArithmeticError("C_infinity includes zero")
    return {"abs_log_tau_prime": abs(lp),
            "abs_log_C_infinity_prime": abs(cinf.derivative / cinf.value),
            "relative_C_error": coefficient_error(data, sh, errors, sharp_errors, cinf),
            "abs_C_infinity": abs(cinf.value), "q": max(q, qs),
            "disk_margin": min(data.disk_margin, sh.disk_margin)}
