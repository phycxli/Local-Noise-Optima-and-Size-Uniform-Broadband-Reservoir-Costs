"""Independent floating-point audits of the exact boundary/modal identities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from audit_boundary_argument_principle_transfer import profile
from audit_multiband_crossing_boundary import drift
from boundary_evans import (KAPPA, boundary_values, coefficient_and_phase,
                            crossing_numerator, limiting_coefficient, modal_boundary_values,
                            modal_data, scaled_modes)
from boundary_evans_interval import (certified_modal_data, complex_box,
                                     complex_leading_coefficient, float_down, float_up)

ROOT = Path(__file__).resolve().parents[1]


def midpoint(value):
    return complex((float(value.real.a) + float(value.real.b)) / 2,
                   (float(value.imag.a) + float(value.imag.b)) / 2)


def main():
    rows = []
    for n in (8, 24, 40):
        m = drift(n, False) + drift(n, False).conj().T
        for z in (0.0, 0.0145, 0.15, 0.2, 0.25, 0.15 + 0.003j, 0.2 - 0.003j):
            direct = np.asarray(boundary_values(z, n))
            modal = np.asarray(modal_boundary_values(modal_data(z), n))
            modal_error = float(np.max(np.abs(modal - direct) / np.maximum(1e-30, np.abs(direct))))
            w, wsh = profile(z, n), profile(z, n, sharp=True)
            reference = complex(wsh.ravel() @ (m @ w.ravel()))
            f = crossing_numerator(z, n)
            a, b, d = direct
            ash, bsh, dsh = boundary_values(z, n, sharp=True)
            scale = KAPPA * (abs(ash * a) + abs(bsh * b)) + abs(dsh * b) + abs(bsh * d)
            identity_error = float(abs(f - reference) / scale)
            c, tau, data, sh = coefficient_and_phase(z, n)
            an, _, _ = scaled_modes(data, n)
            ans, _, _ = scaled_modes(sh, n)
            scalar_reference = KAPPA - c * tau ** (n - 2)
            scalar_exact = f / (ash * a)
            scalar_error = float(abs(scalar_exact - scalar_reference)
                                 / (KAPPA + abs(c * tau ** (n - 2))))
            port_error = 0.0
            if complex(z).imag == 0:
                source = np.zeros(2 * n, complex)
                source[-2] = 1
                resolvent = -1j * z * np.eye(2 * n) - drift(n, True).conj().T
                y = np.linalg.solve(resolvent, source)
                q = complex(np.vdot(y, m @ y))
                corrected = -2 * y[-2].real + KAPPA * (abs(y[0]) ** 2 + abs(y[-2]) ** 2)
                port_error = float(abs(q - corrected)
                                   / (2 * abs(y[-2]) + KAPPA * (abs(y[0]) ** 2 + abs(y[-2]) ** 2)))
            rows.append({"N": n, "z_real": complex(z).real, "z_imag": complex(z).imag,
                         "modal_relative_error": modal_error,
                         "boundary_identity_term_normalised_error": identity_error,
                         "scalar_reduction_term_normalised_error": scalar_error,
                         "port_identity_term_normalised_error": port_error})
    if max(max(row[key] for key in row if key.endswith("error")) for row in rows) > 1e-9:
        raise ArithmeticError("An independent boundary identity audit failed")
    coefficient_rows = []
    for z in (0.0, 0.005, 0.05, 0.15, 0.2, 0.25, 1.0, 2.0, 0.2 + 0.004j):
        eps = 1e-9
        box = complex_box(complex(z).real - eps, complex(z).real + eps,
                          complex(z).imag - eps, complex(z).imag + eps)
        data, sh = certified_modal_data(box), certified_modal_data(box, sharp=True)
        approximate = modal_data(z)
        coeff_error = max(abs(midpoint(enclosure.value) - reference)
                          / max(1e-8, abs(reference))
                          for enclosures, references in ((data.a, approximate.a),
                                                         (data.b, approximate.b),
                                                         (data.d, approximate.d))
                          for enclosure, reference in zip(enclosures, references))
        cn = complex_leading_coefficient(data, sh)
        cref, _ = limiting_coefficient(z)
        c_error = abs(midpoint(cn.value) - cref) / abs(cref)
        if coeff_error > 1e-5 or c_error > 1e-6:
            raise ArithmeticError(f"Residue formula mismatch at z={z}")
        delta = 1e-6
        cplus, _ = limiting_coefficient(z + delta)
        cminus, _ = limiting_coefficient(z - delta)
        derivative = (cplus - cminus) / (2 * delta)
        derivative_error = abs(midpoint(cn.derivative) - derivative)
        if derivative_error > 2e-4:
            raise ArithmeticError(f"Analytic coefficient derivative mismatch at z={z}")
        coefficient_rows.append({"z_real": complex(z).real, "z_imag": complex(z).imag,
                                 "residue_midpoint_relative_error": coeff_error,
                                 "C_infinity_midpoint_relative_error": c_error,
                                 "C_infinity_derivative_absolute_error": derivative_error,
                                 "certified_root_disk_margin": float_down(data.disk_margin)})
    for name, records in (("boundary_evans_identity_audit", rows),
                          ("boundary_evans_residue_audit", coefficient_rows)):
        path = ROOT / "results" / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(path)
    print("Boundary, modal, corrected port, residue and derivative audits PASSED")


if __name__ == "__main__":
    main()
