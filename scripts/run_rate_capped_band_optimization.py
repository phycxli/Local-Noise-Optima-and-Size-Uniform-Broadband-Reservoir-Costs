"""Fixed-reservoir noise optimization with a full gain/loss rate cap.

The scalar endpoint-normalized response is a polynomial.  Its symmetric-band
Gram is integrated using rational Legendre coefficients, so no ill-conditioned
Gram inverse or frequency-dependent reservoir enters the shared problem.
Stored primal matrices and dual factors give independently checkable bounds.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import time

import cvxpy as cp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh_tridiagonal

from audit_high_gain_band_bound import response_vector
from run_finite_frequency_local_optimality import scalar_drift, scalar_kernel
from run_rate_constrained_local_cp import alpha_star, normalize_kernel

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
EPS = np.finfo(float).eps


def sym(a: np.ndarray) -> np.ndarray:
    return (a + a.conj().T) / 2


def normalized_response(n: int, omega: float) -> np.ndarray:
    u = np.zeros(n, dtype=complex)
    u[0] = 1
    u[1] = 1.3 - 1j * omega
    for j in range(1, n - 1):
        u[j + 1] = (1.2 - 1j * omega) * u[j] - 0.25 * u[j - 1]
    return u / np.sqrt(0.2)


def rational_band_gram(n: int, halfwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """c[j,l] multiplies (-i)^l P_l(omega/halfwidth) in u_j."""
    width = Fraction(str(halfwidth))
    c = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    c[0][0] = Fraction(1)
    c[1][0], c[1][1] = Fraction(13, 10), width
    for j in range(1, n - 1):
        for ell in range(j + 2):
            value = Fraction(6, 5) * c[j][ell] - c[j - 1][ell] / 4
            if ell:
                value += width * Fraction(ell, 2 * ell - 1) * c[j][ell - 1]
            if ell + 1 < n:
                value -= width * Fraction(ell + 1, 2 * ell + 3) * c[j][ell + 1]
            c[j + 1][ell] = value
    gram = np.zeros((n, n))
    error = np.zeros_like(gram)
    for i in range(n):
        for j in range(i + 1):
            exact = sum(
                (5 * c[i][ell] * c[j][ell] / (2 * ell + 1)
                 for ell in range(min(i, j) + 1)), Fraction(0)
            )
            value = float(exact)
            rounding = float(abs(exact - Fraction.from_float(value)))
            gram[i, j] = gram[j, i] = value
            error[i, j] = error[j, i] = rounding * (1 + 8 * EPS)
    return gram, error


def quadrature_gram(n: int, halfwidth: float, order: int) -> np.ndarray:
    nodes, weights = leggauss(order)
    columns = np.column_stack([
        np.sqrt(weight / 2) * normalized_response(n, halfwidth * node)
        for node, weight in zip(nodes, weights)
    ])
    return sym(columns @ columns.conj().T).real


def minimum_band_gain(n: int, halfwidth: float) -> float:
    diagonal = np.full(n, 1.2)
    diagonal[[0, -1]] += 0.1
    decay_rates = eigh_tridiagonal(diagonal, np.full(n - 1, 0.5), eigvals_only=True)
    if decay_rates.min() <= 0:
        raise ValueError("The probed scalar drift must be stable.")
    # |det(A-iw)|^2 = product_j(lambda_j^2+w^2), increasing with |w|.
    return float(np.exp(np.log(0.04) - np.log(decay_rates**2 + halfwidth**2).sum()))


def nonlocal_pointwise_certificate(n: int, halfwidth: float) -> dict:
    """Prove that L=7(I-uu*) reaches the floor throughout this band.

    M+7I is positive.  Its rank-one subtraction is PSD if
    v*(I-7(M+7I)^-1)v >= 0.  Compute this polynomial with rational arithmetic
    and bound it on [-1,1] by its constant term minus all other coefficients.
    """
    width = Fraction(str(halfwidth))
    coefficients = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    coefficients[0][0] = Fraction(1)
    coefficients[1][0], coefficients[1][1] = Fraction(13, 10), width
    for j in range(1, n - 1):
        for k in range(j + 2):
            coefficients[j + 1][k] = Fraction(6, 5) * coefficients[j][k] - coefficients[j - 1][k] / 4
            if k:
                coefficients[j + 1][k] += width * coefficients[j][k - 1]
    diagonal = [Fraction(23, 5)]
    lower = []
    for j in range(1, n):
        lower.append(Fraction(5, 4) / diagonal[j - 1])
        diagonal.append(Fraction(23, 5) - Fraction(5, 4) * lower[-1])
    transformed = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for k in range(n):
        work = [coefficients[j][k] for j in range(n)]
        for j in range(1, n):
            work[j] -= lower[j - 1] * work[j - 1]
        solution = [work[j] / diagonal[j] for j in range(n)]
        for j in range(n - 2, -1, -1):
            solution[j] -= lower[j] * solution[j + 1]
        for j in range(n):
            transformed[j][k] = coefficients[j][k] - 7 * solution[j]
    polynomial = [Fraction(0) for _ in range(2 * n - 1)]
    for k in range(n):
        for ell in range(k, n):
            if (k - ell) % 2:
                continue
            sign = 1 if ((k - ell) // 2) % 2 == 0 else -1
            value = sum((coefficients[j][k] * transformed[j][ell] for j in range(n)), Fraction(0))
            polynomial[k + ell] += (1 if k == ell else 2) * sign * value
    margin = polynomial[0] - sum(abs(v) for v in polynomial[1:])
    return {"N": n, "halfwidth": halfwidth, "alpha": 7,
            "polynomial_margin": float(margin), "valid": margin > 0,
            "gain_rate_upper": 7.1, "loss_rate": 7,
            "margin_numerator": str(margin.numerator), "margin_denominator": str(margin.denominator),
            "polynomial_coefficients": [str(v) for v in polynomial],
            "scope": "A separately chosen nonlocal reservoir at each frequency, not a shared reservoir."}


def allowed_mask(n: int, radius: int | None) -> np.ndarray:
    return (np.ones((n, n), dtype=bool) if radius is None else
            np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= radius)


def factor_product_error(factor: np.ndarray) -> np.ndarray:
    n = factor.shape[1]
    magnitude = np.abs(factor) @ np.abs(factor.conj().T)
    return 8 * n * EPS * magnitude + np.finfo(float).tiny


def psd_check(matrix: np.ndarray, entry_error: np.ndarray) -> dict:
    """Check positivity through a Cholesky factor and a row-norm error bound."""
    n = len(matrix)
    margin = float(np.linalg.eigvalsh(matrix).min())
    if margin <= 0:
        return {"valid": False, "margin": margin, "checked_margin": margin}
    shift = margin / 2
    factor = np.linalg.cholesky(matrix - shift * np.eye(n))
    product = factor @ factor.conj().T
    residual = np.abs(matrix - shift * np.eye(n) - product)
    error = (entry_error + factor_product_error(factor)
             + 16 * EPS * (np.abs(matrix) + np.abs(product) + shift * np.eye(n)))
    remainder = float(np.max(np.sum(residual + error, axis=1))) * (1 + 8 * n * EPS)
    return {"valid": shift > remainder, "margin": margin,
            "checked_margin": shift - remainder, "shift": shift,
            "residual_bound": remainder, "factor": factor}


def repair_primal(raw: np.ndarray, source: np.ndarray, cap: float,
                  mask: np.ndarray) -> tuple[np.ndarray, float, dict]:
    n = len(source)
    h = sym(raw) * mask
    q = source / cap
    anchor = (np.eye(n) - q) / 2
    anchor_margin = (1 - np.linalg.norm(q, 2)) / 2
    if anchor_margin <= 0:
        raise ValueError("Use a rate cap strictly above ||M|| for the audit.")
    violation = max(0., *(-np.linalg.eigvalsh(a).min() for a in
                         (h, h + q, np.eye(n) - h, np.eye(n) - h - q)))
    safety = 2e-10
    mixing = (violation + safety) / (anchor_margin + violation + safety)
    loss = cap * ((1 - mixing) * h + mixing * anchor)
    gain = sym(loss + source)
    # M's rational entries differ from their binary representations by < EPS.
    model_error = np.full((n, n), 4 * EPS)
    checks = {}
    for name, matrix, error in (
        ("loss", loss, np.zeros((n, n))),
        ("gain", gain, model_error + 4 * EPS * (np.abs(loss) + np.abs(source))),
        ("loss_slack", cap * np.eye(n) - loss, 4 * EPS * (cap * np.eye(n) + np.abs(loss))),
        ("gain_slack", cap * np.eye(n) - gain,
         model_error + 8 * EPS * (cap * np.eye(n) + np.abs(loss) + np.abs(source))),
    ):
        checks[name] = psd_check(matrix, error)
    if not all(check["valid"] for check in checks.values()):
        raise RuntimeError("The repaired reservoir failed a matrix certificate.")
    return loss, mixing, checks


def dual_bound(gram: np.ndarray, gram_error: np.ndarray, source: np.ndarray,
               cap: float, raw_duals: list[np.ndarray], mask: np.ndarray) -> tuple[float, dict]:
    n = len(source)
    factors, matrices, errors = [], [], []
    for raw in raw_duals:
        values, vectors = np.linalg.eigh(sym(raw))
        positive = sym((vectors * np.maximum(values, 0)) @ vectors.conj().T)
        positive += 1e-11 * max(1., np.linalg.norm(positive, 2)) * np.eye(n)
        factor = np.linalg.cholesky(positive)
        matrices.append(factor @ factor.conj().T)
        factors.append(factor)
        errors.append(factor_product_error(factor))
    a, b, c, d = matrices
    residual = (gram - a - b + c + d) * mask
    residual_error = (gram_error + sum(errors)
                      + 16 * EPS * (np.abs(gram) + sum(np.abs(v) for v in matrices))) * mask
    residual_bound = float(np.max(np.sum(np.abs(residual) + residual_error, axis=1)))
    residual_bound *= 1 + 8 * n * EPS
    q = source / cap
    objective = float(np.trace((d - b) @ q).real - np.trace(c + d).real)
    objective_error = float(np.sum((errors[1] + errors[3]) * np.abs(q.T))
                            + np.trace(errors[2] + errors[3]))
    objective_error += 4 * EPS * float(np.sum(np.abs(d - b))) / cap
    objective_error += 128 * n * EPS * (1 + abs(objective)
                           + np.linalg.norm(q, 1) * sum(np.linalg.norm(v, 1) for v in matrices))
    base = float(np.trace(gram @ source).real)
    base_error = float(np.sum(gram_error * np.abs(source.T))) + 64 * n * EPS * (1 + abs(base))
    lower = base + cap * (objective - n * residual_bound - objective_error) - base_error
    return lower, {"factors": factors, "stationarity_row_bound": residual_bound,
                   "dual_objective": base + cap * objective,
                   "dual_roundoff_allowance": cap * objective_error + base_error}


def solve_shared(n: int, cap: float, gram: np.ndarray, gram_error: np.ndarray,
                 source: np.ndarray, radius: int | None, tolerance: float,
                 path: Path) -> dict:
    started = time.perf_counter()
    mask = allowed_mask(n, radius)
    h = cp.Variable((n, n), symmetric=True)
    q, eye = source / cap, np.eye(n)
    constraints = [h >> 0, h + q >> 0, eye - h >> 0, eye - h - q >> 0]
    if radius is not None:
        rows, columns = np.where(np.triu(~mask, 1))
        constraints.append(h[rows, columns] == 0)
    problem = cp.Problem(cp.Minimize(cp.trace(gram @ h)), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=tolerance, tol_gap_rel=tolerance,
                  tol_feas=tolerance, max_iter=300, max_threads=1)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or h.value is None:
        raise RuntimeError(f"Shared SDP failed: {problem.status}")
    loss, mixing, checks = repair_primal(h.value, source, cap, mask)
    gain = sym(loss + source)
    lower, dual = dual_bound(gram, gram_error, source, cap,
                            [v.dual_value for v in constraints[:4]], mask)
    upper = float(np.trace(gram @ gain).real)
    upper_error = float(np.sum(gram_error * np.abs(gain.T)))
    upper_error += 128 * n * EPS * (1 + abs(upper) + cap * np.linalg.norm(gram, 1))
    upper += upper_error
    arrays = {"gain": gain, "loss": loss, "gram": gram, "source": source,
              "gram_entry_error": gram_error, "mask": mask, "cap": np.array(cap)}
    for name, check in checks.items():
        arrays[f"primal_{name}_factor"] = check["factor"]
        arrays[f"primal_{name}_shift"] = np.array(check["shift"])
    for index, factor in enumerate(dual["factors"]):
        arrays[f"dual_factor_{index}"] = factor
    np.savez_compressed(path, **arrays)
    point_floor = float(np.trace(gram @ source).real)
    return {"radius": "all" if radius is None else radius, "rate_cap": cap,
            "lower": lower, "upper": upper, "bound_gap": upper - lower,
            "point_floor_average": point_floor, "extra_lower": lower - point_floor,
            "extra_upper": upper - point_floor, "solver_status": problem.status,
            "primal_mixing": mixing,
            "primal_checked_margin": min(v["checked_margin"] for v in checks.values()),
            "dual_stationarity_bound": dual["stationarity_row_bound"],
            "rate_used": max(np.linalg.eigvalsh(v).max() for v in (gain, loss)),
            "matrix_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "elapsed_seconds": time.perf_counter() - started}


def pointwise_construction(n: int, omega: float) -> dict:
    source = (scalar_drift(n) + scalar_drift(n).conj().T)
    z = normalized_response(n, omega)
    kernel = normalize_kernel(scalar_kernel(z))
    alpha = alpha_star(source, kernel)
    loss = (alpha * (1 + 1e-7) + 1e-9) * kernel
    gain = source + loss
    return {"omega": omega, "rate_required": float(max(np.linalg.eigvalsh(v).max()
                                                      for v in (gain, loss))),
            "floor": float(np.vdot(z, source @ z).real),
            "noise": float(np.vdot(z, gain @ z).real),
            "loss": loss, "gain": gain}


def run_case(n: int, halfwidth: float, caps: list[float], radii: list[int | None],
             tolerance: float, stem: str, pointwise: bool) -> list[dict]:
    source = (scalar_drift(n) + scalar_drift(n).conj().T).real
    gram, gram_error = rational_band_gram(n, halfwidth)
    quad_error = max(np.max(np.abs(gram - quadrature_gram(n, halfwidth, q)))
                     for q in (n + 8, 2 * n + 9))
    nodes, weights = leggauss(max(n + 8, 81))
    floor_samples = [float(np.vdot(normalized_response(n, halfwidth * w),
                                   source @ normalized_response(n, halfwidth * w)).real)
                     for w in nodes]
    min_gain = minimum_band_gain(n, halfwidth)
    high_gain = min_gain >= 1.1
    if high_gain and min(floor_samples) <= 0:
        raise RuntimeError("The high-gain band cannot have a negative fixed-drift branch.")
    points = ([pointwise_construction(n, halfwidth * float(w))
               for w in np.linspace(-1, 1, 17)] if pointwise else [])
    point_certificate = nonlocal_pointwise_certificate(n, halfwidth) if pointwise else {}
    original_gain = np.zeros((n, n))
    for j in range(n - 1):
        original_gain[j:j + 2, j:j + 2] += 0.625
    original_noise = float(np.trace(gram @ original_gain))
    matrix_dir = RES_DIR / f"{stem}_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    if point_certificate:
        (matrix_dir / f"N{n}_B{halfwidth:g}_pointwise.json").write_text(
            json.dumps(point_certificate, indent=2) + "\n", encoding="utf-8")
    rows = []
    for cap in caps:
        for radius in radii:
            label = "all" if radius is None else str(radius)
            path = matrix_dir / f"N{n}_B{halfwidth:g}_R{cap:g}_r{label}.npz"
            row = solve_shared(n, cap, gram, gram_error, source, radius, tolerance, path)
            row.update({"N": n, "halfwidth": halfwidth, "min_band_gain": min_gain,
                        "high_gain_margin_0p1": high_gain, "quadrature_check_error": quad_error,
                        "floor_min_sample": min(floor_samples), "original_noise": original_noise,
                        "sampled_pointwise_rate_max": max((p["rate_required"] for p in points), default=0),
                        "sampled_pointwise_formula_error": max((abs(p["noise"] - p["floor"])
                                                                 for p in points), default=0),
                        "pointwise_floor_attainable_under_cap": bool(point_certificate.get("valid", False)
                                                                    and cap >= 7.1),
                        "pointwise_rational_margin": point_certificate.get("polynomial_margin", 0)})
            if not high_gain:
                # The signed linear baseline is not the sign-adaptive floor.
                true_floor = float(np.dot(weights / 2, np.maximum(floor_samples, 0)))
                row["point_floor_average"] = true_floor
                row["extra_lower"], row["extra_upper"] = row["lower"] - true_floor, row["upper"] - true_floor
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[40, 48, 64])
    parser.add_argument("--halfwidths", nargs="+", type=float, default=[0.05])
    parser.add_argument("--caps", nargs="+", type=float, default=[8, 32, 256])
    parser.add_argument("--radii", nargs="+", default=["all", "1", "2"])
    parser.add_argument("--tolerance", type=float, default=2e-10)
    parser.add_argument("--stem", default="rate_capped_band_optimization")
    parser.add_argument("--skip-pointwise", action="store_true")
    args = parser.parse_args()
    radii = [None if r == "all" else int(r) for r in args.radii]
    RES_DIR.mkdir(exist_ok=True)
    rows = []
    for n in args.sizes:
        for halfwidth in args.halfwidths:
            rows += run_case(n, halfwidth, args.caps, radii, args.tolerance,
                             args.stem, not args.skip_pointwise)
            with (RES_DIR / f"{args.stem}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    metadata = {"arguments": vars(args), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "numpy": np.__version__, "cvxpy": cp.__version__,
                "integration": "Exact rational Legendre polynomial Gram; quadrature is an independent check.",
                "gain_minimum": "Analytic endpoint minimum from the positive symmetric probed drift.",
                "certificate": "Primal Cholesky residual bounds; PSD dual factors and a stationarity row-norm correction.",
                "scope": "Fixed scalar model, finite sizes, symmetric-band mean. No multiband or all-N claim."}
    (RES_DIR / f"{args.stem}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
