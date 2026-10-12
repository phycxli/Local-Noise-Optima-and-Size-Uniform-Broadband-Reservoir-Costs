"""Match both complete spectral caps and the total coupling-strength proxy."""

import csv
import json

import cvxpy as cp
import numpy as np

from physics import ROOT, grams, matrices, parameters


def solve_case(z, source, eta, cap, trace_cap, radius, path):
    n, eye = len(z), np.eye(len(z))
    entries = [cp.Variable(n - j) for j in range(radius + 1)]
    f = cp.diag(entries[0])
    for j in range(1, radius + 1):
        f += cp.diag(entries[j], j) + cp.diag(entries[j], -j)
    shifted = source + eta * eye
    constraints = [f >> 0, f + shifted >> 0, (cap - eta) * eye - f >> 0,
                   cap * eye - f - shifted >> 0,
                   2 * cp.trace(f) + np.trace(source) + 2 * eta * n <= trace_cap]
    problem = cp.Problem(cp.Minimize(cp.trace(z @ f)), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=2e-10, tol_gap_rel=2e-10,
                  tol_feas=2e-10, max_iter=300, max_threads=1)
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    raw = (f.value + f.value.T) / 2
    # A second SDP supplies an interior anchor respecting the additional trace cap.
    anchor_entries = [cp.Variable(n - j) for j in range(radius + 1)]
    anchor_expr = cp.diag(anchor_entries[0])
    for j in range(1, radius + 1):
        anchor_expr += cp.diag(anchor_entries[j], j) + cp.diag(anchor_entries[j], -j)
    slack = cp.Variable()
    anchor_constraints = [anchor_expr - slack * eye >> 0, anchor_expr + shifted - slack * eye >> 0,
                          (cap - eta) * eye - anchor_expr - slack * eye >> 0,
                          cap * eye - anchor_expr - shifted - slack * eye >> 0,
                          2 * cp.trace(anchor_expr) + np.trace(source) + 2 * eta * n + 2 * n * slack <= trace_cap]
    anchor_problem = cp.Problem(cp.Maximize(slack), anchor_constraints)
    anchor_problem.solve(solver="CLARABEL", tol_gap_abs=2e-10, tol_gap_rel=2e-10,
                         tol_feas=2e-10, max_iter=300, max_threads=1)
    assert anchor_problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    anchor = (anchor_expr.value + anchor_expr.value.T) / 2
    margin = min(np.linalg.eigvalsh(t).min() for t in
                 (anchor, anchor + shifted, (cap - eta) * eye - anchor, cap * eye - anchor - shifted))
    trace_margin = trace_cap - (2 * np.trace(anchor) + np.trace(source) + 2 * eta * n)
    assert margin > 0 and trace_margin > 0, (margin, trace_margin)
    violation = max(0., *(-np.linalg.eigvalsh(t).min() for t in
                          (raw, raw + shifted, (cap - eta) * eye - raw, cap * eye - raw - shifted)))
    trace_violation = max(0., 2 * np.trace(raw) + np.trace(source) + 2 * eta * n - trace_cap)
    mixing = max((violation + 2e-9) / (margin + violation + 2e-9),
                 (trace_violation + 2e-9) / (trace_margin + trace_violation + 2e-9))
    programmed = (1 - mixing) * raw + mixing * anchor
    loss, gain = programmed + eta * eye, programmed + shifted
    factors = []
    for c in constraints[:4]:
        raw_dual = (c.dual_value + c.dual_value.T) / 2
        eigenvalues, vectors = np.linalg.eigh(raw_dual)
        positive = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T + 1e-10 * eye
        factors.append(np.linalg.cholesky(positive))
    a, b, c, d = [v @ v.T for v in factors]
    mu = max(0., float(constraints[4].dual_value))
    e = z - a - b + c + d + 2 * mu * eye
    penalty = np.maximum(-np.diag(e), 0).sum()
    for j in range(1, radius + 1):
        penalty += abs(np.diag(e, j)).sum()
    lower = (np.sum((d - b) * shifted.T) - (cap - eta) * np.trace(c) - cap * np.trace(d)
             - mu * (trace_cap - np.trace(source) - 2 * eta * n) - (cap - eta) * penalty
             + np.sum(z * shifted.T) - 1e-7)
    upper = float(np.sum(z * gain.T))
    rate = float(max(np.linalg.eigvalsh(gain).max(), np.linalg.eigvalsh(loss).max()))
    total = float(np.trace(gain) + np.trace(loss))
    assert rate < cap and total < trace_cap and upper > lower
    if radius == 1:
        assert upper - lower < 1e-4
    primal = {}
    for key, target in (("programmed_loss", programmed), ("gain", gain),
                        ("complete_loss_slack", cap * eye - loss), ("gain_slack", cap * eye - gain)):
        shift = np.linalg.eigvalsh(target).min() / 2
        factor = np.linalg.cholesky(target - shift * eye)
        assert shift > np.max(np.sum(abs(target - shift * eye - factor @ factor.T), axis=1)) + 1e-11
        primal[f"primal_{key}_shift"], primal[f"primal_{key}_factor"] = shift, factor
    np.savez_compressed(path, gain=gain, loss=loss, gram=z, source=source, **primal,
                        cap=cap, trace_cap=trace_cap, intrinsic=eta, trace_dual=mu,
                        **{f"dual_factor_{j}": v for j, v in enumerate(factors)})
    return {"cold_lower": float(lower), "cold_upper": upper, "maximum_complete_rate": rate,
            "complete_trace_used": total, "primal_dual_gap": upper - float(lower)}


def main():
    directory = ROOT / "results/trace_budget_witnesses"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("uniform", "dimerized"):
        p, eta, width, nb = parameters(name), .01, .05, .001
        source = matrices(p, 64)[1]
        for linewidth in (80., 100.):
            w, z, *_ = grams(p, 64, width, linewidth, eta, 193)
            for trace_cap in (175., 190., 210., 250., 325.):
                pair = []
                for radius in (1, 2):
                    path = directory / f"{name}_k{linewidth:g}_T{trace_cap:g}_r{radius}.npz"
                    record = solve_case(z, source, eta, 8., trace_cap, radius, path)
                    offset = -nb * np.sum(z * (source + eta * np.eye(64)).T) + nb * eta * np.trace(w)
                    record["thermal_lower"] = (1 + 2 * nb) * record["cold_lower"] + offset
                    record["thermal_upper"] = (1 + 2 * nb) * record["cold_upper"] + offset
                    record["radius"] = radius
                    record["matrix_file"] = path.relative_to(ROOT).as_posix()
                    pair.append(record)
                row = {"model": name, "N": 64, "linewidth": linewidth, "intrinsic": eta,
                       "halfwidth": width, "occupation": nb, "spectral_cap": 8., "trace_cap": trace_cap,
                       "r1_lower": pair[0]["thermal_lower"], "r2_upper": pair[1]["thermal_upper"],
                       "gap_lower_float": pair[0]["thermal_lower"] - pair[1]["thermal_upper"],
                       "r1_trace_used": pair[0]["complete_trace_used"], "r2_trace_used": pair[1]["complete_trace_used"],
                       "local_primal_dual_gap": pair[0]["primal_dual_gap"],
                       "reference_primal_dual_gap": pair[1]["primal_dual_gap"],
                       "r1_matrix_file": pair[0]["matrix_file"], "r2_matrix_file": pair[1]["matrix_file"]}
                rows.append(row)
    with (ROOT / "results/matched_trace_budget.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"scope": "N=64 nominal physical models; both operator norm and complete trace caps matched",
               "rows": len(rows), "minimum_gap_lower_float": min(r["gap_lower_float"] for r in rows),
               "positive_gap_cases": int(sum(r["gap_lower_float"] > 0 for r in rows)),
               "maximum_local_primal_dual_gap": max(r["local_primal_dual_gap"] for r in rows),
               "maximum_reference_primal_dual_gap": max(r["reference_primal_dual_gap"] for r in rows),
               "reference_scope": "The range-two device supplies a verified feasible upper bound; separation does not require its optimality",
               "resource_mapping": "sum |U_g|^2+|U_l|^2 = kappa_b*(Tr(G)+Tr(L)-N*eta)/4; a coupling-strength proxy, not pump watts"}
    (ROOT / "results/matched_trace_budget_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
